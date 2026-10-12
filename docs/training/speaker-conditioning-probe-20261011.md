# Speaker 条件捷径诊断与 refmask 方案评估

日期：2026-10-11。本文是一次性诊断的记录，不是正式实验；诊断脚本和逐句结果**不进入本仓库**，当时放在会话临时目录 `/tmp/claude-1002/-workspace-workspace-yanglin/8aafbba1-e57d-4267-9d3e-8daa175ba13f/scratchpad/probe/`（会话结束后可能不再存在）。脚本只依赖 `lm_tts.data.reader.FeatureDataset`、`lm_tts.data.batch.collate`、`lm_tts.models.loading.load_model` 和已导出的 FP32 权重，按第 2 节的方法可以重写。背景与已有评测见 [all16 训练与 Seed-TTS 评测](all16-training-and-seedtts-evaluation.md) 第 6、9 节，refmask 实验见 [all16-reference-20261009](all16-reference-20261009.md)。

临时目录中的文件：

| 文件 | 内容 |
| --- | --- |
| `speaker_probe.py` | teacher-forcing 换条件探针，输出逐句 JSON |
| `analyze.py` | 汇总 JSON 为表格 |
| `fc17500.json` / `fc47382.json` / `tb17500.json` | 三个 checkpoint 的逐句结果 |
| `summary.md` | `analyze.py` 的完整输出 |
| `text_embedding_compare.py` | TTS 2048 维 vs Qwen3-0.6B-Base 1024 维 text embedding 几何对比 |
| `text_embedding_origin.py` | TTS text embedding / projector / Talker 来源核查 |
| `refstats.py` | reference features 的裁剪长度与 embedding 相似度抽样统计 |

---

## 0. 结论摘要

1. **frozen-conditioning 模型确实在从 speaker embedding 里读内容。** 把目标自身 embedding 换成同说话人另一条，首码本 CE 在 99.3%（17.5k）/ 99.9%（47k）的验证句子上变差。用裁剪片段 embedding 时，片段覆盖的帧 CE 与自身完整 embedding 相同，未覆盖的帧 CE 比用其他说话人的 embedding 还差——embedding 携带的内容在主动误导预测。
2. **依赖随训练加深。** 同说话人换条件的 CE 惩罚从 17.5k 的 +0.30 涨到 47k 的 +0.74；这与 Seed-TTS 评测里 WER 从 17.5k 往后变差的趋势一致。
3. **EOS 也依赖自身 embedding。** 47k 用自身 embedding 时 EOS CE 0.15，换同说话人另一条后 1.77，30% 句子的 EOS 预测不可靠。这对应评测里出现的超过 30 秒长输出。
4. **泄漏集中在首码本。** 残余 15 个码本对 embedding 的依赖主要是说话人身份（换其他说话人 +0.88），裁剪区间内外几乎无差（0.08）。Code Predictor 没有 cross-attention，speaker 信息只经 Talker hidden `h` 进入。
5. **refmask 方案对症。** 它 mask 掉的恰好是裁剪 embedding 覆盖的"便宜"帧，保留的恰好是当前被误导最严重的帧；训练时模型必须学会忽略 embedding 的内容分量。短裁剪（<1 s）的误导最强，而现有 reference features 中 57–83% 的片段不足 1.5 s，训练信号充足。EOS 依赖不会被 mask 直接切断，但 embedding 不再覆盖整句时长信息，预期会明显减弱；评测时需单独看长输出比例。
6. **text embedding 核查**：Qwen3-TTS 0.6B 与 1.7B 共用同一张 2048 维 text embedding（仅 3 个 TTS 控制 token 行不同），它与 Qwen3 0.6B / 1.7B 文本 LM 的 embedding 几乎正交（逐行余弦 0.0001，线性 R² 0.15–0.20），是 TTS 专用、独立训练的表。有效维度约 1,600，Qwen3-0.6B-Base 的 1024 维表约 900。textbase 的劣势同时包含"未经 TTS 训练"和"容量更窄"，现有数据无法分离两者。

---

## 1. 背景

`docs/training/all16-training-and-seedtts-evaluation.md` 第 6 节记录：训练用目标音频自身的 speaker embedding，推理用另一条参考音频的 embedding。在 frozen-conditioning 最终 checkpoint 上固定 16+16 条目标、只替换 speaker 条件（speaker-only，双 greedy）：

| speaker 条件 | 英文 WER | 中文 CER |
| --- | ---: | ---: |
| 目标录音自身 embedding | 0.0% | 1.3% |
| 同说话人另一条录音 | 75.7% | 48.1% |

refmask 方案（`configs/supervised-tts-20260929-all16-refmask-20261009-*.yaml`，数据 `data/builds/supervised-tts-20260929-reference-20261009-all16`）让 embedding 只来自目标音频的一段裁剪 `[a, b)`，并对这段 codec 不计 loss。本报告回答：旧问题的机制是什么、refmask 能否缓解、以及顺带核查 text embedding 的差异与来源。

## 2. 探针方法

- **模型**：UltraEval 中已导出的 FP32 权重 `all16-frozen-conditioning-step-00017500`、`-00047382`、`all16-textbase-step-00017500`，SDPA attention，不训练。
- **数据**：reference build 的 `validation/manifest.json`，每数据集随机抽至多 300 条，共 **3,566 句、358,996 帧**（16 个数据集）。每句读取完整 codec、文本 token、已发布的裁剪区间 `[a,b)` 与裁剪 embedding。
- **四种 speaker 条件**，其余输入完全相同：
  - `自身完整`：原训练条件，来自基线 merged build 的完整音频 embedding；
  - `自身裁剪`：reference build 中 `[a,b)` 片段的 embedding；
  - `同说话人另一条`：同 `speaker_id`、不同 `target_id` 的完整音频 embedding（随机抽一条）；galgame 无标签、wenetspeech4tts 无同说话人样本，该条件缺失（2,959 句有效）；
  - `其他说话人`：随机一条不同 `speaker_id` 的完整音频 embedding；ljspeech 只有一个说话人，缺失。
- **统计**：逐帧首码本 CE / top-1 acc、残余 15 码本平均 CE、EOS CE；按 `[a,b)` 内外、句内相对位置十分位、数据集、裁剪长度分组。全部为帧加权均值。
- **设计要点**：`自身裁剪` 条件下，区间内的帧内容被 embedding 覆盖、区间外没有，而说话人、录音环境、韵律完全相同，因此"内 − 外"之差直接度量内容泄漏；`自身完整` 条件的"内 − 外"是位置偏差对照。

## 3. 探针结果

### 3.1 整体首码本 CE

| speaker 条件 | FC 17.5k | FC 47k | textbase 17.5k |
| --- | ---: | ---: | ---: |
| 自身完整 | 1.115 | 0.912 | 1.146 |
| 同说话人另一条 | 1.411 (+0.30) | 1.650 (+0.74) | 1.361 (+0.21) |
| 其他说话人 | 1.901 (+0.79) | 2.246 (+1.33) | 1.790 (+0.64) |
| 自身裁剪 | 1.524 | 1.846 | 1.485 |

首码本 top-1 acc：FC 47k 自身完整 69.5%，同说话人 56.0%，其他说话人 45.5%。

- 逐句看，`自身完整` 优于 `同说话人另一条` 的比例：FC 17.5k 99.3%，FC 47k 99.9%，textbase 97.5%。仅用标签可靠的数据集（n=2,363）时，逐句 CE 差的中位数：17.5k 0.32，47k 0.79，textbase 0.23。
- 17.5k → 47k，同说话人换条件的惩罚增加约 2.5 倍，其他说话人的惩罚增加 1.7 倍。模型对"自身特有信息"的依赖增长快于对"说话人身份"的依赖。
- textbase 的依赖略弱于同步数的 frozen-conditioning，可能与可训练文本前端有关，差别不大。

### 3.2 裁剪区间内外（首码本 CE）

| 条件 | FC 17.5k 内 / 外 | FC 47k 内 / 外 | textbase 内 / 外 |
| --- | --- | --- | --- |
| 自身完整（对照） | 1.099 / 1.122 | 0.904 / 0.916 | 1.128 / 1.153 |
| **自身裁剪** | **1.105 / 1.706** | **0.998 / 2.215** | **1.152 / 1.630** |
| 同说话人另一条 | 1.371 / 1.429 | 1.563 / 1.688 | 1.322 / 1.377 |
| 其他说话人 | 1.870 / 1.915 | 2.157 / 2.285 | 1.757 / 1.804 |

- 对照组内外差 ≤ 0.03，位置本身无偏差。
- 裁剪 embedding 覆盖的帧 CE 与自身完整 embedding 持平；未覆盖的帧比 `同说话人另一条` 高 0.28（17.5k）/ 0.53（47k），比 `其他说话人` 也不低。差分（裁剪内外差 − 对照内外差）：17.5k −0.58，47k −1.20，textbase −0.45。
- 区间外 acc：47k 从 67.3%（内）掉到 49.7%（外）。

### 3.3 裁剪长度与句长

区间外惩罚 = `自身裁剪` 区间外 CE − `自身完整` 区间外 CE：

| 裁剪片段长度 | n | FC 17.5k | FC 47k |
| --- | ---: | ---: | ---: |
| < 1 s | 867 | 1.29 | 2.81 |
| 1–2 s | 1,062 | 0.95 | 2.03 |
| 2–4 s | 989 | 0.54 | 1.23 |
| > 4 s | 648 | 0.29 | 0.68 |

裁剪占句长的比例（10%–50%）对惩罚几乎没有影响（47k：1.72–1.81）。短片段 embedding 的内容更具体，误导更强；长片段的内容被池化稀释，接近"泛化的说话人向量"。同理，短句（<4 s）的同说话人换条件惩罚最大（47k：1.30），长句（>15 s）最小（0.57）。

### 3.4 句内位置

首码本 CE 按相对位置十分位（FC 47k）：

| 条件 | 0–10% | 10–20% | 中段均值 | 80–90% | 90–100% |
| --- | ---: | ---: | ---: | ---: | ---: |
| 自身完整 | 0.86 | 0.95 | ~0.90 | 0.94 | 0.97 |
| 自身裁剪 | **3.04** | 1.54 | ~1.5 | 1.85 | **2.66** |
| 同说话人另一条 | **2.04** | 1.40 | ~1.5 | 1.74 | 1.98 |

开头没有 codec 历史，模型最依赖 embedding；结尾对应 EOS 判断。ICL 推理提供 codec 前缀，恰好填补开头的空档，这与 speaker+ICL 模式 WER 远好于 speaker-only 的现象一致。

### 3.5 EOS

| 条件 | FC 17.5k EOS CE | FC 47k EOS CE | 47k 中 EOS CE > ln2 的句子比例 |
| --- | ---: | ---: | ---: |
| 自身完整 | 0.40 | **0.15** | 6.9% |
| 自身裁剪 | 0.35 | 1.20 | 20.4% |
| 同说话人另一条 | 0.98 | **1.77** | 29.8% |
| 其他说话人 | 0.89 | 1.73 | 31.7% |

模型用 embedding 判断何时结束。47k 比 17.5k 严重得多，与评测中 47k 出现更多超过 30 秒输出的记录一致。

### 3.6 残余码本与 Code Predictor

残余 15 码本平均 CE（FC 47k）：自身完整 5.11，自身裁剪 5.39，同说话人 5.63（+0.52），其他说话人 6.36（+1.26）。裁剪区间内外差 0.31（17.5k 为 0.08）；对照内外差 0.07。

Code Predictor 的输入是 `[h, emb(code₀), …, emb(code₁₄)]` 这 16 个位置的短序列，经 `small_to_mtp_projection` 与 5 层自注意力（`lm_tts/models/codec.py:110-121`），官方实现亦无 `encoder_hidden_states`，**没有 cross-attention**。speaker 信息只能经 Talker 的 `h` 进入。数据表明这条通路传递的主要是说话人身份（换说话人 CE 显著上升），内容捷径绝大部分在首码本。残余 CE 整体在 5.1 以上（2048 类均匀分布为 7.6），本身就是难预测的声学细节。

### 3.7 分数据集

所有 16 个数据集方向一致。裁剪区间外惩罚最大的是短句、单说话人多句的数据集：aishell3（47k：区间内 1.14 / 外 4.43）、vctk（1.05 / 4.18）；最小的是长句有声书：libriheavy（0.85 / 1.71）、mls_sidon（0.81 / 1.66）。完整表见 `summary.md`。

### 3.8 Embedding 几何

四种条件的向量范数中位数均在 10.4–10.6，与自身完整 embedding 的原始余弦：裁剪 0.990、同说话人 0.986、其他说话人 0.949。排除范数或尺度失配导致 CE 差异的可能；原始余弦区分度极低，需减去均值后才能分辨（见 `refstats.py`：减均值后裁剪 vs 自身 0.77、vs 同说话人另一条 0.50、随机 ~0）。

## 4. 对 refmask 方案的评估

**能缓解，且机制对症。**

- 捷径形成的条件是：embedding 中的信息能降低被监督帧的 loss。旧设置下所有帧都被监督，embedding 覆盖全部内容。refmask 下 embedding 只覆盖 `[a,b)`，而这段不计 loss；被监督的帧的内容不在 embedding 里，模型要降低 loss 只能从文本读内容，并学会忽略 embedding 的内容分量。
- 3.2 节显示，被监督的那些帧正是当前模型被裁剪 embedding 误导最严重的帧（47k 区间外 CE 2.2 对自身 0.9）；3.3 节显示短片段误导最强，而 reference features 中 ref 的中位长度只有 1–2 s（emilia 2.2 s，aishell3 1.0 s，vctk 0.96 s）。训练初期 loss 会明显高于旧实验，属预期。
- 3.5 节的 EOS 依赖不会被 mask 直接切断（ref 位于句尾的比例只有 2–4%），但 embedding 不再包含整句时长，预期会显著减弱。需要在评测中单独统计超过 30 秒的输出比例。
- 3.6 节说明 speaker 身份信息经 `h` 到 Code Predictor 的通路是合理的，refmask 不会破坏它。

**不能解决的部分**：同一录音的录音环境、语速、情绪仍在 embedding 中（减均值后裁剪 vs 自身 0.77，vs 跨句 0.50）；训练 ref 偏短而推理参考偏长；ICL 协议（ref 在句首）只有 2–4% 样本覆盖。按计划，跨句参考放到后续阶段。

**预判**：若 refmask 有效，训到 17.5k 时用本探针应看到：`自身裁剪` 的区间外 CE 接近 `同说话人另一条`（而不是更差），`同说话人另一条` 与 `自身完整` 的差距明显小于 0.30，EOS CE 在换条件后不再成倍上升。探针不需要生成，可以在正式 Seed-TTS 评测之前先跑。

## 5. Text embedding：维度与来源

### 5.1 结构（权重核对）

| | Qwen3-TTS-12Hz-0.6B-Base（frozen-conditioning 用） | Qwen3-0.6B-Base（textbase 用） |
| --- | --- | --- |
| text embedding | 151,936 × 2048 | 151,936 × 1024 |
| text projector | 2048 → 2048 → 1024，预训练 | 1024 → 1024 → 1024，随机初始化 |
| projector 参数 | 6.3M | 2.1M |

### 5.2 几何对比（`text_embedding_compare.py`）

| 指标 | TTS 2048 维 | Base 1024 维 |
| --- | ---: | ---: |
| 有效秩（谱熵） | 1,671 | 932 |
| 解释 90% 方差所需维数 | 1,574 | 840 |
| 解释 99% 方差所需维数 | 1,980 | 996 |
| 行范数均值 | 0.757 | 0.955 |

- 两张表近乎无线性关系：Base→TTS 线性拟合留出集 R² 0.17，反向 0.21；按列块余弦 0.02。
- 词汇相似结构一致：同一 token 的 top-10 近邻重合 51%（随机 0.007%）；随机 token 对的余弦相关 0.15。
- textbase 用到的 `<tts_pad/bos/eod>` 三行在 Base 表中是未训练的默认行（范数均 0.350）。

### 5.3 来源（`text_embedding_origin.py`）

| 对比 | 结果 |
| --- | --- |
| TTS-0.6B vs TTS-1.7B-Base text_embedding | 151,936 行中仅 3 行不同：`<tts_pad>`、`<tts_text_bos>`、`<tts_text_eod>`（151671–151673），其余逐 bit 相等 |
| TTS-1.7B-Base vs TTS-1.7B-CustomVoice | 完全一致 |
| TTS text_embedding vs Qwen3-ASR-1.7B embed_tokens（Qwen3-1.7B 系，2048 维） | 逐行余弦中位数 0.0001；线性 R² 0.20 / 0.15；top-10 近邻重合 53% |
| 参照：Qwen3-0.6B-Base vs Qwen3-ASR-0.6B（一次微调） | 逐行余弦 0.925 |
| 参照：Qwen3-0.6B-Base vs Qwen3-0.6B instruct | 逐行余弦 0.992 |
| TTS-0.6B vs TTS-1.7B text_projection.fc1 | 余弦 0.005，各自独立 |
| TTS-0.6B Talker 各层 vs Qwen3-0.6B-Base 对应层 | 余弦 ≈ 0（参照：ASR-0.6B 微调 vs Base 0.96） |
| TTS-1.7B Talker 各层 vs Qwen3-ASR-1.7B | 余弦 0.08–0.18，有微弱残留 |

结论：

- 官方 TTS 的 text embedding 是**两个尺寸共用的、TTS 专用、独立训练**的 2048 维表；不是 Qwen3 0.6B 或 1.7B 文本 embedding 的继承或微调（微调只会偏离到余弦 0.92–0.99，这里接近正交）。0.6B TTS 用 1.7B 那套 2048 维表，再接一个 0.6B 专属的 2048→1024 projector。
- 官方 0.6B Talker 主干也看不出 Qwen3-0.6B-Base 的痕迹。本工程用 Qwen3-0.6B-Base 的层初始化 Talker，与官方做法不同；与本次问题无直接关系，记录备查。
- 对 frozen-conditioning vs textbase：前者冻结的是一个"为 TTS 训练过的文本前端"，这才是优势所在，而非 LLM 语义；后者从 LLM 语义出发、经随机 projector 重新学 TTS 文本表示，且有效维度约为前者的 55%。"未训练"与"容量窄"在现有实验中无法分离；docs 第 9 节已列出的两组对照（Identity/near-identity projector；开放 TTS 2048 前端训练）仍是分离它们的途径。
- 文本前端直接影响内容（WER）；SIM 的差距更可能是整体建模变差的间接结果，本节数据不支持把 SIM 归因于维度。

## 6. 建议

1. **refmask 按现有配置开训**；17.5k（约 1.1 epoch，与三组已有评测进度一致）时评测。
2. **评测前先跑本探针**（约 15 分钟 / checkpoint，不需生成）：看 `同说话人另一条 − 自身完整` 的差距、裁剪区间外 CE、换条件后的 EOS CE 三项是否收敛。再做 16+16 speaker-only 诊断和全量 Seed-TTS（speaker-only 需恢复，与 speaker+ICL 一起），对照 frozen-conditioning / textbase / no-speaker 的 17.5k。
3. **多看一个趋势信号**：WER 和探针差距随 step 的变化。旧实验 17.5k → 47k 的依赖增长 2.5 倍，refmask 若在 32.5k 不再增长即为证据。
4. **评测侧单独统计长输出比例**，验证 EOS 依赖是否随 refmask 减弱。
5. 低成本叠加项（可与 refmask 同 run，也可另开对照）：speaker 条件随机丢弃（10–15%，可学习空向量），换来推理时 CFG 和纯 ICL 能力；以及把 docs 第 9 节的两组文本前端对照排进计划。
6. 跨句参考放到后续阶段；有可靠说话人标签的数据集（libritts_r、aishell3、vctk、genshin、starrail、hifitts 等）的完整 embedding 已全部算好，构建时按 `speaker_id` 配对即可，不需要重跑特征。

## 7. 局限

- 探针是 teacher-forcing CE，不是生成；CE 差距到 WER 的放大倍数未知，但方向和趋势与已有的 16+16 生成诊断及全量评测一致。
- `同说话人另一条` 依赖数据的 `speaker_id`：emilia / emilia_yodas 标签可靠性未核验，galgame 无标签，wenetspeech4tts 无同说话人样本；仅用可靠子集的结论不变。
- 每数据集最多 300 句，小数据集（csemotions 4、ljspeech 13）数字噪声大。
- 导出权重为 FP32，训练为 BF16；探针用 SDPA 而非 FlashAttention-2。数值差异对比较不构成影响。
- 对 refmask 的判断是机制层面的预判，需要训练后的探针与评测确认。
