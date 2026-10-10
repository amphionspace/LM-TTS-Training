# All16 训练与 Seed-TTS 评测

本报告记录 all16 的两种训练方案：**frozen-conditioning 使用冻结的 TTS 文本 embedding / projector；textbase 使用纯文本 Qwen3-0.6B-Base 初始化的可训练 embedding，以及随机初始化的可训练 projector**。两者共用数据划分和音频建模结构，文本前端的初始化、宽度与冻结范围不同。第 1–4 节说明结构和训练设置，第 5–7 节保留 frozen-conditioning 的评测与诊断，第 8 节汇总两种方案的全部已完成 Seed-TTS 结果。

更新于 **2026-10-09**。Frozen-conditioning 的 17,500 / 32,500 / 47,382，以及 textbase 的 17,500 / 25,000，均已完成 speaker + ICL 双 greedy 和双采样评测，共 **10 组**，全部通过审计。最新一组于 **2026-10-09 07:18 UTC** 完成；总表见第 8 节。speaker-only 保持暂停。

## 1. 实验与目录怎么对应

| 实验 | 配置（相对仓库根目录） | 初始组装模型 | 训练作业 |
| --- | --- | --- | --- |
| frozen-conditioning | `configs/supervised-tts-20260929-all16-bf16-32gpu-lr3e-4-bblr1e-4-ep3-wsd.yaml` | `assets/assembled/qwen3-tts-frozen-conditioning` | `pt-blb39rgw`；最终 step 47382 |
| textbase，可训练文本前端 | `configs/supervised-tts-20260929-all16-textbase-trainable-randproj-bf16-32gpu-lr3e-4-bblr1e-4-ep3-wsd.yaml` | `assets/assembled/qwen3-tts-text-base-trainable-random-projector` | `pt-u0ys9osd`；本报告已评测 step 17500 / 25000 |

两种实验 YAML 的 `train.run_name` 与文件名去掉 `.yaml` 一致；输出统一为 `/workspace/LM-TTS-Training-Runs/<run_name>/`。`/workspace/workspace/yanglin/LM-TTS-Training-Runs` 是其软链接。

| Run 内路径 | 用途 |
| --- | --- |
| `metrics.jsonl`、`tensorboard/` | 训练、验证、吞吐和数据等待指标，以及 TensorBoard |
| `checkpoints/` | 可直接恢复的完整分布式训练状态 |
| `archived-checkpoints/` | frozen-conditioning 的独立 checkpoint 归档，避免被轮转删除 |
| `logs/` | 节点日志与保留的验证证据 |
| `submissions/` | 实际提交配置、代码快照和作业身份；复核历史训练优先读这里 |

Frozen-conditioning 的训练目录轮转保留最近两份，另有每小时归档任务，评测读取 `archived-checkpoints/`；textbase 设置 `keep_checkpoints: null`，保留全部完整 checkpoint，评测直接读取该实验的 `checkpoints/`。下载的基础模型在 `assets/base/`，组装模型在 `assets/assembled/`，评测模型和产物在相邻的 `UltraEval-Audio`，三者各有用途。

## 2. 使用了哪些数据

**两种实验复用同一份 build、16 个数据集和固定 train / validation 划分**，textbase 未重新切分数据。训练数据与固定划分：

- Unified 根目录：`/workspace/data/DATA-TTS-UNIFIED`。
- Selection：`tts-selection-supervised-tts-20260929T151539bjt-01`。
- Merged features 发布后缀：`merged-selected-features-v1-20261001T161244bjt-01`。
- 本仓库 build：`data/builds/supervised-tts-20260929-merged-20261001-all16/`。
- 共 **128,220,178** 条，约 **357,506.83 小时**源音频；训练 **128,091,959** 条，验证 **128,219** 条。

下表直接取 build manifest 的有效行数；验证集按数据集抽取约千分之一，seed 42。

| 数据集 | 训练条数 | 验证条数 |
| --- | ---: | ---: |
| aishell3 | 87,925 | 88 |
| csemotions | 4,156 | 4 |
| emilia | 39,661,042 | 39,701 |
| emilia_yodas | 38,470,635 | 38,509 |
| galgame | 6,700,440 | 6,707 |
| genshin_voice | 570,898 | 571 |
| hifitts | 313,852 | 314 |
| hifitts2 | 12,794,364 | 12,807 |
| libriheavy | 12,429,316 | 12,442 |
| libritts_r | 356,366 | 357 |
| ljspeech | 13,087 | 13 |
| mls_sidon | 12,276,410 | 12,289 |
| starrail_voice | 323,571 | 324 |
| vctk | 88,068 | 88 |
| wenetspeech4tts | 3,927,408 | 3,931 |
| wutheringwaves | 74,421 | 74 |
| **合计** | **128,091,959** | **128,219** |

Merged 表已经包含目标 text、每帧 16 个 codec ID、1024 维 speaker embedding，训练侧不再 join，也不在线转写或编码音频。Text 经组装模型配套 tokenizer 转为 token IDs；speaker embedding 来自**同一条目标音频**的冻结 speaker encoder。

`configs/data.yaml` 描述数据选择；`data/builds/` 保存固定 Lance 版本、来源哈希、文本 token、长度索引和 train/validation 行划分，不复制大型 codec / speaker 字段，所以源 features 仍须保留。源配置的 `evaluation: no_holdout` 只表示源表尚未切分，正式训练实际读取已切分的 train / validation build。

划分保证样本行不重叠，**不保证说话人或文本隔离，也未在此证明与外部 Seed-TTS 完全去重**。自参考 speaker embedding 与零样本克隆时另一段参考音频存在分布差异；其是否导致内容捷径仍待验证，不能仅凭冻结 encoder 排除这一问题。

## 3. 模型初始化与冻结

两种方案均为“文本 LM 主干迁移 + 新音频模块”。组装在本地提前完成，seed 42、FP32 保存；训练直接加载 assembled 权重，恢复时加载 checkpoint，不重新组装。

### 文本前端：两种方案的差异

| 模块 | Frozen-conditioning | Textbase |
| --- | --- | --- |
| 文本 tokenizer | 配套 TTS tokenizer，固定 token IDs，无可训练参数 | 与左侧文件和 token IDs 一致，兼容同一训练 build |
| 文本 embedding | **151,936 × 2,048**；来自 `Qwen3-TTS-12Hz-0.6B-Base` 完整表；**冻结** | **151,936 × 1,024**；共享 token 行来自 `Qwen3-0.6B-Base`，新增 TTS token 行随机初始化；**整张表更新**，峰值 LR `1e-4` |
| 文本 projector | **2,048 → 2,048 → 1,024，SiLU**；来自同一 TTS checkpoint；**冻结** | **1,024 → 1,024 → 1,024，SiLU**；**随机初始化并更新**，峰值 LR `3e-4`，保留两层 MLP |

Textbase 的“扩词表”指在文本 Base 的 **151,669 个有效 token** 上加入 **7 个 TTS token**，有效 token 数变为 151,676。原 embedding 已预留 151,936 行，因此此次**没有增加矩阵行数**；共享 token 的 ID 与初始向量保持一致，新增 token 对应行重新随机初始化。所有文本 embedding 行使用同一个 `1e-4` 参数组，新增行不单独采用 `3e-4`。这些是组装时的初始化规则，训练后的 checkpoint 已包含更新后的 embedding 和 projector。

### Talker、音频与 speaker：两种方案共用的结构

以下模块的结构、初始化来源和训练范围一致；各实验独立组装和训练，权重并不共享。

| 模块 | 结构 / 初始化来源 | 是否更新 |
| --- | --- | --- |
| Talker 主干与 norm | 28 层，hidden 1,024，FFN 3,072；`Qwen3-0.6B-Base` 的 layers / norm | **更新**，峰值 LR `1e-4` |
| 首码本 embedding | 3,072 × 1,024，含音频控制 token；随机初始化 | **更新**，`3e-4` |
| 首码本输出 head | 1,024 → 3,072；随机初始化，与 embedding 不共享权重 | **更新**，`3e-4` |
| 15 张 residual embedding | 每张 2,048 × 1,024；随机初始化 | **更新**，`3e-4` |
| Code Predictor 主干 | 5 层 Transformer + norm，hidden 1,024；随机初始化 | **更新**，`3e-4` |
| 15 个 residual 输出 head | 每个 1,024 → 2,048；随机初始化，与 embedding 不共享权重 | **更新**，`3e-4` |
| Talker → Code Predictor projector | 两侧均为 1,024，实际为 `Identity`（直接传递） | 无参数 |
| Speaker encoder | ECAPA-TDNN，128 mel → 1,024；官方 TTS checkpoint | **冻结**，训练读取预计算向量 |
| Speaker → Talker projector | 维度已匹配，没有单独 projector | 无参数 |
| 音频 codec encoder / decoder | `Qwen3-TTS-Tokenizer-12Hz`，24 kHz、16 码本 | **独立冻结**，不进入训练 optimizer |

两种方案都没有加载官方 TTS 的 Talker / 音频 embedding / Code Predictor 权重，也没有加载纯文本 LM 的文本输出 head。Textbase 中随机初始化的是 text projector、新增 TTS token 行和音频模块；Talker 主干及共享文本 embedding 行来自文本 Base，speaker encoder 与 codec 来自预训练 TTS 组件。

| 主模型参数量（不含独立 codec） | Frozen-conditioning | Textbase |
| --- | ---: | ---: |
| 总参数 | 914,643,008 | 754,865,216 |
| 可训练 | 588,329,216 | 746,010,880 |
| 冻结 | 326,313,792 | 8,854,336（仅 speaker encoder） |

Textbase 的文本前端更窄，因此总参数更少；开放文本前端训练后，可训练参数反而更多。结构与初始化核对依据为各自的 `assembly_report.json`，textbase 的组装和恢复验证见 [textbase 实验记录](all16-textbase-20261003.md)。

### 两种方案的输入融合与联合训练

输入路径：文本 IDs 查表并经过各自的 text projector，均输出 1024 维向量；每个音频帧的 16 组 codec IDs 分别查表，所得 16 个向量相加成一个 1024 维向量。它们按“角色与控制前缀 → speaker → 完整文本 → codec BOS → 历史音频帧”排列，经因果 attention 融合。文本位置加 codec PAD，音频位置加 projected text PAD；这是协议占位向量，与 batch 补齐长度的 padding 不同。

这 16 张音频 embedding 都参与训练，residual 表也复用于 Code Predictor 的码本内预测；它们与输出 head **不 tie**。训练用真实历史帧做 teacher forcing，Talker 与 Code Predictor 同时计算 loss、共同反向传播，二者之间不 detach。

## 4. 怎么训练与恢复

两种实验共用以下训练设置；文本前端的峰值 LR 与冻结范围见第 3 节，checkpoint 保留策略见第 1 节。

| 项目 | 设置与含义 |
| --- | --- |
| 资源 / 精度 | 4 节点 × 8 A800 80 GB，FSDP2，BF16 mixed precision，activation checkpointing |
| Dynamic batching | 每卡最多 32,000 codec frames / 50,000 Talker tokens，accumulation 1 |
| Attention | Talker FA2 varlen，无长度 padding、样本间 attention 隔离；Code Predictor 固定短序列使用 SDPA efficient/math |
| Loss | token 归一化的 `first_ce + 0.3 × residual_ce`；后者为 15 个 residual 码本平均 CE |
| Optimizer / LR | AdamW；新增模块 `3e-4`，迁移的可训练模块 `1e-4` |
| 训练长度 / WSD | 3 epochs；前 0.03 epoch warmup，steady 到 2.7，最后 0.3 epoch 余弦衰减到峰值 10% |
| LR 进度依据 | 已消费样本数占计划总量的比例，不依赖预估 dynamic batch 步数 |
| 读取 | 每 rank 4 workers，prefetch factor 2，长度分桶窗口 65,536，pin memory / persistent workers |
| 验证与保存 | 第 100 步，之后每 2,500 步，以及训练结束；验证集全部 128,219 条 |

Textbase 的 `1e-4` 参数组包含 Talker 主干和整张文本 embedding，共 **596,049,920** 参数；`3e-4` 参数组包含随机 text projector 和音频模块，共 **149,960,960** 参数。两种实验分别维护 optimizer、LR 进度和 checkpoint，不能跨方案 resume。

损坏的单条数据跳过并记录；某 rank 整个 microbatch 无效时全体协调跳过，避免通信挂起。配置错误或模型计算异常不会伪装成坏数据吞掉。训练记录包括数据等待时间 / 占比、吞吐和显存；frozen-conditioning 的预算预检时实际显存约 66–67 GiB/卡。临近结束 step 47380 记录约 7.55 秒/步、10,794 音频秒/秒，属于当时单步测量，不代表含验证与保存的全程平均。

Code Predictor 使用 SDPA，是因为当前环境的大 batch FA2 测试曾出现非法访问 / NaN；Talker 仍使用 FA2。完整验证、32 卡恢复逐位一致证据见 [本轮训练记录](all16-20261001.md)。

以下以 frozen-conditioning 配置演示，命令在 **LM-TTS-Training 根目录**运行；运行 textbase 时将 `CFG` 换成第 1 节对应的 YAML，组装模型路径由配置选择。复现新实验先复制 YAML、修改 `train.run_name`，不要对已有正式 run 重复提交：

```bash
PY=/workspace/workspace/yanglin/envs/lm-tts/bin/python
CFG=configs/supervised-tts-20260929-all16-bf16-32gpu-lr3e-4-bblr1e-4-ep3-wsd.yaml

# 首次准备；已有相同 assembled 模型和 build 时直接复用。
"$PY" -m scripts.prepare_models --config "$CFG"
"$PY" -m scripts.build_unified --config "$CFG"

# 预览请求，然后提交一次 4 节点任务。
"$PY" -m scripts.acp.submit --config "$CFG" --nodes 4
"$PY" -m scripts.acp.submit --config "$CFG" --nodes 4 --submit

# 仅在旧任务已停止且确需接续同一 run 时使用。
"$PY" -m scripts.acp.submit --config "$CFG" --nodes 4 --resume latest --submit
```

实验 YAML 直接继承 `configs/base.yaml`；数据绑定在 `configs/data.yaml`。常改项为 run name、模型组装方案、LR、epochs、精度及 batch 预算。ACP 适配仅在 `scripts/acp/`，平台工具 / 密钥在仓库外 `../acp/`；训练核心不依赖平台。环境准备与本地 torchrun 用法见[仓库 README](../../README.md)。

Resume 恢复 optimizer、LR 进度、epoch、next batch、各 rank RNG 和已提交的数据进度。保持模型、数据、卡数、batch 预算和 LR 计划不变；可以调整 workers / prefetch。不要手工按“已读了多少预取数据”推进位置，也不要把其他实验 checkpoint 接到此 run。

Frozen-conditioning 最终 step 47,382 的 teacher-forcing 验证：首码本 CE **0.906703**、residual CE **5.244773**、加权 loss **2.480135**，跳过 / 连带丢弃样本均为 0。这类验证随训练自动执行；下文自由生成的 ASR / SIM 是独立评测流程，loss 降低不等于 WER/CER 必然良好。

## 5. Seed-TTS 方法与历史 Qwen 对比

本次使用本轮最终归档 `step-00047382`，67 个归档文件已核对清单与 SHA-256。导出为 FP32，以 BF16 + SDPA 推理。导出模型在 `UltraEval-Audio/init_model/all16-frozen-conditioning-step-00047382`，没有修改训练 checkpoint。

评测只用 Seed-TTS：英文 **1,088** 条、中文 **2,020** 条。参考音频为 `audio`、参考转写为 `prompt_text`，待生成文本为 `text`。speaker + ICL 同时提供参考 speaker embedding、参考 codec 和参考文本；目标音频不作为生成输入。speaker-only 仅提供 speaker embedding，目前暂停，没有完整成绩。

- 英文：Whisper-large-v3 WER；中文：SeACo-Paraformer CER。
- 相似度：WavLM-large + ECAPA，SIM 为余弦相似度乘 100。
- WER/CER 使用逐句错误率的算术平均，不是全语料累计编辑距离比例。Step 47,382 原始结果保留全量；后续对照采用下述各组独立的 30 秒过滤口径，历史 Qwen 使用其原有异常过滤口径。
- 当前 Seed-TTS 流程没有 DNSMOS / 主观 MOS；SIM 衡量说话人相似度，不能替代音质评分。

| 模型与解码 | 英文 WER/% ↓ | 中文 CER/% ↓ | 英文 SIM×100 ↑ | 中文 SIM×100 ↑ |
| --- | ---: | ---: | ---: | ---: |
| 历史复现 Qwen3-TTS 0.6B，speaker + ICL，采样，剔除异常 | 1.747 | 1.116 | 70.831 | 76.666 |
| 历史复现 Qwen3-TTS 1.7B，speaker + ICL，采样，剔除异常 | 1.743 | 0.958 | 71.285 | 76.958 |
| 本轮 frozen-conditioning step 47382，speaker + ICL，采样 | **17.072** | **14.169** | **64.253** | **76.158** |
| 同一 checkpoint，speaker + ICL，两个模块均 greedy | **13.327** | **9.721** | **63.646** | **74.908** |

历史值取 [2026-09-07 Qwen 复现报告](../../../UltraEval-Audio/replication/voice_clone_20260907.md)的“表 2：统一剔除异常长输出对应样本”。规则是历史任一已完成配置输出 **超过 160 秒**，即在所有配置中同步剔除该样本，不按错误率筛选。英文剔除 5 条，保留 **1,083/1,088**；中文保留 **2,020/2,020**。本轮仍为全量，英文样本集合不同。历史数值沿用报告，本次未重新评分。

**比较限制：**历史 Qwen 使用显式 English/Chinese language ID 与默认 streaming 文本布局；本轮按训练协议使用 `language="Auto"`、`non_streaming_mode=True`，没有显式 language ID。因此这是现有实验成绩对比，不能将全部差距归因于权重或训练数据。当前模型的内容错误率明显更高，中文 SIM 接近并不表示读对了文本。

采样基线：Talker / Code Predictor 均启用 sampling，`top_k=50, top_p=1.0, temperature=0.9`；Talker `repetition_penalty=1.05, max_new_tokens=2048`。Batch size 1，逐条 seed 为 `42 + split 内 index`。Greedy 只关闭两处采样，其余参数保持相同。

### Greedy 与采样：全量对比

两组使用同一 checkpoint、相同 1,088 条英文和 2,020 条中文、相同参考音频与评分器；仅关闭 Talker 和 Code Predictor 两处采样，其余生成参数不变。两组均保留全量，没有过滤异常样本。

| 指标 | 两个模块均采样 | 两个模块均 greedy | Greedy 相对变化 |
| --- | ---: | ---: | --- |
| 英文 WER/% ↓ | 17.072 | **13.327** | 降低 3.744 个百分点，相对降低 21.9% |
| 中文 CER/% ↓ | 14.169 | **9.721** | 降低 4.449 个百分点，相对降低 31.4% |
| 英文 SIM×100 ↑ | **64.253** | 63.646 | 降低 0.606 |
| 中文 SIM×100 ↑ | **76.158** | 74.908 | 降低 1.251 |
| 英文零错误样本 | 451/1,088 | **611/1,088** | 增加 160 条 |
| 中文零错误样本 | 588/2,020 | **931/2,020** | 增加 343 条 |
| 输出超过 30 秒（英 / 中） | **0 / 0** | 1 / 4 | 出现少量长输出 |

**结论：greedy 在内容准确率上更好，但并非所有指标都更好。** 英文和中文错误率均下降，SIM 略降；没有 MOS 或完整主观试听结果，不能据此认定整体音质更好。Greedy 最长英文 70 秒、中文 105.04 秒；其中 1 条英文触及现有 Whisper 评分器的 30 秒截断，因此该条 WER 未衡量后半段内容，长输出风险需单独看待。

### Step 32,500 对照（greedy 与采样均已完成）

使用同一训练 run 的完整归档 `step-00032500`，67 个文件已通过 SHA-256 校验。导出模型为 `UltraEval-Audio/init_model/all16-frozen-conditioning-step-00032500`；模型配置、文本 tokenizer 与 step 47,382 一致。

按 **greedy → 采样** 顺序评测，两组均为 speaker + ICL、3,108 条输入、8 卡、每推理进程 16 GiB 上限。Greedy 完成生成、ASR/SIM 和审计后才启动采样；任一阶段失败即停止接续。speaker-only 不运行。

**2026-10-08 更新：各组独立排除输出超过 30 秒的样本，不参与 WER/CER 和 SIM，不强制使用相同排除清单。** 对已经生成的长音频只排除评分，原音频保留；后续生成在略超 30 秒时结束并排除，不把截短输出当成正常样本。命令使用 `--max-audio-seconds 30`，采样参数不变，生成上限从 2,048 调整为 378 个 token（按 codec 帧率折算并留出超过阈值的余量）。

`full/summary.json` 分别记录输入、已生成、应评分、已评分和排除数量；`full/audit.json` 保存排除 ID、时长与原因。每组的保留数可能不同，报告按实际分母展示；上文 step 47,382 表格仍为原始全量结果，不能将过滤后的改善全部归因于模型差异。

以下路径相对 UltraEval-Audio：

| 内容 | 路径 |
| --- | --- |
| Greedy 结果 | `res/all16-step32500-seedtts-greedy-20261008-8gpu` |
| 随后采样的结果 | `res/all16-step32500-seedtts-sampling-20261008-8gpu` |
| 顺序执行日志 | `log/seed-tts-step32500-sequence.log` |

**Greedy 于 2026-10-08 10:13 UTC 完成并通过审计**：3,108 条均已生成，英文排除 4 条、中文排除 28 条超过 30 秒的输出；保留的 3,076 条均完成 ASR 和 SIM，无未恢复的生成或评分错误。采样于同日 12:22 UTC 完成并通过审计，3,108 条全部评分、无超过 30 秒的输出；完整指标见第 8 节。

下表对两个 checkpoint 分别应用 **输出超过 30 秒即排除** 的规则；step 47,382 从既有逐条评分重新汇总，不重新生成，不覆盖上文原始全量成绩。

| Checkpoint，均为 speaker + ICL 双 greedy | 英文保留 / 排除 | 英文 WER/% ↓ | 英文 SIM×100 ↑ | 中文保留 / 排除 | 中文 CER/% ↓ | 中文 SIM×100 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Step 47,382 | 1,087 / 1 | 13.255 | **63.665** | 2,016 / 4 | 9.523 | **74.943** |
| Step 32,500 | 1,084 / 4 | **4.803** | 62.203 | 1,992 / 28 | **5.024** | 71.525 |

Step 32,500 的英文 WER 和中文 CER 分别降低 **8.452、4.499 个百分点**，但 SIM 分别降低 **1.462、3.418**，且异常长输出更多。它在保留样本上的内容准确率更好，不能据此认定整体克隆质量更好，也不能仅凭这两组结果确定退化原因。各组保留样本不同；排除的长输出仍须计入失败现象，不能视为模型已经解决这些样本。历史 Qwen 的过滤阈值为 160 秒，与此表也不同。

结果见 [32,500 greedy 汇总](../../../UltraEval-Audio/res/all16-step32500-seedtts-greedy-20261008-8gpu/full/summary.json)和[审计](../../../UltraEval-Audio/res/all16-step32500-seedtts-greedy-20261008-8gpu/full/audit.json)。后台会话为 `seedtts-all16-s32500-sequence`；顺序启动命令保存在 greedy 目录的 `run-sequence.sh`，阶段见 `sequence-stage.txt`。两组完整成绩均已纳入第 8 节。

完整结果见 [采样汇总](../../../UltraEval-Audio/res/all16-step47382-seedtts-20261008-8gpu/full/summary.json)、[greedy 汇总](../../../UltraEval-Audio/res/all16-step47382-seedtts-greedy-20261008-8gpu/full/summary.json)和 [greedy 审计](../../../UltraEval-Audio/res/all16-step47382-seedtts-greedy-20261008-8gpu/full/audit.json)。早期小样本诊断保存在 [评测证据](evidence/seed-tts-step47382.json)，正式结论以上述全量结果为准。

### 一个 epoch 附近的对照：step 17,500

现有完整归档没有 step 15,000 / 16,000；最接近 16,000 的是 **step 17,500**。其 checkpoint 记录已完成 1 个 epoch，再消费 13,686,509 / 128,091,959 条训练样本，约 **1.107 epoch**。

已完成的执行顺序：**32,500 greedy → 32,500 采样 → 17,500 greedy → 17,500 采样**。每组必须完成全量评分并通过审计，才会开始下一组；前序失败则停止接续。17,500 沿用相同 speaker + ICL、数据、采样参数、8 卡和每进程 16 GiB 上限，以及上述各组独立的 30 秒过滤规则，开始前自动核验归档并导出模型。

结果分别写入 UltraEval-Audio 的 `res/all16-step17500-seedtts-greedy-20261008-8gpu` 和 `res/all16-step17500-seedtts-sampling-20261008-8gpu`。Greedy 和采样分别于 2026-10-08 14:39、16:20 UTC 完成并通过审计，指标见第 8 节。调度日志为 `log/seed-tts-step17500-sequence.log`，greedy 目录的 `sequence-stage.txt` 已为 `complete`。

## 6. 当前训练存在的问题：speaker 条件的训推不一致

**训练与验证使用目标音频自身的 speaker embedding，实际克隆推理使用另一条参考音频的 embedding。** 数据中的 codec 目标和 speaker 向量来自同一段录音；推理时目标录音尚不存在，只能从参考录音提取 speaker 条件。两者的条件来源不同，即使属于同一说话人，也不能假设向量完全等价。

冻结 speaker encoder 只保证提取器的参数不更新，不保证其向量只包含说话人身份。向量仍可能携带内容、韵律、语速或录音条件；可训练的 Talker 可能学会依赖这些与目标有关的信息。换成另一句话的参考向量后，这种依赖可能造成错读。这是当前需要验证的训练捷径风险，不能简单归因于推理采样设置。

### 已有证据

在最终 checkpoint 上固定中英文各 16 条目标，仅替换 speaker 条件，使用 speaker-only、双 greedy，并将 speaker 提取精度和预处理对齐训练 pipeline：

| Speaker 条件 | 英文 WER/% ↓ | 中文 CER/% ↓ |
| --- | ---: | ---: |
| 目标录音自身的 embedding | 0.000 | 1.276 |
| 配对说话人的另一条参考录音 | 75.741 | 48.083 |
| 排除当前目标后的两条参考 embedding 均值 | 64.105 | 22.595 |

跨句参考相较自身条件，英文 16/16、中文 13/16 条错误率变差，其余持平。这支持模型对目标自身条件存在依赖的担忧。**自身条件使用了真实目标录音，属于诊断，不能作为零样本克隆成绩。** 同说话人关系来自 Seed-TTS 的原始配对，没有额外独立身份核验；样本量较小，均值组还改变了向量幅度，因此尚不能证明具体泄漏了词汇内容，也不能将全部生成错误归因于此。

完整设置、转写和试听见 [speaker 条件配对诊断](../../../UltraEval-Audio/res/seedtts-speaker-conditioning-20261008/RESULTS.md)。这组测试没有 ICL，参数也与全量评测不同，分数不与上节全量成绩直接横比。

### 对当前结果和后续训练的影响

训练内 holdout 也使用目标自身的 embedding，所以 teacher-forcing 验证 loss 不能直接衡量跨句参考时的生成表现。双 greedy 降低了当前全量 WER/CER，但没有改变 speaker 条件来源，不能视为修复了训推不一致。

后续优先验证：训练时为目标配对**同说话人的另一条录音**作为 speaker 参考，严格排除目标自身；验证时固定跨句参考，同时检查 speaker-only 与 speaker + ICL 的内容准确率、SIM 和长输出。需要先核对数据的说话人身份与可用参考，不能随意换成不同说话人的音频。以上是待验证的改进方向，本报告不代表已经修改训练数据、模型或正在运行的任务。

## 7. 导出、运行 greedy 和查看结果

模型转换入口保留在本仓库。以下命令输出目录必须尚不存在；本次模型已经导出，不必重跑：

```bash
PY=/workspace/workspace/yanglin/envs/lm-tts/bin/python
RUN=/workspace/workspace/yanglin/LM-TTS-Training-Runs/supervised-tts-20260929-all16-bf16-32gpu-lr3e-4-bblr1e-4-ep3-wsd
UE=/workspace/workspace/yanglin/UltraEval-Audio
"$PY" -m scripts.export_checkpoint \
  --checkpoint "$RUN/archived-checkpoints/step-00047382" \
  --assembled-model assets/assembled/qwen3-tts-frozen-conditioning \
  --output "$UE/init_model/all16-frozen-conditioning-step-00047382" \
  --copy-tokenizer
```

导出 `model.safetensors` SHA-256：`1e942af5c2c5ec2d1a26bcaf78d6e231e965c313aa0fa2a4191172f0dd2878c7`。

以下路径相对 **UltraEval-Audio**：

| 路径 | 内容 |
| --- | --- |
| `raw_data/TwinkStart/Seed-TTS-Eval`、`raw_data/seed_tts_manifests` | HF 数据快照、固定输入清单与参考 WAV |
| `init_model/` | 本轮导出模型、Whisper、Paraformer、WavLM / ECAPA；模型与数据通过 HF Mirror 无代理下载 |
| `envs/tts`、`envs/metrics` | 独立推理 / 评分 Conda 环境 |
| `res/all16-step47382-seedtts-20261008-8gpu` | 已完成的采样 ICL；保留暂停的 speaker-only 记录 |
| `res/all16-step47382-seedtts-greedy-20261008-8gpu` | 已完成的全量 greedy ICL |
| `log/seed-tts-icl-greedy.log` | Greedy 调度日志 |

数据 HF revision 为 `185352e5da255d788097b835ea7dd11ae4f74976`。下载版本、文件哈希与环境版本记录保留在 UltraEval。中文评分使用 JunHowie 的完整 FP32 SeACo-Paraformer 镜像；已核对模型加载完整，但未与原始 ModelScope 发布文件逐字节比较。

本次使用 8 个 GPU 分片，每进程 16 GiB 显存上限。运行命令如下；已完成的结果无需重跑，中断时可用同一命令接续：

```bash
cd /workspace/workspace/yanglin/UltraEval-Audio
envs/tts/bin/python -u scripts/run_seed_tts.py \
  --model-path init_model/all16-frozen-conditioning-step-00047382 \
  --run-dir res/all16-step47382-seedtts-greedy-20261008-8gpu \
  --gpus 7,6,5,4,3,2,1,0 --gpu-memory-gib 16 \
  --modes speaker_icl --greedy
```

`--greedy` 同时设置 `do_sample=False`、`subtalker_dosample=False`，仍保留 repetition penalty 1.05。不要去掉 `--modes speaker_icl`，否则默认还会运行已暂停的 speaker-only。解码设置写入运行身份，不能在原采样目录中切换 greedy；恢复会跳过已成功样本，目录锁阻止重复调度。

流程为 **小样本生成 / 评分 / 审计 → 全量生成 → ASR 与 SIM → 完整性审计**，无需手工再启动 ASR。查看：

- `pipeline_status.json`：当前阶段、模式与模型身份。
- `full/results_table.md`、`full/summary.json`：最终汇总；中间阶段可能尚不存在。
- `full/audit.json`：必须 `passed: true` 且覆盖全部 3,108 条；同时 pipeline 为 `complete` 才算所选模式完成。
- `full/speaker_icl/`：逐条生成音频、转写、WER/CER、SIM、分片参数。

本节的 step 47,382 两组均为 `complete`，审计通过；各 checkpoint 的过滤后对比见第 8 节。

## 8. 全部已完成的 Seed-TTS 结果（2026-10-09）

### 口径与实验身份

以下 **10 组均为 speaker + ICL**，每组原始输入英文 1,088 条、中文 2,020 条；全部完成生成、应保留样本的 ASR / SIM 和审计，无未恢复的生成或评分错误。WER/CER 是逐句错误率的算术平均；SIM 为 WavLM-large + ECAPA 余弦相似度 ×100，不是音质 MOS。

- **Frozen-conditioning**：第 3 节所述的冻结 TTS 文本 embedding / projector；评测 17,500（约 1.107 epoch）、32,500 和最终 47,382（3 epochs）。
- **Textbase**：文本 embedding 从 Qwen3-0.6B-Base 初始化并扩词表，维度为 151,936 × 1,024；随机初始化的 `1024 → 1024 → 1024` SiLU projector。文本 embedding 和 projector **均参与训练**，speaker encoder 仍冻结；评测 17,500（约 1.107 epoch）和 25,000（约 1.587 epoch）。其训练设置见 [textbase 实验记录](all16-textbase-20261003.md)。

总表统一按**各组独立排除输出超过 30 秒**汇总，不强制相同排除清单，也不按错误率筛样本。47,382 的两组由已保存的逐条评分重算该过滤视图，不重新生成或覆盖原始汇总；第 5 节仍保留其原始全量成绩。其余各组直接取已过滤的 `full/summary.json`。各组保留样本不同，长输出仍属于需报告的异常，过滤后的分数不能单独代表全体输入表现。

新运行的生成上限为 378 token；47,382 以及 32,500 greedy 在切换规则前已生成的音频使用原 2,048 token 上限。因此总表统一的是评分过滤规则，并非所有历史生成都使用了同一上限。其余协议沿用 Auto、non-streaming、相同参考输入和评分器。历史 Qwen 使用超过 160 秒的统一排除清单及不同 language / streaming 设置，其已过滤成绩保留在第 5 节，不混入这张 30 秒口径的总表。

### 指标总表

“保留/排除”分别列出实际评分数和超过 30 秒的输出数。Greedy 与采样均指 Talker 和 Code Predictor 两个模块同时关闭或开启采样。

| 实验 | Step | 解码 | 英文保留/排除 | WER/% ↓ | 英文 SIM ↑ | 中文保留/排除 | CER/% ↓ | 中文 SIM ↑ |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen-conditioning | 17,500 | Greedy | 1,088/0 | 3.119 | 56.481 | 2,000/20 | 8.386 | 65.474 |
| Frozen-conditioning | 17,500 | 采样 | 1,088/0 | 5.713 | 56.772 | 2,020/0 | 11.350 | 67.166 |
| Frozen-conditioning | 32,500 | Greedy | 1,084/4 | 4.803 | 62.203 | 1,992/28 | 5.024 | 71.525 |
| Frozen-conditioning | 32,500 | 采样 | 1,088/0 | 8.181 | 62.168 | 2,020/0 | 9.218 | 71.915 |
| Frozen-conditioning | 47,382 | Greedy | 1,087/1 | 13.255 | 63.665 | 2,016/4 | 9.523 | 74.943 |
| Frozen-conditioning | 47,382 | 采样 | 1,088/0 | 17.072 | 64.253 | 2,020/0 | 14.169 | 76.158 |
| Textbase | 17,500 | Greedy | 1,087/1 | 3.609 | 45.080 | 1,979/41 | 10.084 | 54.164 |
| Textbase | 17,500 | 采样 | 1,088/0 | 5.153 | 47.100 | 2,020/0 | 12.864 | 57.656 |
| Textbase | 25,000 | Greedy | 1,087/1 | 3.590 | 51.639 | 2,015/5 | 5.249 | 62.483 |
| Textbase | 25,000 | 采样 | 1,088/0 | 4.383 | 53.101 | 2,020/0 | 6.166 | 64.852 |

### 结果说明

1. **两个实验、所有已测 checkpoint 的 greedy 错误率都低于对应采样。** 但 greedy 的长输出更多；textbase 17,500 排除 42 条、25,000 排除 6 条，对应采样均未排除。不能仅凭保留样本的 WER/CER 认为 greedy 在所有方面更好。
2. **Textbase 从 17,500 到 25,000 有改善。** Greedy 英文 WER 基本持平（3.609% → 3.590%），中文 CER 从 10.084% 降到 5.249%，中英文 SIM 均提高；采样的两种错误率也下降。Textbase 25,000 greedy 内容准确率优于本 checkpoint 的采样，但采样的 SIM 更高且没有超过 30 秒的输出。
3. **同为 17,500，textbase 尚未显示全面优势。** Greedy 的中英文错误率均高于 frozen-conditioning，SIM 也更低；采样只有英文 WER 更低，中文 CER 更高，SIM 仍更低。Textbase 25,000 的英文较好也不能直接与另一实验 32,500 或 47,382 等同训练进度比较。
4. **Frozen-conditioning 训练更久并未带来持续降低的生成错误率。** 已测 greedy 中，17,500 英文 WER 最低，32,500 中文 CER 最低；47,382 的 SIM 较高，但内容错误率较差。这些结果不能单独证明过拟合或确定原因。Textbase 同样使用目标自身 speaker embedding 训练，改变文本前端并未消除第 6 节的训推差异；该节配对诊断只测试了 frozen-conditioning 47,382，不能把诊断数值直接套给 textbase。

### 完成时间与原始证据

下表时间为 UTC。每个链接指向该组的原始汇总和审计；逐条转写、评分及生成 WAV 位于同目录下的 `speaker_icl/`。47,382 链接仍为原始未过滤汇总，总表的 30 秒视图从该目录逐条记录计算。

| 实验 / Step / 解码 | 完成时间（UTC） | 结果证据 |
| --- | --- | --- |
| Frozen-conditioning / 17,500 / Greedy | 2026-10-08 14:39:57 | [汇总](../../../UltraEval-Audio/res/all16-step17500-seedtts-greedy-20261008-8gpu/full/summary.json) · [审计](../../../UltraEval-Audio/res/all16-step17500-seedtts-greedy-20261008-8gpu/full/audit.json) |
| Frozen-conditioning / 17,500 / 采样 | 2026-10-08 16:20:04 | [汇总](../../../UltraEval-Audio/res/all16-step17500-seedtts-sampling-20261008-8gpu/full/summary.json) · [审计](../../../UltraEval-Audio/res/all16-step17500-seedtts-sampling-20261008-8gpu/full/audit.json) |
| Frozen-conditioning / 32,500 / Greedy | 2026-10-08 10:13:29 | [汇总](../../../UltraEval-Audio/res/all16-step32500-seedtts-greedy-20261008-8gpu/full/summary.json) · [审计](../../../UltraEval-Audio/res/all16-step32500-seedtts-greedy-20261008-8gpu/full/audit.json) |
| Frozen-conditioning / 32,500 / 采样 | 2026-10-08 12:22:22 | [汇总](../../../UltraEval-Audio/res/all16-step32500-seedtts-sampling-20261008-8gpu/full/summary.json) · [审计](../../../UltraEval-Audio/res/all16-step32500-seedtts-sampling-20261008-8gpu/full/audit.json) |
| Frozen-conditioning / 47,382 / Greedy | 2026-10-08 07:22:50 | [汇总](../../../UltraEval-Audio/res/all16-step47382-seedtts-greedy-20261008-8gpu/full/summary.json) · [审计](../../../UltraEval-Audio/res/all16-step47382-seedtts-greedy-20261008-8gpu/full/audit.json) |
| Frozen-conditioning / 47,382 / 采样 | 2026-10-08 05:02:52 | [汇总](../../../UltraEval-Audio/res/all16-step47382-seedtts-20261008-8gpu/full/summary.json) · [审计](../../../UltraEval-Audio/res/all16-step47382-seedtts-20261008-8gpu/full/audit.json) |
| Textbase / 17,500 / Greedy | 2026-10-09 06:11:28 | [汇总](../../../UltraEval-Audio/res/all16-textbase-step17500-seedtts-greedy-20261009-8gpu/full/summary.json) · [审计](../../../UltraEval-Audio/res/all16-textbase-step17500-seedtts-greedy-20261009-8gpu/full/audit.json) |
| Textbase / 17,500 / 采样 | 2026-10-09 07:18:23 | [汇总](../../../UltraEval-Audio/res/all16-textbase-step17500-seedtts-sampling-20261009-8gpu/full/summary.json) · [审计](../../../UltraEval-Audio/res/all16-textbase-step17500-seedtts-sampling-20261009-8gpu/full/audit.json) |
| Textbase / 25,000 / Greedy | 2026-10-09 03:56:09 | [汇总](../../../UltraEval-Audio/res/all16-textbase-step25000-seedtts-greedy-20261009-8gpu/full/summary.json) · [审计](../../../UltraEval-Audio/res/all16-textbase-step25000-seedtts-greedy-20261009-8gpu/full/audit.json) |
| Textbase / 25,000 / 采样 | 2026-10-09 05:00:28 | [汇总](../../../UltraEval-Audio/res/all16-textbase-step25000-seedtts-sampling-20261009-8gpu/full/summary.json) · [审计](../../../UltraEval-Audio/res/all16-textbase-step25000-seedtts-sampling-20261009-8gpu/full/audit.json) |

## 9. 文本维度与 projector 差异：当前结论和待验证问题

**Textbase 的 1024 维与 frozen-conditioning 的 2048 维差异是真实的结构差异，不是文档笔误或维度接错。** 已直接核对本地基础模型、组装模型及 textbase step 25,000 导出权重；两种方案经过 text projector 后都输出 1024 维，与 Talker 匹配。

| 项目 | Frozen-conditioning | Textbase |
| --- | --- | --- |
| Text embedding | TTS 0.6B：151,936 × **2,048**，冻结 | 纯文本 Qwen3 0.6B Base：151,936 × **1,024**，扩充有效 token 后参与训练 |
| Text projector | **2,048 → 2,048 → 1,024**，继承 TTS 预训练权重，冻结 | **1,024 → 1,024 → 1,024**，随机初始化，参与训练 |
| Projector 参数量 | 6,294,528 | 2,099,200 |
| Talker → Code Predictor projector | **Identity**，1024 维直接传递 | **Identity**，1024 维直接传递 |

两边的 **text projector 都是带 bias 的两层 Linear，中间 SiLU，没有残差连接或 LayerNorm**，计算为 `Linear₂(SiLU(Linear₁(x)))`。Textbase 的输入输出同宽，不代表它是 Identity。此前提到的 Identity 位于 **Talker → Code Predictor**；代码支持 text projector 的 Identity / near-identity 选项，但本次 `textbase-randproj` 训练没有使用它们，见[原组装设计](../design/text-base-trainable-assembly.md)。

### 为什么可能影响结果

Textbase 同时改变了**文本表示维度、初始化来源、projector 初始化和文本前端是否冻结**，并非只比较“冻结或解冻”。更窄的 embedding 和 projector 减少了容量；替换前端也失去了原 TTS embedding 与 projector 共同训练形成的适配。

另一个值得验证的因素是随机 projector：textbase 的共享 embedding 行与 Talker layers 原本来自同一个纯文本 Base，中间插入无残差的随机非线性映射，会改变进入预训练 layers 的向量方向和尺度，可能增加重新适配的难度。Frozen-conditioning 继承的是一对预训练 TTS embedding / projector，但其 Talker layers 同样换成了纯文本 Base，因此也不能声称整个输入到主干的链路已经预训练对齐。

**这些是可能原因，现有结果无法单独证明维度缩小或随机 projector 导致了性能差距。** 同为 17,500 时 textbase 的 SIM 明显更低，但采样英文 WER 更好，并非所有指标都退化。第 6 节的 speaker 条件训推差异也仍然存在，不能用文本维度解释全部问题。

### 后续可做的独立对照

- **检验随机映射的影响**：保留 textbase 的 embedding、数据和训练设置，只将 text projector 换为 Identity，或保留 MLP 结构并改用 near-identity 初始化。前者同时去掉了 projector 参数，后者更适合对照初始化方式；两者均不能直接证明维度效应。
- **检验开放文本前端训练的影响**：保留原 TTS 的 2048 维 embedding 和预训练 projector，仅开放这两者训练，并明确记录文本参数组的 LR。

若要隔离维度本身，还需要控制初始化来源、训练范围和映射方式；直接给 1024 维补零或增加升维层，不等于恢复了原 TTS 的 2048 维表示。以上仅记录后续对照建议，本次未修改现有模型、训练或评测结果。

## 10. No-speaker step 10,000：纯 ICL 评测（2026-10-10，全量已暂停）

本次按指定 checkpoint 评测：`supervised-tts-20260929-all16-no-spk-bf16-16gpu-acc2-lr3e-4-bblr1e-4-ep3-wsd/checkpoints/step-00010000`。它从 **frozen-conditioning assembled 模型重新训练**，不是 textbase，也不是从旧训练 checkpoint 接续。配置为 `model.use_speaker_embedding: false`；训练时删除 speaker 位置，保留冻结的 2048 维 TTS 文本 embedding 与预训练 projector。资源为 16 卡、梯度累积 2；该 checkpoint 已消费 81,422,431 条样本，约 **0.636 epoch**。训练设置见 [no-speaker 实验记录](all16-no-speaker-20261008.md)。

导出模型在 `UltraEval-Audio/init_model/all16-no-spk-step-00010000`，权重 SHA256 为 `f08e5390744df3f72930c1fca6ac6ba1bc31c8dc9a05df4182b7d1a48a56e255`。已检查完整标记、组装身份并完成分布式权重加载；导出配置和元数据均明确禁用 speaker embedding。

### 推理条件与批量设置

- 模式为 **`icl_only`**：参考音频 codec + 参考文本 + 目标文本；跳过 speaker encoder 计算，完全省略 speaker 向量和对应位置，不插入零向量。原生 Qwen wrapper 不识别训练侧的无 speaker 标记，评测适配显式使用其无 speaker 前缀分支，并检查每次生成的 ICL 调用数量。无 speaker 导出若误选 speaker-only 或 speaker + ICL，会被拒绝。
- 沿用 Seed-TTS 英文 1,088 条、中文 2,020 条，以及 Auto、non-streaming、BF16 + SDPA、相同 ASR / SIM 评分器。顺序为双 greedy → 双采样，各组独立排除超过 30 秒输出；生成上限 378 token。
- **正式评测使用 batch size 1，8 卡独立分片生成**，GPU 为 **0–7**。Batch 4 曾尝试将参考编码和输出解码固定为单条、仅批量计算 Talker / Code Predictor，但完整生成出现新增长输出，未采用；具体检查见下文。每个生成进程的 PyTorch 显存分配上限为 **4 GiB**，启动时从可用显存中留出 0.25 GiB 后计算有效上限，不足 4 GiB 则拒绝启动；CUDA 上下文等开销不计入分配器上限。此前 batch 1 小样本分配 / 缓存峰值约为 2.85 / 3.14 GiB。**ASR / SIM 评分单独使用 GPU 2、3、6、7**，避免在余量较小的 0、1 卡加载评分模型。共享 GPU 仍需观察 OOM 和运行速度。
- 正式 batch 1 每条 seed 为 `42 + split 内 index`。批量诊断中，每个分片按原始清单固定组批，seed 为 `42 + 该批首条样本的 index`。恢复时保持原分组；若一批只写入部分结果，则重新生成同一批，只补写缺失记录。Batch size 和分片数必须保持一致。**采样的随机数分配与此前 batch 1 不同**，不属于逐条随机性完全对齐的对照；BF16 批量计算也不保证 greedy 输出逐位一致。

结果路径相对 UltraEval-Audio：

| 内容 | 路径 |
| --- | --- |
| Greedy | `res/all16-no-spk-step10000-seedtts-greedy-bs1-20261010-8gpu` |
| 采样 | `res/all16-no-spk-step10000-seedtts-sampling-bs1-20261010-8gpu` |
| Batch 对照日志 | `log/seed-tts-no-spk-step10000-preflight.log` |
| 正式顺序日志 | `log/seed-tts-no-spk-step10000-bs1-8gpu-sequence.log` |

### 批量推理正确性检查与修正

最初仅通过了路径、ASR 和完整性预检，不能据此认定 batch 推理与单条等价；全量启动后即暂停，尚未写出正式样本。进一步对照发现：**原生 wrapper 将参考音频批量编码时，四条样本各有 101 个 codec ID 与单条编码不同**（每条共 784 或 960 个 ID）。固定同一份 codec 后，批量音频解码也出现约 0.0156 的最大波形差异。因此旧 batch 4 的“199.66 秒 → 100.44 秒”对比改变了参考条件，不能作为修正后路径的性能结论。该次诊断不参与正式汇总。

修正为参考编码和输出解码逐条执行，并对批量运行的这一设置加入身份记录及审计。固定参考 codec 后，对中英文各 2 条、不同输入长度进行单条与 batch 4 对照：

| 检查项 | 结果 |
| --- | --- |
| 去除 padding 后的输入 embedding | 完全一致 |
| Prefill 与缓存续推的位置编号 | 完全一致 |
| FP32 prefill logits 最大绝对差 | 1.72e-5 |
| FP32 residual Code Predictor logits 最大绝对差 | 9.06e-6 |
| FP32 固定历史下缓存续推 logits 最大绝对差 | 1.19e-5 |
| 上述检查中的 argmax | BF16 / FP32 均一致 |
| Speaker encoder 调用数 | 0 |

FP32 对照使用同一份 BF16 舍入后的权重转为 FP32 计算，用于隔离批量计算的数值影响，不是另一次 FP32 音质评测。BF16 下这三处 logits 的最大差异约为 0.125 / 0.128 / 0.094，因此不能承诺整段自回归输出逐位一致。这里只验证了四条样本的 prefill、residual heads 和一次固定历史的缓存续推，不代表覆盖所有长度与生成步骤。22 项回归检查还覆盖了误用 speaker 模式、同批 seed、尾批、ICL 调用、逐条 codec 处理及异常后的 decoder 恢复。

两个 batch 4 诊断目录已按要求删除；本节保留当时的检查结果，诊断输出不参与正式汇总。

**局部正确性检查不等于整段生成等价。** 修正 codec 后的完整四条预检中，英文 index 0 在 batch 1 输出 **3.68 秒**，batch 4 却输出 **30.16 秒**并被排除；index 1 分别为 4.72 / 4.56 秒，两条中文均为 30.16 秒。固定历史的 FP32 对照未发现位置或缓存逻辑错误，但不能据此排除整段自回归路径的问题，也不能把新出现的长输出简单归为无关紧要的舍入差异。

因此本轮正式评测使用 **batch 1、GPU 0–7 独立分片并行**，保持与旧结果一致的逐样本 seed。Batch 4 两次预检暂停时均未写出正式全量样本，相关目录已清理。加速方案需要进一步验证整体生成稳定性后再采用。

运行命令保存在上表 `bs1` greedy 目录的 `run-sequence.sh`，后台会话为 `seedtts-all16-no-spk-s10000-8gpu-resume`。8 卡流程使用新的结果目录，保留分片身份，旧 3 卡预检不混入正式汇总。2026-10-10 03:21 UTC 已通过 4 GiB 上限下的四条小样本生成、评分及审计，并启动全量 greedy 的 8 个分片；原计划完成全量审计后接采样，现已取消自动接续并暂停全量流程。此次小样本缓存显存峰值约 3.15 GiB。生成与评分各自的分片数在恢复时必须固定，可以更换对应的物理 GPU。尚无全量成绩，读取各组 `full/summary.json` 与 `full/audit.json`。该实验的纯 ICL 条件和训练进度与第 8 节旧结果不同，应分别报告。


### 大量输出达到上限，暂停全量评测

2026-10-10 06:17 UTC 按要求停止所有本轮全量生成进程，并取消后续全量采样。已有结果保留，未完成全量评分，不应作为正式 benchmark 结果。

| 已完成的 greedy 输出 | 数量 | 达到 30.16 秒上限 | 占比 |
| --- | ---: | ---: | ---: |
| 英文 | 1,086 | 768 | 70.72% |
| 中文 | 85 | 79 | 92.94% |

30.16 秒对应本轮 378 token 的生成上限附近，大量输出集中到这一长度，说明存在明显的生成终止异常；现有证据尚不能区分模型、输入协议或解码策略的原因。之前的小样本审计检查了文件、输入身份、条件和覆盖率，**不代表生成质量正常**。这些达到上限的样本必须保留在诊断统计中，不能仅剔除后报告剩余样本的评分。

按要求只取 Seed-TTS 中英文各前 5 条做双采样试验，batch size 1；Talker / Code Predictor 均为 `do_sample=true`、`top_k=50`、`top_p=1.0`、`temperature=0.9`，其他条件与 greedy 相同。已有 9 条同样本 greedy 输出可配对，中文 index 4 尚无 greedy 输出；不补跑全量。试验目录为 `UltraEval-Audio/res/all16-no-spk-step10000-seedtts-sampling10-bs1-20261010`，音频在 `output/icl_only/`，逐条时长对照保存在 `comparison.json`。此试验只诊断生成长度，不是 ASR / SIM 评分。

采样试验于 06:23 UTC 完成。10 条生成记录均成功，实际设置已检查为双采样；同样本的文本、参考音频 hash、seed 和无 speaker 条件与 greedy 一致。结果如下：

| 语言 / Index | Greedy 时长（秒） | 双采样时长（秒） |
| --- | ---: | ---: |
| 英文 / 0 | 3.68 | 1.76 |
| 英文 / 1 | 4.72 | 2.64 |
| 英文 / 2 | 30.16 | 8.56 |
| 英文 / 3 | 30.16 | 3.60 |
| 英文 / 4 | 30.16 | 5.36 |
| 中文 / 0 | 30.16 | 9.44 |
| 中文 / 1 | 30.16 | 15.12 |
| 中文 / 2 | 30.16 | 0.32 |
| 中文 / 3 | 30.16 | 4.64 |
| 中文 / 4 | 未生成 | 10.96 |

**双采样 0/10 达到上限；已有 greedy 配对样本是 7/9 达到上限。** 这说明这组样本的生成终止对解码策略敏感，但不代表内容正确。中文 index 2 的目标文本有 23 个字符，采样只生成 **0.32 秒**，明显不足以完整读出；仍存在提前结束等问题。本次未计算 ASR / SIM，不能把时长改善当作可懂度或音质改善。只完成这 10 条诊断，相关生成进程均已退出，全量 greedy 与全量采样保持暂停。
