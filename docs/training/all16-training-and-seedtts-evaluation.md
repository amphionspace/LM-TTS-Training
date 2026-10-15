# All16 训练与 Seed-TTS 评测

本报告记录 all16 的三种训练方案：**frozen-conditioning** 使用冻结的 TTS 文本 embedding / projector；**textbase** 使用纯文本 Qwen3-0.6B-Base 初始化的可训练 embedding，以及随机初始化的可训练 projector；**no-speaker** 保留 frozen-conditioning 的文本前端与音频结构，移除 speaker embedding 条件后独立训练。三者共用同一数据划分。第 1–4 节说明共同设置与主要差异，第 5–9 节保留前两种方案的评测和诊断，第 10 节记录 no-speaker step 10,000，**第 11 节记录 no-speaker step 17,500 的结构、设置和完整结果，第 12 节记录同一实验 step 25,000 的接续评测**。

更新于 **2026-10-14**。Frozen-conditioning 与 textbase 的已完成历史评测见第 8 节。No-speaker step 17,500 的全量双采样与双 greedy 均已完成，结果见第 11 节；No-speaker step 25,000 的双采样结果见第 12 节，双 greedy 已完成、结果待补录。**Emilia2 refmask step 10,000 的 speaker + ICL 双采样已完成，双 greedy 按要求停止**，结果见第 13 节。**LFM230M step 5,000 的全量双采样也已完成并通过审计**，结构、结果和对比见第 14 节。**同一 Qwen Emilia2 refmask 实验的 step 12,500 双采样评测已完成**，见第 15 节。**LFM230M step 6,000 同模式双采样评测已完成**，见第 16 节。Step 10,000 的旧任务与 speaker-only 保持暂停。

## 1. 实验与目录怎么对应

| 实验 | 配置（相对仓库根目录） | 初始组装模型 | 训练作业 |
| --- | --- | --- | --- |
| frozen-conditioning | `configs/supervised-tts-20260929-all16-bf16-32gpu-lr3e-4-bblr1e-4-ep3-wsd.yaml` | `assets/assembled/qwen3-tts-frozen-conditioning` | `pt-blb39rgw`；最终 step 47382 |
| textbase，可训练文本前端 | `configs/supervised-tts-20260929-all16-textbase-trainable-randproj-bf16-32gpu-lr3e-4-bblr1e-4-ep3-wsd.yaml` | `assets/assembled/qwen3-tts-text-base-trainable-random-projector` | `pt-u0ys9osd`；本报告已评测 step 17500 / 25000 |
| no-speaker，冻结文本前端、无 speaker 条件 | `configs/supervised-tts-20260929-all16-no-spk-bf16-16gpu-acc2-lr3e-4-bblr1e-4-ep3-wsd.yaml` | `assets/assembled/qwen3-tts-frozen-conditioning` | `pt-6epa7d0g` 为首次训练作业；step 17500 已完成，当前评测 step 25000 |

三种实验 YAML 的 `train.run_name` 与文件名去掉 `.yaml` 一致；输出统一为 `/workspace/LM-TTS-Training-Runs/<run_name>/`。`/workspace/workspace/yanglin/LM-TTS-Training-Runs` 是其软链接。

| Run 内路径 | 用途 |
| --- | --- |
| `metrics.jsonl`、`tensorboard/` | 训练、验证、吞吐和数据等待指标，以及 TensorBoard |
| `checkpoints/` | 可直接恢复的完整分布式训练状态 |
| `archived-checkpoints/` | frozen-conditioning 的独立 checkpoint 归档，避免被轮转删除 |
| `logs/` | 节点日志与保留的验证证据 |
| `submissions/` | 实际提交配置、代码快照和作业身份；复核历史训练优先读这里 |

Frozen-conditioning 的训练目录轮转保留最近两份，另有每小时归档任务，评测读取 `archived-checkpoints/`；textbase 与 no-speaker 均设置 `keep_checkpoints: null`，保留全部完整 checkpoint，评测直接读取各自的 `checkpoints/`。下载的基础模型在 `assets/base/`，组装模型在 `assets/assembled/`，评测模型和产物在相邻的 `UltraEval-Audio`，三者各有用途。

## 2. 使用了哪些数据

**三种实验复用同一份 build、16 个数据集和固定 train / validation 划分**，textbase 与 no-speaker 均未重新切分数据。训练数据与固定划分：

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

Merged 表已经包含目标 text、每帧 16 个 codec ID、1024 维 speaker embedding，训练侧不再 join，也不在线转写或编码音频。Text 经组装模型配套 tokenizer 转为 token IDs；前两种实验使用的 speaker embedding 来自**同一条目标音频**的冻结 speaker encoder。No-speaker 复用同一 build，但读取器跳过 speaker embedding 向量，模型也不使用它；源表和既有发布筛选条件没有改变，不因此纳入更多样本。

`configs/data.yaml` 描述数据选择；`data/builds/` 保存固定 Lance 版本、来源哈希、文本 token、长度索引和 train/validation 行划分，不复制大型 codec / speaker 字段，所以源 features 仍须保留。源配置的 `evaluation: no_holdout` 只表示源表尚未切分，正式训练实际读取已切分的 train / validation build。

划分保证样本行不重叠，**不保证说话人或文本隔离，也未在此证明与外部 Seed-TTS 完全去重**。自参考 speaker embedding 与零样本克隆时另一段参考音频存在分布差异；其是否导致内容捷径仍待验证，不能仅凭冻结 encoder 排除这一问题。

## 3. 模型初始化与冻结

三种方案均为“文本 LM 主干迁移 + 新音频模块”。组装在本地提前完成，seed 42、FP32 保存；训练直接加载 assembled 权重，恢复时加载 checkpoint，不重新组装。下面两组对照表比较 frozen-conditioning 与 textbase；**no-speaker 的结构、初始化和冻结范围继承 frozen-conditioning，但 speaker encoder 不执行、不提供 speaker 向量，也不保留对应输入位置**。当前 no-speaker checkpoint 的完整说明见第 11 节。

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

这 16 张音频 embedding 都参与训练，residual 表也复用于 Code Predictor 的码本内预测；它们与输出 head **不 tie**。训练用真实历史帧做 teacher forcing，Talker 与 Code Predictor 同时计算 loss、共同反向传播，二者之间不 detach。No-speaker 保留这套音频输入与联合训练路径，只从上述序列中删除 speaker 位置，不插入零向量。

## 4. 怎么训练与恢复

下表列出 frozen-conditioning 与 textbase 的 32 卡训练设置；文本前端的峰值 LR 与冻结范围见第 3 节，checkpoint 保留策略见第 1 节。**No-speaker 使用 16 卡、梯度累积 2**，保留相同的每卡预算、学习率与样本进度调度，具体配置见第 11 节；不应将下表的 32 卡、accumulation 1 直接套给当前模型。

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

> **2026-10-14 更正**：“更窄的 embedding 和 projector 减少了容量”不成立；textbase SIM 低的直接原因是对 speaker embedding 的利用学得慢。见 [文本前端分析](text-frontend-analysis-20261014.md) 第 3、4 节。

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


## 11. No-speaker step 17,500：双采样预检与全量评测（2026-10-12）

按要求测试同一 no-speaker 实验的 `checkpoints/step-00017500`。完整标记、组装身份和 `use_speaker_embedding=false` 均已检查；checkpoint 的 epoch 为 1，已进入第二个 epoch。使用与 step 10,000 相同的导出路径，保存 FP32 权重，推理使用 BF16 + SDPA。配置、文本 tokenizer 和音频 tokenizer 与 step 10,000 一致，只有训练权重改变。

### 当前模型的来源、结构与训练范围

训练配置为 [no-speaker 实验 YAML](../../configs/supervised-tts-20260929-all16-no-spk-bf16-16gpu-acc2-lr3e-4-bblr1e-4-ep3-wsd.yaml)，直接继承 `configs/base.yaml`。本实验从 `qwen3-tts-frozen-conditioning` **初始组装权重独立开始训练**，不是在已训练的 frozen-conditioning checkpoint 上继续微调，也不采用 textbase 的 1024 维文本 embedding 或随机 text projector。

| 模块 | 结构与初始化来源 | 本轮是否更新 |
| --- | --- | --- |
| 文本 embedding | 151,936 × 2,048；完整继承 `Qwen3-TTS-12Hz-0.6B-Base` | **冻结** |
| 文本 projector | 2,048 → 2,048 → 1,024，两层带 bias 的 Linear，中间 SiLU；继承同一 TTS checkpoint | **冻结**；不是 Identity |
| Talker layers / norm | 28 层，hidden 1,024、FFN 3,072；来自纯文本 `Qwen3-0.6B-Base` | **更新**，峰值 LR `1e-4` |
| 首码本 embedding / head | embedding 3,072 × 1,024、head 1,024 → 3,072；随机初始化，二者不 tie | **更新**，`3e-4` |
| Code Predictor 与 residual 表 / heads | 5 层、hidden 1,024；15 张 2,048 × 1,024 embedding 和 15 个 2,048 类输出 head；整个模块随机初始化，表与 head 不 tie | **更新**，`3e-4`，与 Talker 联合训练 |
| Talker → Code Predictor projector | 两端均为 1,024，使用 `Identity` | 无参数 |
| Speaker encoder | TTS checkpoint 的 ECAPA-TDNN，128 mel → 1,024；权重仍保留在模型状态中 | **冻结且不执行**；不提供 speaker 条件 |
| 音频 codec encoder / decoder | `Qwen3-TTS-Tokenizer-12Hz`，24 kHz、16 码本 | **独立冻结**，不进入训练 optimizer |

主模型总参数仍为 **914,643,008**，其中可训练 **588,329,216**、冻结 **326,313,792**，不含独立 codec。关闭 speaker 条件没有删除 encoder 权重，因此不会减少保存的主模型参数量。随机初始化描述的是训练开始时的状态；step 17,500 的 Talker 和音频模块均已更新。

### 训练输入、数据与配置

文本使用配套 tokenizer，经过冻结的 text embedding / projector 得到 1,024 维向量。每帧 16 个 codec ID 分别查询 16 张可训练音频 embedding，再求和为一个 1,024 维向量。训练输入为 **角色与控制前缀 → 完整目标文本 → codec BOS → 真实历史音频帧**；完全省略 speaker 位置。文本位置加 codec PAD，音频位置加 projected text PAD，这些协议向量与 batch 长度补齐的 padding 不同。

训练继续使用第 2 节的全部 16 个数据集和固定 build：`supervised-tts-20260929-merged-20261001-all16`，共 **128,091,959 条训练、128,219 条验证**。仅跳过 speaker embedding 大字段，不重建数据、不改变文本或 codec，也没有新增跨音频参考采样或 reference loss mask。

| 项目 | 当前 no-speaker 实验 |
| --- | --- |
| 资源 / 精度 | 2 节点 × 8 A800 80 GB，FSDP2、BF16、activation checkpointing |
| 动态 batching / attention | 每卡 32,000 codec frames / 50,000 Talker tokens；Talker FA2 varlen、样本间隔离；Code Predictor SDPA |
| 梯度累积 | 2；16 卡 × 2 个 microbatch 对应原基线 32 卡 × 1 的预算，不保证每次更新的实际样本组合一致 |
| Loss | 完整 codec teacher forcing，`first_ce + 0.3 × residual_ce`；residual 为 15 个码本的平均 CE，按所有卡与整个累积窗口的监督 token 归一化 |
| Optimizer | AdamW；Talker layers / norm 为 `1e-4`，音频模块为 `3e-4`；weight decay 0.01，grad clip 1.0 |
| 训练长度 / LR 调度 | 3 epochs；按消费样本进度 WSD，前 0.03 epoch warmup、steady 到 2.7、最后 0.3 epoch 余弦衰减到峰值 10%；不预设固定总 step |
| 数据读取 | 每 rank 4 workers，prefetch factor 2，shuffle / 长度分桶窗口 65,536；保留坏样本跳过与跨 rank 协调机制 |
| 验证 / 保存 | 第 100 步、每 2,500 步及训练结束；训练内验证为固定 holdout 上的 teacher-forcing loss，`eval.batch_size=32`；保留全部完整 checkpoint |

Run 名称为 `supervised-tts-20260929-all16-no-spk-bf16-16gpu-acc2-lr3e-4-bblr1e-4-ep3-wsd`，输出目录为 `/workspace/LM-TTS-Training-Runs/<run_name>/`，也可通过 `yanglin/LM-TTS-Training-Runs` 软链接访问。当前读取 `checkpoints/step-00017500`；其元数据记录 `world_size=16`、`epoch=1`（从 0 计数）、本 epoch 已消费 **13,685,862** 条样本，处于 WSD steady 阶段，两组 LR 仍为 `1e-4 / 3e-4`。上述设置依据 run 内实际 `config.yaml`、checkpoint `metadata.json` 和初始 `assembly_report.json` 核对；提交与恢复方法见 [no-speaker 训练记录](all16-no-speaker-20261008.md)。

### 当前评测的输入条件

推理模式为 **`icl_only`**：输入参考文本、参考音频 codec 和目标文本，不输入 speaker embedding，也不运行 speaker encoder。参考音频仍通过 codec 前缀提供声音条件；“无 speaker embedding”不等于“无参考音频”。导出配置带 `lm_tts_use_speaker_embedding=false`，评测适配显式省略 speaker 位置，不能直接把该 checkpoint 当作普通 speaker + ICL 模型调用。

该改动去掉了目标音频 speaker embedding 这一条件通道，但没有证明所有训推差异都消失：训练仍是单条完整音频的 teacher forcing，外部克隆评测会额外提供参考文本和参考 codec 前缀，再自由生成目标音频。不能仅因关闭 speaker encoder 就断言推理质量已解决。训练内 `eval.batch_size=32` 也不等于自由生成的 batch size；本轮生成始终为 **batch size 1**。全量样本规模、评分器、生成参数与结果目录见本节末的“全量评测：双采样后接双 greedy”。

### 十条双采样预检

本次沿用上一轮的 **Seed-TTS 中英文各前 5 条**，相同参考音频、参考文本、目标文本及逐条 seed；纯 ICL、Auto、non-streaming、batch size 1，Talker 和 Code Predictor 均开启采样（top-k 50、top-p 1.0、temperature 0.9），生成上限仍为 378 token。五个进程各生成两条。此小试检查生成长度和结束行为，不计算 ASR / SIM，也不自动恢复任何全量评测。

- 导出：`UltraEval-Audio/init_model/all16-no-spk-step-00017500`；权重 SHA256：`d23148cab2837738d3c01eed79cbcc0888430eb59c67fb3799dbbf1e63ae02e6`。
- 结果：`UltraEval-Audio/res/all16-no-spk-step17500-seedtts-sampling10-bs1-20261012`；音频在 `output/icl_only/`，逐条对照在 `comparison.json`，复现命令在 `run-trial.py`。

测试于 01:31 UTC 完成；十条记录覆盖、双采样参数、相同文本 / 参考音频 hash / seed / 无 speaker 条件均已核对。

| 语言 / Index | Step 10,000 双采样（秒） | Step 17,500 双采样（秒） |
| --- | ---: | ---: |
| 英文 / 0 | 1.76 | 2.56 |
| 英文 / 1 | 2.64 | 2.48 |
| 英文 / 2 | 8.56 | 5.84 |
| 英文 / 3 | 3.60 | 3.28 |
| 英文 / 4 | 5.36 | 4.40 |
| 中文 / 0 | 9.44 | 14.96 |
| 中文 / 1 | 15.12 | 15.84 |
| 中文 / 2 | 0.32 | 7.36 |
| 中文 / 3 | 4.64 | 6.08 |
| 中文 / 4 | 10.96 | 11.04 |

两轮采样均为 **0/10 达到 30 秒上限**。Step 17,500 的时长为 **2.48–15.84 秒**，没有低于 1 秒的输出；此前中文 index 2 的 0.32 秒变为 7.36 秒。这十条中未再观察到上一轮的极短输出，但仅凭长度不能确认是否完整读对，也不能证明全量生成稳定。没有进行 ASR、SIM 或人工听测；本次不报告可懂度或音质改善。全部十条已生成，测试进程已结束，旧全量任务继续暂停。


### 全量评测：双采样后接双 greedy

十条预检完成后，按后续要求启动 **step 17,500 全量 Seed-TTS 双采样评测**，英文 1,088 条、中文 2,020 条。沿用纯 ICL、Auto、non-streaming、BF16 + SDPA、batch size 1、逐条 seed 和上述采样参数；按后续要求，双采样完成生成、评分及审计后，自动接同一 checkpoint 的全量双 greedy。Greedy 将 Talker 与 Code Predictor 的采样同时关闭，其余输入和评分设置保持一致。

- 生成：GPU 0–7，共 8 个分片；每进程 PyTorch 分配上限 4 GiB，启动检查保留 0.5 GiB 余量。
- 评分：GPU 0、1、6、7；英文 Whisper large-v3 WER、中文 SeACo-Paraformer CER、WavLM large + ECAPA 说话人相似度。先完成小样本评分和审计，再自动生成全量、评分、汇总和审计。
- 统计：保留所有生成记录；超过 30 秒的音频按既有规则排除评分，并单独报告排除数量与比例，不能只看保留样本的均分。
- 结果目录：`UltraEval-Audio/res/all16-no-spk-step17500-seedtts-sampling-bs1-20261012-8gpu`。复现命令为目录内的 `run.sh`，主日志 `run.log`，阶段状态 `pipeline_status.json`；最终结果为 `full/summary.json`、`full/audit.json`，音频及逐条评分在 `full/icl_only/`。
- 后台会话：`seedtts-no-spk-s17500-sampling-full`。Step 10,000 的旧任务仍保持暂停。

四条评分预检和审计已通过：英文两条 WER 均为 0，中文两条平均 CER 为 36.36%；样本太少，不能代表全量结果。首次全量启动因 5 卡可用显存略低于 4 GiB + 1 GiB 启动门槛而退出，随后维持 4 GiB 上限，将启动余量调整为 0.5 GiB 并恢复；复用原预检记录。全量双采样已完成，结果如下。


### Step 17,500 全量双采样结果

本组为 **no-speaker、纯 ICL、Talker / Code Predictor 双采样**，于 **2026-10-12 03:37:31 UTC** 完成。3,108 条全部生成，3,103 条完成评分、5 条按时长规则排除，完整性审计通过，无未恢复的生成或评分错误。

| 模型 | 数据集 | 合成/总数 | 评分/总数 | CER或WER/% ↓ | SIM/% ↑ | DNSMOS P808 ↑ | DNSMOS OVRL ↑ | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| icl_only | seed_tts_eval_en | 1088/1088 | 1088/1088 | 4.398 | 44.778 | — | — | 完整 |
| icl_only | seed_tts_eval_zh | 2020/2020 | 2015/2015 | 22.659 | 48.587 | — | — | 完整（排除 5 条） |

英文报告 WER，中文报告 CER，均为逐句错误率的算术平均；SIM 为余弦相似度乘 100。中文 5 条输出均为 30.16 秒，因超过 30 秒排除，占中文总数约 **0.25%**；`2015/2015` 的评分分母为排除后的有效样本数。DNSMOS 本轮未计算，`—` 不表示零分。

原始证据：[全量汇总](../../../UltraEval-Audio/res/all16-no-spk-step17500-seedtts-sampling-bs1-20261012-8gpu/full/summary.json) · [完整性审计及排除清单](../../../UltraEval-Audio/res/all16-no-spk-step17500-seedtts-sampling-bs1-20261012-8gpu/full/audit.json)。逐条转写、评分和 WAV 位于同目录的 `icl_only/`。

Greedy 的接续任务使用独立目录 `UltraEval-Audio/res/all16-no-spk-step17500-seedtts-greedy-bs1-20261012-8gpu`；同样为 8 卡、batch size 1、纯 ICL、3,108 条。两组独立按 30 秒规则排除长音频，汇总时分别报告排除数量。

接续脚本为 greedy 目录内的 `wait-and-run.py`，实际运行命令为 `run.sh`，后台会话为 `seedtts-no-spk-s17500-greedy-queued`。`queue_status.json` / `queue.log` 记录等待与接续状态；greedy 启动后的状态和日志为 `pipeline_status.json` / `run.log`，最终结果同样在 `full/summary.json`、`full/audit.json`。脚本等待 sampling 释放运行锁，再确认 checkpoint 身份、完成状态、审计通过以及 3,108 条全部已评分或按规则排除；若 sampling 失败或不完整，则停止接续，不启动 greedy。


### Step 17,500 全量双 greedy 结果

本组为同一 **no-speaker step 17,500、纯 ICL** 模型，Talker 与 Code Predictor 的采样均关闭，其余设置沿用上一组。于 **2026-10-12 08:51:52 UTC** 完成，接续流程与完整性审计均通过。3,108 条全部生成，其中 **546 条评分、2,562 条因超过 30 秒排除**；无未恢复的生成或评分错误。

| 模型 | 数据集 | 合成/总数 | 评分/总数 | CER或WER/% ↓ | SIM/% ↑ | DNSMOS P808 ↑ | DNSMOS OVRL ↑ | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| icl_only | seed_tts_eval_en | 1088/1088 | 197/197 | 44.859 | 26.335 | — | — | 完整（排除 891 条） |
| icl_only | seed_tts_eval_zh | 2020/2020 | 349/349 | 91.902 | 9.121 | — | — | 完整（排除 1671 条） |

英文排除 **891/1088（81.89%）**，中文排除 **1671/2020（82.72%）**，总排除率 **82.43%**。被排除的输出集中在约 30.16 秒的生成上限；`197/197` 和 `349/349` 是过滤后的评分覆盖，不是原始测试集的覆盖率。表中“完整”仅表示生成、评分及排除记录齐全，**不代表生成质量正常**。DNSMOS 未计算。

与双采样的英文 0 条、中文 5 条超长输出相比，双 greedy 出现了大量无法在上限前结束的输出，且保留样本的内容错误率仍很高。两组按各自时长独立过滤，评分子集差异很大，不能把过滤后的均分差值视为同一批样本上的严格配对提升。

原始证据：[结果表](../../../UltraEval-Audio/res/all16-no-spk-step17500-seedtts-greedy-bs1-20261012-8gpu/full/results_table.md) · [全量汇总](../../../UltraEval-Audio/res/all16-no-spk-step17500-seedtts-greedy-bs1-20261012-8gpu/full/summary.json) · [完整性审计及排除清单](../../../UltraEval-Audio/res/all16-no-spk-step17500-seedtts-greedy-bs1-20261012-8gpu/full/audit.json)。逐条转写、评分和 WAV 位于同目录的 `icl_only/`。两组全量评测及自动接续现已完成。


## 12. No-speaker step 25,000：全量双采样与双 greedy（2026-10-13）

本轮读取同一 no-speaker run 的 `checkpoints/step-00025000`，不是 textbase step 25,000。`COMPLETE` 标记、组装身份与 `use_speaker_embedding=false` 已核对。Checkpoint 记录 `world_size=16`、`epoch=1`（从 0 计数），本 epoch 已消费 **75,174,926** 条样本。结构、冻结范围、全部 16 个数据集、固定划分和训练设置均沿用第 11 节；只更换 checkpoint 权重。

评测继续使用 **纯 ICL、Auto、non-streaming、BF16 + SDPA、batch size 1**，英文 1,088 条、中文 2,020 条。先运行 Talker / Code Predictor 双采样（top-k 50、top-p 1.0、temperature 0.9），完成生成、评分与完整性审计后自动接双 greedy；逐条 seed、输入及评分规则与 step 17,500 一致。两组均先进行中英文各 2 条生成、评分和审计预检。

- 导出模型：`UltraEval-Audio/init_model/all16-no-spk-step-00025000`，保存 FP32 权重及配套 tokenizer；无 speaker 标记要求推理显式省略 speaker 位置。
- 生成使用 GPU 0–7；每进程分配上限 4 GiB、启动预留 0.5 GiB。评分使用 GPU 0–3，以避开其他卡上已有任务的显存压力。
- 指标仍为英文 Whisper large-v3 WER、中文 SeACo-Paraformer CER、WavLM large + ECAPA SIM；不计算 DNSMOS。
- 各组独立排除超过 30 秒的输出，同时保留并报告排除数量与比例。尤其需要检查 step 17,500 双 greedy 的大量超长输出是否仍存在；过滤后的评分不能代替全量生成成功率。

| 内容 | `UltraEval-Audio/` 下的路径 |
| --- | --- |
| 全量双采样 | `res/all16-no-spk-step25000-seedtts-sampling-bs1-20261013-8gpu/` |
| 全量双 greedy | `res/all16-no-spk-step25000-seedtts-greedy-bs1-20261013-8gpu/` |
| 每组命令、状态、日志 | 各目录内的 `run.sh`、`pipeline_status.json`、`run.log` |
| 完成后的结果与审计 | 各目录内的 `full/results_table.md`、`full/summary.json`、`full/audit.json` |
| 生成音频与逐条评分 | 各目录内的 `full/icl_only/` |
| Greedy 接续控制 | Greedy 目录内的 `wait-and-run.py`、`queue_status.json`、`queue.log` |

接续脚本等待双采样释放运行锁，再验证 checkpoint 身份、双采样设置、审计通过以及全部 3,108 条均已评分或按规则排除；若前一阶段失败则不启动 greedy。四条预检的生成、ASR / SIM 评分和完整性审计均已通过：英文两条平均 WER **0.000%**、SIM **50.753%**，中文两条平均 CER **6.818%**、SIM **69.232%**；无超长排除，音频时长 2.08–9.84 秒。这只是预检，不代表全量质量。全量双采样已完成，结果见下文；双 greedy 随后自动接续并已完成，结果待补录。

导出权重 SHA256：`1f2abbedf3f0cbdee1397ec34ba798a0f09a897a692135b08953dbdaf491bf89`。导出配置、文本 tokenizer 和音频 tokenizer 权重均与 step 17,500 一致。后台会话分别为 `seedtts-no-spk-s25000-sampling-full`、`seedtts-no-spk-s25000-greedy-queued`；状态以各目录中的 JSON 文件为准。


### Step 25,000 全量双采样结果

本组为 **no-speaker、纯 ICL、Talker / Code Predictor 双采样**，于 **2026-10-13 05:10:30 UTC** 完成。**3,108 条全部生成并评分，无超过 30 秒的排除项**；完整性审计通过，无未恢复的生成或评分错误。

| 模型 | 数据集 | 合成/总数 | 评分/总数 | CER或WER/% ↓ | SIM/% ↑ | DNSMOS P808 ↑ | DNSMOS OVRL ↑ | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| icl_only | seed_tts_eval_en | 1088/1088 | 1088/1088 | 3.508 | 47.483 | — | — | 完整 |
| icl_only | seed_tts_eval_zh | 2020/2020 | 2020/2020 | 7.325 | 56.077 | — | — | 完整 |

英文报告 WER，中文报告 CER，均为逐句错误率的算术平均；SIM 为余弦相似度乘 100。英文最长音频 **26.48 秒**，中文最长 **25.28 秒**，两组评分分母均为完整测试集。DNSMOS 未计算，`—` 不表示零分。

与第 11 节 step 17,500 的同设置双采样相比，英文 WER **4.398% → 3.508%**、SIM **44.778% → 47.483%**；中文 CER **22.659% → 7.325%**、SIM **48.587% → 56.077%**，中文超长排除从 **5 条降为 0 条**。本轮汇总指标更好；中文前一轮只评分 2,015 条，本轮评分全部 2,020 条，因此中文均分并非严格相同子集上的配对比较。

原始证据：[结果表](../../../UltraEval-Audio/res/all16-no-spk-step25000-seedtts-sampling-bs1-20261013-8gpu/full/results_table.md) · [全量汇总](../../../UltraEval-Audio/res/all16-no-spk-step25000-seedtts-sampling-bs1-20261013-8gpu/full/summary.json) · [完整性审计](../../../UltraEval-Audio/res/all16-no-spk-step25000-seedtts-sampling-bs1-20261013-8gpu/full/audit.json)。逐条转写、评分和 WAV 位于同目录的 `icl_only/`。

双 greedy 于 **2026-10-13 05:10:31 UTC** 自动接续，现已完成并通过审计，完整结果待补录；本节上述数值仅属于双采样。


## 13. Emilia2 refmask step 10,000：speaker + ICL 评测（2026-10-14）

本轮 checkpoint 为 `supervised-tts-20261012-emilia2-refmask-bf16-16gpu-acc1-lr3e-4-bblr1e-4-ep3-wsd/checkpoints/step-00010000`，位于同一 `LM-TTS-Training-Runs` 根目录。`COMPLETE` 和组装身份检查通过，元数据为 `epoch=0`、已消费 **66,839,436** 条样本，仍处于第一个 epoch、WSD steady 阶段。该实验单独使用 Emilia2，不属于前述 all16 数据配方；记录在这里便于统一查阅 Seed-TTS 结果。

### 模型、数据与训练差异

模型来自同一 `assets/assembled/qwen3-tts-frozen-conditioning`。文本 embedding（151,936 × 2,048）和 text projector（2,048 → 2,048 → 1,024）继承 TTS checkpoint 并冻结；Talker layers / norm、首码本 embedding / head、Code Predictor 及 residual embeddings / heads 参与训练，初始化来源沿用第 11 节。**本轮开启 speaker embedding**，使用冻结的 speaker encoder 所提取的参考片段向量；音频 codec 仍独立冻结。

| 项目 | 本轮设置 |
| --- | --- |
| 训练配置 | `configs/supervised-tts-20261012-emilia2-refmask-bf16-16gpu-acc1-lr3e-4-bblr1e-4-ep3-wsd.yaml` |
| 数据配方 / build | `configs/data-emilia2-reference.yaml` / `supervised-tts-20261012-emilia2-reference` |
| 数据来源 | selection `tts-selection-supervised-tts-20261012T123927bjt-01`；仅绑定 Emilia2 reference features |
| 固定划分 | 训练 **76,607,724** 条，验证 **76,684** 条；seed 42、约千分之一留出 |
| Speaker 条件 / loss mask | `use_speaker_embedding=true`、`mask_reference=true`；参考来自目标录音内的合法裁剪片段，屏蔽该区间所有 16 个码本的 loss，保留 EOS 监督 |
| 训练资源 / loss | 16 卡、BF16、accumulation 1；每卡 32,000 frames / 50,000 tokens；token loss：`first_ce + 0.3 × residual_ce` |
| LR / 长度 | Talker 主干 `1e-4`，其余可训练音频模块 `3e-4`；3 epochs，样本进度 WSD，warmup 1%、decay 10%、最低为峰值 10% |

训练仍提供完整目标文本与真实历史 codec；**refmask 仅取消参考区间的 loss，并不删除输入帧，也不构造独立的参考文本 / codec 前缀**。因此本次 speaker + ICL 是外部克隆生成测试，不能把 refmask 直接等同于完整 ICL 训练，或仅凭这项改动就认定所有训推差异已解决。

### 评测设置与产物

本次保留模型的 speaker 条件，采用 **`speaker_icl`**：参考音频提取 speaker embedding，同时提供参考文本、参考 codec 和目标文本。沿用 Seed-TTS 英文 **1,088** 条、中文 **2,020** 条、Auto、non-streaming、BF16 + SDPA、batch size 1；先双采样（Talker / Code Predictor 均为 top-k 50、top-p 1.0、temperature 0.9），完成评分及审计后自动接双 greedy。Speaker-only 不在本轮运行范围内。

生成使用 GPU 0–7，每进程 PyTorch 分配上限 8 GiB、启动余量 2 GiB；评分使用 GPU 0–3。指标为英文 Whisper large-v3 WER、中文 SeACo-Paraformer CER、WavLM large + ECAPA SIM，DNSMOS 不计算。继续按每组自身的 30 秒时长规则排除超长输出，并单独报告排除数；和 no-speaker 结果比较时，需同时考虑训练数据、speaker 条件、消费样本量和梯度累积的变化，不能归因于 refmask 单一因素。

| 内容 | `UltraEval-Audio/` 下的路径 |
| --- | --- |
| 独立导出模型 | `init_model/emilia2-refmask-step-00010000/` |
| 全量双采样 | `res/emilia2-refmask-step10000-seedtts-sampling-bs1-20261014-8gpu/` |
| 全量双 greedy | `res/emilia2-refmask-step10000-seedtts-greedy-bs1-20261014-8gpu/` |

每组的 `run.sh` 保存复现命令，`pipeline_status.json` / `run.log` 记录阶段与日志；最终评分表、汇总、审计在 `full/results_table.md`、`full/summary.json`、`full/audit.json`，音频及逐条评分在 `full/speaker_icl/`。Greedy 目录中的 `wait-and-run.py` / `queue_status.json` 负责接续：只有双采样 checkpoint 身份匹配、全部 3,108 条评分或按规则排除且审计通过后才启动。

导出权重 SHA256：`eaf81e93de6ef498843d018d9b5e95236b7cf72bb0f8e25b3032601aab86ba54`。导出保存 FP32 权重及配套 tokenizer，推理使用 BF16；已确认 speaker 标记为 true。中英文各两条生成、评分及完整性审计均已通过：英文 WER **0.000%**、SIM **65.933%**，中文 CER **15.909%**、SIM **77.900%**；四条均未超过 30 秒，运行断言确认 speaker embedding 与 ICL 均实际使用。上述仅为预检，不能代表全量质量。全量双采样已完成，结果见下文；双 greedy 按后续要求停止，不再推进全量评分。后台会话为 `seedtts-emilia2-refmask-s10000-sampling-full` 与 `seedtts-emilia2-refmask-s10000-greedy-queued`。


### Emilia2 refmask step 10,000 全量双采样结果

本组为 **speaker embedding + ICL、Talker / Code Predictor 双采样**，于 **2026-10-14 02:36:45 UTC** 完成。**3,108 条全部生成并评分，无超长排除**；完整性审计通过，无未恢复的生成或评分错误。

| 模型 | 数据集 | 合成/总数 | 评分/总数 | CER或WER/% ↓ | SIM/% ↑ | DNSMOS P808 ↑ | DNSMOS OVRL ↑ | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| speaker_icl | seed_tts_eval_en | 1088/1088 | 1088/1088 | 5.016 | 53.470 | — | — | 完整 |
| speaker_icl | seed_tts_eval_zh | 2020/2020 | 2020/2020 | 4.594 | 65.478 | — | — | 完整 |

英文报告 WER，中文报告 CER，均为逐句错误率的算术平均；SIM 为余弦相似度乘 100。英文最长音频 **11.20 秒**，中文最长 **15.68 秒**，均未超过 30 秒，两组评分分母均为完整测试集。DNSMOS 未计算，`—` 不表示零分。

与第 12 节 no-speaker step 25,000 的双采样汇总相比，本轮英文 WER **3.508% → 5.016%**，中文 CER **7.325% → 4.594%**；英文 SIM **47.483% → 53.470%**，中文 SIM **56.077% → 65.478%**。即中文错误率更低、中英文相似度更高，但英文错误率更高。两轮都没有时长排除；训练数据、speaker 条件和训练进度不同，这不是 refmask 的单变量对照。

原始证据：[结果表](../../../UltraEval-Audio/res/emilia2-refmask-step10000-seedtts-sampling-bs1-20261014-8gpu/full/results_table.md) · [全量汇总](../../../UltraEval-Audio/res/emilia2-refmask-step10000-seedtts-sampling-bs1-20261014-8gpu/full/summary.json) · [完整性审计](../../../UltraEval-Audio/res/emilia2-refmask-step10000-seedtts-sampling-bs1-20261014-8gpu/full/audit.json)。逐条转写、评分和 WAV 位于同目录的 `speaker_icl/`。

双 greedy 于 **2026-10-14 02:36:54 UTC** 启动，后于 **03:10 UTC** 按要求停止，接续进程与八个生成分片均已退出。停止时保留 **827** 条成功生成记录，其中 **130** 条超过 30 秒；未完成全量生成或评分，不作为正式结果。状态文件中的 `failed` 来自主动中断，接续状态为 `stopped`。上述完整结果仅属于双采样。


## 14. Emilia2 LFM230M step 5,000：仅双采样评测（2026-10-14）

读取 `supervised-tts-20261012-emilia2-lfm230m-text2048-spk-refmask-bf16-8gpu-acc2-lr3e-4-bblr1e-4-ep3-wsd/checkpoints/step-00005000`。完整标记与组装身份已核验，checkpoint 为 `epoch=0`、已消费 **34,642,179** 条样本。按要求只跑 sampling，不接 greedy。

### 模型与训练条件

| 模块 | 结构、初始化与本轮更新范围 |
| --- | --- |
| 文本 tokenizer / embedding | 使用 LFM tokenizer；65,536 × 2,048 embedding，以 LFM 的 1,024 维权重初始化为 `[E, -E]`，TTS 控制 token 使用预留词表行；**可训练**，LR `1e-4` |
| 文本 projector | 2,048 → 2,048 → 1,024、SiLU；采用 paired-SiLU 初始化，初始投影组合还原 LFM embedding；**可训练**，LR `3e-4`，不是随机 projector |
| Talker 主干 | `LFM2.5-230M-Base`，14 层、hidden 1,024，混合短卷积与 attention；**可训练**，LR `1e-4` |
| Codec embedding / head、Code Predictor | 首码本 3,072 类、16 张音频 embedding、5 层 Code Predictor 与 15 个 residual heads；训练开始时随机初始化，**可训练**，LR `3e-4` |
| Speaker encoder | 从 Qwen3-TTS 模板继承 ECAPA，1,024 维输出；**冻结并参与条件输入** |
| 音频 codec | Qwen3-TTS-Tokenizer-12Hz，24 kHz、16 码本；**独立冻结** |

主模型共 **459,812,672** 个参数，其中 **450,958,336** 可训练，不含独立 codec。组装目录为 `assets/assembled/lfm2.5-230m-base-text2048-spk`；训练配置为 `configs/supervised-tts-20261012-emilia2-lfm230m-text2048-spk-refmask-bf16-8gpu-acc2-lr3e-4-bblr1e-4-ep3-wsd.yaml`。

使用与第 13 节相同的 Emilia2 划分：训练 **76,607,724** 条、验证 **76,684** 条，build 为 `supervised-tts-20261012-emilia2-reference-lfm2.5-230m`，文本 token 改为 LFM tokenizer。BF16、8 卡 × accumulation 2、每卡 32,000 frames / 50,000 tokens；token loss、reference mask、峰值 LR 与样本进度 WSD 沿用上一组，目标 3 epochs。Step 5,000 仍处于 steady 阶段；消费样本量不同，不能直接把与 Qwen step 10,000 的分数差异归因于主干结构。

### 推理适配与评测范围

LFM checkpoint 使用 `lm_tts` 的生成路径：沿用训练输入构造，加入参考音频的 speaker embedding、参考文本 / codec 前缀，再生成目标音频；每帧 16 个码本参与后续帧的输入。文本 BOS / EOS / PAD 使用 LFM 组装配置中的 ID，参考文本和目标文本分别 tokenize 后拼接。参考与生成 codec 一起解码，再裁掉参考部分。

本次为这条路径补齐 speaker 条件，speaker 音频前处理与官方 Qwen 方法一致；通过“生成与训练协议逐帧计算一致”和“speaker 提取与官方实现一致”的测试。LFM 暂使用 **完整前缀计算、无 KV cache**；计算速度不能直接与 Qwen 的缓存推理相比。本次修改限于推理和评测适配，不改变训练模型参数结构、optimizer 或 resume 状态。

评测为 **speaker + ICL、batch size 1、Auto、non-streaming、BF16 + SDPA**；Talker / Code Predictor 均采样，top-k 50、top-p 1.0、temperature 0.9，Talker repetition penalty 1.05。英文 1,088 条、中文 2,020 条；8 卡生成、GPU 0–3 评分，每进程生成显存上限 8 GiB、预留 2 GiB。继续使用相同的 WER / CER、WavLM SIM 和独立 30 秒排除规则，DNSMOS 不计算。

- 导出：`UltraEval-Audio/init_model/emilia2-lfm230m-text2048-spk-refmask-step-00005000/`；权重 SHA256 `dd8c2a7d757bd8d408a86be27799b80ba1dbf1904055110f4662b7843ff2978d`。
- 结果：`UltraEval-Audio/res/emilia2-lfm230m-text2048-spk-refmask-step5000-seedtts-sampling-bs1-20261014-8gpu/`。
- 复现命令为结果目录的 `run.sh`；阶段及日志为 `pipeline_status.json` / `run.log`；完整结果在 `full/results_table.md`、`full/summary.json`、`full/audit.json`，音频和逐条评分在 `full/speaker_icl/`。
- 后台会话：`seedtts-emilia2-lfm230m-s5000-sampling-full`。中英文各两条生成、评分与审计均已通过；3,108 条全量音频已全部生成并评分，修复评分中断后完成审计（见下文）。未安排 greedy，本轮评测已完成。

预检四条时长 **2.48、2.80、5.28、6.48 秒**，单条生成耗时约 **3.18–7.11 秒**；英文平均 WER **13.393%**、SIM **45.209%**，中文平均 CER **9.091%**、SIM **61.383%**。无超长排除，speaker 与 ICL 条件均通过运行校验。这只是流程预检，不能代表全量质量或用于模型优劣结论。


### LFM 评分中断与恢复（2026-10-14）

全量音频已全部生成。评分于 **04:02:17 UTC** 中断：中文 index **282** 的生成音频只有 **0.08 秒**，SeACo-Paraformer 预测不到识别 token，返回空结果列表；原适配器直接读取 `[0]["text"]` 导致 `IndexError`。一个评分分片失败后，协调进程按既定机制终止其他评分分片。日志里的 `FireRedASR failed` 是原错误信息写错模型名，实际使用的是 Paraformer。

已修复空假设处理：**返回空转写并正常计分**，对非空目标文本计为全部删除（CER 100%），保留该样本及 SIM 评分。没有把短音频排除，也没有重生成或改动已有 WAV；真实模型异常仍会抛出。回归验证覆盖空转写错误率、正常转写不变及模型异常不被吞掉，连同既有评测测试共 **29 项通过**。

从原目录的 `run.sh` 恢复，保留中断前 **616 条**成功评分和历史错误记录；所有生成分片均确认 `remaining=0`。**04:14:26 UTC** 已重新进入全量评分。故障样本 index 282 随后成功评分：空转写、**CER 100.000%**、**SIM -14.922%**，保留在统计内；四个评分分片均已越过中断位置，确认时成功评分已从 616 条增至 **713 条**。随后于 **04:28:22 UTC** 完成全部评分、汇总和审计。审计保留 1 条历史评分错误记录，该样本已成功重试，无未恢复错误。修复提交：UltraEval-Audio `0cfbb2b`。


### LFM230M step 5,000 全量双采样结果

本组为 **LFM230M、speaker embedding + ICL、Talker / Code Predictor 双采样**，于 **2026-10-14 04:28:22 UTC** 完成。**3,108 条全部生成并评分，无超长排除**；完整性审计通过，输入身份、分片覆盖、音频格式及推理条件均已核对。下表与原始结果表一致，均分另从 3,108 条唯一成功评分记录重新计算核对。

| 模型 | 数据集 | 合成/总数 | 评分/总数 | CER或WER/% ↓ | SIM/% ↑ | DNSMOS P808 ↑ | DNSMOS OVRL ↑ | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| speaker_icl | seed_tts_eval_en | 1088/1088 | 1088/1088 | 8.682 | 39.195 | — | — | 完整 |
| speaker_icl | seed_tts_eval_zh | 2020/2020 | 2020/2020 | 9.399 | 55.248 | — | — | 完整 |

英文为逐句 WER 的算术平均，中文为逐句 CER 的算术平均；SIM 为余弦相似度乘 100。DNSMOS 未计算，`—` 不表示零分。英文平均时长 **3.979 秒**、最长 **11.12 秒**；中文平均 **5.279 秒**、最长 **12.40 秒**。

**无超长输出不代表没有提前结束问题。** 中文仍有 5 条低于 0.2 秒：index 282、992、1797 为 0.08 秒，844、1322 为 0.16 秒；这 5 条均保留在评分中，CER 均为 100%，对应 SIM 也计入均分。其中 4 条 ASR 为空转写，844 的转写为“嗯”。此前触发评分中断的 282 已包含在最终 2,020 条中文结果中，没有通过删样本改善分数。

与第 13 节 **Qwen Emilia2 refmask step 10,000** 的同模式双采样比较：

| Checkpoint | 英文 WER/% ↓ | 英文 SIM/% ↑ | 中文 CER/% ↓ | 中文 SIM/% ↑ | 评分条数 / 超长排除 |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen Emilia2 refmask step 10,000 | 5.016 | 53.470 | 4.594 | 65.478 | 3,108 / 0 |
| LFM230M Emilia2 refmask step 5,000 | 8.682 | 39.195 | 9.399 | 55.248 | 3,108 / 0 |

本次 LFM checkpoint 的中英文错误率均更高、相似度均更低。两轮使用相同的 Emilia2 样本划分与完整 Seed-TTS 测试集，但 LFM 仅消费 **34,642,179** 条训练样本，Qwen 已消费 **66,839,436** 条；主干、tokenizer、文本前端初始化及冻结范围也不同。因此这反映的是两个当前 checkpoint 的结果，不能据此单独判定 LFM 主干或 refmask 方案的优劣。

原始证据：[结果表](../../../UltraEval-Audio/res/emilia2-lfm230m-text2048-spk-refmask-step5000-seedtts-sampling-bs1-20261014-8gpu/full/results_table.md) · [全量汇总](../../../UltraEval-Audio/res/emilia2-lfm230m-text2048-spk-refmask-step5000-seedtts-sampling-bs1-20261014-8gpu/full/summary.json) · [完整性审计](../../../UltraEval-Audio/res/emilia2-lfm230m-text2048-spk-refmask-step5000-seedtts-sampling-bs1-20261014-8gpu/full/audit.json)。逐条转写、评分和 WAV 位于同目录的 `speaker_icl/`。本轮仅运行双采样，没有后续 greedy 任务。


## 15. Emilia2 refmask step 12,500：每卡四个 worker 的双采样评测（2026-10-14）

Checkpoint 为 `supervised-tts-20261012-emilia2-refmask-bf16-16gpu-acc1-lr3e-4-bblr1e-4-ep3-wsd/checkpoints/step-00012500`。模型结构、初始化、冻结范围、Emilia2 数据划分与 reference mask 训练方式均沿用第 13 节；本次仅更换同一训练实验的 checkpoint。完整标记已核验，训练进度为 `epoch=1`（从零计数，即第二轮）、`next_batch=1081`，本轮已消费 **7,249,267** 条样本。导出权重 SHA256 为 `3361d68ab3040f3df5baffe5a7070a42a5f418b09cf07cb313653aedce52a688`；配套配置与文本 tokenizer 的文件哈希均与 step 10,000 一致，speaker 条件启用。

本次使用 **speaker embedding + ICL、Talker / Code Predictor 双采样**，沿用 Auto、non-streaming、BF16 + SDPA 和 batch size 1；top-k 50、top-p 1.0、temperature 0.9，Talker repetition penalty 1.05。英文 1,088 条、中文 2,020 条；只安排 sampling。

生成并行度为 **8 卡 × 每卡 4 个 worker，共 32 个进程 / 分片**。每个进程处理独立样本，仍使用 `42 + 原始样本 index` 的随机种子；提高并行度不改变 batch size 或采样参数。每进程 PyTorch 显存上限 8 GiB、启动预留 2 GiB；评分阶段使用 GPU 0–3，中英文各两个评分分片。流程自动依次执行四条预检、预检评分与审计、全量生成、全量评分与审计。指标及独立 30 秒超长排除规则与第 13 节一致，DNSMOS 不计算。

| 内容 | `UltraEval-Audio/` 下的路径 |
| --- | --- |
| 独立导出模型 | `init_model/emilia2-refmask-step-00012500/` |
| 本轮运行目录 | `res/emilia2-refmask-step12500-seedtts-sampling-bs1-20261014-8gpu-4wpg/` |
| 复现 / 恢复命令 | 上述运行目录的 `run.sh` |
| 阶段 / 日志 | `pipeline_status.json`、`run.log`、`logs/` |
| 正式结果 | `full/results_table.md`、`full/summary.json`、`full/audit.json` |
| 逐条音频与评分 | `full/speaker_icl/` |

后台会话为 `seedtts-emilia2-refmask-s12500-sampling-4wpg`。恢复时重跑同目录 `run.sh`，保留 **32 个生成分片**及原评分分片数，已成功生成与评分的记录会跳过。中英文各两条预检生成、评分及审计已通过：英文 WER **0.000%**、SIM **58.765%**；中文 CER **6.818%**、SIM **78.155%**，四条均无超长排除。全量生成阶段已核实 GPU 0–7 各有四个生成进程；正式全量结果如下。


### Qwen Emilia2 refmask step 12,500 全量双采样结果

于 **2026-10-14 07:22:32 UTC** 完成，**3,108 条全部生成并评分，无超长排除**，完整性审计通过，无生成或评分错误。

| 模型 | 数据集 | 合成/总数 | 评分/总数 | CER或WER/% ↓ | SIM/% ↑ | DNSMOS P808 ↑ | DNSMOS OVRL ↑ | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| speaker_icl | seed_tts_eval_en | 1088/1088 | 1088/1088 | 4.420 | 53.353 | — | — | 完整 |
| speaker_icl | seed_tts_eval_zh | 2020/2020 | 2020/2020 | 4.352 | 66.503 | — | — | 完整 |

英文为逐句 WER 均值，中文为逐句 CER 均值，SIM 为余弦相似度乘 100；DNSMOS 未计算。英文最长 **9.28 秒**，中文最长 **13.52 秒**。相较同一实验 step 10,000，英文 WER **5.016% → 4.420%**，中文 CER **4.594% → 4.352%**；英文 SIM **53.470% → 53.353%**，中文 SIM **65.478% → 66.503%**。两组均使用完整测试集。

原始证据：[结果表](../../../UltraEval-Audio/res/emilia2-refmask-step12500-seedtts-sampling-bs1-20261014-8gpu-4wpg/full/results_table.md) · [全量汇总](../../../UltraEval-Audio/res/emilia2-refmask-step12500-seedtts-sampling-bs1-20261014-8gpu-4wpg/full/summary.json) · [完整性审计](../../../UltraEval-Audio/res/emilia2-refmask-step12500-seedtts-sampling-bs1-20261014-8gpu-4wpg/full/audit.json)。


## 16. Emilia2 LFM230M step 6,000：双采样评测（2026-10-14）

Checkpoint 为 `supervised-tts-20261012-emilia2-lfm230m-text2048-spk-refmask-bf16-8gpu-acc2-lr3e-4-bblr1e-4-ep3-wsd/checkpoints/step-00006000`，完整标记已确认；`epoch=0`，已消费 **41,507,406** 条训练样本。模型结构、文本前端初始化、冻结范围、数据划分和训练设置与第 14 节 LFM step 5,000 一致，仍处于 WSD steady 阶段。

沿用第 14 节的 **speaker embedding + ICL、Talker / Code Predictor 双采样、Auto、non-streaming、BF16 + SDPA、batch size 1**；top-k 50、top-p 1.0、temperature 0.9，Talker repetition penalty 1.05。英文 1,088 条、中文 2,020 条，仅运行 sampling。推理仍为 LFM 完整前缀计算、无 KV cache；本次使用 **8 卡 × 每卡 4 个 worker，共 32 分片**，每条随机种子为 `42 + 原始 index`。每进程显存上限 8 GiB、预留 2 GiB；评分最初使用 GPU 0–3、中英文各两分片；后按要求切换为 **8 卡 × 每卡 4 个评分 worker，共 32 个**：英文 16 分片使用 GPU 2–5，中文 16 分片使用 GPU 0、1、6、7。英文 ASR 进程显存占用更高，优先放在空闲显存充足的卡上。四条预检评分及审计通过后自动进入全量生成、评分与审计，沿用相同指标和独立 30 秒超长排除规则。

- 导出模型：`UltraEval-Audio/init_model/emilia2-lfm230m-text2048-spk-refmask-step-00006000/`。
- 结果目录：`UltraEval-Audio/res/emilia2-lfm230m-text2048-spk-refmask-step6000-seedtts-sampling-bs1-20261014-8gpu-4wpg/`。
- 复现 / 恢复：结果目录的 `run.sh`；恢复时保持 32 个生成分片；评分可改变并行度，读取所有已有评分分片并跳过成功记录。切换并行度前先停止旧评分进程，避免同一批样本被两个调度器重复处理。
- 阶段与日志：`pipeline_status.json`、`run.log`、`logs/`；正式结果为 `full/results_table.md`、`full/summary.json`、`full/audit.json`，音频及逐条评分在 `full/speaker_icl/`。

导出权重 SHA256：`aeeaeeb05a04f0166a1045549152d4aba1f5a8efb95fb6a4bf1041de2861ea42`，配套配置与 tokenizer 文件哈希均与 step 5,000 一致。后台会话为 `seedtts-emilia2-lfm230m-s6000-sampling-4wpg`，中英文各两条预检生成、评分及审计已通过：英文 WER **7.143%**、SIM **48.113%**；中文 CER **13.636%**、SIM **64.592%**，均无超长排除。全量评测已完成，结果见下文；预检分数仅验证流程。


### 评分并行度调整

全量生成完成后，评分由 4 个进程切换为 **32 个进程**，保留已完成的 **514 条**成功评分及全部 WAV。`run.sh` 已加入 `--score-gpus 2,3,4,5,0,1,6,7 --score-workers-per-gpu 4`；后台恢复会话为 `seedtts-emilia2-lfm230m-s6000-score32`。生成和评分的每卡 worker 数分别由 `--workers-per-gpu`、`--score-workers-per-gpu` 控制。中英文仍分别使用 Whisper WER、Paraformer CER 和 WavLM SIM，指标定义不变。

恢复适配通过评分跨旧分片复用、失败记录重试、32 worker 路由及既有审计测试，共 **30 项通过**。后续恢复无需手动删除或重新分配旧评分文件；完整性审计仍检查每个样本只有一条成功评分。


### LFM230M step 6,000 全量双采样结果

于 **2026-10-14 09:50:14 UTC** 完成，**3,108 条全部生成并评分，无超长排除**，完整性审计通过。改为 32 worker 后保留旧评分，最终中英文分别有 1,088 / 2,020 条唯一成功记录，无重复或未恢复错误；均分从逐条记录重新核对。

| 模型 | 数据集 | 合成/总数 | 评分/总数 | CER或WER/% ↓ | SIM/% ↑ | DNSMOS P808 ↑ | DNSMOS OVRL ↑ | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| speaker_icl | seed_tts_eval_en | 1088/1088 | 1088/1088 | 9.417 | 43.251 | — | — | 完整 |
| speaker_icl | seed_tts_eval_zh | 2020/2020 | 2020/2020 | 8.727 | 58.582 | — | — | 完整 |

与同一实验 step 5,000 相比，英文 WER **8.682% → 9.417%**、SIM **39.195% → 43.251%**；中文 CER **9.399% → 8.727%**、SIM **55.248% → 58.582%**。相似度均提升，中文错误率下降，英文错误率上升。WER / CER 为逐句算术平均，SIM 为余弦相似度乘 100，DNSMOS 未计算。

原始证据：[结果表](../../../UltraEval-Audio/res/emilia2-lfm230m-text2048-spk-refmask-step6000-seedtts-sampling-bs1-20261014-8gpu-4wpg/full/results_table.md) · [全量汇总](../../../UltraEval-Audio/res/emilia2-lfm230m-text2048-spk-refmask-step6000-seedtts-sampling-bs1-20261014-8gpu-4wpg/full/summary.json) · [完整性审计](../../../UltraEval-Audio/res/emilia2-lfm230m-text2048-spk-refmask-step6000-seedtts-sampling-bs1-20261014-8gpu-4wpg/full/audit.json)。


## 17. Qwen Emilia2 refmask step 15,000：自动等待与双采样评测

每 **30 分钟**检查一次 `supervised-tts-20261012-emilia2-refmask-bf16-16gpu-acc1-lr3e-4-bblr1e-4-ep3-wsd/checkpoints/step-00015000/`，仅在 `COMPLETE` 与 `metadata.json` 存在且元数据 step 为 15,000 时开始。确认后自动以第 13 节组装模型导出 FP32 权重，再运行 **speaker + ICL、双采样、BF16 推理、batch size 1** 的 Seed-TTS 中英文全量评测；生成和评分均为 **8 卡 × 每卡 4 个 worker，共 32 个**。采样参数、评分模型和 30 秒超长排除规则沿用第 15 节。

- 后台会话：`seedtts-emilia2-refmask-s15000-watch`。
- 运行目录：`UltraEval-Audio/res/emilia2-refmask-step15000-seedtts-sampling-bs1-20261014-8gpu-4wpg/`。
- 等待 / 自动启动脚本：运行目录的 `watch-and-run.py`；检查状态与历史为 `watch_status.json`、`watch_events.jsonl`，日志为 `watch.log`。
- 固定评测命令：同目录 `run.sh`；模型为 `UltraEval-Audio/init_model/emilia2-refmask-step-00015000/`。
- 导出日志为 `export.log`，评测状态 / 日志为 `pipeline_status.json` / `run.log`；正式结果在 `full/results_table.md`、`full/summary.json`、`full/audit.json`。

检查器持有单实例锁，避免重复启动。若导出或评测失败，会记录错误并在 30 分钟后重试；评测沿用原目录，成功音频和评分自动跳过。完成全量评分并通过审计后，检查器退出。创建时 step 15,000 尚未出现；随后已自动完成导出、全量生成、评分和审计。


### Qwen Emilia2 refmask step 15,000 全量双采样结果

于 **2026-10-14 13:05:18 UTC** 完成，3,108 条全部生成；按独立 30 秒规则排除英文 3 条、中文 2 条，其余 **3,103 条全部评分**，完整性审计通过，无未恢复错误。

| 模型 | 数据集 | 合成/总数 | 评分/总数 | CER或WER/% ↓ | SIM/% ↑ | DNSMOS P808 ↑ | DNSMOS OVRL ↑ | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| speaker_icl | seed_tts_eval_en | 1088/1088 | 1085/1085 | 5.418 | 55.153 | — | — | 完整（排除 3 条） |
| speaker_icl | seed_tts_eval_zh | 2020/2020 | 2018/2018 | 4.870 | 67.760 | — | — | 完整（排除 2 条） |

英文为逐句 WER 均值，中文为逐句 CER 均值；SIM 为余弦相似度乘 100，DNSMOS 未计算。与 step 12,500 比较时须考虑本轮排除了 5 条超长输出，评分分母不同。

原始证据：[结果表](../../../UltraEval-Audio/res/emilia2-refmask-step15000-seedtts-sampling-bs1-20261014-8gpu-4wpg/full/results_table.md) · [汇总及排除清单](../../../UltraEval-Audio/res/emilia2-refmask-step15000-seedtts-sampling-bs1-20261014-8gpu-4wpg/full/summary.json) · [审计](../../../UltraEval-Audio/res/emilia2-refmask-step15000-seedtts-sampling-bs1-20261014-8gpu-4wpg/full/audit.json)。


## 18. Qwen Emilia2 refmask step 17,500 / 20,000：每十分钟自动检查

监控与第 17 节相同训练 run 下的 `checkpoints/step-00017500/`、`checkpoints/step-00020000/`，每 **600 秒（10 分钟）**各检查一次。目标目录必须有 `COMPLETE`、`metadata.json`，且元数据 step 匹配。确认后自动导出模型并跑 **speaker + ICL、双采样、BF16 推理、batch size 1** 的 Seed-TTS 中英文全量评测，采样参数与评分指标沿用 step 15,000；生成与评分均为 **8 卡 × 每卡 4 个 worker，共 32 个**。

| 目标 | 后台会话 | `UltraEval-Audio/res/` 下的目录 |
| --- | --- | --- |
| step 17,500 | `seedtts-emilia2-refmask-s17500-watch` | `emilia2-refmask-step17500-seedtts-sampling-bs1-20261014-8gpu-4wpg/` |
| step 20,000 | `seedtts-emilia2-refmask-s20000-watch` | `emilia2-refmask-step20000-seedtts-sampling-bs1-20261014-8gpu-4wpg/` |

两轮按顺序运行：17,500 等 15,000 完成并通过审计，20,000 等 17,500 完成并通过审计，避免并行占用同一组 GPU。完整 checkpoint 的检查、等待前轮、导出、评测和失败重试均由各目录的 `watch-and-run.py` 执行；单实例锁防止重复启动。导出或评测失败后记录错误，10 分钟后重试，复用已有成功音频与评分。已完成并通过审计的任务不会因重启检查器而重复执行。

各目录的 `watch_status.json` 为当前状态，`watch_events.jsonl` 为检查历史，`watch.log` / `export.log` / `run.log` 保存日志；固定评测命令为 `run.sh`。正式结果仍在 `full/results_table.md`、`full/summary.json`、`full/audit.json`，音频在 `full/speaker_icl/`。导出模型分别为 `UltraEval-Audio/init_model/emilia2-refmask-step-00017500/` 与 `emilia2-refmask-step-00020000/`。创建时两份 checkpoint 均未出现，首次检查状态为 `waiting_checkpoint`。


### Step 22,500 自动接续（2026-10-15）

已将同一 Qwen Emilia2 refmask 训练的 `checkpoints/step-00022500/` 加入自动评测，每 **10 分钟**检查一次。完整标记与元数据核验通过、且 step 20,000 已完成并通过审计后，自动导出并运行相同的 **speaker + ICL 双采样**全量评测：BF16 推理、batch size 1，生成和评分各 32 worker（8 卡 × 每卡 4 个）。失败后每 10 分钟重试并复用成功记录。

后台会话为 `seedtts-emilia2-refmask-s22500-watch`，运行目录为 `UltraEval-Audio/res/emilia2-refmask-step22500-seedtts-sampling-bs1-20261015-8gpu-4wpg/`。检查器、固定评测命令、当前状态和历史分别为该目录的 `watch-and-run.py`、`run.sh`、`watch_status.json`、`watch_events.jsonl`；日志与结果布局同上。导出模型为 `UltraEval-Audio/init_model/emilia2-refmask-step-00022500/`。设置时 step 20,000 已完成，step 22,500 尚未出现。


## 19. Step 17,500 / 20,000 等权平均模型（2026-10-15）

按 **50% × step 17,500 + 50% × step 20,000** 对同一 Qwen Emilia2 refmask 实验的已导出模型进行参数平均。使用 `scripts/average_exports.py`，在 CPU 上以 FP32 计算，对全部对应浮点张量平均；非浮点 buffer 必须相同并原样保留。先核对源导出文件哈希、配置 / tokenizer、tensor keys、shape、dtype 与有限性，再写入独立目录。原 checkpoint 与源导出模型均保留。

- 平均模型：`UltraEval-Audio/init_model/emilia2-refmask-average-step17500-step20000/`。
- 权重 SHA256：`81b6f374e7ca8ee10f7f69ca6d7e37de218287767a0f6f5a9d48b37780297770`。
- `export.json` 记录两份源导出的完整身份及各自 `weight=0.5`；合并产物是推理模型，不包含可恢复训练的 optimizer / scheduler 状态。
- **478 张量逐张量核验通过**：81 张源张量相同且合并后不变，397 张不同并严格符合 FP32 等权平均。冻结的文本前端、speaker encoder 保持对应源权重，独立音频 codec 复用已有冻结模型。
- 回归测试 **7 项通过**，覆盖平均值、原张量不被修改、冻结权重 / 整数 buffer 保留、不兼容或非有限输入拒绝、完整导出身份与加载校验；Ruff 检查通过。

评测沿用 **speaker + ICL、Talker / Code Predictor 双采样、BF16 + SDPA、batch size 1、Auto、non-streaming**；top-k 50、top-p 1.0、temperature 0.9，Talker repetition penalty 1.05。Seed-TTS 英文 1,088 条、中文 2,020 条，生成与评分各 **8 卡 × 每卡 4 worker，共 32 个**。指标及独立 30 秒超长排除规则不变，DNSMOS 不计算。

运行目录为 `UltraEval-Audio/res/emilia2-refmask-average-step17500-step20000-seedtts-sampling-bs1-20261015-8gpu-4wpg/`，后台会话为 `seedtts-emilia2-refmask-avg17500-20000`。`merge.sh` 保存合并命令（目标目录必须不存在），`merge_verification.json` 保存逐张量核验；`run.sh` 为评测 / 恢复命令，阶段和日志为 `pipeline_status.json`、`run.log`、`logs/`，正式结果在 `full/results_table.md`、`full/summary.json`、`full/audit.json`。中英文各两条预检生成、评分及审计通过：英文 WER **0.000%**、SIM **64.755%**；中文 CER **6.818%**、SIM **78.992%**，无超长排除。全量评测随后已完成，见下文；预检仅验证流程。

Step 22,500 的每十分钟自动检查继续运行；其启动条件改为本轮平均模型评测完成并通过审计，再接续评测，避免与本轮争抢 GPU。


### 平均模型的源模型对照

| 模型（均为双采样 speaker + ICL） | 英文 WER/% ↓ | 英文 SIM/% ↑ | 中文 CER/% ↓ | 中文 SIM/% ↑ | 超长排除（英 / 中） |
| --- | ---: | ---: | ---: | ---: | --- |
| step 17,500 | 3.368 | 58.359 | 3.272 | 67.995 | 0 / 1 |
| step 20,000 | 3.367 | 59.638 | 3.244 | 69.735 | 0 / 1 |

两份源模型均生成全部 3,108 条音频，英文评分 1,088 条、中文评分 2,019 条。来源：[step 17,500 结果表](../../../UltraEval-Audio/res/emilia2-refmask-step17500-seedtts-sampling-bs1-20261014-8gpu-4wpg/full/results_table.md)、[step 20,000 结果表](../../../UltraEval-Audio/res/emilia2-refmask-step20000-seedtts-sampling-bs1-20261014-8gpu-4wpg/full/results_table.md)。平均模型须完成独立全量评分后再比较，分别保留各自超长排除数。


### 两模型平均的全量结果

于 **2026-10-15 01:58:13 UTC** 完成，3,108 条全部生成并评分，无超长排除，完整性审计通过；均分从全部唯一成功评分记录重新核验。

| 模型 | 数据集 | 合成/总数 | 评分/总数 | CER或WER/% ↓ | SIM/% ↑ | DNSMOS P808 ↑ | DNSMOS OVRL ↑ | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| speaker_icl | seed_tts_eval_en | 1088/1088 | 1088/1088 | 3.035 | 59.818 | — | — | 完整 |
| speaker_icl | seed_tts_eval_zh | 2020/2020 | 2020/2020 | 2.662 | 70.013 | — | — | 完整 |

相较单独 step 20,000，两模型平均的英文 WER **3.367% → 3.035%**、SIM **59.638% → 59.818%**，中文 CER **3.244% → 2.662%**、SIM **69.735% → 70.013%**。平均模型未排除超长样本，单独 step 20,000 排除 1 条中文，比较时需考虑分母变化。

原始证据：[结果表](../../../UltraEval-Audio/res/emilia2-refmask-average-step17500-step20000-seedtts-sampling-bs1-20261015-8gpu-4wpg/full/results_table.md) · [汇总](../../../UltraEval-Audio/res/emilia2-refmask-average-step17500-step20000-seedtts-sampling-bs1-20261015-8gpu-4wpg/full/summary.json) · [审计](../../../UltraEval-Audio/res/emilia2-refmask-average-step17500-step20000-seedtts-sampling-bs1-20261015-8gpu-4wpg/full/audit.json)。


## 20. Step 17,500 / 20,000 / 22,500 三模型平均：自动接续

新增三份原始模型的**等权平均，各占 1/3**，不会将两模型平均结果与 step 22,500 再做 50/50 平均。`scripts/average_exports.py` 已支持至少两份源导出，校验每份模型的配置、tokenizer、形状、dtype、哈希与有限值；浮点张量以 FP32 等权累加，所有源中完全相同的张量直接保留，避免冻结权重因 1/3 舍入而变化，整数 buffer 必须一致。三模型均分、第三份模型不兼容拒绝、冻结权重严格不变及既有两模型行为测试共 **10 项通过**，Ruff 检查通过。

当前尚缺 step 22,500，检查器每 **10 分钟**检查源导出；先由已有的 step 22,500 检查器完成导出、单模型评测和审计，再自动运行三模型合并、逐张量核验和评测，避免任务重叠。逐张量核验将平均结果与 FP64 参考均值比较，检查相同源张量严格不变，通过后才开始评测。任一步失败会记录错误并在 10 分钟后重试，已有成功音频和评分会复用。

- 后台会话：`seedtts-emilia2-refmask-avg17500-20000-22500-watch`。
- 独立模型：`UltraEval-Audio/init_model/emilia2-refmask-average-step17500-step20000-step22500/`。
- 运行目录：`UltraEval-Audio/res/emilia2-refmask-average-step17500-step20000-step22500-seedtts-sampling-bs1-20261015-8gpu-4wpg/`。
- `watch-and-run.py` 自动等待与接续；`watch_status.json` / `watch_events.jsonl` 记录状态和检查历史；`watch.log` / `merge.log` / `run.log` 保存日志。
- `merge.sh` 为合并命令（先写入 `.exporting` 目录再由检查器重命名）；`verify-merge.py` / `merge_verification.json` 为逐张量核验及记录；`run.sh` 为固定评测与恢复命令。
- 评测继续使用 **speaker + ICL 双采样、BF16 + SDPA、batch size 1、Auto、non-streaming**；采样参数与第 19 节一致，生成和评分各 **32 worker（8 卡 × 每卡 4 个）**。正式结果在 `full/results_table.md`、`full/summary.json`、`full/audit.json`，音频在 `full/speaker_icl/`，使用相同指标和独立 30 秒排除规则。
