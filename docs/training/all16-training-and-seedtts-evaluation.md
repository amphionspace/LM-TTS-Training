# All16 frozen-conditioning 训练与 Seed-TTS 评测

本报告记录 all16 frozen-conditioning 的训练设置、数据、模型初始化与冻结范围，以及最终 checkpoint **`step-00047382`** 的 Seed-TTS 评测。

更新于 **2026-10-08 07:22 UTC**。本轮已完成 3 个 epoch；speaker + ICL 的采样和双 greedy 两组全量评测均已完成并通过审计，每组 3,108 条。speaker-only 保持暂停。

## 1. 实验与目录怎么对应

| 实验 | 配置（相对仓库根目录） | 初始组装模型 | 训练作业 |
| --- | --- | --- | --- |
| frozen-conditioning，本轮 | `configs/supervised-tts-20260929-all16-bf16-32gpu-lr3e-4-bblr1e-4-ep3-wsd.yaml` | `assets/assembled/qwen3-tts-frozen-conditioning` | `pt-blb39rgw`；最终 step 47382 |

本实验 YAML 的 `train.run_name` 与文件名去掉 `.yaml` 一致；输出统一为 `/workspace/LM-TTS-Training-Runs/<run_name>/`。`/workspace/workspace/yanglin/LM-TTS-Training-Runs` 是其软链接。

| Run 内路径 | 用途 |
| --- | --- |
| `metrics.jsonl`、`tensorboard/` | 训练、验证、吞吐和数据等待指标，以及 TensorBoard |
| `checkpoints/` | 可直接恢复的完整分布式训练状态 |
| `archived-checkpoints/` | 本轮的独立 checkpoint 归档，避免被轮转删除 |
| `logs/` | 节点日志与保留的验证证据 |
| `submissions/` | 实际提交配置、代码快照和作业身份；复核历史训练优先读这里 |

本轮训练目录轮转保留最近两份，另有每小时归档任务；评测读取归档，不依赖轮转目录。下载的基础模型在 `assets/base/`，组装模型在 `assets/assembled/`，评测模型和产物在相邻的 `UltraEval-Audio`，三者各有用途。

## 2. 使用了哪些数据

训练数据与固定划分：

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

本轮是“文本 LM 主干迁移 + 新音频模块”，不是整个模型从零训练。组装在本地提前完成，seed 42、FP32 保存；训练直接加载 assembled 权重，恢复时加载 checkpoint，不重新组装。

| 模块 | 结构 / 初始化来源 | 是否更新 |
| --- | --- | --- |
| 文本 tokenizer | 固定分词规则与 TTS 特殊 token，配套组装模型 | 无可训练参数 |
| 文本 embedding | 151,936 × 2,048；`Qwen3-TTS-12Hz-0.6B-Base` 完整表 | **冻结** |
| 文本 projector | 2,048 → 2,048 → 1,024，SiLU；同一 TTS checkpoint | **冻结** |
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

主模型总参数 **914,643,008**，其中可训练 **588,329,216**、冻结 **326,313,792**；codec 不计入此总数。没有加载官方 TTS 的 Talker / 音频 embedding / Code Predictor 权重，也没有加载纯文本 LM 的文本输出 head。

输入路径：文本 IDs 查表并经过 text projector；每个音频帧的 16 组 codec IDs 分别查表，所得 16 个向量相加成一个 1024 维向量。它们按“角色与控制前缀 → speaker → 完整文本 → codec BOS → 历史音频帧”排列，经因果 attention 融合。文本位置加 codec PAD，音频位置加 projected text PAD；这是协议占位向量，与 batch 补齐长度的 padding 不同。

这 16 张音频 embedding 都参与训练，residual 表也复用于 Code Predictor 的码本内预测；它们与输出 head **不 tie**。训练用真实历史帧做 teacher forcing，Talker 与 Code Predictor 同时计算 loss、共同反向传播，二者之间不 detach。

## 4. 怎么训练与恢复

本轮训练设置：

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

损坏的单条数据跳过并记录；某 rank 整个 microbatch 无效时全体协调跳过，避免通信挂起。配置错误或模型计算异常不会伪装成坏数据吞掉。训练记录包括数据等待时间 / 占比、吞吐和显存；预算预检时实际显存约 66–67 GiB/卡。临近结束 step 47380 记录约 7.55 秒/步、10,794 音频秒/秒，属于当时单步测量，不代表含验证与保存的全程平均。

Code Predictor 使用 SDPA，是因为当前环境的大 batch FA2 测试曾出现非法访问 / NaN；Talker 仍使用 FA2。完整验证、32 卡恢复逐位一致证据见 [本轮训练记录](all16-20261001.md)。

以下命令在 **LM-TTS-Training 根目录**运行。复现新实验先复制 YAML、修改 `train.run_name`，不要对已完成的正式 run 再提交一次：

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

本轮最终 teacher-forcing 验证：首码本 CE **0.906703**、residual CE **5.244773**、加权 loss **2.480135**，跳过 / 连带丢弃样本均为 0。这类验证随训练自动执行；下文自由生成的 ASR / SIM 是独立评测流程，loss 降低不等于 WER/CER 必然良好。

## 5. Seed-TTS 方法与历史 Qwen 对比

本次使用本轮最终归档 `step-00047382`，67 个归档文件已核对清单与 SHA-256。导出为 FP32，以 BF16 + SDPA 推理。导出模型在 `UltraEval-Audio/init_model/all16-frozen-conditioning-step-00047382`，没有修改训练 checkpoint。

评测只用 Seed-TTS：英文 **1,088** 条、中文 **2,020** 条。参考音频为 `audio`、参考转写为 `prompt_text`，待生成文本为 `text`。speaker + ICL 同时提供参考 speaker embedding、参考 codec 和参考文本；目标音频不作为生成输入。speaker-only 仅提供 speaker embedding，目前暂停，没有完整成绩。

- 英文：Whisper-large-v3 WER；中文：SeACo-Paraformer CER。
- 相似度：WavLM-large + ECAPA，SIM 为余弦相似度乘 100。
- WER/CER 使用逐句错误率的算术平均，不是全语料累计编辑距离比例。本轮结果保留全量；历史 Qwen 使用下述异常过滤口径。
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

### Step 32,500 对照（2026-10-08 启动）

使用同一训练 run 的完整归档 `step-00032500`，67 个文件已通过 SHA-256 校验。导出模型为 `UltraEval-Audio/init_model/all16-frozen-conditioning-step-00032500`；模型配置、文本 tokenizer 与 step 47,382 一致。

按 **greedy → 采样** 顺序评测，两组均为 speaker + ICL、完整 3,108 条、8 卡、每推理进程 16 GiB 上限。Greedy 完成生成、ASR/SIM 和完整审计后才启动采样；任一阶段失败即停止接续。两组保持本节其余生成参数不变，speaker-only 不运行。

以下路径相对 UltraEval-Audio：

| 内容 | 路径 |
| --- | --- |
| Greedy 结果 | `res/all16-step32500-seedtts-greedy-20261008-8gpu` |
| 随后采样的结果 | `res/all16-step32500-seedtts-sampling-20261008-8gpu` |
| 顺序执行日志 | `log/seed-tts-step32500-sequence.log` |

后台会话为 `seedtts-all16-s32500-sequence`；顺序启动命令保存在 greedy 目录的 `run-sequence.sh`，阶段见 `sequence-stage.txt`。各组完成后分别读取 `full/summary.json` 和 `full/audit.json`。该对照尚无完整成绩，不混入上表 step 47,382 的结果。

完整结果见 [采样汇总](../../../UltraEval-Audio/res/all16-step47382-seedtts-20261008-8gpu/full/summary.json)、[greedy 汇总](../../../UltraEval-Audio/res/all16-step47382-seedtts-greedy-20261008-8gpu/full/summary.json)和 [greedy 审计](../../../UltraEval-Audio/res/all16-step47382-seedtts-greedy-20261008-8gpu/full/audit.json)。早期小样本诊断保存在 [评测证据](evidence/seed-tts-step47382.json)，正式结论以上述全量结果为准。

### 一个 epoch 附近的对照：step 17,500

现有完整归档没有 step 15,000 / 16,000；最接近 16,000 的是 **step 17,500**。其 checkpoint 记录已完成 1 个 epoch，再消费 13,686,509 / 128,091,959 条训练样本，约 **1.107 epoch**。

已安排自动顺序：**32,500 greedy → 32,500 采样 → 17,500 greedy → 17,500 采样**。每组必须完成全量评分并通过审计，才会开始下一组；前序失败则停止接续。17,500 沿用相同 speaker + ICL、数据、采样参数、8 卡和每进程 16 GiB 上限，开始前自动核验归档并导出模型。

结果分别写入 UltraEval-Audio 的 `res/all16-step17500-seedtts-greedy-20261008-8gpu` 和 `res/all16-step17500-seedtts-sampling-20261008-8gpu`。后台会话 `seedtts-all16-s17500-sequence` 当前等待 32,500 两组完成；调度日志为 `log/seed-tts-step17500-sequence.log`，阶段在 greedy 目录的 `sequence-stage.txt`。本段为已启动的接续安排，尚无该 checkpoint 的评测成绩。

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

本次两组均为 `complete`，审计通过；上表使用全量汇总结果。
