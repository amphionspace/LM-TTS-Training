# All16 frozen-conditioning 训练与 Seed-TTS 评测

本报告记录 all16 frozen-conditioning 的训练设置、数据、模型初始化与冻结范围，以及最终 checkpoint **`step-00047382`** 的 Seed-TTS 评测。

状态快照：**2026-10-08 05:34 UTC**。本轮已完成 3 个 epoch；其 speaker + ICL 采样评测已完成，两个模块均 greedy 的全量对照正在运行。speaker-only 按要求暂停，不继续生成或评分。运行状态以后文结果目录中的文件为准。

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
- WER/CER 使用逐句错误率的算术平均，不是全语料累计编辑距离比例。不剔除长音频或高错误率样本。
- 当前 Seed-TTS 流程没有 DNSMOS / 主观 MOS；SIM 衡量说话人相似度，不能替代音质评分。

| 模型与解码 | 英文 WER/% ↓ | 中文 CER/% ↓ | 英文 SIM×100 ↑ | 中文 SIM×100 ↑ |
| --- | ---: | ---: | ---: | ---: |
| 历史复现 Qwen3-TTS 0.6B，speaker + ICL，采样 | 4.521 | 1.116 | 70.787 | 76.666 |
| 历史复现 Qwen3-TTS 1.7B，speaker + ICL，采样 | 1.735 | 0.958 | 71.278 | 76.958 |
| 本轮 frozen-conditioning step 47382，speaker + ICL，采样 | **17.072** | **14.169** | **64.253** | **76.158** |
| 同一 checkpoint，speaker + ICL，两个模块均 greedy | 运行中 | 运行中 | 运行中 | 运行中 |

历史值取 [2026-09-07 Qwen 复现报告](../../../UltraEval-Audio/replication/voice_clone_20260907.md)的“表 1：不剔除异常，全部样本”，不是论文成绩，也不是过滤后的更好成绩。历史原始音频 / 逐条评分未在当前仓库保留，本次没有重新审计历史数值。

**比较限制：**历史 Qwen 使用显式 English/Chinese language ID 与默认 streaming 文本布局；本轮按训练协议使用 `language="Auto"`、`non_streaming_mode=True`，没有显式 language ID。因此这是现有实验成绩对比，不能将全部差距归因于权重或训练数据。当前模型的内容错误率明显更高，中文 SIM 接近并不表示读对了文本。

采样基线：Talker / Code Predictor 均启用 sampling，`top_k=50, top_p=1.0, temperature=0.9`；Talker `repetition_penalty=1.05, max_new_tokens=2048`。Batch size 1，逐条 seed 为 `42 + split 内 index`。Greedy 只关闭两处采样，其余参数保持相同。

### Greedy 对照的依据

独立抽取的中英文各 48 条中，两处采样都关闭后，WER/CER 从 **15.20% / 13.33%** 降至 **11.52% / 7.44%**，因此继续进行全量对照。这只是小样本结果，尚无该子集的 SIM / 音质对比，不能作为全量成绩。

已核对导出权重、输入协议及 ASR 评分链路，未发现足以解释当前高错误率的实现错误；speaker 自参考依赖仍是待验证的训练问题。详细检查保存在 [评测证据](evidence/seed-tts-step47382.json)，这里不展开诊断过程。

## 6. 导出、运行 greedy 和查看结果

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
| `res/all16-step47382-seedtts-greedy-20261008-8gpu` | 正在运行的全量 greedy ICL |
| `log/seed-tts-icl-greedy.log` | Greedy 调度日志 |

数据 HF revision 为 `185352e5da255d788097b835ea7dd11ae4f74976`。下载版本、文件哈希与环境版本记录保留在 UltraEval。中文评分使用 JunHowie 的完整 FP32 SeACo-Paraformer 镜像；已核对模型加载完整，但未与原始 ModelScope 发布文件逐字节比较。

当前后台会话为 `seedtts-all16-s47382-icl-greedy`，8 个 GPU 分片，每进程 16 GiB 显存上限。启动命令如下；任务活跃时不要重复启动，意外中断后可用同一命令接续：

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

文档中的运行状态是上述时间点的快照；greedy 完成后应从通过审计的 `full/summary.json` 更新比较表，不能使用小样本结果代替。
