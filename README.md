# LM-TTS-Training

基于 Qwen3-TTS 的非流式训练框架，读取 `tts-data-pipeline` 发布的 unified Lance features。默认冻结 text frontend 和 speaker encoder，训练 Talker 和音频预测模块。支持 FSDP2 多机多卡、BF16 / FP32、断点恢复，以及独立的生成音频评分。

日常流程：**修改实验配置 → 准备模型 → 构建数据索引 → 本地或 ACP 训练 → 查看日志与评分**。下面的命令均在仓库根目录执行。

> 当前正式数据尚未全部发布。`configs/data.yaml` 中的路径是预期发布位置，不能直接假定全部可用；构建前需要确认所选数据集的 codec 和 speaker manifest 都已完成。

## 1. 选择实验和配置

```bash
cd /workspace/workspace/yanglin/LM-TTS-Training
PY=/workspace/workspace/yanglin/envs/lm-tts/bin/python
CFG=configs/train-bf16.yaml
```

使用 FP32 时，将 `CFG` 改为 `configs/train-fp32.yaml`。新实验可以复制一个 `train-*.yaml` 到 `configs/`，修改 `train.run_name`，再让 `CFG` 指向新文件。

所有配置直接继承一个 [base.yaml](configs/base.yaml)，无需追踪多级依赖。ACP 和训练参数放在同一份实验配置中；数据绑定、生成和评分各有自己的入口。

| 要改什么 | 修改位置 |
| --- | --- |
| 实验名称、精度、学习率、batch 预算 | 实验 YAML 的 `train` 区块 |
| ACP 节点数、镜像 | 同一实验 YAML 的 `acp` 区块；提交时也可用 `--nodes` |
| 公共路径、通信设置、训练默认值 | `configs/base.yaml` |
| selection、各数据集 codec / speaker manifest | [configs/data.yaml](configs/data.yaml) |
| 新数据索引的名称 | 实验 YAML 的 `data.build_id` |
| 生成参数、生成设备和精度 | [configs/synthesis.yaml](configs/synthesis.yaml) |
| 评分项目、评分模型、试听数量 | [configs/evaluation.yaml](configs/evaluation.yaml) |

例如，一个实验文件可以只包含：

```yaml
extends: base.yaml

train:
  run_name: unified-bf16-exp01
  precision: bf16
  max_batch_frames: 6000
  max_batch_tokens: 16000

acp:
  nodes: 4
```

配置规则：`extends` 相对于当前 YAML；字典合并，列表整体替换；`${paths.project}` 等引用在合并后解析。数据构建会沿用实验配置解析后的 `paths`。独立运行的生成、评分入口读取各自的 YAML，因此多流程共用的路径建议改在 base 中。

**输出目录统一为 `paths.runs / train.run_name`**。默认是 `/workspace/LM-TTS-Training-Runs/<run_name>/`，checkpoint、TensorBoard 和日志自动派生，不用重复填写路径。`yanglin/LM-TTS-Training-Runs` 是指向该目录的软链接。

## 2. 准备环境和模型

已有环境和已完成的组装模型可以复用。首次准备环境：

```bash
export ENV_PREFIX=/workspace/workspace/yanglin/envs/lm-tts
export CONDA_BIN=/home/yanglin/miniforge3/bin/conda
bash scripts/setup_env.sh
```

训练模型在本地提前组装：

```bash
$PY -m scripts.prepare_models --config "$CFG"
```

该命令从 `/workspace/model` 复制基础权重，校验后组装；可用 `--source` 指定其他本地来源。训练启动时只加载带 `ASSEMBLY_COMPLETE` 标记的产物。

| 目录 | 内容 |
| --- | --- |
| `assets/base/` | 复制的基础模型、官方 TTS 模板和 codec |
| `assets/assembled/` | 可直接加载训练的组装模型 |
| `assets/evaluation/` | Whisper、DNSMOS、WavLM 评分模型 |
| `../envs/lm-tts/` | 仓库外的 Conda 环境 |

评分模型缺失时执行 `$PY -m scripts.download_eval_models`，默认写入 `assets/evaluation/`；自定义位置使用 `--output`，并同步修改 `paths.evaluation_models`。训练与评分模型分开存放，`assets/` 已被 gitignore 排除。

环境使用 Python 3.11、PyTorch 2.8 / CUDA 12.6，直接依赖版本见 [requirements.lock.txt](requirements.lock.txt)。

## 3. 构建训练数据

检查 `configs/data.yaml` 的 `selection_manifest` 和 `bindings`。每个绑定选择同一数据集、同一 release 的 codec 和 speaker embedding manifest，必须为 `status: complete`，不能指向 `.incomplete`。先使用部分数据时，保留已完成的数据集绑定即可。

- **text / language** 已包含在发布后的 codec 表中，文本来自固定 selection 的 `selected_text` 或原始 samples 的 `text`；训练侧不生成转写。
- **codec** 是每帧 16 个码本的离散 token。
- **speaker embedding** 是官方冻结 encoder 生成的 1024 维向量。

```bash
$PY -m scripts.build_unified --config "$CFG"
```

完成后生成 `data/builds/<build_id>/manifest.json`，训练通过实验配置的 `data.build` 读取它。Build 固定 feature 版本，建立 speaker 定位和 text token 缓存，只写训练仓库；它仍引用 unified 的源文件，因此源文件需要保留。

更换 selection、feature 版本或 tokenizer 时，修改 `data.build_id` 创建新 build。已有 build 不会被覆盖；同一 build 可供 BF16 和 FP32 训练复用。

**`data.evaluation: no_holdout` 表示这份数据没有划出独立验证集。** 训练仍记录 loss，但不会产生 validation loss，`train.eval_every` 此时不起作用。当前构建器只生成这种 build；启用训练中验证需要另外发布真正隔离的训练 / 验证 build，并设置 `data.val_build`，不能只改这个字符串。下文的独立生成评分仍可使用。

## 4. 启动训练

本地与 ACP 选择一种启动方式。同一 run 不应同时启动多个训练任务。

### 本地训练

```bash
NPROC_PER_NODE=8 bash scripts/run_train.sh --config "$CFG"
# 中断后，保持相同实验配置和卡数
NPROC_PER_NODE=8 bash scripts/run_train.sh --config "$CFG" --resume latest
```

单卡将 `NPROC_PER_NODE` 设为 `1`。多机时，各节点运行同一命令，设置相同的 `NNODES`、`MASTER_ADDR`、`MASTER_PORT` 和 `NPROC_PER_NODE`，并为各节点设置不同的 `NODE_RANK`（从 `0` 开始）。

### ACP 提交

平台 CLI、凭据、profile 和密钥位于仓库外 `../acp/`，提交与适配脚本位于 `scripts/acp/`。初始化说明见 [ACP README](scripts/acp/README.md)。

```bash
# 检查实际请求，不创建任务
$PY -m scripts.acp.submit --config "$CFG" --nodes 4
# 确认配置后创建任务：当前规格每节点 8 张 A800
$PY -m scripts.acp.submit --config "$CFG" --nodes 4 --submit
# 恢复同一个 run
$PY -m scripts.acp.submit --config "$CFG" --nodes 4 --resume latest --submit
```

SSH 免密和平台 TensorBoard 默认开启。提交器将代码快照、有效训练配置、请求和 job ID 保存在该 run 的 `submissions/<timestamp>/`，任务运行这份代码快照。`acp` 区块只由提交器读取，训练核心不依赖平台代码。

### 精度与恢复约束

BF16 使用 FlashAttention，FP32 使用 SDPA；两者都保留 FP32 master weights、梯度归约和 cross entropy，关闭 TF32。组批预算按每张卡计算，显存不足时降低 `max_batch_frames` / `max_batch_tokens`。数据提取精度与训练精度分别记录。

恢复时保持相同 world size、模型、数据、batch、优化器和学习率计划；可以增加 `max_steps`，但不能超过原有 `schedule_steps`。改变这些训练语义时使用新 run。

通信默认值集中在 base 的 `environment`：`NCCL_IB_TIMEOUT=22`、`NCCL_IB_RETRY_CNT=13`、`NCCL_IB_AR_THRESHOLD=0`，按当前[商汤 ACP 建议](https://www.sensecore.cn/help/docs/cloud-foundation/compute/acp/acpUserGuide/acpEnvironmentVariable)设置。本地显式 shell 环境优先；ACP 将配置值注入各节点。

## 5. 查看日志与 TensorBoard

每个 run 的文件用途如下：

| 位置 | 内容 |
| --- | --- |
| `checkpoints/step-XXXXXXXX/` | 完整训练状态，默认保留最近两份 |
| `checkpoints/latest` | 保存最后一个完整 checkpoint 的目录名 |
| `tensorboard/` | 训练和评分事件 |
| `logs/` | 每个节点的启动及训练输出 |
| `metrics.jsonl` | 可用于重建曲线的指标记录 |
| `config.json`、`config.yaml` | 有效配置和源实验 YAML |
| `submissions/` | ACP 提交记录及代码快照 |

平台 TensorBoard 挂载统一的 Runs 目录，页面中按 run 名选择实验。也可以在开发机打开单个 run：

```bash
$PY -m scripts.tensorboard --run unified-bf16
```

命令打印访问地址，远程访问时转发 `6006` 端口。修改过 `train.run_name` 时使用对应名称，也可传绝对路径。若不需要平台页面，将 `acp.tensorboard` 改为 `local`。平台模式要求 `paths.runs` 位于 AFS 一级目录；独立仪表盘可使用嵌套目录。

默认训练曲线按三组组织，可在 `train.tensorboard.groups` 中选择，空列表关闭训练 scalar：

| 组 | 用来看什么 |
| --- | --- |
| `optimization` | 总 loss、首码本 / residual CE、两组学习率、梯度范数 |
| `performance` | 音频秒吞吐、每步耗时、数据等待、rank 0 每步峰值显存 |
| `batch` | 全局样本数、音频秒数、codec token 数，包含梯度累积 |

`Custom Scalars` 提供 loss、学习率和耗时组合图；`Text` 显示配置。执行下一节评分后，会增加 WER / CER、音质、相似度和参考 / 生成音频试听。音质指标不会随训练自动生成。

对于有指标记录但缺少事件的 run：

```bash
$PY -m scripts.tensorboard --run unified-bf16 --rebuild-only
```

重建结果在 `tensorboard/restored/`，原始训练事件保留。重复执行会重新读取当前记录、更新新增步数和评分预览；同一步的重复训练记录采用最后一次值。重建后重新启动独立仪表盘，选择 restored 对应的 run 查看。日志中未记录的指标无法恢复，建议在训练或评分停止后重建。

## 6. 生成与评分

评分独立于训练进程，在保存 checkpoint 后执行。准备固定的 `prompts.json`，例如：

```json
[
  {
    "id": "zh-001",
    "text": "这是用于检查语音生成效果的一句话。",
    "language": "zh",
    "reference_audio": "references/speaker-001.wav"
  }
]
```

`reference_audio` 相对于 prompts 文件，使用同说话人的另一条录音。`language` 使用 `zh`、`en` 等语言代码，用于 ASR 评分与语言汇总；生成侧使用自动语言识别。评估样本应与训练隔离，并固定集合以便比较不同 checkpoint。

```bash
# 使用实际的 run 名和已保存的 checkpoint 步数
RUN=/workspace/LM-TTS-Training-Runs/unified-bf16
$PY -m scripts.export_checkpoint --checkpoint "$RUN/checkpoints/step-00000500" \
  --assembled-model assets/assembled/qwen3-tts-frozen-conditioning --output "$RUN/export-500"
$PY -m scripts.synthesize --model "$RUN/export-500" --config configs/synthesis.yaml \
  --prompts prompts.json --output "$RUN/generated-500"
$PY -m scripts.evaluate --config configs/evaluation.yaml --pairs "$RUN/generated-500/pairs.json" \
  --output "$RUN/evaluations/step-500" --tensorboard "$RUN/tensorboard" --step 500
```

导出时必须使用该 checkpoint 训练时的 assembled 模型。生成、导出和评分输出目录均需为新目录，重复运行时更换名称。

| 评分 | 含义 |
| --- | --- |
| WER / CER | Whisper small 转写后的词 / 字符错误率，越低越好；中文主要看 CER |
| DNSMOS SIG / BAK / OVRL | 预测的语音、背景和总体音质，越高越好；不等同人工 MOS |
| Speaker similarity | WavLM x-vector 余弦相似度，越高通常表示越相似，没有通用及格阈值 |

报告包含逐条结果、按语言汇总、模型 / 音频哈希及参数。TensorBoard 默认试听前两对音频的前 20 秒，在 `evaluation.yaml` 的 `tensorboard` 中调整；裁剪只影响试听，完整音频参与评分。已有生成音频也可直接评分：在 prompts 的字段基础上增加 `audio`，组成 `pairs.json`。

## 开发与验证

代码按职责划分：`qwen3_train/data/` 负责构建、读取和组批，`models/` 负责模型与输入协议，`objectives/` 负责 loss，`training/` 负责分布式更新和恢复，`evaluation/` 负责生成音频评分。`scripts/` 是操作入口，`scripts/acp/` 是平台适配层。代码与配置注释使用英文，操作文档使用中文。

```bash
$PY -m pytest -q
$PY -m ruff check qwen3_train scripts tests
$PY -m ruff format --check qwen3_train scripts tests
```

需要重新验证集群启动和恢复时，可以显式提交微型模型验证任务：

```bash
$PY -m scripts.acp.submit --validate --nodes 4 --run-name check-32gpu --submit
```

已完成的 ACP 4 节点 32 张 A800 作业 `pt-ughagq38` 验证了 BF16 / FP32 连续训练与 2+2 步恢复：权重逐位一致，冻结模块未变，评分接口通过。报告在 `/workspace/LM-TTS-Training-Runs/acp-20260930-32gpu/result.json`。该结果证明微型模型功能，不代表完整模型吞吐。另已用独立完成的 CSEMOTIONS features 验证真实数据读取，正式 selection 仍待处理完成。

更多证据见[验证记录](docs/training/unified-refactor.md)。历史 run 按[保留规则](runs/README.md)保存，旧实验文档仅作历史记录，当前操作以本 README 为准。
