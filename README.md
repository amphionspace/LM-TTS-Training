# LM-TTS-Training

基于 Qwen3-TTS 的非流式训练框架，接入 `tts-data-pipeline` 发布的 unified Lance features。默认冻结官方 text frontend 和 speaker encoder，训练 Talker 和音频预测模块。训练使用 FSDP2，支持单机、多机、BF16 和 FP32；旧 JSONL / NPZ 训练入口已移除。

## 目录与资源

| 位置 | 用途 |
| --- | --- |
| `qwen3_train/data/` | feature 检查、训练 build、读取和组批 |
| `qwen3_train/models/`、`objectives/` | 模型与输入协议、训练目标 |
| `qwen3_train/training/`、`evaluation/` | 分布式训练、恢复、独立生成评分 |
| `scripts/acp/` | ACP 提交和平台变量适配；训练模块不依赖 ACP |
| `assets/base/` | 从 `/workspace/model` 复制的基础模型 |
| `assets/evaluation/` | Whisper、DNSMOS、WavLM 评分模型 |
| `assets/assembled/` | 本地预先组装好的训练模型 |
| `data/builds/` | 训练侧的固定版本索引与 shallow clone |
| `../acp/` | 平台 CLI 下载、凭据、配置和密钥 |
| `../envs/lm-tts/` | 训练 Conda 环境 |
| `/workspace/LM-TTS-Training-Runs/<run_name>/` | checkpoint、TensorBoard、日志和配置快照 |

`assets/`、`data/` 和运行产物均已 gitignore。基础、评估、assembled 模型相互独立；训练只加载已有 assembled 模型，缺少完成标记会失败，不会启动后组装或下载。

## 配置怎么改

所有配置直接继承同一个 `base.yaml`，没有多级继承。ACP 参数放在同一个训练配置的 `acp` 区块；数据绑定、生成和评分分别使用各自配置。所有入口使用相同的 YAML 继承规则：`extends` 相对于当前文件，字典合并，列表整体替换；`${paths.project}` 等引用在合并后解析。

| 配置 | 通常需要改什么 |
| --- | --- |
| `configs/base.yaml` | 唯一共享默认配置，按路径、通信、模型、数据、训练、ACP 分块 |
| `configs/data.yaml` | 各数据集明确的 codec / speaker manifest，selection 和数据策略 |
| `configs/train-bf16.yaml`、`train-fp32.yaml` | 实验名称、精度；需要时覆盖训练参数和 `acp.nodes` / `acp.image` |
| `configs/synthesis.yaml`、`evaluation.yaml` | 生成参数、评分模型和音频预览数量 |

新实验复制一份实验 YAML，继承 `base.yaml`，更换 `train.run_name`。恢复保持同名 run。`runs_root` 定义一次，checkpoint / TensorBoard / 日志从 run 目录派生，不需要在多个 YAML 重复写路径。修改训练精度、数据、batch 或优化器语义需要新实验；恢复要求相同 world size。

## 准备环境和模型

```bash
export ENV_PREFIX=/workspace/workspace/yanglin/envs/lm-tts
export CONDA_BIN=/home/yanglin/miniforge3/bin/conda
bash scripts/setup_env.sh
PY=$ENV_PREFIX/bin/python
# 本地复制基础权重、校验复制结果，然后组装并重新加载检查；不下载训练权重
$PY -m scripts.prepare_models --config configs/train-bf16.yaml
# 只有评分模型缺失时才需要下载
$PY -m scripts.download_eval_models
```

准备步骤在提交训练之前单独执行。已完成的模型可重复使用。环境为 Python 3.11、torch 2.8.0 / CUDA 12.6、transformers 4.57.3、FlashAttention 2.8.3.post1、pylance 12.0.0；直接依赖在 `requirements.lock.txt`，已安装环境快照在 `../envs/lm-tts.freeze.txt`。

## 数据到训练

当前正式 feature 尚未全部发布。`configs/data.yaml` 已列出计划发布路径，只有 codec 和 speaker 两份 manifest 均 `status=complete` 的绑定才能 build。先训练部分数据时保留已就绪的数据集绑定；不要指向 `.incomplete`。text / language 来自发布后的 codec 表，文本源于固定 selection 的 `selected_text` 或原始 samples 的 `text`，训练侧不生成转写。speaker embedding 是官方冻结 encoder 的 1024 维向量。

```bash
# 从实验配置派生数据配置和 build 输出路径
$PY -m scripts.build_unified --config configs/train-bf16.yaml
# 本地训练
NPROC_PER_NODE=8 bash scripts/run_train.sh --config configs/train-bf16.yaml
# FP32：使用独立样本 SDPA；BF16：使用变长 FlashAttention
NPROC_PER_NODE=8 bash scripts/run_train.sh --config configs/train-fp32.yaml
# 断点恢复
NPROC_PER_NODE=8 bash scripts/run_train.sh --config configs/train-bf16.yaml --resume latest
```

更换 selection、feature 版本或 tokenizer 时，修改 `data.build_id` 生成新 build；同一份 build 可以供 BF16 和 FP32 训练复用。

Build 只写训练 repo 内的 `data/builds/`，不修改 unified 源表或 branch。Lance shallow clone 引用固定版源 payload，训练侧添加 speaker 定位符和 token 缓存；源文件必须保留。未绑定、缺失或失败的 feature 都有覆盖计数，身份冲突直接失败。训练按就绪 utterance 均匀无放回采样，使用 mmap 索引与有界 shuffle，读取时批量投影 codec / speaker，不在线重算 embedding。

`base.yaml` 的 `environment` 统一定义通信默认值：`NCCL_IB_TIMEOUT=22`、`NCCL_IB_RETRY_CNT=13`、`NCCL_IB_AR_THRESHOLD=0`，按当前 [商汤 ACP 建议](https://www.sensecore.cn/help/docs/cloud-foundation/compute/acp/acpUserGuide/acpEnvironmentVariable) 设置。本地训练保留显式 shell 环境覆盖，ACP 提交将这些值注入所有节点；更换网络环境时需按对应 NCCL / 驱动要求调整。

多机在各节点运行同一入口，设置 `NNODES`、不同的 `NODE_RANK`、相同的 `MASTER_ADDR` / `MASTER_PORT`、`NPROC_PER_NODE`。两种精度都保留 FP32 master weights、梯度归约和 cross entropy，关闭 TF32。feature 提取精度与训练精度分别记录。

## ACP 提交

```bash
# 查看实际请求，不创建任务
$PY -m scripts.acp.submit --config configs/train-bf16.yaml --nodes 4
# 创建 4 节点任务；当前规格每节点 8 张 A800
$PY -m scripts.acp.submit --config configs/train-bf16.yaml --nodes 4 --submit
# 恢复同一个 run
$PY -m scripts.acp.submit --config configs/train-bf16.yaml --nodes 4 --resume latest --submit
# 验证启动、BF16 / FP32、断点恢复和评分接口，使用微型模型
$PY -m scripts.acp.submit --validate --nodes 4 --run-name check-32gpu --submit
```

SSH 免密和平台 TensorBoard 默认开启。同一个实验 YAML 中的 `acp` 区块只由提交器读取，训练核心收到的配置不包含它。每次提交保存代码快照、解析后的实验配置、请求与返回的 job ID 到 `submissions/<timestamp>/`，任务使用该代码快照。凭据只从 `../acp/.credentials.json` 读取。

所有 run 直接保存在 `/workspace/LM-TTS-Training-Runs/<run_name>`。ACP 将这个统一 Runs 目录挂载给 TensorBoard，不再创建 `/workspace/lmtts-<run_name>`；页面中按 run 名选择实验。平台只扫描本项目 Runs，训练、数据、模型目录不作为 TensorBoard 根目录。

也可以在开发机用下方命令启动独立仪表盘，只读取一个 run。若不需要平台页面，将 `acp.tensorboard` 改成 `local`。平台模式要求 `paths.runs` 是 AFS 一级目录；选择其他嵌套路径时使用独立仪表盘即可。历史 Runs 旧路径保留链接以维护已发布 build 的绝对引用，新配置统一使用新路径。

## 日志与 TensorBoard

每个 run 包含 `checkpoints/`、`tensorboard/`、`logs/`、`metrics.jsonl`、有效 `config.json` 和源 `config.yaml`。`checkpoints/latest` 指向最后一个完整 checkpoint；恢复会继续同名 run。每节点启动输出分别保存到 `logs/`。

默认写入三组训练指标，在 `train.tensorboard.groups` 中选择；空列表关闭 scalar 组。

| 组 | 内容 |
| --- | --- |
| `optimization` | 总 loss、首码本 / residual CE、两组学习率、梯度范数 |
| `performance` | 音频秒吞吐、每步耗时、数据等待、rank 0 每步峰值显存 |
| `batch` | 全局有效样本数、音频秒数、codec token 数；包含梯度累积 |

`Custom Scalars` 提供 loss、学习率、耗时组合图；`Scalars` 查看单曲线，`Text` 查看训练配置。真正隔离的 validation build 可写入 validation loss 和两项 CE。当前 selection 未划分 holdout，默认 `no_holdout`，不会伪造验证曲线。

评估写入 WER / CER、DNSMOS SIG / BAK / OVRL 和 speaker cosine；多语言额外显示适合该语言的误差。`Audio` 对照参考 / 生成音频，`Text` 显示目标文本和 ASR 转写。默认预览前两对音频的前 20 秒，可在 `evaluation.yaml` 的 `tensorboard` 中调整；预览裁剪不影响完整音频评分。DNSMOS 是预测音质分，不等于人工 MOS。

```bash
# 在开发机打开指定 run，打印地址；远程访问需转发 6006 端口
$PY -m scripts.tensorboard --run unified-bf16
# 旧 run 曲线较少时，从日志恢复；不重新训练
$PY -m scripts.tensorboard --run acp-20260930-32gpu --rebuild-only
```

恢复日志写在 `tensorboard/restored/`，保留原始事件。选择 restored 对应的 run 查看。历史日志没有记录的指标不会被补造。

## 生成与评分

生成评分独立于训练进程，在 checkpoint 保存后单独执行，避免评分模型阻塞分布式更新。`prompts.json` 是固定集合的数组，每条包含 `id`、`text`、ISO `language`、`reference_audio`；reference 使用同说话人的另一条录音。评估集应与训练隔离。

```bash
RUN=/workspace/LM-TTS-Training-Runs/unified-bf16
$PY -m scripts.export_checkpoint --checkpoint "$RUN/checkpoints/step-00000500" \
  --assembled-model assets/assembled/qwen3-tts-frozen-conditioning --output "$RUN/export-500"
$PY -m scripts.synthesize --model "$RUN/export-500" --config configs/synthesis.yaml \
  --prompts prompts.json --output "$RUN/generated-500"
$PY -m scripts.evaluate --config configs/evaluation.yaml --pairs "$RUN/generated-500/pairs.json" \
  --output "$RUN/evaluations/step-500" --tensorboard "$RUN/tensorboard" --step 500
```

Whisper small 用于内容误差，DNSMOS P.835 用于音质，WavLM base-plus-sv 用于归一化 x-vector cosine。评分报告保存每条结果、语言汇总、权重 / 音频哈希及参数。相似度没有通用及格阈值。已有音频也可直接评分：pairs 在 prompts 字段基础上增加 `audio`。

## 验证与边界

```bash
$PY -m pytest -q
```

真实 ACP 4 节点 32 张 A800 作业 `pt-ughagq38` 已成功：BF16 和 FP32 各完成连续训练与 2+2 步恢复，全部权重逐位一致，冻结模块未变，评分接口通过。报告在 `/workspace/LM-TTS-Training-Runs/acp-20260930-32gpu/result.json`。微型模型验证功能，不能代表完整模型吞吐。CSEMOTIONS 的独立完整 feature 用于真实数据读取验证；正式 selection 仍等待处理完成。

更多数值证据见 [验证记录](docs/training/unified-refactor.md)。历史 run 按 [保留规则](runs/README.md) 保存，旧文档不作为当前启动说明。
