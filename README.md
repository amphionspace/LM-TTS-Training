# LM-TTS-Training

基于 Qwen3-TTS 的非流式预训练框架。从 Qwen3 text Base 初始化 Talker，默认冻结官方 text frontend 和 speaker encoder，训练音频预测模块。当前训练数据入口是 `tts-data-pipeline` 发布的 unified Lance features；旧 JSONL / NPZ 训练入口和实验专用脚本已移除。

| 目录 | 职责 |
| --- | --- |
| `qwen3_train/data/` | 发布检查、固定版本 build、批量读取、动态组批 |
| `qwen3_train/models/` | 权重组装、输入协议、模型前向 |
| `qwen3_train/objectives/` | EOS 与 residual codebook 目标、loss 归一化 |
| `qwen3_train/training/` | 精度、FSDP2、优化器、调度器、checkpoint / resume |
| `qwen3_train/evaluation/` | 验证 loss、ASR、DNSMOS、WavLM、报告 |
| `configs/` | 数据配方、BF16 / FP32 训练、生成与评分配置 |
| `scripts/` | 少量独立入口 |

ACP CLI 位于 `/workspace/workspace/yanglin/acp`；Conda 环境位于 `/workspace/workspace/yanglin/envs/lm-tts`。训练代码只读取标准 PyTorch 分布式变量，不依赖 ACP。凭据不进入本仓库。

## 环境与模型

```bash
export ENV_PREFIX=/workspace/workspace/yanglin/envs/lm-tts
export CONDA_BIN=/home/yanglin/miniforge3/bin/conda
bash scripts/setup_env.sh
export PYTHONPATH="$PWD"
export HF_HOME=/workspace/workspace/yanglin/.cache/huggingface
export NUMBA_CACHE_DIR=/workspace/workspace/yanglin/.cache/numba
PY=$ENV_PREFIX/bin/python
$PY -m scripts.assemble_qwen3_tts --output pretrained/assembled-qwen3-tts-frozen-conditioning
$PY -m scripts.download_eval_models
```

Python 3.11、torch 2.8.0 CUDA 12.6、FlashAttention 2.8.3.post1、transformers 4.57.3、pylance 12.0.0。已安装环境的完整快照在 `../envs/lm-tts.freeze.txt`；`requirements.lock.txt` 固定项目直接依赖。

## Unified 数据

当前 selection：`tts-selection-supervised-tts-20260929T151539bjt-01`，128,220,178 条、约 357,507 小时。数据还在处理：codec 的 `.incomplete` 分片已能核对 K=16、词表 2048、12.5 frames/s、24kHz；最终文本列在 feature 发布时补齐。训练必须等待对应 codec 和 speaker manifest 的 `status=complete`。

在 `configs/data_recipe.yaml` 填入**明确发布版本**的绑定，例如：

```yaml
bindings:
  - codec_manifest: datasets/ljspeech/v0.1/features/codec/<published-run>/manifest.json
    speaker_manifest: datasets/ljspeech/v0.1/features/speaker_embedding/<published-run>/manifest.json
```

```bash
$PY -m scripts.build_unified --recipe configs/data_recipe.yaml --output /workspace/data/DATA-TTS-UNIFIED/builds/tts-build-lmtts-20260930T140000bjt-01
```

Build 对源 feature 创建新 Lance branch，添加 speaker 行定位符、可训练状态、token 缓存；不复制 codec、文本、原音频或 embedding。源 main 快照不变。manifest 固定 branch / version / profile / selection / tokenizer 哈希，未绑定数据集也记入未就绪覆盖；行定位符是该快照的逻辑行偏移，不是 `_rowid`。本阶段需要统一数据根目录的写权限以创建 branch。

一次性构建使用本地 SQLite 校验 target ID、音频哈希、原生采样率和区间一致性；训练阶段只按批次投影读取 codec 与 speaker，不做全量 join 或重算 embedding。当前 plan 按所有就绪 utterance 均匀采样，不做语言 / 质量加权；这些采样策略与 feature 构建分离。抽样元数据使用 mmap，shuffle 和 rank 分配使用有界窗口；不把亿级样本列表或整 epoch 批次放入 Python 内存。缺失 / 失败 feature 有独立覆盖计数，身份冲突直接失败。中断 build 不发布完整 manifest，使用新 build ID 重建；无需重算已有 feature。

当前 selection 未隔离评估集，配置明确声明 `no_holdout`。不会从训练集复制一份冒充验证集；验证 loss 需要另行发布隔离后的训练 / 验证 build。新框架不继续旧格式 run 的 checkpoint，历史实验及其数据、音频和源码归档仍按 [保留规则](runs/README.md) 保留。

## 训练与恢复

```bash
NPROC_PER_NODE=8 bash scripts/run_train.sh --config configs/unified.yaml
# 原生 FP32，自动选择 SDPA
NPROC_PER_NODE=8 bash scripts/run_train.sh --config configs/unified-fp32.yaml
# 相同 world size、数据、精度、batch、优化器与 schedule 下恢复
NPROC_PER_NODE=8 bash scripts/run_train.sh --config configs/unified.yaml --resume latest
```

多机时每台运行同一入口，设置 `NNODES`、各自的 `NODE_RANK`、相同的 `MASTER_ADDR` / `MASTER_PORT`、`NPROC_PER_NODE`。支持跨节点 FSDP2。ACP 变量转换由外部 `../acp/launch.sh` 完成。BF16 使用变长 FlashAttention，FP32 使用独立样本的 SDPA padding；两者 master weights、梯度归约和 cross entropy 保持 FP32。TF32 关闭。离线 feature 的提取精度属于 feature profile，与训练精度分别记录。

按全体 rank 与所有累积 microbatch 的真实分母归一化 loss，支持 token / sample / sqrt。checkpoint 保存优化器、scheduler、各 rank RNG、epoch 和下一批位置；只有完整写入后才更新 latest。操作参数允许调整，改变数据或训练语义会拒绝恢复。当前要求恢复时保持 world size。

## 生成与评分

训练中的验证 loss 与生成评分分开。ASR / 音质 / 相似度模型独立加载，避免阻塞 FSDP 的更新和集合通信。可以在保存 checkpoint 后另行调度评估。

`prompts.json` 是固定评估集合的 JSON 数组，每条包含 `id`、`text`、ISO `language` 和 `reference_audio`。引用同说话人的另一条录音；目标文本不作为 ICL 参考文本。评估集应与训练数据隔离，当前 `no_holdout` 的训练本身不作泛化声明。

```bash
$PY -m scripts.export_checkpoint --checkpoint runs/unified-bf16/checkpoints/step-00000500 \
  --assembled-model pretrained/assembled-qwen3-tts-frozen-conditioning --output artifacts/export-500
$PY -m scripts.synthesize --model artifacts/export-500 --config configs/synthesis.yaml \
  --prompts prompts.json --output artifacts/generated-500
$PY -m scripts.evaluate --config configs/evaluation.yaml --pairs artifacts/generated-500/pairs.json \
  --output artifacts/scores-500 --tensorboard runs/unified-bf16/tensorboard --step 500
```

评分也接受已有生成音频：pairs 数组增加 `audio` 字段即可。报告保留每条结果、各语言 corpus WER / CER、所有计数、权重文件哈希、音频哈希和评分参数。

- ASR：固定版本多语言 Whisper small，中文 / 日文 / 韩文重点看 CER，其他语言看 WER。
- 音质：[Microsoft DNSMOS P.835](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS)，保存 SIG / BAK / OVRL，仪表盘只显示 OVRL。16kHz、9.01 秒窗、1 秒 hop、短录音重复、官方非个性化校准；采用整数样本边界。它是模型预测分数，不是人工 MOS。
- 相似度：[Microsoft WavLM base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv) 的归一化 x-vector cosine。使用独立参考录音，16kHz、FP32；长录音分均匀段提取后平均归一化向量，不设置通用的相似度及格阈值。该模型的 VoxCeleb 适配分布也应纳入跨语言结果解释。

TensorBoard 只保留 `train/loss`、`train/lr`、`train/audio_seconds_per_second`、可用的 `val/loss`、各语言一个内容误差、`eval/dnsmos_ovrl`、`eval/speaker_similarity`。codebook 明细、梯度、显存、计数和等待时间保存在 JSON 日志。日志目录在 AFS，ACP TensorBoard 可读取同一路径。

## 验证

```bash
$PY -m pytest -q
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=.:tests $PY -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/check_distributed_equivalence.py --qwen-protocol --speaker --frozen-speaker --frozen-frontend
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=.:tests $PY tests/check_training_resume.py \
  --output artifacts/resume-check --precision fp32
```

[本次验证和限制](docs/training/unified-refactor.md)。`docs/` 中其他 Emilia / JSONL / NPZ 文档记录历史实验，不再作为当前启动说明。
