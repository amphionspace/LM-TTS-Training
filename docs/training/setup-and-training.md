# 环境、数据准备、训练与评估

> 当前主线和完整自动运行入口见 [Emilia 1,000 小时基线](emilia-baseline.md)。本页保留早期 LJSpeech 调试记录。2026-09-09 已将旧配置归档至本机 `runs/cleanup-20260909/configs/`；旧实验初始化权重已清理，重新从初始化训练前需重建对应模型。历史 checkpoint 读取所需的配置、tokenizer 和 codec 保留。

## 旧配置归档

本机 `runs/` 归档不提交到 Git。其他环境执行下列命令，可从清理前的提交恢复本页历史命令使用的配置：

```bash
mkdir -p runs/cleanup-20260909
git archive dd2922bb0b41d0541365d42143d835b14d1dd252 configs | tar -x -C runs/cleanup-20260909
```

独立工程：使用官方音频 codec 和模型组件，新写训练、数据、恢复和评估代码。
原生 PyTorch FSDP2，当前支持单机多卡。首版针对 LJSpeech 做非流式文本到语音验证。

## 环境

本机已经建立 `.venv`，不继承系统 site-packages。
Python 3.10；PyTorch 2.8.0+cu126；Transformers 4.57.3；qwen-tts 0.1.1。
GPU：2 × A800 80GB。默认使用 SDPA；未安装外部 flash-attn 扩展。

```bash
cd /119010446/LM-TTS-Training
source .venv/bin/activate
# 重建环境（要求基础 Python 提供 pip）：
bash scripts/setup_env.sh
# 下载固定 commit 的官方 backbone 和 codec：
HF_HOME="$PWD/.cache/huggingface" HF_HUB_DISABLE_XET=1 python scripts/download_models.py
```

直接依赖在 `requirements.txt`，已安装环境的完整版本锁在 `requirements.lock.txt`。

## 数据与旧基线模型

用户提供的数据：`/ai_sds_wuzz/DATA_TTS/LJSpeech/LJSpeech-1.1`。
本地 README 声明为同一位录音者，metadata 有 13,100 条，采用第三列 normalized transcription。
每次生成一段语音。旧基线没有 speaker 条件；assembled 配置通过同 speaker 的另一条录音提取音色向量并联合训练。

- 冻结 Qwen3-TTS-Tokenizer-12Hz，离线编码为 `[frames, 16]` 的 uint16 codes。
- 采用官方 Talker 和 5 层 Code Predictor 类。
- Talker 的 Transformer layers、final norm、text embedding 从 Qwen3-0.6B 严格加载；音频 embedding/head 和 predictor 新初始化。
- 文本 embedding 宽度设为 1024，text projection 使用 Identity，以适配 Qwen3-0.6B。
- 输入协议为 `[完整文本, audio BOS, 历史音频帧]`。Talker 预测当前帧的第 0 码本，depth predictor 自回归预测其余 15 个码本。
- 采用普通一维位置序列；不声称复现官方流式双轨训练协议或兼容官方生成 checkpoint 格式。
- 当前数据读取使用 JSONL + 单样本 NPZ，适合验证；几十万小时训练前需要实现分片流式读取和预处理分布式调度。

```bash
# 首轮真实数据测试：28 train + 4 val，固定种子，不修改原始录音。
bash scripts/prepare_ljspeech.sh --limit 32 --val-count 4 --max-seconds 6
# 全量编码请使用新输出目录，并在训练配置中更新 manifest 路径。
bash scripts/prepare_ljspeech.sh --limit 0 --val-count 128 --output data/ljspeech-full
```

准备过程同时输出 `preparation.json`、音频 codes 校验值和 `codec-reconstruction.wav`。
同一输出目录要求准备参数一致；代码缓存可重用。默认模型路径位于 `pretrained/`。

## 组装模型训练与恢复

推荐配置为 `runs/cleanup-20260909/configs/ljspeech-assembled.yaml`。先运行独立组装脚本：

```bash
python scripts/assemble_qwen3_tts.py --output pretrained/assembled-qwen3-tts-0.6b
bash scripts/run_train.sh --config runs/cleanup-20260909/configs/ljspeech-assembled.yaml
bash scripts/run_train.sh --config runs/cleanup-20260909/configs/ljspeech-assembled.yaml --resume latest
```

新路径使用 Qwen3-0.6B-Base、完整 TTS tokenizer、ResizeMLP 和可训练的预训练 Qwen ECAPA。
输出路径必须不存在；来源与校验记录在 assembly_report.json。
两步真实数据测试配置为 ljspeech-assembled-integration.yaml。
详见 [组装设计](../design/model-assembly.md)。
下面的旧配置命令用于复现最初无 speaker 的基线。

## 基线训练与恢复

```bash
NPROC_PER_NODE=2 bash scripts/run_train.sh --config runs/cleanup-20260909/configs/ljspeech.yaml
NPROC_PER_NODE=2 bash scripts/run_train.sh --config runs/cleanup-20260909/configs/ljspeech.yaml --resume latest
# --max-steps 可以提前停止，保持配置中的 schedule_steps 学习率曲线不变。
```

训练使用 block 级 FSDP2、BF16 计算、FP32 梯度归约、activation checkpointing 和梯度累积。
两个 loss 分别按整个累积窗口、全部 rank 的有效目标数量归一化，再加权相加。
按照固定种子和 epoch 生成长度分桶 batch；当前丢弃不足一个 global microbatch 的 epoch 尾部。

Checkpoint 使用 Distributed Checkpoint 保存模型和 AdamW 状态，另外保存 scheduler、step、epoch、下一个 batch、各 rank RNG。
完成写入后才发布 `COMPLETE` 和 `latest`。保存先于周期评估，避免评估失败丢失该检查点。
当前训练启用确定性 CUDA 算法及固定 cuBLAS workspace；ECAPA 使用与官方数值等价的确定性 reflection padding。
恢复协议为 v2，早期 v1 测试 checkpoint 需要用对应旧版本代码恢复。精确恢复要求硬件与软件栈、world size、batch、累积步数、数据、模型、学习率计划等一致；不支持中途改卡数的精确恢复。
只加载本工程生成的可信 checkpoint。未实现正式训练的自动 checkpoint 清理。

## TensorBoard 与 evaluation

```bash
.venv/bin/tensorboard --logdir runs --port 6006
```

- 训练：两个 CE、梯度范数、学习率，以及 assembled 配置的 speaker 梯度范数、每步时间、音频秒数吞吐。
- 验证：固定验证集，首码本和其余各码本 CE；跨 rank 按有效目标数归约，不重复计入补齐样本。
- 生成：默认固定前 4 条验证文本，greedy 生成并通过官方 decoder 输出 WAV，TensorBoard 同时写入原始和生成录音。
- 内容准确性：CPU INT8 faster-whisper small.en 转写，计算 WER/CER、替换/漏词/插词数；原始录音也经过同一个 ASR 作为基线。
- 同时报告是否生成 EOS、是否达到长度上限、时长比例。未生成音频按全漏读处理，不默认为成功。
- 每条结果在 `runs/<run>/evaluation/step-*/sample-*/metrics.json`，试听文件为 `generated.wav`；单条模式直接写在 step 目录。汇总指标在 `summary.json`。

汇总 WER/CER 使用所有样本的总 edit counts / 总 reference lengths；不会平均每句错误率。
当前固定少量样本仍是诊断集，不是代表整个模型能力的基准。可以用 eval.num_samples 扩大生成评估集。
英文评估只做大小写/标点规范化，CER 忽略空格；未接入复杂数字标准化或中文专用文本处理。
ASR 有自身误差，WER/CER 不能替代自然度、音色相似度和人工试听。
早期随机初始化的语音模块不会立即生成可理解语音。通过工程测试不等于语音质量达标。
生成暂未优化 KV cache，周期性试听应控制频率和最大帧数。

## 独立评估已有 checkpoint

```bash
bash scripts/run_train.sh --config runs/cleanup-20260909/configs/ljspeech-integration.yaml --resume latest --eval-only --eval-samples 2
```

此命令不执行 optimizer update。`runs/cleanup-20260909/configs/ljspeech-integration.yaml` 是已运行的两步真实数据测试配置，
生成上限刻意设为 4 帧，验证的是链路；正常训练配置 `ljspeech.yaml` 使用 256 帧上限。

## 准确性检查

```bash
# 一键执行整个小模型验证流程：
bash scripts/verify_framework.sh
# 或分别运行：
python -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=. .venv/bin/torchrun --standalone --nproc_per_node=2 tests/check_distributed_equivalence.py
python scripts/make_smoke_fixture.py
bash scripts/run_train.sh --config configs/smoke.yaml --output runs/continuous
bash scripts/run_train.sh --config configs/smoke.yaml --output runs/interrupted --max-steps 1
bash scripts/run_train.sh --config configs/smoke.yaml --output runs/interrupted --resume latest
python scripts/compare_checkpoints.py runs/continuous/checkpoints/step-00000003 runs/interrupted/checkpoints/step-00000003
```

已验证：标签因果关系、padding 不影响 loss、EOS 计数、初始空音频生成、所有训练模块有梯度、WER/CER 编辑计数。
严格 FP32 下，双卡变长 batch + 两次梯度累积与单卡全局 batch 的最大梯度绝对误差约 `1.99e-7`。
微型模型中，连续训练与中断恢复后的模型、优化器及 RNG 共 320 个张量完全一致，数据位置跨 epoch 也一致。
完整模型和真实音频的检查结果另外记录于 [验证记录](../validation/framework-validation.md)。

## 参考

- [Qwen3-TTS 官方工程](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS 技术报告](https://arxiv.org/abs/2601.15621)
- [PyTorch FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [PyTorch CUDA wheels](https://pytorch.org/get-started/previous-versions/)
