# 验证记录（2026-09-08）

## 环境

- 独立 venv：`/119010446/LM-TTS-Training/.venv`，Python 3.10。
- PyTorch 2.8.0+cu126；Transformers 4.57.3；qwen-tts 0.1.1。
- 2 × NVIDIA A800-SXM4-80GB，driver 570.124.06。
- NCCL + FSDP2 + BF16 的前向、反向和参数更新检查通过。
- pip check 通过；完整依赖见 requirements.lock.txt。

## 实现准确性

1. 初始化映射：Qwen3 的 layers、norm、text embedding 严格加载到 Talker；微型结构上同一文本的隐状态匹配。
2. 因果性：修改目标帧不会改变预测该帧所用的 Talker 状态。
3. Padding 与 EOS：拼成变长 batch 后 loss 与逐样本相加一致，每条样本恰好有一个 EOS 目标。
4. 梯度：全部可训练模块收到梯度。
5. 生成：空音频历史可产生合法的 16 码本帧。
6. 采样：不足 global batch 的样本从随机序列尾部丢弃，不会固定丢掉最长样本。
7. 内容指标：WER/CER 编辑计数，以及 corpus WER 按 reference 长度加权的测试。
8. 单卡/多卡：严格 FP32，变长数据、两次梯度累积；最大梯度绝对误差 1.993e-7。
9. 断点恢复：连续训练 3 步，与 1 步后退出再恢复到第 3 步逐张量比较；模型、AdamW、RNG 共 320 个张量完全一致。scheduler、epoch 和 next_batch 也一致。

测试采用小模型检查数值，不能据此声称完整 0.6B 的所有数值路径均与单卡逐位一致。
容器初始 FP32 matmul precision 为 high；严格数值检查显式改为 highest，避免 TF32 误差干扰。

## 真实 LJSpeech 闭环

- 原目录：`/ai_sds_wuzz/DATA_TTS/LJSpeech/LJSpeech-1.1`。
- metadata 13,100 条，本地 README 声明为同一位读者的录音。
- 固定 seed 42，抽取不超过 6 秒的 32 条录音：28 train、4 val，合计约 0.03852 小时。
- 冻结官方 codec，完成编码、解码重建与 codes 范围/shape 检查。
- Qwen3-0.6B backbone + 新音频模块，共 743,911,680 个可训练参数。
- `configs/ljspeech-integration.yaml`：双卡训练第 1 步，保存退出，然后从 latest 恢复到第 2 步。
- 完成完整模型/优化器的分片保存与加载、validation loss、WAV 输出和 ASR 评分。
- 独立 eval-only 在该 checkpoint 上完成两条样本生成及 corpus WER/CER；验证汇总计数与逐条结果相等，checkpoint 仍停留在第 2 步。
- 已读取 TensorBoard event 文件，确认训练标量、各码本 validation loss、WER/CER、原始/生成音频存在，训练 step 到 2。

模型版本：

- Qwen/Qwen3-0.6B @ c1899de289a04d12100db370d81485cdf75e47ca
- Qwen/Qwen3-TTS-Tokenizer-12Hz @ 7dd38ad4e9bad454aae9cd937d0cd577604fe229

## 内容准确性诊断

固定样本 LJ030-0150，faster-whisper small.en，CPU INT8，英文文本规范化：

| 音频 | WER | CER | 说明 |
|---|---:|---:|---|
| 原始录音 | 0 | 0 | 17 个 reference words |
| Codec 重建 | 0 | 0 | 同一条录音的编解码基线 |
| 仅训练两步的生成音频 | 1.0 | 1.0 | 4 帧、0.32 秒，ASR 返回空文本 |

生成链路正确报告 `truncated=true` 和 `eos_reached=false`。
这次生成刻意限长，仅用于验证失败计数和端到端输出，不是语音质量测试。
模型的语音模块尚未收敛；还需要小数据过拟合、较充分训练、更多固定文本和人工试听才能判断实际准确性。

证据：

- `runs/resume-reference/` 与 `runs/resume-split/`：微型恢复检查。
- `runs/ljspeech-integration/`：完整模型训练、checkpoint、TensorBoard 和 evaluation。
- `data/ljspeech-test/asr-baseline.json`：原始/重建录音转写及计数。
- `data/ljspeech-test/codec-reconstruction.wav`：codec 重建试听。

## 当前范围

支持单机 FSDP2、同 world size 的精确恢复、TensorBoard、固定多样本生成及 corpus WER/CER。
本记录描述首版无 speaker 基线。新增组装模型已接入可训练 ECAPA；仍未实现大规模分片数据流、跨卡数精确恢复、流式生成和生产推理优化。
