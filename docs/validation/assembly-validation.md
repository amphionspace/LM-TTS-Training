# 组装模型与 ECAPA 联合训练验证

环境：Python 3.10，PyTorch 2.8.0+cu126，Transformers 4.57.3，qwen-tts 0.1.1，2 × A800 80GB。

## 组装结果

本地输出：pretrained/assembled-qwen3-tts-0.6b。
总参数 754,865,216；Talker 746,010,880；speaker encoder 8,854,336。
官方 Qwen3TTSModel.from_pretrained 已成功重载文本 processor、Talker、ECAPA 和 24 kHz codec。
文本宽度 1024，使用官方 ResizeMLP；7 个新增文本 token 利用原有 embedding 预留行。
共享的 151,669 个 token ID 全部核对；另以 32 条 LJSpeech 文本和 5 条多语言/空白符样例验证实际分词 ID 一致。

## 正确性

- 14 项单元测试通过：包括权重来源、保存重载、产物损坏检测、因果性、padding、speaker 条件影响、speaker 参数实际更新及 WER/CER 统计。
- 确定性 reflection padding 与官方实现的前向完全一致，输入和权重梯度在 FP64 精度下相符。
- 严格 FP32 下，双卡变长 batch、两次梯度累积与全局 batch 对照通过；最大绝对梯度误差 1.7166e-5，检查阈值为 atol=2e-6、rtol=2e-4。
- 完整模型连续两步与中断后恢复两步相比，模型、AdamW、各 rank RNG 共 1,916 个张量完全一致；scheduler 和数据位置也一致。

原先未限制 CUDA 算法的 BF16 路径能恢复，但后续训练数值不同。现已启用确定性算法、固定 cuBLAS workspace，并用 index_select 实现等价的 reflection padding，避免 CUDA 原始 reflection backward 的原子累加。
新 checkpoint 协议为 v2；旧 v1 测试产物需使用对应旧代码恢复。

复现命令：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=. .venv/bin/torchrun --standalone --nproc_per_node=2 tests/check_distributed_equivalence.py --speaker
# 旧配置恢复方法见 docs/training/setup-and-training.md 的“旧配置归档”。
bash scripts/run_train.sh --config runs/cleanup-20260909/configs/ljspeech-assembled-integration.yaml --output runs/check-split --max-steps 1
bash scripts/run_train.sh --config runs/cleanup-20260909/configs/ljspeech-assembled-integration.yaml --output runs/check-split --resume latest
bash scripts/run_train.sh --config runs/cleanup-20260909/configs/ljspeech-assembled-integration.yaml --output runs/check-continuous
python scripts/compare_checkpoints.py runs/check-continuous/checkpoints/step-00000002 runs/check-split/checkpoints/step-00000002
```

## 两步真实数据检查

28 条 train、4 条 val。第一步 speaker 梯度范数 77.9197，第二步 41.8945。
第二步训练 first CE 9.24319、residual CE 7.84070。
生成、官方 codec 解码、参考音频记录、TensorBoard 和 ASR 评估均运行成功。
该配置仅生成最多 4 帧，即 0.32 秒；生成被截断，WER/CER 为 1，原录音 ASR 为 0。这里只验证链路，不能据此评价语音质量。
