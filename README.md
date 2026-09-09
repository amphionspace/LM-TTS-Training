# LM-TTS-Training

对齐 Qwen3-TTS-12Hz-0.6B 结构：冻结公开 text embedding / projector / ECAPA，从 Qwen3-0.6B-Base 初始化 Talker，复用官方 codec 和 ECAPA，训练新的音频预测模块。
支持单机多卡 FSDP2、断点续训、TensorBoard、验证 loss 和 ASR WER/CER。LJSpeech 用于诊断；当前主线准备约 1,000 小时 Emilia2 独立 short，验证官方非流式训练基线。

## 快速开始

```bash
source .venv/bin/activate
# 组装初始化权重，默认使用脚本中固定的官方 revision：
python scripts/assemble_qwen3_tts.py --output pretrained/assembled-qwen3-tts-frozen-conditioning
# 按 docs/training/emilia-baseline.md 完成数据准备和显存测试后：
PYTHONPATH=. python scripts/run_emilia_baseline.py
bash scripts/run_train.sh --config runs/emilia-official-frozen-1000h/baseline-config.yaml --resume latest
tensorboard --logdir runs --port 6006
```

组装脚本支持本地来源目录；只核对配置和词表可使用 `--audit-only --output runs/assembly-audit.json`。
ECAPA 从 Qwen 公开 TTS checkpoint 加载，使用同说话人的另一条训练录音提供音色条件，输出音色条件；权重冻结，不参与训练。
组装成功不代表语音质量达标。LJSpeech 不能检验跨说话人的 zero-shot 泛化。

## 文档

- [当前模型详细结构与冻结策略](docs/design/final-architecture.md)
- [官方 0.6B / 1.7B 结构、维度与初始化证据](docs/design/official-qwen3-tts.md)
- [Emilia2 数据中间层与 1,000 小时基线](docs/training/emilia-baseline.md)
- [环境、数据准备、训练和评估](docs/training/setup-and-training.md)
- [模型组装、官方差异、ECAPA 来源与 ICL](docs/design/model-assembly.md)
- [从头训练的 reference：Nar、VALL-E、Fish、F5 对比](docs/design/reference-training.md)
- [组装模型、ECAPA 与精确恢复验证](docs/validation/assembly-validation.md)
- [LJSpeech 500 步训练与生成评估](docs/validation/ljspeech-medium-500.md)
- [框架正确性与真实数据验证记录](docs/validation/framework-validation.md)

## 检查

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=. .venv/bin/torchrun --standalone --nproc_per_node=2 tests/check_distributed_equivalence.py --speaker
```

当前 JSONL / NPZ 读取方式用于验证。几十万小时正式训练前，还需接入分片流式数据生产与读取。

训练正确性、恢复验证及历史 LJSpeech 诊断见 [验证记录](docs/validation/training-diagnosis.md)。
