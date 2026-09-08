# LM-TTS-Training

从 Qwen3-0.6B-Base 初始化 Talker，复用官方 Qwen3-TTS 音频 tokenizer 和预训练 ECAPA，训练新的 TTS 模型。
支持单机多卡 FSDP2、断点续训、TensorBoard、验证 loss 和 ASR WER/CER。当前使用 LJSpeech 验证非流式训练链路。

## 快速开始

```bash
source .venv/bin/activate
# 组装初始化权重，默认使用脚本中固定的官方 revision：
python scripts/assemble_qwen3_tts.py --output pretrained/assembled-qwen3-tts-0.6b
# 本机已有 28 train + 4 val 的测试数据：
bash scripts/run_train.sh --config configs/ljspeech-assembled.yaml
bash scripts/run_train.sh --config configs/ljspeech-assembled.yaml --resume latest
tensorboard --logdir runs --port 6006
```

组装脚本支持本地来源目录；只核对配置和词表可使用 `--audit-only --output runs/assembly-audit.json`。
ECAPA 从 Qwen 公开 TTS checkpoint 加载，使用同说话人的另一条训练录音提供音色条件，并继续参与训练。
组装成功不代表语音质量达标。LJSpeech 不能检验跨说话人的 zero-shot 泛化。

## 文档

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
