# LM-TTS-Training

对齐 Qwen3-TTS-12Hz-0.6B 结构：冻结公开 text embedding / projector / ECAPA，从 Qwen3-0.6B-Base 初始化 Talker，复用官方 codec 和 ECAPA，训练新的音频预测模块。
支持单机多卡 FSDP2、断点续训、TensorBoard、验证 loss 和 ASR WER/CER。LJSpeech 用于诊断；当前主线为 Emilia2 中英 10,000 小时、动态组批的非流式预训练。

中英 1kh 实验已完成 5,000 步，见 [数据、模型、训练与评估报告](docs/training/emilia-1kh-report.md)。**该run和当前10kh正式run完整保留；清理前先读 `runs/README.md`，依赖范围见[保留说明](docs/training/run-retention.md)。**

## 快速开始

```bash
source .venv/bin/activate
# 环境、组装和数据准备见 docs/training/setup-and-training.md。
# 当前正式run确认原进程结束后，通过launcher从latest恢复：
python runs/emilia-en-zh-dynamic-10000h/launch.py
tensorboard --logdir runs --port 6006
```

组装脚本支持本地来源目录；只核对配置和词表可使用 `--audit-only --output /tmp/assembly-audit.json`。
ECAPA 从 Qwen 公开 TTS checkpoint 加载并冻结。训练和验证 loss 使用完整目标录音提取音色条件；生成评估使用同说话人的另一条训练录音。
组装成功不代表语音质量达标。LJSpeech 不能检验跨说话人的 zero-shot 泛化。

## 文档

- [实验产物保留清单：清理前必读](docs/training/run-retention.md)
- [中英 1kh 实验报告：数据组成、模型结构、训练配置与评估结论](docs/training/emilia-1kh-report.md)
- [10kh 正式训练计划与巡检记录](docs/training/emilia-10kh-supervision.md)
- [训练故障处置：诊断、修复与恢复边界](docs/training/training-incident-playbook.md)
- [动态组批与全局token归一化](docs/training/dynamic-batching.md)
- [ICL 与 speaker-only 生成评估：输入、配对和结果边界](docs/training/icl-evaluation.md)
- [中英预训练完整 pipeline：数据读取、模型输入与 loss 推导](docs/training/pretraining-pipeline.md)
- [中英 10,000 小时数据准备：独立 short 与 long 标注短句](docs/training/emilia-10kh.md)
- [当前模型详细结构与冻结策略](docs/design/final-architecture.md)
- [官方 0.6B / 1.7B 结构、维度与初始化证据](docs/design/official-qwen3-tts.md)
- [环境、数据准备、训练和评估](docs/training/setup-and-training.md)
- [模型组装、官方差异、ECAPA 来源与 ICL](docs/design/model-assembly.md)
- [从头训练的 reference：Nar、VALL-E、Fish、F5 对比](docs/design/reference-training.md)

## 检查

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=. .venv/bin/torchrun --standalone --nproc_per_node=2 tests/check_distributed_equivalence.py --speaker
```

当前 JSONL / NPZ 读取方式用于验证。几十万小时正式训练前，还需接入分片流式数据生产与读取。
