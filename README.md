# LM-TTS-Training

对齐 Qwen3-TTS-12Hz-0.6B 结构：冻结公开 text embedding / projector / ECAPA，从 Qwen3-0.6B-Base 初始化 Talker，复用官方 codec 和 ECAPA，训练新的音频预测模块。
支持单机多卡 FSDP2、断点续训、TensorBoard、验证 loss 和 ASR WER/CER。LJSpeech 用于诊断；当前主线为 Emilia2 中英 10,000 小时、动态组批的非流式预训练。

中英 1kh 实验已完成 5,000 步，见 [数据、模型、训练与评估报告](docs/training/emilia-1kh-report.md)。**该 run 及关联产物必须保留；任何清理任务先阅读 [实验产物保留清单](docs/training/run-retention.md)。**

## 快速开始

```bash
source .venv/bin/activate
# 组装初始化权重，默认使用脚本中固定的官方 revision：
python scripts/assemble_qwen3_tts.py --output pretrained/assembled-qwen3-tts-frozen-conditioning
# 详细流程见 docs/training/pretraining-pipeline.md：
PYTHONPATH=. python scripts/run_emilia_baseline.py
# 对新的动态组批 run 续训（替换 RUN_DIR）：
NPROC_PER_NODE=4 bash scripts/run_train.sh --config RUN_DIR/config.yaml --resume latest
tensorboard --logdir runs --port 6006
```

组装脚本支持本地来源目录；只核对配置和词表可使用 `--audit-only --output runs/assembly-audit.json`。
ECAPA 从 Qwen 公开 TTS checkpoint 加载并冻结。训练和验证 loss 使用完整目标录音提取音色条件；生成评估使用同说话人的另一条训练录音。
组装成功不代表语音质量达标。LJSpeech 不能检验跨说话人的 zero-shot 泛化。

## 文档

- [实验产物保留清单：清理前必读](docs/training/run-retention.md)
- [中英 1kh 实验报告：数据组成、模型结构、训练配置与评估结论](docs/training/emilia-1kh-report.md)
- [10kh 正式训练计划与巡检记录](docs/training/emilia-10kh-supervision.md)
- [训练故障处置：诊断、修复与恢复边界](docs/training/training-incident-playbook.md)
- [动态组批与四卡试训](docs/training/dynamic-batching.md)
- [ICL 与 speaker-only 生成评估：输入、配对和结果边界](docs/training/icl-evaluation.md)
- [中英预训练完整 pipeline：数据读取、模型输入与 loss 推导](docs/training/pretraining-pipeline.md)
- [中英 10,000 小时数据准备：独立 short 与 long 标注短句](docs/training/emilia-10kh.md)
- [当前模型详细结构与冻结策略](docs/design/final-architecture.md)
- [官方 0.6B / 1.7B 结构、维度与初始化证据](docs/design/official-qwen3-tts.md)
- [历史 Emilia2 英语基线与数据中间层](docs/training/emilia-baseline.md)
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
