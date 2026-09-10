# 环境、训练与评估入口

当前主线为中英10kh动态组批预训练，配置见 [`configs/emilia-10kh-pretrain.yaml`](../../configs/emilia-10kh-pretrain.yaml)。模型与loss见[完整pipeline](pretraining-pipeline.md)，数据生产见[10kh数据准备](emilia-10kh.md)。

## 环境与初始化

本机使用独立 `.venv`：Python3.10、PyTorch2.8.0+cu126、Transformers4.57.3、qwen-tts0.1.1、Flash Attention2.8.3.post1。依赖和wheel校验值固定在 `requirements.txt`，完整安装版本记录在 `requirements.lock.txt`。

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python scripts/download_models.py
python scripts/assemble_qwen3_tts.py --output pretrained/assembled-qwen3-tts-frozen-conditioning
```

组装输出必须为新目录；已存在的正式初始化权重直接复用。默认冻结公开文本前端与ECAPA，Talker layers/norm从文本Base加载，音频模块新初始化。完整来源见[模型组装](../design/model-assembly.md)。

## 正式运行与恢复

当前run由带文件锁的launcher管理，并记录PID身份和退出状态。确认原训练已结束后，使用：

```bash
.venv/bin/python runs/emilia-en-zh-dynamic-10000h/launch.py
```

它自动从latest完整checkpoint恢复模型、优化器、scheduler、RNG和数据游标。不要在训练存活时重复启动，也不要直接修改checkpoint签名。人工干预需与当前巡检协调，步骤见[故障处置](training-incident-playbook.md)。

```bash
.venv/bin/tensorboard --logdir runs --port 6006
```

训练记录首/残余CE、梯度、LR、实际样本/帧数、预算填充和吞吐。验证按全局有效token计算CE。生成分别写入 `evaluation/speaker_only/step-*/` 与 `evaluation/icl/step-*/`，包括音频、逐句metrics和summary；最短生成长度、参考配对与ASR口径见[ICL评估](icl-evaluation.md)。

## 验证入口

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=. .venv/bin/torchrun --standalone --nproc_per_node=4 \
  tests/check_distributed_equivalence.py \
  --speaker --qwen-protocol --frozen-frontend --frozen-speaker
```

分布式检查需要可用GPU资源，避免与正式训练争抢显存。1kh历史固定batch checkpoint需使用其对应源码，不能当作当前动态组批run恢复；源码归档位于该run的 `provenance/`。
