# 10kh 已完成分片上的从头试训

试训目录：[`runs/emilia-10kh-partial-trial-20260909`](../../runs/emilia-10kh-partial-trial-20260909/)。配置：[当时的实际配置](../../runs/emilia-10kh-partial-trial-20260909/train/config.yaml)。本次用于验证长句数据读取、显存、优化更新，以及新增的 speaker-only/ICL 生成链路。

## 数据范围

2026-09-09 12:09 UTC 选取快照时，10kh 数据已有 531/832 个 prepared 分片发布。只复制当时最前 4 个和最后 4 个完整 JSONL 分片，复用其绝对路径指向的 codec 缓存；后续数据准备不会改变此快照。该选择特意包含独立 short 和 long 中的短句视图，不代表完整 10kh 的随机样本。

| 项目 | 数量 |
|---|---:|
| 总录音数 / 时长 | 65,614 / 89.3725 h |
| 训练录音数 / 时长 | 65,550 / 89.2936 h |
| 验证录音数 | 64，英语和中文各 32 |
| 英语录音数 | 28,832 |
| 中文录音数 | 36,782 |
| 独立 short | 32,768 |
| long 短句视图 | 32,846 |
| 最长音频 / codec 帧数 | 30 秒 / 375 |
| 最长原始文本 token 数 | 174 |

完整分片路径及 SHA256 位于 [snapshot.json](../../runs/emilia-10kh-partial-trial-20260909/data/snapshot.json)，划分统计位于 [preparation.json](../../runs/emilia-10kh-partial-trial-20260909/data/preparation.json)。快照内部检查 ID 唯一、train/val 规范化文本隔离，验证样本同 speaker 保留至少两条训练录音。它只服务此次短测，正式 ICL evaluation 数据等完整 10kh 完成后再统一构造。

## 初始化、学习率与计算配置

从 `pretrained/assembled-qwen3-tts-frozen-conditioning` 加载原始组装模型，未传入 `--resume`，也未读取 1kh 的训练 checkpoint。Talker 主干来自 Qwen3-0.6B-Base，音频新模块为原始随机初始化；冻结的文本前端和 ECAPA 沿用公开 TTS 权重。

| 项目 | 配置 |
|---|---|
| GPU | 4 × A100 80GB，FSDP2 |
| 计算 | BF16、Flash Attention 2；关闭 activation checkpointing |
| 每卡 batch / 梯度累积 | 16 / 3 |
| 有效全局 batch | 16 × 4 × 3 = 192 |
| DataLoader | 每 rank 4 个 worker，prefetch_factor=2，spawn、persistent workers、pinned memory |
| 主干峰值 LR / 新模块峰值 LR | 1e-4 / 3e-4 |
| warmup | 1,000 步 |
| 本次优化更新 | 100 步 |
| 本次 scheduler horizon | 1,000 步，仅用于这次 warmup 内短测，不是完整 10kh 的训练预算 |
| 验证 / 保存 | 第 100 步；checkpoint 不自动清理 |
| 生成检查 | 两种条件各生成英语、中文各 1 条；最多新增 8 帧；关闭 ASR |

100 步均处于 warmup，实际使用的 LR 低于设定峰值。本次无法检验峰值 LR 的收敛稳定性，也不以短生成的音质或截断率判断正式训练质量。源码哈希和启动设置记录于 [trial-start.json](../../runs/emilia-10kh-partial-trial-20260909/trial-start.json)。

## 显存压力测试

使用快照中最长音频，配上最长文本，并让每个 microbatch 的全部样本都达到这一长度；每档测试执行完整前向、反向和 AdamW 更新。

| 每卡录音数 | 结果 |
|---:|---|
| 48 | CUDA OOM |
| 24 | CUDA OOM |
| 16 | 两次完整优化更新通过；峰值 allocated 57.15 GiB，reserved 63.73 GiB |

对应日志位于 run 根目录的 `memory-batch48.log`、`memory-batch24.log`、`memory-batch16.log`。实际随机混合 batch 通常更短，显存低于这个极限组合；这次选 16 并用累积 3 次保持有效 batch，不是简单把最长时长增加倍数当作显存倍数。

## 运行与结果记录

历史启动命令（当前训练已切换动态组批；该固定 batch 配置仅作为运行记录保留）：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 NPROC_PER_NODE=4 \
  bash scripts/run_train.sh --config runs/emilia-10kh-partial-trial-20260909/train/config.yaml \
  > runs/emilia-10kh-partial-trial-20260909/train.log 2>&1
```

运行日志在 [train.log](../../runs/emilia-10kh-partial-trial-20260909/train.log)。`train/data_wait_seconds` 是 rank 0 获取本步 microbatch 的等待时间；其他 rank 的读取阻塞可能表现为之后的分布式同步等待，因此不能把它当作全局数据等待的完整统计。

ICL 代码及边界验证见 [ICL evaluation 说明](icl-evaluation.md)。两种生成条件复用同一目标/参考配对，ICL 增加参考文本与 codec 前缀，解码后裁去参考波形。生成检查只验证完整链路，正式评测需要独立构造的数据和更充分的生成长度。

本次 run 与 1kh 完全分开。已有 1kh 产物保持原样，保护范围见 [保留清单](run-retention.md)。

固定 batch 试训已按用户要求停止：100 次优化更新及验证 loss 已完成并保存 checkpoint，生成评估被中断。后续改用 [动态组批试训](dynamic-batching.md)。
