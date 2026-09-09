# 动态组批

训练入口已统一使用动态组批：PyTorch `DataLoader` 负责 worker、预取和 pinned memory，`DistributedTokenBatchSampler` 只负责生成样本索引列表。配置不再接受固定的 `train.batch_size` 作为组批依据，旧 checkpoint 布局迁移分支已删除。

## 配置与行为

```yaml
train:
  max_batch_frames: 6000
  max_batch_tokens: 9000
  accumulation: 1
  num_workers: 4
  prefetch_factor: 2
eval:
  batch_size: 8
```

两个预算均为**每卡、每个 microbatch 的总量上限**。音频帧限制约束 codec 预测的计算量，Talker token 限制包含文本、控制符、speaker、audio BOS 和目标 codec 帧。当前组装模型每条样本额外计入 10 个 token（其中两个是文本 BOS/EOS）。预算限制总长度，不限制每批录音条数；不裁剪音频或文本。

各 rank 根据同一 seed 和 epoch 重建全局分配顺序。每组先给每卡一条样本，再将下一条样本交给可容纳它且预算占用率最低的 rank；全部容纳不下时开始下一组。各 rank 的 batch 数一致，样本数和实际 token 数可不同。无重复填充；epoch 尾部若不足以给每卡各一条，只丢弃最后不足 world size 的随机样本。单条样本超过预算会明确报错。

loss 仍以当前优化更新中所有 rank、全部累积 microbatch 的实际有效目标数归一化。checkpoint 保存已消费的 epoch 和 batch 游标，恢复通过 Accelerate `skip_first_batches` 跳过索引，不重新解码已消费样本。最后一个训练 epoch 的索引流限制在本次剩余更新所需的 microbatch 数，避免 worker 在训练结束时仍解码额外预取数据；该边界不改变已消费样本的顺序。

TensorBoard 增加 `train/global_samples`、`global_audio_frames`、`global_talker_tokens`、`frame_budget_fill` 和 `token_budget_fill`。这些是整个优化更新的全局统计。现有 `peak_memory_gib` 和 `data_wait_seconds` 为 rank 0 的记录。

## 验证

证据目录：[`runs/emilia-10kh-dynamic-trial-20260909`](../../runs/emilia-10kh-dynamic-trial-20260909/)。

- 单元测试覆盖预算边界、各 rank 步数一致、无重复、尾部样本、epoch 重排、恢复后的 batch 边界，以及 spawned worker 读取实际音频和 codec。
- 四卡 BF16/FA2、变长样本和变动 batch 样本数、两次梯度累积，与单模型全局 batch 梯度对照通过；相对梯度误差约 0.00318。
- 两卡训练连续运行三步，与一步后恢复至第三步对照：320 个 checkpoint/RNG tensor 逐值一致，优化器、scheduler、进度和 RNG 元数据也一致。
- 6,000 帧 / 9,000 token 预算的最长输入压力测试通过两次实际优化更新，最大 allocated 57.15 GiB、reserved 63.73 GiB。

## 真实数据试训

使用 [动态试训配置](../../configs/emilia-10kh-dynamic-trial.yaml)，从原始组装模型重新开始，不读取旧训练 checkpoint。数据复用固定试训的 65,550 条、约 89.29 小时训练快照；它不是完整 10kh 数据集。64 条验证样本保留英语、中文各 32 条。初始化、数据与学习率背景见 [快照说明](emilia-10kh-partial-trial.md)。

第一个 epoch 共 172 组四卡 microbatch，零丢弃。平均帧预算填充率 98.03%，每卡每批 33–120 条，平均 95.28 条；同一步各 rank 的帧数最大差为 207 帧。完整统计见 [batch-plan.json](../../runs/emilia-10kh-dynamic-trial-20260909/batch-plan.json)。

最初每卡 4 个 worker 的动态组批出现 CPU 解码供数停顿，测量日志保留在 `train-workers4.log`；后续试训改为每卡 16 个原生 DataLoader worker，输出独立写入 `train-workers16/`。该 worker 数依据本机约 76 核 CPU 配额设置，并非所有机器的默认值。

本次已完成 100 次优化更新，累积为 1；完整 checkpoint、验证 loss，以及 speaker-only / ICL 的英语和中文短生成链路均已记录。生成最多新增 8 帧且关闭 ASR，只验证工程链路，不评价音质或收敛。运行日志为 [train.log](../../runs/emilia-10kh-dynamic-trial-20260909/train.log)。

固定 batch 试训已停止，旧配置只保留在其 run 的实际配置记录中。1kh 和此前试训产物保留，未用于本次输出。

## 100 步结果

性能按第 2–100 步的音频总时长除以训练步耗时总和计算，包含读取及同步等待，排除初始化、首步、checkpoint 保存和评估。两次配置使用同一数据快照，但样本分组和每步数据量不同；此对照反映**动态组批与增加 worker 的整体效果**。

| 配置 | 每卡解码 worker | 吞吐：音频秒/墙钟秒 | rank 0 峰值 allocated |
|---|---:|---:|---:|
| 原固定 batch 16、累积 3 | 4 | 103.21 | 19.71 GiB |
| 动态 6,000 帧 / 9,000 token、累积 1 | 16 | 451.95 | 57.15 GiB |

整体吞吐比约 **4.38×**。动态组批 100 步共处理 38,169 条录音，每步全局 345–425 条；平均帧预算填充率 **98.27%**，Talker token 预算填充率 **96.55%**。测量步骤的中位耗时约 2.20 秒，平均耗时约 4.17 秒；其他 rank 的解码等待仍可能使单步变慢。

第 100 步验证首 CE 为 6.96717，残余 CE 为 7.58934。生成链路结果如下；训练仍在 warmup，不能据此评价收敛或音质。

| 条件 | 英语新增帧数 | 中文新增帧数 |
|---|---:|---:|
| speaker-only | 6，遇到 EOS | 8，达到检查上限 |
| ICL | 0，立即 EOS | 8，达到检查上限 |

两种条件使用同一组目标/参考录音。ICL 英语没有写出空音频文件，而是记录提前 EOS 和空输出指标。完整数据见 [trial-result.json](../../runs/emilia-10kh-dynamic-trial-20260909/trial-result.json)。

首次 100 步运行的训练与评估完成、主进程退出码为 0，但 worker 在销毁时出现 abort 警告。随后已限制最终 epoch 的索引流，避免超出停止步的预取；同一四卡、每卡 16 worker 配置从第 100 步续跑至 103 步并保存完整 checkpoint，**退出码为 0，未再出现 worker abort 或 traceback**。修复后的两卡跨 epoch 累积与恢复对照仍为 320 个 tensor 逐值一致。记录见 `bounded-resume-real.log`、`bounded-resume-comparison.json`；原始异常日志保留。

已清理固定 batch 训练函数、旧布局迁移分支、旧固定试训入口、未使用的 speaker 裁剪配置，以及旧试训的两个一次性脚本。清理记录见 `code-cleanup.json`；快照、日志、checkpoint 和评估音频保留。
