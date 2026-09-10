# 动态组批

训练入口已统一使用动态组批：PyTorch `DataLoader` 负责 worker、预取和 pinned memory，`DistributedTokenBatchSampler` 只负责生成样本索引列表。配置不再接受固定的 `train.batch_size` 作为组批依据，旧 checkpoint 布局迁移分支已删除。

## 配置与行为

```yaml
train:
  max_batch_frames: 6000
  max_batch_tokens: 9000
  accumulation: 1
  num_workers: 16
  prefetch_factor: 2
eval:
  batch_size: 8
```

两个预算均为**每卡、每个 microbatch 的总量上限**。音频帧限制约束 codec 预测的计算量，Talker token 限制包含文本、控制符、speaker、audio BOS 和目标 codec 帧。当前组装模型每条样本额外计入 10 个 token（其中两个是文本 BOS/EOS）。预算限制总长度，不限制每批录音条数；不裁剪音频或文本。

各 rank 根据同一 seed 和 epoch 重建全局分配顺序。每组先给每卡一条样本，再将下一条样本交给可容纳它且预算占用率最低的 rank；全部容纳不下时开始下一组。各 rank 的 batch 数一致，样本数和实际 token 数可不同。无重复填充；epoch 尾部若不足以给每卡各一条，只丢弃最后不足 world size 的随机样本。单条样本超过预算会明确报错。

loss 仍以当前优化更新中所有 rank、全部累积 microbatch 的实际有效目标数归一化。checkpoint 保存已消费的 epoch 和 batch 游标，恢复通过 Accelerate `skip_first_batches` 跳过索引，不重新解码已消费样本。最后一个训练 epoch 的索引流限制在本次剩余更新所需的 microbatch 数，避免 worker 在训练结束时仍解码额外预取数据；该边界不改变已消费样本的顺序。

TensorBoard 增加 `train/global_samples`、`global_audio_frames`、`global_talker_tokens`、`frame_budget_fill` 和 `token_budget_fill`。这些是整个优化更新的全局统计。现有 `peak_memory_gib` 和 `data_wait_seconds` 为 rank 0 的记录。

## 验证与当前运行

单元测试覆盖预算边界、各rank步数一致、无重复、epoch重排、恢复边界和spawned worker实际读取。历史四卡BF16/FA2对照包含变长样本、各卡不同样本数及两次梯度累积，与单模型全局batch相比相对梯度误差约0.00318；两卡连续运行与中断恢复对照的320个checkpoint/RNG tensor逐值一致。旧快照试训产物已按2026-09-10清理要求删除，这些数值是历史验证记录。

当前正式配置为每卡6000帧/9000 token、16个worker、prefetch_factor=2。完整10kh数据的预检、最长输入显存压测、实际训练与恢复证据均保存在 `runs/emilia-en-zh-dynamic-10000h/`，见[正式训练记录](emilia-10kh-supervision.md)。

原生DataLoader负责预取；long短句在线读取通过tar成员边界和时间seek仅解码目标附近片段。增加worker和动态组批并不能代替这一解码修复。`data_wait_seconds`为rank 0取批等待，不能独自代表所有卡的等待时间。
