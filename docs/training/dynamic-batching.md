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

默认 token loss 以当前优化更新中所有 rank、全部累积 microbatch 的实际有效目标数归一化；另可选择下面的 sample 或 sqrt 聚合。checkpoint 保存已消费的 epoch 和 batch 游标，恢复通过 Accelerate `skip_first_batches` 跳过索引，不重新解码已消费样本。最后一个训练 epoch 的索引流限制在本次剩余更新所需的 microbatch 数，避免 worker 在训练结束时仍解码额外预取数据；该边界不改变已消费样本的顺序。

TensorBoard 增加 `train/global_samples`、`global_audio_frames`、`global_talker_tokens`、`frame_budget_fill` 和 `token_budget_fill`。这些是整个优化更新的全局统计。现有 `peak_memory_gib` 和 `data_wait_seconds` 为 rank 0 的记录。

启动脚本默认设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，减少动态 batch 尺寸变化造成的显存碎片；显式传入该环境变量可覆盖默认值。实际恢复曾在 Flash Attention 反向申请5.85 GiB时OOM，现场仍有21.60 GiB的PyTorch保留但未使用显存，因此不应只看总显存或据此直接减小batch预算。此分配策略不改变训练目标、batch序列或学习率计划，机制见[PyTorch 2.8显存管理说明](https://docs.pytorch.org/docs/2.8/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)。

## Loss 聚合

通过 `train.loss_reduction` 选择 `token`（默认）、`sample` 或 `sqrt`：

```yaml
train:
  loss_reduction: sample
```

设样本 i 的有效目标数为 n_i，样本内平均交叉熵为 meanCE_i。三种模式统一为：

```text
L = Σ_i n_i^α × meanCE_i / Σ_i n_i^α
token:  α = 1     按有效 token 数加权
sample: α = 0     每条样本等权
sqrt:   α = 1/2   按有效 token 数的平方根加权
```

首码本和残余码本分别聚合，再计算 `L_first + residual_weight × L_residual`。首项 `n_i = T_i + 1`，包含 EOS；残余项 `n_i = 15 × T_i`，包含全部 15 个残余码本。文本、prompt 和控制符不计入 n_i。`sqrt` 是调整样本权重，并非对 CE 本身开平方。

分子、分母均覆盖一次更新中的所有 rank 和全部累积 microbatch，FSDP 反传乘 `world_size` 抵消梯度平均。验证在整个验证集上按同一规则聚合，维持调用次数的占位样本不参与统计。

`train/first_ce`、`train/residual_ce`、对应的 `val/` 指标及逐码本 CE 始终保留 token 平均含义，方便跨实验对比。选择 `sample` 或 `sqrt` 后，额外记录 `first_sample_ce` / `residual_sample_ce` 或 `first_sqrt_ce` / `residual_sqrt_ce`，训练与验证均有；两项按 `residual_weight` 组合即所选目标。

显式设置的 `loss_reduction` 参与 checkpoint 配置签名，恢复时必须保持一致。旧配置不添加该字段即可继续以原 token 目标恢复。切换聚合模式需新建训练实验，不能直接修改配置续训已有 checkpoint。当前正式训练配置未切换。

回归检查用逐条样本的独立 token CE 均值对照三种模式的 loss 和全部参数梯度。两卡、两次梯度累积、变长且每卡样本数不同的检查中，`sample` / `sqrt` 的 FP32 相对梯度误差分别约 `3.58e-7` / `3.10e-7`；带冻结 speaker/frontend 的 BF16/FA2 对照分别约 `0.00452` / `0.00404`，验证集检查包含零权重占位样本。可在有空闲 GPU 时复跑：

```bash
PYTHONPATH=. .venv/bin/torchrun --standalone --nproc_per_node=2 \
  tests/check_distributed_equivalence.py --loss-reduction sample \
  --bf16 --speaker --qwen-protocol --frozen-frontend --frozen-speaker
```

将 `sample` 替换为 `sqrt` 即检查另一模式；去掉 `--bf16` 可检查 FP32。

## 验证与当前运行

单元测试覆盖预算边界、各rank步数一致、无重复、epoch重排、恢复边界和spawned worker实际读取。历史四卡BF16/FA2对照包含变长样本、各卡不同样本数及两次梯度累积，与单模型全局batch相比相对梯度误差约0.00318；两卡连续运行与中断恢复对照的320个checkpoint/RNG tensor逐值一致。旧快照试训产物已按2026-09-10清理要求删除，这些数值是历史验证记录。

当前正式配置为每卡6000帧/9000 token、16个worker、prefetch_factor=2。完整10kh数据的预检、最长输入显存压测、实际训练与恢复证据均保存在 `runs/emilia-en-zh-dynamic-10000h/`，见[正式训练记录](emilia-10kh-supervision.md)。

原生DataLoader负责预取；long短句在线读取通过tar成员边界和时间seek仅解码目标附近片段。增加worker和动态组批并不能代替这一解码修复。`data_wait_seconds`为rank 0取批等待，不能独自代表所有卡的等待时间。
