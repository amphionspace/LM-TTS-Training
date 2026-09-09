# Padding-free 训练实现与验证

本文保留当时的迁移与恢复实验记录。当前训练已统一转向 [dynamic batching](dynamic-batching.md)，旧布局迁移兼容代码已删除；下文的历史恢复命令不再作为当前入口。1kh 运行已完成 5,000 步，见 [最终报告](emilia-1kh-report.md)。

当前实现已归入主目录 `/119010446/LM-TTS-Training`，临时工作目录已移除。原四卡训练已暂停，完整第 1000 步 checkpoint 和评估音频保留。当前没有正式 padding-free 训练任务，重启步骤见 [当前进展与恢复说明](restart-progress.md)。

## 数据和模型行为

`collate(rows)` 将文本 token、codec 帧和 speaker mel 分别连续存储，长度单独记录：

| 字段 | 形状 |
|---|---|
| `text_ids` | `[sum(text_lengths)]` |
| `codes` | `[sum(frame_lengths), 16]` |
| `speaker_mels` | `[sum(speaker_lengths), 128]` |
| 三组 lengths | 各为 `[样本数]` |

每条样本的协议、文本内容、speaker 条件和音频帧保持原样。speaker encoder 按相同长度分组处理完整录音，既不跨录音拼接计算，也不接收 mel padding。

Talker 的每条样本仍为角色、控制 token、speaker、文本、codec BOS、音频帧。所有样本的有效 embedding 合成 `[1, sum(sequence_lengths), hidden]`。每条样本的 position ID 从 0 开始；int32 累积长度 `cu_seq_lens_q/k` 与最大样本长度显式传给 FA2。每个 attention 段内部保持因果关系，样本之间不相互关注。QKV 投影、attention、MLP 均不处理补齐 token。

这使用已安装 Transformers 4.57.3 的 [变长 attention 路径](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/modeling_flash_attention_utils.py) 和 Flash Attention 2.8.3.post1 的 [`flash_attn_varlen_func`](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3.post1/flash_attn/flash_attn_interface.py#L1370)。没有修改第三方包。

Talker 输出只取每条样本用于预测音频的 `T+1` 个位置：BOS 预测第 0 帧，随后逐帧预测，最后一个位置预测 EOS。首码本 head 不计算文本或 padding 位置的 logits。Code Predictor 继续接收 `[有效帧总数, 16, hidden]`，其残余码本目标不变。

## 训练入口与恢复

训练、验证和生成统一采用 packed 数据，不再保留 padded 分支或布局开关。旧的固定时长参考音频读取接口也已移除。正式 CUDA 配置使用 FA2 varlen；CPU／SDPA 检查在同样的展平数据上使用分段因果 mask。生成时只更新连续 codec 帧和每条样本的 frame_lengths。

每步仍取原来的样本集合，采样、epoch、每卡样本数和 accumulation 均未改变。这次没有引入 token-budget 动态 batch。首码本全局分母仍为 `sum(T_i + 1)`，残余码本全局分母仍为 `15 * sum(T_i)`；跨卡和跨 microbatch 的归一化保持原样。

新 checkpoint 记录 `batch_layout: packed`。`load_checkpoint()` 允许已有 padded／未记录布局的 checkpoint 迁移到 packed；其他签名字段和 world size 必须完全一致。模型参数、AdamW 状态、scheduler、epoch/next_batch 和各 rank RNG 均完整恢复，迁移会写入日志。原 checkpoint 不被修改。此迁移不承诺与旧 padded 训练逐值相同；数值轨迹的变化见下文。四卡恢复短测已通过，尚未启动正式 padding-free 长训。

## 验证结果（2026-09-09）

最终日志和 JSON 结果位于本目录的 `runs/padding-free-validation/`；一次性诊断脚本已清理。

统一 packed 版本已完成：

- 34 项测试全部通过，包含 CUDA BF16 的样本隔离、因果关系和单样本／多样本 loss、梯度对照。结果见 `unified-tests.log`。
- 双卡 FSDP2、变长输入、两次梯度累积与未分片 packed 参考对照，相对梯度差 **0.4285%**；冻结参数逐值不变。参考模型仅将参数转换为 BF16，保留 FP32 RoPE buffer，匹配训练的混合精度方式。结果见 `unified-distributed.log`。
- 读取原训练的完整 **step 500** checkpoint，恢复 397 项 AdamW 参数状态、scheduler、`epoch=0 / next_batch=500` 和各 rank RNG。使用四个 Gloo 进程在 **CPU FP32** 上对两条固定真实录音完成一次更新，临时进度到 501；loss **7.44583**、梯度范数 **26.25346**，均有限。保存后再次恢复，抽查的参数、优化器、scheduler 和随机状态一致；不匹配的 batch size 或学习率仍被拒绝。测试临时 checkpoint 已清理，原 checkpoint 未修改。结果见 `existing-resume.json`。

- 四张 A100 80GB、FSDP2、BF16、FA2、每卡 48 条，从正式 **step 1000** 迁移并完成两次更新到 **1002**，随后保存完整 checkpoint 并退出。首 CE **3.26503 / 3.33601**，残余 CE **7.24999 / 7.27881**，梯度范数 **3.79125 / 5.16615**，均有限；rank 0 峰值 allocated **20.58 / 50.90 GiB**。第二步耗时 **3.89 秒**，仅两步数据，不作为稳定吞吐基准。结果见 `resume-1000.log`。
- 从新的 packed checkpoint 重新启动四卡评估，恢复进度为 `step=1002 / next_batch=1002`；完整验证集首 CE **3.15693**、残余 CE **7.23500**。生成了 2 条验证样本及 2 条训练样本音频，其中一条训练样本达到 160 帧上限；这是生成长度截断，不是运行错误。结果见 `eval-1002.log`。

正式训练仍暂停。上述验证覆盖了真实四卡恢复、更新、保存和再次加载，没有检验长期收敛。原第 1000 步 checkpoint 未修改。

前一版双路径对照的历史结果如下，旧实现仅保存在审计记录中：

- CPU 输入 embedding 对照逐条未补齐样本，覆盖两种输入协议、变长文本/音频/speaker mel，以及生成时空音频历史。
- CUDA BF16 对照 padded/packed loss 与梯度，检查其他样本不影响本样本、未来音频不影响此前预测。
- 双卡 FSDP2、每卡每个 microbatch 两条变长样本、两次梯度累积，对照未分片 padded 参考；相对梯度差约 **0.502%**，冻结参数逐值不变。
- 实际组装模型、两条真实录音（文本长度 10/48，codec 帧数 25/125）：FP32 CPU 分段因果 attention 数学参考下，loss 差约 **1.9e-6**，相对梯度差约 **0.00608%**。
- 同一对真实录音、step 500 权重、BF16 + 实际 FA2：首 CE 为 padded **6.14898**、packed **6.15014**；残余 CE 为 **7.41477 / 7.41466**，相对梯度差约 **2.63%**。
- 同一小批量前向/反向的单进程峰值 allocated 从 **3.470 GiB** 降为 **3.208 GiB**，约减少 **7.55%**。这是完整 BF16 模型、两条录音、不含优化器的测量，不代表每卡 batch 48 的 FSDP 训练峰值。Talker 有效位置从 padded 的 362 个降为 packed 的 224 个。

### 数值差异和测量边界

不能宣称 BF16 下逐值等价。实际初始化模型的 BF16 loss 差异较小，但相对梯度差达到 **41.8%**；逐层检查发现输入 embedding 相对差约 0.063%，在主干中逐层放大。关闭 BF16 GEMM 的 reduced-precision reduction 也未消除这个现象。已有 padded batch 与逐条推理同样不逐值相同。FP32 数学参考验证了序列、位置、因果关系和 loss 的语义；step 500 权重的 BF16 对照改善到上述 2.63%。切换布局会改变有限精度下的训练轨迹，仍需训练与收敛观察。

前期 CUDA 数值对照与原训练并发进行，小模型测试限制 PyTorch 分配为单卡容量的 1%，实际模型对照为 7%。速度记录受原训练及生成评估争用影响，不能作为独占 GPU 的吞吐结论。这些前期对照没有中断原训练，也没有并发执行 batch 48/64 的大显存压力测试。原训练暂停后另行完成了上面的四卡 batch 48 恢复短测；最大 batch 和稳定吞吐仍需独立测量。

可重复运行：

```bash
PYTHONPATH=.:tests .venv/bin/python -m unittest discover -s tests
PYTHONPATH=. .venv/bin/torchrun --standalone --nproc_per_node=2 \
  tests/check_distributed_equivalence.py --bf16 --speaker \
  --qwen-protocol --frozen-frontend --frozen-speaker
```
