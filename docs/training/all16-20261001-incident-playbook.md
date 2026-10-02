# all16 BF16 训练巡检与恢复手册

本文件是后台 Codex agent 每次巡检必须重新读取的指令。用户已授权每小时检查、自主诊断必要修复和恢复，OOM 可先按原配置尝试一次重启；无需等待前台盯到 2500 步。结论和动作追加到 [巡检记录](all16-20261001-supervision.md)，不要覆盖旧记录。

## 本轮身份与不变量

- 仓库：`/workspace/workspace/yanglin/LM-TTS-Training`。
- Run：`/workspace/LM-TTS-Training-Runs/supervised-tts-20260929-all16-bf16-32gpu-lr3e-4-bblr1e-4-ep3-wsd`，同名配置在 `configs/`，初始 ACP job 为 `pt-blb39rgw`。
- 原始提交：`submissions/20261001T120352364999Z`，训练源码快照 `7a4c303`；重启后以各 submission/job.json 和现场平台状态交叉核对当前 job。
- 4 节点 × 8 卡、BF16、token loss、3 epochs，全部 16 个数据集；每卡 32000 codec frames / 50000 tokens，accumulation=1。WSD 随已消费样本进度，峰值新模块 LR=3e-4、backbone=1e-4，warmup 前 0.03 epoch，末 0.3 epoch decay。不得自行改成固定步数或更改实验目标。
- Talker 使用外部 FA2 varlen；CodePredictor 使用原生 SDPA efficient/math。大批量 FA2/原生 Flash 曾出现 NaN/非法访存，不恢复那些不稳定路径。
- 首次 step 100 已完成 128219 条 holdout eval 和完整 checkpoint；此后每 2500 步以及训练结束自动 eval/save。训练 eval 是 teacher-forcing loss/CE，不是自动音频生成、WER/CER、DNSMOS 或相似度评分。
- 用户允许跳过损坏样本的既有实现继续生效；巡检不得再改清单、划分、跳过模型计算异常或重写签名绕过恢复校验。

## 每小时检查内容

1. 先读本文、巡检记录最后几段、`supervision/agent-state.json`（若存在）、最新 submission 的 job.json。平台 job ID 与 run、display_name、启动脚本必须匹配。通过 `qwen3_train.config.read_yaml` 和 `scripts.acp.api.request/jobs_url` 查询 ACP；不要输出凭据文件或认证头。
2. 从 `metrics.jsonl` 提取上次至今的 train/val，核对 step、epoch、LR、有限 loss/grad、skipped/discarded、data_wait_seconds/fraction、step_seconds、吞吐与峰值显存。日志每 10 步记录的是单步数值，不能冒充连续全部训练步统计。跳过半行 JSON，不能用空日志认定训练正常。
3. 检查 `logs/` 下四节点 GPU CSV 的新鲜度、32 卡覆盖、显存/利用率、stdout/worker 错误，结合平台状态判断是否训练、排队、eval、保存或退出。API 返回失败时记录“平台不可确认”，不把旧 GPU 样本说成当前利用率。
4. 检查最近完整 checkpoint 的 COMPLETE、metadata.json、DCP `.metadata`、32 个 RNG 文件与分片，核对进度/world_size=32；新 eval 与 checkpoint 应同 step，正在写入时给出合理等待时间。**每轮优先执行下一节的固定归档命令，保存所有新完整 checkpoint，不能只记录存在而不归档。** TensorBoard 在 `tensorboard/`，指标原始值在 `metrics.jsonl`。
5. 检查可用磁盘与异常数据增长。正常状态执行 checkpoint 归档并记录摘要，不停止、重启或为了满显存调整配置。持续低利用率先区分数据 CPU 组批、AFS I/O、eval/save 等阶段；当前基线 NVML 约 66.5–66.8 GiB/卡，训练阶段利用率高，正常 train step 约 7–8 秒，数据等待约几个百分点。
6. 在本文关联记录追加 UTC、job、step/epoch、最近 eval/ckpt、资源/数据等待、判断、动作和后续事项。详细证据写 run/supervision/ 下带 UTC 时间戳 JSON；更新 agent-state.json 的 current_job、last_step、last_checked、status、已用重试及事件。凭据不得进入记录。当前主会话正在提交文档时不要并发 git commit；后续每轮无需自动 commit/push，工作树中的追加记录即持久化。必要代码修复在针对性测试及真实恢复验证通过后可按既有授权 commit/push（`lucky9cyou <lucky@lucky9.cyou>`），只包含本次修复文件，不得 force push。

## 完整 checkpoint 的永久归档（2026-10-02 用户新增要求）

用户要求 supervisor 每次发现新的完整 checkpoint，就执行固定命令保存到轮转之外。目标固定为：

```text
/workspace/LM-TTS-Training-Runs/supervised-tts-20260929-all16-bf16-32gpu-lr3e-4-bblr1e-4-ep3-wsd/archived-checkpoints/step-XXXXXXXX/
```

每次巡检在仓库根目录执行下面同一条命令，不改目标、不重新实现复制、不只归档 latest：

```bash
/workspace/workspace/yanglin/envs/lm-tts/bin/python -m scripts.archive_checkpoints \
  --run /workspace/LM-TTS-Training-Runs/supervised-tts-20260929-all16-bf16-32gpu-lr3e-4-bblr1e-4-ep3-wsd
```

该命令只读取完整的 `checkpoints/step-XXXXXXXX`，核对 run 签名、step、DCP metadata 和各 rank RNG，复制所有 checkpoint 文件。逐文件比较源读取流与目标文件的 SHA256，一致后写入 `ARCHIVE_COMPLETE.json`，再原子发布归档目录。**是独立复制，不是 mv、软链接或硬链接**；不删除源 checkpoint、不改 latest、不修改 keep_checkpoints。训练仍只保留两个工作 checkpoint，归档目录保留所有成功归档的版本，不能自行轮转或清理。

命令有互斥锁和重复执行检查。已有归档会核对结构、文件清单 / 大小与 metadata，然后返回 `already_archived`，不会每小时重拷贝大文件。需要重新核对归档内容时，在同一命令末尾加 `--verify`，它会重新计算所有归档文件 SHA256，包括源 checkpoint 已被训练轮转删除的版本。首次归档已执行全文件 SHA256 校验；不能把日常跳过重复归档描述为再次全量校验。

将命令结果、归档 step、目标路径和失败原因写入每轮巡检记录。复制 / 校验失败不能记录为成功；先定位 I/O、容量或源 checkpoint 已被轮转的问题，再有限重试，不得手工补归档回执，也不得为释放空间删除已有归档。仅凭目录存在不能认定归档完成，必须有有效回执。开始归档前已被轮转删除的版本不能凭日志恢复。

归档和训练输出目前在同一 AFS，用于防止训练轮转删除，**不是异地容灾备份**。每份完整 checkpoint 当前约 7.86 GiB，记录归档总量与可用空间。每小时巡检需要正常运行才能发现新版本；长期巡检中断可能错过已被轮转删除的 checkpoint，应如实记录缺口。

需要从归档恢复时先加 `--verify` 完成全量校验，再使用归档 checkpoint 的绝对路径作为 `--resume` 参数；仍须满足下述完整恢复步骤，不更改训练签名。

## 故障处置

| 现场 | agent 的判断与允许动作 |
| --- | --- |
| 单次 loss 波动、短暂无进度或低 GPU 利用率 | 先排除正常 eval/save、epoch 边界、排队和数据窗口切换。结合至少两个时间点与对应日志；不要因一个采样中断。 |
| CUDA OOM | 保存首个报错 rank/阶段、峰值显存、最近完整 checkpoint。确认原 job 和所有 rank 已退出，无重复训练；若有完整 checkpoint，允许**本轮全程一次**原配置新 job 恢复，记录 `oom_original_config_retries=1`。它可以排除瞬时外部占用/碎片等因素，不能保证解决稳定复现的容量问题。同样 OOM 再现，停止无变化重启，定位训练/eval/加载阶段根因。不得擅自减 batch 预算绕过签名。 |
| 节点掉线、AFS/网络瞬态、NCCL 错误 | 找首个根因，不把其他 rank 被动报错当主因。确认外部故障已消失且旧 job 终态后，可原配置完整恢复；同一事件最多一次，全轮这类重试累计最多三次，超过则记录阻塞。不能无限增加 timeout 或无限重试。 |
| NaN/Inf、非法访存 | 保存第一处异常及配置/attention 路径；检查输入与梯度/权重有限性。需可复现根因、必要修复、针对性测试和真实恢复证据；不能直接重启碰运气、跳更新、清空 optimizer 或任意降 LR。不可证明修复时保留完整恢复点并记录阻塞。 |
| data worker / CPU 内存 / 吞吐持续退化 | 排除 eval/save 和样本长度差异，比较连续多个正常训练日志点。workers/prefetch 不改变组批序列且不在恢复签名内；有证据可调这两项后恢复并比较数据等待/吞吐/内存。无证据不要调参。 |
| 数据损坏 | 既有按样本跳过会记录计数；突增时定位数据集/文件与挂载情况。不得删源数据、重新随机划分或用零值伪造样本。必要的读取实现修复需验证不改变正常样本或 sampler 顺序。 |
| checkpoint 写失败或损坏 | 不手工补 COMPLETE。不完整 checkpoint 不可 resume。检查上一个完整可加载恢复点并记录回退步数；无完整 checkpoint 时不自动从头重跑。不删除现有完整 checkpoint、源数据、模型或失败现场。 |
| 持续卡死 | 两次采样确认 train、eval/save 文件与平台日志均无进展，并结合 GPU/IO/进程状态。收集现场后才停止本 run 的准确 job，确认所有 rank 退出，修复根因再恢复。严禁停止其他实验。 |
| ACP POST 响应不明 | 先查平台同 display_name/启动脚本和 submission/request.json，判断是否实际创建成功。**不得直接重复 POST**。只有确认没有分配 job 才重新提交；保留查询与判断证据。 |
| Codex 认证、服务或本机进程中断 | 调度状态记录 review_failed；下一小时继续同一 session。没有成功 agent 审查就不能称“训练正常”。主机/容器重启不会由 tmux 自动复活；重新运行调度命令沿用 session。训练在 ACP 独立继续。 |

## 完整恢复步骤

1. 保存故障现场与每次重试计数，确认失败 job 已终态，查询同 run 的其他 job，确保不存在 RUNNING/PENDING 的重复任务。需要停止时先查本机 CLI/API 帮助，按**准确 job ID** 操作并复核终态。
2. 选择本 run 最近的有效 COMPLETE checkpoint（必要时上一份），检查 metadata 进度、32 ranks、优化器/scheduler/RNG 分片。对照原提交实验配置和最新配置，保持模型、数据、动态 batch、LR、world size 等恢复语义一致；不编辑旧 metadata、不修改签名版本、不只加载模型冒充完整 resume。
3. 从仓库执行：

   ```bash
   /workspace/workspace/yanglin/envs/lm-tts/bin/python -m scripts.acp.submit \
     --config configs/supervised-tts-20260929-all16-bf16-32gpu-lr3e-4-bblr1e-4-ep3-wsd.yaml \
     --resume latest --submit
   ```

   回退旧 checkpoint 时使用该已核对 checkpoint 的绝对路径替换 `latest`。提交产生新的 submission 与 job.json；立即登记新 job ID、旧 job、恢复 step、原因和变更。原 job 正常运行时绝不执行此命令。提交前必须确认工作树源码修复已经验证；当前生产运行使用快照，仓库改动不会热更新进程。
4. 等待新 job 启动，核对加载 checkpoint、恢复 step/epoch/next_batch/samples_in_epoch、optimizer/scheduler/RNG，无签名跳过；观察至少 20 个 optimizer update（两个日志点），loss/grad 有限、LR 连续、数据游标推进、吞吐恢复。一次恢复不通过不得谎称完成。
5. 若需要改变实验语义、缺失数据或凭据等不能自主解决，只记录具体阻塞及保留的恢复点，不反复消费资源，不发送 Slack/邮件等外部消息。每小时可继续检查阻塞条件是否改变。

## 完成与调度控制

3 epochs 结束后，核对 ACP 成功终态、`completion.json` 的最终进度、最后 checkpoint 完整且进度达到目标、对应最终 val 与 TensorBoard，执行固定命令确保最终 checkpoint 也已归档，记录最终摘要。只有 agent 核实完成后才创建 `supervision/COMPLETE`，让调度器退出；不能仅因为 job 退出、API 不可用或训练 step 暂停就写此标记。无需额外启动 3 epochs 之外的训练或新评估实验。

调度每 3600 秒唤醒固定持久 Codex session，单轮最多 3000 秒且有独占锁；代码只负责唤醒和记录 CLI 退出，不做训练规则判定或自动 restart。`supervision/STOP` 只停止后续巡检，不影响训练；删除该标记后重新启动调度即可接续同一 session。宿主机/容器需持续运行，CLI 认证需可用；不保证跨宿主机重启自动唤醒。CLI 机制见 [OpenAI 非交互模式说明](https://learn.chatgpt.com/docs/non-interactive-mode)。
