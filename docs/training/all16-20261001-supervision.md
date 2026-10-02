# all16 BF16 训练巡检记录

训练说明见 [本轮配置与验证](all16-20261001.md)，每次 agent 巡检遵循 [处置与恢复手册](all16-20261001-incident-playbook.md)。用户于 2026-10-01 授权后台 agent 每小时巡检，代替前台持续盯到 2500 步；正常训练不干预，允许有界故障恢复。

原始 job：`pt-blb39rgw`。Run：`/workspace/LM-TTS-Training-Runs/supervised-tts-20260929-all16-bf16-32gpu-lr3e-4-bblr1e-4-ep3-wsd`。所有 session、CLI 原始输出与结构化现场保存在该 run 的 `supervision/`，本文保存可读摘要。eval 自动随训练执行，结果在 `metrics.jsonl` 的 `val` 记录与 `tensorboard/`。

## 后台入口

```bash
/workspace/workspace/yanglin/envs/lm-tts/bin/python -m scripts.acp.schedule_agent \
  --run /workspace/LM-TTS-Training-Runs/supervised-tts-20260929-all16-bf16-32gpu-lr3e-4-bblr1e-4-ep3-wsd \
  --instructions docs/training/all16-20261001-incident-playbook.md
```

入口在独立 tmux 中运行；互斥锁拒绝重复调度，固定 session ID 写在 `supervision/session-id`。每小时 agent 自己读现场、诊断、执行授权动作并追加记录。`scheduler-status.json` 是 CLI 调度状态，**不等于训练健康状态**。停止后续巡检可创建 `supervision/STOP`；训练不受影响。宿主机或容器退出后需重启这一入口，现有 session 和记录仍保留。

2026-10-01 已实际启动 tmux `all16-agent-supervision`，调度 PID `1892623`；固定 Codex session 为 `01a0f787-d1cc-7c91-9fc2-b34131b34a04`。首轮 12:54:27–12:58:33 UTC 成功完成，CLI exit code=0；下一轮 13:54:27 UTC，此后每小时执行。已实测 `exec resume` 返回同一 session ID 且保留首轮结论，见 `supervision/resume-check.jsonl`。第二个调度实例被互斥锁拒绝，没有并发巡检；旧 45 秒 observer 已按进程身份核对后停止。采用 `workspace-write` 和自动审批审核，不关闭沙箱。模型诊断、修复和恢复由每轮 agent 执行，调度脚本只负责定时唤醒。

## 2026-10-01 12:54:27 UTC：首次定时巡检，正常训练

- **身份与平台**：本次巡检开始于 `2026-10-01T12:54:27.024158+00:00`。此前无 `agent-state.json`，因此建立首次基线，检查已有全部日志。最新且唯一 submission 为 `20261001T120352364999Z`，源码快照 `7a4c303`。12:56:11 UTC 经 `read_yaml` 与 `scripts.acp.api.request/jobs_url` 查询，ACP job `pt-blb39rgw` 为 **RUNNING**，run、display_name、启动脚本均匹配，4 节点 × 8 卡。首次网络沙箱请求被拒，授权的只读外部查询成功，未泄露凭据。
- **进度与数值**：检查期间先见 step 370，12:57:03 UTC 证据采样时已到 **step 380 / epoch 0.0244299644**，metrics 最近写入为 12:56:29 UTC。最新 objective **6.448041**、first CE **4.279502**、residual CE **7.228463**、grad norm **1.084082**；backbone LR **8.14332147e-5**、新模块 LR **2.44299644e-4**，符合前 0.03 epoch warmup。已解析 step 10–380 共 38 条 train 和 1 条 val，所有数值有限，采样记录的 skipped/discarded 均为 0，无半行 JSON。日志每 10 步仅记该单步，以上不代表未记录的每一步均已检查。
- **吞吐与资源**：最近十个日志点（step 290–380）单步 **7.456–7.917 秒**（均值 7.679），数据等待 **0.082–0.487 秒 / 1.10%–6.15%**（比例均值 **2.28%**），吞吐 **10287–10932 音频秒/秒**。PyTorch 记录峰值显存 **60.30–60.64 GiB**；四份 GPU CSV 共 32 个不同 UUID，采样距检查 **4.7–12.9 秒**，最新采样利用率全部 **100%**、NVML 显存 **66.47–66.81 GiB/卡**，与训练基线一致。四节点 stdout 未见 OOM、NaN/Inf、非法访存、NCCL 错误或 traceback；仅见启动阶段 FA2 未指定 dtype 的警告。非主节点 stdout 自启动后不增长，与 rank 0 输出训练指标的实现及各节点新鲜 GPU 采样相符。
- **评估与恢复点**：最近 eval 和 checkpoint 同为 **step 100**。日志记载 holdout 128219 条，teacher-forcing val loss **9.576484**、first CE **7.319771**、residual CE **7.522377**，skipped/discarded 为 0；TensorBoard 的 step 100 val 与 step 380 epoch 已交叉核对。`checkpoints/step-00000100` 的 COMPLETE、metadata、DCP `.metadata`、32 个非空 RNG 文件（rank 0–31）及 32 个非空分片齐全，world_size=32，signature 与 run 一致；恢复进度为 step=100、epoch=0、next_batch=100、samples_in_epoch=777104。本次为结构完整性检查，未执行完整加载。尚未到下次 step 2500 的自动 eval/save。
- **磁盘与增长**：共享文件系统可用约 **37.69 TiB**（使用约 62.31%），checkpoint 约 **7.86 GiB**、日志约 **1.13 MiB**、TensorBoard 约 **50.1 KiB**；目前无异常体积迹象。首次巡检无上次增长基线，后续以本次字节数比较。
- **判断与动作**：正常训练，保持原配置，不停止、不重启、不修改训练代码或数据；OOM 原配置重试 **0**，外部瞬态故障重试 **0**。已建立 `supervision/agent-state.json`；未创建 COMPLETE/STOP，未启动额外 supervisor，未执行 git commit/push。后续按既定调度检查进度、warmup 结束后的 LR、数据等待及 step 2500 eval/checkpoint。
- **详细证据**：run 下 `supervision/20261001T125427Z-evidence.json`（训练采样、四节点 GPU、错误扫描、checkpoint 文件与磁盘基线）和 `supervision/20261001T125427Z-platform.json`（实时 ACP 身份/状态）。

## 2026-10-01 13:54:27 UTC：第二次定时巡检，正常训练

- **身份与平台**：本次开始于 `2026-10-01T13:54:27.024282+00:00`，已重读当前手册、历史记录及 agent-state。最新且唯一 submission 仍为 `20261001T120352364999Z`；13:55:12 UTC 通过 `read_yaml` 与 `scripts.acp.api.request/jobs_url` 查询确认 **`pt-blb39rgw` RUNNING**，run、display_name、启动脚本均匹配，4 节点 × 8 卡。
- **进度与区间检查**：相对上次 step 380，本次到 **step 830 / epoch 0.0537093433**（metrics 写入于 13:54:40 UTC）。新增 step 390–830 共 **45 个 train 日志采样点**，无新增 val；epoch 持续推进，所有已记录数值有限，skipped/discarded 均为 0，无损坏或半行 JSON。最新 objective **4.594216**、first CE **2.509808**、residual CE **6.948028**、grad norm **0.198908**。已核对本区间 LR 符合样本进度 warmup，首次记录平台期为 step 470；当前 backbone LR **1e-4**、新模块 LR **3e-4**。
- **速度与数据等待**：45 个新增采样点的单步耗时 **7.433–8.471 秒**、均值 **7.734 秒**；data wait **0.066–0.670 秒**，比例 **0.89%–8.20%**、均值 **2.50%**；吞吐 **9598–10984 音频秒/秒**，PyTorch 峰值显存 **60.22–60.87 GiB**。最近十点（step 740–830）单步均值 **7.687 秒**、等待比例均值 **2.78%**，接近上次 7.679 秒 / 2.28%，无持续退化证据。以上均为每 10 步记录的单步采样统计，不代表全部连续训练步。
- **GPU 与日志**：13:55:12 UTC 采集的四份 GPU CSV 覆盖 **32 个不同 UUID**，最新样本年龄 **7.4–13.0 秒**，利用率全部 **100%**，NVML 显存 **66.58–66.83 GiB/卡**。四节点日志未发现 OOM、NaN/Inf、非法访存、NCCL 错误、worker 异常或 traceback；非主节点仍仅有启动时 FA2 dtype 警告，其 GPU 采样持续更新。已停止的旧 observer 所写 `monitor.jsonl` 仍停留在 12:53，未将它用作当前健康证据。
- **评估、checkpoint 与 TensorBoard**：最近仍为 **step 100** 的完整 checkpoint 与对应 val（loss **9.576484**、first CE **7.319771**、residual CE **7.522377**），尚未到 step 2500 的下一次自动 eval/save。重新确认 COMPLETE、metadata、非空 DCP `.metadata`、rank 0–31 的 32 个非空 RNG 文件及 32 个非空分片，world_size=32、signature 与 run 一致，恢复游标仍为 epoch=0 / next_batch=100 / samples_in_epoch=777104。只做结构检查，未执行恢复加载。TensorBoard step 830 train loss/epoch 和 step 100 val 均与 metrics 一致。
- **磁盘与增长**：共享文件系统可用约 **37.65 TiB**。相对上次，本 run checkpoint 与 submission 字节数不变，日志增加 **1,235,417 字节**、TensorBoard 增加 **56,970 字节**，符合正常追加；checkpoint 仍约 **7.86 GiB**。共享盘已用量增加约 **46.08 GiB**，不是本 run 上述目录增长量，不能归因于本训练；当前容量充足。
- **判断、动作与后续**：判定正常训练，继续原配置，未停止、重启或修改代码/数据；OOM 原配置重试 **0**、瞬态故障重试 **0**。追加本记录并更新 agent-state，未创建 COMPLETE/STOP、未启动另一 supervisor、未 commit/push。下轮检查进度和稳态 LR、数据等待，继续等待预定 step 2500 eval/save。
- **详细证据**：run 下 `supervision/20261001T135427Z-evidence.json` 与 `supervision/20261001T135427Z-platform.json`，包含区间采样统计、GPU 新鲜度、日志扫描、checkpoint 结构、TensorBoard 与磁盘增长。

## 2026-10-01 14:54:27 UTC：第三次定时巡检，正常训练

- **身份与平台**：开始于 `2026-10-01T14:54:27.024385+00:00`，已重读当前手册、监督记录、agent-state，并读取最新且唯一 submission `20261001T120352364999Z` 的 job.json。14:55:01 UTC 经 `read_yaml` 和 `scripts.acp.api.request/jobs_url` 查询，**`pt-blb39rgw` RUNNING**，run、display_name、启动脚本匹配，4 节点 × 8 卡。
- **进度与数值**：相对上次 step 830，14:55:02 UTC 主采样到 **step 1290**，14:55:52 UTC 复核到 **step 1300 / epoch 0.0845664090**（metrics 写入于 14:55:28 UTC），本次共检查新增 **47 个 train 日志点**，无新增 val。新增已记录数值均有限，skipped/discarded 为 0；主采样无半行 JSON，epoch 持续推进。step 1300 objective **3.784264**、first CE **1.712066**、residual CE **6.907324**、grad norm **0.289613**；backbone LR **1e-4**、新模块 LR **3e-4**，符合稳态配置。TensorBoard 同步到 step 1300，loss/epoch 与 metrics 交叉核对一致。
- **吞吐与等待**：主采样新增 46 点（step 840–1290）单步 **7.466–8.289 秒**、均值 **7.783 秒**；data wait **0.101–0.721 秒**，比例 **1.35%–8.95%**、均值 **3.01%**；吞吐 **9781–10912 音频秒/秒**，PyTorch 峰值显存 **60.30–60.79 GiB**。最近十点（1200–1290）单步均值 **7.657 秒**、等待比例均值 **2.17%**；随后 step 1300 为 **7.979 秒 / 3.95%**。未见持续退化，无需调 workers/prefetch。统计仅针对每 10 步记录的单步采样，不是全部训练步。
- **GPU 与日志**：主采样覆盖四节点 **32 个不同 GPU UUID**，CSV 最新样本年龄 **11.2–13.8 秒**，利用率 **98%–100%**，NVML 显存 **66.58–66.94 GiB/卡**。四节点 stdout 未见 OOM、NaN/Inf、非法访存、NCCL 错误或 traceback；非主节点仍仅有既有启动 FA2 dtype 警告，GPU 采样正常更新。
- **评估与 checkpoint**：最近仍为 **step 100** 的 val 与完整 checkpoint；teacher-forcing val loss **9.576484**、first CE **7.319771**、residual CE **7.522377**，TensorBoard 一致。重新确认 COMPLETE、metadata、非空 DCP `.metadata`、32 个非空 RNG 文件（rank 0–31）和 32 个非空分片，world_size=32、signature 与 run 相符；恢复游标 step=100 / epoch=0 / next_batch=100 / samples_in_epoch=777104。检查限于结构，未做恢复加载。尚未达到下次 step 2500 自动 eval/save。
- **磁盘与增长**：共享盘可用 **37.60 TiB**；本 run checkpoint 约 **7.86 GiB**，与上次字节数相同，submission 也未增长。日志增加 **1,271,218 字节**、TensorBoard 增加 **58,236 字节**，符合正常追加。共享盘已用量增长 **47.44 GiB**，不能归因于本 run，当前容量充足。
- **判断与动作**：正常训练，保持原配置；未停止、重启、改代码或数据。OOM 原配置重试 **0**、瞬态故障重试 **0**，无新故障事件。追加本记录、保存证据并更新 agent-state；未创建 COMPLETE/STOP、未启动另一 supervisor、未 commit/push。后续按既定调度检查稳态进度及 step 2500 的 eval/checkpoint。
- **详细证据**：run 下 `supervision/20261001T145427Z-evidence.json`（含 step 1300 复核）与 `supervision/20261001T145427Z-platform.json`。

## 2026-10-01 15:54:27 UTC：第四次定时巡检，正常训练

- **身份与平台**：开始于 `2026-10-01T15:54:27.024469+00:00`，已重读当前手册、监督记录及 agent-state，读取最新且唯一 submission `20261001T120352364999Z` 的 job.json。15:55:00 UTC 经 `read_yaml` 和 `scripts.acp.api.request/jobs_url` 查询，**`pt-blb39rgw` RUNNING**，run、display_name、启动脚本匹配，4 节点 × 8 卡。
- **进度与数值**：15:55:01 UTC 证据采样到 **step 1760 / epoch 0.1115208254**，metrics 写入于 15:54:33 UTC；相对上次 step 1300 增加 460 步。新增 **46 个日志点（1310–1760）** 的数值均有限，skipped/discarded 全为 0，epoch 持续推进，无半行 JSON，无新增 val。最新 objective **4.132059**、first CE **2.060745**、residual CE **6.904378**、grad norm **0.142102**；区间内 backbone LR 恒为 **1e-4**、新模块 LR 恒为 **3e-4**，符合稳态计划。最新 loss 的波动处于本区间 **3.113–4.757** 的采样范围内，无数值故障迹象。
- **吞吐与等待**：新增 46 个单步采样的 step time **7.426–8.533 秒**、均值 **7.712 秒**；data wait **0.063–0.731 秒**，比例 **0.85%–8.57%**、均值 **2.40%**；吞吐 **9530–10984 音频秒/秒**，PyTorch 峰值显存 **60.24–60.70 GiB**。最近十点（1670–1760）单步均值 **7.666 秒**、等待比例均值 **1.84%**，无持续吞吐退化。上述仅为每 10 步记录的该单步统计，不能代表全部连续训练步。
- **GPU 与日志**：四节点 CSV 共 **32 个不同 GPU UUID**，最新样本距主采样 **8.9–10.9 秒**；利用率 **98%–100%**，NVML 显存 **66.58–66.94 GiB/卡**。四节点 stdout 未发现 OOM、NaN/Inf、非法访存、NCCL 错误或 traceback；非主节点仍仅有既有启动 FA2 dtype 警告，各节点 GPU 采样持续更新。
- **评估与恢复点**：最近仍是 **step 100** 的 val 与 checkpoint，尚未到 step 2500 自动 eval/save。teacher-forcing val loss **9.576484**、first CE **7.319771**、residual CE **7.522377**；TensorBoard 的该 val 及 step 1760 train loss/epoch 与 metrics 一致。checkpoint 的 COMPLETE、metadata、非空 DCP `.metadata`、32 个非空 RNG 文件（rank 0–31）及 32 个非空分片齐全，world_size=32、signature 与 run 一致；恢复游标 step=100 / epoch=0 / next_batch=100 / samples_in_epoch=777104。仅验证结构完整性，未执行恢复加载。
- **磁盘与增长**：共享盘可用 **37.55 TiB**；本 run checkpoint 仍约 **7.86 GiB**，checkpoint 与 submission 字节数不变。日志比上次增加 **1,274,682 字节**、TensorBoard 增加 **59,502 字节**，符合正常追加；共享盘已用量增加 **47.56 GiB**，不能归因于本 run，当前容量充足。
- **判断与动作**：正常训练，保持原配置；未停止、重启或修改训练代码/数据。OOM 原配置重试 **0**、瞬态故障重试 **0**，无新故障事件。追加本记录、保存证据并更新 agent-state；未创建 COMPLETE/STOP、未启动另一 supervisor、未 commit/push。下轮继续检查稳态进度、数据等待和预定 step 2500 的 eval/checkpoint。
- **详细证据**：run 下 `supervision/20261001T155427Z-evidence.json` 与 `supervision/20261001T155427Z-platform.json`。

## 2026-10-01 16:54:27 UTC：第五次定时巡检，正常训练

- **身份与平台**：开始于 `2026-10-01T16:54:27.024563+00:00`，已重读手册、监督记录和 agent-state，并读取最新且唯一 submission `20261001T120352364999Z` 的 job.json。16:55:00 UTC 经 `read_yaml` 与 `scripts.acp.api.request/jobs_url` 确认 **`pt-blb39rgw` RUNNING**，run、display_name、启动脚本均匹配，4 节点 × 8 卡。
- **进度与数值**：主采样为 step 2220，16:55:42 UTC 复核已到 **step 2230 / epoch 0.1405696824**（metrics 写入于 16:55:04 UTC），相对上次 step 1760 增加 470 步。新增 **47 个 train 日志点（1770–2230）** 数值均有限，skipped/discarded 全为 0，epoch 持续推进，无半行 JSON，无新增 val。最新 objective **3.909320**、first CE **1.891726**、residual CE **6.725315**、grad norm **0.262006**；区间内 backbone LR **1e-4**、新模块 LR **3e-4**，符合稳态计划。TensorBoard step 2230 train loss/epoch 与 metrics 一致。
- **速度与数据等待**：新增采样单步 **7.407–8.188 秒**、均值 **7.742 秒**；data wait **0.069–0.740 秒**，比例 **0.93%–9.15%**、均值 **2.72%**；吞吐 **9949–10990 音频秒/秒**，PyTorch 峰值显存 **60.30–60.67 GiB**。主采样最近十点（2130–2220）单步均值 **7.856 秒**、等待比例均值 **3.58%**，复核 step 2230 为 **7.709 秒 / 2.09%**，无持续退化证据。上述是每 10 步记录的单步采样统计，不代表连续全部训练步。
- **GPU 与日志**：16:55:01 UTC 主采样覆盖四节点 **32 个不同 GPU UUID**，CSV 最新样本年龄 **5.4–8.9 秒**，利用率全部 **100%**，NVML 显存 **66.60–66.94 GiB/卡**。四节点 stdout 未发现 OOM、NaN/Inf、非法访存、NCCL 错误或 traceback；非主节点仍仅有启动时 FA2 dtype 警告，各节点 GPU 采样持续更新。
- **评估与恢复点**：最近仍为 **step 100** 的 val 与 checkpoint，teacher-forcing val loss **9.576484**、first CE **7.319771**、residual CE **7.522377**，TensorBoard 一致。COMPLETE、metadata、非空 DCP `.metadata`、rank 0–31 的 32 个非空 RNG 文件及 32 个非空分片齐全，world_size=32、signature 与 run 一致；恢复游标 step=100 / epoch=0 / next_batch=100 / samples_in_epoch=777104。仅检查结构，未执行恢复加载。当前距离下一次 step 2500 自动 eval/save 尚有 **270 步**；下轮重点核对同 step 的 val 和完整 checkpoint。
- **磁盘与增长**：共享盘可用 **37.51 TiB**；checkpoint 仍约 **7.86 GiB**，checkpoint/submission 字节数未变。日志比上次增加 **1,272,993 字节**、TensorBoard 增加 **58,236 字节**，符合正常追加；共享盘已用量增加 **47.55 GiB**，不能归因于本 run，当前容量充足。
- **判断与动作**：正常训练，维持原配置；未停止、重启或修改训练代码/数据。OOM 原配置重试 **0**、瞬态故障重试 **0**，无新故障事件。已追加本记录、保存详细证据并更新 agent-state；未创建 COMPLETE/STOP、未启动另一 supervisor、未 commit/push。本次巡检结束，不等待下一小时。
- **详细证据**：run 下 `supervision/20261001T165427Z-evidence.json`（含 step 2230 复核）与 `supervision/20261001T165427Z-platform.json`。

## 2026-10-01 17:54:27 UTC：第六次定时巡检，step 2500 评估与保存完成

- **身份与平台**：开始于 `2026-10-01T17:54:27.024663+00:00`，已重读当前手册、监督记录、agent-state 及最新且唯一 submission `20261001T120352364999Z` 的 job.json。17:54:59 UTC 经 `read_yaml` 与 `scripts.acp.api.request/jobs_url` 确认 **`pt-blb39rgw` RUNNING**，run、display_name、启动脚本匹配，4 节点 × 8 卡。
- **新评估**：step **2500** 的全量 **128219 条 holdout** 评估已完成，skipped/discarded 均为 0；teacher-forcing val loss **3.469182**（step 100 为 9.576484）、first CE **1.482672**（此前 7.319771）、residual CE **6.621699**（此前 7.522377），全部有限。TensorBoard 三项 val 与 metrics 一致。这是 teacher-forcing 指标，不是音频生成或 WER/音质评估。
- **新恢复点**：`checkpoints/step-00002500` 的 COMPLETE 写入于 **17:31:27 UTC**，latest 已指向该目录；metadata、DCP `.metadata`、rank 0–31 的 **32 个非空 RNG 文件和 32 个非空分片**齐全，world_size=32、signature 与 run 一致。恢复游标 **step=2500 / epoch=0 / next_batch=2500 / samples_in_epoch=20286810**，scheduler.last_epoch=2500，LR 为 1e-4 / 3e-4。DCP 索引可读取，含 **478 个 model 键、1217 个 optimizer 键**，引用 32 个分片，所有索引字节范围均在对应文件大小内；本次没有完整加载权重或执行恢复。旧 step 100 checkpoint 仍保留。
- **训练进度**：主采样 step 2680，17:56:21 UTC 复核到 **step 2690 / epoch 0.1706024966**（metrics 写入于 17:56:00 UTC），说明评估/保存后正常继续。相对上次 step 2230 新增 **46 个 train 日志点（2240–2690）及 1 条 val**；数值均有限，skipped/discarded 全为 0，epoch 持续推进，无半行 JSON。最新 objective **3.543643**、first CE **1.697861**、residual CE **6.152607**、grad norm **0.382559**；backbone LR **1e-4**、新模块 LR **3e-4**，TensorBoard step 2690 train loss/epoch 与 metrics 一致。
- **吞吐与资源**：新增训练采样单步 **7.445–8.699 秒**、均值 **7.763 秒**；data wait **0.064–0.992 秒**，比例 **0.86%–11.40%**、均值 **2.98%**；吞吐 **9328–10962 音频秒/秒**，PyTorch 峰值显存 **60.32–60.87 GiB**。评估后的最近十点（2590–2680）单步均值 **7.641 秒**、等待比例均值 **1.84%**，无持续退化。以上是每 10 步记录的单步采样，不能代表连续全部训练步或 eval/save 耗时。
- **GPU 与日志**：主采样四节点共 **32 个不同 UUID**，样本年龄 **2.0–14.9 秒**，NVML 显存 **66.60–66.94 GiB/卡**；rank-0 节点瞬时利用率 **63%–92%**，其余节点 **100%**。后续复核 rank-0 已恢复 **100%**，rank-16 为 **48%–92%**、rank-24 为 **100%**、rank-8 为 **97%–100%**，样本年龄 **3.9–16.5 秒**。结合训练持续推进和正常吞吐，未见持续低利用率证据，不据单次波动干预。四节点 stdout 无 OOM、NaN/Inf、非法访存、NCCL 错误或 traceback，仍只有既有启动 dtype 警告。
- **磁盘与增长**：共享盘可用 **37.45 TiB**；checkpoint 总计约 **15.72 GiB**，比上次增加 **8,438,656,593 字节**，对应新增 step 2500 checkpoint。日志增加 **1,274,316 字节**、TensorBoard 增加 **58,388 字节**，submission 未增长，均可解释。共享盘总已用量增加 **55.44 GiB**，不全部归因于本 run。
- **判断与动作**：正常训练，自动 eval/save 按计划完成，保持原配置；未停止、重启或修改代码/数据。OOM 原配置重试 **0**、瞬态故障重试 **0**。已将最近 eval/完整 checkpoint 更新为 **2500**，登记该正常完成事件、追加本记录并保存证据。未创建 COMPLETE/STOP、未启动另一 supervisor、未 commit/push。后续继续每小时检查进度，下一次预定自动 eval/save 为 step **5000**；本次巡检结束。
- **详细证据**：run 下 `supervision/20261001T175427Z-evidence.json`（含 DCP 索引检查和 GPU 二次采样）与 `supervision/20261001T175427Z-platform.json`。

## 2026-10-01 18:54:27 UTC：第七次定时巡检，正常训练

- **身份与平台**：开始于 `2026-10-01T18:54:27.024746+00:00`，已重读当前手册、监督记录、agent-state 和最新且唯一 submission `20261001T120352364999Z` 的 job.json。18:55:01 UTC 经 `read_yaml` 与 `scripts.acp.api.request/jobs_url` 确认 **`pt-blb39rgw` RUNNING**，run、display_name、启动脚本匹配，4 节点 × 8 卡。
- **进度与数值**：主采样 step 3140，18:55:47 UTC 复核到 **step 3150 / epoch 0.1993046418**（metrics 写入于 18:55:12 UTC），相对上次 step 2690 增加 460 步。新增 **46 个 train 日志点（2700–3150）** 数值均有限，skipped/discarded 全为 0，epoch 持续推进，无半行 JSON，无新增 val。最新 objective **2.940108**、first CE **1.001181**、residual CE **6.463088**、grad norm **0.310387**；区间 backbone LR **1e-4**、新模块 LR **3e-4**，符合稳态计划。TensorBoard step 3150 train loss/epoch 已与 metrics 核对。
- **吞吐与数据等待**：新增单步采样耗时 **7.433–8.214 秒**、均值 **7.718 秒**；data wait **0.064–0.777 秒**，比例 **0.86%–9.50%**、均值 **2.92%**；吞吐 **9920–10949 音频秒/秒**，PyTorch 峰值显存 **60.30–60.87 GiB**。主采样最近十点（3050–3140）单步均值 **7.669 秒**、等待均值 **1.91%**，复核 step 3150 为 **7.999 秒 / 7.82%**，未见连续多个日志点退化，不据单点等待增加调参。统计仅为每 10 步记录的单步采样，不代表全部连续训练步。
- **GPU 与日志**：四节点 CSV 共 **32 个不同 GPU UUID**，18:55:01 UTC 样本年龄 **4.8–11.6 秒**；利用率 **87%–100%**（rank-24 节点有瞬时波动，其余均 100%），NVML 显存 **66.65–66.94 GiB/卡**。四节点 stdout 未见 OOM、NaN/Inf、非法访存、NCCL 错误或 traceback；非主节点仍仅有既有启动 FA2 dtype 警告，GPU 采样持续更新。
- **评估与恢复点**：最近仍为 **step 2500** 的 val 与完整 checkpoint，teacher-forcing val loss **3.469182**、first CE **1.482672**、residual CE **6.621699**，TensorBoard 一致。重新确认 COMPLETE、metadata、非空 DCP `.metadata`、rank 0–31 的 32 个非空 RNG 文件及 32 个非空分片；world_size=32、signature 与 run 一致、latest 指向 step-00002500。恢复游标 step=2500 / epoch=0 / next_batch=2500 / samples_in_epoch=20286810。本次为结构检查，未执行完整恢复加载。下一次自动 eval/save 仍为 step **5000**。
- **磁盘与增长**：共享盘可用 **37.27 TiB**；本 run checkpoint 约 **15.72 GiB**，checkpoint 与 submission 字节数未变。日志新增 **1,273,077 字节**、TensorBoard 新增 **58,236 字节**，正常追加。共享盘总已用量增加 **184.73 GiB**，高于此前每小时约 47–55 GiB，但不是本 run 上述目录增长量，不能归因于本训练；当前容量充足，下轮继续比较。
- **判断与动作**：正常训练，保持原配置，未停止、重启或修改代码/数据；OOM 原配置重试 **0**、瞬态故障重试 **0**，无新故障事件。追加本记录、保存详细证据并更新 agent-state，保留最近 eval/checkpoint=2500 及历史事件；未创建 COMPLETE/STOP、未启动另一 supervisor、未 commit/push。本轮结束。
- **详细证据**：run 下 `supervision/20261001T185427Z-evidence.json`（含 step 3150 复核）与 `supervision/20261001T185427Z-platform.json`。

## 2026-10-01 19:54:27 UTC：第八次定时巡检，正常训练

- **身份与平台**：开始于 `2026-10-01T19:54:27.024826+00:00`，已重读当前手册、监督记录、agent-state，以及最新且唯一 submission `20261001T120352364999Z` 的 job.json。19:55:04 UTC 经 `read_yaml` 与 `scripts.acp.api.request/jobs_url` 确认 **`pt-blb39rgw` RUNNING**，run、display_name、启动脚本匹配，4 节点 × 8 卡。
- **进度与数值**：主采样 step 3610，19:56:01 UTC 复核到 **step 3620 / epoch 0.2274011127**（metrics 写入于 19:55:33 UTC），相对上次 step 3150 增加 470 步。新增 **47 个 train 日志点（3160–3620）** 数值均有限，skipped/discarded 全为 0，epoch 持续推进，无半行 JSON，无新增 val。最新 objective **3.437354**、first CE **1.768007**、residual CE **5.564490**、grad norm **0.484673**；backbone LR **1e-4**、新模块 LR **3e-4**，符合稳态计划。TensorBoard step 3620 train loss/epoch 与 metrics 一致。
- **速度与等待**：新增单步采样耗时 **7.421–8.395 秒**、均值 **7.728 秒**；data wait **0.065–0.748 秒**，比例 **0.87%–8.91%**、均值 **3.12%**；吞吐 **9673–10985 音频秒/秒**，PyTorch 峰值显存 **60.24–60.72 GiB**。最近十点（3520–3610）单步均值 **7.691 秒**、等待均值 **2.73%**，复核 step 3620 为 **7.925 秒 / 1.98%**，未见持续退化。统计仅覆盖每 10 步记录的单步采样，不代表连续全部训练步。
- **GPU 与日志**：主采样覆盖 **32 个不同 UUID**，样本年龄 **3.2–9.8 秒**，NVML 显存 **66.68–66.94 GiB/卡**；rank-8 节点瞬时利用率 **0%–67%**，其余节点均 **100%**。约 19:56:01 UTC 二次采样中 **32 卡全部恢复 100%**，样本年龄 **2.8–9.3 秒**；最近 120 秒各节点设备采样的平均利用率约 **93.92%–100%**。结合训练继续推进和正常耗时，判定为短暂波动，不作干预。四节点 stdout 无 OOM、NaN/Inf、非法访存、NCCL 错误或 traceback，仍仅有既有启动 FA2 dtype 警告。
- **评估与恢复点**：最近仍为 **step 2500** 的 val 与完整 checkpoint；teacher-forcing val loss **3.469182**、first CE **1.482672**、residual CE **6.621699**，TensorBoard 一致。COMPLETE、metadata、非空 DCP `.metadata`、32 个非空 RNG 文件（rank 0–31）和 32 个非空分片齐全，world_size=32、signature 与 run 一致、latest 指向 step-00002500。恢复游标 step=2500 / epoch=0 / next_batch=2500 / samples_in_epoch=20286810；本次为结构检查，未执行恢复加载。下一次自动 eval/save 为 step **5000**。
- **磁盘与增长**：共享盘可用 **37.09 TiB**；本 run checkpoint 约 **15.72 GiB**，checkpoint/submission 字节数未变，日志增加 **1,277,236 字节**、TensorBoard 增加 **59,502 字节**，正常追加。共享盘总已用量增加 **189.50 GiB**，接近上一小时 184.73 GiB，不能归因于本 run；容量仍充足，后续继续比较。
- **判断与动作**：正常训练，保持原配置；未停止、重启或修改代码/数据。OOM 原配置重试 **0**、瞬态故障重试 **0**，无新故障事件。已追加记录、保存证据并更新 agent-state，保留最近 eval/checkpoint=2500 和历史事件；未创建 COMPLETE/STOP、未启动另一 supervisor、未 commit/push。本轮结束。
- **详细证据**：run 下 `supervision/20261001T195427Z-evidence.json`（含 step 3620 与 GPU 二次采样）及 `supervision/20261001T195427Z-platform.json`。

## 2026-10-01 20:54:27 UTC：第九次定时巡检，正常训练

- **身份与平台**：开始于 `2026-10-01T20:54:27.024914+00:00`，已重读当前手册、监督记录、agent-state 及最新且唯一 submission `20261001T120352364999Z` 的 job.json。20:55:01 UTC 经 `read_yaml` 与 `scripts.acp.api.request/jobs_url` 确认 **`pt-blb39rgw` RUNNING**，run、display_name、启动脚本匹配，4 节点 × 8 卡。
- **进度与数值**：本次从最初读取的 step 4070 推进至 **step 4080 / epoch 0.2560848648**，20:55:48 UTC 复核 metrics 最近写入为 20:54:46 UTC。相对上次 step 3620 增加 460 步，新增 **46 个 train 日志点（3630–4080）**，数值均有限，skipped/discarded 全为 0，epoch 持续推进，无半行 JSON，无新增 val。最新 objective **3.573122**、first CE **1.614691**、residual CE **6.528103**、grad norm **0.205173**；backbone LR **1e-4**、新模块 LR **3e-4**，符合稳态计划。TensorBoard step 4080 train loss/epoch 已与 metrics 核对。
- **吞吐与等待**：新增单步采样耗时 **7.434–8.403 秒**、均值 **7.757 秒**；data wait **0.079–0.823 秒**，比例 **1.03%–9.83%**、均值 **3.18%**；吞吐 **9662–10957 音频秒/秒**，PyTorch 峰值显存 **60.30–60.85 GiB**。最近十点（3990–4080）单步均值 **7.716 秒**、等待均值 **2.71%**，与此前基线相近，无持续退化。统计仅针对每 10 步记录的单步采样，不代表全部连续训练步。
- **GPU 与日志**：20:55:02 UTC 四节点采样覆盖 **32 个不同 GPU UUID**，最新样本年龄 **3.7–15.4 秒**，利用率全部 **100%**，NVML 显存 **66.68–66.94 GiB/卡**。四节点 stdout 无 OOM、NaN/Inf、非法访存、NCCL 错误或 traceback；非主节点仍仅有既有启动 FA2 dtype 警告，其 GPU 采样持续更新。
- **评估与恢复点**：最近仍为 **step 2500** 的 val 与完整 checkpoint，teacher-forcing val loss **3.469182**、first CE **1.482672**、residual CE **6.621699**，TensorBoard 一致。重新确认 COMPLETE、metadata、非空 DCP `.metadata`、rank 0–31 的 32 个非空 RNG 文件及 32 个非空分片；world_size=32、signature 与 run 一致、latest 指向 step-00002500。恢复游标 step=2500 / epoch=0 / next_batch=2500 / samples_in_epoch=20286810；本次为结构检查，未执行恢复加载。下一次自动 eval/save 为 step **5000**。
- **磁盘与增长**：共享盘可用 **36.93 TiB**；本 run checkpoint 总计约 **15.72 GiB**，checkpoint/submission 字节数未变。日志增加 **1,272,842 字节**、TensorBoard 增加 **59,502 字节**，正常追加。共享盘总已用量增加 **164.47 GiB**，低于上次 189.50 GiB，不能归因于本 run；容量仍充足，后续继续比较。
- **判断与动作**：正常训练，保持原配置，未停止、重启或修改代码/数据；OOM 原配置重试 **0**、瞬态故障重试 **0**，无新故障事件。追加本记录、保存证据并更新 agent-state，保留最近 eval/checkpoint=2500 和历史事件；未创建 COMPLETE/STOP、未启动另一 supervisor、未 commit/push。本轮结束。
- **详细证据**：run 下 `supervision/20261001T205427Z-evidence.json` 与 `supervision/20261001T205427Z-platform.json`。

## 2026-10-01 21:54:27 UTC：第十次定时巡检，正常训练

- **身份与平台**：开始于 `2026-10-01T21:54:27.024993+00:00`，已重读当前手册、监督记录、agent-state 及最新且唯一 submission `20261001T120352364999Z` 的 job.json。21:54:59 UTC 经 `read_yaml` 与 `scripts.acp.api.request/jobs_url` 确认 **`pt-blb39rgw` RUNNING**，run、display_name、启动脚本匹配，4 节点 × 8 卡。
- **进度与数值**：主采样 step 4540，21:55:47 UTC 复核到 **step 4550 / epoch 0.2858807710**（metrics 写入于 21:55:09 UTC）；相对上次 step 4080 增加 470 步。新增 **47 个 train 日志点（4090–4550）** 数值均有限，skipped/discarded 全为 0，epoch 持续推进，无半行 JSON，无新增 val。最新 objective **2.682348**、first CE **0.894774**、residual CE **5.958582**、grad norm **0.416673**；backbone LR **1e-4**、新模块 LR **3e-4**，符合稳态计划。TensorBoard step 4550 train loss/epoch 与 metrics 一致。
- **吞吐与等待**：新增单步采样耗时 **7.436–8.239 秒**、均值 **7.750 秒**；data wait **0.092–0.757 秒**，比例 **1.19%–9.37%**、均值 **3.11%**；吞吐 **9818–10957 音频秒/秒**，PyTorch 峰值显存 **60.31–60.86 GiB**。主采样最近十点（4450–4540）单步均值 **7.762 秒**、等待均值 **2.91%**；随后 step 4550 为 **7.537 秒 / 2.59%**，未见持续退化。统计仅为每 10 步记录的单步采样，不代表连续全部训练步。
- **GPU 与日志**：21:55:00 UTC 四节点采样覆盖 **32 个不同 UUID**，样本年龄 **7.8–10.6 秒**，利用率全部 **100%**，NVML 显存 **66.68–66.94 GiB/卡**。四节点 stdout 未见 OOM、NaN/Inf、非法访存、NCCL 错误或 traceback；非主节点仍只有既有启动 FA2 dtype 警告，其 GPU 采样持续更新。
- **评估与恢复点**：最近仍为 **step 2500** 的 teacher-forcing val 与完整 checkpoint，val loss **3.469182**、first CE **1.482672**、residual CE **6.621699**，TensorBoard 一致。重新确认 COMPLETE、metadata、非空 DCP `.metadata`、rank 0–31 的 32 个非空 RNG 文件及 32 个非空分片；world_size=32、signature 与 run 一致、latest 指向 step-00002500。恢复游标 step=2500 / epoch=0 / next_batch=2500 / samples_in_epoch=20286810。本次为结构检查，未执行恢复加载。距离下一次 step **5000** 自动 eval/save 尚有 **450 步**，下轮重点核对阶段及对应 val/checkpoint。
- **磁盘与增长**：共享盘可用 **36.76 TiB**；本 run checkpoint 约 **15.72 GiB**，checkpoint/submission 字节数不变。日志增加 **1,273,000 字节**、TensorBoard 增加 **58,236 字节**，正常追加。共享盘总已用量增加 **174.03 GiB**，不能归因于本 run；容量仍充足，后续继续比较。
- **判断与动作**：正常训练，保持原配置；未停止、重启或修改代码/数据。OOM 原配置重试 **0**、瞬态故障重试 **0**，无新故障事件。已追加本记录、保存证据并更新 agent-state，保留最近 eval/checkpoint=2500 和历史事件；未创建 COMPLETE/STOP、未启动另一 supervisor、未 commit/push。本轮结束。
- **详细证据**：run 下 `supervision/20261001T215427Z-evidence.json`（含 step 4550 复核）与 `supervision/20261001T215427Z-platform.json`。

## 2026-10-01 22:54:27 UTC：第十一次定时巡检，step 5000 评估与保存完成

- **身份与平台**：开始于 `2026-10-01T22:54:27.025079+00:00`，已重读当前手册、监督记录、agent-state 及最新且唯一 submission `20261001T120352364999Z` 的 job.json。22:55:00 UTC 经 `read_yaml` 与 `scripts.acp.api.request/jobs_url` 确认 **`pt-blb39rgw` RUNNING**，run、display_name、启动脚本匹配，4 节点 × 8 卡。
- **新评估**：step **5000** 的 **128219 条 holdout** 评估已完成，skipped/discarded 为 0，所有指标有限。teacher-forcing val loss **3.166461**（step 2500 为 3.469182）、first CE **1.302180**（此前 1.482672）、residual CE **6.214271**（此前 6.621699）；TensorBoard 与 metrics 一致。这些是 teacher-forcing 指标，不是音频生成质量评分。
- **新恢复点与保留策略**：`checkpoints/step-00005000` 的 COMPLETE 写入于 **22:54:21 UTC**，latest 指向该目录；metadata、非空 DCP `.metadata`、rank 0–31 的 **32 个非空 RNG 文件与 32 个非空分片**齐全，world_size=32、signature 与 run 一致。恢复游标 **step=5000 / epoch=0 / next_batch=5000 / samples_in_epoch=40295457**，scheduler.last_epoch=5000，LR 为 1e-4 / 3e-4。DCP 索引含 **478 个 model 键、1217 个 optimizer 键**，引用 32 个分片，所有索引字节范围在对应文件大小内；未执行完整加载或恢复。原提交 experiment.yaml 与 run config.json 均设置 `keep_checkpoints: 2`，训练日志明确记录自动清理 `step-00000100`，当前保留 **2500、5000**；该清理由训练既有逻辑执行，本 agent 未删除文件。
- **训练继续推进**：主采样停留于已完成 eval/save 的 step 5000，22:56:15 UTC 复核已到 **step 5010 / epoch 0.3152734045**（metrics 写入于 22:55:39 UTC），确认正常继续训练。相对上次 step 4550 新增 **46 个 train 日志点（4560–5010）与 1 条 val**；数值均有限，skipped/discarded 全为 0，epoch 持续推进，无半行 JSON。最新 objective **3.468637**、first CE **1.568546**、residual CE **6.333639**、grad norm **0.215894**；backbone LR **1e-4**、新模块 LR **3e-4**，TensorBoard step 5010 train loss/epoch 与 metrics 一致。
- **吞吐与资源**：新增单步采样耗时 **7.426–8.274 秒**、均值 **7.745 秒**；data wait **0.067–0.634 秒**，比例 **0.88%–7.91%**、均值 **2.89%**；吞吐 **9815–10987 音频秒/秒**，PyTorch 峰值显存 **60.23–60.87 GiB**。eval/save 后 step 5010 为 **7.976 秒 / 4.55%**，无持续退化证据。统计是每 10 步记录的单步采样，不能代表连续全部训练步或 eval/save 耗时。
- **GPU 与日志**：22:55:00 UTC 四节点 CSV 覆盖 **32 个不同 UUID**，最新样本年龄 **4.4–9.8 秒**，利用率 **90%–100%**，NVML 显存 **66.68–66.94 GiB/卡**。四节点 stdout 无 OOM、NaN/Inf、非法访存、NCCL 错误或 traceback，仍仅有既有启动 FA2 dtype 警告；非主节点 GPU 采样持续更新。
- **磁盘与增长**：共享盘可用 **36.59 TiB**；checkpoint 总计仍约 **15.72 GiB**（比上次少 26 字节），符合新增 5000 同时自动淘汰 100 的两份保留策略。日志增加 **1,274,294 字节**、TensorBoard 增加 **58,388 字节**，submission 未变。共享盘总已用量增加 **169.86 GiB**，不全部归因于本 run，容量仍充足。
- **判断与动作**：正常训练，自动 eval/save 完成，保持原配置；未停止、重启或修改代码/数据。OOM 原配置重试 **0**、瞬态故障重试 **0**。登记 eval/checkpoint=**5000** 和训练自动保留事件，追加本记录、保存证据并更新 agent-state；未创建 COMPLETE/STOP、未启动另一 supervisor、未 commit/push。后续继续按既定调度巡检，下一次自动 eval/save 为 step **7500**；本轮结束。
- **详细证据**：run 下 `supervision/20261001T225427Z-evidence.json`（含 DCP 索引、自动保留日志及 step 5010 复核）与 `supervision/20261001T225427Z-platform.json`。

## 2026-10-01 23:54:27 UTC：第十二次定时巡检，正常训练；日志瞬时读取异常已复核

- **身份与平台**：开始于 `2026-10-01T23:54:27.025157+00:00`，已重读手册、监督记录、agent-state 及最新且唯一 submission `20261001T120352364999Z` 的 job.json。23:55:05 UTC 经 `read_yaml` 与 `scripts.acp.api.request/jobs_url` 确认 **`pt-blb39rgw` RUNNING**，run、display_name、启动脚本匹配，4 节点 × 8 卡。
- **进度与数值**：本轮从最初读取的 step 5460 推进至 **step 5470 / epoch 0.3459713424**，metrics 最近写入为 **23:55:03 UTC**；相对上次 step 5010 增加 460 步。新增 **46 个 train 日志点（5020–5470）**，数值均有限，skipped/discarded 全为 0，epoch 持续推进，无半行 JSON，无新增 val。最新 objective **2.948915**、first CE **1.118914**、residual CE **6.100005**、grad norm **0.291022**；backbone LR **1e-4**、新模块 LR **3e-4**，符合稳态计划。TensorBoard step 5470 train loss/epoch 与 metrics 一致。
- **吞吐与等待**：新增单步采样耗时 **7.477–8.697 秒**、均值 **7.794 秒**；data wait **0.081–1.049 秒**，比例 **1.08%–12.06%**、均值 **3.09%**；吞吐 **9331–10891 音频秒/秒**，PyTorch 峰值显存 **60.19–60.76 GiB**。最近十点（5380–5470）单步均值 **7.762 秒**、等待比例均值 **2.76%**，无持续退化。统计仅覆盖每 10 步记录的单步采样，不代表连续全部训练步。
- **GPU 与日志复核**：23:55:06 UTC 四节点采样覆盖 **32 个不同 UUID**，样本年龄 **6.6–11.2 秒**，利用率全部 **100%**，NVML 显存 **66.70–66.96 GiB/卡**。四节点 stdout 未匹配到 OOM、NaN/Inf、非法访存、NCCL 错误或 traceback，非主节点仍只有既有启动 FA2 dtype 警告。首次读取主节点 stdout 时尾部出现一段空字节，现场已保存；**23:55:21 UTC** 重读无空字节且 step 5470 记录完整，**23:56:13 UTC** 再核对仍无空字节、无错误匹配，区间 5020–5470 的 train 日志点均齐全。并发追加或共享文件系统可见性仅为可能解释，原因未确认；metrics/TensorBoard 正常，不据这次已消失的读取异常重启训练。下轮关注是否复现。
- **评估与恢复点**：最近仍为 **step 5000** 的全量 **128219 条 holdout** teacher-forcing val，loss **3.166461**、first CE **1.302180**、residual CE **6.214271**，TensorBoard 与 metrics 一致。latest 指向 `checkpoints/step-00005000`，重新确认 COMPLETE、metadata、非空 DCP `.metadata`、rank 0–31 的 **32 个非空 RNG 文件与 32 个非空分片**；world_size=32、signature 与 run 一致。恢复游标 **step=5000 / epoch=0 / next_batch=5000 / samples_in_epoch=40295457**。当前保留 **2500、5000** 两份 checkpoint，本次仅检查结构，未执行完整加载或恢复。下一次预定自动 eval/save 为 step **7500**。
- **磁盘与增长**：共享盘可用 **36.47 TiB**；本 run checkpoint 约 **15.72 GiB**，checkpoint/submission 字节数不变。日志增加 **1,277,277 字节**、TensorBoard 增加 **59,502 字节**，符合正常追加；共享盘总已用量增加 **120.45 GiB**，不能归因于本 run，容量仍充足。
- **判断与动作**：正常训练，保持原配置；未停止、重启或修改训练代码/数据/日志。OOM 原配置重试 **0**、瞬态故障重试 **0**。登记已消失的 stdout 读取异常，保留历史事件与重试计数，追加本记录、保存证据并更新 agent-state，最近 eval/完整 checkpoint 仍为 **5000**。未创建 COMPLETE/STOP、未启动另一 supervisor、未 commit/push。本轮结束，不等待下一小时。
- **详细证据**：run 下 `supervision/20261001T235427Z-evidence.json`、`supervision/20261001T235427Z-platform.json` 与 `supervision/20261001T235427Z-log-recheck.json`。

## 2026-10-02 00:54:27 UTC：第十三次定时巡检，正常训练

- **身份与平台**：开始于 `2026-10-02T00:54:27.025250+00:00`，已重读手册、监督记录、agent-state 及最新 submission `20261001T120352364999Z` 的 job.json。00:55:12 UTC 经 `read_yaml` 与 `scripts.acp.api.request/jobs_url` 确认 **`pt-blb39rgw` RUNNING**，run、display_name、启动脚本匹配，4 节点 × 8 卡。
- **进度与数值**：主采样 step 5930，00:56:23 UTC 复核推进至 **step 5940 / epoch 0.3749607811**（metrics 写入于 00:55:26 UTC）。相对上次 step 5470 增加 470 步，新增 **47 个 train 日志点（5480–5940）**，数值均有限，skipped/discarded 全为 0，epoch 持续推进，无半行 JSON，无新增 val。最新 objective **3.401452**、first CE **1.535009**、residual CE **6.221477**、grad norm **0.190916**；backbone LR **1e-4**、新模块 LR **3e-4**，符合稳态计划。TensorBoard step 5940 train loss/epoch 与 metrics 一致。
- **吞吐与等待**：新增单步采样耗时 **7.450–8.234 秒**、均值 **7.722 秒**；data wait **0.072–0.684 秒**，比例 **0.95%–8.31%**、均值 **2.69%**；吞吐 **9908–10940 音频秒/秒**，PyTorch 峰值显存 **60.31–60.82 GiB**。主采样最近十点（5840–5930）单步均值 **7.739 秒**、等待均值 **3.13%**，后续 step 5940 为 **7.515 秒 / 1.64%**，无持续退化。统计仅覆盖每 10 步记录的单步采样，不代表连续全部训练步。
- **GPU 与日志**：00:55:12 UTC 四节点采样覆盖 **32 个不同 UUID**，样本年龄 **5.6–12.4 秒**，利用率全部 **100%**，NVML 显存 **66.70–67.04 GiB/卡**。四节点 stdout 未匹配到 OOM、NaN/Inf、非法访存、NCCL 错误或 traceback；非主节点仍只有既有启动 FA2 dtype 警告，GPU 采样持续更新。四节点日志空字节计数均为 **0**，主节点区间 5480–5940 的 train 日志点齐全；上轮瞬时空字节读取异常本轮未复现。
- **评估与恢复点**：最近仍为 **step 5000** 的 teacher-forcing val 与完整 checkpoint，val loss **3.166461**、first CE **1.302180**、residual CE **6.214271**，TensorBoard 一致。latest 指向 `step-00005000`，COMPLETE、metadata、非空 DCP `.metadata`、rank 0–31 的 **32 个非空 RNG 文件与 32 个非空分片**齐全；world_size=32、signature 与 run 一致。恢复游标 **step=5000 / epoch=0 / next_batch=5000 / samples_in_epoch=40295457**。当前保留 **2500、5000** 两份，本次为结构检查，未执行完整加载或恢复。下一次预定自动 eval/save 为 step **7500**。
- **磁盘与增长**：共享盘可用 **36.43 TiB**；本 run checkpoint 约 **15.72 GiB**，checkpoint/submission 字节数不变。日志增加 **1,276,547 字节**、TensorBoard 增加 **58,236 字节**，正常追加；共享盘总已用量增加 **47.36 GiB**，不能归因于本 run，容量仍充足。
- **判断与动作**：正常训练，保持原配置；未停止、重启或修改训练代码/数据。OOM 原配置重试 **0**、瞬态故障重试 **0**，无新增故障事件。追加记录、保存证据并更新 agent-state，保留历史事件和最近 eval/checkpoint=5000；未创建 COMPLETE/STOP、未启动另一 supervisor、未 commit/push。本轮结束，不等待下一小时。
- **详细证据**：run 下 `supervision/20261002T005427Z-evidence.json`（含 step 5940 复核）与 `supervision/20261002T005427Z-platform.json`。

## 2026-10-02 01:54:27 UTC：第十四次定时巡检，正常训练；瞬时日志读取异常再次复现并恢复

- **身份与平台**：开始于 `2026-10-02T01:54:27.025327+00:00`，已重读手册、监督记录、agent-state 及最新 submission `20261001T120352364999Z` 的 job.json。01:54:59 UTC 经 `read_yaml` 与 `scripts.acp.api.request/jobs_url` 确认 **`pt-blb39rgw` RUNNING**，run、display_name、启动脚本匹配，4 节点 × 8 卡。
- **进度与数值**：01:55:00 UTC 主采样为 **step 6400 / epoch 0.4066003394**（metrics 写入于 01:54:55 UTC），01:55:23 UTC 复核一致；相对上次 step 5940 增加 **460 步**。新增 **46 个 train 日志点（5950–6400）**，数值均有限，skipped/discarded 全为 0，epoch 持续推进，无半行 JSON，无新增 val。最新 objective **2.606659**、first CE **0.948391**、residual CE **5.527560**、grad norm **0.129658**；backbone LR **1e-4**、新模块 LR **3e-4**，符合稳态计划。TensorBoard step 6400 train loss/epoch 与 metrics 一致。
- **吞吐与等待**：新增单步采样耗时 **7.492–8.274 秒**、均值 **7.778 秒**；data wait **0.081–0.747 秒**，比例 **1.08%–9.11%**、均值 **3.30%**；吞吐 **9850–10871 音频秒/秒**，PyTorch 峰值显存 **60.26–60.69 GiB**。最近十点（6310–6400）单步均值 **7.622 秒**、等待比例均值 **2.34%**，无持续退化。统计仅覆盖每 10 步记录的单步采样，不代表连续全部训练步。
- **GPU**：主采样覆盖四节点 **32 个不同 UUID**，样本年龄 **4.9–13.4 秒**，NVML 显存 **66.70–67.04 GiB/卡**；rank-0 利用率 **78%–100%**、rank-24 **80%–100%**，其余均 **100%**。01:55:23 UTC 二次采样年龄 **5.3–8.2 秒**，利用率 **96%–100%**（31 卡为 100%），结合正常单步耗时，未见持续低利用率。
- **日志读取异常复核**：四节点 stdout 未匹配 OOM、NaN/Inf、非法访存、NCCL 错误或 traceback，非主节点仍只有既有启动 FA2 dtype 警告。首次读取主节点 stdout 尾部出现 **708 个空字节**，与 10 月 1 日 23:54 巡检现象类似；01:55:23 UTC 两次读取均恢复正常，四节点空字节计数均为 **0**，区间 5950–6400 的主节点日志点齐全，step 6400 完整。原始异常与重读证据均保留。并发追加或共享文件系统可见性只是可能解释，原因未确认；未发现持久日志损坏或训练受影响，不据此重启。后续继续关注复现频率及是否伴随指标/进度异常。
- **评估与恢复点**：最近仍为 **step 5000** 的 teacher-forcing val 与完整 checkpoint，val loss **3.166461**、first CE **1.302180**、residual CE **6.214271**，TensorBoard 一致。latest 指向 `step-00005000`；COMPLETE、metadata、非空 DCP `.metadata`、rank 0–31 的 **32 个非空 RNG 文件与 32 个非空分片**齐全，world_size=32、signature 与 run 一致。恢复游标 **step=5000 / epoch=0 / next_batch=5000 / samples_in_epoch=40295457**。当前保留 **2500、5000** 两份；本次仅验证结构，未执行完整加载或恢复。下一次预定自动 eval/save 为 step **7500**。
- **磁盘与增长**：共享盘可用 **36.38 TiB**；本 run checkpoint 约 **15.72 GiB**，checkpoint/submission 字节数不变。日志增加 **1,268,605 字节**、TensorBoard 增加 **59,502 字节**，符合正常追加；共享盘总已用量增加 **47.28 GiB**，不能归因于本 run，容量仍充足。
- **判断与动作**：正常训练，保持原配置；未停止、重启或修改训练代码/数据/日志。OOM 原配置重试 **0**、瞬态故障重试 **0**。登记本次已恢复的 stdout 读取异常，追加记录、保存证据并更新 agent-state，保留历史事件和最近 eval/checkpoint=5000；未创建 COMPLETE/STOP、未启动另一 supervisor、未 commit/push。本轮结束，不等待下一小时。
- **详细证据**：run 下 `supervision/20261002T015427Z-evidence.json`、`supervision/20261002T015427Z-platform.json` 与 `supervision/20261002T015427Z-recheck.json`。

## 2026-10-02 02:54:27 UTC：第十五次定时巡检，正常训练

- **身份与平台**：开始于 `2026-10-02T02:54:27.025403+00:00`，已重读手册、监督记录、agent-state 及最新 submission `20261001T120352364999Z` 的 job.json。02:54:59 UTC 经 `read_yaml` 与 `scripts.acp.api.request/jobs_url` 确认 **`pt-blb39rgw` RUNNING**，run、display_name、启动脚本匹配，4 节点 × 8 卡。
- **进度与数值**：主采样及 02:55:10 UTC 复核为 **step 6860 / epoch 0.4346612187**（metrics 写入于 02:54:07 UTC），相对上次 step 6400 增加 **460 步**。新增 **46 个 train 日志点（6410–6860）**，数值均有限，skipped/discarded 全为 0，epoch 持续推进，无半行 JSON，无新增 val。最新 objective **2.908000**、first CE **1.246764**、residual CE **5.537451**、grad norm **0.197466**；backbone LR **1e-4**、新模块 LR **3e-4**，符合稳态计划。TensorBoard step 6860 train loss/epoch 与 metrics 一致。
- **吞吐与等待**：新增单步采样耗时 **7.455–8.365 秒**、均值 **7.708 秒**；data wait **0.061–0.713 秒**，比例 **0.82%–8.52%**、均值 **2.61%**；吞吐 **9703–10938 音频秒/秒**，PyTorch 峰值显存 **60.29–60.75 GiB**。最近十点（6770–6860）单步均值 **7.716 秒**、等待比例均值 **2.28%**，无持续退化。统计仅覆盖每 10 步记录的单步采样，不代表连续全部训练步。
- **GPU 与日志**：02:54:59 UTC 四节点采样覆盖 **32 个不同 UUID**，样本年龄 **9.2–10.6 秒**，利用率全部 **100%**，NVML 显存 **66.70–67.04 GiB/卡**。四节点 stdout 未匹配 OOM、NaN/Inf、非法访存、NCCL 错误或 traceback；非主节点仍只有既有启动 FA2 dtype 警告，GPU 采样持续更新。四节点日志空字节计数均为 **0**，区间 6410–6860 主节点 train 日志点齐全，上轮瞬时读取异常本轮未复现。
- **评估与恢复点**：最近仍为 **step 5000** 的 teacher-forcing val 与完整 checkpoint，val loss **3.166461**、first CE **1.302180**、residual CE **6.214271**，TensorBoard 一致。latest 指向 `step-00005000`；COMPLETE、metadata、非空 DCP `.metadata`、rank 0–31 的 **32 个非空 RNG 文件与 32 个非空分片**齐全，world_size=32、signature 与 run 一致。恢复游标 **step=5000 / epoch=0 / next_batch=5000 / samples_in_epoch=40295457**。当前保留 **2500、5000** 两份；本次仅验证结构，未执行完整加载或恢复。距离下一次预定 step **7500** 自动 eval/save 尚有 **640 步**。
- **磁盘与增长**：共享盘可用 **36.33 TiB**；本 run checkpoint 约 **15.72 GiB**，checkpoint/submission 字节数不变。日志增加 **1,273,011 字节**、TensorBoard 增加 **58,236 字节**，符合正常追加；共享盘总已用量增加 **47.45 GiB**，不能归因于本 run，容量仍充足。
- **判断与动作**：正常训练，保持原配置；未停止、重启或修改训练代码/数据。OOM 原配置重试 **0**、瞬态故障重试 **0**，无新增故障事件。追加记录、保存证据并更新 agent-state，保留历史事件和最近 eval/checkpoint=5000；未创建 COMPLETE/STOP、未启动另一 supervisor、未 commit/push。后续继续核对训练进度及预定 eval/save；本轮结束，不等待下一小时。
- **详细证据**：run 下 `supervision/20261002T025427Z-evidence.json` 与 `supervision/20261002T025427Z-platform.json`。

## 2026-10-02 03:54:27 UTC：第十六次定时巡检，正常训练，接近 step 7500 评估与保存

- **身份与平台**：开始于 `2026-10-02T03:54:27.025494+00:00`，已重读手册、监督记录、agent-state 及最新 submission `20261001T120352364999Z` 的 job.json。03:55:06 UTC 经 `read_yaml` 与 `scripts.acp.api.request/jobs_url` 确认 **`pt-blb39rgw` RUNNING**，run、display_name、启动脚本匹配，4 节点 × 8 卡。
- **进度与数值**：03:55:07 UTC 主采样及 03:55:16 UTC 复核为 **step 7330 / epoch 0.4640762423**（metrics 写入于 03:54:38 UTC），相对上次 step 6860 增加 **470 步**。新增 **47 个 train 日志点（6870–7330）**，数值均有限，skipped/discarded 全为 0，epoch 持续推进，无半行 JSON，无新增 val。最新 objective **3.540048**、first CE **1.668149**、residual CE **6.239663**、grad norm **0.204247**；backbone LR **1e-4**、新模块 LR **3e-4**，符合稳态计划。最新 objective 在本区间 **2.445–3.542** 范围内，未见数值故障。TensorBoard step 7330 train loss/epoch 与 metrics 一致。
- **吞吐与等待**：新增单步采样耗时 **7.437–8.325 秒**、均值 **7.749 秒**；data wait **0.090–0.724 秒**，比例 **1.21%–8.77%**、均值 **2.73%**；吞吐 **9751–10941 音频秒/秒**，PyTorch 峰值显存 **60.31–60.74 GiB**。最近十点（7240–7330）单步均值 **7.743 秒**、等待比例均值 **2.20%**，无持续退化。统计仅覆盖每 10 步记录的单步采样，不代表连续全部训练步。
- **GPU 与日志**：03:55:07 UTC 四节点采样覆盖 **32 个不同 UUID**，样本年龄 **6.3–16.1 秒**，利用率全部 **100%**，NVML 显存 **66.70–67.04 GiB/卡**。四节点 stdout 未匹配 OOM、NaN/Inf、非法访存、NCCL 错误或 traceback；非主节点仍只有既有启动 FA2 dtype 警告，GPU 采样持续更新。四节点日志空字节计数均为 **0**，区间 6870–7330 主节点 train 日志点齐全，此前瞬时读取异常本轮未复现。
- **评估与恢复点**：最近仍为 **step 5000** 的 teacher-forcing val 与完整 checkpoint，val loss **3.166461**、first CE **1.302180**、residual CE **6.214271**，TensorBoard 一致。latest 指向 `step-00005000`；COMPLETE、metadata、非空 DCP `.metadata`、rank 0–31 的 **32 个非空 RNG 文件与 32 个非空分片**齐全，world_size=32、signature 与 run 一致。恢复游标 **step=5000 / epoch=0 / next_batch=5000 / samples_in_epoch=40295457**。当前保留 **2500、5000** 两份；本次仅验证结构，未执行完整加载或恢复。距离下一次预定 step **7500** 自动 eval/save 尚有 **170 步**，下轮重点核对对应 val、完整 checkpoint 及保存后进度。
- **磁盘与增长**：共享盘可用 **36.29 TiB**；本 run checkpoint 约 **15.72 GiB**，checkpoint/submission 字节数不变。日志增加 **1,277,990 字节**、TensorBoard 增加 **59,502 字节**，符合正常追加；共享盘总已用量增加 **42.13 GiB**，不能归因于本 run，容量仍充足。
- **判断与动作**：正常训练，保持原配置；未停止、重启或修改训练代码/数据。OOM 原配置重试 **0**、瞬态故障重试 **0**，无新增故障事件。追加记录、保存证据并更新 agent-state，保留历史事件和最近 eval/checkpoint=5000；未创建 COMPLETE/STOP、未启动另一 supervisor、未 commit/push。本轮结束，不等待下一小时。
- **详细证据**：run 下 `supervision/20261002T035427Z-evidence.json` 与 `supervision/20261002T035427Z-platform.json`。

## 2026-10-02 04:54:27 UTC：第十七次定时巡检，step 7500 评估与保存完成

- **身份与平台**：开始于 `2026-10-02T04:54:27.025572+00:00`，已重读手册、监督记录、agent-state 及最新 submission `20261001T120352364999Z` 的 job.json。04:55:08 UTC 经 `read_yaml` 与 `scripts.acp.api.request/jobs_url` 确认 **`pt-blb39rgw` RUNNING**，run、display_name、启动脚本匹配，4 节点 × 8 卡。
- **新评估**：step **7500** 的全量 **128219 条 holdout** 评估完成，skipped/discarded 均为 0，所有指标有限。teacher-forcing val loss **2.983602**（step 5000 为 3.166461）、first CE **1.199507**（此前 1.302180）、residual CE **5.946984**（此前 6.214271），TensorBoard 与 metrics 一致。这些为 teacher-forcing 指标，不是音频生成质量评分。
- **新恢复点与保留策略**：`checkpoints/step-00007500` 的 COMPLETE 写入于 **04:17:48 UTC**，latest 指向该目录；metadata、非空 DCP `.metadata`、rank 0–31 的 **32 个非空 RNG 文件与 32 个非空分片**齐全，world_size=32、signature 与 run 一致。恢复游标 **step=7500 / epoch=0 / next_batch=7500 / samples_in_epoch=60806212**，scheduler.last_epoch=7500，LR 为 1e-4 / 3e-4。DCP 索引可读取，含 **478 个 model 键、1217 个 optimizer 键**，引用 32 个分片，索引字节范围均在对应文件大小内；未完整加载权重或执行恢复。run 配置 `keep_checkpoints: 2`，训练日志明确记录自动清理 `step-00002500`，当前保留 **5000、7500**；本 agent 未删除文件。
- **训练进度**：主采样 step 7780，04:55:22 UTC 复核到 **step 7790 / epoch 0.4931935189**（metrics 写入于 04:55:13 UTC），确认 eval/save 后正常继续。相对上次 step 7330 增加 **460 步**，新增 **46 个 train 日志点（7340–7790）和 1 条 val**。数值均有限，skipped/discarded 全为 0，epoch 持续推进，无半行 JSON。最新 objective **2.856728**、first CE **1.056736**、residual CE **5.999973**、grad norm **0.071504**；backbone LR **1e-4**、新模块 LR **3e-4**，符合稳态计划。TensorBoard step 7790 train loss/epoch 与 metrics 一致。
- **吞吐与等待**：新增单步采样耗时 **7.452–8.313 秒**、均值 **7.764 秒**；data wait **0.061–0.721 秒**，比例 **0.82%–8.98%**、均值 **3.23%**；吞吐 **9820–10934 音频秒/秒**，PyTorch 峰值显存 **60.17–60.76 GiB**。主采样最近十点（7690–7780）单步均值 **7.775 秒**、等待比例均值 **3.35%**；复核 step 7790 为 **7.478 秒 / 1.06%**，无持续退化。统计仅覆盖每 10 步记录的单步采样，不代表连续全部训练步或 eval/save 耗时。
- **GPU 与日志**：04:55:09 UTC 四节点采样覆盖 **32 个不同 UUID**，样本年龄 **10.5–16.4 秒**，利用率 **98%–100%**，NVML 显存 **66.70–67.04 GiB/卡**。四节点 stdout 未匹配 OOM、NaN/Inf、非法访存、NCCL 错误或 traceback；非主节点仍只有既有启动 FA2 dtype 警告，GPU 采样持续更新。四节点日志空字节计数均为 **0**，区间主节点 train 日志点齐全，此前瞬时读取异常本轮未复现。
- **磁盘与增长**：共享盘可用 **36.29 TiB**；checkpoint 总计仍约 **15.72 GiB**，字节数不变，符合新增 7500 同时自动淘汰 2500 的两份保留策略。日志增加 **1,273,534 字节**、TensorBoard 增加 **57,122 字节**，submission 未变。共享盘总已用量增加 **6.44 GiB**，不能归因于本 run，容量仍充足。
- **判断与动作**：正常训练，自动 eval/save 按计划完成，保持原配置；未停止、重启或修改训练代码/数据。OOM 原配置重试 **0**、瞬态故障重试 **0**。登记 eval/checkpoint=**7500** 和训练自动保留事件，追加记录、保存证据并更新 agent-state，保留历史事件；未创建 COMPLETE/STOP、未启动另一 supervisor、未 commit/push。下一次预定自动 eval/save 为 step **10000**；本轮结束，不等待下一小时。
- **详细证据**：run 下 `supervision/20261002T045427Z-evidence.json`（含 DCP 索引、保留日志和 step 7790 复核）与 `supervision/20261002T045427Z-platform.json`。

## 2026-10-02 05:54:27 UTC：第十八次定时巡检，正常训练

- **身份与平台**：开始于 `2026-10-02T05:54:27.025678+00:00`，已重读手册、监督记录、agent-state 及最新 submission `20261001T120352364999Z` 的 job.json。05:55:07 UTC 经 `read_yaml` 与 `scripts.acp.api.request/jobs_url` 确认 **`pt-blb39rgw` RUNNING**，run、display_name、启动脚本匹配，4 节点 × 8 卡。
- **进度与数值**：主采样及 05:55:18 UTC 复核为 **step 8250 / epoch 0.5216723089**（metrics 写入于 05:54:11 UTC），相对上次 step 7790 增加 **460 步**。新增 **46 个 train 日志点（7800–8250）**，数值均有限，skipped/discarded 全为 0，epoch 持续推进，无半行 JSON，无新增 val。最新 objective **3.093241**、first CE **1.288569**、residual CE **6.015572**、grad norm **0.170766**；backbone LR **1e-4**、新模块 LR **3e-4**，符合稳态计划。TensorBoard step 8250 train loss/epoch 与 metrics 一致。
- **吞吐与等待**：新增单步采样耗时 **7.437–8.087 秒**、均值 **7.687 秒**；data wait **0.084–0.740 秒**，比例 **1.12%–9.19%**、均值 **2.51%**；吞吐 **10064–10961 音频秒/秒**，PyTorch 峰值显存 **60.29–60.86 GiB**。最近十点（8160–8250）单步均值 **7.759 秒**、等待比例均值 **3.21%**，无持续退化。统计仅覆盖每 10 步记录的单步采样，不代表连续全部训练步。
- **GPU 与日志**：05:55:07 UTC 四节点采样覆盖 **32 个不同 UUID**，样本年龄 **6.8–13.2 秒**，利用率全部 **100%**，NVML 显存 **66.70–67.04 GiB/卡**。四节点 stdout 未匹配 OOM、NaN/Inf、非法访存、NCCL 错误或 traceback；非主节点仍只有既有启动 FA2 dtype 警告，GPU 采样持续更新。四节点日志空字节计数均为 **0**，区间 7800–8250 主节点 train 日志点齐全，此前瞬时读取异常本轮未复现。
- **评估与恢复点**：最近仍为 **step 7500** 的 teacher-forcing val 与完整 checkpoint，val loss **2.983602**、first CE **1.199507**、residual CE **5.946984**，TensorBoard 一致。latest 指向 `step-00007500`；COMPLETE、metadata、非空 DCP `.metadata`、rank 0–31 的 **32 个非空 RNG 文件与 32 个非空分片**齐全，world_size=32、signature 与 run 一致。恢复游标 **step=7500 / epoch=0 / next_batch=7500 / samples_in_epoch=60806212**。当前保留 **5000、7500** 两份；本次仅验证结构，未执行完整加载或恢复。下一次预定自动 eval/save 为 step **10000**。
- **磁盘与增长**：共享盘可用 **36.28 TiB**；本 run checkpoint 约 **15.72 GiB**，checkpoint/submission 字节数不变。日志增加 **1,273,856 字节**、TensorBoard 增加 **59,502 字节**，符合正常追加；共享盘总已用量增加 **6.59 GiB**，不能归因于本 run，容量仍充足。
- **判断与动作**：正常训练，保持原配置；未停止、重启或修改训练代码/数据。OOM 原配置重试 **0**、瞬态故障重试 **0**，无新增故障事件。追加记录、保存证据并更新 agent-state，保留历史事件和最近 eval/checkpoint=7500；未创建 COMPLETE/STOP、未启动另一 supervisor、未 commit/push。后续继续检查稳态进度与预定 eval/save；本轮结束，不等待下一小时。
- **详细证据**：run 下 `supervision/20261002T055427Z-evidence.json` 与 `supervision/20261002T055427Z-platform.json`。

## 2026-10-02 06:54:27 UTC：第十九次定时巡检，正常训练，复核 GPU 间歇波动

- **身份与平台**：开始于 `2026-10-02T06:54:27.025757+00:00`，已重读手册、监督记录、agent-state 及最新 submission `20261001T120352364999Z` 的 job.json。06:55:02 UTC 经 `read_yaml` 与 `scripts.acp.api.request/jobs_url` 确认 **`pt-blb39rgw` RUNNING**，run、display_name、启动脚本匹配，4 节点 × 8 卡。
- **进度与数值**：主采样 step 8720，06:56:20 UTC 复核推进到 **step 8730 / epoch 0.5528949713**（metrics 写入于 06:55:54 UTC），相对上次 step 8250 增加 **480 步**。新增 **48 个 train 日志点（8260–8730）**，数值均有限，skipped/discarded 全为 0，epoch 持续推进，无半行 JSON，无新增 val。最新 objective **2.999034**、first CE **1.187527**、residual CE **6.038356**、grad norm **0.131059**；backbone LR **1e-4**、新模块 LR **3e-4**，符合稳态计划。TensorBoard step 8730 train loss/epoch 与 metrics 一致。
- **吞吐与等待**：新增单步采样耗时 **7.390–8.535 秒**、均值 **7.723 秒**；data wait **0.076–0.767 秒**，比例 **1.01%–8.99%**、均值 **2.96%**；吞吐 **9522–10993 音频秒/秒**，PyTorch 峰值显存 **60.31–60.70 GiB**。主采样最近十点（8630–8720）单步均值 **7.788 秒**、等待比例均值 **3.34%**；复核 step 8730 为 **7.648 秒 / 2.28%**，无持续退化。统计仅覆盖每 10 步记录的单步采样，不代表连续全部训练步。
- **GPU 波动复核**：06:55:03 UTC 四节点采样覆盖 **32 个不同 UUID**，样本年龄 **7.7–14.0 秒**，NVML 显存 **66.70–67.04 GiB/卡**。rank-8 瞬时利用率 **21%–100%**，其余三节点均 **100%**；06:55:31 UTC 二次采样 rank-8 为 **0%–31%**，其余 **98%–100%**，各节点最新样本年龄不超过 **9.4 秒**。该时点最近 120 秒 rank-8 平均利用率 **86.05%**，其余节点 **98.23%–99.78%**。进一步保存最近 240 秒窗口：rank-8 在 06:54:52、06:55:22 出现低值，随后 06:55:02/12 和 06:55:32/42/52 均回到 **100%**，06:56:02 又出现 **0%** 单次采样。结合 step 8730 正常推进、单步时间和数据等待，判定间歇波动，未见持续卡死或吞吐退化证据；原因尚未定位，不据瞬时低值重启或调参。后续继续比较波动频率与训练耗时。
- **日志**：四节点 stdout 未匹配 OOM、NaN/Inf、非法访存、NCCL 错误或 traceback；非主节点仍只有既有启动 FA2 dtype 警告，GPU 采样持续更新。四节点日志空字节计数均为 **0**，区间 8260–8730 主节点 train 日志点齐全，此前瞬时读取异常本轮未复现。
- **评估与恢复点**：最近仍为 **step 7500** 的 teacher-forcing val 与完整 checkpoint，val loss **2.983602**、first CE **1.199507**、residual CE **5.946984**，TensorBoard 一致。latest 指向 `step-00007500`；COMPLETE、metadata、非空 DCP `.metadata`、rank 0–31 的 **32 个非空 RNG 文件与 32 个非空分片**齐全，world_size=32、signature 与 run 一致。恢复游标 **step=7500 / epoch=0 / next_batch=7500 / samples_in_epoch=60806212**。当前保留 **5000、7500** 两份；本次仅验证结构，未执行完整加载或恢复。下一次预定自动 eval/save 为 step **10000**。
- **磁盘与增长**：共享盘可用 **36.24 TiB**；本 run checkpoint 约 **15.72 GiB**，checkpoint/submission 字节数不变。日志增加 **1,273,690 字节**、TensorBoard 增加 **59,502 字节**，符合正常追加；共享盘总已用量增加 **44.57 GiB**，不能归因于本 run，容量仍充足。
- **判断与动作**：正常训练，保持原配置；未停止、重启或修改训练代码/数据。OOM 原配置重试 **0**、瞬态故障重试 **0**，无新增故障事件。追加记录、保存 GPU 复核及完整巡检证据并更新 agent-state，保留历史事件和最近 eval/checkpoint=7500；未创建 COMPLETE/STOP、未启动另一 supervisor、未 commit/push。本轮结束，不等待下一小时。
- **详细证据**：run 下 `supervision/20261002T065427Z-evidence.json`（含 step 8730、GPU 二次采样及 rank-8 窗口复核）与 `supervision/20261002T065427Z-platform.json`。

## 2026-10-02 07:54:27 UTC：第二十次定时巡检，正常训练；日志瞬时读取异常复核恢复

- **身份与平台**：开始于 `2026-10-02T07:54:27.025835+00:00`，已重读手册、监督记录、agent-state 及最新 submission `20261001T120352364999Z` 的 job.json。07:55:09 UTC 经 `read_yaml` 与 `scripts.acp.api.request/jobs_url` 确认 **`pt-blb39rgw` RUNNING**，run、display_name、启动脚本匹配，4 节点 × 8 卡。
- **进度与数值**：主采样及 07:55:43 UTC 复核为 **step 9190 / epoch 0.5821908696**（metrics 写入于 07:55:10 UTC），相对上次 step 8730 增加 **460 步**。新增 **46 个 train 日志点（8740–9190）**，数值均有限，skipped/discarded 全为 0，epoch 持续推进，无半行 JSON，无新增 val。最新 objective **2.908833**、first CE **1.146474**、residual CE **5.874530**、grad norm **0.089358**；backbone LR **1e-4**、新模块 LR **3e-4**，符合稳态计划。TensorBoard step 9190 train loss/epoch 与 metrics 一致。
- **吞吐与等待**：新增单步采样耗时 **7.428–8.447 秒**、均值 **7.719 秒**；data wait **0.090–0.702 秒**，比例 **1.18%–8.31%**、均值 **2.70%**；吞吐 **9635–10973 音频秒/秒**，PyTorch 峰值显存 **60.29–60.79 GiB**。最近十点（9100–9190）单步均值 **7.578 秒**、等待比例均值 **1.78%**，无持续退化。统计仅覆盖每 10 步记录的单步采样，不代表连续全部训练步。
- **GPU 复核**：07:55:10 UTC 四节点采样覆盖 **32 个不同 UUID**，样本年龄 **0.7–12.0 秒**，利用率全部 **100%**，NVML 显存 **66.70–67.04 GiB/卡**。07:55:44 UTC 二次采样仍为 **32 卡 100%**，最新样本年龄不超过 **13.1 秒**。该时点最近 120 秒 rank-8 平均利用率 **100%**，其余节点 **99.76%–99.93%**；上轮 rank-8 间歇低值在本窗口未复现，吞吐正常。
- **日志读取异常**：四节点 stdout 未匹配 OOM、NaN/Inf、非法访存、NCCL 错误或 traceback；非主节点仍只有既有启动 FA2 dtype 警告。首次读取主节点 stdout 尾部出现 **701 个空字节**，采样恰逢 step 9190 日志追加；07:55:43 和 07:55:44 UTC 重读均恢复，四节点空字节计数为 **0**，区间 8740–9190 主节点 train 日志点齐全，step 9190 完整。原始异常和复核尾部均已保存；并发追加或共享文件系统可见性是可能解释，原因未确认，不据已消失的读取异常重启。TensorBoard 文件 mtime 比主采样开始时间晚约 0.045 秒，是采集期间追加，后续标量核对正常。
- **评估与恢复点**：最近仍为 **step 7500** 的 teacher-forcing val 与完整 checkpoint，val loss **2.983602**、first CE **1.199507**、residual CE **5.946984**，TensorBoard 一致。latest 指向 `step-00007500`；COMPLETE、metadata、非空 DCP `.metadata`、rank 0–31 的 **32 个非空 RNG 文件与 32 个非空分片**齐全，world_size=32、signature 与 run 一致。恢复游标 **step=7500 / epoch=0 / next_batch=7500 / samples_in_epoch=60806212**。当前保留 **5000、7500** 两份；本次仅验证结构，未执行完整加载或恢复。下一次预定自动 eval/save 为 step **10000**。
- **磁盘与增长**：共享盘可用 **36.10 TiB**；本 run checkpoint 约 **15.72 GiB**，checkpoint/submission 字节数不变。日志增加 **1,278,007 字节**、TensorBoard 增加 **59,502 字节**，符合正常追加；共享盘总已用量增加 **140.74 GiB**，高于上轮 44.57 GiB，不能归因于本 run，容量仍充足，后续继续比较。
- **判断与动作**：正常训练，保持原配置；未停止、重启或修改训练代码/数据/日志。OOM 原配置重试 **0**、瞬态故障重试 **0**。登记已恢复的 stdout 读取异常，追加记录、保存完整证据并更新 agent-state，保留历史事件和最近 eval/checkpoint=7500；未创建 COMPLETE/STOP、未启动另一 supervisor、未 commit/push。后续关注日志读取异常是否持续及预定 eval/save；本轮结束，不等待下一小时。
- **详细证据**：run 下 `supervision/20261002T075427Z-evidence.json`（含日志与 GPU 二次复核）与 `supervision/20261002T075427Z-platform.json`。

## 2026-10-02 08:54:27 UTC：第二十一次定时巡检，正常训练，接近 step 10000 评估与保存

- **身份与平台**：开始于 `2026-10-02T08:54:27.025916+00:00`，已重读手册、监督记录、agent-state 及最新 submission `20261001T120352364999Z` 的 job.json。08:55:12 UTC 经 `read_yaml` 与 `scripts.acp.api.request/jobs_url` 确认 **`pt-blb39rgw` RUNNING**，run、display_name、启动脚本匹配，4 节点 × 8 卡。
- **进度与数值**：主采样及 08:55:21 UTC 复核为 **step 9650 / epoch 0.6122166966**（metrics 写入于 08:54:32 UTC），相对上次 step 9190 增加 **460 步**。新增 **46 个 train 日志点（9200–9650）**，数值均有限，skipped/discarded 全为 0，epoch 持续推进，无半行 JSON，无新增 val。最新 objective **3.154445**、first CE **1.517943**、residual CE **5.455005**、grad norm **0.328582**；backbone LR **1e-4**、新模块 LR **3e-4**，符合稳态计划。TensorBoard step 9650 train loss/epoch 与 metrics 一致。
- **吞吐与等待**：新增单步采样耗时 **7.418–8.441 秒**、均值 **7.736 秒**；data wait **0.078–0.753 秒**，比例 **1.02%–9.15%**、均值 **2.67%**；吞吐 **9525–10977 音频秒/秒**，PyTorch 峰值显存 **60.16–60.87 GiB**。最近十点（9560–9650）单步均值 **7.669 秒**、等待比例均值 **1.97%**，无持续退化。统计仅覆盖每 10 步记录的单步采样，不代表连续全部训练步。
- **GPU 与日志**：08:55:13 UTC 四节点采样覆盖 **32 个不同 UUID**，样本年龄 **8.2–13.3 秒**，利用率全部 **100%**，NVML 显存 **66.70–67.04 GiB/卡**。四节点 stdout 未匹配 OOM、NaN/Inf、非法访存、NCCL 错误或 traceback；非主节点仍只有既有启动 FA2 dtype 警告，GPU 采样持续更新。四节点日志空字节计数均为 **0**，区间 9200–9650 主节点 train 日志点齐全，上轮瞬时读取异常本轮未复现。
- **评估与恢复点**：最近仍为 **step 7500** 的 teacher-forcing val 与完整 checkpoint，val loss **2.983602**、first CE **1.199507**、residual CE **5.946984**，TensorBoard 一致。latest 指向 `step-00007500`；COMPLETE、metadata、非空 DCP `.metadata`、rank 0–31 的 **32 个非空 RNG 文件与 32 个非空分片**齐全，world_size=32、signature 与 run 一致。恢复游标 **step=7500 / epoch=0 / next_batch=7500 / samples_in_epoch=60806212**。当前保留 **5000、7500** 两份；本次仅验证结构，未执行完整加载或恢复。距离下一次预定 step **10000** 自动 eval/save 尚有 **350 步**，下轮重点核对同 step 的 val、完整 checkpoint 和保存后进度。
- **磁盘与增长**：共享盘可用 **35.96 TiB**；本 run checkpoint 约 **15.72 GiB**，checkpoint/submission 字节数不变。日志增加 **1,273,829 字节**、TensorBoard 增加 **58,236 字节**，符合正常追加；共享盘总已用量增加 **141.38 GiB**，接近上轮 140.74 GiB，不能归因于本 run，容量仍充足，后续继续比较。
- **判断与动作**：正常训练，保持原配置；未停止、重启或修改训练代码/数据。OOM 原配置重试 **0**、瞬态故障重试 **0**，无新增故障事件。追加记录、保存证据并更新 agent-state，保留历史事件和最近 eval/checkpoint=7500；未创建 COMPLETE/STOP、未启动另一 supervisor、未 commit/push。本轮结束，不等待下一小时。
- **详细证据**：run 下 `supervision/20261002T085427Z-evidence.json` 与 `supervision/20261002T085427Z-platform.json`。

## 2026-10-02 09:54:27 UTC：第二十二次定时巡检，step 10000 评估与保存完成

- **身份与平台**：开始于 `2026-10-02T09:54:27.025999+00:00`，已重读手册、监督记录、agent-state 及最新 submission `20261001T120352364999Z` 的 job.json。09:55:05 UTC 经 `read_yaml` 与 `scripts.acp.api.request/jobs_url` 确认 **`pt-blb39rgw` RUNNING**，run、display_name、启动脚本匹配，4 节点 × 8 卡。
- **新评估**：step **10000** 的全量 **128219 条 holdout** 评估完成，skipped/discarded 均为 0，所有指标有限。teacher-forcing val loss **2.881729**（step 7500 为 2.983602）、first CE **1.142551**（此前 1.199507）、residual CE **5.797259**（此前 5.946984），TensorBoard 与 metrics 一致。这些为 teacher-forcing 指标，不是音频生成质量评分。
- **新恢复点与保留策略**：`checkpoints/step-00010000` 的 COMPLETE 写入于 **09:41:01 UTC**，latest 指向该目录；metadata、非空 DCP `.metadata`、rank 0–31 的 **32 个非空 RNG 文件与 32 个非空分片**齐全，world_size=32、signature 与 run 一致。恢复游标 **step=10000 / epoch=0 / next_batch=10000 / samples_in_epoch=81426791**，scheduler.last_epoch=10000，LR 为 1e-4 / 3e-4。DCP 索引可读取，含 **478 个 model 键、1217 个 optimizer 键**，引用 32 个分片，索引字节范围均在对应文件大小内；未完整加载权重或执行恢复。run 配置 `keep_checkpoints: 2`，训练日志明确记录自动清理 `step-00005000`，当前保留 **7500、10000**；本 agent 未删除文件。
- **训练进度**：主采样 step 10100，09:55:41 UTC 复核到 **step 10110 / epoch 0.6439851857**（metrics 写入于 09:55:17 UTC），确认 eval/save 后正常继续。相对上次 step 9650 增加 **460 步**，新增 **46 个 train 日志点（9660–10110）和 1 条 val**。数值均有限，skipped/discarded 全为 0，epoch 持续推进，无半行 JSON。最新 objective **2.829726**、first CE **1.047051**、residual CE **5.942250**、grad norm **0.134493**；backbone LR **1e-4**、新模块 LR **3e-4**，符合稳态计划。TensorBoard step 10110 train loss/epoch 与 metrics 一致。
- **吞吐与等待**：新增单步采样耗时 **7.431–8.275 秒**、均值 **7.761 秒**；data wait **0.104–0.745 秒**，比例 **1.37%–9.00%**、均值 **3.22%**；吞吐 **9860–10957 音频秒/秒**，PyTorch 峰值显存 **60.23–60.74 GiB**。eval/save 后最近十点（10010–10100）单步均值 **7.788 秒**、等待比例均值 **3.34%**；复核 step 10110 为 **7.599 秒 / 1.78%**，无持续退化。统计仅覆盖每 10 步记录的单步采样，不代表连续全部训练步或 eval/save 耗时。
- **GPU 与日志**：09:55:05 UTC 四节点采样覆盖 **32 个不同 UUID**，样本年龄 **7.0–14.1 秒**，利用率全部 **100%**，NVML 显存 **66.70–67.04 GiB/卡**。四节点 stdout 未匹配 OOM、NaN/Inf、非法访存、NCCL 错误或 traceback；非主节点仍只有既有启动 FA2 dtype 警告，GPU 采样持续更新。四节点日志空字节计数均为 **0**，区间主节点 train 日志点齐全，此前瞬时读取异常本轮未复现。
- **磁盘与增长**：共享盘可用 **35.85 TiB**；checkpoint 总计仍约 **15.72 GiB**（增加 4 字节），符合新增 10000 同时自动淘汰 5000 的两份保留策略。日志增加 **1,270,998 字节**、TensorBoard 增加 **57,122 字节**，submission 未变。共享盘总已用量增加 **112.82 GiB**，不能归因于本 run，容量仍充足。
- **判断与动作**：正常训练，自动 eval/save 按计划完成，保持原配置；未停止、重启或修改训练代码/数据。OOM 原配置重试 **0**、瞬态故障重试 **0**。登记 eval/checkpoint=**10000** 和训练自动保留事件，追加记录、保存证据并更新 agent-state，保留历史事件；未创建 COMPLETE/STOP、未启动另一 supervisor、未 commit/push。下一次预定自动 eval/save 为 step **12500**；本轮结束，不等待下一小时。
- **详细证据**：run 下 `supervision/20261002T095427Z-evidence.json`（含 DCP 索引、保留日志和 step 10110 复核）与 `supervision/20261002T095427Z-platform.json`。

## 2026-10-02 10:54:27 UTC：第二十三次定时巡检，正常训练

- **身份与平台**：开始于 `2026-10-02T10:54:27.026105+00:00`，已重读手册、监督记录、agent-state 及最新 submission `20261001T120352364999Z` 的 job.json。10:55:08 UTC 经 `read_yaml` 与 `scripts.acp.api.request/jobs_url` 确认 **`pt-blb39rgw` RUNNING**，run、display_name、启动脚本匹配，4 节点 × 8 卡。
- **进度与数值**：主采样及 10:55:19 UTC 复核为 **step 10570 / epoch 0.6724712829**（metrics 写入于 10:54:21 UTC），相对上次 step 10110 增加 **460 步**。新增 **46 个 train 日志点（10120–10570）**，数值均有限，skipped/discarded 全为 0，epoch 持续推进，无半行 JSON，无新增 val。最新 objective **3.375361**、first CE **1.551026**、residual CE **6.081119**、grad norm **0.178185**；backbone LR **1e-4**、新模块 LR **3e-4**，符合稳态计划。TensorBoard step 10570 train loss/epoch 与 metrics 一致。
- **吞吐与等待**：新增单步采样耗时 **7.424–8.392 秒**、均值 **7.702 秒**；data wait **0.088–0.720 秒**，比例 **1.19%–8.62%**、均值 **2.70%**；吞吐 **9712–10958 音频秒/秒**，PyTorch 峰值显存 **60.31–60.66 GiB**。最近十点（10480–10570）单步均值 **7.802 秒**、等待比例均值 **3.21%**，无持续退化。统计仅覆盖每 10 步记录的单步采样，不代表连续全部训练步。
- **GPU 与日志**：10:55:09 UTC 四节点采样覆盖 **32 个不同 UUID**，样本年龄 **6.5–13.8 秒**，利用率全部 **100%**，NVML 显存 **66.70–67.04 GiB/卡**。四节点 stdout 未匹配 OOM、NaN/Inf、非法访存、NCCL 错误或 traceback；非主节点仍只有既有启动 FA2 dtype 警告，GPU 采样持续更新。四节点日志空字节计数均为 **0**，区间 10120–10570 主节点 train 日志点齐全，此前瞬时读取异常本轮未复现。
- **评估与恢复点**：最近仍为 **step 10000** 的 teacher-forcing val 与完整 checkpoint，val loss **2.881729**、first CE **1.142551**、residual CE **5.797259**，TensorBoard 一致。latest 指向 `step-00010000`；COMPLETE、metadata、非空 DCP `.metadata`、rank 0–31 的 **32 个非空 RNG 文件与 32 个非空分片**齐全，world_size=32、signature 与 run 一致。恢复游标 **step=10000 / epoch=0 / next_batch=10000 / samples_in_epoch=81426791**。当前保留 **7500、10000** 两份；本次仅验证结构，未执行完整加载或恢复。下一次预定自动 eval/save 为 step **12500**。
- **磁盘与增长**：共享盘可用 **35.79 TiB**；本 run checkpoint 约 **15.72 GiB**，checkpoint/submission 字节数不变。日志增加 **1,275,490 字节**、TensorBoard 增加 **59,502 字节**，符合正常追加；共享盘总已用量增加 **57.64 GiB**，低于上轮 112.82 GiB，不能归因于本 run，容量仍充足。
- **判断与动作**：正常训练，保持原配置；未停止、重启或修改训练代码/数据。OOM 原配置重试 **0**、瞬态故障重试 **0**，无新增故障事件。追加记录、保存证据并更新 agent-state，保留历史事件和最近 eval/checkpoint=10000；未创建 COMPLETE/STOP、未启动另一 supervisor、未 commit/push。后续继续检查稳态进度与预定 eval/save；本轮结束，不等待下一小时。
- **详细证据**：run 下 `supervision/20261002T105427Z-evidence.json` 与 `supervision/20261002T105427Z-platform.json`。

## 2026-10-02 11:45 UTC：主会话增加固定 checkpoint 归档并完成首批复制

- 用户新增要求：supervisor 每次发现新完整 checkpoint 后，执行固定命令保留到轮转之外。已更新[巡检手册](all16-20261001-incident-playbook.md)中的目标和命令，入口为 `scripts.archive_checkpoints`，目标固定为本 run 的 `archived-checkpoints/step-XXXXXXXX/`。
- 已实际完成 `step-00007500`、`step-00010000` 的独立复制，分别在 **11:43:28 UTC** 和 **11:44:37 UTC** 写入归档回执并发布。每份 **67 个文件**、约 **7.86 GiB**；所有文件的源读取流与目标重读 SHA256 一致，详细哈希保存在各归档的 `ARCHIVE_COMPLETE.json`。
- 复查所有归档文件均不是软链接 / 硬链接，源文件仍存在；latest 仍为 `step-00010000`，未改变训练或 keep_checkpoints。再次执行同一命令，两份均返回 `already_archived`，没有重复复制。归档保留全部版本，不参与训练目录的两份轮转。
- 两项针对性回归通过：源 checkpoint 删除后归档仍可校验、重复执行幂等、同大小内容损坏可被全量校验发现、复制失败不发布目录、缺失 RNG 拒绝归档。Ruff 与 diff 检查通过。复制本身的校验不等于执行完整训练恢复加载。
- 调度器 PID `1892623` 存活，session 未变，下一轮 **11:54:27 UTC**；已核对它指向更新后的手册且每轮重新读取。后续 agent 必须执行固定命令并记录结果，包括训练结束后的最终 checkpoint。没有启动第二个巡检或重启训练。
- 更早已被轮转删除的 step 100 / 2500 / 5000 不在本次归档中。本目录与训练使用同一 AFS，用于防轮转删除，不是异地备份。
