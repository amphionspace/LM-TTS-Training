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
