# 10kh 正式训练与巡检记录

用户授权：完整数据检查通过后，从初始组装模型启动正式训练，训练两个 epoch；每半小时检查训练过程，发现具体问题可以调整，但必须详细记录证据、原因、修改和结果。现有 1kh 和试训产物继续保留。

## 固定训练计划

- 配置：`configs/emilia-10kh-pretrain.yaml`；输出：`runs/emilia-en-zh-dynamic-10000h/`。
- 完整训练数据 6,869,742 条、9,999.19476 小时；验证集 512 条（中英各 256）。完整清单 SHA256、样本核查及精确 epoch 规划见 run 下 `preflight.json`。
- 4 × A100 80GB，BF16、FSDP、Flash Attention 2；每卡动态预算 6,000 音频帧 / 9,000 talker token，累积 1，16 DataLoader workers / 卡，prefetch 2。
- seed 42 下两个 epoch 分别 19,270、19,269 次更新，总计 38,539；schedule_steps 同为 38,539。warmup 1,000，主干峰值 LR 1e-4、新参数峰值 LR 3e-4。
- 从 `pretrained/assembled-qwen3-tts-frozen-conditioning` 初始化；不接续 1kh 或试训。训练仍只使用 speaker embedding，不新增 ICL 训练任务。
- 每 500 步保存和评估，保留最近两个完整 checkpoint；最后第 38,539 步同样保存并执行验证 loss 和两种生成评估。本次补齐末步评估条件，避免末步不是 500 的倍数时缺少最终结果。

## 验证口径

512 条均参与 teacher-forcing 验证 loss。`evaluation-pairs.json` 固定记录 511 条合格目标/参考配对：同 speaker、同语言、不同 ID 和规范化文本；参考取自训练集。英语目标 `emilia2:933a1fd2d0817ae5_005_000` 无合格参考，只从生成配对排除。这是已见说话人、留出文本评估，不代表未见说话人泛化。

在线每次生成选固定 8 条目标（4 EN + 4 ZH），分别运行 `speaker_only` 和 `icl`，使用同一组参考。ICL 额外输入参考文本及 codec 前缀，目标 codec 不输入生成；评估裁剪后的目标音频。最大新增 400 帧；ASR 为 multilingual Whisper small，英语规范化开启。输出分别位于 `evaluation/<mode>/step-XXXXXXXX/`。这 8 条用于持续诊断，不能把小样本 WER/CER 波动当作全验证集质量结论。

## 启动与恢复

在项目根目录运行 `python runs/emilia-en-zh-dynamic-10000h/launch.py`。后台启动需要将 stdin 关闭，并把 launcher stdout/stderr 写到本 run 的 `launcher.log`；launcher 有运行锁，训练 PID、进程启动标识和命令写入 `training-process.json`，退出码写入 `training-exit.json`，训练日志追加至 `train.log`。

有完整 checkpoint 时 launcher 自动 `--resume latest`。已有启动记录而没有完整 checkpoint 时拒绝自动从头重跑，需要核查失败现场。不要修改 checkpoint 元数据或绕过恢复签名；修改学习率/动态预算等训练语义前，必须先验证恢复路径和两个 epoch 的计划。所有人工或自动干预追加到本文。

## 2026-09-09 启动前核查

- 14:26 UTC：完整 train/val 清单哈希匹配准备报告；核查数量、ID/规范化文本隔离，以及全部 512 条验证 codec 和 30 条采样/极限训练 codec，共 542 个文件。精确规划得到上述两个 epoch 步数，rank 0 音频预算填充率约 97.96%。
- 显存压力测试：最长 375 帧音频 + 最长 398 token 文本（含特殊 token），每卡 11 条，4 卡完成两次真实优化更新。峰值 allocated 44.92 GiB，reserved 50.05 GiB，无 OOM，进程正常退出。证据：`memory-probe.log`。
- CLI 巡检可用性：实际 `codex exec` 已成功读取本地预检报告并输出结果，见 `scheduler-smoke-result.md`。当前容器不能启动 CLI 的 read-only sandbox，因此使用本会话已有的 danger-full-access 执行权限和 never 审批策略；巡检范围由下述任务提示固定。CLI 认证沿用本机已配置账号，不复制凭据。

## 定时监督与前 2,000 步观察

用户已选择固定独立持久化巡检 session。`scripts/monitor_training.py` 每 1,800 秒执行一次 Codex，首次创建会话，之后根据 `supervision/session-id` 用 `codex exec resume <ID>` 延续；不使用 `--last`，不使用 ephemeral。单次最长 1,500 秒，有独占锁避免重叠。快照、CLI 日志、回复、状态全部保存在 run 的 `supervision/`。会话机制参考 [OpenAI 非交互模式说明](https://learn.chatgpt.com/docs/non-interactive-mode)。

主会话负责观察前 2,000 步以及 500/1,000/1,500/2,000 的 checkpoint 和双模式生成。`manual-observation.json` 的 active=true 期间，巡检仅记录和建议，避免同时修改训练；主会话完成观察后改为 false，巡检按用户授权自主处理有明确证据的问题。具体判断、修复与停止条件见 [故障处置表](training-incident-playbook.md)。

脚本需要此机器和进程持续运行，不能跨主机重启自动恢复；Codex 服务/认证不可用时会记录巡检失败，训练进程独立继续，不能把机械快照当作已经完成 AI 审查。停止巡检可创建 `supervision/STOP`；恢复需移除此显式停止标记并重新启动同一脚本，沿用原 session。它不会自动向外部发消息，后台回复是否同步到当前对话窗口未作保证。


## 2026-09-09 14:37:25 UTC 单次巡检：启动前预检

- 范围与依据：仅 `runs/emilia-en-zh-dynamic-10000h`。已读取本文全部既有记录（最新为启动前核查；本次前没有独立巡检报告）、`manual-observation.json` 和最新且唯一快照 `supervision/20260909T143445Z.json`（14:34:45 UTC）。14:36:30、14:36:43 UTC 复核现场，manual 状态仍为 `active=true, through_step=2000, phase=preflight, owner=main conversation`，训练控制由主会话负责。
- 前后 step：快照 0 → 现场无已记录训练 step（按监控口径仍为 0），新增 0。`training-process.json`、`training-exit.json`、`train.log`、`launcher.log` 均不存在；扫描 `/proc/*/cmdline` 未发现本配置训练或本 run launcher。没有可核对的训练 PID/start_ticks、没有退出码；这是尚未启动，不是退出故障或卡死，没有发送任何信号。
- 最近完整 checkpoint：`checkpoints/` 尚不存在，无 COMPLETE checkpoint。`evaluation/` 尚不存在，speaker_only、icl 均无 summary 或逐句 metrics；两模式 EN/ZH WER/CER、英语规范化实际输出、截断、目标时长/时长比与异常输出均暂不可评估，teacher-forcing val 也尚无结果。未试听音频。`evaluation-tests.log` 中 step=10、target、临时目录样本是测试输出（9 tests / OK），不能算正式训练或正式生成结果。
- 近半小时指标：正式 loss、grad_norm、LR、动态 batch 实际填充、吞吐和数据等待均无日志，无法计算窗口趋势。仅核对既有预检证据：`preflight.json` 的两个 epoch 为 19270 + 19269 = 38539 steps，rank0 规划帧预算填充约 97.9585% / 97.9610%；这些不是运行时测量。`memory-probe.log` 记录四卡两次更新，loss 12.27744 → 10.45749，最高 allocated 44.92156 GiB、reserved 50.05469 GiB；压力测试也不是本 run 训练进度。此次只读取已有验证记录，没有重新执行训练或压力测试。
- 资源现场（14:36:30 UTC）：4 × A100 80GB，各卡显存 4 / 81920 MiB、GPU utilization 0%；compute-apps 查询为空。磁盘 `df -h`：GPFS 使用 46%，可用约 572 TiB；14:36:43 UTC 精确可用 585112.20 GiB。无当前 GPU 竞争或磁盘容量压力证据。
- 判断与处理：保持现有训练代码和配置，包括 batch 6000 帧 / 9000 token、workers=16、prefetch=2、LR 和两 epoch 计划。理由是预检阶段没有实际训练异常证据，且 manual active=true 禁止本巡检启动、停止、重启或修改训练。没有恢复操作；没有完整 checkpoint 时也不会自行从头重跑。
- 未解决事项 / 给主会话的具体建议：`scripts/monitor_training.py:124-125,132-133` 目前仅以 training-exit PID 匹配且 exit_code=0 判为 complete 并终止外层循环，未检查最终 step 38539 COMPLETE checkpoint、最终 val、speaker_only/icl summary 或 include-speaker 冻结权重核验，也未要求本轮巡检成功。这与完成验收要求不一致，可能在验收缺失或巡检失败时停止监督。建议主会话在停止巡检前加入上述最终产物及冻结核验成功条件；本轮不修改脚本，不把进程退出当成完成。启动及前 2000 步观察仍待主会话完成；本轮无需要立即修复的训练故障。
- 执行命令与修改：只读使用 `pwd`、`rg --files`、`cat` / `tail` / `sed` / `nl -ba`（约定、配置、launcher、monitor、快照和预检日志）、`ls -lat`（run 与 supervision）、`date -u`、`git status --short`、`nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv`、`nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv`、`df -h runs/emilia-en-zh-dynamic-10000h`，以及只读 Python 检查文件存在性、manual JSON、`/proc` 进程身份和 `shutil.disk_usage`。最初尝试 supervision/manual-observation.json 不存在，随即按文件列表改读 run 根目录正确文件。唯一人工写入是本段追加；未动已有工作区修改，未运行 timer / Codex / subagent，未提交、推送、发消息或清理文件。本次巡检结束，由既有外层脚本安排下一次。


## 2026-09-09T14:40:29.400155+00:00 主会话：巡检复用与提交前验证

固定巡检 session `01a08697-ba78-7cd3-bb9e-b01eba12d724` 已完成首次实际检查和第二次 `resume`，JSON 事件中的 thread ID 相同，第二次能正确回忆首轮 run/step/前2000步观察目标。证据：run 下 `scheduler-resume.log` 和 `scheduler-resume-result.md`。首轮指出的提前结束巡检问题已修复：必须本轮巡检成功、训练退出码为0且PID匹配，并有核查最终checkpoint、val、两种生成和冻结权重后的 `final-verification.json` passed=true/PID匹配。快照采集异常会记录 snapshot_error 交给巡检处理，避免单次文件/设备读取失败直接终止巡检。

完整单元测试45项通过（`precommit-tests.log`），ICL/Qwen输入协议9项通过（`evaluation-tests.log`）；此前四卡梯度对照及320个tensor精确恢复结果见动态组批文档。正式训练尚未启动。
