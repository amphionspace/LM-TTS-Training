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


## 2026-09-09T14:51:29.763706+00:00 正式启动

用户已同意先训练、待 GitHub 认证恢复后补推送。本地提交 `78cfc6d`，完整 Git bundle 和启动脚本/配置/提示词哈希已保存在 run 下。后台训练 launcher PID=441406，巡检 daemon PID=441407，固定巡检 session `01a08697-ba78-7cd3-bb9e-b01eba12d724`。主会话接管前 2,000 步故障处理，巡检按半小时节奏记录；两 epoch 总预算 38,539 步。启动命令：`python runs/emilia-en-zh-dynamic-10000h/launch.py`；`python scripts/monitor_training.py --run-dir runs/emilia-en-zh-dynamic-10000h`，均脱离当前终端且日志追加到本 run。


## 2026-09-09 14:53:42 UTC 单次巡检：已启动，正在加载训练清单

- 依据与权限：先读取 `training-incident-playbook.md`、本文、run 根目录 `manual-observation.json`、最新快照 `supervision/20260909T145129Z.json`，以及最新已完成巡检 `supervision/20260909T143445Z.md` / `status.json`。现场 14:52:49 UTC manual 仍为 `active=true, through_step=2000, phase=initial-observation`；前 2000 步干预由主会话负责。
- 前后 step：上一轮 0、本轮 14:51:29 快照 0 → 14:52:49 现场仍无已记录训练 step（监控口径为 0）。与上一轮未启动不同，本轮训练已于 14:51:29 UTC 启动；截至现场核验约 79 秒，正在初始化加载，不是长期无进展。
- 进程身份与退出：`training-process.json` PID=441408、start_ticks=`80639545`，与 `/proc/441408/stat` 精确匹配；当前 cmdline 为项目 `.venv/bin/python .venv/bin/torchrun --standalone --nproc_per_node=4 -m qwen3_train.train --config configs/emilia-10kh-pretrain.yaml`。记录的 bash 命令通过 `scripts/run_train.sh` 的 exec 转为 torchrun，PID/start_ticks 不变，身份一致。launcher PID=441406、PPID=1，四个子 rank PID=441631/441632/441633/441634（启动 ticks 分别 80639693/80639693/80639694/80639694），命令均匹配本配置；四 rank 均为 R 状态，torchrun 为 S 等待状态。`training-exit.json` 不存在，train.log / launcher.log 尚为空，未见 traceback；未将无日志解读为训练已完成或失败。
- 启动进展证据：14:52:12 → 14:52:49 UTC，各 rank 的累计 rchar 约 2.11 GB → 4.41–4.43 GB；CPU utime+stime ticks 分别 4128→7793、3990→7655、3999→7663、3992→7657；RSS 约 9577 MiB → 19727–19813 MiB。第二次采样四 rank 的 fd 81 均打开正式 train.jsonl，文件游标约 4.26–4.29 GB。`qwen3_train/train.py` 在模型构建及 initialized 日志前执行 CodeDataset；`qwen3_train/data.py:13` 起逐行读取、计算哈希并解析完整清单。因此当前有明确读取和 CPU 工作进展，判断为训练前清单加载阶段，不是训练前反向、teacher-forcing 验证或生成/codec/ASR 故障；不需要停止或重启。
- 近半小时指标：正式训练仅刚启动且尚未输出首条 train 指标，loss / grad_norm / 实际 LR / 动态 batch 填充 / 吞吐 / data_wait_seconds 均不可用，不能据此计算半小时趋势。现配置仍为两个 epoch / 38539 steps、warmup=1000、backbone_lr=1e-4、lr=3e-4、预算 6000 帧 / 9000 token、workers=16、prefetch=2；这些是配置值，不冒充观测指标。
- GPU / 主机 / 磁盘：14:52:12 与 14:52:49 UTC 四张 A100 均为 761 / 81920 MiB、utilization=0%；compute-apps 仅见上述本 run 四个 rank，各占 752 MiB，无其他 GPU 计算进程。GPU 暂闲与 CPU 清单加载相符。14:52:12 主机内存共 2.0 TiB、已用 72 GiB、available 1.9 TiB，无 swap；14:52:49 磁盘可用 585077.12 GiB（df 约 572 TiB、使用 46%）。没有资源不足证据。
- Checkpoint 与评估：尚无 checkpoints 或 evaluation 产物，无 COMPLETE checkpoint、val 或 speaker_only/icl summary / 逐句 metrics。两模式 EN/ZH WER/CER、英语规范化实际结果、截断、目标时长比及异常输出暂不可评价；尚未到首次 500 步保存评估。未试听音频，也没有用先前测试的虚拟样本替代训练评估。后续仅 8 条诊断样本，不据小样本或早期 ICL 波动调参。
- 判断、操作和结果：保持现有训练代码/配置及训练进程；理由是主会话观察权限仍有效，且两次现场采样证实启动加载持续进展，没有需修复的具体故障。只读执行 `cat` / `ls -lat`（约定、记录、快照、配置、进程身份）、`tail`（train.log）、`sed` / `rg -n`（run_train.sh、train.py、data.py、monitor_training.py）、`git status --short`、`date -u`、两次 `nvidia-smi --query-gpu=... --format=csv`、一次 `nvidia-smi --query-compute-apps=... --format=csv`、`df -h`、`free -h`，以及 Python 读取 `/proc` 的 stat/cmdline/io/fd/fdinfo、manual/退出记录、checkpoint/summary 列表和磁盘空间。唯一写入是追加本段；未发信号、未启动或恢复训练、未新增定时器/Codex/subagent、未提交推送或向外发送消息。
- 未解决事项：本轮尚无法验证首个成功优化步骤及训练质量，由主会话继续观察启动和前 2000 步。上一轮提出的巡检提前结束问题已由主会话修复，并只读确认 monitor 现要求本轮成功、训练退出码0/PID匹配及 `final-verification.json` passed=true/PID匹配。当前训练未完成，不执行最终冻结检查，不写 final-verification.json；最终仍须实际核查 38539 COMPLETE、最终 val、双模式 summary，并将 include-speaker 冻结检查保存在 final-frozen-check.log 后才允许写 passed=true。本次巡检结束。


## 2026-09-09T15:00:13.673392+00:00 首次启动：完整长录音重复解码导致供数阻塞

四rank模型已初始化，64个workers持续读取/解码，GPU保持初始化占用约5GiB且util=0，尚无step日志或checkpoint。定位 sources._decode_emilia_carrier 对每条long短句先解码完整carrier。例如 `emilia2:31b7f5adcb75141d_000_364` 的4.31秒目标来自116377518/44100≈2638.95秒carrier，重复计算放大约612倍。PID/start_ticks/cmdline核验后向torchrun PID=441408发送SIGTERM，保存现场；下一步验证seek+片段解码与原完整解码的样本等价性，再从同一初始化重启。未跳过数据、修改训练预算或绕过checkpoint签名。


## 2026-09-09T15:11:44.201567+00:00 长录音片段读取修复与验证

`decode_emilia_audio` 对 long 短句改用 FFmpeg subfile 的 tar 成员边界，按目标时间向前 seek，解码 1 秒 preroll，再按原采样率的精确样本边界裁剪，最后重采样至 24kHz。避免重复解码完整长录音，也避免把完整压缩 member 读入 BytesIO。独立 short 和数据准备时的整条 carrier 分组解码保持原路径。预算、学习率、数据清单、codec 和样本顺序未改。

AAC 的噪声合成依赖解码器历史，因此随机 seek 与从头解码不逐样本相同；不把它描述为位级等价。FFmpeg 的 [AAC 噪声合成实现](https://github.com/FFmpeg/FFmpeg/blob/master/libavcodec/aac/aacdec_proc_template.c) 在噪声频带迭代 random_state；读取方式参考 [FFmpeg subfile](https://ffmpeg.org/ffmpeg-protocols.html#subfile) 和 [PyAV seek](https://pyav.org/docs/stable/api/container.html#av.container.InputContainer.seek)。已给训练恢复签名加入 `audio_decoder=emilia_native_seek_v1`，防止旧解码语义的 checkpoint 被静默接续；本 run 尚无 checkpoint，修复后从相同初始权重重新启动。

实测23条真实long目标，每条验证目标区间、开头和末尾，重复解码逐值相同。最低波形SNR 48.28dB，冻结ECAPA embedding最低余弦相似度 0.9999989271。23条完整解码累计 44.057s，最终subfile+seek累计 0.578s，逐条加速中位数 71.70倍；这是单条读取测试，不冒充训练吞吐。证据：`seek-subfile-validation.jsonl`、`seek-validation-summary.json`。

新增AAC真实编码回归用例覆盖非零tar offset及成员尾界、目标起止/末尾、重复确定性和单样本时间偏移；原WAV区间仍逐样本对照。最初测试错误地要求噪声较重的合成AAC也达到40dB波形SNR，失败后改为直接核查目标时间对齐及能量，避免把AAC噪声当作错位。source测试7项通过。首次启动PID已退出，四卡显存释放，无残留workers。


2026-09-09T15:13:06.666419+00:00：最终全套46项测试通过（`seek-final-tests.log`）。首次启动的训练日志、PID/退出记录、launcher日志和初始化配置已保存在 `attempt-1-full-carrier-decode/`，无删除；本次无checkpoint可恢复，核查原PID已退出后由主会话从相同初始化重启。


2026-09-09T15:13:06.898087+00:00：修复提交 `f1cf9beaa83eade0502838f9fec7790e4017fabc`，启动第二次正式训练，launcher PID=621409。原半小时巡检daemon PID=441407继续运行，session不变。主会话继续负责前2000步，GitHub推送仍待认证恢复。


2026-09-09T15:19:20.513972+00:00：重启后首次实际训练日志 `{"step": 10, "train": {"first_ce": 9.837877486057463, "residual_ce": 7.829211072943733, "grad_norm": 42.787845611572266, "peak_memory_gib": 53.473777294158936, "audio_seconds_per_second": 787.5562085331356, "step_seconds": 2.3065782230114564, "data_wait_seconds": 0.0002762631047517061, "global_samples": 334.0, "global_audio_frames": 22707.0, "global_talker_tokens": 32874.0, "frame_budget_fill": 0.946125, "token_budget_fill": 0.9131666666666667, "lr_backbone": 1.1e-06, "lr_new": 3.2999999999999993e-06}}`。首个记录步10已完成，CPU供数阻塞解除；这是单步观测，稳态统计继续收集。


## 2026-09-09 15:23:38 UTC 单次巡检：seek 修复后训练持续推进

- 依据与权限：已读取故障处置表、本文、manual、最新快照 `supervision/20260909T152129Z.json`、最新已完成巡检 `supervision/20260909T145129Z.md` / status，并以当前 `/proc`、train.log 和产物为准。15:22:33 UTC manual 仍为 `active=true, through_step=2000, phase=initial-observation-after-seek-fix`；本巡检仅检查、记录和建议，主会话负责干预。
- 前后 step：上轮巡检 0 → 本轮快照（15:21:29 UTC）80 → 15:22:16 现场100 → 15:22:33 现场110。已确认本轮内推进30步；最近半小时包含首次启动供数阻塞、主会话修复和第二次启动，不能把此窗口全部算成稳态训练时间。
- 进程与历史退出：当前 launcher PID=621409（PPID=1），torchrun PID=621410、start_ticks=`80769258` 与 training-process.json 精确匹配；cmdline 为本项目 `.venv/bin/python .venv/bin/torchrun --standalone --nproc_per_node=4 -m qwen3_train.train --config configs/emilia-10kh-pretrain.yaml`，与 run_train.sh 的 exec 启动方式一致。四个 rank PID=621435/621436/621437/621438，启动 ticks 均为80769421，父PID/命令均匹配；均存活，现场 S 状态伴随 GPU 高利用率属正常执行。rank 子进程合计68（含辅助进程，不能全部计为DataLoader workers）；采样状态 S=51、D=12、R=5，当前供数等待很低，不据单次 D 状态判阻塞。当前 training-exit.json 不存在。旧PID441408已不存在；归档 `attempt-1-full-carrier-decode/training-exit.json` 为旧PID exit_code=1、15:00:19 UTC，结合主会话15:00:13记录为其SIGTERM处置结果，不能当成当前训练退出。
- 已有修复证据：主会话记录第二次于15:13:06启动，train.log 的 initialized 明确 progress={step:0, epoch:0, next_batch:0}、world_size=4。只读核查 `seek-validation-summary.json`：23条真实目标、最低SNR48.2757dB、speaker余弦最低0.999998927、重复确定性true；`seek-final-tests.log` 为46项通过。本轮没有重做这些测试，也没有启动训练。解码语义改变及 `audio_decoder=emilia_native_seek_v1` 签名处理详见主会话记录；本巡检未修改签名或旧元数据。
- 训练质量指标：10个已记录点step10–100全部数值有限。first_ce 9.837877→6.873501，residual_ce 7.829211→7.572307；grad_norm 42.787846→2.589349（范围1.714203–42.787846，中位2.340894）。代码显示grad_norm为clip前范数，grad_clip=1；早期较高范数后下降不构成修改LR理由。最终复核step110：first_ce=6.793987、residual_ce=7.560811、grad_norm=2.504368，均有限。日志未见 traceback/OOM/Non-finite；初始化的4条Flash Attention dtype警告保留，未观察到相应运行失败。
- LR及填充：step10–100 backbone LR 1.1e-6→1.01e-5、新参数3.3e-6→3.03e-5；step110为1.11e-5 / 3.33e-5。与warmup=1000及scheduler.step后记录下一次LR的代码口径一致。step10–100帧预算填充94.6125%–99.7125%，中位98.4917%；token填充91.3167%–97.5333%，中位96.1444%；global samples 329–389。step110帧填充99.2958%、token97.5889%。
- 吞吐与等待：step10–100记录点 step_seconds 2.0665–2.4406s，中位2.1750s；全局 audio_seconds_per_second 773.545–913.390，中位859.780；data_wait_seconds 0.0002574–0.0003262s，中位0.0002702s。step110为2.240688s、850.846音频秒/墙钟秒、等待0.0002672s。代码只每10步记录当步指标，以上是记录点统计，不是整个半小时所有步均值；CPU供数不再是当前观测瓶颈，连续推进且无持续吞吐恶化证据。
- 显存与主机：15:22:16 GPU0–3分别67537/77551/75477/77151 MiB（总量各81920），利用率93/94/100/100%；compute-apps仅本run四rank，无其他GPU进程。rank0日志peak allocated最高56.6649GiB，不代表其他rank峰值，也不能把nvidia-smi总占用全视作活跃tensor。当前最高总占用GPU1约75.73GiB，外部可用4369MiB；主会话15:21:18记录GPU2曾占78409MiB。暂无OOM，但首次500步保存/teacher-forcing/双模式生成的实际峰值仍需主会话核对；不能仅据总占用预先降低预算。主机15:22:16内存已用236GiB、available1.7TiB、无swap；15:22:33磁盘可用584927.18GiB（约572TiB，使用46%），空间充足。
- Checkpoint / 双模式评估：checkpoints/latest及checkpoint目录产物仍不存在，尚无COMPLETE；evaluation无产物，val、speaker_only/icl summary及逐句metrics尚未产生，因为未到首次500步。两模式EN/ZH WER/CER、英语规范化实际输出、截断、目标时长/时长比、异常输出均暂不可评价，不记成0、不声称试听。只有8条诊断样本，后续亦不因噪声或早期ICL较差调超参。
- 判断及未解决事项：本轮训练持续推进，主会话修复后的供数已恢复；保持当前配置（预算6000/9000、workers16/prefetch2、LR、两epoch/38539目标）和进程，无修改或恢复。首次500步COMPLETE、val及双模式评估尚待验证；建议主会话在该阶段继续区分各rank allocated/reserved和额外codec/ASR占用，仅出现具体失败时定位根因。当前未完成训练，未执行最终冻结检查、未写final-verification.json。
- 命令与修改：只读 `cat` / `tail` / `head` / `ls -lt` / `sed -n`（约定、快照、当前及归档退出记录、日志、配置和训练指标实现）、`date -u`、`nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv`、`nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv`、`df -h`、`free -h`，以及Python解析train.log计算min/median/max/finite、读取/proc进程身份和父子关系、检查checkpoint/eval及shutil.disk_usage。唯一写入为追加本段；未发信号、未改训练代码/配置、未启动训练/定时器/Codex/subagent、未提交推送发消息或清理。本次巡检结束。


## 2026-09-09 15:55:31 UTC 单次巡检：500步保存与验证完成，ICL评估进行中

- 依据与权限：已读故障处置表、本文、manual、最新快照 `supervision/20260909T155129Z.json` 和最新完成巡检 `supervision/20260909T152129Z.md` / status。15:54:05 UTC manual仍为active=true、through_step=2000、initial-observation-after-seek-fix；主会话继续负责干预。
- 前后step与阶段：前次快照80、上次现场110 → 本轮快照500 → 本轮结束前现场500。近两次快照增加420步；本次step暂不变是500步生成评估。speaker_only summary已完成，ICL从3条metrics推进到4条（15:54:05）；sample-03 WAV于15:53:33新增，之后metrics已写入。GPU持续活动，无长期无进展证据。
- 身份与退出：training-process.json PID621410/start_ticks80769258与/proc匹配，当前cmdline为项目`.venv/bin/python .venv/bin/torchrun --standalone --nproc_per_node=4 -m qwen3_train.train --config configs/emilia-10kh-pretrain.yaml`，符合run_train.sh的exec。launcher621409（PPID1），四rank621435/621436/621437/621438（start_ticks均80769421、PPID621410）命令正确且存活；现场四rank为R，launcher/torchrun为S。当前退出记录不存在，日志未见Traceback/OOM/Non-finite/Error/Aborted；未发送信号。
- 半小时训练统计：按上次快照step80之后的42个记录点step90–500统计，每10步只记录当步，以下不是全窗口所有更新的均值。first_ce 6.977822→5.196846，residual_ce 7.565062→7.326595；所有记录指标有限。grad_norm范围1.230105–9.085606、中位2.546858，step500为9.085606（clip前）；单次偏高不改LR。backbone LR 9.1e-6→5.01e-5、新参数2.73e-5→1.503e-4，符合warmup；预算仍6000帧/9000token、workers16/prefetch2、总38539步。
- 填充/吞吐/等待：帧填充94.8833%–99.7125%，中位98.4146%；token填充91.7139%–97.9472%，中位95.6583%；global samples318–383。step_seconds范围1.9985–2.3539s、中位2.1568s；全局音频秒/墙钟秒807.173–918.324、中位868.581；data_wait_seconds 0.0002516–0.0004611s、中位0.0002721s。step500为2.150868s、851.154音频秒/秒、等待0.0002838s。与上一轮中位2.1750s/等待0.0002702s相近，未见供数或训练吞吐持续恶化；本窗口包含保存/验证/逐句生成耗时，不将420/1800当作纯训练吞吐。
- 最近完整checkpoint：latest指向`checkpoints/step-00000500`；COMPLETE mtime为15:37:23.964950 UTC，metadata progress={step:500, epoch:0, next_batch:500}，world_size4，scheduler.last_epoch500/_step_count501/_last_lr与日志一致。四个distributed分片均存在且每片约2.093GB，`.metadata`1424887字节、rng-0..3各14613字节；检查标记/元数据/文件存在性，不声称本轮做过优化器恢复加载。signature包含`audio_decoder=emilia_native_seek_v1`、原train/val哈希、预算/模型/eval及schedule38539，未改任何元数据。
- 验证及已有冻结检查：step500 val first_ce=5.207924、residual_ce=7.292686；15个残差codebook CE均有限（5.823551–7.509618）。`frozen-step-500.log`已有`status=passed, exact_frozen_tensors=81, include_speaker=true, parameters=326313792`，本轮只读取结果，未重复运行。这仅是500步冻结检查，不替代最终38539步验收。
- speaker_only完整summary（8条=EN4/ZH4）：EN基础WER=1.0000/CER=0.818182，英语规范化后WER=1.055556/CER=0.818182；ZH WER=1.0000/CER=1.107143；合计WER=1.0000/CER=0.925110，截断2/8=25%。已逐条核对metrics与日志；目录编号00/01/02/03/04/05/06/12来自验证集索引及语言配额选择，仍是8条，不是13条或漏掉7–11。

| speaker_only样本目录（step-00000500下） | 语言 | 输出时长 / 目标时长比 | 基础WER / CER | 具体诊断 |
|---|---|---|---|---|
| sample-00 | EN | 2.00s / 1.105 | 1.50 / 1.167 | 目标“The the liquid spears.”，ASR为“and he'll literally lose the eyes.”；英语规范化展开he'll，WER变1.75 |
| sample-01 | ZH | 1.44s / 1.469 | 1.00 / 1.00 | “煮至奶白色后”被识别为“再見啦” |
| sample-02 | EN | 32.00s / 6.667 | 1.00 / 0.735 | 400帧未EOS、截断；ASR反复“hey”，目标约4.8s |
| sample-03 | ZH | 32.00s / 5.096 | 1.00 / 1.00 | 400帧未EOS、截断；长目标仅识别为“因此”，目标约6.28s |
| sample-04 | ZH | 2.48s / 1.078 | 1.00 / 1.692 | ASR反复“接身上讓你” |
| sample-05 | EN | 1.36s / 0.840 | 1.25 / 1.385 | “W.A. Flinders Uni.”被识别为“We're going to follow you.”；英语规范化WER1.50/CER1.308 |
| sample-06 | EN | 4.16s / 1.169 | 0.75 / 0.636 | Jeffries tube等内容替换，未截断 |
| sample-12 | ZH | 3.76s / 0.712 | 1.00 / 1.00 | 目标含video，被识别为“行行行行” |

- 英语/ASR口径：summary.by_language.en为基础评分，summary.english才是额外英语规范化评分，不能混用。代码对生成与目标音频均评分；`reference_asr`在此指目标真值音频的ASR，而不是speaker参考句。由8条metrics复算目标音频EN基础WER0.055556/CER0.048951，ZH WER1.375/CER0.202381；中文空格分词WER解释性有限，重点看CER，目标音频也有ASR误差。以上输出诊断依据ASR文本，不声称听过音频。
- ICL进行中（15:54:05，仅4/8，summary尚不存在）：sample-00（EN）0帧立即EOS，基础/英语规范化WER/CER均1，未写WAV、未给duration字段；与代码空生成分支一致，不是音频文件丢失。sample-01（ZH）400帧/32s/时长比32.653，目标约0.98s，未EOS截断且ASR为空；sample-02（EN）33帧/2.640042s/比0.550，ASR“This is just my act for a while.”，基础及英语规范化WER0.875/CER0.794118；sample-03（ZH）400帧/32s/比5.096，截断且ASR为空。仅这4条临时复算EN2条WER0.9000/CER0.837209、ZH2条WER1/CER1，不能当8条最终summary或与speaker_only全量直接比较；已完成4条中2条截断。两模式相同目标的speaker_reference_id一致，ICL参考帧sample00/01=36、sample02=65。代码先解码参考+目标再裁剪参考波形；读取到的speaker_only全部8个WAV和当时ICL两个非空WAV头均为24kHz单声道，时长匹配metrics，未试听。
- GPU/磁盘：15:52:54四卡占用81125/77997/75925/77437 MiB（各81920），利用率73/31/57/57%；compute-apps仅本run四rank。GPU0外部可用795MiB，属于显存余量较小，尚无OOM；rank0训练日志peak allocated最高56.7952GiB不能代表生成阶段或各rank总占用，不把nvidia-smi数字等同活跃tensor。主机内存已用237GiB/available1.7TiB，无swap；15:53:41磁盘可用584852.13GiB、使用46%，可容纳后续checkpoint。
- 判断、保持配置原因与未解决事项：当前训练/保存/val已成功，正在正常产出ICL；早期生成有明显内容错误、重复、空输出和超长，8条诊断样本及首次500步不足以据此改变LR/预算/训练目标。保持全部现配置及进程；主会话应继续完成ICL剩余4条及summary、确认返回训练后loss/step推进，并在下次相同目标评估跟踪上述异常和GPU0余量。如出现实际OOM再定位生成/codec/ASR阶段的峰值与引用，不能仅凭显存总占用改签名覆盖的配置。本run远未达38539；未执行最终冻结检查，未写final-verification.json。
- 命令/修改/结果：只读cat/tail/head读取约定、manual、快照、最新巡检、训练与冻结日志；rg --files列举checkpoint/eval，rg -n及sed -n核对train.py评分、ICL裁剪、样本选择和summary实现；date -u、nvidia-smi GPU及compute-apps查询、df -h、free -h；Python读取/proc身份、解析完整train.log统计finite/min/median/max、读checkpoint metadata/分片大小/时间、读全部已有metrics与summary并复算分语言错误率，wave只读检查WAV头。唯一写入是本段追加；未修改训练代码/配置、未启动停止恢复训练、未运行定时器/Codex/subagent，未提交推送发消息或清理。本次巡检结束。


## 2026-09-09T15:56:46.249553+00:00 主会话：第500步完整检查

完整checkpoint约7.80GiB，COMPLETE/world_size=4/step=500/epoch=0/next_batch=500/解码器版本/两清单哈希均核对通过。冻结前端与speaker encoder共81个张量、326313792元素，与初始化逐值一致（frozen-step-500.log）。验证first CE=5.207924，residual CE=7.292686。

- speaker_only：8条（4EN/4ZH），空输出0，截断2；英语规范化WER=1.055556，中文CER=1.107143。
- icl：8条（4EN/4ZH），空输出2，截断2；英语规范化WER=0.944444，中文CER=1.166667。

已核对两模式使用同一组目标/参考，ICL参考文本与codec帧数存在，非空生成WAV时长与metrics一致；空输出没有伪造空WAV。无OOM、非有限值或读取/保存错误。模型内容和结束行为仍差，保持当前训练目标和学习率，比较后续1000/1500/2000检查点，不用8条早期样本贸然调参。详细逐条信息见milestone-500.json和两mode评估目录。


## 2026-09-09 16:24:27 UTC 单次巡检：1000步完整保存，speaker_only改善，ICL评估进行中

- 依据与权限：先读故障处置表、本文、manual、最新快照 `supervision/20260909T162129Z.json`、最新完成巡检 `supervision/20260909T155129Z.md` / status，并核对当前/proc及产物。16:22:52 UTC manual仍active=true、through_step=2000、initial-observation-after-seek-fix；继续由主会话干预。
- 前后step：上一轮快照/现场500 → 本轮快照1000 → 16:22:24现场1000；半小时新增500更新，新增50个train日志点。500步ICL已全部完成且随后实际训练至1000，上一轮“待确认返回训练”事项已解决；当前1000步停留为下一轮评估。16:21:26 speaker_only最后样本完成、16:21:56 ICL sample00 metrics完成、16:22:47 ICL sample01 WAV新增，GPU持续活动，不判卡死。
- 进程与退出：training-process.json PID621410/start_ticks80769258与/proc精确匹配；cmdline为项目`.venv/bin/python .venv/bin/torchrun --standalone --nproc_per_node=4 -m qwen3_train.train --config configs/emilia-10kh-pretrain.yaml`。launcher621409/PPID1，四rank621435–621438/PPID621410/start_ticks80769421，命令均正确且存活；四rank R、launcher及torchrun S。training-exit.json不存在，扫描当前完整train.log未见Traceback/OOM/Non-finite/Error/Aborted；无停止或恢复操作。
- 半小时训练指标：step510–1000共50个记录点（每10步的当步指标，不是500步均值），first_ce 5.113428→2.240499、residual_ce 7.326219→7.125836；全部指标有限。grad_norm范围1.322671–8.967528、中位2.305302、末值1.409376（clip前）。LR主干5.11e-5→1e-4、新参数1.533e-4→3e-4，1000步正常到达warmup峰值；990→1000的first_ce小幅上浮不构成调整依据。
- 填充与吞吐：帧填充95.5333%–99.7583%，中位98.3375%；token填充92.4167%–97.9250%，中位95.9111%；global samples330–415。step_seconds 2.0361–2.3510s，中位2.1812s（上一窗口2.1568s）；音频秒/墙钟秒810.368–937.020，中位864.252（上一窗口868.581）；data_wait_seconds 0.0001709–0.0004916s，中位0.0002790s。无持续吞吐恶化或供数阻塞证据。rank0训练peak allocated最高57.0089GiB；不代表生成阶段或其他rank峰值。
- 最近完整checkpoint：latest=`step-00001000`，COMPLETE于16:13:54.630272 UTC生成，metadata progress={step:1000, epoch:0, next_batch:1000}、world_size4、scheduler.last_epoch1000、LR[0.0001,0.0003]。四distributed分片各约2.093GB、`.metadata`1424887字节、四rng文件各14613字节均存在；同时保留500步COMPLETE，符合keep_checkpoints=2。本轮实比500/1000 signature完全相同（含解码版本emilia_native_seek_v1及38539 schedule）；未编辑元数据，未做恢复加载，文件检查不冒充优化器恢复验证。
- Teacher-forcing：val first_ce从500步5.207924降至1000步2.304850，residual_ce从7.292686降至7.096445；1000步15个残差codebook CE均有限（5.077887–7.503183）。500步已有include-speaker冻结检查通过，当前仅见frozen-step-500.log；本轮未重新运行，也不将其当最终冻结验收。

| 生成评估（每份完整summary为EN4/ZH4） | EN基础WER / CER | EN英语规范化WER / CER | ZH WER / CER | 截断 / 空输出 |
|---|---|---|---|---|
| speaker_only step500 | 1.000000 / 0.818182 | 1.055556 / 0.818182 | 1.000000 / 1.107143 | 2/8 / 0/8 |
| speaker_only step1000 | 0.444444 / 0.307692 | 0.444444 / 0.307692 | 1.375000 / 0.785714 | 1/8 / 0/8 |
| icl step500（本轮核查完整结果） | 0.944444 / 0.853147 | 0.944444 / 0.853147 | 1.500000 / 1.166667 | 2/8 / 2/8 |

- 1000步speaker_only逐句核查：sample00 EN时长1.44s/目标比0.796，ASR“the liquid spars.”，WER0.50/CER0.2222；sample02 EN由500步32s截断变为4.32s/比0.9且正常EOS，WER0.1875/CER0.1471（McLaughlin被识别为Michael Acklin）。sample03 ZH仍400帧/32s/比5.096截断，CER0.9167，ASR“又是虚老丹产的早逝”；这是应继续追踪的固定目标。sample01 ZH比1.143/CER0.8333，ASR“組織大白絲後”；sample04 ZH比0.870/CER0.5385；sample05 EN比1.481/WER1.25/CER0.9231，ASR“Weigh a flounder, you and me.”；sample06 EN比0.966/WER0.50/CER0.4091；sample12 ZH比0.712/CER0.7241，末尾ASR重复“二位都有”。8个非空WAV头均为24kHz单声道，时长与metrics一致；未试听，以上都是ASR文本诊断。
- 评分与配对：逐条比较已出现的两模式/500与1000目标ID、文本、speaker_reference_id，均一致。1000步EN4条同时有english_content，基础与英语规范化汇总恰好相同，不是未运行规范化。speaker_only目标真值音频ASR在500/1000一致：EN WER0.055556/CER0.048951，ZH WER1.375/CER0.202381。中文WER受空格分词影响，ZH WER上升但CER下降不能简单描述为整体退化；繁简等ASR表示差异也影响CER。不将reference_asr误认为speaker参考句评分。
- ICL500补齐上一轮剩余结果：sample04 ZH为3.52s/比1.530，ASR反复“神秘”及“啊”、CER2.0769；sample05 EN仅2帧/0.16s/比0.099却ASR“Thank you for watching!”，可能有短音频ASR幻觉，不能据文本认定真的说出这句话；sample06 EN2.96s/比0.831、WER1/CER0.7727；sample12 ZH立即EOS0帧，与sample00 EN合计两条空输出，无WAV符合空生成分支。两个400帧截断仍是ZH sample01与03。该完整8条评估质量差，但仅早期诊断样本，不能据此增加ICL训练目标或调LR。
- ICL1000截至本段追加时已有3/8条metrics，完整summary存在=False；先前16:22:52只见1条metrics和新增sample01 WAV，已确认文件推进。最新已写metrics如下（不足8条时不伪造完整summary，不与完整8条直接比较）：
  - sample-00，emilia2:8e38c4aea5a42b10_847_000（en）：18帧，truncated=False，时长=1.44s，比=0.7955801104972375；WER=0.75、CER=0.6111111111111112，英语规范化WER/CER=0.75/0.6111111111111112；ASR='but the leaking sprayers.'。
  - sample-01，emilia2:1ac48948a9178d19_007_005（zh）：154帧，truncated=False，时长=12.32s，比=12.571428571428571；WER=1.0、CER=1.0，英语规范化WER/CER=None/None；ASR='这是哪一代财后'。
  - sample-02，emilia2:9f840426d205a7a6_446_000（en）：46帧，truncated=False，时长=3.68s，比=0.7666666666666667；WER=0.1875、CER=0.1323529411764706，英语规范化WER/CER=0.1875/0.1323529411764706；ASR='This is a Mechaloft win group in which like there are winners and losers yet, you know.'。
- 资源现场（16:22:24 UTC）：GPU0–3占用81125/77997/75925/77437 MiB，各总量81920，利用率58/30/65/75%；compute-apps仅本run四rank，GPU0外部余量795MiB，仍需关注但无OOM。此总占用与上一轮评估相同，不等同活跃tensor。主机内存已用237GiB、available1.7TiB、无swap；16:22:52磁盘可用584742.67GiB（df约572TiB，使用46%），足以容纳后续checkpoint。
- 判断、结果与未解决事项：保持当前代码/配置/进程（预算6000/9000、workers16/prefetch2、既定LR和两epoch38539），未执行干预；原因是训练loss/val改善、吞吐稳定，speaker_only在同组诊断样本上也改善，当前正常执行ICL。主会话继续核查1000步ICL完整summary及评估后step推进，关注持续截断的ZH sample03、空输出是否减少和GPU0显存余量；不因8条噪声/早期ICL质量差改超参。训练尚未完成，不运行最终冻结检查、不写final-verification.json。
- 执行命令与修改：只读cat/tail读取处置约定、巡检、manual及status；Python解析指定完整快照、train.log（50点统计/finite/val/错误）、两checkpoint metadata及分片大小、全部已有两模式metrics/summary，比较配对与signature，wave读取非空WAV头；读取/proc/stat/cmdline核对PID/启动ticks/父子关系；date -u、nvidia-smi GPU和compute-apps查询、df -h、free -h、shutil.disk_usage。唯一人工写入为追加本段；未发信号、改训练代码/配置、启动停止恢复训练，未新增定时器/Codex/subagent，未提交推送发消息或删除清理。本次巡检结束。


## 2026-09-09T16:30:48.723167+00:00 主会话：第1000步完整检查

第1000步COMPLETE、四个distributed分片和四rank RNG文件齐全，约7.80GiB；world_size=4、progress={step:1000,epoch:0,next_batch:1000}，signature与500步完整相同。验证first CE=2.304850、residual CE=7.096445，均较500步下降；warmup结束时backbone/new LR为1e-4/3e-4，1010步按cosine计划连续下降。

- speaker_only：8条（4EN/4ZH），空输出0、截断1；英语规范化WER=0.444444，中文CER=0.785714。
- icl：相同8条目标/参考，空输出0、截断1；英语规范化WER=0.388889，中文CER=0.892857。短中文sample-01生成154帧约12.32秒，时长仍明显过长；两模式长中文sample-03仍400帧截断。

已逐项检查metrics、两模式参考ID和ICL reference_frames，并读取全部16个WAV头核对24kHz单声道和时长；未声称试听。证据milestone-1000.json及evaluation目录。1010步first/residual CE=2.205299/7.136432，grad_norm=1.112640，step_seconds=2.332144、data_wait=0.000256s，无OOM/非有限值/退出。保持配置和进程，因为训练/评估成功且整体改善，8条诊断仍不足以支持调参。16:21轮独立巡检成功、session不变。另重试普通git push origin main仍因GitHub SSH publickey认证失败，未推送、未改远端配置。后续继续观察1500和2000步。


## 2026-09-09 16:54:33 UTC 单次巡检：1500步完整保存，正在生成评估

- 依据与权限：已读故障处置表、本文、manual、最新快照`supervision/20260909T165129Z.json`、最新完成巡检`supervision/20260909T162129Z.md`及status；现场16:53:02 UTC manual仍active=true、through_step=2000、initial-observation-after-seek-fix，主会话负责干预。
- 前后step与阶段：上一轮快照/现场1000 → 本轮快照1500 → 16:52:32现场1500，半小时新增500更新。1000步两模式评估已结束，之后训练确实连续推进；1500步当前执行speaker_only，尚未到ICL。快照最后日志16:49:43，现场新增sample03 WAV（16:51:53）/metrics（16:52:23），随后sample04 metrics出现；当前共5/8条，GPU持续工作，step不变不是卡死。
- 进程身份与退出：training-process.json PID621410/start_ticks80769258与/proc精确一致；cmdline是本项目`.venv/bin/python .venv/bin/torchrun --standalone --nproc_per_node=4 -m qwen3_train.train --config configs/emilia-10kh-pretrain.yaml`。launcher621409/PPID1、四rank621435–621438/PPID621410/start_ticks80769421均匹配且存活；rank为R，torchrun/launcher为S。当前training-exit.json不存在；完整train.log未见Traceback/OOM/Non-finite/Error/Aborted。本轮未发信号。
- 近半小时训练：step1010–1500共50个每10步记录的当步指标全部有限；first_ce 2.205299→1.873166（范围1.835288–2.205299），residual_ce 7.136432→6.874695。grad_norm范围0.830500–1.425752、中位1.109562、末值1.157482（clip前）。主干LR从9.999998424e-5降至9.996060932e-5、新参数从2.999999527e-4降至2.998818280e-4，与warmup后的cosine连续接续一致，无需改LR。
- 填充/吞吐/供数：帧填充94.0458%–99.8167%，中位98.3292%；token填充92.2444%–97.6722%，中位95.7125%；global samples293–396。step_seconds范围1.9759–2.3714s，中位2.14485s；全局音频秒/墙钟秒805.123–951.159，中位879.254；data_wait_seconds范围0.0002358–0.0004669s，中位0.0002725s。与上一窗口中位2.18121s/864.252音频秒/秒/0.0002790s相近，无持续恶化；这些是记录点统计，不是包含评估停留的整个30分钟平均。rank0训练peak allocated最高56.8660GiB，不冒充其他rank或生成峰值。
- Checkpoint：latest指向step-00001500，COMPLETE于16:47:24.532330 UTC生成；progress={step:1500,epoch:0,next_batch:1500}、world_size4、scheduler.last_epoch1500、LR与日志一致。四distributed分片各约2.093GB，`.metadata`1424887字节及四rng文件各14613字节均存在。当前保留1000和1500两个COMPLETE，二者signature逐项相等；500已按既有轮转消失，本巡检未清理。核查文件和metadata，不声称执行过优化器恢复加载；未改签名或元数据。
- Teacher-forcing：1000→1500 val first_ce 2.304850→1.927205，residual_ce 7.096445→6.861559；15个残差codebook CE均有限，1500步范围4.579915–7.442406。训练及验证均改善。

| 最新完整生成summary（均step1000、EN4/ZH4） | EN基础WER / CER | EN英语规范化WER / CER | ZH WER / CER | 截断 / 空输出 |
|---|---|---|---|---|
| speaker_only | 0.444444 / 0.307692 | 0.444444 / 0.307692 | 1.375000 / 0.785714 | 1/8 / 0/8 |
| icl | 0.388889 / 0.293706 | 0.388889 / 0.293706 | 1.125000 / 0.892857 | 1/8 / 0/8 |

- 补齐1000步ICL：本轮实读summary及全部8条metrics，非空WAV均24kHz单声道、时长匹配。相比500步英语规范化WER0.944444、ZH CER1.166667，均改善，空输出2→0、截断2→1；不是全验证集质量结论。仍有ZH sample01 154帧/12.32s/目标比12.571、CER1.0；ZH sample03 400帧/32s/比5.096、ASR空、CER1.0。其余新补齐的sample04 ZH1.68s/比0.730/CER0.8462，sample05 EN2s/比1.235/WER0.75，sample06 EN3.36s/比0.944/WER0.4167，sample12 ZH3.84s/比0.727/CER0.7586，均正常EOS。英语规范化字段实际存在，汇总与基础值相同；中文WER受分词影响，重点观察CER。
- 1500步部分输出（最后现场5/8，无speaker_only summary，ICL尚无产物）：sample00 EN24帧/1.92s/比1.061、WER0.50/CER0.6111，ASR“The leak creates spears.”；sample01 ZH13帧/1.04s/比1.061、WER1/CER0.6667，ASR“坐著那白色喉”；sample02 EN50帧/4.00s/比0.833、WER0.125/CER0.1029，McLaughlin仍有专名替换；sample03 ZH400帧/32s/比5.096，未EOS截断，WER1/CER0.75，ASR“预期了半项锁,现在是下午两点午时。”；sample04 ZH29帧/2.32s/比1.009、WER1/CER0.5385，ASR“習氏人家畢竟見識了多麼多年”。已写5条中1条截断、0空输出，仅为部分结果，不外推8条汇总；EN已完成两条英语规范化与基础WER/CER相同。已有目标ID/文本/参考ID与1000步匹配。
- 持续截断目标核查：sample03 ID=`emilia2:205e83c292110537_086_000`，val清单duration=6.28s、num_frames=79、text_ids长度25；对应缓存目标WAV实读为6.280s、24kHz单声道、FLOAT。其目标长度远低于400帧生成上限，不能用“目标本来超过上限”解释连续500/1000/1500步speaker_only截断；当前配对/时长未见不一致。建议主会话在1500双模式及2000步继续对照该目标的EOS与后半句输出；若持续出现再做针对性复现，不靠提高max_frames或仅凭这一条改训练超参。ASR结果不等于听感，本轮未试听音频。
- 资源（16:52:32 UTC）：GPU0–3占用81125/77997/75925/77439 MiB（总81920），利用率71/26/51/44%；compute-apps仅本run四rank，无其他GPU任务。GPU0外部余量795MiB，与近期评估一致、尚无OOM，不能把总占用当活跃tensor。主机内存已用237GiB/available1.7TiB、无swap；磁盘可用584631.87GiB，df约571TiB、使用46%，足以继续保存。
- 判断与未解决事项：保持现配置/进程（预算6000帧/9000token、workers16/prefetch2、既定LR与两epoch38539），理由是训练/验证改善、吞吐稳定、评估产物推进；8条早期诊断及目标个别异常不足以支持调参。1500步speaker_only剩余样本、两模式完整summary、评估后训练推进仍待主会话观察，GPU0余量和sample03持续截断继续跟踪；本run未完成，不执行最终冻结检查、不写final-verification.json。
- 执行命令/修改：cat/tail读取约定、manual、快照、最新记录/status；Python解析train.log统计finite/min/median/max/错误、读两checkpoint文件大小和metadata、比较signature/样本配对、读summary与逐句metrics、读取/proc/stat和cmdline；date -u、nvidia-smi GPU及compute-apps、df -h、free -h、shutil.disk_usage。生成WAV用wave读取头；目标真值WAV首次用系统wave遇到`wave.Error: unknown format: 3`（巡检工具不支持FLOAT WAV，非训练报错），立即改用`.venv/bin/python`的soundfile.info成功核验，无数据修改。唯一人工写入为追加本段；未启动停止恢复训练、未改训练代码/配置、未新增timer/Codex/subagent、未提交推送发消息或清理。本次巡检结束。


## 2026-09-09T17:03:54.629920+00:00 主会话：第1500步完整检查

第1500步COMPLETE及四rank分片/RNG齐全，约7.80GiB，progress={step:1500,epoch:0,next_batch:1500}，world_size=4，signature与1000步相同。训练器按keep_checkpoints=2保留1000/1500；主会话未清理文件。验证first CE=1.927205、residual CE=6.861559，15个残差codebook均有限。1010–1500的50个记录点全部指标有限，step_seconds中位2.144848，data_wait中位0.000273s，音频吞吐中位879.254秒/秒，帧预算填充中位98.3292%，grad_norm范围0.8305–1.4258。以上是每10步记录点统计，不是所有步均值。

- speaker_only：8条、空输出0、截断1；英语规范化WER=0.333333、中文CER=0.678571。
- icl：8条、空输出0、截断1；英语规范化WER=0.277778、中文CER=0.797619。上轮154帧的短中文sample-01这轮14帧，时长改善；长中文sample-03两模式仍400帧截断，ASR内容仍不完整。

Python读取summary/逐句metrics并核对固定目标/参考、ICL reference_frames，wave核对16个WAV采样率/声道/时长，结果均通过；未试听。详细证据milestone-1500.json。1520步已恢复训练，first/residual CE=1.855447/6.879958，step_seconds=2.194998，无OOM/NaN/退出。16:51轮独立巡检成功且session不变。保持预算、LR、目标和进程，因为训练/验证/生成完整、整体继续改善；长中文样本的持续截断继续在2000及后续相同评估跟踪，不能把该诊断集当全验证质量。主会话继续观察到2000步评估完成及训练恢复。


## 2026-09-09T17:23:15.054603+00:00 主会话：第2000步冻结检查与巡检命令修正

第2000步COMPLETE已生成、progress/world_size/signature与计划和1500步匹配，验证first CE=1.779803、residual CE=6.717015。首次直接执行`.venv/bin/python scripts/check_frozen_frontend.py ... --include-speaker`因项目根目录未加入模块路径报ModuleNotFoundError，未触及训练。按现有README/验证命令约定加入`PYTHONPATH=.`补跑成功，81个冻结张量、326313792个元素逐值相同；原失败和补跑成功输出均保留在frozen-step-2000.log，补跑进程退出码0。只修正run下supervision-prompt.md的最终冻结检查调用方式，前后SHA256保存在supervision-prompt-frozen-command-fix.json；巡检每轮重新读取提示词，无须重启daemon，训练代码/配置/进程未改。当前2000步双模式生成尚未完成，manual仍active=true，不将此阶段检查当作38539步最终验收。


## 2026-09-09 17:24:32 UTC 单次巡检：2000步checkpoint与val完成，主会话仍在观察评估

- 依据与权限：已读故障处置表、本文、manual、最新快照`supervision/20260909T172129Z.json`、最新完成巡检`supervision/20260909T165129Z.md`及status。17:23:03 UTC manual仍active=true、through_step=2000、initial-observation-after-seek-fix；不能仅因step到2000就自动视为交接，本轮继续只检查/记录，由主会话完成双模式评估及恢复训练观察。
- 前后step/阶段：上次快照及现场1500 → 本次快照2000 → 17:22:35现场2000，新增500次更新/50个日志点。1500步评估已完整结束并继续训练；当前2000步执行speaker_only，快照后新增sample01 metrics（17:21:48）和sample02 metrics（17:22:31），当前3/8，ICL尚未开始。GPU持续工作，step暂停属于评估，不是卡死。
- 进程与退出：training-process.json PID621410/start_ticks80769258和/proc精确匹配，cmdline为项目`.venv/bin/python .venv/bin/torchrun --standalone --nproc_per_node=4 -m qwen3_train.train --config configs/emilia-10kh-pretrain.yaml`。launcher621409/PPID1、四rank621435–621438/PPID621410/start_ticks80769421均正确且存活，rank R、launcher/torchrun S。当前training-exit.json不存在；完整train.log无Traceback/OOM/Non-finite/Error/Aborted，全部已记录训练指标有限。未发信号、未恢复或重启。
- 近半小时指标（step1510–2000，50个每10步记录的当步点）：first_ce 1.937868→1.757516，范围1.623174–1.937868；residual_ce 6.880911→6.725744；grad_norm范围0.725152–1.558834、中位0.923146、末值0.938135（clip前）。主干LR9.995901818e-5→9.984250623e-5、新参数2.998770545e-4→2.995275187e-4，按cosine连续下降；不因1990→2000单点loss变化调LR。
- 填充/吞吐/等待：帧填充94.5667%–99.6917%，中位98.2083%；token填充90.7639%–97.8139%，中位95.3097%；global samples298–414。step_seconds范围1.97130–2.27391s，中位2.13477s；音频秒/墙钟秒831.224–933.409，中位883.110；data_wait_seconds 0.0002399–0.0004661s，中位0.0002764s。相比上一窗口中位2.14485s、879.254音频秒/秒、等待0.0002725s稳定；不是包含评估耗时的全窗口平均。rank0训练peak allocated最高56.7576GiB，不代表生成/其他rank总占用。
- 最近完整checkpoint：latest=step-00002000，COMPLETE时间17:20:09.511173 UTC；metadata progress={step:2000,epoch:0,next_batch:2000}、world_size4、scheduler.last_epoch2000及LR与日志一致。四distributed分片各约2.093GB、`.metadata`1424887字节和四rng文件各14613字节均存在；当前仅保留1500/2000 COMPLETE，两者signature完全一致。1000已按既有keep_checkpoints=2轮转，本巡检未清理。仅查结构/元数据，不声称恢复加载了优化器，未编辑签名或metadata。
- 验证：1500→2000 val first_ce 1.927205→1.779803，residual_ce 6.861559→6.717015，15个残差codebook CE均有限（2000范围4.402321–7.383985），训练/val继续改善。
- 已有2000步冻结检查：读取`frozen-step-2000.log`，其先保留工具导入失败，完整报错为：

```text
Traceback (most recent call last):
  File "/119010446/LM-TTS-Training/scripts/check_frozen_frontend.py", line 7, in <module>
    from qwen3_train.assembly import load_prefix
ModuleNotFoundError: No module named 'qwen3_train'
```

随后同一日志已有单进程checkpoint加载warning及`{"status":"passed","exact_frozen_tensors":81,"include_speaker":true,"parameters":326313792}`。这是主会话独立检查工具先失败后成功的记录，非训练rank故障；本轮未重复执行，且未根据日志推断未记录的修复命令。2000步通过记录不替代最终38539步检查，最终工具调用仍需确保项目包可导入并保留完整日志。

| 最新完整生成summary（均step1500，EN4/ZH4） | EN基础WER / CER | EN英语规范化WER / CER | ZH WER / CER | 截断 / 空输出 |
|---|---|---|---|---|
| speaker_only | 0.333333 / 0.237762 | 0.333333 / 0.237762 | 1.375000 / 0.678571 | 1/8 / 0/8 |
| icl | 0.277778 / 0.153846 | 0.277778 / 0.153846 | 1.375000 / 0.797619 | 1/8 / 0/8 |

- 两模式1500步完整质量核查：实读summary及8条metrics，EN均有english_content（数值与基础评分相同），与1000步相比EN WER/CER及ZH CER整体下降；ZH WER有分词影响，不作单独质量结论。固定目标/文本/参考ID两模式及已出现2000样本一致。speaker_only新补齐sample05 EN2.08s/比1.284、WER1/CER0.5385，ASR“Davia Flanders, UAE.”；sample06 EN3.36s/比0.944、WER0.3333/CER0.2045；sample12 ZH4.24s/比0.803、CER0.6552，video被识别为B6。ICL短ZH sample01从1000步12.32s缩至1.12s/比1.143、CER0.8333；长ZH sample03仍32s/比5.096/400帧截断，ASR空、CER1。ICL sample04 ZH2.4s/比1.043、CER0.2308，sample05 EN2.4s/比1.481、WER1.25，sample06 EN4.08s/比1.146、WER0.1667，sample12 ZH4s/比0.758、CER0.7931且ASR含重复“弟弟”。本輪读到的1500 ICL全部8个WAV头为24kHz单声道、时长匹配；未试听，不把ASR文本当实际听感。
- 2000步部分生成（17:23:03仍speaker_only3/8，双模式summary均未出现）：sample00 EN22帧/1.76s/比0.972，基础/英语规范化WER0.50、CER0.2778，ASR“Now the liquid spares.”；sample01 ZH15帧/1.20s/比1.224、WER1/CER0.8333，ASR“入卷待北索后”；sample02 EN54帧/4.32s/比0.9、基础/英语规范化WER/CER均0，ASR完整匹配目标。三条均EOS、未截断、非空WAV为24kHz单声道且时长匹配。ICL未产生本步metrics，不以缺失记0；不把部分3条与完整8条汇总直接比较。持续截断的长ZH sample03本步结果尚待产生；上一轮核验的目标6.28s/79帧远低于400帧上限，不贸然增大上限。
- 资源（17:22:35 UTC）：GPU0–3占用81125/77997/75925/77439 MiB（每卡总81920），利用率58/29/41/58%；compute-apps仅本run四rank。GPU0外部余量795MiB，与近期评估一致、无OOM；不能把nvidia-smi总占用等同活跃tensor。主机内存已用238GiB、available1.7TiB、无swap；磁盘可用584440.34GiB（df约571TiB，使用46%），足够继续保存。
- 判断、处理与未解决事项：保持现配置和进程（预算6000/9000、workers16/prefetch2、既定LR与两epoch38539），因loss/val改善、吞吐稳定、评估产物推进且无具体运行故障；仅8条诊断及早期ICL偏差不能支持改超参。主会话仍需完成2000步两种summary、核对长ZH输出及评估后step继续，然后由其更新manual观察状态；本巡检不代改。GPU0余量及长ZH持续截断后续继续跟踪。本run仅2000/38539，未完成训练，未执行最终冻结检查、未写final-verification.json。
- 命令/修改/结果：只读cat/tail读取处置约定、manual、快照、巡检/status及frozen-step-2000.log；Python解析train.log统计finite/min/median/max、val和错误，读取/proc/stat/cmdline、checkpoint文件大小/metadata/signature、两模式summary/metrics并比对配对；wave只读生成WAV头，date -u、nvidia-smi GPU与compute-apps、df -h、free -h、shutil.disk_usage。唯一人工写入为本段追加；未改训练代码/配置、未启停/恢复训练、未新增timer/Codex/subagent，未提交推送发消息或清理。本次巡检结束。


## 2026-09-09T17:36:54.133367+00:00 主会话：前2000步观察完成，交接持续监督

第2000步checkpoint约7.80GiB，COMPLETE、四rank分片、progress={step:2000,epoch:0,next_batch:2000}和signature核查通过。最终本阶段验证first CE=1.779803、residual CE=6.717015；冻结文本前端和speaker encoder 81个张量逐值一致，见上一节。1510–2000的50个记录点全部指标有限，单步中位2.134773s，供数等待中位0.000276s，音频吞吐中位883.110秒/秒，帧预算填充中位98.2083%，grad_norm范围0.7252–1.5588。

两种模式本轮均为8条（4EN/4ZH），空生成0、截断1；固定目标/参考、ICL reference_frames以及16个24kHz单声道WAV时长检查通过。四个节点对比如下（EN为英语规范化WER、ZH为中文CER；仅在线8条诊断）：

| step | val first CE | val residual CE | speaker_only EN / ZH | ICL EN / ZH | speaker_only / ICL 截断条数 |
|---|---|---|---|---|---|
| 500 | 5.2079 | 7.2927 | 1.0556 / 1.1071 | 0.9444 / 1.1667 | 2 / 2 |
| 1000 | 2.3049 | 7.0964 | 0.4444 / 0.7857 | 0.3889 / 0.8929 | 1 / 1 |
| 1500 | 1.9272 | 6.8616 | 0.3333 / 0.6786 | 0.2778 / 0.7976 | 1 / 1 |
| 2000 | 1.7798 | 6.7170 | 0.1667 / 0.7857 | 0.3056 / 0.7857 | 1 / 1 |

质量未达到可用性结论：固定长中文sample-03在两模式持续400帧截断；2000步ICL该句ASR为空（音频非空）。其他中文也有内容替换及ASR繁简字差异，本轮speaker_only中文CER较1500回升；ICL英语有小幅波动。参考配对、目标时长、WAV头和评估输出结构均核查，当前无已证实实现故障；总体loss和多项生成指标改善，保留原训练配置。后续巡检继续逐句比较；若持续严重异常且不再改善，应按故障处置表扩大诊断，不能仅以8条早期样本更改训练目标。未试听音频。

评估后已核查2010/2020/2030/2040多个日志点，loss/梯度有限，2040 first/residual CE=1.672666/6.717085、step_seconds=2.070029；未见OOM、NaN、数据读取/保存错误或训练退出。证据milestone-2000.json、initial-observation-after-seek.jsonl、train.log和两模式评估目录。第500/1000/1500/2000全部保存、val和双模式生成均完成；这仅完成用户要求的前期观察，不是38539步训练验收。

已原子将manual-observation.json设为active=false、observed_through_step=2040，并写入证据与待跟踪事项。现场核对原巡检daemon PID441407的命令和run匹配，固定session仍为01a08697-ba78-7cd3-bb9e-b01eba12d724；最新17:21:29轮退出码0，间隔1800s，无STOP标记。后续巡检可按既有用户授权自主处理证据明确的故障并记录。训练继续两个epoch，总38539步；最终仍需完整checkpoint/val/双模式summary和include-speaker冻结核验后才结束巡检。机器和巡检进程需持续运行。

本阶段没有改LR、预算或数据；已修复的真实供数问题见15:00–15:19各节，修复提交f1cf9be、完整46项测试通过。GitHub SSH认证尚未恢复，普通push仍报publickey，未发布远端；本地提交与run下Git bundle保留。
