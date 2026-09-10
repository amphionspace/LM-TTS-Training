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


## 2026-09-09 17:56:20 UTC 单次巡检：交接后正常推进至2500步，补查长中文静音段

- 依据/交接：已读故障处置表、本文、manual、最新快照`supervision/20260909T175129Z.json`、最新完成巡检`supervision/20260909T172129Z.md`及status。manual现在active=false、phase=initial-observation-complete、observed_through_step=2040、owner=persistent supervision session，主会话已完成前2000步检查并交接。当前可按既定范围自主修复真实故障，但本轮无需要停止/修改训练的已证实运行故障。
- 前后step：上一轮2000 → 本轮快照2470 → 17:53:23现场2500。快照后再完成30步、2500 COMPLETE及val；2000评估后新增500更新/50个日志点。2500生成评估刚开始，17:53:24尚无metrics，17:54:23已出现speaker_only sample00，当前阶段有产物推进，不判停滞。
- 身份/退出：training-process.json PID621410/start_ticks80769258与/proc精确一致，cmdline为项目`.venv/bin/python .venv/bin/torchrun --standalone --nproc_per_node=4 -m qwen3_train.train --config configs/emilia-10kh-pretrain.yaml`；launcher621409/PPID1、四rank621435–621438/PPID621410/start_ticks80769421均匹配且存活。rank为R，torchrun/launcher为S；当前training-exit.json不存在，train.log无Traceback/OOM/Non-finite/Error/Aborted。未发信号、未启停/恢复训练。
- 半小时训练（step2010–2500的50个记录点）：first_ce 1.729550→1.658270，范围1.601483–1.796015；residual_ce 6.731385→6.613011；grad_norm范围0.663941–1.192303、中位0.865190、末值0.984629（clip前）。全部指标有限。主干LR9.983934249e-5→9.964589751e-5、新参数2.995180275e-4→2.989376925e-4，按既定cosine连续变化。
- 填充/吞吐/数据等待：帧预算填充94.6542%–99.8375%，中位97.4667%；token填充91.5750%–97.3667%，中位94.6625%；global samples308–392。step_seconds范围2.01588–2.34172s，中位2.11414s；音频秒/墙钟秒799.243–927.208，中位879.727；data_wait_seconds 0.0001840–0.0005156s，中位0.0002699s。与上窗口中位2.13477s/883.110音频秒/秒/等待0.0002764s相近，无持续恶化。这是每10步当步记录点统计，不是含评估的全30分钟均值。rank0训练peak allocated最高56.7675GiB，不代表各rank/生成峰值。
- Checkpoint/val：latest=step-00002500，COMPLETE时间17:52:37.735874 UTC，progress={step:2500,epoch:0,next_batch:2500}、world_size4、scheduler.last_epoch2500及LR与日志一致；四distributed分片各约2.093GB、`.metadata`1424887字节及四rng文件各14613字节齐全。当前2000/2500两COMPLETE，二者signature完全相同，既有keep_checkpoints=2轮转继续，本巡检未清理或编辑metadata。核查结构不冒充优化器恢复加载。val first_ce 2000步1.779803→2500步1.699108，residual_ce 6.717015→6.616145；2500步15个残差codebook CE均有限（4.309896–7.325610），训练及验证继续改善。

| 最新完整生成summary（step2000，各EN4/ZH4） | EN基础WER / CER | EN英语规范化WER / CER | ZH WER / CER | 截断 / 空输出 |
|---|---|---|---|---|
| speaker_only | 0.166667 / 0.069930 | 0.166667 / 0.069930 | 1.000000 / 0.785714 | 1/8 / 0/8 |
| icl | 0.333333 / 0.132867 | 0.305556 / 0.125874 | 1.375000 / 0.785714 | 1/8 / 0/8 |

- 2000步逐句及规范化：已实读全部16条metrics和WAV头（24kHz单声道、时长一致），目标ID/文本/参考ID两模式及已出现2500样本一致；没有试听。speaker_only sample02 EN WER/CER0、sample06 EN WER0.0833；ICL sample06 EN WER/CER0，但sample02的McLaughlin仍被替换，ASR含“they're”，基础WER0.4375/CER0.1765，经展开为they are后0.375/0.1618，解释了summary.english与by_language.en差异，不混用两种口径。ICL短ZH sample01 13帧/1.04s/比1.061，CER0.3333；此前12.32s异常已消失。speaker_only同句CER0.8333、sample04 ZH CER0.6923；sample12两模式分别3.84s/比0.727/CER0.7931与3.52s/比0.667/CER0.6552，有video等内容替换。EN/总体val多项继续改善，ZH有波动，仅8条诊断，不据此调LR或训练目标。
- 固定长ZH sample03仍两模式400帧/32s/比5.096（目标6.28s）：speaker_only2000 ASR“尤其了保姜氏,现在是三五两点五四”，CER0.8056；ICL2000 ASR为空、CER1。为避免仅看WAV存在/32s时长漏掉异常，本轮对500/1000/1500/2000的该目标共8个既有WAV作只读数值诊断：

| step | speaker_only RMS / 精确零样本比例 | ICL RMS / 精确零样本比例 |
|---|---|---|
| 500 | 0.0139204 / 90.7133% | 6.93849e-7 / 99.9613% |
| 1000 | 0.00835747 / 96.1449% | 2.41263e-7 / 99.9938% |
| 1500 | 0.00890832 / 91.8316% | 1.39293e-7 / 99.9979% |
| 2000 | 0.0224329 / 77.3647% | 0.0203782 / 87.1556% |

八个WAV均有限。500–1500 ICL几乎全为零，不是仅仅ASR漏识别可解释；2000已有部分能量，但仍大段零值，不能视作内容恢复。2000 ICL每4秒RMS约[0.000007,0.000005,0,0,0.051326,0.026226,0,0]；speaker_only约[0.030342,0.000010,0.000007,0,0.000001,0.055725,0,0]，数组显示值作六位小数舍入。目标真值音频实读6.28s、RMS0.0544648、peak0.575738、零比例0；speaker参考源按实际tar解码路径只读解码3.02s、RMS0.0389158、peak0.312425、零比例0，二者有限，排除输入本身全零这一解释。该检查不等于试听或证明模型/codec具体哪层出错。
- 参考路径核查及工具结果：metrics中的speaker_reference_audio缓存路径不存在，直接soundfile.read曾报LibsndfileError/System error；检查`data.py`、`speaker.py::audio_mel`及`train.py::generate_sample`确认实际speaker特征优先从audio_source tar_member解码，日志字段保留原audio路径。随后按speaker_reference_source用`decode_emilia_audio`在内存中成功解码上述3.02s参考，未物化WAV、未修改数据。此为本巡检读取非物化缓存路径的失败，不是训练丢失参考/ASR失败；不因此添加伪造缓存或重启。
- 当前2500评估：17:54:23仅speaker_only sample00已写，25帧/2s/比1.105，EOS、无截断，基础/英语规范化WER0.50/CER0.2778，ASR“the liquid spares.”；两模式summary均尚无，ICL未开始。保留该时间截面，不以不完整样本伪造汇总；2500长ZH结果需下轮继续核对。
- 资源（17:53:23 UTC）：GPU0–3占用81125/77997/75925/77439 MiB（各总81920），利用率29/52/57/54%；compute-apps仅本run四rank。GPU0外部余量795MiB，近期评估总占用稳定、无OOM，不能把总占用当活跃tensor。主机内存已用238GiB、available1.7TiB、无swap；磁盘可用584343.11GiB（约571TiB，使用46%），足以继续保存。
- 判断/处理/后续：保持现代码/配置/进程（预算6000/9000、workers16/prefetch2、LR及两epoch38539），因为优化/验证健康且当前评估有推进，尚无已证实需改动的实现故障。已按交接事项扩大长ZH诊断至波形零段及真实输入，现证据定位为生成输出质量异常，不能仅靠提高max_frames或改LR解决；若2500及后续该目标仍大段零值，应进一步对照生成codec序列及解码前浮点波形，区分生成阶段与codec输出，保留当前连续训练目标。只读数值分析不声称完成该根因定位。训练远未完成，最终必须使用`PYTHONPATH=. .venv/bin/python scripts/check_frozen_frontend.py ... --include-speaker`保留final-frozen-check.log并核查38539 COMPLETE/val/两summary，之后才能写passed=true；本轮未运行最终检查、未写final-verification.json。
- 命令与修改：cat/tail读取约定、manual、快照、巡检/status；Python解析train.log作finite/min/median/max/val/错误检查，读/proc身份、两checkpoint结构/metadata/signature；date -u、nvidia-smi GPU与compute-apps、df -h、free -h、shutil.disk_usage。`.venv/bin/python`用soundfile/numpy读取16条2000步WAV信息、8个历史long-ZH WAV数值及目标真值；sed/rg读取data.py、train.py、speaker.py、sources.py确认实际参考路径，并单次CPU内存解码tar参考。唯一人工写入为本段追加；未改代码/配置/数据，未发信号、启停恢复训练、启动timer/Codex/subagent、提交推送发消息或删除清理。本次巡检结束。


## 2026-09-09 18:25:24 UTC 单次巡检：训练持续推进，长中文speaker_only首次不截断

- 依据/状态：已读故障处置表、本文、manual、最新快照`supervision/20260909T182129Z.json`、最新完成巡检`supervision/20260909T175129Z.md`及status。18:23:55 UTC manual仍active=false，交接状态不变；本轮没有需要修复并重启的已证实训练故障。
- 前后step：上一快照2470、上一巡检现场2500 → 本轮快照2910 → 18:23:09现场2960 → 18:23:55最终采样2980。本轮现场较快照再推进70步；2500两模式评估已完成且随后连续更新。最终采样仍在训练、3000 checkpoint/eval尚未出现，尚未到该记录步，不能记成保存/评估失败。
- 进程身份与退出：PID621410/start_ticks80769258与training-process.json及/proc精确一致，cmdline为项目`.venv/bin/python .venv/bin/torchrun --standalone --nproc_per_node=4 -m qwen3_train.train --config configs/emilia-10kh-pretrain.yaml`。launcher621409/PPID1、四rank621435–621438/PPID621410/start_ticks80769421均匹配存活；采样rank R、torchrun/launcher S，GPU高利用率且step持续增加。当前training-exit.json不存在；完整train.log无Traceback/OOM/Non-finite/Error/Aborted。无信号/恢复/重启。
- 近半小时指标：按上次快照2470之后截至2960的49个记录点统计（每10步当步值，不是全窗口所有步均值），first_ce 1.657437→1.556140、residual_ce 6.632066→6.533313；grad_norm范围0.658160–1.237921、中位0.830769；全部记录指标有限。主干LR9.965526529e-5→9.939597483e-5、新参数2.989657959e-4→2.981879245e-4，按cosine连续下降。最终2980：first_ce=1.517535、residual_ce=6.519288、grad_norm=0.728892、LR=9.938361326e-5/2.981508398e-4，均正常，不因小波动调LR。
- 填充/吞吐/供数：上述49点帧填充94.6417%–99.7583%，中位97.9625%；token填充91.4889%–97.8000%，中位95.3889%；global samples310–388。step_seconds范围1.99859–2.27785s，中位2.12553s；全局音频秒/墙钟秒825.725–922.814，中位882.314；data_wait_seconds 0.0002257–0.0005120s，中位0.0002786s。与上一轮中位2.11414s/879.727音频秒/秒/等待0.0002699s相近，无持续恶化。2980当步2.083789s、896.252音频秒/秒、等待0.0002596s。rank0训练peak allocated最高56.9730GiB，不代表其他rank总占用。
- 最近完整checkpoint：最终采样latest仍step-00002500，COMPLETE时间17:52:37.735874 UTC；progress={step:2500,epoch:0,next_batch:2500}、world_size4、scheduler.last_epoch2500及LR符合该步。四distributed分片各约2.093GB、`.metadata`1424887字节、四rng文件各14613字节均齐全。2000/2500两COMPLETE均保留，signature逐项相等；未编辑metadata或清理，未做恢复加载，不把结构核查当作优化器恢复验证。
- 最近val仍2500步：first_ce=1.699108、residual_ce=6.616145，15个残差codebook CE均有限，较2000步下降；截至18:23:55尚无3000步val。

| 2500步完整生成summary（每模式EN4/ZH4） | EN基础WER / CER | EN英语规范化WER / CER | ZH WER / CER | 截断 / 空输出 |
|---|---|---|---|---|
| speaker_only | 0.194444 / 0.111888 | 0.222222 / 0.111888 | 1.125000 / 0.345238 | 0/8 / 0/8 |
| icl | 0.277778 / 0.167832 | 0.277778 / 0.167832 | 1.000000 / 0.678571 | 1/8 / 0/8 |

- 评估核查：实读两summary、全部16条metrics及WAV头，全部24kHz单声道、时长匹配；目标ID/文本/speaker_reference_id与2000步一致，ICL参考帧仍存在。speaker_only EN规范化WER由2000的0.1667波动至0.2222；sample06 ASR“In the Jefferies 2B...”中2B被规范化分成2 b，使该句WER0.1667→0.25，解释了基础与英语规范化汇总差异。ICL EN规范化WER0.3056→0.2778；ZH CER speaker_only0.7857→0.3452、ICL0.7857→0.6786。仅8条诊断，英语波动不能当整体退化、ZH改善也不等同全验证集已可用。
- 固定长ZH sample03明显分化：speaker_only从2000步400帧/32s截断变为2500步229帧/18.32s，首次EOS且未截断，目标6.28s、时长比仍2.917；ASR已覆盖到后半句“现在剩余的点量是百分之八射”，CER从0.8056降至0.2222。只读波形检查finite=true、RMS0.0245661、peak0.300934、精确零比例67.4848%（上轮77.3647%）；每4秒RMS约[0.027880,0,0,0.035787,0.034889]，末段不足4秒。仍有长零段，不能仅凭不截断宣告修复。
- 同一目标ICL2500仍400帧/32s/比5.096、ASR空/CER1，波形finite=true、RMS4.92475e-8、peak3.05176e-5、精确零比例99.9997396%（768000样本中仅2个非零），几乎全零；2000步为87.1556%零，该单目标ICL没有稳定改善。其他ICL句仍有内容输出，不能说整套ASR或codec完全失败。此前目标和tar参考输入有限且非零，配对/时长未见不一致；当前只有生成后WAV/metrics，没有生成codec序列及解码前浮点输出可供直接对照，本轮不能据此确定模型预测还是codec环节根因。保持训练继续，后续比较3000同目标及其他长句；如持续则需在保留训练目标的前提下针对原始codec输出作可复现诊断，不用延长max_frames掩盖问题。
- 其余逐句指标：speaker_only EN sample02 4.56s/比0.95、WER/CER0；sample05 1.36s/比0.840、WER0.75，ASR“Delicious uni”，短专名仍差。ZH sample01 1.04s/比1.061/CER0.6667、sample04 2.16s/比0.939/CER0.3846（ASR繁体“其實人家畢竟堅持了這麼多年”）、sample12 3.68s/比0.697/CER0.4138。ICL sample01 0.96s/比0.980/CER0.5，旧12.32s异常未复发；sample02 EN4.72s/比0.983/WER0.0625，winners被识别为whalers；sample04 ZH2s/比0.870/CER0.5385、sample05 EN1.36s/比0.840/WER1、sample06 EN3.36s/比0.944/WER0.25、sample12 ZH3.44s/比0.652/CER0.3793。全部判断基于ASR和数值分析，未试听音频。
- 资源（18:23:09 UTC）：GPU0–3占用81125/77997/75925/77439 MiB（各总81920），利用率98/99/99/99%；compute-apps仅本run四rank。GPU0外部余量795MiB，总占用与近期一致，未见OOM；不将总占用等同活跃tensor。主机内存已用238GiB、available1.7TiB、无swap；磁盘可用584162.08GiB（df约571TiB，使用46%），足以继续checkpoint。
- 判断/处理/待办：保持现代码/配置/进程（预算6000/9000、workers16/prefetch2、LR与两epoch38539）。训练/val及多数诊断指标改善，speaker_only长ZH已有结束及内容改善；保留ICL该句几乎全零为未解决质量问题，不据8条早期样本改训练目标。3000步保存/评估待后续核查，仍关注GPU0余量和长零段。未执行最终冻结检查、未写final-verification.json；完成标准仍为38539 COMPLETE、最终val/两summary及`PYTHONPATH=.` include-speaker冻结检查实际通过。
- 命令/修改：cat/tail读取约定、manual、快照、巡检/status；Python读完整train.log作finite/min/median/max/val/错误统计、/proc/stat和cmdline、两checkpoint结构/metadata/signature；date -u、nvidia-smi GPU与compute-apps、df -h、free -h、shutil.disk_usage；`.venv/bin/python`用soundfile/numpy读取16个2500步WAV信息、两个长ZH WAV的RMS/峰值/零比例，并比较配对。18:23:55再次只读核查最新step/checkpoint/eval和manual。唯一写入是本段追加；未改训练代码/配置/数据、未发信号或启停恢复训练、未新增timer/Codex/subagent、未提交推送发消息或清理。本次巡检结束。


## 2026-09-09 19:00:36 UTC 单次巡检：3500步保存与验证完成，3000步中文生成异常仍在

- 依据/权限：已读故障处置表、本文、manual、最新快照`supervision/20260909T185129Z.json`及上轮`20260909T182129Z.md`和status。manual仍active=false、主会话已观察至2040并交接。上一快照2910/上一巡检最终2980 → 本轮18:51:30快照3360 → 18:57:17现场3500；3500已完成保存/val，18:58:39正在生成评估。本轮只执行一次巡检。
- 进程/退出：training-process.json PID621410/start_ticks80769258与/proc一致，cmdline为项目`.venv/bin/python .venv/bin/torchrun --standalone --nproc_per_node=4 -m qwen3_train.train --config configs/emilia-10kh-pretrain.yaml`。launcher621409/start80769255/PPID1、四rank621435–621438/start80769421/PPID621410均存活；采样torchrun/launcher S、四rank R。training-exit.json不存在，完整train.log无Traceback/OOM/Non-finite/Error/Aborted。未发信号、未启动/停止/恢复训练。
- 近半小时及现场延伸窗口：2920–3500共59个每10步当步记录点，全部数值有限（不是含评估的全窗口均值）。first_ce首末1.607984→1.560717，范围1.510260–1.632507、中位1.564855；residual_ce首末6.561438→6.481836，范围6.454731–6.572088；grad_norm范围0.576432–0.929013、中位0.730082、末0.803487（clip前）。主干LR9.942032482e-5→9.901867678e-5、新参数2.982609745e-4→2.970560303e-4，沿既定cosine下降。
- 填充/吞吐/数据等待：帧填充94.5583%–99.7292%，中位98.0917%；token填充90.3472%–98.4639%，中位95.2472%；global samples297–415。step_seconds范围1.93731–2.29118s、中位2.13234s；全局音频秒/墙钟秒828.347–953.116、中位876.946；data_wait_seconds范围0.0001629–0.0004413s、中位0.0002723s。相对上一窗口2.12553s/882.314音频秒每秒/等待0.0002786s基本稳定，没有供数恶化证据。rank0训练peak allocated最高56.8801GiB，不等于各卡总占用。
- 最近完整checkpoint：latest=step-00003500，COMPLETE时间18:56:16.167384 UTC；progress={step:3500,epoch:0,next_batch:3500}、world_size4、scheduler.last_epoch3500、LR与3500日志相等。3000/3500四个distributed分片各约2.093GB、`.metadata`1424887字节、四rng文件各14613字节齐全，两份signature相等。2500由训练器既有keep_checkpoints=2轮转删除，巡检未清理或改metadata。这里只验证结构/一致性，没有执行优化器恢复加载。
- 验证：3000步first/residual CE=1.642074/6.535412 → 3500步1.597373/6.466996，继续下降；3500全部15个残差codebook CE有限，范围4.149448–7.222983。

| 最新完整生成summary：3000步，每模式EN4/ZH4 | EN基础WER / CER | EN英语规范化WER / CER | ZH WER / CER | 截断 / 零帧输出 |
|---|---|---|---|---|
| speaker_only | 0.083333 / 0.069930 | 0.083333 / 0.069930 | 1.375000 / 0.750000 | 1/8 / 0/8 |
| icl | 0.250000 / 0.111888 | 0.250000 / 0.111888 | 1.000000 / 0.892857 | 1/8 / 1/8 |

- 两summary及全部16条metrics实读；15个存在的WAV均24kHz单声道、数值有限，时长与metrics一致。各目标ID/文本/speaker_reference_id与2500步相同。本轮EN两口径相等；2500→3000英语规范化WER speaker_only0.222222→0.083333、ICL0.277778→0.25，但ZH CER分别0.345238→0.75、0.678571→0.892857。仅8条固定诊断不能推断全体质量，也不能忽略明确的逐句失败。未试听音频。
- 长ZH sample03（目标6.28s）两模式均400帧/32s/时长比5.09554、未EOS且截断。speaker_only从2500的229帧/18.32s未截断退回上限；ASR“又起了半個手勢”、CER0.916667，RMS0.0114921、peak0.533997、精确零比例94.2107%（2500为67.4848%），每4秒RMS[0.030589,0,0,0,0,0.010995,0,0]。ICL仍ASR空/CER1，RMS0.0105747、peak0.157623、精确零比例95.8595%，每4秒RMS[0,0,0,0,0.029910,0,0,0]；相比2500几乎全零已有局部能量，但不能视作内容改善。上述数组作六位小数舍入。
- 新的零帧输出：ICL sample12（目标5.28s、reference_frames80）在3000步0.369804s内立即EOS，frames0、truncated=false、WER/CER1；无generated.wav和duration字段，是生成前即结束的真实空输出，不是WAV丢失或ASR运行错误。2500同配对为3.44s/CER0.37931。实读`model.py:144–160`确认stop来自第一码本合法码与EOS的greedy argmax，`train.py:87–97,146–154`在stop后不追加帧并将空结果评分；故这个空输出可定位到生成EOS阶段，尚不能确定其训练/模型根因，不用补零音频或强制最小时长掩盖。
- 其余逐句：speaker_only sample02 EN4.16s/比0.867/WER0.0625（省略like）；sample05 EN2s/比1.235/WER0，sample06 EN3.6s/比1.011/WER0.0833（Jeffries专名差异）。ZH sample01 1.12s/比1.143/CER0.5、sample04 2.16s/比0.939/CER0.5385、sample12 4s/比0.758/CER0.6897。ICL sample01 0.88s/比0.898/CER0.8333，旧12.32s异常未复发；sample02 EN4.000042s/比0.833/WER0.1875（McLaughlin替换），sample04 ZH2s/比0.870/CER0.3846且ASR与目标真值ASR相同，含繁简差异；sample05 EN2s/WER0.75但CER0.0769（U.N.A.分词）；sample06 EN3.6s/WER0.0833。
- 扩大静音诊断：只读检查当前训练生成、裁剪、写WAV及已安装codec路径。speaker_only prefix长度为0、cut=0，所以该模式的长零段不可能由ICL前缀裁剪产生；ICL按帧比例裁剪，3000 sample03参考38帧对应3.04s、剩余400帧对应32s，未见整段误裁掉。已安装12Hz codec的chunked_decode按300帧分块、25帧左上下文裁掉后拼接，wrapper返回float32再由sf.write写成PCM16。3000 speaker_only零段在4–20s、即首个300帧块内部已存在，不能只归因于第二块拼接边界。此前目标及tar参考输入非零证据仍有效。现存产物没有原始预测码或解码前浮点波形，PCM16零值不能证明浮点严格为零，当前不足以区分预测退化与codec数值问题；未声称复现或定位完成。后续确需复现时，应保留该目标预测码及同次浮点解码，对比真值码解码、逐码本重复率与能量；本轮不在GPU0仅795MiB余量时加载另一套模型，也不为增加诊断而打断健康训练。
- 3500评估进展：18:57附近speaker_only已有sample00/01，18:58:39新增sample02，三条分别21/13/53帧，均EOS且未截断，sample02 EN WER/CER0。两模式3500 summary均未完成，ICL尚无metrics；评估产物实际推进，当前step3500不动是正常评估阶段，不判卡死。完整3500生成及长ZH/ICL空输出是否复发留待下轮核查。
- 资源：GPU0–3占用81125/77997/75925/77439 MiB（各81920），现场利用率0/100/100/100%；compute-apps仅本run四rank。GPU0外部余量795MiB、总占用与上轮一致，尚无OOM；不能把总占用或一次rank0低利用率单独当作泄漏/卡死。主机available1.7TiB、已用238GiB、无swap；磁盘可用584092.60GiB（约570.4TiB），足以保存。
- 判断/处理/待办：保持现训练代码、配置、进程及两epoch38539目标（预算6000/9000、workers16/prefetch2、LR不变）。优化/验证健康、评估有推进，无已证实需要修复重启的运行故障；长ZH长零段/截断和ICL sample12立即EOS是未解决质量异常，须继续分别记录，不据8条早期样本调LR或改变评估上限。训练未完成，未执行最终冻结检查、未写final-verification.json；仍需最终38539 COMPLETE、val、两summary、PYTHONPATH=. include-speaker冻结核验实际通过后才能写passed=true。
- 命令/修改：cat/tail读文档、manual、快照、上轮记录/status；Python解析train.log作finite/min/median/max/val/错误检查、读/proc身份、checkpoint目录/metadata/signature及shutil.disk_usage；nvidia-smi两类查询、free -h；`.venv/bin/python`用soundfile/numpy读metrics/WAV并比较配对/零比例/RMS；rg/sed读model.py、train.py和已安装codec源码。初次rg猜测的evaluation.py、Python3.11/3.12依赖路径及pyproject.toml不存在，之后rg --files和pyvenv.cfg确认实际Python3.10路径并读取成功，属于巡检路径查询失败，不是训练错误。唯一人工写入为本段追加；未改训练代码/配置/数据，未启停/恢复、发信号、新增timer/Codex/subagent、提交推送、发消息或删除清理。本次巡检结束。


## 2026-09-09 19:24:46 UTC 单次巡检：推进至3880步，3500评估完成，ICL两条立即EOS

- 依据/权限：已读故障处置表、本文、manual、当前快照`supervision/20260909T192129Z.json`、最新完成巡检`20260909T185129Z.md`及status。manual仍active=false、主会话observed_through_step2040。上一快照3360/上一巡检3500 → 本轮19:21:30快照3830 → 19:21:51现场3840 → 19:22:30指标截止3860 → 19:23:18最终采样3880。3500评估已全部结束，随后连续训练；4000保存/评估尚未到步，不是缺失故障。
- 进程/退出：training-process.json PID621410/start_ticks80769258与/proc一致，cmdline为项目`.venv/bin/python .venv/bin/torchrun --standalone --nproc_per_node=4 -m qwen3_train.train --config configs/emilia-10kh-pretrain.yaml`。launcher621409/start80769255/PPID1、四rank621435–621438/start80769421/PPID621410全部匹配存活；torchrun/launcher S、四rank R。training-exit.json不存在，完整train.log无Traceback/OOM/Non-finite/Error/Aborted。没有停止/恢复或发信号。
- 近半小时指标：3370–3860共50个每10步当步记录点，全部数值有限。first_ce首末1.584487→1.493246、范围1.474181–1.626156、中位1.538437；residual_ce首末6.495502→6.441274、范围6.408484–6.504675；grad_norm范围0.573997–0.887842、中位0.731162、末0.796993（clip前）。主干LR9.911775473e-5→9.871715376e-5、新参数2.973532642e-4→2.961514613e-4，既定cosine连续下降。最终3880 first/residual CE1.498772/6.426274、grad0.650106、LR9.869923636e-5/2.960977091e-4，均有限。
- 动态batch/吞吐：上述记录点帧填充94.1667%–99.7875%，中位97.8521%；token填充89.7611%–98%，中位94.8708%；global samples315–400。step_seconds范围2.03377–2.29883s、中位2.15172s；全局音频秒/墙钟秒816.417–930.606、中位870.773；data_wait_seconds范围0.0002359–0.0004323s、中位0.0002759s。与上轮中位2.13234s/876.946音频秒每秒/等待0.0002723s接近，无持续供数下降证据。这是稀疏当步记录，不是含评估的30分钟所有步均值。rank0训练peak allocated最高57.2515GiB，不代表各卡总占用。
- Checkpoint/val：latest仍step-00003500，COMPLETE时间18:56:16.167384 UTC，progress={step:3500,epoch:0,next_batch:3500}、world_size4、scheduler.last_epoch3500、LR与3500日志一致。3000/3500两份COMPLETE、四distributed分片各约2.093GB、`.metadata`1424887字节及四rng各14613字节齐全，signature相等；未编辑/清理，未冒充恢复加载验证。最新val3500 first/residual CE1.597373/6.466996、15个codebook CE有限（4.149448–7.222983），较3000下降。

| 3500步完整summary，每模式EN4/ZH4 | EN基础WER / CER | EN英语规范化WER / CER | ZH WER / CER | 截断 / 零帧输出 |
|---|---|---|---|---|
| speaker_only | 0.222222 / 0.090909 | 0.222222 / 0.090909 | 1.000000 / 0.690476 | 1/8 / 0/8 |
| icl | 0.194444 / 0.083916 | 0.194444 / 0.083916 | 1.000000 / 0.845238 | 1/8 / 2/8 |

- 生成核验：读取两summary、全部16条metrics及14个存在的WAV，WAV均24kHz单声道、数值有限、时长与metrics一致；ICL两零帧样本不应有WAV。全部目标ID/文本/speaker_reference_id与3000相同。本轮EN基础与英语规范化口径相等。3000→3500 EN WER speaker_only0.083333→0.222222、ICL0.25→0.194444；ZH CER分别0.75→0.690476、0.892857→0.845238。仅8条固定诊断，不能由总体小幅改善掩盖空输出增多，也不能据英语波动改LR。未试听音频。
- 长ZH sample03仍两模式400帧/32s/目标6.28s、比5.09554，未EOS且截断。speaker_only ASR“有气了,帮你收拾 下了大概10公里 先升至下午2点50”，CER0.833333（3000为0.916667），仍有内容替换/顺序异常；RMS0.0192218、peak0.442902、精确零比例86.9447%（3000为94.2107%），每4秒RMS[0.026151,0,0,0.047126,0.007149,0,0,0]。ICL本轮ASR不再空，但重复“现在是小五两点五十”两次、CER0.722222；RMS0.0102064、peak0.279938、精确零比例92.7026%（3000为95.8595%），每4秒RMS[1.39293e-7,0,0.026152,0.012224,0,0,0,0]。有局部内容不等于长零段或停止行为恢复；原始预测码和解码前浮点波形仍未保存，当前不能继续缩小模型预测/codec数值的根因范围。
- ICL空输出扩大：sample12连续3000/3500立即EOS，3500耗时0.367636s，frames0、reference_frames80、目标5.28s。sample04本轮首次立即EOS，耗时0.368012s、frames0、reference_frames19、目标2.3s；3000同配对25帧/2s，speaker_only3500仍32帧/2.56s。两条truncated=false、WER/CER1、无duration/WAV，是生成EOS阶段失败，不是ASR或落盘丢失。空输出数由1/8增至2/8（均ZH），列为明确未解决质量问题。
- 针对空输出进一步只读核查：实读model.input_embeddings、evaluate_audio和data.py，参考选择按同speaker/同language且不同ID/文本，ICL使用参考音频码前缀与参考+目标文本，既有生成逻辑由greedy EOS决定停止。流式定位两目标和两参考manifest行，并读四份NPZ：sample04参考(19,16)/目标(29,16)，sample12参考(80,16)/目标(66,16)；全部codes_sha256与manifest一致，码范围均在0–2047内，文本非空，speaker/language配对相符。参考文本长度7/33、目标8/17；参考各码本唯一值数分别全19及71–80，排除这两个参考为空、纯常量码或hash损坏这一解释。未发现据此可直接修复的输入结构错误，也未执行完整checkpoint推理复现。下一步需要同目标的EOS与最佳非EOS logits、生成码和浮点解码对照；不在现有GPU0余量仅795MiB时另载训练模型，也不通过禁EOS、强制最短长度或更换参考来掩盖失败。
- 其他逐句指标：sample02 EN两模式均WER/CER0，speaker_only4.24s/比0.883、ICL4.080042s/比0.850。speaker_only sample05“WA Flanders Uni”因WA分词及专名差异WER0.75但CER0.0769；sample06“Jeffreeze toupee...swap”WER0.3333/CER0.2045，是英语汇总回升的主要贡献。ICL sample05“W.F. Flangers, U.N.”WER1/CER0.3077，sample06 WER0.1667/CER0.1136。ZH短sample01 speaker_only1.04s/CER1、ICL0.96s/CER0.5，旧长时异常未复发；speaker_only sample04 CER0.4615、sample12 3.52s/比0.667/CER0.5517，仍有繁简/内容差异及video缺失。
- 资源（19:22:30附近）：GPU0–3占用81125/77997/75925/77439 MiB（各81920），现场利用率23/23/100/23%，快照98/98/99/99%；compute-apps仅本run四rank。GPU0外部余量795MiB、占用与前轮稳定，没有OOM或持续失活证据。CPU内存已用238GiB、available1.7TiB、无swap；磁盘可用584319.39GiB（约570.6TiB），足以继续保存。
- 判断/处理/后续：保持现训练代码/配置/进程（预算6000/9000、workers16/prefetch2、LR及两epoch38539不变），因为训练/val健康、step连续推进，未定位到需要干预的实现故障；ICL空输出增加和长ZH静音/重复仍需独立跟踪，不能宣告生成质量已恢复。4000 checkpoint/val/两生成待下一轮。未执行最终冻结检查、未写final-verification.json；38539完整checkpoint、最终val/两summary和PYTHONPATH=. include-speaker冻结检查实际通过仍是最终验收条件。
- 命令/修改/结果：cat/tail读取约定、manual、快照、最新巡检/status和配置；Python解析完整train.log作finite/min/median/max/val/错误统计、读/proc身份、两checkpoint文件/metadata/signature、shutil.disk_usage；nvidia-smi GPU/compute-apps与free -h；`.venv/bin/python`用soundfile/numpy检查14个WAV数值和全部metrics，比较2500/3000/3500两个立即EOS目标；sed/rg只读模型/生成/data逻辑，流式读train/val manifest及四NPZ/hash；最后复核3880/退出文件。所有命令成功，唯一人工写入是本段追加。未改代码/配置/数据，未发信号或启停恢复训练，未新增timer/Codex/subagent、提交推送、发消息或删除清理。本次巡检结束。


## 2026-09-09 19:54:31 UTC 单次巡检：推进至4320步，4000评估完成，ICL sample12空输出持续

- 依据/权限与进度：已读故障处置表、本文、manual、最新快照`supervision/20260909T195129Z.json`、最新完成巡检`20260909T192129Z.md`及status。manual仍active=false、主会话observed_through_step2040。上一快照3830/上一巡检3880 → 本轮19:51:29快照4270 → 19:52:07现场4290 → 19:52:43指标截止4310 → 19:53:03最终采样4320。4000保存、验证和两种生成均完成，之后持续训练；4500尚未到步，未把未到期产物当作缺失故障。
- 进程/退出：training-process.json PID621410/start_ticks80769258与/proc相符，cmdline为项目`.venv/bin/python .venv/bin/torchrun --standalone --nproc_per_node=4 -m qwen3_train.train --config configs/emilia-10kh-pretrain.yaml`。launcher621409/start80769255/PPID1、四rank621435–621438/start80769421/PPID621410均匹配存活。本次采样六进程均S，但GPU持续高利用率、step两次采样继续增加，不能据一次S状态判卡死。training-exit.json不存在；完整train.log无Traceback/OOM/Non-finite/Error/Aborted。未发信号或启停恢复。
- 近半小时训练：3840–4310共48个每10步当步记录点，全部数值有限。first_ce首末1.534149→1.500115、范围1.437679–1.568302、中位1.499444；residual_ce首末6.414985→6.381359、范围6.360045–6.443961；grad_norm范围0.546647–0.842291、中位0.683667、末0.765927（clip前）。主干LR9.873494868e-5→9.828448720e-5、新参数2.962048460e-4→2.948534616e-4，既定cosine连续下降。最终4320 first/residual CE1.467735/6.370825、grad0.665887、LR9.827417278e-5/2.948225183e-4，均有限。
- 动态batch/吞吐/等待：上述48点帧填充94.3625%–99.7875%，中位97.7188%；token填充90.8444%–97.7639%，中位95.1153%；global samples300–401。step_seconds范围1.95883–2.33490s、中位2.14287s；全局音频秒/墙钟秒802.603–935.540、中位875.929；data_wait_seconds范围0.0002138–0.0006081s、中位0.0002784s。相对上轮2.15172s/870.773音频秒每秒/等待0.0002759s基本稳定。4310为本窗口最慢2.33490s，但4320已回2.14240s，不能据此改workers。以上为稀疏当步值，不是含评估的全窗口均值；rank0训练peak allocated最高57.0155GiB，不代表各卡总显存。
- Checkpoint/val：latest=step-00004000，COMPLETE时间19:27:19.545029 UTC；progress={step:4000,epoch:0,next_batch:4000}、world_size4、scheduler.last_epoch4000及LR9.858916285e-5/2.957674885e-4符合4000步。3500/4000两COMPLETE的四distributed分片各约2.093GB、`.metadata`1424887字节及四rng各14613字节齐全，signature相等；3000由训练器既有keep_checkpoints=2轮转，巡检未删除或改metadata。只核查结构与一致性，未做恢复加载。4000 val first/residual CE1.557591/6.414087，较3500的1.597373/6.466996下降；15个codebook CE有限（4.106878–7.176060）。

| 4000步完整summary，每模式EN4/ZH4 | EN基础WER / CER | EN英语规范化WER / CER | ZH WER / CER | 截断 / 零帧输出 |
|---|---|---|---|---|
| speaker_only | 0.111111 / 0.104895 | 0.138889 / 0.104895 | 1.000000 / 0.726190 | 1/8 / 0/8 |
| icl | 0.138889 / 0.083916 | 0.138889 / 0.083916 | 1.000000 / 0.916667 | 1/8 / 1/8 |

- 评估核验：两summary、全部16条metrics及15个存在的WAV均实读；WAV全部24kHz单声道、数值有限、时长与metrics一致。目标ID/文本/speaker_reference_id与3500相同。ICL sample12零帧无WAV符合生成立即EOS分支，不是落盘或ASR失败。EN英语规范化WER较3500 speaker_only0.222222→0.138889、ICL0.194444→0.138889；ZH CER分别0.690476→0.726190、0.845238→0.916667，不能以英语改善代表中文恢复。仅8条固定诊断，未据此调整超参；未试听音频。
- 英语规范化差异：speaker_only sample05 ASR“W.A. Flengder's Uni.”，基础归一化保留flengder's，WER0.25；英语归一化展开为flengder is，WER0.5，故EN summary基础0.111111与english0.138889不同，CER同为0.104895。该口径差异有逐句证据，未混用两者。sample02 speaker_only McLaughlin被识别为McLean，4.72s/比0.983/WER0.0625；ICL同句4.72s/WER/CER0。sample06两模式均Jeffreeze专名差异、WER0.0833，分别4s/比1.124和4.160042s/比1.169；ICL sample05 1.68s/比1.037/WER0.5。
- 长ZH sample03仍两模式400帧/32s/目标6.28s、时长比5.09554、未EOS且截断。speaker_only ASR仅“又期待半个小时”，CER0.861111，RMS0.00821021、peak0.344818、精确零比例95.7935%（3500为86.9447%）；每4秒RMS[0.023222,0,0,0,0,0,0,0]。ICL在3500出现重复短句后本轮又ASR空/CER1；RMS0.0150501、peak0.164886、精确零比例93.2669%（3500为92.7026%），每4秒RMS[0,0,0,0,0.038771,0.017574,0,0]。波形仍有局部能量，不能叫全零，也不能视作内容有效或停止行为恢复。数组按六位小数舍入。
- ICL空输出跟踪：sample04由3500零帧恢复到4000的24帧/1.92s/目标2.3s、比0.835/CER0.461538，仍有内容/繁简差异。sample12连续3000/3500/4000三次立即EOS，本轮耗时0.358703s、frames0、reference_frames80、目标5.28s、WER/CER1。空输出总数由2/8降到1/8，但该固定目标未改善；其speaker_only4000仍有41帧/3.28s/比0.621输出，ASR“首先你跟似乎都看過我寄給你們的JVLV”，CER0.689655。两模式短ZH sample01分别1.12s/CER0.5与1.04s/CER1，旧超长时长问题未复发。
- 异常判断与诊断边界：持续长零段/截断、ICL sample12过早EOS依然是未解决生成质量问题；之前已对照真实目标/参考非零音频、参考码hash/形状/有效值/非空文本、同speaker/language配对和输入/裁剪逻辑，未确认实现错误。本轮同配对逐句比较显示sample04可恢复而sample12持续，不能推断整体ICL或ASR损坏。现存产物仍无原始预测码、EOS与非EOS logits或解码前浮点波形，未声称完成根因复现；不重复把PCM16零比例当浮点严格为零，也不为单目标另载模型挤占当前GPU资源。需要这些证据才能进一步区分模型停止/预测退化与codec数值问题；不以禁EOS、强制最小时长、改参考或延长max_frames掩盖异常。
- 资源（19:52:43附近）：GPU0–3占用81125/77997/75925/77439 MiB（各81920），利用率95/93/95/100%；compute-apps仅本run四rank。GPU0外部余量795MiB、总占用与近期相同，未见OOM。主机内存已用238GiB、available1.7TiB、无swap；磁盘可用584273.65GiB（约570.6TiB），足以继续checkpoint。
- 判断/处理/后续：保持现训练代码/配置/进程（预算6000/9000、workers16/prefetch2、LR及两epoch38539目标），训练/验证继续改善、供数及吞吐稳定，暂无有证据支持的根因修复或重启动作。继续分别跟踪4500保存/评估、长ZH和ICL sample12；生成质量尚不能验收。未执行最终冻结检查、未写final-verification.json；最终仍须38539 COMPLETE、最终val/两summary及PYTHONPATH=. include-speaker冻结检查实际通过后才能写passed=true。
- 命令/修改/结果：cat/tail读故障处置表、本文、manual、当前快照、最新巡检/status；Python解析完整train.log统计finite/min/median/max/val/错误、读/proc身份、checkpoint目录/metadata/signature及shutil.disk_usage；nvidia-smi GPU/compute-apps与free -h；`.venv/bin/python`用soundfile/numpy读取全部metrics与15个WAV，比较3500配对、RMS/零比例/时长/英语口径；最后只读复核4320、latest、manual和退出/验收文件。命令均成功，唯一人工写入为本段追加；未改训练代码/配置/数据、未发信号或启停恢复、未新增timer/Codex/subagent、提交推送、发消息或删除清理。本次巡检结束。


## 2026-09-09 20:24:18 UTC 单次巡检：推进至4770步，4500步ICL长中文首次不截断

- 依据/权限与进度：已读故障处置表、本文、manual、当前快照`supervision/20260909T202129Z.json`、最新完成巡检`20260909T195129Z.md`及status。manual仍active=false，主会话observed_through_step2040。上一快照4270/上一巡检4320 → 本轮20:21:30快照4730 → 20:21:59现场4740 → 20:22:36指标截止4760 → 20:22:51最终采样4770。4500保存、val和两生成均完整，评估后连续更新；5000尚未到步，未把未到期产物视为故障。
- 进程/退出：training-process.json PID621410/start_ticks80769258与/proc精确一致，cmdline为项目`.venv/bin/python .venv/bin/torchrun --standalone --nproc_per_node=4 -m qwen3_train.train --config configs/emilia-10kh-pretrain.yaml`。launcher621409/start80769255/PPID1、四rank621435–621438/start80769421/PPID621410均匹配存活；torchrun/launcher S、四rank R。training-exit.json不存在，完整train.log无Traceback/OOM/Non-finite/Error/Aborted。未发信号或启停恢复。
- 近半小时训练：4280–4760共49个每10步当步记录点，所有指标有限。first_ce首末1.494303→1.396822、范围1.396822–1.529268、中位1.468484；residual_ce首末6.401439→6.314886、范围6.310151–6.401439；grad_norm范围0.535077–0.767859、中位0.635016、末0.756003（clip前）。主干LR9.831524855e-5→9.779043874e-5、新参数2.949457457e-4→2.933713162e-4，按既定cosine下降。最终4770 first/residual CE1.448908/6.344409、grad0.609852、LR9.777876791e-5/2.933363037e-4，均有限，不据单步回升调LR。
- 动态batch/吞吐/等待：上述49点帧填充94.8292%–99.6458%，中位98.2708%；token填充91.7417%–98.0167%，中位95.4194%；global samples313–409。step_seconds范围2.00120–2.33490s、中位2.15437s；全局音频秒/墙钟秒802.603–943.439、中位878.475；data_wait_seconds范围0.0001677–0.0004099s、中位0.0002748s。与上轮2.14287s/875.929音频秒每秒/等待0.0002784s相近，未见供数恶化。这是每10步当步记录点统计，不是含评估的全窗口均值；rank0训练peak allocated最高56.9363GiB，不代表各卡总占用。最终4770当步2.10766s/909.709音频秒每秒/等待0.0004942s，无持续变慢。
- Checkpoint/val：latest=step-00004500，COMPLETE时间19:59:27.304678 UTC；progress={step:4500,epoch:0,next_batch:4500}、world_size4、scheduler.last_epoch4500、LR9.808333750e-5/2.942500125e-4符合该步。4000/4500两COMPLETE四distributed分片各约2.093GB、`.metadata`1424887字节及四rng各14613字节齐全，signature相等。3500由训练器既有keep_checkpoints=2轮转，巡检未清理或编辑metadata；仅验证结构/一致性，没有做恢复加载。4500 val first/residual CE1.525012/6.368667，较4000的1.557591/6.414087下降；15个codebook CE有限（4.066110–7.136250）。

| 4500步完整summary，每模式EN4/ZH4 | EN基础WER / CER | EN英语规范化WER / CER | ZH WER / CER | 截断 / 零帧输出 |
|---|---|---|---|---|
| speaker_only | 0.250000 / 0.195804 | 0.277778 / 0.195804 | 1.000000 / 0.785714 | 1/8 / 0/8 |
| icl | 0.055556 / 0.034965 | 0.055556 / 0.034965 | 1.125000 / 0.690476 | 0/8 / 1/8 |

- 评估核验：实读两summary、全部16条metrics及15个存在的WAV，WAV全部24kHz单声道、数值有限、时长与metrics一致；目标ID/文本/speaker_reference_id与4000相同。ICL sample04零帧无WAV符合立即EOS分支，不是落盘或ASR失败。EN英语规范化WER 4000→4500 speaker_only0.138889→0.277778、ICL0.138889→0.055556；ZH CER分别0.726190→0.785714、0.916667→0.690476，两模式分化。仅8条固定诊断，不能由ICL本轮改善或speaker_only波动推断整体可用性/退化；未试听音频。
- ICL长ZH sample03首次正常EOS：从4000步400帧/32s截断变为4500步252帧/20.16s、未截断，目标6.28s、时长比仍3.21019。ASR覆盖到后半句“现在生育的点量是80%”，但“现在是下午两点五十/现在向右两点五十”重复且有词替换，CER0.666667（4000为1）。RMS0.0253845、peak0.357849、精确零比例68.2668%（4000为93.2669%），每4秒RMS[0,0,0.033876,0.018800,0.041762,0.008029]，末段仅0.16s；仍有至少前8秒零段。未截断与内容增加是局部改善，不代表时长/静音问题已解决。
- 同目标speaker_only仍400帧/32s/比5.09554、无EOS且截断，ASR仅“又起了搬甲室”，CER0.944444；RMS0.0100396、peak0.335205、精确零比例96.2688%（4000为95.7935%），每4秒RMS[0.028396,2.36798e-6,1.45092e-6,0,0,0,0,0]。该模式没有同步改善，继续作为明确未解决质量问题。以上波形判断为数值诊断，PCM16零样本不等于已证明解码前浮点严格为零。
- ICL立即EOS交替：sample12经历3000/3500/4000三次空输出后，4500恢复43帧/3.44s/目标5.28s、比0.6515，CER0.551724、ASR仍有“为了为了”及video内容遗漏；不能记成连续第四次空输出。sample04在4000恢复后，本轮又0帧立即EOS、耗时0.359378s、reference_frames19、目标2.3s、WER/CER1。空输出总数仍1/8，但目标发生变化，说明停止行为尚不稳定，不能宣告ICL已修复。
- 英语与其余逐句：speaker_only sample00 ASR“I'm the league lead spurs.”，基础WER1、英语展开i am后WER1.25，解释EN汇总0.25与0.277778差异；sample05 ASR“to debut a flandest uni.”，WER0.75/CER0.7692，二者是英语回升主要贡献。sample02两模式均WER/CER0，分别4.72s/比0.983和4.56s/比0.95；ICL sample05 1.84s/比1.136、sample06 4.24s/比1.191也均WER/CER0。speaker_only sample06 4s/比1.124/WER0.1667，Jeffreeze与his swamp有差异。短ZH sample01两模式均1.04s，CER分别1和0.8333，旧超长时长未复发；speaker_only sample04 2.72s/比1.183/CER0.6154、sample12 3.28s/比0.621/CER0.6207，仍有内容替换及繁简差异。
- 资源（20:22:36附近）：GPU0–3占用81127/77997/75925/77439 MiB（各81920），利用率25/21/22/23%；compute-apps仅本run四rank。GPU0较上轮增加2MiB、外部余量793MiB，其他卡相同；结合step稳定连续推进，无证据据此判显存泄漏或卡死。主机内存已用238GiB、available1.7TiB、无swap；磁盘可用584151.26GiB（约570.5TiB），足以继续保存。
- 判断/处理/后续：保持现训练代码/配置/进程（预算6000/9000、workers16/prefetch2、LR与两epoch38539目标）。训练/val、供数和吞吐健康，ICL长ZH及sample12本轮有实际改善；speaker_only长零段/截断及ICL sample04立即EOS仍未解决。先前目标/参考音频、NPZ/hash、文本和配对/裁剪检查没有确认实现错误，本轮未因8条诊断波动改超参或为了增加诊断而打断训练。原始预测码/logits及解码前浮点波形仍缺，未声称完成模型/codec根因复现。后续检查5000保存/评估及相同目标停止行为和长零段是否持续改善。未执行最终冻结检查、未写final-verification.json；最终验收仍需38539 COMPLETE、最终val/两summary及PYTHONPATH=. include-speaker冻结核验实际通过。
- 命令/修改/结果：cat/tail读约定、本文、manual、快照、最新巡检/status；Python解析完整train.log做finite/min/median/max/val/错误统计、读/proc身份、checkpoint文件/metadata/signature及shutil.disk_usage；nvidia-smi GPU/compute-apps与free -h；`.venv/bin/python`用soundfile/numpy读16条metrics和15个WAV、比较4000配对、时长/波形零值/RMS及英语规范化；最后只读复核4770、latest、manual、退出/验收文件。所有命令成功，唯一人工写入为本段追加；未改训练代码/配置/数据、发信号或启停恢复，未新增timer/Codex/subagent、提交推送、发消息或删除清理。本次巡检结束。


## 2026-09-09 20:54:59 UTC 单次巡检：推进至5250步，5000评估完整，ICL长中文再次截断

- 依据/权限与进度：已读故障处置表、本文、manual、最新快照`supervision/20260909T205129Z.json`、最新完成巡检`20260909T202129Z.md`及status。manual仍active=false、主会话observed_through_step2040。上一快照4730/上一巡检4770 → 本轮20:51:29快照5190 → 20:52:37现场5220 → 20:53:17指标截止5240 → 20:53:31最终采样5250。5000保存、val和两生成已完成，之后连续更新；5500尚未到步，不是缺失故障。
- 进程/退出：training-process.json PID621410/start_ticks80769258与/proc一致，cmdline为项目`.venv/bin/python .venv/bin/torchrun --standalone --nproc_per_node=4 -m qwen3_train.train --config configs/emilia-10kh-pretrain.yaml`。launcher621409/start80769255/PPID1、四rank621435–621438/start80769421/PPID621410全部匹配存活；torchrun/launcher S、四rank R。training-exit.json不存在；完整train.log无Traceback/OOM/Non-finite/Error/Aborted。未发信号或启停恢复。
- 近半小时训练：4740–5240共51个每10步当步记录点，全部数值有限。first_ce首末1.514787→1.463362、范围1.396822–1.514787、中位1.446739；residual_ce首末6.356975→6.301922、范围6.276364–6.397304；grad_norm范围0.494113–0.756003、中位0.639005、末0.623338（clip前）。主干LR9.781369049e-5→9.719658824e-5、新参数2.934410715e-4→2.915897647e-4，按既定cosine连续下降。最终5250 first/residual CE1.451883/6.317072、grad0.620745、LR9.718348887e-5/2.915504666e-4，均有限。
- 填充/吞吐/等待：上述51点帧填充94.9875%–99.8625%，中位98.1792%；token填充92%–98.4028%，中位95.5472%；global samples321–423。step_seconds范围2.03367–2.35502s、中位2.15149s；全局音频秒/墙钟秒795.679–912.273、中位875.361；data_wait_seconds范围0.0002431–0.0006014s、中位0.0002784s。相对上轮2.15437s/878.475音频秒每秒/等待0.0002748s基本稳定，无持续供数恶化证据。以上为稀疏当步值，不是含评估的全30分钟均值；rank0训练peak allocated最高56.8485GiB，不代表各卡总占用。
- Checkpoint/val：latest=step-00005000，COMPLETE时间20:31:02.488366 UTC；progress={step:5000,epoch:0,next_batch:5000}、world_size4、scheduler.last_epoch5000、LR9.750208629e-5/2.925062589e-4符合该步。4500/5000两COMPLETE四distributed分片各约2.093GB、`.metadata`1424887字节及四rng各14613字节齐全，signature相等。4000由训练器既有keep_checkpoints=2轮转，巡检未清理或改metadata；结构检查不等于实际恢复加载。5000 val first/residual CE1.516816/6.332561，较4500的1.525012/6.368667下降；15个codebook CE有限（4.037282–7.103788）。

| 5000步完整summary，每模式EN4/ZH4 | EN基础WER / CER | EN英语规范化WER / CER | ZH WER / CER | 截断 / 零帧输出 |
|---|---|---|---|---|
| speaker_only | 0.277778 / 0.230769 | 0.277778 / 0.230769 | 1.250000 / 0.773810 | 1/8 / 0/8 |
| icl | 0.194444 / 0.167832 | 0.194444 / 0.167832 | 1.125000 / 0.666667 | 1/8 / 2/8 |

- 评估核验：实读两summary、16条metrics和14个存在的WAV，WAV全部24kHz单声道、数值有限、时长与metrics一致；全部目标ID/文本/speaker_reference_id与4500相同。两ICL零帧无WAV符合立即EOS分支，不是ASR或落盘丢失。EN基础与英语规范化本轮相等；4500→5000英语规范化WER speaker_only0.277778→0.277778（CER0.195804→0.230769）、ICL0.055556→0.194444。ZH CER分别0.785714→0.773810、0.690476→0.666667，小幅汇总改善不能掩盖截断/空输出。仅8条诊断，未据此改超参；未试听音频。
- 长ZH sample03：两模式重新均400帧/32s/目标6.28s、比5.09554、未EOS且截断。speaker_only ASR“就起了半個小時”，CER0.916667、RMS0.0121732、peak0.444427、精确零比例96.0904%（4500为96.2688%），每4秒RMS[0.034431,0,0,0,0,0,0,0]。ICL从4500的252帧/20.16s未截断退回400帧，ASR再次空/CER1；RMS0.0136513、peak0.165588、精确零比例93.5740%（4500为68.2668%），每4秒RMS[0,0,0,0,0.033563,0.019089,0,0]。4500停止/内容改善没有稳定保持；两模式长零段仍未解决。波形有局部能量，不称全零；PCM16零比例不能证明解码前浮点严格为零。
- ICL空输出变化：sample04连续4500/5000立即EOS，本轮0.365312s、frames0、reference_frames19、目标2.3s、WER/CER1。新增英语sample00立即EOS，0.357690s、frames0、reference_frames36、目标1.81s、WER/CER1；4500同配对18帧/1.44s，因此过早停止现象不局限ZH。空输出数1/8→2/8（EN1/ZH1），解释ICL英语回升中的4个删除错误。该现象定位于生成EOS分支，不能由重新跑ASR修复；根因尚未确定。
- ICL sample12继续有输出：4500恢复后5000为44帧/3.52s/目标5.28s、比0.6667，CER0.172414（4500为0.551724），ASR“相信你个师傅都看过我寄给你们那卷VRVIDEO VRVIDEO”。内容明显增加但仍有替换/重复，且中文WER2受空格分词影响，故分开记录CER和WER。不能将其误记为持续空输出。speaker_only同句48帧/3.84s/比0.7273/CER0.7241，ASR“然后先你跟师父都看过我 寄给您的范围 飞动飞动”，仍有明显内容异常。
- 其余逐句：speaker_only sample05“Tubertepe, flying as you need.”，2.08s/比1.284/WER1.25/CER1.3846，是本模式EN错误中5/10词错误、18/33字符错误的主要来源；ICL该句1.76s/比1.086/WER0.5/CER0.2308。sample02 speaker_only4.24s/比0.883/WER0.0625（省略like），ICL4.080042s/比0.850/WER/CER0。sample06分别3.44s/比0.966/WER0.1667与3.200042s/比0.899/WER0.0833，仍有Jeffreeze/tooth差异。短ZH sample01分别1.04s/CER0.6667及0.96s/CER0.3333，旧超长时长未复发；speaker_only sample04 2.4s/比1.043/CER0.5385，含繁简及词替换。
- 资源（20:53:17附近）：GPU0–3占用81127/77997/75925/77439 MiB（各81920），利用率20/20/21/22%；compute-apps仅本run四rank。GPU0外部余量793MiB，显存与上轮一致，无OOM；结合多次step连续增加及稳定step_seconds，单次低利用率不构成停滞证据。主机内存已用238GiB、available1.7TiB、无swap；磁盘可用584120.48GiB（约570.4TiB），足以继续保存。
- 判断/处理/后续：保持现训练代码/配置/进程（预算6000/9000、workers16/prefetch2、LR及两epoch38539目标）。训练/val及吞吐健康；生成异常持续但在目标间分化，sample12内容改善、长ZH和两个立即EOS目标仍未解决。先前目标/参考真实音频、NPZ/hash、文本与配对/输入/裁剪核查没有确认可直接修复的实现错误，本轮未把8条诊断波动当调LR或重启依据。未获取原始预测码、EOS/非EOS logits及解码前浮点波形，未声称完成根因复现；这些仍是进一步定位所需证据。后续核对5500保存/评估，关注新增EN sample00立即EOS、sample04及长ZH。训练尚未完成，未执行最终冻结检查、未写final-verification.json；最终仍须38539 COMPLETE、最终val/两summary及PYTHONPATH=. include-speaker冻结核验实际通过。
- 命令/修改/结果：cat/tail读取故障处置表、本文、manual、快照、最新巡检/status；Python读完整train.log做finite/min/median/max/val/错误统计、/proc身份、两checkpoint文件/metadata/signature及shutil.disk_usage；nvidia-smi GPU/compute-apps与free -h；`.venv/bin/python`用soundfile/numpy读取16条metrics和14个WAV、比较4500配对、时长/RMS/零值及英语口径；最后只读复核5250、latest、manual、退出/验收文件。命令全部成功，唯一人工写入为本段追加；未改训练代码/配置/数据、发信号或启停恢复，未新增timer/Codex/subagent、提交推送、发消息或删除清理。本次巡检结束。


## 2026-09-09 21:24:16 UTC 单次巡检：推进至5660步，5500步ICL长中文WAV全部为零

- 依据/权限与进度：已读故障处置表、本文、manual、快照`supervision/20260909T212129Z.json`、最新完成巡检`20260909T205129Z.md`及status。manual仍active=false、主会话observed_through_step2040。上一快照5190/上一巡检5250 → 本轮21:21:29快照5620 → 21:21:47现场5630 → 21:22:24指标截止5650 → 21:22:48最终采样5660。5500保存、val和两生成完整，随后连续训练；6000尚未到步，不是缺失故障。
- 进程/退出：training-process.json PID621410/start_ticks80769258与/proc一致，cmdline为项目`.venv/bin/python .venv/bin/torchrun --standalone --nproc_per_node=4 -m qwen3_train.train --config configs/emilia-10kh-pretrain.yaml`。launcher621409/start80769255/PPID1、四rank621435–621438/start80769421/PPID621410均匹配存活。采样均S，但四GPU利用率99%–100%、step持续增加，不据一次进程状态判停滞。training-exit.json不存在；完整train.log无Traceback/OOM/Non-finite/Error/Aborted。未发信号或启停恢复。
- 近半小时训练：5200–5650共46个每10步当步记录点，全部指标有限。first_ce首末1.473660→1.406144、范围1.356257–1.522780、中位1.435676；residual_ce首末6.344226→6.259316、范围6.231207–6.344226；grad_norm范围0.489792–0.704592、中位0.586338、末0.529847（clip前）。主干LR9.724869010e-5→9.663538757e-5、新参数2.917460703e-4→2.899061627e-4，按既定cosine连续下降。最终5660 first/residual CE1.375537/6.281040、grad0.473692、LR9.662108463e-5/2.898632539e-4，均有限。
- 动态batch/吞吐/等待：上述46点帧填充95.4417%–99.4958%，中位97.7188%；token填充91.9528%–97.5750%，中位95.1750%；global samples318–399。step_seconds范围2.01562–2.30180s、中位2.14616s；全局音频秒/墙钟秒824.365–939.304、中位876.810；data_wait_seconds范围0.0002462–0.0004568s、中位0.0002861s。与上轮2.15149s/875.361音频秒每秒/等待0.0002784s相近，无持续供数下降证据。以上是稀疏当步记录，不是含评估的全窗口均值；rank0训练peak allocated最高56.8110GiB，不代表各卡总占用。
- Checkpoint/val：latest=step-00005500，COMPLETE时间21:02:31.256527 UTC；progress={step:5500,epoch:0,next_batch:5500}、world_size4、scheduler.last_epoch5500、LR9.684642680e-5/2.905392804e-4符合该步。5000/5500两COMPLETE四distributed分片各约2.093GB、`.metadata`1424887字节及四rng各14613字节齐全，signature相等。4500由训练器既有keep_checkpoints=2轮转，巡检未清理或编辑metadata；结构检查不冒充实际恢复加载。5500 val first/residual CE1.485941/6.283517，较5000的1.516816/6.332561下降；15个codebook CE有限（3.990388–7.059859）。

| 5500步完整summary，每模式EN4/ZH4 | EN基础WER / CER | EN英语规范化WER / CER | ZH WER / CER | 截断 / 零帧输出 |
|---|---|---|---|---|
| speaker_only | 0.250000 / 0.118881 | 0.250000 / 0.118881 | 1.125000 / 0.654762 | 1/8 / 0/8 |
| icl | 0.055556 / 0.034965 | 0.055556 / 0.034965 | 1.000000 / 0.797619 | 1/8 / 1/8 |

- 评估核验：两summary、全部16条metrics和15个存在的WAV均实读；WAV全部24kHz单声道、数值有限、时长与metrics一致。全部目标ID/文本/speaker_reference_id与5000相同；本轮EN基础/英语规范化相等。5000→5500 EN WER speaker_only0.277778→0.25、ICL0.194444→0.055556；ZH CER分别0.773810→0.654762、0.666667→0.797619。仅8条诊断，不据汇总波动改超参；数值有限不等于音频内容有效，特别是下述全零WAV。未试听音频。
- ICL长ZH sample03本轮全零：仍400帧/32s/目标6.28s、比5.09554，未EOS且截断，ASR空/CER1。浮点读取RMS=0、peak=0、精确零比例100%，每4秒RMS全0；另用int16原始PCM复核768000个采样、非零计数0、min=max=0，文件1536044字节、PCM_16、24kHz单声道。该WAV确实全部为零，不是只有ASR漏识别，也不是frames0或文件缺失；其生成循环实际走完400帧。5000同句零比例93.5740%、RMS0.0136513，因此本轮输出能量进一步消失。现存PCM16只能证明落盘采样全零，不能证明写入前浮点严格为零或确定模型预测/codec哪一层根因。
- 同目标speaker_only有局部内容改善但仍截断：400帧/32s/比5.09554，ASR“又起了5小时,现在也下午两点午时 收收收收,关芝八十”，CER0.666667（5000为0.916667），含词替换和重复。RMS0.0192766、peak0.395142、精确零比例60.1497%（5000为96.0904%），每4秒RMS[0.035223,4.46825e-6,2.22651e-6,0.010754,0.017915,0.035992,0,0]。更多时段有能量不代表完整内容或EOS恢复，结尾仍长零段。
- ICL立即EOS/恢复：sample04连续4500/5000/5500立即EOS，本轮0.371804s、frames0、reference_frames19、目标2.3s、WER/CER1，无WAV符合该分支；是明确未解决的生成停止问题。英语sample00从5000空输出恢复19帧/1.52s/目标1.81s、比0.8398/WER0.25（省略重复the），因此总空输出2/8→1/8。sample12连续4500/5000/5500有输出，本轮48帧/3.84s/比0.7273/CER0.517241，较5000的0.172414回升，ASR“相信你跟似乎都看過我寄給你們的眷屋有videos”，内容替换/繁简仍影响结果，未误记为空输出。
- 其余逐句：sample02两模式均将yeah识别为yet，WER0.0625/CER0.0294，speaker_only4.4s/比0.917、ICL3.76s/比0.783。speaker_only sample05“TWA, Flanders-Uni.”，1.92s/比1.185/WER0.75但CER0.1538，较5000的1.3846明显下降，分词及专名仍有差异；ICL该句1.84s/比1.136/WER/CER0，sample06 3.44s/比0.966也WER/CER0。speaker_only sample06“And...Jeffreeze...into”3.84s/比1.079/WER0.25/CER0.1818。短ZH sample01分别1.04s/CER0.6667、1.12s/CER0.5，旧超长时长未复发；speaker_only sample04 2.56s/比1.113/CER0.6154、sample12 3.76s/比0.712/CER0.6552，仍有内容替换。
- 资源（21:22:24附近）：GPU0–3占用81127/77997/75925/77439 MiB（各81920），利用率99/99/99/100%；compute-apps仅本run四rank。GPU0外部余量793MiB、各卡占用与上轮相同，无OOM。主机内存已用238GiB、available1.7TiB、无swap；磁盘可用583980.03GiB（约570.3TiB），足以继续保存。
- 判断/处理/后续：保持现训练代码/配置/进程（预算6000/9000、workers16/prefetch2、LR及两epoch38539目标），训练/val与吞吐健康，尚未确认可直接修复的实现根因。ICL全零WAV和连续立即EOS不能视为已恢复；同目标/其他样本有反复，不能据8条早期ICL结果改变训练目标或用延长生成上限/禁EOS掩盖异常。先前真实输入音频、NPZ/hash、文本/同speaker配对及裁剪逻辑检查未发现明确输入故障；原始预测码、EOS/非EOS logits和解码前浮点波形仍缺，未声称复现已完成。6000评估继续核对长ZH能量/内容/停止行为和sample04/00/12。训练尚未完成，未执行最终冻结检查、未写final-verification.json；最终仍须38539 COMPLETE、最终val/两summary及PYTHONPATH=. include-speaker冻结检查实际通过。
- 命令/修改/结果：cat/tail读故障处置表、本文、manual、快照、最新巡检/status；Python解析完整train.log作finite/min/median/max/val/错误统计、读取/proc身份、checkpoint文件/metadata/signature及shutil.disk_usage；nvidia-smi GPU/compute-apps与free -h；`.venv/bin/python`用soundfile/numpy读取16条metrics和15个WAV、比较5000配对、时长/RMS/零比例/英语口径，并以int16复核ICL长ZH全零；最后只读复核5660、latest、manual、退出/验收文件。命令均成功，唯一人工写入为本段追加；未改代码/配置/数据、发信号或启停恢复，未新增timer/Codex/subagent、提交推送、发消息或删除清理。本次巡检结束。


## 2026-09-09 21:54:25 UTC 单次巡检：推进至6160步，speaker_only长中文时长恢复，ICL仍全零

- 依据/权限与进度：已读故障处置表、本文、manual、快照`supervision/20260909T215129Z.json`、最新完成巡检`20260909T212129Z.md`及status。manual仍active=false、主会话observed_through_step2040。上一快照5620/上一巡检5660 → 本轮21:51:29快照6120 → 21:51:54现场6130 → 21:52:31指标截止6150 → 21:52:57最终采样6160。6000保存、val和两生成完整，之后连续更新；6500尚未到步，不是缺失故障。
- 进程/退出：training-process.json PID621410/start_ticks80769258与/proc一致，cmdline为项目`.venv/bin/python .venv/bin/torchrun --standalone --nproc_per_node=4 -m qwen3_train.train --config configs/emilia-10kh-pretrain.yaml`。launcher621409/start80769255/PPID1、四rank621435–621438/start80769421/PPID621410匹配且存活；torchrun/launcher S、四rank R。training-exit.json不存在，完整train.log无Traceback/OOM/Non-finite/Error/Aborted。未发信号或启停恢复。
- 近半小时训练：5630–6150共53个每10步当步记录点，全部指标有限。first_ce首末1.468233→1.401084、范围1.361580–1.485819、中位1.419533；residual_ce首末6.292551→6.234668、范围6.199880–6.292551；grad_norm范围0.473692–0.711452、中位0.574185、末0.513666（clip前）。主干LR9.666390595e-5→9.588473260e-5、新参数2.899917179e-4→2.876541978e-4，既定cosine连续下降。最终6160 first/residual CE1.425695/6.210646、grad0.661378、LR9.586898483e-5/2.876069545e-4，均有限。
- 动态batch/吞吐/等待：上述53点帧填充95.4417%–99.7625%，中位98.4167%；token填充91.9528%–97.1500%，中位95.6833%；global samples312–394。step_seconds范围2.01922–2.30881s、中位2.14117s；全局音频秒/墙钟秒827.683–925.261、中位879.333；data_wait_seconds范围0.0002343–0.0004973s、中位0.0002771s。与上轮2.14616s/876.810音频秒每秒/等待0.0002861s相近，无持续供数下降证据。以上为稀疏当步值，不是含评估的全窗口均值；rank0训练peak allocated最高56.9181GiB，不代表各卡总占用。
- Checkpoint/val：latest=step-00006000，COMPLETE时间21:34:50.306065 UTC；progress={step:6000,epoch:0,next_batch:6000}、world_size4、scheduler.last_epoch6000、LR9.611750689e-5/2.883525207e-4符合该步。5500/6000两COMPLETE四distributed分片各约2.093GB、`.metadata`1424887字节及四rng各14613字节齐全，signature相等。5000由训练器既有keep_checkpoints=2轮转，巡检未清理或编辑metadata；结构检查不等于实际恢复加载。6000 val first/residual CE1.464722/6.241244，较5500的1.485941/6.283517下降；15个codebook CE有限（3.965875–7.019991）。

| 6000步完整summary，每模式EN4/ZH4 | EN基础WER / CER | EN英语规范化WER / CER | ZH WER / CER | 截断 / 零帧输出 |
|---|---|---|---|---|
| speaker_only | 0.250000 / 0.167832 | 0.250000 / 0.167832 | 1.125000 / 0.428571 | 0/8 / 0/8 |
| icl | 0.222222 / 0.111888 | 0.222222 / 0.111888 | 1.125000 / 0.797619 | 1/8 / 1/8 |

- 评估核验：两summary、16条metrics及15个存在的WAV均实读，WAV全部24kHz单声道、数值有限、时长与metrics一致；全部目标ID/文本/speaker_reference_id与5500相同。本轮EN基础与英语规范化相等。5500→6000 EN WER speaker_only0.25不变（CER0.118881→0.167832），ICL0.055556→0.222222；ZH CER分别0.654762→0.428571、0.797619不变。仅8条诊断，不由汇总波动推断整体质量或调LR；未试听音频。
- speaker_only长ZH sample03本轮明显改善：从5500的400帧/32s截断变为6000的90帧/7.2s、正常EOS且不截断，目标6.28s、时长比1.14650。ASR覆盖至“现在剩余的点掉是80%”，仍有“都七/七的/点掉”等替换，CER0.361111（5500为0.666667）。RMS0.0282205、peak0.256073、精确零比例11.5260%（5500为60.1497%）；两段RMS为0.031102（前4秒）/0.024140（后3.2秒）。追加int16检查172800采样中152883非零、最长连续精确零段仅0.03125s，故本次WAV的历史长零段未复发。此前2500也曾不截断但仍18.32s且长零段，本轮时长/连续能量有实质改善；单次改善不等于持续恢复或内容已正确。
- ICL同目标连续5500/6000全零：6000仍400帧/32s/比5.09554、无EOS且截断，ASR空/CER1；RMS0、peak0、零比例100%。int16再次核查768000采样全部0，最长连续零段32s、文件1536044字节，排除仅ASR漏识别或缺文件这一解释。生成循环走完400帧，与立即EOS空输出不同；现有PCM16证据不能进一步区分写入前极低浮点输出与模型/codec根因。speaker_only同目标和其他ICL句可产生有效能量，不能判断整套codec/ASR全局损坏。
- ICL立即EOS与其他跟踪目标：sample04连续4500/5000/5500/6000四次立即EOS，本轮耗时0.362698s、frames0、reference_frames19、目标2.3s、WER/CER1，无WAV符合生成分支，尚未恢复。sample00继5500后继续有输出，17帧/1.36s/比0.7514/WER0.5（liquid spars及重复the省略）；sample12连续4500以来有输出，6000为44帧/3.52s/比0.6667/CER0.586207，ASR“相信你个舒服都看我寄给你们的卷 微调微调”，仍有替换/重复，不误记为空。
- 其余逐句：ICL短ZH sample01 12帧/0.96s/比0.980，ASR“煮至奶白色後”，CER0.166667仅后/後差异；speaker_only同句0.96s/CER0.6667，旧超长时长未复发。sample02两模式4.16s/比0.867，speaker_only WER0.0625（yeah→and）、ICL WER0.125（另有McLasselin专名差异）。sample05 speaker_only1.52s/比0.938/WER0.75、ICL1.44s/比0.889/WER0.75，专名/分词仍不稳；sample06分别3.84s/WER0.25（and/Jeffreeze/okay插入）和3.04s/WER0.0833。speaker_only sample04 2.4s/比1.043/CER0.4615、sample12 4.32s/比0.818/CER0.4483，仍有繁简/内容差异。
- 资源（21:52:31附近）：GPU0–3占用81127/77997/75925/77439 MiB（各81920），现场利用率27/23/23/22%，快照99/99/99/99%；compute-apps仅本run四rank。GPU0外部余量793MiB、占用与前轮相同，无OOM；结合step连续推进和稳定step_seconds，瞬时利用率差异不构成停滞证据。主机内存已用238GiB、available1.7TiB、无swap；磁盘可用583949.30GiB（约570.3TiB），足以继续保存。
- 判断/处理/后续：保持现训练代码/配置/进程（预算6000/9000、workers16/prefetch2、LR及两epoch38539目标），训练/val和吞吐健康，speaker_only长ZH出现接近目标时长且无长零段的实际改善。ICL同目标全零及sample04连续EOS仍是明确未解决质量异常；先前真实输入音频、NPZ/hash、文本/配对与输入/裁剪逻辑核查未发现可直接修复的实现错误，不能仅凭早期8条ICL结果改训练目标。原始预测码、EOS/非EOS logits及解码前浮点波形仍缺，未声称完成根因复现，也未为取得诊断而另载GPU模型或打断健康训练。后续6500需确认speaker_only改善能否保持，并分别追踪ICL全零/EOS。尚未执行最终冻结检查、未写final-verification.json；最终仍须38539 COMPLETE、最终val/两summary和PYTHONPATH=. include-speaker冻结核验实际通过。
- 命令/修改/结果：cat/tail读故障处置表、本文、manual、快照、最新巡检/status；Python解析train.log作finite/min/median/max/val/错误统计、读/proc身份、checkpoint文件/metadata/signature及shutil.disk_usage；nvidia-smi GPU/compute-apps与free -h；`.venv/bin/python`用soundfile/numpy读16条metrics和15个WAV、比较5500配对/时长/RMS/零比例/英语口径，并用int16复核两模式长ZH非零采样数与最长零段；最终只读复核6160、latest、manual、退出/验收文件。命令均成功，唯一人工写入为本段追加；未改代码/配置/数据、发信号或启停恢复，未新增timer/Codex/subagent、提交推送、发消息或删除清理。本次巡检结束。


## 2026-09-09 22:23:27 UTC 单次巡检：6660步，6500评估完成

- 已读故障处置表、本文、manual、快照supervision/20260909T222129Z.json及最新巡检20260909T215129Z.md/status。manual仍active=false、主会话观察至2040。上一巡检6160→本轮快照6640→22:21:49现场6650→22:22:35现场6660；6500保存/val/两生成均完成，评估后正常推进。
- 身份：training-process.json PID621410/start_ticks80769258与/proc一致，cmdline为项目.venv/bin/python .venv/bin/torchrun --standalone --nproc_per_node=4 -m qwen3_train.train --config configs/emilia-10kh-pretrain.yaml。launcher621409/start80769255/PPID1、四rank621435–621438/start80769421/PPID621410均存活匹配。瞬时均S但GPU98–99%且step连续增加，不判卡死。training-exit.json不存在，完整train.log无Traceback/OOM/Non-finite/Error/Aborted。
- 6130–6660共54个每10步当步记录均有限：first_ce1.410872→1.405663，范围1.321250–1.464001；residual_ce6.209313→6.191740；grad_norm范围0.447032–0.661378、中位0.553451（clip前）。LR主干9.591614224e-5→9.504533582e-5、新参数2.877484267e-4→2.851360075e-4，按既定cosine下降。
- 帧/token填充中位98.3417%/95.7736%，范围94.8833–99.7542%/92.1889–98.4722%；global samples305–421。step中位2.14191s、范围1.99123–2.30462s；音频秒/墙钟秒中位879.961；数据等待中位0.0002803s、最高0.0004649s，与上轮稳定。上述为稀疏当步统计，不是含评估窗口均值。rank0 peak allocated最高56.9465GiB。
- latest=step-00006500，COMPLETE22:04:57.803771 UTC，progress step6500/epoch0/next_batch6500、world_size4、scheduler.last_epoch6500；6000/6500签名相同，四约2.093GB distributed分片、1424887字节.metadata及四14613字节rng齐全。未做实际恢复加载。5500由训练器既有轮转，巡检未清理。val first/residual CE1.452424/6.202600，继续下降；15码本CE有限3.938615–6.981762。

|6500，每模式EN4/ZH4|EN基础及英语规范化WER/CER（本轮相同）|ZH WER/CER|截断/零帧|
|---|---|---|---|
|speaker_only|0.222222/0.153846|1.125/0.392857|0/8、0/8|
|icl|0.083333/0.062937|1/0.940476|1/8、2/8|

- 实读两summary、16metrics和14WAV；配对ID/文本/参考与6000相同，WAV均24kHz单声道、有限、时长匹配，未试听。speaker_only长ZH83帧/6.64s/目标6.28s，比1.0573、EOS，CER0.25；RMS0.0418024、零比例6.2287%，前4秒/后2.64秒RMS0.043323/0.039386，延续上轮改善，但仍有词替换。
- ICL长ZH连续5500/6000/6500的32s WAV全零：400帧截断、RMS/peak0、零比例100%、ASR空/CER1。sample04连续五次立即EOS，本轮0.357241s；sample12在4500–6000有输出后本轮再次0帧EOS，0.348691s；无WAV符合生成分支。sample00有1.44s输出；英语02/05/06 WER0.0625/0/0，短ZH01仅后/後差异CER0.1667。不能将英语改善视为ICL中文恢复。
- GPU占用81127/77997/75925/77439 MiB，仅本run四rank，GPU0余793MiB；RAM available1.7TiB，磁盘583833.39GiB，无资源故障。
- 保持代码/配置/进程和两epoch38539目标：训练/val健康，8条诊断不足以调超参；ICL全零/EOS根因仍未确定，原始预测码/logits/解码前浮点证据仍缺，未声称完成复现。后续7000继续跟踪。未运行最终冻结检查或写final-verification.json。
- 命令：cat/tail、Python解析日志/快照/proc/checkpoint签名、nvidia-smi、free、shutil.disk_usage及soundfile/numpy读metrics/WAV。全部成功；唯一写入本段。未改训练配置/代码/数据、启停恢复、发信号、启动timer/subagent、提交推送发消息或清理。本次结束。


## 2026-09-09 22:54:37 UTC 单次巡检：7230步，7000步ICL长中文恢复部分能量但仍有8.835秒连续零段

- 依据/进度：已读故障处置表、本文、manual、当前快照`supervision/20260909T225129Z.json`、最新巡检`20260909T222129Z.md`与status。该轮独立回复仅为约定确认，具体检查证据以本文22:23:27记录为准。manual仍active=false、主会话observed_through_step2040。上一现场6660/上一快照6640 → 本轮22:51:29快照7180 → 22:52:15现场7200 → 22:53:11最终采样7230。7000 checkpoint/val/两生成已完成，随后连续更新；7500尚未到步，不是缺失故障。
- 进程/退出：training-process.json PID621410/start_ticks80769258与/proc精确匹配，cmdline为项目`.venv/bin/python .venv/bin/torchrun --standalone --nproc_per_node=4 -m qwen3_train.train --config configs/emilia-10kh-pretrain.yaml`；launcher621409/start80769255/PPID1、四rank621435–621438/start80769421/PPID621410均匹配存活。现场torchrun/launcher S、四rank R。training-exit.json不存在，完整train.log无Traceback/OutOfMemory/out of memory/Non-finite/Error/Aborted匹配行。未发信号或启停恢复。
- 近半小时指标：6650–7200共56个每10步当步记录点全部有限。first_ce首末1.430656→1.376896、范围1.323934–1.436198、中位1.386315；residual_ce首末6.143091→6.180426、范围6.086693–6.214174、中位6.167648；grad_norm范围0.452626–0.673384、中位0.535055（clip前）。主干LR9.506250086e-5→9.407709646e-5、新参数2.851875026e-4→2.822312894e-4，既定cosine连续下降。最终7230 first/residual CE1.339928/6.143281、grad0.505706、LR9.402094672e-5/2.820628402e-4、step2.109726s，均有限。
- 动态batch/吞吐：上述56点帧填充94.9167%–99.7875%、中位98.3375%；token填充92.0111%–97.3250%、中位95.4056%；global samples317–386。step_seconds中位2.12509s、范围1.98147–19.32972s；音频秒/墙钟秒中位879.681、范围98.692–951.334；data_wait_seconds中位0.0002754s、范围0.0002299–0.0005068s。rank0训练peak allocated最高57.1468GiB。以上为稀疏当步值，不是包含评估的全窗口均值。
- 慢步单独核查：6730为19.329721s、吞吐98.6915、data_wait仅0.0002299s、loss/grad有限；6710/6720为2.12056/2.06185s，6740/6750为2.00374/2.23710s。窗口其余55点最高2.30291s，中位及供数与上轮2.14191s/879.961音频秒每秒/等待0.0002803s相近。该步不在500步保存评估边界，不能直接归因评估；日志不足以定位单次延迟源，也不能由rank0等待小排除其他rank延迟。因后续正常且未持续三个记录点恶化，不调workers或重启，后续跟踪是否复发。
- Checkpoint/val：latest=step-00007000，COMPLETE时间22:35:14.882673 UTC；progress={step:7000,epoch:0,next_batch:7000}、world_size4、scheduler.last_epoch7000、LR9.444511634e-5/2.833353490e-4。6500/7000两COMPLETE签名相等，四distributed分片各约2.093GB、`.metadata`1424887字节及四rng各14613字节齐全；结构核查不冒充实际恢复加载。6000由训练器既有keep_checkpoints=2轮转，巡检未清理或编辑metadata。7000 val first/residual CE1.434841/6.167901，较6500的1.452424/6.202600继续下降；15个码本CE均有限（3.921649–6.946683）。

|7000步，每模式EN4/ZH4|EN基础WER/CER|EN英语规范化WER/CER|ZH WER/CER|截断/零帧|
|---|---|---|---|---|
|speaker_only|0.305556/0.160839|0.305556/0.160839|1.125/0.452381|0/8、0/8|
|icl|0.277778/0.195804|0.250000/0.188811|1/0.880952|0/8、3/8|

- 实读两summary、16条metrics及13个存在的WAV；WAV均24kHz单声道、有限且时长匹配，目标ID/文本/speaker_reference_id/reference_text与6500一致。未试听。ICL英语基础与规范化差异来自sample02 ASR“they're”展开为“they are”：该句WER/CER0.125/0.044118→0.0625/0.029412，仍有there/they差异；不混用两种口径。6500→7000 EN规范化WER speaker_only0.222222→0.305556、ICL0.083333→0.25；ZH CER0.392857→0.452381、0.940476→0.880952。仅8条诊断，不据此推断全体验证质量或改LR。
- speaker_only长ZH sample03连续6000/6500/7000不截断：本轮86帧/6.88s、目标6.28s、比1.09554、正常EOS；CER0.444444（6500为0.25），ASR有“取/骑”“先是/现在是”“先生”等替换及阿拉伯数字/汉字口径差异。RMS0.0368747、peak0.3203125、精确零比例10.7607%；int16非零147352/165120，最长连续零段0.018667s，前4秒/后2.88秒RMS0.040649/0.030876。时长和连续能量保持恢复，但内容仍未正确。
- ICL同长ZH从6500全零32s改为147帧/11.76s、比1.87261、EOS且不截断；ASR“这大概是公里县生鱼的电量是80%”、CER0.777778，有局部内容但大段缺失。RMS0.0243877、peak0.423706、零比例75.8507%；int16非零68159/282240，最长连续零段8.835s，前两个4秒段RMS精确0、末3.76秒RMS0.0431302。因此不再全零或触及max_frames不等于异常解决；长零段仍明确存在。现存PCM16无法进一步判定写入前浮点或预测码/codec哪一层根因。
- ICL立即EOS数2→3：sample04连续4500至7000六次0帧，本轮0.354181s、reference_frames19、目标2.3s；sample12继6500后仍0帧、0.351013s、reference_frames80、目标5.28s；EN sample00在5500–6500有输出后本轮再次0帧、0.362268s、reference_frames36、目标1.81s。三者WER/CER1且无WAV符合生成分支，不能靠重评分ASR恢复内容。短ZH sample01仍有0.88s/比0.898输出，但CER0.666667（6500为0.166667），ASR“組織奶白伺候”；speaker_only同句0.96s/CER1。ICL sample05/06分别1.84s/WER0.5和3.6s/WER0.1667，仍有专名/分词/词替换。speaker_only sample00插词使WER1；sample02 WER0.0625（which→Twitch），sample04 2.48s/CER0.4615，sample12 4.16s/比0.7879/CER0.3448并有末段缺失。
- 资源：GPU0–3占用81127/77997/75925/77439 MiB（各81920），现场利用率49/26/25/21%，compute-apps仅本run四rank；GPU0外部余量793MiB，与前轮一致且无OOM。主机已用236GiB、available1.7TiB、无swap；22:52:15磁盘可用583741.37GiB，足以继续保存。结合连续step与吞吐，单次利用率不判卡死。
- 判断/处理/未解决：保持当前代码/配置/进程（预算6000/9000、workers16/prefetch2、LR、两epoch38539目标），训练/val健康且单次慢步已恢复。ICL长零段与立即EOS仍是未解决质量异常，先前真实输入/NPZ/hash/配对/输入裁剪核查未证实可直接修复的实现错误；本轮能量有变化但不称根因已定位。原始预测codec、EOS/非EOS logits、解码前浮点波形仍缺，未为收集证据另载GPU模型或打断训练。后续7500继续分别核对长零段、三个EOS目标及慢步复发。训练尚未完成，未执行最终冻结检查、未写final-verification.json；最终仍须38539 COMPLETE、最终val/两summary和`PYTHONPATH=. ...check_frozen_frontend.py --include-speaker`实际通过。
- 命令/修改/结果：cat/tail/ls读约定、manual、快照、巡检/status、进程记录和metrics；Python解析完整日志作finite/min/median/max/错误及慢步上下文检查、读/proc和两checkpoint文件/metadata/signature、datetime与shutil.disk_usage；nvidia-smi GPU/compute-apps、free -h；`.venv/bin/python`用soundfile/numpy核查16metrics/13WAV、配对、时长、RMS/零比例及两长ZH int16最长零段；最后只读复核7230、manual、latest、退出/验收文件。命令均成功；唯一人工写入为本段追加。未改训练代码/配置/数据、发信号或启停恢复，未启动timer/Codex/subagent、提交推送、发消息或删除清理。本次巡检结束。


## 2026-09-09 23:24:10 UTC 单次巡检：7730步，7500步ICL长中文再次全零，训练与供数稳定

- 依据/进度：已读故障处置表、本文、manual、快照`supervision/20260909T232129Z.json`、最新巡检`20260909T225129Z.md`及status（上轮review_exit_code0）。manual仍active=false，主会话observed_through_step2040。上一现场7230/上一快照7180 → 本轮23:21:30快照7700 → 23:22:12现场7720 → 23:22:47最终采样7730。7500 checkpoint、val和两生成均已完成，之后正常更新；8000尚未到步，不是产物缺失故障。
- 进程/退出：training-process.json PID621410/start_ticks80769258与/proc精确匹配，cmdline为项目`.venv/bin/python .venv/bin/torchrun --standalone --nproc_per_node=4 -m qwen3_train.train --config configs/emilia-10kh-pretrain.yaml`。launcher621409/start80769255/PPID1、四rank621435–621438/start80769421/PPID621410均匹配存活。现场均S，但GPU98–99%且step持续增加，不判停滞。training-exit.json不存在；完整train.log无Traceback/OutOfMemory/out of memory/Non-finite/Error/Aborted匹配行。未发信号或启停恢复。
- 半小时训练：7190–7720共54个每10步当步记录，所有指标有限。first_ce首末1.436198→1.333740、范围1.295483–1.439324、中位1.371725；residual_ce首末6.154286→6.082571、范围6.071592–6.188478、中位6.131229；grad_norm范围0.467737–0.647680、中位0.533026（clip前）。主干LR9.409575833e-5→9.306927761e-5、新参数2.822872750e-4→2.792078328e-4，按既定cosine连续下降。最终7730 first/residual CE1.373110/6.157051、grad0.603680、LR9.304918368e-5/2.791475510e-4，均有限。
- 动态batch/吞吐/等待：上述54点帧填充95.0792%–99.8208%、中位97.7729%；token填充92.2778%–98.1361%、中位95.2167%；global samples317–404。step_seconds范围2.04817–2.44447s、中位2.14193s；音频秒/墙钟秒778.804–913.204、中位880.317；data_wait_seconds范围0.0002278–0.0004840s、中位0.0002739s。rank0训练peak allocated最高56.7731GiB，不代表各rank总占用。与上一窗口中位2.12509s/879.681音频秒每秒/等待0.0002754s相近；本轮记录无>3s慢步，6730的19.33s异常未在此窗口记录点复发。以上为稀疏当步统计，不是含评估的全窗口均值，不据此排除未记录步的瞬时延迟。
- Checkpoint/val：latest=step-00007500，COMPLETE时间23:02:54.944613 UTC；progress={step:7500,epoch:0,next_batch:7500}、world_size4、scheduler.last_epoch7500、LR9.350457354e-5/2.805137206e-4。7000/7500两COMPLETE签名相等，四distributed分片各约2.093GB、`.metadata`1424887字节、四rng各14613字节齐全。结构核查不冒充实际恢复加载。6500由训练器既有keep_checkpoints=2轮转，巡检未清理或改metadata。7500 val first/residual CE1.420105/6.134384，较7000的1.434841/6.167901继续下降；15个码本CE均有限，范围3.894393–6.915975。

|7500步，每模式EN4/ZH4|EN基础WER/CER|EN英语规范化WER/CER|ZH WER/CER|截断/零帧|
|---|---|---|---|---|
|speaker_only|0.361111/0.237762|0.361111/0.237762|1.125/0.500000|0/8、0/8|
|icl|0.194444/0.181818|0.194444/0.181818|1/0.940476|1/8、3/8|

- 评估核验：实读两summary、16条metrics和13个存在的WAV，所有WAV均24kHz单声道、有限、时长与metrics一致；目标ID/文本/speaker_reference_id/reference_text与7000一致。本轮两英语口径相同。7000→7500 EN规范化WER speaker_only0.305556→0.361111、ICL0.25→0.194444；ZH CER0.452381→0.5、0.880952→0.940476。只有8条诊断，不把汇总波动当整体质量结论或调LR依据；未试听音频。
- ICL长ZH sample03再次400帧/32s/目标6.28s、比5.09554，未EOS且截断，ASR空/CER1。RMS/peak0、精确零比例100%；int16复核768000个采样非零0、最长零段32s，八个4秒段RMS全0。上轮147帧/11.76s且部分能量没有保持；这次与5500–6500全零表现一致，不能解释为仅ASR漏识别，也不同于0帧立即EOS。现有PCM16仍不足以确定预测码或codec层根因，不声称写入前浮点严格全零。
- speaker_only同长ZH连续6000/6500/7000/7500不截断：92帧/7.36s、比1.17197、正常EOS；CER0.361111（7000为0.444444）。RMS0.039853、peak0.428894、零比例8.5281%；int16非零161576/176640、最长连续零段0.004583s，前4秒/后3.36秒RMS0.044392/0.033660。ASR“尤其的半个小时现在是下午两点五时 起了大概十公里 现生鱼的电量是80%”，仍有词替换与数字口径差异，但历史长零段未复发。
- ICL立即EOS三个目标未恢复：sample04连续4500–7500七次0帧，本轮0.365957s/reference_frames19/目标2.3s；sample12连续6500/7000/7500为0帧，本轮0.359803s/reference_frames80/目标5.28s；EN sample00连续7000/7500为0帧，本轮0.356054s/reference_frames36/目标1.81s。均WER/CER1，无WAV符合生成分支。ICL另一个ZH sample01有1.04s/比1.061输出，CER0.166667，仅“后/後”差异；不能由短句恢复推断长句/EOS恢复。
- 其余逐句及汇总来源：speaker_only EN sample05本轮2.64s/目标1.62s、比1.62963，ASR“Cheetah Debbie 2, I flun this you-nay.”，WER2/CER1.692308，共8/13个EN词错误及22/34个EN字符错误，主导本轮EN汇总退化；其余sample02为4.72s/WER/CER0，sample00为1.76s/WER0.5，sample06为3.6s/WER0.25。ICL sample02 4.24s/WER0.0625（省略like），sample05 1.6s/WER0.25（Flendis专名），sample06 3.76s/WER0.0833（swamp→swap）。speaker_only sample12从4.16s/CER0.3448变为3.6s/比0.6818/CER0.6897，ASR缺失video末段；sample04 2.56s/CER0.4615仍有见识/坚持与繁简差异，sample01 1.04s/CER0.5。保留这些具体变化，不以总体均值掩盖异常。
- 资源：GPU0–3占用81127/77997/75925/77439 MiB（各81920），利用率99/99/98/98%；compute-apps仅本run四rank，GPU0外部余量793MiB，与前轮一致且无OOM。主机已用236GiB、available1.7TiB、无swap；23:22:12磁盘可用583671.94GiB，足以继续保存。
- 判断/处理/未解决：保持现有训练代码/配置/进程（预算6000/9000、workers16/prefetch2、LR和两个epoch38539目标），因为训练/val与供数健康，当前未证实需修改实现或重启的运行故障。ICL长ZH全零复发和三个提前EOS仍未解决；既有输入音频/NPZ/hash/配对与裁剪核查没有确认可直接修复的实现问题，本轮波形检查未完成原始codec预测、EOS/非EOS logits或解码前浮点根因定位，未另载GPU模型或打断健康训练。后续8000继续跟踪上述异常及speaker_only sample05/12退化能否保持；不据这8条改变训练目标。尚未执行最终冻结检查、未写final-verification.json；最终仍须38539 COMPLETE、最终val/两summary及`PYTHONPATH=. ...check_frozen_frontend.py --include-speaker`实际通过才可验收。
- 命令/修改/结果：cat/tail读取约定、manual、最新巡检/status；Python解析当前快照和完整train.log作finite/min/median/max/慢步与错误扫描、/proc身份、checkpoint文件/metadata/signature、datetime/shutil.disk_usage；nvidia-smi GPU/compute-apps、free -h；`.venv/bin/python`用soundfile/numpy读取16metrics和13WAV，对照7000配对/英语口径、时长/RMS/零比例及两长ZH int16最长零段；最后复核7730、manual、latest、退出/验收文件。命令均成功；唯一人工写入是本段追加，未改代码/配置/数据、发信号或启停恢复，未启动timer/Codex/subagent、提交推送、发消息或删除清理。本次巡检结束。


## 2026-09-09 23:54:31 UTC 单次巡检：8310步，8000步ICL长中文恢复接近目标时长与有效能量

- 依据/进度：已读故障处置表、本文、manual、快照`supervision/20260909T235129Z.json`、最新巡检`20260909T232129Z.md`及status（上轮退出码0、固定session不变）。manual仍active=false，主会话observed_through_step2040。上一现场7730/上一快照7700 → 本轮23:51:30快照8270 → 23:52:29现场8300 → 23:53:05最终采样8310。8000 checkpoint、val、双模式summary完整，之后正常更新；8500尚未到步，不是保存或评估缺失。
- 进程/退出：training-process.json PID621410/start_ticks80769258与/proc精确匹配，cmdline为项目`.venv/bin/python .venv/bin/torchrun --standalone --nproc_per_node=4 -m qwen3_train.train --config configs/emilia-10kh-pretrain.yaml`。launcher621409/start80769255/PPID1、四rank621435–621438/start80769421/PPID621410均匹配存活。采样均S但GPU99–100%且step连续推进，不判断为卡死。training-exit.json不存在，完整train.log无Traceback/OutOfMemory/out of memory/Non-finite/Error/Aborted匹配行。未发信号或启停恢复。
- 近半小时训练：7710–8300共60个每10步当步记录，全部指标有限。first_ce首末1.364031→1.318221、范围1.294704–1.409667、中位1.355627；residual_ce首末6.163629→6.065195、范围6.059798–6.163629、中位6.107248；grad_norm范围0.442441–0.603680、中位0.512547（clip前）。主干LR9.308934487e-5→9.186022186e-5、新参数2.792680346e-4→2.755806656e-4，按既定cosine连续下降。最终8310 first/residual CE1.316397/6.039712、grad0.488608、LR9.183860616e-5/2.755158185e-4、step2.197387s，均有限。
- 动态batch/吞吐/供数：上述60点帧填充95.1083%–99.7417%、中位98.1542%；token填充91.9944%–97.7250%、中位95.6319%；global samples310–397。step_seconds范围2.01773–2.34700s、中位2.14669s；音频秒/墙钟秒806.648–935.786、中位879.576；data_wait_seconds范围0.0002359–0.0004172s、中位0.0002805s。与上轮2.14193s/880.317音频秒每秒/等待0.0002739s相近；本窗口记录无>3s慢步，6730慢步未在此窗口记录点复发。rank0训练peak allocated最高56.9171GiB，不代表各rank总占用。以上为稀疏当步统计，不是包含评估的全窗口均值。
- Checkpoint/val：latest=step-00008000，COMPLETE时间23:32:13.422690 UTC；progress={step:8000,epoch:0,next_batch:8000}、world_size4、scheduler.last_epoch8000、LR9.249662090e-5/2.774898627e-4。7500/8000两COMPLETE签名相等，四distributed分片各约2.093GB、`.metadata`1424887字节、四rng各14613字节齐全。结构检查不等于实际恢复加载；7000由训练器既有keep_checkpoints=2轮转，巡检未清理或编辑metadata。8000 val first/residual CE1.412018/6.108515，较7500的1.420105/6.134384继续下降；15个码本CE有限，范围3.882938–6.890419。

|8000步，每模式EN4/ZH4|EN基础WER/CER|EN英语规范化WER/CER|ZH WER/CER|截断/零帧|
|---|---|---|---|---|
|speaker_only|0.222222/0.132867|0.222222/0.132867|1.125/0.416667|0/8、0/8|
|icl|0.138889/0.146853|0.138889/0.146853|1/0.785714|0/8、3/8|

- 评估核验：实读两summary、16条metrics及13个存在的WAV；WAV均24kHz单声道、有限、时长与metrics一致；目标ID/文本/speaker_reference_id/reference_text与7500一致。本轮两英语口径相同；7500→8000 EN规范化WER speaker_only0.361111→0.222222、ICL0.194444→0.138889；ZH CER0.5→0.416667、0.940476→0.785714。仅8条诊断，不能把汇总改善当作全验证集质量结论或异常根因已解决。未试听。
- ICL长ZH sample03本轮实质改善：从7500的400帧/32s全零变为88帧/7.04s、目标6.28s、比1.12102、正常EOS且不截断；ASR覆盖目标前后内容，为“又起了半個小時,現在是下午2點50,起了大概10公里,現在生魚的電量是80%”，CER0.555556，仍有同音词替换、繁简及数字口径差异。RMS0.0408770、peak0.292542、零比例14.7414%；int16非零144053/168960，最长连续零段0.705125s位于开头[0,0.705125]，第二长仅0.049833s位于[3.265167,3.315]，前4秒/后3.04秒RMS0.042467/0.038685。相比7000虽不截断却有8.835s零段，本轮时长及能量分布有进一步改善；单次变化不能确认后续不会复发，也未定位此前全零的生成/codec根因。
- speaker_only同目标连续6000至8000五次不截断：99帧/7.92s、比1.26115、EOS、CER0.416667（7500为0.361111）。RMS0.0382779、peak0.434052、零比例3.8258%；int16非零182808/190080、最长零段0.007625s，前4秒/后3.92秒RMS0.045233/0.029538。ASR有“其他大概是/最余”等词缺失或替换，内容仍不完整；历史长零段未复发。
- ICL三个立即EOS目标未恢复：sample04连续4500–8000八次0帧，本轮0.362377s、reference_frames19、目标2.3s；sample12连续6500–8000四次0帧，本轮0.364339s、reference_frames80、目标5.28s；EN sample00连续7000–8000三次0帧，本轮0.359599s、reference_frames36、目标1.81s。均WER/CER1，无WAV符合生成分支，不当成ASR/落盘错误。ICL EN汇总5个词错误中的4个来自sample00全删除；其余sample02/05本轮WER/CER均0，sample06仅Jeffreeze专名差异WER0.0833，不能让这三条的改善掩盖sample00空输出。
- 其他跟踪：speaker_only sample05从2.64s/比1.6296/WER2/CER1.6923变为1.84s/比1.1358/WER0.75/CER0.6923，ASR“Tibi Ue, Flintis Uni.”，长时长和插入减轻但专名仍错；sample12从3.6s/CER0.6897变为3.84s/比0.7273/CER0.2414，ASR已有“videos啊videos”末段但仍有替换/插词。sample02两模式分别4.64/4.400042s，均WER/CER0；speaker_only sample00 2.16s/WER0.5、sample06 3.6s/WER0.25。短ZH sample01 speaker_only0.96s/CER1、ICL1.04s/CER0.6667，仍有同音替换；speaker_only sample04 2.48s/CER0.5385，有“監視了賬目多年”等内容差异。
- 资源：GPU0–3占用81127/77997/75927/77439 MiB（各81920），利用率99/99/99/100%，compute-apps仅本run四rank；GPU0外部余量793MiB，与前轮一致，无OOM。主机已用236GiB、available1.7TiB、无swap；23:52:29磁盘可用583564.125GiB，足以继续保存。
- 判断/处理/未解决：保持现训练代码/配置/进程（预算6000/9000、workers16/prefetch2、LR、两epoch38539目标），训练/val与供数正常，且长ZH本轮能量明显改善。三个ICL立即EOS仍持续，原始codec预测、EOS/非EOS logits及解码前浮点证据仍缺；既有输入/配对/裁剪检查未确认可直接修复的实现错误，本轮不声称已完成根因复现，也未另载GPU模型或打断健康训练。8500继续验证长ZH改善是否保持及三个EOS目标，不能因8条波动改变训练任务或LR。训练尚未完成，未执行最终冻结检查、未写final-verification.json；最终仍须38539 COMPLETE、最终val/双summary及`PYTHONPATH=. ...check_frozen_frontend.py --include-speaker`实际通过。
- 命令/修改/结果：cat/tail读取约定、manual、最新巡检/status；Python解析快照和完整日志作finite/min/median/max/慢步/错误扫描、/proc身份、两checkpoint文件/metadata/signature、datetime/shutil.disk_usage；nvidia-smi GPU/compute-apps、free -h；`.venv/bin/python`用soundfile/numpy读取16metrics和13WAV，对照7500配对/英语口径及RMS/时长/零比例，两长ZH int16最长零段并补查ICL零段位置；最后复核8310、manual、latest、退出/验收文件。均成功；唯一人工写入为本段追加。未改代码/配置/数据、发信号或启停恢复，未启动timer/Codex/subagent、提交推送、发消息或删除清理。本次巡检结束。


## 2026-09-10 00:24:51 UTC 单次巡检：8850步，8500步ICL英语空输出恢复，长中文15.75秒开头零段复发

- 依据/进度：已读故障处置表、本文、manual、快照`supervision/20260910T002129Z.json`、最新巡检`20260909T235129Z.md`及status（上轮退出码0、固定session不变）。manual仍active=false，主会话observed_through_step2040。上一现场8310/上一快照8270 → 本轮00:21:29快照8790 → 00:22:41现场8830 → 00:23:25最终采样8850。8500 checkpoint/val/两summary完整，之后继续更新；9000尚未到步，不是产物缺失故障。
- 进程/退出：training-process.json PID621410/start_ticks80769258与/proc精确一致，cmdline为项目`.venv/bin/python .venv/bin/torchrun --standalone --nproc_per_node=4 -m qwen3_train.train --config configs/emilia-10kh-pretrain.yaml`；launcher621409/start80769255/PPID1、四rank621435–621438/start80769421/PPID621410均匹配存活。现场均S但GPU均99%且step持续增加，不判停滞。training-exit.json不存在，完整train.log无Traceback/OutOfMemory/out of memory/Non-finite/Error/Aborted匹配行。未发信号或启停恢复。
- 近半小时训练：8280–8830共56个每10步当步记录均有限。first_ce首末1.387033→1.359505、范围1.283357–1.390756、中位1.341007；residual_ce首末6.098785→6.058897、范围6.027755–6.126024、中位6.075035；grad_norm范围0.394212–0.640365、中位0.486395（clip前）。主干LR9.190337579e-5→9.067939628e-5、新参数2.757101274e-4→2.720381888e-4，既定cosine连续下降。最终8850 first/residual CE1.320933/6.066902、grad0.520056、LR9.063344760e-5/2.719003428e-4，均有限。
- 动态batch/吞吐/供数：上述56点帧填充95.2208%–99.7292%、中位98.3396%；token填充92.0833%–97.9694%、中位95.7653%；global samples318–405。step_seconds范围2.01440–2.40544s、中位2.15803s；音频秒/墙钟秒787.550–935.148、中位870.971；data_wait_seconds范围0.0002491–0.0004571s、中位0.0002755s。与前轮2.14669s/879.576音频秒每秒/等待0.0002805s相近，无持续供数恶化；记录无>3s慢步。rank0训练peak allocated最高56.9171GiB，不代表各rank总占用。以上为稀疏当步值，不是含评估的全窗口均值。
- Checkpoint/val：latest=step-00008500，COMPLETE时间2026-09-09 23:59:40.542404 UTC；progress={step:8500,epoch:0,next_batch:8500}、world_size4、scheduler.last_epoch8500、LR9.142302304e-5/2.742690691e-4。8000/8500两COMPLETE签名相等，四distributed分片各约2.093GB、`.metadata`1424887字节、四rng各14613字节齐全。结构核查不冒充实际恢复加载；7500由训练器既有keep_checkpoints=2轮转，巡检未清理或编辑metadata。8500 val first/residual CE1.398912/6.080750，较8000的1.412018/6.108515继续下降；15个码本CE有限，范围3.854830–6.862309。

|8500步，每模式EN4/ZH4|EN基础WER/CER|EN英语规范化WER/CER|ZH WER/CER|截断/零帧|
|---|---|---|---|---|
|speaker_only|0.222222/0.132867|0.222222/0.132867|1.125/0.571429|0/8、0/8|
|icl|0.111111/0.055944|0.111111/0.055944|1.25/0.916667|0/8、2/8|

- 评估核验：实读两summary、16条metrics及14个存在的WAV；全部WAV为24kHz单声道、有限且时长匹配。目标ID/文本/speaker_reference_id/reference_text与8000一致。本轮英语基础与规范化口径相同；8000→8500 speaker_only EN WER/CER不变、ZH CER0.416667→0.571429；ICL EN WER0.138889→0.111111/CER0.146853→0.055944，ZH CER0.785714→0.916667。仅8条诊断，不据汇总波动判断全体验证质量或调LR；未试听。
- ICL长ZH sample03改善未保持：从8000的88帧/7.04s变为264帧/21.12s、目标6.28s、比3.36306，仍EOS且不截断，但RMS0.0226384、peak0.418732、零比例74.8209%；int16非零127628/506880，最长连续零段位于开头[0,15.75375]秒，第二长仅0.006375s。每4秒RMS约[0,0,0,0.019472,0.046128,0.026663]（末段1.12s）。8000最长开头零段仅0.705125s，本轮重新大幅延长；因此0截断率不等于恢复。ASR“上升後5m IOS SOS現在設計當點57.3M質 但往前10公里 現在剩餘的電量是80%”，WER1.5/CER0.944444，内容错误与插入明显。不是全零WAV或立即EOS，也不能归因只有ASR漏识别；预测码/codec/写入前浮点根因仍未确定。
- speaker_only同长ZH连续6000至8500六次不截断，88帧/7.04s/比1.12102、EOS；RMS0.0416431、零比例5.2012%，int16非零160172/168960，最长零段0.036125s位于尾部，前4秒/后3.04秒RMS0.041099/0.042349。历史长零段未复发，但CER0.555556（8000为0.416667），ASR“就其他棒格石现在是下午两点五时 其他大概10公里 现身顺域的电量是80%”，仍有明显词替换。
- ICL零帧数3→2：EN sample00在7000/7500/8000立即EOS后本轮恢复18帧/1.44s/比0.79558，RMS0.0560278，ASR“The liquid spears.”，WER0.25/CER0.166667，仅省略重复the；不能保证后续稳定。ZH sample04连续4500至8500九次0帧，本轮0.364797s/reference_frames19/目标2.3s；sample12连续6500至8500五次0帧，本轮0.368091s/reference_frames80/目标5.28s。二者WER/CER1且无WAV符合生成分支，仍未恢复。
- 其余跟踪：sample02两模式分别4.64s/4.080042s，WER/CER均0；ICL短ZH sample01 0.96s/比0.9796/CER0.166667（煮均/煮至），speaker_only同句0.96s/CER0.5。speaker_only EN sample05 2.24s/比1.3827、WER1.25/CER0.846154（8000为0.75/0.6923），ASR“T-puture, F-line this uni.”，仍有专名和插入错误；sample06改善到3.6s/WER0.0833（In→And），抵消sample05退化，使EN汇总相同。ICL sample05 1.52s/WER0.5（W.F. Lenders-Uni），sample06 3.84s/WER0.0833（Jeffreeze）。speaker_only sample04 2.16s/CER0.6923，sample12 4.88s/比0.9242/CER0.5517、ASR含videos三次及词替换；更接近目标时长不代表内容更好。
- 资源：GPU0–3占用81127/77997/75927/77439 MiB（各81920），利用率均99%，compute-apps仅本run四rank；GPU0外部余量793MiB，无OOM。主机已用238GiB、available1.7TiB、无swap；00:22:41磁盘可用583520.76GiB，足以继续保存。
- 判断/处理/未解决：保持当前代码/配置/进程（预算6000/9000、workers16/prefetch2、LR及两epoch38539目标），训练/val和供数健康，没有证实需改实现或重启的运行故障。ICL长零段复发与两条中文立即EOS仍是未解决质量异常；既有输入音频/NPZ/hash/配对/裁剪核查未确认可直接修复的实现错误，原始预测codec、EOS/非EOS logits与解码前浮点证据仍缺，本轮不声称已根因复现，也未另载GPU模型或打断训练。后续9000继续看sample00恢复能否保持、长零段与两个EOS目标；仅8条结果不改变训练目标。未运行最终冻结检查、未写final-verification.json；最终仍须38539 COMPLETE、最终val/双summary及`PYTHONPATH=. ...check_frozen_frontend.py --include-speaker`实际通过。
- 命令/修改/结果：cat/tail读约定、manual、最新巡检/status；Python解析当前快照和完整日志作finite/min/median/max/慢步/错误扫描、读/proc身份、两checkpoint文件/metadata/signature、datetime/shutil.disk_usage；nvidia-smi GPU/compute-apps与free -h；`.venv/bin/python`用soundfile/numpy读取16metrics/14WAV，对照8000配对/英语口径、时长/RMS/零比例及两长ZH int16最长零段位置；最后复核8850、manual、latest、退出/验收文件。均成功；唯一人工写入为本段追加。未改代码/配置/数据、发信号或启停恢复，未启动timer/Codex/subagent、提交推送、发消息或删除清理。本次巡检结束。


## 2026-09-10 00:54:37 UTC 单次巡检：9360步，9000步ICL sample12恢复，长中文仍有10.435秒开头零段

- 依据/进度：已读故障处置表、本文、manual、快照`supervision/20260910T005129Z.json`、最新巡检`20260910T002129Z.md`及status（上轮退出码0、固定session不变）。manual仍active=false，主会话observed_through_step2040。上一现场8850/上一快照8790 → 本轮00:51:29快照9310 → 00:52:20现场9340 → 00:53:05最终采样9360。9000 checkpoint/val/两summary完整，之后持续更新；9500尚未到步，不是保存评估缺失。
- 进程/退出：training-process.json PID621410/start_ticks80769258与/proc精确一致，cmdline为项目`.venv/bin/python .venv/bin/torchrun --standalone --nproc_per_node=4 -m qwen3_train.train --config configs/emilia-10kh-pretrain.yaml`。launcher621409/start80769255/PPID1、四rank621435–621438/start80769421/PPID621410均匹配存活。现场均S，但GPU98–99%且step增加，不判停滞。training-exit.json不存在，完整train.log无Traceback/OutOfMemory/out of memory/Non-finite/Error/Aborted匹配行。未发信号或启停恢复。
- 近半小时训练：8800–9340共55个每10步当步记录，全部指标有限。first_ce首末1.362235→1.346877、范围1.275145–1.402784、中位1.329215；residual_ce首末6.075331→6.003974、范围5.998589–6.092394、中位6.054770；grad_norm范围0.409245–0.574997、中位0.489612（clip前）。主干LR9.074813182e-5→8.947684093e-5、新参数2.722443955e-4→2.684305228e-4，既定cosine连续下降。最终9360 first/residual CE1.345588/6.073481、grad0.470214、LR8.942838762e-5/2.682851629e-4、step2.101856s，均有限。
- 动态batch/吞吐/供数：上述55点帧填充95.1542%–99.6125%、中位98.2083%；token填充92.5%–97.6167%、中位95.3944%；global samples315–387。step_seconds范围1.99600–2.33872s、中位2.14031s；音频秒/墙钟秒817.609–935.148、中位878.979；data_wait_seconds范围0.0002337–0.0004984s、中位0.0002760s。与前轮2.15803s/870.971音频秒每秒/等待0.0002755s相近，记录无>3s慢步，无持续供数恶化。rank0训练peak allocated最高56.6985GiB，不代表各rank总占用；以上为稀疏当步值，不是含评估的全窗口均值。
- Checkpoint/val：latest=step-00009000，COMPLETE时间00:29:03.306928 UTC；progress={step:9000,epoch:0,next_batch:9000}、world_size4、scheduler.last_epoch9000、LR9.028565950e-5/2.708569785e-4。8500/9000两COMPLETE签名相等，四distributed分片各约2.093GB、`.metadata`1424887字节、四rng各14613字节齐全。结构核查不等于实际恢复加载；8000由训练器既有keep_checkpoints=2轮转，巡检未清理或改metadata。9000 val first/residual CE1.393668/6.064613，较8500的1.398912/6.080750继续下降；15个码本CE有限，范围3.853430–6.842832。

|9000步，每模式EN4/ZH4|EN基础WER/CER|EN英语规范化WER/CER|ZH WER/CER|截断/零帧|
|---|---|---|---|---|
|speaker_only|0.277778/0.195804|0.277778/0.195804|1.125/0.476190|0/8、0/8|
|icl|0.250000/0.167832|0.250000/0.167832|1.125/0.666667|0/8、2/8|

- 评估核验：实读两summary、16条metrics及14个存在的WAV，均24kHz单声道、有限且时长与metrics匹配；目标ID/文本/speaker_reference_id/reference_text与8500一致。本轮英语基础/规范化口径相同；8500→9000 EN WER speaker_only0.222222→0.277778、ICL0.111111→0.25；ZH CER0.571429→0.476190、0.916667→0.666667。仅8条诊断，不据此判断全验证集质量或调LR。未试听。
- ICL长ZH sample03：260帧/20.8s、目标6.28s、比3.31210、EOS且不截断；RMS0.0267509、peak0.370819、零比例52.9499%，int16非零234874/499200，最长零段仍位于开头[0,10.435]秒。每4秒RMS约[0,0,0.028408,0.027898,0.043699,0.033636]（末段0.8s）。较8500的15.75375秒开头零段缩短，但仍严重超长；不把EOS或非全零视为恢复。ASR“最高速速有24c 其实近期还慢 到速速甚至下午2点50其他也是公里 现在剩余的电量是80%”，CER0.888889，仍有明显内容错误。
- 新增参考内容对照：9000 ASR开头“最高速速有24c 其实近期还慢”与该配对reference_text“最高时速只有二十四，比骑自行车还慢。”有相似内容，8500/9000 reference_id及38帧前缀完全一致；这是可能重复参考内容的线索，不是经试听或预测码证明的泄漏。只读复核`qwen3_train/train.py:74`起ICL拼接参考+目标文本、输入参考codec；`:117–122`将参考codec与新增codec一起解码，再按参考帧占比裁掉前缀波形后写WAV。代码确有裁剪而非直接保存全部前缀，当前7.04s/20.8s等输出时长也符合新增帧口径，未发现可直接修复的漏裁剪错误。现有WAV/ASR不能区分模型重复参考、codec行为或ASR插入，需要原始预测码及解码前浮点进一步定位；未据此改协议/训练目标。
- speaker_only同长ZH连续6000–9000七次不截断：88帧/7.04s/比1.12102、EOS，RMS0.0399488、零比例7.6024%，int16非零156115/168960、最长零段0.003958s，前4秒/后3.04秒RMS0.039308/0.040776。CER0.472222（8500为0.555556），ASR有“推起/先吃/切了”等替换及数字口径差异；历史长零段未复发，内容仍不完整。
- ICL零帧总数仍2但目标改变：sample12在6500–8500五次立即EOS后恢复46帧/3.68s/比0.69697，RMS0.0541453，ASR“相信你跟师傅都看过我寄给你们的卷位 videos”，CER0.310345，有尾部缺失及词替换；不称持续恢复。EN sample00继8500恢复后再次0帧，0.356974s/reference_frames36/目标1.81s、WER/CER1。ZH sample04连续4500–9000十次0帧，本轮0.366920s/reference_frames19/目标2.3s、WER/CER1。两零帧无WAV符合生成分支；不能单看2/8不变漏掉目标互换。
- 其余逐句：ICL sample02 4.080042s/WER/CER0，sample05 1.6s/WER0.75/CER0.1538（WF Flanders Uni、分词及专名差异），sample06 3.44s/WER0.1667（Jeffreeze/swap）；短ZH sample01 1.04s/CER0.3333（组织/煮至），speaker_only同句1.04s/CER1。speaker_only sample05 1.6s/比0.98765/WER1.25/CER1.15385、ASR“to keep it a good joy.”，近目标时长仍内容错误；sample02 4.24s/WER0.0625（省略like）、sample06 3.6s/WER0.1667、sample04 2.32s/CER0.5385、sample12 4.08s/比0.7727/CER0.3448，仍有替换/缺失。speaker_only sample00生成26帧耗时18.9954s，比其通常约0.33s/帧偏慢，但后续13帧4.749s/53帧18.105s及训练吞吐已正常，日志无对应错误，单次生成延迟不构成停滞证据，未声称已定位原因。
- 资源：GPU0–3占用81127/77997/75927/77439 MiB（各81920），利用率98/98/99/98%；compute-apps仅本run四rank，GPU0外部余量793MiB，无OOM。主机已用238GiB、available1.7TiB、无swap；00:52:20磁盘可用583456.23GiB，足以继续保存。
- 判断/处理/未解决：保持当前训练代码/配置/进程（预算6000/9000、workers16/prefetch2、LR及两个epoch38539目标），训练/val与供数健康，未证实需改实现或重启的运行故障。ICL长零段、可能参考内容重复、sample04持续EOS及sample00反复EOS仍待定位；本轮扩大了参考文本和实际裁剪代码对照，但未获取原始预测codec、EOS/非EOS logits、解码前浮点，也未声称完成根因复现。未另载GPU模型或打断训练，不据8条波动改训练任务。9500继续验证sample12恢复、sample00/04及长ZH；尚未执行最终冻结检查或写final-verification.json，最终仍须38539 COMPLETE、最终val/两summary及`PYTHONPATH=. ...check_frozen_frontend.py --include-speaker`实际通过。
- 命令/修改/结果：cat/tail读约定、manual、最新巡检/status；Python解析快照/完整日志作finite/min/median/max/慢步/错误扫描、/proc身份、checkpoint文件/metadata/signature、datetime/shutil.disk_usage；nvidia-smi GPU/compute-apps、free -h；`.venv/bin/python`用soundfile/numpy读16metrics/14WAV、对照8500配对/英语口径、时长/RMS/零比例及两长ZH int16最长零段位置；rg/sed只读复核train.py生成/解码裁剪，并对照8500/9000参考文本；最后复核9360、manual、latest、退出/验收文件。均成功；唯一人工写入为本段追加。未改代码/配置/数据、发信号或启停恢复，未启动timer/Codex/subagent、提交推送、发消息或删除清理。本次巡检结束。


## 2026-09-10 01:24:33 UTC 单次巡检：9890步，9500步ICL长中文时长恢复，sample04持续立即EOS

- 依据/进度：已读故障处置表、本文、manual、快照`supervision/20260910T012129Z.json`、最新巡检`20260910T005129Z.md`和status（上轮review_exit_code0、固定session不变）。manual仍active=false、主会话observed_through_step2040。上一现场9360/上一快照9310 → 本轮01:21:29快照9850 → 01:22:22现场9870 → 01:23:06最终采样9890。9500 checkpoint/val/两summary完整，之后继续更新；10000尚未到步，不是缺失故障。
- 进程/退出：training-process.json PID621410/start_ticks80769258与/proc精确一致，cmdline为项目`.venv/bin/python .venv/bin/torchrun --standalone --nproc_per_node=4 -m qwen3_train.train --config configs/emilia-10kh-pretrain.yaml`；launcher621409/start80769255/PPID1、四rank621435–621438/start80769421/PPID621410均匹配存活。现场均S，但四GPU利用率100%且step增加，不判卡死。training-exit.json不存在，完整train.log无Traceback/OutOfMemory/out of memory/Non-finite/Error/Aborted匹配行；未发信号或启停恢复。
- 近半小时训练：9320–9870共56个每10步当步记录，所有指标有限。first_ce首末1.337499→1.345208、范围1.273927–1.393008、中位1.329555；residual_ce首末6.046884→5.994820、范围5.974000–6.081336、中位6.035519；grad_norm范围0.420446–0.565849、中位0.472310（clip前）。主干LR8.952519764e-5→8.816061928e-5、新参数2.685755929e-4→2.644818578e-4，按既定cosine下降。最终9890 first/residual CE1.335030/6.026995、grad0.490736、LR8.810965669e-5/2.643289701e-4、step2.207808s，均有限。
- 动态batch/吞吐/供数：上述56点帧填充95.1292%–99.5500%、中位98.3813%；token填充91.6861%–97.7000%、中位95.2611%；global samples321–391。step_seconds范围2.01152–2.29438s、中位2.13570s；音频秒/墙钟秒807.698–935.253、中位883.651；data_wait_seconds范围0.0002411–0.0006585s、中位0.0002754s。与上轮2.14031s/878.979音频秒每秒/等待0.0002760s相近，无持续供数恶化，记录无>3s慢步。rank0训练peak allocated最高56.9851GiB，不代表各rank总占用；以上为稀疏当步值，不是含评估的全窗口均值。
- Checkpoint/val：latest=step-00009500，COMPLETE时间00:58:05.541339 UTC；progress={step:9500,epoch:0,next_batch:9500}、world_size4、scheduler.last_epoch9500、LR8.908652146e-5/2.672595644e-4。9000/9500两COMPLETE签名相等，四distributed分片各约2.093GB、`.metadata`1424887字节、四rng各14613字节齐全。结构核查不等于实际恢复加载；8500由训练器既有keep_checkpoints=2轮转，巡检未清理或改metadata。9500 val first/residual CE1.376615/6.040666，较9000的1.393668/6.064613继续下降；15个码本CE有限，范围3.831623–6.819021。

|9500步，每模式EN4/ZH4|EN基础WER/CER|EN英语规范化WER/CER|ZH WER/CER|截断/零帧|
|---|---|---|---|---|
|speaker_only|0.222222/0.083916|0.222222/0.083916|1.125/0.595238|0/8、0/8|
|icl|0.194444/0.125874|0.194444/0.125874|1.125/0.547619|0/8、1/8|

- 评估核验：实读两summary、16条metrics、15个存在的WAV；所有WAV为24kHz单声道、有限、时长与metrics一致，目标ID/文本/speaker_reference_id/reference_text与9000一致。本轮英语两口径相同；9000→9500 EN WER speaker_only0.277778→0.222222、ICL0.25→0.194444；ZH CER0.476190→0.595238、0.666667→0.547619。仅8条诊断，不由汇总变化推断全验证质量或调LR；未试听。
- ICL长ZH sample03：从9000的260帧/20.8s变为92帧/7.36s、目标6.28s、比1.17197、EOS且不截断。RMS0.0332679、peak0.260254、精确零比例15.5774%，int16非零149124/176640；最长连续零段为开头[0,0.529042]秒（上轮10.435s），第二长为尾部0.035708s。前4秒/后3.36秒RMS0.030939/0.035843，历史多秒零段本轮未出现。ASR“尤其的半個小時,現在是下午2點50,起了大概10公里,現在剩餘的電量是80%”，CER0.583333，含同音替换、繁简与数字口径差异；未见9000那段类似“最高时速只有二十四，比骑自行车还慢”的参考内容，但ASR未出现不能证明模型从未重复参考。8000也曾改善后复发，故仅记录本次恢复，不宣布根因解决。
- speaker_only同目标连续6000–9500八次不截断：90帧/7.2s、比1.14650、EOS，RMS0.0380704、零比例5.9427%，int16非零162531/172800、最长零段0.022417s；前4秒/后3.2秒RMS0.038594/0.037406。CER0.583333（9000为0.472222），ASR有“有期/期的/4公里”等替换，时长与连续能量保持正常但内容仍错。
- ICL零帧2→1：EN sample00继9000立即EOS后再次恢复18帧/1.44s/比0.79558，RMS0.0616439、ASR“the liquid spears.”，WER0.25/CER0.166667，省略重复the；恢复具有反复史。ZH sample12连续9000/9500有输出，本轮43帧/3.44s/比0.65152、RMS0.0649208、CER0.344828，ASR“相信你跟似乎都看过我寄给你们那卷 Vidio”，仍有词替换与末段缺失。sample04连续4500–9500十一次立即EOS，本轮0.365270s/reference_frames19/目标2.3s，WER/CER1，无WAV符合零帧分支，仍未恢复。
- 其余逐句：ICL sample02从WER0变为0.1875/CER0.1471，4.080042s，ASR“Michael Aufflin”替代McLaughlin且省略yeah；sample05 1.68s/WER0.25（W.I.）、sample06 3.76s/WER0.1667（Jeffreeze/swap）；短ZH sample01 1.04s/CER0.3333（阻止/煮至）。speaker_only sample05 2.16s/比1.3333、WER0.75/CER0.1538（TWA Flanders UNI），较9000内容改善但仍有专名/分词差异；sample02 4.32s/WER0.0625（yeah→at），sample06 3.68s/WER0.25；sample01 1.2s/CER0.8333、sample04 2.4s/CER0.6154、sample12 4.32s/比0.8182/CER0.5517，中文仍有替换、插词及繁简差异。speaker_only sample00 28帧生成9.4931s，9000的26帧18.9954s延迟未持续。
- 资源：GPU0–3占用81127/77997/75927/77439 MiB（各81920），利用率均100%，compute-apps仅本run四rank；GPU0外部余量793MiB，无OOM。主机已用237GiB、available1.7TiB、无swap；01:22:22磁盘可用583353.42GiB，足以继续保存。
- 判断/处理/未解决：保持当前训练代码/配置/进程（预算6000/9000、workers16/prefetch2、LR及两个epoch38539目标），训练/val及供数正常，ICL部分目标本轮有实际改善，没有证实需改实现或重启的运行故障。sample04持续EOS、sample00反复EOS及长ZH历史长零段/参考内容重复线索仍待定位；原始预测codec、EOS/非EOS logits和解码前浮点证据仍缺，未声称根因已复现或消除，也未另载GPU模型或打断训练。下轮10000核对恢复能否保持，不据8条诊断改变训练任务。未执行最终冻结检查或写final-verification.json；最终仍须38539 COMPLETE、最终val/两summary及`PYTHONPATH=. ...check_frozen_frontend.py --include-speaker`实际通过。
- 命令/修改/结果：cat/tail读约定、manual、最新巡检/status；Python解析快照/完整日志作finite/min/median/max/慢步/错误扫描、/proc身份、两checkpoint文件/metadata/signature、datetime/shutil.disk_usage；nvidia-smi GPU/compute-apps、free -h；`.venv/bin/python`用soundfile/numpy读16metrics/15WAV，对照9000配对/英语口径、时长/RMS/零比例和两长ZH int16最长零段位置；最后复核9890、manual、latest、退出/验收文件。均成功；唯一人工写入为本段追加。未改代码/配置/数据、发信号或启停恢复，未启动timer/Codex/subagent、提交推送、发消息或删除清理。本次巡检结束。


## 2026-09-10T01:39:15.077570+00:00 主会话：用户授权暂停，修复生成初期EOS规则

用户明确要求修复与官方生成不一致的地方，并允许暂停后resume。实际9000步CPU/FP32对照确认：输入/位置/首步logits与官方一致，但本地允许第一步EOS，官方min_new_tokens=2；两个空样本补齐规则后生成有效目标音频。当前10000 COMPLETE齐全、尚在该步评估，主会话已将manual.active=true接管干预。核对PID621410/start_ticks80769258/cmdline/进程组后发送TERM，现场、日志及checkpoint metadata保存在run下eos-minimum-fix/。保持数据、训练loss、LR、预算及38539步计划，后续从完整checkpoint恢复优化器/调度器/游标；不编辑旧metadata或跳过签名。


2026-09-10T01:42:24.143998+00:00: EOS修复代码和47项完整测试通过。model.next_frame接受suppress_eos，选首码本argmax前屏蔽EOS；generate_sample在新增帧数小于2时启用，两种模式一致，ICL参考帧不计数。生成metrics/summary新增generation_policy={decoding:greedy,min_new_frames:2}，区分修复前口径。测试验证EOS最高时输出两帧最佳有效音频码再EOS，防止只忽略停止标志而反馈占位code。训练loss/输入/参数与配置完全未改，checkpoint signature仍严格逐项匹配，无旧metadata编辑。已启动既有--eval-only入口，准备从10000恢复四卡模型及优化器（签名/加载结果待现场确认），输出到eos-minimum-fix/evaluation-check；修复前该步speaker_only8条完整、ICL6条未完成，原评估尚保留。评估完成后再正常launch.py --resume latest接续训练。


## 2026-09-10 01:56:49 UTC 单次巡检：10000步由主会话主动暂停，最小生成长度修复正在四卡评估

- 依据/权限：已读故障处置表、本文（含主会话01:39暂停及01:42修复记录）、manual、快照`supervision/20260910T015129Z.json`、最新巡检`20260910T012129Z.md`与status。manual现active=true、phase=eos-minimum-fix、owner=main conversation，intervention_started_at=01:39:15；本轮只读检查并追加记录，训练控制由主会话负责。上一现场9890/快照9850 → 当前快照及01:54:47复核仍10000；是用户授权暂停后的修复评估，不是训练意外卡死或完成。
- 原训练退出/阶段：training-process.json仍记PID621410/start_ticks80769258；training-exit.json同PID、exit_code1、finished_at01:39:21.008977。/proc确认原launcher621409、torchrun621410和四rank621435–621438全部不存在。`eos-minimum-fix/pause.json`记录主会话核对PID/start/cmdline/pgid后01:39:15发送TERM，阶段为10000生成评估，未观察到checkpoint后更新。train.log明确“Received Signals.SIGTERM death signal”，四rank收到TERM，最终`SignalException: Process 621410 got signal: 15`；不是OOM/NaN首发。完整Traceback和关闭时每rank“56 leaked semaphore objects”警告已保留在train.log及`eos-minimum-fix/after-pause-train.log`，退出记录也已归档，本轮实读该尾部；未删除现场、未发信号。
- 当前实际进程为修复评估：`eos-minimum-fix/evaluation-process.json` PID4186036/start_ticks84544864与/proc匹配，命令为4卡torchrun、本配置、`--resume .../checkpoints/step-00010000 --eval-only --output .../eos-minimum-fix/evaluation-check`。四rank4186095–4186098/start84545026/PPID4186036均存活且现场R，torchrun S。原training-process.json不描述这个独立eval-only进程，不能因原PID不在就启动第二份训练。evaluation.log已记录initialized=true/world_size4/progress={step:10000,epoch:0,next_batch:10000}，val与暂停前逐项一致；有真实恢复初始化证据，但本轮未独立比较所有优化器tensor。尚无该评估退出记录，主会话尚未交还manual控制。
- 近半小时有效训练窗口仅9860–10000的15个每10步记录点，后续为评估/暂停，不能冒充整半小时稳态吞吐。全部指标有限：first_ce首末1.280629→1.310585、范围1.254419–1.346802、中位1.300881；residual_ce首末5.992513→6.022567、范围5.992513–6.041941、中位6.015175；grad_norm范围0.410068–0.490736、中位0.466494（clip前）。主干LR8.818606574e-5→8.782770825e-5、新参数2.645581972e-4→2.634831248e-4，按既定cosine下降。帧填充95.8833%–99.3958%、中位97.8792%；token填充93.8417%–96.55%、中位95.5611%；global samples333–389。step范围2.08822–2.26907s、中位2.15398s，音频秒/墙钟秒828.215–913.887、中位868.315，data_wait中位0.0002760s、范围0.0002548–0.0003767s。rank0训练peak allocated最高56.5480GiB。暂停前供数与上轮相近，无训练指标故障。
- Checkpoint/val：latest=step-00010000，COMPLETE时间01:26:49.770991 UTC；progress={step:10000,epoch:0,next_batch:10000}、world_size4、scheduler.last_epoch10000、LR8.782770825e-5/2.634831248e-4。9500/10000签名完全相同，四distributed分片各约2.093GB、`.metadata`1424887字节、四rng各14613字节齐全。9000由既有训练器轮转，本巡检未清理；未改旧metadata或绕过签名。10000 val first/residual CE1.370062/6.024738，较9500的1.376615/6.040666下降，15码本CE均有限（3.818170–6.800961）；独立eval-only重算结果与原val相同。
- 修复证据更新：读取主会话归档`eos-minimum-fix/audit-REPORT.md`、audit-silence-analysis.json、audit-min2-asr.json与测试日志。本地CPU/FP32、checkpoint9000对照报告确认sample00/04初始EOS概率0.927745/0.991161，前两步屏蔽EOS后生成17/25帧，并有目标ASR内容；sample04前八个完整16码帧与官方缓存生成相等。该报告描述输入embedding最大差1.1921e-7、初始logits差1.7167e-5、位置相同，但不等于现场BF16/FSDP位级验证。此前巡检仅有WAV而缺原始码/logits的证据限制已有主会话实验补充，不再将所有过早EOS泛称为无法复现。
- 当前代码只读核验：model.py:149在有效codec+EOS候选argmax之前将EOS置-inf，train.py:87以len(generated)<2传suppress_eos，参考帧不计入新增帧；metrics与summary记录generation_policy={decoding:greedy,min_new_frames:2}。已读取full-tests.log“47 tests/OK”及final-generation-tests.log“2 tests/OK”，这些由主会话执行，本轮未重跑或改代码。训练目标/LR/预算/38539步计划保持，后续实际恢复和完成验证由主会话推进。
- 长近静音新增根因证据：归档CPU/FP32实验通过同checkpoint9000的官方缓存生成，在greedy、repetition_penalty1、min_new_tokens2下仍产生263帧/21.04s，前10秒浮点RMS1.1082904e-5、min9.2790842e-6/max1.4783165e-5、无精确零；前130帧仅10种完整codec向量，首码1657出现101次。由此支持近静音在PCM16写入之前已存在、重复预测码是直接线索，不能仅用裁剪或WAV写入解释；这是主会话已保存实验，不是本巡检新生成的音频。采样/repetition penalty的影响未验证，不自行改解码策略，最小长度修复也不保证长静音消失。

|本轮可用完整summary（每模式EN4/ZH4）|EN基础WER/CER|EN英语规范化WER/CER|ZH WER/CER|截断/零帧|
|---|---|---|---|---|
|10000原speaker_only|0.444444/0.321678|0.472222/0.321678|1/0.345238|0/8、0/8|
|10000 min2复评speaker_only|0.444444/0.321678|0.472222/0.321678|1/0.345238|0/8、0/8|
|9500原ICL（最近完整，旧口径）|0.194444/0.125874|0.194444/0.125874|1.125/0.547619|0/8、1/8|

- 原10000 ICL仅6条metrics、无summary，是主会话暂停中断，不能拼接9500后两条或伪造10000汇总。01:54:24修复复评已有speaker_only8条及summary、ICL3条且无summary；01:54:47计数仍8/3。speaker_only summary于01:51:53完成，随后ICL sample00/01/02于01:52:22/01:52:50/01:53:32依次落盘，四rank持续GPU工作；当前正在耗时长样本的阶段，短时无新metrics不判停滞。未把复评的step10000当成新增训练更新。
- 实读原10000的14条metrics/WAV及修复复评11条metrics/WAV，共25份WAV均24kHz单声道、有限且时长一致；复评11个已完成目标/文本/参考均与原对应样本一致，11份WAV的SHA256逐份相等。修复speaker_only全部8句音频保持相同，说明这些目标在该checkpoint未受前两帧EOS屏蔽影响，不代表EOS修复无效；该步原ICL sample04本来就已恢复，不能把它的首次有效输出归功于尚未应用的修复。未试听。
- 原speaker_only10000异常主要为EN sample05：54帧/4.32s/目标1.62s、比2.6667，ASR“2BW terse, to push the terse to take a DDDB young.”，基础WER2.5/CER2.4615，英语规范化拆分2BW为2 bw使WER2.75；解释summary两种EN WER差异。其余EN sample00 1.84s/WER0.75、02 4.4s/WER0.0625、06 3.6s/WER0.1667。ZH sample03 94帧/7.52s/比1.1975/CER0.3056，RMS0.0390152、零比例10.2061%、最长零段0.005375s，历史多秒零段未复发；sample04 2.4s/CER0但WER1受分词影响；sample01 1.2s/CER0.5、sample12 4.08s/比0.7727/CER0.5172，仍有词替换。只有8条诊断，不据波动调超参。
- 原ICL10000 sample03重新400帧/32s/比5.09554、截断、CER1；RMS0.0194336、零比例63.0680%，最长开头零段15.753625s，ASR“止難地轉入內地 首上以年 首上以年 首上以年”，既有长零段/内容重复未恢复。sample04原策略已24帧/1.92s/比0.8348、ASR“其實人家畢竟堅持了這麼多年”、CER0.3846仅繁简差异，相比9500的连续EOS发生实际变化；sample00原策略及复评均14帧/1.12s/比0.6188/WER0.25；sample01均1.04s/CER0.3333；sample02均4.000042s/WER0.0625；原sample05 1.76s/WER/CER0。复评sample03/04及完整ICL仍待主会话验证，不提前宣布修复后结果。
- 资源（01:53:37附近）：四GPU占用7457/7787/7539/6687 MiB、利用率61/51/31/37%，compute-apps仅当前eval四rank；较训练阶段大幅降低与eval-only相符。主机已用198GiB、available1.8TiB、无swap；磁盘可用583314.05GiB。无当前OOM或资源短缺证据。
- 判断/未解决/给主会话建议：保持当前评估进程与现配置，由主会话完成两模式复评、核对generation_policy和异常样本，再按既有launch.py从10000完整恢复。恢复时应确认游标epoch0/next_batch10000、scheduler10000/LR连续，并至少观察两个训练日志点及后续10000后更新；保留原10000未完成评估，避免旧策略与min2混成同一口径。长期近静音/参考重复与EN sample05异常仍需单独分析；不能用最小长度修复宣称全部解决。本轮未启停/恢复训练、未改代码或参数。manual仍active=true；训练仅10000/38539，未执行最终冻结验收，final-verification.json不存在，未写passed=true。
- 命令/结果：cat/tail/ls读约定、快照、状态、干预/审计/测试记录和完整终止traceback；Python解析日志统计、扫描/proc及checkpoint文件/metadata/signature、datetime/shutil.disk_usage；nvidia-smi、free；soundfile/numpy读取WAV与零段，hashlib比较复评/原WAV；rg/sed核验EOS屏蔽与policy。一次rg附带不存在的qwen3_train/evaluation.py返回2（model/train匹配正常），随后只查实际model.py/train.py成功；这是巡检路径搜索失误，不是训练错误。唯一人工写入为本段追加；未发信号、改训练代码/配置/数据、启动timer/Codex/subagent、提交推送发消息或删除清理。本次巡检结束。


## 2026-09-10T02:00:05.954847+00:00 主会话：EOS规则修复复评通过，从10000步恢复

- 修复仅对首两帧新增音频在首码本选择前屏蔽EOS，ICL参考帧不计数；训练loss、模型参数、LR、动态预算及38539步目标未变。独立checkpoint生成入口同步修正。47项完整测试通过，随后增加独立入口断言的2项定向测试也通过，git diff --check通过。代码差异、哈希、测试日志与先前9000步官方对照保存在run的eos-minimum-fix目录。
- 四卡BF16/FA2/FSDP从10000 COMPLETE严格加载的eval-only退出码0。512条验证first/residual CE为1.3700621786308829/6.024738094039755，连同15个残差码本CE与暂停前完全一致。两模式各8条metrics/WAV及summary齐全，24kHz、有限、时长一致，均记录greedy/min_new_frames=2。
- speaker-only 8份WAV与原10000逐文件SHA256相同；ICL原先已完成的6份WAV也逐文件相同，剩余2份为本次完成。原10000 ICL没有summary，不能编造完整前后指标对照。当前speaker-only英语规范化WER47.22%、ZH CER34.52%；ICL英语WER11.11%、ZH CER66.67%，无零帧，ICL长中文sample03仍400帧/32秒截断。仅8条在线诊断，不据此判断整体质量。
- 归因边界：10000的ICL sample04在旧策略下本来已恢复24帧/1.92秒，不能把该步恢复归因于本次修复。直接干预证据来自9000步相同权重实验：sample00/04原先立即EOS，补齐官方最短长度后分别17/25帧。近静音可通过官方贪心生成复现，仍需独立跟踪，最短长度修复不宣称解决此问题。
- 原10000评估完整归档至eos-minimum-fix/before-evaluation；新评估经校验复制到正式evaluation两模式目录，源副本保留evaluation-check。逐条校验见evaluation-verification.json。旧退出记录保存before-resume-training-exit.json。现由既有launch.py --resume latest启动，launcher PID5004、torchrun PID5005/start_ticks84647923，从10000/epoch0/next_batch10000恢复优化器、调度器与游标，尚待初始化后实际更新确认。
- 用户补问的loss已核对：每步跨所有rank和累积microbatch汇总有效目标数，first按总帧数+总样本数平均（含EOS）、residual按15倍总帧数平均，再以0.3加权；反传乘world_size抵消FSDP梯度平均。各卡batch不同仍按token加权，文本token不在loss分母内。既有四卡变长/不同样本数/两次累积对照相对梯度误差0.00318，见dynamic-batching.md。


## 2026-09-10T02:11:21.953523+00:00 主会话：恢复验证通过，清理结束并交回巡检

从10000完整恢复后已实际推进至10170步，17个每10步日志点全部指标有限；LR逐点符合原38539步cosine计划。最新first/residual CE为1.297313/6.019990，grad norm 0.513839；恢复后记录点step中位2.1306秒，rank0取批等待中位0.0002675秒。torchrun PID5005/start_ticks84647923一致，四卡继续运行，无新的退出记录。恢复验证见eos-minimum-fix/resume-verification.json。manual.active已改回false，交回原持久化巡检PID441407、session 01a08697-ba78-7cd3-bb9e-b01eba12d724，每1800秒继续检查；长期ICL静音/截断尚未解决。

用户明确要求runs只留两个正式实验，并清理过时文档。实际移除34个旧顶层条目，文件分配空间合计74.0854GiB；1kh所有原有文件大小/修改时间未变。源码归档/历史清理记录归入1kh的provenance，10kh数据准备日志归入当前run的provenance/data-preparation。两个正式实验的checkpoint、数据、初始化及原始音频依赖均保留。runs顶层另按用户要求保留README.md，明确两个目录完整保护。清单见cleanup-20260910/。旧试训/迁移/LJSpeech等9份过时文档与编辑器副本删除，当前模型、数据、ICL、动态组批、环境与pipeline说明更新；本文历史巡检中的旧路径保留为当时记录，不代表产物仍存在。用户已授权commit/push，正在单独完成版本管理。


## 2026-09-10 02:27 UTC 单次巡检：恢复后10500步保存验证通过，生成评估进行中

- 依据/权限：已读故障处置表、本文（含EOS修复、02:11恢复交接）、manual、快照`supervision/20260910T022129Z.json`、最近巡检`20260910T015129Z.md`及status。manual.active=false，phase=eos-fix-complete-training-resumed，主会话已观察至10170。上一快照10000（主动暂停）→本轮02:21:29快照10440→02:25:14现场10500；02:26:29仍10500但生成metrics继续落盘，属于正常评估阶段。原目标仍两个epoch/38539步。
- 进程/恢复：PID5005/start_ticks84647923与/proc匹配，cmdline为项目torchrun、4 ranks、原config、--resume latest；launcher5004/start84647921/PPID1，四rank5030–5033/start84648072/PPID5005均存活。现场rank为R，torchrun为S。train.log实读initialized/world_size4/progress={step:10000,epoch:0,next_batch:10000}及随后50个训练日志点。已读主会话resume-verification.json，确认其10170交接记录。training-exit.json不存在；完整日志中的唯一Traceback来自旧PID621410的主动TERM，最终SignalException signal15，已对照上下文，不能当成新退出。以本次恢复初始化为边界扫描，未见新Traceback/OOM/Non-finite/Error/Aborted。
- 恢复后10010–10500共50个每10步当步值全部有限：first_ce范围1.234879–1.359359、中位1.307179、末1.315523；residual_ce范围5.959869–6.064899、中位6.012993、末5.993888；grad_norm范围0.396307–0.561454、中位0.473285、末0.430976（clip前）。主干LR8.780193827e-5→8.651142368e-5、新参数2.634058148e-4→2.595342710e-4，恢复后按既定计划连续下降，未清空或改scheduler。
- 动态batch/吞吐：frame填充94.5458%–99.6708%、中位98.3854%；token填充91.8972%–98.0056%、中位95.3333%；global samples291–410。step中位2.130537s、范围1.965201–19.647187s；音频秒/墙钟秒中位881.655、范围96.967–942.886；data_wait中位0.0002731s、范围0.0002150–0.0004573s。唯一>3s记录是10450的19.647187s，取批等待仅0.0002974s、loss/梯度有限；前10440为2.140936s、后10460为2.136809s，末10500为2.145701s。没有持续3点恶化证据，不能归因为供数或checkpoint（该步不是保存步），原因未定位，本轮未重启。恢复最初10010/10020为2.44/2.61s，随后稳定。rank0训练peak allocated最高56.9387GiB；上述为稀疏当步记录，不是包含暂停/初始化/评估的半小时平均。
- Checkpoint/val：latest=step-00010500，COMPLETE时间02:24:07.077596 UTC；progress={step:10500,epoch:0,next_batch:10500}、world_size4、scheduler.last_epoch10500、LR8.651142368e-5/2.595342710e-4。10000/10500两COMPLETE签名逐项相等，各自四distributed分片约2.093GB、.metadata1424887字节、四rng各14613字节齐全；目录结构核查不等于本轮实际试加载10500。9500由训练器keep_checkpoints=2轮转，本轮未清理或编辑metadata。10500 val first/residual CE1.357246/6.007607，较10000的1.370062/6.024738下降；15码本CE均有限（3.798398–6.784663）。

|最近完整10000步复评，每模式EN4/ZH4，greedy/min_new_frames=2|EN基础WER/CER|EN英语规范化WER/CER|ZH WER/CER|截断/零帧|
|---|---|---|---|---|
|speaker_only|0.444444/0.321678|0.472222/0.321678|1/0.345238|0/8、0/8|
|icl|0.111111/0.076923|0.111111/0.076923|2.125/0.666667|1/8、0/8|

- 完整评估核验：实读正式10000两summary、16条metrics/16WAV，全部24kHz单声道、有限、时长与metrics一致；目标ID/文本/speaker_reference_id/reference_text与9500一致，均记录修复后policy。与9500的旧策略口径有变化，不能把恢复直接归因于修复；10000原ICL sample04本来已有24帧，直接EOS干预证据仍是主会话9000同权重实验。两模式只有8条诊断，不代表全验证集质量，未试听。
- 长ZH ICL sample03仍400帧/32s、目标6.28s、比5.09554、未EOS且截断；RMS0.01943356、peak0.291229、零比例63.0680%，最长零段[0,15.753625]秒。每4秒RMS约[0,0,0,0.016111,0.039917,0.026853,0.021148,0.000003525]。当前ASR为“丹主肃请。丹主肃! 京 Additional … 张晓敏 …”，WER3.25/CER1.083333。实读before-evaluation中旧10000同目标metrics，原ASR“止難地轉入內地 首上以年 …”、WER/CER1；逐文件SHA256确认两WAV相同。评分变化存在ASR重复运行差异，不能报告为EOS修复改变音频或模型质量进一步下降；两次ASR都不能忠实评价近静音。原始预测码重复、PCM16前近静音由主会话9000官方生成实验已复现，最短长度修复不解决该问题，本轮未重复加载模型。
- 同目标speaker_only为94帧/7.52s/比1.19745、EOS、不截断；RMS0.0390152、零比例10.2061%、最长零段0.005375s，CER0.305556，历史多秒零段未复发但有替换及数字口径差异。ICL sample00有14帧/1.12s/WER0.25；sample04为24帧/1.92s/CER0.384615（繁简差异）；sample12为45帧/3.6s/比0.681818/CER0.344828，仍有尾部缺失/词替换。其余ICL EN sample02/05/06 WER0.0625/0/0.166667，短ZH sample01 CER0.333333。
- speaker_only英语异常仍主要sample05：54帧/4.32s/目标1.62s、比2.66667，基础WER2.5/CER2.461538，英语规范化WER2.75；ASR“2BW terse, to push the terse to take a DDDB young.”，2BW拆分使两种WER不同。其余EN sample00/02/06 WER0.75/0.0625/0.166667；ZH sample04 CER0、sample01/12 CER0.5/0.517241。不能因时长正常即判内容恢复。
- 10500评估进展：02:25:29 speaker_only sample00已29帧/2.32s、EOS、min2、WAV有限；基础WER0.75，英语规范化去除um后WER0.5。02:26:10 sample01 metrics新落盘，37帧/2.96s、目标0.98s、比3.0204、EOS，提示该短ZH本次变长，尚未据未完成评估下结论。02:26:29最终计数speaker_only2条、ICL0条、两summary均尚无，是刚开始的新一轮评估，不是文件丢失；不拼接10000结果构造10500汇总，也不等待整轮评估再结束本次巡检。
- 资源（02:25附近）：GPU0–3总占用77519/77209/76023/77859 MiB（各81920），利用率0/100/100/100%，compute-apps仅本run四rank。生成评估期间单次rank0 GPU低利用率不足判停滞，metrics随后新增。主机已用237GiB、available1.7TiB、无swap，磁盘可用583227.98GiB。没有新OOM或资源短缺证据。
- 判断/处理：保持代码、预算6000/9000、workers16/prefetch2、LR、训练进程及38539步目标。恢复、后续保存和val均通过现场核验，无证实需干预的运行故障。继续跟踪10500完整评估、ICL长近静音/截断、EN sample05及短ZH变长；慢步若复发再按连续证据诊断。当前10500/38539，final-verification.json不存在，未执行或宣称最终冻结验收。最终仍须38539 COMPLETE、最终val/双summary和包含speaker的冻结权重检查实际通过。
- 命令/修改：cat/tail/ls读约定与状态，Python解析快照、train.log、/proc、checkpoint metadata/分片、统计finite和min/median/max、扫描恢复后错误、读取磁盘，nvidia-smi/free，soundfile/numpy读取WAV/零段，hashlib对照旧新同目标音频。一次寻找监督脚本的rg通配路径不存在返回2，不涉及训练错误；未据此修改配置。唯一人工写入为本段追加；未启停恢复/发信号、修改代码配置数据、创建timer/Codex/subagent、提交推送发消息或删除清理。本次巡检结束。


## 2026-09-10 02:53 UTC 单次巡检：10990步，10500步ICL长中文本次恢复正常时长

- 依据/进度：已读故障处置表、本文、manual、快照`supervision/20260910T025129Z.json`、最近巡检`20260910T022129Z.md`与status（上轮退出码0、固定session不变）。manual.active=false、主会话已观察至10170。上一现场10500/快照10440→本轮02:51:30快照10950→02:52:34现场10980→02:52:59最终采样10990。10500 checkpoint/val/双summary完整；最终采样时11000尚未到步，无对应产物不是故障。
- 进程/退出：training-process.json PID5005/start_ticks84647923与/proc匹配，cmdline为项目torchrun、4 ranks、原config、--resume latest；launcher5004/start84647921/PPID1、四rank5030–5033/start84648072/PPID5005均存活。torchrun/launcher为S、rank为R，step持续增加。training-exit.json不存在；以10000恢复初始化为界扫描完整后续日志，无Traceback/OOM/Non-finite/Error/Aborted。快照仍收录旧主动TERM的Traceback，已在上轮核实为PID621410的历史退出，不当作当前异常。
- 近半小时10450–10980共54个每10步当步值全部有限：first_ce范围1.250051–1.384805、中位1.303462；residual_ce范围5.927572–6.038371、中位5.994222；grad_norm范围0.389444–0.558983、中位0.462407（clip前）。主干LR8.664557344e-5→8.519585960e-5、新参数2.599367203e-4→2.555875788e-4，既定cosine连续下降。最终10990 first/residual CE1.270140/5.987254、grad0.477924、LR8.516792644e-5/2.555037793e-4，step2.204338s，均有限。
- 动态batch/吞吐：上述54点帧填充94.9167%–99.7792%、中位98.0292%；token填充91.7694%–97.5306%、中位95.3069%；global samples297–385。step中位2.112482s、范围1.939990–19.647187s；音频秒/墙钟秒中位887.286、范围96.967–949.160；data_wait中位0.0002823s、范围0.0002345–0.0007438s。唯一>3s点仍是上轮已记录的10450慢步，不是新复发；后续无持续恶化。与上轮中位2.130537s/881.655音频秒每秒/等待0.0002731s接近。rank0训练peak allocated最高56.7562GiB，不代表各rank总占用；稀疏当步记录不能当作含评估的窗口平均。
- Checkpoint/val：02:52:59 latest仍step-00010500，COMPLETE时间02:24:07.077596 UTC；progress={step:10500,epoch:0,next_batch:10500}、world_size4、scheduler.last_epoch10500、LR8.651142368e-5/2.595342710e-4。10000/10500签名相等，四distributed分片各约2.093GB、.metadata1424887字节、四rng各14613字节齐全；仅结构核查，未实际试加载。10500 val first/residual CE1.357246/6.007607，较10000的1.370062/6.024738下降，15码本CE有限（3.798398–6.784663）。本轮未清理checkpoint或改metadata。

|10500步，每模式EN4/ZH4，greedy/min_new_frames=2|EN基础WER/CER|EN英语规范化WER/CER|ZH WER/CER|截断/零帧|
|---|---|---|---|---|
|speaker_only|0.277778/0.111888|0.250000/0.097902|1.125/0.559524|0/8、0/8|
|icl|0.111111/0.055944|0.111111/0.055944|1.125/0.476190|0/8、0/8|

- 评估核验：实读两summary、16条metrics/16WAV，全部24kHz单声道、有限、时长与metrics匹配；目标ID/文本/speaker_reference_id/reference_text与10000一致，均记录相同min2策略。10000→10500英语规范化WER speaker_only0.472222→0.25，ICL0.111111不变；ZH CER0.345238→0.559524、0.666667→0.476190。SO英语两口径差异由sample00的um在英语规范化中去除导致。每模式仅8条诊断，不据汇总波动改变超参，未试听。
- 长ZH ICL sample03：从10000的400帧/32秒截断变为86帧/6.88s、目标6.28s、比1.09554、EOS、不截断。RMS0.0340204、peak0.277557、零比例11.8580%；最长连续零段位于开头[0,0.529667]秒，较上轮15.753625秒显著缩短；前4秒/后2.88秒RMS0.035887/0.031243。ASR“又騎了半個小時,現在是下午2點50,騎了大概10公里,先生與電量是80%”，CER0.611111，包含繁简/数字口径与实际词替换。未出现类似reference_text“最高时速只有二十四，比骑自行车还慢”的开头内容，但不据ASR证明绝无重复。9500也曾恢复而10000复发，故仅确认本次音频时长和能量改善，不宣布近静音根因已解决。
- speaker_only同长ZH：89帧/7.12s/比1.13376、EOS、不截断，RMS0.0386415、零比例10.2499%、最长零段为末尾0.019625s；前4秒/后3.12秒RMS0.038837/0.038389，历史多秒零段未复发。ASR“客气的半个小时…气的大概十公里…现生域的电量百分之八十”，CER0.277778（10000为0.305556），仍有替换与缺词。
- 上轮新见SO短ZH sample01异常已完整核验：37帧/2.96s，目标0.98s、比3.0204、EOS、不截断，RMS0.0637783、零比例14.4637%、最长零段0.200667s；ASR“脏血血的年轻后 随着久久久久久久久的温柳”，CER3，明显超长且评分含重复内容，不能只看EOS判正常。同目标ICL仍13帧/1.04s/CER0.5（組織/煮至及繁简差异），最长零段末尾0.197083s。配对未变、WAV可读，尚无直接实现故障证据；保留异常，后续11000核对是否持续，不据单句暂停训练。
- SO EN sample05从10000的54帧/4.32s、基础WER2.5恢复为26帧/2.08s/比1.28395，ASR“TWA, Flanders Uni.”、WER0.75/CER0.153846；仍有专名与分词差异，但未见前轮长串错误内容。SO sample00/02/06基础WER0.75/0.0625/0.25，sample04/12 CER0.615385/0.379310；SO00规范化WER0.5。ICL sample00持续有19帧/1.52s/WER0.25；sample04有23帧/1.84s/比0.8/CER0.076923（坚实/坚持），未立即EOS；sample12为46帧/3.68s/比0.69697/CER0.482759，尾部仍不完整。ICL sample02/05/06 WER0/0.25/0.166667。两模式无零帧不等于所有内容正确。
- 资源（02:52:34附近）：GPU0–3总占用77519/77211/76023/77859 MiB（各81920），利用率98/93/21/22%，compute-apps仅本run四rank。step持续推进，不由单次各rank利用率差判卡死。主机已用237GiB、available1.7TiB、无swap；磁盘可用583210.63GiB，无新OOM或资源短缺证据。
- 判断/处理/未解决：保持训练代码、配置、预算6000/9000、workers16/prefetch2、LR与两个epoch38539步计划；训练供数正常，长ICL及SO英语异常本次部分改善，无证实需重启或修复实现的运行故障。长近静音历史反复、短ZH SO新超长及其他内容缺失仍须跟踪；主会话9000官方生成实验已证明近静音可在PCM16前出现，本轮不重复加载模型或自行改采样/惩罚策略。当前尚未完成训练，未执行最终冻结检查，final-verification.json不存在，未写passed=true；最终仍需38539 COMPLETE、最终val/双summary及include-speaker冻结检查实际通过。
- 命令/修改/结果：cat/tail读约定、manual、快照和最新review/status；Python解析日志统计finite/min/median/max/慢步、扫描恢复后错误、核对/proc身份、checkpoint分片/metadata/signature、datetime/shutil.disk_usage；nvidia-smi GPU/compute-apps、free -h；soundfile/numpy读16WAV/metrics、核验配对/时长并测长ZH及短ZH连续零段；最终复核10990、latest、manual、退出/验收文件。全部成功；唯一人工写入为本段追加。未改代码/配置/数据、启停恢复或发信号、启动timer/Codex/subagent、提交推送发消息或删除清理。本次巡检结束。


## 2026-09-10 03:24 UTC 单次巡检：11500步已保存验证，11000步长ICL连续两次无截断

- 依据/进度：已读故障处置表、本文、manual、快照`supervision/20260910T032129Z.json`、最近巡检`20260910T025129Z.md`与status（上轮退出码0、固定session不变）。manual.active=false，主会话交接观察至10170。上一现场10990/快照10950→本轮03:21:30快照11470→03:23:49现场11500，03:24:07复核处于11500生成评估，metrics持续新增。11000 checkpoint/val/双summary完成；11500 checkpoint/val完成、生成尚在进行，不能当作卡死或缺失故障。
- 进程/退出：training-process.json PID5005/start_ticks84647923与/proc匹配，cmdline为项目torchrun、4 ranks、原config、--resume latest；launcher5004/start84647921/PPID1、四rank5030–5033/start84648072/PPID5005均存活。launcher/torchrun为S、rank为R。training-exit.json不存在；以10000恢复初始化为界扫描后续完整日志，无Traceback/OOM/Non-finite/Error/Aborted。快照收录的旧Traceback仍为已记录的主动TERM历史，不代表当前进程出错。
- 近半小时10960–11500共55个每10步当步值全部有限：first_ce范围1.243198–1.338702、中位1.291119、末1.329843；residual_ce范围5.927708–6.034699、中位5.976868、末5.974521；grad_norm范围0.389546–0.584246、中位0.464089、末0.525584（clip前）。主干LR8.525166246e-5→8.371575465e-5、新参数2.557549874e-4→2.511472640e-4，既定cosine连续下降。
- 动态batch/吞吐：帧填充94.5167%–99.7542%、中位98.0125%；token填充91.6111%–97.8306%、中位95.2722%；global samples324–401。step范围1.981670–2.243841s、中位2.108031s；音频秒/墙钟秒830.005–936.138、中位889.630；data_wait范围0.0001405–0.0004789s、中位0.0002761s。记录无>3s慢步，旧10450慢步未复发；较上轮2.112482s/887.286音频秒每秒/等待0.0002823s相近，供数正常。rank0训练peak allocated最高56.8644GiB；这些是稀疏当步值，不是含评估的窗口均值。
- Checkpoint/val：latest=step-00011500，COMPLETE时间03:22:19.358855 UTC；progress={step:11500,epoch:0,next_batch:11500}、world_size4、scheduler.last_epoch11500、LR8.371575465e-5/2.511472640e-4。11000/11500签名相等，每份四distributed分片各约2.093GB、.metadata1424887字节、四rng各14613字节齐全。仅结构核查，未实际试加载；10500由训练器keep_checkpoints=2轮转，本轮未清理或编辑metadata。11000 val first/residual CE1.346106/5.993912，11500进一步降至1.341276/5.979663；11500的15码本CE均有限（3.780324–6.756024）。

|最近完整11000步，每模式EN4/ZH4，greedy/min_new_frames=2|EN基础WER/CER|EN英语规范化WER/CER|ZH WER/CER|截断/零帧|
|---|---|---|---|---|
|speaker_only|0.333333/0.181818|0.333333/0.181818|1.375/0.392857|0/8、0/8|
|icl|0.138889/0.062937|0.138889/0.062937|1.125/0.333333|0/8、0/8|

- 完整评估核验：实读11000两summary、16条metrics/16WAV，均24kHz单声道、有限、时长与metrics匹配；目标ID/文本/speaker_reference_id/reference_text与10500一致，生成policy相同。10500→11000英语规范化WER speaker_only0.25→0.333333、ICL0.111111→0.138889；ZH CER0.559524→0.392857、0.476190→0.333333。本轮两种英语口径相同。仅8条诊断，不据其波动推断全验证集质量或调超参，未试听。
- 长ZH ICL sample03连续10500/11000两次无截断：88帧/7.04s、目标6.28s、比1.12102、EOS，RMS0.0339098、peak0.300507、零比例14.8550%，最长零段开头[0,0.361]秒（10500为0.529667秒）；前4秒/后3.04秒RMS0.032608/0.035550。ASR“又起了半个小时,现在是下午2点50,起了大概10公里,现在剩余的点量是80%”，CER0.361111，较10500的0.611111降低，但含繁简口径变化、数字写法以及起/骑、点/电等词差异。长零段本轮未复发；历史9500恢复后10000复发，仍不宣布根因解决。
- speaker_only同长ZH为85帧/6.8s/比1.08280、EOS，不截断；RMS0.0381077、零比例7.2034%、最长零段0.003875s，前4秒/后2.8秒RMS0.040488/0.034423。CER0.333333（10500为0.277778），ASR有“最起/起”等替换和数字口径差异，时长/连续能量正常不等于内容完全正确。
- 短ZH sample01上轮SO超长未延续：11000为13帧/1.04s/比1.06122，较10500的37帧/2.96s回到目标时长附近，RMS0.0711111、零比例21.1939%、最长零段0.207042s；ASR“早就來白色後”，CER0.666667，仍有内容错误但未见上轮长串重复。ICL同句12帧/0.96s/CER0.5（组织拿/煮至奶），最长尾部零段0.158167s。两模式配对未变，没有直接实现错误证据。
- 英语问题仍集中少量目标：SO sample00仍29帧/2.32s/比1.28177，ASR“I that lamine and sparse spears.”、WER1.25/CER0.833333，较10500错误增多；SO sample05为26帧/2.08s、ASR“TWA, Flanders, SUNY.”、WER1/CER0.307692，未恢复10000那种超长但专名仍错。SO sample02 WER/CER0、sample06 WER0.25；sample04/12 CER0.538462/0.344828。SO12为64帧/5.12s/比0.96970，WER4受中英混合文本分词和插词影响，应同时看CER及ASR，不能单看ZH WER推断全面退化。
- ICL sample00为16帧/1.28s/WER0.25，sample04为24帧/1.92s/CER0.153846（歇时/坚持），仍未立即EOS；sample12为46帧/3.68s/比0.69697/CER0.344828，尾部内容仍缺失或替换。ICL sample02/05/06 WER0/0.5/0.166667。无零帧与无截断不等于所有目标已恢复。
- 11500评估进展：03:24:07复核speaker_only两条metrics、ICL零条、双summary尚无；SO00于03:23:31落盘22帧/1.76s/比0.97238、ASR“The liquid spears.”、WER0.25，相比11000明显改善；SO01于03:23:59落盘14帧/1.12s/比1.14286、CER0.666667，超长未再次出现。这两WAV均24kHz、有限、时长匹配，EOS、不截断、min2。新metrics说明评估持续推进；未把两条结果拼接旧summary，也未等待整轮评估后才结束本次巡检。
- 资源（03:23:49附近）：GPU0–3总占用77519/77211/76025/77859 MiB（各81920），利用率0/100/100/100%，compute-apps仅本run四rank。生成评估阶段rank0瞬时低利用率且文件继续新增，不判停滞。主机已用237GiB、available1.7TiB、无swap；磁盘可用583265.45GiB，无新OOM或资源短缺证据。
- 判断/处理/未解决：保持训练代码、配置、预算6000/9000、workers16/prefetch2、LR及两个epoch38539步计划；训练、供数、保存验证正常，生成部分异常改善，无证实需干预的运行故障。持续跟踪长ICL历史反复及SO00/05和其他内容缺失；主会话9000实验已证明PCM16前近静音可通过官方贪心生成复现，本轮不自行改解码策略或重复加载GPU模型。当前11500/38539，training-exit与final-verification均不存在，未执行或宣称最终冻结验收；最终仍需38539 COMPLETE、最终val/双summary及include-speaker冻结检查实际通过。
- 命令/修改/结果：cat/tail读约定、manual、快照与最新review/status；Python解析完整日志统计finite/min/median/max/慢步、扫描恢复后错误、核对/proc身份、checkpoint分片/metadata/signature、datetime/shutil.disk_usage；nvidia-smi GPU/compute-apps、free -h；soundfile/numpy核验11000的16条metrics/WAV、配对/时长/长短ZH零段，并复核11500两条WAV/metrics和manual/latest/退出验收文件。全部成功；唯一人工写入为本段追加。未改代码/配置/数据、启停恢复或发信号、启动timer/Codex/subagent、提交推送发消息或删除清理。本次巡检结束。


## 2026-09-10 03:53 UTC 单次巡检：12000步保存验证完成，11500步SO短英文内容异常复发

- 依据/进度：已读故障处置表、本文、manual、快照`supervision/20260910T035129Z.json`、最近巡检`20260910T032129Z.md`与status（上轮退出码0、固定session不变）。manual.active=false，交接观察至10170。上一现场11500/快照11470→本轮03:51:29快照12000→03:52:42现场及03:53:01复核12000。当前处于生成评估，文件继续新增；11500双summary完整，12000 checkpoint/val完成，尚未完成的生成不当作停滞或缺失故障。
- 进程/退出：training-process.json PID5005/start_ticks84647923与/proc精确一致，cmdline为项目torchrun、4 ranks、原config、--resume latest。launcher5004/start84647921/PPID1、四rank5030–5033/start84648072/PPID5005均存活，launcher/torchrun为S、rank为R。training-exit.json不存在；以10000恢复初始化为界扫描日志，无新Traceback/OOM/Non-finite/Error/Aborted。快照中的旧Traceback仍对应此前已记录的主动TERM，不能当成当前故障。
- 近半小时11480–12000共53个每10步当步值全部有限：first_ce范围1.218950–1.378092、中位1.280091、末1.306826；residual_ce范围5.898612–6.026089、中位5.960677、末5.960677；grad_norm范围0.403036–0.525584、中位0.455410、末0.480992（clip前）。主干LR8.377370547e-5→8.224126458e-5、新参数2.513211164e-4→2.467237937e-4，按既定cosine连续下降。
- 动态batch/吞吐：帧填充94.9958%–99.9625%、中位98.3833%；token填充91.6972%–97.5306%、中位95.5444%；global samples310–391。step范围2.008735–2.333160s、中位2.124192s；音频秒/墙钟秒819.455–927.865、中位889.066；data_wait范围0.0002332–0.0004798s、中位0.0002735s。无>3s记录，旧慢步未复发；与上轮2.108031s/889.630音频秒每秒/等待0.0002761s接近，无持续供数下降。rank0训练peak allocated最高57.3033GiB；以上为稀疏当步值，不是含评估的窗口均值或全部rank显存。
- Checkpoint/val：latest=step-00012000，COMPLETE时间03:51:09.106357 UTC；progress={step:12000,epoch:0,next_batch:12000}、world_size4、scheduler.last_epoch12000、LR8.224126458e-5/2.467237937e-4。11500/12000签名相等，各自四distributed分片约2.093GB、.metadata1424887字节、四rng各14613字节齐全；结构核验不等于本轮实际恢复加载。11000由训练器keep_checkpoints=2轮转，本巡检未清理或编辑metadata。12000 val first/residual CE1.337219/5.965698，较11500的1.341276/5.979663下降，15码本CE有限（3.773642–6.738522）。

|最近完整11500步，每模式EN4/ZH4，greedy/min_new_frames=2|EN基础WER/CER|EN英语规范化WER/CER|ZH WER/CER|截断/零帧|
|---|---|---|---|---|
|speaker_only|0.388889/0.300699|0.388889/0.300699|1.125/0.547619|0/8、0/8|
|icl|0.194444/0.118881|0.194444/0.118881|1/0.428571|0/8、0/8|

- 评估核验：实读11500两summary、16条metrics/16WAV，均24kHz单声道、有限、时长与metrics一致；目标ID/文本/speaker_reference_id/reference_text与11000一致，policy相同。本轮两种英语口径相同。11000→11500 EN WER speaker_only0.333333→0.388889、ICL0.138889→0.194444；ZH CER0.392857→0.547619、0.333333→0.428571。8条诊断不足判断全验证集退化，逐句归因如下；未试听。
- 长ZH ICL sample03连续10500/11000/11500三次无截断：87帧/6.96s、目标6.28s、比1.10828、EOS，RMS0.0346063、peak0.215698、零比例15.7549%，最长连续零段开头[0,0.446708]秒（上轮0.361秒），前4秒/后2.96秒RMS0.036008/0.032617。ASR“又起了半個小時,現在是下午2點50 起了大概10公里,先剩餘的電量是80%”，CER0.555556，包含繁简/数字口径及同音替换和缺词。未复发多秒零段；仍不据三次恢复宣布历史近静音根因消失。
- speaker_only同目标82帧/6.56s/比1.04459、EOS、不截断；RMS0.0393162、零比例13.6782%、最长零段0.019583s，前4秒/后2.56秒RMS0.038719/0.040232，连续能量正常。ASR“最起的半個小時,先是下午2點50,起的大概10公里,先剩許電量是80%”，CER0.694444，较11000的0.333333升高，含繁简、数字变化及替换/遗漏；不能仅凭正常时长判内容恢复。
- SO EN sample05内容异常复发：38帧/3.04s、目标1.62s、比1.87654，EOS、不截断；ASR“to bear to purse bouquet that it cage you with.”、WER2.5/CER2.461538，RMS0.0624154、零比例0.1014%、最长零段0.0000833s，明显内容错误并非长零段。10000也曾4.32s/WER2.5，10500/11000曾回到2.08s及专名近似输出；本次重新变长。目标/参考配对未变，同目标ICL仍21帧/1.68s、ASR“W. Flanders, uni.”、WER0.5/CER0.153846，未见同样长串内容。已扩大至两模式同目标和波形对照，但没有证实可直接修复的配对、裁剪或音频写出错误；保留异常，后续12000核对，不据此改LR或解码策略。
- SO sample00维持上轮已见改善：22帧/1.76s、WER0.25；sample02为58帧/4.64s、WER/CER0，sample06为48帧/3.84s、WER0.25。短ZH sample01为14帧/1.12s/比1.14286、CER0.666667，最长尾部零段0.171792s，10500超长未复发；ICL同句13帧/1.04s/CER0.333333、最长尾部零段0.214208s。SO sample04/12 CER0.538462/0.344828；sample12缩短至49帧/3.92s/比0.74242，尾部仍缺失。ICL sample00为17帧/1.36s/WER0.25；sample04为25帧/2s/CER0.384615，未立即EOS；sample12为42帧/3.36s/比0.63636/CER0.310345，虽CER降低但时长更短，不能视为内容完整。ICL sample02/06 WER0.125/0.166667（sample02末尾you know变为yeah，sample06专名和swamp仍错）。
- 12000现场评估：03:53:01 speaker_only两条metrics、ICL零条、两summary均尚无。SO00于03:52:21落盘24帧/1.92s/WER0.5，SO01于03:52:49落盘12帧/0.96s/CER0.666667；两WAV均24kHz、有限、时长匹配、EOS、不截断、min2。快照时四GPU利用率0%发生在刚完成保存的时间点；之后rank1–3为100%、metrics不断新增，排除仅据快照判卡死。未拼接部分结果构造新summary，也未等待完整评估后才结束巡检。
- 资源（03:52:42附近）：GPU0–3占用77519/77211/76025/77859 MiB（各81920），利用率0/100/100/100%，compute-apps仅本run四rank。主机已用238GiB、available1.7TiB、无swap，磁盘可用583202.25GiB；无新OOM或资源短缺证据。
- 判断/处理/未解决：保持训练代码、配置、预算6000/9000、workers16/prefetch2、LR与两个epoch38539步目标。训练、供数、保存验证正常，未证实需启停或修复实现的运行故障。继续跟踪长ICL历史近静音反复、SO05再次超长和其他内容缺失；主会话9000实验已补充官方贪心生成可复现PCM16前近静音的证据，本轮未重复加载GPU模型。当前12000/38539，退出及final-verification文件均不存在；未执行最终冻结检查、未写passed=true，最终仍须38539 COMPLETE、最终val/双summary和include-speaker冻结检查实际通过。
- 命令/修改/结果：cat/tail读约定、manual、快照与最近review/status；Python解析日志统计finite/min/median/max/慢步、扫描恢复后错误、核对/proc身份、checkpoint分片/metadata/signature、datetime/shutil.disk_usage；nvidia-smi GPU/compute-apps、free -h；soundfile/numpy核验11500的16条metrics/WAV、配对/时长及长ZH/短ZH/EN05零段，并复核12000两条metrics/WAV、manual/latest和退出验收文件。全部成功；唯一人工写入为本段追加。未改代码/配置/数据、启停恢复或发信号、启动timer/Codex/subagent、提交推送发消息或删除清理。本次巡检结束。


## 2026-09-10 04:23 UTC 单次巡检：12500步保存验证完成，12000步长ICL连续四次无截断

- 依据/进度：已读故障处置表、本文、manual、快照`supervision/20260910T042129Z.json`、最近巡检`20260910T035129Z.md`和status（上轮退出码0、固定session不变）。manual.active=false、交接观察至10170。上一现场/快照12000→本轮04:21:29快照12500→04:22:34现场及04:22:52复核12500，正在生成评估，metrics持续新增。12000双summary完整，12500 checkpoint/val已完成，不能将评估时step不变当作卡死。
- 进程/退出：training-process.json PID5005/start_ticks84647923与/proc匹配，cmdline为项目torchrun、4 ranks、原config、--resume latest；launcher5004/start84647921/PPID1、四rank5030–5033/start84648072/PPID5005均存活，launcher/torchrun为S、rank为R。training-exit.json不存在；以10000恢复初始化为界扫描日志，无新Traceback/OOM/Non-finite/Error/Aborted。快照旧Traceback是已记录的主动TERM历史，不是本次新错误。
- 近半小时12010–12500共50个每10步当步值全部有限：first_ce范围1.210874–1.346308、中位1.282574、末1.210874；residual_ce范围5.903436–5.992973、中位5.951023、末5.922611；grad_norm范围0.397385–0.508671、中位0.436212、末0.443391（clip前）。主干LR8.221127958e-5→8.071908330e-5、新参数2.466338387e-4→2.421572499e-4，按既定cosine连续下降。
- 动态batch/吞吐：帧填充94.4833%–99.8333%、中位98.0625%；token填充91.6333%–97.5500%、中位95.0736%；global samples309–394。step范围1.985113–2.260113s、中位2.118143s；音频秒/墙钟秒839.289–940.866、中位884.230；data_wait范围0.0002192–0.0005910s、中位0.0002691s。无>3s慢步，与上轮2.124192s/889.066音频秒每秒/等待0.0002735s相近，无持续供数恶化。rank0训练peak allocated最高56.6604GiB；以上为稀疏当步值，不代表含评估的窗口均值或全部rank峰值。
- Checkpoint/val：latest=step-00012500，COMPLETE时间04:20:09.267095 UTC；progress={step:12500,epoch:0,next_batch:12500}、world_size4、scheduler.last_epoch12500、LR8.071908330e-5/2.421572499e-4。12000/12500签名逐项相等，各自四distributed分片约2.093GB、.metadata1424887字节、四rng各14613字节齐全；结构核查不等于本轮实际试加载。11500由训练器keep_checkpoints=2轮转，本轮未清理或编辑metadata。12500 val first/residual CE1.330940/5.950863，较12000的1.337219/5.965698下降，15码本CE均有限（3.761661–6.721716）。

|最近完整12000步，每模式EN4/ZH4，greedy/min_new_frames=2|EN基础WER/CER|EN英语规范化WER/CER|ZH WER/CER|截断/零帧|
|---|---|---|---|---|
|speaker_only|0.388889/0.174825|0.388889/0.174825|1.25/0.428571|0/8、0/8|
|icl|0.166667/0.048951|0.166667/0.048951|1/0.547619|0/8、0/8|

- 评估核验：实读12000两summary、16条metrics/16WAV，均24kHz单声道、有限、时长与metrics匹配；目标ID/文本/speaker_reference_id/reference_text与11500一致，policy相同。本轮两种英语口径相同。11500→12000 EN WER speaker_only0.388889不变、ICL0.194444→0.166667，SO EN CER0.300699→0.174825；ZH CER0.547619→0.428571、0.428571→0.547619。每模式只有8条诊断，汇总包含繁简、数字、分词及实际错词影响，不能据此推断整体退化或调超参；未试听。
- 长ZH ICL sample03连续10500–12000四次不截断：88帧/7.04s、目标6.28s、比1.12102、EOS，RMS0.0378150、peak0.275482、零比例14.6982%，最长连续零段开头[0,0.510042]秒（11500为0.446708秒），前4秒/后3.04秒RMS0.035458/0.040709。ASR“又起了半個小時,現在是下午2點50,起了大概11公里,現在剩餘的電量是80%”，CER0.527778，有繁简/数字口径差异及起/骑、11/十等实际错误。多秒零段未复发，但不宣布历史根因已消失。
- speaker_only同目标89帧/7.12s/比1.13376、EOS、不截断，RMS0.0396150、零比例11.7609%、最长尾部零段0.031458s，前4秒/后3.12秒RMS0.038208/0.041349；CER0.5，ASR有“起/五时”等替换和繁简/数字差异。时长与连续能量正常不等于内容完全正确。
- SO EN sample05连续11500/12000两次明显内容错误：本轮29帧/2.32s、目标1.62s、比1.43210，较上轮3.04s缩短；ASR“t-bed tech at the Fly in the Suny.”、WER2.25/CER1.307692（上轮2.5/2.461538），RMS0.0721389、零比例0.1221%、最长零段0.0000833s，不是长静音。不能把时长/CER下降写成恢复。同目标ICL仍21帧/1.68s，ASR“WA Flanders Uni”、WER0.75/CER0.076923，WER升高包含WA与W A分词差异，内容比SO接近。两模式配对一致，未确认可修复的输入、裁剪或写WAV错误；保留复发记录，待12500完整结果验证是否持续，未自行改解码策略。
- 短ZH SO sample01为12帧/0.96s/CER0.666667，最长零段0.030750s；ICL同句13帧/1.04s、ASR“煮至奶白色後”、CER0.166667，仅后/後口径差异，最长零段0.023917s。旧超长未复发。SO sample00/02/06 WER0.5/0/0.25；sample04/12 CER0.615385/0.206897，sample12为50帧/4s/比0.75758、ASR尾部“videos videos”，仍有词/语气缺失，WER3受中英混合分词影响。ICL sample00为16帧/1.28s/WER0.5（Lilliquid），sample04为24帧/1.92s/CER0.538462（缺词及繁简），未立即EOS；sample12为46帧/3.68s/比0.69697，ASR“相信你跟師傅都看過我寄給你們的捐威尿”、CER0.655172，末段明显错/缺，不能把中文汇总变化全部解释成繁简。ICL sample02/06 WER0/0.083333，后者仍有Jeffreeze专名差异。
- 12500评估现场：04:22:52 speaker_only三条metrics、ICL零条、两summary尚无；SO00于04:21:33落盘23帧/1.84s/WER0.5，SO01于04:22:02落盘14帧/1.12s/CER0.833333，SO02于04:22:49落盘58帧/4.64s/WER0.1875（末尾yeah you know变为Yay）。三WAV均24kHz、有限、时长匹配、EOS、不截断、min2。文件持续新增证明当前评估推进，未拼接部分结果构造summary，也未等待整轮评估结束。
- 资源（04:22:34附近）：GPU0–3占用77519/77211/76025/77859 MiB（各81920），利用率0/100/100/100%，compute-apps仅本run四rank；生成评估时rank0瞬时低利用率且文件持续更新，不判卡死。主机已用238GiB、available1.7TiB、无swap，磁盘可用583155.04GiB，无新OOM或资源短缺证据。
- 判断/处理/未解决：保持训练代码、配置、预算6000/9000、workers16/prefetch2、LR和两个epoch38539步目标。训练供数、保存验证正常，没有证实需启停或修复实现的运行故障。继续跟踪SO05连续内容异常、ICL12末段缺失及长ICL历史近静音；主会话9000官方贪心生成复现PCM16前近静音的证据仍有效，本轮未重复加载模型或改采样/惩罚策略。当前12500/38539，退出及final-verification文件不存在；未执行最终冻结检查、未写passed=true，最终仍需38539 COMPLETE、最终val/双summary与include-speaker冻结检查实际通过。
- 命令/修改/结果：cat/tail读约定、manual、快照与最近review/status；Python解析日志作finite/min/median/max/慢步及恢复后错误扫描，核对/proc身份、checkpoint分片/metadata/signature、datetime/shutil.disk_usage；nvidia-smi GPU/compute-apps、free -h；soundfile/numpy核验12000的16条metrics/WAV、配对/时长及长ZH/短ZH/EN05零段，最后复核12500三条metrics/WAV、manual/latest、退出验收文件。全部成功；唯一人工写入为本段追加。未改代码/配置/数据、启停恢复或发信号、启动timer/Codex/subagent、提交推送发消息或删除清理。本次巡检结束。


## 2026-09-10 04:53 UTC 单次巡检：13000步保存验证完成，12500步SO短英文异常部分改善

- 依据/进度：已读故障处置表、本文、manual、快照`supervision/20260910T045129Z.json`、最近巡检`20260910T042129Z.md`及status（上轮退出码0、固定session不变）。manual.active=false，交接观察至10170。上一现场/快照12500→本轮04:51:29快照13000→04:52:36现场及04:52:59复核13000，当前生成评估持续落盘。12500双summary完整；13000 checkpoint/val完成，不能把评估中step不变判为卡死。
- 进程/退出：training-process.json PID5005/start_ticks84647923与/proc精确一致，cmdline为项目torchrun、4 ranks、原config、--resume latest；launcher5004/start84647921/PPID1、四rank5030–5033/start84648072/PPID5005均存活。launcher/torchrun为S、rank为R。training-exit.json不存在；以10000恢复初始化为界扫描后续日志，无新Traceback/OOM/Non-finite/Error/Aborted；快照收录的旧Traceback仍为此前主动TERM历史。
- 近半小时12510–13000共50个每10步当步值全部有限：first_ce范围1.181593–1.309338、中位1.267505、末1.260264；residual_ce范围5.879136–5.979296、中位5.923925、末5.929438；grad_norm范围0.381394–0.559982、中位0.443423、末0.437677（clip前）。主干LR8.068817139e-5→7.915187570e-5、新参数2.420645142e-4→2.374556271e-4，既定cosine连续下降。
- 动态batch/吞吐：帧填充95.0542%–99.7292%、中位98.4979%；token填充91.9778%–98.6444%、中位95.5347%；global samples318–417。step范围2.004859–2.296896s、中位2.127734s；音频秒/墙钟秒823.964–945.782、中位887.031；data_wait范围0.0002253–0.0005118s、中位0.0002746s。无>3s慢步，与上轮2.118143s/884.230音频秒每秒/等待0.0002691s相近，供数稳定。rank0训练peak allocated最高56.9690GiB；以上为稀疏当步记录，不是含评估的窗口平均或全部rank峰值。
- Checkpoint/val：latest=step-00013000，COMPLETE时间04:49:11.050575 UTC；progress={step:13000,epoch:0,next_batch:13000}、world_size4、scheduler.last_epoch13000、LR7.915187570e-5/2.374556271e-4。12500/13000签名相等，每份四distributed分片约2.093GB、.metadata1424887字节、四rng各14613字节齐全；仅结构核查，未实际试加载。12000由训练器keep_checkpoints=2轮转，本轮未清理或编辑metadata。13000 val first/residual CE1.325073/5.939527，较12500的1.330940/5.950863下降，15码本CE有限（3.755999–6.706539）。

|最近完整12500步，每模式EN4/ZH4，greedy/min_new_frames=2|EN基础WER/CER|EN英语规范化WER/CER|ZH WER/CER|截断/零帧|
|---|---|---|---|---|
|speaker_only|0.305556/0.174825|0.305556/0.174825|1/0.547619|0/8、0/8|
|icl|0.166667/0.062937|0.166667/0.062937|1/0.273810|0/8、0/8|

- 评估核验：实读12500两summary、16条metrics/16WAV，均24kHz单声道、有限、时长与metrics匹配；目标ID/文本/speaker_reference_id/reference_text与12000一致，policy相同。本轮英语两口径相同。12000→12500 EN WER speaker_only0.388889→0.305556、ICL0.166667不变；ZH CER0.428571→0.547619、0.547619→0.273810。只有8条诊断，不能据汇总推断全验证质量或调整超参；未试听。
- 长ZH ICL sample03连续10500–12500五次不截断：86帧/6.88s、目标6.28s、比1.09554、EOS；RMS0.0377514、peak0.311584、零比例12.5164%，最长连续零段开头[0,0.514208]秒，与12000的0.510042秒相近；前4秒/后2.88秒RMS0.037054/0.038699。ASR“又起了半小小时,现在是下午两点五时,起了大概十公里,现在剩余的电量是百分之八十。”，CER0.111111，较上轮0.527778降低，既有简体/汉字数字口径改善，也有实际少量起/骑、小字重复、五时/五十差异。历史多秒零段未复发，但不据此声称根因消失。
- speaker_only同目标82帧/6.56s/比1.04459、EOS、不截断，RMS0.0376958、零比例8.8097%、最长零段0.009667s，前4秒/后2.56秒RMS0.038545/0.036329。ASR“就起了半小時限時下午2點50,起了大概10公里,現在剩餘的電量是80%”，CER0.611111，仍有替换/缺失和繁简/数字差异，时长及连续能量正常不等于内容正确。
- SO EN sample05本次部分改善：25帧/2s、目标1.62s、比1.23457，较12000的2.32s缩短；ASR“Tita pay Flanders uni.”、WER0.75/CER0.538462，较上轮2.25/1.307692改善，RMS0.0698573、零比例0.1271%、最长零段0.000125s。没有上一轮长串错误，但仍明显偏离W.A. Flinders Uni.，不能称完全恢复。ICL同目标20帧/1.6s、ASR“WF Flanders, uni”、WER0.75/CER0.153846，含首字母和专名/分词差异。配对保持一致，本轮未确认需修改输入或生成实现的直接证据。
- 短ZH SO sample01为14帧/1.12s/CER0.833333，最长零段0.000792s；ICL同句12帧/0.96s/CER0.333333（阻止/煮至），最长尾部零段0.166083s。旧超长未复发。SO sample00/02/06 WER0.5/0.1875/0.25，sample02末尾yeah you know变为Yay；SO sample04/12 CER0.615385/0.379310，sample12缩至46帧/3.68s/比0.69697，末段仍缺失。ICL sample00为18帧/1.44s/WER0.25；sample04为24帧/1.92s/CER0.461538（先迟/坚持及繁简），未立即EOS；sample12为43帧/3.44s/比0.65152、ASR“相信你跟师父都看过我寄给你们的绝位 videos”、CER0.379310，较上轮0.655172降低但仍缩短且缺尾部。ICL sample02/06 WER0/0.166667。
- 额外波形边界核验：12500 SO sample02峰值恰为1，进一步用int16计数：111360个采样中仅1个为-32768、无32767、最长连续边界点1个（1/24000秒）。该点不足以证明持续削波或解释末尾内容缺失；未改幅度或重新写WAV，保留现有产物。
- 13000评估进展：04:52:59 SO四条metrics、ICL零条、两summary尚无；SO00/01/02/03分别于04:50:25、04:50:55、04:51:38、04:52:32落盘，24kHz、有限、时长匹配、EOS、不截断。依次为27帧/2.16s/WER0.75、14帧/1.12s/CER0.5、53帧/4.24s/WER0、82帧/6.56s/CER0.583333。新文件证明生成持续推进；未拼接部分结果构造summary或等待整轮结束。
- 资源（04:52:36附近）：GPU0–3占用77519/77211/76025/77861 MiB（各81920），利用率72/26/52/57%，compute-apps仅本run四rank。主机已用238GiB、available1.7TiB、无swap；磁盘可用583047.69GiB，无新OOM或资源短缺证据。
- 判断/处理/未解决：保持训练代码、配置、预算6000/9000、workers16/prefetch2、LR及两个epoch38539步目标。训练供数、保存验证正常，部分生成异常改善，未证实需启停或修复实现的运行故障。继续跟踪SO05反复、ICL12末段缺失与长ICL历史近静音；主会话9000官方贪心生成复现PCM16前近静音的证据仍有效，本轮未重复加载模型或自行改解码策略。当前13000/38539，退出及final-verification文件不存在；未执行最终冻结检查、未写passed=true，最终仍需38539 COMPLETE、最终val/双summary和include-speaker冻结检查实际通过。
- 命令/修改/结果：cat/tail读约定、manual、快照、最近review/status；Python解析日志统计finite/min/median/max/慢步及恢复后错误，核对/proc身份、checkpoint分片/metadata/signature、datetime/shutil.disk_usage；nvidia-smi GPU/compute-apps、free -h；soundfile/numpy核验12500的16条metrics/WAV、配对/时长/长ZH短ZH和EN05零段、SO02 int16边界点，并复核13000四条metrics/WAV与manual/latest/退出验收文件。全部成功；唯一人工写入为本段追加。未改代码/配置/数据、启停恢复或发信号、启动timer/Codex/subagent、提交推送发消息或删除清理。本次巡检结束。


## 2026-09-10 05:24 UTC 单次巡检：13500步运行正常，13000步出现短英文重复与ICL最短长度后EOS

- 依据/进度：已读故障处置表、本文、manual、快照`supervision/20260910T052129Z.json`、最近巡检`20260910T045129Z.md`及status（上轮退出码0、固定session不变）。manual.active=false、交接观察至10170。上一现场/快照13000→本轮05:21:30快照13500→05:23:07现场13500；05:23:57仍13500但SO评估已新增到六条。13000双summary完整，13500 checkpoint/val完成、生成评估进行中，不判停滞。
- 进程/退出：training-process.json PID5005/start_ticks84647923与/proc匹配，cmdline为项目torchrun、4 ranks、原config、--resume latest。launcher5004/start84647921/PPID1、四rank5030–5033/start84648072/PPID5005均存活，launcher/torchrun为S、rank为R。training-exit.json不存在；以10000恢复初始化为界扫描后续日志，无新Traceback/OOM/Non-finite/Error/Aborted。快照旧Traceback是此前主动TERM记录，不是本次故障。
- 近半小时13010–13500共50个每10步当步值全部有限：first_ce范围1.185244–1.325296、中位1.252684、末1.244649；residual_ce范围5.871644–5.949180、中位5.914920、末5.903801；grad_norm范围0.371320–0.526097、中位0.434343、末0.434240（clip前）。主干LR7.912009100e-5→7.754238549e-5、新参数2.373602730e-4→2.326271565e-4，既定cosine连续下降。
- 动态batch/吞吐：帧填充94.4417%–99.9208%、中位98.1438%；token填充91.1861%–97.8361%、中位95.3833%；global samples305–403。step范围1.951762–2.235470s、中位2.116134s；音频秒/墙钟秒818.867–958.560、中位890.643；data_wait范围0.0002453–0.0005308s、中位0.0002751s。无>3s慢步，与上轮2.127734s/887.031音频秒每秒/等待0.0002746s相近，无持续供数恶化。rank0训练peak allocated最高57.0376GiB；以上是稀疏当步值，不是含评估的窗口平均或全部rank峰值。
- Checkpoint/val：latest=step-00013500，COMPLETE时间05:18:41.171224 UTC；progress={step:13500,epoch:0,next_batch:13500}、world_size4、scheduler.last_epoch13500、LR7.754238549e-5/2.326271565e-4。13000/13500签名相等，各自四distributed分片约2.093GB、.metadata1424887字节、四rng各14613字节齐全；仅结构核查，未实际试加载。12500由训练器keep_checkpoints=2轮转，本巡检未清理或编辑metadata。13500 val first/residual CE1.316129/5.922186，较13000的1.325073/5.939527下降，15码本CE有限（3.742644–6.687010）。

|最近完整13000步，每模式EN4/ZH4，greedy/min_new_frames=2|EN基础WER/CER|EN英语规范化WER/CER|ZH WER/CER|截断/零帧/仅2帧|
|---|---|---|---|---|
|speaker_only|0.805556/0.706294|0.833333/0.706294|1.125/0.523810|0/8、0/8、0/8|
|icl|0.083333/0.048951|0.083333/0.048951|1/0.595238|0/8、0/8、1/8|

- 评估核验：实读13000两summary、16条metrics/16WAV，全部24kHz单声道、有限、时长匹配；目标ID/文本/speaker_reference_id/reference_text与12500一致、policy相同。SO英语规范化WER较基础值多1/36，来自sample00基础WER0.75→规范化1（ASR含didn't）；SO sample05两口径相同。12500→13000 SO EN规范化WER0.305556→0.833333，主要受单条大量插词拖累；ICL EN WER0.166667→0.083333，ZH CER0.273810→0.595238。每模式8条不足以判断整体退化，不能由汇总直接改训练超参；未试听。
- SO EN sample05显著复发：108帧/8.64s、目标1.62s、比5.33333，最终EOS、不截断；ASR“Ti da brutte tā EMBUM dadi dadi … da ליis shetime”，大量重复，WER5.75/CER6.538462（23词错误、85字符错误）。RMS0.0620383、peak0.337646、零比例0.07282%、最长零段0.0000833s，前4秒/次4秒/末0.64秒RMS0.066125/0.059038/0.053345，属于有能量的长重复，不能解释成近静音或正常停止。12500刚缩短到2秒，本次再次增长，说明恢复有反复。同目标ICL为22帧/1.76s、ASR“W.A. Flinders-Uni”、WER/CER0，配对一致。未保存本次预测codec序列，现有波形/ASR尚不足以确认重复预测的具体机制。
- ICL ZH sample04过早结束复发：仅2帧/0.16s、目标2.3s、比0.069565，随后EOS、不截断，RMS3.0374e-5、peak0.0001831、零比例73.0469%，是近静音短输出。ASR“字幕by索兰娅”、WER/CER1，不能将其视为生成了有效目标语音。该输出满足min_new_frames=2的下界，却没有有效完成内容；零帧计数为0不能掩盖异常。与修复前立即EOS不同，当前证据不指向EOS屏蔽规则失效，不自行继续提高最短长度。12500同句24帧/1.92s，本次明显退化；13500 ICL尚未开始，后续必须跟踪该目标。
- 为上述两异常扩展输入核验：实读当前val清单并按目标ID定位，speaker.py确认tar参考走decode_emilia_audio，本轮CPU解码实际参考源而非只检查名义缓存路径。EN05目标emilia2:4318cc5dac2cbeca_599_000为24kHz/1.62s、RMS0.0773468、peak0.624962；参考_429_001为4.16s、RMS0.0694285、peak0.729818。ZH04目标emilia2:8fa50bd7ff55f0d6_2827_000为24kHz/2.3s、RMS0.0771031、peak0.494960；参考_2543_000为1.49s、RMS0.1042421、peak0.450368。四音频均有限且非空，两参考精确零比例0；未materialize或改写缓存。EN05原目标reference_asr为“W.A. Flinders-Uni”、WER/CER0，ZH04原目标ASR内容与文本对应、CER0.384615仅繁简差异。没有支持输入损坏、空参考或错误目标配对的证据；这不等于已复现模型生成根因。
- 长ZH ICL sample03连续10500–13000六次不截断：82帧/6.56s、目标6.28s、比1.04459、EOS，RMS0.0345933、零比例11.4914%、最长开头零段0.53075s（上轮0.514208s），前4秒/后2.56秒RMS0.033642/0.036029。CER0.527778，较12500的0.111111升高含繁简及数字口径变化，也有起/骑等替换；多秒零段未复发。SO同句82帧/6.56s，RMS0.0385873、零比例8.4057%、最长零段0.017167s，CER0.583333；时长与能量正常不等于内容正确。
- 其余样本：SO00 27帧/2.16s、ASR“I didn't know the liquid spears.”，有多余词；SO02 WER0，SO06 WER0.25，短ZH01 14帧/1.12s/CER0.5、旧超长未复发；SO04 CER0.307692、SO12 62帧/4.96s/比0.93939/CER0.551724，时长接近目标但末段仍错。ICL00 19帧/1.52s/WER0.25、01 13帧/1.04s/CER0.333333、02 WER0、06 WER0.166667；ICL12 42帧/3.36s/比0.63636/CER0.551724，仍有尾部缺失和替换。
- 13500现场进展：05:23:57 SO六条metrics、ICL零条、两summary尚无。SO00/01/02/03/04/05落盘时间分别05:19:59、05:20:28、05:21:12、05:22:08、05:22:42、05:23:16，证明生成持续推进。SO05已缩至24帧/1.92s，ASR“Too bad you find us uni.”、WER1.25/CER0.846154，长重复本次未出现，但仍错；SO00为31帧/2.48s/WER2（多词），短ZH01仍14帧/1.12s/CER1.166667（内容错但未超长）；SO02 WER0，03/04 CER0.527778/0.615385。此处为metrics检查，未重新读取13500六WAV或构造不完整summary，不等待整轮结束。
- 资源（05:23:07附近）：GPU0–3占用77519/77211/76025/77861 MiB（各81920），利用率0/100/100/100%，compute-apps仅本run四rank；rank0瞬时低利用率与生成阶段相符且文件持续新增，不判卡死。主机已用238GiB、available1.7TiB、无swap；磁盘可用582963.59GiB，无新OOM或资源短缺证据。
- 判断/处理/未解决：保持训练代码、配置、预算6000/9000、workers16/prefetch2、LR与两个epoch38539步目标。训练供数、保存验证正常，质量异常已逐项记录并扩展到实际输入读取，尚未证实可直接修复的实现/数据故障；不靠增加最短生成长度或改LR掩盖问题。后续优先核对13500 ICL04是否仍只生成下界帧数，并继续跟踪SO05/00重复、ICL12尾部缺失。原始生成codec/logits尚缺，若这些异常持续，需要同checkpoint生成诊断才能选择干预；本轮未另载GPU模型或打断训练。当前13500/38539，退出及final-verification文件不存在，未执行最终冻结检查、未写passed=true；最终仍须38539 COMPLETE、最终val/双summary和include-speaker冻结检查实际通过。
- 命令/修改/结果：cat/tail读约定、manual、快照/最近review/status；Python解析完整日志统计finite/min/median/max/慢步及恢复后错误，核对/proc身份、checkpoint分片/metadata/signature、datetime/shutil.disk_usage；nvidia-smi/free；soundfile/numpy读13000的16metrics/WAV，比较配对/时长与长ZH/短ZH/EN05零段；rg/sed读sources.py/speaker.py及config，CPU decode_emilia_audio核验EN05/ZH04实际tar参考与目标音频；最后读13500六metrics、manual/退出验收文件。全部成功；唯一人工写入为本段追加。未改代码/配置/数据、启停恢复或发信号、启动timer/Codex/subagent、提交推送发消息或删除清理。本次巡检结束。


## 2026-09-10 05:53 UTC 单次巡检：14000步保存验证完成，13500步两帧近静音从ICL中文转到英文目标

- 依据/进度：已读故障处置表、本文、manual、快照`supervision/20260910T055129Z.json`、最近巡检`20260910T052129Z.md`及status（上轮退出码0、固定session不变）。manual.active=false，交接观察至10170。上一现场/快照13500→本轮05:51:30快照14000→05:52:35现场及05:52:54复核14000，当前生成评估持续落盘。13500双summary完整；14000 checkpoint/val完成，生成阶段step不变不是停滞。
- 进程/退出：training-process.json PID5005/start_ticks84647923与/proc匹配，cmdline为项目torchrun、4 ranks、原config、--resume latest；launcher5004/start84647921/PPID1、四rank5030–5033/start84648072/PPID5005均存活，launcher/torchrun为S、rank为R。training-exit.json不存在；以10000恢复初始化为界扫描日志，无新Traceback/OOM/Non-finite/Error/Aborted；快照旧Traceback仍为已记录的主动TERM历史。
- 近半小时13510–14000共50个每10步当步值全部有限：first_ce范围1.197799–1.310238、中位1.260540、末1.233182；residual_ce范围5.860354–5.941077、中位5.910566、末5.908734；grad_norm范围0.365358–0.525468、中位0.426228、末0.413514（clip前）。主干LR7.750978363e-5→7.589343039e-5、新参数2.325293509e-4→2.276802912e-4，既定cosine连续下降。
- 动态batch/吞吐：帧填充95.5583%–99.6583%、中位98.4042%；token填充91.9472%–98.1750%、中位95.4194%；global samples301–408。step范围2.023065–2.512952s、中位2.128165s；音频秒/墙钟秒760.444–927.662、中位885.795；data_wait范围0.0002227–0.0005073s、中位0.0002745s。无>3s慢步，较上轮2.116134s/890.643音频秒每秒/等待0.0002751s相近，无持续供数下降。rank0训练peak allocated最高56.9478GiB；以上是稀疏当步值，不是含评估的窗口平均或全部rank峰值。
- Checkpoint/val：latest=step-00014000，COMPLETE时间05:47:32.226801 UTC；progress={step:14000,epoch:0,next_batch:14000}、world_size4、scheduler.last_epoch14000、LR7.589343039e-5/2.276802912e-4。13500/14000签名相等，各自四distributed分片约2.093GB、.metadata1424887字节、四rng各14613字节齐全；仅结构核查，未实际试加载。13000由训练器keep_checkpoints=2轮转，本巡检未清理或编辑metadata。14000 val first/residual CE1.310338/5.909446，较13500的1.316129/5.922186下降，15码本CE有限（3.731853–6.671785）。

|最近完整13500步，每模式EN4/ZH4，greedy/min_new_frames=2|EN基础WER/CER|EN英语规范化WER/CER|ZH WER/CER|截断/零帧/仅2帧|
|---|---|---|---|---|
|speaker_only|0.388889/0.251748|0.388889/0.251748|1.125/0.5|0/8、0/8、0/8|
|icl|0.138889/0.146853|0.138889/0.146853|1/0.523810|0/8、0/8、1/8|

- 评估核验：实读13500两summary、16条metrics/16WAV，均24kHz单声道、有限、时长与metrics匹配；目标ID/文本/speaker_reference_id/reference_text与13000一致、policy相同。本轮两种英语口径相同。13000→13500 SO EN规范化WER0.833333→0.388889，主要来自sample05长重复缩短；ICL EN WER0.083333→0.138889，包含sample00本轮失去有效输出。只有8条诊断，不能把汇总变化当成全验证集趋势或据此调超参；未试听。
- ICL两帧近静音目标互换：上轮中文sample04本轮恢复24帧/1.92s/比0.83478，RMS0.0931227、peak0.415131、最长尾部零段0.051667s，ASR“其實人家畢竟堅持了這麼多年”、CER0.384615主要为繁简差异。但EN sample00从13000的19帧/1.52s变为仅2帧/0.16s、目标1.81s、比0.088398，随后EOS，RMS6.7381e-5、peak0.0003357、零比例34.1667%，ASR空、WER/CER1。两帧计数总数仍1/8不能掩盖目标变换；零帧0且不截断也不能称内容正常。该样本同样满足min2下界而没有生成有效内容，不能声称EOS规则修复解决了所有过早结束。上轮EN05/ZH04实际输入核验未发现损坏；本轮配对仍一致，未重复读取已验证输入或臆测新的数据故障。
- SO短英文sample05本轮24帧/1.92s/目标1.62s、比1.18519，RMS0.0714580、零比例0.5946%、最长零段0.000333s，ASR“Too bad you find us uni.”、WER1.25/CER0.846154。13000的108帧/8.64s长重复未复发，但内容仍错。同目标ICL为21帧/1.68s、ASR“W.A. Flinders-Uni.”、WER/CER0；两模式差异和输入配对证据不支持为单句改训练目标。SO sample00本轮31帧/2.48s/比1.37017、ASR“I know some juror I know in the spears.”、WER2/CER1.166667，成为另一个短句内容异常，不能只盯sample05。
- 长ZH ICL sample03连续10500–13500七次不截断：82帧/6.56s、目标6.28s、比1.04459、EOS，RMS0.0364976、零比例13.9533%、最长开头零段0.438583s（上轮0.53075s），前4秒/后2.56秒RMS0.036063/0.037167。ASR“月氣了半小時,先是下午2點50,氣了大概10公里,現在升去的電量是80%”，CER0.611111，仍有错词、缺词与繁简/数字口径差异。SO同目标82帧/6.56s，RMS0.0383373、零比例12.1875%、最长零段0.184s（5.52425–5.70825秒），前后能量相近，CER0.527778；均未复发历史多秒零段，但不宣称根因消失。
- 其余内容：SO短ZH01仍14帧/1.12s但CER1.166667（“咱们就先来一杯色后”），时长正常不能掩盖错词；SO02 WER0、SO06 WER0.083333（Enter/In差异），SO04 CER0.615385，SO12 50帧/4s/比0.75758/CER0.275862，仍有替换/语气缺失。ICL01 12帧/0.96s/CER0.333333（组织/煮至）、02 WER0、06 WER0.083333（Jeffreeze）；ICL12 44帧/3.52s/比0.66667、ASR“相信你跟舒服都看过我寄给你们的卷微调”、CER0.517241，尾部仍错/缺。
- 14000评估进展：05:52:54 SO七条metrics、ICL零条、两summary尚无。SO00至06分别于05:48:36、05:49:04、05:49:48、05:50:44、05:51:18、05:51:50、05:52:32落盘，持续推进。SO05为24帧/1.92s、ASR“TWA, Flanders, UNI.”、WER0.75/CER0.153846，比13500改善但未完全正确；SO00 24帧/1.92s/WER0.75，SO01 13帧/1.04s/CER0.666667，SO02 WER0.0625，SO03/04 CER0.5/0.538462，SO06 WER0.25。这里只检查新metrics，未重新读取14000七WAV；未拼接部分结果构造summary，不等待整轮结束。
- 资源（05:52:35附近）：GPU0–3占用77519/77211/76025/77861 MiB（各81920），利用率70/31/40/32%，compute-apps仅本run四rank。主机已用238GiB、available1.7TiB、无swap；磁盘可用582838.86GiB，无新OOM或资源短缺证据。
- 判断/处理/未解决：保持训练代码、配置、预算6000/9000、workers16/prefetch2、LR及两个epoch38539步目标。训练供数、保存验证正常；生成异常有实际恢复也有目标间反复，尚无证实可直接修复的实现/数据故障，不自行加大最短长度或改LR。下一轮优先核对14000 ICL00/04是否仍仅两帧，以及SO00/05和ICL12内容；若最低帧输出反复持续，同checkpoint的原始codec/EOS logits诊断是选择干预所需证据，不能用0零帧汇总宣布解决。本轮未另载模型或打断训练。当前14000/38539，退出及final-verification文件不存在；未执行最终冻结检查、未写passed=true，最终仍需38539 COMPLETE、最终val/双summary及include-speaker冻结检查实际通过。
- 命令/修改/结果：cat/tail读约定、manual、快照/最近review/status；Python解析完整日志统计finite/min/median/max/慢步及恢复后错误，核对/proc身份、checkpoint分片/metadata/signature、datetime/shutil.disk_usage；nvidia-smi/free；soundfile/numpy实读13500的16metrics/WAV、比较配对/时长及長ZH/ICL04/EN05零段；最后核对14000七metrics和manual/退出验收文件。全部成功；唯一人工写入为本段追加。未改代码/配置/数据、启停恢复或发信号、启动timer/Codex/subagent、提交推送发消息或删除清理。本次巡检结束。


## 2026-09-10 06:17 UTC 主会话进度核查与评估规模说明

- 当前已到14500步，checkpoint和验证loss完成；first/residual CE为1.298761/5.895098，双模式音频评估正在进行。14000步双summary完整；英语规范化WER为speaker_only 0.305556、ICL 0.083333，中文CER为0.488095、0.428571。
- 全部512条验证数据（中英各256）参与验证loss；其中511条有合格的同speaker、同语言、不同文本训练参考。当前eval.num_samples=8，每个模式固定生成8条（中英各4），两种模式共16条音频；生成WER/CER并非全验证集统计。
- 用户提到一个epoch后停止，随后明确先不停止当前训练。本轮只核查既有停止/恢复逻辑，未修改max_steps=38539或schedule_steps=38539，未停止、重启或发送训练信号，也未安排未来自动重启。第一epoch为19270步，但该停止点尚未应用。当前训练进程及原巡检继续运行。
- 按用户明确的commit/push请求提交当前巡检文档；无训练代码、配置或checkpoint元数据变更。
