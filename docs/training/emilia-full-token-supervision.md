# Emilia 全量 token-loss 训练监督记录

本轮目标和筛选标准见 [训练说明](emilia-full-token.md)。用户已授权排除 ASMR、考虑经质量筛选的 dialogue、完成去重后直接启动 token-loss 训练，并在后台监督。

## 2026-09-12 启动前检查

- 先前流水线在准备阶段停止，正式训练没有开始；321 个暂存 raw 分片含 ASMR，准备重新生成，保留旧清单作追溯。
- 全量 ID 清点完成。short/long 合计 23,317,092 条、约 42,333.71 小时；dialogue 质量过滤前新增 11,535,807 条，检测到 225,953 次重复 ID。
- dialogue 的元数据筛选、实际切片 DNSMOS 复核、跨说话人标注重叠排除及全局 ID 去重已接入准备流程。
- 96 条实际音频完成评分与 ASR 抽查；评分实现与修正整数窗口边界后的官方参考一致。相关报告在 `runs/emilia-full-token-scratch/preflight/dialogue-quality/`。
- 既有四卡 20 次更新与冻结权重预检通过。正式全量数据发布后，流水线仍会计算实际步数并对最长音频、最长文本执行四卡显存检查。
- 当前还处于数据准备启动前；不能将上述预检或旧 19,000-step 实验算作本轮正式训练。

## 2026-09-12 13:04:13 UTC 单次巡检

- 阶段：`prepare/running`，正式训练 step 0 → 0。读取指定 `20260912T130214Z.json`、现场状态与此前记录；流水线于 13:02:14 UTC 启动，本次约 13:03–13:04 UTC 观察，尚无半小时训练指标可比较。`manual-observation.active=true`（13:02:13 UTC 更新），因此仅核对并追加巡检证据。
- 进程身份：入口 PID 355930/start_ticks 105903983；质量筛选 355993、导出 355994、codec supervisor 355995 的 start_ticks 均为 105904004，cmdline 与本轮数据及状态文件一致；三个准备阶段 returncode 均为 null。codec rank 0–3 为 PID 355997–356000/start_ticks 105904009，父进程 355995，命令为四卡 encode；质量筛选父进程下确有 32 个 spawn worker，另有 resource tracker，均非僵尸。完整现场身份记录：`runs/emilia-full-token-scratch/supervision/20260912T130413Z-inspection.json`。
- 导出推进：13:03:24 左右 raw 为 156 分片，13:03:40 为 194 分片；后者 EN 837,695 条/1,017.16 小时，ZH 751,553 条/854.60 小时。同期 exporter CPU ticks 2523 → 3046，wchar 721,328,398 → 897,805,319。此为导出中间统计，不是最终数据规模或平衡训练预算。
- dialogue：13:03:40 已有 19 个 stats、19 个完成 score JSONL，以及 32 个处理中 score 文件，score 总大小 233,561 字节。`dialogue-quality/status.json` 尚未生成；代码每完成 25 个分片才发布状态，因此与当前现场一致。代表 worker 356221/356254 的 CPU ticks 分别 6460 → 7981、6419 → 7947。配方确认 metadata≥3.4、OVRL≥3.4、SIG≥3.5、BAK≥4.0、clipped_fraction≤0.001，排除其他标注说话人重叠。未修改筛选、ASMR 排除或去重规则；音质筛选不等于完整文本对齐保证，未将缺失全量 ASR 分数认定为故障。
- codec：四个 worker CPU ticks 各增加约 1532（约 15 秒 CPU），rchar 从约 3.78–3.95 GB 增至 4.60–4.84 GB；四进程文件描述符均打开旧 `emilia-short-en-zh-10000h/train.jsonl`。结合加载缓存完成后才输出 cache_records 的代码，判断仍在加载旧缓存索引；prepared/codes 文件数为 0，四份 worker status 和 codec 日志尚无输出，不能据此判断卡死。
- 日志与资源：本 run pipeline、prepare 及 codec 日志未检出 Traceback/Error/Exception/OOM/NaN。`prepare-0.log` 最后修改于 12:12:02 UTC，早于本次启动，其中 short 导出行属于历史内容，不用作本轮进展依据。13:03:05 四卡各用 423 MiB、GPU 利用率 0%；主机已用约 105 GiB、可用约 1.9 TiB，无 swap；项目和数据所在 GPFS 可用约 564 TiB（47% 已用）。CPU/I/O 和新分片提供了维持运行的依据。
- 完成条件：`PREPARATION_COMPLETE`、`EXPORT_COMPLETE.json`、`DIALOGUE_QUALITY_COMPLETE.json`、`preparation.json` 均不存在；本轮 `training-process.json`、`training-plan.json`、`train.log`、checkpoints 均不存在。loss/grad/LR、训练吞吐/data_wait、val 和生成指标尚不适用，未生成 startup/final verification。
- 判断与动作：准备工作有明确进展，本次无需恢复；未改代码、配置或数据，未启停进程。只保存本次检查 JSON 并追加本文，然后结束本轮巡检。
- 剩余事项：继续等待质量筛选、short/long 导出及缓存加载/编码完成；exporter 后续需等待 dialogue 完成再汇入合格数据。正式数据发布后由原入口计算一个按音频时长中英 1:1 的平衡 epoch 真实步数，核对原始 assembled 初始化与 token loss；正式训练开始后再完成至少 20 次更新的启动验证。保留手动观察限制，后续以现场标记为准。

## 2026-09-12T13:08:20.930584+00:00 后台接管

- 修正后的流水线于 13:02:14 UTC 启动，PID 355930；质量筛选、导出、四卡编码并行运行，子进程身份已核对。当前导出 1095 个 raw 分片、完成 122 个 prepared 分片；质量报告覆盖 100/8059 个来源分片。
- 新准备数据前4份分片共32,768条全部检查ASMR路径排除和分片内ID唯一，抽查12份NPZ的SHA-256、形状和token范围通过。四个codec worker都已产出，当前首先复用既有匹配缓存。报告：preparation-startup-verification.json。
- 错误旧raw清单保留于数据目录 selection-history/20260912T130213Z-before-asmr-exclusion；没有混入新分片。全量ID去重在export，最终发布再次检查全局ID唯一。
- 监督PID 355981，每1800秒巡检，首轮review退出码0、状态monitoring，固定session 01a095b6-1b5c-7d42-8676-4a0347a52160。manual-observation.active现为false，后台接管必要诊断和恢复。流水线tmux为emilia-full-token，监督tmux为emilia-full-supervision。
- 当前正式训练尚未开始；编码和划分完成后自动计算一个平衡epoch目标步数，通过四卡显存检查再从原assembled模型以token loss开始。后台负责前20次更新核验、后续checkpoint/验证/生成检查和最终验收。
- 验证：相关原21项测试通过；新增长音频尾部窗口回归后，3项inventory/筛选测试通过；96条评分实现数值核对通过；真实来源分片筛选与恢复检查通过。详细状态：background-handoff.json。

## 2026-09-12 13:34:19 UTC 单次巡检

- 阶段与范围：`prepare/running`，正式训练 step 0 → 0；先读取 manual、pipeline、指定 `20260912T133214Z.json` 和最近巡检/接管记录。manual 自 13:08:20 UTC 起为 false，本次现场仍为 false。快照的 process=null 指正式训练尚未启动，并非准备入口退出。
- 身份和退出码：入口 PID 355930/start_ticks 105903983，cmdline 为本轮 `run_emilia_full.py`；准备三个子进程 PID 355993/355994/355995、start_ticks 105904004，逐一与 pipeline 保存的命令和本轮路径核对一致，returncodes=[null,null,null]。四个 codec worker PID 355997–356000、start_ticks 105904009，父进程 355995，rank 0–3/world_size=4。质量评分父进程下 32 个 spawn worker 全部运行；13:33:30 进程树共77个进程，无僵尸，质量 worker 合计 CPU 58,711.84 秒、RSS 9.82 GiB。完整身份、状态、日志尾部及资源证据：`runs/emilia-full-token-scratch/supervision/20260912T133419Z-inspection.json`。
- 近半小时数据进展：上轮13:04 raw为270分片，本轮为2,825；13:08接管记录 prepared为122，本次完成304份（另有处理中临时文件）。导出器日志已遍历 short/long，统计 EN 17,569.79小时、ZH 24,763.92小时，随后打印 `Waiting for actual-audio dialogue quality filtering`；raw 最新发布时间13:17:19。现场 exporter 处于S状态，且代码明确等待 `DIALOGUE_QUALITY_COMPLETE.json`，因此raw暂不增长是正常等待。`export-status.json`为最近一次flush的23,309,060条，尚含未发布尾部与后续dialogue的不完整统计，不能当作最终数据量。
- dialogue：相较13:08的100/8059，最新状态为525/8059；现场完成stats/score为527份，另有32份score处理中，文件持续更新。发布状态统计已评分 EN 36,084、ZH 4,242，保留 EN 24,788、ZH 2,708。配方阈值仍为metadata/OVRL≥3.4、SIG≥3.5、BAK≥4.0、clipped_fraction≤0.001，排除其他标注说话人重叠；配方 source SHA256 与当前评分脚本一致=True。export配方排除ASMR及2060个recording ID，按short→long→合格dialogue的短句ID优先级去重，无小时配额；未改变这些规则。音质标准不保证完整文本对齐，全量ASR缺失仍为已知设计边界。
- codec：四卡已越过旧缓存索引加载阶段；本次worker状态合计复用1,474,560条、新编码1,015,808条，共2,490,368条。rank 0–3最近完成chunk分别为[300, 293, 306, 307]。最近各worker日志5个分片间隔测得新增编码吞吐合计约745.1条/秒（不同rank各自时间窗口，不是训练吞吐）。13:33:06所读累计decode_wait约306–333秒、codec约1220–1233秒、write_wait约13–17秒；峰值allocated约13.16GiB/卡。13:33:30各codec累计CPU约2012–2058秒、rchar约110–117GB，与不断产生prepared的证据一致。
- 资源和异常：13:33:06 GPU各14,109MiB、利用率7/37/27/34%；主机已用约125GiB、可用约1.8TiB、无swap，GPFS空闲约563.4TiB、已用47%。扫描本轮pipeline、prepare、四worker日志，异常匹配0条。先前 preparation-startup-verification passed=true（32,768条检查ASMR排除及分片内ID唯一、12份NPZ抽验）作为已有验证证据，本轮未重复运行测试。
- 训练/检查点：数据四个完成标记和preparation发布报告尚未齐备（本轮检查均不存在），training-process、training-plan、train.log及本轮checkpoints也均不存在；loss/grad/LR、训练吞吐/data_wait、val/生成评估尚不适用。尚不满足正式启动或最终验收条件，未写startup/final-verification。
- 判断、动作与结果：准备阶段有明确CPU/I/O、评分文件与编码分片进展，维持原流水线运行，无需修复或恢复；本轮只写巡检JSON并追加本文，未更改代码、配置、数据，未启停进程。
- 剩余事项：等待dialogue质量完成后导出器汇入合格数据，并等待四卡编码和最终数据发布；由原入口计算中英按时长1:1、一个平衡epoch的实际目标步数，从原始assembled模型使用token loss启动，之后核验至少20次更新。本次巡检到此结束。

## 2026-09-12 14:03:03 UTC 单次巡检

- 阶段/step：`prepare/running`，正式训练 step 0 → 0；先读manual、pipeline、指定 `20260912T140214Z.json` 和最近记录。manual.active=false。快照process=null表示正式训练尚未启动，不代表准备流水线退出。完整本次证据保存于 `runs/emilia-full-token-scratch/supervision/20260912T140303Z-inspection.json`，对比基线为13:34:19巡检，间隔28.73分钟。
- 进程核对：入口355930/start_ticks=105903983，cmdline为本轮 `run_emilia_full.py`；质量筛选355993、exporter355994、codec supervisor355995的start_ticks均105904004，现场cmdline与pipeline记录逐一匹配，三个returncode均null。codec worker 355997–356000/start_ticks=105904009，父进程355995，rank 0–3/world_size=4及本轮路径匹配。32个质量评分worker均R状态，进程身份与上轮匹配者合计新增CPU约54,021秒；四codec同身份进程各新增CPU约1,887–1,930秒、rchar约129–149GB、wchar约1.19–1.33GB。未发现进程退出故障。
- 导出：raw仍2,825份，最新修改13:17:19 UTC；`export-status.json`仍EN 10,401,862条/17,518.65小时、ZH 12,907,198条/24,762.99小时，是最近flush的中间统计。`prepare-1.log`末尾明确等待实际音频dialogue筛选，当前 `DIALOGUE_QUALITY_COMPLETE.json`不存在；代码71–72行每5秒等待该标记。short/long已遍历，尾部及合格dialogue尚待发布，因此raw不变符合设计。
- dialogue：完成score/stats从527增至996（+469），另32份score处理中，score约38.74MB，最近修改14:03:02 UTC。发布状态按25分片更新，目前975/8059；已评分EN 67,895/ZH 8,145，保留EN 46,630/ZH 5,315，累计保留70.78/9.47小时。配方SHA256与当前脚本一致；metadata≥3.4、OVRL≥3.4、SIG≥3.5、BAK≥4.0、clipped_fraction≤0.001及跨说话人标注重叠排除均未变。export仍排除ASMR，按短句ID优先short→long→合格dialogue去重，不加小时配额。音质门槛不能充当完整文本对齐保证，缺失全量ASR分数仍为设计边界。
- codec：prepared完成304 → 457（+153），最近修改14:03:02 UTC。worker最近chunk为[452, 441, 462, 463]；累计复用1,474,560条、新编码2,269,184条，共3,743,744条；相较上轮新编码增加1,253,376条，按两轮采样间隔折算约727.0条/秒（分片发布统计有时间粒度，不是训练吞吐）。各worker状态数值有限，累计decode_wait约419–451秒、write_wait约30–33秒、codec约2809–2876秒，peak_allocated均13.16GiB/卡。
- 资源/日志：14:03:03 GPU各14,109MiB、利用率36/46/49/7%；主机MemAvailable约1877.5GiB，SwapTotal=0，14:02:36主机已用约127GiB；磁盘空闲约562.8TiB、占用47%。检查pipeline.log、三个prepare日志及四codec日志，Traceback/Error/Exception/OOM/NaN匹配0条。GPU瞬时低利用率与持续编码进展并存，无据判定卡死。
- 训练/检查点：PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均未发布；本轮training-process、training-exit、training-plan、train.log、checkpoints均不存在。正式loss/grad/LR、吞吐/data_wait、500步val与2000步生成尚不适用；本轮未写startup/final-verification。
- 判断与动作：评分与编码有明确新增产出，维持原流水线；无需修复或恢复。仅保存巡检JSON并追加本文，未改代码/配置/数据，未启停进程。剩余为dialogue筛选、合格数据汇入、编码及发布；完成后由原入口计算按时长中英1:1的一个平衡epoch真实步数，以原始assembled模型和token loss启动，再核验至少20次更新。未改变目标或套用旧19,000步预算；本轮巡检结束。

## 2026-09-12 14:32:54 UTC 单次巡检

- 阶段/step：`prepare/running`，正式训练step 0 → 0。已先读取manual、pipeline、指定 `20260912T143214Z.json` 和最近记录；manual.active=false。对比14:03:03巡检，间隔29.84分钟。完整现场证据：`runs/emilia-full-token-scratch/supervision/20260912T143254Z-inspection.json`。快照process=null是尚无正式训练进程，准备入口仍在运行。
- 进程/退出码：入口PID 355930/start_ticks 105903983，cmdline `.venv/bin/python -u scripts/run_emilia_full.py`；三个prepare进程355993/355994/355995、start_ticks 105904004，现场命令与pipeline逐一匹配，returncodes均null。四codec worker 355997–356000/start_ticks 105904009，父进程355995、本轮路径、rank 0–3/world_size=4均匹配。32个质量worker均R状态；与上轮同身份进程合计新增CPU约56,433秒。各codec新增CPU约1,833–1,893秒、rchar约74–93GB、wchar约1.97–2.10GB；rank 1瞬时D状态，但持续产出和I/O增量未显示持续无进展。
- 导出/等待：raw仍2,825份，最新发布时间13:17:19 UTC；export-status仍23,309,060条、EN 17,518.65/ZH 24,762.99小时，为最近flush统计。prepare-1.log明确等待实际音频dialogue筛选，DIALOGUE_QUALITY_COMPLETE尚不存在；short/long遍历后的尾部与筛选通过dialogue须后续发布，因此维持等待，无须重启exporter。
- dialogue：score/stats完成996 → 1308（+312），另32份score处理中，评分文件约56.97MB，最新修改14:32:53 UTC。发布状态为1300/8059，实际评分EN 98,813/ZH 14,335，保留EN 68,179/ZH 9,602，约107.54/17.73小时。配方与脚本SHA256相符；metadata≥3.4、OVRL≥3.4、SIG≥3.5、BAK≥4.0、clipped_fraction≤0.001、跨说话人标注重叠排除均维持。export配方仍排除ASMR及其2060个recording ID，按短句ID优先short→long→合格dialogue去重，无小时配额。本轮未改任何质量/筛选规则；音质门槛不等于文本对齐保证，全量ASR分数缺失仍属设计边界。
- codec：prepared完成457 → 1052（+595），最新修改14:32:54 UTC。四worker最近chunk为[1048, 1041, 1054, 1055]；累计复用6,056,225条、新编码2,608,984条，共8,665,209条；本观察间隔新增复用4,581,665条、新编码339,800条。合计处理约2749条/秒，其中新编码约190条/秒，反映此次缓存命中占比增加，不能只比较新编码条数判断吞吐退化。grouped_carriers_decoded分别[1672, 1269, 1995, 1988]。各worker状态数值有限=True；累计decode_wait约1285–1396秒、write_wait约33–36秒、codec约3457–3589秒；peak_allocated仍13.16GiB/卡。此为准备阶段指标，不是训练data_wait。
- 资源/日志：14:32:54四GPU各14,111MiB，瞬时利用率16/0/50/0%；主机已用约192GiB（上轮127GiB），MemAvailable约1812.3GiB，无swap；codec单进程RSS约26.8–28.4GiB（上轮14.6–14.8GiB）。内存增加已记录，当前仍有充足余量，无OOM证据，不据此调整参数。项目与数据GPFS可用约562.7TiB、47%已用。pipeline、prepare和worker日志未检出Traceback/Error/Exception/OOM/NaN/Inf；评分及编码文件持续更新，GPU瞬时0%不构成卡死依据。
- 训练/检查点：PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均不存在；本轮training-process、training-exit、training-plan、train.log和checkpoints亦不存在。正式loss/grad/LR、吞吐/data_wait、val及speaker_only/icl评估尚不适用；未生成startup/final-verification。
- 判断与执行结果：准备工作有明确进展，维持原流水线；本轮无需修复或恢复。仅保存巡检证据并追加本文，未更改代码/配置/数据或启停进程。剩余为dialogue筛选、合格数据汇入、编码和最终数据发布；随后由原入口计算一个中英按时长1:1平衡epoch的真实目标步数，以原始assembled模型和token loss启动，再验证至少20次更新。目标与授权范围不变，本次巡检结束。

## 2026-09-12 15:03:15 UTC 单次巡检

- 阶段/step：`prepare/running`，正式step 0 → 0。先读manual、pipeline、指定 `20260912T150214Z.json` 和最近巡检记录；manual.active=false。与14:32:54的观察间隔30.35分钟。本次现场证据：`runs/emilia-full-token-scratch/supervision/20260912T150315Z-inspection.json`。
- 进程及退出码：入口355930/start_ticks 105903983，命令为本轮 `run_emilia_full.py`；quality/export/codec supervisor分别355993/355994/355995，start_ticks均105904004，cmdline与pipeline记录完全匹配，三个returncode均null。codec 355997–356000/start_ticks 105904009、父进程355995、rank 0–3和本轮输入/输出路径与上轮现场一致。32个质量worker在运行（30R、2D）；同身份评分进程近半小时合计增加CPU约57,359秒，四codec分别增加CPU约1,915–1,943秒、rchar约71–76GB、wchar约0.82–0.87GB。没有进程退出或持续无进展证据，瞬时D状态结合持续产出不作卡死判断。
- 数据进展：raw仍2,825份/23,309,060条最近flush统计，EN 17,518.65、ZH 24,762.99小时；不是最终发布总量。exporter日志仍为 `Waiting for actual-audio dialogue quality filtering`，dialogue完成标记尚缺，维持正常等待。prepared完成1,052 → 1,202（+150），最新修改15:03:07 UTC；dialogue完成score/stats 1,308 → 1,549（+241），另32份score处理中，score约76.01MB，最新修改15:03:11 UTC。发布状态为1,525/8,059（每25份更新），已评分EN 125,705/ZH 23,551，保留EN 87,043/ZH 16,140，保留时长约144.73/31.00小时。
- 配方：评分脚本SHA256与配方相符；metadata≥3.4、OVRL≥3.4、SIG≥3.5、BAK≥4.0、clipped_fraction≤0.001及跨说话人标注重叠排除不变。export配方仍排除ASMR及其recording ID、按短句ID优先short→long→合格dialogue去重、无小时配额。本轮未改筛选或去重；音质门槛不保证完整文本对齐，缺失全量ASR分数是既定设计边界。
- codec指标：rank 0–3最近chunk为[1200, 1189, 1210, 1199]，累计复用6,670,967条、新编码3,236,156条，共9,907,123条；本间隔新增复用614,742条、新编码627,172条，按采样间隔合计约682条/秒（含复用，分片发布粒度）。grouped_carriers_decoded分别[8625, 7961, 8828, 8447]。状态数值有限=True；累计decode_wait约1854–2014秒、write_wait约37–41秒、codec约4633–4776秒；峰值allocated仍13.16GiB/卡。这些是准备指标，正式训练loss/grad/LR、吞吐/data_wait、val和生成指标尚不适用。
- 资源/日志：15:03:15四卡各14,111MiB，瞬时利用率23/0/22/5%；主机已用约202GiB（上轮192GiB），MemAvailable约1803.1GiB，无swap，codec RSS约27.2–29.2GiB/进程，仍有充足余量。磁盘空闲约562.6TiB、47%已用。pipeline、准备及codec日志未检出Traceback/Error/Exception/OOM/NaN/Inf。CPU/I/O和评分、编码文件增量支持维持运行，不以GPU瞬时0%判故障。
- 完成条件/检查点：PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json仍均不存在；本轮training-process、training-exit、training-plan、train.log及checkpoints亦不存在，尚未正式开训。未生成startup/final-verification。
- 判断、动作与剩余：原准备流水线持续推进，无需修复或恢复；本轮仅保存巡检JSON、追加本文，未更改代码/配置/数据或启停进程。等待dialogue筛选完成、合格数据汇入、四卡编码和数据发布，再由原入口计算中英按时长1:1的一个完整平衡epoch真实步数，以原始assembled模型、token loss启动并核验至少20次更新。未变更目标或预算，本次巡检结束。

## 2026-09-12 15:33:26 UTC 单次巡检

- 阶段与step：`prepare/running`，正式训练step 0 → 0。先读manual、pipeline、指定 `20260912T153214Z.json` 和最近记录；manual.active=false。与15:03:15的观察间隔30.18分钟。现场进程、文件、日志、配方及资源证据保存在 `runs/emilia-full-token-scratch/supervision/20260912T153326Z-inspection.json`；快照process=null仍表示正式训练尚未开始。
- 进程/退出码：入口355930/start_ticks 105903983；quality/export/codec supervisor为355993/355994/355995、start_ticks均105904004，现场cmdline与pipeline记录匹配，三个returncode均null。四codec 355997–356000/start_ticks 105904009，父进程355995，rank 0–3/world_size=4及本轮输入输出路径匹配。质量32个worker均R，近半小时同身份进程合计增加CPU约57,271秒；各codec增加CPU约1,707–1,760秒、rchar约110–129GB、wchar约1.05–1.20GB。codec采样瞬间3S/1R，但各rank最后完成分片持续前进，无进程退出或卡死证据。
- 数据/导出：raw保持2,825份；export-status为最近flush的23,309,060条，EN 17,518.65/ZH 24,762.99小时，不能作最终总量。exporter末条日志仍为 `Waiting for actual-audio dialogue quality filtering`；DIALOGUE_QUALITY_COMPLETE缺失，继续等待符合设计。prepared完成1,202 → 1,352（+150），最新修改15:33:16 UTC；dialogue完成score/stats 1,549 → 1,744（+195），另32份处理中，score约91.79MB，最新修改15:32:30 UTC。发布状态1,725/8,059，已评分EN 148,763/ZH 31,902，保留EN 103,264/ZH 21,935，约177.25/42.34小时；状态文件每25个分片更新，少于现场已完成数正常。
- 配方核对：当前评分脚本与配方SHA256一致；metadata≥3.4、OVRL≥3.4、SIG≥3.5、BAK≥4.0、clipped_fraction≤0.001及其他说话人标注重叠排除未变。export仍排除ASMR及其recording ID，按短句ID优先short→long→合格dialogue去重，无小时配额。本轮没有改变筛选、去重或配方；音质筛选不保证完整文本对齐，缺失全量ASR分数仍是设计边界。
- codec指标：各rank最近chunk为[1356, 1329, 1362, 1351]；累计复用6,870,254条、新编码4,277,448条，共11,147,702条。本间隔新增复用199,287条、新编码1,041,292条，合计约685条/秒（含复用、受分片发布粒度影响）。grouped_carriers_decoded为[19862, 18537, 21166, 20232]。各worker数值有限=True，累计decode_wait约2376–2494秒、write_wait约43–47秒、codec约5912–6017秒，峰值allocated仍13.16GiB/卡。
- 资源/异常：15:33:26四GPU各14,111MiB、瞬时利用率0/0/7/0%；15:32:14指定快照为47/67/66/52%。结合CPU/I/O增量和新分片，不据瞬时低GPU利用率判故障。主机已用约216GiB（上轮202GiB），MemAvailable约1788.8GiB，无swap；codec RSS约28.7–31.3GiB/进程。仍有充足内存，继续记录趋势。GPFS可用约562.5TiB、47%已用。pipeline、prepare及codec日志未检出Traceback/Error/Exception/OOM/NaN/Inf。
- 正式训练/检查点：PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均不存在；本轮training-process、training-exit、training-plan、train.log、checkpoints亦不存在。loss/grad/LR、训练吞吐/data_wait、500步val及2000步speaker_only/icl评估尚不适用；本轮未写startup/final-verification。
- 判断、动作与剩余：准备阶段有持续产出，本轮无需修复、恢复或参数调整，维持原流水线。仅保存巡检证据并追加本文，未更改代码/配置/数据或启停进程。等待dialogue筛选、合格数据汇入、编码及数据发布；完成后由原入口计算按音频时长中英1:1的一个平衡epoch真实步数，以原始assembled模型、token loss启动，再验证至少20次更新。本次巡检结束。

## 2026-09-12 16:03:00 UTC 单次巡检

- 阶段/step：`prepare/running`，正式训练step 0 → 0。先读取manual、pipeline、指定 `20260912T160214Z.json` 和最近巡检；manual.active=false。本次与15:33:26间隔29.57分钟，完整证据保存于 `runs/emilia-full-token-scratch/supervision/20260912T160300Z-inspection.json`。准备入口运行，快照process=null表示尚无正式训练进程。
- 进程/退出码：入口355930/start_ticks 105903983；quality/export/codec supervisor 355993/355994/355995、start_ticks均105904004，现场cmdline与pipeline匹配，returncodes=[null,null,null]。四codec 355997–356000/start_ticks 105904009，父进程355995，rank 0–3/world_size=4及本轮路径均与此前现场一致，身份检查全部通过。32个质量worker均R，近半小时同身份进程合计新增CPU约56,420秒；各codec新增CPU约1,607–1,630秒、rchar约141–149GB、wchar约1.23–1.29GB，无退出或持续无进展证据。
- 数据：prepared完成1,352 → 1,489（+137），最新修改16:02:53 UTC；dialogue完成score/stats 1,744 → 1,975（+231），另32份score处理中，score约103.10MB，最新修改16:02:59 UTC。发布状态为1,975/8,059，已评分EN 166,233/ZH 40,691，保留EN 115,277/ZH 27,713、约211.73/65.13小时。raw仍2,825份；export-status仍为最近flush的23,309,060条、EN 17,518.65/ZH 24,762.99小时，尚非最终总量。exporter日志明确等待实际音频dialogue质量筛选，DIALOGUE_QUALITY_COMPLETE未发布，因此raw不变是正常等待。
- 配方：评分脚本SHA256与保存配方一致，metadata≥3.4、OVRL≥3.4、SIG≥3.5、BAK≥4.0、clipped_fraction≤0.001、跨说话人标注重叠排除不变；export仍排除ASMR及其recording ID，按短句ID优先short→long→合格dialogue去重，无小时配额。本轮未改任何输入输出配方或筛选；音质阈值不保证完整文本对齐，缺失全量ASR分数仍为既定设计边界。
- codec指标：rank 0–3最近chunk为[1492, 1465, 1498, 1491]；累计复用6,870,254条、新编码5,407,648条，共12,277,902条。本间隔复用增量0、新编码增量1,130,200，约637条/秒（按分片发布计数与采样间隔折算）。grouped_carriers_decoded为[32956, 31065, 34817, 33846]。worker状态数值有限=True，累计decode_wait约2842–2964秒、write_wait约54–61秒、codec约7119–7254秒，峰值allocated仍13.16GiB/卡。
- 资源/日志：16:03:00四GPU各14,111MiB，瞬时利用率47/0/65/54%；主机已用约221GiB（上轮216GiB），MemAvailable约1784.3GiB，无swap，codec RSS约29.5–30.7GiB/进程。内存仍有充足余量；GPFS空闲约562.4TiB、47%已用。pipeline、prepare及四worker日志未检出Traceback/Error/Exception/OOM/NaN/Inf。CPU/I/O和新分片是维持运行依据，GPU瞬时0%不作卡死判据。
- 正式训练/检查点：PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均未发布；本轮training-process、training-exit、training-plan、train.log和checkpoints均不存在。正式loss/grad/LR、吞吐/data_wait、val及speaker_only/icl评估尚不适用，未写startup/final-verification。
- 判断与执行：准备工作持续推进，无需修复或重启；仅保存现场JSON并追加本文，未改代码/配置/数据或启停进程。剩余为dialogue筛选、合格数据汇入、编码和发布；随后由原入口计算一个中英按音频时长1:1平衡epoch真实步数，以原始assembled模型、token loss开训，再核验至少20次更新。本次巡检结束。

## 2026-09-12 16:33:13 UTC 单次巡检

- 阶段/step：`prepare/running`，正式训练step 0 → 0。先读manual、pipeline、指定 `20260912T163214Z.json` 及最近记录；manual.active=false。与16:03:00观察间隔30.22分钟，完整进程、指标、日志及资源证据保存于 `runs/emilia-full-token-scratch/supervision/20260912T163313Z-inspection.json`。
- 进程及退出码：入口355930/start_ticks 105903983；quality/export/codec supervisor为355993/355994/355995，start_ticks均105904004，现场cmdline与pipeline记录匹配，三个returncode均null。codec 355997–356000/start_ticks 105904009、父进程355995、rank 0–3/world_size=4及本轮路径身份核对均通过。32个质量worker均R，近半小时CPU合计增加约57,373秒；各codec CPU增加约1,602–1,661秒、rchar约168–189GB、wchar约1.23–1.29GB。瞬时codec为1R/3S，结合持续产出不判断卡死。
- 数据推进：prepared完成1,489 → 1,589（+100），最新修改16:33:02 UTC；dialogue完成score/stats 1,975 → 2,466（+491），另32份score处理中，score约124.33MB、最新修改16:33:04 UTC。按25分片更新的质量状态为2,450/8,059，已评分EN 201,968/ZH 45,339，保留EN 140,064/ZH 30,834，保留时长约262.49/73.61小时。raw仍2,825份，export-status为最近flush的23,309,060条、EN 17,518.65/ZH 24,762.99小时；未发布尾部及合格dialogue，不能当最终规模。exporter明确打印等待实际音频dialogue筛选，完成标记尚未生成，raw不增长符合设计。
- 配方：评分脚本SHA256与保存配方相符；metadata≥3.4、OVRL≥3.4、SIG≥3.5、BAK≥4.0、clipped_fraction≤0.001及其他标注说话人重叠排除未变。export仍排除ASMR及其recording ID，按短句ID优先short→long→合格dialogue去重，无小时配额。本轮未改代码、配方或筛选；音质门槛不保证完整文本对齐，全量ASR分数缺失仍属已知设计边界。
- codec指标：各rank最近chunk为[1588, 1577, 1590, 1591]。累计复用6,870,254条（本间隔无新增）、新编码6,234,650条，总计13,104,904条；近半小时新增编码827,002条，约456条/秒，新增已处理音频约2035.64小时。条数速度受分片音频组成及发布粒度影响，不据一次变化调整参数。grouped_carriers_decoded为[44513, 43783, 47603, 46469]。各worker状态数值有限=True；累计decode_wait约3337–3518秒、write_wait约64–73秒、codec约8334–8493秒，峰值allocated仍13.16GiB/卡。
- 资源/异常：16:33:13四GPU各14,111MiB、瞬时利用率53/0/0/0%；16:32:14快照为0/71/70/71%。CPU/I/O、日志和新分片共同显示持续处理，不以GPU瞬时空闲判故障。主机已用约234GiB（上轮221GiB）、MemAvailable约1770.5GiB、无swap；codec RSS约32.1–33.3GiB/进程，仍有充足内存。GPFS可用约562.4TiB，47%已用。pipeline、prepare及四worker日志未检出Traceback/Error/Exception/OOM/NaN/Inf。
- 训练/检查点：数据PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均未发布；本轮training-process、training-exit、training-plan、train.log和checkpoints均不存在。正式训练尚未启动，loss/grad/LR、训练吞吐/data_wait、500步val与2000步speaker_only/icl指标尚不适用，未生成startup/final-verification。
- 判断、动作及剩余：准备工作有明确进展，维持原流水线，无需修复或重启。仅保存现场JSON并追加本文，未更改代码/配置/数据或启停进程。剩余为dialogue筛选、合格数据汇入、编码和最终发布；随后由原入口计算一个按音频时长中英1:1平衡epoch的真实步数，以原始assembled模型、token loss开训，并核验至少20次更新。本次巡检结束。

## 2026-09-12 17:04:30 UTC 单次巡检

- 阶段/step：`prepare/running`，正式训练step 0 → 0。先读取manual、pipeline、指定 `20260912T170214Z.json` 和最近记录；manual.active=false。与16:33:13巡检间隔31.29分钟，完整现场证据：`runs/emilia-full-token-scratch/supervision/20260912T170430Z-inspection.json`。
- 进程：入口355930/start_ticks 105903983；quality/export/codec supervisor分别355993/355994/355995、start_ticks均105904004，现场命令与pipeline匹配，returncodes均null。codec 355997–356000/start_ticks 105904009、父进程355995、rank 0–3/world_size=4及本轮路径匹配。32个质量worker均R，近半小时同身份进程新增CPU合计约59,386秒；各codec新增CPU约1,707–1,757秒、rchar约154–174GB、wchar约1.28–1.39GB，无退出或持续无进展迹象。
- 数据进展：prepared完成1,589 → 1,724（+135），最新修改17:04:27 UTC；dialogue完成score/stats 2,466 → 3,012（+546），另32份score处理中，score约145.91MB、最新修改17:04:30 UTC。质量发布状态3,000/8,059，已评分EN 241,804/ZH 49,861，保留EN 167,610/ZH 33,807，约306.55/78.96小时。raw仍2,825份，export-status的23,309,060条、EN 17,518.65/ZH 24,762.99小时为最近flush统计；exporter末条日志明确等待实际音频dialogue筛选，完成标记尚缺，故raw不变正常，不能作最终数据规模。
- codec：各rank最近chunk为[1724, 1681, 1742, 1739]；累计复用6,870,254条（本间隔无新增）、新编码7,351,024条，总计14,221,278条。新增编码1,116,374条、对应音频约1951.34小时，折算约595条/秒（分片发布统计）。worker数值有限=True；累计decode_wait约3854–4016秒、write_wait约70–79秒、codec约9691–9835秒，峰值allocated仍13.16GiB/卡。
- 资源与异常：17:04:30四GPU各14,111MiB、瞬时利用率41/41/44/0%；主机已用约242GiB（上轮234GiB），MemAvailable约1762.5GiB，无swap，codec RSS约32.1–33.2GiB/进程，余量充足。GPFS可用约562.3TiB、47%已用。pipeline、prepare及四worker日志未检出Traceback/Error/Exception/OOM/NaN/Inf；CPU/I/O和新分片支持维持运行，不以GPU瞬时0%判断卡死。
- 配方/训练：评分脚本与保存配方SHA256相符，metadata/OVRL≥3.4、SIG≥3.5、BAK≥4.0、clipped_fraction≤0.001及跨说话人标注重叠排除不变；export仍排除ASMR、按短句ID优先short→long→合格dialogue去重、无小时配额。音质标准不保证完整文本对齐，全量ASR分数缺失仍为设计边界。PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均未发布；本轮training-process、training-exit、training-plan、train.log及checkpoints均不存在。正式loss/grad/LR、吞吐/data_wait、val/生成指标不适用，未生成startup/final-verification。
- 判断与执行结果：准备工作持续推进，无需修复或重启。仅保存本次JSON并追加本文，未更改代码/配置/数据或启停进程。剩余为dialogue筛选、合格数据汇入、编码及发布；随后由原入口计算中英按时长1:1、一个平衡epoch的真实步数，以原始assembled模型、token loss启动并核验至少20次更新。本次巡检结束。

## 2026-09-12 17:33:00 UTC 单次巡检

- 阶段/step：`prepare/running`，正式step 0 → 0。先读取manual、pipeline、指定 `20260912T173214Z.json` 和最近记录；manual.active=false。与17:04:30巡检间隔28.50分钟。完整现场证据：`runs/emilia-full-token-scratch/supervision/20260912T173300Z-inspection.json`。
- 进程/退出码：入口355930/start_ticks 105903983；quality/export/codec supervisor为355993/355994/355995、start_ticks均105904004，现场cmdline与pipeline匹配，returncodes均null。四codec 355997–356000/start_ticks 105904009、父进程355995、rank 0–3及本轮路径均匹配，现场均R；32个质量worker均R，近半小时同身份进程新增CPU合计约53,869秒。各codec新增CPU约1,513–1,576秒、rchar约120–134GB、wchar约1.15–1.21GB，无进程退出或持续无进展证据。
- 数据推进：prepared完成1,724 → 1,864（+140），最新修改17:32:32 UTC；dialogue完成score/stats 3,012 → 3,242（+230），另32份处理中，score约162.40MB，最新修改17:32:50 UTC。每25分片发布的质量状态为3,225/8,059，已评分EN 266,094/ZH 56,490，保留EN 184,756/ZH 38,414，约341.34/88.93小时。raw仍2,825份；export-status仍23,309,060条、EN 17,518.65/ZH 24,762.99小时的最近flush统计，不是最终总量。exporter末条日志明确等待实际音频dialogue筛选；DIALOGUE_QUALITY_COMPLETE未发布，维持等待正常。
- codec：rank 0–3最近chunk为[1860, 1821, 1886, 1879]。累计复用6,870,254条（本间隔无新增）、新编码8,509,185条，总计15,379,439条；新增编码1,158,161条、对应音频约1461.20小时，折算约677条/秒（分片发布统计）。worker指标有限=True，累计decode_wait约4247–4462秒、write_wait约102–115秒、codec约10862–11047秒，峰值allocated仍13.16GiB/卡。
- 资源/异常：17:33:00四卡各14,111MiB、利用率43/39/44/41%；主机已用约247GiB（上轮242GiB），MemAvailable约1757.4GiB，无swap；codec RSS约32.1–33.9GiB/进程，内存充足。GPFS空闲约562.1TiB、47%已用。pipeline、prepare及四worker日志未检出Traceback/Error/Exception/OOM/NaN/Inf。CPU/I/O、新评分和编码分片均支持维持运行。
- 配方/训练：评分脚本SHA256与保存配方一致；metadata/OVRL≥3.4、SIG≥3.5、BAK≥4.0、clipped_fraction≤0.001及跨说话人标注重叠排除未变；export仍排除ASMR及其recording ID、按短句ID优先short→long→合格dialogue去重、无小时配额。音质门槛不保证完整文本对齐，全量ASR分数缺失仍是既定设计边界。数据PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均未发布；本轮training-process、training-exit、training-plan、train.log和checkpoints均不存在。正式训练尚未开始，loss/grad/LR、训练吞吐/data_wait、val与speaker_only/icl评估尚不适用；未写startup/final-verification。
- 判断、动作及剩余：准备流水线持续推进，本轮无需修复或重启。仅保存检查JSON并追加本文，未更改代码/配置/数据或启停进程。等待dialogue筛选、合格数据汇入、编码和最终发布；随后原入口计算中英按音频时长1:1的一个平衡epoch真实步数，以原始assembled模型、token loss启动并核验至少20次更新。本次巡检结束。

## 2026-09-12 18:02:56 UTC 单次巡检

- 阶段/step：`prepare/running`，正式step 0 → 0。先读取manual、pipeline、指定 `20260912T180214Z.json` 和最近巡检记录；manual.active=false。与17:33:00观察间隔29.92分钟。完整现场证据保存于 `runs/emilia-full-token-scratch/supervision/20260912T180256Z-inspection.json`。
- 进程/退出码：入口355930/start_ticks 105903983；quality/export/codec supervisor分别355993/355994/355995、start_ticks均105904004，现场cmdline与pipeline记录匹配，三个returncode均null。codec 355997–356000/start_ticks 105904009、父进程355995、rank 0–3及本轮路径均匹配；质量worker32个（31R/1D），同身份进程新增CPU合计约56,785秒。四codec新增CPU约1,635–1,667秒、rchar约137–149GB、wchar约1.18–1.27GB，没有退出或持续无进展证据。
- 数据：prepared完成1,864 → 2,001（+137），最新修改18:02:43 UTC；dialogue完成score/stats 3,242 → 3,468（+226），另32份score处理中，score约180.83MB，最新修改18:02:41 UTC。质量发布状态3,450/8,059，已评分EN 294,241/ZH 65,580，保留EN 204,514/ZH 44,791、约381.63/102.27小时。raw仍2,825份；export-status的23,309,060条、EN 17,518.65/ZH 24,762.99小时仍是最近flush统计。exporter日志明确等待实际音频dialogue筛选，DIALOGUE_QUALITY_COMPLETE尚缺，继续等待正常；不能以该中间统计计算最终训练预算。
- codec：各rank最近chunk为[1992, 1957, 2026, 2019]。累计复用6,870,254条（本间隔无新增）、新编码9,639,239条，共16,509,493条；新增编码1,130,054条、对应音频约1640.97小时，约629条/秒（按分片发布计数）。worker指标有限=True；累计decode_wait约4702–4933秒、write_wait约109–122秒、codec约12146–12358秒，peak_allocated仍13.16GiB/卡。
- 资源/日志：18:02:56四GPU各14,111MiB、利用率39/0/29/31%；主机已用约257GiB（上轮247GiB）、MemAvailable约1748.2GiB、无swap，codec RSS约31.7–33.8GiB/进程，仍有充足余量。GPFS可用约561.9TiB、47%已用。pipeline、prepare及四worker日志未检出Traceback/Error/Exception/OOM/NaN/Inf；CPU/I/O和新分片提供维持运行依据，不以GPU瞬时0%判断卡死。
- 配方/训练：评分脚本SHA256与保存配方一致，metadata/OVRL≥3.4、SIG≥3.5、BAK≥4.0、clipped_fraction≤0.001及跨说话人标注重叠排除未变；export仍排除ASMR及其recording ID、按短句ID优先short→long→合格dialogue去重、无小时配额。本轮未改变配方或筛选；音质标准不保证完整文本对齐，全量ASR分数缺失仍为设计边界。数据PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均未发布；本轮training-process、training-exit、training-plan、train.log和checkpoints均不存在。正式loss/grad/LR、吞吐/data_wait、val及speaker_only/icl评估不适用，未写startup/final-verification。
- 判断、动作及剩余：准备工作持续推进，无需修复或重启。仅保存巡检JSON并追加本文，未更改代码/配置/数据或启停进程。等待dialogue筛选、合格数据汇入、编码和最终发布；之后原入口计算按音频时长中英1:1、一个平衡epoch的真实步数，以原始assembled模型、token loss启动并核验至少20次更新。本次巡检结束。

## 2026-09-12 18:32:58 UTC 单次巡检

- 阶段/step：`prepare/running`，正式step 0 → 0。先读取manual、pipeline、指定 `20260912T183214Z.json` 和最近记录；manual.active=false。与18:02:56观察间隔30.03分钟。完整现场证据：`runs/emilia-full-token-scratch/supervision/20260912T183258Z-inspection.json`。
- 进程/退出码：入口355930/start_ticks 105903983；quality/export/codec supervisor为355993/355994/355995、start_ticks均105904004，现场cmdline与pipeline匹配，returncodes均null。codec 355997–356000/start_ticks 105904009、父进程355995、rank 0–3及本轮路径匹配。32个质量worker（31R/1D），同身份进程新增CPU合计约57,165秒；四codec新增CPU约1,639–1,683秒、rchar约138–179GB、wchar约1.19–1.31GB，各rank已完成分片持续增加，无进程退出或持续无进展证据。
- 数据：prepared完成2,001 → 2,119（+118），最新修改18:32:57 UTC；dialogue完成score/stats 3,468 → 3,620（+152），另32份score处理中，score约194.95MB，最新修改18:31:30 UTC。质量发布状态3,600/8,059，已评分EN 313,430/ZH 71,937，保留EN 218,056/ZH 49,252、約409.83/111.80小时。raw仍2,825份，export-status的23,309,060条、EN 17,518.65/ZH 24,762.99小时是最近flush统计，不能当最终规模。exporter明确等待实际音频dialogue筛选，DIALOGUE_QUALITY_COMPLETE尚缺，raw不增长符合设计。
- codec：rank 0–3最近chunk为[2116, 2089, 2134, 2127]。累计复用6,870,254条（本间隔无新增）、新编码10,612,111条，共17,482,365条；新增编码972,872条、对应音频约1849.19小时，折算约540条/秒（按分片发布统计）。worker数值有限=True；累计decode_wait约5145–5467秒、write_wait约115–128秒、codec约13401–13677秒，峰值allocated仍13.16GiB/卡。
- 资源/日志：18:32:58四GPU各14,111MiB、瞬时利用率3/0/0/0%；18:32:14快照为60/0/70/69%。CPU/I/O和新分片支持持续处理，不以GPU瞬时空闲判断卡死。主机已用约248GiB（上轮257GiB），MemAvailable约1756.4GiB，无swap；codec RSS约32.1–33.7GiB/进程，余量充足。GPFS空闲约561.8TiB、47%已用。pipeline、prepare及四worker日志未检出Traceback/Error/Exception/OOM/NaN/Inf。
- 配方/训练：评分脚本SHA256与保存配方一致；metadata/OVRL≥3.4、SIG≥3.5、BAK≥4.0、clipped_fraction≤0.001及跨说话人标注重叠排除不变；export仍排除ASMR及其recording ID、按短句ID优先short→long→合格dialogue去重、无小时配额。音质标准不保证完整文本对齐，全量ASR分数缺失仍为设计边界。数据PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均未发布；本轮training-process、training-exit、training-plan、train.log及checkpoints均不存在。正式loss/grad/LR、吞吐/data_wait、val与speaker_only/icl评估尚不适用，未写startup/final-verification。
- 判断、动作与剩余：准备工作持续推进，无需修复或重启。仅保存检查JSON并追加本文，未更改代码/配置/数据或启停进程。等待dialogue筛选、合格数据汇入、编码和最终发布；后续由原入口计算按音频时长中英1:1的一个平衡epoch真实步数，以原始assembled模型、token loss启动，并核验至少20次更新。本次巡检结束。

## 2026-09-12 19:02:57 UTC 单次巡检

- 阶段/step：`prepare/running`，正式step 0 → 0。先读取manual、pipeline、指定 `20260912T190214Z.json` 和最近记录；manual.active=false。与18:32:58巡检间隔30.00分钟，完整现场证据保存于 `runs/emilia-full-token-scratch/supervision/20260912T190257Z-inspection.json`。
- 进程/退出码：入口355930/start_ticks 105903983；quality/export/codec supervisor分别355993/355994/355995、start_ticks均105904004，现场cmdline与pipeline匹配，returncodes均null。codec 355997–356000/start_ticks 105904009、父进程355995、rank 0–3和本轮路径匹配。32个质量worker均R，近半小时同身份进程CPU合计增加约56,457秒；各codec CPU增加约1,621–1,674秒、rchar约156–194GB、wchar约1.20–1.30GB，无退出或持续无进展证据。
- 数据：prepared完成2,119 → 2,219（+100），最新修改19:02:46 UTC；dialogue完成score/stats 3,620 → 3,882（+262），另32份处理中，score约207.70MB，最新修改19:02:55 UTC。质量发布状态3,875/8,059，已评分EN 334,798/ZH 80,850，保留EN 232,945/ZH 54,993、约449.31/133.51小时。raw仍2,825份；export-status的23,309,060条、EN 17,518.65/ZH 24,762.99小时仍为最近flush统计。exporter明确打印等待实际音频dialogue筛选，完成标记尚缺，维持等待正常；该中间状态不能当最终数据量或训练预算。
- codec：各rank最近chunk为[2200, 2165, 2266, 2235]；累计复用6,870,254条（本间隔无新增）、新编码11,438,302条，共18,308,556条。本间隔新增编码826,191条、对应音频约1984.98小时，折算约459条/秒（分片发布统计）。worker数值有限=True；累计decode_wait约5664–5925秒、write_wait约146–159秒、codec约14651–14868秒，峰值allocated仍13.16GiB/卡。
- 资源/日志：19:02:57四卡各14,111MiB、利用率43/37/38/32%；主机已用约250GiB（上轮248GiB），MemAvailable约1754.4GiB，无swap，codec RSS约32.2–33.1GiB/进程，内存余量充足。GPFS可用约564.1TiB、47%已用。pipeline、prepare及四worker日志未检出Traceback/Error/Exception/OOM/NaN/Inf。CPU/I/O、新评分及编码分片共同支持维持运行。
- 配方/训练：评分脚本SHA256与保存配方一致，metadata/OVRL≥3.4、SIG≥3.5、BAK≥4.0、clipped_fraction≤0.001及跨说话人标注重叠排除未变；export仍排除ASMR及其recording ID、按短句ID优先short→long→合格dialogue去重、无小时配额。音质筛选不保证完整文本对齐，缺失全量ASR分数仍为设计边界。数据PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均未发布；本轮training-process、training-exit、training-plan、train.log及checkpoints均不存在。正式loss/grad/LR、吞吐/data_wait、val及speaker_only/icl评估尚不适用，未生成startup/final-verification。
- 判断、动作及剩余：准备流水线持续推进，无需修复或重启；仅保存现场JSON并追加本文，未更改代码/配置/数据或启停进程。等待dialogue筛选、合格数据汇入、编码及最终发布；之后原入口计算中英按音频时长1:1、一个平衡epoch真实步数，以原始assembled模型、token loss启动并核验至少20次更新。本次巡检结束。

## 2026-09-12 19:32:59 UTC 单次巡检

- 阶段/step：`prepare/running`，正式step 0 → 0。先读取manual、pipeline、指定 `20260912T193214Z.json` 和最近记录；manual.active=false。与19:02:57观察间隔30.03分钟，现场证据：`runs/emilia-full-token-scratch/supervision/20260912T193259Z-inspection.json`。
- 进程/退出码：入口355930/start_ticks 105903983；quality/export/codec supervisor 355993/355994/355995、start_ticks均105904004，现场cmdline与pipeline匹配，三个returncode均null。codec 355997–356000/start_ticks 105904009、父进程355995、rank 0–3及本轮路径匹配，均R状态。32个质量worker均R，近半小时同身份进程CPU合计增加约57,000秒；各codec CPU增加约1,626–1,701秒、rchar约130–143GB、wchar约1.22–1.31GB，无退出或持续无进展证据。
- 数据：prepared完成2,219 → 2,372（+153），最新修改19:32:50 UTC；先枚举dialogue score/stats得到4,399份（上轮3,882，+517），32份处理中，随后读取的状态已发布4,400/8,059。文件枚举和状态读取不是原子快照，这一条差异发生在持续发布期间。score约229.91MB、最新修改19:32:59 UTC。质量状态已评分EN 375,402/ZH 86,099，保留EN 261,188/ZH 58,530、约504.04/142.41小时。raw仍2,825份，export-status为最近flush的23,309,060条、EN 17,518.65/ZH 24,762.99小时；exporter明确打印等待实际音频dialogue筛选，完成标记未发布，raw不变符合设计，不能用此中间统计计算最终步数。
- codec：各rank最近chunk为[2352, 2325, 2418, 2383]；累计复用6,870,254条（无新增）、新编码12,702,364条，共19,572,618条。近半小时新增编码1,264,062条、对应音频约1580.08小时，折算约701条/秒（按分片发布统计）。worker指标有限=True，累计decode_wait约6104–6403秒、write_wait约156–168秒、codec约15921–16179秒，peak_allocated仍13.16GiB/卡。
- 资源/日志：19:32:59四GPU各14,111MiB、利用率6/42/43/0%；主机已用约250GiB（与上轮相近），MemAvailable约1755.1GiB，无swap；codec RSS约32.1–32.9GiB/进程，内存充足。GPFS可用约564.0TiB、47%已用。pipeline、prepare及四worker日志未检出Traceback/Error/Exception/OOM/NaN/Inf。CPU/I/O和分片增量提供维持运行依据，不据GPU瞬时空闲判断卡死。
- 配方/训练：评分脚本SHA256与保存配方相符；metadata/OVRL≥3.4、SIG≥3.5、BAK≥4.0、clipped_fraction≤0.001及跨说话人标注重叠排除未变。export仍排除ASMR及其recording ID、按短句ID优先short→long→合格dialogue去重、无小时配额。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。数据PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均未发布；本轮training-process、training-exit、training-plan、train.log、checkpoints均不存在。正式loss/grad/LR、吞吐/data_wait、val及speaker_only/icl评估尚不适用，未写startup/final-verification。
- 判断、动作及剩余：准备流水线持续推进，无需修复或重启。仅保存巡检JSON并追加本文，未更改代码/配置/数据或启停进程。等待dialogue筛选、合格数据汇入、编码和最终发布；后续原入口计算按时长中英1:1、一个完整平衡epoch真实步数，以原始assembled模型、token loss启动，再核验至少20次更新。本次巡检结束。

## 2026-09-12 20:03:00 UTC 单次巡检

- 阶段/step：`prepare/running`，正式step 0 → 0。先读取manual、pipeline、指定 `20260912T200214Z.json` 和最近巡检；manual.active=false。与19:32:59观察间隔30.01分钟，完整现场证据：`runs/emilia-full-token-scratch/supervision/20260912T200300Z-inspection.json`。
- 进程/退出码：入口355930/start_ticks 105903983；quality/export/codec supervisor为355993/355994/355995、start_ticks均105904004，现场cmdline与pipeline匹配，returncodes均null。codec 355997–356000/start_ticks 105904009、父进程355995、rank 0–3及本轮路径匹配。32个质量worker均R，近半小时同身份进程CPU合计增加约56,971秒；各codec CPU增加约1,640–1,678秒、rchar约137–150GB、wchar约1.18–1.30GB，无退出或持续无进展证据。
- 数据：prepared完成2,372 → 2,510（+138），最新修改20:02:54 UTC；dialogue完成score/stats为4,927份（上轮枚举4,399，+528），另32份处理中，score约251.18MB，最新修改20:02:48 UTC。质量发布状态4,925/8,059，已评分EN 413,402/ZH 90,453，保留EN 287,257/ZH 61,377、约543.76/147.34小时。raw仍2,825份，export-status仍为最近flush的23,309,060条、EN 17,518.65/ZH 24,762.99小时；exporter明确打印等待实际音频dialogue筛选，完成标记未发布，等待正常。原有raw编码接近赶上导出进度，后续可能同样等待dialogue，不能仅因GPU空闲判断故障。
- codec：各rank最近chunk为[2488, 2473, 2558, 2511]。累计复用6,870,254条（无新增）、新编码13,843,835条，共20,714,089条；新增编码1,141,471条、对应音频约1620.18小时，折算约634条/秒（分片发布统计）。worker指标有限=True；累计decode_wait约6548–6876秒、write_wait约165–175秒、codec约17180–17453秒，峰值allocated仍13.16GiB/卡。
- 资源/日志：20:03:00四GPU各14,111MiB、利用率58/56/45/0%；主机已用约255GiB（上轮250GiB），MemAvailable约1749.4GiB，无swap，codec RSS约32.4–33.6GiB/进程，余量充足。GPFS可用约563.9TiB、47%已用。pipeline、prepare及四worker日志未检出Traceback/Error/Exception/OOM/NaN/Inf；CPU/I/O和评分、编码产出支持维持运行。
- 配方/训练：评分脚本SHA256与保存配方一致，metadata/OVRL≥3.4、SIG≥3.5、BAK≥4.0、clipped_fraction≤0.001及跨说话人标注重叠排除未变；export仍排除ASMR及其recording ID、按短句ID优先short→long→合格dialogue去重、无小时配额。音质门槛不保证完整文本对齐，缺失全量ASR分数仍为设计边界。数据PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均未发布；本轮training-process、training-exit、training-plan、train.log及checkpoints均不存在。正式loss/grad/LR、吞吐/data_wait、val与speaker_only/icl评估尚不适用，未写startup/final-verification。
- 判断、动作及剩余：准备流水线持续推进，无需修复或重启；仅保存现场JSON并追加本文，未更改代码/配置/数据或启停进程。等待dialogue筛选、合格数据汇入、编码及最终发布；随后原入口计算按音频时长中英1:1、一个平衡epoch真实步数，以原始assembled模型、token loss启动，再核验至少20次更新。本次巡检结束。

## 2026-09-12 20:33:57 UTC 单次巡检

- 阶段/step：`prepare/running`，正式step 0 → 0。先读取manual、pipeline、指定 `20260912T203214Z.json` 和最近记录；manual.active=false。与20:03:00观察间隔30.94分钟，完整现场证据：`runs/emilia-full-token-scratch/supervision/20260912T203357Z-inspection.json`。
- 进程/退出码：入口355930/start_ticks 105903983；quality/export/codec supervisor分别355993/355994/355995、start_ticks均105904004，cmdline与pipeline匹配，三个returncode均null。四codec 355997–356000/start_ticks 105904009、父进程355995、rank 0–3及本轮路径匹配。32个质量worker（31R/1D），同身份进程CPU合计增加约58,740秒；各codec CPU增加约1,244–1,272秒、rchar约105–118GB、wchar约0.89–0.98GB，无退出证据。
- 数据：prepared完成2,510 → 2,614（+104），最新修改20:33:48 UTC；dialogue完成score/stats 4,927 → 5,172（+245），另32份处理中，score约270.04MB、最新修改20:33:57 UTC。质量发布状态5,150/8,059，已评分EN 438,831/ZH 99,833，保留EN 304,996/ZH 68,049、约579.09/161.58小时。raw仍2,825份；export-status的23,309,060条、EN 17,518.65/ZH 24,762.99小时为最近flush统计。exporter日志明确等待实际音频dialogue筛选，完成标记仍缺，保持等待正常。
- codec：各rank最近chunk为[2596, 2581, 2662, 2607]；累计复用6,870,254条（无新增）、新编码14,703,268条，共21,573,522条。新增编码859,433条、对应音频约1259.59小时，约463条/秒（按分片发布统计）。worker指标有限=True；累计decode_wait约6882–7219秒、write_wait约653–685秒、codec约18073–18477秒，峰值allocated仍13.16GiB/卡。
- 写入等待专项核对：本间隔各rank累计write_wait增加约478–515秒，较上轮明显上升；定位在codec异步写入等待计时，并非训练/验证/生成故障。日志单分片写等待峰值：rank0 chunk2524为172.45秒、rank1 chunk2509为224.04秒、rank2 chunk2594为184.33秒、rank3 chunk2543为228.97秒。20:34左右再次检查四worker日志，最近各5个已完成分片的写等待均降至约0.17–0.22秒；rank3已进一步完成2611。说明先前写入延迟后仍有持续进展，尚无持续卡死证据；底层延迟原因未证实，不臆测为特定存储故障，也不据此重启或调整参数。
- 资源/异常：20:33:57四GPU各14,111MiB、利用率0/31/66/66%；主机已用约256GiB（上轮255GiB），MemAvailable约1748.6GiB，无swap；codec RSS约32.1–34.4GiB/进程，余量充足。GPFS空闲约563.9TiB、47%已用。pipeline、prepare及四worker日志未检出Traceback/Error/Exception/OOM/NaN/Inf。
- 配方/训练：评分脚本SHA256与保存配方一致，metadata/OVRL≥3.4、SIG≥3.5、BAK≥4.0、clipped_fraction≤0.001及跨说话人标注重叠排除不变；export仍排除ASMR及其recording ID、按短句ID优先short→long→合格dialogue去重、无小时配额。音质标准不保证完整文本对齐，全量ASR分数缺失仍为设计边界。数据PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均未发布；本轮training-process、training-exit、training-plan、train.log和checkpoints均不存在。正式loss/grad/LR、吞吐/data_wait、val/生成指标尚不适用，未写startup/final-verification。
- 判断、动作与剩余：准备工作持续产出，写入等待增加已定位并记录，当前无需修复或重启。仅保存巡检JSON并追加本文，未更改代码/配置/数据或启停进程。后续关注写入耗时；等待dialogue筛选、合格数据汇入、编码和最终发布，之后原入口计算按时长中英1:1、一个平衡epoch的真实步数，以原始assembled模型、token loss开训并核验至少20次更新。本次巡检结束。

## 2026-09-12 21:02:58 UTC 单次巡检

- 阶段/step：`prepare/running`，正式step 0 → 0。先读取manual、pipeline、指定 `20260912T210214Z.json` 和最近记录；manual.active=false。与20:33:57间隔29.03分钟，完整现场证据：`runs/emilia-full-token-scratch/supervision/20260912T210258Z-inspection.json`。
- 进程/退出码：入口355930/start_ticks 105903983；quality/export/codec supervisor分别355993/355994/355995、start_ticks均105904004，现场cmdline与pipeline匹配，returncodes均null。四codec 355997–356000/start_ticks 105904009、父进程355995、rank 0–3和本轮路径匹配，采样均R。32个质量worker（30R/2D），同身份进程CPU合计增加约55,112秒；各codec CPU增加约1,594–1,622秒、rchar约172–223GB、wchar约1.14–1.23GB，无退出或持续无进展证据。
- 数据：prepared完成2,614 → 2,692（+78），最新修改21:01:50 UTC；dialogue完成score/stats 5,172 → 5,403（+231），另32份处理中，score约288.75MB，最新修改21:02:44 UTC。质量发布状态5,400/8,059，已评分EN 468,261/ZH 110,719，保留EN 325,611/ZH 75,876、约620.94/177.47小时。raw仍2,825份，export-status的23,309,060条、EN 17,518.65/ZH 24,762.99小时为最近flush统计；exporter明确等待实际音频dialogue筛选，完成标记未发布，raw不变正常，不能当最终预算。
- codec：各rank最近chunk为[2684, 2677, 2710, 2687]，累计复用6,870,254条（无新增）、新编码15,346,510条，共22,216,764条。本间隔新增编码643,242条、对应音频约2080.36小时，约369条/秒（分片发布统计）。worker指标有限=True；累计decode_wait约7342–7800秒、write_wait约655–690秒、codec约19242–19663秒，峰值allocated仍13.16GiB/卡。
- 写入等待复查：本间隔rank 0–3 write_wait增量分别为[4.14, 4.6, 2.29, 4.45]秒，较上轮各约478–515秒大幅回落；最近各5份已完成分片写等待均约0.18–0.23秒。最近分片总耗时多为147–169秒，不能把变慢归因于仍在写入卡顿；CPU/I/O和音频时长持续增加，本轮没有持续停滞证据，保留日志而不调整参数。
- 资源/异常：21:02:14快照四GPU利用率全0%，21:02:58现场恢复39/58/60/54%，显存各14,111MiB；说明瞬时空闲不能作故障依据。主机已用约254GiB（上轮256GiB），MemAvailable约1750.5GiB，无swap；codec RSS约32.6–34.1GiB/进程，内存充足。GPFS可用约563.8TiB、47%已用。pipeline、prepare及四worker日志未检出Traceback/Error/Exception/OOM/NaN/Inf。
- 配方/训练：评分脚本SHA256与保存配方一致；metadata/OVRL≥3.4、SIG≥3.5、BAK≥4.0、clipped_fraction≤0.001及跨说话人标注重叠排除未变；export仍排除ASMR及其recording ID、按短句ID优先short→long→合格dialogue去重、无小时配额。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。数据PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均未发布；本轮training-process、training-exit、training-plan、train.log、checkpoints均不存在。正式loss/grad/LR、吞吐/data_wait、val与speaker_only/icl评估尚不适用，未写startup/final-verification。
- 判断、动作及剩余：准备流水线有持续产出，写等待已回落，无需修复或重启。仅保存巡检JSON并追加本文，未更改代码/配置/数据或启停进程。等待dialogue筛选、合格数据汇入、编码及最终发布；之后原入口计算按音频时长中英1:1、一个完整平衡epoch真实步数，以原始assembled模型、token loss启动并核验至少20次更新。本次巡检结束。

## 2026-09-12 21:33:18 UTC 单次巡检

- 阶段/step：`prepare/running`，正式step 0 → 0；先读manual、pipeline、指定 `20260912T213214Z.json` 和最近记录，manual.active=false。与21:02:58观察间隔30.33分钟。完整现场证据：`runs/emilia-full-token-scratch/supervision/20260912T213318Z-inspection.json`。
- 进程/退出码：入口355930/start_ticks 105903983；quality/export/codec supervisor为355993/355994/355995、start_ticks均105904004，cmdline与pipeline匹配，returncodes均null。四codec 355997–356000/start_ticks 105904009、父进程355995、rank 0–3与本轮路径匹配。32个质量worker（31R/1D）新增CPU合计约57,831秒；四codec CPU各增加约1,685–1,695秒、rchar约213–222GB、wchar约1.25–1.27GB，均有进展，无退出故障。
- 数据：prepared完成2,692 → 2,740（+48），最新修改21:32:38 UTC；dialogue score/stats完成5,403 → 5,580（+177），另32份处理中，score约297.83MB、最新修改21:32:56 UTC。质量发布状态5,575/8,059，保留EN 334,086/ZH 79,401、约639.30/187.51小时。raw仍2,825份；export-status仍为最近flush的23,309,060条，EN 17,518.65/ZH 24,762.99小时，不是最终总量。exporter日志明确等待dialogue实际音频筛选，完成标记尚缺，继续等待符合设计。
- codec：rank最近chunk为[2732, 2725, 2758, 2735]，累计复用6,870,254条（无新增）、新编码15,741,075条，共22,611,329条。本间隔新增编码394,565条、对应音频约2578.14小时，约217条/秒（按分片发布统计）。worker指标有限=True；累计decode_wait约7943–8378秒、write_wait约657–692秒、codec约20450–20858秒，peak_allocated仍13.16GiB/卡。本间隔写等待增量[2.44, 2.39, 2.47, 2.59]秒；各rank最近5份分片写等待约0.19–0.23秒、总耗时约149–162秒，写入延迟未复发，音频处理持续推进。
- 资源/日志：四GPU各14,111MiB，利用率0/44/70/0%；主机已用约264GiB，MemAvailable约1740.4GiB，无swap，codec RSS约32.7–33.9GiB/进程；GPFS空闲约563.6TiB、47%已用。pipeline、prepare及四worker日志无Traceback/Error/Exception/OOM/NaN/Inf匹配。内存充足，CPU/I/O及新分片支持维持运行，不以GPU瞬时空闲判卡死。
- 配方/训练：评分脚本与配方SHA256一致；metadata/OVRL≥3.4、SIG≥3.5、BAK≥4、clipped_fraction≤0.001及其他标注说话人重叠排除不变。export仍排除ASMR及其recording ID、按短句ID优先short→long→合格dialogue去重、无小时配额。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。四个数据完成/发布文件均缺；本轮training-process、training-exit、training-plan、train.log及checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和speaker_only/icl评估不适用，未写startup/final-verification。
- 判断/动作/剩余：准备工作持续产出，无需修复或重启；仅保存现场JSON并追加本文，未更改代码/配置/数据或启停进程。等待dialogue筛选、合格数据汇入、编码和最终发布，由原入口计算中英按时长1:1一个平衡epoch真实步数，以原始assembled模型、token loss启动并核验至少20次更新。本轮结束。

## 2026-09-12 22:03:16 UTC 单次巡检

- 阶段/step：`prepare/running`，正式step 0 → 0；先读manual、pipeline、指定 `20260912T220214Z.json` 和最近记录，manual.active=false。距上轮现场约29.96分钟；完整证据保存于 `runs/emilia-full-token-scratch/supervision/20260912T220316Z-inspection.json`。
- 进程/退出码：入口355930/start_ticks 105903983；quality/export/codec supervisor为355993/355994/355995、start_ticks均105904004，现场cmdline与pipeline相符，returncodes均null。四codec 355997–356000/start_ticks 105904009、父进程355995、rank 0–3和本轮路径匹配。32个质量评分worker均R（另有1个S状态resource_tracker辅助进程），同身份进程CPU合计增加约57,037秒；四codec各增加约1,663–1,703 CPU秒、rchar约209–215GB、wchar约1.23–1.25GB，无退出或持续停滞证据。
- 数据：prepared完成2,740 → 2,786（+46），最新修改22:03:02 UTC；dialogue score/stats完成5,580 → 5,925（+345），另32份处理中，score约316.64MB、最新修改22:03:15 UTC。质量状态发布5,925/8,059，已评分EN 510,242/ZH 125,270，保留EN 355,075/ZH 85,788、约693.19/209.11小时。raw仍2,825份，export-status为最近flush的23,309,060条、EN 17,518.65/ZH 24,762.99小时；exporter明确等待实际音频dialogue筛选，完成标记未发布，此等待正常，中间统计不能作最终预算。
- codec：rank最近chunk为[2776, 2773, 2806, 2779]；累计复用6,870,254条（无新增）、新编码16,118,809条，共22,989,063条。本间隔新增编码377,734条、对应音频约2473.87小时，约210条/秒（按分片发布统计）。worker数值有限；累计decode_wait约8485–8940秒、write_wait约660–695秒、codec约21718–22067秒，峰值allocated仍13.16GiB/卡。write_wait增量[2.42, 2.44, 2.68, 2.29]秒，最近各5分片写等待约0.19–0.23秒，未再现先前长时间写入等待。
- 资源/日志：四GPU各14,111MiB，利用率70/0/0/63%；主机已用约262GiB（上轮264GiB）、MemAvailable约1743.5GiB，无swap；codec RSS约32.6–33.4GiB/进程。GPFS可用约563.5TiB、47%已用。pipeline、prepare及四worker日志未检出Traceback/Error/Exception/OOM/NaN/Inf；CPU/I/O、评分及编码产出支持持续运行，不以瞬时GPU空闲判断卡死。
- 配方/训练：评分脚本SHA256与保存配方相符；metadata/OVRL≥3.4、SIG≥3.5、BAK≥4、clipped_fraction≤0.001及其他标注说话人重叠排除不变。export仍排除ASMR及其recording ID，按短句ID优先short→long→合格dialogue去重，无小时配额。音质门槛不保证完整文本对齐，缺少全量ASR分数仍是设计边界。PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均未发布；training-process、training-exit、training-plan、train.log和checkpoints不存在，正式loss/grad/LR、吞吐/data_wait及val/生成评估尚不适用，未写startup/final-verification。
- 判断/动作/剩余：准备工作持续推进，无需修复或重启；仅保存巡检JSON并追加本文，未更改代码/配置/数据或启停进程。等待dialogue筛选、合格数据汇入、编码及发布；之后原入口计算中英按时长1:1的一个完整平衡epoch真实步数，以原始assembled模型、token loss启动，再核验至少20次更新。本次巡检结束。

## 2026-09-12 22:33:09 UTC 单次巡检

- 阶段/step：`prepare/running`，正式step 0 → 0；先读manual、pipeline、指定 `20260912T223214Z.json` 和最近记录，manual.active=false。距上轮现场29.88分钟，证据保存于 `runs/emilia-full-token-scratch/supervision/20260912T223309Z-inspection.json`。
- 进程/退出码：入口355930/start_ticks 105903983；quality/export/codec supervisor为355993/355994/355995、start_ticks均105904004，现场cmdline与pipeline相符，三个returncode均null。四codec 355997–356000/start_ticks 105904009、父进程355995、rank 0–3与本轮路径匹配。32个质量worker均R，近半小时CPU合计增加约56,718秒；四codec CPU分别增加1609.56/1657.25/585.52/1371.63秒，rchar约204/220/69/186GB，均有处理进展后进入等待。
- 数据：prepared完成2,786 → 2,825（+39），最新修改22:32:47 UTC；与raw的000000–002824文件名集合完全相等。dialogue score/stats完成5,925 → 6,480（+555），另32份处理中，score约338.42MB、最新修改22:32:57 UTC。质量状态最近发布6,475/8,059，已评分EN 548,307/ZH 130,413，保留EN 381,132/ZH 89,185、约738.19/215.61小时。raw仍2,825份；export-status最近flush仍为23,309,060条、EN 17,518.65/ZH 24,762.99小时，不能视为最终数据总量或训练预算。
- codec等待核验：四rank最近chunk为[2824, 2821, 2822, 2823]，下一分片均未出现；EXPORT_COMPLETE缺失且export-error不存在。现场四worker均S、wchan为do_select；代码 `scripts/prepare_emilia_streaming.py:234-242` 明确在下一raw缺失且尚无导出完成标记时每2秒等待。exporter日志同样明确等待实际音频dialogue筛选。因此本次GPU全0%和各rank先后停止编码符合当前数据已处理完的状态，无持续卡死证据；后续prepared不增加也需结合dialogue产出判断。
- 编码指标：累计复用6,870,254条（无新增）、新编码16,438,806条，共23,309,060条。本间隔新增编码319,997条、对应音频约2089.96小时，按整个观察间隔约178条/秒（包含等待时间）。worker指标有限；累计decode_wait约9059–9137秒、write_wait约661–697秒、codec约22150–23262秒，peak_allocated仍13.16GiB/卡。write_wait增量[2.54, 2.4, 0.87, 2.12]秒，最后各5分片写等待约0.18–0.22秒，未复发长写等待。
- 资源/日志：四GPU各14,111MiB，利用率均0%；主机已用约263GiB（上轮262GiB），MemAvailable约1742.5GiB，无swap，codec RSS约32.5–33.8GiB/进程。GPFS可用约563.4TiB、47%已用。pipeline、prepare及四worker日志无Traceback/Error/Exception/OOM/NaN/Inf匹配。
- 配方/训练：评分脚本SHA256与配方一致；metadata/OVRL≥3.4、SIG≥3.5、BAK≥4、clipped_fraction≤0.001及其他标注说话人重叠排除不变。export仍排除ASMR及其recording ID、按短句ID优先short→long→合格dialogue去重、无小时配额。音质标准不保证完整文本对齐，缺少全量ASR分数仍为设计边界。四个数据完成/发布文件均未出现；本轮training-process、training-exit、training-plan、train.log及checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和speaker_only/icl评估尚不适用，未生成startup/final-verification。
- 判断/动作/剩余：codec已追上当前导出，dialogue评分持续推进，无需修复或重启；仅保存巡检JSON并追加本文，未更改代码/配置/数据或启停进程。等待筛选完成、合格dialogue汇入及编码发布，之后原入口计算中英按时长1:1一个平衡epoch真实步数，以原始assembled模型、token loss开训并核验至少20次更新。本次巡检结束。

## 2026-09-12 23:03:17 UTC 单次巡检

- 阶段/step：`prepare/running`，正式step 0 → 0。先读取manual、pipeline、指定 `20260912T230214Z.json` 和最近记录；manual.active=false。距上轮现场30.13分钟，完整证据保存于 `runs/emilia-full-token-scratch/supervision/20260912T230317Z-inspection.json`。
- 进程/退出码：入口355930/start_ticks 105903983；quality/export/codec supervisor为355993/355994/355995、start_ticks均105904004，现场cmdline与pipeline匹配，三个returncode均null。四codec 355997–356000/start_ticks 105904009、父进程355995、rank 0–3及本轮路径匹配。32个质量worker为31R/1D，同身份进程CPU合计增加约57,322秒，rchar增加约71.13GB、wchar增加约45.32MB，持续推进；未发现退出。
- dialogue：score/stats完成6,480 → 6,877（+397），另32份处理中，score约358.37MB，最新修改23:02:59 UTC。状态最近发布6,875/8,059，已评分EN 582,108/ZH 136,760，保留EN 404,625/ZH 93,508、约777.77/224.50小时。质量日志有新增输出；文件计数与状态批量发布略有差异，均支持持续进展。
- 导出/codec：raw和prepared仍各2,825份、文件名集合一致，prepared最新修改仍22:32:47 UTC。累计复用6,870,254条、新编码16,438,806条，共23,309,060条；本间隔新增编码0。export-status仍为最近flush的EN 17,518.65/ZH 24,762.99小时，不是最终训练预算。四rank最后chunk仍[2824, 2821, 2822, 2823]，下一raw均不存在；EXPORT_COMPLETE及export-error均缺。四worker均S、wchan=do_select，CPU仅增加约0.65–0.75秒/进程且rchar/wchar不变，全部worker状态及日志保持不变；exporter日志明确等待实际音频dialogue筛选。结合当前raw已全部处理完、上轮核对的缺失分片等待代码及评分持续产出，GPU持续空闲是正常等待，不是已证实的故障。
- 资源/异常：四GPU各14,111MiB、利用率均0%；worker既有指标有限，峰值allocated仍13.16GiB/卡。主机已用约263GiB（与上轮相同）、MemAvailable约1742.6GiB，无swap，codec RSS约32.5–33.8GiB/进程。GPFS可用约563.4TiB、47%已用。pipeline、prepare及四worker日志未检出Traceback/Error/Exception/OOM/NaN/Inf。
- 配方/训练：评分脚本SHA256与配方一致，metadata/OVRL≥3.4、SIG≥3.5、BAK≥4、clipped_fraction≤0.001及其他标注说话人重叠排除未变。export仍排除ASMR及其recording ID、按短句ID优先short→long→合格dialogue去重，无小时配额。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均未发布；本轮training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和speaker_only/icl评估尚不适用，未写startup/final-verification。
- 判断/动作/剩余：评分工作持续推进，codec正常等待新增数据，无需修复或重启。仅保存巡检JSON并追加本文，未更改代码/配置/数据或启停进程。等待筛选完成、合格dialogue汇入、编码及数据发布；之后原入口计算中英按音频时长1:1一个完整平衡epoch真实步数，以原始assembled模型、token loss启动并核验至少20次更新。本次巡检结束。

## 2026-09-12 23:33:04 UTC 单次巡检

- 阶段/step：`prepare/running`，正式step 0 → 0。先读manual、pipeline、指定 `20260912T233214Z.json` 和最近记录，manual.active=false。距上轮现场29.78分钟；完整证据：`runs/emilia-full-token-scratch/supervision/20260912T233304Z-inspection.json`。
- 进程/退出码：入口355930/start_ticks 105903983；quality/export/codec supervisor为355993/355994/355995、start_ticks均105904004，现场cmdline与pipeline匹配，returncodes均null。四codec 355997–356000/start_ticks 105904009、父进程355995、rank 0–3及本轮路径匹配。32个质量worker均R，同身份进程CPU合计增加约56,648秒，rchar增加约70.69GB、wchar约45.50MB，无退出或持续无进展证据。
- dialogue：score/stats完成6,877 → 7,110（+233），另32份处理中，score约378.08MB，最新修改23:33:01 UTC。状态最近发布7,100/8,059，已评分EN 611,076/ZH 146,138，保留EN 425,092/ZH 100,154、约818.13/238.19小时。质量日志持续新增，文件计数与批量状态发布略有差异；CPU/I/O和新score共同确认推进。
- 导出/codec：raw和prepared仍各2,825份、文件名集合相等；prepared最后修改22:32:47 UTC。累计复用6,870,254条、新编码16,438,806条，共23,309,060条，本间隔新增0。export-status仍为最近flush的EN 17,518.65/ZH 24,762.99小时，不是最终训练预算。rank最后chunk仍[2824, 2821, 2822, 2823]，下一分片均缺；EXPORT_COMPLETE及export-error不存在。四worker均S、wchan=do_select，CPU仅增加0.66–0.80秒/进程、rchar/wchar不变，worker状态与日志未变。exporter日志仍明确等待实际音频dialogue筛选；当前raw已处理完且评分继续产出，因此维持等待，不以GPU空闲或训练step 0认定故障。
- 资源/日志：四GPU各14,111MiB、利用率均0%，worker既有数值有限、peak_allocated仍13.16GiB/卡。主机已用约263GiB，与上轮相近；MemAvailable约1742.4GiB，无swap，codec RSS约32.5–33.8GiB/进程。GPFS可用约563.3TiB、47%已用。pipeline、prepare及四worker日志未检出Traceback/Error/Exception/OOM/NaN/Inf。
- 配方/训练：评分脚本SHA256与保存配方一致，质量和导出配方与上轮相同；metadata/OVRL≥3.4、SIG≥3.5、BAK≥4、clipped_fraction≤0.001及其他标注说话人重叠排除未变。ASMR及其recording ID仍排除，短句ID优先short→long→合格dialogue去重，无小时配额。音质门槛不保证完整文本对齐，缺失全量ASR分数仍为设计边界。四个数据完成/发布文件均缺；training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait及val/生成指标不适用，未写startup/final-verification。
- 判断/动作/剩余：准备阶段持续推进，codec正常等待，无需修复或重启；仅保存现场JSON并追加本文，未更改代码/配置/数据或启停进程。等待dialogue筛选、合格数据汇入、编码和数据发布；之后原入口计算中英按时长1:1一个完整平衡epoch真实步数，以原始assembled模型、token loss启动并核验至少20次更新。本次巡检结束。

## 2026-09-13 00:02:58 UTC 单次巡检

- 阶段/step：`prepare/running`，正式step 0 → 0。先读manual、pipeline、指定 `20260913T000214Z.json` 和最近记录，manual.active=false；距上轮现场29.90分钟，完整证据：`runs/emilia-full-token-scratch/supervision/20260913T000258Z-inspection.json`。
- 进程/退出码：入口355930/start_ticks 105903983；quality/export/codec supervisor为355993/355994/355995、start_ticks均105904004，现场cmdline与pipeline匹配，returncodes均null。四codec 355997–356000/start_ticks 105904009、父进程355995、rank 0–3及本轮路径匹配。32个质量worker为31R/1D，同身份进程CPU合计增加约56,899秒、rchar约71.65GB、wchar约42.27MB，无退出或持续无进展证据。
- dialogue：score/stats完成7,110 → 7,337（+227），另32份处理中，score约396.40MB，最新修改00:02:51 UTC。状态最近发布7,325/8,059，已评分EN 637,357/ZH 155,715，保留EN 443,596/ZH 106,902、约855.49/252.00小时。质量日志新增输出，文件计数与批量发布状态略有差异；CPU/I/O和新score确认持续推进。
- 导出/codec：raw和prepared仍各2,825份、文件名集合相等，prepared最后修改为09-12 22:32:47 UTC。累计复用6,870,254条、新编码16,438,806条，共23,309,060条，本间隔新增0；export-status仍为最近flush的EN 17,518.65/ZH 24,762.99小时，不是最终训练预算。rank最后chunk仍[2824, 2821, 2822, 2823]，下一raw均缺，EXPORT_COMPLETE和export-error不存在。四worker均S、wchan=do_select，CPU仅增加0.66–0.74秒/进程、rchar/wchar不变，状态与日志未变。exporter明确等待实际音频dialogue筛选；当前raw已全部编码且评分继续产出，因此继续等待符合设计，不以GPU空闲或训练step 0断言故障。
- 资源/日志：四GPU各14,111MiB、利用率均0%；既有worker指标有限，peak_allocated仍13.16GiB/卡。主机已用约257GiB（上轮263GiB），MemAvailable约1748.8GiB，无swap，codec RSS约32.5–33.8GiB/进程。GPFS可用约563.2TiB、47%已用。pipeline、prepare及四worker日志未检出Traceback/Error/Exception/OOM/NaN/Inf。
- 配方/训练：评分脚本SHA256与配方一致，质量和导出配方与上轮相同；metadata/OVRL≥3.4、SIG≥3.5、BAK≥4、clipped_fraction≤0.001及其他标注说话人重叠排除未变。ASMR及其recording ID仍排除，短句ID优先short→long→合格dialogue去重，无小时配额。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均缺；training-process、training-exit、training-plan、train.log及checkpoints均不存在。正式loss/grad/LR、吞吐/data_wait及val/生成评估尚不适用，未写startup/final-verification。
- 判断/动作/剩余：评分持续推进，codec正常等待，无需修复或重启；仅保存现场JSON并追加本文，未更改代码/配置/数据或启停进程。等待筛选完成、合格dialogue汇入、编码及发布；之后原入口计算中英按时长1:1一个完整平衡epoch真实步数，以原始assembled模型、token loss启动并核验至少20次更新。本次巡检结束。

## 2026-09-13 00:33:18 UTC 单次巡检

- 阶段/step：`prepare/running`，正式step 0 → 0。先读manual、pipeline、指定 `20260913T003214Z.json` 和最近记录，manual.active=false；距上轮现场30.33分钟，完整证据：`runs/emilia-full-token-scratch/supervision/20260913T003318Z-inspection.json`。
- 进程/退出码：入口355930/start_ticks 105903983；quality/export/codec supervisor为355993/355994/355995、start_ticks均105904004，现场cmdline与pipeline匹配，三个returncode均null。四codec 355997–356000/start_ticks 105904009、父进程355995、rank 0–3和本轮路径匹配。32个质量worker均R，同身份进程CPU合计增加约57,972秒、rchar约74.43GB、wchar约20.43MB，无退出故障。
- dialogue：score/stats完成7,337 → 7,472（+135），另32份处理中，score约404.87MB，最新修改00:31:20 UTC。质量状态最近发布7,450/8,059，已评分EN 644,655/ZH 160,955，保留EN 448,719/ZH 110,400、约869.88/264.13小时。完成分片增量较上轮227减少，但CPU、I/O、新score及质量日志均有增长，未出现持续停滞证据；不能用分片数量线性推断剩余耗时。状态按批量发布，略落后于完成文件计数。
- 导出/codec：raw和prepared仍各2,825份、文件名集合相等；prepared最后修改为09-12 22:32:47 UTC。累计复用6,870,254条、新编码16,438,806条，共23,309,060条，本间隔新增0。export-status仍为最近flush的EN 17,518.65/ZH 24,762.99小时，不是最终预算。rank最后chunk仍[2824, 2821, 2822, 2823]，下一raw均缺，EXPORT_COMPLETE和export-error不存在。四worker均S、wchan=do_select，CPU仅增加0.60–0.72秒/进程、rchar/wchar不变，状态和日志未变。exporter明确等待实际音频dialogue筛选；当前raw已全部编码且评分持续推进，四卡空闲符合等待状态。
- 资源/日志：四GPU各14,111MiB、利用率均0%；既有worker指标有限，peak_allocated仍13.16GiB/卡。主机已用约250GiB（上轮257GiB）、MemAvailable约1755.3GiB，无swap，codec RSS约32.5–33.8GiB/进程。GPFS可用约563.2TiB、47%已用。pipeline、prepare及四worker日志未检出Traceback/Error/Exception/OOM/NaN/Inf。
- 配方/训练：评分脚本SHA256与配方相符，质量和导出配方与上轮相同；metadata/OVRL≥3.4、SIG≥3.5、BAK≥4、clipped_fraction≤0.001及其他标注说话人重叠排除未变。仍排除ASMR及其recording ID，按短句ID优先short→long→合格dialogue去重，无小时配额。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。四个数据完成/发布文件均缺；training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val及speaker_only/icl评估不适用，未写startup/final-verification。
- 判断/动作/剩余：评分持续推进，codec正常等待，无需修复或重启；仅保存现场JSON并追加本文，未更改代码/配置/数据或启停进程。等待dialogue筛选、合格数据汇入、编码和发布，之后原入口计算中英按音频时长1:1一个完整平衡epoch真实步数，以原始assembled模型、token loss启动并核验至少20次更新。本次巡检结束。

## 2026-09-13 01:07:32 UTC 单次巡检

- 阶段/step：`prepare/running`，正式step 0 → 0。先读manual、pipeline、指定 `20260913T010214Z.json` 和最近记录；manual.active=false。距上轮现场34.25分钟，完整证据：`runs/emilia-full-token-scratch/supervision/20260913T010732Z-inspection.json`。
- 进程/退出码：入口355930/start_ticks 105903983；quality/export/codec supervisor为355993/355994/355995、start_ticks均105904004，现场cmdline与pipeline匹配，三个returncode均null。四codec 355997–356000/start_ticks 105904009、父进程355995、rank 0–3及本轮路径匹配。32个质量worker均R，同身份进程CPU合计增加约65,476秒、rchar约79.86GB、wchar约26.95MB，持续推进，无退出证据。
- dialogue：score/stats完成7,472 → 7,622（+150），另32份处理中，score约415.81MB，最新修改01:07:13 UTC。状态最近发布7,600/8,059，已评分EN 657,393/ZH 170,656，保留EN 457,806/ZH 116,806、约910.71/291.18小时。质量日志新增；完成文件和批量发布状态略有差异，CPU/I/O及新score共同支持继续运行，未据分片数线性估计剩余耗时。
- 导出/codec：raw和prepared仍各2,825份、文件名集合相等；prepared最后修改09-12 22:32:47 UTC。累计复用6,870,254条、新编码16,438,806条，共23,309,060条，本间隔新增0。export-status仍为最近flush的EN 17,518.65/ZH 24,762.99小时，不是最终训练预算。rank最后chunk仍[2824, 2821, 2822, 2823]，下一raw均缺；EXPORT_COMPLETE和export-error不存在。四worker均S、wchan=do_select，CPU仅增加0.74–0.86秒/进程、rchar/wchar不变，worker状态和日志未变。exporter明确等待实际音频dialogue筛选，当前raw已全部编码且评分持续推进，四卡空闲符合等待状态。
- 资源/日志：四GPU各14,111MiB、利用率均0%；既有worker指标有限、peak_allocated仍13.16GiB/卡。主机已用约241GiB（上轮250GiB），MemAvailable约1765.2GiB，无swap，codec RSS约32.5–33.8GiB/进程。GPFS可用约563.1TiB、47%已用。pipeline、prepare及四worker日志未检出Traceback/Error/Exception/OOM/NaN/Inf。
- 配方/训练：评分脚本SHA256与配方一致，质量和导出配方与上轮相同；metadata/OVRL≥3.4、SIG≥3.5、BAK≥4、clipped_fraction≤0.001及其他标注说话人重叠排除未变。ASMR及其recording ID仍排除，按短句ID优先short→long→合格dialogue去重，无小时配额。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。四个数据完成/发布文件均缺；training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait及val/生成评估不适用，未写startup/final-verification。
- 判断/动作/剩余：评分持续推进，codec正常等待，无需修复或重启。仅保存现场JSON并追加本文，未更改代码/配置/数据或启停进程。等待筛选完成、合格dialogue汇入、编码和发布；之后原入口计算中英按时长1:1一个完整平衡epoch真实步数，以原始assembled模型、token loss启动并核验至少20次更新。本次巡检结束。

## 2026-09-13 01:33:27 UTC 单次巡检

- 阶段/step：`prepare/running`，正式step 0 → 0。先读manual、pipeline、指定 `20260913T013214Z.json` 和最近记录，manual.active=false；距上轮现场25.92分钟，完整证据：`runs/emilia-full-token-scratch/supervision/20260913T013327Z-inspection.json`。
- 进程/退出码：入口355930/start_ticks 105903983；quality/export/codec supervisor为355993/355994/355995、start_ticks均105904004，现场cmdline与pipeline匹配，三个returncode均null。四codec 355997–356000/start_ticks 105904009、父进程355995、rank 0–3和本轮路径匹配。32个质量worker均R，同身份进程CPU合计增加约49,559秒、rchar约61.29GB、wchar约18.92MB，持续推进，无退出证据。
- dialogue：score/stats完成7,622 → 7,747（+125），另32份处理中，score约423.55MB，最新修改01:32:59 UTC。状态最近发布7,725/8,059，已评分EN 665,684/ZH 175,933，保留EN 463,626/ZH 120,157、约937.39/306.26小时。质量日志新增，批量状态略落后于完成文件计数；CPU/I/O和新score提供继续运行依据，未据分片数线性估计剩余耗时。
- 导出/codec：raw和prepared仍各2,825份、文件名集合相等；prepared最后修改09-12 22:32:47 UTC。累计复用6,870,254条、新编码16,438,806条，共23,309,060条，本间隔新增0；export-status仍为最近flush的EN 17,518.65/ZH 24,762.99小时，不是最终训练预算。rank最后chunk仍[2824, 2821, 2822, 2823]，下一raw均缺，EXPORT_COMPLETE和export-error不存在。四worker均S、wchan=do_select，CPU仅增加0.66–0.74秒/进程、rchar/wchar不变，状态和日志未变。exporter明确等待实际音频dialogue筛选；当前raw已全部编码且评分继续推进，四卡空闲符合等待状态。
- 资源/日志：四GPU各14,111MiB、利用率均0%；既有worker指标有限、peak_allocated仍13.16GiB/卡。主机已用约255GiB（上轮241GiB），MemAvailable约1749.8GiB，无swap，codec RSS约32.5–33.8GiB/进程，内存余量充足。GPFS可用约563.1TiB、47%已用。pipeline、prepare及四worker日志未检出Traceback/Error/Exception/OOM/NaN/Inf。
- 配方/训练：评分脚本SHA256与配方一致，质量和导出配方与上轮相同；metadata/OVRL≥3.4、SIG≥3.5、BAK≥4、clipped_fraction≤0.001及其他标注说话人重叠排除未变。仍排除ASMR及其recording ID、按短句ID优先short→long→合格dialogue去重，无小时配额。音质门槛不保证完整文本对齐，缺失全量ASR分数仍为设计边界。四个数据完成/发布文件均缺；training-process、training-exit、training-plan、train.log和checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val及speaker_only/icl评估不适用，未写startup/final-verification。
- 判断/动作/剩余：评分持续推进，codec正常等待，无需修复或重启。仅保存现场JSON并追加本文，未更改代码/配置/数据或启停进程。等待dialogue筛选、合格数据汇入、编码及发布；之后原入口计算中英按时长1:1一个完整平衡epoch真实步数，以原始assembled模型、token loss启动并核验至少20次更新。本次巡检结束。

## 2026-09-13 02:02:58 UTC 单次巡检

- 阶段/step：`prepare/running`，正式step 0 → 0。先读manual、pipeline、指定 `20260913T020214Z.json` 和最近记录，manual.active=false；距上轮现场29.51分钟，完整证据：`runs/emilia-full-token-scratch/supervision/20260913T020258Z-inspection.json`。
- 进程/退出码：入口355930/start_ticks 105903983；quality/export/codec supervisor为355993/355994/355995、start_ticks均105904004，现场cmdline与pipeline匹配，三个returncode均null。四codec 355997–356000/start_ticks 105904009、父进程355995、rank 0–3和本轮路径匹配。32个质量worker均R，同身份进程CPU合计增加约56,427秒、rchar约62.54GB、wchar约21.87MB，无退出或持续停滞证据。
- dialogue：score/stats完成7,747 → 7,787（+40），另32份处理中，score约432.51MB，最新修改02:02:53 UTC。状态最近发布7,775/8,059，已评分EN 677,922/ZH 185,097，保留EN 472,206/ZH 126,198、约978.17/331.89小时。虽完成分片增量减少，状态中的已评分片段较上轮增加21,402条、score约增加8.96MB，CPU/I/O及质量日志持续增长，不构成卡死证据，也不能用分片数线性推断剩余耗时。批量发布状态略落后于完成文件计数。
- 导出/codec：raw和prepared仍各2,825份、文件名集合相等；prepared最后修改09-12 22:32:47 UTC。累计复用6,870,254条、新编码16,438,806条，共23,309,060条，本间隔新增0。export-status仍为最近flush的EN 17,518.65/ZH 24,762.99小时，不是最终训练预算。rank最后chunk仍[2824, 2821, 2822, 2823]，下一raw均缺；EXPORT_COMPLETE和export-error不存在。四worker均S、wchan=do_select，CPU仅增加0.61–0.73秒/进程、rchar/wchar不变，状态与日志未变。exporter明确等待实际音频dialogue筛选；当前raw已全部编码且评分继续推进，四卡空闲符合等待状态。
- 资源/日志：四GPU各14,111MiB、利用率均0%；既有worker指标有限、peak_allocated仍13.16GiB/卡。主机已用约254GiB（上轮255GiB），MemAvailable约1750.4GiB，无swap，codec RSS约32.5–33.8GiB/进程。GPFS可用约563.0TiB、47%已用。pipeline、prepare及四worker日志未检出Traceback/Error/Exception/OOM/NaN/Inf。
- 配方/训练：评分脚本SHA256与配方一致，质量和导出配方与上轮相同；metadata/OVRL≥3.4、SIG≥3.5、BAK≥4、clipped_fraction≤0.001及其他标注说话人重叠排除未变。ASMR及其recording ID仍排除，按短句ID优先short→long→合格dialogue去重，无小时配额。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。四个数据完成/发布文件均缺；training-process、training-exit、training-plan、train.log和checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val及speaker_only/icl评估不适用，未写startup/final-verification。
- 判断/动作/剩余：评分持续推进，codec正常等待，无需修复或重启；仅保存现场JSON并追加本文，未更改代码/配置/数据或启停进程。等待筛选完成、合格dialogue汇入、编码及发布；之后原入口计算中英按音频时长1:1一个完整平衡epoch真实步数，以原始assembled模型、token loss启动并核验至少20次更新。本次巡检结束。


## 2026-09-13T08:33:01.924285+00:00 环境中断后的恢复

- 用户报告机器中断后重启，要求检查进度并用后台矩阵乘法保持各卡有占用。旧准备、监督进程均已退出；最后流水线状态写入约 02:34:56 UTC。现有两张 A800-SXM4-80GB，原先是四张 A100。中断根因未知，没有证据确认由 GPU 空闲导致。
- 数据保留：raw/prepared 均为 2,825 个分片，已编码 23,309,060 条、约 42,281.64 小时。quality 完成 8,014/8,059 个来源分片，剩余 45 个；完成 stats 合计保留时长 1393.07 小时，尚未与 short/long 做最终去重。四个发布完成文件尚未齐备，正式训练没有启动。
- 恢复 quality PID 1935（32 CPU workers）、export PID 55462、codec supervisor PID 55543（两卡、rank0/1），各自独立 tmux。完整 start_ticks/cmdline、日志及现场记录见 recovery-20260913.json。不是旧入口父进程恢复；pipeline-status 已记录独立进程模式并保留旧状态于 pipeline-history/20260913-before-recovery.json。
- codec world_size 改为 2 仅重新分配待处理分片索引，所有已有 prepared 直接跳过；筛选、编码配方未修改。质量脚本 SHA 与 export recipe 一致。旧 rank2/3 统计是历史值，不能同恢复后的 rank0/1 新增统计相加；最终以全部 prepared 分片和发布报告为准。export 重放时逐分片比较既有内容。
- 新增 scripts/gpu_idle_matmul.py，每卡 4096×4096 FP16 矩阵，空闲时计算 1 秒再等待，每卡约 557 MiB（含 CUDA context）。两张卡的 worker 与实时状态已核对；py_compile 通过。程序在 capacity/training 阶段暂停计算，监督在容量测试或训练前还须停止程序并释放 context。该程序不保证平台不会关机。
- 每 1800 秒的原监督会话已恢复，PID 64753，session 01a095b6-1b5c-7d42-8676-4a0347a52160；manual.active=false。已更新监督指令以检查独立准备进程及两卡现场。正式训练卡数的澄清待答复；两卡时可完成准备，不得启动旧四卡入口或擅自改成两卡训练。
- ETA 暂估还需 4–6 小时（约 12:30–14:30 UTC），包括剩余质量筛选、约 1,400 小时累计合格 dialogue 的两卡编码及完整清单发布。45 个未完成分片中元数据过线最多 18,226 条/88.19 小时；恢复后的新吞吐尚未测得，因此该区间为粗估，不能由分片完成率线性换算。

## 补记：2026-09-13 02:34:39 UTC 已采集观察

- 此为恢复前已保存现场 `supervision/20260913T023439Z-inspection.json` 的补记，不能代表恢复后的进程。旧四卡prepare/running，step 0 → 0，manual=false；score/stats 7,787 → 8,014，32份处理中，raw/prepared各2,825。原阶段进程身份匹配，评分CPU/I/O增加，日志无异常，未执行修复或重启。随后收到用户提供的环境中断及两卡独立恢复说明，以以下新现场为准；旧中断原因未知。

## 2026-09-13 08:33:32–08:33:57 UTC 恢复后单次巡检

- 阶段/step：`prepare/running`，独立tmux恢复模式，正式step 0 → 0。先读manual、pipeline、recovery、指定 `20260913T083137Z.json` 和最近记录。首次manual.active=true，按只核对和记录执行；复读显示主会话于08:33:01解除观察锁，active=false。完整现场：`runs/emilia-full-token-scratch/supervision/20260913T083332Z-inspection.json`。没有run_emilia_full父进程符合恢复方案，不作为故障。
- 进程/退出状态：quality 1935/start_ticks 155944140，export 55462/155976081，codec supervisor 55543/155976101，矩阵主进程4832/155946508，现场cmdline均与recovery记录匹配；对应tmux pane均存活，无已发布退出码，不能据此填写exit_code=0。codec仅两个worker：55576/55577、start_ticks均155976117、父PID55543，cmdline分别rank0/1、world_size=2，本轮路径匹配。GPU进程表确认分别位于GPU0/1。短间隔复核同PID/start_ticks/cmdline保持一致。
- 质量评分：已完成score/stats仍8,014/8,059，32份处理中，32个评分worker均R；恢复记录以来尚无新完成文件。25秒观察中评分进程CPU合计增加671.21秒、rchar约1.081GB；虽wchar暂未增长，实际CPU/I/O支持正在处理，未证明停滞。发布status仍8,000，保留EN 482,971/ZH 133,732，约1028.49/364.24小时；该重载状态不能当新评分数量。质量日志无异常。
- export：已有raw和prepared仍各2,825份（本轮新增编码分片0）。恢复导出重新遍历并逐字节校验已有前缀；代码 `scripts/export_emilia_full.py:52-66` 对存在的raw比对payload，不覆盖成功分片。现场export-status从320（初读）→416→487，25秒内CPU增加11.34秒、rchar约848MB；stdout随后报告short遍历完成。状态计数回到较小值是恢复遍历进度，不是已有数据丢失，也不能作为最终总量。
- codec：25秒中rank0/1 CPU分别增加21.66/21.44秒、rchar约1.421/1.396GB，RSS约10.91/10.67GiB。两进程打开的输入均为旧缓存池train.jsonl；与 `prepare_emilia_streaming.py:157-170` 缓存索引加载流程一致，目前未打印新的cache_records或编码状态，不能据GPU低显存判断失败。现有worker-0/1状态mtime也早于恢复，不能当本次产出；rank2/3仅历史，不计入本次新增。四份保存codec recipe一致，最终训练数据量须由全部prepared分片汇总，禁止混加跨world_size的worker计数。
- 矩阵程序：tmux emilia-gpu-idle存活；GPU0/1 worker为6667/6668、start_ticks 155947569/155947570、父PID4832，现场spawn命令、GPU进程映射及短间隔身份一致。gpu-0/1.json新鲜度约0.54秒，matrix_work=true；25秒CPU各增加约10.1秒。两卡均A800-SXM4-80GB，采样显存各977MiB、利用率82/63%；每卡矩阵worker约548MiB、codec约414MiB。此利用率包含用户授权矩阵计算，不能当codec吞吐，也不能证明平台不会关闭或解释先前中断。容量测试/正式训练前必须停止矩阵程序并确认其显存释放；仅自动跳过计算不会释放已分配矩阵。
- 资源/日志/配方：主机已用约67GiB、MemAvailable约1938.0GiB，无swap；数据盘可用约562.2TiB。三个recovery日志、矩阵stdout及当前rank0/1日志无Traceback/Error/Exception/OOM/NaN/Inf匹配。评分源码SHA256与配方一致，固定metadata/OVRL≥3.4、SIG≥3.5、BAK≥4、clipped_fraction≤0.001及跨标注说话人重叠排除不变；ASMR排除、short→long→合格dialogue短句ID去重和无配额保持。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 训练/判断/动作：数据四个完成/发布文件均缺，export-error缺；training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、验证/生成尚不适用，未写startup/final-verification。独立准备任务在短观察中有进展，无需修复或重启；仅保存证据和追加文档，未启停进程或更改代码/配置/数据。当前只有两卡且training_topology仍pending，允许完成数据准备和发布；完成后应记录awaiting_training_topology，不运行四卡总入口，也不擅自两卡训练。等待用户卡数答复或四卡环境恢复，并以最新现场记录为准。本次巡检结束。

## 2026-09-13 09:02:19–09:02:33 UTC 两卡恢复后单次巡检

- 阶段/step：独立tmux `prepare/running`，正式step 0 → 0。先读manual、pipeline、指定 `20260913T090137Z.json` 和最近记录，再核对recovery；manual.active=false，training_topology仍pending。距上轮主采样28.78分钟；完整证据：`runs/emilia-full-token-scratch/supervision/20260913T090219Z-inspection.json`。不存在总入口父进程符合恢复方案。
- 进程/退出状态：quality 1935/start_ticks 155944140、export 55462/155976081、codec supervisor 55543/155976101、矩阵主进程4832/155946508，均与恢复记录cmdline匹配，对应tmux pane存活，无已发布退出码。codec worker 55576/55577、start_ticks均155976117、父PID55543、rank0/1与world_size=2匹配；矩阵worker 6667/6668、start_ticks 155947569/155947570、父PID4832，均保持身份与GPU映射。
- dialogue：完成score/stats 8,014 → 8,057/8,059（+43），只剩2份incomplete，score约450.92MB，最新完成时间08:59:53 UTC。32个评分worker为2R/30S；近间隔同身份CPU合计增加约31,818秒、rchar约39.22GB、wchar约18.71MB。进一步14.68秒核对剩余两个R进程1960/1968，start_ticks未变，各增加14.68 CPU秒且rchar增加约1.15/3.01MB，确认仍在处理，不能把其余worker等待当故障。质量状态批量发布8,050，保留EN 486,915/ZH 137,659、约1046.32/381.38小时；完成标记仍未发布。
- export/codec：export从上轮416个已校验前缀推进至2,825；CPU增加323.49秒、rchar约28.45GB，日志已报告short/long遍历完成并等待dialogue实际音频筛选。raw/prepared仍各2,825、文件名集合相等，本次新增编码分片0；export-status当前23,309,060条、EN 17,518.65/ZH 24,762.99小时仍非最终数据总量。两个codec日志已各打印cache_records=6,870,254，缓存加载完成；进程均S、wchan=do_select，下一rank分片2826/2825均缺，等待新增raw符合代码语义。RSS各约13.4GiB。worker-0/1状态仍是恢复前记录，rank2/3只作历史，不参与本次新增计数，最终按全部prepared分片汇总。
- 矩阵/GPU：两张A800-SXM4-80GB，显存各1,675MiB、采样利用率88/0%；每卡矩阵worker约548MiB、codec约1,112MiB。矩阵状态复核新鲜度1.62/0.48秒、matrix_work=true，身份保持，近间隔CPU约增加813/811秒。利用率包含矩阵计算，不能当codec吞吐。容量预检/正式训练前须停止矩阵并确认显存释放；此程序不保证平台不关闭，也没有证据认定先前中断由GPU利用率导致。
- 资源/日志/配方：主机已用约74GiB（上轮67GiB）、MemAvailable约1930.9GiB，无swap；数据盘空闲约562.1TiB。三个recovery日志、当前rank0/1日志及矩阵日志无Traceback/Error/Exception/OOM/NaN/Inf匹配。质量、导出、codec配方与上轮相同，评分源码SHA256匹配；metadata/OVRL≥3.4、SIG≥3.5、BAK≥4、clipped_fraction≤0.001与跨说话人标注重叠排除不变，仍排除ASMR，按short→long→合格dialogue短句ID去重，无配额。音质不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 判断/动作/剩余：准备推进正常，剩余评分任务有实际CPU/I/O，暂无修复或重启需要。数据四个完成/发布文件均缺、export-error缺；training-process、training-exit、training-plan、train.log和checkpoints均不存在，正式loss/grad/LR、val/生成不适用，未写startup/final-verification。仅保存证据并追加文档，未改变代码/配置/数据或启停进程。等待最后评分、dialogue汇入和编码发布；准备完成后记录awaiting_training_topology。仅两卡且用户尚未确认训练卡数，不启动四卡入口或擅自两卡训练。本次巡检结束。

## 2026-09-13 09:33:09–09:33:35 UTC 两卡恢复后单次巡检

- 阶段/step：独立准备任务进入合格dialogue编码，正式step 0 → 0；先读manual、pipeline、指定 `20260913T093137Z.json` 和最近记录，manual.active=false、training_topology仍pending。距上轮主采样30.85分钟；完整证据：`runs/emilia-full-token-scratch/supervision/20260913T093309Z-inspection.json`。
- 质量/导出完成：DIALOGUE_QUALITY_COMPLETE于09:23:42 UTC发布，score/stats为8,059/8,059、无incomplete，保留EN 490,281/ZH 138,361、约1063.51/384.24小时。EXPORT_COMPLETE于09:25:35发布，共2,901分片、23,931,617条，EN 18,607.63/ZH 25,137.70小时；去重后dialogue贡献EN 481,087/ZH 133,438，共614,525条，跳过14,117个重复短句ID。完成报告包含原配方，ASMR仍排除。quality PID1935、export PID55462及对应tmux已退出；日志有最终报告、无异常，但没有保留exit记录，不能声称已核实exit_code=0。pipeline-status仍列旧两个PID/running，是未随独立任务退出刷新的状态；结合完成文件判断两阶段已完成，无需重启。
- 在运行身份：codec supervisor55543/start_ticks155976101、worker55576/55577/start_ticks均155976117、父PID55543，rank0/1、world_size=2及本轮路径与恢复记录匹配；矩阵主PID4832/start_ticks155946508、worker6667/6668/start_ticks155947569/155947570、父PID4832，身份和GPU映射未变，tmux存活。无总入口父进程符合独立恢复模式。
- 编码进展：raw 2,825 → 2,901；prepared 2,825 → 2,829（+4），尚余72份未完成。当前rank0/1最后chunk为2828/2827，两份状态mtime均属本次恢复，累计新编码32,784条、音频约90.47小时，指标有限；不与旧rank2/3或旧rank0/1计数相加。最终按全部分片汇总。近间隔两codec CPU分别增加132.24/225.32秒、rchar约11.65/25.94GB；追加25.49秒观察中codec及其直接解码子进程CPU合计增加398.47秒、rchar约6.06GB、wchar约0.80GB，支持正在处理。prepared在这段短观察内不增不构成停滞。
- GPU/矩阵：两张A800，采样显存59,673/72,607MiB，可用21,482/8,548MiB；其中codec分别59,110/72,044MiB，矩阵各548MiB。显存较上轮显著升高，定位到codec主进程；最近已完成分片的peak_allocated仅12.88/13.16GiB，该历史分片指标不足以解释当前驱动显存，未证明是分配缓存或其他特定原因。当前无OOM/退出，CPU/I/O持续，记录高占用待后续关注，不擅自调参数。GPU利用率98/0%包含矩阵计算，不能当codec吞吐。矩阵状态新鲜度1.35/0.74秒、matrix_work=true，按采样到的空闲状态短时计算；没有据此声称始终无竞争。容量预检/正式训练前须停止矩阵并确认内存释放。
- 主机/日志/配方：主机已用约104GiB（上轮74GiB）、MemAvailable约1900.6GiB，无swap，codec RSS约19.87/32.16GiB，数据盘可用约562.2TiB。三个recovery日志、当前rank0/1日志及矩阵日志未检出Traceback/Error/Exception/OOM/NaN/Inf。质量、导出、codec配方与上轮一致，评分源码SHA256匹配，固定DNSMOS/削波/重叠标准及short→long→合格dialogue去重不变，无配额。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界；之前环境中断原因未知。
- 判断/动作/剩余：评分和导出已完成，两卡codec持续编码，无需修复或重启；仅保存证据并追加本文，未改代码/配置/数据或启停进程。PREPARATION_COMPLETE和preparation.json尚缺；本轮training-process、training-exit、training-plan、train.log及checkpoints均不存在，正式loss/grad/LR、val/生成不适用，未写startup/final-verification。等待剩余编码与train/val发布，再记录awaiting_training_topology；仅两卡且卡数答复未到，不启动四卡入口或擅自两卡训练。本次巡检结束。

## 2026-09-13 10:03:12 UTC 两卡恢复后单次巡检

- 阶段/step：独立tmux `prepare/encoding`，正式step 0 → 0；先读manual、pipeline、指定 `20260913T100137Z.json` 和最近记录。manual.active=false、training_topology仍pending，距上轮现场30.05分钟；完整证据：`runs/emilia-full-token-scratch/supervision/20260913T100312Z-inspection.json`。
- 进程/退出状态：codec supervisor55543/start_ticks155976101、worker55576/55577/start_ticks均155976117、父PID55543，rank0/1、world_size=2、本轮路径及cmdline保持匹配；矩阵主PID4832/start_ticks155946508、worker6667/6668/start_ticks155947569/155947570、父PID4832，身份和GPU映射相符，tmux存活。quality1935/export55462仍已退出，完成文件和最终日志保持，无保留exit code；pipeline仍列旧PID/running，不据此重启已完成阶段。无总入口父进程符合恢复方案。
- 数据/编码：评分8,059/8,059和EXPORT_COMPLETE保持完成，无score incomplete；raw共2,901，最终导出23,931,617条、EN 18,607.63/ZH 25,137.70小时，去重后dialogue614,525条。prepared 2,829 → 2,847（+18），尚余54份，最新修改10:02:30 UTC。当前rank0/1最后chunk2846/2845，恢复后累计新编码180,329条，本间隔新增147,545条、音频约298.74小时（累计389.21小时），指标有限；旧rank2/3仅作历史，未混加计数，最终按全部分片汇总。
- 进展依据：两codec CPU分别增加581.73/570.03秒、rchar约50.98/49.11GB、wchar约420/408MB；近间隔新分片、CPU/I/O及日志均有进展。当前累计decode_wait约1843/1753秒，codec约463/540秒，write_wait约7.35/3.71秒，peak_allocated约13.16GiB；解码等待期间GPU空闲不构成故障。
- GPU/矩阵：两张A800显存48,343/72,607MiB，可用32,812/8,548MiB；codec分别47,780/72,044MiB，矩阵各548MiB。GPU0较上轮回落，GPU1高占用保持；当前日志无OOM/退出且有持续产出，不擅自调参或归因于特定分配器行为。采样利用率98/98%包含矩阵工作，不能作codec吞吐。矩阵状态复核新鲜度1.44/1.44秒，状态持续更新；容量测试/正式训练前须停止矩阵并确认显存释放。不能据此保证平台不会关闭，先前中断原因仍未知。
- 资源/配方/日志：主机已用约120GiB（上轮104GiB），MemAvailable约1884.5GiB，无swap；codec RSS约27.20/34.13GiB，数据盘可用约562.2TiB。三个recovery日志、当前rank0/1日志及矩阵日志无Traceback/Error/Exception/OOM/NaN/Inf匹配。质量/导出/codec配方未变、评分源码SHA256匹配；固定DNSMOS/削波/重叠门槛、ASMR排除及short→long→合格dialogue短句ID去重保持，无配额。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 判断/动作/剩余：编码正常推进，保持现有任务，无需修复或重启；仅保存证据并追加本文，未更改代码/配置/数据或启停进程。PREPARATION_COMPLETE和preparation.json仍缺，export-error缺；training-process、training-exit、training-plan、train.log及checkpoints均不存在，正式loss/grad/LR和验证/生成尚不适用，未写startup/final-verification。等待剩余编码及数据发布，完成后记录awaiting_training_topology；只有两卡且训练卡数未确认，不启动四卡入口或擅自两卡训练。本次巡检结束。

## 2026-09-13 10:32:23 UTC 两卡恢复后单次巡检

- 阶段/step：独立tmux `prepare/encoding`，正式step 0 → 0；先读manual、pipeline、指定 `20260913T103137Z.json` 和最近记录。manual.active=false、training_topology仍pending，距上轮现场29.18分钟；完整证据：`runs/emilia-full-token-scratch/supervision/20260913T103223Z-inspection.json`。
- 进程/退出状态：codec supervisor55543/start_ticks155976101、worker55576/55577/start_ticks均155976117、父PID55543，rank0/1、world_size=2及本轮cmdline匹配；矩阵主PID4832/start_ticks155946508、worker6667/6668/start_ticks155947569/155947570、父PID4832，身份与GPU映射未变，tmux存活。quality1935/export55462保持退出，完成报告及最终日志存在，无保留exit code；pipeline中的旧PID/running不代表阶段失败，无需重启。无总入口父进程符合独立恢复模式。
- 数据/编码：评分8,059/8,059和导出完成标记保持；raw共2,901，最终导出23,931,617条、EN 18,607.63/ZH 25,137.70小时，去重后dialogue614,525条。prepared 2,847 → 2,864（+17），尚余37份，最新修改10:31:43 UTC。当前rank0/1最后chunk2862/2863，恢复后累计新编码319,694条，本间隔新增139,365条、音频约289.09小时（累计678.30小时），指标有限。旧rank2/3不参与新增计数，最终按全部prepared分片汇总。
- 进展依据：两codec CPU分别增加530.46/585.02秒、rchar约48.37/53.14GB、wchar约382/429MB，分片和日志持续新增。累计decode_wait约3142/3128秒、codec约793/927秒、write_wait约8.65/5.90秒，peak_allocated约13.16GiB；采样S状态与解码等待不构成持续停滞。
- GPU/矩阵：两张A800显存48,343/72,607MiB，与上轮相同，可用32,812/8,548MiB；codec47,780/72,044MiB，矩阵各548MiB。GPU1高占用未继续增长，当前无OOM/退出且持续编码，不擅自调参或认定具体分配器原因。采样利用率0/98%包含矩阵计算，不能当codec吞吐。矩阵状态复核新鲜度0.31/0.21秒，程序身份正确且状态更新；容量测试/正式训练前须停止并确认显存释放。不能据此保证平台不会关闭，之前中断原因未知。
- 资源/日志/配方：主机已用约125GiB（上轮120GiB），MemAvailable约1880.0GiB，无swap；codec RSS约28.41/34.94GiB，数据盘可用约562.1TiB。三个recovery日志、当前rank0/1日志及矩阵日志未检出Traceback/Error/Exception/OOM/NaN/Inf。质量/导出/codec配方未变、评分源码SHA256匹配；固定DNSMOS/削波/重叠标准、ASMR排除、short→long→合格dialogue短句ID去重与无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 判断/动作/剩余：编码正常推进，无需修复或重启；仅保存证据并追加本文，未更改代码/配置/数据或启停进程。PREPARATION_COMPLETE和preparation.json仍缺，export-error缺；training-process、training-exit、training-plan、train.log及checkpoints均不存在，正式loss/grad/LR与验证/生成尚不适用，未写startup/final-verification。等待剩余编码与发布，完成后记录awaiting_training_topology；仅两卡且训练卡数未确认，不启动四卡入口或擅自两卡训练。本次巡检结束。

## 2026-09-13 11:02:37 UTC 两卡恢复后单次巡检

- 阶段/step：独立tmux `prepare/encoding`，正式step 0 → 0；先读manual、pipeline、指定 `20260913T110137Z.json` 和最近记录。manual.active=false、training_topology仍pending，距上轮现场30.22分钟；完整证据：`runs/emilia-full-token-scratch/supervision/20260913T110237Z-inspection.json`。
- 进程/退出状态：codec supervisor55543/start_ticks155976101、worker55576/55577/start_ticks均155976117、父PID55543，rank0/1、world_size=2及本轮cmdline匹配；矩阵主PID4832/start_ticks155946508、worker6667/6668/start_ticks155947569/155947570、父PID4832，身份和GPU映射保持，tmux存活。quality1935/export55462已退出，完成报告及日志保持、无保留exit code；pipeline仍列旧PID/running，不据此重启已完成阶段。无总入口父进程符合独立恢复模式。
- 数据/编码：评分8,059/8,059与导出完成标记保持；raw共2,901，最终导出23,931,617条、EN 18,607.63/ZH 25,137.70小时，去重后dialogue614,525条。prepared 2,864 → 2,881（+17），尚余20份，最新修改10:59:27 UTC。当前rank0/1最后chunk2880/2879，恢复后累计新编码459,009条，本间隔新增139,315条、音频约287.40小时（累计965.69小时），指标有限。旧rank2/3仅作历史，未混加计数，最终按全部prepared分片汇总。
- 进展依据：两codec CPU分别增加589.65/569.04秒、rchar约52.97/50.34GB、wchar约430/398MB，分片和日志持续新增，采样两进程均R。累计decode_wait约4537/4419秒、codec约1168/1276秒、write_wait约10.13/7.13秒，peak_allocated约13.16GiB。未发现持续停滞证据。
- GPU/矩阵：两张A800显存48,343/72,607MiB，与上轮相同，可用32,812/8,548MiB；codec47,780/72,044MiB，矩阵各548MiB。GPU1高占用保持但未增长，当前无OOM/退出且持续编码，不擅自调参或认定特定分配器原因。采样利用率0/18%包含矩阵影响，不能当codec吞吐。矩阵身份正确；初次GPU1状态observed_utilization=44、matrix_work=false，记录到忙时暂停计算；后续状态新鲜度1.06/1.12秒。容量测试/正式训练前仍须停止矩阵并确认显存释放，仅暂停计算不释放内存。程序不保证平台不会关闭，之前中断原因未知。
- 资源/日志/配方：主机已用约127GiB（上轮125GiB），MemAvailable约1877.9GiB，无swap；codec RSS约28.43/34.95GiB，数据盘可用约562.0TiB。三个recovery日志、当前rank0/1日志及矩阵日志无Traceback/Error/Exception/OOM/NaN/Inf匹配。质量/导出/codec配方未变，评分源码SHA256匹配；固定DNSMOS/削波/重叠标准、ASMR排除、short→long→合格dialogue短句ID去重及无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 判断/动作/剩余：编码持续推进，无需修复或重启；仅保存证据并追加本文，未更改代码/配置/数据或启停进程。PREPARATION_COMPLETE和preparation.json仍缺，export-error缺；training-process、training-exit、training-plan、train.log及checkpoints均不存在，正式loss/grad/LR与验证/生成尚不适用，未写startup/final-verification。等待剩余编码及数据发布，完成后记录awaiting_training_topology；仅两卡且训练卡数未确认，不启动四卡入口或擅自两卡训练。本次巡检结束。


## 2026-09-13 11:36:04 UTC 编码完成、数据汇总中单次巡检

- 阶段/step：独立准备任务由encoding进入`finalizing`，正式step 0 → 0。已核对manual、pipeline、指定`20260913T113137Z.json`和最近记录；manual.active=false、training_topology仍pending。距上轮主采样33.45分钟；完整证据：`runs/emilia-full-token-scratch/supervision/20260913T113604Z-inspection.json`。
- 数据/编码完成：raw与prepared均为2,901份，文件名集合相等；prepared较上轮2,881新增20份，最后修改11:30:08 UTC。rank0/1最后chunk为2900/2899，两份WORKER_COMPLETE已发布；恢复后累计新编码622,557条，本间隔新增163,548条、约498.00小时，恢复后累计约1463.69小时，指标有限。未混加旧rank2/3状态；最终总量仍须以全部分片汇总报告为准。质量评分8,059/8,059及导出完成保持，导出共23,931,617条，EN 18,607.63/ZH 25,137.70小时，其中去重后dialogue614,525条。
- 进程/汇总进展：codec worker55576/55577已退出；supervisor55543/start_ticks155976101及cmdline匹配，状态为R，codec pipeline为finalizing。代码仅在两worker退出码均为0后进入该阶段（`scripts/prepare_emilia_streaming.py` supervise），因此状态和完成标记共同支持编码正常结束；未另存独立worker退出码文件。20秒复核中supervisor CPU增加16.30秒、rchar增加约1.12GB，仍打开prepared/001964.jsonl，RSS由5.11至5.34GiB，确认汇总持续读取。父进程跨worker退出的累计I/O含子进程回收影响，不把该长间隔大增量当成本次汇总吞吐。finalize会扫描全量数据检查ID/数量、选验证集并发布train/val，目前未出现train/val清单、PREPARATION_COMPLETE或preparation.json；不将暂时无输出视为停滞。quality1935/export55462仍已退出且完成报告保持，无保留exit code；外层pipeline旧PID/running未刷新，不据此重启。
- GPU/矩阵/资源：codec显存已释放，两张A800各仅557MiB，其中矩阵worker6667/6668各548MiB；两worker的start_ticks、父PID4832及cmdline与恢复记录匹配，主程序4832身份正确、tmux存活。矩阵状态新鲜度1.42/1.30秒，采样利用率98/98%来自矩阵，不能当训练或codec吞吐。容量测试/正式训练前须停止矩阵并确认释放。主机已用约41GiB（上轮127GiB），无swap，数据盘可用约561.96TiB。所检恢复日志、当前worker日志和矩阵日志未见Traceback/Error/Exception/OOM/NaN/Inf；无须修复或重启。
- 配方/训练边界：质量/导出/codec配方与上轮相同，评分源码SHA256匹配；固定DNSMOS/削波/重叠门槛、ASMR排除、short→long→合格dialogue短句ID去重及无配额保持。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。training-process、training-exit、training-plan、train.log、checkpoints均不存在；正式loss/grad/LR、验证和生成指标不适用，未写startup/final-verification。
- 判断/动作/剩余：所有编码分片已产出，最终数据汇总持续推进；本轮仅保存证据并追加本文，未改代码/配置/数据或启停进程。等待train/val及最终完成标记发布后记录awaiting_training_topology；当前只有两卡且卡数答复未到，不启动四卡入口或擅自两卡训练。本次巡检结束。


## 2026-09-13 12:02:51 UTC 数据发布完成、等待训练卡数单次巡检

- 阶段/step：数据汇总已完成，正式step 0 → 0。先读manual、pipeline、指定`20260913T120137Z.json`及最近记录；manual.active=false，现场无新训练卡数指令，距上轮主采样26.79分钟。完整证据：`runs/emilia-full-token-scratch/supervision/20260913T120251Z-inspection.json`。
- 发布与全量核验：PREPARATION_COMPLETE和preparation.json于11:56:52 UTC发布，codec pipeline为complete；raw/prepared均2,901份且文件名集合相等，无train/val incomplete。重新流式读取整个train/val，SHA256均与报告相符，行数分别23,931,105和512；train约31.37GB，核验用时48.88秒。总数23,931,617、中英文合计及来源数均与导出报告匹配。train EN 10,890,567/ZH 13,040,538条，约18,607.17/25,137.17小时，训练集总计43,744.34小时；val EN/ZH各256条，512个唯一ID及512个唯一speaker，时长约0.99小时。全量来源short 3,994,914、long 19,322,178、合格dialogue 614,525条；跳过14,117个重复短句ID。报告最大text_tokens=484、duration=30秒、codec_frames=375；这些是观测最大值，不是新增截断。发布流程已通过全量ID唯一性和导出数量校验；报告文本与训练集不重叠，但speaker不与训练集隔离，不能称为未见说话人评估。
- 进程/退出状态：quality1935、export55462、codec supervisor55543和worker55576/55577均已退出，对应准备tmux已退出。质量/导出完成文件、codec complete状态、最终日志和匹配的清单共同支持准备成功；独立任务未保留父进程exit code，不声称核实各父进程exit_code=0。评分仍8,059/8,059；worker恢复计数仍622,557条，无新编码，旧rank2/3不参与新增计数。外层pipeline此前仍显示旧PID/running，已保存原状态到pipeline-history并刷新为waiting/awaiting_training_topology，prepare阶段completed；task-state同步记录awaiting_training_topology，保留原进程身份与核验路径。
- GPU/矩阵/资源：仍仅两张A800，显存各557MiB，其中矩阵worker6667/6668各548MiB；主PID4832/start_ticks155946508及worker start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。两worker近间隔CPU各增加约782.02秒，初次状态新鲜度0.79/0.46秒、matrix_work=true；采样GPU利用率0/0%属于短时计算间隙，不能据此断言矩阵失效，亦不能视为训练吞吐。主机已用约36GiB（上轮41GiB）、无swap，数据盘可用约561.81TiB。所检恢复日志、当前worker及矩阵日志无Traceback/Error/Exception/OOM/NaN/Inf。
- 配方/训练：质量、导出及codec配方与上轮相同，评分源码SHA256匹配；固定DNSMOS/削波/重叠门槛、ASMR排除、short→long→合格dialogue短句ID去重及无配额保持。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。training-process、training-exit、training-plan、train.log、checkpoints均不存在；正式loss/grad/LR、val loss和生成指标尚不适用，未写startup/final-verification。中英按时长1:1的训练目标步数仍须根据最终训练拓扑与动态预算计算，不能用未平衡的43,744.34小时直接充当训练预算。
- 判断/动作/剩余：准备成功并通过发布核验，无需恢复或修复。本轮保存证据、历史状态、更新任务阶段并追加本文，未改代码/配置/数据或启停进程。等待训练卡数答复；若四卡恢复且无相反指令，按原授权执行容量核验及原始assembled/token loss训练。任何容量测试或正式训练前须停止矩阵并确认显存释放。目前保持矩阵和原有半小时监督，本次巡检结束。


## 2026-09-13 12:33:25 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260913T123137Z.json`和最近记录；manual.active=false，训练卡数仍pending，无新卡数指令，距上轮主采样30.56分钟。完整证据：`runs/emilia-full-token-scratch/supervision/20260913T123325Z-inspection.json`。
- 数据/发布：四个数据完成/发布文件保持存在，codec pipeline为complete；raw/prepared均2,901份、文件名集合相等，评分score/stats各8,059份，无新增编码。train/val文件大小和mtime、完成报告均与上轮相同，无incomplete；沿用12:02巡检的全量SHA256/行数核验，本轮未重复扫描31.37GB清单。训练集23,931,105条、43,744.34小时（EN 18,607.17/ZH 25,137.17小时），验证集中英文各256条；全量来源short 3,994,914、long 19,322,178、合格dialogue 614,525条，总数23,931,617，跳过14,117个重复短句ID。恢复worker计数保持622,557条，不混加旧rank2/3。质量/导出/codec配方与上轮一致，评分源码SHA256匹配。
- 进程/退出状态：quality1935、export55462、codec supervisor55543及worker55576/55577仍已退出，准备tmux不存在；完成报告、最终日志及codec complete状态保持，无需重启。未新增独立进程退出码记录，不能声称核实这些父进程exit_code=0。pipeline的prepare阶段为completed，task-state保持awaiting_training_topology；无总入口父进程符合独立恢复后的等待状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB，其中矩阵worker6667/6668各548MiB。主PID4832/start_ticks155946508、worker start_ticks155947569/155947570、cmdline及父PID均匹配，tmux存活；两worker近间隔CPU分别增加892.20/892.21秒、状态新鲜度1.28/1.51秒，matrix_work=true。采样利用率98/98%属于矩阵工作，不能当训练吞吐。主机已用约36GiB，与上轮相近，无swap，数据盘可用约561.71TiB。所检恢复日志、当前worker日志及矩阵日志无Traceback/Error/Exception/OOM/NaN/Inf。
- 训练/边界：本轮training-process、training-exit、training-plan、train.log、checkpoints仍不存在，正式loss/grad/LR、吞吐/data_wait、val loss及speaker_only/icl指标不适用，未写startup/final-verification。ASMR排除、固定DNSMOS/削波/其他说话人重叠门槛、short→long→合格dialogue短句ID去重及无配额保持；音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。实际训练步数须在卡数确定后按中英时长1:1完整平衡epoch及动态预算计算。
- 判断/动作/剩余：准备完成且发布文件无变化，矩阵程序运行正常，无需修复或重启。本轮仅保存现场证据并追加本文，未改代码/配置/数据/任务状态或启停进程。继续等待训练卡数答复；未答复且仅两卡时不启动训练，若四卡恢复且无相反指令可按原授权继续。容量测试/正式训练前须停止矩阵并确认显存释放。保留现有半小时监督，本次巡检结束。


## 2026-09-13 13:02:25 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260913T130137Z.json`及最近记录；manual.active=false，训练卡数仍pending，无新卡数指令，距上轮主采样29.00分钟。完整证据：`runs/emilia-full-token-scratch/supervision/20260913T130225Z-inspection.json`。
- 数据/配方：PREPARATION_COMPLETE、preparation.json、EXPORT_COMPLETE及DIALOGUE_QUALITY_COMPLETE保持存在，codec状态complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份。完成报告、train/val大小与mtime均未变，无incomplete，沿用12:02的全量SHA256/行数核验，本轮未重复扫描。train 23,931,105条、43,744.34小时（EN 18,607.17/ZH 25,137.17），val中英文各256条；合格dialogue 614,525条，跳过重复ID 14,117个。worker状态未变，本间隔新增编码0，旧rank2/3不混入恢复计数。质量/导出/codec配方未变，评分源码SHA256匹配；ASMR排除、固定DNSMOS/削波/重叠门槛、short→long→合格dialogue短句ID去重及无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577仍已退出，准备tmux不存在；完成报告和最终日志保持，未新增独立父进程exit code记录，不声称核实exit_code=0。pipeline的prepare阶段completed、task-state等待卡数，无总入口父进程符合当前状态，无需恢复。所检三个恢复日志、当前worker及矩阵日志无Traceback/Error/Exception/OOM/NaN/Inf。
- GPU/矩阵/资源：仍仅两张A800，各557MiB，其中矩阵worker6667/6668各548MiB。主PID4832/start_ticks155946508、worker start_ticks155947569/155947570、cmdline与父PID匹配，tmux存活；两worker CPU较上轮增加847.10/847.03秒，状态新鲜度1.41/0.96秒，matrix_work=true。采样利用率98/0%与短时计算模式相符，不能用单次0%断言失效，也不能当训练吞吐。主机已用约35GiB（上轮36GiB）、无swap，数据盘可用约561.71TiB。
- 训练/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在；正式loss/grad/LR、吞吐/data_wait、val及生成指标不适用，未写startup/final-verification。数据发布保持、矩阵程序正常，无需修复或重启；本轮仅保存证据并追加本文，未改代码/配置/数据/任务状态或启停进程。继续等待训练卡数答复，未答复且仅两卡不启动训练；四卡恢复且无相反指令时可按原授权继续，从原始assembled开始token loss、中英时长1:1完整平衡epoch。容量测试/正式训练前须停止矩阵并确认释放；实际步数随最终拓扑与动态预算计算。保留原有半小时监督，本次巡检结束。


## 2026-09-13 13:32:22 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先核对manual、pipeline、task-state、指定`20260913T133137Z.json`及最近记录；manual.active=false，训练卡数仍pending，现场无新指令，距上轮29.96分钟。证据：`runs/emilia-full-token-scratch/supervision/20260913T133222Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件保持存在，codec complete，raw/prepared各2,901份且文件名集合相等，score/stats各8,059份。完成报告和train/val大小、mtime未变，无incomplete；沿用12:02已通过的全量SHA256/行数核验，本轮未重新扫描。训练集23,931,105条、43,744.34小时，验证集中英文各256条；合格dialogue614,525条，跳过重复ID14,117个。本间隔编码新增0，旧rank2/3不混入恢复计数。质量/导出/codec配方不变、评分源码SHA256匹配；ASMR排除、固定DNSMOS/削波/其他说话人重叠门槛、short→long→合格dialogue短句ID去重及无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543和worker55576/55577仍已退出，准备tmux不存在，完成报告及最终日志保持。独立父进程退出码未留存，不声称核实exit_code=0。pipeline prepare=completed、task-state等待卡数，无总入口父进程符合当前状态；所检三个恢复日志、当前worker及矩阵日志无Traceback/Error/Exception/OOM/NaN/Inf，无需恢复。
- GPU/矩阵/资源：仍仅两张A800，各557MiB，矩阵worker6667/6668各548MiB；主PID4832/start_ticks155946508、worker start_ticks155947569/155947570及cmdline/父PID均匹配，tmux存活。两worker CPU近间隔增加876.14/875.12秒，状态新鲜度1.11/1.00秒、matrix_work=true。采样GPU利用率59/0%与短时计算模式相符，单次0%不构成故障，矩阵利用率不代表训练吞吐。主机已用约35GiB，与上轮相近，无swap；数据盘可用约561.58TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints仍不存在，正式loss/grad/LR、吞吐/data_wait、验证及生成指标不适用，未写startup/final-verification。数据完成状态保持、矩阵程序正常，无需修复或重启；本轮仅保存证据并追加本文，未改代码/配置/数据/任务状态或启停进程。继续等待训练卡数答复，未答复且仅两卡不启动训练；若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行。容量测试/正式训练前须停止矩阵并确认显存释放；实际步数按最终拓扑与动态预算计算。保留现有半小时监督，本次巡检结束。


## 2026-09-13 14:02:23 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先核对manual、pipeline、task-state、指定`20260913T140137Z.json`及最近记录；manual.active=false，训练卡数仍pending，无新指令，距上轮30.02分钟。证据：`runs/emilia-full-token-scratch/supervision/20260913T140223Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件保持存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份。完成报告和train/val大小、mtime未变，无incomplete；沿用12:02全量SHA256/行数核验，本轮未重复扫描。train 23,931,105条、43,744.34小时，val中英文各256条，合格dialogue614,525条，跳过重复ID14,117个；worker状态保持，本间隔编码新增0，旧rank2/3不混入恢复计数。质量/导出/codec配方不变、评分源码SHA256匹配；ASMR排除、固定DNSMOS/削波/其他说话人重叠门槛、short→long→合格dialogue短句ID去重与无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577仍已退出，准备tmux不存在，完成报告及最终日志保持。独立父进程exit code未留存，不声称核实exit_code=0；pipeline prepare=completed和task-state等待卡数一致，无总入口父进程符合当前状态。所检三个恢复日志、当前worker及矩阵日志无Traceback/Error/Exception/OOM/NaN/Inf，无需恢复。
- GPU/矩阵/资源：仍仅两张A800，各557MiB，其中矩阵worker6667/6668各548MiB；主PID4832/start_ticks155946508、worker start_ticks155947569/155947570及cmdline/父PID均匹配，tmux存活。两worker近间隔CPU增加877.15/876.98秒，状态新鲜度0.92/0.28秒、matrix_work=true；本轮快照利用率98/46%，现场0/0%，符合短时计算采样差异，不能凭单次0%断言失效，也不能当训练吞吐。主机已用约35GiB，与上轮相近，无swap；数据盘可用约561.48TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints仍不存在；正式loss/grad/LR、吞吐/data_wait、val及生成指标不适用，未写startup/final-verification。数据完成状态保持、矩阵程序正常，无需修复或重启；本轮仅保存证据并追加本文，未改代码/配置/数据/任务状态或启停进程。继续等待训练卡数答复，未答复且仅两卡不启动训练；四卡恢复且无相反指令时按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行。容量测试/正式训练前须停止矩阵并确认释放；实际步数按最终拓扑与动态预算计算。保留现有半小时监督，本次巡检结束。


## 2026-09-13 14:32:23 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先核对manual、pipeline、task-state、指定`20260913T143137Z.json`及最近记录；manual.active=false，训练卡数仍pending，无新指令，距上轮29.99分钟。证据：`runs/emilia-full-token-scratch/supervision/20260913T143223Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件仍存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份。完成报告与train/val大小、mtime未变，无incomplete；沿用12:02全量SHA256/行数核验，本轮未重复扫描。train 23,931,105条、43,744.34小时，val中英文各256条，合格dialogue614,525条、跳过重复ID14,117个；worker状态保持，编码新增0，旧rank2/3不混入恢复计数。质量/导出/codec配方未变，评分源码SHA256匹配；ASMR排除、固定DNSMOS/削波/其他说话人重叠门槛、short→long→合格dialogue短句ID去重与无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577仍已退出，准备tmux不存在，完成报告和最终日志保持。独立父进程exit code未留存，不声称核实exit_code=0。pipeline prepare=completed、task-state等待卡数，无总入口父进程符合当前状态，无需恢复。所检三个恢复日志、当前worker及矩阵日志无Traceback/Error/Exception/OOM/NaN/Inf。
- GPU/矩阵/资源：仍仅两张A800，各557MiB，其中矩阵worker6667/6668各548MiB；主PID4832/start_ticks155946508、worker start_ticks155947569/155947570及cmdline/父PID均匹配，tmux存活。两worker近间隔CPU增加876.17/875.48秒，状态新鲜度1.49/1.69秒、matrix_work=true。快照GPU利用率0/0%、现场98/98%与短时计算模式相符，不以单次0%认定失效，矩阵利用率不代表训练吞吐。主机已用约35GiB，与上轮相近，无swap；数据盘可用约561.45TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints仍不存在；正式loss/grad/LR、吞吐/data_wait、val及生成指标不适用，未写startup/final-verification。数据完成状态保持、矩阵程序正常，无需修复或重启；本轮仅保存证据并追加本文，未改代码/配置/数据/任务状态或启停进程。继续等待训练卡数答复，未答复且仅两卡不启动训练；四卡恢复且无相反指令时按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行。容量测试/正式训练前须停止矩阵并确认释放；实际步数按最终拓扑与动态预算计算。保留现有半小时监督，本次巡检结束。


## 2026-09-13 15:02:20 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先核对manual、pipeline、task-state、指定`20260913T150137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮29.95分钟。证据：`runs/emilia-full-token-scratch/supervision/20260913T150220Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件保持存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份。完成报告与train/val大小、mtime未变，无incomplete；沿用12:02全量SHA256/行数核验，本轮未重新扫描。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。本间隔编码新增0，旧rank2/3不混入恢复计数。质量/导出/codec配方不变、评分源码SHA256匹配；ASMR排除、固定DNSMOS/削波/其他说话人重叠门槛、short→long→合格dialogue短句ID去重及无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577仍已退出，准备tmux不存在，完成报告与最终日志保持；独立父进程退出码未留存，不声称核实exit_code=0。pipeline prepare=completed、task-state等待卡数，无总入口父进程符合当前状态，无需恢复。所检三个恢复日志、当前worker及矩阵日志无Traceback/Error/Exception/OOM/NaN/Inf。
- GPU/矩阵/资源：仍仅两张A800，各557MiB，其中矩阵worker6667/6668各548MiB；主PID4832/start_ticks155946508、worker start_ticks155947569/155947570及cmdline/父PID均匹配，tmux存活。两worker CPU近间隔增加874.07/874.05秒，状态新鲜度0.95/1.62秒、matrix_work=true；采样利用率0/97%与短时计算模式相符，不以单次0%认定失效，矩阵利用率不代表训练吞吐。主机已用约36GiB（上轮35GiB）、无swap；数据盘可用约561.31TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints仍不存在；正式loss/grad/LR、吞吐/data_wait、val及生成指标不适用，未写startup/final-verification。数据完成状态保持、矩阵程序正常，无需修复或重启；本轮仅保存证据并追加本文，未改代码/配置/数据/任务状态或启停进程。继续等待训练卡数答复，未答复且仅两卡不启动训练；四卡恢复且无相反指令时按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行。容量测试/正式训练前须停止矩阵并确认释放；实际步数按最终拓扑与动态预算计算。保留现有半小时监督，本次巡检结束。


## 2026-09-13 15:32:37 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260913T153137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮30.29分钟。证据：`runs/emilia-full-token-scratch/supervision/20260913T153237Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件保持存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份。完成报告及train/val大小、mtime未变，无incomplete；沿用12:02全量SHA256/行数核验，本轮未重复扫描。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条，跳过重复ID14,117个。本间隔编码新增0，旧rank2/3不混入恢复计数。质量/导出/codec配方未变、评分源码SHA256匹配；ASMR排除、固定DNSMOS/削波/其他说话人重叠门槛、short→long→合格dialogue短句ID去重及无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543和worker55576/55577仍已退出，准备tmux不存在，完成报告与最终日志保持；独立父进程exit code未留存，不声称核实exit_code=0。pipeline prepare=completed和task-state等待卡数一致，无总入口父进程符合当前状态，无需恢复。所检三个恢复日志、当前worker及矩阵日志无Traceback/Error/Exception/OOM/NaN/Inf。
- GPU/矩阵/资源：仍仅两张A800，各557MiB，其中矩阵worker6667/6668各548MiB；主PID4832/start_ticks155946508、worker start_ticks155947569/155947570及cmdline/父PID均匹配，tmux存活。两worker CPU近间隔增加884.98/883.72秒，状态新鲜度0.64/0.57秒、matrix_work=true；快照GPU利用率98/46%、现场0/0%，符合短时计算采样差异，不以单次0%认定失效，矩阵利用率不代表训练吞吐。主机已用约37GiB（上轮36GiB），无swap；数据盘可用约561.25TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints仍不存在；正式loss/grad/LR、吞吐/data_wait、val及生成指标不适用，未写startup/final-verification。数据完成状态保持、矩阵程序正常，无需修复或重启；本轮仅保存证据并追加本文，未改代码/配置/数据/任务状态或启停进程。继续等待训练卡数答复，未答复且仅两卡不启动训练；四卡恢复且无相反指令时按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行。容量测试/正式训练前须停止矩阵并确认释放；实际步数按最终拓扑与动态预算计算。保留现有半小时监督，本次巡检结束。


## 2026-09-13 16:02:53 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先核对manual、pipeline、task-state、指定`20260913T160137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮30.26分钟。证据：`runs/emilia-full-token-scratch/supervision/20260913T160253Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件保持存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份。完成报告与train/val大小、mtime未变，无incomplete；沿用12:02全量SHA256/行数核验，本轮未重复扫描。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条，跳过重复ID14,117个。worker状态保持，本间隔编码新增0，旧rank2/3不混入恢复计数。质量/导出/codec配方未变、评分源码SHA256匹配；ASMR排除、固定DNSMOS/削波/其他说话人重叠门槛、short→long→合格dialogue短句ID去重及无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577仍已退出，准备tmux不存在，完成报告及最终日志保持；独立父进程exit code未留存，不声称核实exit_code=0。pipeline prepare=completed和task-state等待卡数一致，无总入口父进程符合当前状态，无需恢复。所检三个恢复日志、当前worker及矩阵日志无Traceback/Error/Exception/OOM/NaN/Inf。
- GPU/矩阵/资源：仍仅两张A800，各557MiB，其中矩阵worker6667/6668各548MiB；主PID4832/start_ticks155946508、worker start_ticks155947569/155947570及cmdline/父PID均匹配，tmux存活。两worker CPU近间隔增加883.29/882.36秒，状态新鲜度1.37/1.43秒、matrix_work=true；采样利用率98/97%来自矩阵，不能当训练吞吐。主机已用约36GiB（上轮37GiB），无swap；数据盘可用约561.15TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints仍不存在；正式loss/grad/LR、吞吐/data_wait、val及生成指标不适用，未写startup/final-verification。数据完成状态保持、矩阵程序正常，无需修复或重启；本轮仅保存证据并追加本文，未改代码/配置/数据/任务状态或启停进程。继续等待训练卡数答复，未答复且仅两卡不启动训练；四卡恢复且无相反指令时按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行。容量测试/正式训练前须停止矩阵并确认释放；实际步数按最终拓扑与动态预算计算。保留现有半小时监督，本次巡检结束。


## 2026-09-13 16:33:15 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260913T163137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮30.36分钟。证据：`runs/emilia-full-token-scratch/supervision/20260913T163315Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件仍存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份。完成报告与train/val大小、mtime未变，无incomplete，沿用12:02全量SHA256/行数核验，本轮未重复扫描。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。worker状态不变，编码新增0，旧rank2/3不混入恢复计数。质量/导出/codec配方未变、评分源码SHA256匹配；ASMR排除、固定DNSMOS/削波/其他说话人重叠门槛、short→long→合格dialogue短句ID去重及无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577仍已退出，准备tmux不存在，完成报告和最终日志保持；独立父进程退出码未留存，不声称核实exit_code=0。pipeline prepare=completed与task-state等待卡数一致，无总入口父进程符合当前状态，无需恢复。所检三个恢复日志、当前worker及矩阵日志无Traceback/Error/Exception/OOM/NaN/Inf。
- GPU/矩阵/资源：仍仅两张A800，各557MiB，矩阵worker6667/6668各548MiB；主PID4832/start_ticks155946508、worker start_ticks155947569/155947570及cmdline/父PID均匹配，tmux存活。两worker CPU近间隔增加886.78/886.90秒，状态新鲜度0.05/0.14秒、matrix_work=true；采样利用率33/45%来自矩阵，不能当训练吞吐。主机已用约37GiB（上轮36GiB），无swap；数据盘可用约561.01TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints仍不存在；正式loss/grad/LR、吞吐/data_wait、val及生成指标不适用，未写startup/final-verification。数据完成状态保持、矩阵程序正常，无需修复或重启。本轮仅保存证据并追加本文，未改代码/配置/数据/任务状态或启停进程。继续等待训练卡数答复，未答复且仅两卡不启动训练；四卡恢复且无相反指令时按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行。容量测试/正式训练前须停止矩阵并确认释放；实际步数按最终拓扑与动态预算计算。保留现有半小时监督，本次巡检结束。


## 2026-09-13 17:02:22 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260913T170137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮29.12分钟。证据：`runs/emilia-full-token-scratch/supervision/20260913T170222Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件保持存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份。完成报告及train/val大小、mtime未变，无incomplete；沿用12:02全量SHA256/行数核验，本轮未重复扫描。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。worker状态不变，编码新增0，旧rank2/3不混入恢复计数。质量/导出/codec配方未变、评分源码SHA256匹配；ASMR排除、固定DNSMOS/削波/其他说话人重叠门槛、short→long→合格dialogue短句ID去重及无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577仍已退出，准备tmux不存在，完成报告与最终日志保持；独立父进程exit code未留存，不声称核实exit_code=0。pipeline prepare=completed与task-state等待卡数一致，无总入口父进程符合当前状态，无需恢复。所检三个恢复日志、当前worker及矩阵日志无Traceback/Error/Exception/OOM/NaN/Inf。
- GPU/矩阵/资源：仍仅两张A800，各557MiB，矩阵worker6667/6668各548MiB；主PID4832/start_ticks155946508、worker start_ticks155947569/155947570及cmdline/父PID均匹配，tmux存活。两worker CPU近间隔增加849.65/849.51秒，状态新鲜度1.91/1.90秒、matrix_work=true；采样利用率98/98%来自矩阵，不能当训练吞吐。主机已用约38GiB（上轮37GiB），无swap；数据盘可用约560.87TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints仍不存在；正式loss/grad/LR、吞吐/data_wait、val及生成指标不适用，未写startup/final-verification。数据完成状态保持、矩阵程序正常，无需修复或重启。本轮仅保存证据并追加本文，未改代码/配置/数据/任务状态或启停进程。继续等待训练卡数答复，未答复且仅两卡不启动训练；四卡恢复且无相反指令时按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行。容量测试/正式训练前须停止矩阵并确认释放；实际步数按最终拓扑与动态预算计算。保留现有半小时监督，本次巡检结束。


## 2026-09-13 17:33:22 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先核对manual、pipeline、task-state、指定`20260913T173137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮30.99分钟。证据：`runs/emilia-full-token-scratch/supervision/20260913T173322Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件保持存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份。完成报告及train/val大小、mtime未变，无incomplete；沿用12:02全量SHA256/行数核验，本轮未重复扫描。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条，跳过重复ID14,117个。worker状态不变，编码新增0，旧rank2/3不混入恢复计数。质量/导出/codec配方未变、评分源码SHA256匹配；ASMR排除、固定DNSMOS/削波/其他说话人重叠门槛、short→long→合格dialogue短句ID去重及无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577仍已退出，准备tmux不存在，完成报告与最终日志保持；独立父进程exit code未留存，不声称核实exit_code=0。pipeline prepare=completed与task-state等待卡数一致，无总入口父进程符合当前状态，无需恢复。所检三个恢复日志、当前worker及矩阵日志无Traceback/Error/Exception/OOM/NaN/Inf。
- GPU/矩阵/资源：仍仅两张A800，各557MiB，矩阵worker6667/6668各548MiB；主PID4832/start_ticks155946508、worker start_ticks155947569/155947570及cmdline/父PID均匹配，tmux存活。两worker CPU近间隔增加904.99/904.07秒，状态新鲜度0.96/1.46秒、matrix_work=true；采样利用率0/97%符合短时计算模式，不能以单次0%认定失效，矩阵利用率不代表训练吞吐。主机已用约43GiB（上轮38GiB），无swap、内存余量充足；数据盘可用约560.70TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints仍不存在；正式loss/grad/LR、吞吐/data_wait、val及生成指标不适用，未写startup/final-verification。数据完成状态保持、矩阵程序正常，无需修复或重启。本轮仅保存证据并追加本文，未改代码/配置/数据/任务状态或启停进程。继续等待训练卡数答复，未答复且仅两卡不启动训练；四卡恢复且无相反指令时按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行。容量测试/正式训练前须停止矩阵并确认释放；实际步数按最终拓扑与动态预算计算。保留现有半小时监督，本次巡检结束。


## 2026-09-13 18:08:27 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先核对manual、pipeline、task-state、指定`20260913T180137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮35.09分钟。证据：`runs/emilia-full-token-scratch/supervision/20260913T180827Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件保持存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份。完成报告及train/val大小、mtime未变，无incomplete；沿用12:02全量SHA256/行数核验，本轮未重复扫描。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条，跳过重复ID14,117个。worker状态未变，编码新增0，旧rank2/3不混入恢复计数。质量/导出/codec配方未变、评分源码SHA256匹配；ASMR排除、固定DNSMOS/削波/其他说话人重叠门槛、short→long→合格dialogue短句ID去重及无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577仍已退出，准备tmux不存在，完成报告与最终日志保持；独立父进程exit code未留存，不声称核实exit_code=0。pipeline prepare=completed与task-state等待卡数一致，无总入口父进程符合当前状态，无需恢复。所检三个恢复日志、当前worker及矩阵日志无Traceback/Error/Exception/OOM/NaN/Inf。
- GPU/矩阵/资源：仍仅两张A800，各557MiB，矩阵worker6667/6668各548MiB；主PID4832/start_ticks155946508、worker start_ticks155947569/155947570及cmdline/父PID均匹配，tmux存活。两worker CPU近间隔增加1024.85/1023.94秒，状态新鲜度2.01/0.05秒、matrix_work=true；采样利用率62/58%来自矩阵，不能当训练吞吐。主机已用约37GiB（上轮43GiB），无swap；数据盘可用约560.55TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints仍不存在；正式loss/grad/LR、吞吐/data_wait、val及生成指标不适用，未写startup/final-verification。数据完成状态保持、矩阵程序正常，无需修复或重启。本轮仅保存证据并追加本文，未改代码/配置/数据/任务状态或启停进程。继续等待训练卡数答复，未答复且仅两卡不启动训练；四卡恢复且无相反指令时按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行。容量测试/正式训练前须停止矩阵并确认释放；实际步数按最终拓扑与动态预算计算。保留现有半小时监督，本次巡检结束。


## 2026-09-13 18:32:58 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260913T183137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮24.50分钟。证据：`runs/emilia-full-token-scratch/supervision/20260913T183258Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件保持存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份。完成报告及train/val大小、mtime未变，无incomplete；沿用12:02全量SHA256/行数核验，本轮未重复扫描。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。worker状态不变，编码新增0，旧rank2/3不混入恢复计数。质量/导出/codec配方未变、评分源码SHA256匹配；ASMR排除、固定DNSMOS/削波/其他说话人重叠门槛、short→long→合格dialogue短句ID去重及无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577仍已退出，准备tmux不存在，完成报告与最终日志保持；独立父进程exit code未留存，不声称核实exit_code=0。pipeline prepare=completed与task-state等待卡数一致，无总入口父进程符合当前状态，无需恢复。所检三个恢复日志、当前worker及矩阵日志无Traceback/Error/Exception/OOM/NaN/Inf。
- GPU/矩阵/资源：仍仅两张A800，各557MiB，矩阵worker6667/6668各548MiB；主PID4832/start_ticks155946508、worker start_ticks155947569/155947570及cmdline/父PID均匹配，tmux存活。两worker CPU近间隔增加715.37/715.24秒，状态新鲜度0.77/0.46秒、matrix_work=true；快照GPU利用率0/84%、现场0/0%，符合短时计算采样差异，不以单次0%认定失效，矩阵利用率不代表训练吞吐。主机已用约37GiB，与上轮相近，无swap；数据盘可用约560.43TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints仍不存在；正式loss/grad/LR、吞吐/data_wait、val及生成指标不适用，未写startup/final-verification。数据完成状态保持、矩阵程序正常，无需修复或重启。本轮仅保存证据并追加本文，未改代码/配置/数据/任务状态或启停进程。继续等待训练卡数答复，未答复且仅两卡不启动训练；四卡恢复且无相反指令时按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行。容量测试/正式训练前须停止矩阵并确认释放；实际步数按最终拓扑与动态预算计算。保留现有半小时监督，本次巡检结束。


## 2026-09-13 19:02:55 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260913T190137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮29.96分钟。证据：`runs/emilia-full-token-scratch/supervision/20260913T190255Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件保持存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份。完成报告及train/val大小、mtime未变，无incomplete；沿用12:02全量SHA256/行数核验，本轮未重复扫描。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。worker状态不变，编码新增0，旧rank2/3不混入恢复计数。质量/导出/codec配方未变、评分源码SHA256匹配；ASMR排除、固定DNSMOS/削波/其他说话人重叠门槛、short→long→合格dialogue短句ID去重及无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577仍已退出，准备tmux不存在，完成报告与最终日志保持；独立父进程exit code未留存，不声称核实exit_code=0。pipeline prepare=completed与task-state等待卡数一致，无总入口父进程符合当前状态，无需恢复。所检三个恢复日志、当前worker及矩阵日志无Traceback/Error/Exception/OOM/NaN/Inf。
- GPU/矩阵/资源：仍仅两张A800，各557MiB，矩阵worker6667/6668各548MiB；主PID4832/start_ticks155946508、worker start_ticks155947569/155947570及cmdline/父PID均匹配，tmux存活。两worker CPU近间隔增加865.12/864.92秒，状态新鲜度1.35/0.38秒、matrix_work=true；采样利用率98/0%符合短时计算模式，不能以单次0%认定失效，矩阵利用率不代表训练吞吐。主机已用约37GiB，与上轮相近，无swap。数据盘可用约566.29TiB，比上轮增加；本轮未清理任何数据，未对共享盘空间变化归因。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints仍不存在；正式loss/grad/LR、吞吐/data_wait、val及生成指标不适用，未写startup/final-verification。数据完成状态保持、矩阵程序正常，无需修复或重启。本轮仅保存证据并追加本文，未改代码/配置/数据/任务状态或启停进程。继续等待训练卡数答复，未答复且仅两卡不启动训练；四卡恢复且无相反指令时按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行。容量测试/正式训练前须停止矩阵并确认释放；实际步数按最终拓扑与动态预算计算。保留现有半小时监督，本次巡检结束。


## 2026-09-13 19:32:53 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先核对manual、pipeline、task-state、指定`20260913T193137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮29.97分钟。证据：`runs/emilia-full-token-scratch/supervision/20260913T193253Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件保持存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份。完成报告及train/val大小、mtime未变，无incomplete；沿用12:02全量SHA256/行数核验，本轮未重复扫描。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。worker状态不变，编码新增0，旧rank2/3不混入恢复计数。质量/导出/codec配方未变、评分源码SHA256匹配；ASMR排除、固定DNSMOS/削波/其他说话人重叠门槛、short→long→合格dialogue短句ID去重及无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577仍已退出，准备tmux不存在，完成报告与最终日志保持；独立父进程exit code未留存，不声称核实exit_code=0。pipeline prepare=completed与task-state等待卡数一致，无总入口父进程符合当前状态，无需恢复。所检三个恢复日志、当前worker及矩阵日志无Traceback/Error/Exception/OOM/NaN/Inf。
- GPU/矩阵/资源：仍仅两张A800，各557MiB，矩阵worker6667/6668各548MiB；主PID4832/start_ticks155946508、worker start_ticks155947569/155947570及cmdline/父PID均匹配，tmux存活。两worker CPU近间隔增加875.12/875.23秒，状态新鲜度0.52/0.34秒、matrix_work=true；采样利用率0/0%符合短时计算间隙，不能据此认定失效，也不将矩阵利用率当训练吞吐。主机已用约37GiB，与上轮相近，无swap；数据盘可用约566.18TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints仍不存在；正式loss/grad/LR、吞吐/data_wait、val及生成指标不适用，未写startup/final-verification。数据完成状态保持、矩阵程序正常，无需修复或重启。本轮仅保存证据并追加本文，未改代码/配置/数据/任务状态或启停进程。继续等待训练卡数答复，未答复且仅两卡不启动训练；四卡恢复且无相反指令时按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行。容量测试/正式训练前须停止矩阵并确认释放；实际步数按最终拓扑与动态预算计算。保留现有半小时监督，本次巡检结束。


## 2026-09-13 20:02:26 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260913T200137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮29.54分钟。证据：`runs/emilia-full-token-scratch/supervision/20260913T200226Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件保持存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份。完成报告及train/val大小、mtime未变，无incomplete；沿用12:02全量SHA256/行数核验，本轮未重复扫描。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条，跳过重复ID14,117个。worker状态不变，编码新增0，旧rank2/3不混入恢复计数。质量/导出/codec配方未变、评分源码SHA256匹配；ASMR排除、固定DNSMOS/削波/其他说话人重叠门槛、short→long→合格dialogue短句ID去重及无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577仍已退出，准备tmux不存在，完成报告与最终日志保持；独立父进程exit code未留存，不声称核实exit_code=0。pipeline prepare=completed与task-state等待卡数一致，无总入口父进程符合当前状态，无需恢复。所检三个恢复日志、当前worker及矩阵日志无Traceback/Error/Exception/OOM/NaN/Inf。
- GPU/矩阵/资源：仍仅两张A800，各557MiB，矩阵worker6667/6668各548MiB；主PID4832/start_ticks155946508、worker start_ticks155947569/155947570及cmdline/父PID均匹配，tmux存活。两worker CPU近间隔增加863.25/862.13秒，状态新鲜度1.55/0.78秒、matrix_work=true；采样利用率98/0%符合短时计算模式，不能以单次0%认定失效，矩阵利用率不代表训练吞吐。主机已用约37GiB，与上轮相近，无swap；数据盘可用约566.09TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints仍不存在；正式loss/grad/LR、吞吐/data_wait、val及生成指标不适用，未写startup/final-verification。数据完成状态保持、矩阵程序正常，无需修复或重启。本轮仅保存证据并追加本文，未改代码/配置/数据/任务状态或启停进程。继续等待训练卡数答复，未答复且仅两卡不启动训练；四卡恢复且无相反指令时按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行。容量测试/正式训练前须停止矩阵并确认释放；实际步数按最终拓扑与动态预算计算。保留现有半小时监督，本次巡检结束。


## 2026-09-13 20:32:20 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260913T203137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮29.90分钟。证据：`runs/emilia-full-token-scratch/supervision/20260913T203220Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件保持存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份。完成报告及train/val大小、mtime未变，无incomplete；沿用12:02全量SHA256/行数核验，本轮未重复扫描。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。worker状态不变，编码新增0，旧rank2/3不混入恢复计数。质量/导出/codec配方未变、评分源码SHA256匹配；ASMR排除、固定DNSMOS/削波/其他说话人重叠门槛、short→long→合格dialogue短句ID去重及无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577仍已退出，准备tmux不存在，完成报告与最终日志保持；独立父进程exit code未留存，不声称核实exit_code=0。pipeline prepare=completed与task-state等待卡数一致，无总入口父进程符合当前状态，无需恢复。所检三个恢复日志、当前worker及矩阵日志无Traceback/Error/Exception/OOM/NaN/Inf。
- GPU/矩阵/资源：仍仅两张A800，各557MiB，矩阵worker6667/6668各548MiB；主PID4832/start_ticks155946508、worker start_ticks155947569/155947570及cmdline/父PID均匹配，tmux存活。两worker CPU近间隔增加873.96/873.85秒，状态新鲜度0.54/0.12秒、matrix_work=true；采样利用率0/18%符合短时计算模式，不能以单次低利用率认定失效，矩阵利用率不代表训练吞吐。主机已用约37GiB，与上轮相近，无swap；数据盘可用约565.97TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints仍不存在；正式loss/grad/LR、吞吐/data_wait、val及生成指标不适用，未写startup/final-verification。数据完成状态保持、矩阵程序正常，无需修复或重启。本轮仅保存证据并追加本文，未改代码/配置/数据/任务状态或启停进程。继续等待训练卡数答复，未答复且仅两卡不启动训练；四卡恢复且无相反指令时按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行。容量测试/正式训练前须停止矩阵并确认释放；实际步数按最终拓扑与动态预算计算。保留现有半小时监督，本次巡检结束。


## 2026-09-13 21:02:28 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260913T210137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮30.13分钟。证据：`runs/emilia-full-token-scratch/supervision/20260913T210228Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件保持存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份。完成报告及train/val大小、mtime未变，无incomplete；沿用12:02全量SHA256/行数核验，本轮未重复扫描。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。worker状态不变，编码新增0，旧rank2/3不混入恢复计数。质量/导出/codec配方未变、评分源码SHA256匹配；ASMR排除、固定DNSMOS/削波/其他说话人重叠门槛、short→long→合格dialogue短句ID去重及无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577仍已退出，准备tmux不存在，完成报告与最终日志保持；独立父进程exit code未留存，不声称核实exit_code=0。pipeline prepare=completed与task-state等待卡数一致，无总入口父进程符合当前状态，无需恢复。所检三个恢复日志、当前worker及矩阵日志无Traceback/Error/Exception/OOM/NaN/Inf。
- GPU/矩阵/资源：仍仅两张A800，各557MiB，矩阵worker6667/6668各548MiB；主PID4832/start_ticks155946508、worker start_ticks155947569/155947570及cmdline/父PID均匹配，tmux存活。两worker CPU近间隔增加880.11/879.37秒，状态新鲜度0.72/0.66秒、matrix_work=true；快照利用率98/98%、现场0/0%，符合短时计算采样差异，不以单次0%认定失效，矩阵利用率不代表训练吞吐。主机已用约37GiB，与上轮相近，无swap；数据盘可用约565.90TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints仍不存在；正式loss/grad/LR、吞吐/data_wait、val及生成指标不适用，未写startup/final-verification。数据完成状态保持、矩阵程序正常，无需修复或重启。本轮仅保存证据并追加本文，未改代码/配置/数据/任务状态或启停进程。继续等待训练卡数答复，未答复且仅两卡不启动训练；四卡恢复且无相反指令时按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行。容量测试/正式训练前须停止矩阵并确认释放；实际步数按最终拓扑与动态预算计算。保留现有半小时监督，本次巡检结束。


## 2026-09-13 21:33:09 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260913T213137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮30.69分钟。证据：`runs/emilia-full-token-scratch/supervision/20260913T213309Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件保持存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份。完成报告及train/val大小、mtime未变，无incomplete；沿用12:02全量SHA256/行数核验，本轮未重复扫描。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。worker状态不变，编码新增0，旧rank2/3不混入恢复计数。质量/导出/codec配方未变、评分源码SHA256匹配；ASMR排除、固定DNSMOS/削波/其他说话人重叠门槛、short→long→合格dialogue短句ID去重及无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577仍已退出，准备tmux不存在，完成报告与最终日志保持；独立父进程exit code未留存，不声称核实exit_code=0。pipeline prepare=completed与task-state等待卡数一致，无总入口父进程符合当前状态，无需恢复。所检三个恢复日志、当前worker及矩阵日志无Traceback/Error/Exception/OOM/NaN/Inf。
- GPU/矩阵/资源：仍仅两张A800，各557MiB，矩阵worker6667/6668各548MiB；主PID4832/start_ticks155946508、worker start_ticks155947569/155947570及cmdline/父PID均匹配，tmux存活。两worker CPU近间隔增加895.12/895.16秒，状态新鲜度1.32/1.40秒、matrix_work=true；采样利用率98/98%来自矩阵，不能当训练吞吐。主机已用约37GiB，与上轮相近，无swap；数据盘可用约565.82TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints仍不存在；正式loss/grad/LR、吞吐/data_wait、val及生成指标不适用，未写startup/final-verification。数据完成状态保持、矩阵程序正常，无需修复或重启。本轮仅保存证据并追加本文，未改代码/配置/数据/任务状态或启停进程。继续等待训练卡数答复，未答复且仅两卡不启动训练；四卡恢复且无相反指令时按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行。容量测试/正式训练前须停止矩阵并确认释放；实际步数按最终拓扑与动态预算计算。保留现有半小时监督，本次巡检结束。


## 2026-09-13 22:32:59 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260913T223137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令。距最近已记录巡检59.83分钟，本轮不补写中间时点的观察。证据：`runs/emilia-full-token-scratch/supervision/20260913T223259Z-inspection.json`。
- 数据/配方：PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、worker状态均未变；沿用12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；纳入合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；阶段状态及完成报告一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志均未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度1.06/0.63秒，matrix_work=true；距最近记录CPU分别增加1748.24/1748.23秒。快照利用率98/98%、现场57/0%，符合短时计算采样差异，不能以单次0%判为失效，也不当作训练吞吐。主机已用约36GiB，无swap；数据盘可用约565.67TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-13 23:02:23 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260913T230137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮29.40分钟。证据：`runs/emilia-full-token-scratch/supervision/20260913T230223Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度0.07/1.89秒，matrix_work=true；近间隔CPU分别增加858.78/858.53秒。快照利用率0/0%、现场94/97%，符合短时计算采样差异，不能以单次0%判为失效，也不当作训练吞吐。主机已用约36GiB，无swap；数据盘可用约565.64TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-13 23:32:35 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260913T233137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮30.21分钟。证据：`runs/emilia-full-token-scratch/supervision/20260913T233235Z-inspection.json`。
- 数据/配方：PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度1.57/1.72秒，matrix_work=true；近间隔CPU分别增加881.47/881.76秒。快照利用率0/0%、现场98/97%，符合短时计算采样差异，不能以单次0%判为失效，也不当作训练吞吐。主机已用约36GiB，无swap；数据盘可用约565.58TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 00:02:48 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T000137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮30.21分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T000248Z-inspection.json`。
- 数据/配方：PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度1.93/1.99秒，matrix_work=true；近间隔CPU分别增加882.59/882.62秒。快照利用率0/0%、现场98/98%，符合短时计算采样差异，不能以单次0%判为失效，也不当作训练吞吐。主机已用约36GiB，无swap；数据盘可用约565.55TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 00:32:53 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T003137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮30.09分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T003253Z-inspection.json`。
- 数据/配方：PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度1.53/1.07秒，matrix_work=true；近间隔CPU分别增加878.77/878.57秒。快照利用率25/0%、现场98/7%，符合短时计算采样差异，不能以单次低利用率判为失效，也不当作训练吞吐。主机已用约36GiB，无swap；数据盘可用约565.53TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 01:02:19 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T010137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮29.43分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T010219Z-inspection.json`。
- 数据/配方：PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度0.92/0.51秒，matrix_work=true；近间隔CPU分别增加860.05/860.07秒。快照利用率98/29%、现场0/0%，符合短时计算采样差异，不能以单次0%判为失效，也不当作训练吞吐。主机已用约36GiB，无swap；数据盘可用约565.50TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 01:33:56 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T013137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮31.61分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T013356Z-inspection.json`。
- 数据/配方：PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度0.55/0.16秒，matrix_work=true；近间隔CPU分别增加924.08/922.70秒。快照利用率0/0%、现场0/35%，符合短时计算采样差异，不能以单次低利用率判为失效，也不当作训练吞吐。主机已用约36GiB，无swap；数据盘可用约565.78TiB，比上轮增加；本轮未清理数据，不对共享盘空间变化归因。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 02:33:37 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、最新`20260914T023137Z.json`和此前`20260914T020137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令。两份快照均无训练进程/指标/检查点。本次只执行一次现场巡检，距最近实际巡检59.68分钟，不补写02:01时点的现场观察。证据：`runs/emilia-full-token-scratch/supervision/20260914T023337Z-inspection.json`。
- 数据/配方：PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度0.24/0.67秒，matrix_work=true；距最近实际巡检CPU分别增加1744.19/1742.70秒。最新快照利用率98/96%、现场0/0%，符合短时计算采样差异，不能以单次0%判为失效，也不当作训练吞吐。主机已用约36GiB，无swap；数据盘可用约565.74TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 03:03:11 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T030137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮29.57分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T030311Z-inspection.json`。
- 数据/配方：PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度0.57/1.20秒，matrix_work=true；近间隔CPU分别增加864.27/863.11秒。快照利用率71/9%、现场0/98%，符合短时计算采样差异，不能以单次0%判为失效，也不当作训练吞吐。主机已用约35GiB，无swap；数据盘可用约565.72TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 03:34:36 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T033137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮31.42分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T033436Z-inspection.json`。
- 数据/配方：PREPARATION_COMPLETE、EXPORT_COMPLETE.json、DIALOGUE_QUALITY_COMPLETE.json、preparation.json均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度1.88/1.10秒，matrix_work=true；近间隔CPU分别增加917.63/917.19秒。快照利用率98/0%、现场98/17%，符合短时计算采样差异，不能以单次低利用率判为失效，也不当作训练吞吐。主机已用约36GiB，无swap；数据盘可用约565.57TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 04:02:24 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T040137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮27.80分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T040224Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度0.93/1.43秒，matrix_work=true；近间隔CPU分别增加811.60/812.03秒。快照利用率83/98%、现场0/98%，符合短时计算采样差异，不能以单次0%判为失效，也不当作训练吞吐。主机已用约36GiB，无swap；数据盘可用约565.54TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 04:32:53 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T043137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮30.49分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T043253Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度0.97/1.05秒，matrix_work=true；近间隔CPU分别增加890.19/890.24秒。快照利用率0/0%、现场0/58%，符合短时计算采样差异，不能以单次0%判为失效，也不当作训练吞吐。主机已用约36GiB，无swap；数据盘可用约565.42TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 05:02:37 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T050137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮29.74分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T050237Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度0.53/0.04秒，matrix_work=true；近间隔CPU分别增加868.21/867.89秒。快照利用率98/0%、现场0/57%，符合短时计算采样差异，不能以单次0%判为失效，也不当作训练吞吐。主机已用约36GiB，无swap；数据盘可用约565.36TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 05:35:26 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T053137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮32.82分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T053526Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度1.58/1.15秒，matrix_work=true；近间隔CPU分别增加958.38/957.65秒。快照利用率0/98%、现场98/98%，符合短时计算采样差异，不能以单次0%判为失效，也不当作训练吞吐。主机已用约36GiB，无swap；数据盘可用约565.24TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 06:02:41 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T060137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮27.24分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T060241Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度1.68/1.92秒，matrix_work=true；近间隔CPU分别增加796.13/795.51秒。快照利用率37/98%、现场98/98%，符合短时计算采样差异，也不当作训练吞吐。主机已用约36GiB，无swap；数据盘可用约565.13TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 06:34:04 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T063137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮31.38分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T063404Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度0.42/1.53秒，matrix_work=true；近间隔CPU分别增加916.84/915.79秒。快照利用率98/0%、现场0/98%，符合短时计算采样差异，不能以单次0%判为失效，也不当作训练吞吐。主机已用约36GiB，无swap；数据盘可用约565.01TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 07:02:41 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T070137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮28.62分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T070241Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度1.87/1.48秒，matrix_work=true；近间隔CPU分别增加834.55/835.15秒。快照利用率48/98%、现场98/98%，符合短时计算采样差异，也不当作训练吞吐。主机已用约36GiB，无swap；数据盘可用约565.02TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 07:32:30 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T073137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮29.83分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T073230Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度1.83/0.73秒，matrix_work=true；近间隔CPU分别增加871.10/871.18秒。快照利用率98/0%、现场97/0%，符合短时计算采样差异，不能以单次0%判为失效，也不当作训练吞吐。主机已用约36GiB，无swap；数据盘可用约564.98TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 08:03:08 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T080137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮30.63分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T080308Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度1.46/1.16秒，matrix_work=true；近间隔CPU分别增加894.87/894.22秒。快照利用率0/93%、现场98/80%，符合短时计算采样差异，不能以单次0%判为失效，也不当作训练吞吐。主机已用约36GiB，无swap；数据盘可用约564.87TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 08:32:22 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T083137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮29.23分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T083222Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度0.36/0.03秒，matrix_work=true；近间隔CPU分别增加853.99/853.76秒。快照及现场利用率均0/98%，单次0%不代表失效，矩阵利用率不当作训练吞吐。主机已用约36GiB，无swap；数据盘可用约564.85TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 09:02:39 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T090137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮30.28分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T090239Z-inspection.json`。
- 数据/配方：四个数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度0.82/0.75秒，matrix_work=true；近间隔CPU分别增加884.21/883.59秒。快照及现场利用率均0/0%，结合状态刷新及CPU增量符合短时计算间隙，不能以单次0%判为失效；矩阵利用率不当作训练吞吐。主机已用约36GiB，无swap；数据盘可用约564.72TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 09:34:06 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。已读manual、pipeline、task-state、指定`20260914T093137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮31.46分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T093406Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度0.68/0.48秒，matrix_work=true；近间隔CPU分别增加918.22/918.25秒。快照利用率98/98%、现场0/0%，结合状态刷新及CPU增量符合短时计算间隙，不能以单次0%判为失效；矩阵利用率不当作训练吞吐。主机已用约36GiB，无swap；数据盘可用约564.55TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 10:05:20 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T100137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮31.24分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T100520Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度1.26/1.54秒，matrix_work=true；近间隔CPU分别增加911.36/911.72秒。快照利用率0/0%、现场0/98%，结合状态刷新及CPU增量符合短时计算间隙，不能以单次0%判为失效；矩阵利用率不当作训练吞吐。主机已用约36GiB，无swap；数据盘可用约564.45TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 10:32:22 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T103137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮27.03分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T103222Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度0.96/0.68秒，matrix_work=true；近间隔CPU分别增加788.85/788.60秒。快照及现场利用率均0/0%，结合状态刷新及CPU增量符合短时计算间隙，不能以单次0%判为失效；矩阵利用率不当作训练吞吐。主机已用约36GiB，无swap；数据盘可用约564.38TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 11:02:22 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T110137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮约30分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T110222Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度0.71/1.42秒，matrix_work=true；近间隔CPU分别增加876.14/875.55秒。快照及现场利用率均0/98%，结合状态刷新及CPU增量符合短时计算采样差异，不能以单次0%判为失效；矩阵利用率不当作训练吞吐。主机已用约36GiB，无swap；数据盘可用约564.25TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 11:32:47 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T113137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮约30.4分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T113247Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度1.18/0.66秒，matrix_work=true；近间隔CPU分别增加888.28/887.83秒。快照利用率2/0%、现场35/0%，结合状态刷新及CPU增量符合短时计算采样差异，不能以单次0%判为失效；矩阵利用率不当作训练吞吐。主机已用约37GiB，无swap；数据盘可用约564.23TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 12:02:22 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T120137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮约29.6分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T120222Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度0.48/2.00秒，matrix_work=true；近间隔CPU分别增加863.97/864.09秒。快照利用率94/98%、现场0/98%，结合状态刷新及CPU增量符合短时计算采样差异，不能以单次0%判为失效；矩阵利用率不当作训练吞吐。主机已用约37GiB，无swap；数据盘可用约564.16TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 12:32:24 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T123137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮约30分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T123224Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度0.60/0.67秒，matrix_work=true；近间隔CPU分别增加876.26/876.37秒。快照利用率73/60%、现场0/0%，结合状态刷新及CPU增量符合短时计算间隙，不能以单次0%判为失效；矩阵利用率不当作训练吞吐。主机已用约50GiB（上轮37GiB），仍可用约1.9TiB，无swap；未据此归因于本run或操作其他任务。数据盘可用约564.11TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 13:02:22 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T130137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮约30分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T130222Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度1.09/1.07秒，matrix_work=true；近间隔CPU分别增加870.38/870.42秒。快照利用率98/98%、现场16/65%，符合短时计算采样差异；矩阵利用率不当作训练吞吐。主机已用约53GiB（上轮50GiB），仍可用约1.9TiB，无swap；未据此归因于本run或操作其他任务。数据盘可用约564.04TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 13:32:18 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T133137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮约30分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T133218Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度1.07/1.04秒，matrix_work=true；近间隔CPU分别增加872.44/872.46秒。快照利用率0/0%、现场31/62%，符合短时计算采样差异，不能以单次0%判为失效；矩阵利用率不当作训练吞吐。主机已用约53GiB，仍可用约1.9TiB，无swap；数据盘可用约564.03TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 14:02:21 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T140137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮约30分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T140221Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度1.73/1.68秒，matrix_work=true；近间隔CPU分别增加876.07/876.09秒。快照利用率0/0%、现场98/98%，符合短时计算采样差异，不能以单次0%判为失效；矩阵利用率不当作训练吞吐。主机已用约53GiB，仍可用约1.9TiB，无swap；数据盘可用约564.07TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 14:32:17 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T143137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮约30分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T143217Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度1.66/1.71秒，matrix_work=true；近间隔CPU分别增加872.22/872.30秒。快照利用率0/0%、现场98/98%，符合短时计算采样差异，不能以单次0%判为失效；矩阵利用率不当作训练吞吐。主机已用约65GiB（上轮53GiB），仍可用约1.9TiB，无swap；未据此归因于本run或操作其他任务。数据盘可用约564.02TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 15:02:23 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T150137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮约30分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T150223Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度1.64/1.66秒，matrix_work=true；近间隔CPU分别增加877.41/877.51秒。快照利用率0/0%、现场98/97%，符合短时计算采样差异，不能以单次0%判为失效；矩阵利用率不当作训练吞吐。主机已用约65GiB，仍可用约1.9TiB，无swap；数据盘可用约563.94TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 15:33:59 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T153137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮约31.6分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T153359Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度1.59/1.70秒，matrix_work=true；近间隔CPU分别增加919.45/919.57秒。快照利用率98/98%、现场98/97%，矩阵利用率不当作训练吞吐。主机已用约71GiB（上轮65GiB），仍可用约1.9TiB，无swap；未据此归因于本run或操作其他任务。数据盘可用约563.87TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 16:02:37 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T160137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮约28.6分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T160237Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度1.36/0.57秒，matrix_work=true；近间隔CPU分别增加832.23/831.81秒。快照利用率0/98%、现场0/97%，结合状态刷新及CPU增量，单次0%不代表失效；矩阵利用率不当作训练吞吐。主机已用约72GiB，仍可用约1.9TiB，无swap；数据盘可用约563.80TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 16:32:22 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T163137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮约29.8分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T163222Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度0.40/0.75秒，matrix_work=true；近间隔CPU分别增加865.09/864.44秒。快照利用率0/4%、现场0/0%，结合状态刷新及CPU增量符合短时计算间隙，不能以单次0%判为失效；矩阵利用率不当作训练吞吐。主机已用约72GiB，仍可用约1.9TiB，无swap；数据盘可用约563.60TiB。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 17:02:16 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T170137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮约29.9分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T170216Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度1.62/1.69秒，matrix_work=true；近间隔CPU分别增加870.10/870.13秒。快照利用率3/0%、现场98/97%，符合短时计算采样差异，不能以单次0%判为失效；矩阵利用率不当作训练吞吐。主机已用约78GiB（上轮72GiB），仍可用约1.9TiB，无swap；数据盘可用约563.25TiB。资源变化未归因于本run，未操作其他任务。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 17:32:37 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T173137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮约30.4分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T173237Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度0.50/0.05秒，matrix_work=true；近间隔CPU分别增加884.07/883.96秒。快照利用率98/97%、现场0/32%，符合短时计算采样差异，不能以单次0%判为失效；矩阵利用率不当作训练吞吐。主机已用约78GiB，仍可用约1.9TiB，无swap；数据盘可用约562.86TiB。资源变化未归因于本run，未操作其他任务。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 18:02:36 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T180137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮约30分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T180236Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度1.37/1.31秒，matrix_work=true；近间隔CPU分别增加872.89/872.72秒。快照利用率98/98%、现场98/97%，矩阵利用率不当作训练吞吐。主机已用约60GiB，仍可用约1.9TiB，无swap；数据盘可用约562.58TiB。资源变化未归因于本run，未操作其他任务。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 18:32:36 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T183137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮约30分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T183236Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度0.51/1.03秒，matrix_work=true；近间隔CPU分别增加874.21/874.06秒。快照利用率39/98%、现场0/0%，结合状态刷新及CPU增量符合短时计算间隙，不能以单次0%判为失效；矩阵利用率不当作训练吞吐。主机已用约60GiB，仍可用约1.9TiB，无swap；数据盘可用约562.52TiB。资源变化未归因于本run，未操作其他任务。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 19:02:53 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T190137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮约30.3分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T190253Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度0.58/0.47秒，matrix_work=true；近间隔CPU分别增加866.40/866.31秒。快照利用率56/26%、现场0/0%，结合状态刷新及CPU增量符合短时计算间隙，不能以单次0%判为失效；矩阵利用率不当作训练吞吐。主机已用约54GiB，仍可用约1.9TiB，无swap；数据盘可用约564.83TiB（快照约563.66TiB）。资源变化未归因于本run，未操作其他任务。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 19:33:54 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T193137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮约31分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T193354Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度0.62/0.16秒，matrix_work=true；近间隔CPU分别增加902.38/902.37秒。快照利用率98/98%、现场0/19%，结合状态刷新及CPU增量符合短时计算采样差异，不能以单次0%判为失效；矩阵利用率不当作训练吞吐。主机已用约55GiB，仍可用约1.9TiB，无swap；数据盘可用约564.71TiB。资源变化未归因于本run，未操作其他任务。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 20:02:26 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T200137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮约28.5分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T200226Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度0.22/2.00秒，matrix_work=true；近间隔CPU分别增加830.26/830.24秒。快照利用率0/0%、现场0/65%，结合状态刷新及CPU增量符合短时计算采样差异，不能以单次0%判为失效；矩阵利用率不当作训练吞吐。主机已用约55GiB，仍可用约1.9TiB，无swap；数据盘可用约564.53TiB。资源变化未归因于本run，未操作其他任务。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 20:32:54 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T203137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮约30.5分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T203254Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度1.54/1.07秒，matrix_work=true；近间隔CPU分别增加886.86/886.45秒。快照利用率0/67%、现场98/26%，符合短时计算采样差异，不能以单次0%判为失效；矩阵利用率不当作训练吞吐。主机已用约52GiB，仍可用约1.9TiB，无swap；数据盘可用约564.42TiB。资源变化未归因于本run，未操作其他任务。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 21:02:18 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T210137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮约29.4分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T210218Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度1.91/1.80秒，matrix_work=true；近间隔CPU分别增加856.58/856.97秒。快照利用率94/98%、现场98/98%，矩阵利用率不当作训练吞吐。主机已用约51GiB，仍可用约1.9TiB，无swap；数据盘可用约564.39TiB。资源变化未归因于本run，未操作其他任务。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 21:32:34 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定`20260914T213137Z.json`及最近记录；manual.active=false，卡数仍pending，无新指令，距上轮约30.3分钟。证据：`runs/emilia-full-token-scratch/supervision/20260914T213234Z-inspection.json`。
- 数据/配方：数据完成/发布文件均存在，codec complete；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。完成报告、train/val大小与mtime、阶段和worker状态均未变；沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val中英文各256条；合格dialogue614,525条、跳过重复ID14,117个。编码新增0，旧rank2/3不计本次增量。质量/导出/codec配方不变，评分源码SHA256匹配；ASMR排除、固定质量门槛及short→long→合格dialogue短句ID去重、无配额保持。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec supervisor55543及worker55576/55577均已退出，准备tmux不存在；完成报告与阶段状态一致，无需恢复。独立父进程退出码未留存，不能声称核实exit_code=0。三个恢复日志、当前rank0/1日志和矩阵日志内容及元数据未变，未发现Traceback/Error/Exception/OOM/NaN/Inf。无总入口父进程符合独立恢复后完成的状态。
- GPU/矩阵/资源：仍仅两张A800，各557MiB；矩阵主PID4832/start_ticks155946508、worker6667/6668的start_ticks155947569/155947570、cmdline和父PID均匹配，tmux存活。各worker显存548MiB，状态新鲜度1.65/1.34秒，matrix_work=true；近间隔CPU分别增加881.93/880.82秒。快照利用率0/0%、现场98/98%，符合短时计算采样差异，不能以单次0%判为失效；矩阵利用率不当作训练吞吐。主机已用约52GiB，仍可用约1.9TiB，无swap；数据盘可用约564.34TiB。资源变化未归因于本run，未操作其他任务。
- 判断/动作/剩余：training-process、training-exit、training-plan、train.log、checkpoints均不存在，正式loss/grad/LR、吞吐/data_wait、val和生成指标不适用；未写startup/final-verification。数据完成状态保持、矩阵持续工作，无需修复或重启。本轮仅保存证据并追加记录，未修改代码/配置/数据/任务状态或启停进程。仍等待训练卡数答复；未答复且仅两卡时不启动训练。若四卡恢复且无相反指令，按原授权从原始assembled以token loss、中英时长1:1完整平衡epoch执行，实际步数按最终拓扑和动态预算计算；容量测试/训练前先停止矩阵并确认显存释放。保留现有监督，本次巡检结束。


## 2026-09-14 22:03:28 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。manual.active=false，pipeline和task-state未变，卡数仍pending。核对指定快照`20260914T220137Z.json`及现场，证据为`runs/emilia-full-token-scratch/supervision/20260914T220328Z-inspection.json`。
- 数据：质量/导出/codec完成状态、完成报告和train/val大小及mtime均未变；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。沿用9月13日12:02全量SHA256及行数核验，本轮未重哈希大清单。已发布train 23,931,105条、43,744.34小时，val 512条；评分源码哈希匹配。ASMR排除、固定质量门槛和短句ID去重配方保持；音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec55543和worker55576/55577均不在，准备已完成无需恢复；独立父进程退出码未留存，不能认定exit_code=0。恢复日志和当前rank0/1、矩阵日志大小及mtime未变，错误扫描无匹配。旧rank2/3不计本次增量，编码新增0。
- GPU/资源：仍为两张A800，各557MiB；矩阵主4832及worker6667/6668的start_ticks、cmdline、父PID均匹配，tmux存活。状态新鲜度1.20/0.63秒，matrix_work=true，两worker较上轮CPU各增加899.86/899.98秒，各占548MiB。现场利用率74/0%符合短时计算采样，结合状态和CPU增量无失效证据。主机已用52GiB、可用1.9TiB，无swap；数据盘可用577,793.44GiB。
- 判断/动作/剩余：训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐、val和生成指标不适用；未写startup/final-verification。仅保存证据和追加文档，未启停进程或修改代码/配置/数据。继续等待卡数答复，仅两卡且未答复时不启动训练；四卡恢复且无相反指令时按原授权执行，容量测试/训练前先停止矩阵并确认显存释放。本次巡检结束。


## 2026-09-14 22:32:51 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。已先读manual、pipeline、task-state、指定快照及最近记录；manual.active=false，卡数仍pending，无新指令。证据：`runs/emilia-full-token-scratch/supervision/20260914T223251Z-inspection.json`。
- 数据/配方：质量/导出/codec完成状态、完成报告、worker状态及配方均未变；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。train/val大小与mtime未变，沿用9月13日12:02全量SHA256和行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val 512条；ASMR排除、固定质量门槛、short→long→合格dialogue短句ID去重及无配额保持；评分源码哈希匹配。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec55543和worker55576/55577均不在；完成文件及状态一致，无需恢复。独立父进程退出码未留存，不能声称exit_code=0。恢复日志、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。各rank状态未变，编码新增0，旧rank2/3仅作为历史。
- GPU/资源：仍两张A800，各557MiB。矩阵主4832和worker6667/6668的start_ticks、cmdline及父PID均匹配，tmux存活；状态新鲜度0.25/0.01秒，matrix_work=true，近29.4分钟两worker CPU分别增加853.05/852.23秒，各占548MiB。现场利用率0/2%与短时计算采样相符，结合状态刷新和CPU增量，无失效证据。主机已用52GiB、可用1.9TiB，无swap；数据盘可用577,739.86GiB。
- 判断/动作/剩余：训练进程/退出记录、计划、日志、checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用，未写startup/final-verification。数据完成状态保持、矩阵持续运行，本轮仅保存证据并追加文档，未修改代码/配置/数据或启停进程。仍待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权执行。容量测试/训练前先停止矩阵并确认显存释放。本次巡检结束。


## 2026-09-14 23:02:49 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定快照和最近记录；manual.active=false，卡数仍pending，无新指令。证据：`runs/emilia-full-token-scratch/supervision/20260914T230249Z-inspection.json`。
- 数据/配方：质量/导出/codec完成状态、完成报告、worker状态、配方均未变；raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。train/val大小与mtime未变，沿用9月13日12:02全量SHA256及行数核验，本轮未重新哈希大清单。train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定质量门槛、short→long→合格dialogue短句ID去重和无配额保持，评分源码哈希匹配。音质筛选不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec55543及worker55576/55577均不在，完成报告与状态一致，无需恢复。独立父进程退出码未留存，不能声称exit_code=0。恢复日志、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。worker状态无变化，编码新增0，旧rank2/3仅作历史。
- GPU/资源：仍两张A800，各557MiB。矩阵主4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度0.56/0.04秒，matrix_work=true；近30分钟两worker CPU分别增加873.22/873.26秒，各占548MiB。现场利用率0/49%，结合状态刷新及CPU增量符合短时计算采样，无失效证据。主机已用52GiB、可用1.9TiB，无swap；数据盘可用577,755.88GiB。
- 判断/动作/剩余：训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait和评估指标不适用；未写startup/final-verification。数据完成状态保持，矩阵持续运行，本轮仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权执行。容量测试/训练前先停止矩阵并确认显存释放。本次巡检结束。


## 2026-09-14 23:33:01 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。已先读manual、pipeline、task-state、指定快照和最近记录；manual.active=false，卡数仍pending。证据：`runs/emilia-full-token-scratch/supervision/20260914T233301Z-inspection.json`。
- 数据/配方：质量/导出/codec完成状态、完成报告、各worker状态和配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。train/val大小及mtime未变，沿用9月13日12:02全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条；ASMR排除、固定音质门槛和short→long→合格dialogue短句ID去重、无配额保持，评分源码哈希匹配。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec55543及worker55576/55577均不在，完成文件与状态一致，无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复日志、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。编码新增0，旧rank2/3只作历史。
- GPU/资源：仍两张A800，各557MiB；矩阵主4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度0.18/1.76秒，matrix_work=true；近30.2分钟worker CPU分别增加880.25/879.96秒，各占548MiB。现场利用率0/98%，结合状态刷新和CPU增量符合短时计算采样，无失效证据。主机已用52GiB、可用1.9TiB，无swap；数据盘可用577,741.32GiB。
- 判断/动作/剩余：训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持，矩阵持续运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权执行。容量测试/训练前先停止矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 00:02:18 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定快照和最近记录；manual.active=false，卡数仍pending。证据：`runs/emilia-full-token-scratch/supervision/20260915T000218Z-inspection.json`。
- 数据/配方：质量/导出/codec完成状态、完成报告、各worker状态和配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。train/val大小及mtime未变，沿用9月13日12:02全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条；ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重及无配额保持，评分源码哈希匹配。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec55543及worker55576/55577均不在，完成文件与状态一致，无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复日志、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。编码新增0，旧rank2/3仅作历史。
- GPU/资源：仍两张A800，各557MiB。矩阵主4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度1.27/0.48秒，matrix_work=true；近29.3分钟worker CPU分别增加853.42/852.54秒，各占548MiB。现场利用率98/0%，结合状态刷新和CPU增量符合短时计算采样，无失效证据。主机已用52GiB、可用1.9TiB，无swap；数据盘可用577,681.99GiB。
- 判断/动作/剩余：训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持，矩阵持续运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权执行。容量测试/训练前先停止矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 00:32:32 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定快照和最近记录；manual.active=false，卡数仍pending。证据：`runs/emilia-full-token-scratch/supervision/20260915T003232Z-inspection.json`。
- 数据/配方：质量/导出/codec完成状态、完成报告、worker状态及配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。train/val大小及mtime未变，沿用9月13日12:02全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条；ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重和无配额保持，评分源码哈希匹配。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec55543及worker55576/55577均不在，完成文件与状态一致，无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复日志、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。编码新增0，旧rank2/3仅作历史。
- GPU/资源：仍两张A800，各557MiB。矩阵主4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度2.06/1.52秒，matrix_work=true；近30.2分钟worker CPU分别增加880.98/880.74秒，各占548MiB。现场利用率85/98%，状态刷新和CPU增量均支持持续运行。主机已用52GiB、可用1.9TiB，无swap；数据盘可用577,668.13GiB。
- 判断/动作/剩余：训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持，矩阵持续运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权执行。容量测试/训练前先停止矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 01:02:18 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定快照和最近记录；manual.active=false，卡数仍pending。证据：`runs/emilia-full-token-scratch/supervision/20260915T010218Z-inspection.json`。
- 数据/配方：质量/导出/codec完成状态、完成报告、各worker状态和配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。train/val大小及mtime未变，沿用9月13日12:02全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条；ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重及无配额保持，评分源码哈希匹配。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec55543及worker55576/55577均不在，完成文件与状态一致，无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复日志、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。编码新增0，旧rank2/3仅作历史。
- GPU/资源：仍两张A800，各557MiB。矩阵主4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度0.13/1.85秒，matrix_work=true；近29.8分钟worker CPU分别增加867.23/867.57秒，各占548MiB。现场利用率36/98%，状态刷新和CPU增量均支持持续运行。主机已用52GiB、可用1.9TiB，无swap；数据盘可用577,403.59GiB。磁盘可用量变化未归因于本run，未操作其他数据。
- 判断/动作/剩余：训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持，矩阵持续运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权执行。容量测试/训练前先停止矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 01:33:58 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定快照和最近记录；manual.active=false，卡数仍pending。证据：`runs/emilia-full-token-scratch/supervision/20260915T013358Z-inspection.json`。
- 数据/配方：质量/导出/codec完成状态、完成报告、各worker状态和配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。train/val大小及mtime未变，沿用9月13日12:02全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条；ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重及无配额保持，评分源码哈希匹配。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec55543及worker55576/55577均不在，完成文件与状态一致，无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复日志、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。编码新增0，旧rank2/3仅作历史。
- GPU/资源：仍两张A800，各557MiB。矩阵主4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度0.71/0.59秒，matrix_work=true；近31.7分钟worker CPU分别增加922.29/922.54秒，各占548MiB。现场利用率0/0%，结合状态刷新与CPU增量符合短时计算采样，不能据单次0%判为失效。主机已用45GiB、可用1.9TiB，无swap；数据盘可用576,955.67GiB。资源变化未归因于本run，未操作其他任务或数据。
- 判断/动作/剩余：训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持，矩阵持续运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权执行。容量测试/训练前先停止矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 02:02:59 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定快照和最近记录；manual.active=false，卡数仍pending。证据：`runs/emilia-full-token-scratch/supervision/20260915T020259Z-inspection.json`。
- 数据/配方：质量/导出/codec完成状态、完成报告、worker状态及配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。train/val大小及mtime未变，沿用9月13日12:02全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条；ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重及无配额保持，评分源码哈希匹配。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec55543及worker55576/55577均不在，完成文件与状态一致，无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复日志、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。编码新增0，旧rank2/3仅作历史。
- GPU/资源：仍两张A800，各557MiB。矩阵主4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度1.81/0.77秒，matrix_work=true；近29分钟worker CPU分别增加846.95/846.16秒，各占548MiB。现场利用率75/0%，结合状态刷新与CPU增量符合短时计算采样，无失效证据。主机已用59GiB、可用1.9TiB，无swap；数据盘可用576,622.87GiB。资源变化未归因于本run，未操作其他任务或数据。
- 判断/动作/剩余：训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持，矩阵持续运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权执行。容量测试/训练前先停止矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 02:33:08 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定快照和最近记录；manual.active=false，卡数仍pending。证据：`runs/emilia-full-token-scratch/supervision/20260915T023308Z-inspection.json`。
- 数据/配方：质量/导出/codec完成状态、完成报告、各worker状态和配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。train/val大小及mtime未变，沿用9月13日12:02全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条；ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重及无配额保持，评分源码哈希匹配。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec55543及worker55576/55577均不在，完成文件与状态一致，无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复日志、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。编码新增0，旧rank2/3仅作历史。
- GPU/资源：仍两张A800，各557MiB。矩阵主4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度1.36/1.40秒，matrix_work=true；近30.2分钟worker CPU分别增加879.74/879.61秒，各占548MiB。现场利用率97/98%，状态刷新和CPU增量均支持持续运行。主机已用61GiB、可用1.9TiB，无swap；数据盘可用576,297.21GiB。资源变化未归因于本run，未操作其他任务或数据。
- 判断/动作/剩余：训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持，矩阵持续运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权执行。容量测试/训练前先停止矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 03:02:17 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定快照和最近记录；manual.active=false，卡数仍pending。证据：`runs/emilia-full-token-scratch/supervision/20260915T030217Z-inspection.json`。
- 数据/配方：质量/导出/codec完成状态、完成报告、各worker状态和配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。train/val大小及mtime未变，沿用9月13日12:02全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条；ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重及无配额保持，评分源码哈希匹配。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec55543及worker55576/55577均不在，完成文件与状态一致，无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复日志、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。编码新增0，旧rank2/3仅作历史。
- GPU/资源：仍两张A800，各557MiB。矩阵主4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度1.86/1.66秒，matrix_work=true；近29.2分钟worker CPU分别增加850.75/850.38秒，各占548MiB。现场利用率97/97%，状态刷新和CPU增量均支持持续运行。主机已用62GiB、可用1.9TiB，无swap；数据盘可用575,984.28GiB。资源变化未归因于本run，未操作其他任务或数据。
- 判断/动作/剩余：训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持，矩阵持续运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权执行。容量测试/训练前先停止矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 03:33:25 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定快照和最近记录；manual.active=false，卡数仍pending。证据：`runs/emilia-full-token-scratch/supervision/20260915T033325Z-inspection.json`。
- 数据/配方：质量/导出/codec完成状态、完成报告、各worker状态和配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。train/val大小及mtime未变，沿用9月13日12:02全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条；ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重及无配额保持，评分源码哈希匹配。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec55543及worker55576/55577均不在，完成文件与状态一致，无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复日志、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。编码新增0，旧rank2/3仅作历史。
- GPU/资源：仍两张A800，各557MiB。矩阵主4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度1.36/0.21秒，matrix_work=true；近31.1分钟worker CPU分别增加907.79/906.77秒，各占548MiB。现场利用率97/0%，结合状态刷新和CPU增量符合短时计算采样，无失效证据。主机已用63GiB、可用1.9TiB，无swap；数据盘可用575,539.99GiB。资源变化未归因于本run，未操作其他任务或数据。
- 判断/动作/剩余：训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持，矩阵持续运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权执行。容量测试/训练前先停止矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 04:02:33 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定快照和最近记录；manual.active=false，卡数仍pending。证据：`runs/emilia-full-token-scratch/supervision/20260915T040233Z-inspection.json`。
- 数据/配方：质量/导出/codec完成状态、完成报告、各worker状态和配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。train/val大小及mtime未变，沿用9月13日12:02全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条；ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重及无配额保持，评分源码哈希匹配。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec55543及worker55576/55577均不在，完成文件与状态一致，无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复日志、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。编码新增0，旧rank2/3仅作历史。
- GPU/资源：仍两张A800，各557MiB。矩阵主4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度1.27/1.38秒，matrix_work=true；近29.1分钟worker CPU分别增加850.15/849.52秒，各占548MiB。现场利用率97/98%，状态刷新和CPU增量均支持持续运行。主机已用63GiB、可用1.9TiB，无swap；数据盘可用575,150.70GiB。资源变化未归因于本run，未操作其他任务或数据。
- 判断/动作/剩余：训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持，矩阵持续运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权执行。容量测试/训练前先停止矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 04:32:31 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定快照和最近记录；manual.active=false，卡数仍pending。证据：`runs/emilia-full-token-scratch/supervision/20260915T043231Z-inspection.json`。
- 数据/配方：质量/导出/codec完成状态、完成报告、各worker状态和配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。train/val大小及mtime未变，沿用9月13日12:02全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条；ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重及无配额保持，评分源码哈希匹配。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 进程/日志：quality1935、export55462、codec55543及worker55576/55577均不在，完成文件与状态一致，无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复日志、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。编码新增0，旧rank2/3仅作历史。
- GPU/资源：仍两张A800，各557MiB。矩阵主4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度0.69/1.51秒，matrix_work=true；近30分钟worker CPU分别增加875.00/874.40秒，各占548MiB。现场利用率0/97%，结合状态刷新与CPU增量符合短时计算采样，无失效证据。主机已用64GiB、可用1.9TiB，无swap；数据盘可用574,817.06GiB。资源变化未归因于本run，未操作其他任务或数据。
- 判断/动作/剩余：训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持，矩阵持续运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权执行。容量测试/训练前先停止矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 05:03:15 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定快照和最近记录；manual.active=false，卡数仍pending。证据：`runs/emilia-full-token-scratch/supervision/20260915T050315Z-inspection.json`。
- 数据/配方：质量/导出/codec完成状态、完成报告、各worker状态和配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。train/val大小及mtime未变，沿用9月13日12:02全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条；ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重及无配额保持，评分源码哈希匹配。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 准备进程/日志：quality1935、export55462、codec55543及worker55576/55577均不在，完成文件与状态一致，无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复日志、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。编码新增0，旧rank2/3仅作历史。
- GPU/矩阵变化：仍两张A800，但新增GPU进程2776122/2776123（start_ticks均171976382，父PID2776108）。只读核对cmdline、cwd与output-dir表明属于`/119010446/LITs-distill`的`majestic100h_meanflow_distill_t16_s2_20260915`训练，不属于本run；未进一步审计或操作该任务。现场两GPU总显存16685/15143MiB、利用率92/93%，新增进程占16122/14580MiB。矩阵主4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活，各worker仍占548MiB。状态新鲜度0.77/0.74秒，observed_utilization92/93、matrix_work=false，符合GPU繁忙时自动暂停计算；近30.7分钟CPU增加734.34/732.90秒包含此前空闲计算，不能当作当前持续矩阵计算证据。无需重启矩阵或强制计算。
- 资源：主机已用70GiB、可用1.9TiB，无swap；数据盘可用574,247.60GiB。资源变化未归因于Emilia，未操作其他任务或数据。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持，矩阵按设计待机，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场GPU占用执行。容量测试/训练前先停止本run矩阵并确认显存释放，不得干预其他项目进程。本次巡检结束。


## 2026-09-15 05:32:52 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定快照和最近记录；manual.active=false，卡数仍pending。证据：`runs/emilia-full-token-scratch/supervision/20260915T053252Z-inspection.json`。
- 数据/配方：质量/导出/codec完成状态、完成报告、各worker状态和配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。train/val大小及mtime未变，沿用9月13日12:02全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条；ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重及无配额保持，评分源码哈希匹配。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 准备进程/日志：quality1935、export55462、codec55543及worker55576/55577均不在，完成文件与状态一致，无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复日志、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。编码新增0，旧rank2/3仅作历史。
- GPU/矩阵：仍两张A800。其他项目LITs-distill的GPU进程2776122/2776123，start_ticks171976382、父PID2776108、cmdline/cwd与上轮匹配，不属于Emilia；未审计或干预该任务。现场总显存17567/15263MiB、利用率87/94%，该任务占17004/14700MiB。矩阵主4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活，各worker仍占548MiB。状态新鲜度0.45/0.46秒，observed_utilization91/92、matrix_work=false；近29.6分钟CPU仅增3.76/1.75秒，符合GPU繁忙时暂停计算并继续监测，无需重启。
- 资源：主机已用70GiB、可用1.9TiB，无swap；数据盘可用573,622.80GiB。资源变化未归因于Emilia，未操作其他任务或数据。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常待机，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放，不得干预其他项目进程。本次巡检结束。


## 2026-09-15 06:02:25 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定快照和最近记录；manual.active=false，卡数仍pending。证据：`runs/emilia-full-token-scratch/supervision/20260915T060225Z-inspection.json`。
- 数据/配方：质量/导出/codec完成状态、完成报告、各worker状态和配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。train/val大小及mtime未变，沿用9月13日12:02全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条；ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重及无配额保持，评分源码哈希匹配。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 准备进程/日志：quality1935、export55462、codec55543及worker55576/55577均不在，完成文件与状态一致，无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复日志、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。编码新增0，旧rank2/3仅作历史。
- GPU/矩阵：仍两张A800。其他项目LITs-distill的GPU进程2776122/2776123，start_ticks171976382、父PID2776108、cmdline/cwd与上轮匹配，不属于Emilia；未审计或干预该任务。现场总显存17567/15319MiB、利用率80/92%，该任务占17004/14756MiB。矩阵主4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活，各worker仍占548MiB。状态新鲜度0.04/0.15秒，observed_utilization95/94、matrix_work=false；近29.6分钟CPU仅增3.75/1.73秒，符合GPU繁忙时暂停计算并继续监测，无需重启。
- 资源：主机已用70GiB、可用1.9TiB，无swap；数据盘可用573,045.59GiB。资源变化未归因于Emilia，未操作其他任务或数据。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常待机，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放，不得干预其他项目进程。本次巡检结束。


## 2026-09-15 06:32:51 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定快照和最近记录；manual.active=false，卡数仍pending。证据：`runs/emilia-full-token-scratch/supervision/20260915T063251Z-inspection.json`。
- 数据/配方：质量/导出/codec完成状态、完成报告、各worker状态和配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。train/val大小及mtime未变，沿用9月13日12:02全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条；ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重及无配额保持，评分源码哈希匹配。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 准备进程/日志：quality1935、export55462、codec55543及worker55576/55577均不在，完成文件与状态一致，无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复日志、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。编码新增0，旧rank2/3仅作历史。
- GPU/矩阵变化：仍两张A800，各557MiB；上轮其他项目GPU进程2776122/2776123已不在/proc，nvidia-smi仅见本run矩阵worker。未审计其他任务退出原因或结果。矩阵主4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活，各worker占548MiB。状态新鲜度均0.33秒，observed_utilization均0、matrix_work=true，表明GPU空闲后自动恢复计算；近30.4分钟CPU增加675.34/631.35秒。现场利用率0/0%，结合新鲜状态和CPU增量符合短时计算采样，不据单次0%判为失效，无需重启。
- 资源：主机已用64GiB、可用1.9TiB，无swap；数据盘可用572,498.12GiB。资源变化未归因于Emilia，未操作其他任务或数据。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵自动恢复空闲计算，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放，不得干预其他项目进程。本次巡检结束。


## 2026-09-15 07:02:16 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定快照和最近记录；manual.active=false，卡数仍pending。证据：`runs/emilia-full-token-scratch/supervision/20260915T070216Z-inspection.json`。
- 数据/配方：质量/导出/codec完成状态、完成报告、各worker状态和配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。train/val大小及mtime未变，沿用9月13日12:02全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条；ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重及无配额保持，评分源码哈希匹配。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 准备进程/日志：quality1935、export55462、codec55543及worker55576/55577均不在，完成文件与状态一致，无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复日志、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。编码新增0，旧rank2/3仅作历史。
- GPU/矩阵变化：仍两张A800，新增GPU进程2802858/2802859（start_ticks172587064，父PID2802843）。只读cmdline/cwd/output-dir表明属于`/119010446/LITs-distill-streaming`的`majestic100h_meanflow_streaming_t16_s2_20260915`训练，不属于Emilia；未审计或干预该任务。现场总显存14875/13137MiB、利用率32/31%，该任务占14312/12574MiB。矩阵主4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活，各worker占548MiB。状态新鲜度1.00/1.01秒，observed_utilization83/46、matrix_work=false，符合GPU繁忙时暂停计算。近29.4分钟CPU增加192.57/192.54秒，包含此前空闲计算，不作为当前持续计算证据。无需重启。
- 资源：主机已用70GiB、可用1.9TiB，无swap；数据盘可用571,824.45GiB。资源变化未归因于Emilia，未操作其他任务或数据。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常待机，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放，不得干预其他项目进程。本次巡检结束。


## 2026-09-15 07:32:21 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定快照和最近记录；manual.active=false，卡数仍pending。证据：`runs/emilia-full-token-scratch/supervision/20260915T073221Z-inspection.json`。
- 数据/配方：质量/导出/codec完成状态、完成报告、各worker状态和配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。train/val大小及mtime未变，沿用9月13日12:02全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条；ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重及无配额保持，评分源码哈希匹配。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 准备进程/日志：quality1935、export55462、codec55543及worker55576/55577均不在，完成文件与状态一致，无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复日志、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。编码新增0，旧rank2/3仅作历史。
- GPU/矩阵：仍两张A800。其他项目LITs-distill-streaming的GPU进程2802858/2802859，start_ticks172587064、父PID2802843、cmdline/cwd与上轮匹配，不属于Emilia；未审计或干预该任务。现场总显存14875/13943MiB、利用率44/44%，该任务占14312/13380MiB。矩阵主4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活，各worker占548MiB。状态新鲜度0.64/0.93秒，observed_utilization31/32、matrix_work=false；近30.1分钟CPU仅增2.83/3.83秒，符合GPU繁忙时暂停计算并继续监测，无需重启。
- 资源：主机已用84GiB、可用1.9TiB，无swap；数据盘可用571,298.36GiB。资源变化未归因于Emilia，未操作其他任务或数据。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常待机，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放，不得干预其他项目进程。本次巡检结束。


## 2026-09-15 08:02:52 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定快照和最近记录；manual.active=false，卡数仍pending。证据：`runs/emilia-full-token-scratch/supervision/20260915T080252Z-inspection.json`。
- 数据/配方：质量/导出/codec完成状态、完成报告、各worker状态和配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。train/val大小及mtime未变，沿用9月13日12:02全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条；ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重及无配额保持，评分源码哈希匹配。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 准备进程/日志：quality1935、export55462、codec55543及worker55576/55577均不在，完成文件与状态一致，无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复日志、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。编码新增0，旧rank2/3仅作历史。
- GPU/矩阵：仍两张A800。其他项目LITs-distill-streaming的GPU进程2802858/2802859，start_ticks172587064、父PID2802843、cmdline/cwd与上轮匹配，不属于Emilia；未审计或干预该任务，不据单次GPU低利用率判断其卡死。现场总显存14879/13943MiB、利用率58/0%，该任务仍占14316/13380MiB。矩阵主4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活，各worker占548MiB。状态新鲜度1.04/0.87秒，observed_utilization均0、matrix_work=true，表明采样空闲时恢复短时计算；近30.5分钟CPU增加44.01/43.96秒。与上轮待机不同，但符合既有空闲计算逻辑，无需重启。
- 资源：主机已用91GiB、可用1.9TiB，无swap；数据盘可用570,851.01GiB。资源变化未归因于Emilia，未操作其他任务或数据。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持，矩阵正常运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放，不得干预其他项目进程。本次巡检结束。


## 2026-09-15 08:32:37 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定快照和最近记录；manual.active=false，卡数仍pending。证据：`runs/emilia-full-token-scratch/supervision/20260915T083237Z-inspection.json`。
- 数据/配方：质量/导出/codec完成状态、完成报告、各worker状态和配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。train/val大小及mtime未变，沿用9月13日12:02全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条；ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重及无配额保持，评分源码哈希匹配。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 准备进程/日志：quality1935、export55462、codec55543及worker55576/55577均不在，完成文件与状态一致，无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复日志、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。编码新增0，旧rank2/3仅作历史。
- GPU/矩阵变化：仍两张A800；上轮其他项目PID2802858/2802859已不在，当前GPU进程2828208/2828209（start_ticks173160339，父PID2828197）的cmdline/cwd表明属于`LITs-distill-streaming-performance`，output-dir为其他项目`majestic100h_meanflow_streaming_t16_s2_20260915`，不属于Emilia。仅确认归属，未审计其退出/恢复结果或干预。现场总显存18171/17383MiB、利用率94/94%，其他任务占17608/16820MiB。矩阵主4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活，各worker占548MiB。状态新鲜度0.79/0.61秒，observed_utilization82/95、matrix_work=false，符合繁忙时暂停计算；近29.8分钟CPU增加242.79/248.76秒包含此前空闲计算，不当作当前持续计算证据。无需重启。
- 资源：主机已用91GiB、可用1.9TiB，无swap；数据盘可用570,545.02GiB。资源变化未归因于Emilia，未操作其他任务或数据。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常待机，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放，不得干预其他项目进程。本次巡检结束。


## 2026-09-15 09:02:21 UTC 等待训练卡数单次巡检

- 阶段/step：`waiting/awaiting_training_topology`，正式step 0 → 0。先读manual、pipeline、task-state、指定快照和最近记录；manual.active=false，卡数仍pending。证据：`runs/emilia-full-token-scratch/supervision/20260915T090221Z-inspection.json`。
- 数据/配方：质量/导出/codec完成状态、完成报告、各worker状态和配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，无incomplete。train/val大小及mtime未变，沿用9月13日12:02全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条；ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重及无配额保持，评分源码哈希匹配。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。
- 准备进程/日志：quality1935、export55462、codec55543及worker55576/55577均不在，完成文件与状态一致，无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复日志、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。编码新增0，旧rank2/3仅作历史。
- GPU/矩阵：仍两张A800。其他项目LITs-distill-streaming-performance的GPU进程2828208/2828209，start_ticks173160339、父PID2828197、cmdline/cwd与上轮匹配，不属于Emilia；未审计或干预该任务。现场总显存20259/17439MiB、利用率90/84%，该任务占19696/16876MiB。矩阵主4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活，各worker占548MiB。状态新鲜度0.06/0.08秒，observed_utilization96/82、matrix_work=false；近29.7分钟CPU仅增4.89/3.89秒，符合GPU繁忙时暂停计算并继续监测，无需重启。
- 资源：主机已用79GiB、可用1.9TiB，无swap；数据盘可用570,353.69GiB。资源变化未归因于Emilia，未操作其他任务或数据。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常待机，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放，不得干预其他项目进程。本次巡检结束。


## 2026-09-15 09:36:46 UTC 等待训练卡数单次巡检

- 阶段/证据：`waiting/awaiting_training_topology`，step 0 → 0，manual.active=false，卡数仍pending。已核对manual、pipeline、task-state、指定09:31快照和最近记录；现场证据为`runs/emilia-full-token-scratch/supervision/20260915T093646Z-inspection.json`。
- 数据/进程：质量、导出、codec完成报告和状态、各worker状态及配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份。train/val大小和mtime未变，沿用9月13日全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重和无配额保持；评分源码哈希匹配。音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不在，阶段脚本进程扫描无匹配，完成状态无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复及rank0/1日志结束于完成统计，错误扫描无匹配。编码新增0，旧rank2/3仅作历史。
- GPU/矩阵：仍两张A800，各557MiB，现场利用率90/62%。上轮其他项目PID2828208/2828209已不在，当前GPU仅见本run矩阵worker；未审计其他任务退出原因或结果。矩阵主4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活，各worker占548MiB。状态新鲜度1.48/0.41秒，matrix_work均true；近34.4分钟CPU增加731.73/680.67秒，支持空闲计算已恢复，无需重启。主机已用74GiB、可用1.9TiB，无swap；数据盘可用570,265.99GiB。
- 判断/动作：本run训练进程/退出记录、计划、日志、checkpoint均不存在，训练和评估指标不适用；未写startup/final-verification。本次仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 10:02:18 UTC 等待训练卡数单次巡检

- 阶段/证据：`waiting/awaiting_training_topology`，step 0 → 0，manual.active=false，卡数仍pending。已核对manual、pipeline、task-state、指定10:01快照和最近记录；证据为`runs/emilia-full-token-scratch/supervision/20260915T100218Z-inspection.json`。
- 数据/进程：质量、导出、codec完成报告和状态、各worker状态及配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，编码新增0；旧rank2/3仅作历史。train/val大小和mtime未变，沿用9月13日全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重和无配额保持，评分源码哈希匹配；音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不在，阶段脚本扫描无匹配，完成状态无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。
- GPU/矩阵：仍两张A800，各557MiB，当前GPU仅见矩阵worker6667/6668，各占548MiB；父4832/start_ticks155946508及worker的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度1.17/1.03秒，matrix_work均true；近25.5分钟CPU增加743.91/743.01秒，支持矩阵持续按既有逻辑运行。现场利用率0/0%属于瞬时采样，结合新鲜状态和CPU增量不判为失效，无需重启。主机已用66GiB、可用1.9TiB，无swap；数据盘可用570,227.59GiB。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 10:32:47 UTC 等待训练卡数单次巡检

- 阶段/证据：`waiting/awaiting_training_topology`，step 0 → 0，manual.active=false，卡数仍pending。已核对manual、pipeline、task-state、指定10:31快照和最近记录；证据为`runs/emilia-full-token-scratch/supervision/20260915T103247Z-inspection.json`。
- 数据/进程：质量、导出、codec完成报告和状态、各worker状态及配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，编码新增0；旧rank2/3仅作历史。train/val大小和mtime未变，沿用9月13日全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重和无配额保持，评分源码哈希匹配；音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不在，阶段脚本扫描无匹配，完成状态无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。
- GPU/矩阵：仍两张A800，现场总显存16859/17623MiB、利用率90/99%。新增GPU进程2855484/2855485（start_ticks173901785/173901786，父2855473）和2858891（start_ticks173978359，父2857437），cmdline/cwd/output表明属于其他项目LITs-distill-teacher-matched训练和评估，不属于Emilia；仅核对归属，未审计或干预。矩阵父4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活，各worker仍占548MiB。状态新鲜度0.02/0.18秒，observed_utilization90/95、matrix_work均false，符合GPU繁忙时暂停计算并继续监测；近30.5分钟CPU增加473.46/473.43秒包含此前空闲计算，不作为当前持续计算证据。无需重启。主机已用75GiB、可用1.9TiB，无swap；数据盘可用570,115.06GiB。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常待机，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放，不干预其他项目进程。本次巡检结束。


## 2026-09-15 11:02:26 UTC 等待训练卡数单次巡检

- 阶段/证据：`waiting/awaiting_training_topology`，step 0 → 0，manual.active=false，卡数仍pending。已核对manual、pipeline、task-state、指定11:01快照和最近记录；证据为`runs/emilia-full-token-scratch/supervision/20260915T110226Z-inspection.json`。
- 数据/进程：质量、导出、codec完成报告和状态、各worker状态及配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，编码新增0；旧rank2/3仅作历史。train/val大小和mtime未变，沿用9月13日全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重和无配额保持，评分源码哈希匹配；音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不在，阶段脚本扫描无匹配，完成状态无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。
- GPU/矩阵：仍两张A800，现场总显存17561/15275MiB、利用率88/92%。其他项目LITs-distill-teacher-matched训练进程2855484/2855485的start_ticks173901785/173901786、父2855473、cmdline/cwd与上轮匹配，不属于Emilia，未审计或干预；上轮评估PID2858891不再出现在本次GPU进程列表，不据此判定其评估结果。矩阵父4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活，各worker占548MiB。状态新鲜度0.05/0.98秒，observed_utilization均88、matrix_work均false；近29.7分钟CPU仅增4.74/1.71秒，符合GPU繁忙时暂停计算并继续监测，无需重启。主机已用72GiB、可用1.9TiB，无swap；数据盘可用570,009.86GiB。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常待机，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放，不干预其他项目进程。本次巡检结束。


## 2026-09-15 11:32:22 UTC 等待训练卡数单次巡检

- 阶段/证据：`waiting/awaiting_training_topology`，step 0 → 0，manual.active=false，卡数仍pending。已核对manual、pipeline、task-state、指定11:31快照和最近记录；证据为`runs/emilia-full-token-scratch/supervision/20260915T113222Z-inspection.json`。
- 数据/进程：质量、导出、codec完成报告和状态、各worker状态及配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，编码新增0；旧rank2/3仅作历史。train/val大小和mtime未变，沿用9月13日全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重和无配额保持，评分源码哈希匹配；音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不在，阶段脚本扫描无匹配，完成状态无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。
- GPU/矩阵：仍两张A800，现场总显存17561/15275MiB、利用率91/92%。其他项目LITs-distill-teacher-matched训练进程2855484/2855485的start_ticks173901785/173901786、父2855473、cmdline/cwd与上轮匹配，不属于Emilia，未审计或干预。矩阵父4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活，各worker占548MiB。状态新鲜度0.52/0.50秒，observed_utilization91/92、matrix_work均false；近29.9分钟CPU仅增2.73/1.73秒，符合GPU繁忙时暂停计算并继续监测，无需重启。主机已用72GiB、可用1.9TiB，无swap；数据盘可用569,976.60GiB。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常待机，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放，不干预其他项目进程。本次巡检结束。


## 2026-09-15 12:02:28 UTC 等待训练卡数单次巡检

- 阶段/证据：`waiting/awaiting_training_topology`，step 0 → 0，manual.active=false，卡数仍pending。已核对manual、pipeline、task-state、指定12:01快照和最近记录；证据为`runs/emilia-full-token-scratch/supervision/20260915T120228Z-inspection.json`。
- 数据/进程：质量、导出、codec完成报告和状态、各worker状态及配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，编码新增0；旧rank2/3仅作历史。train/val大小和mtime未变，沿用9月13日全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重和无配额保持，评分源码哈希匹配；音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不在，阶段脚本扫描无匹配，完成状态无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。
- GPU/矩阵：仍两张A800，各557MiB；上轮其他项目PID2855484/2855485已不在/proc，当前GPU仅见本run矩阵worker，未审计其他任务退出原因或结果。矩阵父4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活，各worker占548MiB。状态新鲜度1.12/1.19秒，observed_utilization均0、matrix_work均true，表明GPU空闲后恢复计算；近30.1分钟CPU增加794.30/721.40秒。现场利用率0/12%，结合新鲜状态和CPU增量符合短时计算采样，不判为失效，无需重启。主机已用66GiB、可用1.9TiB，无swap；数据盘可用569,915.64GiB。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 12:32:21 UTC 等待训练卡数单次巡检

- 阶段/证据：`waiting/awaiting_training_topology`，step 0 → 0，manual.active=false，卡数仍pending。已核对manual、pipeline、task-state、指定12:31快照和最近记录；证据为`runs/emilia-full-token-scratch/supervision/20260915T123221Z-inspection.json`。
- 数据/进程：质量、导出、codec完成报告和状态、各worker状态及配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，编码新增0；旧rank2/3仅作历史。train/val大小和mtime未变，沿用9月13日全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重和无配额保持，评分源码哈希匹配；音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不在，阶段脚本扫描无匹配，完成状态无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。
- GPU/矩阵：仍两张A800，各557MiB，利用率97/97%；GPU仅见本run矩阵worker，各占548MiB。矩阵父4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度1.94/1.85秒，observed_utilization均0、matrix_work均true；近29.9分钟CPU增加872.84/871.73秒，支持矩阵持续按空闲计算逻辑运行，无需重启。主机已用66GiB、可用1.9TiB，无swap；数据盘可用569,896.67GiB。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 13:03:02 UTC 等待训练卡数单次巡检

- 阶段/证据：`waiting/awaiting_training_topology`，step 0 → 0，manual.active=false，卡数仍pending。已核对manual、pipeline、task-state、指定13:01快照和最近记录；证据为`runs/emilia-full-token-scratch/supervision/20260915T130302Z-inspection.json`。
- 数据/进程：质量、导出、codec完成报告和状态、各worker状态及配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，编码新增0；旧rank2/3仅作历史。train/val大小和mtime未变，沿用9月13日全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重和无配额保持，评分源码哈希匹配；音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不在，阶段脚本扫描无匹配，完成状态无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。
- GPU/矩阵：仍两张A800，各557MiB，利用率97/69%；GPU仅见本run矩阵worker，各占548MiB。矩阵父4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度0.10/0.17秒，observed_utilization均0、matrix_work均true；近30.7分钟CPU增加895.62/894.70秒，支持矩阵持续按空闲计算逻辑运行，无需重启。主机已用68GiB、可用1.9TiB，无swap；数据盘可用569,865.84GiB。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 13:32:22 UTC 等待训练卡数单次巡检

- 阶段/证据：`waiting/awaiting_training_topology`，step 0 → 0，manual.active=false，卡数仍pending。已核对manual、pipeline、task-state、指定13:31快照和最近记录；证据为`runs/emilia-full-token-scratch/supervision/20260915T133222Z-inspection.json`。
- 数据/进程：质量、导出、codec完成报告和状态、各worker状态及配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，编码新增0；旧rank2/3仅作历史。train/val大小和mtime未变，沿用9月13日全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重和无配额保持，评分源码哈希匹配；音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不在，阶段脚本扫描无匹配，完成状态无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。
- GPU/矩阵：仍两张A800，各557MiB，利用率97/97%；GPU仅见本run矩阵worker，各占548MiB。矩阵父4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度1.52/1.87秒，observed_utilization均0、matrix_work均true；近29.3分钟CPU增加855.66/854.93秒，支持矩阵持续按空闲计算逻辑运行，无需重启。主机已用67GiB、可用1.9TiB，无swap；数据盘可用569,797.77GiB。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 14:02:23 UTC 等待训练卡数单次巡检

- 阶段/证据：`waiting/awaiting_training_topology`，step 0 → 0，manual.active=false，卡数仍pending。已核对manual、pipeline、task-state、指定14:01快照和最近记录；证据为`runs/emilia-full-token-scratch/supervision/20260915T140223Z-inspection.json`。
- 数据/进程：质量、导出、codec完成报告和状态、各worker状态及配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，编码新增0；旧rank2/3仅作历史。train/val大小和mtime未变，沿用9月13日全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重和无配额保持，评分源码哈希匹配；音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不在，阶段脚本扫描无匹配，完成状态无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。
- GPU/矩阵：仍两张A800，各557MiB，利用率97/98%；GPU仅见本run矩阵worker，各占548MiB。矩阵父4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度1.65/1.54秒，observed_utilization均0、matrix_work均true；近30.0分钟CPU增加874.41/873.95秒，支持矩阵持续按空闲计算逻辑运行，无需重启。主机已用73GiB、可用1.9TiB，无swap；数据盘可用569,728.29GiB。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 14:32:21 UTC 等待训练卡数单次巡检

- 阶段/证据：`waiting/awaiting_training_topology`，step 0 → 0，manual.active=false，卡数仍pending。已核对manual、pipeline、task-state、指定14:31快照和最近记录；证据为`runs/emilia-full-token-scratch/supervision/20260915T143221Z-inspection.json`。
- 数据/进程：质量、导出、codec完成报告和状态、各worker状态及配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，编码新增0；旧rank2/3仅作历史。train/val大小和mtime未变，沿用9月13日全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重和无配额保持，评分源码哈希匹配；音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不在，阶段脚本扫描无匹配，完成状态无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。
- GPU/矩阵：仍两张A800，各557MiB，GPU仅见本run矩阵worker，各占548MiB。矩阵父4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度0.56/0.68秒，observed_utilization均0、matrix_work均true；近30.0分钟CPU增加871.81/871.98秒，支持矩阵持续按空闲计算逻辑运行。现场利用率0/0%为瞬时采样，不据此判为失效，无需重启。主机已用81GiB、可用1.9TiB，无swap；数据盘可用572,174.70GiB。资源变化未归因于Emilia，未操作其他任务或数据。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 15:02:24 UTC 等待训练卡数单次巡检

- 阶段/证据：`waiting/awaiting_training_topology`，step 0 → 0，manual.active=false，卡数仍pending。已核对manual、pipeline、task-state、指定15:01快照和最近记录；证据为`runs/emilia-full-token-scratch/supervision/20260915T150224Z-inspection.json`。
- 数据/进程：质量、导出、codec完成报告和状态、各worker状态及配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，编码新增0；旧rank2/3仅作历史。train/val大小和mtime未变，沿用9月13日全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重和无配额保持，评分源码哈希匹配；音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不在，阶段脚本扫描无匹配，完成状态无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。
- GPU/矩阵：仍两张A800，各557MiB，利用率93/98%；GPU仅见本run矩阵worker，各占548MiB。矩阵父4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度1.49/1.76秒，observed_utilization均0、matrix_work均true；近30.1分钟CPU增加874.62/873.89秒，支持矩阵持续按空闲计算逻辑运行，无需重启。主机已用74GiB、可用1.9TiB，无swap；数据盘可用572,082.81GiB。资源变化未归因于Emilia，未操作其他任务或数据。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 15:33:04 UTC 等待训练卡数单次巡检

- 阶段/证据：`waiting/awaiting_training_topology`，step 0 → 0，manual.active=false，卡数仍pending。已核对manual、pipeline、task-state、指定15:31快照和最近记录；证据为`runs/emilia-full-token-scratch/supervision/20260915T153304Z-inspection.json`。
- 数据/进程：质量、导出、codec完成报告和状态、各worker状态及配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，编码新增0；旧rank2/3仅作历史。train/val大小和mtime未变，沿用9月13日全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重和无配额保持，评分源码哈希匹配；音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不在，阶段脚本扫描无匹配，完成状态无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。
- GPU/矩阵：仍两张A800，各557MiB，利用率42/97%；GPU仅见本run矩阵worker，各占548MiB。矩阵父4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度0.24/1.93秒，observed_utilization均0、matrix_work均true；近30.7分钟CPU增加892.99/892.41秒，支持矩阵持续按空闲计算逻辑运行，无需重启。主机已用74GiB、可用1.9TiB，无swap；数据盘可用572,032.02GiB。资源变化未归因于Emilia，未操作其他任务或数据。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 16:02:22 UTC 等待训练卡数单次巡检

- 阶段/证据：`waiting/awaiting_training_topology`，step 0 → 0，manual.active=false，卡数仍pending。已核对manual、pipeline、task-state、指定16:01快照和最近记录；证据为`runs/emilia-full-token-scratch/supervision/20260915T160222Z-inspection.json`。
- 数据/进程：质量、导出、codec完成报告和状态、各worker状态及配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，编码新增0；旧rank2/3仅作历史。train/val大小和mtime未变，沿用9月13日全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重和无配额保持，评分源码哈希匹配；音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不在，阶段脚本扫描无匹配，完成状态无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。
- GPU/矩阵：仍两张A800，各557MiB，利用率20/18%；GPU仅见本run矩阵worker，各占548MiB。矩阵父4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度0.41/0.42秒，observed_utilization均0、matrix_work均true；近29.3分钟CPU增加852.23/852.52秒，支持矩阵持续按空闲计算逻辑运行，无需重启。主机已用74GiB、可用1.9TiB，无swap；数据盘可用571,999.03GiB。资源变化未归因于Emilia，未操作其他任务或数据。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 16:33:06 UTC 等待训练卡数单次巡检

- 阶段/证据：`waiting/awaiting_training_topology`，step 0 → 0，manual.active=false，卡数仍pending。已核对manual、pipeline、task-state、指定16:31快照和最近记录；证据为`runs/emilia-full-token-scratch/supervision/20260915T163306Z-inspection.json`。
- 数据/进程：质量、导出、codec完成报告和状态、各worker状态及配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，编码新增0；旧rank2/3仅作历史。train/val大小和mtime未变，沿用9月13日全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重和无配额保持，评分源码哈希匹配；音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不在，阶段脚本扫描无匹配，完成状态无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。
- GPU/矩阵：仍两张A800，各557MiB，利用率97/97%；GPU仅见本run矩阵worker，各占548MiB。矩阵父4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度1.86/1.81秒，observed_utilization均0、matrix_work均true；近30.7分钟CPU增加893.95/893.92秒，支持矩阵持续按空闲计算逻辑运行，无需重启。主机已用92GiB、可用1.9TiB，无swap；数据盘可用571,916.91GiB。资源变化未归因于Emilia，未操作其他任务或数据。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 17:02:24 UTC 等待训练卡数单次巡检

- 阶段/证据：`waiting/awaiting_training_topology`，step 0 → 0，manual.active=false，卡数仍pending。已核对manual、pipeline、task-state、指定17:01快照和最近记录；证据为`runs/emilia-full-token-scratch/supervision/20260915T170224Z-inspection.json`。
- 数据/进程：质量、导出、codec完成报告和状态、各worker状态及配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，编码新增0；旧rank2/3仅作历史。train/val大小和mtime未变，沿用9月13日全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重和无配额保持，评分源码哈希匹配；音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不在，阶段脚本扫描无匹配，完成状态无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。
- GPU/矩阵：仍两张A800，各557MiB，GPU仅见本run矩阵worker，各占548MiB。矩阵父4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度1.05/0.83秒，observed_utilization均0、matrix_work均true；近29.3分钟CPU增加845.68/845.74秒，支持矩阵持续按空闲计算逻辑运行。现场利用率0/0%为瞬时采样，不据此判为失效，无需重启。主机已用92GiB、可用1.9TiB，无swap；数据盘可用571,870.63GiB。资源变化未归因于Emilia，未操作其他任务或数据。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 17:34:04 UTC 等待训练卡数单次巡检

- 阶段/证据：`waiting/awaiting_training_topology`，step 0 → 0，manual.active=false，卡数仍pending。已核对manual、pipeline、task-state、指定17:31快照和最近记录；证据为`runs/emilia-full-token-scratch/supervision/20260915T173404Z-inspection.json`。
- 数据/进程：质量、导出、codec完成报告和状态、各worker状态及配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，编码新增0；旧rank2/3仅作历史。train/val大小和mtime未变，沿用9月13日全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重和无配额保持，评分源码哈希匹配；音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不在，阶段脚本扫描无匹配，完成状态无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。
- GPU/矩阵：仍两张A800，各557MiB，利用率97/97%；GPU仅见本run矩阵worker，各占548MiB。矩阵父4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度0.03/2.06秒，observed_utilization均0、matrix_work均true；近31.7分钟CPU增加914.54/914.46秒，支持矩阵持续按空闲计算逻辑运行，无需重启。主机已用110GiB、可用1.8TiB，无swap；数据盘可用571,705.80GiB。资源变化未归因于Emilia，未操作其他任务或数据。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 18:02:55 UTC 等待训练卡数单次巡检

- 阶段/证据：`waiting/awaiting_training_topology`，step 0 → 0，manual.active=false，卡数仍pending。已核对manual、pipeline、task-state、指定18:01快照和最近记录；证据为`runs/emilia-full-token-scratch/supervision/20260915T180255Z-inspection.json`。
- 数据/进程：质量、导出、codec完成报告和状态、各worker状态及配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，编码新增0；旧rank2/3仅作历史。train/val大小和mtime未变，沿用9月13日全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重和无配额保持，评分源码哈希匹配；音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不在，阶段脚本扫描无匹配，完成状态无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。
- GPU/矩阵：仍两张A800，各557MiB，GPU仅见本run矩阵worker，各占548MiB。矩阵父4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度0.74/0.73秒，observed_utilization均0、matrix_work均true；近28.8分钟CPU各增加832.62秒，支持矩阵持续按空闲计算逻辑运行。现场利用率0/0%为瞬时采样，不据此判为失效，无需重启。主机已用112GiB、可用1.8TiB，无swap；数据盘可用571,563.02GiB。资源变化未归因于Emilia，未操作其他任务或数据。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 18:32:22 UTC 等待训练卡数单次巡检

- 阶段/证据：`waiting/awaiting_training_topology`，step 0 → 0，manual.active=false，卡数仍pending。已核对manual、pipeline、task-state、指定18:31快照和最近记录；证据为`runs/emilia-full-token-scratch/supervision/20260915T183222Z-inspection.json`。
- 数据/进程：质量、导出、codec完成报告和状态、各worker状态及配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，编码新增0；旧rank2/3仅作历史。train/val大小和mtime未变，沿用9月13日全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重和无配额保持，评分源码哈希匹配；音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不在，阶段脚本扫描无匹配，完成状态无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。
- GPU/矩阵：仍两张A800，各557MiB，GPU仅见本run矩阵worker，各占548MiB。矩阵父4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度均1.05秒，observed_utilization均0、matrix_work均true；近29.4分钟CPU增加849.53/849.49秒，支持矩阵持续按空闲计算逻辑运行。现场利用率0/0%为瞬时采样，不据此判为失效，无需重启。主机已用100GiB、可用1.9TiB，无swap；数据盘可用571,491.41GiB。资源变化未归因于Emilia，未操作其他任务或数据。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 19:03:20 UTC 等待训练卡数单次巡检

- 阶段/证据：`waiting/awaiting_training_topology`，step 0 → 0，manual.active=false，卡数仍pending。已核对manual、pipeline、task-state、指定19:01快照和最近记录；证据为`runs/emilia-full-token-scratch/supervision/20260915T190320Z-inspection.json`。
- 数据/进程：质量、导出、codec完成报告和状态、各worker状态及配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，编码新增0；旧rank2/3仅作历史。train/val大小和mtime未变，沿用9月13日全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重和无配额保持，评分源码哈希匹配；音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不在，阶段脚本扫描无匹配，完成状态无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。
- GPU/矩阵：仍两张A800，各557MiB，利用率97/97%；GPU仅见本run矩阵worker，各占548MiB。矩阵父4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度1.79/1.62秒，observed_utilization均0、matrix_work均true；近31.0分钟CPU增加883.96/883.82秒，支持矩阵持续按空闲计算逻辑运行，无需重启。主机已用100GiB、可用1.9TiB，无swap；数据盘可用576,203.70GiB。资源变化未归因于Emilia，未操作其他任务或数据。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 19:32:35 UTC 等待训练卡数单次巡检

- 阶段/证据：`waiting/awaiting_training_topology`，step 0 → 0，manual.active=false，卡数仍pending。已核对manual、pipeline、task-state、指定19:31快照和最近记录；证据为`runs/emilia-full-token-scratch/supervision/20260915T193235Z-inspection.json`。
- 数据/进程：质量、导出、codec完成报告和状态、各worker状态及配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，编码新增0；旧rank2/3仅作历史。train/val大小和mtime未变，沿用9月13日全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重和无配额保持，评分源码哈希匹配；音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不在，阶段脚本扫描无匹配，完成状态无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。
- GPU/矩阵：仍两张A800，各557MiB，利用率97/97%；GPU仅见本run矩阵worker，各占548MiB。矩阵父4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度1.82/1.77秒，observed_utilization均0、matrix_work均true；近29.3分钟CPU增加844.51/844.56秒，支持矩阵持续按空闲计算逻辑运行，无需重启。主机已用100GiB、可用1.9TiB，无swap；数据盘可用576,054.07GiB。资源变化未归因于Emilia，未操作其他任务或数据。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 20:02:49 UTC 等待训练卡数单次巡检

- 阶段/证据：`waiting/awaiting_training_topology`，step 0 → 0，manual.active=false，卡数仍pending。已核对manual、pipeline、task-state、指定20:01快照和最近记录；证据为`runs/emilia-full-token-scratch/supervision/20260915T200249Z-inspection.json`。
- 数据/进程：质量、导出、codec完成报告和状态、各worker状态及配方均未变。raw/prepared各2,901份且文件名集合相等，score/stats各8,059份，编码新增0；旧rank2/3仅作历史。train/val大小和mtime未变，沿用9月13日全量SHA256和行数核验，本轮未重哈希大清单。train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定音质门槛、short→long→合格dialogue短句ID去重和无配额保持，评分源码哈希匹配；音质门槛不保证完整文本对齐，全量ASR分数缺失仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不在，阶段脚本扫描无匹配，完成状态无需恢复；独立父进程退出码未留存，不能声称exit_code=0。恢复、rank0/1及矩阵日志大小和mtime未变，错误扫描无匹配。
- GPU/矩阵：仍两张A800，各557MiB，利用率97/97%；GPU仅见本run矩阵worker，各占548MiB。矩阵父4832/start_ticks155946508及worker6667/6668的start_ticks155947569/155947570、cmdline、父PID均匹配，tmux存活。状态新鲜度1.81/1.82秒，observed_utilization均0、matrix_work均true；近30.2分钟CPU增加872.42/872.43秒，支持矩阵持续按空闲计算逻辑运行，无需重启。主机已用82GiB、可用1.9TiB，无swap；数据盘可用575,961.48GiB。资源变化未归因于Emilia，未操作其他任务或数据。
- 判断/动作/剩余：本run训练进程/退出记录、计划、日志和checkpoint均不存在，loss/grad/LR、吞吐/data_wait及评估指标不适用；未写startup/final-verification。数据完成状态保持、矩阵正常运行，仅保存证据并追加文档，未启停进程或修改代码/配置/数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 20:32:11 UTC 等待训练卡数单次巡检

- 阶段/证据：已核对manual、pipeline、task-state及指定20:31快照，仍为`waiting/awaiting_training_topology`，manual.active=false，step 0 → 0。证据为`runs/emilia-full-token-scratch/supervision/20260915T203137Z-inspection.json`，20:34:52补充进程父子关系、tmux及资源检查。
- 数据/日志：质量、导出、准备及worker0/1完成报告与20:02记录相同；train/val、PREPARATION_COMPLETE及preparation.json大小和mtime均未变化。沿用9月13日全量SHA256及行数核验：train 23,931,105条、43,744.34小时，val 512条，本轮未重哈希大清单或重新统计分片。恢复、rank0/1及矩阵日志大小和mtime均未变化，指定快照无错误行。准备PID1935/55462/55543及worker55576/55577均已不存在，完成阶段无需恢复；独立父进程退出码未留存。
- GPU/矩阵：仍两张A800，各557MiB，现场利用率97/97%。GPU仅见本run矩阵worker6667/6668，各548MiB；父4832及两worker的start_ticks、cmdline、父PID均匹配，tmux存活。worker状态约1.5秒内更新，matrix_work均true；相较20:02，CPU时间增加847.27/847.25秒，矩阵运行正常。主机已用83GiB、可用1.9TiB，无swap；补充检查时数据盘可用575,825.32GiB。
- 判断/动作：训练进程/退出记录、计划、日志、checkpoint及startup/final-verification均不存在，loss/grad/LR、吞吐/data_wait和评估指标不适用。本轮仅保存证据并追加文档，未启停进程或修改代码、配置、数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 21:04:18 UTC 等待训练卡数单次巡检

- 阶段/证据：已核对manual、pipeline、task-state、指定21:01快照及最近记录，仍为`waiting/awaiting_training_topology`，manual.active=false，step 0 → 0。证据：`runs/emilia-full-token-scratch/supervision/20260915T210419Z-inspection.json`。
- 数据/进程：质量、导出、codec及各worker状态、完成报告和配方均与20:02完整基线一致，评分源码哈希匹配。raw/prepared各2,901份、名称集合相等，score/stats各8,059份，无新增编码；旧rank2/3仅作历史。train/val大小与mtime未变，沿用9月13日全量SHA256及行数核验，本轮未重哈希大清单：train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定质量门槛、短句ID去重和无配额保持；音质筛选不等于完整文本对齐，缺失全量ASR分数仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不存在，完成状态无需恢复；独立父进程退出码未留存。恢复、rank0/1及矩阵日志元数据未变，错误扫描无匹配。
- GPU/矩阵：两张A800各557MiB、现场利用率97/97%，GPU仅见本run矩阵worker6667/6668，各548MiB。父4832与worker的start_ticks、cmdline、父PID均匹配，tmux存活；状态新鲜度2.04/1.98秒，matrix_work均true。相较20:32，worker CPU时间增加927.02/927.01秒，支持持续运行，无需重启。主机已用84GiB、可用1.9TiB，无swap；数据盘可用575,766.48GiB。
- 判断/动作/剩余：训练进程/退出记录、计划、日志、checkpoint及startup/final-verification均不存在，loss/grad/LR、吞吐/data_wait和评估指标不适用。本轮仅保存证据并追加文档，未启停进程或修改代码、配置、数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 21:33:01 UTC 等待训练卡数单次巡检

- 阶段/证据：已核对manual、pipeline、task-state、指定21:31快照及最近记录，仍为`waiting/awaiting_training_topology`，manual.active=false，step 0 → 0。证据：`runs/emilia-full-token-scratch/supervision/20260915T213302Z-inspection.json`。
- 数据/进程：质量、导出、codec及各worker状态、完成报告和配方均与21:04一致，评分源码哈希匹配。raw/prepared各2,901份、名称集合相等，score/stats各8,059份，无新增编码；旧rank2/3仅作历史。train/val大小与mtime未变，沿用9月13日全量SHA256及行数核验，本轮未重哈希大清单：train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定质量门槛、短句ID去重和无配额保持；音质筛选不等于完整文本对齐，缺失全量ASR分数仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不存在，完成状态无需恢复；独立父进程退出码未留存。恢复、rank0/1及矩阵日志元数据未变，错误扫描无匹配。
- GPU/矩阵：两张A800各557MiB，GPU仅见本run矩阵worker6667/6668，各548MiB。父4832与worker的start_ticks、cmdline、父PID均匹配，tmux存活；状态新鲜度1.26/0.99秒，matrix_work均true。相较21:04，worker CPU时间增加833.45/833.33秒，支持持续运行。利用率83/0%为瞬时采样，结合心跳与CPU推进，无需重启。主机已用84GiB、可用1.9TiB，无swap；数据盘可用575,740.26GiB。
- 判断/动作/剩余：训练进程/退出记录、计划、日志、checkpoint及startup/final-verification均不存在，loss/grad/LR、吞吐/data_wait和评估指标不适用。本轮仅保存证据并追加文档，未启停进程或修改代码、配置、数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 22:03:40 UTC 等待训练卡数单次巡检

- 阶段/证据：已核对manual、pipeline、task-state、指定22:01快照及最近记录，仍为`waiting/awaiting_training_topology`，manual.active=false，step 0 → 0。证据：`runs/emilia-full-token-scratch/supervision/20260915T220341Z-inspection.json`。
- 数据/进程：质量、导出、codec及各worker状态、完成报告和配方均与21:33一致，评分源码哈希匹配。raw/prepared各2,901份、名称集合相等，score/stats各8,059份，无新增编码；旧rank2/3仅作历史。train/val大小与mtime未变，沿用9月13日全量SHA256及行数核验，本轮未重哈希大清单：train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定质量门槛、短句ID去重和无配额保持；音质筛选不等于完整文本对齐，缺失全量ASR分数仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不存在，完成状态无需恢复；独立父进程退出码未留存。恢复、rank0/1及矩阵日志元数据未变，错误扫描无匹配。
- GPU/矩阵：两张A800各557MiB，GPU仅见本run矩阵worker6667/6668，各548MiB。父4832与worker的start_ticks、cmdline、父PID均匹配，tmux存活；状态新鲜度0.56/0.30秒，matrix_work均true。相较21:33，worker CPU时间增加894.05/893.24秒，支持持续运行。利用率0/13%为瞬时采样，结合心跳与CPU推进，无需重启。主机已用83GiB、可用1.9TiB，无swap；数据盘可用575,685.55GiB。
- 判断/动作/剩余：训练进程/退出记录、计划、日志、checkpoint及startup/final-verification均不存在，loss/grad/LR、吞吐/data_wait和评估指标不适用。本轮仅保存证据并追加文档，未启停进程或修改代码、配置、数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 22:32:17 UTC 等待训练卡数单次巡检

- 阶段/证据：已核对manual、pipeline、task-state、指定22:31快照及最近记录，仍为`waiting/awaiting_training_topology`，manual.active=false，step 0 → 0。证据：`runs/emilia-full-token-scratch/supervision/20260915T223218Z-inspection.json`。
- 数据/进程：质量、导出、codec及各worker状态、完成报告和配方均与22:03一致，评分源码哈希匹配。raw/prepared各2,901份、名称集合相等，score/stats各8,059份，无新增编码；旧rank2/3仅作历史。train/val大小与mtime未变，沿用9月13日全量SHA256及行数核验，本轮未重哈希大清单：train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定质量门槛、短句ID去重和无配额保持；音质筛选不等于完整文本对齐，缺失全量ASR分数仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不存在，完成状态无需恢复；独立父进程退出码未留存。恢复、rank0/1及矩阵日志元数据未变，错误扫描无匹配。
- GPU/矩阵：两张A800各557MiB、现场利用率97/97%，GPU仅见本run矩阵worker6667/6668，各548MiB。父4832与worker的start_ticks、cmdline、父PID均匹配，tmux存活；状态新鲜度0.12/1.69秒，matrix_work均true。相较22:03，worker CPU时间增加834.17/833.81秒，支持持续运行，无需重启。主机已用84GiB、可用1.9TiB，无swap；数据盘可用575,602.34GiB。
- 判断/动作/剩余：训练进程/退出记录、计划、日志、checkpoint及startup/final-verification均不存在，loss/grad/LR、吞吐/data_wait和评估指标不适用。本轮仅保存证据并追加文档，未启停进程或修改代码、配置、数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 23:03:14 UTC 等待训练卡数单次巡检

- 阶段/证据：已核对manual、pipeline、task-state、指定23:01快照及最近记录，仍为`waiting/awaiting_training_topology`，manual.active=false，step 0 → 0。证据：`runs/emilia-full-token-scratch/supervision/20260915T230315Z-inspection.json`。
- 数据/进程：质量、导出、codec及各worker状态、完成报告和配方均与22:32一致，评分源码哈希匹配。raw/prepared各2,901份、名称集合相等，score/stats各8,059份，无新增编码；旧rank2/3仅作历史。train/val大小与mtime未变，沿用9月13日全量SHA256及行数核验，本轮未重哈希大清单：train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定质量门槛、短句ID去重和无配额保持；音质筛选不等于完整文本对齐，缺失全量ASR分数仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不存在，完成状态无需恢复；独立父进程退出码未留存。恢复、rank0/1及矩阵日志元数据未变，错误扫描无匹配。
- GPU/矩阵：两张A800各557MiB，GPU仅见本run矩阵worker6667/6668，各548MiB。父4832与worker的start_ticks、cmdline、父PID均匹配，tmux存活；状态新鲜度0.90/0.65秒，matrix_work均true。相较22:32，worker CPU时间增加901.26/901.62秒，支持持续运行。利用率0/0%为瞬时采样，结合心跳与CPU推进，无需重启。主机已用84GiB、可用1.9TiB，无swap；数据盘可用575,576.52GiB。
- 判断/动作/剩余：训练进程/退出记录、计划、日志、checkpoint及startup/final-verification均不存在，loss/grad/LR、吞吐/data_wait和评估指标不适用。本轮仅保存证据并追加文档，未启停进程或修改代码、配置、数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-15 23:32:30 UTC 等待训练卡数单次巡检

- 阶段/证据：已核对manual、pipeline、task-state、指定23:31快照及最近记录，仍为`waiting/awaiting_training_topology`，manual.active=false，step 0 → 0。证据：`runs/emilia-full-token-scratch/supervision/20260915T233231Z-inspection.json`。
- 数据/进程：质量、导出、codec及各worker状态、完成报告和配方均与23:03一致，评分源码哈希匹配。raw/prepared各2,901份、名称集合相等，score/stats各8,059份，无新增编码；旧rank2/3仅作历史。train/val大小与mtime未变，沿用9月13日全量SHA256及行数核验，本轮未重哈希大清单：train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定质量门槛、短句ID去重和无配额保持；音质筛选不等于完整文本对齐，缺失全量ASR分数仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不存在，完成状态无需恢复；独立父进程退出码未留存。恢复、rank0/1及矩阵日志元数据未变，错误扫描无匹配。
- GPU/矩阵：两张A800各557MiB，GPU仅见本run矩阵worker6667/6668，各548MiB。父4832与worker的start_ticks、cmdline、父PID均匹配，tmux存活；状态新鲜度0.57/0.20秒，matrix_work均true。相较23:03，worker CPU时间增加853.21/853.17秒，支持持续运行。利用率0/29%为瞬时采样，结合心跳与CPU推进，无需重启。主机已用84GiB、可用1.9TiB，无swap；数据盘可用575,558.16GiB。
- 判断/动作/剩余：训练进程/退出记录、计划、日志、checkpoint及startup/final-verification均不存在，loss/grad/LR、吞吐/data_wait和评估指标不适用。本轮仅保存证据并追加文档，未启停进程或修改代码、配置、数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-16 00:03:00 UTC 等待训练卡数单次巡检

- 阶段/证据：已核对manual、pipeline、task-state、指定00:01快照及最近记录，仍为`waiting/awaiting_training_topology`，manual.active=false，step 0 → 0。证据：`runs/emilia-full-token-scratch/supervision/20260916T000301Z-inspection.json`。
- 数据/进程：质量、导出、codec及各worker状态、完成报告和配方均与9月15日23:32一致，评分源码哈希匹配。raw/prepared各2,901份、名称集合相等，score/stats各8,059份，无新增编码；旧rank2/3仅作历史。train/val大小与mtime未变，沿用9月13日全量SHA256及行数核验，本轮未重哈希大清单：train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定质量门槛、短句ID去重和无配额保持；音质筛选不等于完整文本对齐，缺失全量ASR分数仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不存在，完成状态无需恢复；独立父进程退出码未留存。恢复、rank0/1及矩阵日志元数据未变，错误扫描无匹配。
- GPU/矩阵：两张A800各557MiB、现场利用率96/96%，GPU仅见本run矩阵worker6667/6668，各548MiB。父4832与worker的start_ticks、cmdline、父PID均匹配，tmux存活；状态新鲜度1.62/1.51秒，matrix_work均true。相较23:32，worker CPU时间增加888.74/888.65秒，支持持续运行，无需重启。主机已用84GiB、可用1.9TiB，无swap；数据盘可用575,497.69GiB。
- 判断/动作/剩余：训练进程/退出记录、计划、日志、checkpoint及startup/final-verification均不存在，loss/grad/LR、吞吐/data_wait和评估指标不适用。本轮仅保存证据并追加文档，未启停进程或修改代码、配置、数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-16 00:32:23 UTC 等待训练卡数单次巡检

- 阶段/证据：已核对manual、pipeline、task-state、指定00:31快照及最近记录，仍为`waiting/awaiting_training_topology`，manual.active=false，step 0 → 0。证据：`runs/emilia-full-token-scratch/supervision/20260916T003224Z-inspection.json`。
- 数据/进程：质量、导出、codec及各worker状态、完成报告和配方均与00:03一致，评分源码哈希匹配。raw/prepared各2,901份、名称集合相等，score/stats各8,059份，无新增编码；旧rank2/3仅作历史。train/val大小与mtime未变，沿用9月13日全量SHA256及行数核验，本轮未重哈希大清单：train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定质量门槛、短句ID去重和无配额保持；音质筛选不等于完整文本对齐，缺失全量ASR分数仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不存在，完成状态无需恢复；独立父进程退出码未留存。恢复、rank0/1及矩阵日志元数据未变，错误扫描无匹配。
- GPU/矩阵：两张A800各557MiB、现场利用率97/66%，GPU仅见本run矩阵worker6667/6668，各548MiB。父4832与worker的start_ticks、cmdline、父PID均匹配，tmux存活；状态新鲜度0.09/0.13秒，matrix_work均true。相较00:03，worker CPU时间增加856.71/855.75秒，支持持续运行，无需重启。主机已用84GiB、可用1.9TiB，无swap；数据盘可用575,428.88GiB。
- 判断/动作/剩余：训练进程/退出记录、计划、日志、checkpoint及startup/final-verification均不存在，loss/grad/LR、吞吐/data_wait和评估指标不适用。本轮仅保存证据并追加文档，未启停进程或修改代码、配置、数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-16 01:02:59 UTC 等待训练卡数单次巡检

- 阶段/证据：已核对manual、pipeline、task-state、指定01:01快照及最近记录，仍为`waiting/awaiting_training_topology`，manual.active=false，step 0 → 0。证据：`runs/emilia-full-token-scratch/supervision/20260916T010259Z-inspection.json`。
- 数据/进程：质量、导出、codec及各worker状态、完成报告和配方均与00:32一致，评分源码哈希匹配。raw/prepared各2,901份、名称集合相等，score/stats各8,059份，无新增编码；旧rank2/3仅作历史。train/val大小与mtime未变，沿用9月13日全量SHA256及行数核验，本轮未重哈希大清单：train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定质量门槛、短句ID去重和无配额保持；音质筛选不等于完整文本对齐，缺失全量ASR分数仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不存在，完成状态无需恢复；独立父进程退出码未留存。恢复、rank0/1及矩阵日志元数据未变，错误扫描无匹配。
- GPU/矩阵：两张A800各557MiB，GPU仅见本run矩阵worker6667/6668，各548MiB。父4832与worker的start_ticks、cmdline、父PID均匹配，tmux存活；状态新鲜度0.84/0.08秒，matrix_work均true。相较00:32，worker CPU时间增加891.24/891.23秒，支持持续运行。利用率0/98%为瞬时采样，结合心跳与CPU推进，无需重启。主机已用84GiB、可用1.9TiB，无swap；数据盘可用575,381.15GiB。
- 判断/动作/剩余：训练进程/退出记录、计划、日志、checkpoint及startup/final-verification均不存在，loss/grad/LR、吞吐/data_wait和评估指标不适用。本轮仅保存证据并追加文档，未启停进程或修改代码、配置、数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-16 01:33:01 UTC 等待训练卡数单次巡检

- 阶段/证据：已核对manual、pipeline、task-state、指定01:31快照及最近记录，仍为`waiting/awaiting_training_topology`，manual.active=false，step 0 → 0。证据：`runs/emilia-full-token-scratch/supervision/20260916T013301Z-inspection.json`。
- 数据/进程：质量、导出、codec及各worker状态、完成报告和配方均与01:02一致，评分源码哈希匹配。raw/prepared各2,901份、名称集合相等，score/stats各8,059份，无新增编码；旧rank2/3仅作历史。train/val大小与mtime未变，沿用9月13日全量SHA256及行数核验，本轮未重哈希大清单：train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定质量门槛、短句ID去重和无配额保持；音质筛选不等于完整文本对齐，缺失全量ASR分数仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不存在，完成状态无需恢复；独立父进程退出码未留存。恢复、rank0/1及矩阵日志元数据未变，错误扫描无匹配。
- GPU/矩阵：两张A800各557MiB，GPU仅见本run矩阵worker6667/6668，各548MiB。父4832与worker的start_ticks、cmdline、父PID均匹配，tmux存活；状态新鲜度1.42/1.05秒，matrix_work均true。相较01:02，worker CPU时间增加875.59/874.26秒，支持持续运行。利用率94/0%为瞬时采样，结合心跳与CPU推进，无需重启。主机已用85GiB、可用1.9TiB，无swap；数据盘可用575,369.91GiB。
- 判断/动作/剩余：训练进程/退出记录、计划、日志、checkpoint及startup/final-verification均不存在，loss/grad/LR、吞吐/data_wait和评估指标不适用。本轮仅保存证据并追加文档，未启停进程或修改代码、配置、数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-16 02:02:13 UTC 等待训练卡数单次巡检

- 阶段/证据：已核对manual、pipeline、task-state、指定02:01快照及最近记录，仍为`waiting/awaiting_training_topology`，manual.active=false，step 0 → 0。证据：`runs/emilia-full-token-scratch/supervision/20260916T020214Z-inspection.json`。
- 数据/进程：质量、导出、codec及各worker状态、完成报告和配方均与01:33一致，评分源码哈希匹配。raw/prepared各2,901份、名称集合相等，score/stats各8,059份，无新增编码；旧rank2/3仅作历史。train/val大小与mtime未变，沿用9月13日全量SHA256及行数核验，本轮未重哈希大清单：train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定质量门槛、短句ID去重和无配额保持；音质筛选不等于完整文本对齐，缺失全量ASR分数仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不存在，完成状态无需恢复；独立父进程退出码未留存。恢复、rank0/1及矩阵日志元数据未变，错误扫描无匹配。
- GPU/矩阵：两张A800各557MiB，GPU仅见本run矩阵worker6667/6668，各548MiB。父4832与worker的start_ticks、cmdline、父PID均匹配，tmux存活；状态新鲜度0.37/0.78秒，matrix_work均true。相较01:33，worker CPU时间增加850.85/851.18秒，支持持续运行。利用率0/0%为瞬时采样，结合心跳与CPU推进，无需重启。主机已用85GiB、可用1.9TiB，无swap；数据盘可用575,321.42GiB。
- 判断/动作/剩余：训练进程/退出记录、计划、日志、checkpoint及startup/final-verification均不存在，loss/grad/LR、吞吐/data_wait和评估指标不适用。本轮仅保存证据并追加文档，未启停进程或修改代码、配置、数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-16 02:32:38 UTC 等待训练卡数单次巡检

- 阶段/证据：已核对manual、pipeline、task-state、指定02:31快照及最近记录，仍为`waiting/awaiting_training_topology`，manual.active=false，step 0 → 0。证据：`runs/emilia-full-token-scratch/supervision/20260916T023238Z-inspection.json`。
- 数据/进程：质量、导出、codec及各worker状态、完成报告和配方均与02:02一致，评分源码哈希匹配。raw/prepared各2,901份、名称集合相等，score/stats各8,059份，无新增编码；旧rank2/3仅作历史。train/val大小与mtime未变，沿用9月13日全量SHA256及行数核验，本轮未重哈希大清单：train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定质量门槛、短句ID去重和无配额保持；音质筛选不等于完整文本对齐，缺失全量ASR分数仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不存在，完成状态无需恢复；独立父进程退出码未留存。恢复、rank0/1及矩阵日志元数据未变，错误扫描无匹配。
- GPU/矩阵：两张A800各557MiB，GPU仅见本run矩阵worker6667/6668，各548MiB。父4832与worker的start_ticks、cmdline、父PID均匹配，tmux存活；状态新鲜度0.28/0.33秒，matrix_work均true。相较02:02，worker CPU时间增加885.31/885.35秒，支持持续运行。利用率24/0%为瞬时采样，结合心跳与CPU推进，无需重启。主机已用86GiB、可用1.9TiB，无swap；数据盘可用575,304.21GiB。
- 判断/动作/剩余：训练进程/退出记录、计划、日志、checkpoint及startup/final-verification均不存在，loss/grad/LR、吞吐/data_wait和评估指标不适用。本轮仅保存证据并追加文档，未启停进程或修改代码、配置、数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-16 03:02:22 UTC 等待训练卡数单次巡检

- 阶段/证据：已核对manual、pipeline、task-state、指定03:01快照及最近记录，仍为`waiting/awaiting_training_topology`，manual.active=false，step 0 → 0。证据：`runs/emilia-full-token-scratch/supervision/20260916T030223Z-inspection.json`。
- 数据/进程：质量、导出、codec及各worker状态、完成报告和配方均与02:32一致，评分源码哈希匹配。raw/prepared各2,901份、名称集合相等，score/stats各8,059份，无新增编码；旧rank2/3仅作历史。train/val大小与mtime未变，沿用9月13日全量SHA256及行数核验，本轮未重哈希大清单：train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定质量门槛、短句ID去重和无配额保持；音质筛选不等于完整文本对齐，缺失全量ASR分数仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不存在，完成状态无需恢复；独立父进程退出码未留存。恢复、rank0/1及矩阵日志元数据未变，错误扫描无匹配。
- GPU/矩阵：两张A800各557MiB、现场利用率97/33%，GPU仅见本run矩阵worker6667/6668，各548MiB。父4832与worker的start_ticks、cmdline、父PID均匹配，tmux存活；状态新鲜度1.61/1.10秒，matrix_work均true。相较02:32，worker CPU时间增加865.80/865.26秒，支持持续运行，无需重启。主机已用87GiB、可用1.9TiB，无swap；数据盘可用575,275.08GiB。
- 判断/动作/剩余：训练进程/退出记录、计划、日志、checkpoint及startup/final-verification均不存在，loss/grad/LR、吞吐/data_wait和评估指标不适用。本轮仅保存证据并追加文档，未启停进程或修改代码、配置、数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-16 03:32:31 UTC 等待训练卡数单次巡检

- 阶段/证据：已核对manual、pipeline、task-state、指定03:31快照及最近记录，仍为`waiting/awaiting_training_topology`，manual.active=false，step 0 → 0。证据：`runs/emilia-full-token-scratch/supervision/20260916T033232Z-inspection.json`。
- 数据/进程：质量、导出、codec及各worker状态、完成报告和配方均与03:02一致，评分源码哈希匹配。raw/prepared各2,901份、名称集合相等，score/stats各8,059份，无新增编码；旧rank2/3仅作历史。train/val大小与mtime未变，沿用9月13日全量SHA256及行数核验，本轮未重哈希大清单：train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定质量门槛、短句ID去重和无配额保持；音质筛选不等于完整文本对齐，缺失全量ASR分数仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不存在，完成状态无需恢复；独立父进程退出码未留存。恢复、rank0/1及矩阵日志元数据未变，错误扫描无匹配。
- GPU/矩阵：两张A800各557MiB、现场利用率97/65%，GPU仅见本run矩阵worker6667/6668，各548MiB。父4832与worker的start_ticks、cmdline、父PID均匹配，tmux存活；状态新鲜度1.83/0.08秒，matrix_work均true。相较03:02，worker CPU时间增加878.45/879.22秒，支持持续运行，无需重启。主机已用86GiB、可用1.9TiB，无swap；数据盘可用575,303.24GiB。
- 判断/动作/剩余：训练进程/退出记录、计划、日志、checkpoint及startup/final-verification均不存在，loss/grad/LR、吞吐/data_wait和评估指标不适用。本轮仅保存证据并追加文档，未启停进程或修改代码、配置、数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-16 04:02:20 UTC 等待训练卡数单次巡检

- 阶段/证据：已核对manual、pipeline、task-state、指定04:01快照及最近记录，仍为`waiting/awaiting_training_topology`，manual.active=false，step 0 → 0。证据：`runs/emilia-full-token-scratch/supervision/20260916T040221Z-inspection.json`。
- 数据/进程：质量、导出、codec及各worker状态、完成报告和配方均与03:32一致，评分源码哈希匹配。raw/prepared各2,901份、名称集合相等，score/stats各8,059份，无新增编码；旧rank2/3仅作历史。train/val大小与mtime未变，沿用9月13日全量SHA256及行数核验，本轮未重哈希大清单：train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定质量门槛、短句ID去重和无配额保持；音质筛选不等于完整文本对齐，缺失全量ASR分数仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不存在，完成状态无需恢复；独立父进程退出码未留存。恢复、rank0/1及矩阵日志元数据未变，错误扫描无匹配。
- GPU/矩阵：两张A800各557MiB、现场利用率97/97%，GPU仅见本run矩阵worker6667/6668，各548MiB。父4832与worker的start_ticks、cmdline、父PID均匹配，tmux存活；状态新鲜度1.95/1.89秒，matrix_work均true。相较03:32，worker CPU时间增加868.37/868.06秒，支持持续运行，无需重启。主机已用86GiB、可用1.9TiB，无swap；数据盘可用575,220.69GiB。
- 判断/动作/剩余：训练进程/退出记录、计划、日志、checkpoint及startup/final-verification均不存在，loss/grad/LR、吞吐/data_wait和评估指标不适用。本轮仅保存证据并追加文档，未启停进程或修改代码、配置、数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-16 04:33:08 UTC 等待训练卡数单次巡检

- 阶段/证据：已核对manual、pipeline、task-state、指定04:31快照及最近记录，仍为`waiting/awaiting_training_topology`，manual.active=false，step 0 → 0。证据：`runs/emilia-full-token-scratch/supervision/20260916T043309Z-inspection.json`。
- 数据/进程：质量、导出、codec及各worker状态、完成报告和配方均与04:02一致，评分源码哈希匹配。raw/prepared各2,901份、名称集合相等，score/stats各8,059份，无新增编码；旧rank2/3仅作历史。train/val大小与mtime未变，沿用9月13日全量SHA256及行数核验，本轮未重哈希大清单：train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定质量门槛、短句ID去重和无配额保持；音质筛选不等于完整文本对齐，缺失全量ASR分数仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不存在，完成状态无需恢复；独立父进程退出码未留存。恢复、rank0/1及矩阵日志元数据未变，错误扫描无匹配。
- GPU/矩阵：两张A800各557MiB，GPU仅见本run矩阵worker6667/6668，各548MiB。父4832与worker的start_ticks、cmdline、父PID均匹配，tmux存活；状态新鲜度1.18/0.74秒，matrix_work均true。相较04:02，worker CPU时间增加897.50/896.48秒，支持持续运行。利用率25/0%为瞬时采样，结合心跳与CPU推进，无需重启。主机已用87GiB、可用1.9TiB，无swap；数据盘可用575,194.60GiB。
- 判断/动作/剩余：训练进程/退出记录、计划、日志、checkpoint及startup/final-verification均不存在，loss/grad/LR、吞吐/data_wait和评估指标不适用。本轮仅保存证据并追加文档，未启停进程或修改代码、配置、数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-16 05:02:42 UTC 等待训练卡数单次巡检

- 阶段/证据：已核对manual、pipeline、task-state、指定05:01快照及最近记录，仍为`waiting/awaiting_training_topology`，manual.active=false，step 0 → 0。证据：`runs/emilia-full-token-scratch/supervision/20260916T050243Z-inspection.json`。
- 数据/进程：质量、导出、codec及各worker状态、完成报告和配方均与04:33一致，评分源码哈希匹配。raw/prepared各2,901份、名称集合相等，score/stats各8,059份，无新增编码；旧rank2/3仅作历史。train/val大小与mtime未变，沿用9月13日全量SHA256及行数核验，本轮未重哈希大清单：train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定质量门槛、短句ID去重和无配额保持；音质筛选不等于完整文本对齐，缺失全量ASR分数仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不存在，完成状态无需恢复；独立父进程退出码未留存。恢复、rank0/1及矩阵日志元数据未变，错误扫描无匹配。
- GPU/矩阵：两张A800各557MiB，GPU仅见本run矩阵worker6667/6668，各548MiB。父4832与worker的start_ticks、cmdline、父PID均匹配，tmux存活；状态新鲜度1.13/0.33秒，matrix_work均true。相较04:33，worker CPU时间增加862.15/861.22秒，支持持续运行。利用率8/0%为瞬时采样，结合心跳与CPU推进，无需重启。主机已用86GiB、可用1.9TiB，无swap；数据盘可用575,344.56GiB。
- 判断/动作/剩余：训练进程/退出记录、计划、日志、checkpoint及startup/final-verification均不存在，loss/grad/LR、吞吐/data_wait和评估指标不适用。本轮仅保存证据并追加文档，未启停进程或修改代码、配置、数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。


## 2026-09-16 05:33:06 UTC 等待训练卡数单次巡检

- 阶段/证据：已核对manual、pipeline、task-state、指定05:31快照及最近记录，仍为`waiting/awaiting_training_topology`，manual.active=false，step 0 → 0。证据：`runs/emilia-full-token-scratch/supervision/20260916T053307Z-inspection.json`。
- 数据/进程：质量、导出、codec及各worker状态、完成报告和配方均与05:02一致，评分源码哈希匹配。raw/prepared各2,901份、名称集合相等，score/stats各8,059份，无新增编码；旧rank2/3仅作历史。train/val大小与mtime未变，沿用9月13日全量SHA256及行数核验，本轮未重哈希大清单：train 23,931,105条、43,744.34小时，val 512条。ASMR排除、固定质量门槛、短句ID去重和无配额保持；音质筛选不等于完整文本对齐，缺失全量ASR分数仍为设计边界。准备PID1935/55462/55543及worker55576/55577均不存在，完成状态无需恢复；独立父进程退出码未留存。恢复、rank0/1及矩阵日志元数据未变，错误扫描无匹配。
- GPU/矩阵：两张A800各557MiB、现场利用率88/91%，GPU仅见本run矩阵worker6667/6668，各548MiB。父4832与worker的start_ticks、cmdline、父PID均匹配，tmux存活；状态新鲜度1.54/1.58秒，matrix_work均true。相较05:02，worker CPU时间增加886.65/885.76秒，支持持续运行，无需重启。主机已用86GiB、可用1.9TiB，无swap；数据盘可用575,322.25GiB。
- 判断/动作/剩余：训练进程/退出记录、计划、日志、checkpoint及startup/final-verification均不存在，loss/grad/LR、吞吐/data_wait和评估指标不适用。本轮仅保存证据并追加文档，未启停进程或修改代码、配置、数据。继续等待训练卡数答复；仅两卡且未答复时不启动训练，四卡恢复且无相反指令时按原授权并结合现场占用执行。容量测试/训练前先停止本run矩阵并确认显存释放。本次巡检结束。
