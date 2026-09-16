# Emilia 10kh sample loss训练巡检

## 2026-09-11T02:23:03.024494+00:00 — 排队，等待sqrt完成19000步

用户要求sqrt训练到19000步后，接着进行同设置sample loss实验。配置configs/emilia-10kh-pretrain-sample.yaml与sqrt逐字段对照，只改变train.loss_reduction=sample和train.output；sample逐句等权，保留四卡、6000/9000预算、accumulation1、workers16/prefetch2、max_steps19000、schedule_steps38539及其余训练/评估设置。

本run从原assembled model的step0初始化，不从sqrt检查点续训。现有后台session 01a08697-ba78-7cd3-bb9e-b01eba12d724在sqrt最终验收、正常退出及所有rank退出后，使用本run launch.py顺序启动；启动器保持独占锁和前序19000验收检查。至少20次真实更新、初始化/配置/四卡allocator/损失梯度/LR和资源验证通过后才完成交接。后台随后沿用同session每1800秒监管本run；manual.active=false。

当前仅准备与排队，尚未启动，也不把主会话写入请求当作后台已接收；确认收到及启动证据由原后台写入sqrt run的next-run-ack.json。本run启动/恢复和最终验收按supervision-prompt.md及现有故障处置表执行；无后续排队实验。

- 2026-09-11T02:24:21.008549+00:00 主会话预检通过：配置仅loss/output差异；启动器三个隔离场景通过（缺前序验收不得启动、前序仍存活不得启动、验收退出后四卡无resume启动）；现有交接三项回归通过，git diff --check通过。当前sqrt PID3343588、后台PID4145999身份均保持；sample尚未启动。证据sqrt run sample-handoff/preflight-verification.json。等待原后台下一次巡检写received，未冒充后台确认。


### 2026-09-11T07:19:34.531838+00:00 — 从原assembled model首次启动，验证进行中

前序sqrt已完成19000 COMPLETE、512条val、SO/ICL各8条生成、同PID exit0及include-speaker冻结检查，final-verification.json已通过；全部旧torchrun/rank退出。sample配置逐字段核验仅loss_reduction=sample和output不同，四卡6000/9000、accum1、workers16/prefetch2、max19000/schedule38539、原学习率/残余0.3/网络及评估保持。以.venv Python的subprocess.Popen([sys.executable, sample绝对路径/launch.py], cwd项目根, start_new_session=True, stdin=DEVNULL, stdout/stderr追加launcher.log)首次启动，无resume。launcher737569、torchrun737573/start95204815，started_at=2026-09-11T07:19:02.496046+00:00，manual=false。初始化及至少20次真实更新仍待验证，ack=starting，禁止重复启动。


### 2026-09-11T07:28:17.121409+00:00 — 首次启动验证通过，接管sample

- 已确认initialized={step:0,epoch:0,next_batch:0}、world4、914643008参数；initialization.json与原sqrt的assembled来源/assembly_report SHA一致，model_config相同，实际config与sample源配置一致，仅loss/output不同。torchrun737573/start95204815、四rank737607–610/start95205032身份、父子关系、无resume及expandable_segments:True核验通过，launcher737569/start95204812独立运行，无training-exit.json或训练异常。
- 已完成50次真实更新、5个连续每10步日志点，普通CE/sample CE/grad均有限，LR逐点符合warmup=(step+1)/1000；目标first_sample_ce+0.3 residual_sample_ce，普通CE为token口径。第10步token=9.845714230/7.829417140、sample=9.930549667/7.829629864、grad49.094051、LR1.1e-6/3.3e-6；第20步token=9.215414511/7.827262984、sample=9.287774172/7.826522866、grad27.873039、LR2.1e-6/6.3e-6。大梯度为clip前记录，配置仍grad_clip=1.0，没有把有限大梯度当作NaN或任意调LR。
- 验收时step50：token=7.795348293/7.721099880，sample=7.758584808/7.717225791，目标=10.073752545，grad=2.707817，LR=5.1e-06/1.53e-05。startup日志step中位2.169522s，wait中位0.0002837s；第10/20/30步为2.316752/2.381069/2.164999s，吞吐784.098/792.887/874.901音频秒每墙钟秒，wait0.0009786/0.0005167/0.0002615s，供数恢复。首次启动前段为清单/模型加载及64数据worker初始化，保留日志原有Flash Attention dtype提示，未发生训练失败、重试或恢复。
- 07:27资源采样四卡63599/63219/63221/63281MiB（各81920）、利用率97–99%，仅四sample rank；RAM用238GiB、可用1.7TiB、无swap。验收最新资源完整保存startup-verification.json，磁盘余579305.16GiB。动态组批和显存样本记录包含在startup metrics；训练设置维持四卡6000/9000、accum1、workers16/prefetch2、残余0.3、max19000/schedule38539。
- 已原子写startup-verification.json并完成本记录；随后原子写sqrt next-run-ack.json为verified，附匹配run_dir/PID/start_ticks、verified_at/last_step/verified_updates和证据，供原外层沿用同session切到sample。manual.active=false；不建新监控/Codex/定时器/subagent，不写STOP。首次checkpoint、512条验证和SO/ICL生成计划在500步，当前未产生，启动通过不等于生成质量通过。后续持续观察短句额外内容/过早EOS、长句截断/低能量及ICL尾缺词，不把sqrt异常直接当作sample已发生的结论。无代码/配置修改、提交/push/PR、删除或外发。


### 2026-09-11T07:31:40.273177+00:00 — sample首次常规巡检，150步

- 已读playbook、本文、manual、进程/退出状态、073032快照和sqrt next-run-ack。外层已实际切到sample：本run supervision/session-id仍为01a08697-ba78-7cd3-bb9e-b01eba12d724、monitor.pid仍4145999。本目录尚无上一轮review/status，是切换后的首轮；前序ack verified、step50/PID737573/start95204815与本run一致。前次启动验收50步，本快照120步、现场130→150步。manual.active=false，旧phase标签queued-after-sqrt不代表仍未启动，以实际进程和启动证据为准，未修改manual。
- launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系和本sample命令正确，无resume；torchrun及四rank均expandable_segments:True。没有training-exit.json或Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted。实际config与sample源配置一致，仍sample、独立output、四卡6000/9000、accum1、workers16/prefetch2、残余0.3、max19000/schedule38539和原学习率/双模式；initialized记录为step0/epoch0/next_batch0、world4，初始化源为原assembled，assembly_report SHA=32abaefa8cc23b4a82612db59bfc0bc3ba6e837d1cfeffa8274a91624a50def2。
- 当前无COMPLETE/checkpoints/latest、val日志或两模式summary，符合save/eval/audio_every=500计划。尚不能核验sample checkpoint signature、验证CE或生成质量；没有以sqrt产物代替本run，也未提前触发评估。首次500步checkpoint、512条val及SO/ICL各8条评估待后续实际产生。
- 至140步全部14日志点普通CE/sample CE/梯度及其他数值均有限，LR逐点匹配warmup=(step+1)/1000和新参数三倍倍率；收尾150步再次核验有限值、LR和五进程身份。60–140共9点：token first范围6.704248–7.644036、中位7.131748，residual7.548168–7.669140、中位7.592452；sample first6.607360–7.572324、中位7.021815，residual7.557864–7.663595、中位7.584985；grad1.642577–11.652675、中位2.516221。普通CE仍token平均，当前目标first_sample_ce+0.3 residual_sample_ce，不能把两口径混用。
- 第80步step3.215645s，随后90–150回到2.032399–2.284190s；第120步clip前grad11.652675，130/140/150为4.358062/1.686188/2.208141。都没有持续恶化或非有限证据，保持grad_clip=1.0及原设置。60–140 step中位2.242070s、范围2.032399–3.215645，吞吐中位853.889音频秒/墙钟秒、范围592.180–921.787；wait中位0.0003166s、范围0.0002603–0.0003989s。
- 同区间samples341–385、中位359；frame填充95.3333–99.7125%、中位99.1792%，token92.4222–97.6500%、中位96.4917%。日志峰值显存54.004–56.466GiB；四卡63601/63319/63341/63281MiB（各81920），利用率94–100%，compute-apps仅四sample rank。RAM用239GiB、可用1.7TiB、无swap，磁盘余579301.95GiB；未见供数、显存、内存或磁盘压力。
- 收尾150步：token first/residual=6.715783612/7.553920210，sample=6.581952474/7.552807617，目标8.847794759，grad2.208141，LR1.51e-5/4.53e-5，step2.148975s、wait0.0002698s，吞吐852.797。较启动验收50步的token7.795348293/7.721099880、sample7.758584808/7.717225791下降，但不能据早期训练loss证明生成能力。
- 实际执行cat/tail、Python JSON/YAML/proc/初始化与配置核验、全日志有限值/LR/统计/文件状态/磁盘检查及nvidia-smi/free，命令均成功；仅追加本文。训练正常，无需修复、信号、重启或恢复，未改代码/配置/manual/metadata/数据/ack，未建监控/Codex/定时器/subagent，未提交/push/PR、删除或外发。本轮没有音频或ASR证据，不将前序sqrt的短句额外内容、提前EOS、长句低能量或ICL尾缺词当作本sample已发生的问题；这些是首次生成后需检查的项目。继续由原外层监管到19000，本次单轮巡检结束。


## 2026-09-11T07:54:30.484955+00:00 — 主会话加入现有TensorBoard

用户要求更新TensorBoard以查看sample。核对tmux 1:0.0原TensorBoard PID3343951/start88104072，仅在该pane重启TensorBoard；端口仍32001，保留pretrain_1000h、dynamic_10000h、dynamic_10000h_sqrt，新增dynamic_10000h_sample。新PID3600072/start95406338。启动初始目录列表先就绪，scalars尚在加载时一次HTTP404，随后重新读取通过；四个run均可见，sample普通CE/sample CE四条scalar均有数据（最新step500）。证据本run tensorboard-addition.json；训练PID737573身份仍匹配、存活，未启停训练或后台。


### 2026-09-11T08:02:43.676488+00:00 — 首个500步checkpoint/验证完成，ICL评估进行中

- 已读playbook、本文、manual、process/退出状态、080032快照及073032 review/status。前轮快照120/现场150，本轮快照及现场500步；manual=false。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系、本sample命令及无resume正确，torchrun/四rank均expandable_segments:True。无training-exit.json或Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted，上一review exit0。注意主会话07:54:30新增TensorBoard sample的记录，本巡检未操作TensorBoard或训练。
- 首个500 COMPLETE于07:46:06.406370写成，latest指向step-00000500，progress=step500/epoch0/next_batch500、world4、scheduler500/_step_count501，LR5.01e-5/1.503e-4。四distcp各约2.093GB、distributed/.metadata1424894B、四rng各14613B齐全。metadata SHA256=3e049f1e5f31e8cad00da6ae7b6802bf92b224d1cafa7fbe320e642257fd98fc。signature settings/model/eval/seed与实际配置匹配，源/run配置一致；与sqrt19000 signature对照只差loss_reduction=sample，manifest/assembly等相同。这是结构/签名检查，没有恢复加载测试，也未将sqrt checkpoint作为恢复源。
- 全部50个训练日志点普通CE/sample CE/grad和其余数值有限，LR逐点匹配warmup=(step+1)/1000、新参数三倍倍率。500 train token first/residual=5.474766159/7.343992345，sample=5.215709339/7.329707018，目标first_sample_ce+0.3 residual_sample_ce=7.414621445，clip前grad6.876153。160–500共35点：token first范围5.474766–6.714429、中位6.478720，residual7.322506–7.558008、中位7.413643；sample first5.215709–6.613504、中位6.279406，residual7.315862–7.549295、中位7.406676；grad1.434998–10.868082、中位4.863180。均为有限值，保留原grad_clip=1.0，不凭早期梯度波动调参。
- 同区间step中位2.129457s、范围1.979617–2.311514；吞吐中位880.775音频秒/墙钟秒、范围796.465–931.723；wait中位0.0002848s、范围0.0002365–0.0004633。samples318–383、中位363，frame填充94.8833–99.6792%、中位98.2625%，token91.7139–97.9472%、中位95.6028%，峰值显存53.105–56.664GiB。评估现场四卡64813/64003/63963/64023MiB（各81920）、利用率28–60%，compute-apps仅四sample rank；RAM用241GiB、可用1.7TiB、无swap，磁盘余579198.52GiB。没有训练供数/吞吐或资源压力证据。
- val清单512条，500首次验证token first/residual=5.531987723/7.311469188，sample=5.245586902/7.327121645；15残余码本CE有限、范围5.919141–7.512544。普通CE仍token平均，sample逐句等权；本run首轮val，无前轮val趋势可比较，训练CE下降不代表已能正确生成。

|500生成进度|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断|
|---|---|---|---|---|
|speaker_only 8条完成、各4EN/4ZH|1.277778/1.041958|1.361111/1.055944|1.125000/1.047619|3/8|
|icl仍进行中|尚无完整summary|尚无完整summary|尚无完整summary|待全部完成|

- SO summary于07:58:54.380337完成；8metrics/8WAV均读取通过，24kHz单声道、finite、时长相符，目标ID/text/speaker_reference_id/greedy min2与前序sqrt同评估列表匹配，确认参考选择没有突变。以下为ASR及波形统计，未主观试听；低能量定义10ms RMS<0.001。本轮首次8句结果不能代表全验证集或据此决定超参。
- SO03/05/06全部400帧32.00s、truncated=true、未达EOS。SO03（目标6.28s）RMS0.007059、低能量93.4375%、最长连续29.51s，ASR“嗯 wazzz…”；SO05（目标1.62s）RMS0.008141、低能量98.3125%、最长31.44s，ASR“We'll see you in the next video. Bye. You”；SO06（目标3.56s）RMS0.010996、低能量97%、最长31.03s，ASR“And just to make this shit…”重复。实证是长低能量段和触顶，不将ASR幻觉/转写当作这些段落存在有效语音的证据；也不把未达EOS称作提前EOS。这些是sample首次实际观察到的质量问题，后续检查是否随训练持续。
- SO00为23帧1.84s，“I think I'm gonna be weird to be off.”、WER2.25，短句明显额外/错内容。SO01为15帧1.20s，“噢噢噢噢噢噢噢噢噢噢!”、CER1.666667；SO02为72帧5.76s，“This is a little bit of a...”、WER0.8125，时长较长但目标内容未正确转写；SO04为21帧1.68s，“起舟 起舟 起舟”、CER1。SO12为55帧4.40s，“嘻嘻嘻吻吻吻喇”、CER1，当前整体内容未形成，不能仅归为尾部少词。上述非400帧样本最长低能量均不超过0.06s，错误不能都归因于长静音。
- ICL00已完成14帧1.12s，“Here we go.”、WER1/CER0.777778，RMS0.042905、低能量10.7143%、最长0.12s；目标和ICL参考ID/text与前序固定列表匹配。07:59:25仅该条metrics；08:01:34.572540新写成ICL01 generated.wav，1536044B、24kHz单声道32s、finite但所有采样为0（RMS0、100%低能量、最长32s），08:02前检查时metrics/ASR仍待完成。这是本轮需要跟踪的全零输出证据，尚不将未写出的metrics或其EOS/truncation标志推定为已验证。
- 现场到08:01:53仍500步，五进程身份复核通过无退出，ICL01 WAV新增证明评估仍推进；长400帧生成与后处理耗时解释步数暂停，不当作训练卡死，不发信号或重启。后续巡检需补查完整ICL八条summary及评估后真实更新、全零/长低能量是否复现；目前不提前报告双模式评估全部完成。
- 实际执行cat/tail、Python JSON/YAML/proc/checkpoint/SHA/signature/val条数/全日志有限值与LR/区间统计/磁盘/产物时间核验、nvidia-smi/free及soundfile/numpy波形/配对检查，命令均成功；仅追加本文。保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式，未修改代码/配置/manual/metadata/数据/ack，未发信号/恢复/重启，未新建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。本次单轮巡检结束，继续由原外层监管。


### 2026-09-11T08:31:44.750265+00:00 — 500双模式完整评估核验，训练至860步

- 已读playbook、本文、manual/process/退出状态、083032快照与080032 review/status。前轮现场500且ICL进行中，本轮快照830、现场840→860。manual=false；launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系、本sample命令及无resume正确，torchrun/四rank均expandable_segments:True。无training-exit.json或Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted，上一review exit0；实际配置与sample源配置一致，未改设置。
- 最新仍500 COMPLETE，latest、progress500/epoch0/next_batch500、world4、scheduler500/_step_count501通过；四distcp各约2.093GB、distributed/.metadata1424894B、四rng各14613B齐全，metadata SHA256仍3e049f1e5f31e8cad00da6ae7b6802bf92b224d1cafa7fbe320e642257fd98fc，signature settings/model/eval/seed与配置相符。当前还未到1000，无新checkpoint属计划行为。本轮未做恢复加载测试。
- 全历史85日志点至850步普通CE/sample CE/梯度和其他数值有限，LR逐点符合warmup=(step+1)/1000、新参数三倍倍率；收尾860再通过有限值/LR和五进程身份复核。510–850共35点：token first范围2.712783–5.504166、中位4.193337，residual7.194668–7.350883、中位7.295224；sample first2.556687–5.228029、中位3.855301，residual7.203635–7.349567、中位7.285946；clip前grad1.733667–9.997499、中位3.621948。训练CE明显下降，保留token与sample口径，不据训练CE声称生成问题已恢复。
- 同区间step中位2.153650s、范围2.023555–2.393142；吞吐中位874.223音频秒/墙钟秒、范围778.868–942.816；wait中位0.0002896s、范围0.0002519–0.0003835。samples335–415、中位364；frame填充95.5333–99.7583%、中位98.6458%，token92.8722–97.9250%、中位95.8972%，峰值显存53.831–56.902GiB。四卡64815/64003/64043/64023MiB（各81920）、利用率58–75%，compute-apps只有四sample rank；RAM用241GiB、可用1.7TiB、无swap，磁盘余579060.85GiB，没有供数/吞吐或资源故障证据。
- 最近val仍500步，清单512条；token first/residual=5.531987723/7.311469188，sample=5.245586902/7.327121645，15残余码本CE有限、范围5.919141–7.512544。无第二轮val，未把相同结果当作新增改善。

|500完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/全零音频/仅2帧|
|---|---|---|---|---|
|speaker_only|1.277778/1.041958|1.361111/1.055944|1.125000/1.047619|3/8、0/8、0/8|
|icl|0.972222/0.902098|0.972222/0.902098|1.000000/1.000000|7/8、1/8、0/8|

- SO/ICL summary分别于07:58:54.380337/08:18:35.031878完成，两模式16metrics/16WAV读取通过，24kHz单声道、finite、时长一致，目标ID/text/speaker_reference_id/reference_text（若适用）/greedy min2与固定评估列表匹配；summary截断率与逐句计数相符。以下只有ASR与波形统计，没有主观试听；低能量定义10ms RMS<0.001。这里是同一轮500评估的补全，不是连续两个训练阶段都复现。
- ICL01现已metrics完整：400帧32s，eos_reached=false、truncated=true，音频所有采样为0、RMS0、ASR为空，确认上轮观察到的全零输出；这是未结束并触顶，不是提前EOS。ICL02为400帧约32s，RMS0.018340、低能量87.4063%、最长27.87s，ASR“MMMMM MMM You”、WER0.9375。ICL03/04/12均400帧32s，RMS分别1.5572e-5/2.8361e-5/1.9965e-6，全部10ms窗口低于阈值、最长32s，但不是所有样本值严格为0；三条ASR“字幕by索兰娅”不能视为有效内容。
- ICL05为400帧32s，RMS0.000167868、低能量99.7188%、最长31.91s，ASR“Thank you. You”；ICL06同为400帧32s，RMS2.1623e-5、低能量99.9688%、最长31.99s，同样ASR“Thank you. You”。这些接近静音的波形不能由ASR短句推定确实说出了这些话。ICL00是唯一未触顶样本：14帧1.12s、EOS=true，“Here we go.”、WER1/CER0.777778，低能量10.7143%、最长0.12s，仍未匹配目标。ICL12整体近静音且触顶，当前问题超出单纯尾部缺词。
- SO同一批500产物复核与上轮一致：03/05/06为400帧32s，最长连续低能量29.51/31.44/31.03s；没有全零WAV。其余00/01/02/04/12分别23/15/72/21/55帧，转写仍是额外/错词、重复内容，详见前轮逐句记录。有限波形与完整summary代表评估流程产物完整，不代表生成质量合格。
- 判断：500步首次完整评估已确认严重低能量/全零及未结束问题；训练进程、数值和供数正常，评估后已继续360次真实更新/36日志点。当前只有一个checkpoint的8句证据，不能据此确定sample目标或实现有故障，也不能套用sqrt后期两帧EOS的机制；保持设置，在1000步同目标/参考下复核是否持续，若持续再扩大有针对性的诊断。未通过调min_new_frames、max_frames、损失或LR掩盖结果。
- 收尾860：token first/residual=2.798260079/7.216409545，sample=2.581160447/7.217905856，目标first_sample_ce+0.3 residual_sample_ce=4.746532203，grad1.860560，LR8.61e-5/2.583e-4，step2.130178s、wait0.0002596s；身份仍匹配、无退出。
- 执行cat/tail、Python JSON/YAML/proc/checkpoint/SHA/signature/val条数/全历史有限值与LR/统计/磁盘核验、nvidia-smi/free、soundfile/numpy配对和波形检查，命令均成功；仅追加本文。未改代码/配置/manual/metadata/数据/ack，未发信号/重启/恢复，未建新监控/Codex/定时器/subagent、提交/push/PR、删除或外发。本次单轮巡检结束，由原外层继续监管到19000。


### 2026-09-11T09:03:02.312309+00:00 — 1000完整评估、静音问题定向核查，训练至1050步

- 已读playbook、本文、manual/process/退出状态、090032快照与083032 review/status。前轮现场860，本快照1000、现场1000→1050；manual=false。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系、本sample命令/无resume正确，torchrun/四rank均expandable_segments:True。无training-exit.json或Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted，上一review exit0。
- 1000 COMPLETE于08:36:31.424672写成，latest指向step-00001000，progress=step1000/epoch0/next_batch1000、world4、scheduler1000/_step_count1001，LR1e-4/3e-4。四distcp各约2.093GB、distributed/.metadata1424894B、四rng各14613B齐全；metadata SHA256=c515c79842236709d01ed2c125bffea757dd342906c08f2a32090ecaeb719b52。1000/500 signature相同，settings/model/eval/seed与配置相符，run/source配置一致；结构/签名检查不等于恢复加载测试。
- 1000 train token first/residual=2.418890000/7.169093322，sample=2.267373501/7.137852335，目标first_sample_ce+0.3 residual_sample_ce=4.408729201，clip前grad1.581995。870–1000共14点token first范围2.376207–2.757518、中位2.542813，residual7.146118–7.213370、中位7.185013；sample first2.267374–2.571858、中位2.384965，residual7.134924–7.196582、中位7.169220；grad1.377175–2.393275、中位1.950195。全部105日志点至1050普通CE/sample CE/梯度有限，warmup到cosine边界及后续LR逐点核验通过，schedule38539和0.1下限保持。
- 同区间step中位2.146806s、范围2.030765–2.411903；吞吐中位879.183音频秒/墙钟秒、范围781.756–906.385；wait中位0.0002952s、范围0.0002613–0.0004455。samples330–400、中位358.5；frame填充95.65–99.4958%、中位98.1938%，token92.4167–96.9%、中位95.8694%，峰值显存54.553–56.478GiB。四卡64815/64003/64043/64023MiB（各81920）、利用率24–99%，compute-apps仅四sample rank；RAM用241GiB、可用1.7TiB、无swap，磁盘余578921.15GiB，无持续供数/吞吐/资源故障。
- 512条val：1000 token first/residual=2.492695905/7.145083522，sample=2.340373287/7.149227045，均低于500的5.531987723/7.311469188和5.245586902/7.327121645；15残余码本CE有限、范围5.293600–7.509696。两口径分开，验证CE下降不能替代生成完整性判断。

|1000完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/全零WAV/仅2帧|
|---|---|---|---|---|
|speaker_only|0.888889/0.587413|0.888889/0.587413|1.375000/0.976190|2/8、0/8、0/8|
|icl|0.777778/0.580420|0.777778/0.559441|1.375000/0.928571|4/8、1/8、0/8|

- SO/ICL summary分别于08:46:49.668207/09:01:15.829381完成，16metrics/16WAV读取通过，24kHz单声道、finite、时长相符，目标ID/text/speaker_reference_id/reference_text（若适用）/greedy min2与500一致。以下为ASR和波形统计，未主观试听；低能量定义10ms RMS<0.001。SO触顶3→2、ICL7→4，英语WER/CER和中文CER均改善；并非全面持续恶化，但部分目标两轮仍静音/触顶，须继续跟踪。
- SO03/06连续两轮400帧32s、未EOS。SO03 RMS0.007028、低能量97.0625%、最长31.04s（500为29.51s），ASR“你試試看”；SO06 RMS0.078478、低能量60.4375%、最长19.33s（500为31.03s），ASR“And if that reads dude You”。后者低能量段缩短但仍不完整。SO05从400帧32s降为22帧1.76s、“Wait, Flindered UD?”、WER1、低能量9.0909%、最长0.13s；长低能量本轮消失，内容仍错。
- ICL00从500的14帧1.12s变为400帧32s全零WAV、RMS0，ASR“you You”是零音频上的转写，不能当真实语音。ICL01/03连续两轮400帧32s，本轮RMS1.1513e-6/3.3401e-7，全部10ms窗口低于阈值、最长32s；01不再是严格全零，但不能因此认定有效语音恢复，两条ASR均空。ICL06仍400帧32s，RMS0.026170、低能量95.9375%、最长30.63s，ASR“in the Japanese too. You”，尾部内容不完整。这四条均未EOS触顶，不是两帧提前EOS。
- ICL02/04/05/12从500触顶恢复为58/26/18/157帧：02为4.64s、“This doesn't make lawful a group…yeah, y'all.”、WER0.5、最长低能量0.09s；04为2.08s、“这十二年毕竟坚持到我们能”、CER0.692308、最长0.05s；05为1.44s、“Wait for it. It's a new”、基础WER1.5/规范化1.75、最长0.11s。12为12.56s/目标5.28s，“这样子你可能只是我懂看过 那对我 对对对 连我要”、CER0.931034，低能量31.6879%、最长3.87s，虽不再触顶但仍额外/错内容和多秒低能量，不能简化为仅尾部缺词。
- SO00为22帧1.76s、“of the Liggins players.”、WER0.75；01为14帧1.12s、“我只拿完之後好”、CER1.166667；02为55帧4.40s、“This is it, make laugh, blend proof…”、WER0.8125；04为28帧2.24s、“其實那冰淨水的走洋片”、CER0.923077；12为104帧8.32s、“再提一個 如果都靠火擊力的 那就為 0R”、CER0.931034、最长低能量0.03s，主要仍是错/额外内容。
- 两轮静音/触顶后作了定向只读核查：ICL00/01/03/06目标WAV均有限非零，时长1.81/0.98/6.28/3.56s，RMS0.090611/0.057914/0.054465/0.130750；对应目标npz SHA匹配manifest，shape为23/13/79/45帧×16、码值均0–2047范围。参考音频按metrics中的speaker_reference_source直接从原tar解码，全部有限非零，RMS0.077649/0.090068/0.038916/0.154914，无写入/物化数据。目标/参考选择前后相同，未见数据损坏或配对突变证据。
- 阅读qwen3_train/train.py生成/codec/WAV写入路径和model.py next_frame：逐帧argmax生成码，rank0广播，拼接reference codes后codec.decode，再按前缀帧比裁去参考音频、sf.write；没有解码失败后填零WAV的异常回退。保存WAV全零是真实产物观察，但未保存生成码或解码前浮点音频，现证据不能区分生成码、codec及量化的具体贡献，也不能完全排除模型相关裁剪影响。此次未占训练GPU重放，不声称已确定根因。总体触顶数与内容指标改善、多个目标恢复，当前没有可验证的实现修复依据，保持设置；后续重点复查00/01/03/06及12低能量，若停滞或反复再扩大诊断。
- 核查中最初按不存在的evaluation.py/codec.py定位，随后在train.py找到实现；直接读取未物化的speaker_reference_audio路径报一次LibsndfileError，改用已有speaker_reference_source原tar只读解码后通过。该路径不存在符合按需音频协议，非训练错误，未补写或修改数据。所有最终核验命令通过。
- 收尾1050：1000双模式评估后50次真实更新/5日志点正常，token first/residual=2.367807830/7.135634754，sample=2.213505080/7.111113307，目标4.346839072，grad1.968408，LR9.999960604e-5/2.999988181e-4，step2.146355s、wait0.0001749s；五进程身份复核通过，无退出。
- 实际执行cat/tail/rg/sed、Python JSON/YAML/proc/checkpoint/SHA/signature/512条数/全日志有限值/LR/统计/磁盘核验、nvidia-smi/free、soundfile/numpy波形/配对和源tar解码诊断；唯一持久修改为追加本文。未改代码/配置/manual/metadata/数据/ack，未发信号、重启或恢复，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。本次单轮巡检结束。


### 2026-09-11T09:34:27.607861+00:00 — 1500 checkpoint/验证与SO完成，ICL进行中

- 已读playbook、本文、manual/process/退出状态、093032快照与090032 review/status。前轮现场1050，本快照及现场1500步；manual=false。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系和本sample命令/无resume正确，torchrun/四rank均expandable_segments:True。09:33:34再次核对五进程身份通过，无training-exit.json或Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted，上一review exit0。
- 1500 COMPLETE于09:19:11.713212写成，latest指向step-00001500，progress=step1500/epoch0/next_batch1500、world4、scheduler1500/_step_count1501，LR9.996060932e-5/2.998818280e-4。四distcp各约2.093GB、distributed/.metadata1424894B、四rng各14613B完整；metadata SHA256=37c0a073f37134b58e238f7af883f0d3039176781c9d34d9f6dc758293a64795。1500/1000 signature一致，settings/model/eval/seed匹配实际配置，run/source配置相同。检查为结构/签名核验，未做恢复加载测试。
- 至1500共150日志点普通CE/sample CE/梯度及其他数值有限，LR按1000 warmup及38539 cosine（0.1下限）逐点通过。1500 train token first/residual=1.991876779/6.962761389，sample=1.860174140/6.930133540，目标first_sample_ce+0.3 residual_sample_ce=3.939214202，clip前grad1.016347。1010–1500共50点token first范围1.950845–2.408812、中位2.130192，residual6.944696–7.184186、中位7.043572；sample first1.824315–2.273409、中位1.978602，residual6.923633–7.144587、中位7.017226；grad0.957779–1.968408、中位1.305260。
- 同区间step中位2.126266s、范围1.952956–2.533930，吞吐中位889.271音频秒/墙钟秒、范围751.307–957.441，wait中位0.0002762s、范围0.0001713–0.0005171。samples293–396、中位361；frame填充94.0458–99.8167%、中位98.3292%，token92.2444–97.6722%、中位95.7125%，日志峰值显存53.497–56.711GiB。评估现场四卡64815/64103/64083/64103MiB（各81920），瞬时rank0利用率0%、其余100%与评估/后处理同步阶段相容，compute-apps仅四sample rank；RAM用241GiB、可用1.7TiB、无swap，磁盘余578800.70GiB，没有训练供数/吞吐/资源压力证据。
- 512条val：1500 token first/residual=2.057269247/6.942872273，sample=1.926674481/6.931915127，均低于1000的2.492695905/7.145083522和2.340373287/7.149227045；15残余码本CE有限，范围4.767490–7.466783。普通CE仍token口径、sample逐句等权，不把验证下降当作生成完整性证明。

|1500生成进度|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断|
|---|---|---|---|---|
|speaker_only 8条完整、各4EN/4ZH|0.305556/0.230769|0.333333/0.230769|1.625000/0.845238|1/8|
|icl尚在评估|未有完整summary|未有完整summary|未有完整summary|不能以部分计数作最终比例|

- SO summary于09:26:50.159007完成；8metrics/8WAV有限、24kHz单声道、时长相符，目标ID/text/speaker_reference_id/greedy min2与1000一致。本轮已完成的ICL00–04同样通过波形、时长与目标/参考配对核验。以下仅ASR和波形统计，未主观试听；低能量定义10ms RMS<0.001。
- SO触顶从1000的2/8降为1/8，英语错误和中文CER下降；SO06从两轮400帧32s恢复至41帧3.28s，“In the Jeffers' Toot…Stapt.”、WER0.25，最长低能量0.16s。SO03仍400帧32s，500/1000/1500连续三轮；RMS0.009132、低能量91.6563%、最长29.30s（1000为31.04s），ASR“夜期的傍向四前再次下午兩點五四”，内容不完整。SO05为17帧1.36s，“Let's play to Zeewee.”、WER1，500触顶在1000/1500未复现，但短句错词仍在。
- SO00为23帧1.84s，“of the liquefied spears.”、WER0.5；01为14帧1.12s，“附近的I-Bike后”、CER1.333333；02为58帧4.64s，“This is a Mick Laughlin group…”、WER0.125（专名分词）；04为27帧2.16s，“非人人家畢竟先吃了這麼重點”、CER0.692308。SO12从104帧8.32s降至49帧3.92s/目标5.28s，“再見,你更服務都看我幾個英文來追問 Bidyo and Bidyo”、CER0.724138，尾部和主体仍错，最长低能量0.03s；时长缩短不等于完整正确。
- ICL00从1000的400帧全零WAV恢复到17帧1.36s，“that the liquid spears.”、WER0.25、RMS0.047843、最长低能量0.07s；全零问题在该目标本轮未复现。ICL01从500/1000的400帧全零/近零变为177帧14.16s、EOS=true，RMS0.056184、低能量82.5565%、最长11.02s，“特别好是这些小胃眼图想翻 手指那一摆再好”、CER3.166667；虽能结束且非零，但相对0.98s目标仍严重过长、额外内容和低能量，不认定已解决。
- ICL02为58帧4.64s，ASR与目标匹配、WER/CER0，最长低能量0.15s。ICL03仍400帧32s、未EOS，三轮持续，RMS7.7867e-8、所有10ms窗口低于阈值、ASR空；不是严格全零WAV，不据微小非零值推断存在有效语音。ICL04为29帧2.32s，“其市占加畢竟正式了在滿洲門前”、CER0.846154，最长低能量0.04s，未恢复500的触顶状态。
- 09:33:34检查时ICL尚只有00–04五条metrics，无1500 summary；ICL05 generated.wav已于09:33:09.860373写成，24kHz/1.68s、finite非零、RMS0.044970、低能量13.0952%、最长0.21s，但metrics/ASR尚未完成，06/12也待评估。不能把1000的ICL summary冒作1500，不能报告1500全部评估通过。新增05 WAV及此前03/04 metrics证明进程仍在推进，1500步暂停是评估阶段，不是训练卡死。
- 判断：SO触顶进一步减少、ICL00/01有局部恢复、ICL02文字明显改善，同时长ZH03的静音/触顶和01的长低能量仍未解决。上轮已核查这些目标/参考音频、目标codec哈希与生成写入路径，未见数据损坏或失败填零回退；本轮没有新的可验证修复依据，不重复相同输入检查或调整超参。后续补查1500完整ICL、评估后更新，继续聚焦03及01/06/12，不据8句定全局结论。
- 实际执行cat/tail、Python JSON/YAML/proc/checkpoint/SHA/signature/512条数/全日志有限值和LR/统计/磁盘/文件时序核验、nvidia-smi/free及soundfile/numpy波形和配对检查，命令均成功；仅追加本文。保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式；未改代码/配置/manual/metadata/数据/ack，未发信号/重启/恢复，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。本次单轮巡检结束。


### 2026-09-11T10:07:28.873041+00:00 — 2000 checkpoint/验证与SO完成，补齐1500 ICL，2000 ICL进行中

- 已读playbook、本文、manual/process/退出状态、100032快照及093032 review/status。前轮现场1500，本轮快照/现场2000；manual=false。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系和sample命令/无resume通过，torchrun/四rank均expandable_segments:True。10:06:38再次复核五进程身份，无training-exit.json或Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted；上一review exit0。
- 2000 COMPLETE于09:52:49.954198写成，latest指向step-00002000，progress=step2000/epoch0/next_batch2000、world4、scheduler2000/_step_count2001，LR9.984250623e-5/2.995275187e-4。四distcp各约2.093GB、distributed/.metadata1424894B、四rng各14613B齐全。metadata SHA256=52a362145c425ca5fdddee71770c44646db7487038aeb0a0b674ac6ec43da03d；2000/1500 signature一致，settings/model/eval/seed与实际配置一致，run/source配置相同。这是结构/签名检查，未做恢复加载测试。
- 全200个训练日志点普通CE/sample CE/梯度及其余数值有限，warmup/cosine LR逐点核验通过。2000 train token first/residual=1.851268026/6.812799214，sample=1.691930135/6.761611599，目标first_sample_ce+0.3 residual_sample_ce=3.720413615，clip前grad1.014102。1510–2000共50点token first范围1.694777–2.059736、中位1.894811，residual6.782375–6.962647、中位6.880858；sample first1.581840–1.934082、中位1.746232，residual6.758149–6.937550、中位6.833722；grad0.792204–1.484406、中位1.109486。
- 同区间step中位2.120036s、范围1.996437–2.470287；吞吐中位889.342音频秒/墙钟秒、范围769.044–941.003；wait中位0.0002797s、范围0.0002333–0.0004940。samples298–414、中位357；frame填充94.5667–99.6917%、中位98.2083%，token90.7639–97.8139%、中位95.3097%；峰值显存53.234–56.591GiB。现场四卡64815/64103/64083/64103MiB（各81920），rank0瞬时0%、其余100%与评估/后处理同步阶段相容，compute-apps只有四sample rank。RAM用242GiB、可用1.7TiB、无swap；磁盘余578737.16GiB，无持续供数/吞吐/资源故障证据。
- 512条val：2000 token first/residual=1.891865617/6.806212704，sample=1.759102546/6.778841458，均低于1500的2.057269247/6.942872273和1.926674481/6.931915127。15残余码本CE有限、范围4.559579–7.423189，全部val日志数值有限。普通CE仍token口径，sample逐句等权，不将验证下降视为生成完整性的证明。

|完整生成，每组各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断|
|---|---|---|---|---|
|1500 icl（补全上轮）|0.277778/0.118881|0.305556/0.118881|1.375000/1.059524|1/8|
|2000 speaker_only|0.250000/0.174825|0.250000/0.174825|1.500000/0.642857|1/8|
|2000 icl尚在评估|暂无完整summary|暂无完整summary|暂无完整summary|不以部分结果推定最终比例|

- 1500 ICL summary于09:34:53.126716、2000 SO summary于10:00:19.477069写成；两组共16metrics/16WAV核验通过，24kHz单声道、finite、时长相符，目标ID/text/speaker_reference_id/reference_text（适用时）/greedy min2与各自前500步一致，summary样本及截断计数相符。另核验2000已完成ICL00–05六条及06 WAV；以下仅ASR和波形统计，未主观试听。低能量定义10ms RMS<0.001。
- 补齐1500 ICL：上轮待完成05为21帧1.68s，“by to a funder's uni.”、基础WER0.75/规范化1、最长低能量0.21s；06从500/1000的400帧恢复为38帧3.04s，“In the Jeff Dries tube, they opened a door and injera swapped.”、WER0.5、最长0.06s，仍有错词；12从1000的157帧12.56s变为44帧3.52s/目标5.28s，ASR“相信你跟四日夫都看過 既定你們絕尾 B06”、CER0.793103、最长0.02s，长低能量消失但内容/尾部仍不完整。1500 ICL无严格全零WAV、无2帧输出，仅03仍400帧近静音，01有11.02s低能量，00/02/04与前轮检查一致。1500至2000已完成500次真实更新。
- 2000 SO：03从500起连续四轮400帧32s、未EOS，RMS0.010330、低能量89.2813%、最长26.77s（1500为29.30s）。ASR“尤其了半个小时,现在是下午两点五时 起了大概十公里”、CER0.472222，比前轮内容有改善，但末尾剩余电量仍未被转写，严重长低能量/触顶未解决。SO06为43帧3.44s，“and the Jeffries too. They open a door and enter a swamp.”、WER0.166667，最长0.28s；连续两轮未再触顶。
- SO00为20帧1.60s、“the liquid spares.”、WER0.5；01为14帧1.12s、“修正禮拜十號”、CER1、最长低能量0.33s；02为47帧3.76s，ASR除winners→whimmers外接近目标、WER0.0625。04为26帧2.08s、“试一下你已经坚持了这门铁”、CER0.692308；05为20帧1.60s、“spruvers, you need.”、WER1，短句错词持续；12为50帧4.00s、“要訓練一個制服都看我我機 給你們那周圍 呃 Middle Amiddle”、CER0.758621，正文和尾部均错。SO无全零/2帧输出，除03外最长低能量不超过0.33s。
- 2000 ICL已完成00–05：00为16帧1.28s、“the liquid spurs.”、WER0.5、最长低能量0.05s，1000的全零问题连续两轮未复现；01从177帧14.16s降为14帧1.12s，RMS0.088899、最长低能量0.24s，“祝质下一百次后”、CER1，长低能量/额外时长本轮消失，内容仍错。02为50帧4.00004s、“This is a McLoughlin group…”、WER0.0625，主要为专名拼写差异。
- ICL03仍400帧32s、未EOS，RMS6.9647e-8，100%窗口低于阈值、最长32s、ASR为空，连续四轮近静音/触顶；非严格全零，不能据微小非零量称为有效语音，也不是提前EOS。04为24帧1.92s、“其实让毕竟坚持了这门天”、CER0.384615、最长0.02s；05为21帧1.68s、“W.A. Flinders Uni.”、基础/规范化WER/CER均0、最长0.02s，较1500改善。06 WAV于10:06:20.307522新增，3.04s、finite非零、RMS0.134029、最长低能量0.16s，但检查时metrics尚未写出，12亦待完成。10:06:38仍只有六条metrics、无2000 ICL summary，不能报告本轮双模式全部通过；产物持续增加，当前暂停在2000属评估阶段，无卡死证据。
- 判断：2000 SO内容指标继续改善，ICL01过长问题和05短句错词有局部恢复；03四轮持续严重静音/触顶仍是重点。1000时已核查03等目标/参考音频、codec哈希/范围及生成写入路径，未见数据损坏、配对突变或失败填零回退；本轮配对仍一致，其他目标与SO03内容改善，没有新的可验证修复依据。不重复同一输入检查或凭8句调LR/loss/生成限制；保留设置，后续补查2000完整ICL与评估后更新，并继续追踪03及12尾部完整性。未保存生成码和写WAV前浮点信号，03根因仍不能区分模型生成、codec与量化等贡献，未宣称解决。
- 实际执行cat/tail、Python JSON/YAML/proc/checkpoint文件与SHA/signature/512条数/全日志有限值/LR/区间统计/磁盘/文件时序检查、nvidia-smi/free及soundfile/numpy波形与配对核验，最终命令均通过；本轮唯一持久修改为追加本文。保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/ack，未发信号、恢复或重启，未新建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。本次单轮巡检结束。


### 2026-09-11T10:33:28.218424+00:00 — 2500 checkpoint/验证通过，2000双模式完整核验，2500生成进行中

- 已读playbook、本文、manual/process/退出状态、103032快照和100032 review/status。前轮2000，本轮快照及现场2500；manual=false。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系、sample命令与无resume正确，torchrun/四rank均expandable_segments:True。10:32:39再次复核五进程身份通过，无training-exit.json或Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted，上一review exit0。
- 2500 COMPLETE于10:26:10.287080写成，latest指向step-00002500，progress=step2500/epoch0/next_batch2500、world4、scheduler2500/_step_count2501，LR9.964589751e-5/2.989376925e-4。四distcp各约2.093GB、distributed/.metadata1424894B、四rng各14613B完整；metadata SHA256=57138f7212d12bc4409a873ed968c7775a6e9ec51cc22201349905e061056d22。2500/2000 signature相同，settings/model/eval/seed匹配配置，run/source配置相同。这是文件结构/签名核验，未做恢复加载测试。
- 全250日志点普通CE/sample CE/梯度及其余数值有限，LR按1000 warmup和38539 cosine逐点核验通过。2500 train token first/residual=1.751254285/6.704684295，sample=1.600884523/6.646126656，目标first_sample_ce+0.3 residual_sample_ce=3.594722520，clip前grad1.069610。2010–2500共50点token first范围1.696106–1.891418、中位1.775279，residual6.693498–6.837445、中位6.759813；sample first1.572608–1.709295、中位1.632023，residual6.646127–6.762604、中位6.709846；grad0.764029–1.415119、中位0.966252。2000评估后已完成500次真实更新。
- 同区间step中位2.112222s、范围2.020044–2.327350；吞吐中位887.171音频秒/墙钟秒、范围804.348–922.333；wait中位0.0002836s、范围0.0002375–0.0005328。samples308–392、中位352.5；frame填充94.6542–99.8375%、中位97.4667%，token91.5750–97.3667%、中位94.6625%；峰值显存53.680–56.571GiB。现场四卡64815/64103/64083/64103MiB（各81920），rank0瞬时0%、其他100%与评估同步阶段相容，compute-apps仅四sample rank；RAM用241GiB、可用1.7TiB、无swap，磁盘余578670.24GiB，无供数/吞吐/资源故障证据。
- 512条val：2500 token first/residual=1.794291811/6.705610066，sample=1.661040543/6.667411186，低于2000的1.891865617/6.806212704和1.759102546/6.778841458；15残余码本CE有限、范围4.411032–7.374776。全部val日志数值有限。普通CE保持token平均、sample逐句等权，不将CE下降当作生成质量验收。

|最近完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断|
|---|---|---|---|---|
|2000 speaker_only|0.250000/0.174825|0.250000/0.174825|1.500000/0.642857|1/8|
|2000 icl（补齐上轮）|0.138889/0.083916|0.138889/0.083916|1.000000/0.845238|1/8|
|2500两模式|尚无完整summary|尚无完整summary|尚无完整summary|不能以部分样本推定比例|

- 2000 SO/ICL summary分别于10:00:19.477069/10:07:26.357331完成，16metrics/16WAV均读取通过，24kHz单声道、finite、时长相符，目标ID/text/speaker_reference_id/reference_text（适用时）/greedy min2与1500一致；summary样本/截断计数吻合，无严格全零WAV、无2帧输出。以下为ASR与波形统计，没有主观试听；低能量定义10ms RMS<0.001。
- 补齐2000 ICL06：38帧3.04s，“In the Jeffreeze 2, they open a door and enter a swamp.”、WER0.166667、CER0.136364，最长低能量0.16s，1500恢复后连续两轮未触顶，尾部主要内容被转写。12为43帧3.44s/目标5.28s，“相信了一个师父都看我几句美男却为RB6”、CER0.827586、最长低能量0.05s，时长短、正文和video尾部仍错，未恢复1000的长低能量。00–05与上轮检查一致，03仍400帧32s近静音、01长低能量消失、05转写完全匹配。ICL整体EN WER从1500的0.277778降至0.138889，ZH CER从1.059524降至0.845238；部分改善不能掩盖03的问题。
- 2500 SO当前已完成00–05六条，WAV有限、时长/目标/参考/策略与2000相符。00为22帧1.76s，“uh... the liquid spares”、WER0.5，基础CER0.222222/规范化0.277778、最长低能量0.09s；01为12帧0.96s，“我是奶白色后”、CER0.333333、最长0.21s，比2000 CER1改善但仍错。02为51帧4.08s，转写少like、WER0.0625；04为25帧2s，“徐依然畢竟建設了這麼多年”、CER0.692308、最长0.04s。05为25帧2s，“W.A. Flunders-Tuny”、基础WER0.5/CER0.230769、最长0.03s，较2000的WER1改善，专名仍错；该WAV于10:31:53.770587新增并已完成metrics，证明评估仍推进。
- SO03连续五轮400帧32s、未EOS；2500 RMS0.010795、低能量96.9688%、最长30.99s，ASR“出去了半小时”、CER0.888889。相较2000最长26.77s/CER0.472222，本轮退化且尾部大部分缺失。不是提前EOS；严格全零在已核验2500样本中未出现。10:32:39仍2500步，SO尚无06/12完整结果、ICL尚无本轮metrics，两模式summary均缺，不能报告2500生成全部通过或沿用2000结果冒充新结果。
- 对03持续严重低能量作了额外只读对照：读取token/sqrt的2000/2500同目标WAV与metrics，核对目标ID/text及speaker_reference_id相同，ICL reference_text相同。sqrt记录的greedy min2与sample一致；token旧文件未记录generation_policy，因此token仅作现象旁证，不能视为严格同策略消融。未占训练GPU或重放生成。

|03历史现象对照|SO帧数/最长连续低能量|ICL帧数/最长连续低能量|
|---|---|---|
|token 2000|400/17.85s|400/17.32s|
|token 2500|229/12.44s|400/32.00s，RMS4.9248e-8，ASR空|
|sqrt 2000|304/17.53s|400/11.32s|
|sqrt 2500|215/10.98s|400/9.02s|
|sample 2000|400/26.77s|400/32.00s，RMS6.9647e-8，ASR空|
|sample 2500|400/30.99s|本轮待完成|

- 上述同目标在旧实验早期亦有长低能量/触顶，token2500 ICL也出现几乎全静音；现象并非sample独有。不过sample2500 SO的结束与低能量表现明显滞后，不能用旧实验存在问题淡化。1000时已核查目标/参考真实音频、codec哈希/码值与生成裁剪/写入路径，未见输入损坏、配对突变或失败填零回退。本轮增加跨run原始产物对照后，仍无可验证的代码修复依据，不能由单一目标确定loss因果；保留配置继续跟踪03的结束/低能量，以及12尾部、短句00/01/05。2500 ICL与评估后更新待下轮补查；若03继续停滞或影响扩展，再扩大定向生成诊断。生成码和写入前浮点音频未保存，根因仍未确定。
- 对照脚本首次直接读取token旧文件generation_policy触发KeyError；改为识别字段未记录后完成核查。这是只读检查脚本的字段假设问题，不是训练异常，未修改历史产物。其余最终核验命令通过。
- 实际执行cat/tail、Python JSON/YAML/proc/checkpoint文件与SHA/signature/512条数/全历史有限值/LR/区间统计/磁盘及产物时序检查、nvidia-smi/free、soundfile/numpy波形与配对及旧实验03对照；本轮唯一持久修改为追加本文。保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/ack，未发信号、恢复或重启，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。本次单轮巡检结束。


### 2026-09-11T11:02:56.850301+00:00 — 3000 checkpoint/验证通过；2500 ICL首次无触顶，03仍有长低能量

- 已读playbook、本文、manual/process/退出状态、110032快照及103032 review/status。前轮2500，本轮快照及现场3000；manual=false。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系、sample命令/无resume通过，torchrun/四rank均expandable_segments:True。11:02:10复核五进程身份通过，无training-exit.json或Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted；上一review exit0。
- 3000 COMPLETE于10:57:41.667992写成，latest指向step-00003000，progress=step3000/epoch0/next_batch3000、world4、scheduler3000/_step_count3001，LR9.937112734e-5/2.981133820e-4。四distcp各约2.093GB、distributed/.metadata1424894B、四rng各14613B齐全。metadata SHA256=5bc42c9b8134127f3511f9d5dd94ea5cf67f5d3209d5c1dc30504d81f45817b2，3000/2500 signature一致，settings/model/eval/seed与配置一致，run/source配置一致。这是文件结构和签名检查，未做恢复加载测试。
- 全300训练日志点普通CE/sample CE/梯度及其余数值有限，LR按1000 warmup/38539 cosine逐点核验通过。3000 train token first/residual=1.666073256/6.642000372，sample=1.538256100/6.576877049，目标first_sample_ce+0.3 residual_sample_ce=3.511319215，clip前grad1.160899。2510–3000共50点token first范围1.591078–1.798903、中位1.689469，residual6.610094–6.723490、中位6.673591；sample first1.454655–1.663754、中位1.552246，residual6.556041–6.658243、中位6.610417；grad0.682480–1.208971、中位0.907018。2500评估后已完成500次真实更新。
- 同区间step中位2.150617s、范围1.955166–2.387388；吞吐中位876.076音频秒/墙钟秒、范围789.918–942.633；wait中位0.0002839s、范围0.0002420–0.0004125。samples310–415、中位360.5；frame填充94.6417–99.7583%、中位97.9146%，token91.4889–98.4639%、中位95.3472%；峰值显存53.468–56.847GiB。现场四卡64815/64103/64083/64103MiB（各81920）、利用率24–71%，compute-apps仅四sample rank。RAM用265GiB、可用1.7TiB、无swap；较前轮用量上升但仍有充足余量，无相伴供数或吞吐故障。磁盘余578559.05GiB。
- 512条val：3000 token first/residual=1.729489848/6.631176513，sample=1.590959707/6.585265085，均低于2500的1.794291811/6.705610066和1.661040543/6.667411186。15残余码本CE有限、范围4.339453–7.333906，全部val日志数值有限。普通CE仍token平均，sample逐句等权，不以CE下降替代生成完整性判断。

|2500完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断|
|---|---|---|---|---|
|speaker_only|0.305556/0.195804|0.305556/0.202797|1.125000/0.797619|1/8|
|icl|0.277778/0.167832|0.277778/0.167832|1.000000/0.607143|0/8|

- 2500 SO/ICL summary分别于10:33:49.002697/10:39:40.219318完成；16metrics/16WAV全部读取通过，24kHz单声道、finite、时长相符，目标ID/text/speaker_reference_id/reference_text（适用时）/greedy min2与2000一致，summary样本与截断计数相符。两模式无严格全零WAV、无2帧输出。以下为ASR和波形统计，没有主观试听；低能量定义10ms RMS<0.001。
- 2500 ICL03从前四轮400帧32s近静音，首次变为193帧15.44s、EOS=true/truncated=false；RMS0.031730、低能量71.9560%、最长10.99s，ASR“这次设施虚授那辆武士,取了大概10公里,现在剩余的剑量是半分之八成”、CER0.611111（2000为空、CER1）。近乎全静音及未结束在本轮消失，但目标6.28s，仍约2.46倍时长、约11s连续低能量且多处错词，不能认定已解决。2500 ICL因这条恢复而首次0/8触顶，ZH CER从2000的0.845238降至0.607143；EN WER反从0.138889升至0.277778，质量变化并非全面改善。
- ICL00为18帧1.44s，“The liquid spares.”、WER0.5、最长低能量0.07s；01为11帧0.88s，“祝大家晚走”、CER1、最长0.09s，连续两轮无1500的长低能量/过长，内容仍错。02为48帧3.84s、“This is a McLeffling group…”、WER0.0625，主要专名差异；04为23帧1.84s、“70年毕竟建设了这么多年”、CER0.461538，最长0.02s。
- ICL05为18帧1.44s，“way flinders you need.”、基础/规范化WER1、CER0.461538、最长低能量0.29s；2000该句WER0，本轮退化，不能把一次正确视作稳定恢复。06为41帧3.28004s，“In the Jeffrey stoop, they open the door and enter a swamp.”、WER0.25、最长0.13s，尾部主要内容仍被转写，无长低能量复发。12为42帧3.36s/目标5.28s，“相信你跟似乎都看过几点点的绝对videos”、CER0.586207、最长0.01s，较2000 CER0.827586改善但正文缺/错词与尾部不完整仍在，不能仅凭短时长确定EOS机制。
- 补齐2500 SO06/12：06为38帧3.04s，“and the Jeffries to pay open a door and they're swapped.”、WER0.5、最长低能量0.08s，较2000 WER0.166667退化但未触顶；12为45帧3.60s，“首先你可是不是都看過我寄給你們的 最後一段視頻”、CER0.827586、最长0.03s，内容仍错。SO00–05与上一轮检查一致：03连续五轮400帧32s、最长30.99s低能量/CER0.888889，05较2000改善但专名仍错。SO整体EN WER从2000的0.25升至0.305556、ZH CER从0.642857升至0.797619，不以ICL03恢复掩盖SO退化。
- 3000现场已完成SO00–02三条metrics/WAV，与2500同目标/参考/策略、24kHz单声道finite且时长一致。00为19帧1.52s，“Now the liquid spears.”、WER0.25、最长低能量0.11s；01为13帧1.04s，“手致奶擺色後”、CER0.666667、最长0.24s；02为50帧4s、“This is a McAflin group…”、WER0.0625、最长0.13s。11:02:10仍只有这三条，SO03尚无WAV/metrics，ICL尚无本轮产物，两模式3000 summary均未生成。当前处于逐句评估，不将步数暂停视为卡死，也不将2500指标当作3000。3000长句及ICL质量留待完成后判断。
- 判断：2500 ICL03出现明确的波形/结束改善，尚不能称为稳定恢复；SO03仍未解决且两模式部分短句有反复。前两轮已做目标/参考/codec核验、生成写入路径阅读与旧实验同目标波形对照，未有输入损坏、配对突变或失败填零证据；本轮03有局部恢复、无新的实现故障线索，不重复相同诊断或改超参。保持设置，重点跟踪3000 SO/ICL03是否结束、低能量是否缩短，以及05正确性和12尾部；生成码/写入前浮点音频未保存，低能量根因仍未确定。待下轮补齐3000双模式与评估后真实更新。
- 实际执行cat/tail、Python JSON/YAML/proc/checkpoint文件与SHA/signature/512条数/全历史有限值/LR/区间统计/磁盘/产物时序检查、nvidia-smi/free及soundfile/numpy波形和配对核验，命令均成功；本轮唯一持久修改为追加本文。保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/ack，未发信号、恢复或重启，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。本次单轮巡检结束。


### 2026-09-11T11:33:44.654536+00:00 — 3500 checkpoint/验证通过，3000 ICL内容改善但03开头低能量变长

- 已读playbook、本文、manual/process/退出状态、113032快照及110032 review/status。前轮3000，本轮快照3490、现场3500；manual=false。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系、sample命令/无resume通过，torchrun/四rank均expandable_segments:True。11:32:46复核五进程身份通过，无training-exit.json或Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted，上一review exit0。
- 3500 COMPLETE于11:30:48.174536写成，latest指向step-00003500，progress=step3500/epoch0/next_batch3500、world4、scheduler3500/_step_count3501，LR9.901867678e-5/2.970560303e-4。四distcp各约2.093GB、distributed/.metadata1424894B、四rng各14613B齐全；metadata SHA256=a4b49da2e6b44bbc1006635fc600a00a4b8a9bf747d580f6cc12b26254aeee9e。3500/3000 signature一致，settings/model/eval/seed匹配配置，run/source配置一致；为结构/签名检查，未做恢复加载测试。
- 全350训练日志点普通CE/sample CE/梯度及其他数值有限，1000 warmup和38539 cosine LR逐点核验通过。3500 train token first/residual=1.657935641/6.586996180，sample=1.532847979/6.513891387，目标first_sample_ce+0.3 residual_sample_ce=3.487015395，clip前grad0.877016。3010–3500共50点token first范围1.594310–1.718964、中位1.649813，residual6.560744–6.646144、中位6.598146；sample first1.452372–1.568073、中位1.509839，residual6.499939–6.592171、中位6.536455；grad0.713061–1.195994、中位0.843144。3000评估后500次真实更新正常。
- 同区间step中位2.185953s、范围2.028411–2.328296；吞吐中位867.467音频秒/墙钟秒、范围796.222–920.321；wait中位0.0002857s、范围0.0002468–0.0004922。samples297–390、中位352.5；frame填充94.5583–99.7292%、中位98.1250%，token90.3472–97.1528%、中位95.1931%；峰值显存53.227–56.691GiB。四卡64815/64103/64083/64103MiB（各81920）、利用率54–95%，compute-apps仅四sample rank。RAM用273GiB、可用1.7TiB、无swap；从前两轮241→265→273GiB的主机总用量变化值得留意，但非训练进程独占统计，当前没有供数等待或资源耗尽证据，不据此认定训练内存泄漏。磁盘余578513.54GiB。
- 512条val：3500 token first/residual=1.692028917/6.570833493，sample=1.556473359/6.520339243，均低于3000的1.729489848/6.631176513和1.590959707/6.585265085；15残余码本CE有限、范围4.283173–7.293887，全部val日志数值有限。保持token/sample两种口径，不把验证下降当作生成问题已解决。

|3000完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断|
|---|---|---|---|---|
|speaker_only|0.250000/0.146853|0.277778/0.146853|1.125000/0.738095|1/8|
|icl|0.166667/0.055944|0.194444/0.055944|1.250000/0.416667|0/8|

- 3000 SO/ICL summary分别于11:06:00.138535/11:12:35.049029完成；16metrics/16WAV读取通过，24kHz单声道、finite、时长相符，目标ID/text/speaker_reference_id/reference_text（适用时）/greedy min2与2500一致，summary样本和截断计数相符。两模式无严格全零WAV、无2帧输出。以下只依据ASR和波形统计，没有主观试听；低能量定义10ms RMS<0.001。
- SO03从500起连续六轮400帧32s、未EOS，3000 RMS0.010707、低能量92.8125%、最长29.68s，ASR“要起了半箱首先在下午2点50”、CER0.833333。较2500最长30.99s/CER0.888889略好，但仍严重长低能量、后续距离/电量内容未转写，不是提前EOS。SO04为30帧2.40s，“其實央家畢竟建設了這麼多年”、CER0.538462、最长低能量0.02s；05为22帧1.76s，“to a flender's uni.”、基础WER0.5/规范化0.75、最长0.05s，专名仍错。
- SO06为48帧3.84s，“and the Jeffries too, they open a door and Intura swapped.”、WER0.416667、最长低能量0.19s；12为49帧3.92s，“讲起你跟似乎都看过我集体给你们的居然威胁 得对我阿飞溜”、CER0.724138、最长0.06s，尾部与正文仍错。00–02与前轮检查相同：00为1.52s/WER0.25，01为1.04s/CER0.666667，02为4s/WER0.0625。SO整体EN WER及ZH CER较2500略降，但03仍未解决。
- ICL03连续第二轮能结束、未触顶，但从2500的193帧15.44s增加为304帧24.32s（目标6.28s约3.87倍），RMS0.029000、低能量73.3553%、最长17.66s（上轮10.99s）。ASR“现在是下午两点五十七了,大概四公里,现在是公里,现在生意的电量是八十八十。”、CER0.527778（上轮0.611111），文字错误略少却时长/低能量变差，不把0/8触顶等同质量合格。
- 定向补查03低能量所在位置：2500/3000 SO最长段分别为[1.01,32.00]s和[2.32,32.00]s，是尾部；2500/3000 ICL最长段分别为[0.00,10.99]s和[0.00,17.66]s，是开头。两模式的长低能量位置不同，不能统一称作“尾部静音”。这来自保存WAV的10ms窗口统计，不能单独判断是生成码、codec还是前缀裁剪造成；未保存生成码和写入前浮点音频，当前根因仍未确定。
- ICL00为17帧1.36s，“that the liquid spears.”、WER0.25、最长低能量0.07s；01为13帧1.04s，“阻止奶白色后”、CER0.333333、最长0.18s，较2500 CER1改善且无长低能量。02为48帧3.84s、目标文本转写匹配、WER/CER0；04为26帧2.08s，“70年年毕竟坚持了这么多年”、CER0.307692、最长0.03s。05为20帧1.60s，“W Flingers Uni”、WER0.5/CER0.153846、最长0.07s，从2500 WER1改善但仍未复现2000的完全匹配。
- ICL06为49帧3.92s，“In the Jeffree's Tude, they open a door and enter a swap.”、基础WER0.25/规范化0.333333、最长低能量0.14s，主要尾词swamp→swap。12为49帧3.92s/目标5.28s，“而且你跟似乎都看过我寄给你们的绝位 videos啊 videos”、CER0.344828（2500为0.586207）、最长0.03s，本轮ASR包含两次videos，尾部有改善但正文仍错，不宣称全文正确。ICL整体EN基础/规范化WER及ZH CER均较2500下降，ZH WER受词切分等影响反升至1.25，保留全部口径。
- 3500评估目前仅SO00/01完整，00为19帧1.52s，“the liquid spears.”、WER0.25、最长低能量0.11s；01为14帧1.12s，“總是來拜色壽”、CER0.833333，RMS0.063049、最长低能量0.32s。两条WAV均finite非零、24kHz单声道、时长与metrics相符，目标/参考/greedy min2匹配3000。01 WAV于11:32:18.115172新增、随后metrics写成，证明评估继续推进。11:32:46仍3500步，SO仅两条metrics，无summary；ICL尚无本轮产物。不可用3000 summary冒充3500或报告双模式全部通过，当前处于评估而非已证实卡死。
- 判断：3000 ICL多句内容改善、03连续两轮结束，但03开头低能量延长，SO03仍持续严重故障表现。此前已检查同目标/参考原始音频、codec哈希/码值、生成裁剪/写入路径以及旧实验同目标波形；未见输入损坏、配对改变或失败填零证据。本轮进一步定位了低能量段位置，仍无可验证的修复依据，保持设置，不凭8句改变loss/LR/生成限制。继续检查3500完整双模式，重点03开头/尾部低能量与EOS、05短句稳定性、12尾部完整性，以及评估后真实更新。
- 实际执行cat/tail、Python JSON/YAML/proc/checkpoint文件与SHA/signature/512条数/全历史有限值/LR/区间统计/磁盘/产物时序检查、nvidia-smi/free及soundfile/numpy波形/配对/低能量区间定位，命令均成功；唯一持久修改为追加本文。保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/ack，未发信号、恢复或重启，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。本次单轮巡检结束。


### 2026-09-11T12:04:00.044893+00:00 — 训练至3960；3500 ICL03全零触顶复发，04首次两帧EOS

- 已读playbook、本文、manual/process/退出状态、120032快照及113032 review/status。前轮现场3500，本快照3890、现场3920→3960；manual=false。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系、sample命令/无resume通过，torchrun/四rank均expandable_segments:True。12:03:03复核五进程身份通过，无training-exit.json或Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted，上一review exit0。
- 本轮检查时最新仍3500 COMPLETE，latest指向step-00003500，progress3500/epoch0/next_batch3500、world4、scheduler3500/_step_count3501、LR9.901867678e-5/2.970560303e-4；四distcp各约2.093GB、distributed/.metadata1424894B、四rng各14613B齐全。metadata SHA256仍a4b49da2e6b44bbc1006635fc600a00a4b8a9bf747d580f6cc12b26254aeee9e；3500/3000 signature一致，settings/model/eval/seed匹配实际配置，run/source配置一致。未到4000时没有新checkpoint符合计划；检查为文件结构/签名核验，未做恢复加载测试。
- 全396日志点至3960普通CE/sample CE/梯度及其余数值有限，warmup/cosine LR逐点核验通过。3510–3920共42点token first范围1.504583–1.714105、中位1.591609，residual6.512180–6.589396、中位6.549113；sample first1.390531–1.551559、中位1.460712，residual6.444340–6.510479、中位6.478707；clip前grad0.663658–1.081354、中位0.810721。保持普通CE/token平均与sample/逐句等权区分。
- 同区间step中位2.180745s、范围2.016848–2.452783；吞吐中位855.962音频秒/墙钟秒、范围775.968–925.087；wait中位0.0003036s、范围0.0002355–0.0027994，最大仍仅2.8ms，未见持续供数瓶颈。samples315–400、中位359；frame填充94.1667–99.7875%、中位97.4646%，token89.7611–98%、中位94.6708%；峰值显存52.876–56.997GiB。现场四卡64875/64103/64083/64103MiB（各81920），compute-apps只有四sample rank；主机RAM用270GiB、较前轮273GiB下降、可用1.7TiB、无swap，磁盘余578441.61GiB，无资源故障证据。
- 最近val仍3500，清单512条，token first/residual=1.692028917/6.570833493，sample=1.556473359/6.520339243；15残余码本CE有限、范围4.283173–7.293887。尚无4000验证，未将重读3500结果当作新增下降。

|3500完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/严格全零WAV/两帧EOS|
|---|---|---|---|---|---|
|speaker_only|0.138889/0.090909|0.166667/0.090909|1.125000/0.750000|1/8、0/8、0/8|
|icl|0.138889/0.090909|0.138889/0.090909|1.625000/0.761905|1/8、1/8、1/8|

- SO/ICL summary分别于11:38:51.397449/11:45:55.779854完成。16metrics/16WAV可读、24kHz单声道、finite、时长相符，目标ID/text/speaker_reference_id/reference_text（适用时）/greedy min2与3000一致，summary样本/截断计数吻合。finite包含全零波形，不代表语音质量合格。以下为ASR和波形统计，未主观试听；低能量定义10ms RMS<0.001。
- SO03连续七轮400帧32s、未EOS；3500 RMS0.020832、低能量91.1875%，最长连续18.58s位于[0.96,19.54]s。与3000连续尾段[2.32,32]s不同，本轮最长段在中间，低能量总占比仍高，不能仅因最长段缩短就认定恢复。ASR“尤其是板枪丝 现在设计的点亮是什么手指”、CER0.861111（3000为0.833333），内容仍严重错误。
- ICL03在2500/3000连续两轮能结束后，3500再次400帧32s、EOS=false/truncated=true，WAV所有采样严格为0、RMS0、低能量100%、ASR空。相比3000的24.32s且开头17.66s低能量，属于明确反复，并非继续改善；这里是未结束触顶，不是提前EOS。该目标前四轮为近零或低能量，本轮严格全零已由波形确认。
- ICL04为2帧0.16s、EOS=true/truncated=false（目标2.3s），RMS0.002428、低能量93.75%，首次在sample run观察到该两帧提前结束。ASR“字幕by索兰娅”不能视为0.16s近静音输出含有这些有效内容。不能将04两帧EOS与03全零400帧混为同一故障机制。上一轮04为26帧2.08s/CER0.307692，退化明显。
- 对新出现两帧EOS的04作定向只读核查：目标WAV2.3s/24kHz、finite非零、RMS0.077103；目标codec NPZ SHA匹配manifest，shape29×16、码值3–2046。参考按metrics中的speaker_reference_source从原tar解码为35760个24kHz采样（1.49s），finite非零、RMS0.104242。目标ID/text、参考ID/text及min2与3000相同，未见输入损坏或配对改变，未物化/重写数据。
- 阅读train.py:generate_sample和model.py:forward的next_frame路径：只在已生成少于两帧时屏蔽EOS，其后在codec码和EOS中argmax，rank0广播stop，再决定是否追加frame。因此04的两帧记录表示解除EOS屏蔽后的首个决策即结束，不是codec将正常长输出截成两帧；这是由实现和metrics作出的判断，未重放logits。生成码经reference前缀拼接/codec解码/按比例裁剪后写WAV，无异常后填零回退。03没有保存生成码或写入前浮点信号，仍不能区分模型码序列、codec与量化/裁剪的贡献，不能声称已找到全零根因。
- SO02为54帧4.32s、ASR目标匹配、WER/CER0；06为45帧3.6s，“In the Jeffries too, they open a door and enter a swamp.”、WER0.083333、最长低能量0.21s，较3000改善。05为21帧1.68s，“to a flounder's bune.”、基础WER0.75/规范化1、最长0.03s，较3000退化。12为52帧4.16s，“我相信你跟師父都看過 我寄給你們那一絕微調”、CER0.689655、最长0.06s，尾部video仍未正确转写。00为1.52s/WER0.25；01为1.12s/CER0.833333；04为28帧2.24s/CER0.538462，均非两帧或长低能量。
- ICL00为15帧1.2s，“the liquid spears.”、WER0.25、最长低能量0.08s；01为11帧0.88s，“數字帶白色後”、CER0.666667、最长0.07s，短句错词仍反复。02为50帧4.00004s、ASR目标匹配，连续两轮WER/CER0。05为23帧1.84s，“W.A. Flangus, Udni.”、WER0.5/CER0.384615、最长0.04s；06为41帧3.28004s，“In the Jeffees too, they open a door and enter a swamp.”、WER0.166667、最长0.13s，尾部主要内容被转写。12为48帧3.84s，“相信你跟似乎都看過 我就給你們拿去為 a video a video”、CER0.379310、最长0.14s，连续两轮转写有两次video，但正文仍错。
- 相较3000，3500 SO英语WER改善、ZH CER略升；ICL英语WER略降，但ZH CER从0.416667升至0.761905，03和04的严重退化不能被英语改善抵消。本轮核查新增04输入及EOS实现边界，没有可验证的数据/实现修复依据；不通过提高min_new_frames、缩短max_frames或改loss/LR掩盖现象。保留设置，下一轮重点复查4000的03全零/触顶与04两帧是否持续，再据连续证据扩大诊断；也跟踪05短句稳定性和12尾部。此前03输入和旧实验同目标波形已核验，不重复相同输入检查。
- 收尾3960：token first/residual=1.590264655/6.516825407，sample=1.475641761/6.443506210，目标3.408693624，grad0.783498，LR9.862634304e-5/2.958790291e-4，step2.257345s、wait0.0005426s。3500完整评估后460次真实更新/46日志点，训练数值持续正常，身份一致无退出；此检查时最新仍3500，没有提前报告4000验收。
- 核查参考解码时最初误以为decode_emilia_audio返回(audio,sr)，解包报ValueError；阅读实现确认直接返回24kHz数组，随后按正确协议重读验证通过。该错误属于只读核查脚本，非训练异常，未修改源数据或历史产物。
- 实际执行cat/tail/rg/sed、Python JSON/YAML/proc/checkpoint/SHA/signature/512条数/全历史有限值/LR/统计/磁盘核验、nvidia-smi/free、soundfile/numpy逐句波形/配对/低能量区间及原tar参考解码检查，最终命令均通过；唯一持久修改为追加本文。保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/ack，未发信号、恢复或重启，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。本次单轮巡检结束。


### 2026-09-11T12:34:27.728582+00:00 — 训练至4390；4000 ICL04恢复，03仍近静音触顶

- 已读playbook、本文、manual/process/退出状态、123032快照及120032 review/status。前轮现场3960，本快照4300、现场4370→4390；manual=false。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系、sample命令/无resume正确，torchrun/四rank均expandable_segments:True。12:33:39五进程身份复核通过，无training-exit.json或Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted，上一review exit0。
- 4000 COMPLETE于12:04:15.328776写成，latest指向step-00004000，progress=step4000/epoch0/next_batch4000、world4、scheduler4000/_step_count4001、LR9.858916285e-5/2.957674885e-4。四distcp各约2.093GB、distributed/.metadata1424894B、四rng各14613B齐全。metadata SHA256=eab2a24941b1966755bb6a5e1e0f8525ed641dac797608d9a813b167cacf0bef，4000/3500 signature一致，settings/model/eval/seed匹配实际配置，run/source配置一致。这是结构/签名核验，未做恢复加载测试。
- 全439训练日志点至4390普通CE/sample CE/梯度及其他数值有限，warmup/cosine LR逐点核验通过。3970–4370共41点token first范围1.500836–1.635560、中位1.578249，residual6.445544–6.540783、中位6.497703；sample first1.366273–1.510879、中位1.434225，residual6.399539–6.474138、中位6.428567；clip前grad0.624107–0.908557、中位0.759083。
- 同区间step中位2.154289s、范围1.997885–2.322238；吞吐中位869.420音频秒/墙钟秒、范围810.559–920.897；wait中位0.0002870s、范围0.0002022–0.0004409。samples300–401、中位359；frame填充95.1375–99.6542%、中位98.0042%，token90.8444–97.7639%、中位95.0139%；峰值显存53.448–56.777GiB。四卡64875/64103/64083/64103MiB（各81920），compute-apps只有四sample rank；RAM用245GiB、较前轮270GiB下降、可用1.7TiB、无swap；磁盘余578338.60GiB。无持续供数、吞吐或资源故障证据。
- 512条val：4000 token first/residual=1.637756906/6.519569350，sample=1.503301620/6.464371122，均低于3500的1.692028917/6.570833493和1.556473359/6.520339243；15残余码本CE有限、范围4.237174–7.256109，全部val日志数值有限。普通CE仍token平均，sample逐句等权；验证下降不代表生成质量合格。

|4000完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/严格全零WAV/两帧EOS|
|---|---|---|---|---|---|
|speaker_only|0.138889/0.069930|0.138889/0.069930|1.250000/0.714286|1/8、0/8、0/8|
|icl|0.194444/0.118881|0.194444/0.118881|1.125000/0.654762|1/8、0/8、0/8|

- SO/ICL summary分别于12:12:14.158655/12:19:29.877784完成，16metrics/16WAV可读、24kHz单声道、finite、时长相符，目标ID/text/speaker_reference_id/reference_text（适用时）/greedy min2与3500一致，summary样本/截断计数吻合。以下为ASR和波形统计，没有主观试听；低能量定义10ms RMS<0.001。
- SO03连续八轮400帧32s、未EOS，RMS0.008682、低能量96.25%，最长连续30.76s位于[1.24,32.00]s，重新表现为长尾部低能量。ASR“又起了半小时”、CER0.861111，与3500同CER但有效内容仍极少；本轮最长段较3500的18.58s中间低能量段更长，不能称为改善。SO02连续两轮ASR目标匹配、WER/CER0；06为45帧3.60s，“And the Jeffries tube, they open a door and enter a swamp.”、WER0.083333、最长低能量0.16s，尾部主要内容保留。
- ICL03为400帧32s、未EOS，RMS3.4823e-8、低能量100%、最长[0,32]s、ASR空。相较3500严格全零，本轮虽有微小非零采样，但仍全程近静音，不能把“严格全零计数降为0”当作恢复。自3500复发后连续两轮再次触顶；与04的提前EOS分开记录。
- ICL04从3500的2帧0.16s恢复为27帧2.16s，EOS=true、未触顶，RMS0.098564、低能量1.8519%、最长0.02s。ASR“其實人家畢竟堅持了這麼多年”、表面内容与目标相符，但现有简繁敏感评分仍WER1/CER0.384615；不把这一评分等同实际漏掉38%的发音。两帧EOS在本轮未复现，不能据一轮恢复认定稳定。上轮已核验04目标/参考音频、codec哈希/码值、同参考配对及EOS边界，本轮配对仍一致，不重复同一输入诊断。
- SO00为24帧1.92s，“of the liquid spears.”、WER0.25、最长低能量0.11s；01为13帧1.04s，“走進來白色後”、CER0.666667、最长0.26s。04为29帧2.32s，“虛實略畢竟堅持了這麼多年”、CER0.615385、最长0.02s；05为28帧2.24s，“to BWUA, Flanders Uni.”、基础/规范化WER0.75，较目标1.62s多内容且专名错误，最长0.06s。12为51帧4.08s，“有消息你跟師傅都看過 我寄給你們的捐威 videos”、CER0.586207、最长0.06s，正文仍错且仅一次videos。
- ICL00为18帧1.44s，“the liquid spears.”、WER0.25、最长低能量0.07s；01为13帧1.04s，“祝之乃白慈厚”、CER0.833333、最长0.17s。02为52帧4.16s，“This is a McLisling group…”、WER0.0625，前两轮完全匹配本轮出现专名差异。05为23帧1.84s，“Dorévé, Flanders, Unis.”、WER1/CER0.615385、最长0.09s，较3500 WER0.5退化。06为45帧3.6s，“In the Jefferies tube, they open a door and enter a swamp.”、WER0.083333、最长0.09s，主要专名拼写差异。
- ICL12为49帧3.92s/目标5.28s，“相信你跟师傅都看过我寄给你们的卷位 videos”、CER0.310345（3500为0.379310）、最长低能量0.07s。正文错误减少，但相较3000/3500转写出两次video，本轮只剩一次，尾部完整性反复；不凭较低CER或短时长断言整体已正确或确定EOS根因。
- 总体SO英语WER持平、CER下降，ZH CER略降；ICL英语WER从0.138889升至0.194444，ZH CER从0.761905降至0.654762。04恢复没有解决03近静音，05错词与12尾部也不稳定。此前03原始目标/参考、codec哈希/范围、生成裁剪/写入路径及旧实验同目标波形已核验，无输入损坏、配对突变或失败填零证据；生成码和写WAV前浮点信号缺失，03根因仍未确定。本轮没有新的可验证修复依据，保留配置，不提高min_new_frames或任意调整loss/LR来掩盖问题。后续重点跟踪4500的03近静音/触顶、04恢复是否持续、05短句及12尾部；若持续严重现象扩展，再据具体证据扩大生成诊断。
- 收尾4390：token first/residual=1.539008982/6.486294194，sample=1.397035180/6.417644222，目标first_sample_ce+0.3 residual_sample_ce=3.322328447，grad0.713101，LR9.820112358e-5/2.946033707e-4，step2.183307s、wait0.0002785s。4000完整评估后390次真实更新/39日志点正常，五进程身份一致、无退出；此检查时最新仍4000，未提前报告4500。
- 实际执行cat/tail、Python JSON/YAML/proc/checkpoint文件与SHA/signature/512条数/全历史有限值/LR/统计/磁盘/产物时序检查、nvidia-smi/free及soundfile/numpy逐句波形/配对/低能量区间定位，命令均成功；唯一持久修改为追加本文。保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/ack，未发信号、恢复或重启，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。本次单轮巡检结束。


### 2026-09-11T13:02:45.146250+00:00 — 训练至4760；4500 SO首次无触顶，03仍有长低能量

- 已读playbook、本文、manual/process/退出状态、130032快照及123032 review/status。前轮现场4390，本快照4720、现场4740→4760；manual=false。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系、sample命令/无resume通过，torchrun/四rank均expandable_segments:True。13:01:55复核五进程身份通过，无training-exit.json或Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted，上一review exit0。
- 4500 COMPLETE于12:38:28.224114写成，latest指向step-00004500，progress=step4500/epoch0/next_batch4500、world4、scheduler4500/_step_count4501、LR9.808333750e-5/2.942500125e-4。四distcp各约2.093GB、distributed/.metadata1424894B、四rng各14613B齐全。metadata SHA256=27e57056f5b6d548ded7cf6519715725fe523cca527d93f5ee8dfd365c867cc8；4500/4000 signature相同，settings/model/eval/seed匹配配置，run/source配置一致。结构/签名核验通过，未做恢复加载测试。
- 全476训练日志点至4760普通CE/sample CE/梯度及其他数值有限，warmup/cosine LR逐点核验通过。4400–4740共35点token first范围1.465311–1.625902、中位1.545964，residual6.414717–6.499854、中位6.464345；sample first1.342914–1.467076、中位1.409606，residual6.367092–6.428136、中位6.393070；clip前grad0.617579–0.915457、中位0.725975。
- 同区间step中位2.167249s、范围2.040082–2.338055；吞吐中位864.432音频秒/墙钟秒、范围778.938–927.007；wait中位0.0002834s、范围0.0002598–0.0004794。samples313–409、中位360；frame填充94.8292–99.6458%、中位98.2667%，token91.7417–98.0167%、中位95.3333%；峰值显存53.596–56.667GiB。四卡64875/64103/64083/64103MiB（各81920）、利用率79–99%，compute-apps只有四sample rank；RAM用267GiB、可用1.7TiB、无swap，磁盘余578286.63GiB。主机总用量有波动但没有持续供数、吞吐或资源耗尽证据。
- 512条val：4500 token first/residual=1.611098392/6.476324700，sample=1.475196311/6.416788235，均低于4000的1.637756906/6.519569350和1.503301620/6.464371122；15残余码本CE有限、范围4.173558–7.223907，全部val日志数值有限。普通CE仍token平均，sample逐句等权，不把CE下降当作生成完整性验收。

|4500完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/严格全零WAV/两帧EOS|
|---|---|---|---|---|---|
|speaker_only|0.138889/0.090909|0.138889/0.090909|1.125000/0.428571|0/8、0/8、0/8|
|icl|0.166667/0.062937|0.166667/0.062937|1.375000/0.642857|1/8、0/8、0/8|

- SO/ICL summary分别于12:45:09.197155/12:52:26.276072完成；16metrics/16WAV可读、24kHz单声道、finite、时长相符，目标ID/text/speaker_reference_id/reference_text（适用时）/greedy min2与4000一致，summary样本和截断计数吻合。以下为ASR和波形统计，没有主观试听；低能量定义10ms RMS<0.001。
- SO03经过连续八轮400帧触顶，4500首次变为205帧16.4s、EOS=true/truncated=false，RMS0.025600、低能量68.1707%，最长10.95s位于[2.60,13.55]s。ASR“估计的本稿室现在是下午两点五十 10个里厂在身处的电量是百分之八十”、CER0.444444（4000为0.861111），后段电量内容被转写。SO首次0/8触顶，但该句目标6.28s，仍约2.61倍时长、约11s中间低能量及多处错词，不能认定已恢复正常。
- ICL03仍400帧32s、EOS=false/truncated=true，自3500复发起连续三轮触顶。本轮RMS0.009011、低能量97.0313%，最长20.61s位于[11.39,32]s，ASR“又寫了百相詩”、CER0.916667。相较4000全程近零，出现了一小段非低能量输出，说明波形有变化；但约97%仍低能量且目标绝大部分内容缺失，改善有限，仍不是质量合格，也不是提前EOS。
- ICL04为25帧2s、EOS=true，RMS0.097087、低能量2.5%、最长0.04s，“其然階畢竟牽涉了這麼多年”、CER0.615385。3500两帧EOS在4000/4500连续两轮未复现，但本轮内容错误比4000更多；不能只看时长恢复认定全文正确。此前已核查该目标/参考和codec哈希/码值、EOS抑制边界，本轮配对相同，无新的输入故障线索。
- SO00为25帧2s，“of the liquid spurs.”、WER0.5、最长低能量0.08s；01为13帧1.04s，“在我之内白色后”、CER0.666667、最长0.25s。02为58帧4.64s、ASR目标匹配、WER/CER0，连续三轮；04为30帧2.4s、“在西瞻家毕竟坚持了这么多年”、CER0.230769、最长0.02s。05为23帧1.84s、“to a Flinders-Yunte.”、WER0.5、最长0.02s，较4000 WER0.75改善但专名仍错。06为48帧3.84s，“In the Jeffreeze tube, they open a door and enter a swamp.”、WER0.083333、最长0.11s，主要专名拼写差异，尾部内容保留。
- SO12为55帧4.4s/目标5.28s，“我相信你跟似乎都看过我寄给你们的卷位 VDLVDL”、CER0.448276（4000为0.586207）、最长低能量0.03s；尾部转写为重复字母串，不能仅凭重复形态称video发音正确。正文也仍有错词。
- ICL00为18帧1.44s，“the liquid spurs.”、WER0.5、最长低能量0.07s；01为12帧0.96s，“組織耐擺收後”、CER1、最长0.17s，短句错词反复。02为57帧4.56s、ASR目标匹配、WER/CER0，4000专名差异本轮消失；05为22帧1.76s，“WA Flingers Uni”、WER0.75/CER0.076923、最长0.08s，WER受W.A.分词与专名影响，不只看WER判断全部内容错误。06为48帧3.84s，“In the Jeffreeze tube, they open a door and enter a swamp.”、WER0.083333、最长0.09s，尾部主要内容仍被转写。
- ICL12为59帧4.72s，“相信你跟似乎都看我我寄给你们那绝 我 video video”、CER0.241379（4000为0.310345）、最长低能量0.03s。ASR再次包含两次video，较4000只转写一次有改善；正文仍有重复/错词，不能宣称全文正确。整体SO ZH CER从0.714286降至0.428571，ICL EN WER从0.194444降至0.166667、ZH CER从0.654762小降至0.642857；8句固定样本仅反映这些目标，不能外推全局效果。
- 判断：SO03首次结束且低能量缩短、ICL03出现少量非低能量输出、04连续两轮无两帧EOS，存在局部改善；严重长低能量仍集中于03，短句00/01/05及12尾部仍有波动。此前目标/参考、codec哈希/范围、生成裁剪/写入路径与旧实验同目标波形已核查，无输入损坏、配对改变或失败填零证据。生成码/写入前浮点音频未保存，03根因未确定；本轮无可验证修复依据，不重复相同输入核查或改loss/LR/min帧数。保持设置，下轮重点复查5000的03结束与低能量、04恢复持续性及短句/尾部内容。
- 收尾4760：token first/residual=1.461623294/6.420736112，sample=1.368333868/6.365162188，目标first_sample_ce+0.3 residual_sample_ce=3.277882525，grad0.741144，LR9.779043874e-5/2.933713162e-4，step2.270421s、wait0.0002602s。4500完整评估后260次真实更新/26日志点正常，五进程身份一致、无退出；检查时最新仍4500，未提前报告5000。
- 实际执行cat/tail、Python JSON/YAML/proc/checkpoint文件与SHA/signature/512条数/全历史有限值/LR/统计/磁盘/产物时序检查、nvidia-smi/free及soundfile/numpy逐句波形/配对/低能量区间定位，命令均成功；唯一持久修改为追加本文。保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/ack，未发信号、恢复或重启，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。本次单轮巡检结束。


### 2026-09-11T13:33:12.666743+00:00 — 训练至5250；5000双模式无触顶，但ICL两帧EOS扩至00/04

- 已读playbook、本文、manual/process/退出状态、133032快照及130032 review/status。前轮4760，本快照5200、现场5240→5250；manual=false。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系、sample命令/无resume正确，torchrun/四rank均expandable_segments:True。13:32:20复核五进程身份通过，无training-exit.json或Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted，上一review exit0。
- 5000 COMPLETE于13:10:39.898159写成，latest指向step-00005000，progress=step5000/epoch0/next_batch5000、world4、scheduler5000/_step_count5001，LR9.750208629e-5/2.925062589e-4。四distcp各约2.093GB、distributed/.metadata1424894B、四rng各14613B齐全；metadata SHA256=088cf00927c4362c1e4b447374f922748063664eabb374e917d3bfb00f9faa88。5000/4500 signature相同，settings/model/eval/seed匹配配置，run/source配置一致。结构/签名核验通过，未做恢复加载测试。
- 全525训练日志点至5250普通CE/sample CE/梯度及其余数值有限，warmup/cosine LR逐点核验通过。4770–5240共48点token first范围1.467441–1.581967、中位1.521484，residual6.383654–6.506426、中位6.433893；sample first1.347576–1.451718、中位1.385492，residual6.322078–6.392711、中位6.359493；clip前grad0.574447–0.889061、中位0.673212。
- 同区间step中位2.157841s、范围2.048004–2.336298；吞吐中位872.557音频秒/墙钟秒、范围795.335–915.467；wait中位0.0002806s、范围0.0002546–0.0004714。samples321–423、中位356；frame填充94.9875–99.8625%、中位98.1479%，token92–98.4028%、中位95.5833%；峰值显存53.658–56.703GiB。四卡64875/64103/64083/64103MiB（各81920），compute-apps只有四sample rank；RAM用252GiB、可用1.7TiB、无swap，磁盘余578219.59GiB。没有持续供数、吞吐或资源故障证据。
- 512条val：5000 token first/residual=1.587385569/6.439202657，sample=1.454842392/6.375659518，均低于4500的1.611098392/6.476324700和1.475196311/6.416788235；15残余码本CE有限、范围4.150872–7.193744。普通CE仍token平均、sample逐句等权，全部val日志数值有限；CE下降不代表生成问题已解决。

|5000完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/严格全零WAV/两帧EOS|
|---|---|---|---|---|---|
|speaker_only|0.166667/0.125874|0.166667/0.125874|1.125000/0.464286|0/8、0/8、0/8|
|icl|0.166667/0.139860|0.166667/0.139860|1.000000/0.630952|0/8、0/8、2/8|

- SO/ICL summary分别于13:17:21.291803/13:23:04.090846完成；16metrics/16WAV可读、24kHz单声道、finite、时长相符，目标ID/text/speaker_reference_id/reference_text（适用时）/greedy min2与4500一致，summary样本和截断计数吻合。以下为ASR和波形统计，没有主观试听；低能量定义10ms RMS<0.001。
- ICL00首次出现2帧0.16s、EOS=true、未触顶，RMS5.7066e-5、所有10ms窗口低能量、ASR空；目标1.81s，上轮18帧1.44s。ICL04在4000/4500两轮正常帧数后再次2帧0.16s、EOS=true，RMS0.0005021、低能量93.75%、最长0.09s，ASR“字幕by索兰娅”不能当作有效语音证据。两条WAV不是严格全零，但都没有形成目标内容。两帧EOS由00首次出现、04复发构成，不能描述为这两条已经连续两轮两帧。
- 前轮已检查04目标/参考音频、codec哈希/码值及EOS边界；00目标/参考曾在1000静音诊断中检查通过，本轮同目标与参考配对仍一致。按已读实现，min_new_frames=2后EOS解除屏蔽，两帧记录说明第一次允许EOS时即停止，不是400帧触顶或codec将正常长输出裁为两帧。现有产物未保留逐步logits，不能进一步确定EOS偏高的训练原因；本轮未重放或提高min帧数掩盖结果。
- SO03连续第二轮能结束，219帧17.52s，RMS0.030795、低能量69.3493%，最长12.06s在[2.41,14.47]s。ASR“就起了打枪时,现在是下午两点五时 起了大概10公里,现在省于的电量是百十八四”、CER0.388889，比4500的0.444444略降；但时长16.4→17.52s、最长低能量10.95→12.06s，不能称为持续全面改善。目标6.28s，本轮仍约2.79倍时长。
- ICL03从3500–4500连续三轮400帧触顶恢复为225帧18s、EOS=true，RMS0.028031、低能量72.3889%，最长12.92s在开头[0,12.92]s。ASR“再刷刷深这时下午2点50起了大概10公里,现在生育的电量是80%”、CER0.611111（4500为0.916667），部分内容恢复。目标6.28s，仍约2.87倍时长且开头近13s低能量；ICL因此降至0/8触顶，但同时两条提前EOS，不能将无触顶作为质量验收。
- SO00为24帧1.92s，“The liquid spears.”、WER0.25、最长低能量0.11s；01为13帧1.04s，“就是在白色後”、CER0.666667、最长0.28s。02为51帧4.08s，“This is a McClaxling group in which there are winners and losers…”、WER0.125，专名和like缺失；前连续三轮WER0本轮退化。04为30帧2.4s，“愛惜上家畢星見識了這麼多年”、CER0.692308；05为25帧2s，“to WA Flengus Uni.”、WER0.75、最长0.04s，短句错误仍有反复。06为45帧3.6s，ASR目标匹配、WER/CER0、最长0.10s。
- SO12为54帧4.32s，“我相信你跟师父都看过我今天这么专为videos videos”、CER0.413793、最长低能量0.04s，尾部转写两次videos但正文错词/缺词仍在。ICL01为11帧0.88s，“周知來白色後”、CER0.666667、最长0.10s；02为51帧4.08004s，ASR目标匹配、连续两轮WER/CER0；05为22帧1.76s，“W. Flingers, uni.”、WER0.5/CER0.153846、最长0.06s。06为43帧3.44s，ASR目标匹配、WER/CER0、最长0.09s。
- ICL12为49帧3.92s，“相信你更似乎都看我寄给你们的权为videos”、CER0.482759（4500为0.241379）、最长低能量0.14s；上轮两次video本轮只剩一次，正文和尾部退化。SO整体EN WER、ZH CER较4500略升；ICL EN WER仍0.166667，但00完全失败被02/05/06的改善部分抵消，EN CER从0.062937升至0.139860。逐句问题不能由聚合WER持平掩盖。
- 判断：长句03重新能结束但多秒低能量仍严重；ICL提前EOS扩至00并在04复发，05/12内容继续波动。现已核对所有本轮目标/参考配对和波形，前述输入、codec哈希及生成路径检查未发现数据损坏、配对改变或失败填零；没有可验证的实现修复依据。保持设置，明确记录生成质量未通过；下轮重点复查5500的00/04是否继续两帧、03低能量/结束是否稳定、12尾部。若提前EOS连续存在或进一步扩散，再针对逐步生成决策扩大诊断，不凭8句任意调整loss/LR或生成上限。
- 收尾5250：token first/residual=1.538514675/6.434437527，sample=1.362107663/6.333387130，目标first_sample_ce+0.3 residual_sample_ce=3.262123801，grad0.616749，LR9.718348887e-5/2.915504666e-4，step2.113821s、wait0.0002733s。5000完整评估后250次真实更新/25日志点正常，身份一致无退出；此时最新仍5000，未提前报告5500。
- 实际执行cat/tail、Python JSON/YAML/proc/checkpoint文件与SHA/signature/512条数/全历史有限值/LR/统计/磁盘/产物时序检查、nvidia-smi/free及soundfile/numpy逐句波形/配对/低能量区间定位，命令均成功；唯一持久修改为追加本文。保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/ack，未发信号、恢复或重启，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。本次单轮巡检结束。


### 2026-09-11T14:02:50.466001+00:00 — 训练至5700；5500 ICL00恢复、04两帧持续，SO长句退化

- 已读playbook、本文、manual/process/退出状态、140032快照及133032 review/status。前轮5250，本快照5660、现场5680→5700；manual=false。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系、sample命令/无resume正确，torchrun/四rank均expandable_segments:True。14:02:01复核五进程身份通过，无training-exit.json或Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted，上一review exit0。
- 5500 COMPLETE于13:41:08.072560写成，latest指向step-00005500，progress5500/epoch0/next_batch5500、world4、scheduler5500/_step_count5501，LR9.684642680e-5/2.905392804e-4。四distcp各约2.093GB、distributed/.metadata1424894B、四rng各14613B完整。metadata SHA256=4a83cc4ea190658588f976a3bea6cffcf3b39921dd48138e968491fb37be0bf7；5500/5000 signature相同，settings/model/eval/seed匹配配置，run/source配置一致。为结构/签名核验，未做恢复加载测试。
- 全570训练日志点至5700普通CE/sample CE/梯度及其他数值有限，warmup/cosine LR逐点核验通过。5260–5680共43点token first范围1.413563–1.603738、中位1.503674，residual6.355236–6.457049、中位6.404982；sample first1.306942–1.426677、中位1.367556，residual6.290245–6.361727、中位6.325784；clip前grad0.564219–0.769514、中位0.669936。
- 同区间step中位2.166025s、范围2.047059–2.355054；吞吐中位862.816音频秒/墙钟秒、范围809.459–924.566；wait中位0.0002880s、范围0.0002606–0.0004607。samples318–399、中位357；frame填充95.4417–99.4958%、中位98.2083%，token91.9528–97.5750%、中位95.2556%；峰值显存53.303–56.647GiB。四卡64875/64103/64083/64103MiB（各81920），compute-apps仅四sample rank；RAM273GiB、可用1.7TiB、无swap，磁盘余578147.45GiB，无持续供数/吞吐或资源故障。
- 512条val：5500 token first/residual=1.567426871/6.402488951，sample=1.427338980/6.336377241，均低于5000的1.587385569/6.439202657和1.454842392/6.375659518；15残余码本CE有限、范围4.104760–7.164548，全部val日志数值有限。普通CE/token平均与sample/逐句等权分开记录。

|5500完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/严格全零WAV/两帧EOS|
|---|---|---|---|---|---|
|speaker_only|0.361111/0.251748|0.361111/0.223776|1.125000/0.726190|0/8、0/8、0/8|
|icl|0.138889/0.069930|0.138889/0.069930|1.000000/0.500000|0/8、0/8、1/8|

- SO/ICL summary于13:48:43.886083/13:54:34.460786完成，16metrics/16WAV可读、24kHz单声道、finite、时长匹配；目标ID/text/speaker_reference_id/reference_text（适用时）/greedy min2与5000一致，summary样本/截断计数吻合。以下为ASR和波形统计，未主观试听；低能量阈值为10ms RMS<0.001。
- ICL00从两帧恢复为17帧1.36s，“The liquid spears.”、WER0.25、最长低能量0.03s。04仍2帧0.16s、EOS=true，RMS0.003073、低能量87.5%、最长0.08s；5000/5500连续两轮提前EOS，ASR“字幕by索兰娅”不作有效语音证据。未触顶不等于质量正常。
- 针对04持续EOS，追加只读历史对照：在token/sqrt/sample的4500/5000/5500共36条00/04、两模式metrics中核对相同目标ID/text/参考ID及ICL reference_text。下表为ICL帧数，三run的SO00/04同期均正常结束且有23–34帧。sqrt/sample均记录greedy min2；token旧文件未记录策略，零帧仅作早停现象旁证，不能与两帧作严格同策略比较。

|run|ICL00：4500/5000/5500|ICL04：4500/5000/5500|
|---|---|---|
|token|18/0/19|0/0/0|
|sqrt|19/18/18|25/22/23|
|sample|18/2/17|25/2/2|

- 该对照表明早停也曾发生于token同一ICL目标，sqrt同期没有；不能将问题直接归因于sample独有实现错误，也不能据此淡化sample退化。00本轮恢复、04持续，尚未全面扩散。此前00/04输入、参考、codec哈希与EOS边界已核验正常；本轮配对相同，未重放逐步logits，未提高min帧数。因没有可验证修复依据，保留设置并继续跟踪04；生成决策根因仍未确定。
- SO03为372帧29.76s、EOS=true，RMS0.008947、低能量97.2446%，最长尾段28.86s位于[0.90,29.76]s；ASR“而且的半箱是”、CER0.944444。较5000的17.52s/12.06s低能量/CER0.388889明显退化，虽未达400帧，仍几乎全程低能量，不能只凭三轮无触顶称其恢复。
- ICL03为212帧16.96s、EOS=true，RMS0.021702、低能量66.5094%，最长开头11.13s，[0,11.13]s；ASR“有气了半小时,现在是下午2点50,气了大概10公里,现在生意的电量是80%”、CER0.444444。较5000的18s/12.92s/CER0.611111改善，但目标6.28s，仍严重过长和开头低能量。两模式变化方向不同。
- SO00为26帧2.08s，“ID on the liquid spurs.”、WER0.75，额外/错内容增加；01为15帧1.2s、“讓就這奶白色後”、CER0.666667；02为56帧4.48s，“This is a McLaughlin group and which…”、WER0.0625。04为30帧2.4s/CER0.461538；05为27帧2.16s，“See the butters three days soon.”、WER1.5、基础CER1.769231/规范化1.461538，波形非零且无低能量窗口，错误不能归因于长静音。06为47帧3.76s，“On the Jeffreeze tube…enter a swap.”、WER0.25。SO12为47帧3.76s、“我相信你跟四伯都看我 我寄给你们那一句微丢啊”、CER0.586207、最长低能量0.03s，正文和video尾部均退化。
- ICL01为12帧0.96s、“祝就奶白伺候”、CER0.666667；02为51帧4.08004s，yeah→yet、WER0.0625；05为21帧1.68s、“W.A. Flanders-Uni.”、WER0.25/CER0.076923；06为44帧3.52004s、“In the Jeffree stoop…enter a swamp.”、WER0.166667。ICL12为51帧4.08s、“相信你跟師傅都看我我寄給你們內訣為video啊video我”、CER0.310345、最长低能量0.14s；两次video再次出现，但重复/额外词和正文错误仍在。
- 相较5000，SO EN WER从0.166667升至0.361111、ZH CER从0.464286升至0.726190，主要03和短句内容退化；ICL EN WER降至0.138889、ZH CER降至0.5，但04完全失败仍在。8句不足以据此调loss/LR。此前03目标/参考、codec、生成裁剪/写入路径均无损坏或失败填零证据，缺少生成码/写入前浮点信号，低能量根因仍未确定。保持配置，重点追踪6000 SO03/05、ICL04及12尾部；本轮没有声称生成质量通过。
- 收尾5700：token first/residual=1.536879056/6.349678327，sample=1.369885738/6.299487872，目标3.259732100，grad0.679913，LR9.656358147e-5/2.896907444e-4，step2.080673s、wait0.0003998s。5500完整评估后200次真实更新/20日志点正常，身份一致无退出；最新仍5500，未提前报告6000。
- 实际执行cat/tail、Python JSON/YAML/proc/checkpoint/SHA/signature/512条数/全历史有限值/LR/统计/磁盘/产物时序、nvidia-smi/free、soundfile/numpy波形/配对/低能量区间和旧实验EOS记录对照，命令均成功；唯一持久修改为追加本文。保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/ack，未发信号/恢复/重启，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。本次单轮巡检结束。


### 2026-09-11T14:40:34.721568+00:00 — 本轮核验至6380；6000 ICL04连续第三轮两帧EOS

- 已读playbook、本文最新记录、manual/process/退出状态、143032快照及140032 review/status。前轮现场5700，本快照6140，现场首次日志6220→6380（14:39:01 UTC）；manual=false。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032的身份、父子关系及sample命令/无resume均正确；torchrun/四rank均expandable_segments:True。收尾五训练进程start_ticks复核通过，无training-exit.json或训练Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted。上一review exit0，同一外层session/1800秒周期，未另建巡检。
- latest文本指向step-00006000；6000 COMPLETE于14:12:47.721272 UTC写成。progress=step6000/epoch0/next_batch6000、world4、scheduler6000/_step_count6001、LR9.611750689e-5/2.883525207e-4。四distcp分别2093307384/2093524691/2093544414/2093533470B，distributed/.metadata1424894B、四rng各14613B齐全。metadata SHA256=7efbf3d41a8f9f5883cd1fd0f8f7922cf3d72a0b3e677da1f899a1eab770257f。6000/5500 signature相同，settings/model/eval/seed匹配配置，run/source配置一致；为结构与签名检查，未做恢复加载测试。sqrt next-run-ack仍确认前序19000验收及sample从assembled step0启动，50次真实更新已验证。
- 全638训练日志点普通CE/sample CE/梯度有限、warmup/cosine LR逐点通过。5710–6220共52点：token first范围1.435177–1.557569、中位1.494608；residual6.327652–6.409593、中位6.374380；sample first1.301289–1.398630、中位1.350739；residual6.248364–6.324974、中位6.285003；clip前grad0.545282–0.759486、中位0.652417。
- 同区间step中位2.150851s、范围2.034251–2.568331，吞吐中位880.285音频秒/墙钟秒、范围745.044–925.714；wait中位0.0002831s、范围0.0002546–0.0005324。samples316–386、中位360；frame填充95.6042–99.7625%、中位98.4667%；token92.2583–98.0222%、中位95.7333%；峰值显存53.675–56.845GiB。四卡64955/64123/64083/64103MiB（各81920），利用率99–100%，compute-apps仅四sample rank；RAM245GiB、可用1.7TiB、无swap，磁盘余578325.02GiB。无持续吞吐下降、供数阻塞或资源耗尽证据。
- 6000 val对应512条清单：token first/residual=1.542169419/6.369539510，sample=1.408357580/6.299466960，均低于5500的1.567426871/6.402488951和1.427338980/6.336377241。15残余码本CE有限，范围4.078073–7.135528；全历史val数值有限。普通CE为token平均、sample为逐句等权，保持分口径记录。

|6000完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/严格全零WAV/两帧EOS|
|---|---|---|---|---|---|
|speaker_only|0.194444/0.139860|0.194444/0.139860|1.000000/0.500000|0/8、0/8、0/8|
|icl|0.138889/0.076923|0.138889/0.076923|1.625000/0.738095|0/8、0/8、1/8|

- SO/ICL summary于14:19:22.037299/14:25:10.907714 UTC完成；16metrics/16WAV可读，24kHz单声道、finite、时长匹配，目标ID/text/参考ID/ICL reference_text及greedy min2与5500一致，summary样本和截断计数吻合。以下为已有ASR与本轮波形统计，未主观试听；低能量定义为10ms RMS<0.001。
- ICL04仍2帧0.16s、EOS=true/truncated=false：5000/5500/6000连续三轮提前EOS。RMS0.001697、低能量87.5%、最长0.08s；ASR“字幕by索兰娅”不作为有效目标语音。00保持16帧1.28s，“that the liquid spears.”、WER0.25，未再次退为两帧。上轮已扩大到token/sqrt/sample同期00/04共36条同目标/参考记录，证明早停非sample独有现象但不能证明loss无关；此前输入/codec/EOS边界已核验。本轮配对未变、失败未扩展到其他目标，没有新增可验证的输入或实现故障依据，不重复同一输入核查或通过提高min帧数掩盖早停。逐步logits根因仍未确定。
- SO03为209帧16.72s、EOS=true，RMS0.025788、低能量68.2416%，最长11.16s位于[2.42,13.58]s。ASR“我觉得半个小时现在是下午两点五时 这些公里现在生育的电量是凡人之八十”、CER0.361111。较5500的29.76s/最长28.86s/CER0.944444明显改善，后段内容重现；但目标6.28s，仍约2.66倍时长及11秒中间低能量，不能称为正常。
- ICL03为213帧17.04s、EOS=true，RMS0.025495、低能量66.6080%，最长10.70s位于开头[0,10.70]s。ASR“现在是下午2点50,起的大概10公里,乘车生育的电量是80%”、CER0.638889。最长低能量较5500的11.13s略短，但总时长略增、开头“又骑了半个小时”未被转写，CER从0.444444升高，不能仅凭低能量缩短认为整体改善。两模式都没有触顶，也仍没有解决03异常长低能量。
- SO00为25帧2s，“I did the liquid spares”、WER0.75，短句额外/错内容持续；01为13帧1.04s，“到最奶白時候”、CER0.666667。02为49帧3.92s，“This is a macaflin group in which there are winners and losers. Yeah, you know.”、WER0.125，专名错误和like缺失。04为27帧2.16s，“可惜人家畢竟監視了這麼多年”、CER0.538462。05为22帧1.76s，“W.A. Flinders-Uni.”、基础与规范化WER/CER均0；5500的无关内容本轮未复现，但仅一轮匹配。06为46帧3.68s，“and the Jeffreeze tube. They open a door and enter a swamp.”、WER0.166667，尾部保留。
- SO12为46帧3.68s/目标5.28s，“我相信你跟师傅都看我集给你们的视频”、CER0.620690、最长低能量0.03s。目标两次video没有以该形式出现在转写，正文亦错，比5500 CER0.586207略差；无长低能量不代表内容完整。
- ICL01为12帧0.96s，“組織哪一拜之後”、CER1.166667；02为51帧4.08004s，本轮ASR目标匹配、WER/CER0。05为19帧1.52s，“Dodie Flenders-Uni.”、WER0.75/CER0.461538，较5500 WER0.25退化；06为44帧3.52004s，“In the Jeffreeze tube, they open a door and enter a swamp.”、WER0.083333，尾部保留。
- ICL12为55帧4.4s/目标5.28s，“相信你跟師傅都看我 我寄給你們的捐味 Vidio Vidio Vidio Vidio”、WER6/CER0.655172、最长低能量0.03s。相较5500两次video，本轮转写出现四次Vidio，说明除尾部缺词也需跟踪额外重复；该现象是ASR证据，未通过试听确定实际重复次数。中文WER受空格及中英混写分词影响，故同时比较CER和原文。ZH CER从0.5升至0.738095，03/01/12退化，不能仅看总体ICL EN WER持平称质量稳定。
- 判断：训练数值、checkpoint及评估产物流程正常，生成质量仍未通过。SO03/05局部改善，ICL04持续失败及12额外重复仍需跟踪。03生成码和写WAV前浮点信号未保存，前序输入/参考/codec/裁剪路径检查未发现损坏或失败填零，根因尚不明确。本轮没有足以支持代码修复或改变loss/LR/解码的证据；保持配置，下轮重点复核6500的ICL04、两模式03、短句00/05及12尾部完整性/额外重复。
- 收尾6380：token first/residual=1.507982769/6.363708247，sample=1.333301689/6.257499115，目标first_sample_ce+0.3 residual_sample_ce=3.210551424，grad0.672424，LR9.551531172e-5/2.865459352e-4，step2.029760s、wait0.0002588s。6000完整评估后380次真实更新/38日志点正常，身份一致无退出；检查时最新仍6000，未提前报告6500。
- 实际命令：cat/tail/ls、rg/sed、Python JSON/YAML/proc/checkpoint/SHA/signature/512清单条数/有限值/LR/统计/磁盘检查、nvidia-smi/free/df、soundfile/numpy逐句波形与低能量区间统计。只读检查曾有两处工具侧问题：初次rg误用了不存在的src目录，改查qwen3_train/train.py；首次独立LR断言将warmup写为step/1000，与源码(step+1)/1000不符，在step10触发检查脚本AssertionError，核对源码后更正并通过全部638日志点。这些不是训练错误，未修改训练代码。latest为文本指针，已读取内容核验，未当作符号链接作结论。
- 唯一持久修改为追加本文；保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/ack，未发信号、恢复或重启，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。尚未到19000，不执行最终冻结验收或写final-verification。本次单轮巡检结束。


### 2026-09-11T15:03:00.656573+00:00 — 训练至6690；6500 ICL04恢复，长低能量与尾部内容仍不稳定

- 已读playbook、本文、manual/process/退出状态、150032快照及143032 review/status。前轮核验6380，本快照6630，现场6650→6690；manual=false。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系、sample命令/无resume通过；torchrun/四rank均expandable_segments:True。收尾PID/start_ticks/cmdline再次核验通过，无training-exit.json或训练Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted；上一review exit0。
- latest文本指向step-00006500，6500 COMPLETE于14:43:16.593243 UTC写成。progress6500/epoch0/next_batch6500、world4、scheduler6500/_step_count6501、LR9.531660270e-5/2.859498081e-4。四distcp分别2093307384/2093524691/2093544414/2093533470B，distributed/.metadata1424894B、四rng各14613B齐全。metadata SHA256=8fe6963274c6efba4bb9e0f613817888af2b81a88418be74b69bfd34d0387805；6500/6000 signature相同，settings/model/eval/seed匹配配置，run/source配置一致。为结构/签名核验，未做恢复加载测试。
- 全669训练日志点至6690普通CE/sample CE/梯度等数值有限，warmup/cosine LR逐点通过。6390–6650共27点token first范围1.409960–1.527660、中位1.471726；residual6.276988–6.364794、中位6.334499；sample first1.291339–1.364362、中位1.331862；residual6.222490–6.284144、中位6.244241；clip前grad0.549688–0.744910、中位0.617705。
- 同区间step中位2.171529s、范围2.024886–2.270507；吞吐中位868.635音频秒/墙钟秒、范围838.785–936.310；wait中位0.0002844s、范围0.0002475–0.0004612。samples319–421、中位354；frame填充94.8833–99.7542%、中位98.2667%；token92.3583–98.4722%、中位95.4%；峰值显存54.082–56.687GiB。四卡64955/64123/64083/64103MiB（各81920），利用率94–100%，compute-apps仅四sample rank；RAM246GiB、可用1.7TiB、无swap，磁盘余578229.76GiB。没有持续吞吐下降、数据等待或资源故障证据。
- 6500 val对应512条清单：token first/residual=1.524084219/6.336277674，sample=1.385572959/6.263338856，均低于6000的1.542169419/6.369539510和1.408357580/6.299466960。15残余码本CE有限、范围4.049111–7.105460，全部val日志数值有限。普通CE仍为token平均，sample逐句等权，不混用两种口径。

|6500完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/严格全零WAV/两帧EOS|
|---|---|---|---|---|---|
|speaker_only|0.277778/0.153846|0.277778/0.153846|1.250000/0.523810|0/8、0/8、0/8|
|icl|0.111111/0.055944|0.111111/0.055944|1.250000/0.488095|0/8、0/8、0/8|

- SO/ICL summary于14:50:04.369052/14:55:49.219748 UTC完成；16metrics/16WAV可读、24kHz单声道、finite、时长相符。目标ID/text/参考ID/ICL reference_text及greedy min2与6000一致，summary样本和截断计数吻合。以下依据已有ASR和本轮波形统计，未主观试听；低能量定义为10ms RMS<0.001。
- ICL04从5000/5500/6000连续三轮两帧恢复为24帧1.92s、EOS=true，RMS0.099434、低能量3.6458%、最长0.04s。ASR“其實人畢竟堅持了這麼多年”、CER0.461538：大部分内容重新出现，现有评分简繁敏感，另有“人家”中的“家”未被转写；不能据本轮时长恢复宣称全文正确或稳定恢复。ICL00为18帧1.44s，“The liquid spares.”、WER0.5，未复发两帧，内容评分反而较6000 WER0.25变差。上轮前已核验同目标/参考、输入codec和EOS边界并扩大三run历史对照，本轮没有新配对或输入故障线索。
- SO03为203帧16.24s、EOS=true，RMS0.025850、低能量58.1281%，最长9.27s位于[3.64,12.91]s。较6000的16.72s/低能量68.24%/最长11.16s缩短；但目标6.28s，仍严重过长和中间低能量。ASR“分析了巴萨侠是再次下午两更午时 水贡旗现在剩余的电量是百分之八十”、CER0.527778，较6000的0.361111退化，不能用低能量缩短替代内容质量判断。
- ICL03为183帧14.64s、EOS=true，RMS0.022480、低能量69.3989%，最长9.87s位于开头[0,9.87]s。较6000的17.04s/最长10.70s缩短，但低能量比例从66.61%上升。ASR“现在是下午2点50,起了大概10公里,现在设备的点量是八分之八”、CER0.5（6000为0.638889）；开头“又骑了半个小时”仍缺失，电量内容亦错。两模式均无触顶，不代表异常长低能量已解决。
- SO00为25帧2s，“I did the liquid spars”、WER0.75，额外/错内容持续；01为13帧1.04s，“讓這臉白色後”、CER0.666667。02为57帧4.56s，ASR匹配目标、WER/CER0，6000缺like及专名错误本轮消失。04为28帧2.24s，“哎 其实人家毕竟坚持了怎么逗你”、CER0.307692，虽评分降低仍有额外“哎”和尾部错词。
- SO05为24帧1.92s，“to the Bay Flanders Cooney.”、WER1.25/CER1，6000单轮匹配未保持；RMS0.065228、低能量3.125%、最长0.04s，不能归因于长静音。06为45帧3.6s，“On the Jeffys tube, they open a door and enter a swamp.”、WER0.166667，尾部主要内容保留。SO12为51帧4.08s/目标5.28s，“我相信你跟师父都看过我鸡给你们那一卷 为了飞钓 飞钓”、CER0.586207、最长低能量0.10s；尾部有重复音译形态，但不能当作两次video正确发音的证明。
- ICL01为11帧0.88s，“組織來擺色後”、CER0.833333；02为51帧4.08004s，ASR匹配、WER/CER0，连续两轮；05为17帧1.36s，“Doya Flinders, uni.”、WER0.5/CER0.230769，仍有专名开头错误；06为50帧4s，ASR目标匹配、WER/CER0，尾部保留。
- ICL12为44帧3.52s/目标5.28s，“相信你跟师父都看过 寄给你们的权威 videos”、CER0.413793、最长低能量0.09s。6000转写的四次Vidio本轮未复现，但又只剩一次videos，正文还缺/错词；不能仅凭CER从0.655172下降判为完整，尾部缺词与额外重复仍反复。ASR文本不等于已试听确认实际缺词/重复次数。
- 总体相较6000，SO EN WER从0.194444升至0.277778、ZH CER从0.5升至0.523810；ICL EN WER从0.138889降至0.111111、ZH CER从0.738095降至0.488095。ICL04恢复是本轮主要改善，SO05再次退化、03长低能量和12尾部错误仍需持续跟踪；固定8句不足以据此调loss/LR。此前03输入/参考/codec/裁剪路径无损坏或失败填零证据，生成码和写WAV前浮点信号未保存，根因仍未确定。本轮没有可验证的实现修复依据，保持配置；下一轮重点复查7000的04恢复持续性、03低能量及00/05/12内容。
- 收尾6690：token first/residual=1.428119595/6.310878261，sample=1.276662989/6.231015712，目标first_sample_ce+0.3 residual_sample_ce=3.145967703，grad=0.599976，LR=9.499367247e-05/2.849810174e-04，step=2.202655s、wait=0.0002490s。6500完整评估后190次真实更新/19日志点正常，身份一致无退出；最新仍6500，未提前报告7000。
- 实际执行cat/tail、Python JSON/YAML/proc/checkpoint/SHA/signature/512清单条数/全历史有限值/LR/统计/磁盘检查、nvidia-smi/free/df、soundfile/numpy逐句波形与配对/低能量区间检查，均成功。唯一持久修改为追加本文；保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/ack，未发信号、恢复或重启，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。尚未到19000，不执行最终冻结验收或写final-verification。本次单轮巡检结束。


### 2026-09-11T15:32:54.764031+00:00 — 训练至7200；7000 ICL04早停复发，SO03长低能量显著缩短

- 已读playbook、本文、manual/process/退出状态、153032快照及150032 review/status。前轮6690，本快照7130，现场7150→7200；manual=false。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系、sample命令/无resume正确；torchrun/四rank均expandable_segments:True。收尾PID/start_ticks/cmdline再次核验通过，无training-exit.json或训练Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted；上一review exit0，同一外层1800秒session。
- latest文本指向step-00007000，7000 COMPLETE于15:13:55.321665 UTC写成。progress7000/epoch0/next_batch7000、world4、scheduler7000/_step_count7001、LR9.444511634e-5/2.833353490e-4。四distcp分别2093307384/2093524691/2093544414/2093533470B，distributed/.metadata1424894B、四rng各14613B齐全。metadata SHA256=96c892fe04bb1bd908ba1099fc66648b846e4b06237a049747ba74f0a1928d3c；7000/6500 signature相同，settings/model/eval/seed匹配配置，run/source配置一致。为结构/签名核验，未做恢复加载测试。
- 全720训练日志点至7200普通CE/sample CE/梯度等数值有限，warmup/cosine LR逐点通过。6700–7150共46点token first范围1.383892–1.513431、中位1.457547；residual6.222298–6.348318、中位6.305131；sample first1.274417–1.377316、中位1.317107；residual6.156500–6.254087、中位6.221340；clip前grad0.515162–0.806470、中位0.604123。
- 同区间step中位2.131531s、范围2.040028–2.497320；吞吐中位880.710音频秒/墙钟秒、范围764.243–922.077；wait中位0.0002892s、范围0.0002393–0.0004392。samples317–386、中位354；frame填充94.9167–99.7875%、中位98.1229%；token92.0111–97.3250%、中位95.4028%；峰值显存53.542–56.916GiB。四卡64955/64123/64083/64103MiB（各81920），compute-apps仅四sample rank；RAM245GiB、可用1.7TiB、无swap，磁盘余578120.20GiB。快照GPU利用率71–98%，现场两次瞬时读数21–24%，但7160→7170持续推进且step2.109/2.145s、吞吐895.84/883.98，未出现持续吞吐下降或等待恶化；不据瞬时利用率判卡死或改workers。
- 7000 val对应512条清单：token first/residual=1.502932881/6.307091693，sample=1.367976677/6.231864855，均低于6500的1.524084219/6.336277674和1.385572959/6.263338856。15残余码本CE有限、范围4.028944–7.083999，全部val日志数值有限。普通CE为token平均，sample逐句等权，保持分口径记录。

|7000完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/严格全零WAV/两帧EOS|
|---|---|---|---|---|---|
|speaker_only|0.194444/0.097902|0.194444/0.097902|1.375000/0.642857|0/8、0/8、0/8|
|icl|0.083333/0.055944|0.083333/0.055944|1.125000/0.571429|0/8、0/8、1/8|

- SO/ICL summary于15:19:52.267917/15:25:32.999730 UTC完成；16metrics/16WAV可读、24kHz单声道、finite、时长匹配。目标ID/text/参考ID/ICL reference_text及greedy min2与6500一致，summary样本和截断计数吻合。以下为已有ASR和本轮波形统计，未主观试听；低能量定义为10ms RMS<0.001。
- ICL04再次变为2帧0.16s、EOS=true/truncated=false，RMS0.002080、低能量87.5%、最长0.08s。ASR“字幕by索兰娅”不作有效目标语音证据。6500的24帧恢复只维持一轮。补查本run全部14轮04帧数：500起依次400/26/29/24/23/26/2/27/25/2/2/2/24/2，其中3500、5000、5500、6000、7000共五轮两帧EOS；现象反复，不能声称稳定恢复。前序输入/codec/EOS边界和三run历史对照已核验，本轮相同目标/参考，无新输入损坏或配对变化证据；尚缺逐步logits诊断，未提高min帧数掩盖现象。
- SO03为81帧6.48s、EOS=true，接近目标6.28s；RMS0.033815、低能量19.7531%，最长0.78s位于[2.24,3.02]s。相较6500的16.24s/58.13%/最长9.27s，异常长低能量显著缩短。但ASR“飞进了棒匣市深圳市下午2点50 起了大概10公里 健身于点量是80%”、CER0.722222，较6500的0.527778更差。时长恢复与内容错误并存，不视为生成质量合格。
- ICL03为199帧15.92s、EOS=true，RMS0.025784、低能量67.7764%，最长10.51s位于开头[0,10.51]s。相较6500的14.64s/最长9.87s重新加长；ASR“这些棒下车现在是下午2点50,切得大概10公里,现在生育的电量是80%”、CER0.583333（6500为0.5）。两模式在长低能量上再次出现不同方向，ICL03仍严重过长。均未触顶不等于正常。
- SO00为24帧1.92s，“under the liquid spars.”、WER0.5，较6500 WER0.75降低但短句内容仍错；01为14帧1.12s，“裝修正奶白之後”、CER0.833333。02为53帧4.24s，“This is a McLaughlin group in which there are winners and losers, yeah, you know.”、WER0.0625，like再次缺失。04为28帧2.24s，“還是人家畢竟見識了這麼多年”、CER0.538462，较6500退化。
- SO05为24帧1.92s，“2W Flenders Uni”、WER0.75/CER0.230769，较6500 WER1.25改善但并未回到6000匹配状态。06为46帧3.68s，“and the Jeffries tube, they open a door and enter a swamp.”、WER0.083333，主要开头差异，尾部保留。SO12为53帧4.24s/目标5.28s，“我相信你跟师父都看过我祭奖你们那九位 呃 非丢啊 非丢”、CER0.551724、最长低能量0.07s，正文仍有错词，尾部两次音译形态不能当作已确认正确的video发音。
- ICL00为16帧1.28s，“the liquid spears.”、WER0.25，未复发两帧；01为11帧0.88s，“煮至奶白絲厚”、CER0.333333，较6500改善。02为49帧3.92s，ASR匹配、WER/CER0，连续三轮；05为17帧1.36s，“to a Flinders uni.”、WER0.25/CER0.153846；06为45帧3.6s，“In the Jeffreeze tube, they open a door and enter a swamp.”、WER0.083333，尾部保留。
- ICL12为47帧3.76s/目标5.28s，“相信你跟似乎都看过我寄给你们的绝位 videos”、CER0.413793、最长低能量0.06s。6500/7000连续两轮转写仅一次videos，本轮CER持平但正文错误构成不同；6000四次Vidio的重复没有再现，也不能据此认定目标两次video已完整。仍以ASR证据记录，未试听确认实际缺词或重复次数。
- 总体相较6500，SO EN WER从0.277778降至0.194444、ZH CER从0.523810升至0.642857；ICL EN WER从0.111111降至0.083333、ZH CER从0.488095升至0.571429。英语改善不能遮蔽中文04早停复发、03内容错误及12尾部不完整。03此前输入/参考/codec/裁剪路径无损坏或失败填零证据，生成码与写WAV前浮点信号未保存，根因仍未确定。8句和单轮变化不足以调整loss/LR；本轮无可验证的实现修复依据，保持配置。下一轮7500重点检查ICL04复发是否持续、SO03时长改善能否保持、ICL03低能量及短句/12内容。
- 收尾7200：token first/residual=1.442179198/6.321330931，sample=1.317770589/6.219442107，目标first_sample_ce+0.3 residual_sample_ce=3.183603222，grad=0.604779，LR=9.407709646e-05/2.822312894e-04，step=2.272653s、wait=0.0004080s。7000完整评估后200次真实更新/20日志点正常，身份一致无退出；最新仍7000，未提前报告7500。
- 实际执行cat/tail、Python JSON/YAML/proc/checkpoint/SHA/signature/512清单条数/全历史有限值/LR/统计/磁盘检查、两次nvidia-smi/free/df、soundfile/numpy逐句波形与配对/低能量区间检查及04全部历史帧数核查，命令均成功。唯一持久修改为追加本文；保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/ack，未发信号、恢复或重启，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。尚未到19000，不执行最终冻结验收或写final-verification。本次单轮巡检结束。


### 2026-09-11T16:03:50.384082+00:00 — 训练至7710；7500 SO03长低能量复发，ICL04两帧持续

- 已读playbook、本文、manual/process/退出状态、160032快照及153032 review/status。前轮7200，本快照7620，现场7660→7710；manual=false。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系、sample命令/无resume正确；torchrun/四rank均expandable_segments:True。收尾PID/start_ticks/cmdline再次核验通过，无training-exit.json或训练Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted；上一review exit0，同一外层1800秒session。
- latest文本指向step-00007500，7500 COMPLETE于15:43:39.468062 UTC写成。progress7500/epoch0/next_batch7500、world4、scheduler7500/_step_count7501、LR9.350457354e-5/2.805137206e-4。四distcp分别2093307384/2093524691/2093544414/2093533470B，distributed/.metadata1424894B、四rng各14613B齐全。metadata SHA256=c16246c5770231f441aacc249fe7292d22a848e9f5c1f7a8904025ea48fd4383；7500/7000 signature相同，settings/model/eval/seed匹配配置，run/source配置一致。为结构/签名核验，未做恢复加载测试。
- 全771训练日志点至7710普通CE/sample CE/梯度等数值有限，warmup/cosine LR逐点通过。7210–7660共46点token first范围1.359167–1.512246、中位1.438553；residual6.210285–6.329034、中位6.268151；sample first1.246149–1.338018、中位1.297193；residual6.143336–6.230875、中位6.180290；clip前grad0.504675–0.679397、中位0.580829。
- 同区间step中位2.163528s、范围2.059925–2.331215；吞吐中位869.992音频秒/墙钟秒、范围819.281–916.260；wait中位0.0002817s、范围0.0002490–0.0004351。samples317–383、中位357.5；frame填充95.0792–99.8208%、中位97.9542%；token92.2778–97.5778%、中位95.6208%；峰值显存53.570–56.588GiB。四卡64955/64123/64083/64103MiB（各81920），利用率91–100%，compute-apps仅四sample rank；RAM243GiB、可用1.7TiB、无swap，磁盘余578027.84GiB。无持续吞吐下降、供数等待或资源耗尽证据。
- 7500 val对应512条清单：token first/residual=1.492540344/6.276413606，sample=1.352886230/6.199410804，均低于7000的1.502932881/6.307091693和1.367976677/6.231864855。15残余码本CE有限、范围4.005576–7.056655，全部val日志数值有限。普通CE为token平均，sample逐句等权，保持分口径记录；CE下降不作为生成质量验收。

|7500完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/严格全零WAV/两帧EOS|
|---|---|---|---|---|---|
|speaker_only|0.222222/0.111888|0.194444/0.104895|1.125000/0.452381|0/8、0/8、0/8|
|icl|0.166667/0.111888|0.166667/0.111888|1.000000/0.619048|0/8、0/8、1/8|

- SO/ICL summary于15:50:18.281906/15:56:02.919290 UTC完成；16metrics/16WAV可读、24kHz单声道、finite、时长匹配。目标ID/text/参考ID/ICL reference_text及greedy min2与7000一致，summary样本和截断计数吻合。以下依据已有ASR和本轮波形统计，未主观试听；低能量定义为10ms RMS<0.001。
- ICL04仍2帧0.16s、EOS=true/truncated=false，7000/7500连续两轮；RMS0.003673、低能量75%、最长0.09s位于[0.07,0.16]s。ASR“字幕by索兰娅”不作有效目标语音。6500恢复只持续一轮，低能量比例从87.5%降为75%也不能视为内容恢复。前序已核验输入/codec/EOS边界、全历史与三run对照，本轮同目标/参考，无新的输入损坏或配对变化证据；逐步logits根因仍未知，未提高min帧数。
- SO03重新变为203帧16.24s、EOS=true，RMS0.030732、低能量65.4557%，最长9.91s位于[4.00,13.91]s。7000的81帧6.48s/最长0.78s改善未保持，重新约2.59倍目标6.28s。ASR“却去半个小时,现在是下午两点五时,起了大概10公里 现在剩余的电量是80%”、CER0.333333（7000为0.722222）明显改善，但长低能量严重退化；两项需分开判断，不能用较低CER掩盖时长问题。
- ICL03为201帧16.08s、EOS=true，RMS0.026466、低能量66.3557%，最长10.43s位于开头[0,10.43]s。较7000的15.92s/最长10.51s，异常长低能量基本持续。ASR“就起了半個小時現在是下午兩點五十七了大概四公里現在生意值的電量是百分之八十”、CER0.388889（7000为0.583333），开头内容重现但仍有距离/电量错词。内容改善不等于低能量恢复，两模式都未触顶也不等于正常。
- SO00为22帧1.76s，“Either the liquid spurs.”、WER0.5，短句起始/专名内容仍错；01为13帧1.04s，“做真愛擺設後”、CER1，较7000退化。02为54帧4.32s，“There's a McLaughlin group in which there are winners and losers, yeah, you know.”、基础WER0.1875，开头变化且like缺失；整体英语基础/规范化指标分别记录，避免将缩写分词差异当作同口径变化。04为28帧2.24s，“愛許人家畢竟監視了這麼多年”、CER0.538462。
- SO05为27帧2.16s，“GWA, Flinders Uni.”、WER0.5/CER0.076923，较7000 WER0.75改善但开头多/错内容仍在，较目标1.62s长；最长低能量0.04s，不能归因于长静音。06为46帧3.68s，“And the Jeffries tube, they open a door and enter a swamp.”、WER0.083333，尾部保留。SO12为51帧4.08s/目标5.28s，“我相信你的师傅都看过我去给你们那卷为 Fiddle啊Fiddle”、CER0.448276、最长低能量0.04s；正文及两次Fiddle仍有发音/ASR歧义，不能宣称两次video正确。
- ICL00为14帧1.12s，“the liquid spurs.”、WER0.5，较7000缩短、内容评分变差，但不是两帧EOS。01为13帧1.04s，“朱志乃白思厚”、CER0.833333，较7000退化。02为54帧4.32004s，“This is a McCleffling group…”、WER0.0625，此前连续三轮ASR匹配本轮出现专名差异。05为18帧1.44s，“to be a Flinders uni.”、WER0.5/CER0.307692，较7000 WER0.25退化；06为45帧3.6s，“In the Jeffries tube, they open a door and enter a swap.”、WER0.083333，主要末词swamp→swap，尾部末词需继续跟踪。
- ICL12为45帧3.6s/目标5.28s，“相信你跟師父都看過我寄給你們的圍丟”、CER0.689655、最长低能量0.06s。6500/7000两轮一次videos之后，本轮只转写“圍丟”，仍未体现目标两次video；正文与简繁/中英混写评分共同影响CER，但不能据无长静音认定尾部完整。此处为ASR证据，未试听确认实际缺词/音译次数。
- 总体相较7000，SO基础EN WER从0.194444升至0.222222，规范化WER仍0.194444；ZH CER从0.642857降至0.452381，但03长低能量复发。ICL EN WER从0.083333升至0.166667、ZH CER从0.571429升至0.619048；03内容改善未抵消04失败及短句/12退化。此前03输入/参考/codec/裁剪路径无损坏或失败填零证据，生成码及写WAV前浮点信号未保存，根因仍未确定。本轮没有可验证的实现修复依据，不凭8句或单轮变化调loss/LR。保持配置，下一轮8000重点跟踪ICL04、两模式03低能量、短句额外内容、06末词及12尾部。
- 收尾7710：token first/residual=1.433986788/6.303345997，sample=1.308851720/6.189561366，目标first_sample_ce+0.3 residual_sample_ce=3.165720130，grad=0.601584，LR=9.308934487e-05/2.792680346e-04，step=2.069709s、wait=0.0002421s。7500完整评估后210次真实更新/21日志点正常，身份一致无退出；最新仍7500，未提前报告8000。
- 实际执行cat/tail、Python JSON/YAML/proc/checkpoint/SHA/signature/512清单条数/全历史有限值/LR/统计/磁盘检查、nvidia-smi/free/df、soundfile/numpy逐句波形与配对/低能量区间检查，命令均成功。唯一持久修改为追加本文；保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/ack，未发信号、恢复或重启，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。尚未到19000，不执行最终冻结验收或写final-verification。本次单轮巡检结束。


### 2026-09-11T16:34:02.410896+00:00 — 训练至8220；8000 ICL04恢复，ICL03重复转写受长低能量影响

- 已读playbook、本文、manual/process/退出状态、163032快照及160032 review/status。前轮7710，本快照8120，现场8140→8220；manual=false。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系、sample命令/无resume正确；torchrun/四rank均expandable_segments:True。收尾PID/start_ticks/cmdline再次核验通过，无training-exit.json或训练Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted；上一review exit0，同一外层1800秒session。
- latest文本指向step-00008000，8000 COMPLETE于16:14:16.176945 UTC写成。progress8000/epoch0/next_batch8000、world4、scheduler8000/_step_count8001、LR9.249662090e-5/2.774898627e-4。四distcp分别2093307384/2093524691/2093544414/2093533470B，distributed/.metadata1424894B、四rng各14613B齐全。metadata SHA256=2d55e5885d7ff7919b7fe0215b09d2196cccd8a43c7730e8486849143ec03153；8000/7500 signature相同，settings/model/eval/seed匹配配置，run/source配置一致。为结构/签名核验，未做恢复加载测试。
- 全822训练日志点至8220普通CE/sample CE/梯度等数值有限，warmup/cosine LR逐点通过。7720–8140共43点token first范围1.366310–1.484817、中位1.417429；residual6.194212–6.306867、中位6.249174；sample first1.237985–1.349033、中位1.290367；residual6.116837–6.186001、中位6.152775；clip前grad0.497854–0.718653、中位0.556184。
- 同区间step中位2.166914s、范围2.046816–2.277818；吞吐中位871.083音频秒/墙钟秒、范围831.146–922.486；wait中位0.0002816s、范围0.0002275–0.0004715。samples310–397、中位358；frame填充95.475–99.6958%、中位98.1875%；token91.9944–97.6167%、中位95.5833%；峰值显存53.329–56.640GiB。四卡64955/64123/64083/64103MiB（各81920），利用率51–98%，compute-apps仅四sample rank；RAM245GiB、可用1.7TiB、无swap，磁盘余577914.38GiB。无持续吞吐下降、供数等待或资源耗尽证据。
- 8000 val对应512条清单：token first/residual=1.479510700/6.249875376，sample=1.343594705/6.168671310，均低于7500的1.492540344/6.276413606和1.352886230/6.199410804。15残余码本CE有限、范围3.992957–7.026361，全部val日志数值有限。普通CE为token平均，sample逐句等权，保持分口径记录；CE下降不作为生成质量验收。

|8000完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/严格全零WAV/两帧EOS|
|---|---|---|---|---|---|
|speaker_only|0.250000/0.174825|0.277778/0.174825|1.375000/0.392857|0/8、0/8、0/8|
|icl|0.166667/0.097902|0.166667/0.097902|1.500000/0.726190|0/8、0/8、0/8|

- SO/ICL summary于16:20:21.773752/16:26:07.441212 UTC完成；16metrics/16WAV可读、24kHz单声道、finite、时长匹配。目标ID/text/参考ID/ICL reference_text及greedy min2与7500一致，summary样本和截断计数吻合。以下依据ASR和波形统计，未主观试听；低能量定义为10ms RMS<0.001。
- ICL04从7000/7500两轮两帧恢复为26帧2.08s、EOS=true，RMS0.106779、低能量0.9615%、最长0.02s。ASR“其實仁面畢竟堅持了這麼多年”、CER0.538462：目标大部分内容出现，仍有错词/简繁敏感评分。早停本轮未复现，但6500也曾仅恢复一轮，不能称稳定解决。前序输入/codec/EOS边界、全历史和三run对照无新故障依据，本轮配对一致，未提高min帧数。
- SO03为86帧6.88s、EOS=true，接近目标6.28s；RMS0.037181、低能量21.8023%，最长0.74s位于[1.08,1.82]s。相较7500的16.24s/最长9.91s明显缩短，再次出现类似7000的时长改善。ASR“27的半个小时 现在是下午2点50 写了大概 公里 现在剩余的电量是80”、CER0.361111，较7500的0.333333略差，距离数值/开头仍错或缺，不能称全文正确。
- ICL03为207帧16.56s、EOS=true，RMS0.028284、低能量66.7271%，最长10.92s位于开头[0,10.92]s。较7500的16.08s/最长10.43s继续加长；原始ASR把“现在是下午两点五时,起了大概10公里 现在生鱼的电量是80%”完整重复两遍，WER1.5/CER1.027778。该重复此前未据音频确认，因此本轮增加一次有界CPU诊断。
- 只读核查qwen3_train/metrics.py的ASRScorer：faster-whisper small、CPU int8/4threads/1worker、beam5、zh、vad_filter=False、condition_on_previous_text=False。使用已缓存Systran faster-whisper-small快照536b0662742c02347bc0e980a01041f333bce120，local_files_only=True；不下载、不占GPU。先按相同设置转写原始generated.wav，原文与原始metrics逐字相同，重现四段：[0,5]和[8,12]是“现在是下午两点五时,起了大概10公里”，[5,8]和[12,15]是“现在生鱼的电量是80%”。其中前两段完全落在本轮波形判定的10.92秒低能量区间内，时间对齐不可信。
- 再仅在内存中取10.92秒后的5.64秒尾段，scipy.signal.resample_poly按2/3重采样到16k，保留相同ASR选项；转写为“去渣班告示,现在是下午两点五时,起了大概10公里 现在生育的电量是80%”，只有一次后半句，回映原WAV时间为[10.92,14.36]和[14.36,16.52]。诊断CER0.5、WER1；原音频转写/尾段转写分别耗时约1.92/1.44秒（不含模型加载）。这一对照支持原始重复转写受长低能量前缀影响，不能把它认定为生成语音实际重复；也不能由此推断低能量生成根因或宣称内容恢复。原始metrics/summary/WAV均保留不改，尾段诊断分数不替代正式全音频分数，不与跨run指标混用，未自动启用VAD或改ASR配置。
- SO00为24帧1.92s，“I didn't know liquid spears.”、基础WER0.75，额外/错内容加重，规范化展开缩写后整体EN WER更高，分别记录。01为13帧1.04s，“裝置打敗之後”、CER1。02为53帧4.24s，“This is a McLaughlin group in which there are winners and to losers, yet, you know.”、WER0.1875，缺like、多to和yeah→yet。04为28帧2.24s，“还许人家彼性坚实了这么多年”、CER0.384615。
- SO05为27帧2.16s，“to W.A. Flinders Uni.”、WER0.25/CER0.153846，专名主体更接近但多to；06为45帧3.6s，“and the Jeffreeze tube. They open a door and enter a swamp.”、WER0.166667，末词保留。SO12为56帧4.48s/目标5.28s，“我相信你跟师傅都看过我机盖 你们拿绝位 videos啊videos”、CER0.310345、最长低能量0.05s，相较7500的Fiddle两次，本轮两次videos进入转写，但正文仍有错词。
- ICL00为13帧1.04s，“They're liquid spears.”、WER0.5，比7500再短一帧，但不是两帧EOS；01为12帧0.96s，“煮至奶白色後”、CER0.166667，仅后/後简繁差异，评分不能当作发音错误比例。02为47帧3.76s，“This is a McLaughlin group in which there are winners and losers dead, you know.”、WER0.125，缺like及yeah→dead；05为20帧1.6s，ASR“W.A. Flinders-Uni.”匹配、WER/CER0；06为42帧3.36s，“And the Jeffreeze tube, they open a door and enter a swamp.”、WER0.166667，7500的末词swap本轮恢复swamp，但开头与专名仍错。
- ICL12为43帧3.44s/目标5.28s，“相信你跟师父都看过 几年的绝位 videos”、CER0.551724、最长低能量0.04s。相较7500的“圍丟”，本轮又出现一次videos，但正文“我寄给你们”等内容没有被正确转写、第二次video仍未体现，不能据较低CER判为完整。短句和12诊断仍以ASR证据记录，未试听确认。
- 相较7500，SO基础EN WER从0.222222升至0.25、规范化从0.194444升至0.277778，ZH CER从0.452381降至0.392857；ICL EN WER持平0.166667、ZH CER从0.619048升至0.726190，后者受03已复现的ASR重复明显影响。04/05局部恢复与03/12问题并存。本轮新增CPU诊断缩小了03重复转写的解释范围，但异常低能量本身、反复EOS的生成根因仍未知；前序输入/参考/codec/裁剪路径无损坏或失败填零证据，缺生成码和写WAV前浮点信号。无可验证训练实现修复依据，保持配置；下一轮8500跟踪04恢复、SO03时长稳定性、ICL03低能量与评分重复、短句及12完整性。
- 收尾8220：token first/residual=1.462934877/6.257356558，sample=1.322979720/6.154107044，目标first_sample_ce+0.3 residual_sample_ce=3.169211833，grad=0.645869，LR=9.203221680e-05/2.760966504e-04，step=2.229286s、wait=0.0002515s。8000完整评估后220次真实更新/22日志点正常，身份一致无退出；最新仍8000，未提前报告8500。
- 实际执行cat/tail、Python JSON/YAML/proc/checkpoint/SHA/signature/512清单条数/有限值/LR/统计/磁盘检查、nvidia-smi/free/df、soundfile/numpy波形/配对/低能量检查、rg/sed查看ASRScorer、读取HF缓存常量定位现存模型、faster-whisper CPU原音频与内存尾段复核。常规检查及CPU复核成功；最初缓存检索未匹配且printenv因HF_HOME/HF_HUB_CACHE未设置返回1，随后用huggingface_hub.constants.HF_HUB_CACHE取得真实缓存路径后成功，与训练无关。
- 唯一持久修改为追加本文；保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/ack/评估原件，未发信号、恢复或重启，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。尚未到19000，不执行最终冻结验收或写final-verification。本次单轮巡检结束。


### 2026-09-11T17:03:52.080255+00:00 — 训练至8640；8500 SO03重新触顶，ICL04早停复发

- 已读playbook、本文、manual/process/退出状态、170032快照及163032 review/status。前轮8220，本快照8550，现场8580→8640；manual=false。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系、sample命令/无resume正确；torchrun/四rank均expandable_segments:True。收尾PID/start_ticks/cmdline再次核验通过，无training-exit.json或训练Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted；上一review exit0，同一外层1800秒session。
- latest文本指向step-00008500，8500 COMPLETE于16:44:19.819372 UTC写成。progress8500/epoch0/next_batch8500、world4、scheduler8500/_step_count8501、LR9.142302304e-5/2.742690691e-4。四distcp分别2093307384/2093524691/2093544414/2093533470B，distributed/.metadata1424894B、四rng各14613B齐全。metadata SHA256=ed94f0aaeba4d63cf9ef06c41cc9497fd6329a742d98319aca48422be87f67fc；8500/8000 signature相同，settings/model/eval/seed匹配配置，run/source配置一致。为结构/签名核验，未做恢复加载测试。
- 全864训练日志点至8640普通CE/sample CE/梯度等数值有限，warmup/cosine LR逐点通过。8230–8580共36点token first范围1.348245–1.461201、中位1.406366；residual6.160929–6.256023、中位6.216100；sample first1.236680–1.330182、中位1.273230；residual6.089232–6.150074、中位6.122313；clip前grad0.503783–0.648073、中位0.548266。
- 同区间step中位2.173185s、范围2.052863–2.366240；吞吐中位864.663音频秒/墙钟秒、范围800.595–929.745；wait中位0.0002907s、范围0.0001894–0.0020505。samples329–405、中位366.5；frame填充95.6333–99.7125%、中位98.5625%；token92.2694–97.9694%、中位95.7222%；峰值显存53.775–56.808GiB。四卡64955/64123/64203/64103MiB（各81920），compute-apps仅四sample rank；RAM250GiB、可用1.7TiB、无swap，磁盘余577877.11GiB。现场GPU瞬时利用率19–24%，但训练持续推进、step/吞吐及等待稳定，无持续供数或资源故障证据；单次2ms等待无需改workers。
- 8500 val对应512条清单：token first/residual=1.462024249/6.216999643，sample=1.326418144/6.134436578，均低于8000的1.479510700/6.249875376和1.343594705/6.168671310。15残余码本CE有限、范围3.961133–6.995953，全部val日志数值有限。普通CE为token平均，sample逐句等权，分口径记录；CE下降不代表生成完整性或质量通过。

|8500完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/严格全零WAV/两帧EOS|
|---|---|---|---|---|---|
|speaker_only|0.333333/0.188811|0.361111/0.188811|1.000000/0.809524|1/8、0/8、0/8|
|icl|0.111111/0.055944|0.111111/0.055944|1.125000/0.488095|0/8、0/8、1/8|

- SO/ICL summary于16:53:00.205854/16:58:39.340328 UTC完成；16metrics/16WAV可读、24kHz单声道、finite、时长匹配。目标ID/text/参考ID/ICL reference_text及greedy min2与8000一致，summary样本和截断计数吻合。以下为ASR和波形统计，未主观试听；低能量定义为10ms RMS<0.001。
- ICL04从8000的26帧再次退为2帧0.16s、EOS=true/truncated=false，RMS0.002253、低能量87.5%、最长0.08s。ASR“字幕by索兰娅”不作为有效语音证据；8000恢复仍仅一轮。与之前反复早停属于同一目标/参考/策略，输入/codec/EOS边界及历史对照已核验，无新配对或输入故障依据，逐步logits根因未确定。
- SO03重新达到400帧32s、EOS=false/truncated=true，8000的6.88s时长改善未保持；RMS0.042679、低能量52.4688%，最长16.75s位于[1.21,17.96]s。ASR“而且他爸想說是 字幕by索兰娅”、CER0.972222，目标6.28s的大部分内容无有效转写。低能量比例比此前某些失败轮次低，并不证明有更多有效语音；本轮后半段有幅度但无目标内容证据，故增加有界CPU诊断。
- SO03分段波形：[0,1.21]s RMS0.045301/peak0.382355；[1.21,17.96]s RMS3.49285e-5/peak0.002197；[17.96,32]s RMS0.063046/peak0.166931，三段均无abs>=0.999削波。末段Welch(nperseg4096)最强五频点约398.44/498.05/222.66/404.30/251.95Hz，合占功率39.17%，谱平坦度0.000706，呈明显集中频谱；这些统计不足以单独确定噪声/语音类别或codec根因，但不能将非低能量段直接称为有效语音。
- 用与原评估相同的faster-whisper small CPU int8、4threads/1worker、beam5/zh、vad_filter=False、condition_on_previous_text=False，对原WAV与内存中17.96秒后的14.04秒尾段分别复核。模型使用现存快照536b0662742c02347bc0e980a01041f333bce120/local_files_only=True，不下载、不占GPU；尾段通过resample_poly 2/3转换16k。原WAV逐字重现原转写，段[0,1.28]为“而且他爸想說是”，[30,32]为“字幕by索兰娅”（no_speech_prob0.8963）；尾段仍仅“字幕by索兰娅”，回映[17.96,31.96]，no_speech_prob0.4417。末段没有得到有效目标内容，不能把无关字幕文本当作真实语音或内容恢复。原始WAV/metrics/summary未修改，诊断没有替代正式评分或改变ASR/VAD配置。
- ICL03为196帧15.68s、EOS=true，RMS0.026642、低能量68.4311%，最长10.51s位于开头[0,10.51]s。较8000的16.56s/最长10.92s略短，但仍约2.50倍目标时长。ASR“这八小时现在是下午两点五十七的大概是公里现在剩余的电量是百分之八十”、CER0.222222。8000的整段重复本轮未复现；上轮原WAV与去低能量前缀CPU对照已说明ASR重复不等于生成重复，本轮不重复同一诊断，也不把CER降低解释为长低能量已修复。
- SO00为26帧2.08s，“I know I'm the liquid in spurs.”、WER1.25，额外内容比8000更多；01为13帧1.04s，“早智奶杯之后”、CER0.666667。02为55帧4.4s，“Does it a McLaughlin group in which like there are winners and losers yet, you know?”、WER0.1875，like重现但开头/yeah仍错。04为28帧2.24s，“還洗身 娘畢竟現實了這麼多年”、CER0.692308。
- SO05为26帧2.08s，“Tikawee Flanders, uni.”、WER0.75/CER0.538462，较8000退化；06为47帧3.76s，“of the Jeffries tube. They open a door and enter a swamp.”、WER0.083333，尾部保留。SO12为46帧3.68s/目标5.28s，“王湘琦你跟四父都看过我集盖你们那权威啊威丢”、CER0.689655、最长低能量0.03s，8000的两次videos本轮未保持，正文与尾部均退化。
- ICL00为15帧1.2s，“The liquid spears.”、WER0.25，未退为两帧；01为13帧1.04s，“組織奶白色後”、CER0.5。02为56帧4.48s，ASR目标匹配、WER/CER0；05为20帧1.6s，“W. Flanders, uni.”、WER0.5/CER0.153846，8000匹配未保持；06为44帧3.52004s，“In the Jeffreeze tube, they open a door and enter a swamp.”、WER0.083333，末词保留。ICL12为50帧4s/目标5.28s，“相信你跟似乎都看过 寄给你们那句完美的视频”、CER0.586207、最长低能量0.09s，目标两次video未在转写中体现，正文仍缺/错词。
- 相较8000，SO基础EN WER从0.25升至0.333333、规范化从0.277778升至0.361111，ZH CER从0.392857升至0.809524；ICL EN WER从0.166667降至0.111111、ZH CER从0.726190降至0.488095，但04完全失败复发，03低能量仍在。SO严重退化主要03重新触顶及00/05/12内容；本轮追加的波形与CPU转写证实末段不能当作有效内容，尚不足以确定生成码/codec/写WAV前浮点信号根因。前序输入/参考/codec/裁剪路径未见损坏或失败填零；没有可验证训练实现修复依据，不凭8句调loss/LR或改生成上限。保持设置，下一轮9000优先检查SO03触顶/末段、ICL04早停、两模式短句及12内容。
- 收尾8640：token first/residual=1.440172526/6.255181813，sample=1.268458633/6.127397227，目标first_sample_ce+0.3 residual_sample_ce=3.106677801，grad=0.609703，LR=9.111090523e-05/2.733327157e-04，step=2.127511s、wait=0.0003133s。8500完整评估后140次真实更新/14日志点正常，身份一致无退出；最新仍8500，未提前报告9000。
- 实际执行cat/tail、Python JSON/YAML/proc/checkpoint/SHA/signature/512清单条数/有限值/LR/统计/磁盘检查、nvidia-smi/free/df、soundfile/numpy逐句波形/配对/低能量检查、scipy Welch分段频谱及faster-whisper CPU原WAV/内存尾段复核，均成功。CPU诊断进程正常结束。唯一持久修改为追加本文；保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/ack/评估原件，未发信号、恢复或重启，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。尚未到19000，不执行最终冻结验收或写final-verification。本次单轮巡检结束。


### 2026-09-11T17:34:27.640866+00:00 — 训练至9120；9000 ICL03全零触顶复发，SO03时长恢复

- 已读playbook、本文、manual/process/退出状态、173032快照及170032 review/status。前轮8640，本快照9010，现场9030→9120；manual=false。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系、sample命令/无resume正确；torchrun/四rank均expandable_segments:True。收尾PID/start_ticks/cmdline复核通过，无training-exit.json或训练Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted；上一review exit0，同一外层1800秒session。
- latest文本指向step-00009000，9000 COMPLETE于17:17:22.758825 UTC写成。progress9000/epoch0/next_batch9000、world4、scheduler9000/_step_count9001、LR9.028565950e-5/2.708569785e-4。四distcp分别2093307384/2093524691/2093544414/2093533470B，distributed/.metadata1424894B、四rng各14613B齐全。metadata SHA256=a62eb6485510bc044565d46044352d92311633472e3ed5a2fe024eea0ddd00e0；9000/8500 signature相同，settings/model/eval/seed匹配配置，run/source配置一致。为结构/签名核验，未做恢复加载测试。
- 全912训练日志点至9120普通CE/sample CE/梯度等数值有限，warmup/cosine LR逐点通过。8650–9030共39点token first范围1.343152–1.474319、中位1.402753；residual6.167977–6.225664、中位6.195981；sample first1.204986–1.317605、中位1.261593；residual6.067783–6.133489、中位6.094873；clip前grad0.488461–0.657028、中位0.547426。
- 同区间step中位2.128807s、范围2.023270–2.318404；吞吐中位877.675音频秒/墙钟秒、范围821.013–928.116；wait中位0.0002960s、范围0.0001819–0.0003860。samples315–384、中位352；frame填充95.2208–99.7292%、中位98.1125%；token92.6–97.6167%、中位95.3111%；峰值显存53.405–56.505GiB。四卡64955/64123/64203/64103MiB（各81920），利用率99–100%，compute-apps仅四sample rank；RAM246GiB、可用1.7TiB、无swap，磁盘余577775.44GiB。无持续吞吐下降、供数等待或资源耗尽证据。
- 9000 val对应512条清单：token first/residual=1.458139527/6.191496482，sample=1.324514269/6.105640516，均低于8500的1.462024249/6.216999643和1.326418144/6.134436578。15残余码本CE有限、范围3.950037–6.971288，全部val日志数值有限。普通CE为token平均，sample逐句等权，分口径记录；CE下降不代表生成质量通过。

|9000完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/严格全零WAV/两帧EOS|
|---|---|---|---|---|---|
|speaker_only|0.194444/0.076923|0.194444/0.076923|1.125000/0.666667|0/8、0/8、0/8|
|icl|0.083333/0.055944|0.083333/0.055944|1.000000/0.845238|1/8、1/8、1/8|

- SO/ICL summary于17:23:18.329774/17:30:02.948421 UTC完成；16metrics/16WAV可读、24kHz单声道、finite、时长匹配。目标ID/text/参考ID/ICL reference_text及greedy min2与8500一致，summary样本和截断计数吻合。以下为ASR和波形统计，未主观试听；低能量定义为10ms RMS<0.001。
- ICL03变为400帧32s、EOS=false/truncated=true，WAV严格全零、RMS0、低能量100%、最长[0,32]s，ASR为空、WER/CER均1。较8500的196帧15.68s/最长10.51s严重退化，不能归为ASR单独漏识别。文件为标准PCM_16单声道24kHz，768000个采样全零、文件1536044B；SHA256=444206525131044130514f503d560a3e33a07e543ac4ba2520de3b86325385b6，与本run3500同目标的全零WAV相同，4000的微小非零WAV哈希不同。相同哈希来自相同长度零样本，不证明生成码相同或复用旧文件。
- 因全零复发，本轮追加当前输入与路径核查。初次直接sf.read(metrics里的speaker_reference_audio)报LibsndfileError：该字段是prepared逻辑路径，实际来源是speaker_reference_source中的tar_member。按qwen3_train/sources.py的decode_emilia_audio读取真实参考emilia2-00158.tar成员205e83c292110537_078_000_spk1.m4a（offset290060288/size34177）成功，24kHz共72480采样/3.02s、finite、RMS0.038916。目标成员205e83c292110537_086_000_spk1.m4a（offset290337792/size66156）读取成功，150720采样/6.28s、finite、RMS0.054465；目标/参考ID/text与本轮metrics一致。目标codec NPZ SHA256=20a8d1513ee2cc49a48a03495ffa0b2ee079228efb998ee33b12a3290231b345，与manifest匹配，codes形状79×16、uint16、范围0–2045。逻辑路径不存在不是训练输入损坏，实际源读取正常。
- 当前qwen3_train/train.py generate_sample路径核查：ICL用38帧参考前缀，400帧实际生成后拼接codec.decode，再按参考帧比例裁剪、直接sf.write；该路径没有失败后填零、nan_to_num或用零音频代替异常的分支。metric记录生成耗时129.84s且随后ASR正常完成，无训练异常。此处WAV默认PCM_16，严格全零只证明持久化样本为零，不能证明写入前float恰好为零，也不能区分生成码、codec输出、参考裁剪或量化贡献。未保存生成码/写WAV前浮点信号，现有证据不足以实施可验证修复；没有通过跳过样本、改上限或重启掩盖问题。
- ICL04仍2帧0.16s、EOS=true/truncated=false，8500/9000连续两轮；RMS0.001073、低能量87.5%、最长0.11s位于[0.05,0.16]s，ASR“字幕by索兰娅”不作有效目标语音。8000的26帧恢复未保持。该句是提前EOS，03是未EOS触顶且全零，两类失败分开记录。
- SO03从8500的400帧32s恢复为84帧6.72s、EOS=true，接近目标6.28s；RMS0.038158、低能量22.3214%，最长0.72s位于[3.98,4.70]s，8500的16.75s长段和后半段无效内容本轮未保持。ASR“回去的辦校室審查是下午2點50,起了大約10公里,先生生意的電量是80%”、CER0.75，仍多错词且简繁敏感；不将时长恢复称为全文正确。此前7000/8000也曾短暂改善，需继续跟踪稳定性。
- SO00为28帧2.24s，“I then the liquid spears.”、WER0.5，较8500 WER1.25改善但仍多/错起始内容，时长比目标1.81s长；01为13帧1.04s，“總監來北市後”、CER1。02为54帧4.32s，ASR目标匹配、WER/CER0；04为26帧2.08s，“徐世人必須堅持了整個年”、CER0.692308。05为31帧2.48s，“Chuta W. Flinders, UNT.”、WER0.75/CER0.461538，仍有额外/专名错误且比目标1.62s长；最长低能量0.04s，不能解释为长静音。
- SO06为46帧3.68s，“And the Jeffries tube, they open a door and enter a swap.”、WER0.166667，末词swamp再次被转写为swap。SO12为56帧4.48s/目标5.28s，“我相信你个师父都看过我给你们的简威 videos”、CER0.482759、最长低能量0.08s，只出现一次videos、正文仍缺/错词。
- ICL00为15帧1.2s，“The liquid spears.”、WER0.25，未退为两帧；01为13帧1.04s，“組織奶杯似乎”、CER0.833333。02为46帧3.68s，“This is a McLaughlin group in which there are winners and losers, yeah, you know.”、WER0.0625，like缺失；05为20帧1.6s，ASR“W.A. Flinders-Uni.”匹配、WER/CER0；06为44帧3.52004s，“In the Jeffries tube, they open a door and enter a swap.”、WER0.083333，末词swap。ICL12为47帧3.76s/目标5.28s，“相信你跟师父都看过我去给你们的捐威视频”、CER0.586207，目标两次video仍未在转写中体现。
- 相较8500，SO EN WER从0.333333降至0.194444、ZH CER从0.809524降至0.666667；ICL EN WER从0.111111降至0.083333、ZH CER从0.488095升至0.845238，03全零触顶和04早停使中文严重退化。此前8000重复转写已由CPU裁剪对照证明不能等同实际语音重复，8500 SO末段也已复核无有效目标转写；本轮全零则可直接由采样验证，无需再跑ASR。输入与codec哈希正常、无失败填零路径，根因仍需生成码/float音频证据，不能凭8句或单轮变化调loss/LR。本轮保持配置，下一轮9500优先核查ICL03全零是否持续、04早停、SO03时长及短句/06末词/12尾部。
- 收尾9120：token first/residual=1.432182886/6.173999431，sample=1.270493968/6.064602970，目标first_sample_ce+0.3 residual_sample_ce=3.089874859，grad=0.608226，LR=9.000342284e-05/2.700102685e-04，step=2.086914s、wait=0.0004429s。9000完整评估后120次真实更新/12日志点正常，身份一致无退出；最新仍9000，未提前报告9500。
- 实际执行cat/tail、Python JSON/YAML/proc/checkpoint/SHA/signature/512清单条数/有限值/LR/统计/磁盘检查、nvidia-smi/free/df、soundfile/numpy逐句波形/配对/低能量检查、rg/sed生成写入与数据来源代码核查、tar原生解码与codec/hash检查。初次猜测evaluation/audio等文件名不存在及直接读取逻辑参考路径失败，随后rg --files定位train.py/sources.py并按真实tar来源完成检查；这些为巡检命令错误，不是训练异常，未据此重启。最终验证均通过。
- 唯一持久修改为追加本文；保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/ack/评估原件，未发信号、恢复或重启，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。尚未到19000，不执行最终冻结验收或写final-verification。本次单轮巡检结束。


### 2026-09-11T18:04:26.581612+00:00 — 训练至9590；9500 ICL03全程近静音持续，04连续三轮两帧

- 已读playbook、本文、manual/process/退出状态、180032快照及173032 review/status。前轮9120，本快照9500，现场9540→9590；manual=false。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系、sample命令/无resume正确；torchrun/四rank均expandable_segments:True。收尾PID/start_ticks/cmdline复核通过，无training-exit.json或训练Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted；上一review exit0。快照时9500仍处于本轮评估，18:01 ICL完成后训练继续，不属于卡死。
- latest文本指向step-00009500，9500 COMPLETE于17:48:16.352602 UTC写成。progress9500/epoch0/next_batch9500、world4、scheduler9500/_step_count9501、LR8.908652146e-5/2.672595644e-4。四distcp分别2093307384/2093524691/2093544414/2093533470B，distributed/.metadata1424894B、四rng各14613B齐全。metadata SHA256=5342608e0a818c3a6cb5fd6f76d194357c6d617d67b40e7d8844e943ec45cc27；9500/9000 signature相同，settings/model/eval/seed匹配配置，run/source配置一致。为结构/签名核验，未做恢复加载测试。
- 全959训练日志点至9590普通CE/sample CE/梯度等数值有限，warmup/cosine LR逐点通过。9130–9540共42点token first范围1.335918–1.467388、中位1.392307；residual6.120712–6.218835、中位6.164184；sample first1.219005–1.295931、中位1.254229；residual6.039963–6.114382、中位6.065780；clip前grad0.463158–0.641753、中位0.541894。
- 同区间step中位2.166848s、范围2.010421–2.332781；吞吐中位868.800音频秒/墙钟秒、范围782.963–924.212；wait中位0.0002969s、范围0.0001681–0.0060078。samples323–391、中位358；frame填充95.1292–99.6125%、中位98.2542%；token91.6861–97.7%、中位95.4486%；峰值显存53.637–56.726GiB。四卡64955/64123/64203/64103MiB（各81920），compute-apps仅四sample rank；RAM246GiB、可用1.7TiB、无swap，磁盘余577643.18GiB。现场瞬时GPU利用率16–39%，但真实更新持续、step/吞吐正常；单次6ms等待没有形成持续瓶颈，无资源耗尽或需要改workers的依据。
- 9500 val对应512条清单：token first/residual=1.439014249/6.165465184，sample=1.305325612/6.076035097，均低于9000的1.458139527/6.191496482和1.324514269/6.105640516。15残余码本CE有限、范围3.924872–6.945151，全部val日志数值有限。普通CE为token平均，sample逐句等权，分口径记录；CE下降不代表生成质量通过。

|9500完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/严格全零WAV/两帧EOS|
|---|---|---|---|---|---|
|speaker_only|0.250000/0.139860|0.250000/0.139860|1.125000/0.630952|0/8、0/8、0/8|
|icl|0.083333/0.055944|0.083333/0.055944|1.000000/0.785714|1/8、0/8、1/8|

- SO/ICL summary于17:54:15.298221/18:01:08.354914 UTC完成；16metrics/16WAV可读、24kHz单声道、finite、时长匹配。目标ID/text/参考ID/ICL reference_text及greedy min2与9000一致，summary样本和截断计数吻合。以下依据ASR和波形统计，未主观试听；低能量定义为10ms RMS<0.001。
- ICL03仍400帧32s、EOS=false/truncated=true，RMS3.482326e-8、低能量100%、最长[0,32]s、ASR为空、WER/CER1。9000的严格全零本轮变为极微小非零，不能认为恢复：直接读PCM_16得到768000采样中仅索引101为-1，其余全零，即单个最低有效位负采样。SHA256=9f4c3bda82d68996fcf0e34a1d15882781896a514bd4938b5a944051209f7355，与本run4000同目标文件完全相同；9000为全零、不同哈希。两轮均全程近静音且触顶，严格全零计数降为0不具备质量改善意义；相同文件内容不证明生成码相同或复用旧文件。
- 上轮已按实际tar_member重读目标/参考音频、核对目标codec SHA及码值，均正常；逻辑speaker_reference_audio路径不一定存在，真实来源在speaker_reference_source。生成路径无失败填零分支，WAV默认PCM_16；本轮目标/参考/策略相同，无新的输入故障线索，不重复已通过的tar解码或给空音频再跑ASR。没有生成码/写入前float音频，仍不能区分生成码、codec输出、参考裁剪或量化贡献，也无可验证修复依据。
- ICL04仍2帧0.16s、EOS=true/truncated=false，8500/9000/9500连续三轮提前EOS。RMS0.002181、低能量87.5%、最长0.08s，ASR“字幕by索兰娅”不作为有效语音。与03的未EOS触顶近静音分开记录；未提高min帧数掩盖问题。
- SO03保持84帧6.72s、EOS=true，9000/9500连续两轮接近目标6.28s且无触顶；RMS0.040922、低能量20.8333%，最长0.79s位于[2.44,3.23]s。较9000的0.72s略长但远短于8500的16.75s。ASR“就去了半高小时间是下午2点50 起了大概10公里 先上雨的点量是80%”、CER0.583333（9000为0.75），仍有开头/电量错词，不能称全文正确或稳定解决。
- SO00为28帧2.24s，“are the liquid spares.”、WER0.5，起始与末词仍错，时长比目标1.81s长；01为13帧1.04s，“放棄奶排售後”、CER0.833333。02为50帧4s，“This is a McLaughlin group in which, like, there were innocent losers, yeah, you know.”、WER0.1875，9000匹配本轮出现正文错误。04为29帧2.32s，“爱惜世侠毕竟见识了这么多年”、CER0.461538。
- SO05为29帧2.32s，“2W Heath Lenders Uni.”、WER0.75/CER0.461538，额外/专名内容仍错，最长低能量0.07s；06为47帧3.76s，“and the Jeffries tube. They open a door and enter a swamp.”、WER0.083333，9000末词swap本轮恢复swamp。SO12为60帧4.8s/目标5.28s，“我相信你跟師父都看過我寄給你們的捐威 威丟威丟威丟”、CER0.724138、最长低能量0.03s。尾部三次音译形态提示需跟踪额外重复，但ASR不等于已确认实际重复；比9000 CER0.482759退化，较长时长也不保证完整正确。
- ICL00为17帧1.36s，“The liquid spears.”、WER0.25，未退为两帧；01为12帧0.96s，“组织奶白色后”、CER0.333333。02为49帧3.92s，“This is a McLaughlin group in which, like, there are winners and losers yet, you know.”、WER0.0625，本轮like重现但yeah→yet；05为22帧1.76s，ASR“W.A. Flinders-Uni.”匹配、WER/CER0，连续两轮。06为42帧3.36s，“In the Jeffreece Tube, they open a door and enter a swamp.”、WER0.083333，9000末词swap本轮恢复swamp，仍有专名差异。
- ICL12为43帧3.44s/目标5.28s，“相信你跟师傅都看过我寄给你们的绝缘”、CER0.517241、最长低能量0.13s。目标两次video均未以对应词形出现；正文更接近使CER比9000的0.586207下降，不代表尾部内容恢复。SO/ICL均没有长低能量的12仍有缺词/重复问题，不归因于静音。
- 相较9000，SO EN WER从0.194444升至0.25、ZH CER从0.666667降至0.630952；ICL EN WER持平0.083333、ZH CER从0.845238降至0.785714，但03全程近静音/触顶及04提前EOS持续。此前8000 CPU对照已说明ASR重复不等于生成实际重复，8500 SO末段也无有效目标转写；本轮PCM计数明确03没有有效波形恢复，继续保留该严重问题。没有新的可验证训练实现修复依据，不凭8句调loss/LR或改生成上限。保持设置，下一轮10000优先核查ICL03/04、SO03时长持续性、短句额外内容及12尾部。
- 收尾9590：token first/residual=1.397366979/6.152737486，sample=1.252566819/6.053879752，目标first_sample_ce+0.3 residual_sample_ce=3.068730744，grad=0.560096，LR=8.886427650e-05/2.665928295e-04，step=2.209024s、wait=0.0003799s。9500完整评估后90次真实更新/9日志点正常，身份一致无退出；最新仍9500，未提前报告10000。
- 实际执行cat/tail、Python JSON/YAML/proc/checkpoint/SHA/signature/512清单条数/有限值/LR/统计/磁盘检查、nvidia-smi/free/df、soundfile/numpy逐句波形/配对/低能量检查及03历史PCM采样/哈希对照，均成功。唯一持久修改为追加本文；保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/ack/评估原件，未发信号、恢复或重启，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。尚未到19000，不执行最终冻结验收或写final-verification。本次单轮巡检结束。


### 2026-09-11T19:03:56.105655+00:00 — 训练至10580；补查10000，10500 ICL长低能量和两帧EOS持续

- 已读playbook、本文首段及最新记录、manual/process/退出状态、190032快照、183032 review/status和sqrt next-run-ack。上次实质巡检收尾9590；18:30快照10000，CLI虽exit0，但review仅确认仓库规则，没有完成指标/音频核验或追加文档，不能算巡检通过。本轮明确补查10000及10500产物，不将当前补查冒充18:30现场记录；原review/日志保留。本次快照10500，现场10510→10580，manual=false，同一外层1800秒session。
- launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系及sample命令/无resume核验通过；torchrun/四rank均expandable_segments:True。收尾身份再核验通过，无training-exit.json，无训练Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted。sqrt ack=verified且本run初始化仍为原assembled/step0；run/source配置相同，未续接sqrt。
- latest指向step-00010500。10000/10500 COMPLETE分别于18:19:29.715419/18:48:57.852290 UTC写成；两份progress的step/next_batch分别10000/10500、epoch0/world4，scheduler last_epoch一致、_step_count分别10001/10501。10500 LR=8.651142368e-5/2.595342710e-4。两份各有四distcp（2093307384/2093524691/2093544414/2093533470B）、distributed/.metadata1424894B及四rng各14613B；metadata SHA分别b60b6a92e825aa771a6f2e3e47af343698958bba2a94d7c7d4c5c50a17f6ca0a、e2fc89f11ad0784cceae8c40220cdadf3a683c631f9abfd7c336e0fc816207fb。10000/10500 signature一致，settings/model/eval/seed匹配配置；当前val清单及assembly_report的SHA重新核验匹配。此为结构/签名检查，没有加载恢复测试，也未重读全量train清单重新计算指纹。
- 全1058训练日志点至10580普通CE/sample CE/梯度及其余训练数值有限，warmup/cosine LR逐点通过。9600–10530共94点：token first范围1.293251–1.438452、中位1.373538，residual6.042464–6.185117、中位6.138500；sample first1.176539–1.292051、中位1.236306，residual5.985892–6.081930、中位6.031197；clip前grad0.438852–0.698959、中位0.525379。
- 同区间step中位2.147759s、范围2.005621–2.468236；吞吐中位877.884音频秒/墙钟秒、范围763.221–926.092；wait中位0.0002828s、最大0.0052042s。samples291–410、中位358.5；frame填充94.5458–99.6708%、中位98.3979%，token91.8972–98.0056%、中位95.4319%；峰值显存53.346–56.870GiB。现场四卡64955/64123/64203/64103MiB（各81920）、利用率94–100%，compute-apps只有四sample rank；RAM245GiB、可用1.7TiB、无swap，现场磁盘余579253.23GiB。无持续吞吐/等待退化或资源耗尽；单次5ms等待无需改workers。
- 10000/10500 val均对应512条清单；21轮val日志全部有限。10000 token first/residual=1.434566574/6.147077498、sample=1.300784388/6.057800524；10500为1.417667220/6.126841496、1.283056890/6.036224693，均较9500持续下降。10500十五残余码本CE范围3.895462–6.902731，均低于10000。普通CE/逐码本CE仍token平均，sample为逐句等权，目标first_sample_ce+0.3 residual_sample_ce；不混用口径或将CE下降当作生成质量通过。

|step/模式，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/严格全零WAV/两帧EOS|
|---|---|---|---|---|
|10000 speaker_only|0.250000/0.097902|0.222222/0.097902|1.250000/0.500000|0/8、0/8、0/8|
|10000 icl|0.111111/0.076923|0.111111/0.076923|1.000000/0.857143|0/8、0/8、1/8|
|10500 speaker_only|0.416667/0.349650|0.444444/0.349650|1.125000/0.511905|0/8、0/8、0/8|
|10500 icl|0.138889/0.062937|0.138889/0.069930|1.000000/0.607143|0/8、0/8、1/8|

- 10000 SO/ICL summary于18:25:30.894634/18:30:52.683321 UTC完成，10500于18:54:55.047131/19:00:48.575627 UTC完成；本次快照时ICL尚未结束，随后正常完成并恢复更新，不是卡死。两轮32份metrics/32份生成WAV全部可读、24kHz单声道、finite、时长匹配；逐轮目标ID/text、参考ID/source/ICL reference_text/reference_frames和greedy min2均与前轮相同。逐语言原始错误计数、英语规范化计数及截断计数与summary吻合。以下仅ASR和波形统计，未主观试听；低能量定义为10ms RMS<0.001。
- ICL03：9500为32s触顶、几乎全零；10000变为132帧10.56s、EOS=true，RMS0.001005、低能量99.6212%，最长[0,10.51]s，ASR为空/CER1。虽不再触顶，仍无有效内容证据。10500为198帧15.84s、EOS=true，RMS0.023565、低能量68.3712%，最长[0,10.54]s；ASR“接着暴水水,线车下午2点50,卸了大概10公里,先剩余的电量是80%”、CER0.638889。部分正文重新进入转写，但时长约目标6.28s的2.52倍且长低能量持续，不能称恢复正常。前序已核查实际tar目标/参考及codec哈希正常、生成无失败填零路径，并有8000/8500 CPU分段ASR诊断；本轮配对无变化、无新原始异常，不重复这些检查或对空音频再跑ASR。缺少生成码/写WAV前float信号，生成码、codec和裁剪环节的根因仍未定位。
- ICL04在10000/10500均2帧0.16s、EOS=true，从8500起连续五轮提前EOS。10500 RMS0.000157、100%低能量；“字幕by索兰娅”不作为有效语音。该问题与03长低能量分开记录，未提高min_new_frames掩盖提前EOS。
- SO03在10000/10500为87/86帧、6.96/6.88s，连续四轮（9000起）接近目标6.28s且无触顶。低能量20.546/20.203%，最长0.82/0.53s。10500 ASR“不久的半个小时 现在是下午2点50 起了大概10公里 先生剩余的电量是80%”、CER0.444444，较10000的0.388889略差；时长改善未解决开头/错词，不称全文正确。
- SO短句：00由10000的27帧2.16s、“I did the liquid spears on”、WER0.75，变为10500的28帧2.24s、“I not know who was shooting the spears.”、WER1.75，额外/错误内容加重。01在10000突然33帧2.64s/目标0.98s，ASR“讓舊真奶白壽後 舊真奶白壽後期算是”、CER2.333333，有重复/额外内容迹象；10500回到13帧1.04s、“張憲奶白是猴”、CER0.666667，时长恢复但内容仍错，未用ASR确认实际重复。05从23帧1.84s、“2A Flinders Uni”、WER0.5变为32帧2.56s、“Chithatir Dabia, Flendis Cuny.”、WER1，最长低能量仅0.02s；不能用长静音解释这次额外内容。
- SO02在10500为51帧4.08s、“This is a McLaughlin group in which like, there are winners and looters. Yeah, you know.”、WER0.0625，losers→looters；04为32帧2.56s、“愛惜時間壓逼近現實了這麼多年”、CER0.846154，比10000的0.153846退化，但简繁也影响CER。06为43帧3.44s、“That's a Jeffreeze tube. They open a door and enter a swamp.”、WER0.25，末词保留、开头仍错。
- ICL短句及尾部：10500的00为20帧1.6s、“Uh, the liquor spears.”、WER0.5，较10000的“The liquid spears.”/WER0.25更差，未退为两帧。01为12帧0.96s、“煮至奶白色厚”、CER0.166667；02为49帧3.92s，连续两轮ASR匹配、WER0。05为20帧1.6s、“W. Flanders, Uni.”、WER0.5，10000的匹配未保持。06为42帧3.36s、“in the Jeffries tube, but they open a door and enter a swamp.”、WER0.083333，10000的末尾“in a swap”本轮回到“enter a swamp”，但多but。
- 12目标5.28s且含两次video：SO在10000/10500为4.24/4.56s，ASR分别“我相信你跟师父都看过我寄给你们的权威 videos”/“我相信你跟似乎都看过我寄给你们的卷位子 videos”，CER均0.413793，均仅一次videos。ICL为3.84/3.76s，ASR分别“相信你跟師傅都看過我寄給你們的捲微軟微軟”/“相信你跟師傅都看過我寄給你們的捲Vidio”，CER0.620690/0.482759，10500只出现一次近似video词形；最长低能量0.13/0.04s，尾缺词未解决。CER降低不能证明两次video或正文完整。
- 10500较10000：SO基础EN WER0.25→0.416667、ZH CER0.5→0.511905；ICL EN WER0.111111→0.138889、ZH CER0.857143→0.607143。ICL中文下降主要伴随03部分可转写内容恢复和01改善，04失败仍在；SO00/05额外内容恶化。持续问题已结合历史输入/codec/分段诊断检查，本轮无新的可验证实现修复依据，不凭8句更改loss、LR或生成上限。下一轮11000继续核查ICL03/04、SO00/01/05、SO03时长和两模式12尾部。
- 收尾10580：token first/residual=1.392698039/6.115651240，sample=1.239289400/6.001844810，目标=3.039842844，grad=0.528620，LR=8.62956373953e-05/0.000258886912186，step=2.103264s、wait=0.0003055s。10500完整评估后80次真实更新/8日志点正常；最新仍10500。
- 实际执行cat/head/tail/ls、Python JSON/YAML/proc身份及allocator、checkpoint文件/签名/SHA、512清单条数、全日志有限值/LR/统计/磁盘检查、nvidia-smi/free/df、soundfile/numpy逐句波形与配对/summary计数核验、rg/sed查看signature/LR/数据fingerprint；检查均通过。仅追加本文并检查文档差异空白。未修改训练代码/配置/manual/metadata/数据/ack/评估原件，未发信号或恢复，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。尚未到19000，不执行最终冻结验收或写final-verification。本次单轮巡检结束。


### 2026-09-11T19:34:38.605915+00:00 — 训练至11120；11000 ICL03长低能量本轮消失，SO05额外内容继续恶化

- 已读playbook、本文最新记录、manual/process/退出状态、193032快照和190032 review/status。前轮10580，本快照11000，现场11050→11120；manual=false。上一巡检exit0且有完整记录，同一外层1800秒session。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系及sample命令/无resume正确；torchrun/四rank均expandable_segments:True。收尾再次核验身份、非僵尸且无training-exit.json；训练日志无Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted。
- latest指向step-00011000，COMPLETE于19:19:26.287509 UTC写成。progress step/next_batch11000、epoch0/world4，scheduler last_epoch11000/_step_count11001、LR8.513997214e-5/2.554199164e-4。四distcp为2093307384/2093524691/2093544414/2093533470B，distributed/.metadata1424894B，四rng各14613B；metadata SHA=bb513557d861892cbbdbee0a11f1123cbc9495c4c1a7527b84af5ae56829309b。11000/10500 signature一致，settings/model/eval/seed匹配配置，run/source配置相同；当前val清单及assembly_report重新SHA核验匹配。此为结构/签名检查，未加载恢复测试，未重读全量train清单计算指纹。
- 全1112训练日志点至11120普通CE/sample CE/梯度和其他训练数值有限，warmup/cosine LR逐点通过。10590–11070共49点：token first范围1.309521–1.426244、中位1.365669，residual6.046286–6.146351、中位6.109236；sample first1.185216–1.269696、中位1.223166，residual5.978267–6.032857、中位6.003015；clip前grad0.445094–0.604279、中位0.516821。
- 同区间step中位2.139934s、范围1.994746–2.368881；吞吐中位881.822音频秒/墙钟秒、范围808.382–923.105；wait中位0.0002855s、最大0.0004522s。samples317–392、中位357；frame填充94.6833–99.7792%、中位98.1458%，token92.2056–97.5361%、中位95.2861%；峰值显存53.656–56.650GiB。四卡64955/64123/64203/64103MiB（各81920），compute-apps仅四sample rank；RAM245GiB、可用1.7TiB、无swap，磁盘余580282.68GiB。瞬时GPU利用率21–24%，但训练持续推进、step/吞吐/等待稳定，没有持续供数或资源故障依据。
- 11000 val对应512条清单，22轮val日志全部有限。token first/residual=1.412721243/6.109988568、sample=1.278664002/6.015832096，均低于10500的1.417667220/6.126841496、1.283056890/6.036224693。十五残余码本CE范围3.890367–6.884413，均较10500下降。普通CE/逐码本CE为token平均，sample逐句等权；优化目标first_sample_ce+0.3 residual_sample_ce，不混用口径或将CE下降当作生成质量通过。

|11000完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/严格全零WAV/两帧EOS|
|---|---|---|---|---|
|speaker_only|0.416667/0.335664|0.444444/0.335664|1.250000/0.630952|0/8、0/8、0/8|
|icl|0.138889/0.083916|0.138889/0.083916|1.375000/0.452381|0/8、0/8、1/8|

- SO/ICL summary于19:25:21.476038/19:30:12.992955 UTC完成；16份metrics/16份WAV可读、24kHz单声道、finite、时长匹配。目标ID/text、参考ID/source/ICL reference_text/reference_frames、greedy min2与10500相同；逐语言和英语规范化错误计数及截断计数与summary吻合。以下为波形统计和ASR证据，未主观试听；低能量定义为10ms RMS<0.001。
- ICL03从10500的198帧15.84s、最长10.54s低能量前缀，变为82帧6.56s、EOS=true，接近目标6.28s。RMS0.037623、低能量24.6951%、最长仅0.55s位于[4.15,4.70]s，本轮长前缀确实未出现。ASR“又起了半小时,现在是下午两点五时,起了大概一次公里,现在又得电量是80%”、CER0.388889，较10500的0.638889改善，但距离词等仍错；不能称全文正确或稳定解决。此前多轮全零/近静音、长低能量和部分恢复反复，下一轮需验证这一改善能否维持。
- ICL04仍2帧0.16s、EOS=true，从8500起连续六轮；RMS0.000708、低能量87.5%、最长0.08s。“字幕by索兰娅”不作为有效语音。该目标早停没有随03改善消失；前序输入、codec、EOS边界/配对检查未见故障，本轮配对也一致，没有新逐步logits证据，不提高min_new_frames掩盖问题。
- SO03为85帧6.8s、EOS=true，9000起连续五轮时长接近目标且无触顶；RMS0.040664、低能量23.5294%、最长0.53s位于[2.08,2.61]s。ASR“特輯的半個時間是下午2點50 其他大概10公里 現在順利的電量是80%”、CER0.694444，比10500的0.444444变差；错词及简繁均影响分数，不将时长稳定当作内容准确。
- SO00为27帧2.16s，“I didn't know the licoride spars.”、WER1.25，较10500的1.75下降，但仍明显多/错内容。01为13帧1.04s，“張靜乃白索厚”、CER0.833333；10000的2.64s额外内容未再复现，时长连续两轮接近目标，但转写仍错。02为53帧4.24s，ASR匹配、WER/CER0；04为28帧2.24s，“愛惜世仍然畢竟堅持了這麼多年”、CER0.692308。06为46帧3.68s，“And the Jeffreeze tube, they open a door and enter a swamp.”、WER0.166667，末词保留，开头/专名仍错。
- SO05从10500的32帧2.56s/目标1.62s，进一步变为34帧2.72s；RMS0.067332、低能量1.1029%、最长仅0.03s。ASR“She the bitch, that bitch who defends his uni.”、WER2/CER2.153846，比10500的“Chithatir Dabia, Flendis Cuny.”/WER1更偏离“W.A. Flinders Uni.”。这些是ASR原文，不据此断言实际语音包含这些词。长低能量不能解释此短句偏离。
- 对SO05实施有界CPU复核：读取本地已缓存faster-whisper-small快照536b0662742c02347bc0e980a01041f333bce120，local_files_only=True，CPU int8、4threads/1worker，沿用评估beam5/en/vad_filter=False/condition_on_previous_text=False。分别重转写10500 SO05、11000 SO05及11000 ICL05三个原WAV，均与原metrics逐字一致；转写耗时1.446/1.255/1.125s（不含加载），no_speech_prob约0.02582/0.02505/0.03931。SO两轮异常可复现，同目标ICL本轮为“W.A. Flinders-Uni”、WER0；原reference_asr同样WER0。该对照排除了简单的旧文本记录错配，说明当前评分下SO持续偏离，但同一个ASR模型重现不是独立试听，也不能单独区分实际错音与ASR误听，no_speech_prob不作内容准确性的证据。ASR分段时间可越出短WAV（ICL段到2.0s而WAV1.68s），未将其当精确对齐。未下载、占用GPU或修改原metrics/summary/WAV，诊断进程正常exit0。
- ICL00为14帧1.12s/目标1.81s，“The Lewis Bears.”、WER0.75，比10500的20帧1.6s/“Uh, the liquor spears.”更短且更错；不是两帧EOS，仍需跟踪。01为12帧0.96s，“组织奶白色后”、CER0.333333；02为47帧3.76s，10000起连续三轮ASR匹配。05为21帧1.68s、WER0，本轮专名恢复，但之前也反复。06为43帧3.44s，“In the Jeffreeze tube, they open a door and enter a swap.”、WER0.166667，10500的swamp本轮又变swap，末词仍不稳定。
- 12目标5.28s：SO为56帧4.48s，“我相信你跟师父都看过我几个 你们的卷位 viteo”、CER0.482759，仍仅一次近似video，正文也错；最长低能量0.08s。ICL为47帧3.76s，时长与10500相同，ASR“相信你跟師傅都看過我寄給你們的捲 VL video video”、CER0.310345，较10500的单个Vidio/CER0.482759改善，本轮明确重新出现两次video；最长低能量0.07s。不能沿用上一轮“第二次video缺失”的结论，但额外VL、填充词/正文及简繁差异仍在，ASR不代表完整发音已验收；需观察两次video能否持续。
- 较10500，SO EN基础/规范化WER均持平0.416667/0.444444，ZH CER0.511905→0.630952；ICL EN WER持平0.138889，ZH CER0.607143→0.452381。SO00/02改善被05恶化抵消；ICL03波形及时长、12两次video的改善与04持续早停并存。前序已核查实际tar输入/参考/codec及分段ASR，本轮增加SO05原WAV复现；没有新的可验证训练实现修复依据，保持loss/LR/生成配置。下一轮11500重点核查ICL03改善是否维持、04/00早停风险、SO05额外内容及两模式12尾部。
- 收尾11120：token first/residual=1.331122697/6.107894604，sample=1.200504979/5.986854422，目标=2.996561305，grad=0.433288，LR=8.48028797105e-05/0.000254408639131，step=2.152709s、wait=0.0002773s。11000完整评估后120次真实更新/12日志点正常，最新仍11000。
- 实际执行cat/tail、Python JSON/YAML/proc身份/allocator、checkpoint文件/签名/SHA、512条清单、全日志有限值/LR/统计/磁盘检查、nvidia-smi/free/df、soundfile/numpy逐句波形及配对/summary计数、rg/sed核查ASRScorer、faster-whisper CPU三段原WAV复核，均成功；仅追加本文并检查文档空白差异。未改训练代码/配置/manual/metadata/数据/ack/评估原件，未发信号、恢复或重启，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。尚未到19000，不执行最终冻结检查或写final-verification。本次单轮巡检结束。


### 2026-09-11T20:03:08.012752+00:00 — 训练至11610；11500 ICL03改善保持，ICL12尾缺词和SO01额外内容复发

- 已读playbook、本文最新记录、manual/process/退出状态、200032快照及193032 review/status。前轮11120，本快照11540，现场11560→11610；manual=false，上一巡检exit0且有完整记录，同一外层1800秒session。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系及sample命令/无resume正确；torchrun/四rank均expandable_segments:True。收尾再核对身份/非僵尸、manual和latest，无training-exit.json或训练Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted。
- latest为step-00011500，COMPLETE于19:48:19.394653 UTC写成。progress step/next_batch11500、epoch0/world4，scheduler last_epoch11500/_step_count11501、LR8.371575465e-5/2.511472640e-4。四distcp为2093307384/2093524691/2093544414/2093533470B，distributed/.metadata1424894B，四rng各14613B；metadata SHA=a01eaf44da76ecc120b94c21e70c34a6b80a7191de800d2a026f1642ef20764b。11500/11000 signature一致，settings/model/eval/seed匹配配置，run/source配置相同；当前val清单及assembly_report SHA匹配签名。检查范围为结构/签名，未加载恢复测试，也未重读全量train清单计算指纹。
- 全1161训练日志点至11610普通CE/sample CE/梯度等数值有限，warmup/cosine LR逐点通过。11130–11570共45点：token first范围1.303232–1.440183、中位1.349456，residual6.052390–6.139682、中位6.088798；sample first1.182272–1.275667、中位1.213254，residual5.948021–6.021736、中位5.983939；clip前grad0.450634–0.624169、中位0.521422。
- 同区间step中位2.142054s、范围1.997233–2.469517；吞吐中位876.794音频秒/墙钟秒、范围756.521–920.151；wait中位0.0002789s、最大0.0003390s。samples326–401、中位356；frame填充94.5167–99.7542%、中位98.0125%，token91.6111–97.3306%、中位95.3361%；峰值显存53.467–56.650GiB。四卡64955/64123/64203/64143MiB（各81920），利用率21–64%，compute-apps仅四sample rank；RAM245GiB、可用1.7TiB、无swap，磁盘余580030.18GiB。训练持续更新、吞吐和等待稳定，无持续资源/供数故障依据，GPU3较前轮多40MiB无需干预。
- 11500 val对应512条清单，23轮val日志全部有限。token first/residual=1.400719574/6.094197495、sample=1.266966153/5.999355964，均低于11000的1.412721243/6.109988568和1.278664002/6.015832096。十五残余码本CE范围3.874558–6.871745，均较11000下降。普通CE/逐码本CE为token平均，sample逐句等权，目标first_sample_ce+0.3 residual_sample_ce；两口径分别记录，CE下降不等于生成质量通过。

|11500完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/严格全零WAV/两帧EOS|
|---|---|---|---|---|
|speaker_only|0.472222/0.342657|0.500000/0.342657|1.125000/0.559524|0/8、0/8、0/8|
|icl|0.027778/0.020979|0.027778/0.020979|1.000000/0.547619|0/8、0/8、1/8|

- SO/ICL summary于19:54:12.124472/19:59:05.890390 UTC完成；16份metrics/16份WAV可读、24kHz单声道、finite、时长匹配。目标ID/text、参考ID/source/ICL reference_text/reference_frames和greedy min2与11000相同；逐语言和英语规范化错误计数、截断计数与summary吻合。以下为波形和ASR证据，未主观试听；低能量定义为10ms RMS<0.001。
- ICL03为90帧7.2s/目标6.28s、EOS=true，RMS0.037255、低能量24.1667%、最长0.47s位于[0,0.47]s，连续两轮没有此前10秒以上低能量前缀；较11000的6.56s略长。ASR“又齊了半個小時,現在是下午2點50,齊了大概10公里,現在剩餘的電量是80%”、CER0.527778，比11000的0.388889高，但本轮10公里/电量等内容出现且有简繁差异，不能把CER上升直接等同语音同幅退化。前缀改善保持，全文正确及稳定解决仍未验收。
- ICL04仍2帧0.16s、EOS=true，从8500起连续七轮；RMS0.002947、低能量87.5%、最长0.09s。ASR“字幕by索兰娅”不作为有效内容。前序输入/codec/EOS边界和配对检查未见实现故障，本轮同目标/参考/策略，无新增逐步logits证据；不提高min_new_frames掩盖早停。
- SO03为91帧7.28s、EOS=true，9000起连续六轮无触顶且时长接近目标；RMS0.037354、低能量23.0769%、最长0.61s位于[2.41,3.02]s。ASR“除去半小时时现在是下午两点五时 切了大概10公里 现在剩余的点量是80%”、CER0.416667，比11000的0.694444低，但开头/错词仍在，不能称全文正确。
- SO00为26帧2.08s，“The liquid spars.”、WER0.5，比11000的1.25改善，额外开头本轮未出现，但起始重复词/末词仍未匹配。SO01由前两轮13帧1.04s变为26帧2.08s/目标0.98s，“裝飾下載載儲值南白色後”、CER1.5，类似10000的时长/额外内容问题复发；低能量24.5192%、最长0.25s，不能归结为长静音，也不将ASR原文视为已试听确认的额外词。
- SO05仍34帧2.72s/目标1.62s，“She pursed up Eva, she dabbled over Tim.”、WER2/CER2.076923，连续两轮WER2，10500起连续三轮较长且严重偏离目标“W.A. Flinders Uni.”。RMS0.056220、低能量4.4118%、最长0.07s，无长低能量解释。上轮已在同设置CPU重现10500/11000 SO原转写，并核对同目标11000 ICL/参考ASR匹配；本轮配对无变化、ICL05仍匹配，无新的评分异常线索，不重复同模型转写或把上轮复核冒充本轮。实际错音与ASR误听的贡献尚未分开，仍缺生成码/逐步分布证据支持实现修复。
- SO02为55帧4.4s，“This is my clackling group in which like there are winners and looters. Yeah, you know.”、WER0.1875，11000匹配本轮专名/losers等退化。04为24帧1.92s/目标2.3s，“愛惜人家必見識了這麼多”、CER0.692308，末尾“年”等未进入转写，应跟踪内容缩短；06为48帧3.84s，“Ascend to Jeffree's tube, they open a door and enter a swap.”、WER0.333333，末词swamp→swap且开头错误。
- ICL00由14帧1.12s变为17帧1.36s，“The liquid spears.”、WER0.25，比11000的“The Lewis Bears.”/0.75改善；仍短于目标1.81s，但不是两帧。01为11帧0.88s，“朱芝奶白色后”、CER0.333333。02为52帧4.16s、ASR匹配，10000起连续四轮；05为21帧1.68s、“W.A. Flinders, Uni.”、WER0，连续两轮匹配。06为48帧3.84s，“In the Jeffries tube, they open a door and enter a swamp.”、WER0，本轮末词及专名恢复。英语ICL四句仅基础计数1个删除/36词，仍只代表这四句当前ASR结果，不推及整体质量。
- 12目标5.28s：SO为49帧3.92s，“我相信你跟似乎都看过我击给你们的权威 videos”、CER0.482759，仍只有一次videos，正文错误，最长低能量0.04s。ICL由47帧3.76s缩为38帧3.04s，“相信你跟师父都看过我集给你们的权威和videos”、CER0.413793，最长低能量0.08s；11000的两次video仅保持一轮，本轮又只转写出一次，尾部恢复不稳定。没有长低能量，不能将缺词归因于长静音。
- 较11000，SO基础EN WER0.416667→0.472222、规范化0.444444→0.5，ZH CER0.630952→0.559524；ICL EN WER0.138889→0.027778，ZH CER0.452381→0.547619。英语ICL改善、03长低能量缓解与04持续早停、12尾缺词复发同时存在；SO00改善未抵消02/06错误及05严重偏离，01时长又反复。前序实际tar/参考/codec、分段ASR及SO05复核已记录，本轮无新的可验证实现故障，不凭8句调整loss、LR或生成上限。下一轮12000检查ICL03持续性、04/12、SO01/05及04缩短。
- 收尾11610：token first/residual=1.315845654/6.113194147，sample=1.195386008/5.980660599，目标=2.989584188，grad=0.485601，LR=8.33955914888e-05/0.000250186774466，step=2.206857s、wait=0.0003038s。11500完整评估后110次真实更新/11日志点正常；最新仍11500。
- 实际执行cat/tail、Python JSON/YAML/proc身份/allocator、checkpoint结构/签名/SHA、512清单条数、全日志有限值/LR/统计/磁盘检查、nvidia-smi/free/df，以及soundfile/numpy逐句波形、配对和summary计数核验，均通过；唯一持久修改为追加本文并检查文档空白差异。保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/ack/评估原件，未发信号、恢复或重启，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。尚未到19000，不执行最终冻结检查或写final-verification。本次单轮巡检结束。


### 2026-09-11T20:32:46.269396+00:00 — 训练至12110；12000 ICL03长低能量复发，04连续八轮两帧EOS

- 已读playbook、本文最新记录、manual/process/退出状态、203032快照及200032 review/status。前轮11610，本快照12050，现场12060→12110；manual=false，上一巡检exit0且有完整记录，同一外层1800秒session。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系及sample命令/无resume正确；torchrun/四rank均expandable_segments:True。收尾重新核验身份/非僵尸、manual和latest，无training-exit.json；训练日志无Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted。
- latest为step-00012000，COMPLETE于20:17:10.773063 UTC写成。progress step/next_batch12000、epoch0/world4，scheduler last_epoch12000/_step_count12001、LR8.224126458e-5/2.467237937e-4。四distcp为2093307384/2093524691/2093544414/2093533470B，distributed/.metadata1424894B，四rng各14613B；metadata SHA=e4a032e39758f7ae9335a30f243e6690b38c6c2a5eab3acfbdc96c15e12de5ae。12000/11500 signature一致，settings/model/eval/seed匹配配置，run/source配置相同；当前val清单及assembly_report SHA与签名一致。此为结构/签名检查，未加载恢复测试，未重读全量train清单计算指纹。
- 全1211训练日志点至12110普通CE/sample CE/梯度等数值有限，warmup/cosine LR逐点通过。11620–12080共47点：token first范围1.279454–1.411848、中位1.336749，residual6.017772–6.118741、中位6.073652；sample first1.162906–1.265028、中位1.198117，residual5.927703–5.995376、中位5.963857；clip前grad0.419062–0.704468、中位0.504551。
- 同区间step中位2.157801s、范围2.004025–2.321147；吞吐中位871.719音频秒/墙钟秒、范围797.227–931.900；wait中位0.0002862s、最大0.0004389s。samples310–391、中位357；frame填充95.1208–99.9625%、中位98.3833%，token91.9389–97.5306%、中位95.5444%；峰值显存53.549–57.060GiB。四卡64955/64123/64203/64143MiB（各81920），利用率23–98%，compute-apps仅四sample rank；RAM245GiB、可用1.7TiB、无swap，磁盘余579970.64GiB。训练持续更新、吞吐及等待稳定，无持续供数或资源耗尽证据。
- 12000 val对应512条清单，24轮val日志全部有限。token first/residual=1.394702123/6.080216800、sample=1.259412820/5.984096020，均低于11500的1.400719574/6.094197495和1.266966153/5.999355964。十五残余码本CE范围3.868577–6.852943，均较11500下降。普通CE/逐码本CE为token平均，sample逐句等权，目标first_sample_ce+0.3 residual_sample_ce；分口径记录，CE下降不代表生成质量通过。

|12000完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/严格全零WAV/两帧EOS|
|---|---|---|---|---|
|speaker_only|0.444444/0.321678|0.444444/0.328671|1.000000/0.428571|0/8、0/8、0/8|
|icl|0.111111/0.055944|0.111111/0.055944|1.250000/0.583333|0/8、0/8、1/8|

- SO/ICL summary于20:23:05.863287/20:28:32.976785 UTC完成；16份metrics/16份WAV可读、24kHz单声道、finite、时长匹配。目标ID/text、参考ID/source/ICL reference_text/reference_frames和greedy min2与11500相同；逐语言/英语规范化错误计数、截断计数与summary吻合。以下是波形和ASR证据，未主观试听；低能量定义为10ms RMS<0.001。
- ICL03从11000/11500的6.56/7.2s、最长0.55/0.47s低能量，重新变为185帧14.8s、EOS=true，约目标6.28s的2.36倍。RMS0.029535、低能量72.1622%，最长10.43s位于[0,10.43]s，与此前10秒以上前缀同类异常再次出现，两轮改善没有保持。ASR“再下为2.57的大概10公里,现在剩余的电量是80%”、CER0.666667，开头内容缺失/错误明显。不是400帧触顶，但无截断不代表有效生成；不能继续报告该长低能量已解决。
- 对该03问题，前序已核验实际tar输入/参考、目标codec哈希、裁剪/写WAV路径，并在8000/8500进行CPU分段ASR；没有输入损坏或失败填零依据，本轮配对/策略亦未变化。现有WAV足以确认低能量复发，但没有生成码及写入前float音频，无法进一步定位生成码、codec、裁剪或量化的贡献；不重复相同ASR检查、不据此改loss/LR或扩大最小生成帧数。
- ICL04仍2帧0.16s、EOS=true，从8500起连续八轮；RMS0.002332、低能量87.5%、最长0.08s。“字幕by索兰娅”不作有效内容证据。前序输入/codec/EOS边界与本轮配对未发现实现故障，尚无逐步logits根因证据，未干预训练。
- SO03为80帧6.4s、EOS=true，9000起连续七轮无触顶且接近目标时长；RMS0.036946、低能量18.75%、最长0.55s位于[2.20,2.75]s。ASR“而且半个小时现在是下午两点五十 骑了大概十公里 现在剩余的点量是80%”、CER0.25，比11500的0.416667低，仍有开头/电量错词，不能称全文正确。
- SO01从11500的26帧2.08s回到13帧1.04s，“张俊乃白色后”、CER0.5，额外时长本轮消失，但内容仍错，不能称稳定恢复。SO00为27帧2.16s，“I do, um, liquid spars.”、WER1，11500仅“The liquid spars.”的局部改善未保持，额外开头又出现。
- SO05为33帧2.64s/目标1.62s，“She can't disobey the windows so you need.”、WER2/CER2.076923。11000起连续三轮WER2，10500起连续四轮较长且严重偏离目标“W.A. Flinders Uni.”；RMS0.071618、低能量1.8939%、最长0.02s，非长静音解释。193032巡检已CPU重现10500/11000原SO05转写并做ICL对照；本轮同目标/参考，无新记录错配或ASR运行异常线索，不重复已完成的同模型复核，也不把ASR词句冒充实际试听。实际错音和ASR误听贡献仍未分开。
- SO02为52帧4.16s，“This isn't a clackling group in which like there are winners and losers, yeah, you know.”、WER0.125，开头/专名仍错；04为26帧2.08s，“愛惜人家必信先實了這麼多年”、CER0.615385，11500缺失的尾字“年”本轮重新出现，但正文未正确。06为48帧3.84s，“On the Jeffreeze Tube, they open a door and enter a swamp.”、WER0.166667，末词从swap恢复swamp，开头/专名仍错。
- ICL00为18帧1.44s，“The liquid spears.”、WER0.25，与11500同转写，未退为两帧。01为11帧0.88s，“煮至奶白色後”、CER0.166667，仅后/後简繁差异，不直接当发音错误比例；02为43帧3.44s，ASR匹配、WER0，10000起连续五轮，但时长较11500的4.16s缩短。05为21帧1.68s，“W.A. Flanders, uni.”、WER0.25，前两轮专名匹配未保持；06为45帧3.6s，“In the Jeffreeze tube, they open a door and enter a swap.”、WER0.166667，11500匹配本轮末词再次swap。
- 12目标5.28s：SO为55帧4.4s，“我那儿相信你跟师傅都看过我机械节目的卷威 videos”、CER0.551724，仍单次videos且正文退化，最长低能量0.09s。ICL为45帧3.6s，“相信你跟师父都看过我 寄给你们的绝位 videos”、CER0.379310，较11500的3.04s略长、正文更接近，但仍只有一次videos；11000双video恢复后已连续两轮未保持，最长低能量0.13s。不能凭时长或CER改善认定尾部恢复。
- 较11500，SO基础EN WER0.472222→0.444444、规范化0.5→0.444444，ZH CER0.559524→0.428571；ICL EN WER0.027778→0.111111，ZH CER0.547619→0.583333。SO部分句子改善与05严重偏离并存；ICL03长低能量复发、04早停和12尾缺词持续。根据当前及既有诊断没有可验证的实现修复依据，不凭8句调整超参。下一轮12500重点核查ICL03复发是否持续、04/12、SO00/01反复及05偏离。
- 收尾12110：token first/residual=1.349355696/6.098105239，sample=1.219836356/5.982526190，目标=3.014594213，grad=0.471596，LR=8.1910384881e-05/0.000245731154643，step=2.030109s、wait=0.0002669s。12000完整评估后110次真实更新/11日志点正常，最新仍12000。
- 实际执行cat/tail、Python JSON/YAML/proc身份/allocator、checkpoint结构/签名/SHA、512清单条数、全日志有限值/LR/统计/磁盘检查、nvidia-smi/free/df，以及soundfile/numpy逐句波形、配对和summary计数核验，均通过；仅追加本文并检查文档空白差异。保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/ack/评估原件，未发信号、恢复或重启，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。尚未到19000，不执行最终冻结检查或写final-verification。本次单轮巡检结束。


### 2026-09-11T21:02:36.213239+00:00 — 训练至12640；12500 ICL03再次恢复短时长，04连续九轮两帧EOS

- 已读playbook、本文最新记录、manual/process/退出状态、210032快照及203032 review/status。前轮12110，本快照12590，现场12600→12640；manual=false，上一巡检exit0且记录完整，同一外层1800秒session。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系及sample命令/无resume正确；torchrun/四rank均expandable_segments:True。收尾再核对身份/非僵尸、manual和latest，无training-exit.json；日志无Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted。
- latest为step-00012500，COMPLETE于20:46:37.328542 UTC写成。progress step/next_batch12500、epoch0/world4，scheduler last_epoch12500/_step_count12501、LR8.071908330e-5/2.421572499e-4。四distcp为2093307384/2093524691/2093544414/2093533470B，distributed/.metadata1424894B，四rng各14613B；metadata SHA=247b5a77270b1a48be7d9b3cb43f5bcd580da15b706e8bbfdef137708b889380。12500/12000 signature一致，settings/model/eval/seed匹配配置，run/source配置相同；当前val清单及assembly_report SHA匹配签名。为结构/签名检查，未加载恢复测试或重读全量train清单计算指纹。
- 全1264训练日志点至12640普通CE/sample CE/梯度等有限，warmup/cosine LR逐点通过。12120–12610共50点：token first范围1.260565–1.413842、中位1.335587，residual6.015458–6.109851、中位6.059289；sample first1.127174–1.238413、中位1.196932，residual5.906023–6.005787、中位5.950015；clip前grad0.427687–0.599634、中位0.505833。
- 同区间step中位2.134114s、范围2.043516–2.341776；吞吐中位877.169音频秒/墙钟秒、范围802.741–932.387；wait中位0.0002820s、最大0.0003698s。samples309–390、中位354.5；frame填充94.4833–99.8333%、中位98.1646%，token91.6333–97.55%、中位95.2208%；峰值显存53.553–56.534GiB。四卡64955/64123/64203/64143MiB（各81920），利用率21–51%，compute-apps仅四sample rank；RAM246GiB、可用1.7TiB、无swap，磁盘余579847.48GiB。更新持续、吞吐/等待稳定，无持续供数或资源压力依据。
- 12500 val对应512条清单，25轮val日志全部有限。token first/residual=1.387445937/6.063935260、sample=1.250385963/5.967129879，均低于12000的1.394702123/6.080216800和1.259412820/5.984096020。十五残余码本CE范围3.855998–6.837386，均较12000下降。普通CE/逐码本CE仍token平均，sample逐句等权，目标first_sample_ce+0.3 residual_sample_ce；分口径记录，不将CE下降当作生成质量通过。

|12500完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/严格全零WAV/两帧EOS|
|---|---|---|---|---|
|speaker_only|0.444444/0.335664|0.472222/0.349650|1.000000/0.559524|0/8、0/8、0/8|
|icl|0.083333/0.048951|0.083333/0.048951|1.500000/0.488095|0/8、0/8、1/8|

- SO/ICL summary于20:52:25.083859/20:57:15.634068 UTC完成；16metrics/16WAV可读、24kHz单声道、finite、时长匹配。目标ID/text、参考ID/source/ICL reference_text/reference_frames及greedy min2与12000相同；逐语言/英语规范化错误计数、截断计数与summary吻合。以下是波形统计和ASR证据，未主观试听；低能量定义为10ms RMS<0.001。
- ICL03由12000的185帧14.8s、10.43s低能量前缀，恢复为82帧6.56s/目标6.28s、EOS=true；RMS0.038067、低能量23.3232%、最长仅0.45s位于[0,0.45]s。ASR“也起了半小小时,现在是下午2点50,起了大概10公里,先剩深渔的电量是80%”、CER0.5，仍有额外字/错词。11000/11500改善后12000复发，本轮再次缓解不能称稳定解决。前序tar输入/参考/codec及裁剪/写WAV核查、8000/8500 CPU分段ASR已记录；本轮没有新输入或后处理故障线索，不重复已通过检查。
- ICL04仍2帧0.16s、EOS=true，从8500起连续九轮；RMS0.000622、低能量87.5%、最长0.08s。ASR“字幕by索兰娅”不是有效内容证据。配对/策略相同，前序EOS边界检查无实现故障依据，缺逐步logits证据，不提高min_new_frames掩盖早停。
- SO03为84帧6.72s、EOS=true，9000起连续八轮无触顶且时长接近目标；RMS0.037206、低能量23.3631%、最长0.5s位于[3.72,4.22]s。ASR“特缺棒杖协生在是下午2点50 骑了大概10公里 先生续的电量是80%”、CER0.611111，比12000的0.25明显变差，开头错误仍在，时长正常不保证内容正确。
- SO00为23帧1.84s，接近目标1.81s，但ASR“I didn't know I'd quiz spars.”、WER1.5，比12000的1更差；不能将缩短时长当作额外内容解决。SO01为13帧1.04s，“張九乃白色後”、CER0.666667，11500额外时长之后连续两轮恢复，但正文仍错，简繁也影响评分。
- SO05仍33帧2.64s/目标1.62s，ASR“Chewbacca out there, defeat Flanders Uni.”、WER1.25/CER1.692308，较前三轮WER2有所下降，仍明显多/错内容；10500起连续五轮较长且严重偏离目标。RMS0.074554、低能量3.0303%、最长0.03s，非长静音解释。193032巡检已CPU复核10500/11000 SO原转写及ICL对照；本轮配对相同、没有新的ASR运行异常或文本错配线索，不重复同模型转写，也不把ASR专名/额外词当作已试听事实。
- SO02为54帧4.32s，“This is a McClacklin group in which, like, they're winners and losers, yeah, you know.”、WER0.1875，专名/there are仍错；04为28帧2.24s，“還是人家畢竟先世了這麼多年”、CER0.538462，尾字“年”连续两轮出现，正文仍错。06为48帧3.84s，“As in the Jeffreeze Tube, they open a door and enter a swamp.”、WER0.166667，末词保留、开头及专名错误持续。
- ICL00为17帧1.36s，“the liquid spears.”、WER0.25，未退为两帧；01为13帧1.04s，“組織奶白色後”、CER0.5。02为46帧3.68s、ASR匹配，10000起连续六轮；05为20帧1.6s、“W.A. Flinders-Uni.”、WER0，本轮专名再次恢复。06为46帧3.68s，“In the Jeffreeze tube, they open a door and enter a swap.”、WER0.166667，末词swap连续两轮、swamp未稳定。
- 12目标5.28s：SO为51帧4.08s，“我相信你跟师傅都看过我寄给你们的卷微调”、CER0.482759，目标两次video均未以对应英文词形出现；正文比12000更接近也不能说明尾部恢复，最长低能量0.04s。ICL为44帧3.52s，“相信你跟师父都看过 寄给你们内卷 我也得 video video”、CER0.241379，两次video本轮重新进入转写，但正文缺“我”等、额外“我也得”等仍在；最长低能量0.06s。11000也曾仅恢复一轮，不能因本轮CER降低或两次词形出现认定完整/稳定发音。
- 较12000，SO基础EN WER持平0.444444、规范化0.444444→0.472222，ZH CER0.428571→0.559524；ICL EN WER0.111111→0.083333，ZH CER0.583333→0.488095。ICL03波形和12词形局部改善，与04持续完全早停并存；SO05分数降低仍严重偏离，SO00/03内容变差。当前及既有诊断没有可验证的训练实现修复依据，不凭8句调整loss/LR/生成上限。下一轮13000跟踪ICL03/12改善持续性、04、SO00/05额外内容及SO03正文。
- 收尾12640：token first/residual=1.271378977/6.017027372，sample=1.183955499/5.940345629，目标=2.966059188，grad=0.483707，LR=8.02846872408e-05/0.000240854061723，step=2.225867s、wait=0.0002819s。12500完整评估后140次真实更新/14日志点正常，最新仍12500。
- 实际执行cat/tail、Python JSON/YAML/proc身份/allocator、checkpoint结构/签名/SHA、512清单条数、全日志有限值/LR/统计/磁盘检查、nvidia-smi/free/df及soundfile/numpy逐句波形、配对/summary计数核验，均通过；仅追加本文并检查文档空白差异。保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/ack/评估原件，未发信号、恢复或重启，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。尚未到19000，不执行最终冻结检查或写final-verification。本次单轮巡检结束。


### 2026-09-11T21:34:30.377178+00:00 — 训练至13180；13000 ICL03长低能量再复发，尾段转写出现参考句相似内容

- 已读playbook、本文最新记录、manual/process/退出状态、213032快照及210032 review/status。前轮12640，本快照13070，现场13090→13180；manual=false，上一巡检exit0且记录完整，同一外层1800秒session。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系和sample命令/无resume正确；torchrun/四rank均expandable_segments:True。收尾再核对身份/非僵尸、manual和latest，无training-exit.json；日志无Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted。
- latest为step-00013000，COMPLETE于21:15:37.699012 UTC写成。progress step/next_batch13000、epoch0/world4，scheduler last_epoch13000/_step_count13001、LR7.915187570e-5/2.374556271e-4。四distcp为2093307384/2093524691/2093544414/2093533470B，distributed/.metadata1424894B，四rng各14613B；metadata SHA=4c78a30c0c4094a201ebec4649b22fc56da6e5ee3ba71fa2520644598391f69d。13000/12500 signature一致，settings/model/eval/seed匹配配置，run/source配置相同；当前val清单及assembly_report SHA匹配签名。为结构/签名检查，未加载恢复测试或重读全量train清单计算指纹。
- 全1318训练日志点至13180普通CE/sample CE/梯度等有限，warmup/cosine LR逐点通过。12650–13110共47点：token first范围1.235537–1.381640、中位1.323914，residual5.990989–6.095702、中位6.035718；sample first1.130207–1.225509、中位1.183931，residual5.902704–5.964026、中位5.929002；clip前grad0.415993–0.616019、中位0.494556。
- 同区间step中位2.148994s、范围2.057349–2.465895；吞吐中位877.743音频秒/墙钟秒、范围769.246–919.125；wait中位0.0002816s、最大0.0004213s。samples318–406、中位355；frame填充95.0542–99.7292%、中位98.5125%，token91.8944–98.6444%、中位95.7556%；峰值显存53.473–56.885GiB。四卡64955/64123/64203/64143MiB（各81920），利用率98–100%，compute-apps仅四sample rank；RAM245GiB、可用1.7TiB、无swap，磁盘余579714.28GiB。更新持续、吞吐/等待稳定，无持续供数或资源压力依据。
- 13000 val对应512条清单，26轮val日志全部有限。token first/residual=1.381199059/6.052829658、sample=1.243420999/5.954830796，均低于12500的1.387445937/6.063935260和1.250385963/5.967129879。十五残余码本CE范围3.849739–6.828002，均较12500下降。普通CE/逐码本CE仍token平均，sample逐句等权，目标first_sample_ce+0.3 residual_sample_ce；分口径记录，CE下降不等于生成质量通过。

|13000完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/严格全零WAV/两帧EOS|
|---|---|---|---|---|
|speaker_only|0.194444/0.111888|0.194444/0.111888|1.000000/0.440476|0/8、0/8、0/8|
|icl|0.111111/0.076923|0.111111/0.076923|1.125000/0.607143|0/8、0/8、1/8|

- SO/ICL summary于21:21:32.018007/21:27:31.742201 UTC完成；16metrics/16WAV可读、24kHz单声道、finite、时长匹配。目标ID/text、参考ID/source/ICL reference_text/reference_frames及greedy min2与12500相同；逐语言/英语规范化错误计数、截断计数与summary吻合。以下为波形和ASR证据，未主观试听；低能量定义为10ms RMS<0.001。
- ICL03由12500的82帧6.56s/最长0.45s低能量，退化为272帧21.76s、EOS=true，约目标6.28s的3.47倍；RMS0.030315、低能量60.4779%、最长13.02s位于[0,13.02]s。12500恢复仅一轮，且本轮前缀长于12000的10.43s。原ASR“现在是数据24比奇迹精确还慢,又起了半个小时,现在是下午2点50,起了大概10公里,现在剩余的电量是80%”、CER0.722222，出现目标没有的开头，需进一步区分长低能量影响与额外生成内容。
- 本轮新增有界CPU诊断：原WAV前13.02s RMS2.42928e-5/peak0.004028，无abs>=0.999削波；其后8.74s RMS0.047834/peak0.327698，同样无削波。使用已缓存faster-whisper-small快照536b0662742c02347bc0e980a01041f333bce120、local_files_only=True、CPU int8/4threads/1worker，沿用beam5/zh/vad_filter=False/condition_on_previous_text=False。原WAV重转写逐字匹配metrics，耗时1.981s；仅在内存中截取13.02s后尾段，经resample_poly 2/3到16k重转写，耗时1.793s，结果“现在是数据24比奇迹精确还慢 又齐了半个小时 现在是下午2点50 齐了大概10公里 现在剩余的电量是80%”。额外开头仍出现，不能仅由长低能量影响解释。尾段ASR将该开头定位到回映[13.02,15.74]s；此为模型时间估计，不是精确对齐或主观确认。
- 进一步核对ICL reference_text为“最高时速只有二十四，比骑自行车还慢。”、reference_frames38、ID emilia2:205e83c292110537_078_000，与本轮额外开头的24/比…还慢相似。按记录的tar_member来源（emilia2-00158.tar，offset290060288、size34177、133182帧/44100Hz）用decode_emilia_audio只读重解码，得到finite的3.02s/24k参考波形、RMS0.038916；同CPU ASR转写“最高時數只有24 比今天這還蠻”，耗时1.310s。这支持“输出可能重新包含参考句内容”的诊断线索；原始参考转写本身也不精确，尚不能据相似词形断言实际复制或模型根因。
- 只读复核train.py generate_sample：ICL拼接reference+target文本、以38参考帧为prefix，generated仅追加新预测帧；decode时拼接prefix+generated，再按参考帧占比裁剪波形。13000记录272新帧对应21.76s，38参考帧按80ms约3.04s；现有代码没有将参考WAV另行附加到成品的分支。该结构与时长不能单独排除codec行为或生成重新说出参考内容；缺生成码和写WAV前float信号，仍不足以认定裁剪bug或制定可验证修复。原WAV/metrics/summary不改，CPU原音频/尾段/参考三项诊断正常exit0，不下载、不占GPU，不改变正式评分或启用VAD。
- ICL04仍2帧0.16s、EOS=true，从8500起连续十轮；RMS0.000673、低能量87.5%、最长0.08s。“字幕by索兰娅”不作有效语音证据。本轮配对/策略同前，未提高min_new_frames掩盖早停。
- SO03为89帧7.12s、EOS=true，9000起连续九轮无触顶且时长接近目标；RMS0.036881、低能量20.6461%、最长0.52s位于[2.18,2.70]s。ASR“客气了半小时,现在是下午2点50,起了大概10公里 现在剩余的电量是80%”、CER0.388889，较12500的0.611111改善，开头仍错，时长正常不等于全文正确。
- SO05由连续五轮约2.56–2.72s明显偏离，缩为26帧2.08s/目标1.62s，“G.W.A. Flinders-Uni”、WER0.25，较12500的1.25显著改善，专名主体接近但多G，仍不能称稳定解决。SO00为23帧1.84s，“I did the liquid spars”、WER0.75，较12500的1.5改善但额外开头/错词持续。01为13帧1.04s，“張靜乃白色後”、CER0.666667，11500额外时长后连续三轮时长恢复，文字仍错。
- SO02为54帧4.32s，“This is a McLaughlin group in which there are winners and losers, yeah, you know.”、WER0.0625，缺like；04为26帧2.08s，“還是然你幸幸失了這麼多年”、CER0.769231，较12500退化。06为47帧3.76s，“Hunt the Jeffreeze tube, they open a door, and enter a swamp.”、WER0.166667，末词保留，开头/专名仍错。
- ICL00为19帧1.52s，“The liquid spears.”、WER0.25；01为13帧1.04s，“煮至奶白色後”、CER0.166667，仅简繁差异，不据此估发音错误比例。02为50帧4.00004s，转写与本轮SO02相同，WER0.0625、缺like；10000起六轮匹配本轮中断。05为20帧1.6s、“W.A. Flinders, Uni.”、WER0，连续两轮匹配。06为47帧3.76s，“In the Jeffreeze tube, they open a door and enter a swap.”、WER0.166667，末词swap连续三轮。
- 12目标5.28s：SO为52帧4.16s，“我相信你跟师父都看过我寄给你们那卷微的videos”、CER0.310345，主体更接近但仍一次videos，最长低能量0.03s。ICL为47帧3.76s，“相信一个师傅都看过我寄给你们内卷位的videos”、CER0.379310，本轮又只有一次videos、正文也变差，最长低能量0.04s。12500两次video恢复仍仅一轮，尾缺词继续反复，不因时长增大认定改善。
- 较12500，SO基础EN WER0.444444→0.194444、ZH CER0.559524→0.440476；ICL EN WER0.083333→0.111111、ZH CER0.488095→0.607143。SO05明显改善与ICL03/12退化并存，ICL04持续早停。本轮原音频/去前缀/真实参考的诊断新增了参考内容重现线索，仍未确定实现根因，不凭8句调整loss/LR/生成上限。下一轮13500跟踪ICL03前缀及参考句相似开头、04/12、SO05改善稳定性及SO00额外内容。
- 收尾13180：token first/residual=1.345753980/6.066491243，sample=1.226496028/5.937923500，目标=3.007873078，grad=0.540665，LR=7.85771845632e-05/0.000235731553689，step=2.081394s、wait=0.0002774s。13000完整评估后180次真实更新/18日志点正常，最新仍13000。
- 实际执行cat/tail、Python JSON/YAML/proc身份/allocator、checkpoint结构/签名/SHA、512清单条数、全日志有限值/LR/统计/磁盘、nvidia-smi/free/df、soundfile/numpy逐句波形及配对/summary计数；另执行CPU原WAV/内存尾段/参考ASR，原生tar参考解码，以及rg/sed只读核查生成/裁剪代码。最初猜测qwen3_train/codec.py不存在导致两条检索exit2，随后用rg --files定位到train.py成功核查；这是巡检检索错误，不是训练异常。其余检查成功，CPU进程均正常结束。唯一持久修改为追加本文并检查文档空白差异；保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/ack/评估原件，未发信号、恢复或重启，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。尚未到19000，不执行最终冻结检查或写final-verification。本次单轮巡检结束。


### 2026-09-11T22:03:08.589846+00:00 — 训练至13680；13500 ICL03长前缀暂退，双模式12转写均出现两次video

- 已读playbook、本文最新记录、manual/process/退出状态、220032快照及213032 review/status。前轮13180，本快照13600，现场13630→13680；manual=false，上一巡检exit0且记录完整，同一外层1800秒session。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系和sample命令/无resume正确；torchrun/四rank均expandable_segments:True。收尾再次核对身份/非僵尸、manual和latest，无training-exit.json；日志无Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted。
- latest为step-00013500，COMPLETE于21:45:51.684150 UTC写成。progress step/next_batch13500、epoch0/world4，scheduler last_epoch13500/_step_count13501、LR7.754238549e-5/2.326271565e-4。四distcp为2093307384/2093524691/2093544414/2093533470B，distributed/.metadata1424894B，四rng各14613B；metadata SHA=17e49bde1c4bdb4a688916222f74ebf6211caf5cec187dcd98d6b5c79d255c53。13500/13000 signature一致，settings/model/eval/seed匹配配置，run/source配置相同；当前val清单及assembly_report SHA匹配签名。为结构/签名检查，未加载恢复测试或重读全量train清单计算指纹。
- 全1368训练日志点至13680普通CE/sample CE/梯度等有限，warmup/cosine LR逐点通过。13190–13640共46点：token first范围1.239510–1.383450、中位1.306590，residual5.989838–6.066192、中位6.031501；sample first1.098005–1.240036、中位1.173671，residual5.885881–5.956456、中位5.919351；clip前grad0.415087–0.629333、中位0.487138。
- 同区间step中位2.158341s、范围1.989373–2.442337；吞吐中位875.256音频秒/墙钟秒、范围776.994–933.681；wait中位0.0002818s、最大0.0006238s。samples301–389、中位352.5；frame填充94.4417–99.9208%、中位98.2104%，token91.1861–97.7361%、中位95.3736%；峰值显存53.810–56.935GiB。四卡64955/64123/64203/64143MiB（各81920），利用率21–100%，compute-apps仅四sample rank；RAM245GiB、可用1.7TiB、无swap，磁盘余579647.53GiB。更新持续、吞吐/等待稳定，无持续供数或资源压力依据。
- 13500 val对应512条清单，27轮val日志全部有限。token first/residual=1.372399192/6.038604299、sample=1.236163851/5.938506231，均低于13000的1.381199059/6.052829658和1.243420999/5.954830796。十五残余码本CE范围3.827772–6.813012，均较13000下降。普通CE/逐码本CE仍token平均，sample逐句等权，目标first_sample_ce+0.3 residual_sample_ce；分口径记录，CE下降不等于生成质量通过。

|13500完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/严格全零WAV/两帧EOS|
|---|---|---|---|---|
|speaker_only|0.305556/0.146853|0.277778/0.132867|1.000000/0.333333|0/8、0/8、0/8|
|icl|0.111111/0.055944|0.111111/0.055944|1.375000/0.511905|0/8、0/8、1/8|

- SO/ICL summary于21:51:42.436599/21:56:37.760059 UTC完成；16metrics/16WAV可读、24kHz单声道、finite、时长匹配。目标ID/text、参考ID/source/ICL reference_text/reference_frames及greedy min2与13000相同；逐语言/英语规范化错误计数、截断计数与summary吻合。以下为波形和ASR证据，未主观试听；低能量定义为10ms RMS<0.001。
- ICL03由13000的272帧21.76s/最长13.02s低能量，回到93帧7.44s、EOS=true，目标6.28s；RMS0.039463、低能量26.7473%、最长0.49s位于[0,0.49]s。ASR“又起了半个小时,限制是下午2点50,起了大概10公里 限制于这电量是80%”、CER0.527778，上轮“24比…还慢”的参考句相似开头本轮未出现在转写中。该改善此前反复、不能称稳定解决；正文仍错。
- 上轮已CPU原WAV/去13.02s前缀尾段/真实tar参考三项对照，发现额外开头在去低能量后仍在且与参考句相似；也检查了reference+target输入、38参考帧和按帧占比裁剪路径，无直接实现故障证据。本轮配对/策略未变、长前缀与额外开头暂退，没有新线索值得重复相同CPU诊断。仍缺生成码/写入前float信号，不能仅据上轮词形相似判定复制或裁剪bug。
- ICL04仍2帧0.16s、EOS=true，从8500起连续十一轮；RMS0.001263、低能量87.5%、最长0.08s。“字幕by索兰娅”不作有效语音。既有输入/codec/EOS边界检查未发现可验证故障，本轮配对仍一致，未提高min_new_frames掩盖早停。
- SO03为89帧7.12s、EOS=true，9000起连续十轮无触顶且时长接近目标；RMS0.039299、低能量23.0337%、最长0.48s位于[1.09,1.57]s。ASR“后期的半个小时 现在是下午两点五时 骑了大概十公里 现在剩余的电量是反值八十”、CER0.194444，较13000的0.388889改善，但开头、五时/五十及百分之等仍错，不能称全文正确。
- SO00从23帧1.84s延长至31帧2.48s/目标1.81s，ASR“I know the um the liquid spares”、WER1，比13000的0.75变差，额外开头加重，最长低能量仅0.08s。SO05由26帧2.08s/“G.W.A. Flinders-Uni”/WER0.25变为28帧2.24s、“GWA, Flanders, UT.”、WER1，13000明显改善未保持；最长低能量0.04s，未回到此前2.64s以上但专名/末词仍错。此前CPU复核SO05异常已有记录，本轮无新增ASR运行失败线索，不重复同模型转写。
- SO01为13帧1.04s，“張俊乃白時候”、CER0.833333，11500额外时长后连续四轮时长恢复但内容仍错。02为51帧4.08s，ASR匹配、WER0，13000缺like本轮恢复；04为29帧2.32s，“愛是人家畢形先世來這麼多年”、CER0.692308。06为48帧3.84s，“Oak and the Jeffreeze tube, they open a door and enter a swamp.”、WER0.25，末词保留，额外/错误开头更多。
- ICL00为17帧1.36s，“The liquid spears.”、WER0.25，未退为两帧；01为12帧0.96s，“組織奶白色後”、CER0.5。02为45帧3.6s、ASR匹配，like本轮恢复。05为18帧1.44s，“W Flanders Uni”、WER0.5，前两轮匹配未保持且A未转写出；06为43帧3.44s，“In the Jeffreeze tube, they open a door and enter a swamp.”、WER0.083333，12000起连续三轮swap后本轮回到swamp，仍有专名差异。
- 12目标5.28s：SO为52帧4.16s，与13000同长，“我逆相信你跟师傅都看过我寄给你们的卷位 videos啊videos”、CER0.241379，两次videos重新出现但额外“我逆”等仍在，最长低能量0.04s。ICL为46帧3.68s，“相信你跟师父都看过 寄给你们的觉为 videos videos”、CER0.275862，也重新出现两次videos，但正文缺“我”等、填充词/专名仍错，最长低能量0.11s。不能继续沿用上一轮仅一次的判断；历史11000/12500都曾只恢复一轮，本轮仍需后续稳定性验证，不将ASR词形当作完整发音已验收。
- 较13000，SO基础EN WER0.194444→0.305556、规范化0.194444→0.277778，ZH CER0.440476→0.333333；ICL EN WER持平0.111111，ZH CER0.607143→0.511905。双模式12和ICL03局部改善，与04持续早停、SO00/05再次变差并存。当前及既有诊断未形成可验证实现修复，不凭8句调整loss/LR/生成上限。下一轮14000跟踪ICL03前缀/参考内容线索、04、双模式12是否维持两次video、SO00/05额外内容。
- 收尾13680：token first/residual=1.289159783/6.046240518，sample=1.155699858/5.918859471，目标=2.931357699，grad=0.499104，LR=7.69531586721e-05/0.000230859476016，step=2.153447s、wait=0.0002891s。13500完整评估后180次真实更新/18日志点正常，最新仍13500。
- 实际执行cat/tail、Python JSON/YAML/proc身份/allocator、checkpoint结构/签名/SHA、512清单条数、全日志有限值/LR/统计/磁盘、nvidia-smi/free/df及soundfile/numpy逐句波形、配对/summary计数核验，均通过；仅追加本文并检查文档空白差异。保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/ack/评估原件，未发信号、恢复或重启，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。尚未到19000，不执行最终冻结检查或写final-verification。本次单轮巡检结束。


### 2026-09-11T22:33:12.927917+00:00 — 训练至14220；14000 ICL05首次两帧EOS，双模式12尾部改善未保持

- 已读playbook、本文最新记录、manual/process/退出状态、223032快照及220032 review/status。前轮13680，本快照14140，现场14150→14220；manual=false，上一巡检exit0且记录完整，同一外层1800秒session。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032身份、父子关系和sample命令/无resume正确；torchrun/四rank均expandable_segments:True。收尾再次核对身份/非僵尸、manual和latest，无training-exit.json；日志无Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted。
- latest为step-00014000，COMPLETE于22:14:41.983109 UTC写成。progress step/next_batch14000、epoch0/world4，scheduler last_epoch14000/_step_count14001、LR7.589343039e-5/2.276802912e-4。四distcp为2093307384/2093524691/2093544414/2093533470B，distributed/.metadata1424894B，四rng各14613B；metadata SHA=442119f6c4b15486617a9669c1e9a6a41270e28ad920a510bf29dee2502a1d13。14000/13500 signature一致，settings/model/eval/seed匹配配置，run/source配置相同；当前val清单及assembly_report SHA匹配签名。为结构/签名检查，未加载恢复测试或重读全量train清单计算指纹。
- 全1422训练日志点至14220普通CE/sample CE/梯度等有限，warmup/cosine LR逐点通过。13690–14170共49点：token first范围1.251939–1.361816、中位1.315716，residual5.961829–6.060242、中位6.022010；sample first1.105535–1.223355、中位1.170898，residual5.853648–5.936704、中位5.907833；clip前grad0.421975–0.644162、中位0.492636。
- 同区间step中位2.155829s、范围2.012754–2.318807；吞吐中位872.648音频秒/墙钟秒、范围810.382–929.792；wait中位0.0002817s、最大0.0004172s。samples318–408、中位356；frame填充94.5083–99.6083%、中位98.3208%，token91.3056–98.175%、中位95.4583%；峰值显存53.390–56.817GiB。四卡64955/64123/64203/64143MiB（各81920），利用率19–69%，compute-apps仅四sample rank；RAM245GiB、可用1.7TiB、无swap，磁盘余579573.98GiB。更新持续、吞吐/等待稳定，无持续供数或资源压力依据。
- 14000 val对应512条清单，28轮val日志全部有限。token first/residual=1.366243097/6.025908917、sample=1.227712761/5.924752235，均低于13500的1.372399192/6.038604299和1.236163851/5.938506231。十五残余码本CE范围3.819947–6.796885，均较13500下降。普通CE/逐码本CE为token平均，sample逐句等权，目标first_sample_ce+0.3 residual_sample_ce；分口径记录，CE下降不等于生成质量通过。

|14000完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/严格全零WAV/两帧EOS|
|---|---|---|---|---|
|speaker_only|0.277778/0.181818|0.277778/0.181818|1.000000/0.511905|0/8、0/8、0/8|
|icl|0.166667/0.132867|0.166667/0.132867|1.000000/0.630952|0/8、0/8、2/8|

- SO/ICL summary于22:20:33.659422/22:25:17.087759 UTC完成；16metrics/16WAV可读、24kHz单声道、finite、时长匹配。目标ID/text、参考ID/source/ICL reference_text/reference_frames及greedy min2与13500相同；逐语言/英语规范化错误计数、截断计数与summary吻合。以下为波形和ASR证据，未主观试听；低能量定义为10ms RMS<0.001。
- ICL05首次退化为2帧0.16s、EOS=true，RMS0.000288、100%低能量、ASR为空/WER1；13500为18帧1.44s，12500/13000还曾匹配。本轮回查全部28轮05 metrics：500为400帧未EOS，1000–13500为17–23帧且EOS=true，14000是本run该句第一次两帧EOS。与ICL04持续早停分开记录，不能把英语其他句子较低WER掩盖这条完全失败。
- 针对新05早停只读核验目标及参考：目标emilia2:4318cc5dac2cbeca_599_000的codec为21×16、码值9–2047，SHA=052447a330197660a776001465b5f1a28a476cca0a4ea12d00416185824b5116，与当前val清单一致。按实际emilia2-00245.tar来源重读原生目标/参考音频，均finite、24k：目标1.62s/RMS0.077347（offset63178240、size17784、71442帧/44100Hz）；参考emilia2:4318cc5dac2cbeca_429_001为4.16s/RMS0.069428（offset61670912、size43764、183456帧/44100Hz）。reference_frames52、reference_text“She appeared distressed and had difficulty coping with our youngest daughter.”、greedy min2及配对与13500一致。没有原音频损坏、目标codec改变或配对变化证据；此检查没有重新编码参考codec或复现逐步logits。空生成不再跑ASR，缺生成码/分布证据，尚无可验证实现修复，不提高min_new_frames掩盖新早停。
- ICL04仍2帧0.16s、EOS=true，从8500起连续十二轮；RMS0.001882、低能量81.25%、最长0.08s。“字幕by索兰娅”不作有效语音。本轮ICL两帧EOS由1/8升至2/8，04持续问题未因其他指标改善解决。
- ICL03为85帧6.8s、EOS=true，RMS0.035391、低能量23.8235%、最长0.55s位于[0,0.55]s，13500/14000连续两轮没有长前缀，也没有13000参考句相似开头的转写。ASR“又起了半個小時,現在是下午2點50,起了大概4公里,現在剩餘的電量是80%”、CER0.5，目标十公里变4公里，仍不完整正确。13000原WAV/裁剪尾段/真实参考的CPU对照及代码核查已记录，本轮无新线索需重复，不能因两轮改善判为稳定解决。
- SO03为87帧6.96s、EOS=true，9000起连续十一轮无触顶且时长接近目标；RMS0.037311、低能量21.5517%、最长0.49s位于[2.21,2.70]s。ASR“具体的半个小时现在是下午两点五十 骑了大概十公里 现在生育的点量是百分之八十”、CER0.166667，较13500的0.194444略低，开头及电量仍错，不能称全文正确。
- SO01从连续四轮13帧1.04s退为32帧2.56s/目标0.98s，“張學乃白色後這一次剪輯剪輯的那堂重量字”、CER2.833333。类似10000/11500的额外内容和时长再次复发；RMS0.070188、低能量19.1406%、最长0.26s，无长低能量可解释额外1.5s；转写不等于已试听确认的实际重复。SO00为28帧2.24s，“I then known the liquid spears.”、WER0.75，较13500略改善但额外开头仍在。
- SO05为30帧2.4s，“She is Dobby Kid, Flanders Uni.”、WER1.25，13000的2.08s/0.25之后连续两轮再变长、错误增多；最长低能量0.04s，仍是内容偏离而非长静音。193032已对早前SO05原WAV做CPU复核，此处不重复同模型转写。02为54帧4.32s、ASR匹配，连续两轮；04为27帧2.16s，“愛惜神家畢竟現實了這麼多年”、CER0.615385。06为45帧3.6s，“the Jeffreeze tube they open a door and enter a swamp”、WER0.166667，末词保留，开头/专名仍错。
- 其余ICL：00为18帧1.44s，“The liquid spears.”、WER0.25，未退为两帧；01为12帧0.96s，“组织奶白色后”、CER0.333333；02为50帧4.00004s、ASR匹配，连续两轮。06为40帧3.20004s，“In the Jeffreeze tube, they open a door and enter a swamp.”、WER0.083333，swamp连续两轮保持，专名仍有差异。
- 12目标5.28s：SO为47帧3.76s，“我相信你跟师父都看我寄给你们的卷位 videos”、CER0.413793，13500双videos本轮又只一次，最长低能量0.03s。ICL为44帧3.52s，“相信你跟师父都看我鸡给鸣鸣的这种视频”、CER0.689655，双videos本轮不再出现且正文明显变差，最长低能量0.05s。两模式均变短、没有长静音，13500的尾部改善都仅保持一轮，不能将无截断计数当作内容完整。
- 较13500，SO基础EN WER0.305556→0.277778、规范化持平0.277778，ZH CER0.333333→0.511905；ICL EN WER0.111111→0.166667，ZH CER0.511905→0.630952。ICL03时长改善与05新早停、04持续早停及12尾部退化并存；SO01严重额外内容、05继续变差。本轮05历史/输入/codec检查无新的可验证实现故障，不凭8句改变loss/LR/生成策略。下一轮14500优先跟踪ICL05是否恢复、04/03及双模式12，SO01/05额外内容。
- 收尾14220：token first/residual=1.279399708/5.986763790，sample=1.127451361/5.878789253，目标=2.891088137，grad=0.418971，LR=7.51561291269e-05/0.000225468387381，step=2.237264s、wait=0.0002809s。14000完整评估后220次真实更新/22日志点正常，最新仍14000。
- 实际执行cat/tail、Python JSON/YAML/proc身份/allocator、checkpoint结构/签名/SHA、512清单条数、全日志有限值/LR/统计/磁盘、nvidia-smi/free/df、soundfile/numpy逐句波形与配对/summary计数，新增05全历史和目标codec哈希/形状/码值及tar原生目标/参考解码检查，均通过。唯一持久修改为追加本文并检查文档空白差异；保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/ack/评估原件，未发信号、恢复或重启，未建监控/Codex/定时器/subagent、提交/push/PR、删除或外发。尚未到19000，不执行最终冻结检查或写final-verification。本次单轮巡检结束。


### 2026-09-11T23:03:12.860156+00:00 — 训练至14750；14500 ICL05早停暂退，12尾缺词持续

- 已读playbook、最新文档、manual/process/退出状态、230032快照及223032 review/status。前轮14220，快照14670，现场14710→14750。manual=false，上一review exit0，同一外层session。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032、父子关系/sample命令/无resume核验通过，四rank及torchrun allocator正确。收尾身份复核通过，无退出文件或训练Traceback/OOM/Non-finite/ChildFailedError/Aborted。
- latest14500 COMPLETE于22:43:21.478036 UTC完成；progress step/next_batch14500、epoch0/world4，scheduler14500/_step_count14501。四distcp、distributed/.metadata及四rng齐全；metadata SHA=5ba6b60b1b1aeabfa7d139136b29c5c8b85bf996eaff72bcb21e8262a677dd83。14500/14000 signature相同，配置settings/model/eval/seed一致，val及assembly_report SHA匹配。为结构核验，未加载恢复测试。
- 全1475训练日志点有限且LR逐点符合原warmup/cosine。14230–14750统计如下（最小/中位/最大）：

- first_ce: 1.23657/1.3043/1.37365
- residual_ce: 5.96663/6.00997/6.0552
- first_sample_ce: 1.12459/1.16531/1.2157
- residual_sample_ce: 5.8586/5.8977/5.92701
- grad_norm: 0.435522/0.476925/0.595059
- step_seconds: 2.02509/2.14885/2.31133
- data_wait_seconds: 0.000237253/0.000284326/0.000605293
- audio_seconds_per_second: 817.684/880.157/921.761
- global_samples: 314/356/400
- frame_budget_fill: 0.949/0.983417/0.998125
- token_budget_fill: 0.920917/0.952694/0.981861
- peak_memory_gib: 53.9429/55.8178/56.6923

- 四卡64955/64123/64203/64143MiB，各81920，仅四sample rank；利用率19–97%。RAM245GiB、可用1.7TiB、无swap，磁盘579461.27GiB。无持续吞吐或资源异常。512条val清单、29轮val数值有限；14500 token CE=1.355666151/6.015670816，sample CE=1.220099888/5.913055338，均较14000下降，15码本CE均下降。普通/逐码本CE为token平均，优化为first_sample_ce+0.3 residual_sample_ce，不混用口径。
- speaker_only：EN基础WER/CER=0.333333/0.230769，规范化=0.416667/0.251748，ZH CER=0.369048。
- icl：EN基础WER/CER=0.083333/0.041958，规范化=0.083333/0.041958，ZH CER=0.595238。
- SO/ICL summary于22:49:14.894493/22:54:05.828164 UTC完成，各4EN/4ZH。16metrics/WAV可读、finite、24k单声道、时长正确，配对/策略与14000一致、逐句错误计数吻合；均无截断/全零，ICL两帧EOS为1/8。仅ASR及波形证据，未试听。
- ICL05恢复18帧1.44s，ASR“W.A. Flenders, Uni.”、WER0.25；上轮已核验输入和codec正常，本轮无新输入故障，不重复诊断。04连续十三轮2帧0.16s，RMS0.036822仍不代表内容恢复。03为7.2s、最长低能量0.55s，连续三轮无长前缀；转写正文较完整但简繁/数字影响CER0.527778，未宣称全文正确。
- SO01回到1.04s但仍错词；05缩为2s、WER0.5仍有额外内容。SO06变4.24s，ASR额外“That’s it, but it’s…”、WER0.416667；SO00仍额外开头、WER1。SO03为6.64s，最长低能量0.49s、CER0.166667，开头仍错。ICL06本轮ASR匹配，05恢复改善英语指标，但不代表整体质量通过。
- SO12仍单次videos；ICL12缩至3.2s，转写结束于“卷位”，两次video均未体现。无长低能量可解释尾缺词。既有输入/codec/裁剪和CPU诊断无可验证修复依据，保持超参；下一轮15000跟踪05恢复、04早停、03及12、SO06额外内容。
- 收尾日志：{"step": 14750, "train": {"first_ce": 1.266929741277958, "residual_ce": 5.9833648512355015, "grad_norm": 0.4738541543483734, "peak_memory_gib": 56.547504901885986, "audio_seconds_per_second": 864.2665690091508, "step_seconds": 2.20265375089366, "data_wait_seconds": 0.000267001916654408, "global_samples": 392.0, "global_audio_frames": 23796.0, "global_talker_tokens": 35003.0, "frame_budget_fill": 0.9915, "token_budget_fill": 0.9723055555555555, "lr_backbone": 7.335233364287027e-05, "lr_new": 0.0002200570009286108, "first_sample_ce": 1.157230416122748, "residual_sample_ce": 5.896701423489318}}；14500评估后250次真实更新正常。实际执行cat/tail、JSON/YAML/proc、checkpoint签名/SHA、全日志有限值/LR统计、nvidia-smi/free/df、soundfile/numpy音频和配对计数检查，均通过。仅追加本文并检查空白差异；未改代码/配置或训练、未重启/发信号、未新建监控或subagent、未提交/删除/外发。尚未19000，不写最终验收。本次单轮结束。


### 2026-09-11T23:33:02.548799+00:00 — 训练至15270；15000 ICL05维持恢复，04连续十四轮两帧EOS

- 已读playbook、本文最新记录、manual/process/退出状态、233032快照及230032 review/status。前轮14750，快照15200，首次现场日志15240→收尾15270。manual=false，前次review exit0，同一外层1800秒session。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032及父子关系/sample命令/无resume核验通过；torchrun和四rank均expandable_segments:True。收尾再次核验身份、非僵尸和manual，无training-exit.json，日志无Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted。
- latest15000 COMPLETE于23:12:12.159788 UTC完成；progress step/next_batch15000、epoch0/world4，scheduler last_epoch15000/_step_count15001，LR7.248873684e-5/2.174662105e-4符合原计划。四distcp分别2093307384/2093524691/2093544414/2093533470B，distributed/.metadata1424894B，四rng各14613B；metadata SHA=583d2224d047a9b875f1c4c3b4f337ecbe7730761d34cfb0e530da74026f0182。15000/14500 signature一致，settings/model/eval/seed与配置匹配，run/source配置相同；当前512条val清单和assembly_report SHA匹配签名。此次为结构/签名核验，未加载恢复测试或重读全量train清单计算指纹。
- 全1526训练日志点及30轮val数值有限，训练LR逐点通过原warmup/cosine核验；收尾15270新增点各值亦有限、LR连续。14760–15260共51点统计（最小/中位/最大）：

|指标|最小|中位|最大|
|---|---:|---:|---:|
|token first CE|1.239233|1.302953|1.346972|
|token residual CE|5.926014|5.990754|6.030316|
|sample first CE|1.108896|1.158728|1.204471|
|sample residual CE|5.845705|5.876528|5.911097|
|clip前grad_norm|0.426416|0.481380|0.595533|
|step秒|2.039044|2.139332|2.557724|
|data wait秒|0.0002448|0.0002832|0.0005188|
|音频秒/墙钟秒|748.165|876.395|926.019|
|global samples|321|357|404|
|frame填充|94.6542%|98.3250%|99.9333%|
|token填充|91.8278%|95.3944%|98.4194%|
|峰值显存GiB|53.5891|55.7843|56.8375|

- 四卡64955/64123/64203/64143MiB，各81920MiB，利用率63–98%，compute-apps仅四sample rank；RAM245GiB、可用1.7TiB、无swap，磁盘余579382.55GiB。训练持续推进，未见持续供数或资源压力；单步最大2.56s不足以认定吞吐故障。
- 15000 val覆盖512条清单：token first/residual=1.351651935/6.005644131，sample=1.214063931/5.902349040，均较14500下降；十五残余码本CE亦全部下降。普通和逐码本CE为token平均，sample为逐句等权，优化目标仍first_sample_ce+0.3 residual_sample_ce，不能混用口径或把loss下降当成生成质量通过。

|15000完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/全零WAV/两帧EOS|
|---|---|---|---|---|
|speaker_only|0.333333/0.209790|0.361111/0.223776|1.125000/0.488095|0/8、0/8、0/8|
|icl|0.138889/0.083916|0.138889/0.083916|1.000000/0.607143|0/8、0/8、1/8|

- SO/ICL summary于23:18:14.864616/23:23:11.893716 UTC完成；16份metrics/WAV可读、finite、24kHz单声道、时长匹配。目标id/text/language、参考id/source、ICL reference_text/reference_frames及greedy min2与14500相同；分语言及英语规范化错误计数、截断计数与summary吻合。以下为已存ASR和波形证据，未试听；低能量为10ms RMS<0.001。
- ICL05为19帧1.52s、EOS=true，ASR“W.A. Flinders-Uni.”、WER/CER0，连续两轮脱离14000的两帧早停，本轮专名也匹配；RMS0.065816、低能量9.21%、最长0.12s。04仍2帧0.16s、EOS=true，从8500起连续十四轮，RMS0.001162、低能量87.5%、最长0.08s；“字幕by索兰娅”不作有效生成内容证据。
- ICL03为85帧6.8s/目标6.28s，RMS0.035469、低能量23.24%、最长0.47s位于[2.71,3.18]s；连续四轮无长低能量前缀，未触顶。ASR“而起了半個小時現在是下午2點50,起了大概10公里 現在剩餘的電量是80%”、CER0.555556，仍有开头错误和简繁/数字口径影响，不能称全文正确。SO03为88帧7.04s，RMS0.034099、低能量22.44%、最长0.52s位于[2.42,2.94]s，ASR“目前要半个小时 现在是下午两点五时 起来大概10公里 现生鱼的电量是80%”、CER0.444444，高于14500的0.166667；时长正常与正文退化并存。
- SO00为25帧2s，“I know in the liquid spears.”、WER0.75，额外开头持续。SO01仍13帧1.04s，“张俊哪麽时候”、CER1；时长没有再扩张，但文本错误严重。SO05缩至24帧1.92s，“She'd not be as flint as uni.”、WER1.5/CER1.076923，从14500 WER0.5再次变差；RMS0.064854、低能量1.56%、最长0.03s，不能用静音解释。此前CPU ASR与原始输入/codec检查已记录，本轮配对无变化，未发现新的可验证修复依据，不重复同一诊断，也未把ASR原文当作实际听感。
- SO02为52帧4.16s，ASR匹配、WER0；04为27帧2.16s，“愛學者畢竟監視了這麼多年”、CER0.692308。SO06为48帧3.84s，“Now sit in the Jeffreeze tube, they open a door and enter a swamp.”、WER0.25；比14500的0.416667低，但额外开头仍在。ICL00为18帧1.44s，“The liquid spheres?”、WER0.5；01为13帧1.04s，“组织来白色后”、CER0.5；02为47帧3.76s，末尾“Yay, you know.”、WER0.0625；06为46帧3.68s，“In the Jeffree stoop, they open a door and enter a swamp.”、WER0.166667，专名本轮再次错误。
- 12目标5.28s：SO54帧4.32s，“我相信你跟师傅都看过我寄给你们的卷位是 videos”、CER0.344828，仍只有一次videos；最长低能量0.08s。ICL44帧3.52s，“相信你跟师父都看过我寄给你们的那些微调”、CER0.517241，仍未完整体现目标尾部的两次video，低能量仅0.57%、最长0.02s，时长比14500增加0.32s不足以认定恢复。
- 较14500，SO基础EN WER持平0.333333，规范化0.416667→0.361111，ZH CER0.369048→0.488095；ICL EN WER0.083333→0.138889，ZH CER0.595238→0.607143。05早停缓解、03无长前缀与04持续早停、12尾缺词及部分句子退化并存。既有输入/codec/裁剪诊断和本轮配对检查未提供新实现故障证据，保持超参。下一轮15500重点跟踪ICL04/05/03/12和SO00/05/06额外内容、SO03正文退化。
- 收尾15270：token first/residual=1.290353825/5.995244290，sample=1.159754443/5.861042111，目标=2.918067076，grad0.559884，LR7.154746537e-5/2.146423961e-4，step2.156267s、wait0.0002670s；15000完整评估后270次真实更新/27日志点持续推进。
- 实际执行cat/tail/ls/rg、Python JSON/YAML/proc身份与allocator、checkpoint文件/签名/SHA/scheduler、512清单条数、全日志有限值/LR和统计、nvidia-smi/free/df、soundfile/numpy逐句波形及配对/summary计数核验，均通过。唯一持久修改为追加本文，并执行git diff --check核验空白。保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/评估原件，未发信号/重启/恢复，未新增监控/Codex/定时器/subagent，未提交/push/PR、删除或外发。尚未到19000，不执行最终冻结检查或写final-verification。本次单轮巡检结束。


### 2026-09-12T00:02:38.253725+00:00 — 训练至15780；15500 ICL04持续早停，12出现双次video音译线索

- 已读playbook、本文最新记录、manual/process/退出状态、000032快照及233032 review/status。前轮15270，快照15720，现场15750→15780；manual=false，上一review exit0，同一外层1800秒session。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032、父子关系/sample命令/无resume正确；torchrun和四rank allocator均expandable_segments:True。收尾再核验身份/非僵尸、manual、latest，无training-exit.json；日志无Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted。
- latest15500 COMPLETE于2026-09-11T23:41:49.615883 UTC完成，progress step/next_batch15500、epoch0/world4；scheduler last_epoch15500/_step_count15501，LR7.073895899e-5/2.122168770e-4符合计划。四distcp分别2093307384/2093524691/2093544414/2093533470B、distributed/.metadata1424894B、四rng各14613B。metadata SHA=2c2538fc7933c8967360501576323c4ac380b85d3d9d4fa9d434b038df0b3492；15500/15000 signature一致，settings/model/eval/seed匹配当前配置，run/source配置一致，val和assembly_report SHA匹配。512条val清单、max19000、workers16/prefetch2核验通过。此为结构/签名核验，未加载恢复测试或重读全量train清单计算指纹。
- 全1577训练日志点及31轮val数值有限，LR逐点符合原warmup/cosine；新增收尾15780点数值亦有限、LR连续。15280–15770共50点统计（最小/中位/最大）：

|指标|最小|中位|最大|
|---|---:|---:|---:|
|token first CE|1.230913|1.296109|1.346058|
|token residual CE|5.933756|5.985847|6.028561|
|sample first CE|1.106450|1.156439|1.196614|
|sample residual CE|5.837615|5.866441|5.902387|
|clip前grad_norm|0.425019|0.489724|0.556700|
|step秒|2.011972|2.154818|2.387173|
|data wait秒|0.0002524|0.0002813|0.0004345|
|音频秒/墙钟秒|799.674|870.558|934.886|
|global samples|315|354.5|396|
|frame填充|95.6625%|97.9021%|99.8375%|
|token填充|91.7778%|95.1972%|98.1083%|
|峰值显存GiB|53.9771|55.7257|56.9679|

- 四卡64955/64123/64203/64143MiB，各81920，现场利用率22–69%、快照99–100%，compute-apps仅四sample rank；RAM245GiB、可用1.7TiB、无swap，磁盘余579333.73GiB。更新、吞吐及等待稳定，无持续供数或资源耗尽证据。
- 15500 val对应512条清单：token first/residual=1.345575199/5.993330805，sample=1.204930139/5.890042275，均较15000下降；十五残余码本CE也全部下降。普通/逐码本CE仍为token平均，优化仍为逐句等权的first_sample_ce+0.3 residual_sample_ce；分口径记录，不以CE下降推断生成质量通过。

|15500完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/全零WAV/两帧EOS|
|---|---|---|---|---|
|speaker_only|0.277778/0.195804|0.277778/0.188811|1.375000/0.678571|0/8、0/8、0/8|
|icl|0.138889/0.083916|0.138889/0.083916|1.000000/0.630952|0/8、0/8、1/8|

- SO/ICL summary于2026-09-11T23:47:31.086797/23:52:27.596665 UTC完成。16metrics/WAV可读、finite、24kHz单声道、时长正确；目标id/text/language、参考id/source、ICL reference_text/reference_frames和greedy min2与15000相同。分语言及英语规范化错误计数、截断计数吻合。以下仅为ASR/波形证据，未主观试听；低能量定义为10ms RMS<0.001。
- ICL04仍2帧0.16s、EOS=true，自8500连续十五轮；RMS0.046070、低能量68.75%、最长0.09s位于[0.07,0.16]s，较高RMS不表示内容恢复。“字幕by索兰娅”不作内容有效证据。05为20帧1.6s、EOS=true，ASR“W.A. Flinders-Uni.”、WER/CER0，连续三轮脱离14000早停，连续两轮专名匹配；RMS0.066340、低能量10%、最长0.1s。
- ICL03为89帧7.12s/目标6.28s，RMS0.041167、低能量20.37%、最长0.54s位于[1.60,2.14]s；连续五轮无长低能量前缀。“又汽了半個小時,現在是下午2點50,汽了大概10公里,現在剩餘的電量是80%”、CER0.527778，较15000的0.555556低，含简繁/数字和同音转写影响，未认定全文准确。SO03为86帧6.88s，RMS0.041044、低能量24.13%、最长0.64s位于[2.29,2.93]s，无触顶或长静音，但ASR“秋期的半個小時先至下午兩點五四 記得大概四公里 先至於省於電量是80%”、CER0.694444，14500→15000→15500为0.166667→0.444444→0.694444。除简繁/数字口径外，转写中的四公里/两点五四等内容也偏离目标；原目标reference_asr本身CER0.138889，实际错音与ASR误听尚不能完全拆分，下一轮重点复核此趋势。
- SO00为24帧1.92s，“The liquid spears.”、WER0.25，本轮额外开头消失，专名仍错。01为13帧1.04s，“張建良拜唆後”、CER1，连续两轮高错误而时长正常。02为52帧4.16s，转写将“there are”变为“they're”、WER0.125。04为28帧2.24s，“愛是人家畢竟堅實了這麼多年”、CER0.538462。05为25帧2s，“Chidat Bia Flendis-Undy”、WER/CER1，低能量4%、最长0.03s；较15000 WER1.5低但仍严重偏离目标，不把缩短时长或分数改善当作恢复。06为47帧3.76s，“That's a Jeffreeze tube. They open a door and enter a swamp.”、WER0.25，与前轮分数持平，开头仍错。
- ICL00为18帧1.44s，“the liquid spears.”、WER0.25；01为12帧0.96s，“入籍奶白色後”、CER0.5；02为48帧3.84s，遗漏“like”、WER0.0625；06为45帧3.6s，“in the Jeffries too, but they open a door and enter a swap.”、WER0.25，比15000的0.166667高，末词再次swap并多出but。
- 12目标尾部为“微呃video啊video”：SO53帧4.24s，“往下下你跟师傅都看过 几个你们的卷威士 威丢啊 威丢”、CER0.655172；ICL46帧3.68s，“相信你跟師傅都看過我寄給你們的捐威迪歐啊威迪歐”、CER0.620690。两种转写都出现两次近似video的汉字音译及中间“啊”，与15000 SO单次videos、ICL“那些微调”不同，是尾部可能改善的证据，不能继续笼统写成本轮两次video均未体现。SO/ICL最长低能量仅0.03/0.02s，时长仍小于目标5.28s；混语被汉字转写、简繁及正文错词使CER升高，不能据此判定尾部更差，也不能仅凭音译认定发音完整或稳定恢复。需下一轮跟踪是否保持；未以ASR冒充试听。
- 较15000，SO基础EN WER0.333333→0.277778，规范化0.361111→0.277778，ZH CER0.488095→0.678571；ICL EN WER持平0.138889，ZH CER0.607143→0.630952。英文部分改善、SO03转写退化与12双次音译线索并存，不能只看汇总分数判断生成。既有原始输入/codec/裁剪和CPU诊断及当前配对检查未给出新的可验证实现修复依据；保持超参，不重复相同检查或凭8句调整loss/LR。下一轮16000跟踪ICL04/05/03、双模式12尾部、SO03正文及短句05/06。
- 收尾15780：token first/residual=1.324034366/5.954545655，sample=1.159071255/5.850933821，目标=2.914351401，grad0.482502，LR6.974684945e-5/2.092405483e-4，step2.083429s、wait0.0002996s。15500完整评估后280次真实更新/28日志点持续推进。
- 实际执行cat/tail、Python JSON/YAML/proc身份及allocator、checkpoint文件/签名/SHA/scheduler、512清单条数、全日志有限值/LR统计、nvidia-smi/free/df、soundfile/numpy逐句波形和配对/summary计数、03/12目标及reference_asr核对，均通过。唯一持久修改为追加本文，并执行git diff --check。未改变sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539和双模式；未改代码/配置/manual/metadata/数据/评估原件，未重启/发信号/恢复，未新增监控/Codex/定时器/subagent，未提交/push/PR、删除或外发。尚未19000，不执行最终冻结检查或写final-verification。本次单轮结束。


### 2026-09-12T00:33:28.098040+00:00 — 训练至16330；16000 SO03转写回升，12双次video线索未保持

- 已读playbook、本文最新记录、manual/process/退出状态、003032快照及000032 review/status。前轮15780，快照16250，现场16310→16330；manual=false，上一review exit0，同一外层1800秒session。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032、父子关系/sample命令/无resume正确；torchrun及四rank allocator为expandable_segments:True。收尾再次核验身份、非僵尸、manual和latest，无training-exit.json，日志无Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted。
- latest16000 COMPLETE于00:10:30.087194 UTC完成，progress step/next_batch16000、epoch0/world4；scheduler last_epoch16000/_step_count16001，LR6.896162699e-5/2.068848810e-4符合原计划。四distcp2093307384/2093524691/2093544414/2093533470B、distributed/.metadata1424894B、四rng各14613B齐全。metadata SHA=863bf391a56c1e8bc62653576831249292debb177068cb774364464c3d6f77f1；16000/15500 signature相同，settings/model/eval/seed与配置一致，run/source配置相同，当前val和assembly_report SHA匹配；512条val清单、max19000、workers16/prefetch2核验通过。为结构/签名核验，未加载恢复测试或重读全量train清单计算指纹。
- 全1633训练日志点及32轮val数值有限，训练LR逐点符合原warmup/cosine，收尾点再次通过有限值/LR检查。15790–16330共55点统计（最小/中位/最大）：

|指标|最小|中位|最大|
|---|---:|---:|---:|
|token first CE|1.225313|1.282577|1.338850|
|token residual CE|5.935158|5.977413|6.011388|
|sample first CE|1.100212|1.143779|1.202453|
|sample residual CE|5.830717|5.856799|5.890609|
|clip前grad_norm|0.418649|0.472958|0.537557|
|step秒|2.037588|2.157170|2.386625|
|data wait秒|0.0002454|0.0002866|0.0004079|
|音频秒/墙钟秒|800.662|874.695|936.701|
|global samples|315|359|425|
|frame填充|94.7375%|98.4708%|99.7583%|
|token填充|91.0111%|95.6222%|97.7472%|
|峰值显存GiB|53.4987|55.8778|56.8325|

- 四卡64955/64123/64203/64143MiB，各81920，利用率93–99%，compute-apps仅四sample rank；RAM245GiB、可用1.7TiB、无swap，磁盘余579255.46GiB。训练更新、等待和吞吐稳定，无持续资源或供数压力。
- 16000 val对应512条清单：token first/residual=1.342793113/5.982146843，sample first/residual=1.205388582/5.878878064。较15500，token两项及sample residual下降，sample first从1.204930139略升0.000458444，不能笼统报告全部CE下降；十五残余码本CE均下降。普通/逐码本CE为token平均，优化为逐句等权first_sample_ce+0.3 residual_sample_ce；这次微小波动不足以调整超参。

|16000完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/全零WAV/两帧EOS|
|---|---|---|---|---|
|speaker_only|0.416667/0.230769|0.472222/0.223776|1.125000/0.511905|0/8、0/8、0/8|
|icl|0.138889/0.069930|0.138889/0.069930|1.125000/0.619048|0/8、0/8、1/8|

- SO/ICL summary于00:16:24.563095/00:21:16.004670 UTC完成。16metrics/WAV可读、finite、24kHz单声道、时长匹配；目标id/text/language、参考id/source、ICL reference_text/reference_frames、greedy min2与15500相同；分语言、英语规范化错误计数及截断计数吻合。以下为ASR与波形证据，未主观试听；低能量为10ms RMS<0.001。
- ICL04仍2帧0.16s、EOS=true，自8500连续十六轮；RMS0.001203、低能量93.75%、最长0.09s，“字幕by索兰娅”不作有效内容证据。05仍20帧1.6s，“W.A. Flinders, Uni.”、WER/CER0，连续四轮无早停、连续三轮专名匹配；RMS0.062441、低能量6.875%、最长0.04s。
- ICL03为86帧6.88s/目标6.28s，RMS0.039583、低能量22.38%、最长0.48s位于[2.62,3.10]s，连续六轮没有长低能量前缀。ASR“又齊了半個小時先是下午2點50,齊了大概10公里,現在剩餘的電量是80%”、CER0.555556，仍有正文错词及简繁/数字影响。SO03为84帧6.72s，RMS0.038371、低能量22.77%、最长0.52s位于[2.17,2.69]s；ASR“最起了半个小时,先是下午两点五时 起了大概10公里 先剩余支电量是80”、CER0.444444，较15500的0.694444回升，四公里错词本轮不再出现。14500→15000→15500→16000为0.166667→0.444444→0.694444→0.444444，持续单向退化未延续，但未恢复到14500水平，也不能称全文正确。
- SO00从24帧1.92s增至34帧2.72s，“I then turn on the liquid spears.”、WER1；15500额外开头消失没有保持，最长低能量仅0.14s，新增时长不能用长静音解释。01为14帧1.12s，“方式整點白色後”、CER0.833333。02为56帧4.48s，“That is a McLaughlin group in which, like, there are winners and two she is. Yeah, you know.”、WER0.25，开头及losers转写错误增加。04为26帧2.08s，“埃西爾亞畢竟現實了這麼多年”、CER0.692308。05增至29帧2.32s，“She W.A. Flanders you and it.”、WER1.25，额外内容及专名错词仍在，低能量2.16%、最长0.03s。06为47帧3.76s，“That's in the jiffy's tube. They open a door and enter a swamp.”、WER0.166667，分数较15500改善但开头/专名仍错。
- ICL00为19帧1.52s，“the liquid spars.”、WER0.5；01为11帧0.88s，“組織奶白伺候”、CER0.666667；02为47帧3.76s，“This is a Make Laughlin group in which like, there are winners and losers. Yeah, you know.”、WER0.125，专名分词错误。06为43帧3.44s，“In the Jeffreeze tube, they open a door and enter a swamp.”、WER0.083333，15500的swap及额外but本轮消失。
- 12目标5.28s：SO60帧4.8s，“我们相信你跟师父都看过我集给你们那卷威尔斯 videos”、CER0.448276，最长低能量0.05s；ICL44帧3.52s，“相信你跟師父都看過 我寄給你們的主要Videos”、CER0.517241，最长低能量0.03s。两种模式均只转写出一次video；15500双次汉字音译线索没有保持。SO时长增加且CER降低仍不能证明尾部完整；ICL尾缺词风险持续。仍不将转写直接等同于听感，亦不因CER较低宣称稳定恢复。
- 较15500，SO基础EN WER0.277778→0.416667，规范化0.277778→0.472222，ZH CER0.678571→0.511905；ICL EN WER持平0.138889，ZH CER0.630952→0.619048。SO03转写回升、05维持生成与SO短句额外内容反复、04早停及12尾部未保持并存。既有输入/codec/裁剪和CPU诊断、当前配对检查未给出新实现故障证据；保持配置，不重复已完成诊断或凭8句调loss/LR。下一轮16500继续跟踪SO00/02/05额外内容、03正文、ICL04/05/03及双模式12。
- 收尾16330：token first/residual=1.305319091/5.985517361，sample=1.145719374/5.867592948，目标=2.905997259，grad0.524478，LR6.777500322e-5/2.033250097e-4，step2.065315s、wait0.0002975s。16000完整评估后330次真实更新/33日志点持续推进。
- 实际执行cat/tail、Python JSON/YAML/proc身份及allocator、checkpoint文件/签名/SHA/scheduler、512清单条数、全日志有限值/LR统计、nvidia-smi/free/df、soundfile/numpy逐句波形与配对/summary计数，均通过。唯一持久修改为追加本文，并执行git diff --check；sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539和双模式保持。未改代码/配置/manual/metadata/数据/评估原件，未重启/发信号/恢复，未新增监控/Codex/定时器/subagent，未提交/push/PR、删除或外发。尚未19000，不执行最终冻结检查或写final-verification。本次单轮结束。


### 2026-09-12T01:03:10.132696+00:00 — 训练至16860；16500 SO英语明显退化，00异常转写CPU复现

- 已读playbook、本文最新记录、manual/process/退出状态、010032快照及003032 review/status。前轮16330，快照16780，现场16810→16860；manual=false，上一review exit0，同一外层1800秒session。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032、父子关系/sample命令/无resume正确；torchrun及四rank allocator为expandable_segments:True。收尾再核验身份、非僵尸、manual及latest，无training-exit.json；日志无Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted。
- latest16500 COMPLETE于00:39:20.243816 UTC完成，progress step/next_batch16500、epoch0/world4；scheduler last_epoch16500/_step_count16501，LR6.715985241e-5/2.014795572e-4符合原计划。四distcp为2093307384/2093524691/2093544414/2093533470B，distributed/.metadata1424894B、四rng各14613B；metadata SHA=6fca86bd5c63a3bc41894c2c24ce0cea307b69ec59010f59e421d30701846977。16500/16000 signature一致，settings/model/eval/seed匹配配置，run/source配置相同，当前val及assembly_report SHA匹配；512条val、max19000、workers16/prefetch2核验通过。为结构/签名核验，未加载恢复测试或重读全量train清单指纹。
- 全1683训练日志点及33轮val数值有限，原warmup/cosine LR逐点通过；收尾重读至16860，新增点亦有限且backbone LR连续。16340–16830共50点统计（最小/中位/最大）：

|指标|最小|中位|最大|
|---|---:|---:|---:|
|token first CE|1.207718|1.282767|1.345717|
|token residual CE|5.912402|5.963404|6.011080|
|sample first CE|1.090687|1.144118|1.194882|
|sample residual CE|5.800523|5.841777|5.886376|
|clip前grad_norm|0.419847|0.468639|0.632447|
|step秒|1.979073|2.149251|2.403098|
|data wait秒|0.0002383|0.0002821|0.0004526|
|音频秒/墙钟秒|796.106|877.913|931.183|
|global samples|296|355.5|392|
|frame填充|95.8417%|97.7438%|99.8792%|
|token填充|91.6528%|95.0986%|97.7389%|
|峰值显存GiB|53.8156|55.4220|56.5304|

- 四卡64955/64123/64203/64143MiB，各81920；现场利用率20–25%、快照28–94%，compute-apps仅四sample rank。RAM245GiB、可用1.7TiB、无swap，磁盘余579209.84GiB。训练持续更新、吞吐/等待稳定，无持续资源或供数异常；单次低GPU利用率不足以判断卡死。
- 16500 val对应512条清单：token first/residual=1.335827405/5.971680827，sample=1.198596979/5.866323799，均较16000下降；十五残余码本CE亦均下降。普通/逐码本CE为token平均，sample逐句等权，优化仍为first_sample_ce+0.3 residual_sample_ce；CE下降与下述SO生成退化并存。

|16500完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/全零WAV/两帧EOS|
|---|---|---|---|---|
|speaker_only|0.777778/0.538462|0.805556/0.573427|1.125000/0.464286|0/8、0/8、0/8|
|icl|0.083333/0.055944|0.083333/0.055944|1.125000/0.535714|0/8、0/8、1/8|

- SO/ICL summary于00:45:16.597929/00:50:08.368338 UTC完成。16metrics/WAV可读、finite、24kHz单声道、时长正确；目标id/text/language、参考id/source、ICL reference_text/reference_frames及greedy min2与16000相同。分语言/英语规范化错误计数、截断计数吻合。以下为ASR/波形证据，未试听；低能量为10ms RMS<0.001。
- SO00为36帧2.88s/目标1.81s，RMS0.064099、低能量7.99%、最长0.17s，非长静音；目标“The the liquid spears.”被转写为“I know, and they're in terms of saying to wrong. This is...”，WER3/CER1.944444。16000为2.72s/WER1，本轮进一步偏离。SO05为27帧2.16s，“should that be a flint to see you in term.”、WER2.25/CER1.846154，较16000 WER1.25恶化，低能量5.09%、最长0.07s。两句贡献SO英语28个基础word errors中的21个；当前退化集中于短句而非所有模式同步失效。
- 为SO00新增有界CPU复核：先读qwen3_train/metrics.py的ASRScorer，使用已缓存faster-whisper small、local_files_only=True、CPU int8/4threads/1worker，沿用beam5/en/vad_filter=False/condition_on_previous_text=False；重转写16000 SO00、16500 SO00、16500 ICL00三个原WAV，均与原metrics逐字相同，耗时1.325/1.312/1.123s（不含加载）。SO16000原文“I then turn on the liquid spears.”、no_speech_prob0.008759，SHA=f8c6b8d29b127865b8363b3bb6eb007507c260b518562724265c08113d503af2；SO16500原文如上、no_speech_prob0.067735，SHA=67a5198a9612b59a8a5154520b08774bf6282417cf00aabf4e09fd1cab8da719；ICL16500为“the liquid spears.”、no_speech_prob0.015681，SHA=9c545b53f31bfe32873770669528862cc4058cc4f1f8fe3a1bdef2feed973442。排除了简单的旧评分文本错配，但同一ASR复现不是独立试听，不能单独区分实际错音与ASR误听。SO16500段末时间3.18s越出2.88s WAV，不作精确对齐；no_speech_prob不作准确率。诊断exit0，无下载、GPU占用或原评估文件修改。
- 本轮初次诊断路径搜索使用不存在的qwen3_train/evaluation*和猜测的whisper缓存目录，rg/ls报告路径不存在；随后rg定位metrics.py并由local_files_only模型加载成功。另一次env筛选无匹配返回1，非训练故障。以上仅巡检命令失误，未影响原现场；未把失败命令报告为检查通过。
- SO01为15帧1.2s，“張憲準點白色後”、CER0.833333；02为59帧4.72s，“This is a McLaughlin group in which, like, there are winners and losers to yay he tan.”、WER0.25，尾部仍异常。04为27帧2.16s，“愛醫師人家畢竟現實了這麼多年”、CER0.615385；06为47帧3.76s，“Asst in the Jeffreeze tube, they open a door and enter a swap.”、WER0.25，末词再次swap。
- ICL04仍2帧0.16s、EOS=true，从8500连续十七轮；RMS0.000954、低能量93.75%、最长0.09s，“字幕by索兰娅”不作有效内容证据。05为19帧1.52s，“W.A. Flinders-Uni”、WER/CER0，连续五轮未早停、连续四轮专名匹配，最长低能量0.04s。ICL00为17帧1.36s，“the liquid spears.”、WER0.25；01为12帧0.96s，“煮至奶白色後”、CER0.166667仅后/後差异，不直接当发音错误。02为47帧3.76s，缺like、WER0.0625；06为46帧3.68s，“In the Jeffries tube, they open a door and enter a swap.”、WER0.083333，专名匹配与末词错误并存。
- ICL03为84帧6.72s/目标6.28s，RMS0.038823、低能量21.58%、最长0.48s位于开头，连续七轮没有长低能量前缀。ASR“又起了半個小時,現在是下午2點50,起了大概10公里,現在生育的電量是80%”、CER0.555556，仍有正文与简繁/数字影响。SO03为78帧6.24s，RMS0.038404、低能量19.87%、最长0.55s位于[2.71,3.26]s；“二小半小时,现在是下午2点50, 洗了大概40公里现在剩余的电量是80%”、CER0.416667，略低于16000的0.444444，但出现40公里，不能以CER改善认定内容恢复。
- 12目标5.28s：SO56帧4.48s，“我们相信你跟师父都看过我寄给你们那卷位置 videos”、CER0.379310，最长低能量0.04s；ICL44帧3.52s，“相信你跟似乎都看过我寄给你们的卷位 videos”、CER0.379310，最长低能量0.05s。均仍单次videos，15500双次音译线索已连续两轮未保持，时长或CER改善不证明尾部恢复。
- 较16000，SO基础EN WER0.416667→0.777778、规范化0.472222→0.805556，ZH CER0.511905→0.464286；ICL EN WER0.138889→0.083333、ZH CER0.619048→0.535714。SO英语持续两轮变差需重点跟踪；本轮00复现及既有05诊断、输入/codec/裁剪与当前配对检查没有给出可验证实现修复依据。保持超参，不凭8句改变loss/LR；下一轮17000优先核对SO00/05严重偏离是否保持，同时跟踪02/06、03正文、ICL04和12。
- 收尾16860：token first/residual=1.324883812/5.968646186，sample=1.176212442/5.846397618，目标=2.930131727，grad0.522714，LR6.584920923e-5/1.975476277e-4，step2.126788s、wait0.0003190s；16500完整评估后360次真实更新/36日志点持续推进。
- 实际执行cat/tail/rg、Python JSON/YAML/proc身份及allocator、checkpoint结构/签名/SHA/scheduler、512清单条数、全日志有限值/LR统计、nvidia-smi/free/df、soundfile/numpy逐句波形和配对/summary计数，及上述CPU ASR诊断。有效检查通过，搜索失误如实记录。唯一持久修改为追加本文，并执行git diff --check；保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539和双模式。未改代码/配置/manual/metadata/数据/评估原件，未重启/发信号/恢复，未新增监控/Codex/定时器/subagent，未提交/push/PR、删除或外发。尚未19000，不执行最终冻结检查或写final-verification。本次单轮结束。


### 2026-09-12T01:32:22.754491+00:00 — 训练至17360；17000 SO短句错误率回落但未恢复，ICL05出现字母错误

- 已读playbook、本文最新记录、manual/process/退出状态、013032快照及010032 review/status。前轮16860，快照17310，现场17330→17360；manual=false，上一review exit0，同一外层1800秒session。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032、父子关系/sample命令/无resume正确；torchrun和四rank allocator为expandable_segments:True。收尾再次核验身份、非僵尸、manual和latest，无training-exit.json，日志无Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted。
- latest17000 COMPLETE于01:08:14.622155 UTC完成；progress step/next_batch17000、epoch0/world4，scheduler last_epoch17000/_step_count17001，LR6.533678961e-5/1.960103688e-4符合原计划。四distcp2093307384/2093524691/2093544414/2093533470B、distributed/.metadata1424894B、四rng各14613B齐全；metadata SHA=5a66eecd53a177165639b5417c2b26f9742101b7c0d3fba3bc33fc1760cc0668。17000/16500 signature一致，settings/model/eval/seed匹配配置，run/source配置相同，当前val和assembly_report SHA匹配；512条val、max19000、workers16/prefetch2核验通过。为结构/签名检查，未加载恢复测试或重读全量train清单指纹。
- 全1736训练日志点及34轮val数值有限，训练两组LR逐点符合原warmup/cosine，收尾再次通过。16870–17360共50点统计（最小/中位/最大）：

|指标|最小|中位|最大|
|---|---:|---:|---:|
|token first CE|1.188192|1.276078|1.336300|
|token residual CE|5.899215|5.944695|5.991153|
|sample first CE|1.088219|1.131426|1.175128|
|sample residual CE|5.797010|5.828422|5.854732|
|clip前grad_norm|0.415481|0.474812|0.569182|
|step秒|2.000639|2.142476|2.342042|
|data wait秒|0.0002415|0.0002811|0.0004620|
|音频秒/墙钟秒|801.284|879.159|932.126|
|global samples|307|352|403|
|frame填充|94.7917%|97.8917%|99.6958%|
|token填充|91.4194%|94.9889%|97.3611%|
|峰值显存GiB|53.6942|55.5405|56.7139|

- 四卡64955/64143/64203/64143MiB，各81920；现场利用率16–37%、快照24–72%，compute-apps仅四sample rank。RAM246GiB、可用1.7TiB、无swap，磁盘余579077.10GiB；训练持续更新，吞吐和等待稳定，无持续资源或供数故障证据。
- 17000 val对应512条清单：token first/residual=1.328171124/5.964460138，sample=1.191574447/5.857989870，均较16500下降；十五残余码本CE亦均下降。普通/逐码本CE为token平均，sample逐句等权，优化仍为first_sample_ce+0.3 residual_sample_ce；不以验证CE下降推断生成质量通过。

|17000完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/全零WAV/两帧EOS|
|---|---|---|---|---|
|speaker_only|0.472222/0.342657|0.472222/0.335664|1.125000/0.595238|0/8、0/8、0/8|
|icl|0.138889/0.069930|0.138889/0.069930|1.125000/0.511905|0/8、0/8、1/8|

- SO/ICL summary于01:14:16.204155/01:19:05.502096 UTC完成。16metrics/WAV可读、finite、24kHz单声道、时长匹配；目标id/text/language、参考id/source、ICL reference_text/reference_frames及greedy min2与16500相同；分语言/英语规范化错误计数及截断计数吻合。以下为ASR/波形证据，未试听；低能量定义为10ms RMS<0.001。
- SO00仍36帧2.88s/目标1.81s，ASR“I learned this in our centaurs.”、WER1.5/CER0.944444，低于16500的3/1.944444，但仍完全偏离目标“The the liquid spears.”；RMS0.069260、低能量4.17%、最长0.12s，非长静音。SO05为26帧2.08s，“She debut way, flying this uni.”、WER1.25/CER1.076923，低于16500的2.25/1.846154，仍有额外内容和严重错词。00/05合计11/17个SO英语基础word errors，仍是主要问题。上一轮已用CPU复现00原WAV、既有05诊断已记录；本轮配对和策略一致，没有新的评分失败线索，不重复同模型转写或把错误率回落称为恢复。
- SO01为13帧1.04s，“蟑螂在白色后”、CER0.5；02为54帧4.32s，“This is a McLaughlin group in which, like, they're winners and losers, yeah. Yattay.”、WER0.25，there are及尾部转写仍错。04为28帧2.24s，“阿姨實驗牙畢性先世了這麼多年”、CER0.846154，比16500的0.615385高；06为48帧3.84s，“That's in the Jeffreeze tube. They open a door and enter a swamp.”、WER0.166667，末词恢复swamp但开头/专名仍错。
- ICL04仍2帧0.16s、EOS=true，自8500连续十八轮；RMS0.001528、低能量87.5%、最长0.08s。“字幕by索兰娅”不作有效内容证据。05为17帧1.36s、EOS=true，连续六轮无两帧早停，但从16500的1.52s缩短，ASR“W.F. Flinders-Uni”、WER0.25/CER0.076923，连续四轮ASR匹配本轮中断；不能继续说专名/字母全部正确。低能量8.82%、最长0.1s，无长静音。
- ICL00为17帧1.36s，“the liquid spheres.”、WER0.5；01为12帧0.96s，“祝质奶白色后”、CER0.333333；02为48帧3.84s，“This is a McLaughlin group in which like there are winners and losers yet, you know.”、WER0.0625；06为45帧3.6s，“In the Jeffries tube, they open a door and enter a swap.”、WER0.083333，连续两轮末词swap。
- ICL03为82帧6.56s/目标6.28s，RMS0.037286、低能量20.73%、最长0.47s位于开头，连续八轮没有长低能量前缀。ASR“尤其的半个小时,现在是下午2点50,起的大概10公里,现在生育的电量是80%”、CER0.472222，低于16500的0.555556，但开头/剩余等正文仍错。SO03为88帧7.04s，RMS0.039323、低能量21.02%、最长0.5s位于[2.28,2.78]s，ASR“現在是下午2點50,起了大概10公里 現在剩餘的電量是80%”、CER0.638889，高于16500的0.416667，开头“又骑了半个小时”未体现。时长增加和无长静音不能证明全文生成；先保留此转写变化，不能直接将ASR遗漏当作实际音频缺失。
- 12目标5.28s：SO58帧4.64s，“我们相信你跟师父都看过我寄给你们的权威是 videos”、CER0.448276，最长低能量0.06s；ICL45帧3.6s，“小姐你跟师傅都看过我寄给你们的卷 videos”、CER0.379310，最长低能量0.02s。两种模式均仍单次videos，15500双次音译线索已连续三轮未保持；时长/CER小幅变化不足以认定尾部恢复。
- 较16500，SO基础EN WER0.777778→0.472222、规范化0.805556→0.472222，ZH CER0.464286→0.595238；ICL EN WER0.083333→0.138889，ZH CER0.535714→0.511905。SO英语单向退化未持续，但00/05严重偏离仍在，SO03/04及ICL05本轮有退化；ICL04早停和12尾缺词风险持续。既有输入/codec/裁剪及CPU诊断、当前配对检查没有给出新可验证实现修复依据，保持配置，不凭8句改变loss/LR。下一轮17500重点跟踪00/05、SO03开头与04正文、ICL05缩短/字母错词、04/03及双模式12。
- 收尾17360：token first/residual=1.245122393/5.937611011，sample=1.133449378/5.851034264，目标=2.888759657，grad0.457395，LR6.401279324e-5/1.920383797e-4，step2.098862s、wait0.0003018s；17000完整评估后360次真实更新/36日志点持续推进。
- 实际执行cat/tail、Python JSON/YAML/proc身份及allocator、checkpoint结构/签名/SHA/scheduler、512清单条数、全日志有限值/LR统计、nvidia-smi/free/df、soundfile/numpy逐句波形与配对/summary计数，均通过。唯一持久修改为追加本文，并执行git diff --check。sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539和双模式保持；未改代码/配置/manual/metadata/数据/评估原件，未重启/发信号/恢复，未新增监控/Codex/定时器/subagent，未提交/push/PR、删除或外发。尚未19000，不执行最终冻结检查或写final-verification。本次单轮结束。


### 2026-09-12T02:03:19.445458+00:00 — 训练至17910；17500 SO05延长且偏离加重，ICL12尾部再次缩短

- 已读playbook、本文最新记录、manual/process/退出状态、020032快照及013032 review/status。前轮17360，快照17830，现场17880→17910；manual=false，上一review exit0，同一外层1800秒session。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032、父子关系/sample命令/无resume正确；torchrun及四rank allocator为expandable_segments:True。收尾再核验身份、非僵尸、manual和latest，无training-exit.json；日志无Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted。
- latest17500 COMPLETE于01:37:26.178548 UTC完成；progress step/next_batch17500、epoch0/world4，scheduler last_epoch17500/_step_count17501，LR6.349563023e-5/1.904868907e-4符合原计划。四distcp2093307384/2093524691/2093544414/2093533470B、distributed/.metadata1424894B、四rng各14613B齐全；metadata SHA=6d8baaf31e93e8db023c88edc2fb7f48813e60b2006326102b4c1fbe1f38c90a。17500/17000 signature一致，settings/model/eval/seed匹配配置，run/source配置相同，当前val及assembly_report SHA匹配；512条val、max19000、workers16/prefetch2核验通过。为结构/签名检查，未加载恢复测试或重读全量train清单指纹。
- 全1790训练日志点及35轮val数值有限，两组LR逐点符合原warmup/cosine；收尾新增17910点亦通过有限值及LR核验。17370–17900共54点统计（最小/中位/最大）：

|指标|最小|中位|最大|
|---|---:|---:|---:|
|token first CE|1.207501|1.268592|1.337708|
|token residual CE|5.871807|5.945939|5.974520|
|sample first CE|1.048381|1.127743|1.172819|
|sample residual CE|5.765318|5.817836|5.870398|
|clip前grad_norm|0.409512|0.462995|0.663804|
|step秒|1.972105|2.156838|2.400612|
|data wait秒|0.0002354|0.0002838|0.0004590|
|音频秒/墙钟秒|772.570|872.736|925.468|
|global samples|314|356.5|385|
|frame填充|95.0583%|98.1979%|99.8958%|
|token填充|91.5306%|95.4736%|97.8222%|
|峰值显存GiB|53.5594|55.5588|56.6428|

- 四卡64955/64143/64203/64143MiB，各81920，现场利用率21–99%，compute-apps仅四sample rank。RAM245GiB、可用1.7TiB、无swap，磁盘余578858.03GiB；更新、吞吐和等待稳定，无持续资源或供数故障。
- 17500 val对应512条清单：token first/residual=1.321818884/5.953832706，sample=1.183621977/5.847852468，均较17000下降；十五残余码本CE也全部下降。普通/逐码本CE为token平均，sample逐句等权，优化仍为first_sample_ce+0.3 residual_sample_ce；分口径记录，不以CE下降判断生成质量通过。

|17500完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/全零WAV/两帧EOS|
|---|---|---|---|---|
|speaker_only|0.583333/0.496503|0.583333/0.496503|1.000000/0.583333|0/8、0/8、0/8|
|icl|0.111111/0.062937|0.111111/0.062937|1.125000/0.654762|0/8、0/8、1/8|

- SO/ICL summary于01:43:29.917906/01:48:24.938526 UTC完成。16metrics/WAV可读、finite、24kHz单声道、时长正确；目标id/text/language、参考id/source、ICL reference_text/reference_frames及greedy min2与17000相同；分语言/英语规范化错误计数与截断计数吻合。以下为ASR/波形证据，未试听；低能量定义为10ms RMS<0.001。
- SO05从17000的26帧2.08s增至45帧3.6s，为目标1.62s的2.22倍；RMS0.071413、低能量1.39%、最长0.03s，额外时长不是长静音。ASR“She died by death, because property owe and he flanders ring.”、WER2.75/CER3，高于17000的1.25/1.076923，再次严重偏离目标“W.A. Flinders Uni.”。本轮单句贡献SO英语21个基础word errors中的11个。既有SO05 CPU复现、原始输入/codec与本轮配对检查未提示记录错配或实现故障；未重复同模型转写，不把ASR词句当作实际听感。
- SO00缩至29帧2.32s，“Ananar and the liquid spares.”、WER0.75/CER0.611111，比17000的2.88s/WER1.5低，但仍额外开头及错词，低能量5.17%、最长0.12s。01为17帧1.36s，“凡姐究竟哪白色厚”、CER1，比前轮时长1.04s和CER0.5退化。02为53帧4.24s，“To this McLaughlin group in which there are winners and bleachers, yeah, you know.”、WER0.3125，开头、like和losers仍错。04为27帧2.16s，“愛是阿畢姐見識了這麼多年”、CER0.769231，仍较高；06为47帧3.76s，“House in the Jeffreeze tube, they open a door and enter a swamp.”、WER0.166667，开头/专名仍错。
- SO03仍88帧7.04s/目标6.28s，RMS0.035910、低能量23.72%、最长0.45s位于[1.05,1.50]s。ASR“休息了半個小時 現在是下午2點50 洗了大概10公里 現在剩餘的電量是80%”、CER0.555556，比17000的0.638889低；上轮未体现的开头本轮出现近似内容，但“休息/洗了”等仍偏离目标，不能称恢复。ICL03为90帧7.2s，RMS0.036246、低能量23.33%、最长0.5s位于[4.59,5.09]s，连续九轮无长低能量前缀；ASR“又起了半個小時,現在是下午2點50,起了大概10公里,現在剩餘的電量是80%”、CER0.527778，简繁/数字和正文错误仍影响指标。
- ICL04仍2帧0.16s、EOS=true，自8500连续十九轮；RMS0.000593、低能量87.5%、最长0.08s，“字幕by索兰娅”不作有效内容证据。05回到19帧1.52s，“W.A. Flinders-Uni”、WER/CER0，连续七轮未早停；17000字母W.F.错词本轮不再出现，但不能视为稳定恢复。
- ICL00为19帧1.52s，“the liquid spares.”、WER0.5；01为13帧1.04s，“逐渐来白色后”、CER0.5；02为45帧3.6s，ASR匹配、WER0。06为46帧3.68s，“In the Jeffreeze Tube, they open a door and enter a swap.”、WER0.166667，末词连续三轮swap，本轮专名也再次错误。
- 12目标5.28s：SO54帧4.32s，“我相信你跟似乎都看过我寄给你们的卷威斯的videos”、CER0.448276，仍单次videos，最长低能量0.05s；ICL从45帧3.6s缩至40帧3.2s，“相信你跟師傅都看過 寄給你們的捲”、CER0.689655，转写未体现任何video，最长低能量仅0.02s。较17000的单次video及CER0.379310，本轮尾部证据变差；15500双次音译线索已连续四轮未保持。不是400帧触顶，无长静音可解释尾部缺失；不将ASR直接冒充试听或把无truncated标记当作内容完整。
- 较17000，SO基础/规范化EN WER0.472222→0.583333，ZH CER0.595238→0.583333；ICL EN WER0.138889→0.111111，ZH CER0.511905→0.654762。SO05严重延长/偏离、ICL12尾部缩短与部分句子改善并存。当前及既有输入/codec/裁剪/CPU诊断没有新的可验证实现修复依据，保持超参，不凭8句改变loss/LR。下一轮18000重点跟踪SO05、00/01、03/04正文、ICL04/05/03及12尾部。
- 收尾17910：token first/residual=1.297756989/5.959107473，sample=1.104551361/5.821075803，目标=2.850874102，grad0.566309，LR6.197463592e-5/1.859239078e-4，step2.067173s、wait0.0002630s；17500完整评估后410次真实更新/41日志点持续推进。
- 实际执行cat/tail、Python JSON/YAML/proc身份及allocator、checkpoint结构/签名/SHA/scheduler、512清单条数、全日志有限值/LR统计、nvidia-smi/free/df、soundfile/numpy逐句波形及配对/summary计数，均通过。唯一持久修改为追加本文，并执行git diff --check；sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539和双模式保持。未改代码/配置/manual/metadata/数据/评估原件，未重启/发信号/恢复，未新增监控/Codex/定时器/subagent，未提交/push/PR、删除或外发。尚未19000，不执行最终冻结检查或写final-verification。本次单轮结束。


### 2026-09-12T02:37:17.593745+00:00 — 训练至18500并进入评估；18000 SO01重复/延长，ICL04持续早停

- 已读playbook、本文最新记录、manual/process/退出状态、023032快照及020032 review/status。前轮17910，快照18360，现场18420→18500；manual=false，上一review exit0，同一外层1800秒session。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032、父子关系/sample命令/无resume正确；torchrun及四rank allocator为expandable_segments:True。现场和收尾再次核验身份、非僵尸、manual；无training-exit.json和训练Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted。
- 初查latest18000 COMPLETE于02:06:33.350922 UTC完成；progress step/next_batch18000、epoch0/world4，scheduler last_epoch18000/_step_count18001，LR6.163959757e-5/1.849187927e-4。四distcp2093307384/2093524691/2093544414/2093533470B、distributed/.metadata1424894B、四rng各14613B齐全；metadata SHA=36ba81713d182f2b1977d3308b1f0f4248283379539000f3916c840c61af8d73。18000/17500 signature一致，settings/model/eval/seed匹配配置，run/source配置相同，当前val及assembly_report SHA匹配；512条val、max19000、workers16/prefetch2核验通过。为结构/签名检查，未加载恢复测试或重读全量train清单指纹。
- 收尾脚本断言latest仍为18000时产生巡检AssertionError（stdin第5行，exec e95416），立即只读保存现场核查：同一组PID/start_ticks仍存活，训练正常推进到18500，latest已更新，无退出文件。该断言失败来自巡检期间状态变化，不是训练故障，未发送信号或恢复。新18500 COMPLETE于02:35:41.343338 UTC完成，metadata SHA=b155f1e904976e536d76bc19fd3004725209f9737a10c90f7a0e1148db90ac97，progress step/next_batch18500、epoch0/world4，scheduler18500/_step_count18501，LR5.977194099e-5/1.793158230e-4；与18000 signature相同，四distcp/.metadata/四rng齐全且大小同上。
- 全1845训练日志点/36轮val初查有限且两组LR逐点符合原warmup/cosine；收尾重新检查全部1850训练点和37轮val通过。17920–18450共54点统计（最小/中位/最大）：

|指标|最小|中位|最大|
|---|---:|---:|---:|
|token first CE|1.188522|1.262939|1.318842|
|token residual CE|5.884754|5.924083|5.973894|
|sample first CE|1.077042|1.122032|1.164832|
|sample residual CE|5.776059|5.804581|5.831287|
|clip前grad_norm|0.410568|0.468401|0.546257|
|step秒|1.990454|2.137219|2.359262|
|data wait秒|0.0002440|0.0002817|0.0004265|
|音频秒/墙钟秒|807.507|872.831|931.933|
|global samples|307|354|395|
|frame填充|94.2542%|98.0875%|99.8500%|
|token填充|91.2528%|95.1431%|97.7083%|
|峰值显存GiB|53.4832|55.5360|56.6746|

- 四卡64955/64143/64203/64223MiB，各81920，现场利用率17–27%，compute-apps仅四sample rank。RAM246GiB、可用1.7TiB、无swap，磁盘余578777.18GiB。训练持续更新，吞吐/等待稳定；收尾18500处于预期评估，不能把step停留视为卡死。
- 日志18420后出现原生解码警告“Audio tail padded for emilia2:52634db1dbf9fd7f_004_000: 11 samples at 44100 Hz”。只读核查sources.py：MP4时间戳尾差允许不足约1ms，11采样约0.249ms在该范围内，超范围会抛错；此为既有解码规则，不是本轮跳过数据或修改metadata。后续18430–18500更新有限，未实施处理。
- 18000 val对应512条清单：token first/residual=1.319551380/5.944678381，sample=1.183279056/5.838260904，均较17500下降，十五残余码本CE亦下降。收尾18500 val已完成：token=1.310134927/5.933481125，sample=1.171109491/5.826554053，继续下降且数值有限。普通/逐码本CE为token平均，sample逐句等权，优化仍为first_sample_ce+0.3 residual_sample_ce；不以CE下降代替生成质量验收。

|18000最近完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/全零WAV/两帧EOS|
|---|---|---|---|---|
|speaker_only|0.444444/0.265734|0.416667/0.258741|1.375000/0.726190|0/8、0/8、0/8|
|icl|0.083333/0.048951|0.083333/0.048951|1.000000/0.607143|0/8、0/8、1/8|

- 18000 SO/ICL summary于02:12:23.257647/02:17:21.681530 UTC完成。16metrics/WAV可读、finite、24kHz单声道、时长匹配；目标id/text/language、参考id/source、ICL reference_text/reference_frames及greedy min2与17500相同；分语言/英语规范化错误计数及截断计数吻合。以下仅为ASR/波形证据，未试听；低能量为10ms RMS<0.001。收尾18500仅SO1份metrics、ICL0份，两模式均无summary，不能把18000结论冒充18500评估通过。
- SO05从45帧3.6s缩至26帧2.08s，“She W.A. flanked his uni.”、WER0.75/CER0.615385，较17500的2.75/3明显改善，但仍额外She及专名错误；RMS0.073201、低能量2.40%、最长0.03s，不能称稳定恢复。SO01从17帧1.36s增至28帧2.24s，为目标0.98s的2.29倍；ASR“咱們去見點白色後期 究竟來白色後”、CER2.166667，重复/额外内容加重，RMS0.076451、低能量25.89%、最长0.29s，额外时长不是连续长静音。该句此前也有额外时长反复，本轮配对/策略不变。
- SO00为25帧2s，“I then then know which earrings.”、WER1.5，时长缩短仍内容偏离；02为52帧4.16s，“That is a McLaughlin group in which, like, they're winners and losers. Keh, yo.”、WER0.375，开头和尾部仍错。04为25帧2s，“還是阿畢竟堅持了這麼多年”、CER0.615385；06为48帧3.84s，“Us in the Jeffries tube, they open a door and enter a swamp.”、WER0.083333，开头仍错。
- SO03仍88帧7.04s，RMS0.040602、低能量22.73%、最长0.56s位于[2.70,3.26]s；ASR“不久的半時間,現在下午2點50,洗了大概10公里,現在剩餘的電量是80%”、CER0.638889，开头仍异常。ICL03为95帧7.6s，RMS0.040306、低能量25.26%、最长0.5s位于[3.56,4.06]s，连续十轮无长低能量前缀；“尤其的半个小时,现在是下午2点50, 吸了大概10公里,现在剩余的电量是80%”、CER0.388889，较17500低但仍错词。
- ICL04仍2帧0.16s、EOS=true，自8500连续二十轮；RMS0.001196、低能量87.5%、最长0.08s，“字幕by索兰娅”不作有效内容证据。05为20帧1.6s，“W.A. Flinders, uni.”、WER/CER0，连续八轮未早停、连续两轮转写匹配；00为18帧1.44s，“the liquid spears.”、WER0.25；01为12帧0.96s，“組織奈白色後”、CER0.666667；02为51帧4.0800417s，ASR匹配、WER0；06为46帧3.68s，“In the Jeffreeze Tube, they open a door and enter a swap.”、WER0.166667，末词连续四轮swap。
- 12目标5.28s：SO58帧4.64s，“王子相信你跟师父都看过我 给给你们那权威 微妙啊微妙啊”、CER0.586207，最长低能量0.04s；尾部出现两次“微妙”，有重复音节线索，但不能当作两次video正确发音。ICL46帧3.68s，“相信你跟師父都看過我寄給你們的圈圍丟”、CER0.689655，最长低能量0.06s；较17500延长0.48s并有近似单次video的“圍丟”线索，但两次目标video仍未可靠体现。仅凭时长或ASR音译不能认定尾部恢复。
- 较17500，SO基础EN WER0.583333→0.444444、规范化0.583333→0.416667，ZH CER0.583333→0.726190；ICL EN WER0.111111→0.083333，ZH CER0.654762→0.607143。SO05改善与01严重额外内容、ICL04早停及12尾部风险并存。当前及既有输入/codec/裁剪/CPU诊断没有新的可验证实现故障依据，保持配置，不凭8句调整loss/LR。下一轮优先检查18500完整评估、SO01/05/00、ICL04/03/12；接近19000但未满足最终验收条件。
- 收尾18500训练点：token first/residual=1.249462115/5.961059105，sample=1.131000016/5.820745950，目标=2.877223801，grad0.468155，LR5.977194099e-5/1.793158230e-4，step2.251460s、wait0.0002752s；18000评估后500次真实更新正常，18500评估进行中。本轮不等待其结束。
- 实际执行cat/tail/sed、Python JSON/YAML/proc身份及allocator、18000和18500 checkpoint结构/签名/SHA/scheduler、512清单条数、全日志有限值/LR统计、nvidia-smi/free/df、soundfile/numpy逐句波形和配对/summary计数。有效检查通过，状态变化导致的巡检断言失败已记录并澄清。唯一持久修改为追加本文并执行git diff --check；保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。未改代码/配置/manual/metadata/数据/评估原件，未重启/发信号/恢复，未新增监控/Codex/定时器/subagent，未提交/push/PR、删除或外发。尚未19000，不执行最终冻结检查或写final-verification。本次单轮结束。


### 2026-09-12T03:03:19.055379+00:00 — 训练至18950；18500 ICL01新增两帧早停，SO05再次延长

- 已读playbook、本文最新记录、manual/process/退出状态、030032快照及023032 review/status。前轮18500评估中，快照18870，现场18910→18950；manual=false，上一review exit0，同一外层1800秒session。launcher737569/start95204812、torchrun737573/start95204815、rank737607–610/start95205032、父子关系/sample命令/无resume正确；torchrun及四rank allocator为expandable_segments:True。收尾身份、非僵尸、manual核验通过，无training-exit.json；日志无Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted。
- latest仍18500 COMPLETE（02:35:41.343338 UTC），progress step/next_batch18500、epoch0/world4，scheduler last_epoch18500/_step_count18501，LR5.977194099e-5/1.793158230e-4符合原计划。四distcp2093307384/2093524691/2093544414/2093533470B、distributed/.metadata1424894B、四rng各14613B齐全，metadata SHA=b155f1e904976e536d76bc19fd3004725209f9737a10c90f7a0e1148db90ac97。18500/18000 signature一致，settings/model/eval/seed匹配配置，run/source配置相同，当前val及assembly_report SHA匹配；512条val、max19000、workers16/prefetch2核验通过。为结构/签名检查，未加载恢复测试或重读全量train清单指纹。
- 全1893训练日志点/37轮val初查有限且两组LR逐点符合原warmup/cosine；收尾全部1895训练点亦通过。18510–18930共43点统计（最小/中位/最大）：

|指标|最小|中位|最大|
|---|---:|---:|---:|
|token first CE|1.208638|1.257236|1.315321|
|token residual CE|5.862396|5.910053|5.970208|
|sample first CE|1.066896|1.115353|1.153100|
|sample residual CE|5.761508|5.789788|5.836935|
|clip前grad_norm|0.403755|0.458715|0.555522|
|step秒|1.978981|2.164086|2.376721|
|data wait秒|0.0002302|0.0002888|0.0003805|
|音频秒/墙钟秒|803.258|874.124|932.646|
|global samples|316|359|389|
|frame填充|93.9708%|98.5250%|99.6417%|
|token填充|90.7333%|95.9389%|97.3444%|
|峰值显存GiB|53.0959|55.7048|56.4398|

- 四卡64955/64143/64203/64223MiB，各81920，现场利用率17–26%，compute-apps仅四sample rank。RAM259GiB、可用1.7TiB、无swap，磁盘余578657.71GiB；吞吐/等待和真实更新正常，无持续资源或供数异常。18500 val对应512条：token first/residual=1.310134927/5.933481125，sample=1.171109491/5.826554053，及十五残余码本CE均较18000下降。普通/逐码本CE为token平均，sample逐句等权，目标仍first_sample_ce+0.3 residual_sample_ce。

|18500完整生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/全零WAV/两帧EOS|
|---|---|---|---|---|
|speaker_only|0.972222/0.461538|1.000000/0.468531|1.250000/0.750000|0/8、0/8、0/8|
|icl|0.138889/0.062937|0.138889/0.062937|1.125000/0.642857|0/8、0/8、2/8|

- SO/ICL summary于02:41:57.637840/02:46:52.159244 UTC完成。16metrics/WAV可读、finite、24kHz单声道、时长匹配；目标id/text/language、参考id/source、ICL reference_text/reference_frames及greedy min2与18000相同；分语言/英语规范化错误计数与截断计数吻合。以下为ASR/波形证据，未试听；低能量为10ms RMS<0.001。
- 新异常ICL01：从18000的12帧0.96s变为2帧0.16s、EOS=true，目标0.98s；RMS3.51422e-5、低能量100%、最长0.16s，虽非严格全零，基本无有效语音证据。“字幕by索兰娅”不作内容证据。配对和生成策略未变化。对目标emilia2:1ac48948a9178d19_007_005及参考1ac48948a9178d19_001_003实施有界原始tar解码：目标23520采样/0.98s/RMS0.057914，参考68160采样/2.84s/RMS0.090068，24kHz输出均finite/nonzero。目标codec为(13,16) uint16，码值16–2028，SHA=4efdb89b60791573f1b0a86e9b69aa79bbe7bf4ffca551be04168d4c3354ebd1，与val清单一致。没有输入损坏或配对错误依据，未改最小帧数/训练目标；当前未保存逐步logits，根因仍未定位。
- ICL04仍2帧0.16s，自8500连续二十一轮；RMS0.015090、低能量75%、最长0.11s，高RMS不表示内容恢复。03为85帧6.8s，RMS0.035407、低能量21.62%、最长0.47s位于开头，连续十一轮无长低能量前缀；“尤其了半個小時先是下午兩點五時, 洗了大概10公里, 先稱屬於這電量是80%”、CER0.638889，正文仍退化。
- SO05为50帧4s/目标1.62s的2.47倍，RMS0.059723、低能量1.5%、最长0.03s，较18000的2.08s再次显著延长。ASR“G-D-B-A-F-L-N-T-S-E-O You've to put W-A-F-L-N-T-S-E-O-N-E”、WER5.75/CER2.076923，包含重复字母串及额外内容。该句贡献SO英语35个基础word errors中的23个，字母拆分也放大WER；不能直接把5.75理解为独立发音错误比例。既有原输入/codec/CPU诊断和本轮配对未给出新的实现故障依据，未重复同模型转写或将其冒充试听。
- SO01仍延长为29帧2.32s，“早期就只可能忘记 祝这奈白之后”、CER2，连续两轮超过目标时长两倍，低能量25%、最长0.29s；SO00为30帧2.4s，“Inus and non, the liquid spears.”、WER0.75，额外开头仍在。02为53帧4.24s，“That is a McLaughlin group in which there are winners and plusers.”、WER0.375，like和尾部未完整体现。04为25帧2s，“所以是人家畢竟現實了這麼多年”、CER0.615385；06为47帧3.76s，“Thus, in the Jeffreeze tube, they open a door and enter a sweat.”、WER0.25，末词再次错误。SO03为82帧6.56s，RMS0.038021、最长低能量0.42s；“特輯的半個時,先是下午2點50,先是大概10公里,先是順於今天是80%”、CER0.777778，较18000的0.638889高，时长正常不代表正文正确。
- ICL00为20帧1.6s，“The liquid spears.”、WER0.25；02为48帧3.84s，遗漏like、WER0.0625；05为18帧1.44s，“WF Flinders Uni.”、WER0.5，仍连续九轮无两帧早停，但字母错误再现；06为44帧3.5200417s，“In the Jeffries tube, they open a door and enter a swap.”、WER0.083333，末词连续五轮swap。
- 12目标5.28s：SO50帧4s，“王子你跟师父都看过我寄给你们那卷 为时为酒啊为酒”、CER0.517241，最长低能量0.03s；尾部两次“为酒”有重复音节线索，不能认定video发音正确。ICL45帧3.6s，“相信你跟师父都看过我寄给你们的捐威 videos”、CER0.379310，最长低能量0.06s，仍单次videos。尾缺词风险未解决，不能凭CER下降或无truncated标记称完整。
- 较18000，SO基础EN WER0.444444→0.972222、规范化0.416667→1，ZH CER0.726190→0.75；ICL EN WER0.083333→0.138889，ZH CER0.607143→0.642857。SO05/01持续严重额外内容及新ICL01早停需要最终评估继续跟踪。新增输入检查没有可验证修复依据，保持配置，不凭8句改变loss/LR；重点待19000检查01是否恢复、04/05/03与12，并完成最终流程。
- 收尾18950：token first/residual=1.269260090/5.908543332，sample=1.142324834/5.812112536，目标=2.885958594，grad0.447041，LR5.808381369e-5/1.742514411e-4，step2.105292s、wait0.0003049s；18500评估后450次真实更新/45日志点正常。latest仍18500，未到最终checkpoint/评估/正常退出，不执行冻结检查或写final-verification。本轮不等待19000。
- 实际执行cat/tail/sed、Python JSON/YAML/proc身份及allocator、checkpoint结构/签名/SHA/scheduler、512清单条数、全日志有限值/LR统计、nvidia-smi/free/df、soundfile/numpy逐句波形和配对/summary计数，以及ICL01原始tar目标/参考解码和codec SHA/形状/码值检查，均通过。唯一持久修改为追加本文并执行git diff --check；sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539和双模式保持。未改代码/配置/manual/metadata/数据/评估原件，未重启/发信号/恢复，未新增监控/Codex/定时器/subagent，未提交/push/PR、删除或外发。本次单轮结束。


### 2026-09-12T03:31:18.713468+00:00 — 19000训练及最终评估完成，执行冻结权重验收

- 已读playbook、本文最新记录、manual/process/退出状态、033032快照及030032 review/status。前轮18950，本快照和现场19000；manual=false，前次review exit0，同一外层1800秒session。training-process仍pid737573/start95204815，training-exit同PID、exit_code0、finished_at=2026-09-12T03:16:19.770700 UTC；原launcher737569、torchrun737573及rank737607–610均已不在/proc，未有PID复用混淆。launch.py的process.wait返回码与该原子退出记录一致，最终summary写入早于退出。没有重启/恢复或19000之后更新。
- latest19000 COMPLETE于03:04:58.470477 UTC完成；progress step/next_batch19000、epoch0/world4，scheduler last_epoch19000/_step_count19001，LR5.789593019e-5/1.736877906e-4仍按38539 schedule衰减，未为停止点更改计划。四distcp2093307384/2093524691/2093544414/2093533470B、distributed/.metadata1424894B、四rng各14613B齐全；metadata SHA=17231c2e27897e43922473c390ba23ce22ddc55b402baf1c8e9fef0e896779a9。19000/18500 signature一致，settings/model/eval/seed与当前配置匹配，run/source配置一致，val及assembly_report SHA匹配。结构/签名通过，未进行完整优化器恢复测试。
- 初始化及sqrt next-run-ack再次核对：同原assembled、sample从step0启动，ack中仅loss_reduction/output不同，当前训练PID与启动证据一致；保持sample、四卡6000/9000、accum1、workers16/prefetch2、原LR/残余0.3、max19000/schedule38539及双模式。
- 全1900训练日志点及38轮val数值有限，两组LR逐点通过原warmup/cosine核验。18960–19000最后50次更新的5个日志点：token first1.210581–1.346935、中位1.248919，residual5.893978–5.952945、中位5.940167；sample first1.091451–1.151352、中位1.114528，residual5.788862–5.810755、中位5.797986；grad0.414579–0.502773、中位0.445126。step本区间实测最小2.015689/中位2.118037/最大2.205273s，wait中位0.00014399s、最大0.00023916s，吞吐867.775–922.750、中位898.945音频秒/墙钟秒；samples336–398、中位366，frame填充94.8875–99.6708%、中位99.1667%，token92.7278–97.9583%、中位95.9278%，峰值显存53.6341–56.3747GiB。末步19000 token1.230437287/5.946152974、sample1.110419643/5.810754995、目标2.853646142、grad0.425549、step2.015689s、wait0.00014399s。
- 退出后四卡均1MiB/81920、利用率0%，compute-apps为空；RAM45GiB、可用1.9TiB、无swap，磁盘余578589.55GiB。训练日志无Traceback/OutOfMemory/Non-finite/ChildFailedError/terminate called/Aborted。最终日志有音频幅值警告（-1.00118/1.01031和-1.06501/1.17125），未定位到具体处理阶段；最终16生成WAV全部finite，abs>=0.999比例均0，未据该警告重跑或修改音频。
- 最终val：token first/residual=1.307292296/5.923950841，sample=1.170239903/5.815643378；四项和十五残余码本CE均较18500下降。val清单为512条且SHA匹配签名；检查validate及val_batches调用路径为四rank各16批、每批8条，实样本仅计一次。普通/逐码本CE仍token平均，sample逐句等权，最终sample验证目标=2.914932916；不能将CE下降当作生成质量通过。

|19000最终生成，各4EN/4ZH|EN基础WER/CER|EN规范化WER/CER|ZH WER/CER|截断/全零WAV/两帧EOS|
|---|---|---|---|---|
|speaker_only|0.333333/0.237762|0.361111/0.237762|1.125000/0.571429|0/8、0/8、0/8|
|icl|0.194444/0.090909|0.194444/0.090909|1.000000/0.702381|0/8、0/8、2/8|

- SO/ICL summary于03:10:50.165972/03:15:37.904471 UTC完成，各8条。16metrics/WAV可读、finite、24kHz单声道、时长正确；目标id/text/language、参考id/source、ICL reference_text/reference_frames及greedy min2与18500一致，分语言/英语规范化错误计数及截断计数吻合。以下为ASR/波形证据，未试听；低能量为10ms RMS<0.001。
- ICL01仍2帧0.16s、EOS=true，连续两轮；RMS2.98100e-5、低能量100%，未恢复。上一轮原目标/参考音频和目标codec检查正常，本轮配对未变。ICL04同样2帧0.16s，自8500连续二十二轮；RMS0.000749、低能量87.5%、最长0.07s。两句“字幕by索兰娅”均不作有效内容证据；最终无truncated标记不代表生成完整。
- ICL12从3.6s缩到36帧2.88s/目标5.28s的54.55%，ASR“相信你各式我都看我寄给你们的卷”、CER0.620690，未体现video；RMS0.059894、低能量3.125%、最长0.06s，尾部缺失风险仍在。SO12为53帧4.24s，“王智霓跟师傅都看过我寄给你们的权威士威迪欧威迪欧”、CER0.620690，尾部两次音译线索但正文错误且未独立试听，不能称正确恢复。
- SO01仍29帧2.32s/目标0.98s的2.37倍，“藏匙植奶白色後 咻咻就見見的好友”、CER2，连续三轮超过目标两倍，额外内容未解决；最长低能量0.22s。SO05从4s缩到29帧2.32s，“Chi Dabioi, Flint as Eunice.”、WER1.25/CER1.076923，仍严重偏离目标；SO00同为29帧2.32s，“I just send the liquid spires.”、WER1，额外开头仍在。SO02为55帧4.4s，ASR匹配、WER0；04为27帧2.16s，“海歇日壓必行寫冊了這麼多年”、CER0.769231；06为49帧3.92s，“At Cynda Jeffery's Tube, they open a door and enter a swamp.”、WER0.25，开头/专名仍错。
- SO03为83帧6.64s，RMS0.036393、低能量22.74%、最长0.49s，“就觉得半小时现在是下午两点五十 起了大概十公里 现在甚至是电量是百分之八十”、CER0.222222，较18500的0.777778改善，仍有正文错误。ICL03为90帧7.2s，RMS0.038340、低能量24.58%、最长0.53s位于[1.53,2.06]s，连续十二轮没有长低能量前缀；“又騎了半個小時,現在是下午2點50,騎了大概10公里,現在剩餘這地方是80%”、CER0.583333，不能据时长正常称全文准确。
- ICL00为17帧1.36s，“The liquid spears.”、WER0.25；02为44帧3.52s，“This is McLaughlin Group in which there are winners and losers, yeah, you know.”、WER0.125，缺a/like；05为19帧1.52s，“WF Lenders Uni”、WER0.75/CER0.153846，连续十轮无两帧早停但字母/专名错误增加；06为46帧3.68s，“In the Jeffreeze tube, they open a door and enter a swamp.”、WER0.083333，连续五轮swap后本轮末词恢复swamp，专名仍错。
- 较18500，SO基础EN WER0.972222→0.333333、规范化1→0.361111，ZH CER0.75→0.571429；ICL EN WER0.138889→0.194444，ZH CER0.642857→0.702381。最终流程产物齐全与生成质量问题并存；不凭8句调整超参或延长训练。保留ICL01/04早停、12尾缺词、SO01/00/05额外内容及正文错词为实验结果和后续诊断事项。无排队后续实验，不启动新run。
- 在上述条件核实后执行用户指定命令：PYTHONPATH=. .venv/bin/python scripts/check_frozen_frontend.py --assembled pretrained/assembled-qwen3-tts-frozen-conditioning --checkpoint runs/emilia-en-zh-dynamic-10000h-sample/checkpoints/step-00019000 --include-speaker；stdout/stderr保存本run final-frozen-check.log。本段写入时检查尚在运行，最终验收结果见后续追加。仅在该命令正常退出并明确passed后原子写final-verification.json，不能提前标记完成。
- 本轮检查命令为cat/tail/sed/rg、JSON/YAML/proc、checkpoint签名/SHA/scheduler、全日志有限值/LR、nvidia-smi/free、soundfile/numpy波形/配对/汇总及上述冻结检查。一次搜索误用了不存在的.venv/lib/python3.11/site-packages路径，rg返回2，属于只读巡检搜索失误；未影响训练或验收检查。未修改训练代码/配置/manual/metadata/数据/原评估，未发信号、重启或恢复，未提交/push/PR、删除或外发，未新增监控/Codex/定时器/subagent。

- 最终验收完成于2026-09-12T03:34:38.990153+00:00：冻结检查进程exit0，final-frozen-check.log明确status=passed，81个冻结张量、326313792参数（含speaker_encoder）与assembled逐元素精确相等；单进程DCP加载提示未初始化分布式属于脚本预期，不是失败。独立执行CodeDataset/val_batches覆盖验证，512个唯一实样本恰好各一次（4rank×16批×8）。上述最终checkpoint、验证、双模式各8条summary及同PID正常退出均实际通过，已用同目录临时文件flush/fsync后os.replace原子写final-verification.json，回读一致：passed=true、pid=737573、step=19000，含路径/哈希/具体证据及质量遗留问题。此次通过指流程和冻结权重验收，不表示生成质量合格。新增持久产物仅final-frozen-check.log、final-verification.json及本文记录；无后续排队实验，外层可据最终标记结束。本次单轮及sample训练验收结束。


## 2026-09-12T08:00:01.549769+00:00 — 环境停止后主会话检查，训练已按计划完成

用户发现机器停止，要求检查任务并准备resume。现场四卡空闲，无训练/监控/TensorBoard进程；当前PID1 tini启动于2026-09-12T07:54:32.460000+00:00，宿主kernel启动时间2026-08-31T06:51:34+00:00。这表明当前容器进程环境在训练结束后重新启动，平台为何重启无法从现有训练记录确定；不能归因为训练OOM或崩溃。

sample已于2026-09-12T03:16:19.770700+00:00正常exit0，03:34:38最终验收通过，早于本次环境启动。独立核实token/sqrt/sample均19000 COMPLETE、对应PID正常退出、final passed；sample最终metadata、两模式summary、冻结检查日志SHA与既有验收记录一致，16WAV均可读、24kHz、有限且时长一致。4个distributed分片/4份RNG、scheduler与数据游标元信息齐全；本次没有实际加载模型和优化器恢复，结构检查不冒充恢复测试。证据本run recovery-check-20260912/inspection.json。

当前max_steps=19000已达到，直接resume会加载后返回，不产生更新。无中断训练需要补跑；尚未启动训练或修改停止点、配置、既有退出/最终验收记录。若用户要延长sample实验，需要先指定新的停止步数，再按同一checkpoint接续并恢复监管。
