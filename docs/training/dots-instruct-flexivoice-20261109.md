# dots.tts 风格指令 TTS：实验记录（2026-11-09 起）

**目标**：在 `dots.tts-soar` 上训练"参考给音色、指令给风格"的 TTS。音色只来自参考音频（x-vector 或 ICL），说话风格（情感、语速、音量、耳语、语气）由自然语言指令决定。**不做 voice design**（不根据文字描述生成音色）。

设计取舍见 [设计文档](../design/dots-instruct-tts-20261108.md)；本页记录实际做法、全部实验、评测结果和分析。两者不一致时以本页为准。

## 0. 现状（2026-11-13）

| 项目 | 状态 |
| --- | --- |
| 当前数据 | **v5**：selection-flexivoice-v4（只保留以风格为主的子集，§1.4）+ NV 五个数据集 + HiFiTTS-2 约 500 h（§1.6）；dots_vae profile v3-ref3s（不做 250 ms 边缘归一化，修复 KeSpeech 内嵌 CAF 头） |
| 当前实验 | v5 · lr 5e-5 · 3 epoch：HiFiTTS-2 特征在 ACP 提取（`pt-snz9ubvd`），发布后自动 build、校验并提交训练 |
| 已完成训练 | step794（nocaf）、step932（raw v3）、lr 2e-5（v3，step 932）、v4（step 376） |
| 评测 | 已完成：step794、step932、release 噪声底的 StyleBench + 回归 suite；v4 的 NV-Bench、Seed-TTS。进行中 / 排队：v4 StyleBench、v4 CV3、lr 2e-5 的 StyleBench 与 suite（§5.3） |
| 主要问题 | 英文 speaker_only 音色相似度低于中文（英文说话人太少，§6.5）、韩语 / 日语发音退化、ICL 下没有风格控制、中文耳语几乎无效（参考全是耳语，§2）、NV 标签准确率下降（§6） |

## 1. 数据

### 1.1 FlexiVoice 接入

FlexiVoice-Instruct（[arXiv 2601.04656](https://arxiv.org/abs/2601.04656)）15 个子集接入为 `DATA-TTS-UNIFIED/datasets/flexivoice/v0.1`，4,199,042 行、约 9,752 h（接入说明见 tts-data-pipeline `docs/flexivoice.md`）。`metadata_json` 含 `instruction` 和上游记录 `upstream`（`wav_path`、`meta` 等）；base 的 `speaker_id` 和多数 `language` 为空。

文本修订 `tts-ann-flexivoice_text-20261110T091454bjt-01`：移除 `[inaudible]`，`[LAUGHTER]`/`[BREATHING]` 映射为 NV 标签，13 行含 `[GARBAGE]` 的转写记为不可用。

流程：text revision → selection（决定子集、时长、采样率、留出、配对）→ dots_vae features（只含入选行，携带 selected_text）→ training build（不再做业务过滤，只校验不变量）。所有发布登记在 tts-data-pipeline `docs/data-contract/PUBLICATIONS.md`。

### 1.2 Selection 版本

| 版本 | selection ID | 入选 | 变化 | 用途 |
| --- | --- | ---: | --- | --- |
| v1 | `…20261110T092455bjt-01` | 1,656,126 行 / 2,742 h | 11 个子集、1–30 s、≥16 kHz、66 个留出说话人键、闭合跨句参考池（≥3 s） | 11-10/11 的早期实验 |
| v2 | `…20261111T162703bjt-01` | 1,330,267 行 / 2,214 h | v1 + 按 103 排除 326,006 条内嵌 CAF 头的 KeSpeech | step794 |
| v3 | `…20261112T112338bjt-01` | 1,655,877 行 / 2,742 h | v1 规则，参考池按 raw 波形时长判断（不排除 CAF，提取时修复） | step932、lr 扫描 |
| **v4** | `…20261113T095517bjt-01` | **452,817 行 / 761 h** | 只保留 WHSP、debatts、ESD/RAVDESS/EMNS、原神、星穹铁道（§1.4） | v4、v5 |
| HiFiTTS-2 | `…20261113T143403bjt-01` | 171,761 行 / 500 h | selection-v2 规则 + 每说话人抽样名单（§1.6） | v5 |

留出说话人（四种语言一起留出，训练中完全不出现）：原神 8 个角色（早柚、莫娜、莱欧斯利、流浪者、珐露珊、可莉、克洛琳德、行秋）、星穹铁道 6 个（玲可、真理医生、银枝、帕姆、阮•梅、驭空）、EARS 7 人、RAVDESS 3 人。

### 1.3 开头咔声与 250 ms 边缘归一化

**现象**：48k step400 试听中"中文·长句·耳语"在约 0.25 s 有强瞬态（"咔"），NV 模型也有类似起音。

**根因 1：KeSpeech 内嵌 CAF 头。** 326,006 条 KeSpeech WAV 的 data 块以 4096 字节的 Apple CAF 文件头开头（`caff`/`desc`/`lpcm`/`free`/`data`），被解码成约 128 ms 的满幅垃圾采样，且是整条音频的峰值。全量字节扫描（tts-data-pipeline `scripts/audit_embedded_caf.py`）只在 kespeech 命中；其他子集和 NV 为 0。

**根因 2：250 ms 边缘静音归一化。** dots 训练预处理把首个超过（峰值 −30 dB）的采样前补零 / 裁剪到恰好 250 ms。CAF 垃圾或从第一个采样就开始发声的录音（FlexiVoice train 4.45%、NV 0.1–9.8%）因此在 0.25 s 处从数字零阶跃到满幅，并且所有起音都被对齐到 0.25 s。只去掉补零不能修复 CAF（咔声只会移到 0 s）。

**处理**：dots_vae waveform 策略 `raw-caf-repair-v1`（profile v3 / v3-ref3s）：解码前用 `wav_repair.py` 识别并跳过内嵌 CAF 头（核对 desc 为 16-bit LE PCM、采样率和声道与 WAV 一致，`start_frame` 记录跳过的帧数）；不做边缘归一化，只补齐到 7680 的整数倍。v3 FlexiVoice 特征中 325,819 条 CAF 行 `start_frame=2048`，所有行边缘偏移为 0。

### 1.4 v4：只保留以风格为主的子集

**指令类型统计**（关键词启发式，每子集抽 3,000 条；"两者都有"表示同一句里既有音色又有风格描述）：

| 子集 | 入选行（v3） | 只音色 | 只风格 | 两者都有 | 指令示例 |
| --- | ---: | ---: | ---: | ---: | --- |
| kespeech | 975,483 | 1% | 0% | 99% | "慢速语速。请使用中年女性的声音，带有冀鲁官话的发音特点。" |
| childmandarin | 39,861 | 99% | 0% | 1% | "用小男孩的声音说出这句话" |
| l2 | 26,826 | 55% | 1% | 43% | "我需要一位带西班牙口音的女声，语气活泼一点" |
| voxceleb | 99,163 | 5% | 0% | 95% | "A husky male… deep booming voice in a noisy environment…" |
| ears / expresso | 53,616 | ≈0 | 0% | ≈100% | ParaSpeechCaps 风格长描述（性别、口音、音高 + 风格） |
| emotion | 45,552 | 0% | 97% | 0% | "Speak it using the emotion of Sarcastic." |
| debatts | 19,732 | 0% | 55% | 0% | "请以辩论中的坚定语气朗读以下内容。" |
| WHSP | 10,949 | 15% | 72% | 0% | "把这句话用气声说出来" |
| 原神 / 星穹铁道 | 384,695 | 2–3% | 77–80% | 12–15% | "直白又理直气壮的讨要语气，带着点玩世不恭的痞气。" |

FlexiVoice 论文说明：KeSpeech 的指令由年龄 / 性别 / 方言三元组让 DeepSeek-V3 写成；ChildMandarin、L2-ARCTIC 按说话人标签写模板；EARS、Expresso、VoxCeleb 的描述来自 ParaSpeechCaps（其标签本身分为说话人固有的 intrinsic 和随句变化的 situational 两类）。即 v3 中约 **69% 的 FlexiVoice 行在用指令描述音色**，与"音色来自参考"的目标冲突。

**v4 决定**：删 kespeech、childmandarin、l2、voxceleb（音色描述为主），ears、expresso（音色混在长描述里），emotion 中的 CREMA-D、IEMOCAP（低质量，见 §6.3）；保留 WHSP、debatts、ESD、RAVDESS、EMNS、原神、星穹铁道。selection 新增 `flexivoice.emotion_corpora` 白名单（不在名单的 emotion 语料记 3004）。

| v4 子集 | 行数 | 时长 |
| --- | ---: | ---: |
| 原神 | 289,242 | 467 h |
| 星穹铁道 | 95,453 | 182 h |
| debatts | 19,732 | 62 h |
| ESD | 35,000 | 29 h |
| WHSP | 10,949 | 18 h |
| EMNS + RAVDESS | 2,441 | 3 h |
| **FlexiVoice 小计** | **452,817** | **761 h** |
| NV 五个数据集 | 230,083 | 731 h |

注意：FlexiVoice 部分 85% 是游戏（中 145 h、英 167 h、日 193 h、韩 144 h）；NV 占总时长约一半。CSEMOTIONS 只作测试，不进训练。

### 1.5 特征版本

| profile | FlexiVoice | NV | 说明 |
| --- | --- | --- | --- |
| v2-ref3s（250 ms 边缘归一化） | 全量 `…20261109T160744`；v1 `…20261110T093409`；v2 `…20261111T164240` | `…20261106T092453` | 早期实验 |
| v3-ref3s（raw-caf-repair-v1） | v3 `…20261112T121538`（本机 GPU 0–4 重新提取）；v4 `…20261113T100902`（由 v3 复用） | `…20261112T121520` | profile_id `a6d56e2c…`，build 要求所有 binding 同一 profile；HiFiTTS-2 `…20261113T144202`（ACP 提取中）同一 profile |

### 1.6 v5：补英文说话人（HiFiTTS-2）

v4 训练集按语言统计（无语言标签的行按文字判断）：

| 语言 | 时长 | 说话人 | 主要来源 |
| --- | ---: | ---: | --- |
| 中文 | 955 h | 4,385+ | emilia_nv 570 h、nonverbalspeech38k 115 h、原神 105 h、debatts 62 h |
| 英文 | 202 h | 170 | 原神 122 h、星穹铁道 47 h（配音演员）、情感语料 16 h |
| 日文 / 韩文 | 195 / 142 h | 133 / 128 | 全部是游戏 |

中文有大量自然、多说话人的数据，英文只剩 170 个、多为游戏配音的说话人——v4 删除的 VoxCeleb、EARS、L2、Expresso 正是英文的多说话人来源。

**v5 补 HiFiTTS-2**（有声书朗读，4,970 个说话人）：`scripts/sample_per_speaker.py` 生成固定抽样名单——每个 base speaker_id 在 3–20 s 的句子里按 `sha256('hifitts2-speaker40-v1:' + sample_id)` 取前 40 句（173,651 行 / 506 h）；selection `selection-hifitts2-speaker40-v1`（selection-v2 规则，只含 hifitts2，base 文本）通过 `sample_evidence` 固定名单哈希，名单外的行记新原因码 **8002 not_sampled**（原因码字典 version 9）。入选 171,761 行 / 500 h（差额为 v2 规则排除的标记文本、上游测试集等）。训练中用默认指令、句内 ref3s 参考（同 NV）。注意：HiFiTTS-2 是 22.05 kHz（有效频宽约 11 kHz），需在评测中关注英文输出音质。

### 1.7 游戏文本核查

FlexiVoice 的原神 / 星穹铁道文本是官方台本，格式正常（没有 `{NICKNAME}`、`<color>` 等占位或标签；连续省略号 2.6%、`Option N:` 221 行、`#76` 编号 46 行）。按"语言 + 游戏内文件名"与已有的 genshin_voice / starrail_voice 对齐 295,534 行（77%），与其 ASR 转写 `tts-ann-transcription-20261010T111659bjt-01` 比较：

| | CER 中位 | 差异 >10% | 差异 >30% |
| --- | ---: | ---: | ---: |
| 英 / 中 / 韩 | 0.00–0.02 | 6–16% | 1–3% |
| 日 | 0.07 | 36–37% | 4–6% |

差异多为 ASR 的错误（专有名词误识，如"南十字之笔"→"难识字之笔"；丢掉笑声、语气词；日文写法差异"ホントか"/"本当か"），官方文本更准，**不替换为 ASR**。少数真正不对应的是舞台说明（如英文 `*panting*`），可在后续 selection 中按差异 >30% 或星号舞台说明排除。

## 2. 输入格式与参考

**序列**：指令前缀在正文前，不参与交错。

```text
interleave：[流式语音合成]Instruction: {instruction}\nText:\n 正文 token / 音频 span 交错 …
basic：     [文本]Instruction: {instruction}\nText:\n{正文}[文本对应语音]<start> S*N <end><eos>
```

- 没有指令（NV 数据）用默认指令 `Speak the following text.`。前缀以换行结尾，保证训练（分开分词）和推理（整串分词）token 一致。
- interleave 与 basic 按 target_id 的哈希固定选择（`interleave_fraction`）；长文本遵循 release 的交错循环，音频先结束时仍补入 `<|text_cond_end|>`。
- 推理时以训练同样的模板注册到 runtime（UltraEval-Audio `scripts/local_dots.py` / `style_bench_infer.py`），每条 schedule 与训练 `Layout.sequence` 逐 token 核对过。

**参考**：

| 数据 | 条件 x-vector |
| --- | --- |
| FlexiVoice | 同一说话人**另一句**（≥3 s）的整句 x-vector（CAM++，取前 10 s）；有风格标签的目标按 0.4 中性 / 0.4 冲突风格 / 0.2 任意抽取，每次访问重新抽；目标整句计 loss |
| NV | 同句内 ref3s 参考段（长度 ≥ min(3.04 s, 半句)）的 x-vector，参考区间 mask |

**中性参考的覆盖**：只有带风格标签的情感语料（v4 中 3.7 万行，8%）按上表抽中性 / 冲突参考；原神、星穹铁道、debatts、WHSP（41.5 万行，92%）没有标签，参考是同一说话人随机另一句，常带情绪。**WHSP 10,949 行全是耳语**（`whsp.wav`），参考也全是耳语，模型从没见过"中性参考 + 耳语指令"，与 StyleBench 中文耳语 8–18% 一致；英文耳语来自 EARS（有中性参考），v4 删掉 EARS 后英文耳语可能变弱。

说话人键从上游路径解析（`lm_tts/data/flexivoice.py: speaker_key`），均为上游官方 ID（KeSpeech、VoxCeleb、ESD、CREMA-D、RAVDESS、EARS、L2、childmandarin、WHSP 的说话人目录或编号；IEMOCAP 为 session + 每句实际说话人 M/F；Expresso 对话按声道；游戏为"语言 × 角色"，每种语言一个配音演员；debatts 为场次 + 自动分离编号，同一人跨场次会被拆开，只缩小参考池、不会配错人）。

## 3. 训练配置

共同设置（`configs/dots-tts-2026111*-flexivoice-*`）：

| 项目 | 设置 |
| --- | --- |
| 起点 | `dots.tts-soar`，nv26 组装（`assets/assembled/dots-tts-soar-nv26`） |
| 冻结 | DiT 及其条件投影：`velocity_field_predictor, hidden_proj, latent_proj, coordinate_proj, xvec_proj` |
| 训练 | LLM、PatchEncoder、EOS head |
| DiT 执行 | packed（按句拼接 + FlexAttention block mask），PatchEncoder / LLM / DiT 全部 activation checkpointing |
| 采样 | `feistel_uniform_token_budget_v3`（全局打乱）；每卡 48,000 tokens / 30,000 patches，accumulation 1 |
| 调度 | WSD：warmup 2%，最后 20% 降到 0.1 倍；save / eval 每 200 步 |
| 指令 dropout | 0；x-vector dropout 0；模型 CFG dropout 0.2 |
| 资源 | ACP 单节点 8×A800，bf16 |

**packed DiT**：原 DiT 按批内最长句补齐并用稠密 mask，长句批次显存会成倍增长；packed 版按真实长度拼接，输出和梯度与原实现一致（CPU / A800 / 双卡 FSDP 均验证）。代码 `lm_tts/models/dots/packed.py`，测试 `tests/test_dots_packed.py`。

**早期采样问题**：首个实验未指定 sampler，用了按 65,536 行分块打乱，step 9000 时 WHSP、EARS、emotion、Expresso 等子集一条都没训到；此后统一用 Feistel 全局打乱。

## 4. 实验一览

| 实验 | ACP | 数据 | 变化 | 状态 / 结果 |
| --- | --- | --- | --- | --- |
| 基线（instr dropout 0.1） | `pt-c42buvr4` | v1 | lr 1e-5，interleave 0.5，分块打乱 | 停止保留；step 9000 试听风格差异弱（采样问题） |
| 32k packed DiT | `pt-ng26vb3i` | v1 | lr 1e-5，1 epoch | 停止保留（step 550）；step 525 试听 |
| 48k lr 5e-5 | `pt-jt0v0he9` | v1 | lr 5e-5，1 epoch | step 400 扩展试听，发现开头咔声 |
| **nocaf**（step794） | `pt-ii8bqdpg` | v2 | 100% interleave，lr 5e-5，2 epoch | 完成，评测见 §5 |
| nocaf 50% interleave | `pt-0xu9i6xh` | v2 | interleave 0.5 | 数据重做后停止删除 |
| **raw v3**（step932） | `pt-x7s1wo9c` | v3 | 50% interleave，lr 5e-5，2 epoch | 完成，评测见 §5 |
| lr 扫描 2e-5 | `pt-fg1ai7mx` | v3 | 只改 lr | 完成（step 932）；验证 loss 与 lr 5e-5 几乎相同（§6.2）；评测排队中 |
| lr 扫描 1e-5 | `pt-p6j3rydu` | v3 | 只改 lr | v4 提交后停止删除（训练到约 step 690），让出 GPU |
| **v4** | `pt-r8wymqby` | v4 | 同 step932，换 v4 数据 | 完成（step 376）；build `dots-flexivoice-v4-selection-nv-ref3s-20261113`；评测见 §6 |
| **v5** | 待提交 | v4 + HiFiTTS-2 | 同 v4，加 500 h 英文多说话人，3 epoch | HiFiTTS-2 特征提取中（`pt-snz9ubvd`），发布后自动 build 并提交 |

run 目录：`/workspace/LM-TTS-Training-Runs/<run_name>/`；评测结果：`UltraEval-Audio/res/<run>/`。

## 5. 评测

### 5.1 StyleBench（UltraEval-Audio）

FlexiVoice 的测试集和测试指令没有公开，按论文的 TR-easy / TR-hard 思路用**训练中没出现过的数据**自己搭（`scripts/prepare_style_bench.py`，数据 `raw_data/style_bench_v1/`，3,187 个 wav）。全部子集分 speaker_only 和 ICL 各跑一遍，`speaker_scale=1.0`（与训练一致；runtime 默认 1.5）。

| 子集 | 数据 | 构造 | 指标 |
| --- | --- | --- | --- |
| 情感 emotion_en | 留出的 EARS 7 人 + RAVDESS 3 人，6 类情感，480 条 | 正文为说话人的中性句；参考为同说话人另一句：easy 中性、hard 冲突情感 | ACC-I（分类器结果 = 指令情感）、ACC-R（= hard 参考情感）、P-I、SIM、WER |
| 情感 emotion_zh | CSEMOTIONS 10 人，4 类（fearful 去掉：真实录音只认出 4%），480 条 | 同上 | 同上 |
| 说话方式 paraling_en / zh | EARS / CSEMOTIONS 中性句，耳语、大声、快、慢、高、低，492 条 | 同参考同正文，分别生成风格指令版和默认指令版 | 相对默认版：响度 +3 dB、浊音比例 −0.2、语速 ×1.15 / ×0.87、F0 ×1.10 / ×0.91 视为成功 |
| 自由描述 free_zh/en/ja/ko | 留出游戏角色，每语言 200 条 | 用该行真实指令和正文，真实录音作锚点，参考为同角色另一句 | 指令版 vs 默认版：emotion2vec embedding 与真实录音的相似度（胜率）、F0 / 语速 / 响度距离 |

- 情感分类器 emotion2vec_plus_large，只在测试情感 + neutral 里取最大值。**真实录音上限：英文 76%、中文 51%**，中文准确率要和上限一起看。
- CSEMOTIONS 与训练集零重叠（音频哈希、归一化文本都核对过）；同说话人 CAM++ / WavLM 相似度：中性-中性 0.79 / 0.80，中性-情感 0.65–0.75，不同人 0.30 / 0.34，标注可靠。
- 局限（独立审查后确认）：默认指令训练时只出现在 NV 数据上，"相对默认指令"的对照带有 NV 风格；EARS、RAVDESS 的句子在其他说话人的训练数据里出现过，英文 WER 偏乐观；ICL + 指令训练时没见过；每格 40–80 条，10 个点以内的差异不可靠；release `--plain` 只是噪声底，不是指令基线。
- 运行：`scripts/run_style_bench.py --model-path init_model/<export> --run-dir res/<run> --gpus 0,1,2,3,4 --bench-dir raw_data/style_bench_v1 --speaker-scale 1.0`（`--plain` 跑 release 噪声底）。

- 自由描述的局限：真实录音只是一种演绎；emotion2vec 只抓情绪，"困倦""痞气"之类不一定体现；参考是同角色另一句，本身带语气，会压小指令版和默认版的差别。胜率在 50% 附近不能证明模型完全没听指令，更可靠的做法是音频大模型裁判（类似 InstructTTSEval）或人工试听。

### 5.2 回归 suite

`scripts/run_dots_suite.py --instruction "Speak the following text."`：NV-Bench（ICL）、Seed-TTS、CV3（speaker_only 与 ICL），模板为 interleave + 默认指令，`speaker_scale` 用 runtime 默认 1.5（与以往 NV 实验一致）。CV3 ICL 有长样本超过每进程 8 GiB 显存上限，现用 `--gpu-memory-gib 16`（每卡仍 4 个进程，保持分片一致以便续跑）。

### 5.3 运行方式

本机 GPU 0–4 常被其他用户的 vLLM 服务占用（推理进程等显存超时会退出），评测改为提交到 ACP（UltraEval-Audio `scripts/acp/submit.py`，StyleBench 与 suite 各一个任务、各一节点）；各评测都支持断点续跑，中途删除重提不丢已完成部分。

## 6. 结果与分析

### 6.1 StyleBench（speaker_only，v1 数据，scale 1.0）

| | release（无指令，噪声底） | step794 | step932 |
| --- | ---: | ---: | ---: |
| 英文情感 ACC-I easy / hard | 6.2 / 8.8 | 46.0 / 45.6 | **53.1 / 53.8** |
| 英文 ACC-R（hard，↓） | 26.0 | 14.8 | **10.2** |
| 中文情感 ACC-I easy / hard | 0.8 / 6.7 | 56.9 / 57.5 | 57.5 / 55.0 |
| 中文 ACC-R（↓） | 29.4 | **21.7** | 23.1 |
| 情感组 SIM 英 / 中 | **0.777 / 0.780** | 0.580 / 0.623 | 0.563 / 0.622 |
| 英文 耳语 / 大声 / 快 / 慢 / 高 / 低（成功率 %） | 0 / 17 / 10 / 7 / 2 / 12 | 93 / 41 / 48 / 86 / 31 / 64 | 81 / 14 / 55 / 79 / 52 / 55 |
| 中文 耳语 / 大声 / 快 / 慢 / 高 / 低 | 0 / 28 / 3 / 3 / 8 / 8 | 18 / 30 / 23 / 43 / 73 / 75 | 8 / 43 / 45 / 58 / 80 / 63 |
| 自由描述胜率 中 / 英 / 日 / 韩 | 48.5 / 50.0 / 54.5 / 46.0 | 50.5 / 52.5 / 48.5 / 53.0 | 58.5 / 49.5 / 54.5 / 52.0 |
| 自由描述 CER 中 / 英 / 日 / 韩 | 4.1 / 2.5 / 7.6 / **8.3** | 4.6 / 2.7 / 15.8 / **51.0** | 6.7 / 6.5 / 11.5 / **62.5** |

- 风格控制学到了：情感、说话方式都远高于噪声底，参考情感泄漏下降。
- **ICL 模式下没有风格控制**：两个模型 ACC-I 只有 3–15%，与 release 相同。
- 自由描述的指令几乎没有作用（胜率在 50% 附近）。
- 英文"大声"在 step932 失效：生成响度反而低于默认指令；训练数据里 EARS 大声比中性高 6.7 dB，问题不在数据本身。
- 中文耳语两个模型都很弱。

### 6.2 回归 suite

| | NV-ref3s（lr 1e-5，无指令训练） | step794 | step932 | **v4** |
| --- | --- | --- | --- | --- |
| Seed-TTS en speaker_only WER / SIM | 2.44 / 69.4 | 2.95 / 62.7 | 7.53 / 52.0 | 6.31 / 58.7 |
| Seed-TTS zh speaker_only | 1.17 / 73.4 | 1.57 / 73.6 | 1.52 / 73.0 | 1.58 / 73.0 |
| Seed-TTS en / zh ICL WER | 2.09 / 1.22 | 2.40 / 1.41 | 3.16 / 1.25 | **2.19** / 1.46 |
| Seed-TTS en / zh ICL SIM | 74.0 / 77.6 | 74.5 / 78.1 | 72.1 / 78.2 | 72.1 / 78.0 |
| CV3 en speaker_only WER / SIM | 4.88 / 72.6 | 5.39 / 67.9 | 8.22 / 62.4 | 排队中 |
| CV3 zh speaker_only | 4.03 / 76.1 | 3.72 / 76.0 | 4.43 / 75.6 | 排队中 |
| NV-Bench CER / PCER zh_single | 2.74 / 22.6 | 3.24 / 21.4 | 5.02 / 28.5 | 3.76 / 30.9 |
| NV-Bench CER / PCER zh_multi | 3.01 / 33.9 | 3.81 / 35.5 | 3.69 / 39.3 | 4.09 / 43.7 |
| NV-Bench CER / PCER en_multi | 6.86 / 39.4 | 10.76 / 48.2 | 15.38 / 55.8 | 14.96 / 54.0 |

- v4 删掉"音色描述"数据后，英文 speaker_only SIM 回升 6.7（52.0 → 58.7），ICL 英文 WER 恢复到 2.19；但仍低于 step794，中文基本不变。
- NV 标签准确率（PCER）没有恢复，中文反而更差：可能与 50% interleave 或风格数据与 NV 数据相互干扰有关。

**EOS（验证 eos_loss）**：

| | step 200 | 最终 |
| --- | ---: | ---: |
| step794（v2 数据，100% interleave） | 0.070 | 0.056 |
| step932（v3，lr 5e-5） | 0.193 | 0.172 |
| lr 2e-5（v3） | 0.201 | 0.174 |
| **v4**（v4 数据，lr 5e-5） | **0.129** | **0.106** |

学习率不影响 EOS，**换成 v4 数据后降了约 40%**，说明 EOS 偏高主要来自被删掉的子集（KeSpeech、VoxCeleb 等）；剩下的差距可能来自去掉 250 ms 静音或 50% interleave。验证集随数据变化，只能参考。StyleBench 英文里错误率超过 20% 的条目 step794 有 3 条、step932 有 26 条（提前截断 17、该停没停 15）；中文 0 → 9 条。

### 6.3 speaker_only 音色相似度下降的原因

1. **学习率**：lr 1e-5 的 NV-ref3s 英文 SIM 69.4；lr 5e-5 的 step794 62.7、step932 52.0。lr 扫描在跑。
2. **指令在描述音色**：v3 中约 69% 的 FlexiVoice 行指令描述性别、年龄、口音、嗓音（§1.4），教模型"音色听指令"，削弱对 x-vector 的依赖；step932 加回 KeSpeech 后英文掉得更多。v4 已删掉这些子集。
3. **训练用 x-vector 与目标句对不上**：同说话人另一句的 CAM++ 余弦，按子集中位：debatts 0.84、l2 0.74、ears 0.73、WHSP 0.70、starrail 0.69、genshin 0.68、kespeech 0.67、voxceleb 0.62、childmandarin 0.59、ESD 0.56、expresso 0.47、IEMOCAP 0.43、CREMA-D 0.40；不同说话人 0.13（>0.5 仅 0.4%）。NV 的句内 ref3s 参考与整句约 0.86。
   - 与 WavLM 在同 240 对上相关 0.91，**不是 CAM++ 的问题**，是音频本身差别大。
   - 主要因素是句长（较短一句 <3 s 时 0.35–0.55，两句都 ≥6 s 时 0.65–0.86）和风格（同说话人换风格低 0.1–0.15）。说话人 embedding 编码情感、情感变化会拉低验证效果，有文献支持（Pappagari et al., ICASSP 2020，"x-vectors meet emotions"）。
   - 改用中性参考几乎没有改善（多数子集 +0~0.06），因为偏差主要来自情感化的**目标句**本身。
   - ICL 下 SIM 基本不变，与"模型少依赖 x-vector"一致。
4. **v4 剩余数据**：每句与本说话人 x-vector 中心（留一）的相似度中位 0.82；低于 0.4 的 0.8%（约 3,600 行），低于 0.5 的 1.9%，集中在游戏数据。是否按阈值过滤，需先试听低分样本再定（尚未筛选）。

### 6.4 日语 / 韩语

韩语 CER 从 release 的 8.3 升到 51–63，ICL 下约 25% 的输出几乎为空（模型立即输出 EOS）；日语 7.6 → 11–16。日韩在训练里只有游戏数据。v4 保留了游戏的日韩部分，需要重点看。

### 6.5 中文正常、英文偏低的原因

speaker_only 只靠 x-vector 还原陌生说话人的音色，需要训练中见过足够多自然的说话人。v4 中文有约 955 h、4,000+ 说话人（emilia_nv、debatts 等自然语音），英文只有 202 h、170 个说话人且 84% 是游戏配音（§1.6）。ICL 下英文 WER 已恢复，说明发音本身没坏，坏的是"只用 x-vector 还原陌生英文音色"。v5 补入 HiFiTTS-2（4,970 个英文朗读者、默认指令、句内参考）验证这一点。

## 7. 下一步

1. v5（+HiFiTTS-2，3 epoch）训练后评测，看英文 speaker_only SIM / WER 能否回到 step794 或 NV-ref3s 水平。
2. 完成 v4 的 StyleBench / CV3 和 lr 2e-5 的评测，确认学习率对英文 SIM 的影响。
3. 中文耳语：WHSP 只有耳语，可查原始语料有无同说话人正常发音作参考；游戏、debatts 的参考可按 emotion2vec 中性概率优先选。
4. NV 标签退化：区分 50% interleave 与数据配比的影响。
5. 游戏数据：排除文本与音频差异 >30% 的行及舞台说明；低音色相似度样本先试听再定是否过滤。
6. 日韩发音、ICL 下的风格控制，视 v5 结果再定。

## 8. 复现

```bash
# selection（tts-data-pipeline）
python scripts/build_selection.py plan --work /tmp/selection-flexivoice-v4 --rules configs/selections/selection-flexivoice-v4.json
python scripts/build_selection.py run --work /tmp/selection-flexivoice-v4 --rules configs/selections/selection-flexivoice-v4.json --workers 16
# 特征：从 v3 复用（不需要 GPU）
PYTHONPATH=src python -m tts_data_pipeline.dots_vae.reselect --root /workspace/data/DATA-TTS-UNIFIED \
  --selection <v4 selection manifest> --features <v3 FlexiVoice features manifest> --work <new work dir>
# build + 提交（LM-TTS-Training）
PYTHONPATH=$PWD python scripts/build_unified.py --config configs/dots-tts-20261113-flexivoice-v4-interleave50-feistel-48k-packeddit-lr5e-5-ep2.yaml
source scripts/acp/env.sh && PYTHONPATH=$PWD python -m scripts.acp.submit --config <config> --submit
# 导出
PYTHONPATH=$PWD:<dots.tts src> python scripts/export_checkpoint.py --checkpoint <run>/checkpoints/<step> \
  --assembled-model assets/assembled/dots-tts-soar-nv26 --output <UltraEval-Audio>/init_model/<name> --precision bf16
```

| 内容 | 位置 |
| --- | --- |
| 统一数据与发布登记 | `/workspace/data/DATA-TTS-UNIFIED`；tts-data-pipeline `docs/data-contract/PUBLICATIONS.md` |
| selection 规则 | tts-data-pipeline `configs/selections/selection-flexivoice-v{1,2,3,4}.json`、`selection-hifitts2-speaker40-v1.json` |
| 数据配置 | `configs/data-flexivoice-{nocaf,raw,v4}-nv-dots-ref3s.yaml`、`data-flexivoice-v5-hifitts2-nv-dots-ref3s.yaml` |
| 评测代码 | UltraEval-Audio `scripts/{prepare_style_bench,style_bench_infer,style_bench_score,style_bench_summary,run_style_bench,run_dots_suite}.py` |
| 设计文档 / 副语言实验 | `docs/design/dots-instruct-tts-20261108.md`；`docs/training/dots-tts-emilia-nv-20261103.md` |
