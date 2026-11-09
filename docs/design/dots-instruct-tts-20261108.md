# dots.tts 风格指令 TTS：FlexiVoice 数据微调（2026-11-08）

> 执行记录与当前进度见 [dots-instruct-flexivoice-20261109](../training/dots-instruct-flexivoice-20261109.md)；两者不一致时以执行记录为准。

目标：在 `dots.tts-soar` 上训练"参考音色 + 风格指令"的 TTS。音色只来自参考音频，说话风格（情感、语速、音量、耳语、语气等）由自然语言指令决定。不做 voice design，也就是不支持"不给参考、靠文字描述生成音色"。

做法：照 FlexiVoice（[arXiv 2601.04656](https://arxiv.org/abs/2601.04656)，ICLR 2026）的第一阶段，用 FlexiVoice-Instruct 数据做监督微调。和原文的区别：

- **只做微调**：不做 S1 DPO、S2 解耦 GRPO、S3 指令 GRPO。dots 是连续 latent 加 flow matching，这些 RL 阶段不能直接照搬，等微调结果出来再决定。
- **第一版不用 Emilia**：既不混入普通 Emilia，也不用 FlexiVoice 里的三个 emilia 指令子集（§1）。副语言部分整个复用 NV 第六轮已经处理好的 ref3s build（5 个数据集，§1）。
- **参考以同一说话人另一句为主**（§3）；复用的副语言数据沿用第六轮的句内参考加 mask。原文的预训练不给参考，所以它的模型能按描述生成音色；我们不要这个能力。

## 1. 数据

`DATA-TTS-UNIFIED/datasets/flexivoice/v0.1`（接入见 tts-data-pipeline `docs/flexivoice.md`，2026-11-08 发布完成），共 15 个子集、4,199,042 行。**第一版约 192 万行 / 约 3,580 h**：FlexiVoice 中能配到同一说话人另一句的行（约 169 万行 / 2,850 h），加上整个复用的 NV 第六轮 ref3s build `dots-nv-canonical-v1-ref3s-20261106`（230,079 行 / 731 h：emilia_nv 571 h、nonverbalspeech38k 119 h、smiip_nv 28 h、disfluencyspeech 9 h、nonverbaltts 4 h，用句内参考加 mask）。FlexiVoice 的 nvspeech 子集就是 emilia_nv，不另外处理。不用的有：emilia 三个子集（说话人要靠完整 Emilia 配对，先不做）、emotion 里的 MSP-PODCAST 和 MELD（路径里没有说话人），以及其他子集里配不到参考的行：

| 子集 | 行数 | 时长（按行累加） | 采样率 | 说话人能否从路径确定 | instruction |
| --- | --- | --- | --- | --- | --- |
| ~~emilia_psc~~（第一版不用） | 932,908 | 2,451 h | 24 / 32k mp3 | 否 | 英文，30% key-value，其余自然语言；描述音色和环境为主 |
| ~~emilia_en_ds~~（第一版不用） | 609,457 | 1,907 h | 24 / 32k mp3 | 否 | 英文场景、语气描述 |
| ~~emilia_zh_ds~~（第一版不用） | 626,457 | 1,712 h | 24–44.1k mp3 | 否 | 中文场景、语气描述 |
| kespeech | 976,054 | 1,542 h | 16k | 是（27,237 人） | 中英模板：年龄、性别、语速、环境 |
| ~~nvspeech~~（= emilia_nv，改为复用 NV ref3s build） | 265,888 | 776 h | 混合 | 否 | 空 |
| genshin | 309,657 | 502 h | 48k | 是（语种/角色，352 组） | 中文自由描述，中英日韩四种语音共用 |
| starrail | 101,794 | 195 h | 44.1k | 是（252 组） | 同上 |
| voxceleb | 100,255 | 222 h | 44.1k（voicefixer 修复过） | 是（530 人） | 英文，音色为主，81% 标为带噪 |
| emotion | 119,648 | 117 h | 16k 为主；RAVDESS、EMNS 为 48k | ESD、CREMA-D、RAVDESS、EMNS 可以（约 4.5 万行）；IEMOCAP、m3ed 待确认；MSP-PODCAST、MELD 不行，不用 | 中英情感模板 |
| ears | 28,776 | 119 h（去重音频 60 h） | 48k | 是（107 人） | 英文，每条音频配 key-value、自然语言两种 |
| expresso | 27,680 | 55 h（去重音频 27 h） | 48k | 是（4 人） | 同上 |
| debatts | 21,740 | 67 h | 16k | 是（场次 + 说话人，中位 3 句） | 中英辩论语气模板 |
| childmandarin | 40,912 | 41 h | 16k | 是（397 人） | 中文童声模板 |
| l2 | 26,867 | 27 h | 44.1k | 是（24 人） | 中英口音描述 |
| WHSP | 10,949 | 18 h | 16k | 是（146 人） | 中英耳语模板 |

预处理（tts-data-pipeline）：

1. **接入**：已于 2026-11-08 发布（4,199,042 行，manifest complete）；adapter 和文档还没提交。
2. **说话人**：不重新发布 base。提特征 / build 时从 `metadata.upstream.wav_path` 按上表解析说话人键（形如 `flexivoice:genshin:ZH:派蒙`，写法参照 `_game_voice.py`、CSEMOTIONS adapter 的 `speaker_id`）；路径里没有说话人的子集留空。风格标签同样从 `metadata.upstream` 的 `wav_path` 和 `meta` 直接读。
3. **副语言：不处理，整个复用 NV 第六轮 ref3s build。** FlexiVoice 的 nvspeech 就是 NVSpeech：18 种标签与 emilia_nv 相同（写法是 `<|X|>`），174,179 条带标签的行与 emilia_nv 全部重合，所以不处理这个子集。新 build 直接复用 `dots-nv-canonical-v1-ref3s-20261106` 的全部 230,079 行（5 个数据集，`[X]` 文本、ref3s 特征和句内参考区间都已就绪）。
4. **selection `selection-flexivoice-v1`**：不收 emilia 三个子集和 MSP-PODCAST、MELD；与已发布的 `genshin_voice`、`starrail_voice` 跨数据集去重；采样率 8k、12k 的 35 条剔除；时长 1–30 s；去重时，同一音频配不同 instruction 的行保留（ears、expresso）；留出评测说话人（§5.2）。
5. **dots_vae 特征**：在 selection 分支上跑，沿用 `dots-tts-soar-vae-xvec-48k-fp32-v2-ref3s` profile。跨句参考用整句 `xvector`，nvspeech 用 ref3s 句内参考段。配不到参考的行在 build 阶段丢掉。

## 2. 输入格式

照 FlexiVoice（论文表 1、§3.1），外层只保留模型原有的文本和音频边界：

```text
[文本]Instruction: {instruction}
Text: {text}[文本对应语音]<start> S*N <end><eos>
```

- `{instruction}` 是 `metadata.instruction` 原文，不改写；可以是中文、英文、key-value 或自然语言。
- 复用的副语言数据没有 instruction，填默认指令 `Speak the following text.`；文本里的 `[X]` 标签照旧（nv26 组装）。
- **指令 dropout 10%**：其余样本把指令换成默认指令，参考不变，保住普通 TTS 的用法。
- 只用 basic 布局，不用 interleave。

## 3. 参考

**以跨句参考为主。** 每条样本的参考是同一说话人另一句（不同文本、时长 ≥ 3 s）的整句 x-vector，送进 DiT，每个 epoch 重新抽；目标句整句计 loss，没有 mask。

**复用的副语言数据例外**：没有说话人信息，沿用第六轮已经处理好的 ref3s 句内参考段 x-vector，参考段不计 FM loss，EOS 正样本保留。指令都是默认指令，不涉及风格冲突。指令前缀在 collate 阶段拼接，复用已有 build 的文本 token，不重新 build。

不用句内参考的原因：句内参考段和目标是同一句话，风格天然一致，训练时参考和指令永远一致，模型可能直接从 x-vector 里拿风格，推理时遇到带情绪的参考就容易跟着参考走（FlexiVoice-Base 冲突参考下有 32–38% 跟着参考走）。

参考怎么抽：

- 有风格标签的（ears、expresso、emotion 中的有标签部分）：中性 0.4，**冲突** 0.4，同一说话人随机一句 0.2。"冲突"指风格与目标相反，例如开心对伤心、耳语对大声、快对慢；没有对应候选时退到随机。这对应 FlexiVoice S2 的冲突参考，只是放进了微调阶段。
- 其他（游戏、kespeech、voxceleb 等）：同一说话人随机一句。游戏每句的描述都不同，随机抽到的参考，风格大多和目标不一样。

每个说话人每个 epoch 设上限，主要压派蒙（每种语言 21,924 句）。具体数值在 selection 统计后确定。

以后要加回 emilia 三个子集（217 万行 / 6,070 h）：FlexiVoice 每个视频只收了两三句，但本地完整的 `emilia/v0.1`（4,026 万行）ID 一致，同一视频的片段多得多，例如 `EN_B00042_S03275` 有 31 段。先取同一视频的其他片段，再用 CAM++ x-vector 余弦确认是同一说话人（一个视频可能有多人）。

## 4. 训练

- 起点：`dots-studio/dots.tts-soar`，nv26 组装。
- 冻结 DiT，训练 LLM 和 PatchEncoder，lr 1e-5，WSD，8×A800。x-vector dropout 为 0。
- epoch 数按时长确定。第一版约 3,580 h，约为 NV 实验的 6 倍，先跑 1 个 epoch，看中间 checkpoint 的评测再决定。
- 训练后可以做 WiSE-FT 插值（α = 0.6 / 0.8），见 [副语言文档 §7](../training/dots-tts-emilia-nv-20261103.md)。

## 5. 评测

### 5.1 测试集

| 测试 | 设置 | 指标 |
| --- | --- | --- |
| **TR-easy / TR-hard**（按 FlexiVoice 复现） | MEAD（英）+ CSEMOTIONS（中），5 类情感，各 500 条；参考取同一说话人的中性句（easy）或冲突情绪句（hard）。这两个语料都不在训练数据里 | ACC-I、ACC-R（emotion2vec-large）、SV（CAM++）、WER / CER |
| **文本冲突** | 中性参考；文本语义情绪与指令相反（没有真实音频） | ACC-I、ACC-T |
| **InstructTTSEval**（APS / DSD / RP，中英各 1,000 条） | 按每题 APS 字段里的性别、年龄、口音挑匹配的中性参考；题目分成"带音色描述"和"只有风格描述"两类分开统计 | 裁判准确率；重点看与 release 的差值 |
| CV3 emotion_zeroshot | 去掉英文里参考说话人出现在训练情感库的 52 条 | 默认指令：是否仍保持参考情绪（回归）；反向指令：ACC-I、ACC-R |
| 留出说话人 | 游戏角色（四种语言一起留出）、ears 7 人；真实描述 | 人工听；日韩只能靠这项 |
| 回归 | Seed-TTS、CV3、NV-Bench（`run_dots_suite.py`），用默认指令格式 | WER、SIM、PCER、DNSMOS |

- **对照：** release soar 用同一个参考、普通模板。
- **不和公开结果直接比较。** FlexiVoice 测 InstructTTSEval 时不给参考，我们是匹配参考的设定。
- **裁判待定：** Gemini 2.5 Pro（外部 API）或本地的 Kimi-Audio-7B-Instruct。同一组对比只用同一个裁判。
- 如果能从 FlexiVoice 作者（hltlab）拿到原始的 TR / TO 测试集，就直接用，不再自己复现。

### 5.2 留出

genshin 8 个、starrail 6 个角色（不含派蒙，四种语言一起留出），ears 7 人，RAVDESS 3 人。

## 6. 风险

- **16 kHz 数据**：kespeech、emotion 主体、debatts、childmandarin、WHSP 都是 16 kHz，可能拉低 48 kHz 音质。回归看 DNSMOS；必要时降低这些子集的权重。
- **带噪和修复音频**：voxceleb 修复过、大部分带噪，kespeech 里也有标为带噪的样本。指令写"嘈杂环境"时，模型可能学会生成噪声。
- **克隆能力退化**：不混普通 Emilia，克隆能力只靠指令数据本身维持。SIM 回归要从第一个 checkpoint 开始盯。
- **自由描述变少**：中英文的自由风格描述主要来自 emilia_ds，去掉后中文自由描述只剩游戏的约 10 万条。InstructTTSEval 的 DSD、RP 可能偏弱。
- **说话人集中**：派蒙、三月七；expresso 只有 4 人。

## 7. 工作顺序

| # | 仓库 | 内容 |
| --- | --- | --- |
| 1 | tts-data-pipeline | 提交当前 adapter（不重新发布） |
| 2 | tts-data-pipeline | selection、dots_vae 特征（只针对跨句参考的子集）；提特征时解析路径说话人。副语言复用已有 build，不处理 |
| 3 | LM-TTS-Training | build：`Instruction: / Text:` 格式，按 `speaker_id` 配跨句参考（配不到的丢掉），复用的副语言 build 走 ref3s（指令前缀在 collate 拼接），按 epoch 抽参考，指令 dropout，说话人上限 |
| 4 | UltraEval-Audio | TR-easy / TR-hard 复现、InstructTTSEval 匹配参考、CV3 emotion；先用 release soar 跑基线 |
| 5 | LM-TTS-Training | 训练与评测 |

第 4 步不依赖数据，可以和 1–3 同时进行。
