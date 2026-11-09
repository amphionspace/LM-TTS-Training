# dots.tts 风格指令 TTS：操作手册与进度（2026-11-09）

目标：在 `dots.tts-soar` 上训练"参考音色 + 风格指令"的 TTS。音色只来自参考音频，说话风格（情感、语速、音量、耳语、语气等）由自然语言指令决定。不做 voice design。

设计取舍见 [设计文档](../design/dots-instruct-tts-20261108.md)；本页记录**实际怎么操作、现在做到哪一步**。设计文档与本页不一致时，以本页为准（见 §9）。

## 0. 进度总览

更新于 2026-11-09 19:00（北京时间）。

| # | 步骤 | 状态 | 产物 / 任务 |
| --- | --- | --- | --- |
| 1 | FlexiVoice 接入 | ✅ 完成 | `DATA-TTS-UNIFIED/datasets/flexivoice/v0.1`，4,199,042 行；tts-data-pipeline `bbbc261` |
| 2 | 配对统计 | ✅ 完成 | 能配到参考 1,685,465 行 / 2,799 h（§3） |
| 3 | dots_vae 特征提取（全量） | ⏳ **运行中** | ACP `pt-yro7gxbj`；18:15 时 755,913 / 4,199,042 行，预计 11-10 凌晨 4 点左右完成 |
| 4 | build / 训练代码 | ✅ 完成，已提交未推送 | LM-TTS-Training `91df100`、`a377982`；24 项测试通过 |
| 5 | 数据配置、训练配置 | ✅ 完成 | 见 §5、§6 |
| 6 | 正式 build | ⬜ 等第 3 步 | `dots-flexivoice-crossref-nv-ref3s-20261109` |
| 7 | 评测集与 release 基线 | ⬜ 未开始 | §7 |
| 8 | 训练 | ⬜ 等第 6、7 步，提交前确认 | §6 |
| 9 | 训练后评测 | ⬜ | §7 |

## 1. 数据接入（完成）

FlexiVoice-Instruct（论文 [arXiv 2601.04656](https://arxiv.org/abs/2601.04656)）的 15 个 JSON 已接入为统一数据集，说明见 tts-data-pipeline `docs/flexivoice.md`。

```bash
cd /workspace/workspace/yanglin/tts-data-pipeline
python scripts/tts_data.py bulk flexivoice \
  --root /home/yanglin/FlexiVoice \
  --output /workspace/data/DATA-TTS-UNIFIED/datasets/flexivoice/v0.1 \
  --workers 4 --batch-mib 1 --shard-mib 1024
```

- 2026-11-08 发布，manifest complete，4,199,042 行，约 9,752 h，676 个分片（standard 验证，未全量解码）。
- 每行的 `metadata_json` 含 `instruction`（指令原文）和 `upstream`（上游完整记录，含 `wav_path`、`meta`）。
- base 的 `speaker_id` 和大部分 `language` 为空。说话人在提特征后的 build 阶段从 `upstream.wav_path` 解析（§3）。

## 2. 特征提取（运行中）

对 FlexiVoice **全部行**提取 dots_vae 特征（latent、整句 x-vector、ref3s 句内参考段 x-vector），profile 与副语言第六轮相同（`dots-tts-soar-vae-xvec-48k-fp32-v2-ref3s`，profile_id `7ee9f58e…`），所以能和副语言特征放进同一个 build。

```bash
cd /workspace/workspace/yanglin/tts-data-pipeline
W=/workspace/workspace/yanglin/work/dots-vae-flexivoice-20261109
export TTS_DOTS_SOURCE=$PWD/cache/third_party/dots.tts

# 1) 计划（本地，只读 base 分片）
/workspace/workspace/yanglin/envs/tts-features/bin/python scripts/extract_dots_vae.py plan \
  --dataset flexivoice --work $W \
  --model /workspace/workspace/yanglin/LM-TTS-Training/assets/base/dots.tts-soar \
  --reference-rule dots-reference-v2:min-3s-on-patches

# 2) 提交 ACP（8 卡节点）。本机经代理访问，Python 需要系统证书，否则 SSL 校验失败
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
/workspace/workspace/yanglin/envs/tts-features/bin/python -m scripts.acp.submit \
  --work $W --name flexivoice-dots-vae-ref3s --submit \
  --cpu-script scripts/extract_dots_vae.py -- run --work $W --gpus 0 1 2 3 4 5 6 7

# 查看进度
cat $W/status.json
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
/workspace/workspace/yanglin/envs/tts-features/bin/python -m scripts.acp.status --work $W
```

- 计划：4,199,042 行，4,471 个任务。速度约 6,000 行/分钟，全程约 12–13 小时。
- 完成后发布到 `datasets/flexivoice/v0.1/features/dots_vae/flexivoice-dots_vae-dots-tts-soar-vae-xvec-48k-fp32-v2-ref3s-20261109T160744bjt-01/`。
- 中断后用同一命令重新提交即可从检查点继续。

**为什么没有先做 selection。** 现有 selection 执行器接不了 FlexiVoice：大部分行语言为空会被 `2001` 排除，不能按子集过滤，同一音频配多条指令的行会被去重，参考协议也只支持 `self`。补这些要改执行器和契约。FlexiVoice 不改写文本，selection 只决定"用哪些行"，不影响特征本身，所以先全量提取，过滤和说话人配对放在 build 里做。以后如需正式的 selection 记录，可以补发布，build 按它筛选，特征直接复用。

## 3. 说话人与跨句参考

**说话人键**（`lm_tts/data/flexivoice.py: speaker_key`），只用上游路径：

| 子集 | 说话人键 | 例子 |
| --- | --- | --- |
| genshin、starrail | 语种 / 角色（不同语种是不同声优） | `genshin:ZH:派蒙` |
| ears、l2 | 目录 | `ears:p043` |
| expresso | `exNN`；对话按 `channel1` / `channel2` 区分两位说话人 | `expresso:ex01` |
| kespeech、voxceleb、childmandarin、WHSP | 说话人 ID | `kespeech:1019466` |
| debatts | 场次 + 说话人 | `debatts:3345_SPEAKER01` |
| emotion | ESD、CREMA-D、RAVDESS 演员号；EMNS 单说话人；IEMOCAP 为 session + 性别 | `IEMOCAP:Session3-F` |
| emilia 三个子集、nvspeech、MSP-PODCAST、MELD、m3ed | 无（不用） | — |

**配对规则**：同一说话人里有另一句、文本不同（`utterance_key` 归一化后比较）、时长 ≥ 3 s，该行才保留。

**风格类别**（`style_class`）：ears 取文件名里的风格，expresso 取目录名，emotion 取上游 `meta` 标签，统一映射到 14 类（neutral、happy、sad、angry、fear、surprise、disgust、whisper、loud、fast、slow、high、low、other）。游戏、kespeech、voxceleb 等没有标签，记为 -1。

**训练时怎么抽参考**（每次访问重新抽）：

| 目标行 | 参考 |
| --- | --- |
| 有风格标签 | 0.4 中性、0.4 冲突（冲突表见 `CONFLICTS`，例如开心对伤心、耳语对大声、快对慢）、0.2 同一说话人任意一句；没有对应候选时退到任意 |
| 无风格标签 | 同一说话人任意一句 |

送进 DiT 的是参考句的**整句 x-vector**，目标句整句计 loss，参考区间为 `(0, 0)`，不做 mask。

**统计结果**（真实 base 元数据上跑 build 的过滤和配对逻辑）：

| 原因 | 行数 |
| --- | --- |
| 不在所选子集（emilia 三个、nvspeech） | 2,434,710 |
| 路径里没有说话人（MSP-PODCAST、MELD、m3ed） | 66,704 |
| 留出评测说话人 | 28,262 |
| 时长不在 1–30 s | 9,417 |
| 配不到参考 | 2,715 |
| 采样率 < 16 kHz | 31 |
| **保留** | **1,657,203 行 / 2,744 h** |

按子集：kespeech 975,548，genshin 289,657，voxceleb 99,163，starrail 95,552，emotion 46,021，childmandarin 39,861，expresso 27,074，l2 26,826，ears 26,542，debatts 20,010，WHSP 10,949。有风格标签的约 10 万行，其中中性 13,720 行。

## 4. 输入格式与代码改动

**格式**（照 FlexiVoice 的 Instruction / Text 两字段）：

```text
[文本]Instruction: {instruction}
Text:
{text}[文本对应语音]<start> S*N <end><eos>
```

- `{instruction}` 是 `metadata.instruction` 原文；没有指令时用 `Speak the following text.`。
- 前缀以换行结尾。训练时前缀和正文分开分词再拼接，推理时 runtime 把整串一起分词；以 `Text: `（空格）结尾时两者在 7 条里有 4 条不一致，换成 `Text:\n` 后在 25,000 条 FlexiVoice 和 NV 文本上全部一致。
- **推理时**直接把 `"Instruction: …\nText:\n" + 正文` 作为 `text` 传给官方 runtime 的默认 `tts` 模板，不需要改 runtime。

**代码**（LM-TTS-Training）：

| 文件 | 内容 |
| --- | --- |
| `lm_tts/data/flexivoice.py`（新） | 说话人键、风格类别、冲突表、指令模板 |
| `lm_tts/data/dots.py` | build：`instructions: flexivoice-v1`；binding 新增 `metadata: flexivoice`、`reference: cross_speaker`、`source_configs`、`duration_seconds`、`min_sample_rate`、`holdout_speakers`；配不到参考的行丢掉；每个跨句 binding 写参考索引 `references-NNN.npz`（按特征表行号索引）；指令前缀长度计入 token 预算。训练：跨句行取参考行的整句 `xvector`；指令 dropout；带指令的样本只用 basic 布局 |
| `lm_tts/data/split.py` | train / validation 切分时把参考索引复制到两个目录 |
| `lm_tts/data/build.py` | recipe 允许 `instructions` |
| `lm_tts/training/flow_task.py` | 新配置项 `train.instruction_dropout` |
| `tests/test_dots_instructions.py`（新） | 前缀布局、说话人和风格解析、配对规则、参考抽样比例 |

没有指令的旧 build 行为不变（原有测试全部通过）。

## 5. Build

数据配置：`configs/data-flexivoice-nv-dots-ref3s.yaml`

- **flexivoice binding**：`metadata: flexivoice`、`reference: cross_speaker`；子集 `WHSP, childmandarin, debatts, ears, emotion, expresso, genshin, kespeech, l2, starrail, voxceleb`；时长 1–30 s；采样率 ≥ 16 kHz；66 个留出说话人键（§7.1）。
- **5 个副语言 binding**：与 `data-nv-canonical-dots-ref3s.yaml` 相同（emilia_nv、nonverbalspeech38k、smiip_nv、disfluencyspeech、nonverbaltts，第六轮 ref3s 特征），句内参考加 mask，统一加默认指令。FlexiVoice 的 nvspeech 子集就是 emilia_nv（18 种标签一致，174,179 条带标签行全部重合），不另外使用。
- validation 1%，`split_seed: 42`。

```bash
cd /workspace/workspace/yanglin/LM-TTS-Training
PYTHONPATH=$PWD /workspace/workspace/yanglin/envs/lm-tts/bin/python scripts/build_unified.py \
  --config configs/dots-tts-20261109-soar-nv26-flexivoice-crossref-instr-frozenhead-bf16-8gpu-lr1e-5-ep1-wsd.yaml
```

**预期**：flexivoice 1,657,203 行（若特征有失败行会略少），副语言 230,079 行，合计约 189 万行、约 3,475 h。build 完成后检查各 binding 的 `coverage.excluded`，与 §3 的数字对照。

**已验证**：只含 emilia_nv 的试跑 build 得到 173,485 行，与第六轮一致；解码后的序列为 `[文本]Instruction: Speak the following text.\nText:\n…[文本对应语音]`，参考区间、token 预算正常。

## 6. 训练

配置：`configs/dots-tts-20261109-soar-nv26-flexivoice-crossref-instr-frozenhead-bf16-8gpu-lr1e-5-ep1-wsd.yaml`，继承冻结 DiT 的 5ds 配置：

| 项目 | 设置 |
| --- | --- |
| 起点 | `dots.tts-soar`，nv26 组装 |
| 冻结 | `velocity_field_predictor, hidden_proj, latent_proj, coordinate_proj, xvec_proj` |
| 学习率 | 1e-5，WSD（warmup 2%，最后 20% 降到 0.1 倍） |
| 数据量 | 约 3,475 h，1 个 epoch，约 19k 步 |
| x-vector dropout | 0 |
| 指令 dropout | 0.1 |
| interleave | 0.5（只作用于默认指令的样本） |
| batch | 每卡 600 patch / 1,000 token |
| checkpoint / 验证 | 每 1,000 步 |
| 资源 | 1 节点 8×A800 |

```bash
cd /workspace/workspace/yanglin/LM-TTS-Training
source scripts/acp/env.sh
CFG=configs/dots-tts-20261109-soar-nv26-flexivoice-crossref-instr-frozenhead-bf16-8gpu-lr1e-5-ep1-wsd.yaml
PYTHONPATH=$PWD /workspace/workspace/yanglin/envs/lm-tts/bin/python -m scripts.acp.submit --config $CFG          # 只渲染
PYTHONPATH=$PWD /workspace/workspace/yanglin/envs/lm-tts/bin/python -m scripts.acp.submit --config $CFG --submit # 提交
```

提交前确认。run 目录在 `/workspace/LM-TTS-Training-Runs/<run_name>/`。

## 7. 评测（未开始）

### 7.1 留出说话人

固定种子挑选，训练中完全不出现（build 里以 `held_out` 排除）：

- genshin 8 个角色，四种语言一起留出：早柚、莫娜、莱欧斯利、流浪者、珐露珊、可莉、克洛琳德、行秋。
- starrail 6 个角色，四种语言一起留出：玲可、真理医生、银枝、帕姆、阮•梅、驭空。
- EARS 7 人：p007、p106、p096、p003、p066、p013、p088。
- RAVDESS 3 人：Actor_12、Actor_17、Actor_10。

### 7.2 测试集

| 测试 | 构造 | 指标 |
| --- | --- | --- |
| **TR-easy / TR-hard（中文）** | 按 FlexiVoice 的方法用 CSEMOTIONS（本地已有，10 人、7 种情感、48 kHz，不在训练数据里）：参考取同一说话人的中性句（easy）或冲突情绪句（hard），指令为目标情绪 | ACC-I、ACC-R（emotion2vec-plus-large）、SV（CAM++）、CER |
| **TR-easy / TR-hard（英文）** | MEAD 在镜像站上找不到（原始数据在 Google Drive，带视频，很大），改用留出的 EARS 7 人和 RAVDESS 3 人：参考取 regular / neutral 句或冲突风格句 | 同上；EARS 的耳语、快慢、高低另用声学量（清音比例、音节速率、F0） |
| **文本冲突** | 中性参考，文本语义情绪与指令相反（LLM 生成，中英各 200 条，没有真实音频） | ACC-I、ACC-T |
| **CV3 emotion_zeroshot** | 中文 150 条；英文去掉参考说话人出现在训练情感库的 52 条后剩 98 条。默认指令测"仍保持参考情绪"（回归），反向指令测指令跟随和参考泄漏 | emotion2vec |
| **InstructTTSEval** | 按每题 APS 字段的性别、年龄、口音挑匹配的中性参考；题目分"带音色描述"和"只有风格描述"两类 | 裁判准确率，看与 release 的差值；裁判待定 |
| **留出游戏角色** | 真实描述，中英日韩 | 人工听；日韩只能靠这项 |
| **回归** | Seed-TTS、CV3、NV-Bench（`run_dots_suite.py`），默认指令格式 | WER、SIM、PCER、DNSMOS |

**对照**：release soar 用同一个参考、普通文本（不加指令）。

**推理**：UltraEval-Audio 的 dots 推理（`scripts/local_dots.py`）用 x-vector 模式（speaker_only），`text` 传 `"Instruction: …\nText:\n" + 正文`；release 基线只传正文。

## 8. 待定与风险

- **InstructTTSEval 的裁判**：Gemini 2.5 Pro（外部 API）或本地 Kimi-Audio-7B-Instruct，需要定。
- **kespeech 占比高**：保留行的 59%，全部 16 kHz，指令是模板。第一轮不设上限；若回归里音质（DNSMOS）或指令跟随受影响，下一轮按比例降采样。
- **16 kHz 数据**：kespeech、ESD、CREMA-D、debatts、childmandarin、WHSP 都是 16 kHz。
- **voxceleb**：voicefixer 修复过，81% 描述为带噪。
- **说话人集中**：派蒙每种语言 2 万余句，expresso 只有 4 人。第一轮不设上限。
- **日韩只出现在游戏指令数据里**，普通克隆和副语言数据都是中英；日韩克隆质量单独看。
- **没有 selection 记录**：过滤只体现在 build manifest 的 `coverage.excluded` 里（§2）。

## 9. 与设计文档的差异

| 设计文档 | 实际 |
| --- | --- |
| 先建 `selection-flexivoice-v1`，再在 selection 分支上提特征 | 全量提特征，过滤和配对在 build 里做（§2） |
| 指令格式 `Instruction: …\nText: ` | `Instruction: …\nText:\n`，保证训练与推理分词一致（§4） |
| 每个说话人设上限、kespeech 降采样 | 第一轮都不设 |
| MEAD + CSEMOTIONS 复现 TR | 中文用 CSEMOTIONS；英文改用留出的 EARS、RAVDESS（§7.2） |

## 10. 相关位置

| 内容 | 位置 |
| --- | --- |
| FlexiVoice 原始数据 | `/home/yanglin/FlexiVoice` |
| 统一数据 | `/workspace/data/DATA-TTS-UNIFIED/datasets/flexivoice/v0.1` |
| 特征提取工作目录 | `/workspace/workspace/yanglin/work/dots-vae-flexivoice-20261109` |
| 副语言 ref3s build（复用其 binding） | `LM-TTS-Training/data/builds/dots-nv-canonical-v1-ref3s-20261106` |
| 设计文档 | `docs/design/dots-instruct-tts-20261108.md` |
| 副语言实验记录 | `docs/training/dots-tts-emilia-nv-20261103.md` |
