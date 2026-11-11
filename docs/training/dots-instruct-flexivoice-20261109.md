# dots.tts 风格指令 TTS：操作手册与进度（2026-11-09）

目标：在 `dots.tts-soar` 上训练"参考音色 + 风格指令"的 TTS。音色只来自参考音频，说话风格（情感、语速、音量、耳语、语气等）由自然语言指令决定。不做 voice design。

设计取舍见 [设计文档](../design/dots-instruct-tts-20261108.md)；本页记录**实际怎么操作、现在做到哪一步**。设计文档与本页不一致时，以本页为准（见 §9）。

## 0. 进度总览

更新于 2026-11-10。当前流程已改为 **文本修订 annotation → selection → features → build**；当前新实验为 packed DiT + 全 interleave + 无 instruction dropout + 全局 Feistel shuffle，单卡预算 48,000 tokens / 30,000 patches、accumulation 1、lr=5e-5，任务 `pt-jt0v0he9`；前一 32k 任务 `pt-ng26vb3i` 按用户要求停止保留，原始基线任务也继续保留。旧 24k 任务及运行目录已删除。

| # | 步骤 | 状态 | 产物 / 任务 |
| --- | --- | --- | --- |
| 1 | FlexiVoice 接入 | ✅ 完成 | `DATA-TTS-UNIFIED/datasets/flexivoice/v0.1`，4,199,042 行；tts-data-pipeline `bbbc261` |
| 2 | 文本修订与 selection | ✅ 完成 | `…20261110T092455bjt-01`；1,656,126 行 / 2,742.5 原生小时，参考全量复核 0 个孤立目标 |
| 3 | dots_vae 特征提取（全量） | ✅ 完成 | ACP `pt-yro7gxbj`；11-10 03:39 发布，4,199,042 / 4,199,042 ok，作为重发的音频张量来源 |
| 4 | build / 训练代码 | ✅ 原实现已提交；本次修复尚未提交，已固定快照 | `91df100`、`a377982` 加参考池/说话人切分修复；52 项相关测试通过 |
| 5 | selection features 与配置 | ✅ 11-10 09:59 发布，1,656,126 行全量校验通过 | `…20261110T093409bjt-01` |
| 6 | 正式 build | ✅ 11-10 10:03 完成，完整校验通过 | 旧 build 因原始标签失败；新 build ID 为 `dots-flexivoice-selection-nv-ref3s-20261110` |
| 7 | 评测集与 release 基线 | ⬜ 未开始 | §7 |
| 8 | 训练 | ✅ 已启动，ACP `RUNNING` | 28k 任务 `pt-fae45akp` 在 step 38 OOM，平台 FAILED；原任务停止保留 |
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
- base 的 `speaker_id` 和大部分 `language` 为空。说话人在 selection 阶段从 `upstream.wav_path` 解析以过滤、配对；build 读取同一分支的元数据用于抽参考（§3）。

## 2. 全量特征与 selection 重发

对 FlexiVoice **全部行**提取 dots_vae 特征（latent、句级 x-vector（最多前 10 秒）、ref3s 句内参考段 x-vector），profile 与副语言第六轮相同（`dots-tts-soar-vae-xvec-48k-fp32-v2-ref3s`，profile_id `7ee9f58e…`），所以能和副语言特征放进同一个 build。

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

**当前流程。** 全量提取是历史产物，保留用于复用音频张量与提供固定的处理后时长证据。新流程先发布 `flexivoice_text` 文本修订，再发布 `selection-flexivoice-v1`：筛选子集、原生时长 1–30 秒、采样率 ≥16 kHz、留出说话人、文本可用性和闭合的跨句参考池；同音频的无冲突指令变体保留。features 只包含入选行，并直接携带 selected_text。

文本修订为 `tts-ann-flexivoice_text-20261110T091454bjt-01`：移除 2 处 `[inaudible]`，统一 34 处 `[LAUGHTER]` 和 9 处 `[BREATHING]`，13 行含 `[GARBAGE]`/`[garbage]` 的转写标为不可用。原文不变。

参考 ≥3 秒使用原始全量 dots_vae 发布中的 `processed_frames / 48000`（profile、manifest、表版本均固定），不使用原生时长代替；原生时长只用于 1–30 秒策略。基于原生时长的初版 selection `…20261110T091733bjt-01` 在复核中发现 657 行不能维持处理后的参考池，因此不用该版训练，也未基于它发布 features。

发布器 `tts_data_pipeline.dots_vae.reselect` 校验 sample_id、音频哈希、时间轴和 profile 后复制已有 latent/x-vector，仅替换为 selection 的最终文本与语言；不重复 GPU 提取。所有新特征全量回读校验张量哈希和文本一致性。原始 features 不覆盖。

最终 selection：`tts-selection-supervised-tts-20261110T092455bjt-01`，1,656,126 行；原生时长 2,742.4787 小时，dots 处理后时长 2,758.1938 小时。参考闭包再次全量检查为 0 个孤立目标。过滤原因如下（主原因互斥）：

| 原因码 | 原因 | 行数 |
| --- | --- | --- |
| 0 | selected | 1,656,126 |
| 1003 | text_source_unavailable | 13 |
| 3001 | below_duration_policy | 7,359 |
| 3002 | above_duration_policy | 2,058 |
| 3003 | below_sample_rate_policy | 31 |
| 3004 | source_config_not_selected | 2,434,710 |
| 4002 | conflicting_duplicate | 515 |
| 5001 | reference_identity_unavailable | 66,704 |
| 5002 | reference_pair_unavailable | 3,264 |
| 7001 | held_out | 28,262 |

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

**配对规则**：由 selection 在所有过滤、重复冲突与留出之后确定闭合的入选集合。每个入选说话人至少有两种不同归一化文本的 ≥3 秒参考（使用固定 dots 实际处理时长）；否则该说话人的所有原本合格行记为 `5002`。不单独保留“仅作参考”的未入选行。build 仅验证不变量并建立抽样索引，不再丢行补救。

**风格类别**（`style_class`）：ears 取文件名里的风格，expresso 取目录名，emotion 取上游 `meta` 标签，统一映射到 14 类（neutral、happy、sad、angry、fear、surprise、disgust、whisper、loud、fast、slow、high、low、other）。游戏、kespeech、voxceleb 等没有标签，记为 -1。

**训练时怎么抽参考**（每次访问重新抽）：

| 目标行 | 参考 |
| --- | --- |
| 有风格标签 | 0.4 中性、0.4 冲突（冲突表见 `CONFLICTS`，例如开心对伤心、耳语对大声、快对慢）、0.2 同一说话人任意一句；没有对应候选时退到任意 |
| 无风格标签 | 同一说话人任意一句 |

送进 DiT 的是参考句的**句级 x-vector**（当前固定提取 profile 最多使用前 10 秒），目标句整句计 loss，参考区间为 `(0, 0)`，不做 mask。

**历史预估**（旧 build 过滤逻辑；当前以已发布 selection manifest 的 reason_counts 为准）：

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
| `lm_tts/data/dots.py` | 消费固定 selection_branch 的 features、指令和说话人元数据；不再清理文本或筛选子集/时长/采样率/留出/配对；配对失败视为上游不变量错误。参考索引、指令前缀、dropout 和训练抽样保留 |
| `lm_tts/data/split.py` | FlexiVoice 按说话人切分，分别写入不相交的参考池；其他 binding 继续按行切分 |
| `lm_tts/data/build.py` | recipe 允许 `instructions` |
| `lm_tts/training/flow_task.py` | 新配置项 `train.instruction_dropout` |
| `tests/test_dots_instructions.py`（新） | 前缀布局、说话人和风格解析、配对规则、参考抽样比例 |

没有指令的旧 build 行为不变（原有测试全部通过）。

## 5. Build

数据配置：`configs/data-flexivoice-nv-dots-ref3s.yaml`

- **flexivoice binding**：`metadata: flexivoice`、`reference: cross_speaker`、`text_kind: flexivoice-text-v1`；其余业务过滤已移至 `tts-data-pipeline/configs/selections/selection-flexivoice-v1.json`。
- **5 个副语言 binding**：与 `data-nv-canonical-dots-ref3s.yaml` 相同（emilia_nv、nonverbalspeech38k、smiip_nv、disfluencyspeech、nonverbaltts，第六轮 ref3s 特征），句内参考加 mask，统一加默认指令。FlexiVoice 的 nvspeech 子集就是 emilia_nv（18 种标签一致，174,179 条带标签行全部重合），不另外使用。
- `split_seed: 42`。FlexiVoice 随机留出约 1% 的说话人键作为 validation，目标和参考都按相同说话人分区；实际行数比例由 build 报告给出，不保证恰好 1%。同一说话人的同音频多指令不会跨分区。其他 5 个 binding 仍按行留出 1%。这份 validation 与 §7 固定留出的 66 个说话人键是两套集合。

```bash
cd /workspace/workspace/yanglin/LM-TTS-Training
PYTHONPATH=$PWD /workspace/workspace/yanglin/envs/lm-tts/bin/python scripts/build_unified.py \
  --config configs/dots-tts-20261109-soar-nv26-flexivoice-crossref-instr-frozenhead-bf16-8gpu-lr1e-5-ep1-wsd.yaml
```

**预期**：flexivoice 1,656,126 行，副语言 230,079 行，合计 1,886,205 行。FlexiVoice 的原生时长与 dots 处理后时长分别记录，训练预算以处理后 patch 数为准。新的 build 不应再因 FlexiVoice 业务过滤排除行。

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
| 指令 dropout | 0.1；替换为默认指令，保留原参考与目标，默认指令表示“不指定风格” |
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

用户暂停旧流程后，已再次明确授权“搞完直接 build，然后启动训练”；新 features 发布及 build 校验通过后自动提交一次。run 目录在 `/workspace/LM-TTS-Training-Runs/<run_name>/`。

### 6.1 自动接续

旧监控位于 `artifacts/flexivoice-launch-20261109/`。11-10 03:43 自动开始旧 build，03:46 因 `[inaudible]` 原始标签失败；没有发出训练提交请求。用户要求暂停后已写入 `STOP`，未再启动该监控。

旧目录的 runtime/experiment 不用于新版 selection 数据。

新接续目录：`artifacts/flexivoice-selection-launch-20261110/`。11-10 09:53 启动后台进程 PID `2173318`，每 60 秒检查新 features 的正式发布；runtime、数据 recipe、实验配置均已固定快照。features 核对 selection ID/hash/入选行数和提取 profile，通过后自动 build → 完整参考/预算校验 → 单次提交 8×A800 训练。`submission-attempt.json` 先于提交请求落盘，响应不明确时不会自动重试。

进度：`status.json`；构建：`build.log`；全量校验报告：`build-validation.json`；提交：`submit.log` 和 `training-job.json`，均位于新目录。

11-10 10:03 正式 build 和提交前校验通过，共 1,886,205 条。FlexiVoice 全部 1,656,126 条进入 build，训练 1,618,141 条（32,526 位说话人）、验证 37,985 条（329 位说话人）；参考池隔离、可配对性、全量序列预算及各 binding 实际取数校验通过。NV 沿用既有参考有效性约束，emilia_nv 和 nonverbaltts 各 2 条无有效参考，详见 `build-validation.json`。

训练已于 11-10 10:03:42 提交一次，ACP 任务 **`pt-c42buvr4`**，10:03:52 进入 **RUNNING**。接续监控已完成其 features → build → submit 工作并退出；启动状态记录在 `training-status.json`。10:07 已确认 8 卡 bf16 / FlashAttention 2 初始化成功并完成 step 30，loss 与梯度均为有限值，skipped/discarded samples 均为 0；实际训练日志和 `metrics.jsonl` 位于 run 目录。

已完成全量 FlexiVoice token 预算审计：1,656,126 行，最大 191 patches、正文加指令最多 227 tokens、序列最多 421 tokens，0 行超过现有 600 patches / 1,000 tokens 预算。报告在 `artifacts/flexivoice-selection-audit-20261110/token-budget.json`。

### 6.2 全 interleave、无 instruction dropout 对照（2026-11-10）

用户要求另起实验，配置 `configs/dots-tts-20261110-flexivoice-allinterleave-instrdrop0.yaml`：`interleave_fraction=1.0`、`instruction_dropout=0.0`，复用同一 selection build，从同一 assembled release 独立开始训练。其余训练参数沿用原实验（8×A800、bf16、冻结 flow head、lr=1e-5、1 epoch、WSD）；x-vector dropout 仍为 0，模型 CFG dropout 仍为 0.2。

ACP 已提交新任务 `pt-ise6stco`，独立 run 为 `dots-tts-20261110-flexivoice-allinterleave-instrdrop0-frozenhead-bf16-8gpu-lr1e-5-ep1-wsd`。

此实验的流式格式为 `[流式语音合成] Instruction: …\nText:\n 正文 token / audio span 交错`，完整指令在正文前，不参与一字一音频的交错。取消带指令和文本长于音频时退回 basic 的限制；长文本遵循 release 的交错循环，音频结束后保留剩余文本，不增加音频 span、不截断正文。旧实验使用已固定 runtime，继续按原格式运行。

提交前校验：原有相关测试通过；全量 train/validation 索引预算通过，六路数据各抽取 train/validation 16 条（共 192 条），确认全部 interleave、指令保留、span/预测位置与音频 patch 数一致、张量有限值。报告：`artifacts/flexivoice-allinterleave-launch-20261110/data-validation.json`。

### 6.3 step 9000 试听

已从原实验 step 9000 导出 bf16 checkpoint，在当前开发节点 GPU 0 生成 10 段 WAV：中英文各默认/开心/悲伤/耳语四段，加原版 release 中英文各一段。相同语言固定参考音频、正文和 seed=42；non-streaming / x-vector-only，10 ODE steps、CFG=1.2，完整 instruction 前缀与训练一致。试听属于人工检查，尚未作效果结论。

产物目录 `artifacts/flexivoice-listen-step9000/`：`index.html` 播放页、`listen-step9000.zip` 下载包、`samples.json` 文本/指令/参数，`audio/` 含生成音频及参考。12 个音频（含两段参考）均可回读、长度有效、数值有限。曾尝试远端训练节点推理，但在导入阶段因 Torch/torchaudio 版本不匹配退出，实际生成全部在当前开发节点完成，未修改训练环境。

### 6.4 step 9000 风格控制弱：输入与采样审计

用户试听认为风格差异不明显后，核查了已导出的 step 9000 与固定训练 runtime。256 条跨数据位置的样本，训练前缀 token 和官方推理生成 schedule 的音频开始前 token 完全一致（0 mismatch）；抽查 LLM 第 0/20 层 q_proj 权重相对初始值分别改变约 7.49%/10.75%，冻结 hidden_proj 无变化。未发现指令缺失或误用初始权重的证据。

实质问题：运行配置未指定 sampler，实际采用 `window_shuffle_token_budget_v2`，以 65,536 行为块打乱；build 本身按来源聚集。根据 checkpoint 的 `samples_in_epoch=675671` 和 seed=42 精确还原顺序，step 9000 对 WHSP、EARS、emotion、Expresso、childmandarin、debatts、l2 的访问数均为 0；已访问 kespeech 402,616、voxceleb 98,443、genshin 65,536、starrail 12,370，其余为部分 NV。因此该试听 checkpoint 尚未接触主要耳语/情绪来源，不足以判断完整训练后的指令能力。新 interleave 实验继承了同一 sampler 设置。

建议把两组对照统一改为已有的 `train.sampler: feistel_uniform_token_budget_v3`，让每批混合全体数据，再比较风格控制。此次仅审计和对照采样，不修改或停止已提交任务。报告：`artifacts/flexivoice-instruction-audit/coverage.json`、`input-audit.json`，审计脚本也保存在该目录。

另一个推理差异是原版 runtime 默认 `speaker_scale=1.5`，训练直接使用未放大的 x-vector。它尚未被证明是风格弱的原因，而且 xvec_proj 后有 LayerNorm，不能简单推断其影响随倍数等比例变化；后续试听应显式固定 1.0 做对照。

### 6.5 全局 shuffle + 大 batch 重启

用户要求停止现有两实验、保留已训练任务/产物、删除未启动任务，重新只启动全 interleave 且无 instruction dropout 的实验。已停止 `pt-c42buvr4`（平台 SUSPENDED，保留 checkpoint）；`pt-ise6stco` 停止后删除，API GET 返回 404。旧任务不再自动接续。

新配置：`configs/dots-tts-20261110-flexivoice-interleave-feistel-batch8x.yaml`，从相同 assembled release 重新训练，复用同一 build；显式 `sampler: feistel_uniform_token_budget_v3`、interleave=1、instruction_dropout=0。每卡预算 2400 patches / 4000 tokens，accumulation=2，有效 batch 预算是原实验 8 倍；lr=1e-5、冻结 flow head、bf16、8×A800、1 epoch、WSD 保持。save/eval 从每 1000 步调为每 125 步，近似保持样本量间隔。

当前节点真实数据显存探测：单次 4800/8000 反向发生 OOM，改用 2400/4000 + 两次累积验证。连续四微批（39/39/44/43 条）全部通过，loss/梯度有限，最后一批将峰值推至 39.76 GiB。探测是单卡未分片模型，不等同生产 FSDP 峰值；生产启动后继续检查实际显存和数值。记录位于 `artifacts/flexivoice-feistel-batch8x-launch-20261110/`。

已被替代的任务：`pt-24j1glhu`，11-10 13:45:07（北京时间）进入 RUNNING；固定提交配置和运行时 training_plan 确认 Feistel、2400/4000、accumulation=2、interleave=1、instruction_dropout=0。

已验证实际完成 step 20：每次更新 620–652 条、约 25,000 个监督 patches，5.93–6.54 秒/步；日志峰值分配显存约 30–33 GiB，nvidia-smi 各卡约 35–37 GiB、利用率 98–100%。FM loss=0.7772、EOS loss=0.1135，skipped/discarded=0，梯度有限。

提交前同时修正全 interleave 的长文本尾部：与本地原版 runtime 一致，在音频先结束时仍补入 `<|text_cond_end|>`。29 项 dots/指令相关测试通过。

### 6.6 36k token 版本（替代 §6.5）

用户要求进一步增至 36k，并清理无用任务。`pt-24j1glhu` 已停止并删除（GET 404）；最初的 `pt-c42buvr4` 和 checkpoint 保留。新任务 `pt-37unlr6f` 使用配置 `configs/dots-tts-20261110-flexivoice-interleave-feistel-36k.yaml`：每卡 24,000 patches / 36,000 tokens、accumulation=1，Feistel、全 interleave、instruction_dropout=0。11-10 13:55:37（北京时间）进入 RUNNING，正在验证实际首步。

更正 §6.5 的单卡探测局限：脚本仅设置 `model.checkpointing=True`，未调用启用 LLM checkpointing 的完整方法，且没有 FSDP，不能用那个 OOM 推断正式 8 卡训练上限。36k 版本直接采用正式 engine 的 activation checkpointing + FSDP 路径，以实际运行结果判定。日志位于 `artifacts/flexivoice-feistel-36k-launch-20261110/` 和对应 run 目录。

### 6.7 32k 实验（已因 OOM 停止删除）

36k 任务首步失败：节点显存接近 80 GB，rank 0 已进入异常清理，其他 rank 仍停在 patch encoder；该任务 `pt-37unlr6f` 已停止删除（GET 404）。检查发现 patch encoder 的 Encoder 层没有接入 activation checkpointing；现已补齐，且以 tiny model 对照验证前向输出、全部参数梯度一致，并验证反向实际重算。11 项 dots 模型测试通过。另在训练异常清理前打印 traceback，避免清理阶段等待掩盖原始错误。

用户随后指定改为 32k。当前配置 `configs/dots-tts-20261110-flexivoice-interleave-feistel-32k.yaml`，任务 **`pt-hvrg2vtt`**，11-10 14:08:03（北京时间）启动：每卡 20,000 patches / 32,000 tokens、accumulation=1、Feistel、全 interleave、instruction_dropout=0，8×A800 bf16，完整 activation checkpointing，从相同 assembled release 重新训练。first_check=5，save/eval 每 25 步。

实际连续完成 step 1–3：每步 2,650–2,664 条，约 104k 监督 patches；初始读数预热后 23.7–24.0 秒/步、约 744–755 音频秒/秒；日志峰值 allocated 65.4–66.8 GiB，loss 和梯度均有限，skipped/discarded=0。这证明初始批次已跑通，不代表整轮所有长样本批次的显存上限已验证。

清理状态：`pt-ise6stco`、`pt-24j1glhu`、`pt-37unlr6f` 均已删除；`pt-c42buvr4` 停止保留及其全部 checkpoint。当前只运行上述 32k 新实验。报告与提交快照索引：`artifacts/flexivoice-feistel-32k-launch-20261110/`。

### 6.8 28k 实验（当前）

32k 实验完成 step 32 后在后续批次的 DiT attention/RoPE 处 OOM：GPU 1 需要再分配 4.31 GiB，空闲仅 4.20 GiB。因此早期 3–5 步通过不足以证明长时间稳定。按用户要求，`pt-hvrg2vtt` 已停止删除（GET 404），整个 32k run 目录连同 checkpoint 已删除；只保留小体积失败日志供定位，原实验 checkpoint 未动。

新任务 **`pt-fae45akp`**，配置 `configs/dots-tts-20261110-flexivoice-interleave-feistel-28k.yaml`：每卡 17,500 patches / 28,000 tokens、accumulation=1，其余沿用 32k 的 Feistel、全 interleave、instruction_dropout=0、完整 activation checkpointing、8×A800 bf16、冻结 flow head、1 epoch、lr=1e-5、WSD。提交记录和失败诊断位于 `artifacts/flexivoice-feistel-28k-launch-20261110/`。

28k 已于 11-10 14:59:24（北京时间）进入 RUNNING，并实际完成 step 1–5。每次更新约 2,288–2,345 条，预热后约 20–21 秒/步，日志 peak allocated 约 56.6–58.5 GiB，loss/梯度均有限、无 skipped/discarded。这里只确认初始运行正常，不据此声称整轮不会再 OOM。

修正异常退出：rank 内未捕获异常先输出并 flush traceback，再以非零状态退出，让 torchrun 终止同组进程，避免 `destroy_process_group()` 等待未到清理阶段的其他 rank 而把失败任务留在 RUNNING。子进程注入 OOM 测试确认返回码 1、保留异常信息、未进入 collective teardown。

## 7. 评测（未开始）

### 7.1 留出说话人

固定种子挑选，训练中完全不出现（selection 中按留出说话人排除）：

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
| **CV3 emotion_zeroshot** | 中文 150 条；英文去掉参考说话人出现在训练情感库的 52 条后剩 98 条。默认指令观察参考情绪跟随率（诊断项，不作为必须保持的回归门槛），反向指令测指令跟随和参考泄漏 | emotion2vec |
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
- **选择证据已前移**：以 selection manifest/rules/reason_counts 为准；音质和指令跟随仍需独立评测。

## 9. 与设计文档的差异

| 设计文档 | 实际 |
| --- | --- |
| 先建 `selection-flexivoice-v1`，再在 selection 分支上提特征 | 已恢复 annotation → selection → features 分工；复用历史全量特征，固定新 selection 后重新发布，不重复 GPU 提取（§2） |
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

### 28k 后续失败核查（11-10）

任务实际完成 step 37，step 38 在 rank 3 的 DiT checkpoint 反向重算（q/k RMSNorm）处 OOM：需 3.82 GiB，空闲 1.60 GiB。快速退出生效，torchrun 终止其他 rank，平台 FAILED；step 5/25 checkpoint 均 COMPLETE。

精确还原 rank 3 相邻两批：step 37 为 288 条、27,955 tokens、12,540 真实 patches、最长 186 patches，补齐后 53,568 patches；step 38 为 292 条、27,876 tokens、12,231 真实 patches、最长 343 patches，补齐后 100,156 patches。总 token 未超预算，但 DiT 按批内最长句补齐，实际张量长度接近翻倍，稠密 attention mask 成本还随最长长度平方增长。仅调低总 token 上限不足以约束此峰值，后续应约束 padded 成本或改变音频分支的 padding 执行方式。证据：`artifacts/flexivoice-feistel-28k-launch-20261110/failure-batch.json`、`replay_failure.py`。


### 24k 重启与 DiT padding 核查（11-10）

按用户要求删除失败的 28k 任务 `pt-fae45akp` 及运行目录；保留失败日志、batch replay 证据。24k 配置为 `configs/dots-tts-20261110-flexivoice-interleave-feistel-24k.yaml`，任务 `pt-0pxwktnv`，运行目录 `dots-tts-20261110-flexivoice-interleave-instrdrop0-feistel-24k-peckpt-bf16-8gpu-lr1e-5-ep1-wsd`。复用原 build 和初始化权重，维持全 interleave、instruction dropout 0、Feistel、8 卡、patch encoder/DiT checkpointing 和快速失败退出。

启动核查：平台 RUNNING，已完成 step 5，checkpoint 有 COMPLETE 标记。step 3–5 每步约 17–18 秒，记录的 peak_memory_gib 约 49–50 GiB；这只证明前几批可运行，尚不能排除后续长句批次 OOM。

代码确认：LLM 的 FA2 分支使用 packed varlen；DiT `velocity()` 使用 `[batch, 10 * max_patches, hidden]` 补齐张量和稠密 attention mask，patch encoder 也会先 pad_sequence。可将 DiT 改为 packed，但必须保持 prefix causal、生成 chunk 只看此前 clean prefix 与自己的整个 noisy chunk、样本隔离、每句时间/说话人条件及 clean/noisy 共享 rotary 位置。普通 causal FA2 不能直接替换此 mask。建议评估 packed + FlexAttention 自定义 block mask；避免构造全局 packed 长度平方的稠密 mask。实施前需做输出/梯度等价与真实 A800、FSDP/checkpoint 显存和吞吐验证；当前仅完成可行性核查，尚未修改 DiT 执行方式。


### Packed DiT 实现与验证（11-10）

已将 `DotsTTSModel.velocity()` 改为按每句真实长度拼接 clean prefix / noisy generation chunks，CUDA attention 使用编译的 FlexAttention。`packed.py` 直接构造块级候选索引，精确 mask 保持 prefix causal、生成 chunk 只看此前 clean prefix 与自身整个 noisy chunk，并隔离不同句子；没有构造全局 token 数平方的稠密 mask。时间/说话人 AdaLN 仍先按句计算，再映射到对应 token；clean/noisy 共享原有 rotary 位置。权重参数名称和形状保持不变。Patch encoder 本次仍使用原来的 padding 路径。

验证结果：

- CPU 与 A800 FP32 对照原 padded 实现，输出、输入梯度、参数梯度通过；分别覆盖开启/关闭 activation checkpointing，跨 128-token tile 边界及多个句子共享 tile。
- 双卡 FSDP + bf16 + 冻结 DiT + checkpoint，对照非分片模型输出和平均梯度通过；连续两步改变每卡句长和样本数。
- A800 两层真实宽度（1024、16 heads）、bf16、冻结 head、checkpoint，32 句（31×35 + 343 patches）前向+反向：padded 6.256 GiB / 0.691 s，packed 0.850 GiB / 0.168 s。计时为首轮预热后两次测量的均值（首轮编译不计入）；这仅为 DiT 局部测试，不能直接换算整机训练收益。BF16 输出相对 L2 差 0.00323，hidden 梯度相对 L2 差 0.00610。

验证脚本和日志在 `artifacts/dots-packed-dit-20261110/`；回归测试为 `tests/test_dots_packed.py`，旧实现只保留在 `tests/dots_padded_reference.py` 作为测试 oracle；分布式验证为 `tests/check_dots_packed_distributed.py`。正在运行的 24k 任务使用其提交时的代码快照，不会自动热更新。

完整 head 压测：精确重放此前 28k rank 3 step 38 的长度分布（292 句、12,231 真实 patches、最长 343），使用随机初始化的完整 18 层 / width 1024 / 16 heads DiT、bf16、冻结 head、checkpoint，连续三轮前向+反向成功。峰值 10.845 GiB，预热后两轮平均 9.680 s。这里测量的是独立 DiT，不包含 LLM、patch encoder、优化器和 FSDP 通信，也不是原 checkpoint 的 loss 重放。结果见 `artifacts/dots-packed-dit-20261110/full-head-failure-batch.json`。

最终相关回归：32 passed、2 个 CUDA 用例在受限 CPU 环境跳过；同一 CUDA 测试文件已另在 A800 运行，4 passed。双卡分布式测试两步均 PASS。Ruff 和 diff whitespace 检查通过。24k 旧快照任务检查时已完成 step 43，本轮未重启或替换该任务。


### 32k packed DiT 正式启动（2026-11-11）

按用户要求，新建 `configs/dots-tts-20261111-flexivoice-interleave-feistel-32k-packeddit.yaml` 并提交任务 `pt-ng26vb3i`，平台已进入 RUNNING。运行目录为 `dots-tts-20261111-flexivoice-interleave-instrdrop0-feistel-32k-packeddit-bf16-8gpu-lr1e-5-ep1-wsd`。使用已验证的 packed DiT 代码快照（model.py/modules.py/packed.py 逐字节核对一致），8×A800，32,000 tokens / 20,000 patches，accumulation 1，全 interleave，instruction_dropout 0，Feistel sampler，冻结 head，bf16，lr=1e-5，1 epoch WSD。复用发布 build，从原 assembled release 初始化，未续训旧 24k。

旧 24k `pt-0pxwktnv` 检查时已 FAILED：最后完成 step 43，随后 OOM 申请 6.49 GiB、仅剩 1.80 GiB。已保存配置、metrics 和节点日志到 `artifacts/flexivoice-feistel-32k-packeddit-launch-20261111/previous-24k-logs/`；删除 ACP 任务后 GET 为 HTTP 404，整个旧运行目录已删除。原始基线任务和 checkpoints 未动。

新任务提交及状态记录：`artifacts/flexivoice-feistel-32k-packeddit-launch-20261111/`。

启动验证完成：新任务已完成 step 5，`checkpoints/step-00000005/COMPLETE` 存在。step 2–5 每步 10.95–11.17 秒，记录的峰值显存 24.08–24.39 GiB；loss、梯度及各数值均有限，skipped/discarded samples 为 0。报告为 `startup-validation.json`。这些是初期批次表现，仍需留意后续 patch encoder padding 引起的长句批次峰值。


### 32k packed DiT step 525 试听（2026-11-11）

用户请求测试最新 checkpoint；开始检查时最新完整 checkpoint 为 `step-00000525`，固定该版本导出 bf16 到 `artifacts/flexivoice-packed32k-listen-step525/model/`。在当前开发节点 GPU 5 生成，未停止训练。中英文各默认、开心、悲伤、耳语四段，加 release 中英文 interleave 对照两段；每种语言固定相同正文、Seed-TTS 参考、seed=42、10 ODE steps、CFG=1.2、speaker_scale=1.0，x-vector-only。使用完整 instruction 前置、仅正文与音频 interleave 的模板；每条生成前将 runtime schedule 与训练 Layout.sequence 精确核对，通过后才推理。

10 段生成 WAV 加 2 段参考均通过回读、非空、有限值检查，生成时长 4.48–9.76 秒、48 kHz。产物为 `artifacts/flexivoice-packed32k-listen-step525/index.html`、`listen-step525.zip`；`samples.json` 记录文本、指令和采样参数，`audio-validation.json` 记录时长、峰值、RMS 与极小量饱和采样点比例。本次为试听和推理完整性检查，未做 ASR/情绪识别或主观效果结论。


### 48k / lr=5e-5 新实验（2026-11-11）

按用户要求提高 token budget 并设 lr=5e-5；本次采用单卡 48,000 tokens / 30,000 patches，相比 32k 增大 50%，accumulation=1。新配置 `configs/dots-tts-20261111-flexivoice-interleave-feistel-48k-packeddit-lr5e-5.yaml`，任务 `pt-jt0v0he9`，运行目录 `dots-tts-20261111-flexivoice-interleave-instrdrop0-feistel-48k-packeddit-bf16-8gpu-lr5e-5-ep1-wsd`。保持 packed DiT、全 interleave、instruction dropout 0、Feistel、冻结 head、8 卡 bf16、1 epoch WSD。从同一 assembled release 初始化，复用原 build，不从前一实验续训。

用户随后要求 checkpoint 每 200 步：`save_every=200`、`eval_every=200`、`first_check_step=null`，避免旧 step 5 或每 25 步评估触发额外 checkpoint；训练结束仍按 engine 正常保存最终 checkpoint。

旧 32k `pt-ng26vb3i` 已执行 stop，保留任务及整个运行目录；检查时有 23 个完整 checkpoints，latest 为 step 550。新提交、配置验证和状态记录在 `artifacts/flexivoice-feistel-48k-packeddit-lr5e-5-launch-20261111/`。


### 48k step400 扩展试听对比（2026-11-11）

固定测试请求开始时最新完整 checkpoint step400，与此前导出的 32k step525 对照；8 段正文（中英文各4段），每段默认加两种风格，共48段生成音频，另附4段参考。覆盖开心/悲伤、生气/耳语、快慢数字播报、长句。各对照固定参考、正文、指令、seed42、interleave、10 ODE steps、CFG1.2、speaker_scale1.0，并逐条验证推理 schedule 与训练 Layout 一致。

产物：`artifacts/flexivoice-48k-step400-expanded/index.html`、`listen-expanded.zip`（约33.4 MB）、`samples.json`、`audio-validation.json`。52个 WAV 全部可回读、非空且数值有限，没有样例达到40秒生成上限；未做ASR或主观质量评分。中文数字慢读：新模型默认8.96s/慢读10.40s，旧模型8.00s/8.00s；新模型英文慢读5.28s短于默认7.20s，需重点听指令跟随或漏读问题，不能仅凭时长定性。

应用户要求启动本机 HTTP 试听服务，`127.0.0.1:18765`（执行会话19992），首页、WAV和ZIP均通过HTTP200检查。通过端口转发访问；仅提供web目录内的试听文件，不提供模型权重。全接口绑定被自动审批拒绝，故使用localhost。


### 中文长句耳语开头瞬态核查（2026-11-11）

用户指出扩展试听“中文·长句·耳语”开头有“咔”声。新模型前100ms峰值约1e-4，约0.245s开始出现强瞬态，0.23–0.28s峰值0.9756；旧模型同一区域也有较强起始峰，但后续语音更强。波形见 `artifacts/flexivoice-48k-step400-expanded/onset-analysis/first-two-seconds.png`。

代码确认本次试听使用整段 latent 一次 VAE decode，未拼接 waveform chunks。固定seed42重新生成，和已保存PCM16原文件最大差3.05e-5（量化范围），复现成功。保持生成latent不变，声码器先解码25帧由数字静音编码得到的latent，再截取目标段：起始瞬态峰值仍0.9080，RMS约0.159（原0.1565）。因此不能用网页播放/流式拼接或单纯decoder冷启动解释；更指向生成首段信号，但尚未区分训练数据起始分布、首段预测、采样误差等原因，也未修复生产路径。

诊断脚本、原始latent、报告保存在 `onset-analysis/`；localhost:18765/onset/ 仅提供前2秒原音频、新旧对照、decoder预热对照和波形图。原试听音频未修改，未用淡入掩盖问题。


### 开头瞬态详细排查报告（2026-11-11）

已整理[详细排查报告](dots-flexivoice-onset-click-investigation-20261111.md)，包含 250ms 预处理算法、真实训练 features 解码、Flex/NV 与子语料统计、采样及参考条件 A/B、证据限制和后续验证方案。当前证据表明数据边界与推理参考条件均需考虑；尚未通过重新提取和重训证明移除处理能完全修复。当前未发布新 features 或提交两 epoch 训练。


### 开头咔声根因：KeSpeech 内嵌 CAF 头（2026-11-11）

上述排查中作为"强起音"例证的训练样本（`b128e87…`），开头并非语音：WAV 的 data 块以一个 4096 字节的 Apple CAF 文件头（`caff`/`desc`/`lpcm`/`free`/`data`）开头，被解码为约 128 ms 的满幅垃圾采样，开头与约 127 ms 处各有一下咔声，而且这是整条音频的峰值。250 ms 边缘静音归一化把它放到每条此类目标的 0.25 s，与生成音频的咔声位置一致。用户试听确认（`artifacts/flexivoice-48k-step400-expanded/onset-analysis/caff-header-listen/`：原样解码、训练视角、剥离头部三版对照）。

字节级全量扫描（tts-data-pipeline `scripts/audit_embedded_caf.py`）：FlexiVoice base 4,199,042 行中命中 326,006 行，全部来自 kespeech（约占其入选行的 1/3、Flex 入选行的 19.7%），头部均为 4096 字节；其他 14 个子集和 NV 五个数据集均为 0。之前按"首采样超阈值"统计的 VoxCeleb、debatts 命中是切得紧的正常语音，40 条抽样无一为垃圾头部。因此只去掉 250 ms 补零无法修复，咔声只会移到 0 s。旧 NV-ref3s 在 speaker_only 下约 0.245 s 的起音不能由此解释，NV 数据中没有 CAF 头，原因待查。

处理（不改 base、不改旧发布）：selection-flexivoice-v2 `tts-selection-supervised-tts-20261111T162703bjt-01` 继承 v1，按 103 排除这些音频，入选 1,330,267 行 / 2,214.4 小时；features `flexivoice-dots_vae-dots-tts-soar-vae-xvec-48k-fp32-v2-ref3s-20261111T164240bjt-01` 由全量特征复用发布（无 GPU），核对后剩余 CAF 样本为 0。发布登记见 tts-data-pipeline `docs/data-contract/PUBLICATIONS.md`。剥离头部后重新提取可找回这批数据，尚未执行。

### 去 CAF 头的 48k 两 epoch 实验（2026-11-11）

配置 `configs/dots-tts-20261111-flexivoice-nocaf-interleave-feistel-48k-packeddit-lr5e-5-ep2.yaml`：与 48k / lr=5e-5 packed DiT 实验相同，换为 build `dots-flexivoice-nocaf-selection-nv-ref3s-20261111`（数据配置 `data-flexivoice-nocaf-nv-dots-ref3s.yaml`），epochs=2，WSD 按整个 run 计算。build：FlexiVoice train 1,318,473 / validation 11,794，NV 不变，合计训练 1,546,251 行；预算与六路抽样张量校验通过（`artifacts/flexivoice-nocaf-launch-20261111/`）。

去掉 kespeech 说话人后按说话人 1% 留出的集合也随之改变（validation 37,985 → 11,794 条），val loss 不能与旧实验直接比较。

ACP 任务 `pt-ii8bqdpg`，2026-11-11 约 18:00 进入 RUNNING。每 epoch 约 396 步，共约 790 步，每步约 16.7 s（约 3.7 小时）。step200：val fm_loss 0.7542、eos_loss 0.0697（旧 48k 同步 0.7494 / 0.0734，验证集不同）；step 321 时训练正常，skipped/discarded 为 0。
