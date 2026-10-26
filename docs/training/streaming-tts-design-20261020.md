# 输入流式（streaming）TTS：现状、数据可行性与训练方案

日期：2026-10-20（PDT）。第一步已于 2026-10-22 实现（见第 9 节），尚未训练。数据统计的脚本和结果在 [evidence/streaming-tts-20261020/](evidence/streaming-tts-20261020/)。论文结论来自原文。凡是标注为"推测"的，都是基于这些资料的推断，尚未经过实验验证。

---

## 0. 结论摘要

1. **"streaming"指两件事。** 一是输出流式，边生成边播放；二是输入流式，文本边到边说，例如接在 LLM 后面。现有模型可以做输出流式：自回归逐帧生成，codec 支持分块解码。输入流式做不了，因为训练只用过 `qwen3_non_streaming`，即完整文本放在 codec 之前。
2. **官方 Qwen3-TTS 的输入流式不需要文本与音频对齐。** 它按固定节奏，每个 codec 位置叠加一个文本 token，文本放在前面，用完后补 `tts_pad`。预测第 j 帧时，模型已经看到前 j+1 个文本 token。对数据只有一个要求：文本要比语音快。
3. **现有数据满足这个要求。** 新 build 中，"文本 token 数 ÷ codec 帧数"的中位数为 0.30（emilia v4）和 0.39（emilia2），p99 为 0.51 和 0.70，大于 1 的样本分别只有 0.0001% 和 0.007%。也就是说，文本平均以语音 2.5–3 倍的速度推进。
4. **业界大致有四类做法**：固定 1:1 叠加、固定 N:M 交错、按词时间戳插 pad、显式时长或单调对齐。后两类需要词级对齐。证据显示：固定节奏的方法硬性要求文本始终快于语音；lookahead 存在一个最佳值；按对齐插 pad 的收益主要在韵律和实时同步，不一定降低 WER。
5. **建议分两步。** 第一步不需要对齐：沿用官方叠加方式，加入可配置的起始领先帧数和随机"文本停顿" pad 段，与 non-streaming 混合训练，并设对照组。第二步视结果决定：用 Qwen3-ForcedAligner 获取词级时间戳，改为 DSM-TTS 式的"真实时间轴 pad + lookahead"。

## 1. 官方两种输入格式

依据：`qwen_tts` 0.1.1 `core/models/modeling_qwen3_tts.py`（本仓库 `envs/lm-tts` 中的版本），generate 约 :2108–2232，`generate_icl_prompt` 约 :1968–2019，talker 逐步生成约 :1666、:1689–1692、:1741。下文的"+"表示在同一位置把 embedding 相加；`cpad` / `cbos` 是 codec pad / bos，`pad` 是文本侧的 `tts_pad`，t₁…t_N 是目标文本 token。

**non-streaming**（`non_streaming_mode=True`，本仓库当前训练用的就是这种）：

```
role | 控制+pad | speaker+pad | bos+cpad  t₁+cpad … t_N+cpad  eos+cpad | cbos+pad  C₀+pad  C₁+pad …
```

**streaming**（`non_streaming_mode=False`）：

```
role | 控制+pad | speaker+pad | cpad+bos | cbos+t₁  C₀+t₂  C₁+t₃ … C_{N-2}+t_N  C_{N-1}+eos  C_N+pad …
```

- 预填（prefill）阶段的最后一个位置是 `cbos + t₁`。其余文本 `[t₂…t_N, eos]` 放在 `trailing_text_hidden` 里。生成计数从 −1 开始，生成第 0 步（输入 C₀）时叠加 `trailing[0] = t₂`，依此类推，超出部分叠加 `tts_pad`。用本仓库的文本行 `[bos, t₁…t_N, eos]` 来表示：`bos` 放在前缀最后的 `cpad` 位上，`text_row[1:]` 逐位置叠加到 `[cbos, C₀, C₁, …]` 上。
- **带 ICL 时**：把"参考文本 + 目标文本 + eos"（不含 bos）按位置叠加到"cbos + 参考 codec"上。文本更长时，多出来的部分进入 trailing，接在生成的第一帧上；文本更短时，用 pad 补齐，trailing 为 `tts_pad`。所以整体规则可以概括为：cbos 之后第 p 个位置叠加文本的第 p 个 token，参考和目标的交界处不重置。
- 语言前缀（think、语言 ID）和 speaker 位在两种格式下完全相同。
- Opus partner 已按官方代码逐行核对了上述偏移。

**与本仓库训练方式的关系。** 我们训练时，参考片段就在同一句话内部，文本是整句。按 streaming 格式从 cbos 开始叠加整句文本，正好就是官方 ICL streaming 的排布。由于文本每帧推进 0.3–0.4 个 token，训练和推理时，大部分目标文本在生成开始之前就已经叠加在参考帧上了，两边是一致的。剩下的差异与 non-streaming 相同：训练时参考和目标是同一句话、一次分词；推理时是两句话、分别分词。参考片段在句中（uniform）还是在句首（prefix），不影响叠加的排布。

## 2. 现有数据是否可用

对新 build `supervised-tts-20261019-emilia-v4-emilia2-reference/train`，每个数据集均匀抽取 300 万条，计算"(文本 token 数 + 2) ÷ codec 帧数"，其中 +2 是 tts_bos 和 tts_eos。结果见 `text_frame_ratio.json`：

| 数据集 | 中位数 | p95 | p99 | > 0.5 | > 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| emilia（selection-v4） | 0.295 | 0.431 | 0.508 | 1.07% | 0.0001% |
| emilia2（selection-v2） | 0.387 | 0.591 | 0.704 | 14.7% | 0.007% |

按语言看，最密的是 emilia2 的 ko、fr、pl，p99 为 0.84–0.88，超过 1 的比例为 0.07–0.21%；zh、en 的 p99 为 0.46–0.65。

结论与限制：

- **不需要逐词对齐，也不需要重新提取特征。** streaming 格式的节奏是固定的，第 j 帧已经看到约 j+1 个 token，只要求文本领先。比值小于 1 说明整句平均来看文本领先，领先量约为 (1 − 比值)·帧数，并随时间线性增长。
- **这只是整句的平均。** 句首几帧，以及局部语速很快的片段，可能领先量接近 0。没有词级时间戳就无法逐帧验证。大多数句子开头有静音可以缓冲（推测）。
- **文本放不下的样本**（`len(text_ids) > frames + 2`，即 N+1 个文本位置多于 T+1 个音频位置）只能走 non-streaming，约占万分之一。

## 3. 业界做法调研

| 类别 | 代表 | 排布 | 对齐 |
| --- | --- | --- | --- |
| A. 固定 1:1 叠加 + 文本前置 + pad | Qwen3-TTS（官方代码）、Kimi-Audio（语义流前加 6 个 blank 作为延迟） | 每帧一个文本 token，文本用完补 pad | 不需要 |
| B. 固定 N:M 块交错 | CosyVoice 2（5:15）、IST-LM（1:3 最好）、StreamMel（音素:mel 1:4 最好）、GLM-4-Voice（13:26）、LLaMA-Omni2（3:10 最好） | 文本块和语音块轮流排列 | 不需要 |
| C. 按词时间戳插 pad | Moshi、Kyutai DSM-TTS、MiniCPM-o 4.5 TAIL | 每个词放在 floor(起始时间 × 12.5) 帧，其余为 pad；音频相对文本延迟 τ；模型预测何时取下一个词 | 需要词时间戳 |
| D. 显式时长 / 单调对齐 | SyncSpeech、VoXtream、SpeakStream、S5-TTS、SMLLE、Boundary-Aware | 每个文本单元生成一段，加少量 lookahead | 大多需要（MFA、WhisperX）；S5-TTS、SMLLE 在模型内部学对齐 |

关键证据（数字来自各论文的表格）：

- **固定节奏下，文本必须快于语音。** StreamMel 在 1:4 时 WER-C 为 2.10、SIM-O 为 0.622，非流式为 1.95；到 1:5，语音追上文本，WER-C 升到 8.44。GLM-4-Voice 选择 13:26，理由也是"保证文本总是领先"。
- **lookahead 存在最佳值。** DSM-TTS 去掉 2 个词的 lookahead 后，英文 WER 从 1.60% 升到 3.51%。SpeakStream 窗口 m=1…6 的 WER 依次为 7.47 / 4.50 / 3.99 / 3.88 / 3.38 / 4.26，全文本为 3.28。S5-TTS 在 k=1/2/3 时为 3.03 / 3.49 / 7.27，非流式为 3.20。IST-LM 观察到，文本与对应语音的平均距离增大时，WER 先降后升。
- **固定比例在短句上接近非流式，难句会掉点。** CosyVoice 2 从 offline 到 streaming：SEED zh CER 1.45 → 1.45，en WER 2.57 → 2.38，hard 6.83 → 8.08，ja 18.79 → 21.41，ko 7.98 → 9.06。IST-LM 1:3 为 3.60，非流式为 3.35。CosyVoice 2 同时训练流式和非流式序列，一个模型支持两种模式。
- **长文本风险。** 有一篇基于 CosyVoice 2 的工作（Prosodic Boundary-Aware）报告，原生 5:15 bistream 在长文本上 WER 达到 70.97%，作者归因于上下文无界增长，以及每个文本 token 对应的语音长度不固定造成漂移。这只是单篇论文的结果。
- **对齐的收益主要在韵律和实时同步。** DSM-TTS 中，"按时间戳的自然 pad"的 SIM 为 0.743，几种"固定 padding"为 0.691–0.715，作者归因于韵律不自然。但 MiniCPM-o 4.5 的 Seed 评测中，固定比例交错（ZH CER 0.86 / EN WER 2.38）优于按时间对齐的动态交错 TAIL（1.04 / 3.93），论文说明 TAIL 是为全双工时序设计的。
- **Qwen3-TTS 技术报告**只写了双轨、在通道维度拼接、因果的 12Hz codec（每包 4 个 token，320ms）、首包延迟 97ms（0.6B），没有写 pad 处理、长度不匹配，也没有流式与非流式的对比。上文的"每帧一个文本 token"来自官方代码。

来源：CosyVoice 2 [2412.10117](https://arxiv.org/html/2412.10117) · IST-LM [2412.16102](https://arxiv.org/html/2412.16102) · StreamMel [2506.12570](https://arxiv.org/html/2506.12570) · DSM-TTS [2509.08753](https://arxiv.org/html/2509.08753) · Moshi [2410.00037](https://arxiv.org/html/2410.00037) · SyncSpeech [2502.11094](https://arxiv.org/html/2502.11094) · SpeakStream [2505.19206](https://arxiv.org/html/2505.19206) · LiveSpeech 2 / SMLLE [2505.19669](https://arxiv.org/html/2505.19669) · S5-TTS [2606.21882](https://arxiv.org/html/2606.21882) · Boundary-Aware [2603.06444](https://arxiv.org/html/2603.06444) · VoXtream [2509.15969](https://arxiv.org/html/2509.15969) · MiniCPM-o 4.5 [2604.27393](https://arxiv.org/html/2604.27393v1) · LLaMA-Omni2 [2505.02625](https://arxiv.org/html/2505.02625) · GLM-4-Voice [2412.02612](https://arxiv.org/html/2412.02612) · Kimi-Audio [2504.18425](https://arxiv.org/html/2504.18425) · Qwen3-TTS [2601.15621](https://arxiv.org/html/2601.15621) · Qwen2.5-Omni [2503.20215](https://arxiv.org/html/2503.20215) · Qwen3-Omni [2509.17765](https://arxiv.org/html/2509.17765)

## 4. 文本速度波动与 pad 的问题

"文本速度波动"包含两种情况，需要分开处理：

1. **语速变化**，即每个文本 token 对应的语音时长不同。在 A 类叠加下，文本领先量随时间线性增长，语速变化只会改变领先多少，不会让文本落后，除非局部比值接近 1。风险集中在句首和高比值样本上。
2. **推理时文本到得比 12.5 token/s 慢**，例如 LLM 卡顿或者首 token 慢。这时文本轨上会出现训练中从未见过的 pad 段，属于分布不匹配。一般 LLM 的输出速度在 30 token/s 以上，大多数时候能跟上，但开头和偶发卡顿无法避免（推测）。

对应的处理办法：针对情况 1，加起始领先帧数 K；针对情况 2，训练时随机插入停顿 pad。两者都不需要对齐。如果要从根本上让文本落在真实时间轴上，就需要 C 类方案和词级时间戳。

## 5. 建议方案

### 第一步：不需要对齐（先做）

1. **排布**：沿用官方每帧一个 token 的叠加，与官方 `generate(non_streaming_mode=False)` 逐位置一致，因此可以直接用官方推理。
2. **起始领先 K 帧**：参考 Kimi-Audio 的 6 帧延迟，训练时让文本在 cbos 之前多领先 K 个位置，K 在 0–8 中随机。推理时等到积累 K 个文本 token 再开始生成，也就是牺牲一点首包延迟，换取句首的 lookahead。注意：K > 0 时，训练格式已经偏离官方，需要配合本仓库自己的推理入口，或者在官方 generate 外面加一层（推测）。
3. **随机停顿 pad**：训练时以一定概率在文本轨中插入长度随机的 pad 段，模拟推理时文本供应中断。插入后要保证文本仍然领先，例如约束"(文本 + 插入的 pad) ÷ 帧数 ≤ 0.8"。没有词时间戳，只能按整句估计余量（推测）。
4. **混合训练**：每条样本按 `sha256("streaming-v1:" + target_id)` 决定是否走 streaming，salt 与 language-auto 的不同，所以两者独立，形成 4 种组合。`len(text_ids) > frames + 2` 的样本强制走 non-streaming。比例可选：两种模式都要部署时用 50%；streaming 只是次要目标时用 30%（Opus 的建议）。
5. **loss 与 refmask 不变**：要预测的目标仍然是 C₀… 和 EOS，mask 只作用在音频帧上。

### 第二步：基于对齐（视第一步结果决定）

触发条件：高比值分桶或长文本明显退化，或者需要真正"边说边接文本"的实时同步。

- **对齐来源**：**Qwen3-ForcedAligner-0.6B**（[2601.21337](https://arxiv.org/abs/2601.21337)），Apache 2.0 许可，基于 LLM 的非自回归时间戳预测模型，支持 11 种语言，覆盖我们除 nl、pl 外的全部语言。调研称它支持词或字级时间戳、最长 300 秒、分辨率 80ms（恰好是 1 个 12.5Hz 帧），这几点在摘要中没有出现，**待确认**。它和现有的 Qwen3-ASR 识别服务是不同的模型，需要另外部署。MOSS diarization 只提供说话人片段，不提供词时间戳。nl、pl 可以考虑 MMS/CTC 对齐。
- **排布**：参考 DSM-TTS，每个词放在 floor(起始时间 × 12.5) 帧，其余为 pad。文本提前 1–2 个词作为 lookahead（DSM-TTS 用 2 个，SpeakStream 3–4 个最好，S5-TTS 1–2 个）。可以再让模型预测"下一步是否取下一个词"（action stream），这样文本快慢都能处理。
- **代价**：要给 1.1 亿条做对齐，需要先在子集上验证对齐质量和收益；排布与官方格式不兼容，需要自有的推理入口。

**不建议**改成固定 N:M 块交错：相对 A 类叠加，没有看到明确的优势，有长文本漂移的报告，而且与官方格式不兼容。

## 6. 实现清单（第一步，按 Opus 评审补全）

- `model.input_layout: non_streaming | mixed`，默认 `non_streaming`，旧 run 的 signature 不变；配合 `train.streaming_fraction`（以及之后的 K、停顿 pad 参数），全部纳入 signature。
- `protocol.build_inputs`：逐样本分支。streaming 时 `bos` 放在 cpad 位，`text_row[1:]` 叠加到 `[cbos, C₀, …]`，之后补 pad；逐样本计算 `prefix_length` 和 `audio_positions`。模型配置为 mixed 时，batch 中必须带逐样本格式标记，缺失就报错。
- **预算**：non-streaming 的长度是上限，sampler 和 `validate_budgets` 不用改。streaming 占 50% 时，batch 实际 token 约少 15%，所以每步耗时不能直接与基线比较。
- **telemetry**：`engine.py` 按"文本 + 帧 + 前缀"统计 token，对 streaming 样本会偏大，要改为按实际长度统计。
- **验证**：使用同一条哈希规则，并按格式分开记录 val loss。
- **导出**：在 `talker_config` 中记录 `lm_tts_input_layout`。
- **评测**：UltraEval `voice_clone_infer.py` 已有 `--non-streaming`，`run_seed_tts.py` 的某条路径写死了 `--non-streaming --language-auto`，需要改成可以选择；对没有训练过 streaming 的导出模型，拒绝 streaming 评测。
- **测试**：与官方 `generate(non_streaming_mode=False)` 逐位置比对，覆盖：x-vector 下 history 为 0、1、3 帧（cbos 位置只看到 t₁）；ICL 下文本短于和长于参考 codec 两种情况；刚好放下的边界（`len(text_ids) == frames + 2`）；显式语言和 Auto 两种前缀。

## 7. 实验计划

1. **继续训练短实验（含对照组）**：从 pt-jnjlqzyb（emilia v4 + emilia2 + codebook init + language ID）一个**未进入衰减段**的 checkpoint 出发，用 `init_from` 跑两组：`streaming_fraction = 0.5` 和 `= 0`，步数和 lr 相同。对照组用来区分"继续训练本身"和"streaming"各自的影响。不要从已衰减的最终 checkpoint 用峰值 lr 继续训，那样会抵消衰减段的效果。
2. **评测**：SeedTTS 分别跑 streaming 和 non-streaming（显式语言加 Auto 子集）；按"文本 ÷ 帧数"分桶看 WER 和 SIM；non-streaming 与对照组比较，确认没有退化；另外测首包延迟（文本逐 token 输入，codec 分块解码）。
3. **决策**：streaming 的结果接近 non-streaming 且没有退化，就把混合格式放进下一轮从头训练；高比值分桶或长文本退化，就进入第二步，做对齐。

## 8. 风险与未决事项

- 句首信息少：开始说话时只看到 1–2 个文本 token。K 帧领先可以缓解，但会增加首包延迟（推测）。
- 混合训练可能让 non-streaming 略有退化。CosyVoice 2 报告 hard 集退化了，需要对照组来量化。
- 高比值语言（ko、fr、pl）的 streaming 风险更大，要单独分桶看。
- K > 0 和停顿 pad 偏离官方格式，需要自有推理入口。
- Qwen3-ForcedAligner 的分辨率、最长时长和语言覆盖细节，待读正文和模型卡确认。

## 9. 第一步的实现（2026-10-22）

与第 5、6 节的计划相比，有三处改动：配置放在 `train` 下，而不是 `model` 下；暂不实现起始领先 K 帧；随机停顿 pad 默认开启，插入位置加了约束。

**配置**（只在设置了 `train.streaming_fraction` 时生效；不设置时，签名、数据和 embedding 与之前逐位一致）：

| 键 | 默认 | 含义 |
| --- | --- | --- |
| `streaming_fraction` | 不设置 | 满足领先条件的样本中，走 streaming 的比例。参考 CosyVoice 2/3 的训练代码（`random.random() < 0.5`），实验用 0.5 |
| `streaming_max_text_ratio` | 0.67 | 只有 (文本 token + 2) ÷ 帧数 ≤ 0.67 的样本才可能走 streaming。CosyVoice 用 5:15、25 Hz，即文本 ≤ 8.3 token/s，换算到 12.5 Hz 为 0.67 |
| `streaming_pause_probability` | 0.3 | streaming 样本中插入一段停顿的比例 |
| `streaming_max_pause_frames` | 12 | 停顿长度在 1–12 帧（0.08–0.96 s）之间均匀抽取 |

- **为什么放在 `train` 下**：`init_from` 要求 `model` 区块完全一致。放在 `train` 下，才能从当前 run 稳定期的 checkpoint 续训。
- **逐样本决定**：用 `sha256("streaming-v1:" + target_id)` 的前 8 字节决定是否走 streaming，后面的字节决定停顿。和 language-auto 用的哈希相互独立，也不依赖 seed，续训、resume、worker 数和验证集上的结果都一致。不满足领先条件的样本强制走 non-streaming。按真实 build 各抽 30 万条估计：emilia v4 有 99.97% 满足，emilia2 有 98.5% 满足。
- **排布**：`protocol.build_inputs` 逐行判断。streaming 行的前缀只保留 `bos+cpad`，`text_ids[1:]` 逐位置叠加到 `[cbos, C0, …]` 上，之后补 `tts_pad`，与官方 `generate(non_streaming_mode=False)` 逐位置一致。refmask 和 loss 不变：两种排布下，音频位置都是 cbos 加 T 帧，预测目标一一对应。
- **停顿的约束**：停顿只推迟文本，录音不会因此停下来。语速均匀时，第 k 个 token 大约在第 k/r 帧才说到，但在第 k+L 帧才到达（r 为每帧的文本 token 数）。所以插入位置 j 必须满足 j ≥ 1.5·L·r/(1−r)，系数 1.5 用来覆盖局部语速偏快的情况。放不下时，停顿缩短或取消；插入后仍要满足文本 ÷ 帧数 ≤ 0.67。估计结果：emilia v4 的 streaming 样本都能放下 12 帧；emilia2 有 98.7% 能放下停顿，短句的停顿会被缩短（p10 为 4 帧）。这只能教会模型“文本晚到时，靠已到手的文本继续说”。“已到手的文本都说完了就停下来等”需要真实停顿的音频，属于第二步。
- **验证**：主指标强制走 non-streaming，可与以往 run 直接比较。另外用同一批样本强制走 streaming（只对满足领先条件的样本），记为 `val/streaming_{loss,first_ce,residual_ce}`。
  - **2026-10-25 起改为与训练一致**：每条样本保持训练时的排布，只跑一遍。`val/{loss,first_ce,residual_ce}` 是全部样本按 token 合并，与 `train/*` 同一算法；同一遍中按排布拆出 `val/streaming_*`（哈希选中的 streaming 样本）和 `val/non_streaming_*`（其余样本）。训练日志同样拆出 `train/streaming_*` 和 `train/non_streaming_*`。三者按 token 精确合并，没有人为权重。之前 s37,500 续训记录的 `val/streaming_*` 是强制 streaming 的结果，`val/*` 是强制 non-streaming 的结果，与新口径不能直接比较。新口径下 `val/non_streaming_*` 不含被选中的 streaming 样本，也不等于主 run 的全样本 `val/*`。
- **其他**：telemetry 按实际序列长度统计 talker token。导出时在 `talker_config` 中写入 `lm_tts_input_layout: mixed`。
- **测试**：`tests/test_qwen_protocol.py` 与官方 streaming 的 prefill 加逐步 trailing 文本逐位置比对，覆盖 x-vector 和 ICL（history 为 0、1、3 帧，文本刚好放下的边界），以及显式语言；mixed 批次中的 non-streaming 行逐位不变；文本放不下时报错；停顿 pad 的位置正确。`tests/test_merged_data.py` 检查 reader 的选择可复现、领先条件和停顿约束。`tests/test_config.py` 检查默认值和非法配置。端到端冒烟（tiny 模型）：non-streaming 源 run → `init_from` streaming 续训（refmask + 停顿）→ 两种验证 → 导出，已跑通。
- **实验配置**：`configs/supervised-tts-20261023-emilia-v4-emilia2-langid-s37500-streaming50-…-ep1-wsd.yaml`。从 pt-jnjlqzyb 的 step 37,500（约 2.0 epoch，稳定期）加载权重和 AdamW 状态，用 streaming 训练代替源 run 的第 3 个 epoch：续训 1 个 epoch，学习率为峰值，最后 0.3 epoch 衰减到 0.1 倍，不 warmup。这样总数据量和 LR 曲线都与源 run 相同，源 run 自己的最终模型就是 non-streaming 对照组，不再单独跑 control。seed 改为 43，以得到新的 epoch 顺序。执行顺序（2026-10-22 决定）：37.5k 写完、验证完成后，先停掉源 run，用它的 2 个节点训练 streaming；streaming 训完后，源 run 从最新的 checkpoint 恢复，训完剩下的部分。
- **尚未完成**：UltraEval 的 streaming 评测入口（`run_seed_tts.py` 目前写死 `--non-streaming`），以及对没训过 streaming 的导出模型拒绝 streaming 评测。

## 10. 第一次 streaming 续训的结果与诊断（2026-10-24）

**运行经过**：pt-ywdi9jb9 从 10-23 11:50 开始，训到约 5,700 步。10-24 00:09 为了先让源 run 训完，手动停掉了它，最新 checkpoint 是 step 5000。源 run 以 pt-5sv2qeif 从 37,500 恢复。streaming 之后可以用 `--resume latest` 从 5000 继续。

**验证 CE（codebook 1）**：

| step | non-streaming | streaming |
| --- | --- | --- |
| 100 | 1.222 | 2.118 |
| 2500 | 1.220 | 1.933 |
| 5000 | 1.219 | 1.565 |

non-streaming 没有被拖累（源 run 37.5k 为 1.213）。streaming 在下降，但和 non-streaming 还差 0.35。

**推理冒烟测试**（SeedTTS en/zh 各 30 条，speaker+ICL，默认采样，官方 `generate(non_streaming_mode=False)`）：

| 模型 | en WER | zh CER |
| --- | --- | --- |
| 源 run 37.5k（没训过 streaming） | 213 | 113（大多超过 30 s 被排除） |
| streaming s2500 | 104 | 105 |
| streaming s5000 | 83 | 95 |
| streaming s5000，x-vector（不用 ICL） | 82 | 81 |
| s5000 non-streaming（对照） | 5.0 | 3.6 |

streaming 输出音色正确，但内容基本不对，目前不能用。

**实现已排除**：用真实的 s5000 模型和 6 条真实验证样本，训练代码 `build_inputs` 构造的 streaming 输入，与官方 prefill 加逐步 trailing 文本（直接对照官方 talker 解码循环 `trailing_text_hidden[:, generation_step]`）构造的输入，逐位置最大差异为 1e-7，teacher-forced CE 逐条相同。x-vector 推理同样不行，说明问题也不在 ICL。所以训练、验证和推理的排布是一致的，问题在模型学到的东西。

**原因（强相关，尚未用对照实验证实）：文本信号相对音频帧太弱。** streaming 位置的输入是「16 码本帧 embedding 之和 + 投影后的文本向量」。在 150 条验证样本上测范数中位数：

| 模型 | 文本向量 | 帧 embedding（16 码本之和） | 文本 ÷ 帧 |
| --- | --- | --- | --- |
| 官方 Qwen3-TTS-12Hz-0.6B-Base | 1.59 | 2.24 | 0.71 |
| 初始化（assembled，codebook init） | 1.59 | 2.53 | 0.63 |
| 源 run 37.5k | 1.59 | 7.41 | 0.22 |
| streaming s5000 | 1.59 | 7.60 | 0.21 |

文本前端是冻结的（`lm_tts_freeze_text_frontend: true`），文本向量一直不变。codec embedding 在 37.5k 步 non-streaming 训练中涨了约 3 倍（codebook 1 从 0.53 涨到 2.08）。non-streaming 不受影响，因为文本单独放在前缀位置，不和帧相加。streaming 下文本只占帧的约 1/5，被音频盖住了。5000 步 streaming 训练中这个比例没有恢复（0.215 → 0.209）。这和看到的现象吻合：CE 降得慢，推理音色对、内容错。

**下一步（待定，未实施）**：
1. 从 streaming s5000 续训 1–2k 步，解冻 `text_projection`（官方推理直接用它，导出后不需要改推理代码），看 streaming CE 是否明显下降得更快，以此验证上面的原因。
2. 以后从头训练时，限制 codec embedding 范数的增长（例如对 embedding 加 weight decay 或加范数上限），让文本 ÷ 帧保持在官方水平附近。
3. UltraEval 的 streaming 入口：`voice_clone_infer.py` 不传 `--non-streaming` 时已经走官方 streaming。还缺 `run_seed_tts.py` 的开关，以及审计中「必须是 non-streaming」检查的放宽。
