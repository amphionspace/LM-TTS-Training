# 输入流式（streaming）TTS：现状、数据可行性与训练方案

日期：2026-10-20（PDT）。第一步已于 2026-10-22 实现（见第 9 节）。2026-10-26 改为交错排布，见第 11 节。数据统计的脚本和结果在 [evidence/streaming-tts-20261020/](evidence/streaming-tts-20261020/)。论文结论来自原文。凡是标注为"推测"的，都是基于这些资料的推断，尚未经过实验验证。

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

**不建议**改成固定 N:M 块交错：相对 A 类叠加，没有看到明确的优势，有长文本漂移的报告，而且与官方格式不兼容。（2026-10-26 已推翻，改为 5:10 交错，见第 11 节。）

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

## 11. 从头训练的 overlay streaming、范数诊断，以及改为交错排布（2026-10-26）

脚本和结果在 [evidence/streaming-interleave-20261026/](evidence/streaming-interleave-20261026/)。脚本使用本机绝对路径，只读 checkpoint 和导出模型。

### 11.1 streaming50：从头训练 overlay streaming

pt-isfahe7o（`configs/supervised-tts-20261025-…-streaming50-…`）沿用语言 ID run 的配方，只多了 `streaming_fraction: 0.5`，并把 `decay_ratio` 改为 0.2。它从 10-25 22:41 训到约 10k 步，10-26 20:4x 停止，换成 11.4 节的交错 run。

Seed-TTS（speaker + ICL，Auto，默认采样，WER/CER 与 SIM，单位 %）：

| step | 新 run non-streaming | 新 run streaming（官方 overlay 推理） | 主 run（只训 non-streaming） |
| --- | --- | --- | --- |
| 2.5k | 27.1 / 34.5 · 33.9 / 53.2 | 92.6 / 32.4 · 89.5 / 52.0 | 9.8 / 36.8 · 15.1 / 53.1 |
| 5k | 10.7 / 45.2 · 14.3 / 60.7 | 33.6 / 43.4 · 41.2 / 60.5 | 6.7 / 46.7 · 8.9 / 61.5 |
| 7.5k | 6.3 / 51.6 · 7.6 / 64.7 | 19.1 / 50.1 · 22.4 / 64.2 | 5.2 / 52.4 · 5.5 / 65.1 |

（每格依次为 en WER / en SIM · zh CER / zh SIM。）

val first_ce（streaming / non-streaming / 主 run）：2.5k 1.86 / 1.59 / 1.54；5k 1.55 / 1.44 / 1.42；7.5k 1.45 / 1.38 / 1.36。训练集上两种格式的 first_ce 差距在 1,250 步左右最大（0.68），之后一直在缩小（3,750 步时 0.15）。streaming 一直在变好，但到 7.5k 时 WER 仍是 non-streaming 的约 3 倍。non-streaming 起步明显慢于主 run，2.5k 时差距约 3 倍，到 7.5k 已基本追上。

### 11.2 排除了实现问题

- **UltraEval 新增 streaming 评测**：`run_seed_tts.py --streaming`，watcher 增加 `default_streaming` 一组，审计检查每条结果的格式。
- **推理代码没问题**：官方 0.6B 用同一套 streaming 推理，4 条冒烟样本的 WER/CER 为 0–4.5%。
- **训练和推理的输入一致**：x-vector streaming 下，训练 `build_inputs` 与官方 prefill 加逐步 trailing 构造的输入，49 个位置的差值全为 0（`layout_diff.py`）。
- **生成过程一致**：s37,500 续训的 5k 模型贪心生成后，再用训练格式 teacher forcing，每帧的 argmax 与生成结果 100% 一致（`greedy_consistency.py`）。所以 11.1 节的问题来自模型本身。

### 11.3 teacher-forced CE（Seed zh 60 句，`tf_ce.py`）

| 模型 | streaming | non-streaming | 换成另一句的文本后（streaming / non-streaming） |
| --- | --- | --- | --- |
| 官方 0.6B | 1.198 | 1.154 | 2.342 / 2.146 |
| s37,500 续训 5k | 1.446 | 1.238 | 1.813 / 1.660 |
| 主 run 最终 ckpt（没训过 streaming） | 2.226 | 1.202 | — |

官方模型两种格式只差 0.04，我们差 0.21。续训模型的 teacher-forced CE 看起来不差，但生成时 WER 达到 55–95%。CE 好看不代表生成可用。

### 11.4 codec 输入表的范数增长

**是什么**：每行的 L2 长度。第 0 层前有 RMSNorm，绝对大小本身意义不大，影响的是相对大小：一是 streaming 帧里文本和音频相加时的比例，二是输入向量和各层输出在残差流里的配比。

**主 run，第 1 层 codec 表**（`norm_growth.py`）：

| step | 100 | 2.5k | 10k | 20k | 30k | 40k | 50k | 56,384 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 行范数中位数 | 0.59 | 1.00 | 1.39 | 1.71 | 1.96 | 2.16 | 2.35 | 2.40 |
| 与最终 ckpt 的方向 cos | 0.54 | 0.82 | 0.91 | 0.96 | 0.98 | 0.99 | 1.00 | 1.00 |

- 10k 之后方向基本固定，长度仍在增长。
- 10k–40k 之间，范数² 每步约增长 9.2e-5，恰好等于 lr²·d = (3e-4)²·1024。
- 行范数与 codec 使用频率几乎无关（Spearman 0.12）。
- 后 15 张表：0.64 → 1.46。

**解释**：模型对输入表的整体尺度几乎不敏感。Adam 每步走固定大小的一步，方向基本与行向量垂直，按勾股定理每步都让长度变大。SGD 的步长会随梯度变小而变小，Adam 不会。weight decay 0.01 的平衡点约为 √(lr·d/2λ) ≈ 3.9，时间常数约 17 万步，所以在整个训练里一直在涨。Fable 补充了几点：
- 严格的尺度不变性并不成立。范数变大，会相对压低每层输出对残差流的修改。
- 漂移是在 16 张表的总和上与梯度垂直，单张表不一定，所以长度会在表之间转移，第 1 层表涨得比其他表快。

**文本 ÷ 音频帧比例**（`norm_balance.py`，150 条验证样本，冻结文本 1.59）：

| | 2.5k | 5k | 7.5k | 最终 |
| --- | --- | --- | --- | --- |
| 主 run | 0.51 | 0.43 | 0.39 | 0.19 |
| streaming50 | 0.53 | 0.49 | 0.46 | — |

官方是 0.71。streaming50 下降得比主 run 慢。两个 run 输出头的 R² 和范数几乎一样（`head_r2_streaming50_vs_main.jsonl`）。

**缩放探针**（`scale_probe.py`，streaming50 7.5k，验证集前 640 条，只缩放 16 张表的音频 code 行）：

| 缩放 | streaming first_ce | non-streaming first_ce | residual_ce |
| --- | --- | --- | --- |
| ×1 | 1.436 | 1.416 | 5.87 |
| ×0.5 | 1.810 | 1.666 | 6.14 |
| ×2 | 1.881 | 1.569 | 6.16 |

两个方向都变差，说明 backbone 已经适应了当前的尺度，导出时不能直接把表缩放到官方的量级。streaming 对缩放更敏感。把文本比例直接提到约 0.9（×0.5）也会变差。

**相关检查**：
- **near-identity 文本前端 run**（emilia2，可训练的 backbone 文本表，`nearid_norms.py`）：
  - 投影后的文本向量 0.92 → 1.31（6k），主要来自 projector（偏离初始化 15–31%）。文本表本身只涨了 4%。
  - tts_pad 0.63 → 1.07，涨得最快。
  - 文本和音频一起变长，比例稳定在 0.34–0.37，但起点很低。
- **语言 ID 行**（codec 表中，可训练）：
  - 大多从 0.63 涨到 1.6–2.4，方向变化很大。
  - **俄语行（2069）完全没被训练过**：cos 1.000，只因 weight decay 略微变短。训练数据里大概率没有带语言 ID 的俄语样本。

### 11.5 考虑过的方案

| 方案 | 结论 |
| --- | --- |
| 只给 codec 表加大 weight decay / 降低 lr | 钝刀：要 λ≥0.3 才压得住，降 lr 只是让增长变慢 |
| 每步把行投影回固定长度 | 目标长度是人为定的。官方的长度属于另一个模型；分叉点的长度也只是漂移到那里而已 |
| 行归一化 × 可学标量（weight norm / nGPT 式，A1） | 能止住漂移、让模型自己选尺度，导出时可以折叠回表里。保留为后续单变量实验 |
| DyT | 漂移会把输入推进 tanh 饱和区，梯度消失 |
| 只给文本加系数 / 解冻文本投影 | 只治比例，不治漂移；near-id run 显示漂移会搬到 projector 和 tts_pad 上 |
| 帧和后加 RMSNorm、concat、门控、fusion MLP | 要么管不住表之间的相对尺度，要么根本不解决漂移 |
| Muon | 输入表通常仍然用 AdamW；Muon 同样是定步长更新 |
| **交错排布（CosyVoice 2/3 bistream）** | **采用**：文本和音频永远不相加，比例问题从结构上消失 |

第 5 节曾写"不建议改成固定 N:M 块交错"，理由是看不到明确的优势，而且与官方格式不兼容。本节的结果推翻了这一判断：overlay 的比例问题是结构性的，而交错只需要自有的推理入口。

### 11.6 交错排布的实现（未 commit，提交训练时打包为快照）

参照 CosyVoice `cosyvoice/llm/llm.py`（`prepare_lm_input_target`、`inference_bistream`，CosyVoice3 的 `mix_ratio: [5, 15]`，25 Hz）。我们的帧率是 12.5 Hz，文本密度约为它的 2 倍，所以改为 5:10。

```
role | 控制+pad | speaker+pad | bos+cpad | t₁ … t₅ (+cpad) | F₁ … F₁₀ (+pad) | t₆ … t₁₀ | F₁₁ … F₂₀ | … | 余文本 eos | 余帧
```

- **排布**：文本保留自己的位置，跨流 pad 与 non-streaming 相同（文本 + cpad，帧 + pad）。没有 cbos，也没有 fill token；推理时每 10 帧固定切回文本。包含 tts_eos 的那一块文本之后，放入全部剩余帧。
- **预测位置**：每一帧都由它前一个 token 预测，最后一个位置预测 EOS。所以每行仍有"帧数 + 1"个预测位置，`tts_loss`、code predictor 和 refmask 都不用改。
- **门槛**：(文本 + 2) ÷ 帧数 ≤ 0.4，比 5:10 的速率留出 1.25 倍余量，并要求帧用完前 tts_eos 已经放下。在真实数据上，合格的约 67%，按原哈希抽一半后，约 33% 的样本走 streaming。不再插入停顿：上游卡顿时，推理只需要等待。
- **配置**：`streaming_layout: interleaved`、`streaming_chunk: [5, 10]`，`streaming_max_text_ratio` 必须显式给出。不写这两个键的 run 仍是 overlay，签名不变。导出时在 `talker_config` 中写入 `lm_tts_streaming_layout` 和 `lm_tts_streaming_chunk`。
- **推理**：`lm_tts/inference/codec.py` 每帧用 `build_inputs` 重算整行，没有 KV cache，所以生成的输入天然与训练一致。tts_eos 放下之前屏蔽 EOS；`penalize_reference=False` 时，重复惩罚只统计生成的帧，与官方一致。ICL 时，文本流是"参考文本 + 目标文本"，参考帧作为已经生成的前缀，从头到尾用同一个计数器排。
- **UltraEval**：`scripts/local_interleaved.py`。交错排布的导出模型在 `--streaming` 下走 lm_tts 解码，settings 记录 `decoder`、LM-TTS-Training 的 git 版本和 dirty 标记。推理前，用训练 checkpoint 的签名核对导出配置中的排布，避免旧导出悄悄退回官方 overlay。审计检查 decoder 与导出的排布一致。用临时导出（streaming50 7.5k 权重，标为交错）完整跑过一遍冒烟（生成、打分、审计 PASS）；这份权重没训过交错，WER 没有意义。
- **测试**：
  - 交错排布逐位置的向量和预测位置；
  - 任意帧数的生成前缀都是训练行的前缀（向量、position_ids、预测位置），这一项抓出过一个部分行 position_ids 的 bug；
  - teacher forcing 与逐帧生成的 logits 一致；
  - tts_eos 放下前屏蔽 EOS，语言 ID 和 streaming 标记都正确传入；
  - 配置校验；导出写入排布。
  
  全部 185 个测试通过。用真实数据和权重在 flash attention 下跑通了训练 forward 和按格式拆分的验证（`interleaved_stage0.py`）。Opus 审查没有发现训练侧的问题（另外对排布函数做了暴力枚举的性质检查），评测侧的建议已经按上面的方式处理。

### 11.7 新 run 与判断标准

pt-ycy9pp26（`configs/supervised-tts-20261027-…-interleaved5x10-…`，ACP 优先级 HIGHEST）于 10-26 20:47 启动。除排布外，与 streaming50 完全相同：同 init、数据、文本侧冻结、refmask、语言 ID、学习率和 decay。

- 跨 run 只比较 Seed WER/CER/SIM。streaming 的损失位置集合变了，val CE 只能在新 run 内部比较两种格式。新 run 的 streaming 样本约是 streaming50 的 0.67 倍，可以同时对照按 streaming 样本数对齐的点（新 run 7.5k ≈ streaming50 5k）。
- **成功**：7.5k 时 non-streaming ≤ 6.3 / 7.6（±0.5），streaming WER 不超过 non-streaming 的 1.3 倍。
- **失败**：non-streaming 退步超过 1 个点，或 streaming 差距仍然超过 1.5 倍。届时检查 ICL 的排布和局部语速违约率，再考虑门槛收紧到 0.33，或改为 4:8。
- **后续单变量实验**：A1（行归一化 × 可学标量）、refmask 以一半概率从句首截取、去掉跨流 pad、文本前端改用 backbone 词表。
- **未决**：交错评测每帧重算整行，复杂度 O(N²)，会比较慢；俄语语言 ID 没有被训练过；本节的代码尚未 commit。

## 12. 交错 run 的评测结果：stage-1 通过（2026-10-27）

pt-ycy9pp26 的 2.5k、5k、7.5k 都由自动评测跑完了 Seed-TTS speaker_icl 全量（3108 条，审计 PASS）。英文是 WER，中文是 CER，SIM 是 % 。

| run | 2.5k en / zh | 5k en / zh | 7.5k en / zh | 7.5k SIM en / zh |
|---|---|---|---|---|
| **interleaved5x10 streaming** | 14.19 / 17.31 | 6.79 / 8.90 | **5.33 / 4.96** | 53.9 / 66.0 |
| **interleaved5x10 non-streaming** | 16.12 / 16.27 | 7.21 / 8.52 | **4.93 / 4.84** | 53.8 / 66.0 |
| 主 run langid（只训 non-streaming） | 9.82 / 15.12 | 6.72 / 8.91 | 5.18 / 5.49 | 52.4 / 65.1 |
| streaming50 overlay non-streaming | 27.10 / 33.90 | 10.73 / 14.33 | 6.35 / 7.60 | 51.6 / 64.7 |
| streaming50 overlay streaming | 92.60 / 89.47 | 33.61 / 41.23 | 19.14 / 22.44 | 50.1 / 64.2 |

- **stage-1 两条标准都通过**：7.5k non-streaming 4.93 / 4.84（标准 ≤ 6.3 / 7.6 ± 0.5）；streaming / non-streaming 为 1.08 / 1.03（标准 ≤ 1.3）。
- streaming / non-streaming 的比值在 2.5k、5k、7.5k 分别是 0.88 / 1.06、0.94 / 1.04、1.08 / 1.03，从一开始就在 1 附近；overlay 一直在 3 倍左右。
- 加入 streaming 没有拖累 non-streaming：2.5k 落后主 run（2 条重复失控的样本把英文从约 11.5 拉到 16.1），5k 追平，7.5k 略好，SIM 高约 1。
- **解码器口径**：交错导出由 UltraEval `scripts/local_qwen.py` 用 lm_tts 解码（两种排布都是，带 KV cache）。在 streaming50 7.5k 上和官方 qwen_tts 对比 non-streaming：lm_tts 6.47 / 7.85，官方 6.35 / 7.60，SIM 相同，差异在采样波动内，所以表中跨 run 可以直接比。
- **ICL 的排布**：参考文本 + 目标文本是一条文本流，参考帧是已生成的前缀，整行按同一个 5:10 计数器交错，与训练（整句交错，refmask 区间在句内）一致。Seed-TTS 参考段的文本密度（(参考 token + 1) ÷ 参考帧数）：en 中位 0.27、最大 0.45，zh 中位 0.24、最大 0.39，全部低于 5:10 的 0.5，所以参考段里文本始终领先音频，没有训练中不存在的"音频跑在文本前面"的情况。

§11.6 之后的实现变化：生成加了 KV cache（`901acce`，与整行重算逐帧一致）；逐帧生成、KV cache 和采样从训练模型类移到 `lm_tts/inference/{frame,sampling}.py`（`2939898`），7.5k 导出在两种排布下 bf16 采样输出与移动前逐位相同；`local_interleaved.py` 改名为 `local_qwen.py`。

## 13. 参考段整段前置（reference lead）：推理侧尝试失败，分析与方案（2026-10-28）

### 13.1 为什么要改参考段的排法

现在 ICL 时，参考文本、目标文本和参考帧按同一个 5:10 计数器交错（§12）。这带来两个问题：

- **参考太密时文本跟不上。** 5:10 相当于 6.25 tok/s（每帧 0.5 token）。§2 的统计里，训练数据中英文中位数为 0.28–0.36，余量只有 1.4–1.8 倍；emilia2 的 ko、fr、pl、de 中位数为 0.44–0.47，超过 0.5 的占 23–39%。Seed-TTS 的参考最大 0.45，测不出问题，但用户给的参考可能超过 0.5，这时参考帧放完了，参考文本还没放完。CosyVoice 有人报告过同样的现象（[issue #967](https://github.com/FunAudioLLM/CosyVoice/issues/967)：`inference_bistream` 会把 prompt 的最后一句再说一遍，原因同样被归于固定的 5:15 比例；维护者没有回应，issue 已 stale）。
- **首包要等。** 交错时，第一帧之前要读入的目标文本 en 中位 13 个 token、zh 16 个（§35）。参考前置后第一帧之前要读入 L + 5 个，L 取得大时并不更快，所以这一条是次要理由，主要解决的是密参考。

直接的想法是推理时把参考文本整段放在参考帧之前，参考段不再交错。

### 13.2 实现

推理侧加了 lead 选项，训练不变：

- `interleaved_layout(..., lead=(k, f))`：第一个 chunk 放 k 个文本 token，接着放 f 帧，之后照常 5:10。
- `generate_codes(reference_lead=k)`：f 为参考帧数，k 为参考 token 数加 L 个目标 token；`CodecSynthesizer.synthesize(reference_lead=L)`。
- UltraEval：`run_seed_tts.py --streaming --reference-lead L`，审计检查 L。
- 提交：LM-TTS `04ce5d3`、`7b13d86`；UltraEval `63c0669`、`7a5aea3`。单测覆盖该排布，以及带 KV cache 与整行重算的生成结果一致。

### 13.3 结果：L = 0、5、10 都不行

用 interleaved5x10 的 12.5k ckpt 跑 Seed-TTS，smoke（en、zh 各 2 条）已能看清问题，三个全量 run 都按要求中途停止（分别生成了 178、72、42 条，都未打分）。中文 smoke 的 CER：

| 排布 | 中文 CER（2 条） | 典型错误 |
| --- | --- | --- |
| 交错 streaming（对照） | 4.5 / 9.1 | 个别字 |
| non-streaming（对照） | 4.5 / 4.5 | 个别字 |
| 参考前置 L=0 | 9.1 / 13.6 | 开头重复参考结尾：参考「…美白**针**」，输出「**真**可是对另外两台……」 |
| 参考前置 L=5 | 36.4 / 27.3 | 句中念错：「惠普把消费领域」→「离丧在与」，「音调失准」→「菜刀湿疹」 |
| 参考前置 L=10 | 31.8 /（未出分） | 句中跳字：整段「怪声怪气的琴」没念 |

英文 4 条只有 L=5 一处小错（「to」→「too.」），但 L=0 的平均时长 3.56 秒，长于交错的 3.04 秒和 non-streaming 的 2.84 秒，可能开头也多念了内容，没有被 ASR 转出来，未核实。

开头重复参考结尾的统计脚本（转写开头的 1–3 个单位等于参考结尾，中文按拼音比较）在交错 streaming 全量上为 en 0/1088、zh 0/2020，在 L=0 的 smoke 上检出 1/2。

### 13.4 分析：落在两种训练分布之间

| 参照 | 训练里见过的 | reference lead 给的 | 差别 |
| --- | --- | --- | --- |
| non-streaming 行 | 整句文本都在第一帧之前，之后不再插入文本 | 只有参考文本 + L 个目标 token，之后还按 5:10 继续插入 | **放少了** |
| 交错 streaming 行 | 每 10 帧插入 5 个 token，领先量从 5 开始慢慢累积 | 第一帧之前一次放 20 多个 token | **放多了** |

此外，streaming 行的前缀只有 bos，模型由此知道这是 streaming 行，紧接着却来了一大段文本，两个信号互相矛盾（推测）。

- L=0：参考帧放完的那一刻，已放入的文本正好全部说完，领先量为 0；随后才放入 5 个目标 token，第一个生成帧之前领先 5 个，而交错时同一位置一般领先 13–16 个。交错训练里没有这种状态，模型把参考结尾又接了一遍。
- L=5、10：边界处有领先，但领先量从一开始就超出交错训练的范围。推测模型靠新文本块插入的位置来跟踪念到哪儿了，领先量不对，对齐就乱了，于是跳字、念错。中文受影响更大，可能因为中文一个 token 约 1–2 个字，指针一偏就念错字（推测）。

所以问题出在推理时换了训练没见过的排布，与参考前置本身无关。overlay（Qwen 官方，每帧 1 个 token，12.5 tok/s）也一样：只要训推排布一致就不会出这个问题，推理时改排布也会坏。overlay 之所以从来没遇到参考文本跟不上，是因为它的速率是语速的 3 倍左右，文本比值超过 1 的样本只占 0.0x%；它自己的问题在 §11（embedding 范数）和对上游速率的要求（§3）。

结论：交错训练的模型不能只在推理时改参考段排布。参考前置必须在训练里覆盖。

### 13.5 可选方案（不用 CTC 强制对齐）

1. **随机化第一个 chunk 的领先量（只改 reader）。** 一部分 streaming 行的第一个 chunk 放 k 个 token（从 5 到大半句），配 f 帧，之后照常 5:10。k = 5 就是现在的交错，k = 全句等价于 non-streaming 的放法，中间正好补上 §13.4 的空白。约束是 f 帧内音频不能追上文本：按整句平均密度估计，取 k ≥ 1.5 × f × 密度，与 §9 停顿插入的思路相同，不需要对齐。`interleaved_layout` 已支持 lead，reader 里抽 k、f 即可。推理用 L = 5 或 10 的 reference lead，使边界处文本仍然领先。局限：训练中领先的那段不一定恰好是参考文本，能否完全补上要实验验证。顺带也让模型对上游文本时快时慢更鲁棒。
2. **同来源相邻片段组成参考 + 目标。** Emilia 本来就是按 VAD/ASR 切好的、各自带转写的片段。selection-v4 里同一来源、MOSS 判为单说话人的相邻两段，前一段做参考，后一段做目标，排布为「参考文本整段 → 参考帧（不算 loss）→ 目标 5:10」，与推理完全一致，参考密度也不再受 5:10 限制。同时解决 refmask 以来一直存在的句内参考与跨句参考的训推差异（`all16-training-and-seedtts-evaluation.md` §13），也与服务时用户任意给参考的场景一致。需要确认片段 ID 能否按来源排序，并处理相邻段说话人不一致的情况。
3. **段内时间戳。** MOSS-Transcribe-Diarize 的输出可能带分句时间戳和对应文本，如果有，句内切分点是现成的。未核实。
4. **停顿切分 + ASR 校验。** 在句内静音处切开，文本按标点切开，对参考那半段跑一遍 ASR，与切出的文本前缀比对，对不上就丢弃。ASR 只做校验，不做精确对齐；需要多跑一遍 ASR，召回率取决于停顿与标点是否对应。

建议先做方案 1：改动最小，可以从当前 ckpt 续训约 2.5k 步，验证 reference lead L = 5/10 能否回到交错 streaming 的水平。不够再做方案 2，它是最干净的长期方案。方案 3 先花几分钟看 MOSS 输出的格式。

在训练覆盖之前，streaming 继续用交错排布；服务时先算参考密度，超过约 0.45 就回退到 non-streaming。
