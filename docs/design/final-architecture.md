# 当前基线：官方结构、冻结公开文本前端和 speaker encoder

## 决定与验证顺序

先尽量对齐 Qwen3-TTS-12Hz-0.6B 的公开结构，在 **Emilia2 英语、中文各约 500 小时 short 数据**上进行预训练，再讨论删掉 projector、解冻或重新初始化。LJSpeech 上的直接连接实验仅是诊断，不作为最终选择的依据。

官方完整结构、0.6B / 1.7B 维度对照及尚未公开的初始化细节见 [官方结构与证据](official-qwen3-tts.md)。数据读取、模型输入、loss 和四卡更新详见 [完整预训练 pipeline](../training/pretraining-pipeline.md)。

## 每个模块从哪里来，是否训练

| 模块 | 当前结构 | 初始化来源 | 当前是否训练 |
|---|---|---|---|
| Text tokenizer | 官方 TTS 词表，151676 项 | Qwen3-TTS-12Hz-0.6B-Base | 不适用 |
| Text embedding | `[151936,2048]` | 同一公开 TTS checkpoint 的完整表，包含已训练 TTS token 行 | **冻结** |
| Text projector | `2048→2048→1024`，SiLU，两层都有 bias | 与 embedding 配套的公开 TTS 权重 | **冻结** |
| Talker decoder / final norm | 28 层，hidden 1024，FFN 3072，16 Q / 8 KV heads，head_dim 128 | **Qwen3-0.6B-Base** 的 layers / norm | 训练，较低学习率 |
| 首码本 embedding / head | `[3072,1024]`，`1024→3072` | 新初始化 | 训练 |
| 15 组残余 embedding | 每张 `[2048,1024]` | 新初始化 | 训练 |
| Code Predictor | 5 层，hidden 1024，15 个 2048 类输出 head | 新初始化 | 训练 |
| Talker→Code Predictor projection | 1024→1024，Identity | 无参数 | 不适用 |
| ECAPA speaker encoder | 24 kHz、128 mel、1024 输出 | 公开 Qwen TTS speaker encoder | **冻结**，不进入优化器，保持 eval 模式 |
| Speech tokenizer encoder/decoder | 12.5 Hz，16 码本，24 kHz 波形 | 官方公开 codec | 冻结，离线编码 |

总参数 **914,643,008**；冻结文本前端 **317,459,456**；冻结 ECAPA **8,854,336**；合计冻结 **326,313,792**，可训练参数 **588,329,216**，不计独立冻结 codec。文本 LM 的 embedding 和输出 head 不在此基线中使用；公开 TTS 的 Talker transformer、audio embedding/head、Code Predictor 权重不加载。

“初始化”不等于“随机初始化”。这里 embedding 和 projector 都明确使用公开预训练权重初始化，并冻结；只有没有采用预训练来源的音频预测模块新初始化。不是把完整公开 TTS 模型拿来微调，也不是所有权重从零开始。

## 为什么暂时保留并冻结 text projector

官方 0.6B 的文本表是 2048 维、Talker 是 1024 维，必须有维度变换；官方 1.7B 同宽仍保留 MLP，说明它还包含非线性特征变换。公开资料未解释 2048 维文本表的确切来源与设计动机，不能认定它来自 1.7B Base。

当前先把已共同训练的 embedding 和 MLP 成套加载并冻结，避免初始化试验和数据规模变化同时混入主线。冻结保证这两组参数不被优化器改动，**不保证它们和 Qwen3-0.6B Base 主干天然匹配**：后者仍须适应这个输入。这个风险要靠基线的训练／验证结果判断。

若以后采用同一个 0.6B text Base 的 embedding 和 decoder，则 `1024→1024` 可以直接连接，无需为了 shape 额外加 MLP。这是另外一种初始化方案，保留在 [直连诊断记录](text-base-ablation.md)，不与当前基线混淆。

## 输入与预测编排

采用官方 `non_streaming_mode=True`、`language=Auto` 的 Base speaker 路径。设 `E_t` 为冻结的文本 embedding **加 projector**，`E_c` 为首码本 embedding，`E_g` 为残余码本 embedding，`s` 为在线 ECAPA 输出。

| 时间顺序 | 该位置的输入 |
|---|---|
| ChatML assistant 角色，3 个位置 | `E_t(role)` |
| 控制，3 个位置 | `E_t(text_pad) + E_c(nothink / think_bos / think_eos)` |
| 音色，1 个位置 | `E_t(text_pad) + s` |
| 文本 BOS，1 个位置 | `E_t(text_bos) + E_c(codec_pad)` |
| 完整文本，L 个位置 | `E_t(text) + E_c(codec_pad)` |
| 文本 EOD，1 个位置 | `E_t(text_eod) + E_c(codec_pad)` |
| 音频 BOS，1 个位置 | `E_t(text_pad) + E_c(audio_bos)` |
| 每个历史音频帧，1 个位置 | `E_t(text_pad) + E_c(c0) + Σ E_g(cg)` |

加号表示同位置逐元素相加，各行沿时间轴拼接；有效长度为 `L+10+T`。padding 不计入有效 position ID。训练和生成复用相同的输入构造。

音频 BOS 的 hidden 预测第 0 帧首码本；历史帧 t 后的 hidden 预测下一帧；最后一帧后预测 EOS。深度模型在当前帧内按 `[h_t, c_t,0, …]` 的已知前缀顺序预测剩余 15 组 code。目标帧不能提前进入预测它的 Talker hidden。

`loss = first CE / (有效帧数+EOS数) + 0.3 × residual CE / (有效帧数×15)`，分母在所有 rank 和 accumulation microbatch 间统计。文本位置、padding、不存在的残余 EOS 都不计音频 loss。

## Reference 与数据边界

每条训练样本为文本加整段目标 codec。训练和验证 loss 的 ECAPA 条件来自完整目标录音；变长 mel 按真实长度分组计算，避免 padding 影响池化。ECAPA 输出在线计算，但权重冻结。训练保留 singleton speaker；生成评估则使用同 speaker 的另一条训练录音，并跳过没有其他参考的样本。

不显式构造独立的 ICL reference/target 训练对。已有 codec 历史的编排经过官方 ICL 函数对照；当前评估从空音频前缀生成，不能据此宣称完成 zero-shot 验证。基线先评价未见文本内容准确率，而非未见说话人能力。

Emilia 仅接收 JSON 顶层 `type=short`，不从 long/dialogue 提取 short 切片。原始 m4a 保留在源 tar 中；中间 manifest 记录 locator；codec 离线缓存；speaker 音频在线解码，评估原音频按需落盘。

## 保存、恢复与验证要求

单机多卡使用 FSDP2，逐 decoder block 分片后包整个模型；BF16 参数计算、FP32 梯度归约，支持 activation checkpointing。冻结参数不进入 AdamW。每个 epoch 的采样计划复用，避免在百万条规模下每步重新全量排序。

DCP 保存模型、optimizer、scheduler、训练进度及每 rank RNG。组装报告 hash、manifest、speaker 条件方式和关键训练设置进入恢复签名。原始 tar 未做全库内容哈希，要求源音频保持不变。更换初始化、冻结策略或输入协议必须使用新 run。

当前官方宽度产物可以由公开 Qwen wrapper 验证权重结构；**冻结策略由本项目加载器执行**，原版 wrapper 本身不认识本项目的冻结配置。训练 checkpoint 仍是 DCP，不能直接当作官方发布模型使用。

验证包含：初始化逐张量来源核对；冻结前端更新前后严格相等；官方非流式／ICL embedding 对照；teacher forcing 与逐帧生成 logits 对齐；变长 padding 与 EOS；FSDP 梯度；真实 Emilia 小样本训练、保存及恢复；正式基线的验证 CE、固定样本音频、原音频 ASR 对照和生成 WER/CER。通过工程检查不等于语音质量达标。

文本 tokenizer 固定 checkpoint 原生 Qwen 规则，显式关闭 Mistral regex 补丁；与当前官方 wrapper 开关的差异及实测见 [分词策略](../training/emilia-baseline.md#固定文本分词规则)。
