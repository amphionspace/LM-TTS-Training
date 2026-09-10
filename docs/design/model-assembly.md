# 模型组装与差异

当前主线见 [基线详细结构](final-architecture.md)，官方逐模块结构见 [官方结构与初始化证据](official-qwen3-tts.md)。

入口：`scripts/assemble_qwen3_tts.py`。默认固定官方来源 revision，也可传入本地目录。
组装产物只是初始化权重，尚未训练的音频预测模块不能直接合成可理解语音。

| 模块 | 官方 Qwen3-TTS 0.6B | 本工程组装 |
|---|---|---|
| Talker decoder | hidden 1024，28 层，16 Q / 8 KV heads，head dim 128 | 从 Qwen3-0.6B-Base 严格加载 layers / norm |
| 文本 embedding | 151936 × 2048 | 151936 × 2048，加载公开 TTS 完整权重并冻结 |
| text projection | ResizeMLP，2048 → 1024 | 保留官方 2048→2048→1024，加载配套权重并冻结 |
| codec embedding / head | 3072 项 | 新初始化，内容 token 为 0–2047 |
| Code Predictor | 5 层，16 码本，每组 2048 项 | 官方结构，新初始化 |
| speaker encoder | ECAPA-TDNN，24 kHz，128 mel，1024 输出 | 加载公开 Qwen TTS 的 speaker_encoder 权重，冻结 |
| speech tokenizer | 12.5 Hz，16 码本，24 kHz 解码 | 复制官方 encoder / decoder，冻结并离线编码 |
| 位置编码 | Talker MRoPE | 保留结构，本工程使用相同轴位置的一维序列 |
| 输入协议 | ChatML、流式双轨等 | 官方非流式 Auto 模式的 ChatML、控制前缀与双通道 embedding |

文本 LM head、Qwen TTS 的 Talker transformer 和 Code Predictor 权重均不加载；文本 embedding / projection 则成套加载。
本地来源会记录路径、配置校验值和可用的 REVISION 标记；默认 revision 字段本身不能证明本地文件来源。

## 扩词表

tokenizer 从 151669 项增加到 151676 项；共享 token 的 ID 全部核对。
embedding 已有 151936 行，因此无需增大矩阵，当前基线加载公开 TTS 中已经训练的新增 7 行；只有 text-base 消融初始化才重新初始化这些行。
依次为：151669 audio_start、151670 audio_end、151671 tts_pad、151672 tts_text_bos、151673 tts_text_eod、151674 tts_text_bos_single、151675 audio_pad；精确 token 字符串见 assembly_report.json。
训练协议采用官方非流式双通道编排，完整时序与预测位置见 [基线结构](final-architecture.md)。
保留全部 special token 不代表当前协议使用每个 token，也不代表复现完整官方训练配方。

## Speaker 来源与联合训练

[Qwen 技术报告 §3.1](https://arxiv.org/html/2601.15621v1#S3.SS1)明确说明 speaker encoder 与 backbone 联合训练。
公开权重已经过 Qwen 训练，但报告未说明最初是否加载过外部说话人识别 checkpoint。

| | Qwen speaker | SpeechBrain VoxCeleb ECAPA |
|---|---|---|
| 采样率 / mel | 24 kHz / 128 | 16 kHz / 80 |
| channels | 512/512/512/512/1536 | 1024/1024/1024/1024/3072 |
| 输出维度 | 1024 | 192 |
| BatchNorm | 未使用 | 使用 |

依据：[Qwen 实现](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/core/models/modeling_qwen3_tts.py)、[SpeechBrain 配置](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/blob/main/hyperparams.yaml)。两者不能直接互载，仅加输出投影也无法解决内部结构差异。
当前采用 Qwen 公开 ECAPA 权重并冻结，不进入优化器，保持 eval 模式。这是当前基线的选择，与官方报告中的联合训练不同。

训练和验证loss使用完整目标录音提供speaker条件，变长mel按真实长度分组送入冻结ECAPA，不裁成固定3秒，也不重复短音频。生成评估使用同speaker、同语言、不同ID和文本的另一条训练录音。参考配对与两种生成模式见[ICL评估](../training/icl-evaluation.md)。

## ICL 与 speaker 参考不同

完整文本条件下的整段音频 teacher forcing 已在学习条件分布 p(audio_t | text, 历史 audio)。
推理时拼接 reference text 与 target text，再以 reference codec 前缀续写，在原理上不要求训练时另存一个 ICL reference WAV；reference codec 后不能插入结束整段音频的 EOS。
这不意味着文本 Base 自动保留了语音 ICL 能力：能力仍需通过语音训练学到，并通过 zero-shot 评估验证。
当前评估同时运行speaker-only与ICL；后者拼接参考文本和参考codec前缀，参考帧不计入新增帧数。
LJSpeech 只能验证工程和内容准确性，不能验证跨说话人 zero-shot 泛化。

## 保存与校验

当前默认脚本核对结构和 token ID 后组装，用公开 Qwen3TTSModel.from_pretrained 重载 processor、Talker、speaker 和 codec。
成功后才原子发布目录及 ASSEMBLY_COMPLETE，写入 assembly_report.json 和产物 SHA256。
qwen-tts 0.1.1 保存 speaker config 时会写入构造器不接受的 model_type；脚本移除该元数据，保证官方接口可重载。
默认初始化权重格式兼容官方加载；冻结策略由项目加载器执行。text-base + Identity 消融仅由项目加载器重载，报告明确区分。训练 checkpoint 为本工程 DCP 格式，输入协议仍为非流式，不能直接当作官方生成模型使用。

从头训练的证据及方案建议见 [reference 训练对比](reference-training.md)。
