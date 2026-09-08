# 诊断候选：同源文本 Base 直连（非当前主线）

当前主线见 [最终基线结构](final-architecture.md)。本页只记录已做的 LJSpeech 诊断候选，不作为最终结构结论。

本页描述 `assembled-qwen3-tts-direct` 路径。旧实验使用的 `legacy_prefix` 和随机 text MLP 仅用于恢复、诊断与对照，不能与本页混用。训练的是新的 TTS 音频预测能力；文本 backbone、speaker encoder 和 codec 使用预训练权重。

## 1. 文本 embedding 为什么不再经过 MLP

实际检查本地固定 revision 的配置与权重：

| 模块 | 官方 Qwen3-TTS 0.6B | 本工程默认组装 |
|---|---|---|
| 文本 embedding | 151936 × 2048 | Qwen3-0.6B-Base：151936 × 1024 |
| Talker hidden size | 1024 | 同一个 Qwen3-0.6B-Base：1024 |
| 文本 projection | Linear(2048,2048) → SiLU → Linear(2048,1024) | Identity，无权重、无额外变换 |

官方 0.6B 的输入宽度与 Talker 不同，确实需要变换。我们的 embedding 与 backbone 同源且同宽，不能再以“维度对齐”为理由插入随机 MLP。默认删除这个变换，完整保留原先 embedding 到 decoder 的连接。旧随机 MLP 对预训练文本向量的影响很大，但这只是初始化疑点，不能单凭它断言之前验证集效果差的根因已经找到。

官方 1.7B TTS 的 text embedding 和 Talker 都是 2048 维，却仍保留 `2048→2048→2048` MLP。因此官方 projection 也包含可学习的特征变换，不能把它的作用全部归结为维度适配。2026-09-08 搜索官方配置、实现、报告及相关 issue 后，未找到选择 2048 维及 embedding 初始化来源的明确解释。两个尺寸使用同一输入宽度可能有利于统一文本接口，但这只是推测，不能据此声称共享权重或取自 1.7B Base。

来源：[0.6B 配置](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/blob/main/config.json)、[1.7B 配置](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/blob/main/config.json)、[官方实现](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/core/models/modeling_qwen3_tts.py)。维度相同意味着不必为了 shape 加 projector，不代表实验已经证明任何可学习 projector 都无益。

这是相对官方 checkpoint 的明确适配，不声称逐参数复刻官方 0.6B；直接加载官方 2048 维文本 embedding 会违背当前采用同源 0.6B 文本初始化的选择。`near-identity` 仅是此前准备的初始化选项，不是默认结构，也不是严格恒等映射。

## 2. 模块、维度与初始化

| 模块 | 结构 / 张量形状 | 初始化与训练 |
|---|---|---|
| Text tokenizer | 151676 项，embedding 容量 151936 | Qwen TTS tokenizer；核对所有共享 ID |
| Text embedding | `[151936, 1024]` | Base 共享行；7 个新增 TTS token 行重新初始化；参与训练 |
| Talker | 28 decoder layers，hidden 1024，FFN 3072；16 Q heads / 8 KV heads；head_dim 128 | Base layers 和 final RMSNorm 严格加载；参与训练 |
| First-code embedding / head | `[3072,1024]` / `1024→3072` | 新初始化，二者不绑权重 |
| Residual embeddings | 15 个 `[2048,1024]` | 新初始化；分别对应码本 1–15 |
| Code Predictor | 官方 5 层深度自回归模型；hidden 1024；15 个 `1024→2048` head | 新初始化，预测当前帧的残余码本 |
| Speaker encoder | 官方 ECAPA-TDNN；24 kHz、128 mel，输出 `[B,1024]` | 加载公开 Qwen TTS speaker 权重，联合训练，不 detach |
| Speech tokenizer | 16 个码本，每组 2048；每帧约 80 ms（12.5 Hz）；24 kHz 输出 | 官方 encoder/decoder 冻结；音频 codes 离线缓存 |

默认模型有 752,766,016 个可训练参数：Talker 743,911,680，ECAPA 8,854,336，不计独立冻结 codec。Talker 保留官方 MRoPE 实现；当前给三个轴相同的一维有效位置，padding 不增加 position ID。Text LM head、已经训练好的 TTS Talker / Code Predictor 均不加载。

Qwen 的公开 ECAPA 权重已经过其训练，不能等同于公开 SpeechBrain VoxCeleb ECAPA；后者的采样率、mel、通道数及输出宽度不同。Qwen 最初是否使用过其他 speaker checkpoint，不从这些公开权重推断。

## 3. 输入编排：官方 non_streaming_mode=True、language=Auto

定义 `E_t` 为文本 embedding，`E_c` 为首码本 embedding，`E_g` 为残余码本 embedding，`s` 为 ECAPA 输出。这里的 `+` 是相同位置的逐元素相加；表中各行沿时间轴拼接。

| 时间顺序 | 文本通道 | 音频 / 音色通道 | Talker 输入 |
|---|---|---|---|
| 角色前缀，3 个位置 | `<\|im_start\|>assistant\n` | 无 | `E_t(role)` |
| 控制，3 个位置 | 每个位置 `<tts_pad>` | `codec_nothink, codec_think_bos, codec_think_eos` | `E_t(pad)+E_c(control)` |
| 音色，1 个位置 | `<tts_pad>` | `s` | `E_t(pad)+s` |
| 文本起点，1 个位置 | `<tts_text_bos>` | `codec_pad` | `E_t(text_bos)+E_c(codec_pad)` |
| 完整目标文本，L 个位置 | 文本 token | 每个位置 `codec_pad` | `E_t(text)+E_c(codec_pad)` |
| 文本终点，1 个位置 | `<tts_text_eod>` | `codec_pad` | `E_t(text_eod)+E_c(codec_pad)` |
| 音频起点，1 个位置 | `<tts_pad>` | `codec_bos` | `E_t(pad)+E_c(codec_bos)` |
| 历史音频帧，每帧 1 个位置 | `<tts_pad>` | 当前历史帧的 16 个 code | `E_t(pad)+E_c(c0)+Σ E_g(cg)` |

完整有效输入长度为 `L+10+T`，T 为已有音频帧数。Batched training 的文本 padding 可能在内部形成被 mask 的空位；有效 token 顺序和 position ID 不变。训练、空前缀生成与已有 codec 前缀续写使用同一个构造函数。

选择非流式是因为当前目标是给完整文本后生成音频。它仍是官方双通道 embedding 编排，不能将“非流式”误解为只有单通道拼接。暂未加入官方流式逐 token 文本输入、显式语言控制分支、文字音色指令和 CustomVoice speaker ID。

## 4. 预测时序和 loss

音频 BOS 所在位置的 Talker hidden `h_0` 预测第 0 帧的首码本。历史帧 t 输入后的 hidden `h_(t+1)` 预测下一帧。最后一帧后预测音频 EOS。当前帧的 codes 不得进入预测该帧首码本的 hidden。

对每个真实帧，Code Predictor 接收 `[h_t, E_c(c_t,0), E_1(c_t,1), …, E_14(c_t,14)]`，按因果顺序预测码本 1–15。训练时使用真实的更早码本，推理时使用已经生成的码本；这两条路径应在相同前缀下给出一致 logits。EOS 不产生残余码本 loss。

`loss = sum(first CE)/(有效音频帧+句末EOS数) + 0.3 × sum(residual CE)/(有效音频帧数×15)`。

首码本 CE 使用完整 3072 项词表，padding 和文本前缀不计 loss；生成时仅允许内容 ID 0–2047 与 EOS 2150。音频 BOS=2149、pad=2148。训练使用全局有效 token 数归一化，包含所有 GPU 和 accumulation microbatch；FSDP 梯度平均所需 world-size 因子显式补偿。

## 5. 音色参考与 ICL

每条训练数据是文本加整段目标音频 codec。另从训练集选择同 speaker 的不同录音，为 ECAPA 提供 3 秒参考；固定中心裁剪，不足长度重复，只缓存 mel，在线计算 speaker 向量并传梯度。验证也只从训练池取音色参考，不使用验证目标音频作参考。

这个参考是 speaker conditioning，不是显式构造的 ICL reference-text/reference-code 训练对。非流式 ICL 推理可以在完整 reference+target 文本后，先喂 reference codec 再续写，参考尾部不插入音频 EOS。输入构造测试覆盖官方 ICL 函数对已有 codec 历史的编排，但当前训练评估入口仍从空音频前缀生成；这不是已经验证 zero-shot 能力的声明。LJSpeech 单说话人数据也不能支持跨说话人结论。

## 6. 分布式、保存和可重复性

单机多卡使用 PyTorch FSDP2：逐 decoder block 分片，再包整个模型；BF16 参数计算、FP32 梯度归约，支持 activation checkpointing。诊断脚本只将参数转为 BF16，RoPE 频率 buffers 保持 FP32，与训练精度一致。

DCP checkpoint 包括模型、AdamW、scheduler、step/epoch/下一 batch、每 rank 的 Python/NumPy/Torch/CUDA RNG。完成标记后才更新 latest。manifest、参考分配及参考音频 hash、组装报告、训练设置进入恢复签名；更换协议或初始化必须新建 run，不能冒充原实验的连续恢复。

直接连接模式通过本项目 `TTSModel.from_assembled` 加载；原版 Qwen 类会重新创建 MLP，不能直接用原版 wrapper 重载此产物。组装报告明确记录 `project_loader_reload=true`、`public_wrapper_reload=false`。保留 MLP 的对照产物可以通过原版 wrapper 做结构重载，但训练 checkpoint 仍为本项目 DCP。

## 7. 对齐的证据与范围

- 安装版本 `qwen-tts==0.1.1` 的 `Qwen3TTSForConditionalGeneration.generate` 和 `generate_icl_prompt` 是输入编排的直接对照对象；测试截获官方函数实际生成的 embedding，并与训练输入比较。
- [官方模型代码](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/core/models/modeling_qwen3_tts.py)包含非流式、流式与 ICL 分支。
- [官方 SFT dataset](https://github.com/QwenLM/Qwen3-TTS/blob/main/finetuning/dataset.py)和[训练脚本](https://github.com/QwenLM/Qwen3-TTS/blob/main/finetuning/sft_12hz.py)是公开微调示例，不是完整预训练配方；不能照搬其 speaker detach、固定 speaker 导出及未经验证的标签索引。
- [技术报告](https://arxiv.org/abs/2601.15621)说明整体架构；公开材料不足以还原全部数据配比、优化策略和 ICL pair 采样。

实现入口：`qwen3_train/model.py`、`qwen3_train/assembly.py`、`scripts/assemble_qwen3_tts.py`。实验结果应另记于 `docs/validation/`，不把结构正确性与语音泛化能力混为一谈。
