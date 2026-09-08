# 从头训练时是否需要 reference

这里讨论从文本 LLM、冻结 codec 和可训练 speaker encoder 出发，重新训练 TTS；不是对现成 TTS 的微调。

结论：对于完整文本前缀的自回归 TTS，整段音频 teacher forcing 已训练音频续写，不必另外制作 ICL reference-target 配对。
但这不是保留了文本 LLM 原有的语音 ICL 能力：语音能力必须通过训练学到，效果需要 unseen-speaker 测试验证。

## Nar TTS 的实际路径

核对版本：ba00e519a73c318a9f875ed013d948ada1a46b7d。
Nar 使用 causal LM 和 Mimi；名字中的 Nar 不表示其预训练使用非自回归模型。

- [预训练入口](https://github.com/kadirnar/nar-tts/blob/ba00e519a73c318a9f875ed013d948ada1a46b7d/nar_tts/training/pretrain.py)：从用户指定的因果 LM 初始化、扩词表，读取 input_ids。
- [序列构造](https://github.com/kadirnar/nar-tts/blob/ba00e519a73c318a9f875ed013d948ada1a46b7d/nar_tts/core/tokens.py#L140)：完整文本、边界 token、完整 Mimi tokens、音频 EOS；没有独立 reference 字段。
- [collator](https://github.com/kadirnar/nar-tts/blob/ba00e519a73c318a9f875ed013d948ada1a46b7d/nar_tts/core/data.py#L6)：无显式 labels 时使用 input_ids，按因果 LM 训练，padding 不计 loss。
- [推理前缀](https://github.com/kadirnar/nar-tts/blob/ba00e519a73c318a9f875ed013d948ada1a46b7d/nar_tts/inference/infer.py#L542)：先把 reference text 与 target text 拼接，再加入 reference speech tokens，后续生成 target speech。

这能证明公开实现采用了这种训练/推理协议；不能单凭代码证明其大规模 zero-shot 效果。
Nar 默认还计算文本 token loss；本工程只监督音频预测，不能把两者整个 loss 配方视为相同。

## 其他模型的证据

| 模型与路径 | 从头训练时如何提供音频条件 | 对本工程的启示 |
|---|---|---|
| VALL-E AR | 普通因果 LM，不显式抽取 prompt；历史 tokens 自然构成条件 | 最直接支持整段训练、前缀续写 |
| VALL-E NAR | 残余码本预测显式使用 acoustic prompt | 不能把 AR 结论直接套给 NAR |
| CosyVoice 2 的 Qwen2LM | 非流式分支为完整 text + speech；另有交错流式分支 | 是否显式分 reference，取决于序列协议 |
| Fish Speech 公开 iterable 数据管线 | 按同 speaker 分组采样，把多个 text/audio 序列放进上下文 | 多段上下文可显式训练跨段依赖，但不必单独保存 reference WAV |
| F5-TTS | 从同一 mel 随机遮盖片段，保留其余片段为条件，对遮盖区域算 flow loss | 需要条件音频，但可从单条样本在线构造；不是纯 AR 续写目标 |

来源：[VALL-E §4.2](https://arxiv.org/pdf/2301.02111)、[CosyVoice Qwen2LM](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/llm/llm.py)、[Fish 数据管线](https://github.com/fishaudio/fish-speech/blob/main/fish_speech/datasets/semantic.py)、[F5-TTS CFM.forward](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/cfm.py)。
公开数据代码能说明实现机制，不应当作所有发布模型完整预训练配方的证明。

## 对当前框架的建议

1. 保留整条文本和完整 codec 的 AR teacher forcing 作为基础训练，不强制所有数据提供单独 ref_text/ref_wav 配对。
2. ECAPA 路径训练时必须获得音频条件，但可从当前录音裁剪，也可取同 speaker 的另一条。两者都不要求另存 WAV；当前原型选择另一条，是防止内容泄漏的策略，不是理论必要条件。
3. 可把同 speaker 的两段拼成一段作为增强：文本合并，codec 合并，中间不插音频 EOS。它有助于覆盖跨句、跨录音边界；是否提升效果应做消融，不能称为 ICL 的必需步骤。
4. 正式 zero-shot 评估使用训练未见的 speaker，reference 与目标录音分离。先比较 speaker-only 与 speaker+ICL；若要测试 ICL-only，还应让训练覆盖无 speaker 向量的条件，避免测试条件突变。
5. ICL 参考音频与 reference text 必须匹配。最初可用一整条较短录音，避免按秒裁剪后仍使用全文。ECAPA 参考不需要 reference text。

以上是设计建议；本版本尚未新增 ICL 推理入口、两段拼接增强或 speaker 条件丢弃。

## Qwen 的分隔符能否证明训练做了 reference 配对

不能。[官方 generate_icl_prompt](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/core/models/modeling_qwen3_tts.py)直接连接 ref_id 与 text_id，再追加一次文本结束 embedding；codec 前缀为 codec BOS 加 ref_code。
调用前，reference 和 target 文本各自的 ChatML 外壳被切掉。reference 与 target 文本之间没有新增专用分隔符，reference codec 末尾也没有 EOS。
非流式路径再把文本轨与 codec padding 相加、音频轨与 text padding 相加。这些模态边界和 padding 不等同于 reference-target 分界。

因此，这个 ICL 推理接口与完整文本加完整音频的 AR 训练兼容；不能据分隔符反推必须显式配对。
但也不能反向断言 Qwen 完整预训练从未做过 reference 增强：论文和公开推理代码没有交代全部数据配方。
