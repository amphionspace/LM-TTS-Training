# ICL 与 speaker-only 生成评估

代码支持同一组目标文本、同一条参考录音下的两种生成模式：

| 模式 | 参考录音提供的条件 |
|---|---|
| `speaker_only` | 冻结 ECAPA 提取的 speaker embedding |
| `icl` | speaker embedding、参考文本、参考音频 codec 前缀 |

配置示例：

```yaml
eval:
  conditioning_modes: [speaker_only, icl]
  num_samples: 8
  train_num_samples: 2
  audio_every: 500
  max_frames: 400
  codec: pretrained/Qwen3-TTS-Tokenizer-12Hz
  codec_device: cuda:0
  asr: true
  asr_model: small
  english_normalization: true
```

未指定 `conditioning_modes` 时只运行 `speaker_only`。ICL 需要 Qwen 非流式协议、speaker encoder，以及包含文本 BOS/EOS 的 prepared 数据。`max_frames` 限制的是新增目标音频帧数，不包含参考 codec。

## 输入与输出边界

ICL 按当前安装的 `qwen-tts==0.1.1` 非流式实现构造上下文：

```text
文本：BOS + 参考文本 token + 目标文本 token + EOS
音频：audio BOS + 参考 codec 帧 + 已生成的目标 codec 帧
音色：参考录音的 speaker embedding
```

目标录音的 codec 不进入生成输入。参考文本末尾与目标文本开头之间不额外插入 BOS/EOS。音频 decoder 一起接收参考 codec 和新增 codec，然后按参考帧数占总帧数的比例裁去参考波形，与 Qwen voice-clone wrapper 的裁剪方式一致。

生成时长、长度比、ASR WER/CER 只针对裁剪后的目标音频；ASR 的比较文本仍是目标文本。每条 `metrics.json` 记录 `conditioning`、参考样本 ID、来源、参考 codec 帧数，以及 ICL 使用的参考文本。

生成采用贪心解码，并与官方 `min_new_tokens=2` 一致：在前两帧新增音频的首码本选择前屏蔽 EOS，之后恢复正常 EOS 判断。参考 codec 不计入新增帧数；`max_frames` 仍是生成上限。此规则同时适用于 speaker-only 和 ICL，避免参考音频结束后立即返回零帧。不能先选出 EOS 再忽略停止标志，否则会把 EOS 占位值当作音频码反馈。

`scripts/inspect_checkpoint.py --generate` 的独立诊断也使用相同的最小生成长度。

`metrics.json` 和 `summary.json` 中的 `generation_policy` 标明 `decoding=greedy`、`min_new_frames=2`。2026-09-10 修复前的结果没有该字段，并允许第一步 EOS；前后空输出率与 WER/CER 应结合生成规则比较。本规则不改变训练 loss 或 checkpoint 张量，也不引入官方默认采样和重复惩罚；长静音问题仍需分别诊断。

## 配对与结果隔离

目前从训练清单选择同 speaker、同语言、ID 和规范化文本均不同的另一条录音。两种模式复用同一对目标/参考，按语言选择固定数量的目标样本。当前选择用于在线诊断；它不等于正式的未见说话人 ICL 测试集。

两种模式的产物和 TensorBoard 标签分开：

```text
<run>/evaluation/speaker_only/step-XXXXXXXX/sample-XX/
<run>/evaluation/icl/step-XXXXXXXX/sample-XX/
<run>/train-evaluation/evaluation/<mode>/step-XXXXXXXX/sample-XX/

TensorBoard: eval/<mode>/...、train_eval/<mode>/...
```

每个 mode/step 有独立 `summary.json`，即使关闭 ASR，也保存样本数量和截断率。`scripts/rescore_english.py` 可以扫描这些目录，分别汇总两种模式中的英文样本。

完整 10kh 已完成配对核查：512 条验证目标中 511 条有合格参考，在线固定使用 8 条（中英各 4）；配对和隔离范围见 [正式训练记录](emilia-10kh-supervision.md)。此前固定快照仅用于测试代码和吞吐。本次改动仅增加生成评估条件，不改变训练 loss 或训练样本的 speaker 条件，也不表示模型已经具备良好的 ICL 能力。

1kh 的历史 run 保持原样；其 speaker-only 结果和保留要求见 [1kh 报告](emilia-1kh-report.md) 与 [保留清单](run-retention.md)。

## 验证

`tests/test_qwen_protocol.py` 将模型输入 embedding 与已安装 Qwen 的非流式 ICL 输入直接比较。`tests/test_icl_evaluation.py` 覆盖完整评估入口中的配对、参考文本/codec 前缀、目标 codec 隔离、解码后的参考音频裁剪、只评分目标文本，以及两种模式分开发布结果。
