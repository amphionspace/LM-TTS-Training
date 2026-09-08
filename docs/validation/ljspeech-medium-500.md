# LJSpeech 500 步训练实验

## 实验范围

训练代码：`8fb7087`。配置：[ljspeech-medium.yaml](../../configs/ljspeech-medium.yaml)。
使用 2 × A800 80GB、FSDP2、BF16 和 activation checkpointing。
模型从 Qwen3-0.6B-Base 文本参数和 Qwen 公开 ECAPA 初始化，音频预测相关新参数重新初始化；冻结官方 codec。
这不是对官方已训练 TTS Talker 的微调。

LJSpeech 按 seed 42 抽取不超过 10 秒的录音，2,048 条训练、128 条验证；分别约 3.7493 和 0.2386 小时。
每卡 batch 1，累积 4 次，全局有效 batch 8；500 次更新相当于约 1.95 轮训练数据。
新参数学习率 1e-4，文本 backbone 和 ECAPA 学习率 2e-5，warmup 25 步，500 步 cosine schedule。
ECAPA 在线读取同说话人的另一条训练录音，截取 3 秒作为音色条件，并参与反向传播。

每 50 步在全部 128 条验证数据上计算 token 加权 CE；每 100 步保存完整训练状态，并对固定的 3 条验证文本贪心生成。
生成上限 192 帧，即 15.36 秒；ASR 为 faster-whisper small.en。
固定样本为 LJ041-0107、LJ021-0146、LJ026-0043，共 67 个归一化单词、332 个字符。
WER/CER 按总错误数除以总参考长度计算，不是句级分数的简单平均。

## 训练结果

500 步及最后一轮生成评估均完成，训练进程退出码为 0。五个定期 checkpoint 均有完成标记，最终 TensorBoard 标量及三条生成/原录音音频均记录到第 500 步。
完整日志保存在 `runs/ljspeech-medium-500/training.log`，机器可读汇总为同目录的 `experiment-summary.json`。

| 步数 | 验证首码本 CE | 验证残余码本 CE | 生成 WER | 生成 CER | 正常 EOS |
|---:|---:|---:|---:|---:|---:|
| 100 | 6.3693 | 7.4671 | 100.00% | 94.28% | 1/3 |
| 200 | 6.3092 | 7.3184 | 98.51% | 93.07% | 3/3 |
| 300 | 6.2771 | 7.2429 | 98.51% | 94.28% | 3/3 |
| 400 | 6.2535 | 7.2060 | 97.01% | 95.78% | 3/3 |
| 500 | 6.2448 | 7.1961 | 123.88% | 89.46% | 3/3 |

最早一次完整验证（第 50 步）CE 为 6.5091 / 7.5150；到第 500 步为 6.2448 / 7.1961。
第 500 步训练 CE 为 6.1642 / 7.1243，ECAPA 梯度范数为 0.1719；所有已记录的训练指标均为有限数值，speaker 梯度均非零。

原录音对照的 ASR WER/CER 始终为 0。最终生成有 83 个词错误 / 67 个参考词，297 个字符错误 / 332 个参考字符；词错误包括 35 个替换、29 个删除和 19 个插入。
WER 可以超过 100%，并非百分比计算错误。最终三条生成时长为 7.52、6.00、7.12 秒，均触发 EOS。
ASR 转写与目标文本仍明显不符；噪声也可能使 ASR 产生重复词，不能把转写当作人工听评。

结论：工程链路完整跑通，但这轮生成内容准确性不达标，不能把 loss 下降或正常 EOS 当作合成质量通过。
建议下一步先用小集合做过拟合检查，结合原音频/codec 重建对照与自由生成，确认能正确读出训练文本，再决定扩大数据与训练时长。
本轮没有另外启动更长训练，也没有测量未见说话人的效果。

## 复现与产物

在项目目录中使用独立 venv：

```bash
source .venv/bin/activate
bash scripts/prepare_ljspeech.sh --limit 2176 --val-count 128 --max-seconds 10 --output data/ljspeech-medium
bash scripts/run_train.sh --config configs/ljspeech-medium.yaml
# 中途退出后，沿用相同配置和学习率计划：
bash scripts/run_train.sh --config configs/ljspeech-medium.yaml --resume latest
tensorboard --logdir runs/ljspeech-medium-500/tensorboard --port 6006
```

本地产物目录：`runs/ljspeech-medium-500/`。权重、音频和数据不提交到 Git。
`checkpoints/` 保存模型、AdamW、scheduler、数据位置及各 rank RNG；`evaluation/step-*/` 保存生成 WAV、逐句指标和汇总。
TensorBoard 包含训练/验证 CE、speaker 梯度、学习率、WER/CER、截断率及生成/原录音音频。
精确断点恢复已在独立的完整模型对照实验中验证，见[组装验证](assembly-validation.md)；本轮用于检查较长训练及生成表现。

## 结果边界

这个实验只有约 3.75 小时训练数据，且音频建模相关参数从头初始化；短程 loss 下降不能证明语音质量合格。
生成评估只有 3 条固定文本，不能替代完整验证集的内容准确性测量。
LJSpeech 只有一个说话人，本轮也未启用 ICL 参考文本/codec 前缀，因此不能据此判断跨说话人 zero-shot 或 ICL 能力。
