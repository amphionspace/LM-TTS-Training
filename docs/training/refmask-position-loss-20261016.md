# uniform refmask 的逐位置 loss：mask 前后与句首

日期：2026-10-16（PDT）。本文是一次性诊断，不训练。脚本与完整输出在 [evidence/refmask-position-loss-20261016/](evidence/refmask-position-loss-20261016/)；逐帧原始结果（约 30 万行）未入库。

背景：emilia2 prefix reference 训练（job `pt-78v2p0f3`，配置 `configs/supervised-tts-20261015-emilia2-prefixref-refmask-bf16-16gpu-acc1-lr3e-4-bblr1e-4-ep3-wsd.yaml`）与 uniform refmask 只差参考区间的位置。prefix 的参考固定从第 0 帧开始，因此每条样本前 10%–50% 的帧（含句首）不计 loss。本文用已训练的 uniform 模型回答两个问题：

1. 被 mask 的参考区间之前 `[0, a)` 与之后 `[b, T)` 的 loss 是否不同（之后的帧在上下文中能看到参考音频）；
2. 句首帧是否显著比后面难，即 prefix 丢掉的是否是更难的监督。

---

## 0. 结论

1. **mask 前后的 loss 基本相同。** 同一样本内配对，first CE 前减后为 +0.007（95% CI −0.017..+0.030，n = 2,593，前高、后高各一半）；residual CE 前低 0.056（CI −0.072..−0.040），统计显著但只占约 1%。
   能看到同句前面的参考音频，并没有让后面的帧更好预测。推测模型主要依靠文本和 speaker embedding，几乎不利用上下文里同句的参考音频。因此"prefix 与 ICL 推理布局一致"这一优点，在 teacher-forcing loss 上可能体现不出来。
2. **句首不比后面难。** 前 3 帧减 `[10, a)` 的配对差：first CE +0.011（CI −0.038..+0.060，不显著），residual CE −0.20（句首更低）。第 0、1 帧的 first CE 是 1.24、1.13，推测多为句首静音。所以 prefix mask 掉句首，丢掉的不是更难的帧。
   但 prefix 训练里 BOS 之后的帧完全没有监督，speaker-only 生成能否正常起音仍需用实际生成验证，本文的 CE 回答不了这个问题。
3. **参考区间刚结束时没有"重新起步"的上升**（`t − b = 0` 时 first CE 为 1.339，与整体相当）。
4. **偏高的是句尾。** 按相对位置，最后 10% 的 first CE 约 1.44，中段为 1.18–1.31；EOS 本身为 0.251。去掉句首后（前段只取 `t ≥ 10`），前段 first CE 反而比后段低 0.045（CI −0.075..−0.015）。这个差距来自后段包含句尾，不是参考上下文造成的。

以上都是 teacher-forcing CE，不等同于生成质量。

## 1. 方法

- 模型：`UltraEval-Audio/init_model/emilia2-refmask-step-00034257/`。它是 `supervised-tts-20261012-emilia2-refmask-bf16-16gpu-acc1-lr3e-4-bblr1e-4-ep3-wsd` 最终 checkpoint 的 FP32 导出，推理时转为 BF16 + flash_attention_2，在 dev host GPU 5 上运行。
- 数据：`data/builds/supervised-tts-20261012-emilia2-reference/validation`（训练从未见过的 holdout）。用 seed 0 均匀抽 4,000 条，共 296,373 帧加 4,000 个 EOS。其中 151 条参考从第 0 帧开始（a = 0）。
- `frame_loss.py`：沿用训练的输入构造与 `collate`，但不施加 loss mask，逐帧计算 first codebook CE 和 15 个 residual codebook CE 的均值，并记录 `(t, T, a, b)`。在 32 条样本上的结果与该 checkpoint 的正式 val loss 一致：first 1.35 / residual 5.32，正式值为 1.306 / 5.448。
- `analyze.py`：分区（前、参考、后、EOS），按离 BOS 的绝对位置、离参考结束的位置、相对位置十分位做统计，并在同一样本内配对相减，以排除样本之间难度不同的影响。配对比较要求两侧各至少 5 帧。

## 2. 结果

### 2.1 分区

| 区域 | 帧数 | first CE | residual CE |
| --- | --- | --- | --- |
| mask 前 `[0, a)` | 99,580 | 1.306 | 5.461 |
| 参考（训练时 mask）`[a, b)` | 92,992 | 1.304 | 5.438 |
| mask 后 `[b, T)` | 103,801 | 1.353 | 5.475 |
| EOS | 4,000 | 0.251 | — |

配对（前减后）：

| 比较 | n | first CE 差（95% CI） | residual CE 差（95% CI） |
| --- | --- | --- | --- |
| 全部前段 − 后段 | 2,593 | +0.007（−0.017..+0.030） | −0.056（−0.072..−0.040） |
| 前段 `t ≥ 10` − 后段 | 1,543 | −0.045（−0.075..−0.015） | −0.002（−0.026..+0.021） |

### 2.2 句首（只统计 mask 前的帧；25 帧 = 2 秒）

| 离 BOS 的帧 | 帧数 | first CE | residual CE |
| --- | --- | --- | --- |
| 0 | 3,849 | 1.240 | 5.078 |
| 1 | 3,683 | 1.133 | 5.263 |
| 2 | 3,524 | 1.541 | 5.389 |
| 3–4 | 6,586 | 1.454 | 5.449 |
| 5–9 | 13,724 | 1.381 | 5.468 |
| 10–24 | 24,640 | 1.309 | 5.472 |
| 25–49 | 17,983 | 1.276 | 5.507 |
| 50–99 | 14,526 | 1.275 | 5.510 |
| ≥ 100 | 11,065 | 1.210 | 5.518 |

第 2–9 帧略高（约 1.38–1.54），推测对应从静音进入语音的过渡。只看前段不少于 4 秒（a ≥ 50）的样本，规律相同。同一样本内前 3 帧减 `[10, a)`（a ≥ 25，n = 1,126）：first CE +0.011（CI −0.038..+0.060），residual CE −0.203（CI −0.237..−0.170）。

### 2.3 参考结束之后与相对位置

`t − b` 为 0、1–2、3–9、10–24、≥ 25 帧时，first CE 分别为 1.339、1.323、1.322、1.299、1.405。

相对位置十分位（first CE；`-` 表示该区域在这一段没有帧）：

| | 0–10% | 10–20% | 20–30% | 30–40% | 40–50% | 50–60% | 60–70% | 70–80% | 80–90% | 90–100% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mask 前 | 1.34 | 1.34 | 1.30 | 1.27 | 1.18 | 1.31 | 1.29 | 1.42 | 1.24 | - |
| mask 后 | - | 1.40 | 1.26 | 1.25 | 1.18 | 1.31 | 1.30 | 1.35 | 1.36 | 1.44 |

## 3. 对 prefix 实验的含义与后续

- uniform 模型几乎不利用同句上下文里的参考音频，句首也不比其他位置难。所以从 loss 看，prefix 与 uniform 的差别预计不大。两者的真正差别要看生成：
  - speaker+ICL：prefix 的布局与推理一致；
  - speaker-only：prefix 从未训练过起音。
- prefix run 的 val loss 与 uniform 不可直接比较，因为 mask 掉的区间不同。对比以 SeedTTS（`speaker_icl`）为准。`scripts/synthesize.py` 对 speaker 模型固定使用 speaker-only，不要用它评测 prefix run。
- prefix run 有 checkpoint 后，可以在同一批 4,000 条上用同一脚本统计，按位置对比两种方案。

## 4. 复现

```bash
cd LM-TTS-Training
E=docs/training/evidence/refmask-position-loss-20261016
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. ../envs/lm-tts/bin/python $E/frame_loss.py \
  --model ../UltraEval-Audio/init_model/emilia2-refmask-step-00034257 \
  --build data/builds/supervised-tts-20261012-emilia2-reference/validation/manifest.json \
  --out /tmp/uniform.npy --samples 4000 --batch 16   # 单卡约 5 分钟
python3 $E/analyze.py /tmp/uniform.npy > $E/results.txt
```
