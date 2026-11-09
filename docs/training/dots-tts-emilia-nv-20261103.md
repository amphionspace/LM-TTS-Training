# dots.tts-soar 副语言微调：Emilia-NV + refmask（2026-11-03）

在 `dots.tts-soar` 上用 Emilia-NV 微调，让模型支持文本内的副语言标签（`[Laughter]`、`[Breathing]` 等），并在 NV-Bench、Seed-TTS、CV3 上与未训练的 soar 及公开模型对比。

## 1. 训练

| 项目 | 设置 |
| --- | --- |
| 起点 | `dots-studio/dots.tts-soar`（Qwen2 1.5B LLM + PatchEncoder + DiT，48 kHz AudioVAE，CAM++ x-vector） |
| 组装 | `assets/assembled/dots-tts-soar-nv18`：18 个 NVSpeech 标签作为新 token 加入 tokenizer，新 embedding 行用标签原子词 embedding 的均值初始化 |
| 数据 | Emilia-NV（`emilia_nv/v0.1`，中文，174,179 条，591 h），dots_vae 特征 v2（`emilia_nv-dots_vae-dots-tts-soar-vae-xvec-48k-fp32-v2-20261103T095118bjt-01`）；build `dots-emilia-nv-reference-20261103`，训练 172,371 条、验证 1,741 条，67 条太短无参考被排除 |
| 参考与 mask | 无配对数据，x-vector 只用句内随机参考段计算，该段不计 FM loss（EOS 正样本保留）。规则同 Qwen refmask `codec-grid-reference-v1`，换到 0.16 s patch 网格：长度均匀取自 [max(0.5 s, 10%), 50%]，再均匀取起点 |
| 布局 | basic 与交错（streaming）1:1，同官方微调配置 |
| 优化 | 全模型（LLM、PatchEncoder、DiT）lr 1e-5，WSD（warmup 2%，最后 20% 余弦降到 1e-6），2 epoch，单节点 8×A800，每卡 600 patch / 1000 token，step 6,448 |

两个 run 只差 x-vector dropout：

| Run | ACP | x-vector dropout | 最终验证 FM / EOS |
| --- | --- | --- | --- |
| refmask | `pt-6e8gc5a5` | 0.2（官方默认） | 0.7488 / 0.0235 |
| refmask-noxvecdrop | `pt-vb4f8ix6` | 0 | 0.7492 / 0.0228 |

CFG dropout（LLM hidden，0.2）两边相同。配置：`configs/dots-tts-20261103-soar-emilia-nv-refmask-bf16-8gpu-lr1e-5-ep2-wsd.yaml`、`configs/dots-tts-20261103-soar-emilia-nv-refmask-noxvecdrop-bf16-8gpu-lr1e-5-ep2-wsd.yaml`。

## 2. NV-Bench

### 口径

- 数据：`CharlesNi/NV-Bench`，zh_single 650、zh_multi 391、en_single 350、en_multi 260 条，每条给定 prompt 音频、prompt 文本和带标签目标文本。dots 以 speaker + ICL 方式生成（官方 runtime，10 步，CFG 1.2）。
- 识别：`CharlesNi/Multilingual-NVASR`（SenseVoice-Small 微调）转写生成音频。
- 指标：每个 `[Tag]` 计为一个符号，其余文字转小写、去标点和空格后按字符计。CER 只比文字，PCER 只比有序标签序列，OCER 比两者合在一起；全部为子集级语料错误率。SIM 为 WavLM-large 与 prompt 音频的余弦相似度。
- 打分脚本是按论文定义自写的（`UltraEval-Audio/scripts/nv_bench_score.py`），没有官方实现。人类录音的中文结果与 Ming 模型卡给出的 NVbench-GT 接近（zh_single 3.44 / 8.62 / 3.51 对 3.86 / 9.38 / 4.07）。英文 CER 的差距较大（3.43 对 6.73），推测是英文文本归一化不同，跨来源比较英文 CER 时不可直接对比，PCER 受影响较小。

### 结果

除标注"我们测"的行外，数字均摘自 [Ming-3.0-tts-tiny 模型卡](https://huggingface.co/inclusionAI/Ming-3.0-tts-tiny#paralinguistic-control-nvbench)（NV-CV3 等行同 NV-Bench 论文表 4）。所有数值越低越好。

| 模型 | zh single CER | PCER | OCER | en single CER | PCER | OCER | zh multi CER | PCER | OCER | en multi CER | PCER | OCER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NVbench-GT（模型卡） | 3.86 | 9.38 | 4.07 | 6.73 | 8.31 | 6.90 | 3.79 | 23.71 | 5.07 | 7.26 | 21.41 | 8.62 |
| NVbench-GT（我们测） | 3.44 | 8.62 | 3.51 | 3.43 | 9.17 | 3.46 | 3.19 | 23.12 | 3.94 | 4.36 | 22.74 | 4.60 |
| Orpheus-TTS | 11.36 | 88.77 | 13.91 | 9.03 | 71.92 | 10.63 | 19.83 | 84.85 | 24.38 | 8.68 | 71.46 | 11.89 |
| SMIIP-NV-CV2 | 8.80 | 75.64 | 11.34 | 17.92 | 56.80 | 19.47 | 10.66 | 77.20 | 14.79 | 20.93 | 54.49 | 23.87 |
| Emilia-NV-CV2 | 5.05 | 40.00 | 6.64 | 12.50 | 55.30 | 13.21 | 5.54 | 48.74 | 8.09 | 11.71 | 60.28 | 14.63 |
| CosyVoice3 | 3.85 | 57.69 | 5.86 | 7.87 | 62.75 | 9.06 | 4.75 | 61.94 | 8.26 | 6.39 | 57.84 | 10.69 |
| NV-FlexiVoice | 6.98 | 31.08 | 8.15 | 11.88 | 50.43 | 13.21 | 8.20 | 39.37 | 10.39 | 9.60 | 51.32 | 13.76 |
| NV-CV3 | 3.80 | 27.69 | 4.90 | 8.33 | 46.13 | 9.44 | 3.44 | 30.04 | 4.84 | 6.70 | 47.13 | 10.10 |
| Ming-3.0-tts-0.5b | 2.70 | 22.80 | 3.20 | 7.60 | 37.00 | 8.90 | 3.50 | 36.60 | 8.10 | 7.90 | 47.30 | 10.40 |
| Ming-3.0-tts-tiny | 2.80 | 17.23 | 3.20 | 5.86 | 32.95 | 7.05 | 4.49 | 32.88 | 5.81 | 5.12 | 46.13 | 7.68 |
| dots.tts-soar（我们测，未训练） | 31.11 | 93.23 | 32.43 | 9.75 | 78.80 | 9.97 | 46.13 | 91.81 | 47.86 | 11.48 | 76.94 | 11.73 |
| dots refmask step 6,448（我们测） | 2.63 | 23.23 | 3.12 | 6.36 | 43.27 | 6.66 | 3.11 | 38.27 | 4.39 | 6.39 | 42.26 | 6.82 |
| dots refmask-noxvecdrop step 6,448（我们测） | 2.76 | 20.62 | 3.10 | 6.40 | 40.40 | 6.63 | 3.70 | 34.79 | 4.71 | 6.39 | 44.19 | 6.81 |

SIM（模型卡未提供 Ming 的 SIM；NV-CV3 取自 NV-Bench 论文表 4）：

| 模型 | zh single | zh multi | en single | en multi |
| --- | ---: | ---: | ---: | ---: |
| NV-CV3 | 0.768 | 0.776 | 0.698 | 0.721 |
| dots.tts-soar（未训练） | 0.719 | 0.713 | 0.780 | 0.807 |
| dots refmask step 6,448 | 0.777 | 0.784 | 0.733 | 0.763 |
| dots refmask-noxvecdrop step 6,448 | 0.773 | 0.782 | 0.733 | 0.767 |

- **未训练的 soar 不会副语言**：PCER 77–93%；它不认识标签，中文 CER 也被拖到 31–46%。
- **微调后中文单标签与 Ming-0.5b 持平，好于 NV-CV3**：zh_single PCER 23.2 / 20.6，对 NV-CV3 27.7、Ming-0.5b 22.8；中文 CER、OCER 是表中最低。仍落后 Ming-tiny（17.2）。
- **多标签句是短板**：zh_multi PCER 34.8–38.3，差于 NV-CV3 的 30.0 和 Ming-tiny 的 32.9。
- **英文**：训练没有英文 NV 数据，PCER 仍好于 NV-CV3，但 en_single 明显落后 Ming-tiny（40–43 对 33）。英文 SIM 从 0.78 / 0.81 降到 0.73 / 0.76，中文 SIM 则上升。
- **去掉 x-vector dropout**：PCER 在四个子集中有三个低 2.6–3.5 点（en_multi 高 1.9），zh_multi CER 高 0.6，SIM 持平。单次生成，差距不大，暂不定论。
- **逐标签召回（refmask，zh_single）**：Surprise-oh 94%，Dissatisfaction-hnn、Question-ei 92%，Laughter、Sigh 90%；Breathing、Surprise-ah 66%；Cough 42%，最差，训练集里 Cough 只有 3,761 条（Breathing 85,560 条）。人类录音各标签在 82–98%。

原始结果：[未训练 soar](../../../UltraEval-Audio/res/dots-studio-soar-default-sampling/nv_bench/results_table.md) · [refmask](../../../UltraEval-Audio/res/dots-soar-emilia-nv-refmask-step6448-default-sampling/nv_bench/results_table.md) · [refmask-noxvecdrop](../../../UltraEval-Audio/res/dots-soar-emilia-nv-refmask-noxvecdrop-step6448-default-sampling/nv_bench/results_table.md) · [人类录音](../../../UltraEval-Audio/res/nv-bench-human-reference/nv_bench/summary_gt.json)。

## 3. Seed-TTS 与 CV3（普通语音克隆，无标签）

用来判断微调是否伤到通用能力。speaker_only 只用 prompt 的 x-vector；speaker_icl 再加 prompt 文本和音频前缀。默认采样，超过 30 s 的输出单独排除。

refmask 的 Seed-TTS 首次运行因审计脚本按 24 kHz 校验 dots 的 48 kHz 输出而失败，已修复（UltraEval `a885ea9`）后重跑。全部评测已完成。

Seed-TTS（英文 WER / 中文 CER，SIM×100）：

| 模型 | 条件 | en WER | en SIM | zh CER | zh SIM |
| --- | --- | ---: | ---: | ---: | ---: |
| dots.tts-soar（未训练） | speaker_only | 2.151 | 72.694 | 0.904 | 77.234 |
| dots.tts-soar（未训练） | speaker_icl | 2.076 | 76.402 | 0.868 | 80.981 |
| dots refmask step 6,448 | speaker_only | 2.145 | 62.987 | 1.143 | 69.665 |
| dots refmask step 6,448 | speaker_icl | 2.169 | 72.756 | 1.260 | 76.860 |
| dots refmask-noxvecdrop step 6,448 | speaker_only | 2.619 | 61.438 | 1.250 | 68.927 |
| dots refmask-noxvecdrop step 6,448 | speaker_icl | 2.488 | 72.338 | 1.504 | 76.483 |

- **Seed-TTS 同样显示相似度明显下降**：speaker_only 降 7.6–11.3 点，speaker_icl 降 3.6–4.5 点；中文 CER 从 0.87–0.90 升到 1.14–1.50，英文 WER refmask 基本持平（2.15 / 2.17），noxvecdrop 升到 2.5–2.6。
- **去掉 x-vector dropout 没有挽回相似度**，内容准确率还略差。训练时始终给 x-vector 也照样下降，说明原因不在 dropout，更可能是训练用的参考段 x-vector 太短太噪（推理时的完整 prompt 向量反而被弱化），或全模型微调冲掉了 SOAR 只在 DiT 上做的相似度提升。下一轮可对照：冻结 DiT、提高参考段最短长度、降低学习率。

CV3（WER / CER 与 SIM×100；括号内为超过 30 s 被排除的条数）：

| 条件 | 子集 | 未训练 soar | | refmask step 6,448 | | refmask-noxvecdrop step 6,448 | |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| | | WER/CER | SIM | WER/CER | SIM | WER/CER | SIM |
| speaker_only | zero_shot_en | 3.995 | 75.970 | 4.789 | 65.914 | 4.752 | 65.495 |
| speaker_only | zero_shot_zh | 3.351 | 79.181 | 3.514 | 72.838 | 3.738 | 72.988 |
| speaker_only | hard_en | 7.420 (6) | 76.564 | 6.829 (1) | 68.116 | 6.669 | 67.562 |
| speaker_only | hard_zh | 11.038 (4) | 77.967 | 11.378 (4) | 71.703 | 11.512 (3) | 71.873 |
| speaker_icl | zero_shot_en | 4.262 | 75.626 | 4.950 | 71.453 | 5.162 | 71.292 |
| speaker_icl | zero_shot_zh | 3.899 | 79.727 | 4.428 | 76.121 | 4.169 | 75.677 |
| speaker_icl | hard_en | 9.823 (9) | 77.252 | 8.493 (1) | 72.505 | 7.794 (1) | 72.997 |
| speaker_icl | hard_zh | 9.958 (6) | 78.034 | 11.915 (2) | 72.893 | 11.032 (3) | 72.880 |

- **微调明显降低了说话人相似度**：refmask 的 SIM 在八项里全部下降，speaker_only 降 6.3–10.1 点（英文最多，75.97 → 65.91），speaker_icl 降 3.6–5.1 点。
- **noxvecdrop 与 refmask 在 CV3 上几乎一样**：SIM 差异在 ±0.6 点内，WER/CER 互有高低，同样没有挽回相似度。
- **内容准确率基本持平**：普通子集 WER/CER 高 0.2–0.8 点；hard 子集互有升降，且微调后超长输出（被排除）明显变少。
- speaker_only 的下降远大于 speaker_icl，说明模型对 x-vector 的依赖变弱了。可能的原因：训练时 x-vector 来自句内的短参考段（约三分之一短于 2 s），比推理时整段 prompt 算出的向量噪声大；全模型微调也可能冲掉了 SOAR（只调 DiT）带来的相似度。noxvecdrop（始终使用 x-vector）的结果可以帮助区分。

原始结果：[refmask Seed-TTS](../../../UltraEval-Audio/res/dots-soar-emilia-nv-refmask-step6448-default-sampling/seed_tts/full/results_table.md) · [noxvecdrop Seed-TTS](../../../UltraEval-Audio/res/dots-soar-emilia-nv-refmask-noxvecdrop-step6448-default-sampling/seed_tts/full/results_table.md) · [未训练 soar Seed-TTS](../../../UltraEval-Audio/res/dots-studio-soar-default-sampling/seed_tts/full/results_table.md) · [未训练 soar CV3](../../../UltraEval-Audio/res/dots-studio-soar-default-sampling/cv3/full/results_table.md) · [refmask CV3](../../../UltraEval-Audio/res/dots-soar-emilia-nv-refmask-step6448-default-sampling/cv3/full/results_table.md) · [noxvecdrop CV3](../../../UltraEval-Audio/res/dots-soar-emilia-nv-refmask-noxvecdrop-step6448-default-sampling/cv3/full/results_table.md)。

## 4. 第二轮：26 个标签、5 个 NV 数据集

### 数据与配置

- **数据**：在 Emilia-NV 之外加入 NonVerbalSpeech-38K（中英，122.5 h）、SMIIP-NV（中文，27.6 h）、DisfluencySpeech（英文，9.6 h）、NonverbalTTS（英文，只保留 Expresso 48 kHz 的 4.0 h），共约 753 h。各数据集的原生标签由数据侧统一成 `nv-canonical-v1` 文本（`tts-data-pipeline/docs/nv-tags.md`）：能对应的映射到原 18 个标签，其余新增 8 个标签 `[Sniff]` `[Throat-clearing]` `[Gasp]` `[Yawn]` `[Groan]` `[Snore]` `[Sneeze]` `[Mouth-noise]`（喷鼻息并入 Sniff，哼声并入 `[Dissatisfaction-hnn]`，咂嘴与吞咽合成 Mouth-noise）。SMIIP 的整段（边笑边说）标注、NonverbalTTS 的 16 kHz VoxCeleb 部分被排除；DisfluencySpeech 的 uh/um 保留为文字。
- **词表**：`assets/assembled/dots-tts-soar-nv26`（`configs/dots-soar-nv26.yaml`），原 18 个标签 ID 不变，新增 151690–151697。
- **build**：`dots-nv-canonical-v1-20261103`（`configs/data-nv-canonical-dots.yaml`）。build 校验每行 `text_kind == nv-canonical-v1`，且文本中出现的标签都必须是词表里的 token。
- **训练**：`configs/dots-tts-20261103-soar-nv26-5ds-refmask-noxvecdrop-bf16-8gpu-lr1e-5-ep2-wsd.yaml`，ACP `pt-7gbkk9q7`。从 soar 全模型训练，x-vector dropout 0，其余同第 1 节；2 epoch = 8,191 步，最终验证 FM 0.7499 / EOS 0.0363。

### NV-Bench

| 子集 | 第一轮 noxvecdrop（终） | 第二轮 step 3,500（≈1 epoch） | 第二轮 step 8,191（终） | NV-CV3 | Ming-tiny |
| --- | ---: | ---: | ---: | ---: | ---: |
| zh_single CER / PCER / OCER | 2.76 / 20.62 / 3.10 | 2.87 / 31.23 / 3.58 | 3.05 / **18.31** / 3.37 | 3.80 / 27.69 / 4.90 | 2.80 / 17.23 / 3.20 |
| zh_multi CER / PCER / OCER | 3.70 / 34.79 / 4.71 | 3.32 / 44.89 / 4.97 | 3.07 / 33.33 / 4.14 | 3.44 / **30.04** / 4.84 | 4.49 / 32.88 / 5.81 |
| en_single CER / PCER / OCER | 6.40 / 40.40 / 6.63 | 5.41 / 56.73 / 5.79 | 5.42 / 34.96 / 5.58 | 8.33 / 46.13 / 9.44 | 5.86 / **32.95** / 7.05 |
| en_multi CER / PCER / OCER | 6.39 / 44.19 / 6.81 | 6.82 / 50.00 / 7.42 | 6.24 / **39.19** / 6.63 | 6.70 / 47.13 / 10.10 | 5.12 / 46.13 / 7.68 |
| SIM zh_s / zh_m / en_s / en_m | 0.773 / 0.782 / 0.733 / 0.767 | 0.778 / 0.782 / 0.740 / 0.768 | 0.768 / 0.773 / 0.730 / 0.764 | 0.768 / 0.776 / 0.698 / 0.721 | — |

- 最终模型四个子集的 PCER 都好于第一轮（低 1.5–5.4 点）；en_multi 是表中所有模型最好，zh_single 与 Ming-tiny 只差 1.1 点，zh_multi 仍差 NV-CV3 约 3 点。
- 1 epoch 时明显落后，主要是 `[Cough]` 召回崩到 22%（中文）/ 18%（英文）；训满 2 epoch 后回到 86% / 60%，其余标签也回升。看中间 checkpoint 不能判断这类混合数据的效果。

### Seed-TTS 与 CV3

Seed-TTS（WER / CER，SIM×100）：

| 模型 | only en | only zh | icl en | icl zh |
| --- | --- | --- | --- | --- |
| dots.tts-soar（未训练） | 2.151 / 72.694 | 0.904 / 77.234 | 2.076 / 76.402 | 0.868 / 80.981 |
| 第一轮 noxvecdrop（终） | 2.619 / 61.438 | 1.250 / 68.927 | 2.488 / 72.338 | 1.504 / 76.483 |
| 第二轮 step 3,500 | 2.078 / 62.493 | 1.427 / 69.407 | 2.355 / 72.789 | 1.574 / 76.630 |
| 第二轮 step 8,191 | 2.560 / 60.871 | 1.225 / 68.464 | 2.107 / 72.070 | 1.424 / 75.942 |

CV3 普通子集（WER / CER，SIM×100）：

| 模型 | only en | only zh | icl en | icl zh |
| --- | --- | --- | --- | --- |
| dots.tts-soar（未训练） | 3.995 / 75.970 | 3.351 / 79.181 | 4.262 / 75.626 | 3.899 / 79.727 |
| 第一轮 noxvecdrop（终） | 4.752 / 65.495 | 3.738 / 72.988 | 5.162 / 71.292 | 4.169 / 75.677 |
| 第二轮 step 3,500 | 4.058 / 65.404 | 3.976 / 72.752 | 4.546 / 71.067 | 4.299 / 75.320 |
| 第二轮 step 8,191 | 4.269 / 64.777 | 4.274 / 72.024 | 5.163 / 70.104 | 4.079 / 75.101 |

- 加入 4 个 NV 数据集没有改善说话人相似度：与第一轮一样，speaker_only 比未训练的 soar 低 9–12 点，speaker_icl 低 4–5 点；最终模型还比 1 epoch 时略低。
- 内容准确率与第一轮相当：Seed-TTS 中文 CER 1.2–1.4，英文 WER 2.1–2.6。

原始结果：[step 3,500](../../../UltraEval-Audio/res/dots-soar-nv26-5ds-noxvecdrop-step3500-default-sampling/) · [step 8,191](../../../UltraEval-Audio/res/dots-soar-nv26-5ds-noxvecdrop-step8191-default-sampling/)。

## 5. 第三轮：混入去重后的游戏语音

为验证普通数据能否挽回说话人相似度，在第二轮数据上混入 4 个游戏语音数据集（原神、星穹铁道、绝区零、鸣潮；中英，ASR 文本，无标签）。

- **去重**（只在训练索引中排除，不修改任何 lance）：游戏样本若满足任一条件即排除：在数据侧给出的游戏↔Emilia-NV 宽松匹配表中（`tts-data-pipeline/reports/nv-game-overlap/loose_pairs.jsonl`，原神 5,783 / 星铁 5,670 条）；归一化文本完全一致（≥5 字）且时长差 ≤0.3 s；或 VAE latent 音频指纹余弦 ≥0.90（与全部 NV 数据集及排在前面的游戏比较）。指纹为标准化后的 VAE 后验均值按 4 个时间段平均。阈值用宽松匹配表的 11,777 对已知重复校准：其中位余弦 0.996，97.4% ≥0.90；而 NV 数据集之间不同录音的最高相似度 p99.9 仅 0.88–0.91。目的是同一段音频不同时以"带标签"和"不带标签"两种文本出现。
- **两个实验**（步数与第二轮相同，按 总小时 / NV 小时 放大每卡 dynamic batching 预算）：
  - 1:1：每个游戏按去重后小时占比截取，合计约 753 h；每卡 1,200 patch / 2,000 token；ACP `pt-e2sq5euo`。
  - 全部：去重后游戏数据 1,102.8 h（约 1.46:1）；每卡 1,480 patch / 2,470 token；ACP `pt-uk27o1b9`。
- 配置：`configs/dots-tts-20261103-soar-nv26-5ds-games-{1to1,all}-refmask-noxvecdrop-bf16-8gpu-lr1e-5-ep2-wsd.yaml`。两组都在 2 epoch 内训完：1:1 共 7,615 步（最终验证 FM 0.7443 / EOS 0.0314），全部共 7,534 步（0.7442 / 0.0294），均少于第二轮的 8,191 步。中间评测点 step 3,000 约为 epoch 0.8。

### NV-Bench PCER

| 子集 | 第二轮终 | 1:1 step 3,000 | 1:1 终 | 全部 step 3,000 | 全部 终 |
| --- | ---: | ---: | ---: | ---: | ---: |
| zh_single | **18.31** | 37.23 | 24.77 | 38.62 | 24.15 |
| zh_multi | **33.33** | 49.27 | 37.37 | 50.28 | 35.91 |
| en_single | **34.96** | 49.00 | 50.43 | 58.74 | 45.27 |
| en_multi | **39.19** | 50.32 | 50.16 | 51.45 | 42.58 |
| zh_single CER | 3.05 | 3.44 | 2.84 | 3.73 | **2.53** |
| SIM zh_single / en_single | **0.768 / 0.730** | 0.769 / 0.731 | 0.763 / 0.721 | 0.761 / 0.723 | 0.763 / 0.725 |

### Seed-TTS（WER / CER，SIM×100）

| 模型 | only en | only zh | icl en | icl zh |
| --- | --- | --- | --- | --- |
| dots.tts-soar（未训练） | 2.151 / **72.694** | 0.904 / **77.234** | 2.076 / **76.402** | **0.868** / **80.981** |
| 第二轮终 | 2.560 / 60.871 | 1.225 / 68.464 | 2.107 / 72.070 | 1.424 / 75.942 |
| 1:1 step 3,000 | 2.045 / 62.402 | 1.238 / 69.204 | 2.185 / 71.785 | 1.247 / 75.937 |
| 1:1 终 | 1.706 / 59.120 | 1.031 / 67.850 | **1.753** / 71.240 | 1.083 / 76.451 |
| 全部 step 3,000 | **1.603** / 57.572 | 1.047 / 64.688 | 1.897 / 71.426 | 1.125 / 76.145 |
| 全部 终 | 1.689 / 58.849 | 1.025 / 66.986 | 1.862 / 71.529 | 1.051 / 76.538 |

### CV3 普通子集（WER / CER，SIM×100）

| 模型 | only en | only zh | icl en | icl zh |
| --- | --- | --- | --- | --- |
| dots.tts-soar（未训练） | 3.995 / **75.970** | 3.351 / **79.181** | 4.262 / **75.626** | 3.899 / **79.727** |
| 第二轮终 | 4.269 / 64.777 | 4.274 / 72.024 | 5.163 / 70.104 | 4.079 / 75.101 |
| 1:1 step 3,000 | 4.057 / 65.664 | **3.162** / 72.618 | 4.755 / 70.862 | 3.744 / 75.792 |
| 1:1 终 | **3.052** / 63.454 | 3.252 / 71.889 | 4.274 / 69.434 | 3.699 / 75.045 |
| 全部 step 3,000 | 3.210 / 62.083 | 3.312 / 70.280 | 4.541 / 69.984 | 3.831 / 74.849 |
| 全部 终 | 3.366 / 63.423 | 3.358 / 70.494 | **4.117** / 69.596 | **3.606** / 74.883 |

### 结论

- **普通数据提高了读字准确率**：Seed-TTS 与 CV3 的 WER/CER 多数好于第二轮，部分好于未训练的 soar（如 1:1 终的 Seed-TTS 英文 1.71–1.75、CV3 英文 3.05）。
- **但削弱了副语言标签**：两组最终的 PCER 都差于第二轮，英文最明显（1:1 en_single 50.4 对 35.0）。可能是不带标签的英文游戏语音冲淡了英文 NV 样本，也可能游戏语音里有未标注的笑声、语气词，削弱了标签与发声的对应。全部数据组的最终 PCER 好于 1:1，原因待查。
- **说话人相似度没有回升，speaker_only 反而更低**：普通数据越多，speaker_only SIM 越低（Seed-TTS 英文：第二轮 60.9 → 1:1 59.1；全部组 step 3,000 已降到 57.6），speaker_icl 基本不变。这说明相似度下降不是缺少普通数据所致，更可能来自训练方式：训练用的 x-vector 来自句内的短参考段（多数 2–4 s），噪声大，数据越多模型越不依赖它；或全模型微调冲掉了 SOAR 只在 DiT 上做的相似度提升。可用冻结 DiT、或参考段最短 3 s 的短实验区分。

原始结果：[1:1 step 3,000](../../../UltraEval-Audio/res/dots-soar-nv26-games-1to1-step3000-default-sampling/) · [1:1 终](../../../UltraEval-Audio/res/dots-soar-nv26-games-1to1-step7615-default-sampling/) · [全部 step 3,000](../../../UltraEval-Audio/res/dots-soar-nv26-games-all-step3000-default-sampling/) · [全部 终](../../../UltraEval-Audio/res/dots-soar-nv26-games-all-step7534-default-sampling/)。


## 6. 评测怎么跑

UltraEval-Audio 中：`init_model/<run-short>-step-<8 位 step>` 为导出模型（`LM-TTS-Training/scripts/export_checkpoint.py` 直接导出为官方 runtime 布局），结果在 `res/<run-short>-step<N>-default-sampling/{nv_bench,seed_tts,cv3}/`。一个模型的全部评测作为一个 ACP 任务提交：

```bash
python -m scripts.acp.submit --name evalsuite-<run> --log-dir res/<run>/logs --submit -- \
  scripts/run_dots_suite.py --model-path init_model/<export> --run-dir res/<run>
```

dots 推理使用 `envs/dots`（官方 dots.tts @ `4cf7d6b`），打分沿用 `envs/metrics`。

## 7. 第四轮：冻结 DiT + LLM LoRA（2026-11-04）

冻结 DiT 让 Seed-TTS speaker_only SIM 有所回升（5ds：en 60.9 → 65.7，zh 68.5 → 70.3），但离未训练的 soar（72.7 / 77.2）仍差很远，speaker_icl 也只到 73.6 / 77.3。所以大部分相似度损失不在 DiT，而在它上游的 LLM / PatchEncoder。本轮在 5ds 冻结 DiT 的基础上，看限制 LLM 的更新幅度能否让 SIM 超过冻结 DiT 的结果。

| | 主实验 | 对照：再冻结 PatchEncoder |
| --- | --- | --- |
| DiT 及其条件投影 | 冻结 | 冻结 |
| LLM attention / MLP（`q/k/v/o_proj`、`gate/up/down_proj`） | LoRA r32、alpha 64、dropout 0 | 同左 |
| LLM embedding | 只训练新加的 26 行标签，原有 151,672 行冻结 | 同左 |
| PatchEncoder | 全量，lr 1e-5 | 冻结 |
| LLM norm、EOS head | 全量，lr 1e-5 | 同左 |

- 学习率：LoRA adapter 和 26 行标签是本轮新引入的参数，用 `train.adapter_lr` 1e-4、不加 weight decay；其余参数仍为 lr 1e-5、weight decay 0.01。数据、步数、WSD 调度都与 5ds frozenhead 相同。
- 配置：`configs/dots-tts-20261104-soar-nv26-5ds-refmask-noxvecdrop-frozenhead-{,frozenpe-}llmlora-r32-addedrows-bf16-8gpu-lr1e-5-adapterlr1e-4-ep2-wsd.yaml`。
- 实现：模型的 `adapt` 按 `train.lora` / `train.embedding_rows` 换入 adapter 层，B 初始化为 0，第 0 步输出与原模型完全一致。导出时按 checkpoint 签名重建这些层、加载权重，再用 `merge_adapters` 合并回原权重，导出结构和参数名与此前完全相同。
- 判断标准：Seed-TTS / CV3 的 speaker_only SIM 高于 5ds frozenhead 终点（Seed-TTS 65.7 / 70.3），NV-Bench PCER 不差于它（20.3 / 32.6 / 42.7 / 40.3）。两个实验都评 epoch 1 和终点。
- 已知风险：评测用 bf16 推理。合并后的 W + ΔW 转成 bf16 时，ΔW 中小于 bf16 精度的部分会被舍掉，而训练时 adapter 支路是单独计算的。导出后要实测合并前后 velocity 的差异，必要时改用 fp32 推理。
- 另一种解释，本轮区分不了：训练用的 x-vector 来自句内的短参考段，噪声较大。冻结 DiT 后，LLM 可能会转而从 teacher-forced 的历史里携带音色，speaker_only 推理时没有历史，SIM 就会下降。如果两个实验都没有明显回升，下一步先做不需要训练的诊断：把冻结 DiT 模型里的 LLM 或 PatchEncoder 换回 release 版本，或者把微调权重和 release 权重做插值。

### 结果

两个实验都跑完 8,191 步（训练 ACP `pt-ek4oegqy` / `pt-wr8afipj`），step 3,500 和 8,191 都评测完成。SIM×100，越高越好：

| 模型 | Seed only en / zh | Seed icl en / zh | CV3 only en / zh |
| --- | --- | --- | --- |
| Base（未微调 soar） | 72.7 / 77.2 | 76.4 / 81.0 | 76.0 / 79.2 |
| 5ds 冻结 DiT · 3,500 | 67.4 / 71.4 | 74.1 / 77.6 | 70.9 / 74.4 |
| 5ds 冻结 DiT · 8,191 | 65.7 / 70.3 | 73.6 / 77.3 | 69.5 / 73.6 |
| LLM LoRA · 3,500 | 66.9 / 71.1 | 73.9 / 77.5 | 70.4 / 73.9 |
| LLM LoRA · 8,191 | 66.0 / 70.8 | 73.7 / 77.7 | 69.6 / 73.7 |
| LLM LoRA + 冻结 PatchEncoder · 3,500 | 66.1 / 70.7 | 74.4 / 77.7 | 69.8 / 73.6 |
| LLM LoRA + 冻结 PatchEncoder · 8,191 | 65.6 / 70.9 | 73.9 / 77.9 | 69.5 / 73.6 |

NV-Bench（8,191 步；zh_single / zh_multi / en_single / en_multi）：

| 模型 | PCER（%）↓ | CER（%）↓ | SIM ↑ |
| --- | --- | --- | --- |
| 5ds 冻结 DiT | 20.3 / 32.5 / 42.7 / 40.3 | 2.86 / 2.83 / 5.88 / 5.74 | .786 / .794 / .760 / .794 |
| LLM LoRA | 18.5 / 32.0 / 38.4 / 41.3 | 2.33 / 2.50 / 5.05 / 5.75 | .786 / .795 / .757 / .787 |
| LLM LoRA + 冻结 PatchEncoder | 20.0 / 32.9 / 41.3 / 43.4 | 2.47 / 2.32 / 4.85 / 5.08 | .788 / .802 / .760 / .786 |

- **SIM 没有回升。** speaker_only SIM 只比冻结 DiT 高 0–0.5 点，仍比 Base 低 6–7 点；冻结 PatchEncoder 也没有帮助。三组都是 step 3,500 略好于终点。
- **PCER 和读字略好。** 主实验 NV-Bench PCER 四项中三项低于冻结 DiT，en_single 42.7 → 38.4；Seed-TTS 中文 CER 1.25 → 0.99。
- **bf16 合并没有吞掉更新。** 合并后的 LoRA 目标层中，只有 1–2% 的元素与 Base 完全相同；原有 151,672 行 embedding 逐元素不变，只有 26 行标签变化。

### 诊断：这组 LoRA 没有约束住更新

按导出权重计算 LLM 各投影层的相对变化 ‖W − W_base‖ / ‖W_base‖（Frobenius 范数，按层类型汇总）：

| 模型 | q / k / v / o | gate / up / down |
| --- | --- | --- |
| 5ds 冻结 DiT（全量，lr 1e-5） | 8.8% / 5.5% / 5.3% / 7.5% | 8.4% / 8.5% / 8.4% |
| LLM LoRA r32（adapter_lr 1e-4） | 17.5% / 12.5% / 9.4% / 14.4% | 17.3% / 16.4% / 11.0% |

LoRA 秩虽低，但 adapter_lr 1e-4、scale（alpha / rank）2 时，权重变化约为全量微调的 2 倍。所以本轮并没有真正检验"限制 LLM 更新能否保住 SIM"。另一方面，更新幅度差一倍、SIM 几乎相同，说明 SIM 下降和 LLM 改了多少关系不大，更符合上面"x-vector 噪声 / teacher-forced 历史"的解释。

### 第五轮：约束住的 LoRA（2026-11-05）

只改 LoRA 的幅度，其余与本节主实验相同：PatchEncoder 全量训练，只训练新加的 26 行标签。

- rank 32 → 16，alpha 64 → 32（scale 仍为 2）；`adapter_lr` 1e-4 → 2e-5，26 行标签随之也用 2e-5。
- 目标是把 LLM 投影层的相对变化压到全量微调的约 1/4（约 2%）。导出后先按上表方法测量；如果明显超出，再降 lr。
- 配置：`configs/dots-tts-20261104-soar-nv26-5ds-refmask-noxvecdrop-frozenhead-llmlora-r16-addedrows-bf16-8gpu-lr1e-5-adapterlr2e-5-ep2-wsd.yaml`。
- 判读：如果 speaker_only SIM 明显回升，说明 LLM 更新幅度确实是主因，再在 SIM 和 PCER 之间找合适的 lr；如果 SIM 仍停在 66 / 71 左右，就基本排除"更新太多"这个原因，下一步转向训练数据：改用同一说话人其他句子的 x-vector，并配合权重插值、换回 release 模块等不需要训练的诊断。

#### 结果：更新压住了，SIM 没有回升

训练 ACP `pt-9249dstr`，跑完 8,191 步。step 3,500 时 LLM 投影层相对变化为 1.3–2.3%，约为全量微调的 1/3、r32 的 1/5，幅度达到了目标。SIM×100：

| 模型 | Seed only en / zh | Seed icl en / zh | CV3 only en / zh |
| --- | --- | --- | --- |
| r16 · 3,500 | 67.8 / 71.8 | 74.2 / 77.8 | 71.3 / 74.9 |
| r16 · 8,191 | 66.5 / 70.9 | 74.2 / 78.0 | —（speaker_icl 分片在 8 GiB 上限下 OOM，待用 16 GiB 重跑） |

NV-Bench PCER（zh_single / zh_multi / en_single / en_multi）：step 3,500 为 35.7 / 47.1 / 58.7 / 49.8，step 8,191 为 23.7 / 35.6 / 49.0 / 50.2，都比冻结 DiT（20.3 / 32.5 / 42.7 / 40.3）差。

更新幅度只有 r32 的 1/5，speaker_only SIM 仍停在 66–68 / 71–72。可见 SIM 下降和 LoRA 更新幅度无关，LoRA 这条线到此为止。

### 不需要训练的诊断：换模块与权重插值（2026-11-05）

以下都在 Seed-TTS speaker_only 子集上测（en、zh 各 60 条，seed 42，num_steps 10，guidance 1.2），打分用 WavLM-large，与正式 simo 打分一致（误差约 0.0003）。微调模型为 5ds 冻结 DiT · 8,191。

| 变体 | SIM en / zh | 相对 release |
| --- | --- | --- |
| release | 74.2 / 77.7 | — |
| 全部微调（LLM + PatchEncoder + eos） | 65.5 / 70.0 | −8.7 / −7.7 |
| 只换微调 LLM | 65.6 / 68.5 | −8.6 / −9.2 |
| 只换微调 PatchEncoder | 74.2 / 77.8 | 0 / 0 |
| 全部微调，speaker_scale 1.5 → 3.0 | 66.8 / 70.0 | −7.4 / −7.7 |
| release，speaker_scale 3.0 | 74.1 / 77.0 | — |

- **SIM 损失全部来自 LLM**，PatchEncoder 微调无害。
- **不是 x-vector 权重不够**：放大 speaker_scale 救不回来。
- 逐句配对分析：82–95% 的句子 SIM 都下降，与 prompt 长度无关，是整体偏移。

于是对 LLM 做 WiSE-FT 插值：W(α) = W_release + α·(W_ft − W_release)；26 行标签、PatchEncoder、eos_proj 直接取微调值，DiT 本来就冻结。NV-Bench 用全集每 4 行取 1 行的子集（414 条，单个子集的 PCER 误差约 ±3–5 点）：

| 模型 | Seed only SIM en / zh | NV PCER zs / zm / es / em ↓ | NV CER ↓ | NV SIM ↑ |
| --- | --- | --- | --- | --- |
| release（α = 0） | 74.2 / 77.7 | 95.1 / 91.1 / 81.8 / 73.5 | 28.65 / 48.39 / 10.55 / 9.99 | .715 / .695 / .795 / .816 |
| α = 0.5 | 74.6 / 76.6 | 28.8 / 37.9 / 50.0 / 53.7 | 1.93 / 3.03 / 5.79 / 3.77 | .803 / .807 / .788 / .810 |
| α = 0.7 | 72.3 / 74.1 | 22.1 / 35.3 / 39.8 / 42.6 | 2.23 / 2.88 / 5.96 / 3.88 | .792 / .798 / .779 / .807 |
| 冻结 DiT · 3,500 | — | 35.6 / 42.4 / 52.3 / 49.4 | 4.01 / 3.79 / 5.30 / 5.69 | .777 / .781 / .771 / .783 |
| 冻结 DiT · 8,191（α = 1） | 65.5 / 70.0 | 19.6 / 28.1 / 47.7 / 39.5 | 2.72 / 2.90 / 6.32 / 4.09 | .783 / .789 / .765 / .800 |

- **插值的效果是非线性的。** α = 0.7 拿回了大部分 PCER 改善，zm 以外与 α = 1 基本持平；Seed SIM 只比 release 低 2–3.6 点，而 α = 1 要低 8–9 点。
- **插值点在训练步数的权衡曲线外侧。** α = 0.5 的 PCER 三项、SIM 全部优于 step 3,500，说明插值不等于少训几步。
- **还不确定的地方：** 子集评测，单个 seed，没有 CV3。下一步拿 α = 0.6 / 0.7 / 0.8 去跑全集评测（NV-Bench、Seed-TTS、CV3）。

#### 全集评测

导出在 `UltraEval-Audio/init_model/dots-soar-nv26-5ds-frozenhead-8191-wise-a{0.6,0.7,0.8}`，`export.json` 里记录了 α 和两端模型；ACP 任务为 `pt-2gxn7y4q` / `pt-2prw10tr` / `pt-o7c70su4`，CV3 另行提交，显存上限 16 GiB。SIM×100：

| 模型 | Seed only en / zh | Seed icl en / zh | CV3 only en / zh | Seed CER en / zh |
| --- | --- | --- | --- | --- |
| release（α = 0） | 72.7 / 77.2 | 76.4 / 81.0 | 76.0 / 79.2 | 2.15 / 0.90 |
| α = 0.6 | 71.7 / 74.8 | 76.0 / 79.2 | 74.6 / 77.1 | 2.37 / 0.92 |
| α = 0.7 | 70.6 / 73.6 | 75.3 / 78.6 | 73.5 / 75.6 | 2.19 / 0.96 |
| α = 0.8 | 69.1 / 72.3 | 74.7 / 78.1 | 72.2 / 74.8 | 1.85 / 1.07 |
| 冻结 DiT · 3,500 | 67.4 / 71.4 | 74.1 / 77.6 | 70.9 / 74.4 | 1.76 / 1.27 |
| 冻结 DiT · 8,191（α = 1） | 65.7 / 70.3 | 73.6 / 77.3 | 69.5 / 73.6 | 1.89 / 1.25 |
| r16 LoRA · 8,191 | 66.5 / 70.9 | 74.2 / 78.0 | 70.4 / 74.3 | 2.02 / 0.89 |

NV-Bench 全集（zh_single / zh_multi / en_single / en_multi）：

| 模型 | PCER（%）↓ | CER（%）↓ | SIM ↑ |
| --- | --- | --- | --- |
| release（α = 0） | 93.2 / 91.8 / 78.8 / 76.9 | 31.11 / 46.13 / 9.75 / 11.48 | .719 / .713 / .780 / .807 |
| α = 0.6 | 24.2 / 36.6 / 46.7 / 45.7 | 2.37 / 2.49 / 4.95 / 5.38 | .798 / .807 / .775 / .800 |
| α = 0.7 | 20.2 / 37.4 / 38.4 / 45.5 | 2.37 / 2.57 / 5.19 / 5.82 | .798 / .804 / .774 / .798 |
| α = 0.8 | 19.7 / 33.9 / 39.8 / 42.6 | 2.43 / 2.73 / 5.06 / 6.05 | .793 / .802 / .765 / .798 |
| 冻结 DiT · 3,500 | 30.2 / 43.2 / 50.7 / 47.6 | 4.47 / 3.80 / 5.18 / 7.17 | .785 / .790 / .762 / .779 |
| 冻结 DiT · 8,191（α = 1） | 20.3 / 32.6 / 42.7 / 40.3 | 2.86 / 2.83 / 5.88 / 5.74 | .787 / .794 / .760 / .794 |
| r16 LoRA · 8,191 | 23.7 / 35.6 / 49.0 / 50.2 | 2.34 / 2.28 / 5.34 / 4.98 | .785 / .797 / .757 / .786 |

- **α = 0.8 几乎全面优于 α = 1。** PCER 四项平均 34.0，与 α = 1 持平：en_single 低 2.9，en_multi 高 2.3。Seed speaker_only SIM 高 3.4 / 2.0，CV3 高 2.7 / 1.2，NV CER 也更低。
- **α = 0.7：** PCER 平均 35.4，比 α = 1 高 1.4 点；zh_multi 和 en_multi 差 5 点，en_single 反而好 4 点。SIM 高 4.9 / 3.4。
- **α = 0.6：** PCER 平均 38.3，比 α = 1 高 4.3 点。Seed SIM 离 release 只差 1.0 / 2.4，CV3 差 1.4 / 2.1。
- 三个插值点在 PCER 和 SIM 上都全面优于 step 3,500 和 r16 LoRA。插值是目前 SIM 和 PCER 权衡里最好的手段。
- 子集预测的趋势在全集上成立，幅度略小：全集上 α = 0.7 的 Seed SIM 是 70.6 / 73.6，子集是 72.3 / 74.1。

#### 稀疏任务向量：TIES 与 DARE（2026-11-06）

插值是把整个更新 δ = W_ft − W_release 按比例缩小。这里换一种做法：只保留 δ 的一部分元素。作用范围与插值相同：LLM 主体逐个 tensor 稀疏化；26 行标签、PatchEncoder、eos_proj 直接取微调值；DiT 本来就冻结。

- **TIES k：** 每个 tensor 只保留 |δ| 最大的 k 比例元素，其余置零，不重新缩放。这里只有一个任务向量，TIES 的符号选举不起作用，只剩 trim 这一步。
- **DARE p：** 每个元素以概率 p 丢弃，保留下来的除以 (1 − p)，期望上等于完整的 δ（随机种子 0）。

导出在 `UltraEval-Audio/init_model/dots-soar-nv26-5ds-frozenhead-8191-{ties-0.1,ties-0.3,dare-0.7,dare-0.9}`，`export.json` 的 `sparse_task_vector` 字段记录了方法、参数、两端模型和实际保留比例。导出脚本 `export_sparse.py` 是一次性脚本，没有入库。只跑了 NV-Bench 和 Seed-TTS，没有跑 CV3。

Seed-TTS 全集（SIM×100）：

| 模型 | 保留元素 | only SIM en / zh | icl SIM en / zh | only WER / CER en / zh |
| --- | --- | --- | --- | --- |
| release（α = 0） | 0 | 72.7 / 77.2 | 76.4 / 81.0 | 2.15 / 0.90 |
| TIES 0.1 | 10.3% | 72.6 / 76.2 | 76.7 / 80.1 | 1.93 / 0.96 |
| α = 0.6 | 100% | 71.7 / 74.8 | 76.0 / 79.2 | 2.37 / 0.92 |
| TIES 0.3 | 29.1% | 69.3 / 72.3 | 74.5 / 78.1 | 1.97 / 1.08 |
| α = 0.8 | 100% | 69.1 / 72.3 | 74.7 / 78.1 | 1.85 / 1.07 |
| DARE 0.9 | 8.6% | 66.3 / 70.9 | 73.7 / 77.7 | 2.64 / 1.42 |
| DARE 0.7 | 25.9% | 65.9 / 70.4 | 73.7 / 77.5 | 2.06 / 1.23 |
| 冻结 DiT · 8,191（α = 1） | 100% | 65.7 / 70.3 | 73.6 / 77.3 | 1.89 / 1.25 |

NV-Bench 全集（zh_single / zh_multi / en_single / en_multi）：

| 模型 | PCER（%）↓ | PCER 平均 ↓ | CER（%）↓ | SIM ↑ |
| --- | --- | --- | --- | --- |
| release（α = 0） | 93.2 / 91.8 / 78.8 / 76.9 | 85.2 | 31.11 / 46.13 / 9.75 / 11.48 | .719 / .713 / .780 / .807 |
| TIES 0.1 | 31.5 / 43.0 / 63.3 / 56.9 | 48.7 | 2.35 / 2.71 / 5.03 / 4.79 | .805 / .814 / .782 / .804 |
| α = 0.6 | 24.2 / 36.6 / 46.7 / 45.7 | 38.3 | 2.37 / 2.49 / 4.95 / 5.38 | .798 / .807 / .775 / .800 |
| TIES 0.3 | 19.2 / 33.3 / 39.3 / 42.1 | 33.5 | 2.35 / 2.72 / 5.05 / 6.05 | .791 / .802 / .766 / .797 |
| α = 0.8 | 19.7 / 33.9 / 39.8 / 42.6 | 34.0 | 2.43 / 2.73 / 5.06 / 6.05 | .793 / .802 / .765 / .798 |
| DARE 0.9 | 19.7 / 34.0 / 42.7 / 43.1 | 34.9 | 3.16 / 3.44 / 6.24 / 6.56 | .791 / .799 / .763 / .794 |
| DARE 0.7 | 20.5 / 34.3 / 39.0 / 38.9 | 33.2 | 2.59 / 2.98 / 5.38 / 6.06 | .789 / .797 / .756 / .797 |
| 冻结 DiT · 8,191（α = 1） | 20.3 / 32.6 / 42.7 / 40.3 | 34.0 | 2.86 / 2.83 / 5.88 / 5.74 | .787 / .794 / .760 / .794 |

- **DARE 收不回 SIM。** 0.7 和 0.9 的 Seed SIM 与 α = 1 相差不到 0.7 点，PCER 也持平。DARE 0.9 只留 8.6% 的元素，读字反而变差：Seed WER / CER 2.64 / 1.42，NV CER 四项都是最高的。DARE 期望上还原完整的 δ，音色损失也一起保留了下来。说明这个损失不集中在少数元素上，而是分散在整个 δ 里，随机丢掉一部分再放大，并不能去掉它。
- **TIES 0.3 与 α = 0.8 几乎重合。** Seed SIM、PCER、NV CER 和 NV SIM 都相差不到 0.6 点。TIES 不重新缩放，实际效果等于把 δ 缩小。结果只看缩小了多少，挑哪些元素保留并没有带来额外收益。
- **TIES 0.1 落在新的权衡点上。** Seed SIM 72.6 / 76.2，ICL 76.7 / 80.1，与 release 只差 0.1 / 1.1 和 +0.2 / −0.9，比 α = 0.6 高 0.9 / 1.4；NV SIM 是所有变体里最高的。代价是 PCER 平均 48.7，比 α = 0.6 高 10.4，en_single 达 63.3。与 step 3,500 相比，PCER 高 5.8，SIM 高 5.3 / 4.7。
- **结论：** 稀疏化没有跳出插值的权衡曲线，插值仍是最简单的选择。要判断 TIES 有没有优势，需要在相同 PCER 下比较，例如 TIES 0.15–0.2 对 α = 0.6，并补跑 CV3。目前都是单个 seed。

#### 只换 26 行标签 embedding（全集）

release 的 LLM 主体、PatchEncoder、eos_proj 不动，只把 26 行标签 embedding 换成冻结 DiT · 8,191 的值（`init_model/dots-soar-nv26-5ds-frozenhead-8191-swap-tags`，`export.json` 的 `swap` 字段记录换了哪些部分）：

- **Seed-TTS 与 release 一致**，差异在 0.01 以内（only SIM 72.7 / 77.2，WER / CER 2.15 / 0.90）。无标签的输入不会用到这些行，这可以作为导出正确性的检查。
- **NV-Bench 上标签基本发不出声：** PCER 92.3 / 91.3 / 94.3 / 93.2。但 CER 降到 2.80 / 4.10 / 4.95 / 4.54，而 release 是 31.11 / 46.13 / 9.75 / 11.48。注意 release 那一行是原版 soar，标签按普通文字切词，会被读出来。换上微调后的标签行，标签不再被当成文字读出；但要真正发出副语言声音，还需要微调后的 LLM 主体。
- 只换 LLM 主体（`swap-llm`、`swap-llm-notags`）和只换 PatchEncoder（`swap-pe`）的全集评测，按要求中途停掉了。只有 NV-Bench 的部分推理结果，没有打分。这几种组合的结论见上面的子集表。

#### 音色在生成过程中漂移

Seed speaker_only 子集里，取所有变体生成时长都 ≥ 4 s 的句子（en 20 条、zh 50 条），把每句切成三段，各段分别和完整的 prompt 算 SIM（WavLM-large，SIM×100；片段越短，绝对值越低，只适合同一列内比较）：

| 模型 | 前 1 s | 前 1/3 | 中 1/3 | 后 1/3 | 整句 |
| --- | --- | --- | --- | --- | --- |
| release · en | 56.4 | 64.4 | 65.5 | 63.0 | 74.9 |
| α = 0.7 · en | 55.2 | 63.4 | 62.1 | 58.1 | 72.3 |
| ft · en | 52.3 | 59.6 | 59.2 | 57.0 | 69.0 |
| release · zh | 56.5 | 69.3 | 69.0 | 66.8 | 78.3 |
| α = 0.7 · zh | 54.5 | 65.0 | 62.7 | 61.8 | 74.0 |
| ft · zh | 52.1 | 62.1 | 57.1 | 57.3 | 70.0 |

ft 相对 release：en 依次 −4.8 / −6.3 / −6.0，zh 依次 −7.2 / −11.9 / −9.5；前 1 s 只低约 4 点。只换 PatchEncoder 时，各段都和 release 持平。只换 LLM（release PatchEncoder）时，后 1/3 段塌到 49.7 / 38.7，说明 PatchEncoder 要和 LLM 配套。

损失从开头就有，但主要在后面累积，说明不是"开头不会起音"，而是音色边生成边漂移。另外 speaker_icl 只比 release 低约 3 点，speaker_only 要低 7–8 点。两者合起来，更符合这样的解释：微调后的 LLM 决定音色时更依赖音频历史。训练时历史是真实音频（参考段在句中，不计 loss，但在输入里）；推理时历史是自己生成的，偏差会越滚越大。

#### 输出音色是否向同一个声音坍缩

同一子集（en、zh 各 60 条，每条的 prompt 来自不同说话人），对每条输出取 WavLM-large 说话人 embedding，计算两类余弦相似度。一是不同 prompt 的输出两两之间的平均值：模型如果只按 x-vector 还原各自的音色，这个值应接近 prompt 之间本身的相似度，微调后升高就说明输出在向某个共同成分靠。二是各输出和自己 prompt 的相似度（SIM）。"和别句 prompt"指每条输出和同语种其他 prompt 的平均相似度。

| 模型 | 输出之间 en / zh | 和自己 prompt en / zh | 和别句 prompt en / zh | 和 release 输出 en / zh |
| --- | --- | --- | --- | --- |
| release | .095 / .234 | .739 / .779 | .084 / .227 | 1 / 1 |
| α = 0.5 | .096 / .249 | .747 / .765 | .084 / .232 | .891 / .882 |
| α = 0.7 | .088 / .261 | .722 / .739 | .078 / .235 | .842 / .854 |
| ft（α = 1） | .100 / .288 | .655 / .701 | .080 / .239 | .766 / .808 |
| 只换 ft 的 LLM | .100 / .305 | .657 / .686 | .080 / .242 | .761 / .789 |
| 只换 ft 的 PatchEncoder | .095 / .233 | .741 / .777 | .083 / .227 | .919 / .936 |

- **英文没有坍缩。** 输出之间的相似度不变，和自己 prompt 的相似度在掉：每条都偏离了自己的 prompt，但方向各不相同，更像音色被打乱。
- **中文部分坍缩。** 输出之间的相似度从 .234 升到 .288，插值可以按比例拉回。幅度只占 SIM 损失的一部分。
- **局限。** "共同成分"不一定是某个说话人，也可能是 5ds 中文数据的录音条件或风格；中文基线（.234）远高于英文（.095），说明这个 embedding 混有语种、录音这类非说话人因素。每个语种只有 60 条。要确认是不是"拉向数据里的音色"，还要比较 ft 输出的平均 embedding 是否明显靠近 5ds 训练集的平均 embedding（尚未做）。

#### 逐层线性探针：LLM hidden 里的说话人与标签信息（2026-11-06）

用 ref3s build 的验证集（`dots-nv-canonical-v1-ref3s-20261106`，不超过 400 patch 的 2,301 条），release 和 ft（冻结 DiT · 8,191）各做一次 teacher-forced 前向。basic 布局，latent 取 posterior mean。每层在预测音频的位置上对 hidden 取均值，每句得到一个向量（第 0 层是 embedding 输出）。按 3:1 随机划分训练和测试，做 ridge 回归：

- **说话人：** 回归到该句参考段的 x-vector，指标为测试集上的平均余弦。
- **标签：** 先把 26 个标签从文本里删掉，这样标签信息只能来自音频历史；回归到标签 multi-hot，指标为 ROC AUC，只算正例不少于 20 条的 9 个标签，取平均。

| 层 | 说话人余弦 release | 说话人余弦 ft | 标签 AUC release | 标签 AUC ft |
| --- | --- | --- | --- | --- |
| 0 | .556 | .543 | .769 | .767 |
| 4 | .596 | .580 | .813 | .799 |
| 8 | .611 | .595 | .836 | .826 |
| 12 | .605 | .593 | .848 | .851 |
| 16 | .592 | .584 | .860 | .867 |
| 17 | .586 | .578 | .857 | .868 |
| 20 | .591 | .583 | .860 | .857 |
| 24 | .589 | .591 | .841 | .850 |
| 28 | .586 | .588 | .817 | .833 |

- **没有看到"微调后 hidden 里说话人信息更多"。** 第 0–20 层，ft 的说话人余弦比 release 低 0.008–0.018；第 21–23 层差距缩小到 0.005 以内，第 24 层以后高 0.002–0.004。
- **标签信息往中后层移。** ft 在第 0–11 层略低，第 12 层以后除第 20 层外都更高（第 17 层 +0.011，第 28 层 +0.016），符合训练目标：去掉文本标签后，中后层更能从音频历史里读出标签。
- **局限：** 线性探针、均值池化、teacher-forced（历史是真实音频，不是自己生成的），差异都在 0.02 以内，只做了一次划分，没有误差估计。它不能排除"推理时依赖生成历史"这种解释，只说明微调没有让线性可读的说话人表示变多。脚本 `layer_probe.py` 未入库。

#### 原因分析与候选方案

- **结构本身没有问题。** CAM++ x-vector 只进 DiT，LLM 看不到它，音色线索只能从音频历史里来。release 也是这个结构、同样的自回归推理，speaker_only SIM 却有 72.7 / 77.2。所以问题出在 5ds 微调让 LLM 开始从历史里带音色，而不是 speaker_only 这种用法注定不行。
- **候选原因（未验证）：** 一是参考段最短 0.64 s，x-vector 噪声大，冻结的 DiT 拿不准音色时，loss 会推着 LLM 用历史里的真实音色去补；二是数据里说话人集中，中文出现部分坍缩。
- **排除掉的方案：**
  - x-vector 改用整句计算：会泄漏，参考段 mask 本来就是为了防这个。
  - 把参考段从输入里裁掉：参考段多在句中，文本和音频会对不上；数据里没有字级时间戳，也没法同步裁掉对应文本。
  - 参考段改成 prefix：训练格式等于 ICL，但 speaker_only 推理时 LLM 看不到任何真实音频，这个不一致仍然在；而且要按新规则重算 `reference_xvector`（tts-data-pipeline `dots_vae`）。
- **候选方案：**
  1. 训练时给输入里的历史 patch 加噪声，让历史音色不可靠，迫使音色交给 x-vector；只改训练代码，扫 2–3 档强度，用上面的分段 SIM 检查后段漂移是否减小。
  2. 只在离标签有一段距离的普通位置上，把 hidden 向冻结的 release LLM 蒸馏，并配一个全位置约束的对照组。
  3. 诊断后再决定：把参考段最短长度提到约 3 s（需重算 x-vector），或者按说话人重新加权采样。

### 第六轮：参考段至少 3 s（2026-11-06）

- **为什么做：** v1 规则下参考段很短。各数据集的中位数为 1.9–2.9 s，一半以上不足 3 s，smiip_nv 达 82%。在 smiip_nv 抽查的 128 条里，参考段 x-vector 和整句 x-vector 的余弦平均只有 0.743；v1 参考段不足 1.5 s 的 60 条只有 0.658。x-vector 不准，冻结的 DiT 就拿不准音色，loss 可能推着 LLM 去从音频历史里补音色。这与前面看到的现象一致：SIM 损失全在 LLM，越往后越大，ICL 掉得少。
- **改了什么：** 新规则 `dots-reference-v2:min-3s-on-patches`。最短长度 lo = max(lo, min(19 patch = 3.04 s, 半句))，其余与 v1 相同（长度均匀取、起点均匀取，参考段不计 loss）。不足 6 s 的句子取半句作参考，有参考段的行与 v1 完全相同。同样 128 条上，新参考段的余弦升到 0.885，不足 1.5 s 的那 60 条升到 0.882。整句 x-vector 与 v1 逐条一致；latent 的差异不超过 1.5e-5，属于 fp32 批次舍入级。代价是被 mask 的 patch 更多，短句一半不计 loss。
- **数据：** tts-data-pipeline `2a7a4f5`、`650580c`。dots_vae profile 为 `dots-tts-soar-vae-xvec-48k-fp32-v2-ref3s`，run id `<dataset>-dots_vae-dots-tts-soar-vae-xvec-48k-fp32-v2-ref3s-20261106T092453bjt-01`，selection 和文本与 v1 相同。提取任务 ACP `pt-i1xc8qih`。
- **训练：** `configs/dots-tts-20261106-soar-nv26-5ds-ref3s-noxvecdrop-frozenhead-bf16-8gpu-lr1e-5-ep2-wsd.yaml`，build `dots-nv-canonical-v1-ref3s-20261106`。相对冻结 DiT 的 5ds 实验只改参考段规则。`ref3s_chain.py`（`/workspace/LM-TTS-Training-Runs/auto-chain/`）依次完成：等特征发布、建 build、提交训练，并在 step 3,500 和最终 checkpoint 自动评测。
- **判读：**
  - speaker_only SIM 明显回升，分段 SIM 里后段掉得少：说明 x-vector 噪声是主因。
  - SIM 不变：排除这个原因，转向给历史 patch 加噪声，或者做按位置约束的正则。
  - 无论哪种结果，都可以再和插值叠加。

#### 结果

评测 ACP 为 `pt-jrag711d`（step 3,500）和 `pt-j0dlwtjw`（step 8,191），导出在 `init_model/dots-soar-nv26-5ds-ref3s-frozenhead-step-0000{3500,8191}`，结果在 `res/dots-soar-nv26-5ds-ref3s-frozenhead-step{3500,8191}-default-sampling/`。对照组"v1"是同一配置、参考段规则为 v1 的冻结 DiT 5ds 实验。SIM×100：

| 模型 | Seed only en / zh | Seed icl en / zh | CV3 only en / zh | CV3 icl en / zh | CV3 hard only en / zh |
| --- | --- | --- | --- | --- | --- |
| release | 72.7 / 77.2 | 76.4 / 81.0 | 76.0 / 79.2 | 75.6 / 79.7 | 76.6 / 78.0 |
| v1 · 3,500 | 67.4 / 71.4 | 74.1 / 77.6 | 70.9 / 74.4 | 73.4 / 77.5 | 73.2 / 73.3 |
| ref3s · 3,500 | 70.1 / 73.6 | 74.9 / 77.8 | 73.2 / 76.4 | 74.8 / 78.0 | 75.2 / 74.3 |
| v1 · 8,191 | 65.7 / 70.3 | 73.6 / 77.3 | 69.5 / 73.6 | 73.2 / 77.0 | 71.6 / 71.8 |
| ref3s · 8,191 | 69.4 / 73.4 | 74.0 / 77.6 | 72.6 / 76.1 | 73.6 / 77.5 | 74.1 / 75.4 |
| v1 · 8,191 插值 α = 0.8 | 69.1 / 72.3 | 74.7 / 78.1 | 72.2 / 74.8 | 74.1 / 77.7 | 72.9 / 73.7 |

错误率与 NV-Bench（zh_single / zh_multi / en_single / en_multi）：

| 模型 | Seed only WER / CER en / zh | NV PCER（%）↓ | PCER 平均 | NV CER（%）↓ | NV SIM ↑ |
| --- | --- | --- | --- | --- | --- |
| v1 · 3,500 | 1.76 / 1.27 | 30.2 / 43.2 / 50.7 / 47.6 | 42.9 | 4.47 / 3.80 / 5.18 / 7.17 | .785 / .790 / .762 / .779 |
| ref3s · 3,500 | 1.94 / 1.40 | 40.3 / 45.7 / 51.6 / 50.2 | 46.9 | 3.77 / 4.96 / 5.43 / 5.87 | .782 / .787 / .757 / .791 |
| v1 · 8,191 | 1.89 / 1.25 | 20.3 / 32.6 / 42.7 / 40.3 | 34.0 | 2.86 / 2.83 / 5.88 / 5.74 | .787 / .794 / .760 / .794 |
| ref3s · 8,191 | 2.44 / 1.17 | 22.6 / 33.9 / 37.8 / 39.4 | 33.4 | 2.74 / 3.01 / 6.04 / 6.86 | .787 / .801 / .757 / .788 |
| v1 · 8,191 插值 α = 0.8 | 1.85 / 1.07 | 19.7 / 33.9 / 39.8 / 42.6 | 34.0 | 2.43 / 2.73 / 5.06 / 6.05 | .793 / .802 / .765 / .798 |

- **speaker_only SIM 明显回升，ICL 基本不变。** 与 v1 相比，step 8,191 的 Seed only 高 3.7 / 3.2，CV3 only 高 3.1 / 2.4，CV3 hard only 高 2.5 / 3.6；ICL 只高 0.3–0.5（CV3 hard zh 高 1.3）。Seed only 与 release 的差距从 7.0 / 7.0 缩小到 3.3 / 3.8，大约收回一半。按上面的判读，参考段 x-vector 噪声是主要原因之一，但不是全部。分段 SIM 还没有测。
- **副语言能力没有受影响。** 最终 PCER 平均 33.4，v1 是 34.0：英文两项更好，中文两项略差。step 3,500 的 PCER 比 v1 高 4 点，可能是被 mask 的 patch 更多、有效监督更少，所以学得慢一些。
- **英文读字略差。** Seed only 英文 WER 1.89 → 2.44，CV3 only 英文 4.68 → 4.88，NV 英文 CER 也略高；中文 Seed CER 1.25 → 1.17。原因还没有查。
- **与 v1 插值 α = 0.8 相比：** PCER 相当；speaker_only SIM 略高（Seed +0.2 / +1.1，CV3 +0.5 / +1.2，CV3 hard +1.2 / +1.7），ICL 略低（Seed −0.7 / −0.5）。两种做法互不冲突，可以叠加。
- **下一步：** 对 ref3s · 8,191 做插值（α = 0.7 / 0.8）；用分段 SIM 检查后段漂移是否减小。

## 8. 冻结 DiT 最终评测汇总（2026-11-04）

本节只比较最终 checkpoint；5ds 为第 4 节的五个 NV 数据集，games 1:1 为第 5 节混入等时长游戏语音的版本。Base 指未微调的 `dots.tts-soar`。两组 frozenhead 都从原始 soar-nv26 组装权重开始，训练 2 epochs，冻结 `velocity_field_predictor`（DiT）和 `hidden_proj`、`latent_proj`、`coordinate_proj`、`xvec_proj`；LLM、PatchEncoder 和 EOS head 仍全量训练。其余训练设置沿用对应未冻结实验。

### 训练与完成状态

| 实验 | 训练 ACP | 最终 step | 验证 FM / EOS / 总 loss | 最终评测 ACP | 评测完成时间（UTC） |
| --- | --- | ---: | --- | --- | --- |
| 5ds 冻结 | `pt-rfqo5avq` | 8,191 | 0.7532 / 0.0267 / 0.7799 | `pt-2mwh19i2` | 2026-11-04 08:33:49 |
| games 1:1 冻结 | `pt-k7ephanh` | 7,615 | 0.7483 / 0.0291 / 0.7774 | `pt-bvul6so7` | 2026-11-04 09:01:16 |

截至 2026-11-04 09:01:16 UTC，两组最终评测均为 `SUCCEEDED`：NV-Bench、Seed-TTS、CV3 全部完成；Seed-TTS 和 CV3 的两种条件均通过完整性检查。

中间点 games 1:1 frozenhead step 3,000 的原评测 `pt-pkgou6e2` 在 CV3 的 `speaker_icl` shard 27 触发单进程 8 GiB 显存上限，未影响训练或已完成的 NV-Bench、Seed-TTS。将上限调到 16 GiB 后，仅续跑 CV3，复用已有输出；补跑 `pt-ef7296gl` 于 2026-11-04 08:47:07 UTC 成功完成。中间点不列入以下最终结果对比。

### NV-Bench：CER / PCER / OCER

每格为 **CER / PCER / OCER（%，均越低越好）**。沿用第 2 节的字符、标签和混合序列口径。Ming 0.5B 数值复用第 2 节收录的模型卡结果，并非本地复测；英文文本归一化可能不同，跨来源英文 CER / OCER 不宜直接判优，PCER 的可比性相对更好。

| 模型 | zh_single | zh_multi | en_single | en_multi |
| --- | --- | --- | --- | --- |
| Ming-3.0-tts-0.5b（模型卡） | 2.70 / 22.80 / 3.20 | 3.50 / 36.60 / 8.10 | 7.60 / 37.00 / 8.90 | 7.90 / 47.30 / 10.40 |
| Base（未微调 soar） | 31.11 / 93.23 / 32.43 | 46.13 / 91.81 / 47.86 | 9.75 / 78.80 / 9.97 | 11.48 / 76.94 / 11.73 |
| 5ds 未冻结 · 8,191 | 3.05 / 18.31 / 3.37 | 3.07 / 33.33 / 4.14 | 5.42 / 34.96 / 5.58 | 6.24 / 39.19 / 6.63 |
| 5ds 冻结 · 8,191 | 2.86 / 20.31 / 3.25 | 2.83 / 32.55 / 3.91 | 5.88 / 42.69 / 6.11 | 5.74 / 40.32 / 6.09 |
| games 1:1 未冻结 · 7,615 | 2.84 / 24.77 / 3.42 | 3.12 / 37.37 / 4.52 | 5.07 / 50.43 / 5.49 | 6.26 / 50.16 / 6.84 |
| games 1:1 冻结 · 7,615 | 2.54 / 27.54 / 3.16 | 3.04 / 38.27 / 4.43 | 4.91 / 51.86 / 5.32 | 5.84 / 46.77 / 6.40 |

NV-Bench SIM（原始余弦值，越高越好；Ming 模型卡未提供）：

| 模型 | zh_single | zh_multi | en_single | en_multi |
| --- | ---: | ---: | ---: | ---: |
| Base（未微调 soar） | 0.7191 | 0.7131 | 0.7796 | 0.8070 |
| 5ds 未冻结 · 8,191 | 0.7675 | 0.7734 | 0.7298 | 0.7643 |
| 5ds 冻结 · 8,191 | 0.7865 | 0.7944 | 0.7599 | 0.7936 |
| games 1:1 未冻结 · 7,615 | 0.7634 | 0.7682 | 0.7209 | 0.7466 |
| games 1:1 冻结 · 7,615 | 0.7811 | 0.7935 | 0.7513 | 0.7830 |

### Seed-TTS

每格为 **英文 WER / 中文 CER（%）↓ / SIM×100 ↑**；错误率为逐句算术平均，SIM 为 WavLM 相似度均值乘 100。`only` 为 speaker_only，`icl` 为 speaker_icl。每个条件英文 1,088 条、中文 2,020 条，表中所有结果均完整且无超长排除；Seed-TTS 未计算 DNSMOS。

| 模型 | only en | only zh | icl en | icl zh |
| --- | --- | --- | --- | --- |
| Base（未微调 soar） | 2.151 / 72.694 | 0.904 / 77.234 | 2.076 / 76.402 | 0.868 / 80.981 |
| 5ds 未冻结 · 8,191 | 2.560 / 60.871 | 1.225 / 68.464 | 2.107 / 72.070 | 1.424 / 75.942 |
| 5ds 冻结 · 8,191 | 1.885 / 65.674 | 1.250 / 70.255 | 2.169 / 73.609 | 1.299 / 77.308 |
| games 1:1 未冻结 · 7,615 | 1.706 / 59.120 | 1.031 / 67.850 | 1.753 / 71.240 | 1.083 / 76.451 |
| games 1:1 冻结 · 7,615 | 1.755 / 59.594 | 1.092 / 65.267 | 1.867 / 73.364 | 1.058 / 78.249 |

### CV3 普通子集

同样每格为 **WER 或 CER（%）↓ / SIM×100 ↑**。每个条件中英各 500 条，全部生成、评分完成，无超长排除。

| 模型 | only en | only zh | icl en | icl zh |
| --- | --- | --- | --- | --- |
| Base（未微调 soar） | 3.995 / 75.970 | 3.351 / 79.181 | 4.262 / 75.626 | 3.899 / 79.727 |
| 5ds 未冻结 · 8,191 | 4.269 / 64.777 | 4.274 / 72.024 | 5.163 / 70.104 | 4.079 / 75.101 |
| 5ds 冻结 · 8,191 | 4.680 / 69.509 | 3.673 / 73.641 | 4.988 / 73.160 | 4.367 / 77.030 |
| games 1:1 未冻结 · 7,615 | 3.052 / 63.454 | 3.252 / 71.889 | 4.274 / 69.434 | 3.699 / 75.045 |
| games 1:1 冻结 · 7,615 | 3.452 / 65.981 | 3.436 / 71.350 | 4.136 / 73.346 | 3.698 / 77.682 |

### CV3 hard 子集

每格为 **WER 或 CER（%）↓ / SIM×100 ↑（排除条数）**。每个条件英文 64 条、中文 60 条；超过 30 s 的输出按各模型分别排除。因此各模型的评分样本集合不完全相同，不能把差值视为同一批样本上的配对改善。

| 模型 | only en | only zh | icl en | icl zh |
| --- | --- | --- | --- | --- |
| Base（未微调 soar） | 7.420 / 76.564（6） | 11.038 / 77.967（4） | 9.823 / 77.252（9） | 9.958 / 78.034（6） |
| 5ds 未冻结 · 8,191 | 7.908 / 67.540（0） | 11.659 / 69.965（4） | 8.783 / 71.080（0） | 11.019 / 71.773（3） |
| 5ds 冻结 · 8,191 | 8.086 / 71.563（0） | 12.372 / 71.812（5） | 8.258 / 75.329（0） | 12.143 / 74.682（5） |
| games 1:1 未冻结 · 7,615 | 6.563 / 66.316（0） | 11.188 / 70.502（4） | 6.936 / 70.321（1） | 11.182 / 71.390（5） |
| games 1:1 冻结 · 7,615 | 5.480 / 70.830（1） | 10.936 / 72.309（4） | 7.234 / 75.153（0） | 10.445 / 76.069（3） |

CV3 还计算了 DNSMOS P808 / OVRL，完整数值见下方各 run 的原始结果表；本节与此前主对比表一致，集中列错误率和 SIM。

### 对照结论

- **冻结 DiT 挽回了部分相似度，但没有恢复到 Base。** 5ds 的 Seed-TTS 四项 SIM 均提高（only en / zh、icl en / zh 分别为 +4.803 / +1.791 / +1.539 / +1.366 点），CV3 普通与 hard 的八项 SIM 也全部提高；两组冻结模型的 Seed-TTS / CV3 SIM 仍全部低于未微调 soar。
- **games 1:1 的 ICL 改善较稳定，speaker_only 仍有问题。** Seed-TTS ICL SIM 英文 / 中文提高 2.124 / 1.798 点，CV3 普通子集提高 3.912 / 2.637 点。但 only 中文 SIM 在 Seed-TTS 下降 2.583 点，在 CV3 普通子集下降 0.539 点。
- **读字准确率没有一致改善。** 5ds 的 Seed-TTS only 英文 WER 2.560 → 1.885、ICL 中文 CER 1.424 → 1.299，另外两项略差；games 1:1 的 Seed-TTS 四项错误率中三项略差，仅 ICL 中文改善。CV3 普通子集中，games 1:1 的 only 中英错误率变差，ICL 英文改善、中文基本持平。
- **副语言能力有取舍。** 两组冻结版的 NV-Bench PCER 都是四项中三项变差：5ds 英文单轮 34.96 → 42.69；games 1:1 仅英文多轮改善（50.16 → 46.77）。5ds 未冻结的四项 PCER 均低于 Ming 0.5B 模型卡；5ds 冻结版有三项低于它，英文单轮例外。不能只看 CER 和 SIM 就认为冻结头全面更好。
- **这些结果尚不能定位相似度下降的唯一原因。** 冻结 DiT 后上游 LLM / PatchEncoder 仍会更新，参考段 x-vector、teacher-forced 历史和模块间配合也可能影响结果。只能确认冻结 DiT 不足以完全保护普通语音克隆能力；当前是单次生成对比，未估计随机波动。

### 原始结果与审计

| 模型 | NV-Bench | Seed-TTS | CV3 | 完整性审计 |
| --- | --- | --- | --- | --- |
| Base（未微调 soar） | [结果](../../../UltraEval-Audio/res/dots-studio-soar-default-sampling/nv_bench/results_table.md) | [结果](../../../UltraEval-Audio/res/dots-studio-soar-default-sampling/seed_tts/full/results_table.md) | [结果](../../../UltraEval-Audio/res/dots-studio-soar-default-sampling/cv3/full/results_table.md) | [Seed-TTS](../../../UltraEval-Audio/res/dots-studio-soar-default-sampling/seed_tts/full/audit.json) · [CV3](../../../UltraEval-Audio/res/dots-studio-soar-default-sampling/cv3/full/audit.json) |
| 5ds 未冻结 · 8,191 | [结果](../../../UltraEval-Audio/res/dots-soar-nv26-5ds-noxvecdrop-step8191-default-sampling/nv_bench/results_table.md) | [结果](../../../UltraEval-Audio/res/dots-soar-nv26-5ds-noxvecdrop-step8191-default-sampling/seed_tts/full/results_table.md) | [结果](../../../UltraEval-Audio/res/dots-soar-nv26-5ds-noxvecdrop-step8191-default-sampling/cv3/full/results_table.md) | [Seed-TTS](../../../UltraEval-Audio/res/dots-soar-nv26-5ds-noxvecdrop-step8191-default-sampling/seed_tts/full/audit.json) · [CV3](../../../UltraEval-Audio/res/dots-soar-nv26-5ds-noxvecdrop-step8191-default-sampling/cv3/full/audit.json) |
| 5ds 冻结 · 8,191 | [结果](../../../UltraEval-Audio/res/dots-soar-nv26-5ds-frozenhead-step8191-default-sampling/nv_bench/results_table.md) | [结果](../../../UltraEval-Audio/res/dots-soar-nv26-5ds-frozenhead-step8191-default-sampling/seed_tts/full/results_table.md) | [结果](../../../UltraEval-Audio/res/dots-soar-nv26-5ds-frozenhead-step8191-default-sampling/cv3/full/results_table.md) | [Seed-TTS](../../../UltraEval-Audio/res/dots-soar-nv26-5ds-frozenhead-step8191-default-sampling/seed_tts/full/audit.json) · [CV3](../../../UltraEval-Audio/res/dots-soar-nv26-5ds-frozenhead-step8191-default-sampling/cv3/full/audit.json) |
| games 1:1 未冻结 · 7,615 | [结果](../../../UltraEval-Audio/res/dots-soar-nv26-games-1to1-step7615-default-sampling/nv_bench/results_table.md) | [结果](../../../UltraEval-Audio/res/dots-soar-nv26-games-1to1-step7615-default-sampling/seed_tts/full/results_table.md) | [结果](../../../UltraEval-Audio/res/dots-soar-nv26-games-1to1-step7615-default-sampling/cv3/full/results_table.md) | [Seed-TTS](../../../UltraEval-Audio/res/dots-soar-nv26-games-1to1-step7615-default-sampling/seed_tts/full/audit.json) · [CV3](../../../UltraEval-Audio/res/dots-soar-nv26-games-1to1-step7615-default-sampling/cv3/full/audit.json) |
| games 1:1 冻结 · 7,615 | [结果](../../../UltraEval-Audio/res/dots-soar-nv26-games-1to1-frozenhead-step7615-default-sampling/nv_bench/results_table.md) | [结果](../../../UltraEval-Audio/res/dots-soar-nv26-games-1to1-frozenhead-step7615-default-sampling/seed_tts/full/results_table.md) | [结果](../../../UltraEval-Audio/res/dots-soar-nv26-games-1to1-frozenhead-step7615-default-sampling/cv3/full/results_table.md) | [Seed-TTS](../../../UltraEval-Audio/res/dots-soar-nv26-games-1to1-frozenhead-step7615-default-sampling/seed_tts/full/audit.json) · [CV3](../../../UltraEval-Audio/res/dots-soar-nv26-games-1to1-frozenhead-step7615-default-sampling/cv3/full/audit.json) |
