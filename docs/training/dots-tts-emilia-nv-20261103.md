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
| 全部 终 | 待补 | | | |

### CV3 普通子集（WER / CER，SIM×100）

| 模型 | only en | only zh | icl en | icl zh |
| --- | --- | --- | --- | --- |
| dots.tts-soar（未训练） | 3.995 / **75.970** | 3.351 / **79.181** | 4.262 / **75.626** | 3.899 / **79.727** |
| 第二轮终 | 4.269 / 64.777 | 4.274 / 72.024 | 5.163 / 70.104 | 4.079 / 75.101 |
| 1:1 step 3,000 | 4.057 / 65.664 | **3.162** / 72.618 | 4.755 / 70.862 | 3.744 / 75.792 |
| 1:1 终 | **3.052** / 63.454 | 3.252 / 71.889 | **4.274** / 69.434 | **3.699** / 75.045 |
| 全部 step 3,000 | 3.210 / 62.083 | 3.312 / 70.280 | 4.541 / 69.984 | 3.831 / 74.849 |
| 全部 终 | 待补 | | | |

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
