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

refmask 的 Seed-TTS 首次运行因审计脚本按 24 kHz 校验 dots 的 48 kHz 输出而失败，已修复（UltraEval `a885ea9`）后重跑。**noxvecdrop 的 CV3 仍在运行**（2026-11-03 02:00 PDT）。

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

CV3（WER / CER 与 SIM×100；括号内为超过 30 s 被排除的条数；noxvecdrop 待补）：

| 条件 | 子集 | 未训练 soar | | refmask step 6,448 | |
| --- | --- | ---: | ---: | ---: | ---: |
| | | WER/CER | SIM | WER/CER | SIM |
| speaker_only | zero_shot_en | 3.995 | 75.970 | 4.789 | 65.914 |
| speaker_only | zero_shot_zh | 3.351 | 79.181 | 3.514 | 72.838 |
| speaker_only | hard_en | 7.420 (6) | 76.564 | 6.829 (1) | 68.116 |
| speaker_only | hard_zh | 11.038 (4) | 77.967 | 11.378 (4) | 71.703 |
| speaker_icl | zero_shot_en | 4.262 | 75.626 | 4.950 | 71.453 |
| speaker_icl | zero_shot_zh | 3.899 | 79.727 | 4.428 | 76.121 |
| speaker_icl | hard_en | 9.823 (9) | 77.252 | 8.493 (1) | 72.505 |
| speaker_icl | hard_zh | 9.958 (6) | 78.034 | 11.915 (2) | 72.893 |

- **微调明显降低了说话人相似度**：refmask 的 SIM 在八项里全部下降，speaker_only 降 6.3–10.1 点（英文最多，75.97 → 65.91），speaker_icl 降 3.6–5.1 点。
- **内容准确率基本持平**：普通子集 WER/CER 高 0.2–0.8 点；hard 子集互有升降，且微调后超长输出（被排除）明显变少。
- speaker_only 的下降远大于 speaker_icl，说明模型对 x-vector 的依赖变弱了。可能的原因：训练时 x-vector 来自句内的短参考段（约三分之一短于 2 s），比推理时整段 prompt 算出的向量噪声大；全模型微调也可能冲掉了 SOAR（只调 DiT）带来的相似度。noxvecdrop（始终使用 x-vector）的结果可以帮助区分。

原始结果：[refmask Seed-TTS](../../../UltraEval-Audio/res/dots-soar-emilia-nv-refmask-step6448-default-sampling/seed_tts/full/results_table.md) · [noxvecdrop Seed-TTS](../../../UltraEval-Audio/res/dots-soar-emilia-nv-refmask-noxvecdrop-step6448-default-sampling/seed_tts/full/results_table.md) · [未训练 soar Seed-TTS](../../../UltraEval-Audio/res/dots-studio-soar-default-sampling/seed_tts/full/results_table.md) · [未训练 soar CV3](../../../UltraEval-Audio/res/dots-studio-soar-default-sampling/cv3/full/results_table.md) · [refmask CV3](../../../UltraEval-Audio/res/dots-soar-emilia-nv-refmask-step6448-default-sampling/cv3/full/results_table.md)。

## 4. 评测怎么跑

UltraEval-Audio 中：`init_model/<run-short>-step-<8 位 step>` 为导出模型（`LM-TTS-Training/scripts/export_checkpoint.py` 直接导出为官方 runtime 布局），结果在 `res/<run-short>-step<N>-default-sampling/{nv_bench,seed_tts,cv3}/`。一个模型的全部评测作为一个 ACP 任务提交：

```bash
python -m scripts.acp.submit --name evalsuite-<run> --log-dir res/<run>/logs --submit -- \
  scripts/run_dots_suite.py --model-path init_model/<export> --run-dir res/<run>
```

dots 推理使用 `envs/dots`（官方 dots.tts @ `4cf7d6b`），打分沿用 `envs/metrics`。
