# 19000-step 模型与 Qwen3-TTS 0.6B 的 Seed-TTS 对比

比较 token、sqrt、sample 三个 step-19000 模型与 UltraEval-Audio 在 2026-09-07 启动、09-08 核对完成的 Qwen3-TTS-12Hz-0.6B-Base 本地复现结果。

`xvec_only` 只使用参考音频的说话人向量；`icl_xvec` 同时使用说话人向量、参考文本和参考音频 token。

## 异常样本排除与比较口径

沿用本地复现的异常长输出规则：生成音频时长 **超过 160 秒**。在下表八个配置中，只要任一配置触发，就从所有配置中统一剔除该样本；不按错误率高低筛选，也不删除原始音频或评分。

英文原有 1,088 条，统一排除 6 条，保留 **1,082 条**；中文原有 2,020 条，无样本触发，保留 **2,020 条**。排除样本使用原始清单的零起始 `index`：

| 数据集 | index | 触发配置 | 输出时长（秒） |
|---|---:|---|---:|
| seed_tts_eval_en | 66 | qwen3-tts-0.6b-base-xvec_only | 163.76 |
| seed_tts_eval_en | 770 | qwen3-tts-0.6b-base-xvec_only | 163.76 |
| seed_tts_eval_en | 771 | qwen3-tts-0.6b-base-xvec_only | 163.76 |
| seed_tts_eval_en | 790 | qwen3-tts-0.6b-base-xvec_only | 163.76 |
| seed_tts_eval_en | 1006 | qwen3-tts-0.6b-base | 163.76 |
| seed_tts_eval_en | 1018 | token-xvec_only | 163.76 |

UltraEval 原先的过滤表仅排除前五条英文样本；本次纳入 token 的 index=1018，八个配置均从逐条评分重新汇总。因此本表的英文数值与原先保留 1,083 条的本地复现表略有不同，不能直接混用。此规则只排除异常长输出，不代表剩余样本没有其他生成错误。

## 对比结果

英文为 Whisper-large-v3 WER，中文为 Paraformer CER，均按逐句错误率百分比取算术平均；SIM 使用 WavLM，按原始相似度均值乘 100。所有指标单位为 %，WER/CER 越低越好，SIM 越高越好。每行均使用相同的英文 1,082 条、中文 2,020 条。

| 结果来源 | 模型 | 条件 | 英文 WER ↓ | 英文 SIM ↑ | 中文 CER ↓ | 中文 SIM ↑ |
|---|---|---|---:|---:|---:|---:|
| 本项目 step 19000 | token | xvec_only | 31.599 | 53.701 | 19.495 | 67.749 |
| 本项目 step 19000 | sqrt | xvec_only | 40.240 | 52.489 | 24.963 | 66.683 |
| 本项目 step 19000 | sample | xvec_only | 31.117 | 48.670 | 22.524 | 63.749 |
| UltraEval 本地复现 | Qwen3-TTS 0.6B Base | xvec_only | 1.687 | 58.578 | 0.843 | 71.853 |
| 本项目 step 19000 | token | icl_xvec | 10.097 | 57.130 | 9.077 | 69.162 |
| 本项目 step 19000 | sqrt | icl_xvec | 11.645 | 55.909 | 10.481 | 68.150 |
| 本项目 step 19000 | sample | icl_xvec | 11.689 | 52.635 | 12.021 | 65.406 |
| UltraEval 本地复现 | Qwen3-TTS 0.6B Base | icl_xvec | 1.749 | 70.837 | 1.116 | 76.666 |

三个自训练模型中，token 的联合模式四项指标均最好；三种 loss 都在加入 ICL 后降低错误率、提高说话人相似度。排除异常长输出后，自训练模型与本地复现的 Qwen3-TTS 0.6B Base 仍有明显差距。

## 自训练模型的实际训练超参

以下设置适用于 token、sqrt、sample 三次 step-19000 实验，已核对各自 checkpoint 保存的配置；不代表 Qwen3-TTS 0.6B Base 的官方训练配方。

| 配置 | 实际值 |
|---|---|
| GPU / 分布式 | 4 × A100 80GB，FSDP2 |
| 计算 | BF16，梯度归约 FP32，Flash Attention 2 |
| Batch 组织 | 动态组批，padding-free |
| 实际每卡 batch / accumulation | 平均约 89 条录音 / 1（按全局 batch 除以 4 折算） |
| 实际有效全局 batch | 平均约 356 条录音/更新；中位数 357，P10–P90 为 330–382 条 |
| 实际全局音频量 | 平均约 31.34 分钟音频/更新 |
| 每卡 batch 预算 | 最多 6,000 codec frames，同时不超过 9,000 Talker tokens；录音条数动态变化 |
| Gradient accumulation | 1 |
| 有效全局 batch 预算 | 最多 24,000 codec frames / 36,000 Talker tokens 每次更新，约 32 分钟音频上限 |
| 优化器 | AdamW，weight decay 0.01 |
| 主干峰值 LR | 1e-4 |
| 新音频模块峰值 LR | 3e-4 |
| 学习率计划 | 1,000 步 warmup；cosine 衰减；总 horizon 38,539 步，约 2 epoch；下限为峰值的 10% |
| 实际训练步数 | 19,000 步，约 1 epoch；提前结束时保留原学习率计划 |
| 结束时 LR | 主干约 5.790e-5；新模块约 1.737e-4 |
| Gradient clipping | 全局梯度范数上限 1.0 |
| 损失组成 | 首码本 CE + 0.3 × 残余码本 CE |
| Loss reduction | 分别为 token / sqrt / sample |
| Activation checkpointing | 关闭 |
| DataLoader | 每卡 16 workers，prefetch factor 2 |
| 保存 / 验证 | 每 500 步，保留最近 2 个 checkpoint |
| 随机种子 | 42 |
| 初始化 | 各自从同一原始 assembled model 开始训练 |

三次实验的主要超参一致，改变的是 loss 的归约方式。动态 batch 的录音条数随长度分布变化；表中的“预算”行表示上限，“实际”行表示日志统计。

实际 batch 统计来自三次训练各自的 `train.log`：对 step 10–19,000 每 10 步记录一次的 1,900 个日志点统计 `global_samples` 和 `global_audio_frames`，重复 step 保留最后一条记录。三次实验的这些统计相同；采样日志点的全局 batch 最小 291 条、最大 425 条。均值是日志采样估计，不是固定 batch 配置；每卡约 89 条是四卡平均，不要求每个 rank 的录音条数相等。

配置来源：[token](../../configs/emilia-10kh-pretrain.yaml)、[sqrt](../../configs/emilia-10kh-pretrain-sqrt.yaml)、[sample](../../configs/emilia-10kh-pretrain-sample.yaml)。

## 推理设置与来源

三个自训练模型使用 Qwen3-TTS 0.1.1 原生 API、BF16、Flash Attention 2、non-streaming、Auto 语言协议。Talker 与 Code Predictor 均使用 temperature=0.9、top_k=50、top_p=1；Talker repetition_penalty=1.05，最多生成 2048 个新 token、最少 2 个。固定分片和批次种子，批次信息保存在 worker JSON 和逐条记录中。

本地 Qwen3-TTS 0.6B 复现使用同一份 Seed-TTS 输入清单及注册评分器，采用 SDPA、batch size 1、显式 English/Chinese 语言控制及逐样本种子。虽然本表统一了保留样本和指标汇总，权重、推理协议、注意力后端、批量和种子仍有差异，跨模型差距不能全部归因于 loss。

**状态：八个配置均已完成全量生成和评分，本表为统一过滤后的汇总。** 汇总前核对了八个配置的样本覆盖、目标文本、参考文本和参考音频源哈希一致；保留样本的必需指标均为有限值。

- 自训练模型原始结果：[全量汇总](../../runs/seed-tts-eval-19000-20260912/summary.json)、[逐条生成与评分](../../runs/seed-tts-eval-19000-20260912/results/)、[评测说明](seed-tts-eval-19000.md)。全量汇总保留未经本表筛选的数值。
- Qwen3-TTS 0.6B 本地复现：[实验记录](../../../UltraEval-Audio/replication/voice_clone_20260907.md)、[全量汇总](../../../UltraEval-Audio/res/voice_clone_20260907/official_summary.json)、[原始结果目录](../../../UltraEval-Audio/res/voice_clone_20260907/)。本表仅使用其中两个 0.6B 配置的 Seed-TTS 结果。

自训练评测运行目录为 `runs/seed-tts-eval-19000-20260912`：`pipeline_status.json` 记录进程和阶段，`logs/` 保存日志，`results/` 保存音频和逐条评分，`exports/` 保存 checkpoint 导出及校验信息。
