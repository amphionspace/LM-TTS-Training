# 实验产物保留清单：清理前必读

更新日期：2026-09-09。根据用户要求，**1kh 实验及相关证据必须保留**。下列保护范围适用于人工清理、脚本清理和后续 agent 的清理任务。

**不能因训练结束、目录较旧、已有 10kh 实验、Git 忽略这些文件或名称含有 validation / integration，就删除或覆盖本清单中的产物。** 普通的“清理临时文件”“释放空间”不包含删除这些实验的授权。只有用户明确指定删除相应实验时，才能变更其保留范围。

项目根目录为 `/119010446/LM-TTS-Training`。以下相对路径均以此为基准。

## 1. 必须长期保留的 1kh 实验与关联证据

| 路径 | 保留范围 | 用途 |
|---|---|---|
| `runs/emilia-en-zh-pretrain-1000h/` | **整个目录，包含所有现存子目录和文件** | 中英各约 500 小时、5,000 步训练的正式实验 |
| `runs/padding-free-validation/` | 整个目录 | 本轮从旧布局迁移至 padding-free 的数值验证、恢复 checkpoint、配置、日志和评估产物 |
| `runs/emilia-official-frozen-1000h/` | 整个目录 | 先前英语 1kh 数据准备、缓存迁移与流程记录；不能与中英正式 run 混为一谈 |
| `runs/restart-source-20260909.tar.gz` | 原文件 | 本轮暂停时的源码归档 |
| `runs/restart-source-20260909.patch` | 原文件 | 对应源码改动记录 |
| `runs/restart-cleanup-20260909.json` | 原文件 | 历史清理清单，解释哪些旧产物此前已被移除 |
| `runs/checkpoint-cleanup-20260909.json` | 原文件 | 历史 checkpoint 清理记录 |

### 正式 run 内具体保护什么

`runs/emilia-en-zh-pretrain-1000h/` 中至少包括以下内容；这不是允许删除其他文件的白名单：

```text
baseline-config.yaml / config.yaml       实际训练配置
initialization.json / model_config.json  初始化与模型结构
train*.log / pipeline*.json / *.log      训练、准备、运行状态和诊断记录
checkpoints/step-00004500/               当前保留的第 4,500 步完整 checkpoint
checkpoints/step-00005000/               最终第 5,000 步完整 checkpoint
checkpoints/latest                      checkpoint 索引
tensorboard/                            全部事件文件
evaluation/                            历次验证生成的音频、逐句指标和汇总
train-evaluation/                       历次训练文本生成的音频和指标
```

必须保留 checkpoint 目录整体，包括分布式权重/优化器分片、scheduler 元数据、各 rank RNG 状态和 `COMPLETE` 标记；仅留下模型权重或 `latest` 不等于保留完整实验。

本轮运行期间曾采用 `keep_checkpoints: 2`，因此目前正式 run 中保留第 4,500 和 5,000 步权重。更早步骤的生成音频和指标仍在，**不能因为对应权重已被历史策略清理，就继续清理这些评估结果**。原有 checkpoint 保留数量不构成对归档实验继续删文件的授权。

本轮已完成，不应把此目录再次作为其他训练任务的输出。1kh 的最终报告见 [模型、数据、训练与评估报告](emilia-1kh-report.md)。

## 2. 与 1kh run 一起保护的数据和模型依赖

只保留 `runs/` 不足以重新读取样本、核查指标或加载模型。以下目录也不能作为“旧缓存”删除：

| 绝对路径或项目相对路径 | 保留原因 |
|---|---|
| `/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_prepared/LM-TTS-Training/emilia-short-en-zh-1000h/` | 1kh train/val manifest、数据统计、part-0/part-1 与 codec/audio 依赖 |
| `/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_prepared/LM-TTS-Training/emilia-short-en-1000h/` | 既有英语 codec 缓存；后续 prepared manifest 可能直接引用其文件 |
| `/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_m4a/` | 原始 tar 和索引；在线 speaker 特征及原音频评估依赖其中被引用的成员 |
| `pretrained/assembled-qwen3-tts-frozen-conditioning/` | 本轮初始组装权重、tokenizer、配置、组装报告及哈希 |
| `pretrained/Qwen3-0.6B-Base/` | 文本主干来源 |
| `pretrained/Qwen3-TTS-12Hz-0.6B-Base/` | 文本前端及 speaker encoder 来源 |
| `pretrained/Qwen3-TTS-Tokenizer-12Hz/` | 音频 codec 来源与解码依赖 |

另外保留同一 prepared 数据根目录下的 `emilia-short-en-500h.raw.jsonl` 和 `emilia-short-zh-500h.raw.jsonl`，它们是本轮筛选输入及 `preparation.json` 中哈希的核查对象。

这些路径中有共享数据和缓存引用。将实验转移到其他磁盘时，必须核对 JSONL 中的绝对路径和 `codes` / `audio_source` 引用，不能把简单移动 run 目录视为已经完成归档。当前要求是原地保留，本次未移动任何产物。

## 3. 当前 10kh 工作也不属于清理对象

| 路径 | 当前用途与状态 |
|---|---|
| `runs/emilia-en-zh-10000h-data/` | 已完成的 10kh 数据准备日志、配方、进度和核查记录 |
| `/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_prepared/LM-TTS-Training/emilia-short-en-zh-10000h/` | 10kh 原始分片、完整 codec 分片、最终 manifest 和准备报告；已整体完成 |
| `runs/emilia-10kh-partial-trial-20260909/` | 固定数据快照、显存压力测试，以及独立从头试训的配置、日志和产物 |
| `runs/emilia-10kh-dynamic-trial-20260909/` | 动态组批、显存与恢复验证、不同 worker 数的测量，以及独立从头试训产物 |
| `runs/emilia-en-zh-dynamic-10000h/` | 完整 10kh 两 epoch 正式训练、预检、配对清单、持久化巡检会话和全部评估记录；训练器按配置轮转最近两个 checkpoint，禁止额外清理 |

10kh 将使用独立 run，从初始组装模型重新训练，不接续 1kh checkpoint。新实验存在或表现更好，都不会自动解除 1kh 的保留要求。

## 4. 后续清理任务如何使用此清单

1. 先读取本页，并把以上目录及其全部子路径排除在自动清理范围外。
2. 不按 `runs/*`、目录年龄、名称或“已有新模型”等规则批量删除。
3. 若其他目录中的文件被本清单实验的 manifest 或报告引用，先核查依赖，不能直接当作临时文件删除。
4. 未列出的 run 也不因未列出就获得删除授权；需要根据具体清理任务另行判断。

`runs/`、`pretrained/` 和数据目录被 `.gitignore` 忽略，**Git 提交不是这些实验产物的备份**。本页是保留约定，不是文件系统写保护或备份证明。
