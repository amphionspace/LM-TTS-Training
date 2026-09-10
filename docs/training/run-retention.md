# 正式实验与依赖保留范围

2026-09-10 用户明确要求 `runs/` 只保留已完成的中英 1kh 正式实验和正在运行的 10kh 正式实验。顶层另保留 `runs/README.md`，供后续清理直接读取。本页据此替代此前对旧试训、迁移验证和历史基线的保留清单。

| 目录 | 保留内容 |
|---|---|
| `runs/emilia-en-zh-pretrain-1000h/` | 整个正式实验：4500/5000完整checkpoint、配置、训练日志、TensorBoard、全部生成音频和指标 |
| `runs/emilia-en-zh-dynamic-10000h/` | 整个当前实验：checkpoint、启动/恢复记录、数据预检、评估、EOS修复证据和持续巡检 |

1kh 原有文件保持原样；原顶层源码归档和历史清理记录归入其 `provenance/`。10kh 数据准备日志由原独立run归入当前实验的 `provenance/data-preparation/`。

旧 LJSpeech、integration、固定/动态快照试训、padding-free迁移验证产物已按本次要求删除，不再提供本地产物链接。清理清单位于当前run的 `cleanup-20260910/runs-cleanup.json`；过时文档清单位于同目录的 `docs-cleanup.json`。

## 数据和模型依赖

本次只清理 `runs/` 和过时文档，下列依赖继续保留：

- `/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_prepared/LM-TTS-Training/emilia-short-en-zh-1000h/`
- 同一prepared根目录下的 `emilia-short-en-1000h/`、`emilia-short-en-zh-10000h/` 及1kh原始筛选清单。
- `/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_m4a/` 原始tar与索引。
- `pretrained/assembled-qwen3-tts-frozen-conditioning/` 及组装来源 `Qwen3-0.6B-Base/`、`Qwen3-TTS-12Hz-0.6B-Base/`、`Qwen3-TTS-Tokenizer-12Hz/`。

prepared清单可能直接引用其他prepared目录的codec文件，在线speaker条件依赖原始tar。只保留checkpoint不能代替保留这些依赖。

当前训练器按配置 `keep_checkpoints: 2` 轮转checkpoint；巡检和人工清理不额外删除正式run内部文件。1kh已完成，不再作为新训练输出。后续若要缩减两个正式实验或其依赖，需要用户明确授权。

`runs/`、`pretrained/` 和数据目录不由Git备份。1kh结论见[正式报告](emilia-1kh-report.md)，10kh现状见[训练与巡检记录](emilia-10kh-supervision.md)。
