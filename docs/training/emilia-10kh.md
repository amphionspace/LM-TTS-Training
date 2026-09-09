# Emilia2 中英 10,000 小时数据准备

目标为英语、中文各约 5,000 小时，达到目标时保留最后一条完整短句；独立 `type=short` 加上 `type=long` 的短句视图去重后若不足，则使用实际可用的有效数据。不额外限制录音时长，不截断文本，也不从 dialogue 提取片段。

入口为 `scripts/prepare_emilia_streaming.py`。当前数据目录：

```text
/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_prepared/LM-TTS-Training/emilia-short-en-zh-10000h/
```

- `raw/`：原始清单分片，约 8,192 条一份，同一 long 的选中片段保留在同一分片。
- `prepared/`：已完成 codec 的清单分片。只有整份分片成功后才发布。
- `codes/`：新编码的 NPZ，按哈希前两位分目录。复用的 NPZ 直接引用已有缓存路径，因此也依赖对应旧缓存目录。
- `train.jsonl`、`val.jsonl`：全部编码完成、全局去重与验证划分通过后发布的训练入口。
- `preparation.json`、`PREPARATION_COMPLETE`：最终统计和完成标记。`EXPORT_COMPLETE.json` 仅表示元数据导出完成。

本次保留切换前已导出的 180 份独立 short 清单，随后扫描独立 short 和 long 的短句视图补足配额。按原始短句 ID 去重；`duplicate_ids_skipped` 包含重新扫描时命中保留清单的记录，不能解释为原始语料的重复率。

## long 片段与解码

片段保留原始短句 ID、文本、语言和 speaker。`source.type=long`、`source.view=short`，并记录载体 ID。`audio_source.frames` 是完整载体的采样点数；`start_frame`、`end_frame` 是该短句在载体内的边界。样本 `duration` 则是片段时长。

导出时排除越界、空文本或缺少 speaker 的 long 片段，具体记录写入导出日志。解码保留既有 AAC 尾部处理规则；先在原始采样率下按标注切片，再重采样到 24 kHz。同一 long 的选中片段由一个 CPU 任务共同处理，避免为每条短句重新读取、解码载体。

训练和评估读取沿用相同的切片函数。在线 speaker 条件目前会按样本重新解码载体；离线 codec 的整组解码优化不代表在线训练吞吐已验证。

## 并发和恢复

本次使用两张 A800 80GB，FP32 codec、每卡 batch 16。元数据导出与 GPU 编码重叠；每卡使用 12 个解码进程、12 个读取线程、8 个写入线程。独立 short 预取 8 个 batch；包含 long 的分片提前提交各载体的解码任务，并让已就绪的批次先编码，避免某个数小时的载体阻塞整个队列。每批仍采用固定的时长分组，减少 padding。更改 batch 组合在实测中会产生少量离散 token 差异，配方记录了精度、batch 和排序方式，不承诺与旧混合时长 batch 逐 token 相同。

元数据导出命令：

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -u scripts/prepare_emilia_streaming.py \
  --stage export \
  --output /ai_sds_wuzz/DATA_TTS/Emilia2_TTS_prepared/LM-TTS-Training/emilia-short-en-zh-10000h \
  --include-long-shorts --allow-shortfall --export-workers 32 --resume-export
```

GPU 编码与最终发布命令：

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -u scripts/prepare_emilia_streaming.py \
  --stage supervise \
  --output /ai_sds_wuzz/DATA_TTS/Emilia2_TTS_prepared/LM-TTS-Training/emilia-short-en-zh-10000h
```

启动前先查看 `runs/emilia-en-zh-10000h-data/` 的 PID、进程和状态，避免重复启动。恢复编码会跳过已发布分片，重新处理未完成分片。配方变化会被拒绝。

最终划分在全体数据上检查 ID 唯一性；每种语言留出 256 条规范化文本唯一的验证样本，保留同说话人的至少两条训练录音。这是已见说话人的未见文本验证。

## 验证证据

`tests/test_sources.py` 覆盖原始采样点切片、整组与单条解码相等、只解码一次载体及越界标注；`tests/test_streaming_preparation.py` 覆盖导出恢复、ID 去重、载体分片边界、短缺报告及全局划分。

真实中英 long 试跑共 679 条、约 1.182 小时，完成双卡编码及 675 train / 4 val 发布。报告位于公共数据目录的 `emilia-long-codec-check/preparation.json`；运行日志、性能压测和本次切换记录位于 `runs/emilia-en-zh-10000h-data/`。
