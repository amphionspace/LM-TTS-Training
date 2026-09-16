# 三种 loss 模型的 Seed-TTS 评测

2026-09-12 启动。评测 token、sqrt、sample 三个已完成训练的 step-19000 checkpoint，每个模型分别运行 speaker only (`xvec_only`) 和 ICL + speaker (`icl_xvec`)。全量结果见 [结果表](seed-tts-eval-19000-results.md)。

## 数据与指标

复用 `/119010446/UltraEval-Audio` 的 Seed-TTS 数据和注册评测器。工具仓库版本为 `43918a0f589926e09fc361674b2d45088054adf6`。

| 数据集 | 条数 | 清单 SHA-256 |
|---|---:|---|
| seed_tts_eval_en | 1,088 | `780b027a08ac7a69c90abb5b03c7fe5a898d5a615e08eca2c88dc4bbb94234b4` |
| seed_tts_eval_zh | 2,020 | `861e6b9ab276713fa74f2ca9196db14cbefebdd92bbf2c26d8f6e292be1fea9c` |

六组共 18,648 条输出。英文评分使用 UltraEval 的 Whisper-large-v3 WER，中文使用 Paraformer CER，两者均为逐句错误率百分比的算术平均。说话人相似度复用 WavLM `simo`，报告均值乘 100。评分调用原有 `scripts/voice_clone_score.py --group seed`，没有改动评分器、文本标准化或 ASR 模型。

推理只读取清单中的目标文本及参考音频/参考文本。清单中的 `prompt_sha256` 是源 Parquet 内音频字节的哈希；UltraEval 将其重新写为 PCM16 WAV，因此不能将该字段直接当成落盘 WAV 的文件哈希。推理另记录实际参考 WAV 的 SHA-256。ICL 解码后由原生 API 裁掉参考部分，仅输出续写的目标音频。评分比较生成音频的转录与目标文本，以及生成音频与参考音频的说话人相似度。

全量 3,108 条参考音频已经与源 Parquet 核对：源音频哈希、目标文本、参考文本以及解码后的 PCM16 采样均一致。运行目录内 `validate_inputs.py` 保存核对脚本，`input_validation.json` 保存核对结果及实际 WAV 哈希；推理逐条复核该哈希。

## 推理条件

| 参数 | 设置 |
|---|---|
| checkpoint | 三个训练 run 各自的 `checkpoints/step-00019000` |
| 引擎 | 训练环境中安装的 `qwen-tts==0.1.1` 原生 `Qwen3TTSModel.generate_voice_clone` |
| 精度与注意力 | BF16；Flash Attention 2；使用原生 KV cache |
| 输入协议 | `non_streaming_mode=True`、`language="Auto"`，与训练协议一致 |
| speaker only | `x_vector_only_mode=True`；不传参考文本；无参考音频 token 前缀 |
| ICL + speaker | `x_vector_only_mode=False`；传参考文本和音频；保留说话人向量 |
| Talker 采样 | `do_sample=True, top_k=50, top_p=1, temperature=0.9, repetition_penalty=1.05` |
| Code Predictor 采样 | `subtalker_dosample=True, subtalker_top_k=50, subtalker_top_p=1, subtalker_temperature=0.9` |
| 长度上限 | `max_new_tokens=2048`；原生最短生成 2 token |
| 随机种子 | 每批 `42 + 首条 index + 中文偏移 100000`；六组规则相同 |

Auto 语言设置和 non-streaming 是这三个训练模型的输入契约。UltraEval 现有官方预训练模型实验使用的语言控制、模型权重、输入模式可能不同，不能将两次实验的差异全部归因于 loss。

checkpoint 导出保留所有 478 个模型张量，共 914,643,008 个元素；不导出优化器。每个导出记录源 checkpoint、metadata SHA-256 和权重 SHA-256。原生加载检查权重键与形状；冒烟检查另逐张量验证加载值。每批运行记录实际使用说话人向量和 ICL 路径的情况，并记录每条是否生成 EOS。未到 EOS 的长输出仍参与评分，不静默过滤。

## 执行与恢复

启动前检查已通过：三份导出均完成原生加载逐张量核对；已有两项输入协议一致性测试通过；36 条中英文试生成全部到 EOS；UltraEval 对其中 12 条完成有限 WER/CER 和 SIM 评分。详细记录位于运行目录的 `smoke-verification.json`、`smoke/` 与 `logs/smoke-score-*.log`。冒烟数据只验证链路，不能用于比较模型总体质量。

全量任务运行于 tmux 会话 `seed-eval-19000`。推理和评分环境的具体包版本分别记录于 `inference_versions.json` 和 `metrics_versions.json`。

在当前训练仓库执行：

```bash
PYTHONPATH=. .venv/bin/python scripts/export_seed_eval.py \
  --output runs/seed-tts-eval-19000-20260912/exports

PYTHONPATH=. .venv/bin/python -u scripts/run_seed_eval.py \
  --run-dir runs/seed-tts-eval-19000-20260912 --workers 16 --batch-size 8
```

默认 16 个独立推理进程使用 4 张 GPU，固定 `index % 16` 分片，每进程 batch size 8。三模型、两种条件按顺序运行，推理全部完成后自动评分和汇总。评分进程以 UltraEval 目录为工作目录，输出仍保存在当前训练仓库。进程、命令、退出码及阶段状态写入运行目录的 `pipeline_status.json`，日志保存在 `logs/`。

同一命令可恢复：生成端跳过已有成功记录；批次部分完成时保留原批次与种子，重新计算后只保存缺失条目。恢复时保持分片数和 batch size 不变。评分器跳过已有成功评分。`smoke/` 中的冒烟样本独立于 `results/` 全量结果。

查看部分进度并更新结果表：

```bash
.venv/bin/python scripts/summarize_seed_eval.py \
  --run-dir runs/seed-tts-eval-19000-20260912 --allow-incomplete
```

汇总验证清单身份、音频帧数/采样率、完整覆盖和有限指标；只有十二个模型/条件/语言组合均覆盖全部样本，才标记全量完成。`summary.json` 保存汇总，`outliers.json` 保存每组错误率最高的十条及音频路径。
