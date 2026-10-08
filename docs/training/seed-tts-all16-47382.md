# All16 step 47382 的 Seed-TTS 外部评测

评测代码、数据、模型、环境和结果位于相邻的 `UltraEval-Audio` 仓库；训练代码无需依赖该仓库。
选用 `supervised-tts-20260929-all16-bf16-32gpu-lr3e-4-bblr1e-4-ep3-wsd` 的最后归档 `step-00047382`，
不是仍在运行的 textbase 实验。归档 67 个文件已通过清单、大小及 SHA-256 验证。

## 转换命令

在 LM-TTS-Training 根目录执行，输出路径必须尚不存在：

```bash
PY=/workspace/workspace/yanglin/envs/lm-tts/bin/python
RUN=/workspace/workspace/yanglin/LM-TTS-Training-Runs/supervised-tts-20260929-all16-bf16-32gpu-lr3e-4-bblr1e-4-ep3-wsd
UE=/workspace/workspace/yanglin/UltraEval-Audio
$PY -m scripts.export_checkpoint \
  --checkpoint "$RUN/archived-checkpoints/step-00047382" \
  --assembled-model assets/assembled/qwen3-tts-frozen-conditioning \
  --output "$UE/init_model/all16-frozen-conditioning-step-00047382" \
  --copy-tokenizer
```

本次已导出，不能直接覆盖重跑。FP32 `model.safetensors` SHA-256：
`1e942af5c2c5ec2d1a26bcaf78d6e231e965c313aa0fa2a4191172f0dd2878c7`。
模型导出后以 BF16 加载做推理，不改原始 checkpoint、optimizer 或训练配置。

## 条件与输入协议

- `speaker_only`：参考音频提取 speaker embedding，不向生成模型提供参考文本或 codec 上下文。
- `speaker_icl`：同一参考音频同时提供 speaker embedding、参考 codec，以及对应参考文本。
- 两组都用 `language="Auto"` 和 `non_streaming_mode=True`，与本轮训练输入布局一致。

官方 SDK 显式设置 Chinese/English 时会加入 language ID；Auto 没有该 ID，且控制前缀不同。
历史 UltraEval 官方模型报告使用显式语言 ID，比较时必须注明该差异。

中英文 Seed-TTS 共 3,108 条输入，每种模式全量评测一次。英文 Whisper-large-v3 WER、中文
SeACo-Paraformer CER、WavLM-large/ECAPA SIM 沿用 UltraEval 评分器，不使用训练内的 Whisper-small 指标。
下载版本、运行参数、逐条结果及审计记录由 UltraEval 保存；报告入口为
`UltraEval-Audio/replication/seed_tts_all16_step47382.md`。
