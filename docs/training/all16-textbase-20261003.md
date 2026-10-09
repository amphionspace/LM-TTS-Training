# all16 可训练文本前端实验（2026-10-03）

本轮采用 [Base text embedding + 随机 projector](../design/text-base-trainable-assembly.md)，使用已有 all16 数据与固定验证划分。与原 frozen-conditioning 实验分别保存模型、配置和运行目录；旧配置解析结果、优化器参数顺序和 checkpoint 格式保持不变。

## 配置与输出

配置：`configs/supervised-tts-20260929-all16-textbase-trainable-randproj-bf16-32gpu-lr3e-4-bblr1e-4-ep3-wsd.yaml`。

- 组装模型：`assets/assembled/qwen3-tts-text-base-trainable-random-projector`，FP32 初始权重，seed 42。
- Run：`/workspace/LM-TTS-Training-Runs/supervised-tts-20260929-all16-textbase-trainable-randproj-bf16-32gpu-lr3e-4-bblr1e-4-ep3-wsd`。
- 正式资源：`cn-sh-01a / cluster-whai`，4 节点 × 8 A800；BF16，Talker FA2 varlen，Code Predictor 原生 SDPA efficient/math。
- 数据：16 个数据集，训练 **128091959** 条、验证 **128219** 条；复用原 build，不改变样本划分。
- 训练：3 epochs，token loss，首码本 CE + 0.3 × residual CE；每卡 **32000 frames / 50000 tokens**，accumulation=1。
- LR：预训练 Talker / text embedding **1e-4**（596049920 参数），随机 projector / 音频模块 **3e-4**（149960960 参数）；WSD 随已消费样本进度，前 0.03 epoch warmup，最后 0.3 epoch decay。
- 自动 eval/save：第 100 步，然后每 2500 步和训练结束。`keep_checkpoints: null`，**保留每一个完整 checkpoint，不自动轮转删除**。实测每份完整 checkpoint 约 **8.39 GiB**；随保存次数累计占用。TensorBoard 在 `tensorboard/`，训练/验证指标在 `metrics.jsonl`，节点日志在 `logs/`。

## 已完成的验证

1. 实际组装后核对 **151669 个共享 token** 的 embedding 与文本 Base 逐值一致；新增 **7 个 TTS token** 对应行重新初始化，矩阵 **151936 × 1024**。随机 MLP 为 `1024 → 1024 → 1024`，保留 SiLU；不是 Identity 或预训练 TTS projector。
2. Talker layers/norm 与文本 Base 对应张量完全一致，speaker 与官方 TTS 对应张量完全一致。主模型 **754865216** 参数，其中 **746010880** 可训练，speaker **8854336** 冻结。公开 wrapper 与项目加载器均可重载，音频 embedding/head 不共享权重。
3. tokenizer 文件与原模型逐字节相同。旧 build 指纹含模型 config；新增兼容检查允许纯模型宽度/冻结变化，但要求 tokenizer 文件、特殊 ID、角色前缀、输入协议及词表容量一致，并继续验证原来源文件未变。没有修改 build 指纹或 manifest。实际读取 16 个数据集样本及 train/val speaker profile 检查通过。
4. CPU 回归 **68 passed / 3 skipped**；跳过 GPU 专用项，GPU 另行验证。最初在受限沙箱中两项进程通信测试因 socket 权限失败，改在正常本地进程通信环境完整重跑通过，没有改测试以绕过失败。Ruff 和格式检查通过。
5. 新旧方案分别执行真实 BF16 双卡连续 / 中断恢复对照，最终权重逐位一致，最大差异 **0**。新方案确认 text embedding/projector 实际更新、speaker 不更新、4 个 checkpoint 全部保留；旧方案确认原冻结范围不变。
6. 完整模型使用真实训练 build 做本地八卡 6 步预检；无 OOM、NaN/Inf 或损坏样本。除去启动和 worker 预热，step 2–6 约 **7.6 秒/步**、数据等待 **0.4%–0.6%**；PyTorch 峰值 **61.18 GiB/卡**，NVML 采样峰值 **67.13 GiB/卡**（约 84%）。第 3、6 步完整保存成功。预检关闭全量 holdout，正式配置保留自动完整 eval。

7. 完整模型八卡从第 3 步恢复到第 6 步：第 4–6 步 loss/CE、梯度范数、LR、样本/音频 token 数逐值一致，最终 checkpoint 元数据完全一致。两份 FP32 全模型导出权重 SHA256 相同：`3360862062e2573fb306ed02122ab9134f6ce24b43c67778bddd3e957dac4b70`。

## 正式提交与清理

- **Job：`pt-u0ys9osd`**，平台创建时间 **2026-10-03 09:25:30 UTC**；首次回查为 **STARTING**，原 `pt-blb39rgw` 仍为 RUNNING。提交只执行一次，没有恢复旧实验或停止其他任务。
- Submission：`submissions/20261003T092528497771Z`；实际源码快照 **`209a70ab01f77aed7983ad2dbc101fa6344dd652`**。已核对保存的 `experiment.yaml`：BF16、3 epochs、4 节点、独立模型与输出、保留全量 checkpoint、正式 holdout 开启。
- 12 份验证报告 / 指标 / GPU 采样已复制并校验到本 run 的 **`logs/verification/`**，清单和 SHA256 在 `verification-files.json`。关键结果为 `assembly-audit.json`、`legacy-bf16-resume.json`、`textbase-bf16-resume.json`、`full-model-8gpu-resume.json`。
- 确认提交后删除本次 `artifacts/textbase-validation/` 下临时数据 fixture、测试模型、预检 checkpoint、导出权重和临时配置，释放 **31.54 GiB**；回归测试代码和验证证据保留。正式模型、训练数据、原任务和新任务的 checkpoint 均未清理。

这里的短预检证明工程可运行，不能证明收敛效果或长时间所有 batch 的峰值。正式 32 卡跨节点运行仍须以平台现场为准。

## 操作

仓库根目录使用同一 YAML：

```bash
PY=/workspace/workspace/yanglin/envs/lm-tts/bin/python
CONFIG=configs/supervised-tts-20260929-all16-textbase-trainable-randproj-bf16-32gpu-lr3e-4-bblr1e-4-ep3-wsd.yaml
"$PY" -m scripts.prepare_models --config "$CONFIG"
"$PY" -m scripts.acp.submit --config "$CONFIG"           # Preview only.
"$PY" -m scripts.acp.submit --config "$CONFIG" --submit  # Submit once.
```

初次提交不传 resume。恢复时只使用这个 run 的 checkpoint，先确认旧任务终态，再追加 `--resume latest`；禁止用原 frozen-conditioning checkpoint 切换模型。任务提交时保存源码快照及展开配置，不随仓库后续清理改变。

## Seed-TTS：step 25,000（2026-10-09）

按指定的 `checkpoints/step-00025000` 评测，约为 1.587 epoch；不自动改用后续 checkpoint。复用 `UltraEval-Audio/init_model/all16-textbase-step-00025000` 的 FP32 导出，以 BF16 + SDPA 推理。已核对完整标记、checkpoint 元数据与组装报告哈希、导出权重哈希，以及配置和 tokenizer 与本实验组装模型的一致性。导出权重 SHA256 为 `41db269684fdd84b8d725ea9466a097fbe6e167451967e0ce04574c32c3e89e2`。

- 仅评测 **speaker + ICL**；英文 1,088 条、中文 2,020 条。参考录音提供 speaker embedding、codec 和参考文本，目标录音不进入生成。
- 顺序为 **双 greedy → 双采样**，每组先完成小样本生成 / 评分 / 审计，再运行全量；greedy 全量通过审计后才开始采样。speaker-only 保持暂停。
- 使用 8 卡，每推理进程 16 GiB 显存上限，batch size 1；`language="Auto"`、`non_streaming_mode=True`，与本轮训练输入协议一致。
- 采样时两个模块均使用 `top_k=50, top_p=1.0, temperature=0.9`；greedy 关闭两处采样。Talker `repetition_penalty=1.05`，逐条 seed 为 `42 + split 内 index`。
- 各组独立排除输出 **超过 30 秒** 的样本，不参与 ASR / SIM；保留排除数量和清单。生成上限 378 token，略超阈值后停止并排除，不把截短音频计为正常样本。
- 英文 Whisper-large-v3 WER、中文 SeACo-Paraformer CER，均按逐句错误率取平均；相似度为 WavLM-large + ECAPA 的余弦相似度 ×100。数据和评分器复用之前 Seed-TTS 评测的本地文件。

以下路径相对 `UltraEval-Audio`：

| 内容 | 路径 |
| --- | --- |
| Greedy 结果 | `res/all16-textbase-step25000-seedtts-greedy-20261009-8gpu` |
| 采样结果 | `res/all16-textbase-step25000-seedtts-sampling-20261009-8gpu` |
| 顺序执行日志 | `log/seed-tts-textbase-step25000-sequence.log` |

后台会话为 `seedtts-all16-textbase-s25000-sequence`。具体命令保存在 greedy 结果目录的 `run-sequence.sh`；`sequence-stage.txt` 记录当前阶段，各组 `pipeline_status.json` 记录运行状态，完成后读取 `full/summary.json` 与 `full/audit.json`。若中断，确认该会话和其评测进程均已退出后，可在 UltraEval-Audio 根目录重跑同一脚本，复用已完成音频与评分。

本次是独立的 textbase 实验，不写入 frozen-conditioning 的训练与评测报告。当前已启动，尚无全量成绩；不能用启动验证或少量样本代替完整结果。
