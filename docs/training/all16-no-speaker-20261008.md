# all16 无 speaker embedding 实验（2026-10-08）

本实验从原冻结文本前端的 assembled 模型重新训练，用于比较去掉整段音频 speaker embedding 后的行为。不是从 step 47382 或其他训练 checkpoint 恢复，也不采用可训练 text-base 前端。

配置为 [supervised-tts-20260929-all16-no-spk-bf16-16gpu-acc2-lr3e-4-bblr1e-4-ep3-wsd.yaml](../../configs/supervised-tts-20260929-all16-no-spk-bf16-16gpu-acc2-lr3e-4-bblr1e-4-ep3-wsd.yaml)，直接继承 `base.yaml`。

| 项目 | 设置 |
| --- | --- |
| 模型初始化 | `assets/assembled/qwen3-tts-frozen-conditioning` |
| 条件开关 | `model.use_speaker_embedding: false` |
| 资源 | `cn-sh-01a / cluster-whai`，2 节点 × 8 A800 |
| 精度与注意力 | BF16；Talker FA2 varlen，Code Predictor SDPA |
| 训练数据 | 原 all16 固定 build；128,091,959 条训练、128,219 条验证 |
| 训练目标 | 完整 codec teacher forcing；token loss；首码本 CE + 0.3 × residual CE |
| 训练量 | 3 epoch，梯度累积 2；每卡 32,000 frames / 50,000 tokens |
| 学习率 | 音频模块 3e-4，Talker backbone 1e-4 |
| 调度 | WSD 随消费样本进度；前 0.03 epoch warmup，最后 0.3 epoch decay，最低比例 0.1 |
| 保存与验证 | 第 100 步、每 2,500 步及训练结束；保留所有 checkpoint |

text embedding、text projector 的权重和冻结策略保持不变。Speaker encoder 的冻结权重仍保留在模型状态中，但不参与输入构造或计算；不使用零向量占位。16 张 codec embedding、Talker、音频 head 和 Code Predictor 的训练方式保持不变。没有增加随机 reference 或 loss mask。

读取器复用原 build 的样本索引、版本和元数据校验，跳过 speaker embedding 大字段；不重建或修改源数据。因此样本范围仍受原 build 的发布筛选约束。动态 batch 长度预算保留原保守上界，避免同时改变分批策略；实际输入每条减少一个位置。Loss 按所有卡、整个累积窗口的监督 token 数归一化。

输出为 `/workspace/LM-TTS-Training-Runs/supervised-tts-20260929-all16-no-spk-bf16-16gpu-acc2-lr3e-4-bblr1e-4-ep3-wsd/`；`yanglin/LM-TTS-Training-Runs` 软链接也可访问。Checkpoint、TensorBoard、节点日志、验证报告分别在 `checkpoints/`、`tensorboard/`、`logs/`、`verification/`。

条件开关进入 checkpoint 的模型签名。旧 YAML 不新增默认键，保持旧 resume 签名；新实验只能恢复本模式、同 world size 和 batch 设置的 checkpoint。导出会保存无 speaker 标记。官方原样的 `generate_voice_clone` 不识别这个扩展标记，后续 codec-prefix ICL 推理需要相应适配；现有 speaker-only 合成脚本会拒绝这种导出，避免带入训练时不存在的 speaker 位置。训练内自动验证是 teacher-forcing loss，不是 Seed-TTS 生成评测。

提交命令（仓库根目录）：

```bash
PY=/workspace/workspace/yanglin/envs/lm-tts/bin/python
CFG=configs/supervised-tts-20260929-all16-no-spk-bf16-16gpu-acc2-lr3e-4-bblr1e-4-ep3-wsd.yaml
$PY -m scripts.acp.submit --config "$CFG"            # Preview.
$PY -m scripts.acp.submit --config "$CFG" --submit   # Submit once.
```

恢复时确认旧任务已停止，再使用同一配置追加 `--resume latest`。提交使用固定源码快照，不随工作区后续修改改变。

## 提交前验证

- 新增条件开关测试 3 项通过：移除 speaker 位置、忽略缺失/无效的 speaker 输入、因果 codec 前缀与完整 teacher forcing 预测一致、训练模块梯度正常、文本端和 speaker 不更新；数据读取不请求 embedding 向量，worker 序列化保留开关。配置对比确认其余基线设置一致。
- 导出测试 5 项通过，包含旧 checkpoint 默认行为与新 checkpoint 无 speaker 标记。既有 merged 读取、配置、checkpoint 保留和 ACP 提交测试通过；修改文件 Ruff 检查通过。
- 实际 assembled 模型完整加载、权重哈希校验通过。Train / validation 的全部 16 个数据集各读取一条真实样本成功；text / codec 和原 build 指纹保持不变。
- 真实训练器使用 tiny 模型进行双卡 BF16、accumulation=2 验证：连续 4 步与训练 2 步后恢复到 4 步，最终权重逐位一致，最大差异 **0**。第 3、4 步 loss、梯度范数、LR 和 token 数一致。冻结模块不更新，Talker 和音频模块正常更新。
- 两份验证 checkpoint 导出权重 SHA256 均为 `2b82726b384513a7958f1c94ebffb537abeeb5ebbe3e842f9bdef045e602e7fe`。这次按要求采用短正确性验证，没有另起完整模型的性能预检；正式两节点任务的吞吐和显存以运行日志为准。

## 正式任务

- Job：`pt-6epa7d0g`；2026-10-08 06:18:03 UTC 创建，06:18:20 UTC 平台进入 RUNNING。
- Submission：`submissions/20261008T061801020882Z`；源码快照 `1834c1f5f1ca84383b9e969c00463bc7f4bfeb7e`，首次启动不传 resume。SSH 免密、平台 TensorBoard 已开启。
- 验证报告、两条路径的训练指标和导出哈希共 8 份文件保存在 run 的 `verification/`；`files.json` 保存校验值。此目录独立于平台以 root 身份创建的节点日志目录。
- 归档后清理本次临时 fixture、测试组装模型、测试 checkpoint、导出权重和预览配置；正式模型、数据和所有正式 checkpoint 保留。

首次正式训练指标已到 **step 10**：objective **11.96857**（处于 warmup 初期），first CE **9.62119**，residual CE **7.82459**，skipped/discarded samples 均为 **0**。该步耗时 **15.03 秒**，最慢 rank 的数据等待 **0.0885 秒 / 0.59%**；PyTorch 峰值分配 **62.99 GiB**，同期 NVML 显存约 **85%–87%**，16 张卡利用率采样均为 **100%**。这是启动阶段观测，不代表全程峰值或收敛结论。`verification/formal-start.json` 保存实际 world size、条件开关、累积设置、checkpoint 保留策略及首条指标。

临时验证产物清理约 **263.5 MiB**；测试代码与验证证据保留。第 100 步的自动 holdout 验证尚未执行，本次没有等待该轮评估完成。
