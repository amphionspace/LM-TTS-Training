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

输出为 `/workspace/LM-TTS-Training-Runs/supervised-tts-20260929-all16-no-spk-bf16-16gpu-acc2-lr3e-4-bblr1e-4-ep3-wsd/`；`yanglin/LM-TTS-Training-Runs` 软链接也可访问。Checkpoint、TensorBoard、节点日志、验证报告分别在 `checkpoints/`、`tensorboard/`、`logs/`、`logs/verification/`。

条件开关进入 checkpoint 的模型签名。旧 YAML 不新增默认键，保持旧 resume 签名；新实验只能恢复本模式、同 world size 和 batch 设置的 checkpoint。导出会保存无 speaker 标记。官方原样的 `generate_voice_clone` 不识别这个扩展标记，后续 codec-prefix ICL 推理需要相应适配；现有 speaker-only 合成脚本会拒绝这种导出，避免带入训练时不存在的 speaker 位置。训练内自动验证是 teacher-forcing loss，不是 Seed-TTS 生成评测。

提交命令（仓库根目录）：

```bash
PY=/workspace/workspace/yanglin/envs/lm-tts/bin/python
CFG=configs/supervised-tts-20260929-all16-no-spk-bf16-16gpu-acc2-lr3e-4-bblr1e-4-ep3-wsd.yaml
$PY -m scripts.acp.submit --config "$CFG"            # Preview.
$PY -m scripts.acp.submit --config "$CFG" --submit   # Submit once.
```

恢复时确认旧任务已停止，再使用同一配置追加 `--resume latest`。提交使用固定源码快照，不随工作区后续修改改变。
