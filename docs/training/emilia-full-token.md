# Emilia 2 全量中英文 token-loss 从头训练

2026-09-12 开始准备。根据 Seed-TTS 六组评测结果选用 token loss，本轮是新的训练实验：从原始组装初始化启动，不加载 token/sqrt/sample 的 19000-step 权重、优化器或数据游标。

## 数据口径

源目录为 `/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_m4a`。主 `data/` 目录有 **8,059 对 tar/idx**。另有 `ASMR/` 的 **13 对**，仅用于清点和建立排除名单，**不进入训练**；同 recording ID 的其他载体也排除。旧 `_dataset_info.json` 的 6,915 个 tar 已不能代表当前目录。

使用 `type=short` 独立短句、`type=long` 中的 short 视图，以及通过下述质量筛选的 `type=dialogue` short 视图；不会将整段多说话人对话当作单说话人目标。仅保留 `language=en/zh`，不设置小时配额，不额外截断短句或文本。旧 10kh 方案不取 dialogue；本轮按用户指令仅将质量合格的分离音频短句作为新增数据。

按原始短句 ID 全局去重，同 ID 优先选择独立 short，其次 long 中的 short，最后通过质量筛选的 dialogue short。相同文本、不同说话人不因文本相同而删除。本轮没有做跨 ID 的语音近重复检测。候选视图小时数不能与载体小时数相加。

全量 ID 核对（质量过滤前）记录于 `runs/emilia-full-token-scratch/preflight/id-audit.json`：

| 来源 | 英文新增条数 | 中文新增条数 | 英文小时 | 中文小时 |
|---|---:|---:|---:|---:|
| 独立 short | 2,081,298 | 1,913,616 | 2,740.17 | 2,362.76 |
| long 的 short | 8,328,438 | 10,993,740 | 14,829.62 | 22,401.16 |
| dialogue 的 short，质量过滤前 | 9,227,338 | 2,308,469 | 12,429.26 | 4,309.23 |

ASMR 中英候选包含 162,349 个不同短句 ID、2,060 个 recording ID；在主目录候选中未发现这些 ID/recording 的重现。short 与 long 部分没有重复 ID；dialogue 部分跳过 225,953 次重复 ID。最终训练量以质量过滤后的 `EXPORT_COMPLETE.json` 和编码发布报告为准。

dialogue 载体来自数据集的独立分离流程；这不保证每条短句都是纯净的单人声。`scripts/filter_emilia_dialogue.py` 对其实施以下筛选：

- 标注 DNSMOS 至少 3.4；缺失或非有限分数不接收。
- 对照载体中所有语言的短句标注，排除与其他说话人有时间重叠的候选。
- 在原采样率下精确切片并重采样到 24 kHz；空、全零或非有限音频不接收。
- 对实际切片重算非个性化 DNSMOS P.835，要求 OVRL ≥3.4、SIG ≥3.5、BAK ≥4.0；绝对幅度 ≥0.999 的采样占比最多 0.1%。

使用 [Microsoft DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS) 的权重和校准多项式。权重已与官方文件 SHA-256 核对一致。短音频重复至足够长度；以 16 kHz 下的 144,160 点窗口、16,000 点步长评估所有完整窗口。采用整数窗口边界和计数，修正参考脚本浮点取整导致后续窗口少一采样点而被跳过的问题。96 条相同波形与作此修正的参考实现数值一致，最大差异约 1.3e-15。

128 个非空分片的分层抽样中英各 48 条，覆盖标注分数 [3,3.2)、[3.2,3.4)、≥3.4。重算后符合音质门槛的英文 14 条、中文 8 条，其 Whisper-large-v3 逐句平均 WER/CER 为 2.51% / 4.37%。这些仅用于校准筛选方案，不代表全量质量；ASR 未作为全量逐条过滤器。原始参考、修正参考、音频、逐条分数、ASR 和一致性报告均保存在 `preflight/dialogue-quality/`。

源 README 记录少量 opus 来源 recording 的短句边界存在约 13–73 ms 偏移。本轮没有全量强制对齐，也不能靠 DNSMOS 保证所有文本和边界正确；重叠检查同样只能发现标注中的重叠。此限制保留在训练数据说明中。

已有中英各约 5,000 小时、合计 6,870,254 条的编码池可复用。仅在 ID、实际音频 locator、codec 权重和编码配方相符时复用，并逐文件校验 NPZ 哈希；选择了不同载体的同 ID 样本重新编码。

全量数据目录：

```text
/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_prepared/LM-TTS-Training/emilia-full-en-zh/
  inventory/                 每个 tar 的实际统计
  candidates/{short,long,dialogue}/
  dialogue-quality/          实际切片评分、过滤后候选及逐分片统计
  DIALOGUE_QUALITY_COMPLETE.json
  INVENTORY_COMPLETE.json    载体与未去重视图统计
  raw/                      按 ID 去重后的编码输入分片
  EXPORT_COMPLETE.json      去重后的语言、时长、条数、speaker 统计
  prepared/、codes/          编码分片与新增 NPZ
  train.jsonl、val.jsonl     编码完成、全局验证划分后发布
  preparation.json、PREPARATION_COMPLETE
```

## 中英文 1:1 与训练预算

1:1 按**音频时长**定义，不是样本条数，也不要求每个 batch 的条数各半。每个平衡 epoch 将较多语言完整遍历一次；较少语言先完整遍历，再重新打乱并重复采样，直到两者音频时长相等。最终差异受最后一条短句及原有分布式尾部处理限制，不丢弃较多语言的大量数据来凑比例。

先训练 **1 个完整的平衡 epoch**。编码完成后，流水线用真实 codec 帧数、文本 token 数和四卡动态组批算法计算实际步数，写入 `training-plan.json`，并据此设置 `max_steps=schedule_steps`。

| 设置 | 本轮计划 |
|---|---|
| 初始化 | `pretrained/assembled-qwen3-tts-frozen-conditioning`，与此前实验相同的初始组装模型 |
| Talker 主干 | Qwen3-0.6B-Base 初始化；音频预测模块为原始随机初始化权重 |
| 冻结部分 | 文本 embedding/projection 与说话人 encoder 保持冻结 |
| loss | `token`；residual weight 0.3 |
| 采样 | 全量中英文、按音频时长 1:1 |
| 硬件/布局 | 四张 A100 80GB，FSDP、BF16、Flash Attention 2、padding-free |
| 每卡预算 | 6,000 codec frames / 9,000 Talker tokens；accumulation 1 |
| 峰值学习率 | 主干 1e-4，新模块 3e-4 |
| 调度 | 1,000 步 warmup，cosine 至峰值的 10% |
| weight decay / grad clip | 0.01 / 1.0 |
| 保存与验证 | 每 500 步 checkpoint 和验证 loss，保留最后两份完整 checkpoint |
| 小样本生成 | 每 2,000 步；speaker only、ICL 各 8 条；最终步同样评估 |
| 验证集 | 每语言 256 条；规范化文本与训练集不重叠，同 speaker 保留至少两条训练参考 |

正式配置 `configs/emilia-full-token-pretrain.yaml` 在编码完成、步数计算后生成。正式训练启动前还会用全量数据中最长音频和最长文本做四卡显存压力检查。

## 已完成验证

- 时长平衡采样测试覆盖全量覆盖、少数语言重复、四卡等步数、预算约束、跨 epoch 打乱和断点恢复；原有动态组批测试同样通过。
- 元数据测试覆盖重叠视图的 ID 去重、载体优先级和越界短句排除；既有来源解码、数据加载和准备测试通过。
- 四卡预检从 step 0 完成 20 次真实更新和验证 loss，指标有限、checkpoint 完整。该预检是独立小数据测试，不计作正式全量训练。
- 预检 checkpoint 的 81 个冻结张量、共 326,313,792 个参数与原始组装模型逐值一致。
- 真实 codec 检查成功复用 4 条 long 短句的既有 NPZ，另编码 4 条 dialogue 短句；复用的路径和哈希逐条一致。

验证产物保存于 `runs/emilia-full-token-scratch/preflight/`。

## 执行与状态

入口：

```bash
PYTHONPATH=. .venv/bin/python -u scripts/run_emilia_full.py
```

流程为清点 → dialogue 质量筛选与 short/long 导出、四卡编码并行 → 汇入通过筛选的 dialogue 并全局去重 → 发布 train/val → 计算训练计划 → 四卡显存检查 → 从原始组装模型训练 → 最终冻结检查。第一次正式启动不带 `--resume`；如果本轮训练之后中断，恢复仅加载本轮自己的完整 checkpoint。

运行目录为 `runs/emilia-full-token-scratch/`。`pipeline-status.json` 记录阶段、命令、进程身份和退出码；`data-preparation/` 保存准备日志；正式训练写入 `train.log`、`tensorboard/`、`checkpoints/` 和 `evaluation/`。只有编码完整发布后才进入正式训练。

用户已明确授权数据处理完成后直接以 token loss 开训并后台监督。巡检沿用 `scripts/monitor_training.py`，每 1,800 秒检查一次，覆盖数据准备和正式训练，记录于 [全量训练监督记录](emilia-full-token-supervision.md)。本轮不使用旧 10kh 的 19,000 步停止点，实际目标步数由一个时长平衡 epoch 计算。
