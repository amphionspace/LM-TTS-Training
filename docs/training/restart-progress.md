# 当前进展与重启恢复（2026-09-09）

本文保留当时的迁移与恢复实验记录。当前训练已统一转向 [dynamic batching](dynamic-batching.md)，旧布局迁移兼容代码已删除；下文的历史恢复命令不再作为当前入口。1kh 运行已完成 5,000 步，见 [最终报告](emilia-1kh-report.md)。

## 已暂停并保留的进度

- 唯一工作目录：`/119010446/LM-TTS-Training`。padding-free 源码已归入此目录，临时 worktree 已移除。
- 暂停时的源码另存为 `runs/restart-source-20260909.tar.gz`，已有文件的 Git diff 另存为 `runs/restart-source-20260909.patch`；正常重启后直接使用工作目录，无需解包。
- 正式训练和四卡短测进程均已停止，没有启动自动恢复任务。
- 正式 checkpoint：`runs/emilia-en-zh-pretrain-1000h/checkpoints/step-00001000`，`COMPLETE` 标记存在，进度为 `step=1000, epoch=0, next_batch=1000`。模型、AdamW、scheduler 和四个 rank 的随机状态均在存档中。第 500 步 checkpoint 也保留。
- 原训练使用四张 A100 80GB、FSDP2、BF16、Flash Attention 2，每卡 48 条、全局 192 条，学习率计划总计 5000 步。恢复测试保持这些设置。
- 英文和中文各约 500 小时的数据已准备完成，不需要重新准备。数据根目录为 `/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_prepared/LM-TTS-Training/emilia-short-en-zh-1000h`。
- `.venv`、预训练模型、Flash Attention 预编译 wheel 和模型缓存均保留。

第 1000 步验证已完成：首码本 CE **3.18335**，残余码本 CE **7.23562**。生成音频为英文、中文各 4 条，位于 `runs/emilia-en-zh-pretrain-1000h/evaluation/step-00001000/`。小样本英文 WER **92.05%**、中文 CER **90.91%**，内容准确性仍不足。

## padding-free 验证状态

34 项测试和双卡 BF16 FSDP2 梯度检查已通过；现有第 500 步完整 checkpoint 的 CPU FP32 恢复、一次更新及保存再恢复检查已通过。详细结果见 [padding-free 说明](padding-free.md) 和 `runs/padding-free-validation/`。

完整四卡、每卡 48 条、从第 1000 步恢复的两步短测已通过，并保存 `runs/padding-free-validation/resume-1000/checkpoints/step-00001002`。该存档重新加载后的验证首 CE 为 **3.15693**，残余 CE 为 **7.23500**。另生成 2 条验证样本及 2 条训练样本音频，位于 `runs/padding-free-validation/eval-1002/`。其中一条训练样本达到 160 帧生成上限。正式 padding-free 长训尚未启动。布局迁移保留优化器与采样进度，但不承诺 BF16 训练轨迹逐值相同。

## 重启后复查与后续恢复

先确认上述项目目录与 `/ai_sds_wuzz` 数据挂载仍在，以及 `nvidia-smi` 能看到四张 GPU。已通过的短测可用下面命令复查；输出使用新目录，避免覆盖已有结果：

```bash
cd /119010446/LM-TTS-Training
NPROC_PER_NODE=4 bash scripts/run_train.sh \
  --config runs/padding-free-validation/resume-1000.yaml \
  --resume /119010446/LM-TTS-Training/runs/emilia-en-zh-pretrain-1000h/checkpoints/step-00001000 \
  --output runs/padding-free-validation/resume-1000-recheck \
  > runs/padding-free-validation/resume-1000-recheck.log 2>&1
```

此配置从 1000 更新到 1002 自动退出，使用原来的 5000 步学习率计划。复查输出仅写入 `runs/padding-free-validation/resume-1000-recheck/`，不会覆盖原训练 checkpoint。预期日志包含布局迁移、恢复到 1000、两次有限 loss／梯度范数，以及完整的 1002 步 checkpoint。

已有 1002 步存档可单独运行验证与生成：

```bash
NPROC_PER_NODE=4 bash scripts/run_train.sh \
  --config runs/padding-free-validation/resume-1000.yaml \
  --resume runs/padding-free-validation/resume-1000/checkpoints/step-00001002 \
  --eval-only --eval-samples 2 \
  --output runs/padding-free-validation/eval-1002 \
  > runs/padding-free-validation/eval-1002.log 2>&1
```

先查看短测及音频结果，再决定正式续训。后续可以从新的 packed 第 1002 步存档继续，也保留了原 padded 第 1000 步存档；应保持四卡、每卡 48 条和原学习率计划。不要重新运行数据准备或初始化流程。

## 清理记录

清理了四个集成测试 run 的临时 checkpoint、一次性诊断脚本、重复日志及旧 SDPA 试跑副本，约 **39.0 GiB**。正式训练数据、模型、历史训练 checkpoint、评估音频及最终验证证据保留。具体清单见 `runs/restart-cleanup-20260909.json`。
