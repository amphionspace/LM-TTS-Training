# 2026-10-09：代码与目录整理验证

本次处理审查中的六项问题，同时整理文件职责。模型网络、初始化来源、参数名称与已有组装产物保持不变；没有提交正式训练，也没有修改现有任务的 submission 快照。

## 修复范围

| 问题 | 处理 |
| --- | --- |
| 坏 text IDs 可能触发 GPU embedding 越界 | 读取阶段检查整数类型、非负和模型词表上界，按坏样本跳过；worker 重建时保留词表限制 |
| 导出产物无法由训练模型加载器重载，LFM 缺少合成入口 | 区分 assembly 与 export，校验导出权重、配置和 tokenizer；Qwen/LFM pure-codec 共用生成入口，支持纯文本和 codec-prefix ICL |
| LFM 配置误走 Qwen 准备流程 | `assembly.family: lfm2` 显式分派；缺省仍为 Qwen；已有模型类型不匹配时，在复制或组装前报错 |
| eval 失败导致同一步更新未保存 | 先保存完整 checkpoint，再执行 eval；状态单独写入 `validation/step-XXXXXXXX.json`，resume 先重试待完成的 eval |
| 训练器依赖具体主干的层、norm 和配置 | 模型提供分片单元、activation checkpointing 和学习率分组接口；LFM 用专门的配置容器，兼容读取旧组装配置 |
| LFM 只覆盖简化训练测试 | 真实 engine 跑双卡动态 batching、accumulation、3 epoch WSD、坏数据、eval 故障、worker/prefetch 变化及导出重载；另测完整 378M 产物 |

目录分为 `models/assembly/`（离线组装）、`models/lfm/`（适配器／主干／配置）、`inference/`（生成）和 `training/validation.py`（eval 状态与恢复）。CLI 路径继续保留在 `scripts/`；具体操作见 [README](../../README.md)。

## 验证结果

使用 `envs/lm-tts`，GPU 测试仅在开发机的空闲设备 2、3 运行。

| 验证 | 结果 |
| --- | --- |
| 完整 pytest，含 FA2 packed 测试 | 110 项通过，无跳过 |
| 后续 ICL 分词边界修正 | 生成测试 9 项通过，包含 1 项新增回归；参考与目标分别 tokenize，再拼接 IDs，避免跨边界合并词语 |
| Qwen 双卡实际 engine | 3 epoch；eval 故障恢复、再次中断恢复、worker 从 1 改为 2、prefetch 改为 3 后，逐步 loss／LR／样本数一致，最终权重最大差 0 |
| LFM 双卡实际 engine | 同上；并验证导出重载与两种 codec 生成路径，最终权重最大差 0 |
| 现有 LFM 378M 产物 | 本地准备入口校验通过；双卡更新与恢复后权重最大差 0，单卡峰值显存 3.415 GiB |
| 真实 LFM 合成 CLI | FP32 导出、BF16 加载；纯文本和 codec-prefix 均生成 24 kHz、5760 采样点的有效音频；参考为测试信号，仅检查流程 |
| 真实 DCP 导出 | Qwen speaker、Qwen pure-codec、LFM pure-codec 均可重载；逐张权重相等，pure-codec 生成结果相等 |

双卡 engine 测试故意破坏一条 codec 和一条 text ID，确认健康样本继续训练；在第一步 eval 注入异常，确认该步 `COMPLETE` checkpoint 已存在，恢复不会重复优化器更新。

本地结果位于 `artifacts/quality-lfm-20261009-v2/result.json`、`artifacts/quality-qwen-20261009/result.json` 和 `artifacts/quality-lfm-real-20261009/validation-rank-*.json`。这些测试产物不纳入 Git。真实模型合成的记录在同目录 `audio-validation.json`，音频在 `generated-final/`。临时的完整模型测试权重和首轮失败的测试目录已清理，保留验证摘要与试听文件。

## 使用边界

Pure-codec 生成逐帧重算完整前缀，尚无 KV cache；可以验证生成正确性，不能视为已完成推理吞吐优化。ICL 必须同时提供参考音频和准确的参考文本。上述测试不构成训练收敛、音质或正式多机吞吐的评估。
