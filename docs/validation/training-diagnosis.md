# 训练正确性与诊断记录

日期：2026-09-08。当前主线是 [公开文本前端冻结的 Emilia 1,000 小时基线](../training/emilia-baseline.md)。以下区分实现正确性与生成质量；小样本通过不能证明未见文本或说话人的泛化。

## 当前基线验证

- 组装模型总参数 914,643,008，可训练 597,183,552。text embedding 与两层 projector 从公开 Qwen3-TTS-0.6B 成套加载，317,459,456 参数冻结；Talker transformer 从 Qwen3-0.6B-Base 初始化。
- 公开 wrapper 可重载组装产物；训练冻结由项目 loader 执行，使用公开 wrapper 进行其他训练时需自行冻结。
- 官方非流式 Auto 输入：测试截获官方生成／ICL 路径的输入 embedding，与项目输入逐项对照；覆盖不同文本长度和 0/1/3 历史音频帧。teacher forcing 与逐帧推理及官方 Code Predictor 的标签对齐通过。
- 双卡 FSDP2 与未分片模型的梯度对照：包含 ECAPA、冻结前端、变长 padding、梯度累积 2；最大绝对误差 1.862645149230957e-7。
- 实际 Emilia short pilot 产生 11 train / 2 val（丢弃 3 条 singleton）；完成 3-step 双卡训练。读取训练后 DCP，5 个文本前端张量共 317,459,456 参数与初始化完全一致。
- 同一 pilot 连续训练和中断恢复到 step 3，1,901 个张量精确相等，进度同为 epoch 1 / next_batch 1。这里只验证训练／恢复流程；4 帧生成上限不能评估语音质量。
- 预处理缓存重跑已通过，train/val manifest 字节一致。数据筛选测试覆盖顶层 long/dialogue 内存在 short 字段但不得入选、伪装成完整 short 的片段必须拒绝；参考验证覆盖训练池外引用、错误说话人、自引用和同音频引用。

日志与产物保存在 `runs/emilia-frozen-integration`，不提交大型 checkpoint 或音频到 Git。27 个单元测试全部通过，`pip check` 无依赖冲突，Python 编译检查通过。单元测试入口为 `PYTHONPATH=. .venv/bin/python -m unittest discover -s tests`。

## LJSpeech 诊断，不能作为最终结构选择

历史实验仅用于定位 loss / 生成 / 过拟合。旧协议将整段文本作为前缀；它与当前官方非流式双轨输入不同。

| 实验 | 数据 / 更新 | 验证首码本 CE | 验证残余 CE | 验证生成原始 WER |
|---|---|---:|---:|---:|
| 旧协议，64 条过拟合 | 64 train / 8 val，400 steps | 11.5684 | 8.0281 | 25/22 = 1.1364（2 条） |
| 旧协议，较大 batch | 2048 train / 128 val，1000 steps | 9.7988 | 9.0193 | 76/67 = 1.1343（3 条） |
| 官方输入，同源 embedding 直连 | 同上，250 steps | 9.0874 | 7.5259 | 73/67 = 1.0896（3 条） |
| 官方输入，同源 embedding + 随机 MLP | 同上，250 steps | 9.8716 | 7.4514 | 90/67 = 1.3433（3 条） |

WER 可因插入错误超过 1。这些生成样本数量小，而且各实验更新次数不同；不能做显著性或跨实验因果结论。直连和随机 MLP 的配对实验使用相同其余 398 个 Talker 张量、采样顺序、每卡 batch 48、学习率和 250 次更新；两者均未得到可靠泛化，不足以认定 projector 是唯一问题。

64 条过拟合实验在固定训练文本上生成 WER=0，teacher-forced 16 码本准确率全部为 1；换成另一个合法训练文本时生成该训练样本的码。反转文本 token 顺序仍能生成记忆中的句子，提示模型记忆而非泛化。该实验从 step 200 恢复到 400 与连续训练的 1,916 个张量精确一致。

1000-step 旧协议实验的两个训练样本经英语规范化后 WER=2/35、CER=0，验证仍很差；其中数字文字与数字、pieman 与 pie man 等 ASR 表达差异应单独看待。项目保留原始 WER/CER，补充英语规范化分数，并对目标原音频做 ASR 对照，不覆盖原始转写。

直连配置从 step 125 恢复到 250 与连续训练的 1,900 个张量精确一致。诊断脚本须保留 RoPE buffer 的 FP32 精度；曾将整个模型包括 buffer 转 BF16 的旧诊断不作为证据，训练结果不受该诊断脚本问题影响。

## 资源与下一步

LJSpeech 合成长文本／长音频压力测试：batch 64 OOM；batch 48 完整优化两步峰值 allocated 58.52 GiB/GPU，reserved 63.54 GiB/GPU。这是旧直连／小文本数据结果，不能代替 Emilia 冻结基线的显存测试。

当前 Emilia 目标 1,000 原始小时，实际训练小时由筛选和划分后的 `preparation.json` 给出。自动流程按导出 → 批量 codec 编码 → 实际数据显存测试 → 5,000-step 训练 → 冻结权重复核执行。是否完成以 `pipeline-status.json`、checkpoint 和评估日志为准。后续再用同一数据和训练预算比较文本前端冻结／解冻、初始化与直连结构。
