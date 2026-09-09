# 训练正确性与诊断记录

日期：2026-09-08。当前主线是 [公开文本前端冻结的 Emilia 1,000 小时基线](../training/emilia-baseline.md)。以下区分实现正确性与生成质量；小样本通过不能证明未见文本或说话人的泛化。

## 2026-09-09：数据恢复与冻结 ECAPA

上次自动任务在编码 364,496 条、约 438.06 小时后失败，未进入正式训练。失败样本 `emilia2:52634db1dbf9fd7f_004_000` 标注 309,259 点，PyAV 解码 309,248 点，44.1 kHz 下相差约 0.25 ms；MP4 movie timescale 为 1,000。现在只对不超过 1 ms 的尾部缺口补零并记录日志，超过阈值继续拒绝。真实失败样本修复后为 24 kHz / 168,305 点，放大长度偏差的反例仍被拒绝。WAV 现在也原子发布，避免中断留下不完整文件。

原始清单共 831,387 条、146,488 个 speaker ID、1,000.000806 小时；筛选后的实际训练规模仍以最终 preparation 报告为准。全部 Emilia 数据迁到公共数据目录，rsync 分组完成后清理源端空目录，清单 SHA256 核对一致。

双卡 codec 编码对照单卡：13 条真实样本的所有 codec 哈希、划分、参考和时长完全一致；双卡缓存重跑的两份清单字节一致。内存波形路径与原 WAV 路径的 13 条 codec 哈希和清单字段也完全一致，缓存重跑一致。30 个单元测试通过。

按最新决定，新产物 `pretrained/assembled-qwen3-tts-frozen-conditioning` 同时冻结文本前端与公开 ECAPA。总参数 914,643,008，可训练 588,329,216。新真实 pilot 完成双卡 3 步训练；训练后 81 个冻结张量、326,313,792 个参数与初始化完全一致。FSDP 梯度对照最大绝对误差 1.862645149230957e-7。连续／恢复到 step 3 的 1,673 个张量精确一致。日志保存在 `runs/emilia-frozen-conditioning-integration/validation/`。

## 2026-09-08：此前冻结文本前端的验证

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

## 2026-09-09：checkpoint 空间管理

清理了 38 个已完成的旧验证／恢复对照与中间 checkpoint，释放约 264.68 GiB。保留六个主要实验的最终 checkpoint，日志、评估和音频保留；被清理 checkpoint 的 metadata 归档到各 run 的 `checkpoint-metadata/`。明细为 `runs/checkpoint-cleanup-20260909.json`。后续训练默认保留最新两个完整 checkpoint，原子发布新的 `latest` 后才清理，未完成写入不受影响；对应删除边界已单元测试。
