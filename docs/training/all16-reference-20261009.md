# All16 裁剪 speaker reference 实验

本轮测试：speaker embedding 只来自目标音频的一段参考片段，且这段 codec 不参与目标监督，能否减轻模型对完整目标 speaker embedding 的依赖。完整文本和 codec 输入保持不变。这仍然是同一条音频内的参考片段实验，不能当作跨音频 speaker-only 泛化已经解决的证据；后续需用 Seed-TTS 的 speaker-only / ICL 生成评测判断。

## 配置与启动

实验文件：[refmask YAML](../../configs/supervised-tts-20260929-all16-refmask-20261009-bf16-16gpu-acc2-lr3e-4-bblr1e-4-ep3-wsd.yaml)。数据绑定：[data-reference.yaml](../../configs/data-reference.yaml)。两者都直接继承 `base.yaml`。

| 项目 | 本轮设置 |
| --- | --- |
| Run | `supervised-tts-20260929-all16-refmask-20261009-bf16-16gpu-acc2-lr3e-4-bblr1e-4-ep3-wsd` |
| 初始化 | `assets/assembled/qwen3-tts-frozen-conditioning`；重新开始，不加载历史训练 checkpoint |
| 冻结 | 官方 TTS text embedding / projector、speaker encoder；codec 不参与训练 |
| 更新 | Talker 主干、16 组 codec embedding、音频 head、Code Predictor |
| 资源 | 2 节点 × 8 A800；BF16；每卡累积 2 次 |
| Batch 预算 | 每卡 32,000 codec frames / 50,000 Talker tokens，保持完整序列预算 |
| LR | 新模块 `3e-4`、backbone `1e-4` |
| 调度 | 3 epoch；WSD 按已消费样本进度：1% warmup、最后 10% decay、最低为峰值的 10% |
| Loss | token reduction；first CE + `0.3 × residual CE` |
| 评估 / 保存 | 第 100 步，随后每 2500 步；保留全部 checkpoint |

```bash
PY=/workspace/workspace/yanglin/envs/lm-tts/bin/python
CFG=configs/supervised-tts-20260929-all16-refmask-20261009-bf16-16gpu-acc2-lr3e-4-bblr1e-4-ep3-wsd.yaml
$PY -m scripts.build_unified --config "$CFG"       # 仅在新 build 尚未生成时执行。
$PY -m scripts.acp.submit --config "$CFG"          # 检查提交配置。
$PY -m scripts.acp.submit --config "$CFG" --submit # 同一个 run 只提交一次。
```

输出均在 `/workspace/LM-TTS-Training-Runs/<run_name>/`，包括 `metrics.jsonl`、`tensorboard/`、`logs/`、`checkpoints/` 与 `submissions/`。只有本轮任务停止且需要恢复时，使用相同配置加 `--resume latest --submit`；不可与旧版全帧 loss 或 no-speaker 模式交叉恢复。

## 数据与划分

16 个数据集使用 `*-merged-reference-features-v1-20261009T000157bjt-01`。原始 selection 有 128,220,178 条，新 merged 保留 128,094,617 条，过滤 125,561 条没有合法参考区间的样本。完整 codec 和 text 来源不变，speaker embedding 替换为参考片段提取结果。

每条样本只有一个已发布的参考片段：占实际音频时长的 10%–50%，最短要求 0.5 秒，并按 80 ms codec 网格对齐，因此最少 7 帧（约 0.56 秒）。片段位置可以在开头、中间或末尾，每个 epoch 保持不变。训练读取已发布的半开区间 `[start, end)`，校验 codec 身份、native audio 坐标和覆盖范围；不重新裁剪音频、不在线计算 speaker embedding。

`reuse_build` 指向原始 all16 完整 build。新构建逐行匹配 target ID，核对 audio hash、text、tokenizer 与 codec/text 来源，复用已有 text token ID；它不影响 forward 中的 embedding / projector 计算。训练 / 验证归属继承原 target ID，过滤样本直接从原归属移除，避免重新抽样导致旧验证样本进入训练。

新 build 位于 `data/builds/supervised-tts-20260929-reference-20261009-all16/`，训练、验证分别读取其 `train/manifest.json`、`validation/manifest.json`。build 独立写入本仓库；codec / speaker 大字段引用 pipeline 的固定 Lance 快照，源数据必须保留。

## Mask 与 CE

`model.use_speaker_embedding: true` 输入裁剪片段的 embedding，`train.mask_reference: true` 屏蔽其对应 codec 目标。两个开关与数据绑定同时校验，防止误用全音频 features 或意外对参考片段计算 CE。旧配置省略 mask 时仍使用旧行为。

设样本完整 codec 长度为 T、参考区间为 `[a,b)`，有效监督帧数 S = T − (b−a)：

- 输入仍包含全部文本、speaker 条件和 T 帧 codec，沿用 teacher forcing；不会删除、重排参考帧，也不会把它们移到 prefix。
- 首码本屏蔽预测目标索引 a 至 b−1；每条样本末尾的 EOS 保留。全局 first CE 分母是所有样本的 Σ(S+1)。
- residual 的 15 个码本对同一批帧全部屏蔽。全局 residual CE 分母是 `15 × ΣS`，不是先取每卡均值再平均。
- 累积窗口内所有 microbatch 与所有 rank 共用全局分母；验证使用同样的 mask 和归一化。
- 屏蔽的是输出目标的 CE，输入历史仍可影响后续预测并接收梯度；计算量和显存预算继续按完整序列计算。

训练中的 `codec_tokens` 统计有效监督 token（包含每条 EOS）；吞吐比较优先看完整 `talker_tokens`、音频秒/秒及 step 耗时。新旧 CE 的监督范围不同，不直接以绝对 loss 高低判断实验优劣。

## 验证与任务记录

已通过 38 项数据、loss、配置、组批、调度与提交回归测试。reference 测试覆盖开头 / 中间 / 末尾 mask、EOS 梯度、三种 reduction、累积窗口归一化、过滤前后验证样本归属；Arrow 缓存路径另外验证不同 batch 边界和被过滤行的 token 对齐。两项需要 IPC 的测试在允许本地进程通信的环境中通过。

两卡 BF16 的真实 trainer / reader / evaluator / checkpoint 路径已通过：连续 4 步，与 2 步后中断并 resume 至 4 步，最终所有权重逐项一致，最大差为 **0**。Talker、codec embedding、head 与 Code Predictor 均有更新，冻结模块完全不变。验证在第 2、4 步执行，未出现坏样本或非有限 loss。

真实 AISHELL 试建保留 87,968 条（训练 87,880、验证 88）。全量构建正在核对全部行；正式任务 ID 与完成后的覆盖数在提交时补充。
