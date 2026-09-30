# Unified 重构验收 · 2026-09-30

当前入口见 [README](../../README.md)。本次移除旧 JSONL / NPZ 训练适配、Emilia 专用数据准备、巡检及实验调度脚本；历史 run 和已有数据、checkpoint、评估音频、源码归档没有清理。

## 已验证

| 检查 | 结果 |
| --- | --- |
| 模型组装、输入协议、teacher forcing、样本隔离、梯度与 loss reduction | 最终完整回归 42 项通过 |
| 真实 Lance build / branch / version、逆序 speaker join、批量读取、spawn worker、发布拒绝、错误身份、重复 ID、未绑定覆盖 | 4 项通过；源 main 不添加训练列 |
| 两卡 FSDP2 FP32 / sqrt loss，冻结 text frontend / speaker | 梯度相对误差 2.6472e-7 |
| 两卡 FSDP2 BF16 / sample loss，FlashAttention | 梯度相对误差 0.0044811，低于测试 0.03 容差 |
| 两个独立 torchrun 节点入口，各一张卡，token loss | rendezvous / global rank / 梯度通过，相对误差 2.5700e-7 |
| 真实训练入口：Lance → DataLoader → FSDP → 累积 → clipping → AdamW → DCP → 导出 | BF16、FP32 各运行 4 步，连续训练与 2+2 步恢复的全部权重逐位相同 |
| DNSMOS / WavLM / 多语言 Whisper 实际权重与真实 unified 录音 | 全部评分接口及 TensorBoard 写入成功 |
| Ruff、shell 语法、Python 编译、pip check | 通过 |

训练链路使用 8 条真实 Lance 格式 fixture、64 维微型 Talker / speaker、每卡一个 spawn worker、两次累积、activation checkpointing。数据索引和数值断点测试有实际读写，不只检查 mock。产物在 Git 忽略的 `artifacts/resume-fp32-v2/`、`artifacts/resume-bf16/`。

多节点测试的两个节点进程运行在**同一物理机**；它验证节点启动与 global/local rank 边界，不能替代真实跨机器 RoCE、交换机和 ACP 容器验证。没有创建收费 ACP 作业。

## 评分 smoke

从 unified `ljspeech/v0.1/samples.lance` 固定 version=2 读取两条原始录音，以其中一条检查内容与音质，另一条作为独立 speaker reference。得到 WER/CER=0、DNSMOS SIG=3.6152 / BAK=4.0137 / OVRL=3.3190、WavLM cosine=0.9296。**这是原始录音评分接口的 smoke，不是新训练模型的生成结果。** 文件、模型和评分配置的哈希在本地 `artifacts/evaluation-smoke/final-results/report.json`。

评估模型固定版本：

- Whisper：`Systran/faster-whisper-small@536b0662742c02347bc0e980a01041f333bce120`。
- WavLM：`microsoft/wavlm-base-plus-sv@feb593a6c23c1cc3d9510425c29b0a14d2b07b1e`。
- DNSMOS：`microsoft/DNS-Challenge@591184a9fcb2cbdec02520fed81a32bbbf9d73ff` 的非个性化 P.835 ONNX。

WavLM 本实现要求至少一秒的有限音频；过短或非法生成不会以零向量替代。评分保存 corpus 误差分母和每条结果；中文、日文、韩文仪表盘使用 CER。音质 / similarity 跨语言的适用性需按模型训练分布解释，不能把相似度当作人工 MOS 或身份真实性保证。

## 数据与生产边界

已检查 pipeline 的 06 / 07 / 11 / 12 数据约定、当前 selection manifest、已写入的 codec checkpoint 分片和已完成 speaker 测试表。codec 是 int16 `[T,16]`，speaker 是 float32 `[1024]`，两者按目标、父样本、音频哈希、原生区间和采样率绑定；不要求两种 frontend 的 encoder input 摘要相同。

当前 128,220,178 条 selection 的 evaluation 为 `not_assigned`。正式 codec / speaker 尚未全部发布；`.incomplete` 中的 codec 尚无最终 text/language，不能按完整 feature table 读取。默认配置采用 `no_holdout`，不伪造 validation 或独立参考克隆结论。

当前 build 只支持 sample、self、speaker-only、冻结 embedding，且每 dataset 一组完整 codec / speaker 表。在线解冻、多 run subset 合并及通用语言 / 质量加权不在本实现中；训练 plan 明确记录均匀 utterance、无放回和不足 rank 数时丢尾。它可以读取当前选择的所有语言。完整 codec binding 对选中 ID 逐条对账，speaker 缺失 / 失败保留未就绪原因，未绑定 dataset 保存固定 selection 源引用与数量。

构建器不复制大列：生成 codec build branch 和 mmap sampling 元信息；一次性 SQLite 位于本地临时目录。失败 build 不发布 manifest，使用新 build ID 重建。本次没有修改 pipeline repo 或未完成的 feature。当前数据仍在处理，因此无法完成亿级就绪数据的真实热读吞吐、生产显存 / batch 预算和跨物理机性能验收；这些需要发布后用真实 batch 测量，不能用本次微型结果外推。

ACP 已只读核对 workspace、集群规格、现有作业镜像和默认 AFS。CLI、适配、凭据和提交配置均在训练 repo 外部的 `../acp/`。默认提交入口只渲染 argv，显式 `--submit` 才创建任务。已安装 CLI 没有 TensorBoard log path 参数；日志在 AFS，可在平台控制台绑定同一路径。
