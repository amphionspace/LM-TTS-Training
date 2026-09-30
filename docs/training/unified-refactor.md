# Unified 重构验证记录

当前入口和操作步骤见根 README。本文只记录证据与限制。最终自动检查为 52 项 pytest 通过，Ruff、shell 语法与 pip check 通过。

## 真实数据

检查了 pipeline 的数据 / feature / speaker 约定、固定 selection，以及处理中和已完成的表。正式 selection 为 128,220,178 条，尚未全部发布，不以 `.incomplete` 作为训练输入。

另外使用 CSEMOTIONS 的完整真实录音验证：在训练侧独立目录调用现有 pipeline codec executor 提取 4,160 条 FP32 codec，绑定已完成的 4,160 条原始 speaker embedding，然后通过正式构建器和 reader 完整读取、组批。结果为 460,833 codec frames、140,571 文本字符、1024 维 speaker embedding；缓存 speaker 权重与本地完整 assembled 模型逐张量哈希一致。未绑定的正式 selection 共 128,216,018 条，明确记录为未就绪覆盖。

报告：`/workspace/LM-TTS-Training-Runs/real-features-csemotions/reader-report.json`。Build：`data/builds/csemotions-real-20260930/manifest.json`。两个 feature 来自不同输出根目录，固定发布 manifest 和源 version；build 只修改训练侧 shallow clone，不修改 unified / pipeline 源表及 branch。边界测试也检查源目录完整文件集合与哈希不变。

## 模型准备

从 `/workspace/model` 复制 Qwen3-0.6B-Base、Qwen3-TTS-12Hz-0.6B-Base、Qwen3-TTS-Tokenizer-12Hz 到 `assets/base/`，逐文件校验复制内容。本地组装 FP32 模型到 `assets/assembled/qwen3-tts-frozen-conditioning`，验证官方 wrapper、tokenizer、speaker encoder、24kHz codec 重新加载。训练直接加载完成产物，不在任务中组装。

## 分布式与恢复

- 两卡 FSDP2 FP32 / sqrt loss 对照参考实现：梯度相对误差 2.6472e-7。
- 两卡 FSDP2 BF16 / sample loss / FlashAttention：相对误差 0.0044811，测试容差 0.03。
- 两个独立 torchrun 节点入口：rendezvous / global rank / 梯度检查通过。
- ACP 两个真实物理节点、16 张 A800：训练和恢复完成，但首次 master 的验收脚本错误地检查了默认 fixture 行数；任务状态 FAILED。修复脚本后，保存结果离线验证全部权重逐位一致。没有将失败任务标为成功。
- ACP 作业 `pt-ughagq38`：四个真实物理节点、32 张不同 UUID 的 A800，状态 SUCCEEDED。BF16 / FP32 各进行连续 4 步和 2+2 步恢复，所有权重差异为 0，四组训练模块发生更新，冻结 text frontend / speaker 保持不变。实际跨机器 NCCL、spawn worker、梯度累积、activation checkpoint、DCP 和导出链路通过。

32 卡报告：`/workspace/LM-TTS-Training-Runs/acp-20260930-32gpu/result.json`。这些任务使用真实 Lance fixture 和 64 维微型模型，不能外推完整模型的显存或吞吐。当前的微型验证入口为 `scripts/acp/submit.py --validate`。

## 评分与 TensorBoard

独立加载固定 Whisper small、官方 DNSMOS P.835、WavLM SV。对真实 LJSpeech 原始录音和另一条独立参考录音评分：WER / CER 0、SIG 3.6152、BAK 4.0137、OVRL 3.3190、cosine 0.9296。这是评分接口 smoke，不是新模型生成表现。逐文件哈希和参数保存在 run 的 `scores/report.json`。

评估模型版本固定：

- `Systran/faster-whisper-small@536b0662742c02347bc0e980a01041f333bce120`
- `microsoft/wavlm-base-plus-sv@feb593a6c23c1cc3d9510425c29b0a14d2b07b1e`
- `microsoft/DNS-Challenge@591184a9fcb2cbdec02520fed81a32bbbf9d73ff`

默认 TensorBoard 写入 optimization / performance / batch 三组诊断、组合图、配置，以及独立评估分数和成对音频。回读测试检查 event 中的真实 step、分组开关、音频长度和重复恢复不重复写训练 scalar。历史 32 卡 run 已从 journal 补回此前未显示的诊断。

本轮默认配置已扩展为 13 项训练指标，单卡真实入口额外验证 4 步连续训练与迁移后的 2+2 步恢复，全部权重差异为 0；HTTP 接口实测可读取曲线、成对音频和组合图。报告位于 `/workspace/LM-TTS-Training-Runs/telemetry-check-fp32/comparison.json`。

实际启动还发现 TensorBoard 2.20.0 使用已从新 setuptools 删除的 `pkg_resources`，写 event 成功不足以证明页面能启动。依赖已升级并固定为官方 2.21.0，单独验证 HTTP 页面数据。

## ACP 和目录

提交适配只在 `scripts/acp/`，平台文件只在 `../acp/`；训练层读取标准 PyTorch 环境。训练与平台使用同一个实验 YAML，直接继承唯一 base；提交器单独读取 acp 区块，只有一处 run_name / runs_root。提交保存固定代码快照和 job ID，后续源码修改不影响已提交任务。

实际 API 支持 SSH 免密和 TensorBoard，但拒绝多层 AFS subdir，且要求 TensorBoard log_path 等于挂载点。用户将统一 Runs 目录选定为 `/workspace/LM-TTS-Training-Runs` 后，提交器可以直接挂载这一层，默认开启平台页面，不再为每个 run 创建一级目录。页面按 run 名区分实验；独立仪表盘仍可只读取指定 run/tensorboard。

32 卡历史任务创建时使用旧挂载，已结束任务的挂载不会随搬迁更新，使用独立仪表盘查看它的保存结果。历史 Runs 旧路径保留符号链接，固定版本 build / checkpoint 的原始引用继续有效。

## 2026-09-30 代码与文档检查

统一代码和配置注释为英文，主 README 按配置、准备、构建、训练、观测和评分顺序改写为中文。YAML 解析值与修改前一致。

本轮修正了实验 paths 未传入数据构建、TensorBoard 重建遗漏新增步数及重复写评分事件、checkpoint 导出未校验 assembled 身份的问题。哈希函数移至无模型 / 数据依赖的公共模块，删除重复实现和无作用的异常捕获；ACP 初始化入口改为显式 main，并遵循 acp.home。

验证：54 项测试通过（93.74 秒），Ruff 检查及格式检查、shell 语法、README 本地链接检查通过；ACP 4 节点请求只做渲染，未创建任务。另检查了 4 份既有 checkpoint 的组装身份匹配。TensorBoard 回读覆盖新增 / 重复训练步数、评分和预览去重、试听配置，以及重建失败时保留已有事件。未重新进行多机训练。

## 尚需正式数据验证的事项

当前 reader、训练、恢复、真实跨机和评分接口已分别验证。正式 selection 未隔离评估集，默认 `no_holdout`；不能从训练集复制样本冒充验证集。亿级数据热读吞吐、完整模型 batch / 显存预算、生成质量和跨语言泛化需要在对应 feature 发布后运行正式训练及隔离评估。当前不支持在线解冻 speaker、同 dataset 多 run subset 合并或语言 / 质量加权采样。
