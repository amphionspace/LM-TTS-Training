# ACP 提交

从训练 repo 根目录运行 `python -m scripts.acp.submit`。`--config` 与本地训练使用同一个实验 YAML；平台参数来自其中的 `acp` 区块。默认只渲染，`--submit` 创建任务，`--nodes` 覆盖节点数，`--resume latest` 恢复同名 run。

平台 CLI 下载、凭据、profile、密钥放在 repo 外的 `../acp/`。`python -m scripts.acp.init --config configs/train-bf16.yaml` 初始化已安装 CLI 的 profile，平台目录读取 `acp.home`；`source scripts/acp/env.sh` 设置平台文件路径。提交使用官方 API，以支持 SSH 免密和可选的专属 TensorBoard 挂载。

`launch.sh` 只把平台变量映射为标准 PyTorch 变量。训练模块不依赖这里的代码。提交保存代码、实验配置、请求和 job ID 到 run 的 `submissions/`，实际任务运行固定代码快照。

完整准备、提交、恢复、run 目录和验证用法见 [主 README](../../README.md)。
