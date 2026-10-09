# LFM Base 的 pure codec 组装

本方案使用本地 `assets/base/LFM2.5-230M-Base` 的 tokenizer、文本 embedding 和完整主干，保留 Qwen 12Hz 的 16 码本协议与 5 层 Code Predictor。Pure codec 指保留文本和 codec、关闭 speaker embedding；指定音色需要参考 codec 条件。

## 初始化和更新范围

| 模块 | 来源 | 训练时更新 |
| --- | --- | --- |
| Text embedding，65536 × 1024 | LFM Base；补充 TTS token 对应行重新随机初始化 | 是 |
| Text projector | Identity，无参数 | — |
| Talker：8 层卷积、6 层 attention、最终 norm | LFM Base | 是 |
| 首码本 embedding/head | 随机初始化，3072 类 | 是 |
| Code Predictor：5 层、hidden 1024 | 随机初始化；15 张 embedding 和 15 个 2048 类 head | 是 |
| Speaker encoder | 不组装、不读取 speaker embedding | — |
| Codec encoder/decoder | Qwen3-TTS-Tokenizer-12Hz，独立存放 | 否 |

组装模型约 378M 参数，不含独立 codec。没有复用 LFM 的文本输出 head；文本 embedding 直接作为输入，与音频输出 head 不绑定。残差 embedding 仍同时用于 Talker 的帧输入求和与 Predictor 的 teacher forcing。

## 组装和加载

在仓库根目录运行，整个过程只读取本地模型：

```bash
PY=/workspace/workspace/yanglin/envs/lm-tts/bin/python
$PY -m scripts.assemble_lfm_tts
```

默认输出 `assets/assembled/lfm2.5-230m-base-pure-codec/`。可通过 `--backbone`、`--tts-template`、`--codec`、`--output` 覆盖路径，已有输出不会被覆盖。默认 FP32 保存，BF16 由训练精度配置控制；LFM 来源权重本身为 BF16。组装器逐张核对预训练权重、特殊 token 行和保存后的完整重载，完成后才发布目录及 `ASSEMBLY_COMPLETE`。

产物包含模型权重、配置、可由当前 AutoTokenizer 加载的 tokenizer、独立 `speech_tokenizer/` 和记录来源哈希的 `assembly_report.json`。这是本仓库的 `lfm2_tts` 格式，不是官方 Qwen3-TTS 模型格式：

```python
from lm_tts.models.loading import load_model

model = load_model(
    "assets/assembled/lfm2.5-230m-base-pure-codec",
    attn_implementation="sdpa",  # BF16 GPU training uses flash_attention_2.
    use_speaker_embedding=False,
)
```

训练入口与 checkpoint 导出按组装配置选择模型；FSDP、loss、优化器和恢复流程共用。已有 `scripts/synthesize.py` 是官方 Qwen speaker-only 入口，不适用于该模型。公共 `mode="next_frame"` 支持完整前缀的 greedy codec 生成，用于正确性验证；尚未提供 LFM 缓存加速的生成服务。

## Packed 正确性和精度

Attention 使用 FA2 的样本边界；每层卷积根据重新从 0 开始的位置 ID，屏蔽跨样本的历史项。只重置 RoPE 或 attention 边界不足以隔离卷积。短卷积使用原始预训练的门控、卷积权重和输出投影，以 FP32 累加卷积项，再回到计算 dtype；不依赖额外安装 causal-conv1d。

BF16 路径保持无 padding 的 packed Talker，Code Predictor 使用固定长度 SDPA。FP32 路径使用逐样本 padding 和 SDPA。训练与完整前缀生成关闭 KV/卷积 cache，显式请求 cache 会报错，避免使用尚未适配的缓存语义。生产吞吐和大 batch 显存仍需要正式训练前实测。

## 数据和实验配置

[LFM 配置](../../configs/lfm2.5-230m-pure-codec.yaml) 直接继承 base，关闭 speaker embedding，预训练 text embedding 与主干使用 `backbone_lr`，其余音频模块使用 `lr`，保存全部 checkpoint。该配置用于后续准备，并不代表任务已提交。

新 tokenizer 必须重新构建文本 token 与长度索引，不能复用 Qwen text IDs。配置继续引用 all16 数据 recipe，新 build 名带 `lfm2.5-230m`，目录布局仍为原来的 root/dataset/train/validation 结构，codec 无需重新提取。准备数据时使用：

```bash
$PY -m scripts.build_unified --config configs/lfm2.5-230m-pure-codec.yaml
```

本次模型接入不执行全量数据构建或提交训练。旧 Qwen 的 checkpoint、组装目录与 submission 代码快照均保持原样；包名调整为 `lm_tts`，旧 submission 快照保留原包名及启动命令。

## 本次验证（2026-10-09）

- 组装参数总数 **377,554,944**，其中 Code Predictor 为 **141,570,304**；组装模型内全部参数参与训练，独立 codec 不计入其中。
- 原生 LFM 与适配主干的 FP32 输出、梯度一致；packed 与逐样本 loss/梯度一致，跨样本和未来音频隔离通过。
- GPU BF16/FA2、activation checkpointing、teacher forcing 与逐码本生成对齐检查通过。
- 完整组装模型在两张 A800 上通过 FSDP2 BF16 更新及恢复验证。保存一步后，连续执行下一步与加载 checkpoint 后重做下一步，权重最大差值为 **0**，loss 完全相同。小 batch 验证每卡峰值 allocated memory 约 **3.42 GiB**，不代表生产 batch 的显存需求。

可复现两卡验证：

```bash
CUDA_VISIBLE_DEVICES=2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
$PY -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m tests.check_lfm_resume \
  --assembled-model assets/assembled/lfm2.5-230m-base-pure-codec \
  --output artifacts/lfm-fsdp-validation
```

GPU 编号需选择空闲设备，输出使用新的目录。该入口只做三次小 batch 更新和恢复比对，不提交 ACP 任务。
