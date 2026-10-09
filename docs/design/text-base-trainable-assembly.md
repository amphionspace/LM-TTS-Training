# 备选组装方案：0.6B 文本 embedding + 可训练随机 projector

## 状态与目标

2026-10-01 记录设计，2026-10-03 接入独立实验配置并生成本地模型。准备、训练和提交共用新实验 YAML；原 frozen-conditioning 配置及其恢复语义不变。实际验证与任务状态见 [本轮记录](../training/all16-textbase-20261003.md)。

目标是从 `Qwen3-0.6B-Base` 同时加载文本 embedding 与 Talker 主干，补齐 TTS token，新增随机初始化的文本 projector。**文本 embedding 和 projector 均参与训练**。音频预测模块仍从新初始化开始学习，speaker encoder 和音频 codec 保持固定。

这属于“基于预训练文本 LM 重新学习 TTS”，不是所有参数随机初始化。它保留预训练 speaker encoder / codec，也保留官方 TTS 的模型结构和 tokenizer 扩展，但不加载官方 TTS 的文本 embedding、text projector 或音频预测权重。

当前基线及输入协议见 [all16 训练说明](../training/all16-20261001.md)，历史组装说明见 [模型组装](model-assembly.md)。本文不替换当前默认方案。

## 模块约定

| 模块 | 结构 | 备选方案的初始化来源 | 未来是否更新 |
| --- | --- | --- | --- |
| Text tokenizer | 原文本词表 + 官方 TTS 特殊 token | 采用官方 TTS tokenizer，严格核对共享 token ID | 固定规则，无梯度参数 |
| Text embedding | `151936 × 1024` | 文本 Base 的 `embed_tokens.weight`；新增 token 行重新初始化 | **更新** |
| Text projector | `1024 → 1024 → 1024`，SiLU，两层带 bias | **随机初始化**，不加载 TTS projector | **更新** |
| Talker layers / final norm | 28 层，hidden 1024，FFN 3072 | 文本 Base 的 `layers` / `norm` | 更新 |
| 首码本 embedding | `3072 × 1024` | 新初始化 | 更新 |
| 首码本输出 head | `1024 → 3072`，无 bias | 新初始化，与 embedding 不共享参数 | 更新 |
| 15 张 residual embedding | 各 `2048 × 1024` | 新初始化 | 更新 |
| Code Predictor layers / norm | 5 层，hidden 1024 | 新初始化 | 更新 |
| 15 个 residual 输出 head | 各 `1024 → 2048`，无 bias | 新初始化，与 embedding 不共享参数 | 更新 |
| Talker → Code Predictor projector | `Identity` | 两侧同宽，无参数 | 不适用 |
| Speaker encoder | ECAPA-TDNN，输出 1024 维 | 官方 Qwen3-TTS Base | 冻结；训练读取缓存 embedding |
| Speech tokenizer encoder / decoder | 16 码本，12.5 Hz，24 kHz 波形 | Qwen3-TTS-Tokenizer-12Hz | 独立冻结 |

**Text projector 是新 MLP；Talker → Code Predictor projector 仍是 Identity。** 两者不要混淆。本方案不使用 text projector 的 `identity` 或 `near-identity` 初始化，也不将官方 TTS 的 2048 维 embedding 截断到 1024 维。

实际组装主模型总参数为 **754,865,216**，其中冻结 ECAPA **8,854,336**，可训练 **746,010,880**，不含独立音频 codec。文本 embedding 为 155,582,464 参数，projector 为 2,099,200 参数；参数量已与 `assembly_report.json` 和项目加载器核对。

## 扩词表与初始化细节

当前本地来源的文本 tokenizer 有 151,669 个 token，TTS tokenizer 有 151,676 个。新增的是以下 7 个逻辑 token：

| Token | ID |
| --- | ---: |
| `<\|audio_start\|>` | 151669 |
| `<\|audio_end\|>` | 151670 |
| `<tts_pad>` | 151671 |
| `<tts_text_bos>` | 151672 |
| `<tts_text_eod>` | 151673 |
| `<tts_text_bos_single>` | 151674 |
| `<\|audio_pad\|>` | 151675 |

Base 的 embedding 已有 151,936 行，因此这些 ID 可以使用原有预留行，**词表扩展 7 项，但矩阵不必新增物理行**。本方案保持所有共享 token 的 ID 和初始 embedding 不变；新增 7 项对应的行即使在原矩阵容量内，也必须重新初始化，不能把预留行当成已经学好的 TTS 表示。

现有 `vocabulary_plan()` 会检查全部共享 token 的 ID；若未来 token ID 超过预留容量，则将矩阵扩到足够行数。现有 `initialize_model()` 先复制 Base embedding，再将新增 token 行以模型 initializer range 重新初始化。当前范围为 `std=0.02` 的零均值正态分布；组装使用 seed 42 和 FP32，以便保留来源权重和新增权重的初始化精度。

Tokenizer 沿用 TTS 版本是为了固定 token ID 与协议一致性，不表示加载它的神经网络文本 embedding。也不能只扩权重而忘记保存 tokenizer、special tokens、模型中的 BOS/EOS/PAD ID。

Projector 保留官方 ResizeMLP 的两层形式，输入 / 中间 / 输出宽度均为 1024。现有构造器初始化 Linear 权重与 bias，随后不覆盖成预训练权重或单位矩阵。模型输出目录中必须记录 `text_initialization=text-base`、`text_projection_init=random` 和 `freeze_text_frontend=false`。

## 输入与梯度路径

沿用当前非流式协议，文本 token 经新文本前端进入 Talker：

```text
文本 → 固定 tokenizer → IDs → Base text embedding〔可训练〕
                             → 随机 text MLP〔可训练〕 → 1024 维

历史 codec IDs → 16 张 embedding〔可训练〕 → 同帧逐元素求和 → 1024 维

角色 / 控制前缀 → speaker → 完整文本 → codec BOS → 历史音频帧
                                    ↓
                             可训练 Talker
                                    ↓
                      首码本 head + Code Predictor
```

文本位置仍加 codec PAD 向量；音频位置仍加投影后的 text PAD 向量。同一帧的 16 个 codec embedding 相加；文本和音频位置沿时间轴拼接，并不将文本 token 数与帧数强制对齐。Talker 与 Code Predictor 共用对应的音频输入 embedding，但 embedding 和输出 head 不 tie。

与当前基线不同，text PAD、角色前缀、普通文本和 TTS 特殊 token 对应的文本前端都不再被冻结。Text PAD 在多个位置复用，梯度会累积到对应 embedding 行和 projector；这属于预期行为，验证时不能只检查普通文本 token 的梯度。

训练联合优化 Talker / MTP，使用同一目标 `first CE + 0.3 × residual CE`，保持 token 归一化及因果边界。Text embedding 和 projector 可通过音频预测 loss 学习，不需要为了让它们更新而额外加入文本 LM loss。Speaker embedding 和 codec IDs 继续作为固定输入，不反传到离线数据流水线。

## 设计收益与需要验证的风险

文本 embedding 与 Talker 主干来自同一 Base，来源明确；文本前端可更新，让表示能适应 TTS。这样还可以避免把官方 TTS 文本前端的已训练权重引入对照实验。

但随机 MLP 会改变 Base embedding 的输出分布：**来源相同，不等于经过随机 projector 后仍保持原主干熟悉的输入**。允许两者更新提供了适配能力，但不能提前保证收敛更快或语音更好。较低初始 loss、训练吞吐或工程检查通过，也不能代替生成音频评估。

本方案同时改变文本权重来源、维度、projector 初始化和冻结范围。它是完整备选方案，不是只检验“是否冻结 projector”的单变量消融。如果之后需要判断具体因素，应另外设计小范围对照，不修改本方案的定义。

同样的 seed 也不保证两套结构的随机音频权重逐值一致：文本矩阵形状变化可能改变随机数消耗顺序。若要求仅文本端不同、其余模块初始权重完全相同，需要显式对齐共同模块、记录来源和逐张量哈希；不能仅写“seed 相同”就宣称严格控制了初始化变量。

## 配置与组装

实验配置为 `configs/supervised-tts-20260929-all16-textbase-trainable-randproj-bf16-32gpu-lr3e-4-bblr1e-4-ep3-wsd.yaml`，直接继承 `base.yaml`。其中 `assembly` 区块仅由准备脚本读取，训练器不会在启动或恢复时组装权重：

```yaml
assembly:
  text_initialization: text-base
  text_projection_init: random
  train_text_frontend: true
  train_speaker_encoder: false
  input_protocol: qwen3_non_streaming
  seed: 42
  dtype: float32
```

从仓库根目录运行：

```bash
PY=/workspace/workspace/yanglin/envs/lm-tts/bin/python
CONFIG=configs/supervised-tts-20260929-all16-textbase-trainable-randproj-bf16-32gpu-lr3e-4-bblr1e-4-ep3-wsd.yaml
"$PY" -m scripts.prepare_models --config "$CONFIG"
```

准备脚本校验已有本地基础模型，调用现有组装实现，输出到 `assets/assembled/qwen3-tts-text-base-trainable-random-projector`。已有完整产物必须匹配初始化、冻结范围、seed、保存精度、输入协议、来源和文件哈希，才允许复用。目录冲突或中断产物不会被静默覆盖。旧配置未指定 `assembly` 时继续使用原冻结方案。

底层入口仍是 [assemble_qwen3_tts.py](../../scripts/assemble_qwen3_tts.py)，支持 `--audit-only`。若手动调用，必须同时传 `--text-initialization text-base --text-projection-init random --train-text-frontend`；只切换 embedding 来源不会自动选择随机 MLP 或解冻。

产物以 FP32 保存，BF16 是训练计算精度。成功发布包含模型权重、config、tokenizer、speech_tokenizer、assembly_report 和 `ASSEMBLY_COMPLETE`。正式训练只加载这个独立目录。

## 组装后的验收

组装自带 public wrapper 重载检查，但 wrapper 可加载不代表项目的冻结策略正确。执行下列针对性验证，实测结果写在新产物之外的验证报告中；不要事后改写已有 `assembly_report.json` 以免改变签名。

| 检查 | 通过条件 |
| --- | --- |
| 文本 tokenizer | 原 151,669 项 token ID 全部一致；新增 7 项及协议 ID 正确；保存 / 重载结果一致 |
| Text embedding 来源 | 全部原共享 token 行与 Base 逐值相等；新增 7 行有限、非零且重新初始化；形状 `[151936,1024]` |
| Projector | 两个带 bias 的 `1024→1024` Linear，中间 SiLU；为随机初始化，不是 Identity / near-identity / TTS 权重 |
| Talker 来源 | 每层及 final norm 与 Base 对应张量严格相等 |
| 音频模块 | 形状正确、未加载 TTS 预训练音频权重；输入表与输出 head 不 tie |
| Speaker / codec | 与指定本地预训练来源一致；训练模型保持 speaker 冻结、eval 模式 |
| 项目重载 | `TTSModel.from_assembled()` 成功；text embedding、projector、Talker、音频模块 `requires_grad=True`，speaker 为 False |
| 参数量 | 与本方案计算值相符；报告中的 trainable 与实际项目加载结果一致 |
| 工程前向 / 反向 | 以微型样本检查文本 embedding / projector 均有有限且非零梯度，speaker 无梯度；训练提交另行记录 |

验证使用独立目录和本地空闲 GPU，不在原正式 run 下运行测试。

## 训练接入与恢复边界

### 学习率分组

新配置显式指定 `train.text_embedding_lr_group: backbone`：预训练 text embedding、Talker layers/norm 使用 `backbone_lr=1e-4`，随机 projector 与音频模块使用 `lr=3e-4`。新增 token 行和原行属于同一个 Parameter，因此整张 embedding 使用同一个 LR；没有按行设置学习率。

[optimizer.py](../../lm_tts/training/optimizer.py) 保留两组及遍历顺序。未指定这一选项的旧实验继续原分组行为；配置加载器不会向旧配置补入该字段，不改变旧 checkpoint 的签名。显式设置属于训练语义，进入新实验的恢复签名。

### 数据 build 是否可复用

新的 embedding 宽度与数值不改变 token IDs，因此无需因此重新生成 codec / speaker features。现有 build 可否直接复用取决于 tokenizer 产物哈希及数据 / speaker 来源是否一致。

旧 build 的 tokenizer 指纹包含整个模型 `config.json`，因此文本宽度或冻结范围变化也会使总哈希不同。训练器仍先验证原 build 的 tokenizer 来源文件未变；总哈希不同时，仅在 tokenizer 文件逐字节相同、特殊 token IDs、角色前缀、输入协议和词表容量一致时允许复用。模型参数的变化仍由独立 assembly 哈希纳入新实验签名。不会改写旧 manifest、sampling index 或旧指纹；真正改变 tokenizer 时必须重新构建数据。

### 独立实验与恢复边界

新 run / YAML 用 `textbase-trainable-randproj` 区分原模型，名字同时包括数据、精度、GPU 数、LR 与 epoch。只在新实验中覆盖 assembled 路径，不改共享 base 默认路径。新实验设 `keep_checkpoints: null`，所有完整 checkpoint 都保留，不参与两份轮转；默认旧实验仍为两份。

不能从当前 frozen-conditioning 的 checkpoint 直接 resume 到本方案：text embedding / projector 形状、冻结范围、optimizer 参数集合和 assembled 签名均改变。新方案第一次训练从自己的组装模型启动；之后只能按其自己的 checkpoint 和签名严格恢复。

### 效果与资源验证

文本 embedding 行宽减半，但从冻结变为可训练，需要梯度和 optimizer 状态，因此不能只根据总参数减少就断言显存更省。当前 32,000 frames / 50,000 tokens 的预算不可未经测量直接照搬，应另做隔离的前向 / 反向 / 保存 / 恢复验证。

效果比较使用相同数据划分、输入协议、训练数据消费量及固定生成样本，报告 token loss 与 WER/CER、音质和说话人相似度。Dynamic batching 下相同步数不一定意味着相同样本量或计算预算，需要同时记录已消费样本 / 音频量。当前训练不会自动产生生成音频评分，未来对照需显式安排该流程。

## 提交与恢复

完成验证后，从同一配置渲染 / 提交：

```bash
"$PY" -m scripts.acp.submit --config "$CONFIG"           # Render only.
"$PY" -m scripts.acp.submit --config "$CONFIG" --submit  # Create one job.
```

新实验首次提交不传 `--resume`。之后恢复同一实验时，使用原配置和 `--resume latest`，先确认旧 job 已终态，避免重复训练。不要从 frozen-conditioning 的 checkpoint 切换到本方案，也不要修改已有 checkpoint 签名绕过检查。源码在每次提交时保存快照；清理临时验证目录不会改变已提交任务。
