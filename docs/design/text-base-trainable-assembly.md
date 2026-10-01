# 备选组装方案：0.6B 文本 embedding + 可训练随机 projector

## 状态与目标

2026-10-01 讨论确定的备选方案，**本次仅记录设计和后续操作方法**：没有修改组装或训练代码，没有生成这套新模型，没有为它启动训练。当前 all16 BF16 作业继续使用原来的 frozen-conditioning 模型，不受本文影响。

目标是从 `Qwen3-0.6B-Base` 同时加载文本 embedding 与 Talker 主干，补齐 TTS token，新增随机初始化的文本 projector。**文本 embedding 和 projector 均参与未来训练**。音频预测模块仍从新初始化开始学习，speaker encoder 和音频 codec 保持固定。

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

按当前结构推算，主模型总参数应为 **754,865,216**，其中冻结 ECAPA **8,854,336**，可训练 **746,010,880**，不含独立音频 codec。文本 embedding 为 155,582,464 参数，projector 为 2,099,200 参数；实际产物必须再核对 `assembly_report.json`，这些推算不是已完成组装的测量结果。

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

未来训练仍应联合优化 Talker / MTP，使用同一目标 `first CE + 0.3 × residual CE`，保持 token 归一化及因果边界。Text embedding 和 projector 可通过音频预测 loss 学习，不需要为了让它们更新而额外加入文本 LM loss。Speaker embedding 和 codec IDs 继续作为固定输入，不反传到离线数据流水线。

## 设计收益与需要验证的风险

文本 embedding 与 Talker 主干来自同一 Base，来源明确；文本前端可更新，让表示能适应 TTS。这样还可以避免把官方 TTS 文本前端的已训练权重引入对照实验。

但随机 MLP 会改变 Base embedding 的输出分布：**来源相同，不等于经过随机 projector 后仍保持原主干熟悉的输入**。允许两者更新提供了适配能力，但不能提前保证收敛更快或语音更好。较低初始 loss、训练吞吐或工程检查通过，也不能代替生成音频评估。

本方案同时改变文本权重来源、维度、projector 初始化和冻结范围。它是完整备选方案，不是只检验“是否冻结 projector”的单变量消融。如果之后需要判断具体因素，应另外设计小范围对照，不修改本方案的定义。

同样的 seed 也不保证两套结构的随机音频权重逐值一致：文本矩阵形状变化可能改变随机数消耗顺序。若要求仅文本端不同、其余模块初始权重完全相同，需要显式对齐共同模块、记录来源和逐张量哈希；不能仅写“seed 相同”就宣称严格控制了初始化变量。

## 后续组装方法：现有入口已支持，以下命令本次未执行

底层 [assemble_qwen3_tts.py](../../scripts/assemble_qwen3_tts.py) 已有 `text-base`、`random` 和 `train-text-frontend` 开关，初始化逻辑位于 [models/assembly.py](../../qwen3_train/models/assembly.py)。后续首先复用该入口，不必复制一整套模型实现。

建议独立产物名：`assets/assembled/qwen3-tts-text-base-trainable-random-projector`。本次没有创建该目录。正式组装前先只生成审计报告：

```bash
cd /workspace/workspace/yanglin/LM-TTS-Training
PY=/workspace/workspace/yanglin/envs/lm-tts/bin/python

# Audit only: no assembled weights and no training job.
"$PY" -m scripts.assemble_qwen3_tts \
  --backbone assets/base/Qwen3-0.6B-Base \
  --tts-template assets/base/Qwen3-TTS-12Hz-0.6B-Base \
  --codec assets/base/Qwen3-TTS-Tokenizer-12Hz \
  --text-initialization text-base \
  --text-projection-init random \
  --train-text-frontend \
  --input-protocol qwen3_non_streaming \
  --seed 42 --dtype float32 \
  --audit-only \
  --output artifacts/assembly-text-base-audit.json
```

核对审计结果后，再执行下面的组装命令；它本身不会启动训练：

```bash
"$PY" -m scripts.assemble_qwen3_tts \
  --backbone assets/base/Qwen3-0.6B-Base \
  --tts-template assets/base/Qwen3-TTS-12Hz-0.6B-Base \
  --codec assets/base/Qwen3-TTS-Tokenizer-12Hz \
  --text-initialization text-base \
  --text-projection-init random \
  --train-text-frontend \
  --input-protocol qwen3_non_streaming \
  --seed 42 --dtype float32 \
  --output assets/assembled/qwen3-tts-text-base-trainable-random-projector
```

使用本地基础模型，不需要下载。不要加 `--train-speaker-encoder`；省略它才能保持 ECAPA 冻结。不要省略 `--text-projection-init random`，因为现有 CLI 在 `text-base` 模式下的默认 projector 是 Identity。不要省略 `--train-text-frontend`，默认仍冻结文本前端。

输出目录已存在时脚本会拒绝覆盖。应核对并选择新目录，不删除当前正式训练正在使用的组装模型。成功发布时包含模型权重、config、tokenizer、speech_tokenizer、assembly_report 和 `ASSEMBLY_COMPLETE`；中途失败的 `.incomplete-*` 目录不是可训练产物。

当前 [prepare_models.py](../../scripts/prepare_models.py) 只调用默认组装参数。**仅更换它的输出目录，不会自动选择本方案**。未来若要提供专用 YAML / 一键入口，应将上述参数显式接入并验证复用已存在产物时的方案身份，保留旧默认行为。本次没有新增这种入口或配置文件。

## 组装后的验收

组装自带 public wrapper 重载检查，但 wrapper 可加载不代表项目的冻结策略正确。完成后还应做下列针对性验证，并把实测结果写在新产物的旁路审计报告中；不要事后改写已有 `assembly_report.json` 以免改变签名。

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
| 工程前向 / 反向 | 后续获准做验证时，以微型样本检查文本 embedding / projector 均有有限且非零梯度，speaker 无梯度；不存在训练任务提交 |

本次仅文档，不执行表中的组装、梯度验证或模型生成。后续纯 CPU 检查 / 小样本反向应与正式训练隔离，不能直接使用正式 run 路径。

## 未来训练前的必要接入检查

### 学习率分组不能只看初始化名称

当前 [training/engine.py](../../qwen3_train/training/engine.py) 只把 `talker.model.layers.*` 和 `talker.model.norm.*` 归入 `backbone_lr`，其他可训练参数归入 `lr`。因此直接使用这份新模型时，**预训练 text embedding 也会进入较高的“新模块”学习率组**。现有 base YAML 注释所说的新模块，不会自动按权重来源判断。

如果未来希望预训练 text embedding 与 Talker 使用较低 LR，需要显式调整和验证 optimizer 分组；随机 projector / 音频模块可放在新参数组。预训练 embedding 内的新 token 行与原行属于同一个 Parameter，普通 optimizer 分组不能简单按行给予不同 LR，不要只在文档里声称做了分行设置。具体 LR、warmup 和分组变更属于后续实验决定，本次不创建训练配置。

### 数据 build 是否可复用

新的 embedding 宽度与数值不改变 token IDs，因此无需因此重新生成 codec / speaker features。现有 build 可否直接复用取决于 tokenizer 产物哈希及数据 / speaker 来源是否一致。

当前训练器会将新 assembled 模型的 tokenizer 哈希与 build 中保存的 tokenizer 哈希比较，也会核对 build 原 tokenizer 路径下的文件。只有这些检查通过才复用现有 build；不要编辑 manifest 的哈希绕过校验。即使词表逻辑一致，tokenizer 文件序列化不同也可能被拒绝，需要核实并按正常构建流程处理。

### 独立实验与恢复边界

未来创建独立 run / YAML，以 `textbase-trainable-randproj` 等明确标识区分当前 `frozen-conditioning` 模型，再加数据、精度、LR、epoch 等实验字段。仅在新实验中覆盖 assembled 路径，不改共享 base 默认路径来切换正在运行的实验。

不能从当前 frozen-conditioning 的 checkpoint 直接 resume 到本方案：text embedding / projector 形状、冻结范围、optimizer 参数集合和 assembled 签名均改变。新方案第一次训练从自己的组装模型启动；之后只能按其自己的 checkpoint 和签名严格恢复。

### 效果与资源验证

文本 embedding 行宽减半，但从冻结变为可训练，需要梯度和 optimizer 状态，因此不能只根据总参数减少就断言显存更省。当前 32,000 frames / 50,000 tokens 的预算不可未经测量直接照搬，应另做隔离的前向 / 反向 / 保存 / 恢复验证。

效果比较使用相同数据划分、输入协议、训练数据消费量及固定生成样本，报告 token loss 与 WER/CER、音质和说话人相似度。Dynamic batching 下相同步数不一定意味着相同样本量或计算预算，需要同时记录已消费样本 / 音频量。当前训练不会自动产生生成音频评分，未来对照需显式安排该流程。

## 交接清单

1. 阅读本方案，确认仍选择“Base embedding + 随机 MLP + 文本端可训练”。
2. 运行 audit-only，核对本地来源、token IDs、尺寸和冻结标记。
3. 输出到新 assembled 目录，完成逐模块来源和项目加载验收。
4. 若要一键化，给现有准备入口增加独立配置接入，避免复制模型实现；这是待办，不是已交付代码。
5. 另行确定 optimizer 的 text embedding 分组、数据 build 兼容性、预算和新 run。
6. 获得启动该实验的明确安排后，再执行训练预检和提交。当前授权范围停留在本文档。
