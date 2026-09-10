# Emilia2 中英预训练：从数据到四卡 loss

本文对应当前实现和 [`configs/emilia-10kh-pretrain.yaml`](../../configs/emilia-10kh-pretrain.yaml)。当前正式实验使用完整中英10kh数据、padding-free布局和动态组批，从Qwen3-0.6B-Base初始化Talker主干，训练新音频模块。此前固定batch的1kh实验已完成，见[1kh报告](emilia-1kh-report.md)。

训练输入包含目标文本、完整目标录音的speaker向量和teacher forcing历史codec；生成使用另一条训练录音作为参考，分别运行speaker-only和ICL。优化目标只有首码本（含EOS）与15个残余码本的交叉熵，没有文本、ASR或speaker loss。

## 1. 当前入口与数据位置

数据准备流程见[10kh数据准备](emilia-10kh.md)，正式启动和恢复使用当前run的 `launch.py`，具体进程核查与运行约定见[训练记录](emilia-10kh-supervision.md)和[故障处置](training-incident-playbook.md)。已有训练进程时不要重复启动。

| 位置 | 内容 |
|---|---|
| `/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_m4a/` | 原始tar与索引 |
| `/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_prepared/LM-TTS-Training/emilia-short-en-zh-10000h/` | 原始/编码分片、最终train/val清单与准备报告 |
| `runs/emilia-en-zh-dynamic-10000h/` | 当前正式实验、预检、checkpoint、评估和巡检 |
| 当前run的 `provenance/data-preparation/` | 已完成的数据准备日志与核查记录 |

## 2. 数据选择和划分

接收独立short与long载体中的标注短句，不提取dialogue。保留原始短句ID、文本、语言、speaker、tar成员offset/size和long片段的原采样点边界；按ID去重，不把完整long录音当作一个训练样本。

完整训练集6,869,742条、约9,999.19小时；验证集512条、中英各256。划分检查ID及规范化文本的train/val隔离。511条验证目标有同speaker、同语言、不同ID和文本的训练参考；剩余1条仍参与验证loss，排除于ICL配对。在线固定选择8条生成目标，不等于未见说话人的zero-shot测试。

## 3. 文本、codec与在线波形

文本使用组装模型的固定tokenizer规则，prepared清单保存原始 `text_ids`，加载时加入文本BOS/EOS。离线codec输出 `[T,16]` 的整数NPZ，清单记录 `codes`、`codes_sha256`、`num_frames` 和 `audio_source`。复用缓存可能直接引用其他prepared目录，需一起保留。

离线准备按long载体分组，一次解码后切出所选短句。在线训练为完整目标短句计算speaker mel：通过tar成员边界和时间seek，只解码目标前约1秒及目标片段，再按原采样率精确裁剪、重采样到24kHz。避免每条短句重复解码几十分钟的载体。AAC随机seek与从头解码有很小数值差异；相应验证及 `audio_decoder=emilia_native_seek_v1` 恢复签名见[训练记录](emilia-10kh-supervision.md)。

## 4. 一个 batch 如何进入模型

实现：[`CodeDataset / collate / DistributedTokenBatchSampler`](../../qwen3_train/data.py)、[`audio_mel`](../../qwen3_train/speaker.py)、[`train.main`](../../qwen3_train/train.py)。

### 4.1 清单与样本读取

每个训练 rank 各自将完整 train/val JSONL 读入内存，记录 manifest SHA256。加载组装模型时，为每个样本的 `text_ids` 加上：

```text
[tts_text_bos=151672, 原始文本 token..., tts_text_eod=151673]
```

每次取样本时读取 NPZ，核对 SHA256，检查 codec 形状和范围。设置 `target_speaker=True` 后，还会解码该样本的**完整目标录音**，转换为 speaker mel。

训练使用 PyTorch `DataLoader`，当前正式配置每个 rank 使用16个worker进程（代码默认4个），每个 worker 提前准备 2 个 microbatch，由 `train.num_workers` 和 `train.prefetch_factor` 调整。worker 使用 `spawn` 启动并跨 epoch 复用，在 CPU 上完成 NPZ 读取、音频解码、mel 和 `collate()`；主进程通过 pinned memory 将 batch 传到 GPU。worker 根据字节索引读取 manifest，不复制主进程的完整样本字典列表。

没有提前缓存 speaker embedding，因此 codec 已缓存仍会发生原始音频读取、解码、mel 和 ECAPA 计算。预取使 CPU 准备数据与 GPU 训练重叠，`train/data_wait_seconds` 记录每步获取 microbatch 的等待时间。验证 loss 目前仍按样本顺序读取。

### 4.2 speaker mel 和向量

完整目标波形在 24 kHz 下按官方 mel 函数处理：`n_fft=1024`、`win_size=1024`、`hop_size=256`、128 mel、`fmin=0`、`fmax=12000`。输出转为 `[M_i, 128]`。

当前不把目标音频裁成固定 3 秒，也不为短录音重复波形。`collate()` 将 mel 连续存储，同时保存 `speaker_lengths`。模型先按长度拆出每条完整录音，再按相同长度分组交给 ECAPA。ECAPA 的全局统计池化和 squeeze-excitation 不会看到补齐帧或其他录音。

ECAPA 输出 `[B, 1024]`，每条录音一个音色向量。它保持 `eval()` 且所有参数 `requires_grad=False`。冻结只表示不更新权重，当前仍在线执行前向；不存在单独 speaker loss。

### 4.3 分布式采样和四卡分配

训练通过 PyTorch `DataLoader(batch_sampler=DistributedTokenBatchSampler(...))` 动态组批。每卡每个 microbatch 同时受 `train.max_batch_frames`（目标 codec 帧总数）和 `train.max_batch_tokens`（完整 Talker 输入 token 总数，包含文本、控制符、speaker 和 audio BOS）限制。超预算的单条样本会报错，不静默跳过。每个 epoch 使用 `seed + epoch` 打乱样本，按当前预算占用率给各 rank 分配可容纳的下一条样本；当所有 rank 都容纳不下时结束这组 batch。因此各 rank 的 batch 数一致、无重复填充，样本数可不同；仅在 epoch 尾部不足以给所有 rank 各分一条时丢弃至多 `world_size - 1` 条随机尾部样本。

checkpoint 保存已消费的 `epoch / next_batch`，不计入 worker 提前读取的 batch。恢复时由 Accelerate 的 `skip_first_batches()` 跳过已消费索引，避免重新解码这些样本。动态 batch 边界由相同数据、seed、epoch 和预算重建。每步样本数可变，loss 分母仍是所有 rank、全部累积 microbatch 的实际目标数。TensorBoard 记录 `global_samples`、`global_audio_frames`、`global_talker_tokens` 和两种 `*_budget_fill`；验证组批大小单独由 `eval.batch_size` 控制（默认 8）。

中英数据按合并清单自然参与采样。中英各约5000小时不是每个batch强制1:1的样本数；两种语言平均录音长度不同时，样本数也可能不同。

### 4.4 batch 张量

设本卡样本数为 `B`，各样本含 BOS/EOD 的文本长度为 `N_i`、codec 帧数为 `T_i`、mel 长度为 `M_i`：

| 张量 | shape | 说明 |
|---|---|---|
| `text_ids` | `[Σ N_i]` | 连续存储有效文本 token |
| `text_lengths` | `[B]` | 每条文本的长度 |
| `codes` | `[Σ T_i, 16]` | 连续存储真实 codec 帧 |
| `frame_lengths` | `[B]` | 每条音频的 codec 帧数 |
| `speaker_mels` | `[Σ M_i, 128]` | 连续存储有效 mel 帧 |
| `speaker_lengths` | `[B]` | 每条 mel 的长度 |

没有文本、音频或 mel 补齐槽位。0 仍是合法 codec ID；样本边界由长度记录，而不是 token 值决定。张量在模型调用前迁移到本 rank 的 GPU。

## 5. 模型从哪里初始化，哪些参数参与训练

实现：[`TTSModel.from_assembled`](../../qwen3_train/model.py)、[`assembly.py`](../../qwen3_train/assembly.py)。组装产物的 `config.json` 和 `assembly_report.json` 是具体权重与维度的依据。

| 模块 | 结构/维度 | 来源 | 是否更新 |
|---|---|---|---|
| 文本 embedding | `[151936, 2048]` | 公开 Qwen3-TTS-0.6B，完整表 | 冻结 |
| 文本 projector | `2048 → 2048 → 1024`，SiLU | 同一 TTS checkpoint | 冻结 |
| Talker layers/norm | 28 层，hidden 1024，FFN 3072，16 Q heads / 8 KV heads，head_dim 128 | Qwen3-0.6B-Base | 更新 |
| 首码本 embedding / head | `[3072,1024]` / `1024→3072` | 新初始化 | 更新 |
| 残余 embedding | 15 张 `[2048,1024]` | 新初始化 | 更新 |
| Code Predictor | 5 层，hidden 1024，15 个 2048 类 head | 新初始化 | 更新 |
| Talker 到 Predictor 的投影 | 同宽 Identity | 无参数 | 不适用 |
| ECAPA | 128 mel → 1024 维 | 公开 Qwen TTS | 冻结 |
| codec encoder/decoder | 24 kHz，16 码本 | 公开 Qwen codec | 独立冻结 |

训练模型共 **914,643,008** 参数，冻结 **326,313,792**，可训练 **588,329,216**，不含独立 codec。这里没有加载公开 TTS 的已训练 Talker 主干和音频预测模块，也没有使用文本 LM 的输出 head。

加载器要求 `ASSEMBLY_COMPLETE`，校验组装配置和权重文件哈希，再严格加载权重。冻结策略由本项目 `TTSModel` 执行；公开 wrapper 是否能加载相同形状，并不代表它会自动执行这些冻结设置。

## 6. 模型真正看到的输入序列

实现：[`TTSModel.input_embeddings / hidden`](../../qwen3_train/model.py)。以下描述的是本轮 `qwen3_non_streaming`、Auto language 路径。

定义：

- `E_t(x)`：文本 embedding 后再经过 projector，最终 1024 维。
- `E_0(c)`：首码本 embedding；`E_g(c)`：第 `g` 个残余码本 embedding。
- `p = E_t(tts_pad)`：投影后的文本 pad 向量。
- `s`：该录音的 1024 维 speaker 向量。
- `c[t,g]`：第 `t` 帧、第 `g` 组 codec 标签，`g=0…15`。

一帧历史音频不是 16 个时间位置，而是 **16 个 embedding 相加成一个时间位置**：

```text
audio_frame(t) = E_0(c[t,0]) + E_1(c[t,1]) + ... + E_15(c[t,15])
```

输入沿时间轴依次拼接：

| 段 | 位置数 | 每个位置的 1024 维输入 |
|---|---:|---|
| assistant 角色 | 3 | `E_t([151644, 77091, 198])` |
| 非流式控制 | 3 | `p + E_0(nothink / think_bos / think_eos)` |
| speaker | 1 | `p + s` |
| 文本 BOS、完整文本、文本 EOD | `L + 2` | `E_t(text_id) + E_0(codec_pad)` |
| 音频 BOS | 1 | `p + E_0(codec_bos)` |
| 历史音频帧 | `T` | `p + audio_frame(t)` |

`L` 是原始文本 token 数，所以单条无 padding 的训练有效长度为 `L + T + 10`。加号表示同位置向量相加；不同表格行表示时间轴拼接。

关键 codec token：`codec_pad=2148`、`codec_bos=2149`、`codec_eos=2150`、`nothink=2155`、`think_bos=2156`、`think_eos=2157`。中文和英文都走 Auto 路径，当前没有额外插入 `codec_language_id[chinese/english]`；清单的 language 字段用于筛选、分组评估与 ASR 语言选择。

**完整文本在音频前面**，因此每个音频位置都能看见本样本的全部文本。各样本的有效 embedding 拼成 `[1, Σ sequence_length_i, 1024]`，每条样本的 position ID 从 0 重新开始。FA2 通过显式 int32 累积长度和最大样本长度计算独立的因果 attention 段，既不能看到未来音频，也不能看到其他样本。CPU／SDPA 检查使用相同 packed 输入和分段因果 mask。

## 7. 首码本和 EOS 的 loss

实现：[`TTSModel.forward`](../../qwen3_train/model.py)。

### 7.1 时间对齐

Talker 输出按索引去掉文本/角色/speaker 前缀，只保留各样本从音频 BOS 开始的 hidden，shape 为 `[Σ (T_i + 1), 1024]`。设一条样本有 `T` 帧：

| hidden 所在输入位置 | 可见的音频历史 | 首 head 的目标 |
|---|---|---|
| 音频 BOS，记作 `h_0` | 无音频帧 | `c[0,0]` |
| 帧 0 之后，`h_1` | 帧 0 | `c[1,0]` |
| 帧 `t-1` 之后，`h_t` | 帧 `0…t-1` | `c[t,0]` |
| 最后一帧之后，`h_T` | 全部 `T` 帧 | EOS |

例如只有两帧时，输入的音频段为 `[BOS, frame_0, frame_1]`，首 head 标签为 `[c[0,0], c[1,0], EOS]`。`h_t` 预测帧 `t` 时尚未看到帧 `t` 的 codec 输入。speaker 向量本来就是目标录音的全局条件；这里的因果约束指 codec 时间序列不会提前提供目标帧。

### 7.2 标签与计算

首 head 输出 `[Σ (T_i + 1), 3072]` logits，转 float32 后调用交叉熵。每条样本的标签依次为全部真实首码本 ID 和一个 EOS；每段末尾位置由 `(frame_lengths + 1).cumsum(0) - 1` 确定。

```text
S_first = Σ_i [Σ_(t=0…T_i-1) CE(head(h_i,t), c_i[t,0])
               + CE(head(h_i,T_i), EOS)]
N_first = Σ_i (T_i + 1)
```

文本位置不进入该 head，每条录音恰好贡献一个 EOS。训练 softmax 覆盖 head 的全部 3072 类，普通音频标签为 0…2047，终止标签为 2150。

## 8. 同一帧内 15 个残余码本的 loss

Talker 用时间自回归预测首码本，Code Predictor 在同一帧内用码本顺序自回归补全其余 15 组。

先从音频 hidden 中排除每条样本预测 EOS 的末尾位置，得到真实帧对应的 `h_t`，与连续存储的 `codes` 一一对应。所有样本的真实帧组成 `F = Σ_i T_i` 条帧内训练序列，不包含 EOS 位置。

每条帧内输入长度为 16：

```text
[h_t, E_0(c[t,0]), E_1(c[t,1]), ..., E_14(c[t,14])]
```

这形成 `[F, 16, 1024]` 输入，经过 5 层因果 Code Predictor。第 `g` 个残余 head 使用帧内位置 `g` 的输出，预测 `c[t,g]`，其中 `g=1…15`：

| 要预测的标签 | 可见帧内输入 |
|---|---|
| `c[t,1]` | `h_t` 和真实 `c[t,0]` |
| `c[t,2]` | `h_t`、真实 `c[t,0]`、真实 `c[t,1]` |
| … | … |
| `c[t,15]` | `h_t` 和真实 `c[t,0…14]` |

这是 teacher forcing：训练中已知前面的真实码本，一次因果前向即可并行计算所有残余位置。目标 `c[t,g]` 不进入预测它的当前位置；`c[t,15]` 不需要作为帧内输入，但它会参与下一时间帧的 Talker 历史 embedding。

每组 head 输出 2048 类 logits，各自转 float32 后计算 CE：

```text
S_residual = Σ_i Σ_(t=0…T_i-1) Σ_(g=1…15) CE(head_g(depth_i,t,g), c_i[t,g])
N_residual = 15 × Σ_i T_i
```

没有“残余码本 EOS”标签。模型还返回 15 个独立的 `group_sums`，用于报告每个残余码本的验证 CE。

## 9. 最终 loss、四卡归一化和参数更新

### 9.1 全局目标

本轮 `residual_weight=0.3`：

```text
L = S_first_global / N_first_global
    + 0.3 × S_residual_global / N_residual_global
```

两项分别按各自有效 token 数平均。残余项已经除以 15，因此 0.3 是“平均残余 CE”的权重，不是把 15 项未经平均的 CE 之和再乘 0.3。

例如一个完整更新只有两条录音，帧数分别为 2 和 1：首项有 `2+1+2=5` 个目标，其中 3 个首码本和 2 个 EOS；残余项有 `3×15=45` 个目标。最终为 `S_first/5 + 0.3×S_residual/45`。这按 token 加权，不是先求每条录音平均再对录音平均；较长录音贡献更多帧标签。

### 9.2 FSDP 和 accumulation 下为什么乘 world size

每次优化更新先收集本 rank 的全部 accumulation microbatch，统计两类有效目标数，再用 `dist.all_reduce()` 将分母在所有卡上求和。每个本地 microbatch 用以下值反向：

```python
loss = world_size * (
    local_first_sum / global_first_count
    + residual_weight * local_residual_sum / global_residual_count
)
loss.backward()
```

FSDP 对各 rank 梯度做平均，前面的 `world_size` 抵消这个平均，最终得到上述全局 token 平均目标的梯度。不能把每个 rank 的局部平均 loss 再简单平均，因为各卡的有效帧数不同。

accumulation 的分母已包含当前更新全部 microbatch，所以不再额外除以 accumulation 次数。前面的 microbatch 关闭梯度同步，最后一个开启同步。当前正式配置 `accumulation=1`。

### 9.3 参数、精度和学习率

每个 Talker decoder block 和 Code Predictor block 分别 FSDP2 分片，最后包住整个模型。计算使用 BF16 mixed precision，梯度归约 FP32；CE logits 显式转 float32，优化器更新使用 FP32 参数分片。当前关闭 activation checkpointing。正式配置通过 `model.attn_implementation: flash_attention_2` 为 Talker 和 Code Predictor 启用 Flash Attention 2；未指定时使用 SDPA。训练与压测均设置 `FLASH_ATTENTION_DETERMINISTIC=1`，启用确定性反向计算。

当前环境使用官方预编译 `flash-attn 2.8.3.post1` wheel，匹配 Linux x86_64、Python 3.10、PyTorch 2.8、CUDA 12、CXX11 ABI=true，没有本地编译。依赖文件固定了 wheel URL 和 SHA256。FA2 的 NVIDIA CUDA 实现支持 BF16/FP16 attention，不接受 FP32 Q/K/V；模型初始构造时的 dtype 提示发生在 FSDP mixed precision 包装之前。安装版本与wheel SHA256固定在 `requirements.txt`。其他环境需安装与其 Python、Torch、CUDA 和 ABI 匹配的 wheel。

优化器只接收 `requires_grad=True` 的参数，使用 AdamW：

| 设置 | 当前值 |
|---|---:|
| Talker 预训练 layers/norm 基础学习率 | `1e-4` |
| 新音频模块基础学习率 | `3e-4` |
| weight decay | `0.01` |
| 全模型 gradient clipping norm | `1.0` |
| warmup | 1000 updates |
| 总训练与调度长度 | 38,539 updates（两个epoch） |

warmup 线性增长，之后 cosine 衰减至基础学习率的 10%。loss 和裁剪前梯度 norm 必须有限，否则直接停止。更新顺序是 `backward → clip_grad_norm_ → optimizer.step → scheduler.step`；日志中的学习率是在 scheduler 前进一步后记录的，对应下一次更新将使用的值。

### 9.4 动态预算如何确定

当前每卡预算6000音频帧/9000 Talker token，累积1次。完整数据预检按相同seed和四卡规划，两个epoch分别19270和19269步，总计38539步；预算不应在恢复时随意修改。最长音频与最长文本的组合压力样本已完成四卡实际优化更新；证据在当前run的 `memory-probe.log`，实际长期吞吐和显存见巡检记录。

## 10. 验证 loss 与生成评估

验证 loss 使用与训练相同的文本、目标 codec 和完整目标音频 speaker 条件，但 `no_grad()`。各卡统计 CE 总和和有效 token 数，归约后计算首 CE、平均残余 CE、15 个码本各自的 CE。最后不足一个全局 batch 时，没有真实样本的 rank 用标记为零权重的占位样本维持 FSDP 调用次数，统计时不计它。

当前每500 updates和最后一步验证loss并生成音频。两种模式各使用8条固定验证目标，中英各4条；不额外生成训练文本样本。

生成与训练的区别：

1. speaker 条件换成同说话人的另一条**训练录音**，不用目标录音作为生成参考。
2. 目标codec清空；speaker-only从audio BOS开始，ICL在参考文本+目标文本条件下从参考codec前缀续写。
3. 首head在0…2047和EOS中贪心选择，前两帧新增音频屏蔽EOS，参考帧不计数。
4. 若不是 EOS，帧内依次生成 15 个残余码本，使用已生成的前置码本。
5. 将完整 16 码本帧加入历史，继续下一帧，直到 EOS 或 `max_frames=400`。
6. rank 0用冻结codec decoder还原24kHz波形；ICL先连同参考codec解码，再裁去参考波形。指标只比较目标音频，完整边界见[ICL评估](icl-evaluation.md)。

所有 rank 参与相同的生成前向以满足 FSDP 通信；保存音频和 ASR 在 rank 0 进行。当前生成未使用 KV cache，长音频生成会重复计算历史，评估期间的训练暂停时间应单独看待。

ASR 使用支持中英的 `faster-whisper small`，按每条样本的 `language` 显式转写，CPU int8 执行。原目标音频也跑 ASR 作为对照。英语另外报告 Whisper 英语规范化后的 WER/CER；中文主要看 CER，当前按空白划词的通用 WER 不等价于中文分词后的 WER。混合指标之外按语言分别汇总。

ASR、生成时长比例、EOS、截断率都只是评估指标，不进入训练反向传播。验证 CE 使用目标音频条件，而生成使用另一条录音条件，二者也不能视为完全相同输入条件下的同一个指标。

## 11. 保存、恢复和可复现边界

实现：[`checkpoint.py`](../../qwen3_train/checkpoint.py)。

每 500 updates 保存一次，也在最后一步保存；保留最近两个完整 checkpoint。每份包含 DCP 模型和优化器状态、scheduler、`step / epoch / next_batch`、每个 rank 的 Python/NumPy/PyTorch/CUDA RNG 状态，以及恢复签名。

先写 `.incomplete`，全部 rank 保存完成后写 `COMPLETE`、rename，再更新 `latest`。只有新 checkpoint 完成后才清理旧的完整 checkpoint。恢复要求相同 world size、模型与冻结设置、manifest 哈希、动态组批算法与预算、accumulation、优化器和调度关键设置。当前签名还记录音频解码版本、`speaker_conditioning=full_target_audio` 与 `generation_conditioning=other_training_utterance`。

```bash
NPROC_PER_NODE=4 bash scripts/run_train.sh \
  --config configs/emilia-10kh-pretrain.yaml \
  --resume latest
```

模型加载会核对组装配置/权重，训练会核对 NPZ 哈希；原始 tar 本身没有做全库内容哈希。由于 speaker 条件在线读取原音频，精确恢复仍要求源 tar 保持不变。只保留 manifest 和 NPZ、却更换原始 tar，并不满足这个条件。

当前 JSONL 会在每个 rank 全量常驻内存，NPZ 是每条录音一个文件。它适用于当前工程基线，尚不是数十万小时语料的分片流式实现。

## 12. 训练前已验证什么

当前覆盖官方非流式/ICL输入对照、teacher forcing与生成时序、变长样本隔离、全局token归一化、冻结参数、动态组批预算和精确恢复。EOS修复的47项完整测试及实际四卡复评记录在当前run的 `eos-minimum-fix/`。历史小规模验证原始产物已按清理要求删除，当前正式运行的预检、恢复和冻结检查证据仍完整保留。

可重复运行的基础检查：

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=. .venv/bin/torchrun --standalone --nproc_per_node=4 \
  tests/check_distributed_equivalence.py \
  --speaker --qwen-protocol --frozen-frontend --frozen-speaker
```
