# 文本前端分析：LFM text2048、Qwen-TTS 前端与 textbase 的 SIM 问题

日期：2026-10-14。本文是一次性诊断记录，不是正式实验。分析脚本和结果 JSON 保存在 [evidence/text-frontend-20261014/](evidence/text-frontend-20261014/)；脚本使用本机的绝对路径，只读 checkpoint 与导出权重，不训练。它修正了 [speaker 条件探针报告](speaker-conditioning-probe-20261011.md) 第 5.2 节和 [all16 评测报告](all16-training-and-seedtts-evaluation.md) 第 9 节中关于“文本维度 / 容量”的推断。

---

## 0. 结论摘要

1. **LFM text2048 前端在功能上等价于一张 1024 维表。** `[E, −E]` 配对初始化下，表的新增分量 `(A+B)/2` 训练到 step 6,000 时能量只有原分量的 1.3e-5；projector 的 SiLU 输入 rms 约 0.03，整个 MLP 可被线性映射精确拟合（R² = 1.0000）。2048 宽度与 MLP 没有带来表达能力，只多出约 7,400 万参数。不影响效果。
2. **官方 Qwen-TTS 文本前端同样近似线性**（R² 0.9999，把 SiLU 换成 `x/2` 输出只差 1.3%）。离散 token 查表后接线性映射，等价于一张 `vocab × 1024` 的表；2048 维表的有效秩（约 1,700）不是 Talker 实际看到的量，projector 输出的有效秩只有约 146。**“TTS 2048 维 vs textbase 1024 维”的容量论证不成立**。官方表与发布的 Qwen3-1.7B 无继承关系（逐行余弦最大 0.098），但连代码类 token 都有 LM 式近邻结构，应是独立初始化、在通用文本上训练后冻结共用的表；TTS 适配更可能在与之共同训练的 projector 中（第 3.1 节）。
3. **textbase SIM 低的直接原因是它对 speaker embedding 的利用学得慢。** step 17,500 时三组样本量相同：以只用 ICL 的 no-speaker 为基线，加 speaker embedding 后英文 SIM frozen-conditioning +12.0，textbase 仅 +2.3。探针一致：换成其他说话人后残余码本 CE frozen +0.88、textbase +0.53；用自身 embedding 时 textbase 的残余 CE 也更高（5.59 vs 5.49）。内容准确率未受同等影响（采样英文 WER 5.15 vs 5.71）。到 25,000 时该增量升到 0.71、自身 CE 已优于 frozen 17,500，是延迟而非上限。
4. **原因尚未隔离。** 维度已排除；最大嫌疑是随机初始化的 projector（开局打乱预训练主干的输入，且前端一直在变）。LFM 前端是保持函数的初始化，按消费样本量看 SIM 明显好于 textbase，可作旁证，但数据、refmask、主干均不同。
5. **text_pad 掩盖 speaker 信号的假设不成立**：speaker 向量范数约 10.5–10.9，text_pad 只有 0.7–1.5。
6. **建议**：Qwen 后续若采用可训练的 LLM 文本前端，使用已有的 `near-identity` projector 初始化，其余照搬 Qwen Emilia2 refmask，作为与冻结 TTS 前端的单变量对照（第 6 节）。LFM 以后改用 native 1024 表加单位阵初始化的线性层，属于清理。

---

## 1. LFM text2048 前端的结构

组装 `assets/assembled/lfm2.5-230m-base-text2048-spk`，`lm_tts/models/lfm/frontend.py` 的 paired-SiLU 初始化：

| 组件 | 初始化 | 作用 |
| --- | --- | --- |
| embedding 65,536 × 2,048 | `[E, −E]`，`E` 为 LFM 1,024 维表 | 存 `e` 和 `−e` |
| fc1 2,048 → 2,048 | 单位阵，bias 0 | `h = [e, −e]` |
| SiLU | — | `[SiLU(e), SiLU(−e)]` |
| fc2 2,048 → 1,024 | `[I, −I]`，bias 0 | `SiLU(e) − SiLU(−e) = e` |

利用 `SiLU(x) − SiLU(−x) = x`，step 0 的前端输出与 LFM 原生 embedding 完全相同。

## 2. LFM 训练后的前端（`lfm_text_embedding.py`）

Run：`supervised-tts-20261012-emilia2-lfm230m-text2048-spk-refmask-bf16-8gpu-acc2-lr3e-4-bblr1e-4-ep3-wsd`。只统计训练中被更新过的行（与初始化余弦 < 1 − 1e-5；65,536 行中 64,405 行为有效 token）。有效秩为奇异值平方归一化后的谱熵指数，均先减去均值。该口径在 Qwen-TTS 表上得到 1,703 / 90% 1,607 维 / 99% 1,985 维，与第 5.2 节的 1,671 / 1,574 / 1,980 相差约 2%，可横向比较。

| 指标 | init | step 1,000 | step 3,000 | step 6,000 |
| --- | ---: | ---: | ---: | ---: |
| 被更新的行 | — | 55,871 | 57,059 | 57,936 |
| embedding 行与 init 余弦（中位数） | 1 | 0.980 | 0.965 | 0.948 |
| embedding 有效秩 / 90% / 99% 维数 | 894 / 848 / 998 | 885 / 845 / 998 | 879 / 845 / 999 | 869 / 844 / 999 |
| 新增分量能量比 `‖(A+B)/2‖² / ‖(A−B)/2‖²` | 0 | 2.6e-6 | 6.6e-6 | 1.3e-5 |
| fc1 偏离单位阵（相对 Frobenius） | 0 | 0.168 | 0.252 | 0.338 |
| fc2 偏离 `[I, −I]` | 0 | 0.120 | 0.176 | 0.231 |
| projector 输出有效秩 / 90% / 99% 维数 | 894 / 848 / 998 | 689 / 747 / 979 | 553 / 647 / 949 | 460 / 555 / 904 |
| projector 输出范数均值 | 0.92 | 1.05 | 1.18 | 1.39 |
| projector 输出与 init 余弦（中位数） | 1 | 0.880 | 0.802 | 0.734 |
| top-10 近邻与 init 重合：embedding / projector 输出 | 1 / 1 | 0.89 / 0.71 | 0.84 / 0.60 | 0.80 / 0.52 |

- **表的后半部分没有长出新信息。** 新增分量经 fc1 同时进入隐层两半，fc2 相减后在线性区被抵消，只剩二阶项；其梯度约为原分量的 `2·SiLU'(h) − 1 ≈ h` 倍（约 3–4%）。实测 step 6,000 时原分量每行变化约 0.4，新增分量约 0.003。
- **文本表示被大幅改写，主要由 projector（LR 3e-4）完成**；表本身（LR 1e-4）变化较小。projector 输出持续压缩到更低维度，方向与官方前端一致（第 3 节），到 step 6,000 尚未饱和。

## 3. MLP 在实际中是线性的（`deep.py`）

对全部有效 token 行计算 projector：

| | SiLU 输入 rms | `|h| > 1` 比例 | SiLU 换成 `x/2` 的输出相对误差 | 线性最小二乘拟合 R² | 输出范数均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LFM init | 0.028 | 0 | 0.0% | 1.0000 | 0.90 |
| LFM step 1,000 | 0.030 | 0 | 0.1% | 1.0000 | 1.02 |
| LFM step 6,000 | 0.038 | 0 | 0.3% | 1.0000 | 1.35 |
| 官方 Qwen-TTS 0.6B | 0.027 | 0 | 1.3% | 0.9999 | 1.57 |

- SiLU 在 `|x| ≈ 0.03` 的范围内近似直线 `x/2`，两层 MLP 退化为一个矩阵。离散 token 查表后接线性映射，等价于一张 `vocab × 1024` 的表；表完全可训练时，任何前端宽度都不改变每个 token 能表达的上限。
- **官方 Qwen-TTS**：projector 输出（Talker 实际看到的量，统计全部 151,643 个常规 token 行）有效秩约 146，90% 方差 133 维、99% 方差 294 维。2048 维表的有效秩约 1,700 只描述中间表示。统计包含大量在 TTS 训练中极少出现的 token，具体数值可能偏低，但远低于 2,048 的结论不变。
- 因此第 5.2 节“有效维度约 1,600 vs 900”、第 9 节“更窄的 embedding 和 projector 减少了容量”不能作为 textbase 劣势的解释。
- 尺度：文本输出范数 0.90 → 1.35，首码本 codec embedding 0.64 → 1.40，两者同步增长、量级相当，无失配。

### 3.1 官方 TTS 文本表的来源（`origin17.py`、`rarity.py`）

本地没有 Qwen3-1.7B-Base，用其微调版 Qwen3-ASR-1.7B 代替（参照：Qwen3-0.6B-Base 与 ASR-0.6B 逐行余弦 0.925，微调基本保留 embedding）。

| TTS-1.7B text_embedding vs Qwen3-ASR-1.7B embed_tokens | 数值 |
| --- | ---: |
| 逐行余弦 中位数 / 90% / 99% / 99.9% | 0.0001 / 0.028 / 0.052 / 0.068 |
| 151,643 行最大值 / 余弦 > 0.1 的行数 | 0.098 / 0 |
| 参照：随机 2,048 维向量余弦标准差 | 0.022 |

TTS 表与 Qwen3-0.6B-Base 的 top-10 近邻重合，按 token ID 十分位（ID 越大大致越罕见）：

| ID 区间 | 0–15k | 45k–61k | 91k–106k | 121k–136k | 136k–152k |
| --- | ---: | ---: | ---: | ---: | ---: |
| 近邻重合 | 70.6% | 60.1% | 53.9% | 33.6% | 33.7% |
| TTS 行范数 | 0.72 | 0.80 | 0.82 | 0.68 | 0.53 |

抽样：代码类 token（`#include`、`</div>`、`self.` 等）56%，常见英文 59%，常见中文 56%。随机为 0.007%。

推断：

- **学过覆盖整个词表的通用文本**：TTS 转写几乎不含代码，若只用 TTS 数据训练，这些行会停在随机初始化附近；实际与常见词同样有 LM 式结构，重合度随罕见程度递减。
- **不是发布的 Qwen3-1.7B 继承或微调**：若由其初始化，罕见 token 行应保留原方向；实际没有任何一行余弦超过 0.1，线性 R² 仅 0.15–0.20。坐标系是独立初始化后训练出来的。
- **训练后冻结、跨模型共用**：TTS-0.6B 与 1.7B 除 3 个控制 token 行外逐 bit 相等，1.7B-Base 与 CustomVoice 完全相同；两模型的 projector 相互独立（fc1 余弦 0.005）。
- 第一阶段是哪个模型无法从权重判断。TTS-1.7B Talker 层对 Qwen3-1.7B 有 0.08–0.18 的残留而 embedding 没有，原因不明。
- 因此官方表本身更像一张通用 LM embedding，与 textbase 用的 Qwen3-0.6B-Base 同类；TTS 适配更可能体现在冻结表上共同训练的 projector（输出有效秩约 146）。

## 4. textbase 为什么较差

### 4.1 同样本量的三组对照

step 17,500 时 frozen-conditioning、textbase、no-speaker 均消费约 1.42 亿条 all16 样本，同一划分，均为双采样（no-speaker 为纯 ICL，其余为 speaker + ICL）：

| 17,500 双采样 | speaker embedding | 文本前端 | 英文 WER | 英文 SIM | 中文 CER | 中文 SIM |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| frozen-conditioning | 有 | TTS，冻结 | 5.71 | **56.8** | 11.35 | **67.2** |
| textbase | 有 | Qwen3-0.6B-Base，可训练，随机 projector | 5.15 | 47.1 | 12.86 | 57.7 |
| no-speaker | 无 | TTS，冻结 | 4.40 | 44.8 | 22.66 | 48.6 |

以 no-speaker 为基线，加入 speaker embedding 后英文 SIM：frozen **+12.0**，textbase **+2.3**。no-speaker 中文 CER 为 22.7%，中文 SIM 的差值不作为依据。textbase 25,000 双采样升到 53.1 / 64.9，仍低于 frozen 17,500。

### 4.2 探针：textbase 对 speaker embedding 的利用更弱

取自探针报告第 3 节（teacher forcing，3,566 句）。数值为换条件后 CE 相对 `自身完整` 的增量：

| 条件 | frozen 17,500 | textbase 17,500 | textbase 25,000 |
| --- | ---: | ---: | ---: |
| 残余码本，其他说话人（主要反映音色） | +0.88 | +0.53 | +0.71 |
| 首码本，其他说话人 | +0.79 | +0.64 | +0.75 |
| 首码本，同说话人另一条（内容捷径） | +0.30 | +0.21 | +0.28 |
| 自身完整：首码本 / 残余码本 CE | 1.115 / 5.488 | 1.146 / 5.591 | 1.079 / 5.441 |

- textbase 对 speaker embedding 的内容捷径和音色两种用法都更弱：前者与英文 WER 更好一致，后者与 SIM 更低一致。用自身 embedding 时的残余 CE 也更高，声学细节建模整体更差。
- **textbase 是“学得慢”，不是“学不到”。** 17,500 → 25,000，残余码本对其他说话人的增量 0.53 → 0.71，首码本 0.64 → 0.75，逐步接近 frozen 17,500（0.88 / 0.79），但在晚 7,500 步时仍未达到；SIM 同步从 47.1 升到 53.1，仍低于 frozen 17,500 的 56.8。用自身 embedding 时的 CE 已优于 frozen 17,500（1.079 / 5.441 vs 1.115 / 5.488），声学细节建模本身已追上，剩余差距集中在 speaker 利用。内容捷径（同说话人另一条）也同步增长到 +0.28，EOS 对自身 embedding 的依赖同样上升（自身 0.60，换条件后约 1.39）。

### 4.3 原因

| 候选 | 判断 |
| --- | --- |
| 维度更窄（1,024 vs 2,048） | **排除**（第 3 节） |
| text_pad 掩盖 speaker 信号 | **排除**：speaker 向量范数 10.9（说话人特有部分 2.8），text_pad 0.71（textbase 17,500）/ 1.54（frozen 17,500）；LFM 分别为 10.5 / 3.0 / 0.78（`spk.py`） |
| 缺少 TTS 训练过的文本知识 | 可解释中文 CER 偏高，难以解释不使用 speaker embedding |
| **随机 projector + 前端可训练** | **最大嫌疑**：开局打乱进入预训练主干的输入，且训练中持续变化，学习被优先用于适配文本输入；未经单变量对照证实 |

### 4.4 LFM 旁证

按消费样本量对齐，LFM（LLM 来源、可训练、保持函数的初始化）明显好于 textbase：

| speaker + ICL 双采样 | 已消费样本 | 英文 WER | 英文 SIM | 中文 CER | 中文 SIM |
| --- | ---: | ---: | ---: | ---: | ---: |
| textbase 17,500 | 约 1.42 亿 | 5.15 | 47.1 | 12.86 | 57.7 |
| LFM 5,000 | 0.35 亿 | 8.68 | 39.2 | 9.40 | 55.2 |
| LFM 6,000 | 0.42 亿 | 9.43 | 43.2 | 8.73 | 58.6 |
| Qwen Emilia2 refmask 10,000 | 0.67 亿 | 5.02 | 53.5 | 4.59 | 65.5 |

LFM 6,000 结果取自 `UltraEval-Audio/res/emilia2-lfm230m-text2048-spk-refmask-step6000-seedtts-sampling-bs1-20261014-8gpu-4wpg/full/results_table.md`，记录时英文 1,086 / 1,088 条已评分、审计尚未写出，数值可能小幅变化。

LFM 6,000 只用约 0.3 倍样本，中文 SIM 已超过 textbase 17,500（58.6 vs 57.7），中文 CER 更低；英文 SIM 仍低 3.9。5,000 → 6,000（约 690 万样本）SIM 英文 +4.0、中文 +3.4，仍在快速上升。数据（Emilia2 vs all16）、refmask、主干均不同，只能作为“保持函数的初始化可能避免 textbase 问题”的旁证。

## 5. 被否定的初始化改法

- **`[E, R]`（第二半随机，fc1 对应列置零）**：保持函数、能学到新分量，但 (1) 线性前端最终等价于一张表，没有收益；(2) 训练中 fc1 的 R 列长大后，未见或罕见 token 的随机 R 会作为噪声进入输出（LFM 中约 6,500 个有效 token 从未被更新）。若一定要加，应为 R 置零、fc1 列小随机。
- **残差 MLP `e + MLP(e)`**：结构更直观，但同样等价于一张表，只属于清理。
- **`[E, −E]` 上加噪声**：新增分量仍在被抵消的方向上，梯度同样弱。

## 6. 后续建议

1. **Qwen 采用 LFM 式前端的单变量对照**：`--text-initialization text-base --text-projection-init near-identity --train-text-frontend`（`lm_tts/models/assembly/qwen.py`，fc1 = I、fc2 = 2I，利用 `2·SiLU(x) ≈ x`，小输入下误差约 1–2%）。数据、refmask、16 卡 acc1、LR、WSD 全部照搬 Qwen Emilia2 refmask，只有文本前端不同。
   - 早期信号：step 2,500 / 5,000 跑本探针，对照 Qwen Emilia2 同 step checkpoint 的“其他说话人”残余码本 CE 增量；明显偏低即在重演 textbase。
   - step 10,000 Seed-TTS 对照基线 5.02 / 53.5 / 4.59 / 65.5。SIM 接近说明 textbase 的问题在随机 projector；仍偏低说明“LLM 来源 + 可训练”本身是问题，Qwen 应继续用冻结 TTS 前端。
   - 不选 `identity`：near-identity 保留共享、LR 3e-4 的变换，与 LFM 实际的改写方式一致。
2. **LFM 以后的组装**：native 1,024 表加单位阵初始化的 1,024 × 1,024 线性层（3e-4 参数组），功能与现有前端等价，省约 7,400 万参数和对应优化器状态；需新增 `linear` projector 选项。现有 run 不需要改。
3. 可选对照：保留 TTS 前端但解冻训练（第 9 节第二项）。

## 7. 局限

- 有效秩口径与第 5.2 节有约 2% 差异；官方 projector 输出统计包含大量罕见 token。
- LFM 与 textbase 的比较受数据、refmask、主干影响，不是对照实验。
- 探针为 teacher-forcing CE，与 SIM 的定量关系未知，方向与评测一致。
- LFM run 在 step 约 6,710 收到 SIGTERM 后停止，按决定不再恢复；最后 checkpoint 为 step 6,000。
