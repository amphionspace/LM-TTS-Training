# dots.tts 风格指令 TTS：实验记录（2026-11-09 起）

在 `dots.tts-soar` 上训练"参考给音色、指令给风格"的 TTS：

- **音色**只来自参考音频（x-vector 或 ICL）。
- **风格**（情感、语速、音量、耳语、语气）由自然语言指令决定。
- **不做 voice design**：不根据文字描述生成音色。

设计取舍见 [设计文档](../design/dots-instruct-tts-20261108.md)。本页记录实际做法、全部实验、评测结果和分析；两者不一致时以本页为准。最后更新：2026-11-17 09:20 UTC（补齐 v5 两组低学习率的全量评测）。

## 1. 结论速览

| | 结论 | 证据 |
| --- | --- | --- |
| 当前最好 | **SC-v6-earsexp**：音色、WER 与 SC-v5-hifi 持平（Seed en 69.6），英文耳语 / 慢 / 高补回到 100%，自由描述胜率最高；短板是英文大声反向、快无效、英文情感比 v5 低 5–9 个点、NV PCER 略差。SC-v5-hifi 英文情感最好 | §3 |
| 英文音色 | 英文 speaker_only SIM 低的主因是**缺少"多说话人 + 默认指令 + 句内高相似度参考"的英文数据**；中文由 emilia_nv（573 h）承担这一角色。补 HiFiTTS-2 后英文 Seed SIM 58.7 → 69.5 | §3.2、§4.1 |
| 指令描述音色 | FlexiVoice v3 中约 69% 的指令描述性别 / 年龄 / 口音，教模型"音色听指令"，叠加高 lr 时英文 SIM 掉得最多 | §4.1、§5.3 |
| 说话方式控制 | 语速、音高、耳语、大声的描述**几乎只在 v4 删掉的子集里**；删掉后这类控制大幅下降。问题在 FlexiVoice 的指令，不在音频 | §4.2 |
| 学习率 | lr 2e-5 比 5e-5 英文 SIM 高 4–7 个点，情感不变 | §3 |
| 外部参照 | CosyVoice3 在 StyleBench 情感上 ACC-I 比 SC-v5-hifi 低 25–45 个点（只看它官方支持的开心 / 伤心 / 生气也一样），但音色相似度更高；ICL 下同样没有风格控制 | §3.6 |
| 学习率 | v6 数据上降到 2e-5 / 1e-5：WER 低 0.3–0.7、SIM 不变、情感控制弱 5–11 个点；风格控制仍以 5e-5 为好 | §3.1 |
| v5 学习率 | 两组低 lr 全套评测均完成。1e-5 的 Seed 四组错误率都最低，CV3 普通测试也多数改善，SIM 基本持平；但两组情感和慢速控制弱于原版，NV PCER 退步，不能全面替换 5e-5 | §3.1–3.4、§4.8 |
| Vccm | 原版 SC-v5-hifi、CosyVoice3、真人各 2,606 条全部完成；CV3 情感 / 独立音高 / WER 三组均更好，v5 SIM 三组均更高。v5 常规 WER 受一条 ASR 重复解码异常明显影响 | §3.7、§4.7 |
| 情感排名反转 | 交叉使用两套分类器后，StyleBench 英文 easy 仍是 v5 更高、Vccm 仍是 CV3 更高；分类器差异与类别比例都不足以单独解释反转 | §4.6 |
| 未解决 | ICL 下没有风格控制；中文耳语无效；韩语 / 日语退化；FlexiVoice 指令整体不可靠（LLM 从元数据生成） | §4.3–4.5 |

## 2. 实验

**命名**：`SC-<数据版本>-<与默认不同的设置>`（SC = style control）。数据版本对应 §5.2 的 selection；默认设置为 v3-ref3s 特征、50% interleave、lr 5e-5、2 epoch。对照组：`base` 为 release `dots.tts-soar`；`NV-ref3s` 为只用 NV 数据的副语言实验（lr 1e-5，见 `dots-tts-emilia-nv-20261103.md`）。

| 名字 | 原叫法 | 数据 | 与默认的差别 | ACP | 最终 step |
| --- | --- | --- | --- | --- | ---: |
| base | release | — | 未微调 | — | — |
| NV-ref3s | ref3s | 只有 NV | lr 1e-5 | — | 8191 |
| SC-v2-il100 | nocaf / step794 | v2 + NV | v2-ref3s 特征（250 ms 归一化）、100% interleave | `pt-ii8bqdpg` | 794 |
| SC-v3-lr5e-5 | raw v3 / step932 | v3 + NV | — | `pt-x7s1wo9c` | 932 |
| SC-v3-lr2e-5 | lr 2e-5 | v3 + NV | lr 2e-5 | `pt-fg1ai7mx` | 932 |
| SC-v4 | v4 | v4 + NV | — | `pt-r8wymqby` | 376 |
| **SC-v5-hifi** | v5 | v4 + NV + HiFiTTS-2 | 3 epoch | `pt-n7jo9ftr` | 723 |
| **SC-v6-earsexp** | — | v6 + NV + HiFiTTS-2 | 3 epoch；EARS / Expresso 用按录音协议重写的风格指令 | `pt-z8rctx9i` | 780 |
| SC-v6-earsexp-lr2e-5 | — | 同 SC-v6-earsexp | lr 2e-5 | `pt-d3dxabeo` | 780 |
| SC-v6-earsexp-lr1e-5 | — | 同 SC-v6-earsexp | lr 1e-5 | `pt-gmkxj6lw` | 780 |
| SC-v5-hifi-lr2e-5 | v5 | 同 SC-v5-hifi | lr 2e-5 | `pt-467lbrpk` | 723 |
| SC-v5-hifi-lr1e-5 | v5 | 同 SC-v5-hifi | lr 1e-5 | `pt-fv681fpm` | 723 |

- 比较时注意：SC-v2-il100 → SC-v3 同时变了数据（+32.6 万条 KeSpeech）、特征、interleave 三项；SC-v3-lr5e-5 → SC-v3-lr2e-5 只变 lr；SC-v3-lr5e-5 → SC-v4 只变 FlexiVoice 子集；SC-v4 → SC-v5-hifi 加 HiFiTTS-2 并多训 1 个 epoch。
- NV 五个数据集在所有 SC 实验里相同，特征版本随 FlexiVoice 一起换。
- 早期停掉的实验：基线 `pt-c42buvr4`（分块打乱，step 9000 时部分子集没训到）、32k packed DiT `pt-ng26vb3i`、48k lr 5e-5 `pt-jt0v0he9`（发现开头咔声）、nocaf 50% interleave `pt-0xu9i6xh`（数据重做后停止）、lr 1e-5 `pt-p6j3rydu`（约 step 690 停止让出 GPU）。

目录（目录名沿用原名）：

| 名字 | run（`/workspace/LM-TTS-Training-Runs/dots-tts-…-wsd`） | 导出（`UltraEval-Audio/init_model/`） | 结果（`UltraEval-Audio/res/`） |
| --- | --- | --- | --- |
| base | — | release | StyleBench `dots-tts-soar-release-plain-stylebench-v1`（`--plain`）；suite 数字取自 NV 文档 |
| NV-ref3s | `20261106-soar-nv26-5ds-ref3s-noxvecdrop-frozenhead-…-lr1e-5-ep2` | `dots-soar-nv26-5ds-ref3s-frozenhead-step-00008191` | `dots-soar-nv26-5ds-ref3s-frozenhead-step8191-default-sampling` |
| SC-v2-il100 | `20261111-flexivoice-nocaf-interleave-…-lr5e-5-ep2` | `dots-flexivoice-nocaf-48k-ep2-step-00000794` | StyleBench `…-step794-stylebench-v1`；suite `…-step794-default-instruction` |
| SC-v3-lr5e-5 | `20261112-flexivoice-raw-interleave50-…-lr5e-5-ep2` | `dots-flexivoice-raw-interleave50-ep2-step-00000932` | `dots-flexivoice-raw-interleave50-ep2-step932-default-instruction` |
| SC-v3-lr2e-5 | `20261112-flexivoice-raw-interleave50-…-lr2e-5-ep2` | `dots-flexivoice-raw-interleave50-lr2e-5-ep2-step-00000932` | `dots-flexivoice-raw-interleave50-lr2e-5-ep2-step932-default-instruction` |
| SC-v4 | `20261113-flexivoice-v4-interleave50-…-lr5e-5-ep2` | `dots-flexivoice-v4-interleave50-ep2-step-00000376` | `dots-flexivoice-v4-interleave50-ep2-step376-default-instruction` |
| SC-v5-hifi | `20261113-flexivoice-v5-hifitts2-interleave50-…-lr5e-5-ep3` | `dots-flexivoice-v5-hifitts2-interleave50-ep3-step-00000723` | `dots-flexivoice-v5-hifitts2-interleave50-ep3-step723-default-instruction` |
| SC-v6-earsexp | `20261115-flexivoice-v6-earsexp-interleave50-…-lr5e-5-ep3` | `dots-flexivoice-v6-earsexp-interleave50-ep3-step-00000780` | `dots-flexivoice-v6-earsexp-interleave50-ep3-step780-default-instruction` |
| SC-v6-earsexp-lr2e-5 / lr1e-5 | `20261116-flexivoice-v6-earsexp-interleave50-…-lr{2e-5,1e-5}-ep3` | `dots-flexivoice-v6-earsexp-lr{2e-5,1e-5}-interleave50-ep3-step-*` | `dots-flexivoice-v6-earsexp-lr{2e-5,1e-5}-interleave50-ep3-step*-default-instruction` |
| SC-v5-hifi-lr2e-5 / lr1e-5 | `20261117-flexivoice-v5-hifitts2-interleave50-…-lr{2e-5,1e-5}-ep3` | `dots-flexivoice-v5-hifitts2-lr{2e-5,1e-5}-interleave50-ep3-step-*` | `dots-flexivoice-v5-hifitts2-lr{2e-5,1e-5}-interleave50-ep3-step*-default-instruction` |

## 3. 结果

评测方法见 §7。粗体为该列最好（base 除外）。

### 3.1 StyleBench（speaker_only，scale 1.0）

**情感**（ACC-I：指令情感命中率 ↑；ACC-R：hard 中照搬参考情感的比例 ↓；真实录音上限英文 76%、中文 51%）

| 模型 | en easy | en hard | en ACC-R | zh easy | zh hard | zh ACC-R | SIM en / zh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base（噪声底） | 6.2 | 8.8 | 26.0 | 0.8 | 6.7 | 29.4 | 0.777 / 0.780 |
| SC-v2-il100 | 46.0 | 45.6 | 14.8 | 56.9 | **57.5** | **21.7** | 0.580 / 0.623 |
| SC-v3-lr5e-5 | 53.1 | **53.8** | **10.2** | **57.5** | 55.0 | 23.1 | 0.563 / 0.622 |
| SC-v3-lr2e-5 | 53.5 | 50.0 | 13.5 | 56.9 | 54.8 | 23.3 | 0.550 / 0.621 |
| SC-v4 | 46.5 | 42.5 | 15.6 | 55.2 | 52.9 | 27.1 | 0.575 / 0.616 |
| SC-v5-hifi | **53.8** | 46.7 | 15.4 | 57.3 | 56.7 | 23.3 | 0.556 / 0.617 |
| SC-v6-earsexp | 45.0 | 41.7 | 13.5 | **59.0** | 57.3 | 22.1 | 0.564 / 0.596 |
| SC-v6-earsexp-lr2e-5 | 35.8 | 36.7 | 16.0 | 52.1 | 53.5 | 24.8 | 0.588 / 0.616 |
| SC-v6-earsexp-lr1e-5 | 37.1 | 37.1 | 14.6 | 48.1 | 46.5 | 27.3 | 0.579 / 0.635 |
| CosyVoice3（外部参照，§3.6） | 27.9 | 27.7 | 18.3 | 15.6 | 29.4 | **17.7** | 0.605 / 0.762 |
| SC-v5-hifi-lr2e-5 | 44.8 | 46.0 | 16.5 | 54.6 | 49.4 | 27.9 | 0.570 / 0.639 |
| SC-v5-hifi-lr1e-5 | 47.3 | 41.7 | 17.5 | 49.4 | 47.7 | 26.9 | 0.591 / 0.649 |

**说话方式**（相对默认指令的成功率 %）

| 模型 | en 耳语 | en 大声 | en 快 | en 慢 | en 高 | en 低 | zh 耳语 | zh 大声 | zh 快 | zh 慢 | zh 高 | zh 低 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0 | 17 | 10 | 7 | 2 | 12 | 0 | 28 | 3 | 3 | 8 | 8 |
| SC-v2-il100 | 93 | **41** | 48 | 86 | 31 | 64 | **18** | 30 | 23 | 43 | 73 | **75** |
| SC-v3-lr5e-5 | 81 | 14 | 55 | 79 | 52 | 55 | 8 | **43** | 45 | 58 | **80** | 63 |
| SC-v3-lr2e-5 | 60 | 12 | **79** | 91 | 31 | **81** | 8 | 15 | **68** | 63 | 73 | 63 |
| SC-v4 | 24 | 5 | 12 | 33 | 52 | 17 | 10 | 18 | 33 | 50 | 53 | 28 |
| SC-v5-hifi | 5 | 2 | 10 | 60 | 41 | 21 | 5 | 35 | 28 | **80** | 55 | 55 |
| SC-v6-earsexp | **100** | 0 | 7 | **100** | **100** | 69 | 10 | 20 | 58 | 55 | 48 | 48 |
| SC-v6-earsexp-lr2e-5 | 98 | 2 | 7 | 98 | 93 | 79 | 15 | 18 | 48 | 53 | 53 | 43 |
| SC-v6-earsexp-lr1e-5 | 93 | 5 | 5 | **100** | 95 | 71 | 18 | 25 | 43 | 43 | 53 | 60 |
| SC-v5-hifi-lr2e-5 | 4.8 | 0.0 | 7.1 | 47.6 | 42.9 | 21.4 | 5.0 | 37.5 | 30.0 | 47.5 | 47.5 | 47.5 |
| SC-v5-hifi-lr1e-5 | 16.7 | 9.5 | 9.5 | 45.2 | 47.6 | 14.3 | 5.0 | 22.5 | 37.5 | 30.0 | 52.5 | 37.5 |

**自由描述**（胜率：指令版比默认版更接近真人录音的比例，50% 即无作用；CER 为指令版）

| 模型 | 胜率 中 / 英 / 日 / 韩 | CER 中 / 英 / 日 / 韩 |
| --- | --- | --- |
| base | 48.5 / 50.0 / 54.5 / 46.0 | 4.1 / 2.5 / 7.6 / 8.3 |
| SC-v2-il100 | 50.5 / 52.5 / 48.5 / 53.0 | 4.6 / 2.7 / 15.8 / 51.0 |
| SC-v3-lr5e-5 | 58.5 / 49.5 / 54.5 / 52.0 | 6.7 / 6.5 / 11.5 / 62.5 |
| SC-v3-lr2e-5 | 53.5 / 53.0 / 46.5 / 60.0 | 6.2 / 5.8 / 10.5 / 56.2 |
| SC-v4 | 56.5 / 47.0 / 53.5 / 56.0 | 6.0 / 4.7 / 10.5 / 51.1 |
| SC-v5-hifi | 55.0 / 57.0 / 49.5 / 55.5 | 6.3 / 6.0 / 10.7 / 68.1 |
| SC-v6-earsexp | **60.5 / 60.5** / 50.0 / 53.5 | 5.8 / 6.1 / 10.1 / 55.4 |
| SC-v6-earsexp-lr2e-5 | 61.0 / **66.5** / 49.5 / 55.5 | 6.4 / 5.1 / 10.0 / 52.2 |
| SC-v6-earsexp-lr1e-5 | **61.5** / 58.0 / 54.5 / 49.5 | 5.7 / 4.9 / 9.4 / 56.4 |
| SC-v5-hifi-lr2e-5 | 58.0 / 62.5 / 54.0 / 55.0 | 6.7 / 5.5 / 11.0 / 55.0 |
| SC-v5-hifi-lr1e-5 | 61.5 / 61.5 / 53.0 / 51.0 | 6.5 / 6.1 / 9.5 / 65.1 |

- 风格控制学到了：情感、说话方式都远高于噪声底，参考情感泄漏下降。
- **ICL 下没有风格控制**：所有模型 ACC-I 只有 3–15%，与 base 相同。
- **SC-v4 说话方式大幅下降**，SC-v5-hifi 中文慢 / 高 / 低回升，但**英文耳语、大声、快几乎失效（2–10%）**（原因见 §4.2）。
- **SC-v6-earsexp**（+ EARS / Expresso，协议风格指令）：英文耳语、慢、高音 100%，低音 69%，v5 丢掉的英文说话方式大部分回来了；但**英文大声 0% 且方向相反**（中位响度比默认指令低 7.2 dB），英文快 7%（语速比 0.99），待查。英文情感比 SC-v5-hifi 低 5–9 个点（angry 86→73、sad 73→60，可能被 EARS 的 22 类细分情感冲淡），中文略升。自由描述胜率中 / 英 60.5%，为所有版本最高。
- **SC-v6 学习率扫描**：lr 2e-5 / 1e-5 的 SIM 与 5e-5 几乎相同（Seed en 69.3–69.6），WER / CER 低 0.3–0.7 个点，但情感控制明显变弱（en 低 5–9、zh 低 5–11 个点），英文说话方式基本不变。与 v3 数据上"降 lr 救英文 SIM"不同：HiFiTTS-2 已稳住英文音色，lr 不再是 SIM 的主因。英文大声 / 快在所有 lr 下都失效（0–7%），与 lr 无关。风格控制仍以 lr 5e-5 为主，2e-5 为偏 WER 的折中。
- **SC-v3-lr2e-5 与 SC-v3-lr5e-5**：情感几乎相同；说话方式互有高低，每格约 40 条，看不出整体优劣。
- 英文"大声"从 SC-v3 起都失效（生成响度反而低于默认指令）；训练数据里 EARS 大声比中性高 6.7 dB，问题不在数据本身。
- 自由描述胜率都在 50% 附近，这个测法区分不出模型差别（§7.1）。

### 3.2 Seed-TTS（WER/CER % ↓ / SIM ×100 ↑）

| 模型 | en speaker_only | zh speaker_only | en ICL | zh ICL |
| --- | --- | --- | --- | --- |
| base | 2.15 / 72.7 | 0.90 / 77.2 | — / 76.4 | — / 81.0 |
| NV-ref3s | **2.44** / 69.4 | **1.17** / 73.4 | 2.09 / 74.0 | **1.22** / 77.6 |
| SC-v2-il100 | 2.95 / 62.7 | 1.57 / **73.6** | 2.40 / 74.5 | 1.41 / 78.1 |
| SC-v3-lr5e-5 | 7.53 / 52.0 | 1.52 / 73.0 | 3.16 / 72.1 | 1.25 / 78.2 |
| SC-v3-lr2e-5 | 6.02 / 59.5 | 1.28 / 72.9 | **2.02** / 73.3 | 1.35 / **78.6** |
| SC-v4 | 6.31 / 58.7 | 1.58 / 73.0 | 2.19 / 72.1 | 1.46 / 78.0 |
| **SC-v5-hifi** | 2.69 / 69.5 | 1.58 / 72.7 | 2.49 / 75.3 | 1.33 / 78.3 |
| SC-v6-earsexp | 2.57 / **69.6** | 1.65 / 72.9 | 2.31 / **75.7** | 1.62 / 78.5 |
| SC-v6-earsexp-lr2e-5 | 2.23 / 69.4 | 1.24 / 72.8 | 1.87 / **76.0** | **1.13** / 78.4 |
| SC-v6-earsexp-lr1e-5 | **2.10** / 69.3 | **1.17** / 73.1 | **1.72** / 75.9 | 1.18 / **78.6** |
| SC-v5-hifi-lr2e-5 | 2.25 / 69.3 | 1.27 / 72.9 | 1.98 / 75.7 | 1.31 / 78.4 |
| SC-v5-hifi-lr1e-5 | 2.05 / 69.2 | 1.19 / 73.1 | 1.84 / 75.7 | 1.07 / 78.8 |

2026-11-17 08:20 UTC 快照：2e-5、1e-5 的 Seed-TTS 四组均完成全量生成和评分（英文每模式 1,088 条，中文 2,020 条）。1e-5 于 08:13:49 UTC 完成，四组错误率均低于 2e-5；SIM 基本持平。

### 3.3 CV3（WER/CER % ↓ / SIM ×100 ↑）

| 模型 | en speaker_only | zh speaker_only | hard SIM en / zh | en ICL | zh ICL |
| --- | --- | --- | --- | --- | --- |
| base | — / 76.0 | — / 79.2 | 76.6 / 78.0 | — / 75.6 | — / 79.7 |
| NV-ref3s | 4.88 / 72.6 | 4.03 / **76.1** | **74.1** / **75.4** | **5.01** / 73.6 | 3.84 / 77.5 |
| SC-v2-il100 | 5.39 / 67.9 | **3.72** / 76.0 | 69.5 / 73.9 | 5.37 / 73.4 | 4.20 / 77.8 |
| SC-v3-lr5e-5 | 8.22 / 62.4 | 4.43 / 75.6 | 64.1 / 74.1 | 7.15 / 72.5 | 4.14 / 77.8 |
| SC-v3-lr2e-5 | 7.12 / 66.4 | 3.92 / 75.2 | 68.9 / 73.4 | 6.59 / 73.2 | **3.83** / 77.6 |
| SC-v4 | 7.46 / 66.1 | 4.07 / 75.4 | 67.2 / 74.1 | 5.21 / 73.8 | 4.18 / 77.8 |
| **SC-v5-hifi** | 4.63 / 72.8 | 4.43 / 75.2 | 73.7 / 73.7 | 5.35 / **75.1** | 4.21 / **78.0** |
| SC-v6-earsexp | 4.51 / 72.9 | 4.47 / 75.1 | 73.0 / 73.9 | 4.94 / 74.8 | 4.08 / 77.7 |
| SC-v6-earsexp-lr2e-5 | **3.82 / 73.2** | 3.86 / 75.4 | 73.7 / 73.9 | 4.56 / 74.9 | **3.55** / 78.2 |
| SC-v6-earsexp-lr1e-5 | 3.88 / 72.8 | **3.56 / 75.5** | 74.1 / 73.6 | **4.35** / 74.8 | 3.60 / 77.9 |
| SC-v5-hifi-lr2e-5 | 4.80 / 72.9 | 3.97 / 75.2 | 73.5 / 73.4 | 4.67 / 74.9 | 3.68 / 77.9 |
| SC-v5-hifi-lr1e-5 | 4.16 / 72.9 | 3.74 / 75.4 | 73.7 / 73.1 | 4.31 / 74.6 | 3.69 / 78.2 |

CV3 hard 每个模型只有 55–64 条有效样本，几个点的差别不可靠。SC-v5-hifi-lr2e-5 的 CV3 全量评分已于 2026-11-17 07:55:47 UTC 完成；其 hard 英文 / 中文 speaker_only WER/CER 为 11.31 / 11.52，原版为 9.54 / 12.62；有效条目分别为 64 / 55 与 63 / 56，不能直接当成逐条配对比较。

### 3.4 NV-Bench（ICL，CER / PCER %，↓）

| 模型 | zh_single | zh_multi | en_single | en_multi |
| --- | --- | --- | --- | --- |
| NV-ref3s | 2.74 / 22.6 | 3.01 / **33.9** | — | 6.86 / **39.4** |
| SC-v2-il100 | 3.24 / **21.4** | 3.81 / 35.5 | **7.41 / 44.1** | 10.76 / 48.2 |
| SC-v3-lr5e-5 | 5.02 / 28.5 | 3.69 / 39.3 | 9.06 / 51.3 | 15.38 / 55.8 |
| SC-v3-lr2e-5 | 3.62 / 33.5 | 5.16 / 42.5 | 9.53 / 53.9 | 16.57 / 56.3 |
| SC-v4 | 3.76 / 30.9 | 4.09 / 43.7 | 9.04 / 52.4 | 14.96 / 54.0 |
| SC-v5-hifi | 3.40 / 24.6 | 3.98 / 36.9 | 8.12 / 47.6 | 13.73 / 50.7 |
| SC-v6-earsexp | 2.98 / 26.0 | 4.48 / 39.1 | 7.74 / 53.3 | 13.83 / 57.6 |
| SC-v6-earsexp-lr2e-5 | 3.45 / 24.5 | 3.59 / 36.5 | 7.98 / 55.0 | 14.03 / 56.0 |
| SC-v6-earsexp-lr1e-5 | 3.18 / 33.9 | 3.87 / 42.3 | 6.96 / 55.3 | 15.21 / 59.4 |
| SC-v5-hifi-lr2e-5 | 3.08 / 26.0 | 3.70 / 37.6 | 7.11 / 49.3 | 13.38 / 51.9 |
| SC-v5-hifi-lr1e-5 | 2.97 / 31.1 | 3.25 / 43.2 | 7.36 / 51.9 | 12.59 / 55.5 |

两组新增 v5 的四个分组均已完成。CER 比较文字字符，PCER 比较 `[Tag]` 副语言事件标签序列，OCER 同时计入二者；具体取舍见 §4.8。

### 3.5 EOS（验证 eos_loss）

| 模型 | step 200 | 最终 |
| --- | ---: | ---: |
| SC-v2-il100 | 0.070 | 0.056 |
| SC-v3-lr5e-5 | 0.193 | 0.172 |
| SC-v3-lr2e-5 | 0.201 | 0.174 |
| SC-v4 | 0.129 | 0.106 |

学习率不影响 EOS；换成 v4 数据后降约 40%，偏高主要来自被删子集（KeSpeech、VoxCeleb 等）。验证集随数据变化，只能参考。StyleBench 英文错误率超过 20% 的条目 SC-v2-il100 有 3 条、SC-v3-lr5e-5 有 26 条（提前截断 17、该停没停 15）。

### 3.6 外部参照：CosyVoice3 的 StyleBench 情感

模型 `Fun-CosyVoice3-0.5B-2512`，推理代码用 UltraEval 的 CosyVoice3 接入（`third_party/CosyVoice`，环境 `envs/cosyvoice_official`），本机 8 卡跑 StyleBench 情感子集（emotion_en / zh，easy / hard，各 1,920 条），打分、汇总与我们的模型完全相同。

- speaker_only：`inference_instruct2(正文, "You are a helpful assistant. <指令><|endofprompt|>", 参考)`，参考给说话人向量与 flow 条件，LM 看不到参考 token。
- ICL：`inference_zero_shot(正文, 同一前缀 + 参考文字, 参考)`。
- **指令用 StyleBench 原样的自由写法**。CosyVoice3 官方示例（`cosyvoice/utils/common.py` 的 `instruct_list`，代码中未被引用，仅作推荐写法）只列了开心 / 伤心 / 生气三种情感的固定句式，fearful、surprised、disgusted 不在其中，它对其他写法与情感的实际能力未知。

| ACC-I（speaker_only） | en easy | en hard | zh easy | zh hard |
| --- | ---: | ---: | ---: | ---: |
| CosyVoice3：全部情感 | 27.9 | 27.7 | 15.6 | 29.4 |
| SC-v5-hifi：全部情感 | **53.8** | **46.7** | **57.3** | **56.7** |
| CosyVoice3：开心 / 伤心 / 生气平均 | 45.0 | 39.6 | 18.1 | 30.5 |
| SC-v5-hifi：开心 / 伤心 / 生气平均 | **76.6** | **65.4** | **62.8** | **61.4** |

- **在这套 StyleBench 上情感控制我们明显更强**：即使只看 CosyVoice3 官方列出的三种情感，speaker_only 也高 25–45 个点；它的中文 easy 只有 15.6（中性参考下指令几乎带不动情感），fearful / disgusted 接近 0。
- **音色相似度 CosyVoice3 更高**（SIM en / zh 0.605 / 0.762 vs 0.556 / 0.617），WER / CER 也略低；它的指令模式把参考同时送进 flow 作条件，保留的音色信息多于我们只用 x-vector。
- ACC-R（hard）CosyVoice3 en 18.3 / zh 17.7，与我们相当（15.4 / 23.3）。
- **ICL 下两者都几乎没有风格控制**（CosyVoice3 ACC-I 0.4–10.8），不是我们独有的问题。
- 结果：UltraEval-Audio `res/cosyvoice3-latest-style_bench-emotion/style_bench/results.md`；生成脚本 `scripts/cosyvoice3_style_bench_infer.py`。环境按官方 requirements 只装推理依赖（去掉 tensorrt / deepspeed / gradio 等，经清华源），torch 2.3.1；onnxruntime 缺 CUDA 11 库，语音 tokenizer 走 CPU。

### 3.7 Vccm / ControlSpeech：SC-v5-hifi、CosyVoice3 与真人

使用 ControlSpeech 的 VccmDataset 与官方情感分类头、音高/能量函数。CV3 指 Fun-CosyVoice3-0.5B-2512。共覆盖原始 2,686 条中的 2,606 条：常规 1,432/1,500，域外风格 91/100，域外音色 1,083/1,086；缺失来源见 [prepare.json](../../../UltraEval-Audio/raw_data/vccm_v1/prepare.json)。全部可用样本评分完成、失败数为 0。

#### 常规

| 音频 | n | 情感 n | 情感 ACC % ↑ | 独立音高 ACC % ↑ | 能量 ACC % ↑ | 官方音高 ACC % ↑ | WER % ↓ | SIM ↑ |
|---|---|---|---|---|---|---|---|---|
| 真人 | 1432 | 133 | 82.71 | 88.97 | 29.26 | 26.26 | 2.44 | 未计算 |
| CV3 | 1432 | 133 | 54.14 | 76.19 | 24.72 | 17.60 | 1.92 | 0.8983 |
| SC-v5-hifi | 1432 | 133 | 40.60 | 66.48 | 37.50 | 24.93 | 6.58 | 0.9210 |

#### 域外风格

| 音频 | n | 情感 n | 情感 ACC % ↑ | 独立音高 ACC % ↑ | 能量 ACC % ↑ | 官方音高 ACC % ↑ | WER % ↓ | SIM ↑ |
|---|---|---|---|---|---|---|---|---|
| 真人 | 91 | 15 | 80.00 | 74.73 | 45.05 | 35.16 | 1.80 | 未计算 |
| CV3 | 91 | 15 | 33.33 | 63.74 | 40.66 | 25.27 | 1.80 | 0.8994 |
| SC-v5-hifi | 91 | 15 | 13.33 | 50.55 | 32.97 | 17.58 | 7.77 | 0.9114 |

#### 域外音色

| 音频 | n | 情感 n | 情感 ACC % ↑ | 独立音高 ACC % ↑ | 能量 ACC % ↑ | 官方音高 ACC % ↑ | WER % ↓ | SIM ↑ |
|---|---|---|---|---|---|---|---|---|
| 真人 | 1083 | 58 | 100.00 | 92.61 | 22.53 | 21.33 | 1.85 | 未计算 |
| CV3 | 1083 | 58 | 44.83 | 82.83 | 14.50 | 12.65 | 1.31 | 0.9149 |
| SC-v5-hifi | 1083 | 58 | 25.86 | 77.29 | 35.36 | 27.79 | 4.45 | 0.9345 |

情感准确率只计算有标签样本。官方音高指标要求能量也判断正确，独立音高准确率则单独统计。真人的能量准确率也低，说明绝对阈值与当前音频/标签的一致性需要检查；v5 超过真人的某个指标不能解释为听感超过真人。WER 用 Whisper-large-v3 与其英文归一化，SIM 为 WavLM base-plus-sv x-vector 余弦。语速准确率依赖 MFA，本次未计算。

原版 v5 音色相似度三组均更高；CV3 的情感、独立音高和 WER 三组均更好。能量与官方联合音高结果随分组变化。

- [dots-flexivoice-v5-hifitts2-interleave50-ep3-step723-default-instruction/vccm/speaker_only/summary.json](../../../UltraEval-Audio/res/dots-flexivoice-v5-hifitts2-interleave50-ep3-step723-default-instruction/vccm/speaker_only/summary.json)
- [cosyvoice3-latest-vccm/vccm/speaker_only/summary.json](../../../UltraEval-Audio/res/cosyvoice3-latest-vccm/vccm/speaker_only/summary.json)
- [真人基线](../../../UltraEval-Audio/raw_data/vccm_v1/reference/summary-reference.json)

## 4. 分析

### 4.1 英文 speaker_only 音色相似度

中文 SIM 在所有 FlexiVoice 模型上都与 NV-ref3s 持平（Seed 73–74，CV3 75–76），比 base 低的 3–4 个点来自 NV 微调本身。问题只在**英文 + 只靠 x-vector** 这个组合，ICL 基本不受影响。叠加了两层：

1. **SC-v3 特有的约 7–10 个点**（相对 SC-v2-il100）：SC-v2 与 SC-v3 的英文数据完全相同，差别是多了 32.6 万条指令 100% 描述音色的 KeSpeech（FlexiVoice 中占比约 49% → 59%）、interleave 100% → 50%、特征版本。降 lr（SC-v3-lr2e-5，+7.5）或删掉音色描述（SC-v4，+6.7）都能找回大半，说明主要是"高 lr + 大量音色描述指令"。
2. **英文缺少"音色只靠参考"的数据**：按语言统计 FlexiVoice（参考按训练规则抽，CAM++ 余弦）：

   | | 说话人 | 参考-目标相似度（中位） | <0.5 占比 |
   | --- | ---: | ---: | ---: |
   | v4 · 中文 | 4,377 | 0.701 | 6.3% |
   | v4 · 英文 | 156 | 0.656 | 9.6% |
   | v3 · 中文 | 31,540 | 0.666 | 5.4% |
   | v3 · 英文 | 904 | 0.635 | 16.3% |

   英文说话人少一到两个数量级、参考一致性更差；同一游戏同一角色，英文也比中文低 0.02–0.03（可能 CAM++ 对英文区分度较弱，未验证）。中文另有 emilia_nv（573 h，默认指令，句内参考相似度约 0.86），英文没有对应数据。**SC-v5-hifi 补 HiFiTTS-2（4,970 个英文说话人、默认指令、句内参考）后，英文 Seed SIM 58.7 → 69.5、CV3 66.1 → 72.8**，与 NV-ref3s 持平，证实了这一点。

其他观察：

- 跨句参考整体只有约 0.67（NV 句内参考约 0.86）。主要因素是句长（较短一句 <3 s 时 0.35–0.55，两句都 ≥6 s 时 0.65–0.86）和风格（同说话人换风格低 0.1–0.15；说话人 embedding 编码情感，见 Pappagari et al., ICASSP 2020）。与 WavLM 在 240 对上相关 0.91，不是 CAM++ 的问题。改用中性参考几乎没有改善（多数子集 +0~0.06），因为偏差主要来自情感化的目标句本身。
- 按子集的参考-目标相似度（中位）：debatts 0.84、l2 0.73、ears 0.71、WHSP 0.70、starrail 0.70、genshin 0.67、kespeech 0.67、voxceleb 0.61、childmandarin 0.61、ESD 0.54、expresso 0.47、IEMOCAP 0.45、CREMA-D 0.39。v4 删掉的子集并不比保留的更差，只有 expresso、IEMOCAP、CREMA-D 明显偏低。
- v4 数据中每句与本说话人 x-vector 中心（留一）的相似度中位 0.82；低于 0.5 的 1.9%，集中在游戏数据，尚未试听和过滤。

### 4.2 说话方式控制的丢失

语速、音高描述几乎只在 v4 删掉的子集里（指令关键词统计）：

| | 语速 | 音量 | 音高 | 音色 |
| --- | ---: | ---: | ---: | ---: |
| v4 保留（45.3 万条） | 2% | 4% | 0.3% | 14% |
| v4 删掉（120.3 万条） | 92% | 6% | 50% | 99% |
| 其中 kespeech（97.5 万条） | 99% | 2% | 49% | 100% |

英文的耳语 / 大声 / 快几乎全部来自 EARS、Expresso、VoxCeleb：

| 英文样本 | v3 | v4 | v3 主要来源 |
| --- | --- | --- | --- |
| 耳语 | 6,244 条 / 17.6 h | 260 条 / 0.5 h | EARS 2,850、Expresso 2,768、VoxCeleb 366 |
| 大声 | 34,130 条 / 78.8 h | 418 条 / 0.4 h | Expresso 13,909、VoxCeleb 13,693、EARS 6,110 |
| 快 | 37,254 条 / 66.5 h | 15,766 条 / 24.9 h | VoxCeleb 17,844、游戏 15,686（多为中文指令，"快"会误配"欢快"，实际更少） |

SC-v5-hifi 英文更差，另一原因是成功率相对默认指令计算，而英文默认指令的输出现在主要由 500 h 平稳朗读决定。

EARS、Expresso 当初被删，是因为 FlexiVoice 给它们的指令是 ParaSpeechCaps 式长描述，几乎 100% 把音色和风格写在一句里；Expresso 另有说话人少（4 人）、参考相似度低（0.47）的问题。但问题在**指令**而非音频：两者的风格是录音协议规定的，写在路径里（EARS `p042/emo_amazement_freeform.wav`，`rainbow_*` / `sentences_*` 为中性朗读；Expresso `…/ex04-ex03/fast/…`），可以按路径标签重新生成只写风格的指令（§8）。

### 4.3 ICL 下没有风格控制

所有模型 ICL 下 ACC-I 只有 3–15%，与 base 相同。训练中 ICL 从没和风格指令一起出现过。

### 4.4 中文耳语

只有带风格标签的情感语料（v4 中 3.7 万行，8%）按 0.4 中性 / 0.4 冲突 / 0.2 任意抽参考；原神、星穹铁道、debatts、WHSP（41.5 万行，92%）的参考是同一说话人随机另一句，常带情绪。**WHSP 10,949 行全是耳语，参考也全是耳语**，模型从没见过"中性参考 + 耳语指令"，所以所有模型中文耳语都只有 5–18%。

### 4.5 日语 / 韩语

韩语 CER 从 base 的 8.3 升到 51–68，ICL 下约 25% 的输出几乎为空（立即输出 EOS），SC-v5-hifi 有 3 条生成超过 30 s 不停；日语 7.6 → 10–16。日韩在训练里只有游戏数据。

### 4.6 情感排名反转：双向分类器交叉复测

对同一份已有生成音频换分类器，没有重新生成。StyleBench 选英文 speaker_only/easy：每模型 480 条；Vccm 选常规集有情感标签的 133 条。Plus-large 取英文六类情感与 neutral 的最大分，ControlSpeech 取八类最大分。

| 数据集 | 模型 | n | Plus-large ACC % | ControlSpeech ACC % |
|---|---|---|---|---|
| style_easy | v5 | 480 | 53.75 | 49.58 |
| style_easy | cv3 | 480 | 27.92 | 34.79 |
| vccm | v5 | 133 | 33.83 | 40.60 |
| vccm | cv3 | 133 | 40.60 | 54.14 |

两个分类器下，排名均随数据集反转。Vccm 六种非中性情感等权平均后，Plus-large 为 CV3 33.88% / v5 26.98%，ControlSpeech 为 CV3 51.17% / v5 38.12%；类别比例也不是唯一解释。

已检查两套 CV3 speaker_only 均调用 inference_instruct2，外层提示模板相同；v5 的模板和主要推理默认参数也相同。仍未拆开的因素包括：单一情感命令与复合属性描述、文本、语料来源、参考音频/说话人、训练数据覆盖。当前证据不能指定其中某项为主因。

下一步若要定位原因，应在同一文本、参考音频和随机种子下，只替换单一情感指令/复合指令，并补少量盲听；训练数据重合也需要独立核查。

- [逐条情感交叉评分](../../../UltraEval-Audio/artifacts/vccm-diagnostics-20261117/emotion-cross.jsonl)
- [情感交叉统计](../../../UltraEval-Audio/artifacts/vccm-diagnostics-20261117/emotion-summary.json)

### 4.7 Vccm WER 异常诊断

原版 v5 常规集为 1,585 次词编辑 / 24,070 个参考词 = 6.58496%。

1. 一条 7.68 秒音频 `test|856c5b731fd75c84` 在原始 Whisper 解码下产生大量 be/ever 重复，计入 430 次编辑；StyleBench 的 ASR pipeline 也复现了同一错误。对同一音频改为 num_beams=5，只剩 1 次编辑。这是该条主要受 ASR 解码异常影响的有力证据。
2. 仅排除该条进行敏感性分析，WER 从 6.58% 降到 4.80%；这是诊断结果，不替代官方分数。其他几个所选严重错误在束搜索下仍存在。
3. Whisper English normalizer 会删掉括号中的文本；常规集 7 条文本含括号/方括号。保留括号正文后，v5 WER 为 6.39%，CV3 为 1.75%，真人为 2.26%。影响较小，不能解释大部分差距。
4. 常规集 v5 音频最长 24.16 秒；全量也无超过 30 秒的 v5 音频，未发现 30 秒 ASR 窗口截断是原因。
5. 常规集按文本长度分组，v5 WER：不超过 10 词为 8.86%，11–30 词为 7.97%，超过 30 词为 3.50%。本次不能把高 WER 主要归因于长句。

上面是六条选例的解码复核，未对两模型全量统一改用束搜索。其余错误需独立听辨或第二个 ASR 验证，才能区分真实漏读/发音问题与识别器误差。正式分数仍使用原始设置。

- [六条 ASR 复核](../../../UltraEval-Audio/artifacts/vccm-diagnostics-20261117/wer-retranscribe.jsonl)
- [归一化敏感性统计](../../../UltraEval-Audio/artifacts/vccm-diagnostics-20261117/wer-normalization.json)：其中 all/CV3 为当时 2,380 条快照，只有 test 三方是完整配对。

### 4.8 SC-v5-hifi 学习率扫描（2026-11-17 09:20 UTC，两组全量完成）

三个模型使用同一 v5 build、3 epoch、step723；新增配置继承 lr5e-5 原版，只覆盖学习率与运行名称。两组训练和导出均完成，目前的评测结论是：

- **文字准确性改善**：NVBench 四组 CER 都低于原版。1e-5 在中文两组与英文 multi 更低，2e-5 在英文 single 更低；2e-5 的 Seed-TTS 四组错误率也均降低，speaker_only 英文 2.692 → 2.253、中文 1.582 → 1.274，SIM 变化很小。
- **副语言标签存在取舍**：NVBench 四组 PCER 均高于原版，且 1e-5 四组都比 2e-5 更差。不能仅凭 CER 改善认定整体风格控制更好。
- **2e-5 情感控制变弱**：StyleBench speaker_only 英文 easy / hard 从 53.8 / 46.7 降为 44.8 / 46.0；中文从 57.3 / 56.7 降为 54.6 / 49.4。easy / hard 每格 480 条，英文 hard 的小差异不宜过度解释。ICL 仍只有 4.2–13.3% 的 ACC-I。
- **2e-5 慢速控制下降**：英文成功率 59.5% → 47.6%，中文 80.0% → 47.5%；耳语仍为英文 4.8%、中文 5.0%，英文大声 0%、快 7.1%。未解决 v5 原有短板。
- **2e-5 CV3 并非全面改善**：普通英文 speaker_only WER 从 4.634% 升至 4.801%，中文 CER 从 4.429% 降至 3.970%；英文 / 中文 ICL 从 5.346 / 4.214 降至 4.673 / 3.680。SIM 变化小，DNSMOS 无一致提升。
- 自由描述有升有降：speaker_only 胜率中 / 英为 58.0 / 62.5（原版 55.0 / 57.0），日 / 韩为 54.0 / 55.0；单个语言仅 200 条，不能仅凭几个点认定可靠提升。
- **1e-5 全量结果**：NVBench、Seed-TTS、CV3、StyleBench 均已完成（CV3 状态 complete 于 08:48:50 UTC）。Seed 英文 / 中文 speaker_only 错误率 2.048 / 1.192%，ICL 为 1.836 / 1.073%，四组均低于 2e-5。CV3 普通英文 / 中文 speaker_only 为 4.164 / 3.743%，ICL 为 4.307 / 3.691%；除中文 ICL 与 2e-5 的 3.680% 接近外，其余普通分组均更低。SIM 变化小，DNSMOS 无一致提升。
- **1e-5 风格取舍**：speaker_only 情感 en easy / hard 为 47.3 / 41.7%，zh 为 49.4 / 47.7%，四项都低于原版；ICL ACC-I 为 2.7–14.4%。慢速成功率英文 45.2%、中文 30.0%（原版 59.5 / 80.0%）；耳语英文 16.7%、中文 5.0%，仍未普遍解决风格控制。相比 2e-5，1e-5 仅英文 easy 情感更高，其他三项情感更低。
- **当前判断**：普通朗读准确性优先可选 1e-5；情感与慢速控制优先仍保留原版 5e-5，两个低 lr 都不能全面替代。2e-5 保住的部分情感与慢速控制多于 1e-5，但朗读错误率多数不及 1e-5。没有多训练随机种子的显著性检验。
- Vccm 本轮只评了原版 lr5e-5，不能将 §3.7 的结果归到两个新学习率。

NVBench 的 OCER / SIM 补充（CER / PCER 已合入 §3.4；SIM 越高越好）：

| 模型 | zh_single | zh_multi | en_single | en_multi |
| --- | --- | --- | --- | --- |
| SC-v5-hifi | 3.90 / 0.7845 | 5.23 / 0.7950 | 8.40 / 0.7500 | 14.17 / 0.7558 |
| SC-v5-hifi-lr2e-5 | 3.63 / 0.7840 | 4.94 / 0.7934 | 7.40 / 0.7485 | 13.83 / 0.7665 |
| SC-v5-hifi-lr1e-5 | 3.65 / 0.7866 | 4.80 / 0.7965 | 7.74 / 0.7475 | 13.10 / 0.7619 |

历史状态（2026-11-17 07:27 UTC）：四个 ACP 评测作业当时均为 RUNNING：2e-5 StyleBench `pt-0bwbc2t8`、suite `pt-eh7rqls9`；1e-5 StyleBench `pt-q6krwuoy`、suite `pt-48zumz01`。

来源：UltraEval-Audio `res/dots-flexivoice-v5-hifitts2-lr{2e-5,1e-5}-interleave50-ep3-step723-default-instruction/{nv_bench,seed_tts/full,cv3/full,style_bench}/`；[作业状态快照](../../../UltraEval-Audio/artifacts/vccm-diagnostics-20261117/v5-lr-job-status.json)。

2e-5 已完成报告：[StyleBench](../../../UltraEval-Audio/res/dots-flexivoice-v5-hifitts2-lr2e-5-interleave50-ep3-step723-default-instruction/style_bench/results.md)、[CV3](../../../UltraEval-Audio/res/dots-flexivoice-v5-hifitts2-lr2e-5-interleave50-ep3-step723-default-instruction/cv3/full/results_table.md)。

## 5. 数据

### 5.1 流程

FlexiVoice-Instruct（[arXiv 2601.04656](https://arxiv.org/abs/2601.04656)）15 个子集接入为 `DATA-TTS-UNIFIED/datasets/flexivoice/v0.1`，4,199,042 行、约 9,752 h（tts-data-pipeline `docs/flexivoice.md`）。`metadata_json` 含 `instruction` 和上游记录；base 的 `speaker_id` 和多数 `language` 为空，说话人键从上游路径解析（`lm_tts/data/flexivoice.py: speaker_key`）。

text revision → selection（子集、时长、采样率、留出、配对）→ dots_vae features（只含入选行）→ training build（只校验不变量）。所有发布登记在 tts-data-pipeline `docs/data-contract/PUBLICATIONS.md`。

文本修订 `tts-ann-flexivoice_text-20261110T091454bjt-01`：移除 `[inaudible]`，`[LAUGHTER]` / `[BREATHING]` 映射为 NV 标签，13 行含 `[GARBAGE]` 记为不可用。

### 5.2 Selection 版本

| 版本 | selection ID | 入选 | 变化 | 用于 |
| --- | --- | ---: | --- | --- |
| v1 | `…20261110T092455bjt-01` | 1,656,126 行 / 2,742 h | 11 个子集、1–30 s、≥16 kHz、66 个留出说话人键、闭合跨句参考池（≥3 s） | 早期实验 |
| v2 | `…20261111T162703bjt-01` | 1,330,267 行 / 2,214 h | v1 − 326,006 条内嵌 CAF 头的 KeSpeech | SC-v2-il100 |
| v3 | `…20261112T112338bjt-01` | 1,655,877 行 / 2,742 h | v1 规则，CAF 在提取时修复而不排除 | SC-v3-* |
| v4 | `…20261113T095517bjt-01` | 452,817 行 / 761 h | 只保留 WHSP、debatts、ESD / RAVDESS / EMNS、原神、星穹铁道 | SC-v4、SC-v5-hifi |
| HiFiTTS-2 | `…20261113T143403bjt-01` | 171,761 行 / 500 h | 每说话人抽样名单（§5.5） | SC-v5-hifi、SC-v6-earsexp |
| v6 | `…20261116T143603bjt-01` | 506,331 行 / 916 h | v4 + ears、expresso，二者指令由 instruction revision 替换（§5.3） | SC-v6-earsexp |

11 个子集：kespeech、voxceleb、childmandarin、l2、ears、expresso、emotion（ESD、RAVDESS、EMNS、CREMA-D、IEMOCAP）、debatts、WHSP、原神、星穹铁道。留出说话人（四种语言一起留出）：原神 8 个角色（早柚、莫娜、莱欧斯利、流浪者、珐露珊、可莉、克洛琳德、行秋）、星穹铁道 6 个（玲可、真理医生、银枝、帕姆、阮•梅、驭空）、EARS 7 人、RAVDESS 3 人。CSEMOTIONS 只作测试。

### 5.3 FlexiVoice 指令的来源与 v4 的筛选

FlexiVoice 的指令**全部由 LLM 从元数据生成，没有对照音频**（FlexiVoice 论文）：

- 游戏：DeepSeek-V3 根据台词和角色名推断"这个角色会怎么说"，与实际配音可能不符，并带角色人设的固定倾向。
- KeSpeech：年龄 / 性别 / 方言三元组写成指令（一半中文、一半英文；音频全是中文），属性前缀几乎只有"慢速"和高 / 低音调。
- ChildMandarin、L2-ARCTIC：按说话人标签写模板。
- EARS、Expresso、VoxCeleb：来自 ParaSpeechCaps 描述（原标签分 intrinsic 与 situational，FlexiVoice 中混在一起）。
- debatts 等单标签数据：预生成模板随机填充。

指令类型（关键词启发式，每子集抽 3,000 条）：

| 子集 | 行数（v3） | 只音色 | 只风格 | 两者都有 | 示例 |
| --- | ---: | ---: | ---: | ---: | --- |
| kespeech | 975,483 | 1% | 0% | 99% | "慢速语速。请使用中年女性的声音，带有冀鲁官话的发音特点。" |
| childmandarin | 39,861 | 99% | 0% | 1% | "用小男孩的声音说出这句话" |
| l2 | 26,826 | 55% | 1% | 43% | "我需要一位带西班牙口音的女声，语气活泼一点" |
| voxceleb | 99,163 | 5% | 0% | 95% | "A husky male… deep booming voice in a noisy environment…" |
| ears / expresso | 53,616 | ≈0 | 0% | ≈100% | ParaSpeechCaps 长描述（性别、口音、音高 + 风格） |
| emotion | 45,552 | 0% | 97% | 0% | "Speak it using the emotion of Sarcastic." |
| debatts | 19,732 | 0% | 55% | 0% | "请以辩论中的坚定语气朗读以下内容。" |
| WHSP | 10,949 | 15% | 72% | 0% | "把这句话用气声说出来" |
| 原神 / 星穹铁道 | 384,695 | 2–3% | 77–80% | 12–15% | "直白又理直气壮的讨要语气，带着点玩世不恭的痞气。" |

即 v3 中约 69% 的行在用指令描述音色。**v4** 删去 kespeech、childmandarin、l2、voxceleb、ears、expresso 及 emotion 中的 CREMA-D、IEMOCAP（selection 新增 `flexivoice.emotion_corpora` 白名单，不在名单的记 3004）：

| v4 子集 | 行数 | 时长 |
| --- | ---: | ---: |
| 原神 | 289,242 | 467 h |
| 星穹铁道 | 95,453 | 182 h |
| debatts | 19,732 | 62 h |
| ESD | 35,000 | 29 h |
| WHSP | 10,949 | 18 h |
| EMNS + RAVDESS | 2,441 | 3 h |
| **FlexiVoice 小计** | **452,817** | **761 h** |
| NV 五个数据集 | 230,083 | 731 h |

FlexiVoice 部分 85% 是游戏（中 145 h、英 167 h、日 193 h、韩 144 h）。按语言（无标签的按文字判断）：中文 955 h / 4,385+ 人（emilia_nv 570 h 等）；英文 202 h / 170 人（游戏 169 h、情感语料 16 h）；日 / 韩 195 / 142 h，全是游戏。

**v6（SC-v6-earsexp）**：在 v4 上加回 EARS、Expresso，但丢掉 FlexiVoice 指令，改用 annotation `tts-ann-flexivoice_instruction-20261116T142927bjt-01`：

- 录音协议风格取自上游路径（EARS `p042/sentences_02_whisper.wav`、`emo_<emotion>_*`、`regular` 为中性；Expresso `…/read/ex01/happy/…` 或对话目录名）。试改发现原 ParaSpeechCaps 描述常与协议矛盾（`highpitch` 文件写成"Speak quickly"，`lowpitch` 写成"high-pitched"，Expresso 同一说话人所有风格都写"loud"），所以不能只删音色词。
- Qwen3.8-27B（sglang，TP4）对每个（协议风格，原描述）组合写一条指令：必须表达协议风格，只保留与之相容的语气细节，不写语速 / 音量 / 音高（除非协议风格就是它们）、性别、口音、音色、录音环境。共 35,923 个组合、56,456 行；仍含说话人 / 录音描述词的 103 行记 failed，selection 以 3005 排除。
- selection-flexivoice-v6 以 v4 为父，`instruction_inputs` 固定这份 annotation，分支列 `selected_instruction` 承载改写后的指令（其余子集为 null，沿用原指令），训练 `flexivoice_rows` 读取该列。入选 506,331 行 / 916 h；特征由 v3 复用。

### 5.4 开头咔声与特征版本

48k step400 试听发现"中文·长句·耳语"在约 0.25 s 有强瞬态。两个根因：

1. **KeSpeech 内嵌 CAF 头**：326,006 条 WAV 的 data 块以 4096 字节的 Apple CAF 文件头开头，被解码成约 128 ms 的满幅垃圾采样（`scripts/audit_embedded_caf.py`，只在 kespeech 命中）。
2. **250 ms 边缘静音归一化**：预处理把首个超过（峰值 −30 dB）的采样前补零 / 裁剪到恰好 250 ms，CAF 垃圾或从第一个采样就发声的录音（FlexiVoice 4.45%、NV 0.1–9.8%）因此在 0.25 s 处从零阶跃到满幅，且所有起音被对齐到 0.25 s。

处理：waveform 策略 `raw-caf-repair-v1`（profile v3 / v3-ref3s）解码前识别并跳过 CAF 头（`start_frame` 记录跳过帧数），不做边缘归一化，只补齐到 7680 的整数倍。

| profile | FlexiVoice | NV | 其他 |
| --- | --- | --- | --- |
| v2-ref3s（250 ms 归一化） | 全量 `…20261109T160744`；v1 `…20261110T093409`；v2 `…20261111T164240` | `…20261106T092453` | — |
| v3-ref3s（raw-caf-repair-v1，profile_id `a6d56e2c…`） | v3 `…20261112T121538`；v4 `…20261113T100902`（由 v3 复用） | `…20261112T121520` | HiFiTTS-2 `…20261113T144202`（ACP `pt-snz9ubvd`） |

### 5.5 HiFiTTS-2

有声书朗读，4,970 个说话人。`scripts/sample_per_speaker.py` 生成固定抽样名单：每个 speaker_id 在 3–20 s 的句子里按 `sha256('hifitts2-speaker40-v1:' + sample_id)` 取前 40 句；selection `selection-hifitts2-speaker40-v1` 用 `sample_evidence` 固定名单哈希，名单外记原因码 **8002 not_sampled**（字典 version 9）。入选 171,761 行 / 500 h，训练中用默认指令、句内 ref3s 参考。注意 HiFiTTS-2 是 22.05 kHz（有效频宽约 11 kHz）。

### 5.6 游戏文本核查

原神 / 星穹铁道文本是官方台本，格式正常。与 genshin_voice / starrail_voice 的 ASR 转写对齐 295,534 行（77%）：英 / 中 / 韩 CER 中位 0.00–0.02（差异 >30% 占 1–3%），日 0.07（4–6%）。差异多为 ASR 错误（专有名词、丢笑声、日文写法），**保留官方文本**；少数是舞台说明（如 `*panting*`），后续可按差异 >30% 或星号排除。

## 6. 训练设置

**序列**：指令前缀在正文前，不参与交错。

```text
interleave：[流式语音合成]Instruction: {instruction}\nText:\n 正文 token / 音频 span 交错 …
basic：     [文本]Instruction: {instruction}\nText:\n{正文}[文本对应语音]<start> S*N <end><eos>
```

- 没有指令的数据（NV、HiFiTTS-2）用默认指令 `Speak the following text.`；前缀以换行结尾，保证训练与推理 token 一致。
- interleave 与 basic 按 target_id 哈希固定选择（`interleave_fraction`）；推理模板与训练 `Layout.sequence` 逐 token 核对过。

**参考**：

| 数据 | 条件 x-vector |
| --- | --- |
| FlexiVoice | 同一说话人另一句（≥3 s、不同文本）的整句 x-vector（CAM++，取前 10 s）；有风格标签的目标按 0.4 中性 / 0.4 冲突 / 0.2 任意抽，每次访问重抽；目标整句计 loss |
| NV、HiFiTTS-2 | 同句内 ref3s 参考段（≥ min(3.04 s, 半句)）的 x-vector，参考区间 mask |

说话人键均为上游官方 ID；游戏为"语言 × 角色"（每种语言一个配音演员）；IEMOCAP 为 session + 实际说话人；Expresso 对话按声道；debatts 为场次 + 自动分离编号。

**共同配置**（`configs/dots-tts-2026111*-flexivoice-*`）：

| 项目 | 设置 |
| --- | --- |
| 起点 | `dots.tts-soar`，nv26 组装（`assets/assembled/dots-tts-soar-nv26`） |
| 冻结 | DiT 及条件投影：`velocity_field_predictor, hidden_proj, latent_proj, coordinate_proj, xvec_proj` |
| 训练 | LLM、PatchEncoder、EOS head |
| DiT 执行 | packed（按句拼接 + FlexAttention block mask；`lm_tts/models/dots/packed.py`，与原实现输出和梯度一致），全部 activation checkpointing |
| 采样 | `feistel_uniform_token_budget_v3`（全局打乱）；每卡 48,000 tokens / 30,000 patches |
| 调度 | WSD：warmup 2%，最后 20% 降到 0.1 倍；save / eval 每 200 步 |
| dropout | 指令 0、x-vector 0、CFG 0.2 |
| 资源 | ACP 单节点 8×A800，bf16 |

## 7. 评测方法

### 7.1 StyleBench

FlexiVoice 的测试集未公开，按论文 TR-easy / TR-hard 思路用**训练中没出现过的数据**自建（UltraEval-Audio `scripts/prepare_style_bench.py`，`raw_data/style_bench_v1/`，3,187 个 wav）。全部子集分 speaker_only 和 ICL 各跑一遍，`speaker_scale=1.0`（与训练一致）。

| 子集 | 数据 | 构造 | 指标 |
| --- | --- | --- | --- |
| emotion_en | 留出的 EARS 7 人 + RAVDESS 3 人，6 类，480 条 | 正文为中性句；参考为同说话人另一句：easy 中性、hard 冲突情感 | ACC-I、ACC-R、P-I、SIM、WER |
| emotion_zh | CSEMOTIONS 10 人，4 类（去掉 fearful：真实录音只认出 4%），480 条 | 同上 | 同上 |
| paraling_en / zh | EARS / CSEMOTIONS 中性句，耳语、大声、快、慢、高、低，492 条 | 同参考同正文，分别生成风格指令版和默认指令版 | 相对默认版的成功率 |
| free_zh/en/ja/ko | 留出游戏角色，每语言 200 条 | 真实指令和正文，真实录音作锚点，参考为同角色另一句 | 胜率、F0 / 语速 / 响度距离 |

生成：模板 `[流式语音合成]Instruction: {instruction}\nText:\n{interleave}`，10 步、CFG 1.2，种子 = sha256(条目 key) 十六进制前 8 位 mod 2³¹（可复现）。打分 `style_bench_score.py`，汇总 `style_bench_summary.py`。

| 逐条量 | 计算 |
| --- | --- |
| 情感 | emotion2vec_plus_large（utterance 级）9 类概率 + 1024 维 embedding；预测情感只在"测试情感 + neutral"里取最大 |
| SIM | WavLM-large 说话人验证（与 Seed-TTS 相同），生成音频与它所用参考的余弦 |
| 识别 | 中文 Paraformer（seaco large），英 / 日 / 韩 Whisper-large-v3（指定语言；>30 s 用 long-form） |
| 文本归一化 | NFKC、小写、繁转简、标点符号换空格；英文按词、中日韩按字 |
| 声学 | 16 kHz，`librosa.effects.split(top_db=40)` 取有声段；响度 = RMS dBFS；F0 = pyin（60–600 Hz，帧 1024，跳 160）浊音帧中位数；浊音比例；语速 = 正文单位 / 有声秒数 |

| 汇总指标 | 计算 |
| --- | --- |
| ACC-I | 预测情感 = 指令情感的比例 |
| ACC-R | 只在 hard 上：预测情感 = 参考情感的比例（越低越好） |
| P-I | 指令情感在候选类里重新归一化后的平均概率 |
| 真实录音上限 | 同一分类器对真人录音（`gt_emotion_*`）的准确率：英文 76%、中文 51% |
| WER / CER | Σ编辑数 / Σ参考长度 ×100 |
| 说话方式成功率 | 风格版对比默认版：大声 Δ响度 ≥ +3 dB；耳语 Δ浊音比例 ≤ −0.20；快 语速比 ≥ 1.15；慢 ≤ 0.87；高 F0 比 ≥ 1.10；低 ≤ 0.91 |
| E-SIM / 胜率 | 与真人录音的 emotion2vec embedding 余弦；胜率 = 指令版 E-SIM > 默认版的比例 |
| F0 / 语速 / 响度距离 | \|log2(f0 比)\|、\|log2(语速比)\|、\|Δ响度\| dB，与真人录音比 |

真人录音单独打分一次（`--reference`），结果在 `raw_data/style_bench_v1/reference/`，各模型共用。

局限：

- 默认指令训练时只出现在 NV / HiFiTTS-2 上，"相对默认指令"的对照带有这些数据的风格。
- EARS、RAVDESS 的句子在其他说话人的训练数据里出现过，英文 WER 偏乐观。
- 每格 40–80 条，10 个点以内的差异不可靠；base `--plain` 只是噪声底。
- 自由描述：真实录音只是一种演绎；emotion2vec 只抓情绪；参考本身带语气。胜率在 50% 附近不能证明指令无效，更可靠的是音频大模型裁判或人工试听。
- CSEMOTIONS 与训练集零重叠（音频哈希、归一化文本已核对）；同说话人 CAM++ / WavLM 相似度：中性-中性 0.79 / 0.80，中性-情感 0.65–0.75，不同人 0.30 / 0.34。

### 7.2 回归 suite

`scripts/run_dots_suite.py --instruction "Speak the following text." --gpu-memory-gib 16`：NV-Bench（ICL）、Seed-TTS、CV3（speaker_only 与 ICL），interleave 模板 + 默认指令，`speaker_scale` 用 runtime 默认 1.5（与以往 NV 实验一致）。

### 7.3 运行注意事项

- 本机 GPU 0–4 常被其他用户占用，评测提交到 ACP（UltraEval-Audio `scripts/acp/submit.py`，StyleBench 与 suite 各一个任务）；都支持断点续跑。
- **ACP 节点无外网**：emotion2vec 目录带 `requirements.txt`，funasr 加载时会 `pip install` 并永久挂起；`style_bench_score.py` 已跳过。
- **跨机器续跑 Seed-TTS / CV3**：样本按 `inference_shards`（GPU 数 × 每卡进程数）固定分片并写入 run identity；本机 5 卡（20 片）开始、ACP 8 卡续跑需加 `--inference-shards 20`。
- StyleBench 必须用 `--speaker-scale 1.0` 的结果；SC-v2-il100 以 `…-step794-stylebench-v1` 为准，`…-default-instruction` 下的 style_bench 是早期未固定 scale 的结果。

### 7.4 Vccm / ControlSpeech

数据准备 `scripts/prepare_vccm_bench.py`，清单 `raw_data/vccm_v1/`；原始 ControlSpeech CSV 的文本、风格指令与标签沿用，音频缺失导致只覆盖 2,606 / 2,686 条。固定选同说话人的另一句参考，情感语料优先中性参考；LibriTTS 参考长 3–12 秒且文本不同。与 StyleBench 的训练留出机制不同，Vccm 与训练数据的交集尚未核查。

生成：`scripts/vccm_infer.py` 与 `scripts/cosyvoice3_vccm_infer.py`；v5 使用 interleave 指令模板、10 步、CFG 1.2、speaker_scale 1.0；CV3 使用 inference_instruct2 与同一外层指令模板。两者均为 speaker_only。

评分 `scripts/vccm_score.py`：

- 情感：emotion2vec_base v2.0.4 帧特征 → ControlSpeech 原始微调八分类头（step180000），argmax 与目标比较；仅统计有情感标签的行。
- 音高 / 能量：直接导入 vendored `energy_pitch_acc.py` 的 get_pitch_energy / judge_pitch / judge_energy；旧版 librosa 位置参数通过适配层映射为关键字，阈值与计算公式不变。
- WER：Whisper-large-v3，指定英文，Whisper English normalizer，Σ词编辑 / Σ参考词；GPU float16 与原脚本 CPU 运行并非逐位一致。归一化和解码异常见 §4.7。
- SIM：microsoft/wavlm-base-plus-sv 的归一化 x-vector 余弦；与 StyleBench / Seed 的 WavLM-large 口径不同，不能跨测试直接比较绝对数值。真人 SIM 本轮未算。
- 语速准确率需要 MFA 对齐，本轮未接入。失败条数单独报告；本轮两模型与真人均为 0。

评分输出：各模型 `vccm/speaker_only/scores-*.jsonl`、`summary.json`；真人为 `raw_data/vccm_v1/reference/`。交叉诊断证据保存在 [artifacts/vccm-diagnostics-20261117](../../../UltraEval-Audio/artifacts/vccm-diagnostics-20261117/)。诊断实验没有改写原始 benchmark 分数。

## 8. 下一步

1. **下一版候选**：v5 数据 + 重新标注的风格数据；lr 扫描已完成，1e-5 偏普通朗读准确性、5e-5 偏情感与慢速控制，后续按目标选择（§4.8）。
2. **EARS / Expresso**：已做，见 SC-v6-earsexp（annotation `flexivoice_instruction`：Qwen3.8-27B 按录音协议风格 + 原描述重写，只写风格；selection-flexivoice-v6）。结果：英文耳语、慢、高补回，大声反向、快无效（§3.1），需查 EARS 大声 / 快样本在训练特征中的实际响度与语速，以及指令是否对上。v6 的 lr 2e-5 / 1e-5 已完成（§3.1），大声 / 快仍失效；v5 的 lr 扫描训练已完成，2e-5 的 NV / Seed / CV3 / StyleBench 全部已出结果，1e-5 的 NV / Seed / CV3 / StyleBench 也全部完成（§4.8）。
3. **重新标注**（对着音频，而不是从元数据推断）：kespeech、voxceleb、游戏等。
   - 语速、音高、音量用声学测量，**相对说话人自己的基线**分档（不按全数据集分档，否则会把音色标成风格）。
   - 情感用 emotion2vec + 音频大模型交叉判断；语气用音频大模型描述，提示词禁止写性别、年龄、口音、嗓音。
   - 指令只写风格；kespeech 按说话人抽样到约 300 h；先做说话人一致性检查（中性句两两相似度、聚类拆分、离群句）。
   - 每个来源抽约 200 条人工试听，验证标签。
4. **参考构造**：抽参考时加相似度门槛（如 ≥0.6，优先长句），或部分样本用说话人平均 x-vector（句子与说话人中心相似度约 0.82，高于单句参考的 0.67，且情感被平均掉）。
5. **中文耳语**：WHSP 需要同说话人的正常发音作参考；游戏、debatts 的参考可按 emotion2vec 中性概率优先选。
6. **未分清的因素**：50% vs 100% interleave 对英文 SIM 和 NV 标签的影响（`pt-0xu9i6xh` 中途停掉）。
7. **ICL + 风格**：需要"ICL 参考 + 不同风格指令"的训练样本。
8. **Vccm 排名反转与 WER**：按 §4.6 的同输入指令消融拆分输入因素；ASR 解码策略若要调整，应对两模型统一全量重评，并另报诊断口径。
9. **游戏数据**：排除文本与音频差异 >30% 的行及舞台说明；低音色相似度样本先试听再定。

公开的风格数据（都需要拆出风格部分才能用）：ParaSpeechCaps（situational 标签与 intrinsic 分开标，人工 282 h + 自动 2,427 h，最干净）、SpeechCraft（约 2,000 h，中英，描述混有音色）、LibriTTS-P（语速 / 响度可用，音高按全数据集分档）、CapSpeech（主要是自动标注）。本地都没有。

## 9. 复现与路径

```bash
# selection（tts-data-pipeline）
python scripts/build_selection.py plan --work /tmp/selection-flexivoice-v4 --rules configs/selections/selection-flexivoice-v4.json
python scripts/build_selection.py run --work /tmp/selection-flexivoice-v4 --rules configs/selections/selection-flexivoice-v4.json --workers 16
# 特征：从 v3 复用（不需要 GPU）
PYTHONPATH=src python -m tts_data_pipeline.dots_vae.reselect --root /workspace/data/DATA-TTS-UNIFIED \
  --selection <v4 selection manifest> --features <v3 FlexiVoice features manifest> --work <new work dir>
# build + 提交（LM-TTS-Training）
PYTHONPATH=$PWD python scripts/build_unified.py --config configs/dots-tts-20261113-flexivoice-v4-interleave50-feistel-48k-packeddit-lr5e-5-ep2.yaml
source scripts/acp/env.sh && PYTHONPATH=$PWD python -m scripts.acp.submit --config <config> --submit
# 导出
PYTHONPATH=$PWD:<dots.tts src> python scripts/export_checkpoint.py --checkpoint <run>/checkpoints/<step> \
  --assembled-model assets/assembled/dots-tts-soar-nv26 --output <UltraEval-Audio>/init_model/<name> --precision bf16
# 评测（UltraEval-Audio）
python scripts/run_style_bench.py --model-path init_model/<export> --run-dir res/<run> --bench-dir raw_data/style_bench_v1 --speaker-scale 1.0
python scripts/run_dots_suite.py --model-path init_model/<export> --run-dir res/<run> --gpu-memory-gib 16 --instruction "Speak the following text."
```

| 内容 | 位置 |
| --- | --- |
| 统一数据与发布登记 | `/workspace/data/DATA-TTS-UNIFIED`；tts-data-pipeline `docs/data-contract/PUBLICATIONS.md` |
| selection 规则 | tts-data-pipeline `configs/selections/selection-flexivoice-v{1,2,3,4}.json`、`selection-hifitts2-speaker40-v1.json` |
| 数据配置 | `configs/data-flexivoice-{nocaf,raw,v4}-nv-dots-ref3s.yaml`、`data-flexivoice-v5-hifitts2-nv-dots-ref3s.yaml` |
| 训练配置 | `configs/dots-tts-2026111*-flexivoice-*.yaml` |
| 评测代码 | UltraEval-Audio `scripts/{prepare_style_bench,style_bench_infer,style_bench_score,style_bench_summary,run_style_bench,run_dots_suite}.py` |
| Vccm 评测代码 | UltraEval-Audio `scripts/{prepare_vccm_bench,vccm_infer,cosyvoice3_vccm_infer,vccm_score,vccm_finish}.py` |
| 交叉诊断证据 | [UltraEval-Audio/artifacts/vccm-diagnostics-20261117](../../../UltraEval-Audio/artifacts/vccm-diagnostics-20261117/) |
| 设计文档 / 副语言实验 | `docs/design/dots-instruct-tts-20261108.md`；`docs/training/dots-tts-emilia-nv-20261103.md` |

最终新增报告：SC-v5-hifi-lr1e-5 的 [StyleBench](../../../UltraEval-Audio/res/dots-flexivoice-v5-hifitts2-lr1e-5-interleave50-ep3-step723-default-instruction/style_bench/results.md)、[CV3](../../../UltraEval-Audio/res/dots-flexivoice-v5-hifitts2-lr1e-5-interleave50-ep3-step723-default-instruction/cv3/full/results_table.md)。
