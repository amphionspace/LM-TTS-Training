# Emilia2：先建立官方结构基线，再做初始化消融

## 当前实验决定

目标是 **约 1,000 小时**，不是 20 小时。第一轮选英语，便于沿用当前英语 ASR 评估；范围为 2–10 秒、JSON **顶层 `type=short`** 的独立录音。不从 long/dialogue 的嵌套 short 视图取数据。

规模按筛选出来的原始 short 时长统计；预处理还会去除空文本、超过 256 token 的文本和缺少同说话人参考的 singleton，再划分验证集，因此实际训练条数／小时必须以 `preparation.json` 为准，不能把目标时长当作完成时长。该子集按确定性 shard 顺序选取，是工程基线，不声称是全库随机代表性样本。

模型详见 [当前基线结构](../design/final-architecture.md)。先固定以下初始化：

- 公开 Qwen3-TTS-0.6B 的 2048 维 text embedding + 配套 projector，成套加载，**全部冻结**。
- Qwen3-0.6B-Base 的 Talker layers / norm，继续训练。
- 新初始化音频 embedding、首 head、Code Predictor，继续训练。
- 公开 Qwen ECAPA，冻结；公开 codec encoder/decoder，冻结。

冻结只是保护已加载的前端，不证明它和 text Base 主干原本匹配。先在更充分的数据上测量这条基线，再决定是否修改结构。

## 数据存放位置

所有 Emilia 原始清单、prepared manifest、codec NPZ 和参考 WAV 存放在 `/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_prepared/LM-TTS-Training/`。原始 tar 仍只读使用 `/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_m4a/`。项目目录保存代码、配置、checkpoint、TensorBoard 和运行日志，不再存放 Emilia 数据。2026-09-09 已搬迁旧缓存并核对原始清单哈希；运行目录中的 `data-relocation.json` 保留搬迁记录。历史 pilot 清单因绝对路径变更而改变哈希，不能直接按原签名续训；正式 1,000 小时任务此前尚无 checkpoint。

## 中间层

```text
源数据适配器（Emilia / LJSpeech）
    ↓
统一 raw JSONL：id、text、speaker、language、duration、audio locator、source
    ↓
解码 / 划分 / 同说话人参考 / 冻结 codec 编码
    ↓
训练 JSONL + codec NPZ + 少量参考 WAV
    ↓
现有 CodeDataset → FSDP2 trainer
```

Emilia 使用 `.tar.idx` 的 member/offset/size 随机读取；不解包整个 tar。8 个 shard 并行读取，返回顺序固定。最终仍以 JSON 的 `type` 过滤，不凭文件名判断。

raw `audio` 是 locator；prepared `audio_source` 保留它，`audio` 是按需落盘位置。m4a 通过 PyAV 使用 FFmpeg 的 MP4 demuxer，处理 edit list；先按原始 44.1 kHz 解码，按 `frames` 裁掉 AAC 尾部 padding，再重采样为 24 kHz。允许最多 1,023 点的 AAC 尾部 padding；对于不超过 1 毫秒的解码尾部缺口，补零到标注长度并在日志记录样本 ID 和采样点数。更大偏差仍失败，不拉伸音频。不使用忽略编码延迟的解码方式。

每 speaker 训练池至少保留两条录音；选两个 anchor，并记录 `speaker_reference_id`。同一条录音不能引用自己，验证不能引用验证池。普通目标音频解码后以 float32 波形直接批量送入 codec，不写临时 WAV；参考 WAV 保留；评估原音频按需重建。codec 批量编码，CPU 解码线程与 GPU 编码配合。CPU 解码使用 8 个 spawn 进程，避免 PyAV/Python 解码被同一进程的 GIL 限制；不从已经初始化 CUDA 的进程 fork。解码进程数与 GPU 数量属于执行参数，不改变缓存 recipe。两个 GPU 各有独立 codec 实例，处理互不重叠的 batch，按原始顺序收集结果；每设备最多一个在途 batch，避免无限预取占内存。GPU 数量不改变清单划分或缓存 recipe；batch size 仍是 recipe 的一部分。

验证文本经规范化后与训练文本不重合；这是同说话人未见文本评估，不是未见说话人的 zero-shot 评估。

预处理 recipe v3 记录 raw manifest、codec 配置／权重、tokenizer 的 SHA256；原始 tar 按 locator 读取，未对整个语料 tar 做内容哈希，要求源文件保持不变。预处理可在相同 recipe 下重跑，已原子完成的 NPZ 会复用；recipe 不一致拒绝覆盖。唯一兼容升级是 v2 → v3 的尾部长度策略：v2 已接受的音频处理结果完全不变，新接受的短尾样本此前未生成过缓存，保留原 recipe 作为审计记录。最终 train/val manifest 原子写入，完成后生成 `PREPARATION_COMPLETE`。不同 codec、筛选、参考方案必须用不同输出目录。

目前训练 JSONL / NPZ 仍是本机验证格式。1,000 小时级别需要关注 inode、索引常驻内存和 sampler 时间；几十万小时正式训练应进一步把 codec 和参考特征写入分片，使用可恢复的数据流读取。这里不宣称当前百万文件实现已满足整个正式语料规模。

## 复现入口

```bash
# 在项目根目录，使用独立 venv。
source .venv/bin/activate
python scripts/assemble_qwen3_tts.py \
  --backbone pretrained/Qwen3-0.6B-Base \
  --tts-template pretrained/Qwen3-TTS-12Hz-0.6B-Base \
  --codec pretrained/Qwen3-TTS-Tokenizer-12Hz \
  --output pretrained/assembled-qwen3-tts-frozen-conditioning

PYTHONPATH=. python scripts/export_manifest.py \
  --dataset emilia-short \
  --source /ai_sds_wuzz/DATA_TTS/Emilia2_TTS_m4a \
  --output /ai_sds_wuzz/DATA_TTS/Emilia2_TTS_prepared/LM-TTS-Training/emilia-short-en-1000h.raw.jsonl \
  --limit 2000000 --target-hours 1000 \
  --language en --min-seconds 2 --max-seconds 10

PYTHONPATH=. python scripts/prepare_manifest.py \
  --manifest /ai_sds_wuzz/DATA_TTS/Emilia2_TTS_prepared/LM-TTS-Training/emilia-short-en-1000h.raw.jsonl \
  --output /ai_sds_wuzz/DATA_TTS/Emilia2_TTS_prepared/LM-TTS-Training/emilia-short-en-1000h \
  --tokenizer pretrained/assembled-qwen3-tts-frozen-conditioning \
  --codec pretrained/Qwen3-TTS-Tokenizer-12Hz \
  --device cuda:0 --secondary-device cuda:1 --batch-size 16 --workers 8 --decode-processes 8 --val-count 512
```

raw manifest 导出后，推荐用控制脚本接管预处理、显存测试及训练：

```bash
PYTHONPATH=. python scripts/run_emilia_baseline.py
# 如果 exporter 尚在运行，可指定真实 Python 进程 PID；控制脚本等待它原子发布 manifest。
PYTHONPATH=. python scripts/run_emilia_baseline.py --export-pid <PID>
```

控制脚本有进程锁，避免同一个 run 被重复启动；状态保存在 `runs/emilia-official-frozen-1000h/pipeline-status.json`，每阶段有独立日志。中断后重跑同一个入口会复用匹配 recipe 的 codec 缓存；已有 checkpoint 时沿用落盘的实际配置并 `--resume latest`。失败会记录原因并停止，不能把启动成功视为阶段完成。

当前计划使用双卡 FSDP2、BF16、关闭 activation checkpointing，先测每卡 batch 48，CUDA OOM 时再测 32。两次完整优化更新覆盖 AdamW 状态分配；只有压力测试通过才开始训练。最终配置写入 run 下 `baseline-config.yaml`，全局 batch 为每卡 batch × 2。训练 5,000 updates，warmup 200，主干学习率 2e-5、新模块 1e-4；每 100 steps 验证 loss，默认 `keep_checkpoints: 2`，只在新 checkpoint 完整落盘后清理旧的完整 checkpoint；设为 null 可关闭自动清理。保留数量不影响恢复签名。每 500 steps 保存 checkpoint 并生成 8 条验证／2 条训练音频。所有验证 loss 使用 512 条验证集；生成内容分数只覆盖固定样本，不能代表完整验证集。

```bash
.venv/bin/tensorboard --logdir runs/emilia-official-frozen-1000h --port 6006
```

训练前必须用实际 prepared manifest 测量最长文本／音频组合的显存，随后记录最终每卡 batch 和 global batch。不能将 LJSpeech 的显存结果直接套到 Emilia。

## 先看什么结果

先确认数据解码长度、codec 重建、参考来源、冻结权重、loss 对齐、FSDP 与恢复正确。正式训练同时观察训练／验证的首码本 CE、各残余码本 CE、生成长度与 EOS，以及固定训练／验证文本的生成音频。

英语内容指标保留原始规范化 WER/CER，新基线在线记录 Whisper 英语规范化分数；也可用 `scripts/rescore_english.py` 给已有转写补充相同分数，避免把 “one hundred forty-five” 与 “145” 记成内容错误。原音频也做 ASR 对照，区分数据／识别器误差与生成误差；其他语言不能直接用当前 `small.en` 评分。

基线稳定后，才按相同数据、采样顺序、batch、更新次数、学习率、验证样本逐项对照：冻结与解冻公开前端；保持形状但更换初始化；同源 0.6B embedding 直连。每次只改变明确的一项，并分别报告文本前端来源和是否训练。不要同时改数据量、结构与学习率后将结果归因于 projector。

## 已完成与尚待完成

真实 short 小样本的 metadata 导出、PyAV 解码、批量 codec 编码、11 train / 2 val 生成及双卡冻结前端训练已跑通。1,000 小时全量筛选／编码／训练的完成状态和指标需依据运行日志另行记录，不能由小样本验证替代。

代码和训练验证记录见 [训练正确性与诊断](../validation/training-diagnosis.md)。
