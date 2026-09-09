import argparse
import hashlib
import json
import math
import os
import random
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
import yaml
from accelerate import skip_first_batches
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import AutoConfig

from .checkpoint import load_checkpoint, save_checkpoint
from .data import CodeDataset, DistributedTokenBatchSampler, collate, val_batches
from .metrics import ASRScorer, aggregate_content, english_metrics, normalize
from .model import TTSModel, make_config


def move(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def configure_fsdp(model, device):
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    model.to(device)
    for blocks in [model.talker.model.layers, model.talker.code_predictor.model.layers]:
        for block in blocks:
            fully_shard(block, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=True)


@torch.no_grad()
def validate(model, dataset, batch_size, device):
    model.eval()
    totals = torch.zeros(19, dtype=torch.float64, device=device)
    for indices, real in val_batches(dataset, batch_size, dist.get_world_size(), dist.get_rank()):
        out = model(move(collate([dataset[i] for i in indices]), device))
        if real:
            totals[0] += out["first_sum"]
            totals[1] += out["first_count"]
            totals[2] += out["residual_sum"]
            totals[3] += out["frame_count"]
            totals[4:] += out["group_sums"]
    dist.all_reduce(totals)
    metrics = {"first_ce": (totals[0] / totals[1]).item(),
               "residual_ce": (totals[2] / (totals[3] * 15)).item()}
    metrics.update({f"codebook_{i + 1}_ce": (totals[4 + i] / totals[3]).item() for i in range(15)})
    model.train()
    return metrics


@torch.no_grad()
def generate_sample(model, dataset, device, settings, output, step, writer, index=0, multi=False,
                    codec=None, scorer=None, tag_prefix="eval", reference=None, conditioning="speaker_only"):
    from qwen_tts import Qwen3TTSTokenizer
    model.eval()
    row = dataset[index]
    prompt = {**row, 'codes': row['codes'][:0]}
    if reference is not None:
        prompt['speaker_mels'] = reference['speaker_mels']
    if conditioning == 'icl':
        # One BOS/EOS pair around reference + target text; audio starts with
        # the reference codes, following Qwen's non-streaming ICL protocol.
        prompt['text_ids'] = reference['text_ids'][:-1] + row['text_ids'][1:]
        prompt['codes'] = reference['codes']
    prefix_codes = prompt['codes']
    tag_prefix = f'{tag_prefix}/{conditioning}'
    tag = f"{tag_prefix}/sample_{index:02d}" if multi else tag_prefix
    batch = move(collate([prompt]), device)
    generated = []
    stopped = False
    start = time.perf_counter()
    for _ in range(settings["max_frames"]):
        frame, stop = model(batch, mode="next_frame")
        # All ranks use the same fixed text and greedy decode, and execute the
        # same FSDP collectives, including at EOS.
        stop_flag = stop.to(torch.int32)
        dist.broadcast(stop_flag, src=0)
        if stop_flag.item():
            stopped = True
            break
        dist.broadcast(frame, src=0)
        generated.append(frame[0].cpu())
        batch["codes"] = torch.cat([batch["codes"], frame], dim=0)
        batch["frame_lengths"] += 1
    result = {"id": row["id"], "text": row["text"], 'language': row.get('language', 'en'), "frames": len(generated),
              "eos_reached": stopped, "truncated": not stopped,
              "generation_seconds": time.perf_counter() - start, 'conditioning': conditioning,
              'reference_frames': len(prefix_codes)}
    if reference is not None:
        result['speaker_reference_audio'] = reference['audio']
        result['speaker_reference_id'] = reference['id']
        result['speaker_reference_source'] = reference.get('audio_source', reference['audio'])
        if conditioning == 'icl':
            result['reference_text'] = reference['text']
    if dist.get_rank() == 0:
        folder = output / "evaluation" / conditioning / f"step-{step:08d}"
        if multi:
            folder = folder / f"sample-{index:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        if generated:
            if codec is None:
                codec = Qwen3TTSTokenizer.from_pretrained(settings["codec"], device_map=settings.get("codec_device", "cpu"))
                codec.model.eval().requires_grad_(False)
            decode_codes = torch.cat([prefix_codes, torch.stack(generated)])
            audio, sr = codec.decode({"audio_codes": [decode_codes]})
            # Decode with reference context, then remove its waveform using the
            # same frame-ratio boundary as Qwen's voice-clone wrapper.
            cut = int(len(prefix_codes) / len(decode_codes) * len(audio[0]))
            waveform = audio[0][cut:]
            audio_path = folder / "generated.wav"
            sf.write(audio_path, waveform, sr)
            result["duration_seconds"] = len(waveform) / sr
            result["duration_ratio"] = result["duration_seconds"] / row["duration"]
            writer.add_audio(f"{tag}/generated", np.asarray(waveform)[None], step, sample_rate=sr)
            from .sources import evaluation_audio
            target_audio = evaluation_audio(row)
            ref_audio, ref_sr = sf.read(target_audio, dtype="float32")
            writer.add_audio(f"{tag}/reference", ref_audio[None], step, sample_rate=ref_sr)
            if settings.get("asr", True):
                if scorer is None:
                    scorer = ASRScorer(settings.get("asr_model", "small.en"))
                result["content"] = scorer.score(audio_path, row["text"], row.get('language', 'en'))
                result["reference_asr"] = scorer.score(target_audio, row["text"], row.get('language', 'en'))
                if settings.get("english_normalization", False) and row.get('language', 'en') == 'en':
                    for key in ["content", "reference_asr"]:
                        result["english_" + key] = english_metrics(row["text"], result[key]["transcript"])
                    for key in ["wer", "cer"]:
                        writer.add_scalar(f"{tag}/english_{key}", result["english_content"][key], step)
                for name in ["wer", "cer", "deletions", "insertions", "substitutions"]:
                    writer.add_scalar(f"{tag}/{name}", result["content"][name], step)
                writer.add_scalar(f"{tag}/reference_asr_wer", result["reference_asr"]["wer"], step)
        else:
            from .metrics import content_metrics
            result["content"] = content_metrics(row["text"], "")
            if settings.get("english_normalization", False) and row.get('language', 'en') == 'en':
                result["english_content"] = english_metrics(row["text"], "")
            writer.add_scalar(f"{tag}/wer", 1.0, step)
            writer.add_scalar(f"{tag}/cer", 1.0, step)
        writer.add_scalar(f"{tag}/truncated", int(result["truncated"]), step)
        writer.add_text(f"{tag}/result", json.dumps(result, ensure_ascii=False, indent=2), step)
        (folder / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        writer.flush()
        print(json.dumps({"step": step, "generation": result}), flush=True)
    dist.barrier()
    model.train()
    return result


def evaluate_audio(model, val_data, device, eval_settings, output, step, writer, train_data=None):
    modes = eval_settings.get('conditioning_modes', ['speaker_only'])
    if not modes or len(set(modes)) != len(modes) or any(mode not in ('speaker_only', 'icl') for mode in modes):
        raise ValueError('conditioning_modes must contain speaker_only and/or icl without duplicates')
    if 'icl' in modes and (getattr(model.config, 'lm_tts_input_protocol', None) != 'qwen3_non_streaming'
                          or model.speaker_encoder is None or train_data is None
                          or train_data.text_special_tokens is None or val_data.text_special_tokens is None):
        raise ValueError('ICL evaluation requires the Qwen non-streaming model and prepared reference text/audio')
    # Evaluation may create models and consume RNG; isolate it from training.
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
        from qwen_tts import Qwen3TTSTokenizer
        codec, scorer = None, None
        if dist.get_rank() == 0:
            codec = Qwen3TTSTokenizer.from_pretrained(eval_settings["codec"], device_map=eval_settings.get("codec_device", "cpu"))
            codec.model.eval().requires_grad_(False)
            if eval_settings.get("asr", True):
                scorer = ASRScorer(eval_settings.get("asr_model", "small.en"))
        selections = [("eval", val_data, output, eval_settings.get("num_samples", 1))]
        if train_data is not None and eval_settings.get("train_num_samples", 0):
            selections.append(("train_eval", train_data, output / "train-evaluation", eval_settings["train_num_samples"]))
        for tag, dataset, folder, count in selections:
            sample_count = min(len(dataset), count)
            if sample_count < 1:
                raise ValueError("Evaluation sample count must be positive")
            speakers = {}
            if model.speaker_encoder is not None:
                for i, row in enumerate(train_data.rows):
                    choices = speakers.setdefault((row['speaker'], row.get('language', 'en')), [])
                    if len(choices) < 2 and all(normalize(train_data.rows[j]['text']) != normalize(row['text']) for j in choices):
                        choices.append(i)
            indices, references = [], []
            languages = sorted({row.get('language', 'en') for row in dataset.rows})
            per_language = {language: 0 for language in languages}
            for i, row in enumerate(dataset.rows):
                language = row.get('language', 'en')
                if per_language[language] >= math.ceil(sample_count / len(languages)):
                    continue
                reference = next((j for j in speakers.get((row.get('speaker'), language), [])
                                  if train_data.rows[j]['id'] != row['id']
                                  and normalize(train_data.rows[j]['text']) != normalize(row['text'])), None)
                if model.speaker_encoder is not None and reference is None:
                    continue
                indices.append(i)
                references.append(train_data[reference] if reference is not None else None)
                per_language[language] += 1
                if len(indices) == sample_count:
                    break
            if not indices:
                raise ValueError('No generation samples have another training utterance for speaker conditioning')
            for conditioning in modes:
                samples = [generate_sample(model, dataset, device, eval_settings, folder, step, writer,
                                           index=i, multi=True, codec=codec, scorer=scorer, tag_prefix=tag,
                                           reference=reference, conditioning=conditioning)
                           for i, reference in zip(indices, references)]
                if dist.get_rank() == 0:
                    mode_tag = f'{tag}/{conditioning}'
                    summary = {'conditioning': conditioning, 'samples': len(samples),
                               'truncation_rate': sum(s['truncated'] for s in samples) / len(samples)}
                    writer.add_scalar(f'{mode_tag}/truncation_rate', summary['truncation_rate'], step)
                    if all('content' in sample for sample in samples):
                        summary.update(aggregate_content([sample['content'] for sample in samples]))
                        summary['by_language'] = {
                            language: aggregate_content([s['content'] for s in samples if s['language'] == language])
                            for language in sorted({sample['language'] for sample in samples})}
                        english = [sample['english_content'] for sample in samples if 'english_content' in sample]
                        if english:
                            summary['english'] = aggregate_content(english)
                            for key in ['wer', 'cer']:
                                writer.add_scalar(f'{mode_tag}/english_{key}', summary['english'][key], step)
                        for key in ['wer', 'cer']:
                            writer.add_scalar(f'{mode_tag}/{key}', summary[key], step)
                        for language, metrics in summary['by_language'].items():
                            for key in ['wer', 'cer']:
                                writer.add_scalar(f'{mode_tag}/{language}/{key}', metrics[key], step)
                    (folder / 'evaluation' / conditioning / f'step-{step:08d}' / 'summary.json').write_text(
                        json.dumps(summary, indent=2))
                    writer.flush()
    random.setstate(python_state)
    np.random.set_state(numpy_state)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", help="Completed checkpoint directory, or 'latest'")
    p.add_argument("--max-steps", type=int, help="Override stopping step, retaining the configured LR schedule")
    p.add_argument("--output", help="Override run output directory")
    p.add_argument("--eval-only", action="store_true", help="Evaluate an existing checkpoint without optimizer updates")
    p.add_argument("--eval-samples", type=int, help="Override generated sample count for eval-only")
    args = p.parse_args()
    if args.eval_only and not args.resume:
        p.error("--eval-only requires --resume")
    if args.eval_samples is not None and (not args.eval_only or args.eval_samples < 1):
        p.error("--eval-samples requires --eval-only and a positive count")
    config = yaml.safe_load(Path(args.config).read_text())
    settings = config["train"]
    settings.setdefault("keep_checkpoints", 2)
    settings.setdefault("num_workers", 4)
    settings.setdefault("prefetch_factor", 2)
    config['eval'].setdefault('batch_size', 8)
    if settings["keep_checkpoints"] is not None and (not isinstance(settings["keep_checkpoints"], int) or settings["keep_checkpoints"] < 1):
        raise ValueError("keep_checkpoints must be a positive integer or null")
    if args.max_steps is not None:
        settings["max_steps"] = args.max_steps
    if args.output:
        settings["output"] = args.output
    for key in ["max_batch_frames", "max_batch_tokens", "accumulation", "max_steps", "schedule_steps", "save_every", "eval_every", "log_every", "prefetch_factor"]:
        if settings[key] < 1:
            raise ValueError(f"{key} must be positive")
    if config['eval']['batch_size'] < 1 or settings['num_workers'] < 0:
        raise ValueError('eval.batch_size must be positive and num_workers must be nonnegative')
    if settings["max_steps"] > settings["schedule_steps"]:
        raise ValueError("max_steps exceeds the fixed schedule_steps horizon")
    if config["eval"].get("audio_every", 0) and config["eval"]["max_frames"] < 1:
        raise ValueError("max_frames must be positive")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=device)
    writer = None
    loader = epoch_loader = reader = None
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        seed = config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        train_data = CodeDataset(config["data"]["train"])
        val_data = CodeDataset(config["data"]["val"])
        train_ids = {r["id"] for r in train_data.rows}
        if train_ids.intersection(r["id"] for r in val_data.rows):
            raise ValueError("Train/validation ID leakage")
        model_cfg = config["model"]
        assembly_fingerprint = None
        if model_cfg.get("assembled_model"):
            assembled = Path(model_cfg["assembled_model"])
            assembly_fingerprint = hashlib.sha256((assembled / "assembly_report.json").read_bytes()).hexdigest()
            model = TTSModel.from_assembled(assembled, load_weights=not args.resume,
                                          attn_implementation=model_cfg.get("attn_implementation", "sdpa"))
            model_config = model.config
            initialization = {"source": str(assembled), "assembly_report_sha256": assembly_fingerprint,
                              "speaker_encoder": "pretrained Qwen ECAPA-TDNN, " + ("frozen" if getattr(model.config, "lm_tts_freeze_speaker_encoder", False) else "jointly trained")}
            full_config = json.loads((assembled / "config.json").read_text())
            for dataset in [train_data, val_data]:
                dataset.text_special_tokens = (full_config["tts_bos_token_id"], full_config["tts_eos_token_id"])
            train_data.target_speaker = True
            val_data.target_speaker = True
        else:
            base_cfg = None if model_cfg["tiny"] else AutoConfig.from_pretrained(model_cfg["backbone"])
            model_config = make_config(base_cfg, tiny=model_cfg["tiny"])
            model = TTSModel(model_config)
            initialization = {"source": "random tiny integration model"}
            if not model_cfg["tiny"] and not args.resume:
                initialization = model.initialize_backbone(model_cfg["backbone"])
        if model_cfg.get("activation_checkpointing", True):
            model.talker.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.talker.code_predictor.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        output = Path(settings["output"]).resolve()
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            if not args.resume and (output / "checkpoints" / "latest").exists():
                raise ValueError("Output already has a checkpoint; use --resume latest or a new output")
        dist.barrier()
        configure_fsdp(model, device)
        pretrained, fresh = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(("talker.model.layers.", "talker.model.norm.", "talker.model.text_embedding.", "speaker_encoder.")):
                pretrained.append(param)
            else:
                fresh.append(param)
        optimizer = torch.optim.AdamW([
            {"params": pretrained, "lr": settings["backbone_lr"]},
            {"params": fresh, "lr": settings["lr"]},
        ], weight_decay=settings["weight_decay"])
        def lr_factor(step):
            warmup = settings["warmup_steps"]
            if step < warmup:
                return (step + 1) / max(1, warmup)
            ratio = min(1.0, (step - warmup) / max(1, settings["schedule_steps"] - warmup))
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * ratio))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        signature = {"protocol": 3, "audio_decoder": "emilia_native_seek_v1", "deterministic": True,
                     "cublas_workspace": os.environ["CUBLAS_WORKSPACE_CONFIG"], "seed": seed, "model": model_cfg,
                     "train_manifest": train_data.fingerprint, "val_manifest": val_data.fingerprint,
                     "settings": {k: v for k, v in settings.items() if k not in ["output", "max_steps", "save_every", "eval_every", "log_every", "keep_checkpoints", "num_workers", "prefetch_factor"]},
                     "eval": config["eval"], "torch": torch.__version__}
        signature['batch_layout'] = 'packed'
        signature['batch_sampler'] = 'distributed_token_budget_v1'
        if assembly_fingerprint:
            signature.update(assembly_report_sha256=assembly_fingerprint,
                             speaker_conditioning='full_target_audio', generation_conditioning='other_training_utterance')
        progress = {"step": 0, "epoch": 0, "next_batch": 0}
        if args.resume:
            resume = args.resume
            if resume == "latest":
                resume = output / "checkpoints" / (output / "checkpoints" / "latest").read_text().strip()
            progress = load_checkpoint(resume, model, optimizer, scheduler, signature)
        if rank == 0:
            writer = SummaryWriter(str(output / "tensorboard"), purge_step=progress["step"] + 1 if args.resume else None)
            (output / "config.yaml").write_text(yaml.safe_dump(config))
            if not args.resume:
                (output / "initialization.json").write_text(json.dumps(initialization, indent=2))
                (output / "model_config.json").write_text(model_config.to_json_string())
            print(json.dumps({"initialized": True, "world_size": world, "progress": progress, "parameters": sum(p.numel() for p in model.parameters())}), flush=True)
        model.train()
        if args.eval_only:
            val = validate(model, val_data, config['eval']['batch_size'], device)
            eval_settings = dict(config["eval"])
            if args.eval_samples is not None:
                eval_settings["num_samples"] = args.eval_samples
            if rank == 0:
                print(json.dumps({"step": progress["step"], "val": val}), flush=True)
            evaluate_audio(model, val_data, device, eval_settings, output, progress["step"], writer, train_data)
            return
        if progress['step'] >= settings['max_steps']:
            return
        prefix_tokens = 1 + int(model.speaker_encoder is not None)
        if getattr(model.config, 'lm_tts_input_protocol', None) == 'qwen3_non_streaming':
            prefix_tokens += len(model.config.lm_tts_role_ids) + 3
        text_special_tokens = 2 if train_data.text_special_tokens is not None else 0
        sampler = DistributedTokenBatchSampler(
            [r['num_frames'] for r in train_data.rows],
            [r['num_frames'] + len(r['text_ids']) + text_special_tokens + prefix_tokens for r in train_data.rows],
            settings['max_batch_frames'], settings['max_batch_tokens'], world_size=world, rank=rank, seed=seed)
        # Bound the index stream so workers finish their last batch before exit,
        # instead of decoding prefetched audio beyond the requested stopping step.
        sampler.set_epoch(progress['epoch'], max_batches=progress['next_batch']
                          + (settings['max_steps'] - progress['step']) * settings['accumulation'])
        workers = settings["num_workers"]
        loader = DataLoader(
            train_data, batch_sampler=sampler, collate_fn=collate, num_workers=workers,
            prefetch_factor=settings["prefetch_factor"] if workers else None,
            persistent_workers=bool(workers), pin_memory=True,
            multiprocessing_context="spawn" if workers else None,
            generator=torch.Generator().manual_seed(seed + rank))
        epoch_loader = skip_first_batches(loader, progress["next_batch"]) if progress["next_batch"] else loader
        reader = iter(epoch_loader)
        while progress["step"] < settings["max_steps"]:
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            batches = []
            for _ in range(settings["accumulation"]):
                try:
                    batch = next(reader)
                except StopIteration:
                    progress["epoch"] += 1
                    progress["next_batch"] = 0
                    sampler.set_epoch(progress['epoch'], max_batches=
                                      (settings['max_steps'] - progress['step']) * settings['accumulation'] - len(batches))
                    reader = epoch_loader = None
                    epoch_loader = loader
                    reader = iter(loader)
                    batch = next(reader)
                batches.append(batch)
                progress["next_batch"] += 1
            data_wait_seconds = time.perf_counter() - start
            counts = torch.tensor([sum(b['frame_lengths'].sum().item() + len(b['frame_lengths']) for b in batches),
                                   sum(b['frame_lengths'].sum().item() * 15 for b in batches),
                                   sum(len(b['frame_lengths']) for b in batches),
                                   sum(b['frame_lengths'].sum().item() + b['text_lengths'].sum().item()
                                       + prefix_tokens * len(b['frame_lengths'])
                                       for b in batches)], dtype=torch.float64, device=device)
            dist.all_reduce(counts)
            optimizer.zero_grad(set_to_none=True)
            sums = torch.zeros(2, dtype=torch.float64, device=device)
            for i, batch in enumerate(batches):
                model.set_requires_gradient_sync(i == len(batches) - 1)
                out = model(move(batch, device))
                loss = world * (out["first_sum"] / counts[0] + settings["residual_weight"] * out["residual_sum"] / counts[1])
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Non-finite loss")
                loss.backward()
                sums += torch.stack([out["first_sum"].detach(), out["residual_sum"].detach()])
            speaker_grad_norm = None
            if model.speaker_encoder is not None and any(p.requires_grad for p in model.speaker_encoder.parameters()):
                squared = torch.zeros((), device=device)
                for parameter in model.speaker_encoder.parameters():
                    if parameter.grad is None:
                        raise RuntimeError("Speaker encoder did not receive a gradient")
                    squared += parameter.grad.to_local().float().square().sum()
                dist.all_reduce(squared)
                speaker_grad_norm = squared.sqrt().item()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"])
            if not torch.isfinite(norm).item():
                raise FloatingPointError("Non-finite gradient norm")
            optimizer.step()
            scheduler.step()
            progress["step"] += 1
            step = progress["step"]
            dist.all_reduce(sums)
            elapsed = time.perf_counter() - start
            metrics = {"first_ce": (sums[0] / counts[0]).item(), "residual_ce": (sums[1] / counts[1]).item(),
                       "grad_norm": norm.item(), "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                       "audio_seconds_per_second": counts[1].item() / 15 / 12.5 / elapsed,
                       "step_seconds": elapsed, "data_wait_seconds": data_wait_seconds,
                       "global_samples": counts[2].item(), "global_audio_frames": counts[1].item() / 15,
                       "global_talker_tokens": counts[3].item(),
                       "frame_budget_fill": counts[1].item() / (15 * world * len(batches) * settings['max_batch_frames']),
                       "token_budget_fill": counts[3].item() / (world * len(batches) * settings['max_batch_tokens']),
                       "lr_backbone": optimizer.param_groups[0]["lr"], "lr_new": optimizer.param_groups[1]["lr"]}
            if speaker_grad_norm is not None:
                metrics["speaker_grad_norm"] = speaker_grad_norm
            if rank == 0 and step % settings["log_every"] == 0:
                for key, value in metrics.items():
                    writer.add_scalar(f"train/{key}", value, step)
                print(json.dumps({"step": step, "train": metrics}), flush=True)
            if step % settings["save_every"] == 0 or step == settings["max_steps"]:
                saved = save_checkpoint(output / "checkpoints", model, optimizer, scheduler, progress.copy(), signature, keep=settings["keep_checkpoints"])
                if rank == 0:
                    writer.flush()
                    print(f"Saved {saved}", flush=True)
            if step % settings["eval_every"] == 0 or step == settings["max_steps"]:
                val = validate(model, val_data, config['eval']['batch_size'], device)
                if rank == 0:
                    for key, value in val.items():
                        writer.add_scalar(f"val/{key}", value, step)
                    print(json.dumps({"step": step, "val": val}), flush=True)
            audio_every = config["eval"].get("audio_every", 0)
            if audio_every and (step % audio_every == 0 or step == settings["max_steps"]):
                evaluate_audio(model, val_data, device, config["eval"], output, step, writer, train_data)
    finally:
        reader = epoch_loader = loader = None
        if writer:
            writer.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
