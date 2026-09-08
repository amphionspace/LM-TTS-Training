import argparse
import json
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
import yaml
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig

from .checkpoint import load_checkpoint, save_checkpoint
from .data import CodeDataset, collate, train_batches, val_batches
from .metrics import ASRScorer, aggregate_content
from .model import TTSModel, make_config


def move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


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
def generate_sample(model, dataset, device, settings, output, step, writer, index=0, multi=False):
    from qwen_tts import Qwen3TTSTokenizer
    model.eval()
    row = dataset[index]
    tag = f"eval/sample_{index:02d}" if multi else "eval"
    batch = move(collate([row]), device)
    batch["codes"] = batch["codes"][:, :0]
    batch["frame_mask"] = batch["frame_mask"][:, :0]
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
        batch["codes"] = torch.cat([batch["codes"], frame[:, None]], dim=1)
        batch["frame_mask"] = torch.ones(batch["codes"].shape[:2], dtype=torch.bool, device=device)
    result = {"id": row["id"], "text": row["text"], "frames": len(generated),
              "eos_reached": stopped, "truncated": not stopped,
              "generation_seconds": time.perf_counter() - start}
    if dist.get_rank() == 0:
        folder = output / "evaluation" / f"step-{step:08d}"
        if multi:
            folder = folder / f"sample-{index:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        if generated:
            # CPU codec avoids keeping frozen codec parameters on training GPUs.
            codec = Qwen3TTSTokenizer.from_pretrained(settings["codec"], device_map="cpu")
            codec.model.eval().requires_grad_(False)
            audio, sr = codec.decode({"audio_codes": [torch.stack(generated)]})
            audio_path = folder / "generated.wav"
            sf.write(audio_path, audio[0], sr)
            result["duration_seconds"] = len(audio[0]) / sr
            result["duration_ratio"] = result["duration_seconds"] / row["duration"]
            writer.add_audio(f"{tag}/generated", np.asarray(audio[0])[None], step, sample_rate=sr)
            ref_audio, ref_sr = sf.read(row["audio"], dtype="float32")
            writer.add_audio(f"{tag}/reference", ref_audio[None], step, sample_rate=ref_sr)
            if settings.get("asr", True):
                scorer = ASRScorer(settings.get("asr_model", "small.en"))
                result["content"] = scorer.score(audio_path, row["text"])
                result["reference_asr"] = scorer.score(row["audio"], row["text"])
                for name in ["wer", "cer", "deletions", "insertions", "substitutions"]:
                    writer.add_scalar(f"{tag}/{name}", result["content"][name], step)
                writer.add_scalar(f"{tag}/reference_asr_wer", result["reference_asr"]["wer"], step)
        else:
            from .metrics import content_metrics
            result["content"] = content_metrics(row["text"], "")
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


def evaluate_audio(model, val_data, device, eval_settings, output, step, writer):
    # Evaluation may create models and consume RNG; isolate it from training.
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[device.index]):
        sample_count = min(len(val_data), eval_settings.get("num_samples", 1))
        if sample_count < 1:
            raise ValueError("eval.num_samples must be positive")
        samples = [generate_sample(model, val_data, device, eval_settings, output, step, writer,
                                   index=i, multi=sample_count > 1) for i in range(sample_count)]
        if dist.get_rank() == 0 and all("content" in sample for sample in samples):
            summary = aggregate_content([sample["content"] for sample in samples])
            summary["truncation_rate"] = sum(s["truncated"] for s in samples) / len(samples)
            for key in ["wer", "cer", "truncation_rate"]:
                writer.add_scalar(f"eval/{key}", summary[key], step)
            (output / "evaluation" / f"step-{step:08d}" / "summary.json").write_text(json.dumps(summary, indent=2))
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
    if args.max_steps is not None:
        settings["max_steps"] = args.max_steps
    if args.output:
        settings["output"] = args.output
    for key in ["batch_size", "accumulation", "max_steps", "schedule_steps", "save_every", "eval_every", "log_every"]:
        if settings[key] < 1:
            raise ValueError(f"{key} must be positive")
    if settings["max_steps"] > settings["schedule_steps"]:
        raise ValueError("max_steps exceeds the fixed schedule_steps horizon")
    if config["eval"].get("audio_every", 0) and config["eval"]["max_frames"] < 1:
        raise ValueError("max_frames must be positive")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=device)
    writer = None
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
            if name.startswith(("talker.model.layers.", "talker.model.norm.", "talker.model.text_embedding.")):
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
        signature = {"protocol": 1, "seed": seed, "model": model_cfg,
                     "train_manifest": train_data.fingerprint, "val_manifest": val_data.fingerprint,
                     "settings": {k: v for k, v in settings.items() if k not in ["output", "max_steps", "save_every", "eval_every", "log_every"]},
                     "eval": config["eval"], "torch": torch.__version__}
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
            val = validate(model, val_data, settings["batch_size"], device)
            eval_settings = dict(config["eval"])
            if args.eval_samples is not None:
                eval_settings["num_samples"] = args.eval_samples
            if rank == 0:
                print(json.dumps({"step": progress["step"], "val": val}), flush=True)
            evaluate_audio(model, val_data, device, eval_settings, output, progress["step"], writer)
            return
        while progress["step"] < settings["max_steps"]:
            start = time.perf_counter()
            batches = []
            for _ in range(settings["accumulation"]):
                plan = train_batches(train_data, settings["batch_size"], world, rank, seed, progress["epoch"])
                if progress["next_batch"] == len(plan):
                    progress["epoch"] += 1
                    progress["next_batch"] = 0
                    plan = train_batches(train_data, settings["batch_size"], world, rank, seed, progress["epoch"])
                indices = plan[progress["next_batch"]]
                batches.append(collate([train_data[i] for i in indices]))
                progress["next_batch"] += 1
            counts = torch.tensor([sum(b["frame_mask"].sum().item() + len(b["codes"]) for b in batches),
                                   sum(b["frame_mask"].sum().item() * 15 for b in batches)], dtype=torch.float64, device=device)
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
                       "grad_norm": norm.item(), "audio_seconds_per_second": counts[1].item() / 15 / 12.5 / elapsed,
                       "step_seconds": elapsed, "lr_backbone": optimizer.param_groups[0]["lr"], "lr_new": optimizer.param_groups[1]["lr"]}
            if rank == 0 and step % settings["log_every"] == 0:
                for key, value in metrics.items():
                    writer.add_scalar(f"train/{key}", value, step)
                print(json.dumps({"step": step, "train": metrics}), flush=True)
            if step % settings["save_every"] == 0 or step == settings["max_steps"]:
                saved = save_checkpoint(output / "checkpoints", model, optimizer, scheduler, progress.copy(), signature)
                if rank == 0:
                    writer.flush()
                    print(f"Saved {saved}", flush=True)
            if step % settings["eval_every"] == 0:
                val = validate(model, val_data, settings["batch_size"], device)
                if rank == 0:
                    for key, value in val.items():
                        writer.add_scalar(f"val/{key}", value, step)
                    print(json.dumps({"step": step, "val": val}), flush=True)
            audio_every = config["eval"].get("audio_every", 0)
            if audio_every and step % audio_every == 0:
                evaluate_audio(model, val_data, device, config["eval"], output, step, writer)
    finally:
        if writer:
            writer.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
