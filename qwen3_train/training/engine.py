"""Optimizer updates and resume orchestration over published unified builds."""

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..artifacts import digest
from ..data.batch import collate
from ..data.build import tokenizer_identity
from ..data.reader import FeatureDataset
from ..data.sampler import TokenBatchSampler
from ..evaluation.validation import validate
from ..models.qwen import TTSModel
from ..objectives.tts import loss_normalizers, tts_loss
from .checkpoint import load_checkpoint, save_checkpoint
from .distributed import batch_health, initialize, move, shard
from .precision import training_precision
from .schedule import finished, learning_rate_factor
from .telemetry import (
    setup_dashboard,
    start_gpu_logging,
    stop_gpu_logging,
    tensorboard_groups,
    write_training,
    write_validation,
)


def verify_conditioning(model, dataset):
    if not getattr(model.config, "lm_tts_freeze_speaker_encoder", False):
        raise ValueError("Cached embeddings require a frozen speaker encoder")
    tokenizer = tokenizer_identity(dataset.manifest["tokenizer"]["path"])
    if tokenizer["sha256"] != dataset.manifest["tokenizer"]["sha256"]:
        raise ValueError("Build tokenizer artifact changed")
    state = model.speaker_encoder.state_dict()
    for binding in dataset.bindings:
        tensors = binding["speaker_profile"]["model"]["tensors"]
        if set(tensors) != {"speaker_encoder." + k for k in state}:
            raise ValueError("Cached speaker profile has a different architecture")
        for name, tensor in state.items():
            expected = tensors["speaker_encoder." + name]
            dtype = getattr(torch, expected["dtype"].removeprefix("torch."))
            value = tensor.detach().cpu().to(dtype).contiguous()
            actual = hashlib.sha256(
                value.reshape(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            if list(value.shape) != expected["shape"] or actual != expected["sha256"]:
                raise ValueError(f"Cached speaker profile weights differ: {name}")


def run(config, resume=None, eval_only=False):
    seed, settings = config["seed"], config["train"]
    policy, backend = training_precision(settings, config["model"])
    random.seed(seed)
    np.random.seed(seed)
    device = initialize(seed)
    loader = reader = writer = gpu_monitor = None
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        assembled = Path(config["model"]["assembled_model"]).resolve()
        full_config = json.loads((assembled / "config.json").read_text())
        specials = (full_config["tts_bos_token_id"], full_config["tts_eos_token_id"])
        train_data = FeatureDataset(config["data"]["build"], specials, verify_integrity=rank == 0)
        if rank == 0:
            train_data.validate_budgets(settings["max_batch_frames"], settings["max_batch_tokens"])
        if train_data.manifest["tokenizer"]["sha256"] != tokenizer_identity(assembled)["sha256"]:
            raise ValueError("Data build and assembled model use different tokenizers")
        val_data = (
            FeatureDataset(config["data"]["val_build"], specials, verify_integrity=rank == 0)
            if config["data"].get("val_build")
            else None
        )
        if val_data is not None:
            # Isolation is a separately published data decision, never inferred from storage split=train.
            if (
                val_data.manifest["evaluation"] != "heldout"
                or train_data.manifest["evaluation"] != "train_isolated"
                or not train_data.manifest.get("split_sha256")
                or train_data.manifest["split_sha256"] != val_data.manifest.get("split_sha256")
            ):
                raise ValueError("Validation requires builds published from an isolated selection")
        model = TTSModel.from_assembled(assembled, load_weights=True, attn_implementation=backend)
        verify_conditioning(model, train_data)
        if val_data is not None:
            verify_conditioning(model, val_data)
        if config["model"].get("activation_checkpointing", True):
            model.talker.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.talker.code_predictor.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        output = Path(settings["output"]).resolve()
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            if not resume and (output / "checkpoints/latest").exists():
                raise ValueError("Run already has checkpoints; resume or choose a new output")
        dist.barrier()
        gpu_monitor = start_gpu_logging(output, rank)
        shard(model, device, policy)
        backbone, fresh = [], []
        for name, param in model.named_parameters():
            if param.requires_grad:
                destination = (
                    backbone
                    if name.startswith(("talker.model.layers.", "talker.model.norm."))
                    else fresh
                )
                destination.append(param)
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone, "lr": settings["backbone_lr"]},
                {"params": fresh, "lr": settings["lr"]},
            ],
            weight_decay=settings["weight_decay"],
        )

        progress = {"step": 0, "epoch": 0, "next_batch": 0, "samples_in_epoch": 0}

        def epoch_fraction():
            return progress["epoch"] + progress.get("samples_in_epoch", 0) / len(train_data)

        def lr_factor(step):
            return learning_rate_factor(settings, step, epoch_fraction())

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        operational = {
            "output",
            "run_name",
            "runs_root",
            "max_steps",
            "save_every",
            "eval_every",
            "first_check_step",
            "log_every",
            "keep_checkpoints",
            "num_workers",
            "prefetch_factor",
            "loader_timeout_seconds",
            "tensorboard",
        }
        signature = {
            "protocol": 5,
            "seed": seed,
            "model": config["model"],
            "assembly_sha256": hashlib.sha256(
                (assembled / "assembly_report.json").read_bytes()
            ).hexdigest(),
            "build": train_data.fingerprint,
            "val_build": val_data.fingerprint if val_data else None,
            "settings": {k: v for k, v in settings.items() if k not in operational},
            "sampler": "window_shuffle_token_budget_v2",
            "torch": torch.__version__,
        }
        if resume:
            checkpoint = (
                output / "checkpoints" / (output / "checkpoints/latest").read_text().strip()
                if resume == "latest"
                else Path(resume)
            )
            progress = load_checkpoint(checkpoint, model, optimizer, scheduler, signature)
        if rank == 0:
            writer = SummaryWriter(
                str(output / "tensorboard"), purge_step=progress["step"] + 1 if resume else None
            )
            groups = tensorboard_groups(settings)
            setup_dashboard(writer, groups)
            writer.add_text(
                "run/config", "```json\n" + json.dumps(config, indent=2) + "\n```", progress["step"]
            )
            (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
            (output / "config.yaml").write_text(Path(config["config_path"]).read_text())
            (output / "signature.json").write_text(json.dumps(signature, indent=2))
            plan = {
                "contract_version": "v0.1",
                "artifact_kind": "training_plan",
                "status": "complete",
                "build_sha256": train_data.fingerprint,
                "recipe": signature,
                "recipe_sha256": digest(signature),
                "world_size": world,
                "sampling": {
                    "unit": "utterance",
                    "weights": "uniform",
                    "replacement": False,
                    "tail": "drop_if_fewer_than_world_size_samples",
                },
                "evaluation": train_data.manifest["evaluation"],
            }
            (output / "training_plan.json").write_text(json.dumps(plan, indent=2))
            print(
                json.dumps(
                    {
                        "initialized": True,
                        "world_size": world,
                        "precision": settings["precision"],
                        "attention": backend,
                        "predictor_attention": model.config.code_predictor_config._attn_implementation,
                        "train_samples": len(train_data),
                        "progress": progress,
                    }
                ),
                flush=True,
            )
        if eval_only:
            if val_data is None:
                raise ValueError("--eval-only requires an isolated validation build")
            metrics = validate(
                model,
                val_data,
                config["eval"]["batch_size"],
                device,
                settings["loss_reduction"],
                settings["residual_weight"],
                loader_settings=settings,
            )
            if rank == 0:
                print(json.dumps({"step": progress["step"], "val": metrics}), flush=True)
            return
        sampler = TokenBatchSampler(
            train_data.frame_lengths,
            train_data.token_lengths,
            settings["max_batch_frames"],
            settings["max_batch_tokens"],
            world_size=world,
            rank=rank,
            seed=seed,
            shuffle_window=settings["shuffle_window"],
            length_bucket_size=settings.get("length_bucket_size", 0),
            pad_to_longest=backend == "sdpa",
        )
        sampler.set_epoch(
            progress["epoch"],
            start_batch=progress["next_batch"],
        )
        workers = settings["num_workers"]
        loader = DataLoader(
            train_data,
            batch_sampler=sampler,
            collate_fn=collate,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=bool(workers),
            prefetch_factor=settings["prefetch_factor"] if workers else None,
            multiprocessing_context="spawn" if workers else None,
            timeout=settings["loader_timeout_seconds"] if workers else 0,
            generator=torch.Generator().manual_seed(seed + rank),
        )
        reader = iter(loader)
        model.train()
        usable_in_epoch = empty_epochs = 0
        skipped_since_log = discarded_since_log = 0
        last_saved_step = progress["step"] if resume else -1
        last_validated_step = -1
        while not finished(settings, progress):
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            batches = []
            data_wait_seconds = 0.0
            while len(batches) < settings["accumulation"]:
                read_started = time.perf_counter()
                try:
                    batch = next(reader)
                except StopIteration:
                    empty_epochs = empty_epochs + 1 if usable_in_epoch == 0 else 0
                    if empty_epochs >= 2:
                        raise RuntimeError(
                            "No usable distributed batches in two consecutive epochs"
                        )
                    usable_in_epoch = 0
                    progress["epoch"] += 1
                    progress["next_batch"] = 0
                    progress["samples_in_epoch"] = 0
                    if finished(settings, progress):
                        break
                    sampler.set_epoch(progress["epoch"])
                    reader = iter(loader)
                    batch = next(reader)
                data_wait_seconds += time.perf_counter() - read_started
                progress["next_batch"] += 1
                skipped_since_log += int(batch["skipped_samples"])
                usable, consumed = batch_health(batch, device)
                progress["samples_in_epoch"] = progress.get("samples_in_epoch", 0) + consumed
                if not usable:
                    discarded_since_log += len(batch.get("frame_lengths", ()))
                    continue
                usable_in_epoch += 1
                batches.append(batch)
            if not batches:
                break
            prepare_seconds = time.perf_counter() - started
            denominators = sum(
                loss_normalizers(b["frame_lengths"], settings["loss_reduction"]) for b in batches
            ).to(device)
            dist.all_reduce(denominators)
            counts = torch.tensor(
                [
                    sum(b["frame_lengths"].sum().item() + len(b["frame_lengths"]) for b in batches),
                    sum(b["frame_lengths"].sum().item() * 15 for b in batches),
                    sum(len(b["frame_lengths"]) for b in batches),
                    sum(b["audio_seconds"].sum().item() for b in batches),
                    skipped_since_log,
                    discarded_since_log,
                    sum((b["text_lengths"] + b["frame_lengths"] + 8).sum().item() for b in batches),
                    sum(
                        (b["text_lengths"] + b["frame_lengths"] + 8).max().item()
                        * len(b["frame_lengths"])
                        for b in batches
                    ),
                ],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(counts)
            optimizer.zero_grad(set_to_none=True)
            sums = torch.zeros(4, dtype=torch.float64, device=device)
            for i, batch in enumerate(batches):
                model.set_requires_gradient_sync(i == len(batches) - 1)
                batch = move(batch, device)
                predictions = model(batch)
                out = tts_loss(predictions, batch, model.eos, settings["loss_reduction"])
                del predictions
                loss = world * (
                    out["first_reduced_sum"] / denominators[0]
                    + settings["residual_weight"] * out["residual_reduced_sum"] / denominators[1]
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss")
                loss.backward()
                sums += torch.stack(
                    [
                        out[k].detach()
                        for k in (
                            "first_sum",
                            "residual_sum",
                            "first_reduced_sum",
                            "residual_reduced_sum",
                        )
                    ]
                )
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"])
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite gradient norm")
            optimizer.step()
            scheduler.step()
            progress["step"] += 1
            step = progress["step"]
            dist.all_reduce(sums)
            if step % settings["log_every"] == 0:
                torch.cuda.synchronize(device)
                # Every rank participates: rank 0 alone can hide an input bottleneck elsewhere.
                timing = torch.tensor(
                    [
                        time.perf_counter() - started,
                        data_wait_seconds,
                        prepare_seconds,
                        torch.cuda.max_memory_allocated(device) / 2**30,
                    ],
                    dtype=torch.float64,
                    device=device,
                )
                wait_mean = timing[1].clone()
                dist.all_reduce(wait_mean)
                dist.all_reduce(timing, op=dist.ReduceOp.MAX)
                elapsed, wait_seconds, prepare_seconds, peak_memory = timing.tolist()
            if rank == 0 and step % settings["log_every"] == 0:
                metrics = {
                    "first_ce": (sums[0] / counts[0]).item(),
                    "residual_ce": (sums[1] / counts[1]).item(),
                    "objective": (
                        sums[2] / denominators[0]
                        + settings["residual_weight"] * sums[3] / denominators[1]
                    ).item(),
                    "global_samples": counts[2].item(),
                    "epoch": epoch_fraction(),
                    "skipped_samples": counts[4].item(),
                    "discarded_samples": counts[5].item(),
                    "audio_seconds": counts[3].item(),
                    "codec_tokens": counts[0].item() + counts[1].item(),
                    "talker_tokens": counts[6].item(),
                    "padding_efficiency": (counts[6] / counts[7]).item()
                    if backend == "sdpa"
                    else 1.0,
                    "grad_norm": norm.item(),
                    "step_seconds": elapsed,
                    "data_wait_seconds": wait_seconds,
                    "data_wait_mean_seconds": wait_mean.item() / world,
                    "data_wait_fraction": wait_seconds / elapsed,
                    "batch_prepare_seconds": prepare_seconds,
                    "audio_seconds_per_second": counts[3].item() / elapsed,
                    "peak_memory_gib": peak_memory,
                    "lr_backbone": optimizer.param_groups[0]["lr"],
                    "lr_new": optimizer.param_groups[1]["lr"],
                }
                print(json.dumps({"step": step, **metrics}), flush=True)
                write_training(writer, metrics, step, groups)
                with (output / "metrics.jsonl").open("a") as journal:
                    journal.write(json.dumps({"step": step, "train": metrics}) + "\n")
            if step % settings["log_every"] == 0:
                skipped_since_log = discarded_since_log = 0
            final = finished(settings, progress)
            first_check = step == settings.get("first_check_step")
            if val_data is not None and (
                first_check or step % settings["eval_every"] == 0 or final
            ):
                if rank == 0:
                    print(
                        json.dumps({"validation_started": step, "samples": len(val_data)}),
                        flush=True,
                    )
                metrics = validate(
                    model,
                    val_data,
                    config["eval"]["batch_size"],
                    device,
                    settings["loss_reduction"],
                    settings["residual_weight"],
                    loader_settings=settings,
                )
                if rank == 0:
                    print(json.dumps({"step": step, "val": metrics}), flush=True)
                    write_validation(writer, metrics, step)
                    with (output / "metrics.jsonl").open("a") as journal:
                        journal.write(json.dumps({"step": step, "val": metrics}) + "\n")
                last_validated_step = step
            if first_check or step % settings["save_every"] == 0 or final:
                save_checkpoint(
                    output / "checkpoints",
                    model,
                    optimizer,
                    scheduler,
                    progress,
                    signature,
                    keep=settings["keep_checkpoints"],
                )
                last_saved_step = step
                if rank == 0:
                    writer.flush()
        # An epoch can end exactly at an accumulation boundary, discovered on the next read.
        if val_data is not None and progress["step"] != last_validated_step:
            metrics = validate(
                model,
                val_data,
                config["eval"]["batch_size"],
                device,
                settings["loss_reduction"],
                settings["residual_weight"],
                loader_settings=settings,
            )
            if rank == 0:
                write_validation(writer, metrics, progress["step"])
                with (output / "metrics.jsonl").open("a") as journal:
                    journal.write(json.dumps({"step": progress["step"], "val": metrics}) + "\n")
        if progress["step"] > last_saved_step:
            save_checkpoint(
                output / "checkpoints",
                model,
                optimizer,
                scheduler,
                progress,
                signature,
                keep=settings["keep_checkpoints"],
            )
        if rank == 0:
            (output / "completion.json").write_text(json.dumps(progress, indent=2))
    finally:
        stop_gpu_logging(gpu_monitor)
        # Stop spawned prefetch workers before tearing down the process group.
        if reader is not None and hasattr(reader, "_shutdown_workers"):
            reader._shutdown_workers()
        loader = reader = None
        if writer is not None:
            writer.close()
        dist.destroy_process_group()
