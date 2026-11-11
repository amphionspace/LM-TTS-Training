"""Optimizer updates and resume orchestration over published unified builds."""

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..artifacts import digest
from ..data.sampler import TokenBatchSampler
from ..models.loading import load_model
from .checkpoint import initialize_from, load_checkpoint, save_checkpoint
from .distributed import batch_health, initialize, move, shard
from .optimizer import parameter_groups
from .precision import training_precision
from .schedule import finished, learning_rate_factor
from .telemetry import (
    setup_dashboard,
    start_gpu_logging,
    stop_gpu_logging,
    tensorboard_groups,
    write_training,
)
from .validation import run_validation, validation_complete


def task_for(model_type):
    if model_type == "dots_tts":
        from .flow_task import FlowTask

        return FlowTask
    from .codec_task import CodecTask

    return CodecTask


SAMPLER_ORDERS = {
    "window_shuffle_token_budget_v2": "window",
    "feistel_uniform_token_budget_v3": "feistel",
}


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
        if not (assembled / "ASSEMBLY_COMPLETE").is_file():
            raise ValueError("Training requires a completed assembly, not an inference export")
        task = task_for(full_config.get("model_type"))(
            config, full_config, assembled, rank, backend
        )
        train_data, val_data = task.train_data, task.val_data
        if val_data is not None:
            # Isolation is a separately published data decision, never inferred from storage split=train.
            if (
                val_data.manifest["evaluation"] != "heldout"
                or train_data.manifest["evaluation"] != "train_isolated"
                or not train_data.manifest.get("split_sha256")
                or train_data.manifest["split_sha256"] != val_data.manifest.get("split_sha256")
            ):
                raise ValueError("Validation requires builds published from an isolated selection")
        model = load_model(
            assembled,
            load_weights=True,
            attn_implementation=backend,
            use_speaker_embedding=config["model"].get("use_speaker_embedding", True),
        )
        task.configure(model)
        if config["model"].get("activation_checkpointing", True):
            model.enable_activation_checkpointing()
        output = Path(settings["output"]).resolve()
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            if not resume and (output / "checkpoints/latest").exists():
                raise ValueError("Run already has checkpoints; resume or choose a new output")
        dist.barrier()
        gpu_monitor = start_gpu_logging(output, rank)
        shard(model, device, policy)
        optimizer = torch.optim.AdamW(
            parameter_groups(model, settings),
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
            "sampler": settings.get("sampler", "window_shuffle_token_budget_v2"),
            "torch": torch.__version__,
        }
        validation_pending = False
        if resume:
            checkpoint = (
                output / "checkpoints" / (output / "checkpoints/latest").read_text().strip()
                if resume == "latest"
                else Path(resume)
            )
            progress = load_checkpoint(checkpoint, model, optimizer, scheduler, signature)
            metadata = json.loads((checkpoint / "metadata.json").read_text())
            validation_pending = metadata.get("validation_pending", False)
        elif settings.get("init_from"):
            origin = initialize_from(settings["init_from"], model, optimizer, signature)
            # Loaded param groups carry the source learning rate; restart this run's schedule.
            for group, base, lr in zip(
                optimizer.param_groups, scheduler.base_lrs, scheduler.get_last_lr(), strict=True
            ):
                group["initial_lr"], group["lr"] = base, lr
            if rank == 0:
                (output / "initialized_from.json").write_text(json.dumps(origin, indent=2))
                print(json.dumps({"initialized_from": origin}), flush=True)
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
                        **task.describe(model),
                        "train_samples": len(train_data),
                        "progress": progress,
                    }
                ),
                flush=True,
            )
        if eval_only:
            if val_data is None:
                raise ValueError("--eval-only requires an isolated validation build")
            metrics = task.validate(model, val_data, device)
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
            order=SAMPLER_ORDERS[settings.get("sampler", "window_shuffle_token_budget_v2")],
        )
        sampler.set_epoch(
            progress["epoch"],
            start_batch=progress["next_batch"],
            start_sample=progress.get("samples_in_epoch"),
        )
        workers = settings["num_workers"]
        loader = DataLoader(
            train_data,
            batch_sampler=sampler,
            collate_fn=task.collate,
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
        if resume and val_data is not None:
            step = progress["step"]
            state = [validation_complete(output, step) if rank == 0 else None]
            dist.broadcast_object_list(state, src=0)
            completed = state[0]
            retry = (
                validation_pending
                or step == settings.get("first_check_step")
                or step % settings["eval_every"] == 0
            )
            if completed:
                last_validated_step = step
            elif retry:
                run_validation(task, model, val_data, device, output, step, writer)
                last_validated_step = step
        while not finished(settings, progress, epoch_fraction()):
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
                    if finished(settings, progress, epoch_fraction()):
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
            counts = torch.tensor(
                [
                    sum(len(b["frame_lengths"]) for b in batches),
                    sum(b["audio_seconds"].sum().item() for b in batches),
                    skipped_since_log,
                    discarded_since_log,
                ],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(counts)
            split = (progress["step"] + 1) % settings["log_every"] == 0
            task.begin_step(model, batches, device, split)
            optimizer.zero_grad(set_to_none=True)
            for i, batch in enumerate(batches):
                model.set_requires_gradient_sync(i == len(batches) - 1)
                loss = task.loss(model, move(batch, device), world)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss")
                loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"])
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite gradient norm")
            optimizer.step()
            scheduler.step()
            progress["step"] += 1
            step = progress["step"]
            task.end_step()
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
                lrs = [group["lr"] for group in optimizer.param_groups]
                metrics = {
                    **task.metrics(),
                    "global_samples": counts[0].item(),
                    "epoch": epoch_fraction(),
                    "skipped_samples": counts[2].item(),
                    "discarded_samples": counts[3].item(),
                    "audio_seconds": counts[1].item(),
                    "grad_norm": norm.item(),
                    "step_seconds": elapsed,
                    "data_wait_seconds": wait_seconds,
                    "data_wait_mean_seconds": wait_mean.item() / world,
                    "data_wait_fraction": wait_seconds / elapsed,
                    "batch_prepare_seconds": prepare_seconds,
                    "audio_seconds_per_second": counts[1].item() / elapsed,
                    "peak_memory_gib": peak_memory,
                    **(
                        {"lr_backbone": lrs[0], "lr_new": lrs[1]}
                        if len(lrs) == 2
                        else {"lr": lrs[0]}
                    ),
                }
                print(json.dumps({"step": step, **metrics}), flush=True)
                write_training(writer, metrics, step, groups)
                with (output / "metrics.jsonl").open("a") as journal:
                    journal.write(json.dumps({"step": step, "train": metrics}) + "\n")
            if step % settings["log_every"] == 0:
                skipped_since_log = discarded_since_log = 0
            final = finished(settings, progress, epoch_fraction())
            first_check = step == settings.get("first_check_step")
            validation_due = val_data is not None and (
                first_check or step % settings["eval_every"] == 0 or final
            )
            if first_check or step % settings["save_every"] == 0 or final or validation_due:
                save_checkpoint(
                    output / "checkpoints",
                    model,
                    optimizer,
                    scheduler,
                    progress,
                    signature,
                    keep=settings["keep_checkpoints"],
                    validation_pending=validation_due,
                )
                last_saved_step = step
                if rank == 0:
                    writer.flush()
            if validation_due:
                run_validation(task, model, val_data, device, output, step, writer)
                last_validated_step = step
        # Epoch exhaustion may only be discovered by reading the next batch.
        validation_due = val_data is not None and progress["step"] != last_validated_step
        if progress["step"] > last_saved_step:
            save_checkpoint(
                output / "checkpoints",
                model,
                optimizer,
                scheduler,
                progress,
                signature,
                keep=settings["keep_checkpoints"],
                validation_pending=validation_due,
            )
        if validation_due:
            run_validation(task, model, val_data, device, output, progress["step"], writer)
        if rank == 0:
            (output / "completion.json").write_text(json.dumps(progress, indent=2))
    except BaseException:
        # A failed rank must let torchrun terminate peers, rather than wait for
        # collectives in destroy_process_group while peers are still in forward.
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(1)
    finally:
        stop_gpu_logging(gpu_monitor)
        # Stop spawned prefetch workers before tearing down the process group.
        if reader is not None and hasattr(reader, "_shutdown_workers"):
            reader._shutdown_workers()
        loader = reader = None
        if writer is not None:
            writer.close()
        dist.destroy_process_group()
