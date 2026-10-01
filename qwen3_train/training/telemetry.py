"""Grouped training diagnostics shared by live logging and journal replay."""

import os
import subprocess
import time
import warnings

GROUPS = {
    "optimization": {
        "train/loss": "objective",
        "train/epoch": "epoch",
        "train/first_ce": "first_ce",
        "train/residual_ce": "residual_ce",
        "train/lr_backbone": "lr_backbone",
        "train/lr_new": "lr_new",
        "train/grad_norm": "grad_norm",
    },
    "performance": {
        "train/audio_seconds_per_second": "audio_seconds_per_second",
        "performance/step_seconds": "step_seconds",
        "performance/data_wait_seconds": "data_wait_seconds",
        "performance/peak_memory_gib": "peak_memory_gib",
    },
    "batch": {
        "batch/global_samples": "global_samples",
        "batch/audio_seconds": "audio_seconds",
        "batch/codec_tokens": "codec_tokens",
        "batch/talker_tokens": "talker_tokens",
        "batch/padding_efficiency": "padding_efficiency",
        "batch/skipped_samples": "skipped_samples",
        "batch/discarded_samples": "discarded_samples",
    },
}


def tensorboard_groups(settings):
    groups = settings.get("tensorboard", {}).get("groups", list(GROUPS))
    if not isinstance(groups, list) or any(g not in GROUPS for g in groups):
        raise ValueError("train.tensorboard.groups must select optimization, performance, batch")
    if len(groups) != len(set(groups)):
        raise ValueError("Duplicate TensorBoard groups")
    return groups


def setup_dashboard(writer, groups):
    charts = {}
    if "optimization" in groups:
        charts["Optimization"] = {
            "Loss": ["Multiline", ["train/loss", "train/first_ce", "train/residual_ce"]],
            "Learning rates": ["Multiline", ["train/lr_backbone", "train/lr_new"]],
        }
    if "performance" in groups:
        charts["Performance"] = {
            "Step and data wait": [
                "Multiline",
                ["performance/step_seconds", "performance/data_wait_seconds"],
            ]
        }
    if charts:
        writer.add_custom_scalars(charts)


def write_training(writer, metrics, step, groups):
    for group in groups:
        for tag, key in GROUPS[group].items():
            if key in metrics:
                writer.add_scalar(tag, metrics[key], step)


def write_validation(writer, metrics, step):
    for key in ("loss", "first_ce", "residual_ce"):
        writer.add_scalar("val/" + key, metrics[key], step)


def start_gpu_logging(output, rank):
    """One NVML poller per node; keep all GPU samples beside the node logs."""
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return None
    directory = output / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    stream = (directory / f"gpu-rank-{rank}-{time.time_ns()}.csv").open("w")
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv",
        "--loop=10",
    ]
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        command += ["--id=" + os.environ["CUDA_VISIBLE_DEVICES"]]
    try:
        return subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT), stream
    except OSError as error:
        stream.close()
        warnings.warn(f"GPU telemetry unavailable: {error}", RuntimeWarning)
        return None


def stop_gpu_logging(monitor):
    if monitor is not None:
        process, stream = monitor
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        stream.close()
