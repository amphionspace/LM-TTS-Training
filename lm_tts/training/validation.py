"""Validation status is independent of immutable optimizer checkpoints."""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ..evaluation.validation import validate
from .telemetry import write_validation


def validation_complete(output, step):
    path = Path(output) / "validation" / f"step-{step:08d}.json"
    return path.is_file() and json.loads(path.read_text())["status"] == "complete"


def run_validation(model, dataset, config, device, output, step, writer):
    rank = dist.get_rank()
    path = Path(output) / "validation" / f"step-{step:08d}.json"

    def status(value, **details):
        if rank == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"step": step, "status": value, **details}, indent=2))
            temporary.replace(path)

    status("running")
    # A retry must leave the next optimizer update's random streams unchanged.
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    was_training = model.training
    try:
        with torch.random.fork_rng(devices=[device]):
            settings = config["train"]
            if rank == 0:
                print(json.dumps({"validation_started": step, "samples": len(dataset)}), flush=True)
            metrics = validate(
                model,
                dataset,
                config["eval"]["batch_size"],
                device,
                settings["loss_reduction"],
                settings["residual_weight"],
                loader_settings=settings,
            )
        if rank == 0:
            print(json.dumps({"step": step, "val": metrics}), flush=True)
            write_validation(writer, metrics, step)
            writer.flush()
            with (Path(output) / "metrics.jsonl").open("a") as journal:
                journal.write(json.dumps({"step": step, "val": metrics}) + "\n")
        status("complete", metrics=metrics)
        return metrics
    except Exception as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        model.train(was_training)
