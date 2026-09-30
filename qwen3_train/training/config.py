"""Validate an experiment before model loading or distributed initialization."""

import math
from pathlib import Path

from ..config import read_yaml
from ..objectives.tts import LOSS_REDUCTION_EXPONENTS
from .precision import training_precision
from .telemetry import tensorboard_groups


def load_config(path):
    config = read_yaml(
        path, keys=("paths", "environment", "seed", "model", "data", "train", "eval")
    )
    model, data, train = (config[k] for k in ("model", "data", "train"))
    if "run_name" in train or "runs_root" in train:
        name = train.get("run_name")
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or Path(name).name != name
            or not train.get("runs_root")
        ):
            raise ValueError("Specify runs_root and a directory-safe run_name")
        output = Path(train["runs_root"]).resolve() / name
        if train.get("output") and Path(train["output"]).resolve() != output:
            raise ValueError("output conflicts with runs_root/run_name")
        train["output"] = str(output)
    elif not train.get("output"):
        raise ValueError("Training requires runs_root/run_name or an explicit output")
    config["config_path"] = str(Path(path).resolve())
    train.setdefault("precision", "bf16")
    training_precision(train, model)
    tensorboard_groups(train)
    if train.get("loss_reduction", "token") not in LOSS_REDUCTION_EXPONENTS:
        raise ValueError("train.loss_reduction must be token, sqrt or sample")
    defaults = {
        "loss_reduction": "token",
        "num_workers": 4,
        "prefetch_factor": 2,
        "keep_checkpoints": 2,
        "shuffle_window": 65536,
        "accumulation": 1,
        "log_every": 10,
        "eval_every": 500,
        "save_every": 500,
        "residual_weight": 0.3,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "loader_timeout_seconds": 300,
    }
    for key, value in defaults.items():
        train.setdefault(key, value)
    for key in (
        "max_batch_frames",
        "max_batch_tokens",
        "max_steps",
        "schedule_steps",
        "accumulation",
        "prefetch_factor",
        "shuffle_window",
        "log_every",
        "eval_every",
        "save_every",
        "loader_timeout_seconds",
    ):
        if type(train[key]) is not int or train[key] < 1:
            raise ValueError(f"train.{key} must be a positive integer")
    if (
        type(train["num_workers"]) is not int
        or train["num_workers"] < 0
        or train["max_steps"] > train["schedule_steps"]
    ):
        raise ValueError("Invalid worker count or stopping/schedule horizon")
    if (
        type(train["warmup_steps"]) is not int
        or not 0 <= train["warmup_steps"] <= train["schedule_steps"]
    ):
        raise ValueError("warmup_steps must be within schedule_steps")
    if type(train["keep_checkpoints"]) is not int or train["keep_checkpoints"] < 1:
        raise ValueError("keep_checkpoints must be a positive integer")
    for key in ("lr", "backbone_lr", "grad_clip", "residual_weight", "weight_decay"):
        if (
            not isinstance(train[key], (int, float))
            or not math.isfinite(train[key])
            or train[key] < 0
        ):
            raise ValueError(f"train.{key} must be finite and nonnegative")
    if not train["lr"] or not train["backbone_lr"] or not train["grad_clip"]:
        raise ValueError("Learning rates and gradient clipping must be positive")
    if data.get("evaluation") != "no_holdout" and not data.get("val_build"):
        raise ValueError("Specify data.evaluation=no_holdout or a separately isolated val_build")
    if not model.get("assembled_model") or not data.get("build"):
        raise ValueError("Training requires an assembled model and a published unified data build")
    config.setdefault("eval", {}).setdefault("batch_size", 8)
    if type(config["eval"]["batch_size"]) is not int or config["eval"]["batch_size"] < 1:
        raise ValueError("eval.batch_size must be a positive integer")
    return config
