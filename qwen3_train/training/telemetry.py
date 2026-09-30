"""Grouped training diagnostics shared by live logging and journal replay."""

GROUPS = {
    "optimization": {
        "train/loss": "objective",
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


def write_training(writer, metrics, step, groups, *, skip_tags=()):
    for group in groups:
        for tag, key in GROUPS[group].items():
            if key in metrics and tag not in skip_tags:
                writer.add_scalar(tag, metrics[key], step)


def write_validation(writer, metrics, step):
    for key in ("loss", "first_ce", "residual_ce"):
        writer.add_scalar("val/" + key, metrics[key], step)
