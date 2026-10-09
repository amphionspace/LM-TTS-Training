"""Learning-rate factors and stopping rules for step- and epoch-based runs."""

import math


def finished(settings, progress):
    return (
        settings.get("max_steps") is not None and progress["step"] >= settings["max_steps"]
    ) or (settings.get("epochs") is not None and progress["epoch"] >= settings["epochs"])


def learning_rate_factor(settings, step, epoch_fraction):
    schedule = settings.get("scheduler", {"name": "cosine"})
    warmup_ratio = schedule.get("warmup_ratio")
    if warmup_ratio is not None:
        fraction = epoch_fraction / settings["epochs"]
        if fraction < warmup_ratio:
            return fraction / warmup_ratio
    else:
        warmup = settings["warmup_steps"]
        if step < warmup:
            return (step + 1) / max(1, warmup)
    if schedule["name"] == "wsd":
        fraction = epoch_fraction / settings["epochs"]
        decay = schedule["decay_ratio"]
        fraction = min(1.0, max(0.0, (fraction - (1 - decay)) / decay))
        floor = schedule["min_lr_ratio"]
    else:
        fraction = min(1.0, (step - warmup) / max(1, settings["schedule_steps"] - warmup))
        floor = 0.1
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * fraction))
