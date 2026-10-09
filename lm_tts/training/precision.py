"""Compute precision is independent of the offline feature extraction precision."""

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy


def training_precision(settings, model_settings):
    precision = settings.setdefault("precision", "bf16")
    if precision not in {"bf16", "fp32"}:
        raise ValueError("train.precision must be bf16 or fp32")
    backend = "flash_attention_2" if precision == "bf16" else "sdpa"
    requested = model_settings.get("attn_implementation")
    if requested is not None and requested != backend:
        raise ValueError(
            f"train.precision={precision} requires model.attn_implementation={backend}"
        )
    model_settings["attn_implementation"] = backend
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    return MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32), backend
