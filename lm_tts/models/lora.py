"""Low-rank adapters that replace full fine-tuning of selected linear layers.

An adapted layer freezes its pretrained weight and learns W + (alpha / rank) * B @ A. B starts at
zero, so training begins from exactly the pretrained function. Exports merge every adapter back,
leaving the original module layout and parameter names."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LoRASpec:
    modules: list[str]  # Top-level modules to adapt.
    targets: list[str]  # Names of their nn.Linear layers, e.g. q_proj.
    rank: int
    alpha: float
    dropout: float

    def __post_init__(self):
        for key in ("modules", "targets"):
            names = getattr(self, key)
            if type(names) is not list or not names or not all(type(n) is str for n in names):
                raise ValueError(f"train.lora.{key} must be a nonempty list of names")
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("train.lora.rank must be a positive integer")
        if not all(
            type(v) in (int, float) and math.isfinite(v) for v in (self.alpha, self.dropout)
        ):
            raise ValueError("train.lora alpha and dropout must be finite numbers")
        if self.alpha <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("train.lora needs alpha > 0 and dropout in [0, 1)")


class LoRALinear(nn.Module):
    def __init__(self, base, spec):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.lora_A = nn.Parameter(base.weight.new_empty(spec.rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, spec.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = spec.alpha / spec.rank
        self.dropout = nn.Dropout(spec.dropout)

    def forward(self, x):
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return self.base(x) + self.scale * update

    @torch.no_grad()
    def merged(self):
        delta = (self.lora_B.double() @ self.lora_A.double()) * self.scale
        self.base.weight.add_(delta.to(self.base.weight.dtype))
        return self.base


def apply_lora(model, spec):
    matched = set()
    for root in spec.modules:
        names = [
            name
            for name, layer in model.get_submodule(root).named_modules()
            if type(layer) is nn.Linear and name.rpartition(".")[2] in spec.targets
        ]
        if not names:
            raise ValueError(f"No LoRA target is a linear layer of {root}")
        for name in names:
            path = f"{root}.{name}"
            model.set_submodule(path, LoRALinear(model.get_submodule(path), spec))
            matched.add(name.rpartition(".")[2])
    if unmatched := set(spec.targets) - matched:
        raise ValueError(f"LoRA targets match no linear layer: {sorted(unmatched)}")


def merge_lora(model):
    for name, layer in list(model.named_modules()):
        if isinstance(layer, LoRALinear):
            model.set_submodule(name, layer.merged())


def lora_parameters(model):
    return [
        p
        for layer in model.modules()
        if isinstance(layer, LoRALinear)
        for p in (layer.lora_A, layer.lora_B)
    ]
