"""Local objective statistics; distributed normalization belongs to the trainer."""

import torch
from torch.nn import functional as F

LOSS_REDUCTION_EXPONENTS = {"token": 1.0, "sample": 0.0, "sqrt": 0.5}


def loss_normalizers(frame_lengths, reduction, residual_groups=15):
    lengths = torch.stack([frame_lengths + 1, frame_lengths * residual_groups]).double()
    return lengths.pow(LOSS_REDUCTION_EXPONENTS[reduction]).sum(dim=1)


def tts_loss(predictions, batch, eos, reduction="token"):
    lengths, codes = batch["frame_lengths"], batch["codes"]
    groups = len(predictions["residual_logits"])
    last = (lengths + 1).cumsum(0) - 1
    labels = codes.new_full((len(codes) + len(lengths),), eos)
    valid = torch.ones(len(labels), dtype=torch.bool, device=codes.device)
    valid[last] = False
    labels[valid] = codes[:, 0]
    first_losses = F.cross_entropy(predictions["first_logits"].float(), labels, reduction="none")
    group_losses = torch.stack(
        [
            F.cross_entropy(logits.float(), codes[:, group + 1], reduction="none")
            for group, logits in enumerate(predictions["residual_logits"])
        ]
    )
    exponent = LOSS_REDUCTION_EXPONENTS[reduction] - 1
    first_weights = (lengths + 1).float().pow(exponent).repeat_interleave(lengths + 1)
    residual_weights = (lengths * groups).float().pow(exponent).repeat_interleave(lengths)
    group_sums = group_losses.sum(dim=1)
    return {
        "first_sum": first_losses.sum(),
        "residual_sum": group_sums.sum(),
        "first_reduced_sum": (first_losses * first_weights).sum(),
        "residual_reduced_sum": (group_losses.sum(dim=0) * residual_weights).sum(),
        "group_sums": group_sums.detach(),
        "first_count": codes.new_tensor(len(labels)),
        "frame_count": codes.new_tensor(len(codes)),
    }
