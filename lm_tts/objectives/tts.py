"""Local objective statistics; distributed normalization belongs to the trainer."""

import torch
from torch.nn import functional as F

LOSS_REDUCTION_EXPONENTS = {"token": 1.0, "sample": 0.0, "sqrt": 0.5}


def loss_normalizers(frame_lengths, reduction, residual_groups=15):
    lengths = torch.stack([frame_lengths + 1, frame_lengths * residual_groups]).double()
    return lengths.pow(LOSS_REDUCTION_EXPONENTS[reduction]).sum(dim=1)


def tts_loss(predictions, batch, eos, reduction="token", samples=None):
    """samples (bool per utterance) adds "selected": the same statistics, detached, over only those
    utterances, taken from the same per-position losses."""
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
    supervised = batch.get("supervised_frame_lengths", lengths)
    if "codec_loss_mask" in batch:
        mask = batch["codec_loss_mask"]
        first_mask = torch.ones_like(labels, dtype=torch.bool)
        first_mask[valid] = mask  # EOS remains supervised for every utterance.
        first_losses = first_losses.masked_fill(~first_mask, 0)
        group_losses = group_losses.masked_fill(~mask.unsqueeze(0), 0)
    exponent = LOSS_REDUCTION_EXPONENTS[reduction] - 1
    first_weights = (supervised + 1).float().pow(exponent).repeat_interleave(lengths + 1)
    residual_weights = (supervised * groups).float().pow(exponent).repeat_interleave(lengths)

    def statistics(first, group, frames):
        group_sums = group.sum(dim=1)
        return {
            "first_sum": first.sum(),
            "residual_sum": group_sums.sum(),
            "first_reduced_sum": (first * first_weights).sum(),
            "residual_reduced_sum": (group.sum(dim=0) * residual_weights).sum(),
            "group_sums": group_sums.detach(),
            "first_count": (frames + 1).sum(),
            "frame_count": frames.sum(),
        }

    out = statistics(first_losses, group_losses, supervised)
    if samples is not None:
        with torch.no_grad():
            out["selected"] = statistics(
                first_losses.masked_fill(~samples.repeat_interleave(lengths + 1), 0),
                group_losses.masked_fill(~samples.repeat_interleave(lengths).unsqueeze(0), 0),
                supervised[samples],
            )
    return out
