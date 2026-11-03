"""Local flow-matching and EOS statistics for continuous-latent models; the trainer normalizes them
by global patch and utterance counts."""

import torch
from torch.nn import functional as F


def supervised_patches(patch_lengths, reference_intervals):
    """Mask over packed patches: False inside each utterance's [start, end) reference span."""
    starts = (patch_lengths.cumsum(0) - patch_lengths).repeat_interleave(patch_lengths)
    position = torch.arange(len(starts), device=patch_lengths.device) - starts
    begin, end = reference_intervals.repeat_interleave(patch_lengths, dim=0).unbind(1)
    return (position < begin) | (position >= end)


def eos_weights(patch_lengths, reference_intervals):
    """Per-patch EOS targets and weights: the patch that ends each utterance is the positive and
    weighs 0.5 even inside a reference span, as the codec reference mask keeps EOS; the other
    supervised patches share 0.5, so every utterance contributes weight 1."""
    last = patch_lengths.cumsum(0) - 1
    targets = torch.zeros(int(patch_lengths.sum()), dtype=torch.long, device=patch_lengths.device)
    targets[last] = 1
    negative = supervised_patches(patch_lengths, reference_intervals)
    negative[last] = False
    counts = torch.zeros(len(patch_lengths), device=patch_lengths.device)
    counts.index_add_(
        0,
        torch.arange(len(patch_lengths), device=patch_lengths.device).repeat_interleave(
            patch_lengths
        ),
        negative.float(),
    )
    weights = negative * (0.5 / counts.clamp_min(1)).repeat_interleave(patch_lengths)
    weights[last] = torch.where(counts > 0, 0.5, 1.0)
    return targets, weights


def flow_loss(predictions, batch):
    lengths, reference = batch["frame_lengths"], batch["reference_intervals"]
    supervised = supervised_patches(lengths, reference)
    errors = (predictions["velocity"].float() - predictions["target"]).pow(2).mean(dim=(1, 2))
    targets, weights = eos_weights(lengths, reference)
    eos = F.cross_entropy(predictions["eos_logits"].float(), targets, reduction="none")
    return {
        "fm_sum": (errors * supervised).sum(),
        "eos_sum": (eos * weights).sum(),
        "patch_count": supervised.sum(),
        "sample_count": torch.tensor(len(lengths), device=lengths.device),
    }
