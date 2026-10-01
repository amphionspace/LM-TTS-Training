"""Validation objective statistics with globally exact sample coverage."""

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from ..data.batch import collate, val_batches
from ..objectives.tts import loss_normalizers, tts_loss
from ..training.distributed import batch_health, move


@torch.no_grad()
def validate(
    model,
    dataset,
    batch_size,
    device,
    loss_reduction="token",
    residual_weight=0.3,
    *,
    loader_settings=None,
):
    model.eval()
    totals = torch.zeros(25, dtype=torch.float64, device=device)
    settings = loader_settings or {}
    workers = settings.get("num_workers", 0)
    plan = list(val_batches(dataset, batch_size, dist.get_world_size(), dist.get_rank()))
    # A tiny holdout cannot amortize spawning workers and filling their queues.
    if len(plan) <= workers * settings.get("prefetch_factor", 2):
        workers = 0
    loader = DataLoader(
        dataset,
        batch_sampler=[indices for indices, _ in plan],
        collate_fn=collate,
        num_workers=workers,
        pin_memory=True,
        prefetch_factor=settings.get("prefetch_factor", 2) if workers else None,
        multiprocessing_context="spawn" if workers else None,
        timeout=settings.get("loader_timeout_seconds", 300) if workers else 0,
        # Validation worker startup must not advance the training RNG state.
        generator=torch.Generator().manual_seed(dist.get_rank()),
    )
    for (_, real), batch in zip(plan, loader, strict=True):
        batch = move(batch, device)
        if real:
            totals[23] += batch["skipped_samples"]
        if not batch_health(batch, device)[0]:
            if real:
                totals[24] += len(batch.get("frame_lengths", ()))
            continue
        out = tts_loss(model(batch), batch, model.eos, loss_reduction)
        if real:
            totals[0] += out["first_sum"]
            totals[1] += out["first_count"]
            totals[2] += out["residual_sum"]
            totals[3] += out["frame_count"]
            totals[4:19] += out["group_sums"]
            totals[19:21] += torch.stack([out["first_reduced_sum"], out["residual_reduced_sum"]])
            totals[21:23] += loss_normalizers(batch["frame_lengths"], loss_reduction)
    dist.all_reduce(totals)
    if totals[1] == 0:
        raise ValueError("No usable validation samples remain")
    metrics = {
        "skipped_samples": totals[23].item(),
        "discarded_samples": totals[24].item(),
        "first_ce": (totals[0] / totals[1]).item(),
        "residual_ce": (totals[2] / (totals[3] * 15)).item(),
        "loss": (totals[19] / totals[21] + residual_weight * totals[20] / totals[22]).item(),
    }
    metrics.update({f"codebook_{i + 1}_ce": (totals[4 + i] / totals[3]).item() for i in range(15)})
    if loss_reduction != "token":
        metrics.update(
            {
                f"first_{loss_reduction}_ce": (totals[19] / totals[21]).item(),
                f"residual_{loss_reduction}_ce": (totals[20] / totals[22]).item(),
            }
        )
    model.train()
    return metrics
