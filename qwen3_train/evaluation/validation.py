"""Validation objective statistics with globally exact sample coverage."""

import torch
import torch.distributed as dist

from ..data.batch import collate, val_batches
from ..objectives.tts import loss_normalizers, tts_loss
from ..training.distributed import move


@torch.no_grad()
def validate(model, dataset, batch_size, device, loss_reduction="token", residual_weight=0.3):
    model.eval()
    totals = torch.zeros(23, dtype=torch.float64, device=device)
    for indices, real in val_batches(dataset, batch_size, dist.get_world_size(), dist.get_rank()):
        batch = move(collate(dataset.__getitems__(indices)), device)
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
    metrics = {
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
