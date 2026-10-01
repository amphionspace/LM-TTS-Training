from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from qwen3_train.training.distributed import batch_health


def worker(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=90)
    )
    try:
        model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(1, 1))
        empty_or_healthy = {"skipped_samples": torch.tensor(1 if rank == 0 else 0)}
        if rank == 1:
            empty_or_healthy["frame_lengths"] = torch.tensor([4])
        usable, consumed = batch_health(empty_or_healthy, "cpu")
        assert not usable and consumed == 2
        healthy = {"skipped_samples": torch.tensor(0), "frame_lengths": torch.tensor([4])}
        usable, consumed = batch_health(healthy, "cpu")
        assert usable and consumed == 2
        model(torch.ones(1, 1)).sum().backward()
        assert torch.isfinite(model.module.weight.grad).all()
    finally:
        dist.destroy_process_group()


def test_empty_rank_skips_collectively_and_next_batch_can_backward(tmp_path):
    mp.spawn(worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)
