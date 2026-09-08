"""Small synthetic FSDP2 environment check; run with torchrun, no model download."""
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
            for _ in range(2)
        ])
        self.head = nn.Linear(64, 8)

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return self.head(x)


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16")
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        device = torch.device("cuda", local_rank)
        value = torch.tensor(float(rank + 1), device=device)
        dist.all_reduce(value)
        assert value.item() == world * (world + 1) / 2
        torch.manual_seed(42)
        model = TinyModel().to(device)
        mesh = init_device_mesh("cuda", (world,))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for block in model.blocks:
            fully_shard(block, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = [p.detach().to_local().clone() for p in model.parameters()]
        torch.manual_seed(100 + rank)
        losses = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            x = torch.randn(2, 16, 64, device=device)
            target = torch.randn(2, 16, 8, device=device)
            loss = (model(x).float() - target).square().mean()
            assert torch.isfinite(loss).item()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            assert torch.isfinite(norm).item()
            optimizer.step()
            dist.all_reduce(loss.detach())
            losses.append(loss.item() / world)
        delta = sum((p.detach().to_local() - old).abs().sum() for p, old in zip(model.parameters(), before))
        dist.all_reduce(delta)
        assert torch.isfinite(delta).item() and delta.item() > 0
        if rank == 0:
            print(json.dumps({
                "status": "passed", "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda, "world_size": world,
                "gpu": torch.cuda.get_device_name(), "backend": dist.get_backend(),
                "steps": len(losses), "mean_losses": losses,
                "parameter_delta_l1": delta.item(),
            }, indent=2), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
