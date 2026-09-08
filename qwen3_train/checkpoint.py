import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict


def save_checkpoint(root, model, optimizer, scheduler, progress, signature):
    rank = dist.get_rank()
    target = Path(root) / f"step-{progress['step']:08d}"
    temporary = target.with_name(target.name + ".incomplete")
    if target.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}")
    if rank == 0:
        temporary.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    model_state, optim_state = get_state_dict(model, optimizer)
    dcp.save({"model": model_state, "optimizer": optim_state}, checkpoint_id=temporary / "distributed")
    torch.save({"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}, temporary / f"rng-{rank}.pt")
    if rank == 0:
        (temporary / "metadata.json").write_text(json.dumps({"progress": progress, "signature": signature,
            "scheduler": scheduler.state_dict(), "world_size": dist.get_world_size()}, indent=2))
    dist.barrier()
    if rank == 0:
        (temporary / "COMPLETE").write_text("ok\n")
        temporary.rename(target)
        latest = Path(root) / "latest.tmp"
        latest.write_text(target.name + "\n")
        latest.replace(Path(root) / "latest")
    dist.barrier()
    return target


def load_checkpoint(path, model, optimizer, scheduler, signature):
    path = Path(path)
    if not (path / "COMPLETE").is_file():
        raise ValueError(f"Not a completed checkpoint: {path}")
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata["signature"] != signature or metadata["world_size"] != dist.get_world_size():
        raise ValueError("Exact resume requires the same model, data, world size, batch, optimizer and schedule settings")
    model_state, optim_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optimizer": optim_state}
    dcp.load(state, checkpoint_id=path / "distributed")
    set_state_dict(model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optimizer"])
    scheduler.load_state_dict(metadata["scheduler"])
    # These pickle files are generated locally by this trainer; only resume trusted runs.
    rng = torch.load(path / f"rng-{dist.get_rank()}.pt", map_location="cpu", weights_only=False)
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state(rng["cuda"])
    return metadata["progress"]
