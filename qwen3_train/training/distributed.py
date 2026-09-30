"""PyTorch distributed setup and FSDP policy, independent of the job platform."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard


def initialize(seed):
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=device)
    return device


def shard(model, device, policy):
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    model.float().to(device)
    for blocks in [model.talker.model.layers, model.talker.code_predictor.model.layers]:
        for block in blocks:
            fully_shard(block, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=True)


def move(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
