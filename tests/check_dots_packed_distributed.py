"""Run with torchrun --nproc_per_node=2: packed DiT + FSDP + checkpoint regression."""

import copy
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from test_dots import batch_of, row, tiny_model
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from lm_tts.models.dots.modules import DiT


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    dist.init_process_group("nccl", timeout=timedelta(seconds=240))
    torch.manual_seed(177)
    reference = tiny_model()
    reference.velocity_field_predictor = DiT(32, 128, {**reference.config["DiT"], "num_heads": 2})
    for module in [
        reference.velocity_field_predictor,
        reference.hidden_proj,
        reference.latent_proj,
        reference.coordinate_proj,
        reference.xvec_proj,
    ]:
        module.requires_grad_(False)
    reference = reference.cuda().bfloat16().train()
    model = copy.deepcopy(reference)
    model.enable_activation_checkpointing()
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for unit in model.sharding_modules():
        fully_shard(unit, mesh=mesh, mp_policy=policy)
    fully_shard(model, mesh=mesh, mp_policy=policy)
    for iteration in range(2):
        lengths = (
            ([1, 29, 2] if rank == 0 else [17, 3])
            if not iteration
            else ([13, 2] if rank == 0 else [1, 7, 26, 2])
        )
        batch = {
            key: value.cuda() if torch.is_tensor(value) else value
            for key, value in batch_of([row(n, 10 + i) for i, n in enumerate(lengths)]).items()
        }
        # Match mixed precision inputs as the trainer does.
        batch = {
            k: v.bfloat16() if torch.is_tensor(v) and v.is_floating_point() else v
            for k, v in batch.items()
        }
        outputs = []
        for current in [reference, model]:
            current.zero_grad(set_to_none=True)
            torch.manual_seed(200 + rank + iteration)
            result = current(batch)
            loss = (
                result["velocity"].float().square().mean()
                + result["eos_logits"].float().square().mean()
            )
            loss.backward()
            outputs.append({k: result[k].detach() for k in ["velocity", "eos_logits"]})
        for key in outputs[0]:
            torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=0.015, rtol=0.015)
        for (name, expected), (_, actual) in zip(
            reference.named_parameters(), model.named_parameters()
        ):
            if expected.grad is None:
                assert actual.grad is None, name
                continue
            grad = expected.grad.float()
            dist.all_reduce(grad)
            grad /= dist.get_world_size()
            observed = actual.grad.full_tensor().float()
            torch.testing.assert_close(observed, grad, atol=0.003, rtol=0.05, msg=name)
        if rank == 0:
            print(
                f"iteration {iteration}: unequal lengths/counts, bf16 frozen DiT, FSDP + checkpoint PASS",
                flush=True,
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
