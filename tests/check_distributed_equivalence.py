"""Compare FSDP gradients against a global-batch unsharded reference."""
import argparse
import copy
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from qwen3_train.data import collate
from qwen3_train.model import TTSModel, make_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    try:
        world = dist.get_world_size()
        assert world == 2
        torch.manual_seed(77)
        speaker_config = None
        if args.speaker:
            from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
            speaker_config = Qwen3TTSSpeakerEncoderConfig(enc_dim=64, mel_dim=8, enc_channels=[16, 16, 16, 16, 48])
        model = TTSModel(make_config(tiny=True), speaker_config).cuda()
        reference = copy.deepcopy(model)
        mesh = init_device_mesh("cuda", (world,))
        policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
        for blocks in [model.talker.model.layers, model.talker.code_predictor.model.layers]:
            for block in blocks:
                fully_shard(block, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        torch.manual_seed(19)
        rows = [{"text_ids": list(range(1, i + 3)), "codes": torch.randint(0, 2048, (i + 1, 16))} for i in range(4)]
        if args.speaker:
            for row in rows:
                row["speaker_mels"] = torch.randn(32, 8)
        def batch(items):
            return {k: v.cuda() for k, v in collate(items).items()}
        total_first = sum(len(r["codes"]) + 1 for r in rows)
        total_residual = sum(len(r["codes"]) * 15 for r in rows)
        out = reference(batch(rows))
        (out["first_sum"] / total_first + 0.3 * out["residual_sum"] / total_residual).backward()
        for micro in range(2):
            model.set_requires_gradient_sync(micro == 1)
            out = model(batch([rows[micro * world + rank]]))
            (world * (out["first_sum"] / total_first + 0.3 * out["residual_sum"] / total_residual)).backward()
        largest = 0.0
        expected = dict(reference.named_parameters())
        for name, param in model.named_parameters():
            actual = param.grad.full_tensor()
            target = expected[name].grad
            torch.testing.assert_close(actual, target, atol=2e-6, rtol=2e-4)
            largest = max(largest, (actual - target).abs().max().item())
        if rank == 0:
            print(json.dumps({"status": "passed", "world_size": world, "accumulation": 2,
                              "variable_lengths": True, "speaker": args.speaker, "max_gradient_absolute_error": largest}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
