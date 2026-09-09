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
    parser.add_argument("--frozen-speaker", action="store_true")
    parser.add_argument("--qwen-protocol", action="store_true")
    parser.add_argument("--frozen-frontend", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.use_deterministic_algorithms(True)
    os.environ['FLASH_ATTENTION_DETERMINISTIC'] = '1'
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    try:
        world = dist.get_world_size()
        assert world >= 2
        torch.manual_seed(77)
        speaker_config = None
        if args.speaker:
            from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
            speaker_config = Qwen3TTSSpeakerEncoderConfig(enc_dim=64, mel_dim=8, enc_channels=[16, 16, 16, 16, 48])
        config = make_config(tiny=True)
        if args.qwen_protocol:
            config.lm_tts_input_protocol = "qwen3_non_streaming"
            config.lm_tts_text_projection = "identity"
            config.lm_tts_pad_token_id = 20
            config.lm_tts_role_ids = [21, 22, 23]
            config.codec_nothink_id = 2151
            config.codec_think_bos_id = 2152
            config.codec_think_eos_id = 2153
        if args.frozen_frontend:
            config.text_hidden_size = 128
            config.lm_tts_text_projection = "mlp"
            config.lm_tts_freeze_text_frontend = True
        config.lm_tts_freeze_speaker_encoder = args.frozen_speaker
        model = TTSModel(config, speaker_config).cuda()
        if args.bf16:
            config._attn_implementation = 'flash_attention_2'
            config.code_predictor_config._attn_implementation = 'flash_attention_2'
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        reference = copy.deepcopy(model)
        if args.bf16:
            for parameter in reference.parameters():
                parameter.data = parameter.data.bfloat16()
        mesh = init_device_mesh("cuda", (world,))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16 if args.bf16 else torch.float32,
                                      reduce_dtype=torch.float32)
        for blocks in [model.talker.model.layers, model.talker.code_predictor.model.layers]:
            for block in blocks:
                fully_shard(block, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        torch.manual_seed(19)
        per_rank = 2 if args.bf16 else 1
        rows = [{"text_ids": list(range(1, i + 3)), "codes": torch.randint(0, 2048, (i + 1, 16))}
                for i in range(2 * world * per_rank)]
        if args.speaker:
            for i, row in enumerate(rows):
                row["speaker_mels"] = torch.randn(24 + 3 * i, 8)
        def batch(items):
            result = {k: v.cuda() for k, v in collate(items).items()}
            if args.bf16 and args.speaker:
                result['speaker_mels'] = result['speaker_mels'].bfloat16()
            return result
        total_first = sum(len(r["codes"]) + 1 for r in rows)
        total_residual = sum(len(r["codes"]) * 15 for r in rows)
        out = reference(batch(rows))
        (out["first_sum"] / total_first + 0.3 * out["residual_sum"] / total_residual).backward()
        for micro in range(2):
            model.set_requires_gradient_sync(micro == 1)
            start = (micro * world + rank) * per_rank
            out = model(batch(rows[start:start + per_rank]))
            (world * (out["first_sum"] / total_first + 0.3 * out["residual_sum"] / total_residual)).backward()
        largest = 0.0
        squared_error, squared_target = 0.0, 0.0
        expected = dict(reference.named_parameters())
        for name, param in model.named_parameters():
            if not param.requires_grad:
                assert param.grad is None and expected[name].grad is None
                torch.testing.assert_close(param.full_tensor(), frozen[name], atol=0, rtol=0)
                continue
            actual = param.grad.full_tensor()
            target = expected[name].grad.float()
            if not args.bf16:
                torch.testing.assert_close(actual, target, atol=2e-6, rtol=2e-4)
            squared_error += (actual - target).square().sum().item()
            squared_target += target.square().sum().item()
            largest = max(largest, (actual - target).abs().max().item())
        relative_error = (squared_error / squared_target) ** .5
        assert relative_error < .03
        if rank == 0:
            print(json.dumps({"status": "passed", "world_size": world, "accumulation": 2,
                              "variable_lengths": True, "bf16": args.bf16, "speaker": args.speaker, "qwen_protocol": args.qwen_protocol, "frozen_frontend": args.frozen_frontend, "frozen_speaker": args.frozen_speaker, "max_gradient_absolute_error": largest,
                              "relative_gradient_error": relative_error}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
