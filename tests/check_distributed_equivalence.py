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
from reference_model import ReferenceTTSModel
from qwen3_train.model import TTSModel, loss_normalizers, make_config
from qwen3_train.train import validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", action="store_true")
    parser.add_argument("--frozen-speaker", action="store_true")
    parser.add_argument("--qwen-protocol", action="store_true")
    parser.add_argument("--frozen-frontend", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--loss-reduction", choices=["token", "sample", "sqrt"], default="token")
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
        model_class = TTSModel if args.bf16 else ReferenceTTSModel
        model = model_class(config, speaker_config).cuda()
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
        batch_sizes = [1 + i % 3 for i in range(2 * world)]
        rows = [{"text_ids": list(range(1, i + 3)), "codes": torch.randint(0, 2048, (i + 1, 16))}
                for i in range(sum(batch_sizes))]
        if args.speaker:
            for i, row in enumerate(rows):
                row["speaker_mels"] = torch.randn(24 + 3 * i, 8)
        def batch(items):
            result = {k: v.cuda() for k, v in collate(items).items()}
            if args.bf16 and args.speaker:
                result['speaker_mels'] = result['speaker_mels'].bfloat16()
            return result
        exponent = {"token": 1, "sample": 0, "sqrt": .5}[args.loss_reduction]
        total_first = sum((len(r["codes"]) + 1) ** exponent for r in rows)
        total_residual = sum((len(r["codes"]) * 15) ** exponent for r in rows)
        out = reference(batch(rows), loss_reduction=args.loss_reduction)
        expected_metrics = {"first_ce": out['first_sum'].item() / sum(len(r['codes']) + 1 for r in rows),
                            "residual_ce": out['residual_sum'].item() / sum(15 * len(r['codes']) for r in rows)}
        if args.loss_reduction != 'token':
            expected_metrics.update({f'first_{args.loss_reduction}_ce': out['first_reduced_sum'].item() / total_first,
                                     f'residual_{args.loss_reduction}_ce': out['residual_reduced_sum'].item() / total_residual})
        (out["first_reduced_sum"] / total_first + 0.3 * out["residual_reduced_sum"] / total_residual).backward()
        batches = []
        for micro in range(2):
            batch_index = micro * world + rank
            start = sum(batch_sizes[:batch_index])
            batches.append(batch(rows[start:start + batch_sizes[batch_index]]))
        normalizers = sum(loss_normalizers(b['frame_lengths'], args.loss_reduction) for b in batches)
        dist.all_reduce(normalizers)
        for micro, local_batch in enumerate(batches):
            model.set_requires_gradient_sync(micro == 1)
            out = model(local_batch, loss_reduction=args.loss_reduction)
            (world * (out["first_reduced_sum"] / normalizers[0]
                      + 0.3 * out["residual_reduced_sum"] / normalizers[1])).backward()
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
        metrics = validate(model, rows, 3, torch.device('cuda', rank), args.loss_reduction)
        for key, expected_value in expected_metrics.items():
            torch.testing.assert_close(metrics[key], expected_value, atol=2e-3 if args.bf16 else 2e-6, rtol=1e-5)
        if rank == 0:
            print(json.dumps({"status": "passed", "world_size": world, "accumulation": 2,
                              "variable_lengths": True, "variable_batch_sizes": True, "bf16": args.bf16, "speaker": args.speaker, "qwen_protocol": args.qwen_protocol, "frozen_frontend": args.frozen_frontend, "frozen_speaker": args.frozen_speaker, "max_gradient_absolute_error": largest,
                              "relative_gradient_error": relative_error, "loss_reduction": args.loss_reduction,
                              "validation_metrics": metrics}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
