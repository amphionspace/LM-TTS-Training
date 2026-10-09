"""Check real assembled LFM FSDP updates and exact optimizer/checkpoint resume on two GPUs."""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy
from transformers import AutoTokenizer

from lm_tts.data.batch import collate
from lm_tts.models.loading import load_model
from lm_tts.objectives.tts import tts_loss
from lm_tts.training.checkpoint import load_checkpoint, save_checkpoint
from lm_tts.training.distributed import initialize, move, shard
from lm_tts.training.optimizer import parameter_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembled-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = initialize(81)
    try:
        torch.cuda.set_per_process_memory_fraction(0.19, device)
        rank = dist.get_rank()
        model = load_model(args.assembled_model, use_speaker_embedding=False)
        model.train()
        for module in (model.talker.model, model.talker.code_predictor.model):
            module.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        tokenizer = AutoTokenizer.from_pretrained(args.assembled_model, local_files_only=True)
        raw = json.loads((args.assembled_model / "config.json").read_text())
        rows = []
        for text, frames in [("你好，这是语音合成测试。", 3 + rank), ("A small speech test.", 2)]:
            rows.append(
                {
                    "text_ids": [
                        raw["tts_bos_token_id"],
                        *tokenizer.encode(text, add_special_tokens=False),
                        raw["tts_eos_token_id"],
                    ],
                    "codes": torch.randint(0, 2048, (frames, 16)),
                }
            )
        batch = move(collate(rows), device)
        normalizers = torch.stack(
            [(batch["frame_lengths"] + 1).sum(), batch["frame_lengths"].sum() * 15]
        ).float()
        dist.all_reduce(normalizers)
        shard(
            model,
            device,
            MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        )
        optimizer = torch.optim.AdamW(
            parameter_groups(
                model,
                {
                    "lr": 3e-4,
                    "backbone_lr": 1e-4,
                    "text_embedding_lr_group": "backbone",
                },
            )
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

        def step():
            optimizer.zero_grad(set_to_none=True)
            stats = tts_loss(model(batch), batch, model.eos)
            loss = dist.get_world_size() * (
                stats["first_sum"] / normalizers[0] + 0.3 * stats["residual_sum"] / normalizers[1]
            )
            assert torch.isfinite(loss)
            loss.backward()
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad.to_local()).all(), name
            optimizer.step()
            scheduler.step()
            return float(loss.detach())

        first = step()
        signature = {"fixture": "lfm-fsdp-resume", "assembly": str(args.assembled_model.resolve())}
        checkpoint = save_checkpoint(
            args.output, model, optimizer, scheduler, {"step": 1}, signature
        )
        expected_loss = step()
        expected = {n: p.detach().to_local().cpu().clone() for n, p in model.named_parameters()}
        restored = load_checkpoint(checkpoint, model, optimizer, scheduler, signature)
        assert restored["step"] == 1
        actual_loss = step()
        largest = max(
            (p.detach().to_local().cpu() - expected[n]).abs().max().item()
            for n, p in model.named_parameters()
        )
        assert largest == 0, largest
        assert actual_loss == expected_loss, (actual_loss, expected_loss)
        result = {
            "rank": rank,
            "world_size": dist.get_world_size(),
            "first_loss": first,
            "resumed_loss": actual_loss,
            "resume_max_weight_difference": largest,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        (args.output / f"validation-rank-{rank}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
