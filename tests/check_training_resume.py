"""Exercise the real trainer/reader/checkpointer; compare resumed and uninterrupted weights."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from safetensors.torch import load_file
from test_merged_data import merged_fixture
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from lm_tts.artifacts import file_hash
from lm_tts.data.build import tokenizer_identity
from lm_tts.data.merged import bind_merged
from lm_tts.models.assembly.qwen import save_model
from lm_tts.models.qwen import make_config


def prepare(
    root,
    precision,
    num_rows=8,
    *,
    train_text_frontend=False,
    use_speaker_embedding=True,
    reference_masking=False,
    model_family="qwen",
):
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(123)
    talker = make_config(tiny=True)
    talker.spk_id = {}
    talker.codec_language_id = {}
    talker.lm_tts_input_protocol = "qwen3_non_streaming"
    talker.lm_tts_pad_token_id = 0
    talker.lm_tts_role_ids = [3, 4, 5]
    talker.codec_nothink_id = 2151
    talker.codec_think_bos_id = 2152
    talker.codec_think_eos_id = 2153
    talker.lm_tts_freeze_speaker_encoder = True
    talker.lm_tts_freeze_text_frontend = not train_text_frontend
    config = Qwen3TTSConfig(
        talker_config=talker.to_dict(),
        tts_model_type="base",
        tts_bos_token_id=200,
        tts_eos_token_id=201,
        speaker_encoder_config={"enc_dim": 64, "mel_dim": 8, "enc_channels": [16, 16, 16, 16, 48]},
    )
    assembled = root / "assembled"
    if model_family == "lfm":
        from safetensors.torch import save_file
        from test_lfm import tiny_config

        from lm_tts.models.lfm import LfmTTSModel

        talker = tiny_config()
        model = LfmTTSModel(talker)
        assembled.mkdir()
        save_file(model.state_dict(), assembled / "model.safetensors")
        (assembled / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "lfm2_tts",
                    "talker_config": model.config.to_dict(),
                    "tts_bos_token_id": 200,
                    "tts_eos_token_id": 201,
                }
            )
        )
        use_speaker_embedding = False
        train_text_frontend = True
    else:
        model = Qwen3TTSForConditionalGeneration(config)
        save_model(model, assembled)
    tokenizer = Tokenizer(
        WordLevel(
            {"[PAD]": 0, "[UNK]": 1, "hello": 2, **{str(i): i + 3 for i in range(num_rows)}},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]"
    )
    tokenizer.save_pretrained(assembled)
    tensors = {}
    for key, value in (
        model.speaker_encoder.state_dict().items() if model.speaker_encoder is not None else []
    ):
        tensors["speaker_encoder." + key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "sha256": hashlib.sha256(
                value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
    profile = {"model": {"tensors": tensors}}
    (assembled / "assembly_report.json").write_text(
        json.dumps(
            {
                "artifact_sha256": {
                    p.name: file_hash(p)
                    for p in assembled.iterdir()
                    if p.suffix in {".json", ".safetensors"}
                }
            }
        )
    )
    (assembled / "ASSEMBLY_COMPLETE").write_text("ok\n")
    recipe, selection, path = merged_fixture(
        root / "unified", embedding_dim=64, speaker_profile=profile, num_rows=num_rows
    )
    bind_merged(
        recipe,
        root / "build",
        selection,
        path,
        file_hash(path),
        tokenizer_identity(assembled),
        tokenizer,
    )
    experiment = {
        "seed": 42,
        "model": {"assembled_model": str(assembled), "activation_checkpointing": True},
        "data": {"build": str(root / "build/manifest.json"), "evaluation": "no_holdout"},
        "train": {
            "precision": precision,
            "max_steps": 4,
            "schedule_steps": 4,
            "warmup_steps": 1,
            "lr": 0.0001,
            "backbone_lr": 0.00001,
            "max_batch_frames": max(12, num_rows + 1),
            "max_batch_tokens": max(100, num_rows + 16),
            "accumulation": 2,
            "num_workers": 1,
            "log_every": 1,
            "save_every": 2,
            "keep_checkpoints": 2,
        },
        "eval": {"batch_size": 2},
    }

    if train_text_frontend:
        experiment["train"].update(
            text_embedding_lr_group="backbone", keep_checkpoints=None, save_every=1
        )
    if not use_speaker_embedding:
        experiment["model"]["use_speaker_embedding"] = False
    if reference_masking:
        from test_reference_training import reference_fixture

        build, _, _ = reference_fixture(
            root / "reference-case",
            speaker_profile=profile,
            tokenizer_info=tokenizer_identity(assembled),
            tokenizer=tokenizer,
        )
        experiment["data"] = {
            "build": str(build / "train/manifest.json"),
            "val_build": str(build / "validation/manifest.json"),
            "evaluation": "train_isolated",
        }
        experiment["train"].update(
            mask_reference=True,
            max_batch_frames=64,
            max_batch_tokens=128,
            eval_every=2,
            keep_checkpoints=None,
        )
    for name in ("continuous", "resumed"):
        experiment["train"]["runs_root"] = str(root)
        experiment["train"]["run_name"] = name
        cfg = root / (name + ".yaml")
        cfg.write_text(yaml.safe_dump(experiment))


def compare(root, precision, world_size):
    weights = []
    for name in ["continuous", "resumed"]:
        destination = root / (name + "-export")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.export_checkpoint",
                "--checkpoint",
                str(root / name / "checkpoints/step-00000004"),
                "--assembled-model",
                str(root / "assembled"),
                "--output",
                str(destination),
            ],
            check=True,
            timeout=180,
        )
        weights.append(load_file(destination / "model.safetensors"))
    largest = 0.0
    for key in weights[0]:
        torch.testing.assert_close(weights[0][key], weights[1][key], atol=0, rtol=0)
        largest = max(largest, (weights[0][key] - weights[1][key]).abs().max().item())
    baseline = load_file(root / "assembled/model.safetensors")
    config = json.loads((root / "assembled/config.json").read_text())
    text_frozen = config["talker_config"]["lm_tts_freeze_text_frontend"]
    text_prefixes = ("talker.model.text_embedding.", "talker.text_projection.")
    frozen = ("speaker_encoder.",) + (text_prefixes if text_frozen else ())
    for key in baseline:
        if key.startswith(frozen):
            torch.testing.assert_close(baseline[key], weights[0][key], atol=0, rtol=0)
    updated = [
        "talker.model.layers.",
        "talker.model.codec_embedding.",
        "talker.codec_head.",
        "talker.code_predictor.",
    ]
    if not text_frozen:
        updated.extend(text_prefixes)
        for name in ("continuous", "resumed"):
            for step in range(1, 5):
                assert (root / name / f"checkpoints/step-{step:08d}/COMPLETE").exists()
    for prefix in updated:
        if not any(
            not torch.equal(baseline[key], weights[0][key])
            for key in baseline
            if key.startswith(prefix)
        ):
            raise AssertionError(f"No optimizer update in {prefix}")
    result = {
        "status": "passed",
        "precision": precision,
        "max_resume_weight_difference": largest,
        "world_size": world_size,
        "updated_modules": updated,
        "frozen_modules_unchanged": True,
        "checkpoint_metadata": {
            name: json.loads((root / name / "checkpoints/step-00000004/metadata.json").read_text())
            for name in ("continuous", "resumed")
        },
    }
    (root / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "checkpoint_metadata"}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="fp32")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--train-text-frontend", action="store_true")
    parser.add_argument("--no-speaker-embedding", action="store_true")
    parser.add_argument("--reference-masking", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    if args.reference_masking and (
        args.rows != 8 or args.world_size > 5 or args.no_speaker_embedding
    ):
        parser.error(
            "reference masking requires 8 fixture rows, at most 5 ranks and speaker conditioning"
        )
    if (
        not 8 <= args.rows <= 128
        or args.world_size < 1
        or (not args.compare_only and args.world_size > args.rows)
    ):
        parser.error("rows must be 8..128 and cover every rank")
    root = args.output.resolve()
    if args.compare_only:
        compare(root, args.precision, args.world_size)
        return
    prepare(
        root,
        args.precision,
        args.rows,
        train_text_frontend=args.train_text_frontend,
        use_speaker_embedding=not args.no_speaker_embedding,
        reference_masking=args.reference_masking,
    )
    if args.prepare_only:
        print(json.dumps({"prepared": str(root), "rows": args.rows}))
        return

    def train(name, steps, resume=False):
        cfg = root / (name + ".yaml")
        argv = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={args.world_size}",
            "-m",
            "lm_tts.train",
            "--config",
            str(cfg),
            "--max-steps",
            str(steps),
        ]
        if resume:
            argv += ["--resume", "latest"]
        with (root / (name + f"-{steps}.log")).open("w") as stream:
            subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=240)

    train("continuous", 4)
    train("resumed", 2)
    train("resumed", 4, resume=True)
    compare(root, args.precision, args.world_size)


if __name__ == "__main__":
    main()
