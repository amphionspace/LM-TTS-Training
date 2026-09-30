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
from test_unified_data import fixture
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from qwen3_train.data.build import bind_features, file_hash, tokenizer_identity
from qwen3_train.models.assembly import save_model
from qwen3_train.models.qwen import make_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="fp32")
    args = parser.parse_args()
    root = args.output.resolve()
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
    talker.lm_tts_freeze_text_frontend = True
    config = Qwen3TTSConfig(
        talker_config=talker.to_dict(),
        tts_model_type="base",
        tts_bos_token_id=200,
        tts_eos_token_id=201,
        speaker_encoder_config={"enc_dim": 64, "mel_dim": 8, "enc_channels": [16, 16, 16, 16, 48]},
    )
    model = Qwen3TTSForConditionalGeneration(config)
    assembled = root / "assembled"
    save_model(model, assembled)
    tokenizer = Tokenizer(
        WordLevel(
            {"[PAD]": 0, "[UNK]": 1, "hello": 2, **{str(i): i + 3 for i in range(8)}},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]"
    )
    tokenizer.save_pretrained(assembled)
    tensors = {}
    for key, value in model.speaker_encoder.state_dict().items():
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
    recipe, selection, path = fixture(root / "unified", embedding_dim=64, speaker_profile=profile)
    bind_features(
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
            "precision": args.precision,
            "max_steps": 4,
            "schedule_steps": 4,
            "warmup_steps": 1,
            "lr": 0.0001,
            "backbone_lr": 0.00001,
            "max_batch_frames": 12,
            "max_batch_tokens": 100,
            "accumulation": 2,
            "num_workers": 1,
            "log_every": 1,
            "save_every": 2,
            "keep_checkpoints": 2,
        },
        "eval": {"batch_size": 2},
    }

    def train(name, steps, resume=False):
        experiment["train"]["output"] = str(root / name)
        cfg = root / (name + ".yaml")
        cfg.write_text(yaml.safe_dump(experiment))
        argv = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "-m",
            "qwen3_train.train",
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
                str(assembled),
                "--output",
                str(destination),
            ],
            check=True,
            timeout=120,
        )
        weights.append(load_file(destination / "model.safetensors"))
    largest = 0.0
    for key in weights[0]:
        torch.testing.assert_close(weights[0][key], weights[1][key], atol=0, rtol=0)
        largest = max(largest, (weights[0][key] - weights[1][key]).abs().max().item())
    print(
        json.dumps(
            {
                "status": "passed",
                "precision": args.precision,
                "max_resume_weight_difference": largest,
                "world_size": 2,
            }
        )
    )


if __name__ == "__main__":
    main()
