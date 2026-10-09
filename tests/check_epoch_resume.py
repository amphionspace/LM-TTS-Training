"""GPU integration check for epoch schedules, corrupt rows and prefetched resume."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import lance
import pyarrow as pa
import torch
import yaml
from check_training_resume import prepare
from safetensors.torch import load_file

from lm_tts.data.split import split_build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--model-family", choices=["qwen", "lfm"], default="qwen")
    args = parser.parse_args()
    root = args.output.resolve()
    prepare(root, args.precision, 16, model_family=args.model_family)
    manifest_path = root / "build/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    binding = manifest["bindings"][0]
    table = lance.dataset(
        binding["merged"]["table_path"], version=binding["merged"]["lance_version"]
    )
    rows = table.to_table().to_pylist()
    rows[0]["codec_codes"][0][0] = 2048
    rows[1]["text_ids"][0] = 256
    updated = lance.write_dataset(
        pa.Table.from_pylist(rows, schema=table.schema),
        binding["merged"]["table_path"],
        mode="overwrite",
    )
    binding["merged"]["lance_version"] = updated.version
    manifest_path.write_text(json.dumps(manifest))
    split_build(manifest_path, fraction=0.125, seed=42)
    for name in ("continuous", "resumed"):
        path = root / f"{name}.yaml"
        config = yaml.safe_load(path.read_text())
        config["train"].update(
            epochs=3,
            max_steps=None,
            schedule_steps=None,
            warmup_steps=None,
            length_bucket_size=7,
            scheduler={"name": "wsd", "warmup_ratio": 0.1, "decay_ratio": 0.1, "min_lr_ratio": 0.1},
            first_check_step=1,
            save_every=2500,
            eval_every=2500,
            keep_checkpoints=3,
        )
        config["data"].update(
            build=str(root / "build/train/manifest.json"),
            val_build=str(root / "build/validation/manifest.json"),
            evaluation="train_isolated",
        )
        path.write_text(yaml.safe_dump(config))

    launcher = root / "fail_validation.py"
    launcher.write_text("""from lm_tts.training import validation
from lm_tts.train import main

def fail(*args, **kwargs):
    raise RuntimeError('injected validation failure')

if __name__ == "__main__":
    validation.validate = fail
    main()
""")

    def train(name, extra=(), *, fail_validation=False):
        with (
            root
            / (
                name
                + ("-failure" if fail_validation else "-partial" if "--max-steps" in extra else "")
                + ".log"
            )
        ).open("w") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=2",
                    *([str(launcher)] if fail_validation else ["-m", "lm_tts.train"]),
                    "--config",
                    str(root / f"{name}.yaml"),
                    *extra,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=not fail_validation,
                timeout=900,
            )

        if fail_validation:
            assert result.returncode != 0
            checkpoint = root / name / "checkpoints/step-00000001"
            assert (checkpoint / "COMPLETE").is_file()
            metadata = json.loads((checkpoint / "metadata.json").read_text())
            assert metadata["progress"]["step"] == 1 and metadata["validation_pending"]
            status = json.loads((root / name / "validation/step-00000001.json").read_text())
            assert status["status"] == "failed"

    train("continuous")
    train("resumed", fail_validation=True)
    train("resumed", ["--max-steps", "2", "--resume", "latest"])
    path = root / "resumed.yaml"
    config = yaml.safe_load(path.read_text())
    # Loader queues are disposable: only batches consumed by completed updates are committed.
    config["train"].update(num_workers=2, prefetch_factor=3)
    path.write_text(yaml.safe_dump(config))
    train("resumed", ["--resume", "latest"])
    weights, histories = [], []
    for name in ("continuous", "resumed"):
        checkpoints = root / name / "checkpoints"
        checkpoint = checkpoints / (checkpoints / "latest").read_text().strip()
        destination = root / (name + "-export")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.export_checkpoint",
                "--checkpoint",
                str(checkpoint),
                "--assembled-model",
                str(root / "assembled"),
                "--output",
                str(destination),
            ],
            check=True,
        )
        weights.append(load_file(destination / "model.safetensors"))
        from lm_tts.models.loading import load_model

        restored = load_model(destination, attn_implementation="sdpa")
        for key, value in restored.state_dict().items():
            torch.testing.assert_close(value, weights[-1][key], atol=0, rtol=0)
        if args.model_family == "lfm":
            from lm_tts.inference.codec import generate_codes

            codes = generate_codes(restored, [200, 2, 201], max_new_tokens=2, min_new_tokens=2)
            assert codes.shape == (2, 16)
            continuation = generate_codes(
                restored, [200, 2, 201], codes, max_new_tokens=1, min_new_tokens=1
            )
            assert continuation.shape == (1, 16)
        del restored
        progress = json.loads((root / name / "completion.json").read_text())
        assert progress["epoch"] == 3
        metrics = [
            json.loads(line) for line in (root / name / "metrics.jsonl").read_text().splitlines()
        ]
        assert any(row["step"] == progress["step"] and "val" in row for row in metrics)
        assert any(row["step"] == 1 and "val" in row for row in metrics)
        assert (checkpoints / "step-00000001/COMPLETE").exists()
        histories.append([row for row in metrics if "train" in row])
    for key in weights[0]:
        torch.testing.assert_close(weights[0][key], weights[1][key], atol=0, rtol=0)
    for left, right in zip(*histories, strict=True):
        assert left["step"] == right["step"]
        for key in (
            "epoch",
            "global_samples",
            "codec_tokens",
            "lr_new",
            "lr_backbone",
            "objective",
            "grad_norm",
        ):
            assert left["train"][key] == right["train"][key], (left["step"], key)
    report = {
        "status": "passed",
        "model_family": args.model_family,
        "validation_failure_retried_without_repeated_update": True,
        "export_reload_verified": True,
        "precision": args.precision,
        "world_size": 2,
        "epochs": 3,
        "max_resume_weight_difference": 0,
        "prefetch_workers_changed_on_resume": True,
        "per_step_lr_loss_samples_identical": True,
        "bad_payload_injected": True,
    }
    (root / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
