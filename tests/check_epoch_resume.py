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
    args = parser.parse_args()
    root = args.output.resolve()
    prepare(root, args.precision, 16)
    manifest_path = root / "build/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    binding = manifest["bindings"][0]
    table = lance.dataset(
        binding["merged"]["table_path"], version=binding["merged"]["lance_version"]
    )
    rows = table.to_table().to_pylist()
    rows[0]["codec_codes"][0][0] = 2048
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

    def train(name, extra=()):
        with (root / (name + ("-partial" if "--max-steps" in extra else "") + ".log")).open(
            "w"
        ) as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=2",
                    "-m",
                    "lm_tts.train",
                    "--config",
                    str(root / f"{name}.yaml"),
                    *extra,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=900,
            )

    train("continuous")
    train("resumed", ["--max-steps", "2"])
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
