"""GPU check: a 2-rank checkpoint initializes a 1-rank anneal run with identical state."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from check_training_resume import prepare
from safetensors.torch import load_file
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

from lm_tts.training.schedule import learning_rate_factor

LAUNCHER = """import sys
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
from lm_tts.training import engine
from lm_tts.train import main

original = engine.initialize_from
destination = sys.argv.pop(1)

def capture(source, model, optimizer, signature):
    origin = original(source, model, optimizer, signature)
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    model_state, optim_state = get_state_dict(model, optimizer, options=options)
    if dist.get_rank() == 0:
        lrs = [group["lr"] for group in optimizer.param_groups]
        torch.save({"model": model_state, "optimizer": optim_state, "lrs": lrs}, destination)
    return origin

if __name__ == "__main__":
    engine.initialize_from = capture
    main()
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    prepare(root, "fp32", 64)
    (root / "capture.py").write_text(LAUNCHER)

    def train(name, ranks, extra=(), capture=None):
        entry = [str(root / "capture.py"), str(capture)] if capture else ["-m", "lm_tts.train"]
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={ranks}",
            *entry,
            "--config",
            str(root / f"{name}.yaml"),
            *extra,
        ]
        with (root / f"{name}-{len(list(root.glob(name + '-*.log')))}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)

    train("continuous", 2, ["--max-steps", "4"])
    source = root / "continuous/checkpoints/step-00000004"
    dcp_to_torch_save(source / "distributed", root / "source.pt")
    expected = torch.load(root / "source.pt", weights_only=False)

    base = yaml.safe_load((root / "continuous.yaml").read_text())
    for name in ("anneal", "anneal-resumed"):
        config = json.loads(json.dumps(base))
        config["train"].update(
            run_name=name,
            init_from={"checkpoint": str(source), "load_optimizer": True},
            epochs=1,
            max_steps=None,
            schedule_steps=None,
            warmup_steps=None,
            scheduler={"name": "anneal", "decay_epochs": 0.5, "min_lr_ratio": 0.1},
            accumulation=4,
            save_every=1,
            keep_checkpoints=None,
        )
        (root / f"{name}.yaml").write_text(yaml.safe_dump(config))
    train("anneal", 1, capture=root / "initialized.pt")
    actual = torch.load(root / "initialized.pt", weights_only=False)
    assert set(actual["model"]) == set(expected["model"])
    for key, value in expected["model"].items():
        torch.testing.assert_close(actual["model"][key], value, atol=0, rtol=0)
    assert set(actual["optimizer"]["state"]) == set(expected["optimizer"]["state"])
    for key, state in expected["optimizer"]["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(
                torch.as_tensor(actual["optimizer"]["state"][key][field]),
                torch.as_tensor(value),
                atol=0,
                rtol=0,
            )
    # The checkpoint carries the source schedule's decayed LR; the new run must not inherit it.
    assert actual["lrs"][1] < base["train"]["lr"]

    rows = [
        json.loads(line)["train"]
        for line in (root / "anneal/metrics.jsonl").read_text().splitlines()
        if "train" in json.loads(line)
    ]
    completion = json.loads((root / "anneal/completion.json").read_text())
    fraction = rows[-1]["epoch"]
    assert completion["epoch"] == 0 and 0.5 <= fraction < 1, (completion, fraction)
    settings = yaml.safe_load((root / "anneal.yaml").read_text())["train"]
    for row in rows:
        factor = learning_rate_factor(settings, 0, row["epoch"])
        assert row["lr_new"] == pytest.approx(settings["lr"] * factor)
        assert row["lr_backbone"] == pytest.approx(settings["backbone_lr"] * factor)
    assert rows[-1]["lr_new"] == pytest.approx(settings["lr"] * 0.1)

    # The anneal run itself resumes through the ordinary signature-checked path.
    train("anneal-resumed", 1, ["--max-steps", "1"])
    train("anneal-resumed", 1, ["--resume", "latest"])
    for name in ("anneal", "anneal-resumed"):
        checkpoints = root / name / "checkpoints"
        checkpoint = checkpoints / (checkpoints / "latest").read_text().strip()
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
                str(root / f"{name}-export"),
            ],
            check=True,
        )
    left = load_file(root / "anneal-export/model.safetensors")
    right = load_file(root / "anneal-resumed-export/model.safetensors")
    for key in left:
        torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)
    report = {
        "status": "passed",
        "source_world_size": 2,
        "target_world_size": 1,
        "model_and_optimizer_identical": True,
        "anneal_steps": completion["step"],
        "final_epoch_fraction": fraction,
        "anneal_resume_weight_difference": 0,
    }
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
