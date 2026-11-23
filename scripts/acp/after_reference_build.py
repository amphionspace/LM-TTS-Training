"""Submit one training run after the strict-reference pipeline completes its build."""

import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from lm_tts.artifacts import digest, file_hash
from lm_tts.config import read_yaml
from lm_tts.data.batch import collate
from lm_tts.training.codec_task import CodecTask
from lm_tts.training.config import load_config

from .api import jobs_url, request


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def validate(config, work):
    result = json.loads((work / "result.json").read_text())
    assert result["status"] == "complete"
    build_root = Path(config["data"]["build_root"])
    assert Path(result["build"]).resolve() == (build_root / "manifest.json").resolve()
    build = json.loads((build_root / "manifest.json").read_text())
    recipe = json.loads((build_root / "data_recipe.json").read_text())
    assert build["recipe_sha256"] == digest(recipe)
    assert read_yaml(config["data"]["config"]) == recipe
    assert build["split"]["method"] == "preserve_baseline_target_membership_v1"
    assembled = Path(config["model"]["assembled_model"])
    assert (assembled / "ASSEMBLY_COMPLETE").exists()
    report = json.loads((assembled / "assembly_report.json").read_text())
    full = json.loads((assembled / "config.json").read_text())
    assert report["text_initialization"] == "qwen-tts"
    assert report["text_projection_init"] == "pretrained"
    assert report["freeze_text_frontend"] and report["freeze_speaker_encoder"]
    assert full["talker_config"]["lm_tts_freeze_text_frontend"]
    assert full["talker_config"]["lm_tts_freeze_speaker_encoder"]
    for name, expected in report["artifact_sha256"].items():
        if name == "config.json" or (
            name.startswith("model") and name.endswith(".safetensors")
        ):
            assert file_hash(assembled / name) == expected, name
    task = CodecTask(config, full, assembled, rank=0, backend="flash_attention_2")
    checked = {}
    for role, dataset in [("train", task.train_data), ("validation", task.val_data)]:
        assert dataset is not None
        dataset.validate_budgets(
            config["train"]["max_batch_frames"], config["train"]["max_batch_tokens"]
        )
        checked[role] = []
        for slot, (binding, index) in enumerate(
            zip(dataset.bindings, dataset.indices, strict=True)
        ):
            assert (
                binding["merged_profile"]["speaker_mode"]
                == "codec_grid_strict3s_reference"
            )
            assert len(index) and len(index) == binding["ready_rows"]
            if role == "train":
                assert len(index) == result["training_rows"][binding["dataset_id"]]
            offset = int(dataset.offsets[slot])
            for i in np.unique(np.linspace(0, len(index) - 1, 8, dtype=int)):
                row = dataset[offset + int(i)]
                assert row is not None
                a, b = row["reference_interval"]
                assert b - a >= 38
                batch = collate([row])
                assert not batch["skipped_samples"].item()
                assert batch["supervised_frame_lengths"].item() == len(row["codes"]) - (
                    b - a
                )
                assert torch.isfinite(batch["speaker_embeddings"]).all()
            checked[role].append(
                dict(
                    dataset=binding["dataset_id"],
                    rows=len(index),
                    build_sha256=dataset.fingerprint,
                )
            )
    return dict(
        text_initialization="qwen-tts",
        text_projection="pretrained",
        freeze_text_frontend=True,
        checks=checked,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--feature-work", type=Path, required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    folder, work = args.folder.resolve(), args.feature_work.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / "monitor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def event(state, **details):
        value = dict(
            at=datetime.now(timezone.utc).isoformat(),
            pid=os.getpid(),
            state=state,
            **details,
        )
        write_json(folder / "status.json", value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    config_sha256 = file_hash(args.config)
    config = load_config(args.config)
    platform = read_yaml(args.config)["acp"]
    project = Path(config["paths"]["project"])
    pinned = json.loads((project / "source_manifest.json").read_text())["files"]
    feature_job = json.loads(
        sorted((work / "submissions").glob("*/job.json"))[-1].read_text()
    )["response"]["name"]
    try:
        while not (folder / "STOP").exists():
            progress = json.loads((work / "status.json").read_text())
            if progress["phase"] == "failed":
                raise RuntimeError(f"Data pipeline failed: {progress}")
            if (work / "result.json").exists():
                break
            try:
                remote = request(
                    "GET", f"{jobs_url(platform)}/{feature_job}", home=platform["home"]
                )
            except Exception as exc:
                event("waiting_for_build", progress=progress, status_error=str(exc))
            else:
                if remote["state"] in {"FAILED", "STOPPED", "CANCELED", "CANCELLED"}:
                    raise RuntimeError(f"Data task {feature_job}: {remote['state']}")
                event(
                    "waiting_for_build",
                    feature_job=feature_job,
                    feature_state=remote["state"],
                    progress=progress,
                )
            for _ in range(args.interval):
                if (folder / "STOP").exists():
                    break
                time.sleep(1)
        if (folder / "STOP").exists():
            event("stopped")
            return
        assert file_hash(args.config) == config_sha256, "Training configuration changed"
        event("validating_build")
        for name, expected in pinned.items():
            assert file_hash(project / name) == expected, name
        checked = validate(config, work)
        write_json(folder / "build-validation.json", checked)
        run = Path(config["train"]["output"])
        marker = folder / "submission-attempt.json"
        if marker.exists() or list((run / "submissions").glob("*/request.json")):
            raise RuntimeError(
                "Submission already attempted; reconcile before retrying"
            )
        write_json(
            marker,
            dict(
                at=datetime.now(timezone.utc).isoformat(),
                run=str(run),
                config_sha256=file_hash(args.config),
            ),
        )
        event("submitting")
        env = {
            **os.environ,
            "PYTHONPATH": str(project),
            "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
        }
        with (folder / "submit.log").open("a") as log:
            subprocess.run(
                [
                    config["paths"]["python"],
                    "-m",
                    "scripts.acp.submit",
                    "--config",
                    str(args.config.resolve()),
                    "--submit",
                ],
                cwd=project,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        record = json.loads(
            sorted((run / "submissions").glob("*/job.json"))[-1].read_text()
        )
        write_json(folder / "training-job.json", record)
        event("submitted", job=record["response"]["name"], run=str(run))
    except BaseException as exc:
        event("failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
