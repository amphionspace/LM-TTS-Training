"""Wait for published features, build and validate once, then submit one training job."""

import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lm_tts.artifacts import file_hash
from lm_tts.config import read_yaml

from .api import jobs_url, request


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def published_features(recipe):
    manifests = []
    for binding in recipe["bindings"]:
        path = Path(recipe["root"]) / binding["dots_manifest"]
        if not path.exists():
            return None
        manifest = json.loads(path.read_text())
        if manifest.get("status") != "complete":
            return None
        if manifest["dataset_id"] != binding["dataset_id"] or manifest["kind"] != "dots_vae":
            raise ValueError(f"Unexpected feature identity: {path}")
        selected = manifest.get("selection", {})
        if selected.get("mode") == "selection_branch":
            selection_path = Path(recipe["root"]) / selected["manifest_path"]
            if file_hash(selection_path) != selected["manifest_sha256"]:
                raise ValueError("Feature selection manifest changed")
            selection = json.loads(selection_path.read_text())
            (output,) = (
                o for o in selection["outputs"] if o["dataset_id"] == binding["dataset_id"]
            )
            if (
                selection["status"] != "complete"
                or selection["selection_id"] != selected["selection_id"]
                or manifest["rows"] != output["selected_rows"]
                or manifest["rows"] != selected["target_count"]
            ):
                raise ValueError("Features differ from their published selection")
        manifests.append(manifest)
    if len({m["profile_id"] for m in manifests}) != 1:
        raise ValueError("Feature profiles differ")
    return manifests


def validate_build(config):
    from transformers import AutoTokenizer

    from lm_tts.data.dots import DotsFeatureDataset, Layout, pairable

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["assembled_model"], local_files_only=True, fix_mistral_regex=False
    )
    report, pools = {}, {}
    for role, key in (("train", "build"), ("validation", "val_build")):
        dataset = DotsFeatureDataset(config["data"][key], Layout(tokenizer))
        dataset.validate_budgets(
            config["train"]["max_batch_frames"], config["train"]["max_batch_tokens"]
        )
        report[role] = []
        pools[role] = {}
        for slot, (binding, index) in enumerate(
            zip(dataset.bindings, dataset.indices, strict=True)
        ):
            if not len(index):
                raise ValueError("Empty training/validation binding")
            ref = dataset.references.get(slot)
            if ref is not None:
                positions = np.searchsorted(ref["rows"], index["row"])
                if np.any(positions >= len(ref["rows"])) or np.any(
                    ref["rows"][positions] != index["row"]
                ):
                    raise ValueError("Targets missing from reference pool")
                if not pairable(ref["group"], ref["text"], ref["long"])[positions].all():
                    raise ValueError("Target without a usable reference")
                pools[role][binding["dataset_id"]] = ref["rows"]
            report[role].append(
                {
                    "dataset": binding["dataset_id"],
                    "rows": len(index),
                    "hours": float(index["duration"].sum() / 3600),
                    "speakers": binding.get("speakers"),
                    "max_patches": int(index["frames"].max()),
                    "max_tokens_with_audio": int(
                        (index["tokens"] + index["frames"]).max() + dataset.layout.overhead
                    ),
                }
            )
            # Exercise real Lance reads, reference selection and collation across each binding.
            from lm_tts.data.dots import DotsCollator

            offset = int(dataset.offsets[slot])
            for i in np.unique(np.linspace(0, len(index) - 1, 8, dtype=int)):
                batch = DotsCollator(dataset.layout)([dataset[offset + int(i)]])
                if not all(
                    np.isfinite(batch[k].numpy()).all()
                    for k in ("vae_mean", "vae_logstd", "xvectors")
                ):
                    raise ValueError("Non-finite feature in build smoke check")
        del dataset
    for name, rows in pools["train"].items():
        if np.intersect1d(rows, pools["validation"][name]).size:
            raise ValueError("Training and validation share reference audio rows")
    return report


def launch(config_path, folder, event):
    config = read_yaml(config_path)
    project = Path(config["paths"]["project"])
    python = config["paths"]["python"]
    env = {
        **os.environ,
        "PYTHONPATH": str(project),
        "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
    }
    marker = folder / "submission-attempt.json"
    run = Path(config["train"]["runs_root"]) / config["train"]["run_name"]
    if marker.exists() or list((run / "submissions").glob("*/request.json")):
        raise RuntimeError("A submission was already attempted; reconcile it before any retry")
    build = Path(config["data"]["build_root"])
    if not build.exists():
        event("building")
        with (folder / "build.log").open("a") as log:
            subprocess.run(
                [python, "scripts/build_unified.py", "--config", str(config_path)],
                cwd=project,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    elif not all((build / role / "manifest.json").exists() for role in ("train", "validation")):
        raise RuntimeError("Partial build exists; inspect it instead of overwriting")
    from lm_tts.artifacts import digest

    recipe = read_yaml(config["data"]["config"])
    # create_build stores only recipe fields, not the inherited training defaults.
    actual_recipe = json.loads((build / "data_recipe.json").read_text())
    if any(recipe.get(k) != v for k, v in actual_recipe.items()):
        raise ValueError("Existing build belongs to another data recipe")
    manifest = json.loads((build / "manifest.json").read_text())
    if manifest["recipe_sha256"] != digest(actual_recipe):
        raise ValueError("Build recipe changed")
    event("validating")
    report = validate_build(config)
    report["source_coverage"] = {b["dataset_id"]: b["coverage"] for b in manifest["bindings"]}
    write_json(folder / "build-validation.json", report)
    # Persist intent BEFORE POST. A lost response must never cause another allocation.
    write_json(marker, {"at": datetime.now(timezone.utc).isoformat(), "run": str(run)})
    event("submitting")
    with (folder / "submit.log").open("a") as log:
        subprocess.run(
            [python, "-m", "scripts.acp.submit", "--config", str(config_path), "--submit"],
            cwd=project,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    records = sorted((run / "submissions").glob("*/job.json"))
    record = json.loads(records[-1].read_text())
    write_json(folder / "training-job.json", record)
    event("submitted", job=record["response"].get("name"), run=str(run))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--feature-work", type=Path, required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("interval must be positive")
    folder = args.folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / "monitor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def event(state, **details):
        value = {
            "at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "state": state,
            **details,
        }
        write_json(folder / "status.json", value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    config = read_yaml(args.config)
    recipe = read_yaml(config["data"]["config"])
    plan = json.loads((args.feature_work / "plan.json").read_text())
    job_path = sorted((args.feature_work / "submissions").glob("*/job.json"))[-1]
    feature_job = json.loads(job_path.read_text())["response"]["name"]
    platform = config["acp"]
    try:
        while not (folder / "STOP").exists():
            manifests = published_features(recipe)
            if manifests is not None:
                flex = next(m for m in manifests if m["dataset_id"] == "flexivoice")
                expected = next(
                    d for d in plan["datasets"] if d["source"]["dataset_id"] == "flexivoice"
                )
                if (
                    flex.get("selection", {}).get("mode") != "selection_branch"
                    and flex["rows"] != expected["target_count"]
                ) or flex["profile_id"] != plan["profile_id"]:
                    raise ValueError("Published FlexiVoice features differ from the monitored plan")
                launch(args.config.resolve(), folder, event)
                return
            progress = json.loads((args.feature_work / "status.json").read_text())
            try:
                job = request("GET", f"{jobs_url(platform)}/{feature_job}", home=platform["home"])
            except Exception as exc:
                event("waiting", progress=progress, status_error=str(exc))
            else:
                if job.get("state") in {"FAILED", "STOPPED", "CANCELED", "CANCELLED"}:
                    raise RuntimeError(f"Feature task {feature_job}: {job['state']}")
                event(
                    "waiting",
                    feature_job=feature_job,
                    feature_state=job.get("state"),
                    progress=progress,
                )
            # A STOP file is observed within 5 seconds while waiting.
            for _ in range(args.interval):
                if (folder / "STOP").exists():
                    break
                time.sleep(1)
        event("stopped")
    except Exception as exc:
        event("failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
