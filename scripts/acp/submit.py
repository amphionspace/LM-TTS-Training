"""Render or submit one experiment; training and platform settings share the experiment YAML."""

import argparse
import datetime
import json
import os
import shlex
import subprocess
from pathlib import Path

import yaml

from lm_tts.config import read_yaml
from lm_tts.training.config import load_config

from .api import jobs_url, request
from .storage import afs_subdir, snapshot

PROJECT = Path(__file__).resolve().parents[2]


def payload(platform, training, runtime, experiment, *, resume=None, validate=False):
    nodes = platform["nodes"]
    if type(nodes) is not int or nodes < 1:
        raise ValueError("nodes must be a positive integer")
    paths, train = training["paths"], training["train"]
    name = train["run_name"]
    directory = Path(train["output"])
    if validate:
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=30s",
            "1200s",
            "bash",
            str(runtime / "scripts/acp/validate.sh"),
            str(directory),
        ]
    else:
        command = ["bash", str(runtime / "scripts/acp/launch.sh"), "--config", str(experiment)]
        if resume:
            command += ["--resume", resume]
    environment = {
        **{key: str(value) for key, value in training.get("environment", {}).items()},
        "PYTHON_BIN": paths["python"],
        "PROJECT_DIR": str(runtime),
        "HF_HOME": str(Path(paths["workspace"]) / ".cache/huggingface"),
        "NUMBA_CACHE_DIR": str(Path(paths["workspace"]) / ".cache/numba"),
    }
    if len(environment) > 10:
        raise ValueError("ACP accepts at most 10 environment variables per job")
    body = {
        "display_name": name,
        "framework": "PYTORCH_DDP" if nodes > 1 else "PYTORCH",
        "roles": [
            {
                "name": "Worker",
                "resource_spec": [{"name": platform["worker_spec"]}],
                "total_replicas": nodes,
                "image_path": platform["image"],
                "startup_script": shlex.join(command),
            }
        ],
        "env": [{"key": k, "value": v} for k, v in environment.items()],
        "mount": [
            {
                "type": "PV_AFS",
                "id": platform["volume_id"],
                "mount_path": platform["mount_path"],
                "subdir": "",
                "zone": platform["zone"],
            },
        ],
        "resource_pool": {"name": platform["cluster"], "zone": platform["zone"]},
        "scheduling": {"priority": "NORMAL", "quota_type": "RESERVED"},
        "ssh": {"auto_key_setup": platform["ssh_auto_key_setup"]},
        "fault_tolerance": {"backoff_limit": 0},
        "root_mapping": True,
    }

    mode = platform.get("tensorboard", "local")
    if mode not in {"local", "platform"}:
        raise ValueError("ACP tensorboard must be local or platform")
    if mode == "platform":
        body["mount"].append(
            {
                "type": "PV_AFS_TENSORBOARD",
                "id": platform["volume_id"],
                "mount_path": "/tensorboard",
                "subdir": afs_subdir(train["runs_root"], platform["mount_path"]),
                "zone": platform["zone"],
            }
        )
        body["tensorboard"] = {"log_path": "/tensorboard"}
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/train-bf16.yaml")
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--resume")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run tiny BF16/FP32 and resume checks instead of production training",
    )
    parser.add_argument(
        "--submit", action="store_true", help="Create and start the job; otherwise only render it"
    )
    args = parser.parse_args()
    training = load_config(args.config)
    if args.run_name:
        if Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
            parser.error("run-name must be a directory basename")
        training["train"]["run_name"] = args.run_name
        training["train"]["output"] = str(Path(training["train"]["runs_root"]) / args.run_name)
    platform = read_yaml(args.config, keys=("acp",))["acp"]
    if args.nodes is not None:
        platform["nodes"] = args.nodes
    if args.validate and (args.resume or platform["nodes"] < 2):
        parser.error(
            "Validation requires at least two nodes and cannot resume an earlier validation"
        )
    run = Path(training["train"]["output"])
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    submission = run / "submissions" / stamp
    runtime, experiment = submission / "runtime", submission / "experiment.yaml"
    body = payload(
        platform, training, runtime, experiment, resume=args.resume, validate=args.validate
    )
    print(
        json.dumps(
            {"submit": args.submit, "url": jobs_url(platform), "run": str(run), "body": body},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if not args.submit:
        return
    if args.validate:
        if (run / "COMPLETE").exists() or any((run / p).exists() for p in ("bf16", "fp32")):
            parser.error("Validation run already exists; choose a new run name")
    else:
        build = json.loads(Path(training["data"]["build"]).read_text())
        if build.get("status") != "complete" or build.get("artifact_kind") != "training_build":
            raise ValueError("The training build is not published")
        if not (Path(training["model"]["assembled_model"]) / "ASSEMBLY_COMPLETE").exists():
            raise ValueError("The assembled training model is not complete")
        if not args.resume and (run / "checkpoints/latest").exists():
            raise ValueError("Run already has checkpoints; resume or choose a new run name")
    run.mkdir(parents=True, exist_ok=True)
    source = snapshot(training["paths"]["project"], runtime)
    experiment.write_text(yaml.safe_dump(training, sort_keys=False))
    (submission / "source.yaml").write_text(args.config.read_text())
    if args.validate:
        for precision in ("bf16", "fp32"):
            subprocess.run(
                [
                    training["paths"]["python"],
                    str(runtime / "tests/check_training_resume.py"),
                    "--prepare-only",
                    "--precision",
                    precision,
                    "--rows",
                    "128",
                    "--output",
                    str(run / precision),
                ],
                check=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(runtime) + os.pathsep + str(runtime / "tests"),
                },
            )
    (submission / "request.json").write_text(json.dumps(body, indent=2))
    # Never retry POST: a lost response can still mean that a job was allocated.
    result = request("POST", jobs_url(platform), body, home=platform["home"])
    record = {
        "run_name": training["train"]["run_name"],
        "run_dir": str(run),
        "submitted_at": stamp,
        "source": source,
        "response": result,
    }
    (submission / "job.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if platform.get("tensorboard", "local") == "local":
        print(
            "Dashboard: "
            + shlex.join(
                [training["paths"]["python"], "-m", "scripts.tensorboard", "--run", str(run)]
            )
        )


if __name__ == "__main__":
    main()
