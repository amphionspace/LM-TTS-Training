"""Resolve the platform log mount and freeze submitted code."""

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


def afs_subdir(directory, mount):
    directory, mount = Path(directory).resolve(), Path(mount).resolve()
    if not directory.is_relative_to(mount):
        raise ValueError("TensorBoard runs_root must be inside the AFS mount")
    relative = directory.relative_to(mount)
    if len(relative.parts) != 1 or not re.fullmatch(
        r"[A-Za-z0-9](?:[A-Za-z0-9_-]{0,61}[A-Za-z0-9])?", relative.name
    ):
        raise ValueError(
            "ACP TensorBoard requires a one-level AFS runs_root; use tensorboard: local for nested paths"
        )
    return "/" + relative.name


def snapshot(project, destination):
    project, destination = Path(project), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("qwen3_train", "scripts", "configs", "tests"):
        shutil.copytree(
            project / name,
            destination / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    pairs = project / "artifacts/evaluation-smoke/pairs.json"
    if pairs.exists():
        target = destination / "artifacts/evaluation-smoke/pairs.json"
        target.parent.mkdir(parents=True)
        shutil.copyfile(pairs, target)
    sources = {
        str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in destination.rglob("*")
        if p.is_file()
    }
    revision = subprocess.check_output(
        ["git", "-C", str(project), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {"git_revision": revision, "files": sources}
    (destination / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
