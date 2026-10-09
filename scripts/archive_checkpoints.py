"""Copy complete checkpoints to a permanent per-run archive without changing retention."""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from lm_tts.artifacts import file_hash

RECEIPT = "ARCHIVE_COMPLETE.json"


def validate_checkpoint(path, signature):
    metadata = json.loads((path / "metadata.json").read_text())
    world = metadata["world_size"]
    if (
        metadata["signature"] != signature
        or path.name != f"step-{metadata['progress']['step']:08d}"
    ):
        raise ValueError(f"Checkpoint identity mismatch: {path}")
    required = [path / "COMPLETE", path / "distributed/.metadata"]
    required.extend(path / f"rng-{rank}.pt" for rank in range(world))
    if not all(p.is_file() and p.stat().st_size > 0 for p in required):
        raise ValueError(f"Checkpoint is incomplete: {path}")
    if not list((path / "distributed").glob("*.distcp")):
        raise ValueError(f"Checkpoint has no distributed shards: {path}")
    return metadata


def verify_archive(path, signature, *, hashes=False):
    receipt = json.loads((path / RECEIPT).read_text())
    validate_checkpoint(path, signature)
    actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
    if actual != set(receipt["files"]) | {RECEIPT}:
        raise ValueError(f"Archive file inventory changed: {path}")
    for name, expected in receipt["files"].items():
        item = path / name
        if item.is_symlink() or item.stat().st_size != expected["bytes"]:
            raise ValueError(f"Archive file size/type changed: {item}")
        if hashes and file_hash(item) != expected["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {item}")
    return receipt


def copy_verified(source, destination):
    """Hash the source stream and reread the independent destination before publishing."""
    checksum = hashlib.sha256()
    size = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            writer.write(chunk)
            checksum.update(chunk)
            size += len(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    if file_hash(destination) != checksum.hexdigest():
        raise ValueError(f"Copied checkpoint checksum mismatch: {destination}")
    return {"bytes": size, "sha256": checksum.hexdigest()}


def archive_one(source, target_root, signature):
    metadata = validate_checkpoint(source, signature)
    target = target_root / source.name
    if target.exists():
        verify_archive(target, signature)
        if file_hash(target / "metadata.json") != file_hash(source / "metadata.json"):
            raise ValueError(f"Conflicting checkpoint already archived: {target}")
        return {"checkpoint": source.name, "status": "already_archived", "target": str(target)}
    files = sorted(p for p in source.rglob("*") if p.is_file())
    if any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError(f"Checkpoint must contain real files, not symlinks: {source}")
    total = sum(p.stat().st_size for p in files)
    if shutil.disk_usage(target_root).free < total + 1024**3:
        raise ValueError("Insufficient free space for checkpoint archive plus 1 GiB reserve")
    temporary = Path(tempfile.mkdtemp(prefix=f".{source.name}.incomplete-", dir=target_root))
    try:
        inventory = {}
        # A staging directory is never a published checkpoint; copy COMPLETE last anyway.
        for item in sorted(files, key=lambda p: (p.name == "COMPLETE", str(p))):
            name = str(item.relative_to(source))
            inventory[name] = copy_verified(item, temporary / name)
        receipt = {
            "source": str(source),
            "step": metadata["progress"]["step"],
            "archived_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "files": inventory,
            "bytes": sum(v["bytes"] for v in inventory.values()),
            "verification": "source stream and destination SHA256 matched for every file",
        }
        with (temporary / RECEIPT).open("w") as stream:
            json.dump(receipt, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.rename(target)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return {
        "checkpoint": source.name,
        "status": "archived",
        "target": str(target),
        "bytes": receipt["bytes"],
    }


def archive_run(run, *, verify=False):
    run = Path(run).resolve()
    target_root = run / "archived-checkpoints"
    target_root.mkdir(exist_ok=True)
    signature = json.loads((run / "signature.json").read_text())
    with (target_root / ".archive.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if verify:
            for path in sorted(target_root.glob("step-*")):
                verify_archive(path, signature, hashes=True)
        results = []
        for source in sorted((run / "checkpoints").glob("step-*")):
            if re.fullmatch(r"step-\d{8}", source.name) and (source / "COMPLETE").is_file():
                result = archive_one(source, target_root, signature)
                results.append(result)
                print(json.dumps(result), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument(
        "--verify", action="store_true", help="Also rehash all previously archived files"
    )
    args = parser.parse_args()
    archive_run(args.run, verify=args.verify)


if __name__ == "__main__":
    main()
