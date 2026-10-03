import fcntl
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.archive_checkpoints import archive_run


def checkpoint_fixture(tmp_path):
    signature = {"protocol": 5}
    (tmp_path / "signature.json").write_text(json.dumps(signature))
    checkpoint = tmp_path / "checkpoints/step-00000010"
    (checkpoint / "distributed").mkdir(parents=True)
    (checkpoint / "distributed/.metadata").write_bytes(b"metadata")
    (checkpoint / "distributed/__0_0.distcp").write_bytes(b"model and optimizer")
    (checkpoint / "rng-0.pt").write_bytes(b"rng")
    (checkpoint / "metadata.json").write_text(
        json.dumps(
            {
                "signature": signature,
                "world_size": 1,
                "progress": {"step": 10},
            }
        )
    )
    (checkpoint / "COMPLETE").write_text("ok\n")
    return checkpoint


def test_archive_survives_source_retention_and_repeated_invocation(tmp_path):
    source = checkpoint_fixture(tmp_path)
    (source.parent / "step-00000020.incomplete").mkdir()
    assert archive_run(tmp_path)[0]["status"] == "archived"
    target = tmp_path / "archived-checkpoints" / source.name
    for path in source.rglob("*"):
        if path.is_file():
            copied = target / path.relative_to(source)
            assert copied.read_bytes() == path.read_bytes()
            assert copied.stat().st_ino != path.stat().st_ino
    assert archive_run(tmp_path)[0]["status"] == "already_archived"
    shutil.rmtree(source)
    assert archive_run(tmp_path, verify=True) == []
    assert (target / "COMPLETE").is_file()
    shard = target / "distributed/__0_0.distcp"
    shard.write_bytes(b"x" * shard.stat().st_size)
    with pytest.raises(ValueError, match="checksum mismatch"):
        archive_run(tmp_path, verify=True)


def test_failed_copy_never_publishes_and_missing_rng_is_rejected(tmp_path, monkeypatch):
    from scripts import archive_checkpoints

    source = checkpoint_fixture(tmp_path)

    def fail(*args):
        raise OSError("simulated interrupted copy")

    monkeypatch.setattr(archive_checkpoints, "copy_verified", fail)
    with pytest.raises(OSError, match="interrupted copy"):
        archive_run(tmp_path)
    assert not (tmp_path / "archived-checkpoints" / source.name).exists()
    assert not list((tmp_path / "archived-checkpoints").glob("*.incomplete-*"))
    (source / "rng-0.pt").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        archive_run(tmp_path)


def test_archive_service_recovers_after_failed_check_and_rejects_duplicate(tmp_path):
    from scripts.schedule_checkpoint_archive import PROJECT, run_check

    source = checkpoint_fixture(tmp_path)
    folder = tmp_path / "archive-service"
    folder.mkdir()
    (source / "rng-0.pt").unlink()
    failed = run_check(tmp_path, folder, interval=3600, timeout=30)
    assert failed["state"] == "archive_failed"
    assert failed["exit_code"] != 0
    (source / "rng-0.pt").write_bytes(b"rng")
    succeeded = run_check(tmp_path, folder, interval=3600, timeout=30)
    assert succeeded["state"] == "waiting"
    assert succeeded["exit_code"] == 0
    assert (tmp_path / "archived-checkpoints" / source.name / "ARCHIVE_COMPLETE.json").is_file()
    history = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
    assert [row["exit_code"] for row in history] == [failed["exit_code"], 0]
    with (folder / "scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        duplicate = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.schedule_checkpoint_archive",
                "--run",
                str(tmp_path),
                "--once",
            ],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert duplicate.returncode != 0
        assert "BlockingIOError" in duplicate.stderr
    assert json.loads((folder / "status.json").read_text()) == succeeded


def test_archive_service_records_timeout_without_claiming_success(tmp_path, monkeypatch):
    from scripts import schedule_checkpoint_archive as service

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("archive", 30)

    monkeypatch.setattr(service.subprocess, "run", timeout)
    status = service.run_check(tmp_path, tmp_path, interval=3600, timeout=30)
    assert status["state"] == "archive_failed"
    assert status["exit_code"] == 124
    assert "retry next hour" in Path(status["log"]).read_text()
