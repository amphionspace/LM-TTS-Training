import json
import shutil

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
