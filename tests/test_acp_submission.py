from pathlib import Path

import pytest

from qwen3_train.config import read_yaml
from qwen3_train.training.config import load_config
from scripts.acp.storage import afs_subdir, snapshot
from scripts.acp.submit import payload


def test_dashboard_modes_keep_training_and_platform_paths_separate(tmp_path):
    root = Path(__file__).resolve().parents[1]
    platform = read_yaml(root / "configs/train-bf16.yaml")["acp"]
    training = load_config(root / "configs/train-bf16.yaml")
    runtime, experiment = tmp_path / "runtime", tmp_path / "experiment.yaml"
    platform["tensorboard"] = "local"
    body = payload(platform, training, runtime, experiment, resume="latest")
    assert len(body["mount"]) == 1 and "tensorboard" not in body
    assert body["ssh"]["auto_key_setup"]
    assert "acp" not in training
    environment = {entry["key"]: entry["value"] for entry in body["env"]}
    assert environment["NCCL_IB_TIMEOUT"] == "22"
    assert environment["NCCL_IB_RETRY_CNT"] == "13"
    assert environment["NCCL_IB_AR_THRESHOLD"] == "0"
    assert str(runtime / "scripts/acp/launch.sh") in body["roles"][0]["startup_script"]
    assert "--resume latest" in body["roles"][0]["startup_script"]
    platform["tensorboard"] = "platform"
    body = payload(platform, training, runtime, experiment)
    assert body["mount"][1]["subdir"] == "/LM-TTS-Training-Runs"
    assert body["tensorboard"]["log_path"] == body["mount"][1]["mount_path"]
    platform["tensorboard"] = "typo"
    with pytest.raises(ValueError, match="local or platform"):
        payload(platform, training, runtime, experiment)


def test_platform_directory_boundary_and_code_snapshot(tmp_path):
    mount = tmp_path / "afs"
    assert afs_subdir(mount / "Runs", mount) == "/Runs"
    with pytest.raises(ValueError, match="one-level"):
        afs_subdir(mount / "nested/Runs", mount)
    with pytest.raises(ValueError, match="inside"):
        afs_subdir(tmp_path / "outside", mount)
    project = Path(__file__).resolve().parents[1]
    source = snapshot(project, tmp_path / "snapshot")
    assert "scripts/acp/launch.sh" in source["files"]
    assert not any(name.startswith(("assets/", "data/", ".git/")) for name in source["files"])
    assert not (tmp_path / "snapshot/assets").exists()
