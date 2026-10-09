from pathlib import Path

import pytest

from lm_tts.config import read_yaml
from lm_tts.training.config import load_config


def test_inheritance_resolves_child_overrides_lists_and_types(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "paths: {workspace: /old, runs: '${paths.workspace}/runs'}\n"
        "train: {run_name: base, steps: 4, keep: 2}\n"
        "output: '${paths.runs}/${train.run_name}'\n"
        "steps: '${train.steps}'\nbindings: [old]\n"
    )
    child = tmp_path / "nested/experiment.yaml"
    child.parent.mkdir()
    child.write_text(
        "extends: ../base.yaml\npaths: {workspace: /new}\n"
        "train: {run_name: child}\nbindings: [new]\n"
    )
    result = read_yaml(child)
    assert result["output"] == "/new/runs/child"
    assert result["steps"] == 4
    assert result["train"]["keep"] == 2
    assert result["bindings"] == ["new"]
    assert read_yaml(child, overrides={"train": {"run_name": "cli"}})["output"] == "/new/runs/cli"
    assert read_yaml(child, keys=("output",)) == {"output": "/new/runs/child"}


@pytest.mark.parametrize(
    "content, message",
    [
        ("extends: a.yaml", "inheritance cycle"),
        ("a: '${b}'\nb: '${a}'", "reference cycle"),
        ("a: '${unknown.path}'", "Unknown configuration reference"),
    ],
)
def test_bad_configuration_references_fail_before_execution(tmp_path, content, message):
    filename = tmp_path / "a.yaml"
    filename.write_text(content)
    with pytest.raises(ValueError, match=message):
        read_yaml(filename)


def test_train_config_has_one_run_identity_and_detects_conflicting_output(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    for precision, file in [("bf16", "train-bf16.yaml"), ("fp32", "train-fp32.yaml")]:
        config = load_config(repo / "configs" / file)
        assert config["train"]["precision"] == precision
        assert Path(config["train"]["output"]).name == config["train"]["run_name"]
    filename = tmp_path / "bad.yaml"
    filename.write_text(
        "model: {}\ndata: {}\ntrain: {run_name: run, runs_root: /tmp, output: /other}"
    )
    with pytest.raises(ValueError, match="conflicts"):
        load_config(filename)


def test_training_applies_communication_defaults_before_initialization(monkeypatch):
    import sys

    from lm_tts import train
    from lm_tts.training import engine

    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("NCCL_IB_TIMEOUT", "21")
    monkeypatch.setenv("NCCL_IB_RETRY_CNT", "temporary")
    monkeypatch.delenv("NCCL_IB_RETRY_CNT")
    monkeypatch.setenv("NCCL_IB_AR_THRESHOLD", "temporary")
    monkeypatch.delenv("NCCL_IB_AR_THRESHOLD")
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(repo / "configs/train-bf16.yaml")])
    observed = {}

    def start(config, **kwargs):
        import os

        assert "acp" not in config
        observed.update({key: os.environ[key] for key in config["environment"]})
        assert observed == config["environment"]

    monkeypatch.setattr(engine, "run", start)
    train.main()
    assert observed == {
        "NCCL_IB_TIMEOUT": "21",
        "NCCL_IB_RETRY_CNT": "13",
        "NCCL_IB_AR_THRESHOLD": "0",
    }
