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


def test_anneal_config_requires_initialization_and_no_warmup():
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root
        / "configs/supervised-tts-20261015-emilia2-refmask-anneal-s22500-bf16-8gpu-acc2-lr3e-4-bblr1e-4-decay0.3ep.yaml"
    )
    train = config["train"]
    assert train["scheduler"] == {"name": "anneal", "decay_epochs": 0.3, "min_lr_ratio": 0.1}
    assert train["init_from"]["load_optimizer"] is True
    assert train["accumulation"] == 2 and read_yaml(config["config_path"])["acp"]["nodes"] == 1


@pytest.mark.parametrize(
    "change, message",
    [
        ({"init_from": None}, "requires train.init_from"),
        ({"warmup_steps": 10}, "no warmup"),
        ({"epochs": 2}, "no warmup"),
        (
            {"scheduler": {"name": "anneal", "decay_epochs": 1.5, "min_lr_ratio": 0.1}},
            "decay_epochs",
        ),
        ({"init_from": {"checkpoint": "x"}}, "init_from needs"),
    ],
)
def test_bad_anneal_settings_fail_before_execution(tmp_path, change, message):
    import yaml

    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "configs/supervised-tts-20261015-emilia2-refmask-anneal-s22500-bf16-8gpu-acc2-lr3e-4-bblr1e-4-decay0.3ep.yaml"
    )
    config = yaml.safe_load(source.read_text())
    config["extends"] = str(root / "configs/base.yaml")
    config["train"].update(change)
    path = tmp_path / "anneal.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=message):
        load_config(path)


LANGID = "configs/supervised-tts-20261020-emilia-v4-emilia2-refmask-cbinit-langid-bf16-16gpu-acc1-lr3e-4-bblr1e-4-ep3-wsd.yaml"


def streaming_config(tmp_path, change):
    import yaml

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / LANGID).read_text())
    config["extends"] = str(root / "configs/base.yaml")
    config["train"].update(change)
    path = tmp_path / "streaming.yaml"
    path.write_text(yaml.safe_dump(config))
    return load_config(path)


def test_streaming_defaults_keep_text_lead_and_train_on_late_text(tmp_path):
    train = streaming_config(tmp_path, {"streaming_fraction": 0.5})["train"]
    assert (
        train["streaming_max_text_ratio"],
        train["streaming_pause_probability"],
        train["streaming_max_pause_frames"],
    ) == (0.67, 0.3, 12)
    assert "streaming_fraction" not in load_config(Path(__file__).parents[1] / LANGID)["train"]


@pytest.mark.parametrize(
    "change, message",
    [
        ({"streaming_fraction": 0}, "streaming_fraction"),
        ({"streaming_fraction": 1.5}, "streaming_fraction"),
        ({"streaming_max_text_ratio": 0.5}, "require train.streaming_fraction"),
        ({"streaming_fraction": 0.5, "streaming_max_text_ratio": 1.2}, "max_text_ratio"),
        (
            {"streaming_fraction": 0.5, "streaming_max_pause_frames": 0},
            "max_pause_frames",
        ),
    ],
)
def test_bad_streaming_settings_fail_before_execution(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        streaming_config(tmp_path, change)
