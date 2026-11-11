import json
import subprocess

import pytest
import yaml

from lm_tts.artifacts import digest, file_hash
from scripts.acp import after_features


def test_waits_for_every_published_feature_and_rejects_mixed_profiles(tmp_path):
    recipe = {
        "root": str(tmp_path),
        "bindings": [
            {"dataset_id": name, "dots_manifest": f"{name}.json"} for name in ("flexivoice", "nv")
        ],
    }

    def publish(name, status="complete", profile="same"):
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                {
                    "dataset_id": name,
                    "kind": "dots_vae",
                    "status": status,
                    "profile_id": profile,
                }
            )
        )

    publish("nv")
    assert after_features.published_features(recipe) is None
    publish("flexivoice", status="running")
    assert after_features.published_features(recipe) is None
    publish("flexivoice")
    assert len(after_features.published_features(recipe)) == 2
    publish("flexivoice", profile="different")
    with pytest.raises(ValueError, match="profiles differ"):
        after_features.published_features(recipe)


def test_selection_features_validate_the_selected_count_not_the_full_extraction_count(tmp_path):
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "selection_id": "selected",
                "outputs": [{"dataset_id": "flexivoice", "selected_rows": 2, "rows": 10}],
            }
        )
    )
    manifest = {
        "status": "complete",
        "kind": "dots_vae",
        "dataset_id": "flexivoice",
        "profile_id": "same",
        "rows": 2,
        "selection": {
            "mode": "selection_branch",
            "selection_id": "selected",
            "manifest_path": "selection.json",
            "manifest_sha256": file_hash(selection_path),
            "target_count": 2,
        },
    }
    path = tmp_path / "features.json"
    path.write_text(json.dumps(manifest))
    recipe = {
        "root": str(tmp_path),
        "bindings": [{"dataset_id": "flexivoice", "dots_manifest": "features.json"}],
    }
    assert after_features.published_features(recipe)[0]["rows"] == 2
    manifest["rows"] = 10
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="published selection"):
        after_features.published_features(recipe)


@pytest.mark.parametrize("lost_response", [False, True])
def test_launch_validates_before_submit_and_never_repeats_post(
    tmp_path, monkeypatch, lost_response
):
    build = tmp_path / "build"
    for role in ("train", "validation"):
        (build / role).mkdir(parents=True)
        (build / role / "manifest.json").write_text("{}")
    recipe = {"root": str(tmp_path), "bindings": []}
    (tmp_path / "recipe.yaml").write_text(yaml.safe_dump(recipe))
    (build / "data_recipe.json").write_text(json.dumps(recipe))
    (build / "manifest.json").write_text(
        json.dumps({"recipe_sha256": digest(recipe), "bindings": []})
    )
    config = {
        "paths": {"project": str(tmp_path), "python": "python"},
        "data": {"config": str(tmp_path / "recipe.yaml"), "build_root": str(build)},
        "train": {"runs_root": str(tmp_path / "runs"), "run_name": "test"},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    calls = []

    def validate(config):
        calls.append("validate")
        return {}

    def submit(*args, **kwargs):
        assert (tmp_path / "submission-attempt.json").exists()
        calls.append("submit")
        if lost_response:
            raise subprocess.CalledProcessError(1, args[0])
        submission = tmp_path / "runs/test/submissions/stamp"
        submission.mkdir(parents=True)
        (submission / "job.json").write_text(json.dumps({"response": {"name": "pt-test"}}))

    monkeypatch.setattr(after_features, "validate_build", validate)
    monkeypatch.setattr(after_features.subprocess, "run", submit)
    if lost_response:
        with pytest.raises(subprocess.CalledProcessError):
            after_features.launch(path, tmp_path, lambda *a, **kw: None)
    else:
        after_features.launch(path, tmp_path, lambda *a, **kw: None)
        assert (
            json.loads((tmp_path / "training-job.json").read_text())["response"]["name"]
            == "pt-test"
        )
    with pytest.raises(RuntimeError, match="already attempted"):
        after_features.launch(path, tmp_path, lambda *a, **kw: None)
    assert calls == ["validate", "submit"]


def test_validation_failure_prevents_submission(tmp_path, monkeypatch):
    # The same launch path must stop before creating a submission marker on a bad build.
    build = tmp_path / "build"
    for role in ("train", "validation"):
        (build / role).mkdir(parents=True)
        (build / role / "manifest.json").write_text("{}")
    recipe = {"bindings": []}
    (tmp_path / "recipe.yaml").write_text(yaml.safe_dump(recipe))
    (build / "data_recipe.json").write_text(json.dumps(recipe))
    (build / "manifest.json").write_text(json.dumps({"recipe_sha256": digest(recipe)}))
    config = {
        "paths": {"project": str(tmp_path), "python": "python"},
        "data": {"config": str(tmp_path / "recipe.yaml"), "build_root": str(build)},
        "train": {"runs_root": str(tmp_path / "runs"), "run_name": "test"},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))

    def fail(config):
        raise ValueError("Target without a usable reference")

    monkeypatch.setattr(after_features, "validate_build", fail)
    with pytest.raises(ValueError, match="usable reference"):
        after_features.launch(path, tmp_path, lambda *a, **kw: None)
    assert not (tmp_path / "submission-attempt.json").exists()
