import pytest
from lm_tts.data.reference import validate_reference


def test_strict_training_checks_full_native_and_codec_interval():
    row = dict(
        codec_native_sample_rate=24000,
        codec_end_frame=240000,
        codec_feature_key="k",
        codec_num_codec_frames=125,
        speaker_reference_codec_start=0,
        speaker_reference_codec_end=38,
        speaker_start_frame=0,
        speaker_end_frame=72960,
        speaker_native_sample_rate=24000,
        speaker_reference_codec_feature_key="k",
        speaker_reference_native_total_frames=240000,
    )
    assert validate_reference(row, strict3s=True) == (0, 38)
    row.update(speaker_reference_codec_end=37, speaker_end_frame=71040)
    assert validate_reference(row) == (0, 37)
    with pytest.raises(ValueError):
        validate_reference(row, strict3s=True)


def test_reference_build_reuses_an_already_filtered_baseline(tmp_path):
    import json
    from test_reference_training import reference_fixture
    from lm_tts.artifacts import digest, file_hash
    from lm_tts.data.reference_build import create_reference_build
    from lm_tts.data.reader import FeatureDataset

    first, _, _ = reference_fixture(tmp_path)
    recipe = json.loads((first / "data_recipe.json").read_text())
    recipe["reuse_build"] = str(first / "manifest.json")
    baseline = json.loads((first / "manifest.json").read_text())
    for role in ("train", "validation"):
        split_path = first / role / "manifest.json"
        split_manifest = json.loads(split_path.read_text())
        split_manifest["split"]["source_build_sha256"] = file_hash(first / "manifest.json")
        split_manifest["split_sha256"] = digest(split_manifest["split"])
        split_path.write_text(json.dumps(split_manifest))
        before = FeatureDataset(first / role / "manifest.json", (200, 201), mask_reference=True)
        assert len(before)
    destination = tmp_path / "second-reference-build"
    create_reference_build(recipe, destination, baseline["tokenizer"], baseline["selection_sha256"])
    for role in ("train", "validation"):
        before = FeatureDataset(first / role / "manifest.json", (200, 201), mask_reference=True)
        after = FeatureDataset(
            destination / role / "manifest.json", (200, 201), mask_reference=True
        )
        assert [before[i]["id"] for i in range(len(before))] == [
            after[i]["id"] for i in range(len(after))
        ]
        assert (
            after.bindings[0]["reference_filtering"]["input_rows"] > after.bindings[0]["ready_rows"]
        )
