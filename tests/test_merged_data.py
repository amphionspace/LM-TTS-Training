import json

import lance
import pyarrow as pa
import pytest
from test_unified_data import Tokenizer, fixture

from lm_tts.artifacts import digest, file_hash
from lm_tts.data.batch import collate
from lm_tts.data.build import feature_reference
from lm_tts.data.merged import bind_merged
from lm_tts.data.reader import FeatureDataset


def merged_fixture(root, *, embedding_dim=1024, speaker_profile=None, num_rows=8):
    recipe, selection, selection_path = fixture(
        root, embedding_dim=embedding_dim, speaker_profile=speaker_profile, num_rows=num_rows
    )
    sources, tables = {}, {}
    for kind, feature_kind in (("codec", "codec"), ("speaker", "speaker_embedding")):
        manifest, ref, table = feature_reference(
            root, recipe["bindings"][0][kind + "_manifest"], feature_kind
        )
        sources[kind] = {"alias": kind, **ref}
        tables[kind] = {r["target_id"]: r for r in table.to_table().to_pylist()}
    text_dir = root / "datasets/test/v0.1/features/text/run"
    text_dir.mkdir(parents=True)
    profile = {"kind": "text"}
    text_manifest = {
        "kind": "text",
        "status": "complete",
        "dataset_id": "test",
        "release_id": "v0.1",
        "rows": num_rows,
        "profile": profile,
        "profile_id": digest(profile),
        "lance_version": 1,
        "selection": {"manifest_sha256": file_hash(selection_path)},
    }
    text_path = text_dir / "manifest.json"
    text_path.write_text(json.dumps(text_manifest))
    sources["text"] = {
        "alias": "text",
        "manifest_path": str(text_path),
        "manifest_sha256": file_hash(text_path),
        "profile_id": digest(profile),
        "lance_version": 1,
    }
    rows = []
    for codec in tables["codec"].values():
        speaker = tables["speaker"][codec["target_id"]]
        row = {k: codec[k] for k in ("target_id", "audio_sha256", "text", "language", "speaker_id")}
        row.update(dataset_id="test", release_id="v0.1")
        for kind, original in (("codec", codec), ("speaker", speaker)):
            row.update(
                {
                    f"{kind}_{k}": v
                    for k, v in original.items()
                    if k not in row and k != "text_revision"
                }
            )
        row["speaker_embedding_dim"] = embedding_dim
        rows.append(row)
    merged_dir = root / "datasets/test/v0.1/features/merged/run"
    merged_dir.mkdir(parents=True)
    table = lance.write_dataset(pa.Table.from_pylist(rows), merged_dir / "features.lance")
    profile = {
        "kind": "merged",
        "sources": {k: s["profile_id"] for k, s in sources.items()},
        "native_duration_tolerance_seconds": 0.02,
        "payload_transform": "none",
    }
    manifest = {
        "kind": "merged",
        "status": "complete",
        "dataset_id": "test",
        "release_id": "v0.1",
        "target_kind": "sample",
        "rows": num_rows,
        "profile": profile,
        "profile_id": digest(profile),
        "lance_version": table.version,
        "table_path": "features/merged/run/features.lance",
        "selection": {"manifest_sha256": file_hash(selection_path)},
        "coverage": {"ok": num_rows},
        "inputs": list(sources.values()),
    }
    manifest_path = merged_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    recipe["bindings"] = [{"dataset_id": "test", "merged_manifest": str(manifest_path)}]
    return recipe, selection, selection_path


def make_build(root, output, **kwargs):
    recipe, selection, path = merged_fixture(root, **kwargs)
    bind_merged(
        recipe, output, selection, path, file_hash(path), {"sha256": "fixture"}, Tokenizer()
    )
    return FeatureDataset(output / "manifest.json", (200, 201))


def test_merged_read_preserves_source_and_uses_one_table(tmp_path):
    root, output = tmp_path / "features", tmp_path / "build"
    recipe, selection, path = merged_fixture(root)
    before = {str(p): file_hash(p) for p in root.rglob("*") if p.is_file()}
    manifest = bind_merged(
        recipe, output, selection, path, file_hash(path), {"sha256": "fixture"}, Tokenizer()
    )
    dataset = FeatureDataset(output / "manifest.json", (200, 201))
    rows = dataset.__getitems__([7, 0, 4])
    assert [r["id"] for r in rows] == [f"{i:064x}" for i in (7, 0, 4)]
    assert [r["speaker_embedding"][0].item() for r in rows] == [8, 1, 5]
    assert set(dataset.tables) == {(0, "merged")}
    assert manifest["coverage"] == {"ready": 8, "dataset_unbound": 0}
    assert before == {str(p): file_hash(p) for p in root.rglob("*") if p.is_file()}


def test_corrupt_payload_is_skipped_without_losing_healthy_peers(tmp_path):
    dataset = make_build(tmp_path / "features", tmp_path / "build")
    table = dataset.table(0, "merged")
    original = table.take

    def corrupt(indices, **kwargs):
        result = original(indices, **kwargs)
        rows = result.to_pylist()
        for index, row in zip(indices, rows):
            if index == 1:
                row["codec_codes"][0][0] = 2048
            if index == 2:
                row["speaker_embedding"][0] = float("nan")
            if index == 4:
                row["text"] = None
        return pa.Table.from_pylist(rows, schema=result.schema)

    table.take = corrupt
    with pytest.warns(RuntimeWarning, match="Skipping bad sample"):
        batch = collate(dataset.__getitems__([0, 1, 2, 3, 4]))
    assert batch["skipped_samples"].item() == 3
    assert batch["frame_lengths"].tolist() == [1, 4]
    with pytest.warns(RuntimeWarning):
        empty = collate(dataset.__getitems__([1, 2]))
    assert set(empty) == {"skipped_samples"}


def test_failed_batched_read_falls_back_to_individual_rows(tmp_path):
    dataset = make_build(tmp_path / "features", tmp_path / "build")
    table = dataset.table(0, "merged")
    original = table.take

    def damaged(indices, **kwargs):
        if 2 in indices:
            raise OSError("unreadable fragment")
        return original(indices, **kwargs)

    table.take = damaged
    with pytest.warns(RuntimeWarning):
        rows = dataset.__getitems__([3, 2, 0])
    assert rows[1] is None
    assert [rows[i]["num_frames"] for i in (0, 2)] == [4, 1]


def test_merged_interval_policy_is_pinned_and_enforced(tmp_path):
    dataset = make_build(tmp_path / "features", tmp_path / "build")
    codec, speaker = dataset._read_rows(0, [0])[0]
    codec["speaker_end_frame"] += 100
    speaker["end_frame"] += 100
    assert dataset._sample(codec, speaker, dataset.bindings[0])["duration"] == 1.0
    codec["speaker_end_frame"] += 1000
    with pytest.raises(ValueError, match="tolerance"):
        dataset._sample(codec, speaker, dataset.bindings[0])
    # Successful early runs used a two-frame policy, which must not be relaxed silently.
    binding = {**dataset.bindings[0], "merged_profile": {"native_frame_tolerance": 2}}
    codec["speaker_end_frame"] = codec["codec_end_frame"] + 3
    with pytest.raises(ValueError, match="tolerance"):
        dataset._sample(codec, speaker, binding)


def test_holdout_is_deterministic_disjoint_and_covers_every_source_row(tmp_path):
    import numpy as np

    from lm_tts.data.split import split_build

    dataset = make_build(tmp_path / "features", tmp_path / "build", num_rows=100)
    split_build(dataset.path, fraction=0.1, seed=42)
    train = FeatureDataset(dataset.path.parent / "train/manifest.json", (200, 201))
    validation = FeatureDataset(dataset.path.parent / "validation/manifest.json", (200, 201))
    assert train.manifest["split_sha256"] == validation.manifest["split_sha256"]
    train_rows = set(train.indices[0]["row"].tolist())
    val_rows = set(validation.indices[0]["row"].tolist())
    assert len(train_rows) == 90 and len(val_rows) == 10
    assert not train_rows & val_rows
    assert train_rows | val_rows == set(range(100))
    assert validation[0]["id"] == f"{int(validation.indices[0][0]['row']):064x}"
    # Rebuilding elsewhere from the same fixed source produces the same partition.
    import shutil

    other = tmp_path / "other"
    other.mkdir()
    shutil.copy2(dataset.path, other / "manifest.json")
    shutil.copy2(dataset.path.parent / "sampling-000.npy", other / "sampling-000.npy")
    split_build(other / "manifest.json", fraction=0.1, seed=42)
    np.testing.assert_array_equal(
        validation.indices[0], np.load(other / "validation/sampling-000.npy")
    )


@pytest.mark.parametrize("bad_token", [-1, 256, True, 3.5, "12", None])
def test_bad_text_tokens_are_skipped_before_collation(tmp_path, monkeypatch, bad_token):
    import pickle

    dataset = make_build(tmp_path / "features", tmp_path / "build")
    dataset = FeatureDataset(dataset.path, (200, 201), text_vocab_size=256)
    # Spawned workers must preserve the vocabulary bound.
    dataset = pickle.loads(pickle.dumps(dataset))
    original = dataset._read_rows

    def corrupt(slot, locators):
        rows = original(slot, locators)
        for locator, (codec, _) in zip(locators, rows):
            if locator == 1:
                codec["text_ids"] = [bad_token]
        return rows

    monkeypatch.setattr(dataset, "_read_rows", corrupt)
    with pytest.warns(RuntimeWarning, match="Invalid text token"):
        batch = collate(dataset.__getitems__([0, 1, 2]))
    assert batch["skipped_samples"].item() == 1
    assert batch["frame_lengths"].tolist() == [1, 3]
    assert batch["text_ids"].max().item() < 256


def test_binding_can_pin_its_own_selection(tmp_path):
    recipe, selection, path = merged_fixture(tmp_path / "features")
    default = path.with_name("other-selection.json")
    default.write_text(json.dumps({**selection, "selection_id": "a-different-selection"}))
    arguments = (json.loads(default.read_text()), default, file_hash(default))
    # The default selection no longer matches the merged feature; the binding's own does.
    with pytest.raises(ValueError, match="fully cover"):
        bind_merged(recipe, tmp_path / "bad", *arguments, {"sha256": "fixture"}, Tokenizer())
    recipe["bindings"][0]["selection_manifest"] = str(path)
    manifest = bind_merged(
        recipe, tmp_path / "build", *arguments, {"sha256": "fixture"}, Tokenizer()
    )
    binding = manifest["bindings"][0]
    assert (binding["selection_manifest"], binding["selection_sha256"]) == (
        str(path),
        file_hash(path),
    )
    assert manifest["selection_sha256"] == file_hash(default)


def test_language_ids_follow_a_fixed_per_target_auto_rule(tmp_path):
    recipe, selection, path = merged_fixture(tmp_path / "features", num_rows=64)
    output = tmp_path / "build"
    bind_merged(
        recipe, output, selection, path, file_hash(path), {"sha256": "fixture"}, Tokenizer()
    )
    languages = json.loads((output / "manifest.json").read_text())["languages"]
    mapping = {language: 2050 + i for i, language in enumerate(sorted(languages))}

    def ids(fraction):
        dataset = FeatureDataset(
            output / "manifest.json",
            (200, 201),
            language_codec_ids=mapping,
            language_auto_fraction=fraction,
        )
        assert (
            dataset.token_lengths[0]
            == dataset.indices[0][0]["tokens"] + dataset.indices[0][0]["frames"] + 11
        )
        rows = dataset.__getitems__(list(range(len(dataset))))
        return [row["language_codec_id"] for row in rows], rows

    explicit, rows = ids(0.0)
    assert explicit == [mapping[row["language"]] for row in rows]
    first, _ = ids(0.5)
    again, _ = ids(0.5)
    assert first == again and 0 < first.count(-1) < len(first)
    assert collate(rows)["language_codec_ids"].tolist() == explicit
    with pytest.raises(ValueError, match="No codec language mapping"):
        FeatureDataset(output / "manifest.json", (200, 201), language_codec_ids={})


def test_streaming_follows_a_fixed_per_target_rule_with_text_lead(tmp_path):
    recipe, selection, path = merged_fixture(tmp_path / "features", num_rows=64)
    output = tmp_path / "build"
    bind_merged(
        recipe, output, selection, path, file_hash(path), {"sha256": "fixture"}, Tokenizer()
    )

    def rows(ratio, pause=0.0):
        dataset = FeatureDataset(
            output / "manifest.json",
            (200, 201),
            streaming_fraction=0.5,
            streaming_max_text_ratio=ratio,
            streaming_pause_probability=pause,
            streaming_max_pause_frames=12 if pause else 0,
        )
        return dataset.__getitems__(list(range(len(dataset))))

    loose = rows(1.0)
    assert [r["streaming"] for r in loose] == [r["streaming"] for r in rows(1.0)]
    for row in loose:
        lead = len(row["text_ids"]) <= len(row["codes"])
        assert row["streaming_eligible"] == lead
        assert not row["streaming"] or row["streaming_eligible"]
    assert 0 < sum(r["streaming"] for r in loose) < sum(r["streaming_eligible"] for r in loose)
    # A tighter lead requirement only removes samples; it never re-draws the 50% choice.
    tight = rows(0.3)
    assert all(a["streaming"] or not b["streaming"] for a, b in zip(loose, tight))
    paused = rows(1.0, pause=1.0)
    assert [r["streaming"] for r in paused] == [r["streaming"] for r in loose]
    for row in paused:
        at, length = row["streaming_pause"]
        if length:
            # The delayed tokens must still arrive before an even-rate speaker reaches them.
            rate = len(row["text_ids"]) / len(row["codes"])
            assert at >= 1.5 * length * rate / (1 - rate)
            assert at <= len(row["text_ids"]) - 2 and 1 <= length <= 12
            assert len(row["text_ids"]) + length <= len(row["codes"])
    assert any(r["streaming_pause"][1] for r in paused)
    batch = collate(paused)
    assert batch["streaming_pause"].shape == (len(paused), 2)
    with pytest.raises(ValueError, match="streaming_max_text_ratio"):
        FeatureDataset(output / "manifest.json", (200, 201), streaming_fraction=0.5)
