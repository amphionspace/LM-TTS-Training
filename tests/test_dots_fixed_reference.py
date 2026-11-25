import json

import lance
import numpy as np
import pyarrow as pa
import pytest

from lm_tts.artifacts import digest, file_hash
from lm_tts.data.dots import DotsFeatureDataset
from lm_tts.data.split import split_build


class Tokenizer:
    eos_token_id = 0

    def __call__(self, texts, **kwargs):
        def encode(text):
            return [ord(c) for c in text]

        return {
            "input_ids": encode(texts) if isinstance(texts, str) else [encode(t) for t in texts]
        }

    def get_added_vocab(self):
        return {}

    def convert_tokens_to_ids(self, token):
        from lm_tts.data import dots

        return list(dots.SPECIAL_TOKENS.values()).index(token) + 1


def features(tmp_path, monkeypatch):
    """Eight rows with distinct x-vectors; row 2 is too short for a reference span of its own."""
    from lm_tts.data import dots

    rule = "dots-reference-v1:codec-grid-v1-on-patches"
    records = []
    for i, patches in enumerate((25, 30, 6, 25, 30, 25, 30, 25)):
        span = dots.reference_interval(f"t{i}", patches, rule)
        frames = patches * 4
        records.append(
            {
                "target_id": f"t{i}",
                "status": "ok",
                "text": f"text {i}",
                "text_kind": "raw",
                "language": "zh",
                "num_latent_frames": frames,
                "processed_frames": patches * 7680,
                "reference_start_patch": None if span is None else span[0],
                "reference_end_patch": None if span is None else span[1],
                "vae_mean": np.zeros((frames, 128), dtype=np.float32).tolist(),
                "vae_logstd": np.zeros((frames, 128), dtype=np.float32).tolist(),
                "xvector": [float(i + 1)] * 512,
                "reference_xvector": [-float(i + 1)] * 512,
            }
        )
    table = pa.Table.from_pylist(records)
    for field in ("vae_mean", "vae_logstd"):
        table = table.set_column(
            table.schema.get_field_index(field),
            field,
            pa.array([r[field] for r in records], type=pa.list_(pa.list_(pa.float32(), 128))),
        )
    for field in ("xvector", "reference_xvector"):
        table = table.set_column(
            table.schema.get_field_index(field),
            field,
            pa.array([r[field] for r in records], type=pa.list_(pa.float32(), 512)),
        )
    source = lance.write_dataset(table, tmp_path / "features.lance")
    profile = {"reference_rule": rule}
    manifest = {
        "dataset_id": "ts004",
        "release_id": "v0.1",
        "rows": len(records),
        "profile": profile,
        "profile_id": digest(profile),
    }
    path = tmp_path / "feature-manifest.json"
    path.write_text(json.dumps(manifest))
    reference = {
        "table_path": str(tmp_path / "features.lance"),
        "lance_version": source.version,
        "manifest_path": str(path),
        "manifest_sha256": file_hash(path),
    }
    monkeypatch.setattr(dots, "feature_reference", lambda *args: (manifest, reference, source))


def recipe(tmp_path, **binding):
    return {
        "root": str(tmp_path),
        "evaluation": "no_holdout",
        "bindings": [{"dataset_id": "ts004", "dots_manifest": "unused", **binding}],
    }


def test_fixed_reference_gives_every_other_row_its_whole_utterance_xvector(tmp_path, monkeypatch):
    from lm_tts.data import dots

    features(tmp_path, monkeypatch)
    tokenizer = Tokenizer()
    output = tmp_path / "build"
    dots.bind_dots(
        recipe(tmp_path, reference="fixed", reference_target="t3"), output, {}, tokenizer
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["bindings"][0]["coverage"]["excluded"] == {"fixed_reference": 1}
    ds = DotsFeatureDataset(output / "manifest.json", dots.Layout(tokenizer))
    # The short row without a reference span of its own stays a target.
    assert sorted(ds[i]["id"] for i in range(len(ds))) == [f"t{i}" for i in (0, 1, 2, 4, 5, 6, 7)]
    for i in range(len(ds)):
        assert ds[i]["reference"] == (0, 0)
        assert (ds[i]["xvector"] == 4.0).all()
    split_build(ds.path, fraction=0.25, seed=42)
    for role in ("train", "validation"):
        split = DotsFeatureDataset(output / role / "manifest.json", dots.Layout(tokenizer))
        assert all((split[i]["xvector"] == 4.0).all() for i in range(len(split)))
        assert "t3" not in {split[i]["id"] for i in range(len(split))}


def test_fixed_reference_must_name_a_feature_row(tmp_path, monkeypatch):
    from lm_tts.data import dots

    features(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="not in the features"):
        dots.bind_dots(
            recipe(tmp_path, reference="fixed", reference_target="missing"),
            tmp_path / "build",
            {},
            Tokenizer(),
        )
    with pytest.raises(ValueError, match="reference_target goes with"):
        dots.bind_dots(recipe(tmp_path, reference_target="t3"), tmp_path / "b2", {}, Tokenizer())
