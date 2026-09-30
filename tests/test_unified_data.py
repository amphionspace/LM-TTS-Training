import json
import pickle
import sqlite3
import tempfile
import unittest
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import DataLoader

from qwen3_train.artifacts import digest, file_hash
from qwen3_train.data.batch import collate
from qwen3_train.data.build import bind_features, feature_reference
from qwen3_train.data.reader import FeatureDataset


class Tokenizer:
    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[ord(c) % 128 for c in text] for text in texts]}


def fixture(root, embedding_dim=1024, speaker_profile=None, num_rows=8):
    release = root / "datasets/test/v0.1"
    release.mkdir(parents=True)
    ids = [f"{i:064x}" for i in range(num_rows)]
    source = lance.write_dataset(
        pa.table({"sample_id": ids, "selection_reason": [0] * num_rows}), release / "samples.lance"
    )
    selected = source.create_branch("selection", source.version)
    selection = {
        "status": "complete",
        "outputs": [
            {
                "dataset_id": "test",
                "table_path": "datasets/test/v0.1/samples.lance",
                "branch": "selection",
                "lance_version": selected.version,
                "selected_rows": num_rows,
            }
        ],
    }
    selection_path = root / "selection.json"
    selection_path.write_text(json.dumps(selection))
    identity = [
        {
            "target_id": uid,
            "parent_sample_id": uid,
            "audio_sha256": f"{i + 20:064x}",
            "start_frame": 0,
            "end_frame": 24000 * (i + 1),
            "native_sample_rate": 24000,
        }
        for i, uid in enumerate(ids)
    ]
    for kind in ["codec", "speaker_embedding"]:
        output = release / "features" / kind / "test-run"
        output.mkdir(parents=True)
        definition = (
            speaker_profile
            if kind == "speaker_embedding" and speaker_profile
            else {"kind": kind, "dimension": embedding_dim}
        )
        profile = digest(definition)
        records = [{**r, "status": "ok", "profile_id": profile} for r in identity]
        if kind == "codec":
            for i, r in enumerate(records):
                r.update(
                    codes=np.full((i + 1, 16), i, dtype=np.int16).tolist(),
                    num_codec_frames=i + 1,
                    text=f"hello {i}",
                    language=["en", "zh", "ja", "fr"][i % 4],
                    speaker_id=None,
                    text_revision="r",
                )
            fields = [
                (k, pa.string())
                for k in [
                    "target_id",
                    "parent_sample_id",
                    "audio_sha256",
                    "status",
                    "profile_id",
                    "text",
                    "language",
                    "speaker_id",
                    "text_revision",
                ]
            ]
            schema = pa.schema(
                fields
                + [
                    ("start_frame", pa.int64()),
                    ("end_frame", pa.int64()),
                    ("native_sample_rate", pa.int32()),
                    ("num_codec_frames", pa.int64()),
                    ("codes", pa.list_(pa.list_(pa.int16(), 16))),
                ]
            )
        else:
            for i, r in enumerate(records):
                r["embedding"] = [float(i + 1)] * embedding_dim
            records.reverse()
            fields = [
                (k, pa.string())
                for k in ["target_id", "parent_sample_id", "audio_sha256", "status", "profile_id"]
            ]
            schema = pa.schema(
                fields
                + [
                    ("start_frame", pa.int64()),
                    ("end_frame", pa.int64()),
                    ("native_sample_rate", pa.int32()),
                    ("embedding", pa.list_(pa.float32(), embedding_dim)),
                ]
            )
        table = lance.write_dataset(
            pa.Table.from_pylist(records, schema=schema), output / "features.lance"
        )
        manifest = {
            "kind": kind,
            "status": "complete",
            "target_kind": "sample",
            "dataset_id": "test",
            "release_id": "v0.1",
            "rows": num_rows,
            "lance_version": table.version,
            "profile_id": profile,
            "profile": definition,
            "table_path": str((output / "features.lance").relative_to(release)),
            "selection": {"manifest_sha256": file_hash(selection_path)},
        }
        (output / "manifest.json").write_text(json.dumps(manifest))
    recipe = {
        "root": str(root),
        "reference_policy": "self",
        "speaker_conditioning_mode": "frozen_embedding",
        "evaluation": "no_holdout",
        "bindings": [
            {
                "codec_manifest": "datasets/test/v0.1/features/codec/test-run/manifest.json",
                "speaker_manifest": "datasets/test/v0.1/features/speaker_embedding/test-run/manifest.json",
            }
        ],
    }
    return recipe, selection, selection_path


class UnifiedDataTests(unittest.TestCase):
    def test_lance_build_batched_reads_identity_and_spawn(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe, selection, path = fixture(root)
            source_files = {
                p.relative_to(root): file_hash(p) for p in root.rglob("*") if p.is_file()
            }
            output = root / "tts-build-test"
            manifest = bind_features(
                recipe,
                output,
                selection,
                path,
                file_hash(path),
                {"sha256": "fixture", "path": str(root)},
                Tokenizer(),
            )
            self.assertEqual(manifest["bindings"][0]["coverage"], {"ready": 8})
            self.assertEqual(
                {
                    p.relative_to(root): file_hash(p)
                    for p in root.rglob("*")
                    if p.is_file() and not p.is_relative_to(output)
                },
                source_files,
            )
            binding = manifest["bindings"][0]
            self.assertTrue(Path(binding["codec"]["table_path"]).is_relative_to(output))
            source_payloads = {
                checksum for p, checksum in source_files.items() if p.suffix == ".lance"
            }
            self.assertTrue(
                all(
                    file_hash(p) not in source_payloads
                    for p in output.rglob("*.lance")
                    if p.is_file()
                )
            )
            ds = FeatureDataset(output / "manifest.json", (200, 201))
            ds.validate_budgets(8, 100)
            with self.assertRaisesRegex(ValueError, "exceeding"):
                ds.validate_budgets(7, 100)
            rows = ds.__getitems__([7, 0, 4])
            for i, r in zip([7, 0, 4], rows):
                self.assertEqual(r["id"], f"{i:064x}")
                self.assertEqual(r["speaker_embedding"][0], float(i + 1))
                self.assertEqual(r["codes"].shape, (i + 1, 16))
                self.assertEqual((r["text_ids"][0], r["text_ids"][-1]), (200, 201))
            self.assertLess(len(pickle.dumps(ds)), 1000)
            loader = DataLoader(
                ds,
                batch_sampler=[[7, 0, 4], [1, 6]],
                collate_fn=collate,
                num_workers=1,
                multiprocessing_context="spawn",
                timeout=60,
            )
            for batch, indices in zip(list(loader), [[7, 0, 4], [1, 6]]):
                expected = collate(ds.__getitems__(indices))
                for key in expected:
                    torch.testing.assert_close(batch[key], expected[key], atol=0, rtol=0)
            _, _, source = feature_reference(root, recipe["bindings"][0]["codec_manifest"], "codec")
            self.assertNotIn("speaker_row", source.schema.names)

    def test_incomplete_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe, _, _ = fixture(root)
            path = root / recipe["bindings"][0]["codec_manifest"]
            value = json.loads(path.read_text())
            value["status"] = "running"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                feature_reference(root, path, "codec")

    def test_unbound_dataset_is_accounted_for(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe, selection, path = fixture(root)
            selection["outputs"].append(
                {
                    "dataset_id": "unbound",
                    "table_path": "unbound/samples.lance",
                    "branch": "selection",
                    "lance_version": 1,
                    "selected_rows": 2,
                }
            )
            path.write_text(json.dumps(selection))
            for binding in recipe["bindings"]:
                for key in ("codec_manifest", "speaker_manifest"):
                    feature_path = root / binding[key]
                    feature = json.loads(feature_path.read_text())
                    feature["selection"]["manifest_sha256"] = file_hash(path)
                    feature_path.write_text(json.dumps(feature))
            manifest = bind_features(
                recipe,
                root / "build",
                selection,
                path,
                file_hash(path),
                {"sha256": "fixture"},
                Tokenizer(),
            )
            self.assertEqual(manifest["selected_rows"], 10)
            self.assertEqual(sum(manifest["coverage"].values()), 10)
            self.assertEqual(len(FeatureDataset(root / "build/manifest.json", (200, 201))), 8)
            self.assertEqual(manifest["unbound_sources"], [selection["outputs"][-1]])

    def test_wrong_audio_identity_and_duplicate_targets_fail_build(self):
        for change in ("audio", "duplicate"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                recipe, selection, path = fixture(root)
                speaker_path = root / recipe["bindings"][0]["speaker_manifest"]
                manifest = json.loads(speaker_path.read_text())
                table_path = root / "datasets/test/v0.1" / manifest["table_path"]
                ds = lance.dataset(table_path)
                rows = ds.to_table().to_pylist()
                if change == "audio":
                    rows[0]["audio_sha256"] = "wrong"
                if change == "duplicate":
                    rows[0]["target_id"] = rows[1]["target_id"]
                updated = lance.write_dataset(
                    pa.Table.from_pylist(rows, schema=ds.schema), table_path, mode="overwrite"
                )
                manifest["lance_version"] = updated.version
                speaker_path.write_text(json.dumps(manifest))
                with self.assertRaises((ValueError, sqlite3.IntegrityError, OSError)):
                    bind_features(
                        recipe,
                        root / "build",
                        selection,
                        path,
                        file_hash(path),
                        {"sha256": "fixture"},
                        Tokenizer(),
                    )
                self.assertFalse((root / "build/manifest.json").exists())


def test_build_command_uses_experiment_paths(tmp_path, monkeypatch):
    import sys

    import yaml
    from tokenizers import Tokenizer as BackendTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from scripts.build_unified import main

    features = tmp_path / "features"
    recipe, _, _ = fixture(features)
    tokenizer_path = tmp_path / "assembled"
    backend = BackendTokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]").save_pretrained(
        tokenizer_path
    )
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump(
            {"paths": {"features": "/missing/features", "assembled_model": "/missing/model"}}
        )
    )
    recipe.update(
        extends="base.yaml",
        root="${paths.features}",
        tokenizer="${paths.assembled_model}",
        selection_manifest="selection.json",
    )
    recipe_path = tmp_path / "data.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe))
    output = tmp_path / "build/manifest.json"
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        yaml.safe_dump(
            {
                "extends": "base.yaml",
                "paths": {"features": str(features), "assembled_model": str(tokenizer_path)},
                "data": {"config": str(recipe_path), "build": str(output)},
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["build_unified", "--config", str(experiment)])
    main()
    dataset = FeatureDataset(output, (200, 201))
    assert len(dataset) == 8
    assert dataset[0]["text_ids"] == [200, 1, 0, 201]
    assert dataset.manifest["tokenizer"]["path"] == str(tokenizer_path)
