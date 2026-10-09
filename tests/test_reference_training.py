"""Reference targets are excluded without removing their causal input history."""

import json
from pathlib import Path

import lance
import pyarrow as pa
import pytest
import torch
from test_merged_data import merged_fixture
from test_unified_data import Tokenizer

from qwen3_train.artifacts import digest, file_hash
from qwen3_train.data.batch import collate
from qwen3_train.data.build import open_snapshot
from qwen3_train.data.merged import bind_merged
from qwen3_train.data.reader import FeatureDataset
from qwen3_train.data.reference_build import aligned_cache, create_reference_build
from qwen3_train.data.split import split_build
from qwen3_train.objectives.tts import loss_normalizers, tts_loss


def reference_fixture(
    root, *, embedding_dim=64, speaker_profile=None, tokenizer_info=None, tokenizer=None
):
    source = root / "unified"
    recipe, selection, selection_path = merged_fixture(
        source, embedding_dim=embedding_dim, speaker_profile=speaker_profile
    )
    old_path = Path(recipe["bindings"][0]["merged_manifest"])
    old = json.loads(old_path.read_text())
    table_path = old_path.parent / "features.lance"
    rows = lance.dataset(table_path).to_table().to_pylist()
    for i, row in enumerate(rows):
        frames = 20 + i
        row.update(
            codec_num_codec_frames=frames,
            codec_codes=[[i] * 16 for _ in range(frames)],
            codec_end_frame=frames * 1920,
            speaker_end_frame=frames * 1920,
            codec_feature_key=digest([i, "codec"]),
        )
    table = lance.write_dataset(pa.Table.from_pylist(rows), table_path, mode="overwrite")
    old["lance_version"] = table.version
    old_path.write_text(json.dumps(old))
    tokenizer_info = tokenizer_info or {"sha256": "fixture"}
    bind_merged(
        recipe,
        root / "baseline",
        selection,
        selection_path,
        file_hash(selection_path),
        tokenizer_info,
        tokenizer or Tokenizer(),
    )
    split_build(root / "baseline/manifest.json", fraction=0.25)

    speaker_source = next(s for s in old["inputs"] if s["alias"] == "speaker")
    speaker_path = Path(speaker_source["manifest_path"])
    speaker = json.loads(speaker_path.read_text())
    speaker["profile"] = {
        **speaker["profile"],
        "reference": {
            "version": "codec-grid-reference-v1",
            "codec_sample_rate": 24000,
            "codec_hop_samples": 1920,
        },
    }
    speaker["profile_id"] = digest(speaker["profile"])
    new_speaker_path = speaker_path.with_name("reference-manifest.json")
    new_speaker_path.write_text(json.dumps(speaker))
    sources = [{**s} for s in old["inputs"]]
    for s in sources:
        if s["alias"] == "speaker":
            s.update(
                manifest_path=str(new_speaker_path),
                manifest_sha256=file_hash(new_speaker_path),
                profile_id=speaker["profile_id"],
            )
    # Remove a training row and a validation row to test ID-preserving partitioning.
    validation = FeatureDataset(root / "baseline/validation/manifest.json", (200, 201))
    validation_rows = set(map(int, validation.indices[0]["row"]))
    removed = {min(validation_rows), min(set(range(8)) - validation_rows)}
    kept = []
    for i, row in enumerate(rows):
        if i in removed:
            continue
        start = (i % 3) * 2
        row.update(
            speaker_profile_id=speaker["profile_id"],
            speaker_start_frame=start * 1920,
            speaker_end_frame=(start + 7) * 1920,
            speaker_reference_codec_start=start,
            speaker_reference_codec_end=start + 7,
            speaker_reference_codec_feature_key=row["codec_feature_key"],
            speaker_reference_native_total_frames=row["codec_end_frame"],
        )
        kept.append(row)
    directory = old_path.parent.parent / "reference"
    directory.mkdir()
    table = lance.write_dataset(pa.Table.from_pylist(kept), directory / "features.lance")
    profile = {
        **old["profile"],
        "speaker_mode": "codec_grid_reference",
        "sources": {**old["profile"]["sources"], "speaker": speaker["profile_id"]},
    }
    merged = {
        **old,
        "rows": len(kept),
        "coverage": {"ok": len(kept)},
        "inputs": sources,
        "profile": profile,
        "profile_id": digest(profile),
        "lance_version": table.version,
        "table_path": "features/merged/reference/features.lance",
        "reference_filtering": {
            "input_rows": 8,
            "retained": len(kept),
            "excluded_reasons": {"no_legal_reference": 2},
        },
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(merged))
    recipe.update(
        reuse_build=str(root / "baseline/manifest.json"),
        workers=1,
        bindings=[{"dataset_id": "test", "merged_manifest": str(path)}],
    )
    create_reference_build(
        recipe, root / "reference-build", tokenizer_info, file_hash(selection_path)
    )
    return root / "reference-build", validation_rows, removed


def test_reference_build_keeps_ids_tokens_and_holdout(tmp_path):
    build, old_validation, removed = reference_fixture(tmp_path)
    train = FeatureDataset(build / "train/manifest.json", (200, 201), mask_reference=True)
    val = FeatureDataset(build / "validation/manifest.json", (200, 201), mask_reference=True)
    old = FeatureDataset(tmp_path / "baseline/manifest.json", (200, 201))

    class SmallBatches:
        def __init__(self, dataset, size):
            self.dataset, self.size = dataset, size

        def scanner(self, **kwargs):
            return self.dataset.scanner(**{**kwargs, "batch_size": self.size})

    # Filtered samples and different chunk boundaries must preserve token-to-row alignment.
    rebuilt = []
    for rows, locators, cached in aligned_cache(
        SmallBatches(open_snapshot(train.bindings[0]["merged"]), 3),
        SmallBatches(open_snapshot(old.bindings[0]["merged"]), 2),
        train.bindings[0],
    ):
        for row, locator, tokens in zip(rows, locators, cached.to_pylist(), strict=True):
            expected = old[int(locator)]
            assert row["target_id"] == expected["id"]
            assert tokens["text_ids"] == expected["text_ids"][1:-1]
            rebuilt.append(row["target_id"])
    assert len(rebuilt) == 6
    assert len(train) + len(val) == 6
    assert {int(val[i]["id"], 16) for i in range(len(val))} == old_validation - removed
    for ds in (train, val):
        for i in range(len(ds)):
            row = ds[i]
            original = old[int(row["id"], 16)]
            assert row["text_ids"] == original["text_ids"]
            torch.testing.assert_close(row["codes"], original["codes"])
            batch = collate([row])
            assert batch["supervised_frame_lengths"].item() == len(row["codes"]) - 7
    with pytest.raises(ValueError, match="requires train.mask_reference"):
        FeatureDataset(build / "train/manifest.json", (200, 201))
    row = open_snapshot(train.bindings[0]["merged"]).take([0]).to_pylist()[0]
    from qwen3_train.data.merged import validate_row

    row["speaker_reference_codec_end"] += 1
    with pytest.raises(ValueError, match="reference-to-codec"):
        validate_row(row, train.bindings[0])


@pytest.mark.parametrize("reduction", ["token", "sample", "sqrt"])
def test_reference_loss_matches_explicit_targets_and_accumulation(reduction):
    torch.manual_seed(123)
    rows = [
        {"text_ids": [1, 2], "codes": torch.randint(0, 8, (n, 16)), "reference_interval": interval}
        for n, interval in [(4, (0, 2)), (6, (2, 4)), (3, (1, 3))]
    ]
    batch = collate(rows)
    predictions = {
        "first_logits": torch.randn(16, 9, requires_grad=True),
        "residual_logits": tuple(torch.randn(13, 8, requires_grad=True) for _ in range(15)),
    }
    out = tts_loss(predictions, batch, eos=8, reduction=reduction)
    normalizers = loss_normalizers(batch["supervised_frame_lengths"], reduction)
    actual = (
        out["first_reduced_sum"] / normalizers[0]
        + 0.3 * out["residual_reduced_sum"] / normalizers[1]
    )
    exponent = {"token": 1, "sample": 0, "sqrt": 0.5}[reduction]
    expected = 0
    offset = audio_offset = 0
    for row in rows:
        n = len(row["codes"])
        a, b = row["reference_interval"]
        count = n - (b - a)
        for t in range(n + 1):
            if a <= t < b:
                continue
            label = row["codes"][t, 0] if t < n else torch.tensor(8)
            ce = torch.nn.functional.cross_entropy(
                predictions["first_logits"][offset + t : offset + t + 1], label.reshape(1)
            )
            expected = expected + ce * (count + 1) ** (exponent - 1) / normalizers[0]
            if t < n:
                for g in range(15):
                    ce = torch.nn.functional.cross_entropy(
                        predictions["residual_logits"][g][audio_offset + t : audio_offset + t + 1],
                        row["codes"][t, g + 1].reshape(1),
                    )
                    expected = expected + 0.3 * ce * (15 * count) ** (exponent - 1) / normalizers[1]
        offset += n + 1
        audio_offset += n
    torch.testing.assert_close(actual, expected)
    assert out["first_count"] == 10 and out["frame_count"] == 7
    variables = [predictions["first_logits"], *predictions["residual_logits"]]
    expected_grad = torch.autograd.grad(expected, variables, retain_graph=True)
    actual_grad = torch.autograd.grad(actual, variables, retain_graph=True)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b)
    # Masked prediction positions get zero direct CE gradient; each EOS remains active.
    assert not actual_grad[0][:2].any()
    assert actual_grad[0][4].abs().sum() > 0
    first = audio = 0
    accumulated = 0
    for group in (rows[:1], rows[1:]):
        micro = collate(group)
        n = int(micro["frame_lengths"].sum())
        k = len(group)
        logits = {
            "first_logits": predictions["first_logits"][first : first + n + k],
            "residual_logits": tuple(v[audio : audio + n] for v in predictions["residual_logits"]),
        }
        result = tts_loss(logits, micro, 8, reduction)
        accumulated += (
            result["first_reduced_sum"] / normalizers[0]
            + 0.3 * result["residual_reduced_sum"] / normalizers[1]
        )
        first += n + k
        audio += n
    torch.testing.assert_close(accumulated, actual)
