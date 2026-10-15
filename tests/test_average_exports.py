"""An averaged export must preserve conditioning and combine corresponding tensors."""

import json
import sys

import pytest
import torch
from safetensors.torch import load_file, save_file

from lm_tts.artifacts import file_hash
from scripts import average_exports


def test_average_preserves_frozen_weights_and_integer_buffers():
    a = {
        "trainable": torch.tensor([2.0, -4.0]),
        "frozen": torch.tensor([3.0]),
        "count": torch.tensor(7),
    }
    b = {
        "trainable": torch.tensor([6.0, 2.0]),
        "frozen": torch.tensor([3.0]),
        "count": torch.tensor(7),
    }
    result = average_exports.average_states(a, b)
    torch.testing.assert_close(result["trainable"], torch.tensor([4.0, -1.0]), rtol=0, atol=0)
    assert torch.equal(result["frozen"], a["frozen"])
    assert torch.equal(result["count"], a["count"])
    assert torch.equal(a["trainable"], torch.tensor([2.0, -4.0]))


@pytest.mark.parametrize(
    "right",
    [
        {"other": torch.tensor([1.0])},
        {"w": torch.tensor([1.0, 2.0])},
        {"w": torch.tensor([float("nan")])},
        {"w": torch.tensor([1.0], dtype=torch.float64)},
    ],
)
def test_rejects_incompatible_or_nonfinite_states(right):
    with pytest.raises(ValueError):
        average_exports.average_states({"w": torch.tensor([1.0])}, right)


def test_rejects_different_integer_buffers():
    with pytest.raises(ValueError, match="Non-floating"):
        average_exports.average_states({"x": torch.tensor(1)}, {"x": torch.tensor(2)})


def test_complete_export_records_sources_and_loads(tmp_path, monkeypatch):
    sources = [tmp_path / "a", tmp_path / "b"]
    for source, value in zip(sources, [2.0, 6.0], strict=True):
        source.mkdir()
        save_file({"weight": torch.tensor([value])}, source / "model.safetensors")
        (source / "config.json").write_text("{}")
        (source / "speech_tokenizer").mkdir()
        (source / "speech_tokenizer/codec").write_bytes(b"codec")
        hashes = {p.name: file_hash(p) for p in source.iterdir() if p.is_file()}
        (source / "export.json").write_text(
            json.dumps(
                {
                    "artifact_sha256": hashes,
                    "weights_sha256": hashes["model.safetensors"],
                    "use_speaker_embedding": True,
                }
            )
        )
    output = tmp_path / "average"
    monkeypatch.setattr(
        sys, "argv", ["average", "--sources", *map(str, sources), "--output", str(output)]
    )
    average_exports.main()
    assert load_file(output / "model.safetensors")["weight"].item() == 4.0
    report = json.loads((output / "export.json").read_text())
    assert report["weights_sha256"] == file_hash(output / "model.safetensors")
    assert [r["weight"] for r in report["sources"]] == [0.5, 0.5]
    assert report["use_speaker_embedding"] is True
    assert (output / "speech_tokenizer/codec").read_bytes() == b"codec"
    from lm_tts.models.loading import verify_export

    verify_export(output)
