"""Exports must preserve weights and optionally remove dependencies on source assets."""

import json
import sys

import pytest
import torch
from safetensors.torch import load_file

from lm_tts.artifacts import file_hash
from scripts import export_checkpoint


@pytest.mark.parametrize("copy_tokenizer", [False, True])
@pytest.mark.parametrize("use_speaker_embedding", [None, False])
def test_export_tokenizer_and_flat_weights(
    tmp_path, monkeypatch, copy_tokenizer, use_speaker_embedding
):
    assembled = tmp_path / "assembled"
    assembled.mkdir()
    (assembled / "assembly_report.json").write_text('{"assembly": "fixture"}')
    (assembled / "model.safetensors.index.json").write_text(
        '{"weight_map": {"weight": "old-shard"}}'
    )
    (assembled / "config.json").write_text("{}")
    (assembled / "speech_tokenizer").mkdir()
    (assembled / "speech_tokenizer" / "weights").write_bytes(b"codec")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "COMPLETE").touch()
    (checkpoint / "metadata.json").write_text(
        json.dumps(
            {
                "signature": {
                    "assembly_sha256": file_hash(assembled / "assembly_report.json"),
                    **(
                        {"model": {"use_speaker_embedding": use_speaker_embedding}}
                        if use_speaker_embedding is not None
                        else {}
                    ),
                },
                "progress": {"step": 123},
            }
        )
    )
    model = torch.nn.Linear(2, 2, bias=False)
    monkeypatch.setattr(export_checkpoint, "load_model", lambda *a, **kw: model)

    def load(state, **kwargs):
        state["model"]["weight"].fill_(0.25)

    monkeypatch.setattr(export_checkpoint.dcp, "load", load)
    output = tmp_path / "output"
    argv = [
        "export",
        "--checkpoint",
        str(checkpoint),
        "--assembled-model",
        str(assembled),
        "--output",
        str(output),
    ]
    if copy_tokenizer:
        argv.append("--copy-tokenizer")
    monkeypatch.setattr(sys, "argv", argv)
    export_checkpoint.main()
    assert torch.equal(load_file(output / "model.safetensors")["weight"], torch.full((2, 2), 0.25))
    assert not (output / "model.safetensors.index.json").exists()
    assert (output / "speech_tokenizer").is_symlink() is not copy_tokenizer
    if copy_tokenizer:
        (assembled / "speech_tokenizer" / "weights").unlink()
    assert (output / "speech_tokenizer" / "weights").read_bytes() == b"codec"
    report = json.loads((output / "export.json").read_text())
    assert report["step"] == 123
    assert report["weights_sha256"] == file_hash(output / "model.safetensors")
    if use_speaker_embedding is False:
        assert report["use_speaker_embedding"] is False
        config = json.loads((output / "config.json").read_text())
        assert config["talker_config"]["lm_tts_use_speaker_embedding"] is False


def test_export_rejects_a_different_assembly_before_loading_weights(tmp_path, monkeypatch):
    checkpoint, assembled = tmp_path / "checkpoint", tmp_path / "assembled"
    checkpoint.mkdir()
    assembled.mkdir()
    (checkpoint / "COMPLETE").touch()
    (checkpoint / "metadata.json").write_text(
        json.dumps({"signature": {"assembly_sha256": "another-assembly"}})
    )
    (assembled / "assembly_report.json").write_text("{}")
    output = tmp_path / "export"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_checkpoint",
            "--checkpoint",
            str(checkpoint),
            "--assembled-model",
            str(assembled),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="different assembly identities"):
        export_checkpoint.main()
    assert not output.exists()


@pytest.mark.parametrize("family,speaker", [("lfm", False), ("qwen", False), ("qwen", True)])
def test_real_dcp_export_reload_and_codec_generation(tmp_path, monkeypatch, family, speaker):
    import torch.distributed.checkpoint as dcp
    from check_training_resume import prepare

    from lm_tts.inference.codec import generate_codes
    from lm_tts.models.loading import load_model

    root = tmp_path / "case"
    prepare(root, "fp32", model_family=family, use_speaker_embedding=speaker)
    assembled = root / "assembled"
    model = load_model(assembled, attn_implementation="sdpa", use_speaker_embedding=speaker).eval()
    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    dcp.save({"model": model.state_dict()}, checkpoint_id=checkpoint / "distributed")
    (checkpoint / "metadata.json").write_text(
        json.dumps(
            {
                "signature": {
                    "assembly_sha256": file_hash(assembled / "assembly_report.json"),
                    "model": {"use_speaker_embedding": speaker},
                },
                "progress": {"step": 7},
            }
        )
    )
    (checkpoint / "COMPLETE").touch()
    output = root / "export"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export",
            "--checkpoint",
            str(checkpoint),
            "--assembled-model",
            str(assembled),
            "--output",
            str(output),
        ],
    )
    export_checkpoint.main()
    restored = load_model(output, attn_implementation="sdpa").eval()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)
    if not speaker:
        for prefix in (None, torch.arange(32).reshape(2, 16)):
            expected = generate_codes(
                model, [200, 2, 201], prefix, max_new_tokens=2, min_new_tokens=2
            )
            actual = generate_codes(
                restored, [200, 2, 201], prefix, max_new_tokens=2, min_new_tokens=2
            )
            assert actual.shape == (2, 16)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        with pytest.raises(ValueError, match="conditioning"):
            load_model(output, attn_implementation="sdpa", use_speaker_embedding=True)
    (output / "config.json").write_text((output / "config.json").read_text() + "\n")
    with pytest.raises(ValueError, match="artifact changed"):
        load_model(output, attn_implementation="sdpa")
