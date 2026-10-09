"""Removing speaker conditioning must not alter targets or require speaker payloads."""

import copy
import pickle
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from test_merged_data import make_build

from lm_tts.data.batch import collate
from lm_tts.data.reader import FeatureDataset
from lm_tts.objectives.tts import tts_loss
from lm_tts.training.config import load_config


def test_no_speaker_removes_position_and_preserves_causal_targets():
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
    from test_qwen_protocol import FrozenFrontendProtocolTests

    from lm_tts.models.qwen import TTSModel

    fixture = FrozenFrontendProtocolTests()
    fixture.setUp()
    speaker = Qwen3TTSSpeakerEncoderConfig(enc_dim=64, mel_dim=8, enc_channels=[16, 16, 16, 16, 48])
    model = TTSModel(fixture.model.config, speaker).eval()
    row = fixture.row
    row = {k: v for k, v in row.items() if k != "speaker_mels"}
    row["speaker_embedding"] = torch.randn(model.config.hidden_size)
    conditioned, _ = model.input_embeddings(collate([row]))
    model.config.lm_tts_use_speaker_embedding = False
    batch = collate([row])
    with patch.object(model.speaker_encoder, "forward", side_effect=AssertionError("encoder used")):
        inputs, positions = model.input_embeddings(batch)
        # Three role positions and three codec controls precede the speaker position.
        expected = torch.cat(
            [conditioned["inputs_embeds"][:, :6], conditioned["inputs_embeds"][:, 7:]], dim=1
        )
        torch.testing.assert_close(inputs["inputs_embeds"], expected, rtol=0, atol=0)
        assert positions.numel() == len(row["codes"]) + 1
        predicted = model(batch)
        clean = {k: v for k, v in batch.items() if k != "speaker_embeddings"}
        poisoned = {
            **batch,
            "speaker_embeddings": torch.full_like(batch["speaker_embeddings"], float("nan")),
        }
        for other in (clean, poisoned):
            actual = model(other)
            torch.testing.assert_close(
                actual["first_logits"], predicted["first_logits"], rtol=0, atol=0
            )
            for a, b in zip(actual["residual_logits"], predicted["residual_logits"]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        # Full teacher forcing and an actual codec prefix predict the same next frame.
        for t in range(len(row["codes"]) + 1):
            prefix = collate([{**row, "codes": row["codes"][:t]}])
            logits = model.talker.codec_head(model.hidden(prefix)[-1:])
            torch.testing.assert_close(
                logits, predicted["first_logits"][t : t + 1], atol=2e-6, rtol=2e-5
            )
        loss = tts_loss(predicted, batch, model.eos)
        (loss["first_sum"] + loss["residual_sum"]).backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.requires_grad
        )
        assert all(p.grad is None for p in model.speaker_encoder.parameters())
        assert all(p.grad is None for p in model.talker.model.text_embedding.parameters())
        assert all(p.grad is None for p in model.talker.text_projection.parameters())


def test_no_speaker_reader_omits_payload_and_survives_worker_serialization(tmp_path):
    original = make_build(tmp_path / "source", tmp_path / "build")
    dataset = FeatureDataset(original.path, (200, 201), use_speaker_embedding=False)
    dataset = pickle.loads(pickle.dumps(dataset))
    table = dataset.table(0, "merged")
    take = table.take

    def without_speaker(indices, *, columns):
        assert "speaker_embedding" not in columns
        return take(indices, columns=columns)

    with patch.object(table, "take", side_effect=without_speaker):
        rows = dataset.__getitems__([0, 4, 7])
    baseline = original.__getitems__([0, 4, 7])
    assert dataset.fingerprint == original.fingerprint
    for actual, expected in zip(rows, baseline):
        assert "speaker_embedding" not in actual
        assert actual["id"] == expected["id"]
        assert actual["text_ids"] == expected["text_ids"]
        torch.testing.assert_close(actual["codes"], expected["codes"], rtol=0, atol=0)
    assert "speaker_embeddings" not in collate(rows)


def test_ablation_config_only_changes_intended_experiment_settings(tmp_path):
    root = Path(__file__).resolve().parents[1]
    baseline = load_config(
        root / "configs/supervised-tts-20260929-all16-bf16-32gpu-lr3e-4-bblr1e-4-ep3-wsd.yaml"
    )
    experiment = load_config(
        root
        / "configs/supervised-tts-20260929-all16-no-spk-bf16-16gpu-acc2-lr3e-4-bblr1e-4-ep3-wsd.yaml"
    )
    assert experiment["model"]["use_speaker_embedding"] is False
    assert experiment["train"]["accumulation"] == 2
    assert experiment["train"]["keep_checkpoints"] is None
    comparable = copy.deepcopy(experiment)
    comparable["model"].pop("use_speaker_embedding")
    for key in ("accumulation", "keep_checkpoints", "run_name", "output"):
        comparable["train"][key] = baseline["train"][key]
    comparable["config_path"] = baseline["config_path"]
    assert comparable == baseline
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        f"extends: {root}/configs/train-bf16.yaml\nmodel:\n  use_speaker_embedding: 'false'\n"
    )
    with pytest.raises(ValueError, match="must be a boolean"):
        load_config(bad)
