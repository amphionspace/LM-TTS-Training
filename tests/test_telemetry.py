import json

import numpy as np
import pytest
import soundfile as sf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from qwen3_train.evaluation.telemetry import write_evaluation
from qwen3_train.training.telemetry import tensorboard_groups, write_training
from scripts.tensorboard import rebuild


def test_dashboard_restores_journal_and_preserves_steps(tmp_path):
    metrics = {
        "objective": 3.0,
        "first_ce": 2.0,
        "residual_ce": 1.0,
        "grad_norm": 0.5,
        "lr_backbone": 1e-5,
        "lr_new": 1e-4,
        "audio_seconds_per_second": 8.0,
        "step_seconds": 2.0,
        "data_wait_seconds": 0.1,
        "peak_memory_gib": 4.0,
        "global_samples": 16,
        "audio_seconds": 16,
        "codec_tokens": 3200,
    }
    (tmp_path / "metrics.jsonl").write_text(
        "\n".join(json.dumps({"step": step, "train": metrics}) for step in (10, 20))
    )
    rebuild(tmp_path)
    with (tmp_path / "metrics.jsonl").open("a") as journal:
        journal.write("\n" + json.dumps({"step": 20, "train": {**metrics, "first_ce": 1.5}}))
        journal.write("\n" + json.dumps({"step": 30, "train": metrics}))
    rebuild(tmp_path)
    rebuild(tmp_path)
    events = EventAccumulator(str(tmp_path / "tensorboard/restored")).Reload()
    assert [event.step for event in events.Scalars("train/first_ce")] == [10, 20, 30]
    assert events.Scalars("train/first_ce")[1].value == 1.5
    assert events.Scalars("batch/global_samples")[0].value == 16
    assert "custom_scalars__config__" in events.Tags()["tensors"]
    with SummaryWriter(str(tmp_path / "limited")) as writer:
        write_training(writer, metrics, 20, ["optimization"])
    limited = EventAccumulator(str(tmp_path / "limited")).Reload()
    assert "train/first_ce" in limited.Tags()["scalars"]
    assert "performance/step_seconds" not in limited.Tags()["scalars"]
    with pytest.raises(ValueError, match="groups"):
        tensorboard_groups({"tensorboard": {"groups": ["unknown"]}})


def test_evaluation_dashboard_has_scores_and_bounded_paired_audio(tmp_path):
    generated, reference = tmp_path / "generated.wav", tmp_path / "reference.wav"
    sf.write(generated, np.zeros(32000), 16000)
    sf.write(reference, np.ones(32000) * 0.1, 16000)
    row = {
        "id": "sample/one",
        "text": "hello",
        "transcript": "hello",
        "audio": str(generated),
        "reference_audio": str(reference),
    }
    summary = {
        "overall": {
            "wer": 0.0,
            "cer": 0.0,
            "dnsmos_sig": 3.0,
            "dnsmos_bak": 4.0,
            "dnsmos_ovrl": 3.5,
            "speaker_similarity": 0.8,
        },
        "languages": {"en": {"wer": 0.0}},
    }
    with SummaryWriter(str(tmp_path / "events")) as writer:
        write_evaluation(writer, summary, [row], 500, {"audio_samples": 1, "audio_seconds": 1})
    events = EventAccumulator(str(tmp_path / "events")).Reload()
    assert events.Scalars("eval/dnsmos_sig")[0].step == 500
    audio = events.Tags()["audio"]
    assert len(audio) == 2
    for tag in audio:
        assert events.Audio(tag)[0].length_frames == 16000
    assert "samples/00-sample_one/text/text_summary" in events.Tags()["tensors"]
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    (evaluation / "samples.json").write_text(json.dumps([row]))
    (evaluation / "report.json").write_text(
        json.dumps(
            {
                **summary,
                "identity": {
                    "step": 500,
                    "config": {"tensorboard": {"audio_samples": 1, "audio_seconds": 1}},
                },
            }
        )
    )
    rebuild(tmp_path)
    rebuild(tmp_path)
    restored = tmp_path / "tensorboard/restored"
    replay = EventAccumulator(str(restored)).Reload()
    assert len(replay.Scalars("eval/wer")) == 1
    for tag in replay.Tags()["audio"]:
        assert len(replay.Audio(tag)) == 1
        assert replay.Audio(tag)[0].length_frames == 16000
    previous = {p.name: p.read_bytes() for p in restored.iterdir()}
    reference.unlink()
    with pytest.raises(sf.LibsndfileError):
        rebuild(tmp_path)
    assert {p.name: p.read_bytes() for p in restored.iterdir()} == previous
