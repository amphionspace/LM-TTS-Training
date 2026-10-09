"""Content, acoustics and paired listening examples from an evaluation report."""

import re

from .audio import load_audio


def write_evaluation(writer, summary, samples, step, settings, *, max_audio_seconds=180):
    count = settings.get("audio_samples", 2)
    seconds = settings.get("audio_seconds", 20)
    if type(count) is not int or count < 0 or not isinstance(seconds, (int, float)) or seconds <= 0:
        raise ValueError("TensorBoard audio_samples must be nonnegative and audio_seconds positive")
    for key in ("wer", "cer", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "speaker_similarity"):
        if key in summary["overall"]:
            writer.add_scalar("eval/" + key, summary["overall"][key], step)
    if len(summary["languages"]) > 1:
        for language, values in summary["languages"].items():
            key = "cer" if language in {"zh", "ja", "ko"} else "wer"
            if key in values:
                writer.add_scalar(f"eval/{key}/{language}", values[key], step)
    for index, row in enumerate(samples[:count]):
        tag = f"samples/{index:02d}-" + re.sub(r"[^\w-]", "_", str(row["id"]))[:64]
        writer.add_text(tag + "/text", row["text"], step)
        if "transcript" in row:
            writer.add_text(tag + "/transcription", row["transcript"], step)
        for name, field in (("generated", "audio"), ("reference", "reference_audio")):
            audio = load_audio(row[field], max_seconds=max_audio_seconds)
            writer.add_audio(
                tag + "/" + name, audio[: int(seconds * 16000)], step, sample_rate=16000
            )
