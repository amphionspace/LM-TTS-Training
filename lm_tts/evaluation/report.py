"""Evaluate a fixed set of generated/reference pairs outside the training loop."""

import json
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from ..artifacts import file_hash
from ..config import read_yaml
from .audio import load_audio
from .metrics import ASRScorer, aggregate_content
from .quality import DNSMOS
from .similarity import WavLMSimilarity, model_identity
from .telemetry import write_evaluation


def read_pairs(path):
    pairs = json.loads(Path(path).read_text())
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("pairs.json must contain a nonempty array")
    seen = set()
    for row in pairs:
        if row["id"] in seen or not row["text"].strip() or not row["language"]:
            raise ValueError("Duplicate IDs, empty text or missing language in evaluation pairs")
        seen.add(row["id"])
        for key in ("audio", "reference_audio"):
            candidate = Path(row[key])
            row[key] = (
                str((Path(path).resolve().parent / candidate).resolve())
                if not candidate.is_absolute()
                else str(candidate)
            )
        if row["audio"] == row["reference_audio"]:
            raise ValueError("Generated and reference audio must be different files")
    return pairs


def evaluate(config_path, pairs_path, output, *, tensorboard=None, step=None):
    config = read_yaml(
        config_path,
        keys=(
            "metrics",
            "asr_model",
            "dnsmos_model",
            "speaker_model",
            "speaker_device",
            "speaker_chunk_seconds",
            "max_audio_seconds",
            "cpu_threads",
            "tensorboard",
        ),
    )
    enabled = config["metrics"]
    if not enabled or not set(enabled) <= {"content", "quality", "similarity"}:
        raise ValueError("metrics must select content, quality and/or similarity")
    if tensorboard is not None and (step is None or step < 0):
        raise ValueError("TensorBoard evaluation requires an explicit checkpoint step")
    torch.set_num_threads(config.get("cpu_threads", 4))
    pairs = read_pairs(pairs_path)
    identity = {
        "config_sha256": file_hash(config_path),
        "pairs_sha256": file_hash(pairs_path),
        "config": config,
        "step": step,
        "models": {},
        "implementation": {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")},
        "dependencies": {
            name: version(name)
            for name in [
                "torch",
                "transformers",
                "faster-whisper",
                "onnxruntime",
                "numpy",
                "scipy",
                "soundfile",
            ]
        },
        "content_normalization": "NFKC_lowercase_unicode_word_tokens_v1",
    }
    asr = quality = similarity = None
    if "content" in enabled:
        identity["models"]["asr"] = model_identity(config["asr_model"])
        asr = ASRScorer(config["asr_model"])
    if "quality" in enabled:
        quality = DNSMOS(config["dnsmos_model"], config.get("cpu_threads", 4))
        identity["models"]["quality"] = quality.identity
    if "similarity" in enabled:
        similarity = WavLMSimilarity(
            config["speaker_model"],
            config.get("speaker_device", "cpu"),
            config.get("speaker_chunk_seconds", 10),
        )
        identity["models"]["similarity"] = similarity.identity
    results = []
    for row in pairs:
        generated = load_audio(row["audio"], max_seconds=config.get("max_audio_seconds", 180))
        result = {
            **row,
            "audio_sha256": file_hash(row["audio"]),
            "duration": len(generated) / 16000,
        }
        if asr:
            result.update(asr.score(row["audio"], row["text"], row["language"]))
        if quality:
            result.update(quality.score(generated))
        if similarity:
            reference = load_audio(
                row["reference_audio"], max_seconds=config.get("max_audio_seconds", 180)
            )
            result["reference_sha256"] = file_hash(row["reference_audio"])
            if result["reference_sha256"] == result["audio_sha256"]:
                raise ValueError("Reference is a byte-identical copy of generated audio")
            result["speaker_similarity"] = similarity.score(generated, reference)
        results.append(result)
    languages = defaultdict(list)
    for row in results:
        languages[row["language"]].append(row)
    acoustic_keys = [
        k
        for k in ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "speaker_similarity")
        if k in results[0]
    ]

    def summarize(rows):
        value = aggregate_content(rows) if asr else {"samples": len(rows)}
        value.update({key: float(np.mean([r[key] for r in rows])) for key in acoustic_keys})
        return value

    summary = {
        "overall": summarize(results),
        "languages": {k: summarize(v) for k, v in languages.items()},
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "samples.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False)
    )
    (output / "report.json").write_text(
        json.dumps({"identity": identity, **summary}, indent=2, ensure_ascii=False, allow_nan=False)
    )
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        with SummaryWriter(str(tensorboard)) as writer:
            write_evaluation(
                writer,
                summary,
                results,
                step,
                config.get("tensorboard", {}),
                max_audio_seconds=config.get("max_audio_seconds", 180),
            )
    return summary
