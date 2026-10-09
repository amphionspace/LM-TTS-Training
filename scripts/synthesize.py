"""Generate a fixed external evaluation set with independent reference recordings."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from lm_tts.artifacts import file_hash
from lm_tts.config import read_yaml


def main():
    from qwen_tts import Qwen3TTSModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="Output of export_checkpoint.py")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_yaml(args.config, keys=("precision", "device", "seed", "generation"))
    if config["precision"] not in {"bf16", "fp32"}:
        raise ValueError("precision must be bf16 or fp32")
    dtype = torch.bfloat16 if config["precision"] == "bf16" else torch.float32
    backend = "flash_attention_2" if dtype == torch.bfloat16 else "sdpa"
    export = json.loads((args.model / "export.json").read_text())
    if not export.get("use_speaker_embedding", True):
        raise ValueError(
            "This speaker-only synthesis entry point requires speaker conditioning; "
            "a no-speaker checkpoint requires codec-prefix inference without a speaker position"
        )
    if file_hash(args.model / "model.safetensors") != export["weights_sha256"]:
        raise ValueError("Export weights changed")
    rows = json.loads(args.prompts.read_text())
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Prompts must have unique IDs and contain at least one sample")
    torch.set_num_threads(4)
    model = Qwen3TTSModel.from_pretrained(
        str(args.model), device_map=config["device"], dtype=dtype, attn_implementation=backend
    )
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for index, row in enumerate(rows):
        reference = Path(row["reference_audio"])
        reference = (args.prompts.resolve().parent / reference).resolve()
        if not row["text"].strip() or not row["language"]:
            raise ValueError("Empty prompt text or missing language")
        seed = config["seed"] + index
        torch.manual_seed(seed)
        wavs, rate = model.generate_voice_clone(
            text=row["text"],
            language="Auto",
            ref_audio=str(reference),
            x_vector_only_mode=True,
            non_streaming_mode=True,
            **config["generation"],
        )
        wav = np.asarray(wavs[0], dtype=np.float32)
        if wav.ndim != 1 or not len(wav) or not np.isfinite(wav).all():
            raise ValueError("Invalid generated waveform")
        audio = (args.output / f"{index:06d}.wav").resolve()
        sf.write(audio, wav, rate, subtype="PCM_16")
        results.append(
            {
                **row,
                "reference_audio": str(reference),
                "audio": str(audio),
                "seed": seed,
                "generated_duration": len(wav) / rate,
                "reference_sha256": file_hash(reference),
            }
        )
        (args.output / "progress.json").write_text(
            json.dumps({"completed": index + 1, "total": len(rows)})
        )
    (args.output / "pairs.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (args.output / "generation.json").write_text(
        json.dumps(
            {
                "export": export,
                "config": config,
                "prompts_sha256": file_hash(args.prompts),
                "pairs_sha256": file_hash(args.output / "pairs.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
