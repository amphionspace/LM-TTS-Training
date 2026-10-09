"""Generate a fixed external evaluation set with independent reference recordings."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from lm_tts.artifacts import file_hash
from lm_tts.config import read_yaml
from lm_tts.inference.codec import CodecSynthesizer, validate_generation
from lm_tts.models.loading import verify_export


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
    export = verify_export(args.model)
    pure_codec = not export.get("use_speaker_embedding", True)
    if pure_codec:
        validate_generation(config["generation"])
    rows = json.loads(args.prompts.read_text())
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Prompts must have unique IDs and contain at least one sample")
    torch.set_num_threads(4)
    model = (
        CodecSynthesizer(args.model, config["device"], dtype, backend)
        if pure_codec
        else Qwen3TTSModel.from_pretrained(
            str(args.model), device_map=config["device"], dtype=dtype, attn_implementation=backend
        )
    )
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for index, row in enumerate(rows):
        reference = row.get("reference_audio")
        reference = (args.prompts.resolve().parent / reference).resolve() if reference else None
        if not row["text"].strip():
            raise ValueError("Empty prompt text")
        seed = config["seed"] + index
        torch.manual_seed(seed)
        if pure_codec:
            wav, rate = model.synthesize(
                row["text"], reference, row.get("reference_text"), **config["generation"]
            )
            wavs = [wav]
        else:
            if reference is None:
                raise ValueError("Speaker-conditioned synthesis requires reference_audio")
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
                "reference_audio": str(reference) if reference else None,
                "audio": str(audio),
                "seed": seed,
                "generated_duration": len(wav) / rate,
                "reference_sha256": file_hash(reference) if reference else None,
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
