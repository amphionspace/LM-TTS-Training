"""Assemble a local LFM2.5 Base into a pure-codec TTS training artifact."""

import argparse
import json
from pathlib import Path

import torch

from lm_tts.models.assembly.lfm import assemble


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=Path, default=root / "assets/base/LFM2.5-230M-Base")
    parser.add_argument(
        "--tts-template", type=Path, default=root / "assets/base/Qwen3-TTS-12Hz-0.6B-Base"
    )
    parser.add_argument("--codec", type=Path, default=root / "assets/base/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument(
        "--output", type=Path, default=root / "assets/assembled/lfm2.5-230m-base-pure-codec"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = assemble(
        args.backbone,
        args.tts_template,
        args.codec,
        args.output,
        seed=args.seed,
        dtype=getattr(torch, args.dtype),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "parameters": report["parameters"],
                "verification": report["verification"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
