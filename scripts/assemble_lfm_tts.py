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
        "--output", type=Path, help="Defaults to a separate directory for each text frontend"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument(
        "--text-frontend",
        choices=["native", "qwen-mlp"],
        default="native",
        help="qwen-mlp doubles the LFM text width and adds a paired-initialized SiLU MLP",
    )
    args = parser.parse_args()
    if args.output is None:
        suffix = "-text2048" if args.text_frontend == "qwen-mlp" else ""
        args.output = root / f"assets/assembled/lfm2.5-230m-base{suffix}-pure-codec"
    torch.set_num_threads(4)
    report = assemble(
        args.backbone,
        args.tts_template,
        args.codec,
        args.output,
        seed=args.seed,
        dtype=getattr(torch, args.dtype),
        text_frontend=args.text_frontend,
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
