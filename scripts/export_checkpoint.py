"""Export one completed FSDP checkpoint for native Qwen3-TTS inference."""

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file

from qwen3_train.data.build import file_hash
from qwen3_train.models.qwen import TTSModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--assembled-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="fp32")
    args = parser.parse_args()
    if not (args.checkpoint / "COMPLETE").is_file():
        raise ValueError("Cannot export an incomplete checkpoint")
    torch.set_num_threads(4)
    model = TTSModel.from_assembled(
        args.assembled_model, load_weights=False, attn_implementation="sdpa"
    )
    state = model.state_dict()
    dcp.load({"model": state}, checkpoint_id=args.checkpoint / "distributed")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    state = {
        key: value.to(dtype) if value.is_floating_point() else value for key, value in state.items()
    }
    args.output.mkdir(parents=True, exist_ok=False)
    save_file(
        {key: value.contiguous() for key, value in state.items()},
        str(args.output / "model.safetensors"),
        metadata={"format": "pt"},
    )
    for source in args.assembled_model.iterdir():
        if source.name == "speech_tokenizer":
            (args.output / source.name).symlink_to(source.resolve(), target_is_directory=True)
        elif (
            source.is_file()
            and source.suffix in {".json", ".txt"}
            and source.name not in {"assembly_report.json", "export.json"}
        ):
            shutil.copy2(source, args.output / source.name)
    metadata = json.loads((args.checkpoint / "metadata.json").read_text())
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "precision": args.precision,
        "checkpoint_metadata_sha256": file_hash(args.checkpoint / "metadata.json"),
        "weights_sha256": file_hash(args.output / "model.safetensors"),
        "step": metadata["progress"]["step"],
    }
    (args.output / "export.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
