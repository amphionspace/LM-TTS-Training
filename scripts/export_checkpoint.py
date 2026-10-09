"""Export one completed FSDP checkpoint as flat model weights and tokenizer artifacts."""

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file

from lm_tts.artifacts import file_hash
from lm_tts.models.loading import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--assembled-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="fp32")
    parser.add_argument(
        "--copy-tokenizer",
        action="store_true",
        help="Copy the speech tokenizer for a self-contained export instead of linking it",
    )
    args = parser.parse_args()
    if not (args.checkpoint / "COMPLETE").is_file():
        raise ValueError("Cannot export an incomplete checkpoint")
    metadata = json.loads((args.checkpoint / "metadata.json").read_text())
    if (
        file_hash(args.assembled_model / "assembly_report.json")
        != metadata["signature"]["assembly_sha256"]
    ):
        raise ValueError("Checkpoint and assembled model have different assembly identities")
    torch.set_num_threads(4)
    model = load_model(
        args.assembled_model,
        load_weights=False,
        attn_implementation="sdpa",
        use_speaker_embedding=metadata["signature"].get("model", {}).get("use_speaker_embedding"),
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
            if args.copy_tokenizer:
                shutil.copytree(source.resolve(), args.output / source.name)
            else:
                (args.output / source.name).symlink_to(source.resolve(), target_is_directory=True)
        elif (
            source.is_file()
            and source.suffix in {".json", ".txt"}
            and source.name
            not in {"assembly_report.json", "export.json", "model.safetensors.index.json"}
        ):
            shutil.copy2(source, args.output / source.name)
    use_speaker = metadata["signature"].get("model", {}).get("use_speaker_embedding", True)
    if not use_speaker:
        config_path = args.output / "config.json"
        config = json.loads(config_path.read_text())
        config.setdefault("talker_config", {})["lm_tts_use_speaker_embedding"] = False
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        print(
            "No-speaker export: inference must omit the speaker position; vanilla Qwen voice_clone does not honor this flag."
        )
    report = {
        "use_speaker_embedding": use_speaker,
        "checkpoint": str(args.checkpoint.resolve()),
        "precision": args.precision,
        "checkpoint_metadata_sha256": file_hash(args.checkpoint / "metadata.json"),
        "weights_sha256": file_hash(args.output / "model.safetensors"),
        "step": metadata["progress"]["step"],
    }
    report["artifact_sha256"] = {
        p.name: file_hash(p)
        for p in sorted(args.output.iterdir())
        if p.is_file() and p.name != "export.json"
    }
    (args.output / "export.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
