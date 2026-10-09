"""Local weight and codec artifact operations shared by assembly and loading."""

import json
import shutil
from pathlib import Path

from safetensors import safe_open

TEXT_SPECIALS = {
    "tts_pad_token_id": "<tts_pad>",
    "tts_bos_token_id": "<tts_text_bos>",
    "tts_eos_token_id": "<tts_text_eod>",
    "im_start_token_id": "<|im_start|>",
    "im_end_token_id": "<|im_end|>",
}


def load_prefix(directory, prefix):
    directory = Path(directory)
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())["weight_map"]
        filenames = sorted({name for key, name in index.items() if key.startswith(prefix)})
    else:
        filenames = [p.name for p in sorted(directory.glob("model*.safetensors"))]
    values = {}
    for name in filenames:
        with safe_open(directory / name, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    short = key[len(prefix) :]
                    if short in values:
                        raise ValueError(f"Duplicate checkpoint tensor: {key}")
                    values[short] = f.get_tensor(key)
    if not values:
        raise ValueError(f"No {prefix} weights found in {directory}")
    return values


def copy_codec(source, destination):
    source, destination = Path(source), Path(destination)
    if not (source / "config.json").exists() or not list(source.glob("*.safetensors")):
        raise ValueError("Codec directory needs config.json and safetensors weights")
    shutil.copytree(
        source, destination, ignore=shutil.ignore_patterns(".cache", ".git", ".ipynb_checkpoints")
    )
