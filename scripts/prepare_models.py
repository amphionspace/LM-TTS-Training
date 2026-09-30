"""Copy local training sources and assemble them without downloading weights."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from qwen3_train.config import read_yaml
from qwen3_train.models.assembly import sha256

SOURCES = {
    "backbone": "Qwen3-0.6B-Base",
    "tts_template": "Qwen3-TTS-12Hz-0.6B-Base",
    "codec": "Qwen3-TTS-Tokenizer-12Hz",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train-bf16.yaml")
    parser.add_argument("--source", type=Path, default=Path("/workspace/model"))
    args = parser.parse_args()
    config = read_yaml(args.config)
    target = Path(config["paths"]["project"]) / "assets/base"
    target.mkdir(parents=True, exist_ok=True)
    for name in SOURCES.values():
        source, destination = args.source / name, target / name
        if not (source / "config.json").exists() or not list(source.glob("model*.safetensors")):
            parser.error(f"Missing local model weights: {source}")
        if not destination.exists():
            temporary = destination.with_name(destination.name + ".incomplete")
            shutil.copytree(
                source, temporary, ignore=shutil.ignore_patterns(".cache", "speech_tokenizer")
            )
            hashes = {}
            for path in temporary.rglob("*"):
                if path.is_file():
                    relative = path.relative_to(temporary)
                    digest = sha256(path)
                    if digest != sha256(source / relative):
                        raise RuntimeError(f"Copy verification failed: {relative}")
                    hashes[str(relative)] = digest
            (temporary / "LOCAL_COPY.json").write_text(
                json.dumps({"source": str(source), "sha256": hashes}, indent=2)
            )
            temporary.rename(destination)
        else:
            marker = json.loads((destination / "LOCAL_COPY.json").read_text())
            for relative, expected in marker["sha256"].items():
                if sha256(destination / relative) != expected:
                    raise RuntimeError(f"Local model copy changed: {destination / relative}")
    assembled = Path(config["model"]["assembled_model"])
    if not (assembled / "ASSEMBLY_COMPLETE").exists():
        command = [
            sys.executable,
            str(Path(__file__).with_name("assemble_qwen3_tts.py")),
            "--output",
            str(assembled),
            "--dtype",
            "float32",
        ]
        for key, name in SOURCES.items():
            command += ["--" + key.replace("_", "-"), str(target / name)]
        subprocess.run(command, check=True)
    print(json.dumps({"sources": str(target), "assembled_model": str(assembled)}, indent=2))


if __name__ == "__main__":
    main()
