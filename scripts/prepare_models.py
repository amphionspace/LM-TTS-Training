"""Copy local training sources and assemble them without downloading weights."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from qwen3_train.artifacts import file_hash
from qwen3_train.config import read_yaml

SOURCES = {
    "backbone": "Qwen3-0.6B-Base",
    "tts_template": "Qwen3-TTS-12Hz-0.6B-Base",
    "codec": "Qwen3-TTS-Tokenizer-12Hz",
}


def assembly_recipe(config):
    recipe = {
        "text_initialization": "qwen-tts",
        "text_projection_init": "pretrained",
        "train_text_frontend": False,
        "train_speaker_encoder": False,
        "input_protocol": "qwen3_non_streaming",
        "seed": 42,
        "dtype": "float32",
    }
    overrides = config.get("assembly", {})
    if not isinstance(overrides, dict) or set(overrides) - set(recipe):
        raise ValueError("Unknown assembly recipe fields")
    recipe.update(overrides)
    if recipe["text_initialization"] not in {"qwen-tts", "text-base"}:
        raise ValueError("Invalid text_initialization")
    if recipe["text_projection_init"] not in {"pretrained", "identity", "random", "near-identity"}:
        raise ValueError("Invalid text_projection_init")
    if (recipe["text_initialization"] == "qwen-tts") != (
        recipe["text_projection_init"] == "pretrained"
    ):
        raise ValueError("Pretrained text projection requires qwen-tts initialization")
    for key in ("train_text_frontend", "train_speaker_encoder"):
        if type(recipe[key]) is not bool:
            raise ValueError(f"assembly.{key} must be boolean")
    if type(recipe["seed"]) is not int or recipe["seed"] < 0:
        raise ValueError("assembly.seed must be a nonnegative integer")
    if recipe["dtype"] not in {"float32", "bfloat16"}:
        raise ValueError("Invalid assembly dtype")
    if recipe["input_protocol"] not in {"qwen3_non_streaming", "legacy_prefix"}:
        raise ValueError("Invalid assembly input_protocol")
    return recipe


def verify_assembly(directory, recipe, sources):
    """Reject a completed artifact from a different recipe instead of silently reusing it."""
    report = json.loads((directory / "assembly_report.json").read_text())
    expected = {
        **{
            key: recipe[key]
            for key in ("seed", "dtype", "text_initialization", "text_projection_init")
        },
        "freeze_text_frontend": not recipe["train_text_frontend"],
        "freeze_speaker_encoder": not recipe["train_speaker_encoder"],
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"Assembled recipe mismatch for {key}: {directory}")
    if report["adaptations"]["input_protocol"] != recipe["input_protocol"]:
        raise ValueError("Assembled input protocol differs")
    for key, source in sources.items():
        binding = report["sources"][key]
        if Path(binding["resolved_directory"]).resolve() != source.resolve() or binding[
            "config_sha256"
        ] != file_hash(source / "config.json"):
            raise ValueError(f"Assembled source differs: {key}")
    for name, expected_hash in report["artifact_sha256"].items():
        if file_hash(directory / name) != expected_hash:
            raise ValueError(f"Assembled artifact changed: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train-bf16.yaml")
    parser.add_argument("--source", type=Path, default=Path("/workspace/model"))
    args = parser.parse_args()
    config = read_yaml(args.config)
    recipe = assembly_recipe(config)
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
                    digest = file_hash(path)
                    if digest != file_hash(source / relative):
                        raise RuntimeError(f"Copy verification failed: {relative}")
                    hashes[str(relative)] = digest
            (temporary / "LOCAL_COPY.json").write_text(
                json.dumps({"source": str(source), "sha256": hashes}, indent=2)
            )
            temporary.rename(destination)
        else:
            marker = json.loads((destination / "LOCAL_COPY.json").read_text())
            for relative, expected in marker["sha256"].items():
                if file_hash(destination / relative) != expected:
                    raise RuntimeError(f"Local model copy changed: {destination / relative}")
    assembled = Path(config["model"]["assembled_model"])
    if not (assembled / "ASSEMBLY_COMPLETE").exists():
        command = [
            sys.executable,
            str(Path(__file__).with_name("assemble_qwen3_tts.py")),
            "--output",
            str(assembled),
        ]
        for key, value in recipe.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            else:
                command += [flag, str(value)]
        for key, name in SOURCES.items():
            command += ["--" + key.replace("_", "-"), str(target / name)]
        subprocess.run(command, check=True)
    verify_assembly(assembled, recipe, {key: target / name for key, name in SOURCES.items()})
    print(json.dumps({"sources": str(target), "assembled_model": str(assembled)}, indent=2))


if __name__ == "__main__":
    main()
