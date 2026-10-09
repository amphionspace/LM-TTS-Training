"""Copy local training sources and assemble them without downloading weights."""

import json
from pathlib import Path

from ...artifacts import file_hash

SOURCES = {
    "backbone": "Qwen3-0.6B-Base",
    "tts_template": "Qwen3-TTS-12Hz-0.6B-Base",
    "codec": "Qwen3-TTS-Tokenizer-12Hz",
}


def assembly_recipe(config):
    overrides = config.get("assembly", {})
    if not isinstance(overrides, dict):
        raise ValueError("assembly must be a mapping")
    family = overrides.get("family", "qwen3")
    if family == "lfm2":
        recipe = {"family": "lfm2", "seed": 42, "dtype": "float32"}
        overrides = config["assembly"]
        if set(overrides) - set(recipe):
            raise ValueError("LFM assembly only accepts family, seed and dtype")
        recipe.update(overrides)
        if type(recipe["seed"]) is not int or recipe["seed"] < 0:
            raise ValueError("assembly.seed must be a nonnegative integer")
        if recipe["dtype"] not in {"float32", "bfloat16"}:
            raise ValueError("Invalid assembly dtype")
        return recipe
    if family != "qwen3":
        raise ValueError(f"Unsupported assembly family: {family}")
    recipe = {
        "text_initialization": "qwen-tts",
        "text_projection_init": "pretrained",
        "train_text_frontend": False,
        "train_speaker_encoder": False,
        "input_protocol": "qwen3_non_streaming",
        "seed": 42,
        "dtype": "float32",
    }
    overrides = {k: v for k, v in config.get("assembly", {}).items() if k != "family"}
    if set(overrides) - set(recipe):
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


def prepare_lfm(config, recipe):
    import torch

    from .lfm import assemble

    target = Path(config["paths"]["project"]) / "assets/base"
    sources = {
        "backbone": target / "LFM2.5-230M-Base",
        "tts_template": target / SOURCES["tts_template"],
        "codec": target / SOURCES["codec"],
    }
    for source in sources.values():
        if not (source / "config.json").is_file() or not list(source.glob("*.safetensors")):
            raise ValueError(f"Missing local model source: {source}")
    assembled = Path(config["model"]["assembled_model"])
    if not (assembled / "ASSEMBLY_COMPLETE").is_file():
        assemble(
            sources["backbone"],
            sources["tts_template"],
            sources["codec"],
            assembled,
            seed=recipe["seed"],
            dtype=getattr(torch, recipe["dtype"]),
        )
    report = json.loads((assembled / "assembly_report.json").read_text())
    for key, value in {
        "model_type": "lfm2_tts",
        "seed": recipe["seed"],
        "dtype": recipe["dtype"],
    }.items():
        if report.get(key) != value:
            raise ValueError(f"Assembled recipe mismatch for {key}: {assembled}")
    for key, source in sources.items():
        binding = report["sources"][key]
        if Path(binding["resolved_directory"]).resolve() != source.resolve():
            raise ValueError(f"Assembled source differs: {key}")
        for name, expected in binding["sha256"].items():
            if file_hash(source / name) != expected:
                raise ValueError(f"Assembled source changed: {source / name}")
    for name, expected in report["artifact_sha256"].items():
        if file_hash(assembled / name) != expected:
            raise ValueError(f"Assembled artifact changed: {name}")
    return assembled
