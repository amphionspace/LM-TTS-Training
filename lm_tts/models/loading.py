"""Load a completed assembly or a verified flat checkpoint export."""

import json
from pathlib import Path

from ..artifacts import file_hash
from .assembly.common import load_prefix


def verify_export(directory):
    directory = Path(directory)
    report = json.loads((directory / "export.json").read_text())
    hashes = report.get("artifact_sha256", {})
    hashes = {**hashes, "model.safetensors": report["weights_sha256"]}
    for name, expected in hashes.items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"Export artifact changed: {name}")
    return report


def load_model(
    directory,
    *,
    load_weights=True,
    attn_implementation="flash_attention_2",
    use_speaker_embedding=None,
):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    family = config.get("model_type")
    if family == "lfm2_tts":
        from .lfm import LfmTTSModel

        implementation = LfmTTSModel
    elif family == "qwen3_tts":
        from .qwen import TTSModel

        implementation = TTSModel
    else:
        raise ValueError(f"Unsupported model_type: {family}")
    if (directory / "export.json").is_file():
        report = verify_export(directory)
        speaker = report.get("use_speaker_embedding", True)
        if use_speaker_embedding is not None and use_speaker_embedding != speaker:
            raise ValueError("Cannot change conditioning of an exported checkpoint")
        model = implementation.from_config(config, attn_implementation, speaker)
        if load_weights:
            model.load_state_dict(load_prefix(directory, ""), strict=True)
        return model
    return implementation.from_assembled(
        directory,
        load_weights=load_weights,
        attn_implementation=attn_implementation,
        use_speaker_embedding=use_speaker_embedding,
    )
