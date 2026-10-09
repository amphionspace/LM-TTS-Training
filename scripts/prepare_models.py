"""Prepare local source models with an explicit Qwen or LFM assembly recipe."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from lm_tts.artifacts import file_hash
from lm_tts.config import read_yaml
from lm_tts.models.assembly.preparation import (
    SOURCES,
    assembly_recipe,
    prepare_lfm,
    verify_assembly,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train-bf16.yaml")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/workspace/model"),
        help="Qwen source root to copy; LFM reads the project assets/base directory",
    )
    args = parser.parse_args()
    config = read_yaml(args.config)
    recipe = assembly_recipe(config)
    assembled = Path(config["model"]["assembled_model"])
    if (assembled / "config.json").exists():
        family = json.loads((assembled / "config.json").read_text()).get("model_type")
        expected = "lfm2_tts" if recipe.get("family") == "lfm2" else "qwen3_tts"
        if family != expected:
            raise ValueError(f"assembly.family disagrees with {assembled}: {family}")
    if recipe.get("family") == "lfm2":
        assembled = prepare_lfm(config, recipe)
        print(json.dumps({"assembled_model": str(assembled), "family": "lfm2"}))
        return
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
