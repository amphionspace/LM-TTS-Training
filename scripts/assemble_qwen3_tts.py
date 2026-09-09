#!/usr/bin/env python3
"""Assemble Qwen3-TTS from a text Base checkpoint, official speaker encoder and codec."""
import argparse
import gc
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from huggingface_hub import snapshot_download
from qwen3_train.assembly import audit_sources, copy_codec, initialize_model, save_model, sha256, validate_saved

DEFAULTS = {
    "backbone": ("Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"),
    "tts_template": ("Qwen/Qwen3-TTS-12Hz-0.6B-Base", "5d83992436eae1d760afd27aff78a71d676296fc"),
    "codec": ("Qwen/Qwen3-TTS-Tokenizer-12Hz", "7dd38ad4e9bad454aae9cd937d0cd577604fe229"),
}


def resolve(source, revision, weights):
    if Path(source).is_dir():
        return Path(source).resolve()
    patterns = ["config.json", "tokenizer*.json", "vocab.json", "merges.txt", "preprocessor_config.json", "generation_config.json"]
    if weights:
        patterns += ["model*.safetensors", "model.safetensors.index.json"]
    return Path(snapshot_download(source, revision=revision, allow_patterns=patterns,
                                  ignore_patterns=["speech_tokenizer/*"], max_workers=4))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name, (repo, revision) in DEFAULTS.items():
        p.add_argument("--" + name.replace("_", "-"), default=repo)
        p.add_argument("--" + name.replace("_", "-") + "-revision", default=revision)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--text-projection-init", choices=["pretrained", "identity", "random", "near-identity"], default=None)
    p.add_argument("--text-initialization", choices=["qwen-tts", "text-base"], default="qwen-tts")
    p.add_argument("--train-speaker-encoder", action="store_true", help="Opt in to joint speaker training; default freezes public ECAPA")
    p.add_argument("--train-text-frontend", action="store_true", help="Unfreeze text embedding and projection; default freezes both")
    p.add_argument("--input-protocol", choices=["qwen3_non_streaming", "legacy_prefix"], default="qwen3_non_streaming")
    p.add_argument("--audit-only", action="store_true", help="Write a JSON report; do not download weights or build a model")
    args = p.parse_args()
    if args.text_projection_init is None:
        args.text_projection_init = "pretrained" if args.text_initialization == "qwen-tts" else "identity"
    if (args.text_initialization == "qwen-tts") != (args.text_projection_init == "pretrained"):
        p.error("qwen-tts initialization requires pretrained projection; other modes require --text-initialization text-base")
    torch.set_num_threads(4)
    destination = Path(args.output).resolve()
    if destination.exists():
        p.error("Output already exists; choose a new output path")
    paths = {}
    # Audit dimensions and tokenizer identities before fetching large weights.
    for name in ["backbone", "tts_template"]:
        paths[name] = resolve(getattr(args, name), getattr(args, name + "_revision"), weights=False)
    config, tokenizer, report = audit_sources(paths["backbone"], paths["tts_template"], args.text_projection_init, args.input_protocol,
                                               args.text_initialization, not args.train_text_frontend, not args.train_speaker_encoder)
    report.update(format_version=2, seed=args.seed, dtype=args.dtype, text_projection_init=args.text_projection_init)
    report["sources"] = {name: {"requested": getattr(args, name), "revision": getattr(args, name + "_revision")}
                         for name in DEFAULTS}
    if args.audit_only:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    for name in DEFAULTS:
        paths[name] = resolve(getattr(args, name), getattr(args, name + "_revision"), weights=True)
        report["sources"][name]["resolved_directory"] = str(paths[name])
        marker = paths[name] / "REVISION"
        if marker.exists():
            report["sources"][name]["local_revision_marker"] = marker.read_text().strip()
        report["sources"][name]["config_sha256"] = sha256(paths[name] / "config.json")
    model = initialize_model(config, paths["backbone"], paths["tts_template"],
                             report["text_vocabulary"]["added_tokens"], getattr(torch, args.dtype), args.seed, args.text_projection_init, args.text_initialization)
    report["parameters"] = {"total": sum(p.numel() for p in model.parameters()),
                            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                            "talker": sum(p.numel() for p in model.talker.parameters()),
                            "speaker_encoder": sum(p.numel() for p in model.speaker_encoder.parameters())}
    temporary = destination.with_name(destination.name + ".incomplete-" + uuid.uuid4().hex[:8])
    temporary.mkdir(parents=True)
    save_model(model, temporary)
    tokenizer.save_pretrained(temporary)
    for name in ["preprocessor_config.json", "generation_config.json"]:
        shutil.copy2(paths["tts_template"] / name, temporary / name)
    copy_codec(paths["codec"], temporary / "speech_tokenizer")
    del model
    gc.collect()
    report["validation"] = validate_saved(temporary, getattr(torch, args.dtype))
    report["artifact_sha256"] = {str(path.relative_to(temporary)): sha256(path)
                                 for path in sorted(temporary.rglob("*")) if path.is_file()}
    (temporary / "assembly_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (temporary / "ASSEMBLY_COMPLETE").write_text("ok\n")
    temporary.rename(destination)
    print(json.dumps({"output": str(destination), "parameters": report["parameters"],
                      "validation": report["validation"], "text_vocabulary": report["text_vocabulary"]}, indent=2))


if __name__ == "__main__":
    main()
