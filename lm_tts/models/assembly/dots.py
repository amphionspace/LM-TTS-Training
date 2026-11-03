"""Assemble a released dots.tts checkpoint, with inline tags as new atomic tokens."""

import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from tokenizers import AddedToken
from transformers import AutoTokenizer

from ...artifacts import file_hash

RELEASE_FILES = (
    "config.json",
    "llm_config.json",
    "latent_stats.pt",
    "model.safetensors",
    "vocoder.safetensors",
    "speaker_encoder.safetensors",
)
EMBEDDING = "llm.model.embed_tokens.weight"


def assemble(source, output, tag_tokens, dtype=torch.float32):
    """Copy the release, append `tag_tokens` to its tokenizer and the tied embedding. A new row
    starts as the mean embedding of the subword tokens that spelled the tag before it was added,
    so tagged text keeps roughly its old meaning while the tag becomes one position."""
    from ..dots import DotsTTSModel

    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    missing = [name for name in RELEASE_FILES if not (source / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete dots.tts release {source}: {missing}")
    if len(set(tag_tokens)) != len(tag_tokens) or not all(tag_tokens):
        raise ValueError("Tag tokens must be distinct, non-empty strings")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    if any(tag in tokenizer.get_vocab() for tag in tag_tokens):
        raise ValueError("A tag token already exists in the release vocabulary")
    spellings = {tag: tokenizer(tag, add_special_tokens=False)["input_ids"] for tag in tag_tokens}
    state = load_file(source / "model.safetensors")
    embedding = state[EMBEDDING].float()
    if len(embedding) != len(tokenizer):
        raise ValueError("Release embedding rows disagree with its tokenizer")
    tokenizer.add_tokens([AddedToken(tag, normalized=False, special=True) for tag in tag_tokens])
    added = {tag: tokenizer.convert_tokens_to_ids(tag) for tag in tag_tokens}
    if sorted(added.values()) != list(range(len(embedding), len(tokenizer))):
        raise ValueError("Tag tokens did not receive consecutive new IDs")
    rows = torch.stack([embedding[spellings[tag]].mean(0) for tag in tag_tokens])
    state = {key: value.to(dtype) for key, value in state.items()}
    state[EMBEDDING] = torch.cat([embedding, rows]).to(dtype)
    llm_config = json.loads((source / "llm_config.json").read_text())
    llm_config["vocab_size"] = len(tokenizer)
    # The release pickles numpy arrays; JSON keeps the assembly readable without unpickling.
    stats = torch.load(source / "latent_stats.pt", weights_only=False)
    stats = {
        key: np.asarray(stats[key], dtype=np.float32).reshape(-1).tolist()
        for key in ("mean", "var")
    }
    report = {
        "format_version": 1,
        "model_type": "dots_tts",
        "dtype": str(dtype).removeprefix("torch."),
        "sources": {
            "release": {
                "resolved_directory": str(source),
                "sha256": {name: file_hash(source / name) for name in RELEASE_FILES},
            }
        },
        "text_vocabulary": {
            "source_token_count": len(embedding),
            "target_embedding_rows": len(tokenizer),
            "added_tokens": added,
            "row_initialization": {tag: spellings[tag] for tag in tag_tokens},
        },
        "frozen_offline": ["vocoder.safetensors (AudioVAE)", "speaker_encoder.safetensors (CAM++)"],
    }
    temporary = output.with_name(output.name + ".incomplete-" + uuid.uuid4().hex[:8])
    temporary.mkdir(parents=True)
    try:
        shutil.copyfile(source / "config.json", temporary / "config.json")
        (temporary / "llm_config.json").write_text(json.dumps(llm_config, indent=2) + "\n")
        (temporary / "latent_stats.json").write_text(json.dumps(stats) + "\n")
        tokenizer.save_pretrained(temporary)
        save_file(state, str(temporary / "model.safetensors"), metadata={"format": "pt"})
        for name in ("vocoder.safetensors", "speaker_encoder.safetensors"):
            shutil.copyfile(source / name, temporary / name)
        report["artifact_sha256"] = {
            p.name: file_hash(p) for p in sorted(temporary.iterdir()) if p.is_file()
        }
        (temporary / "assembly_report.json").write_text(json.dumps(report, indent=2) + "\n")
        (temporary / "ASSEMBLY_COMPLETE").write_text("dots_tts\n")
        model = DotsTTSModel.from_assembled(temporary, attn_implementation="sdpa")
        for name, value in model.state_dict().items():
            if not torch.equal(value.to(state[name].dtype), state[name]):
                raise ValueError(f"Reload changed tensor: {name}")
        reloaded = AutoTokenizer.from_pretrained(temporary, local_files_only=True)
        for tag, token in added.items():
            if reloaded(f"前{tag}后", add_special_tokens=False)["input_ids"].count(token) != 1:
                raise ValueError(f"Reloaded tokenizer does not keep {tag} atomic")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return report
