"""CPU-only weight health of DCP checkpoints against the assembled initialization (no repo code).

Usage: model_health.py ASSEMBLED CHECKPOINT [CHECKPOINT ...] > out.jsonl
Per checkpoint: relative drift from init and update since the previous checkpoint for each module
group, exact equality of frozen modules, non-finite counts, and codec/language/EOS row statistics.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open

assembled, checkpoints = Path(sys.argv[1]), list(map(Path, sys.argv[2:]))
config = json.loads((assembled / "config.json").read_text())["talker_config"]
languages = config["codec_language_id"]
specials = {name: config[name] for name in (
    "codec_pad_id", "codec_bos_id", "codec_eos_token_id", "codec_think_id",
    "codec_nothink_id", "codec_think_bos_id", "codec_think_eos_id")}
FROZEN = ("talker.model.text_embedding.", "talker.text_projection.", "speaker_encoder.")


def group(key):
    if key.startswith("speaker_encoder."):
        return "speaker_encoder"
    m = re.match(r"talker\.(model|code_predictor\.model)\.layers\.(\d+)\.(self_attn|mlp)", key)
    if m:
        return f"{'talker' if m[1] == 'model' else 'predictor'}.L{int(m[2]):02d}.{m[3]}"
    if ".layers." in key:  # layer norms
        return ("talker" if key.startswith("talker.model.") else "predictor") + ".layer_norms"
    return re.sub(r"\.(weight|bias)$", "", key).replace("code_predictor.model.", "predictor.")


def load_init():
    index = json.loads((assembled / "model.safetensors.index.json").read_text())["weight_map"]
    state = {}
    for name in sorted(set(index.values())):
        with safe_open(assembled / name, "pt") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key).float()
    return state


def load(checkpoint, keys):
    reader = dcp.FileSystemReader(checkpoint / "distributed")
    md = reader.read_metadata()
    state = {"model." + k: torch.empty(md.state_dict_metadata["model." + k].size) for k in keys}
    dcp.load(state, checkpoint_id=checkpoint / "distributed", no_dist=True)
    return {k[len("model."):]: v for k, v in state.items()}


def rows(table):
    norms = table.norm(dim=1)
    codes = norms[:2048]
    used = set(range(2048)) | set(specials.values()) | set(languages.values())
    unused = torch.tensor([i for i in range(len(table)) if i not in used])
    return {
        "code_norm_p5_p50_p95": [round(float(q), 4) for q in codes.quantile(torch.tensor([0.05, 0.5, 0.95]))],
        "special_norm": {k: round(float(norms[v]), 4) for k, v in specials.items()},
        "language_norm_mean": round(float(norms[list(languages.values())].mean()), 4),
        "unused_norm_mean": round(float(norms[unused].mean()), 4) if len(unused) else None,
    }


def language_geometry(table, init):
    ids = [languages[name] for name in sorted(languages)]
    vectors = torch.nn.functional.normalize(table[ids], dim=1)
    cosine = vectors @ vectors.T
    off = cosine[~torch.eye(len(ids), dtype=torch.bool)]
    drift = (table[ids] - init[ids]).norm(dim=1) / init[ids].norm(dim=1)
    nothink = torch.nn.functional.normalize(table[specials["codec_nothink_id"]], dim=0)
    return {
        "pairwise_cosine_mean_min_max": [round(float(off.mean()), 4), round(float(off.min()), 4), round(float(off.max()), 4)],
        "relative_drift_by_language": {name: round(float(d), 4) for name, d in zip(sorted(languages), drift)},
        "cosine_to_nothink": {name: round(float(v), 4) for name, v in zip(sorted(languages), vectors @ nothink)},
    }


def eos_head(head):
    norms = head.norm(dim=1)
    eos = head[specials["codec_eos_token_id"]]
    mean_code = head[:2048].mean(0)
    return {
        "eos_norm": round(float(norms[specials["codec_eos_token_id"]]), 4),
        "code_norm_p50": round(float(norms[:2048].median()), 4),
        "eos_cosine_to_mean_code": round(float(torch.nn.functional.cosine_similarity(eos, mean_code, dim=0)), 4),
    }


init = load_init()
keys = sorted(init)
previous = None
for checkpoint in checkpoints:
    state = load(checkpoint, keys)
    drift, update, base = defaultdict(float), defaultdict(float), defaultdict(float)
    nonfinite, frozen_changed = 0, []
    for key in keys:
        value, start = state[key], init[key]
        g = group(key)
        drift[g] += float((value - start).pow(2).sum())
        base[g] += float(start.pow(2).sum())
        if previous is not None:
            update[g] += float((value - previous[key]).pow(2).sum())
        nonfinite += int((~torch.isfinite(value)).sum())
        if key.startswith(FROZEN) and not torch.equal(value, start):
            frozen_changed.append(key)
    record = {
        "checkpoint": checkpoint.name,
        "nonfinite": nonfinite,
        "frozen_changed": frozen_changed,
        "relative_drift": {g: round((drift[g] / max(base[g], 1e-30)) ** 0.5, 5) for g in sorted(drift)},
        "relative_update": {g: round((update[g] / max(base[g], 1e-30)) ** 0.5, 6) for g in sorted(update)},
        "codec_embedding_rows": rows(state["talker.model.codec_embedding.weight"]),
        "codec_embedding_rows_init": rows(init["talker.model.codec_embedding.weight"]),
        "language_ids": language_geometry(state["talker.model.codec_embedding.weight"], init["talker.model.codec_embedding.weight"]),
        "eos_head": eos_head(state["talker.codec_head.weight"]),
        "eos_head_init": eos_head(init["talker.codec_head.weight"]),
        "predictor_embedding_norm_p50": [
            round(float(state[f"talker.code_predictor.model.codec_embedding.{g}.weight"].norm(dim=1).median()), 4)
            for g in range(15)
        ],
    }
    print(json.dumps(record), flush=True)
    previous = state
