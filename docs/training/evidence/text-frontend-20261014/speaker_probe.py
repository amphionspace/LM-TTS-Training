"""Diagnostic only: does the frozen-conditioning Talker read content out of the speaker embedding?

Teacher-forcing CE on the reference validation build under several speaker conditions.
Not part of the training repo. Run with the lm-tts environment from the repo root:

    CUDA_VISIBLE_DEVICES=1 python speaker_probe.py --model <export dir> --out <json>
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import lance
import numpy as np
import pyarrow.compute as pc
import torch
from torch.nn import functional as F

REPO = Path("/workspace/workspace/yanglin/LM-TTS-Training")
sys.path.insert(0, str(REPO))

from lm_tts.data.batch import collate  # noqa: E402
from lm_tts.data.reader import FeatureDataset  # noqa: E402
from lm_tts.models.loading import load_model  # noqa: E402

REF_BUILD = REPO / "data/builds/supervised-tts-20260929-reference-20261009-all16"
BASE_BUILD = REPO / "data/builds/supervised-tts-20260929-merged-20261001-all16"
SPECIALS = (151672, 151673)
CONDITIONS = ("full_self", "crop_self", "cross_same_speaker", "other_speaker")


def pick_rows(dataset, per_dataset, seed):
    rng = np.random.default_rng(seed)
    chosen = []
    for slot, binding in enumerate(dataset.bindings):
        lo, hi = dataset.offsets[slot], dataset.offsets[slot + 1]
        n = min(per_dataset, hi - lo)
        idx = np.sort(rng.choice(np.arange(lo, hi), n, replace=False))
        chosen.extend((binding["dataset_id"], int(i)) for i in idx)
    return chosen


def lookup_embeddings(dataset_id, target_ids, rng):
    """Full-utterance embeddings: self, same-speaker other utterance, other speaker."""
    table = lance.dataset(str(BASE_BUILD / dataset_id / "merged-000.lance"))
    ids = table.to_table(columns=["target_id", "speaker_id"])
    tid = np.asarray(ids["target_id"].to_numpy(zero_copy_only=False))
    spk = np.asarray(ids["speaker_id"].to_numpy(zero_copy_only=False))
    position = {t: i for i, t in enumerate(tid)}
    by_speaker = defaultdict(list)
    for i, s in enumerate(spk):
        by_speaker[s].append(i)
    self_rows, cross_rows, other_rows, speakers = [], [], [], []
    for t in target_ids:
        i = position[t]
        s = spk[i]
        speakers.append(s)
        self_rows.append(i)
        pool = [j for j in by_speaker[s] if j != i]
        cross_rows.append(int(rng.choice(pool)) if pool and s is not None else -1)
        # Unlabelled datasets (galgame: all null) get any other utterance; a single-speaker
        # dataset (ljspeech) has no other speaker at all.
        other = -1
        for _ in range(1000):
            j = int(rng.integers(len(tid)))
            if j != i and (s is None or spk[j] != s):
                other = j
                break
        other_rows.append(other)
    needed = sorted({*self_rows, *[r for r in cross_rows + other_rows if r >= 0]})
    vectors = table.take(needed, columns=["speaker_embedding"])["speaker_embedding"].to_pylist()
    vec = {r: torch.tensor(v, dtype=torch.float32) for r, v in zip(needed, vectors)}
    return (
        [vec[r] for r in self_rows],
        [vec[r] if r >= 0 else None for r in cross_rows],
        [vec[r] if r >= 0 else None for r in other_rows],
        speakers,
    )


@torch.no_grad()
def per_frame_ce(model, batch, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    out = model(batch)
    lengths, codes = batch["frame_lengths"], batch["codes"]
    last = (lengths + 1).cumsum(0) - 1
    labels = codes.new_full((len(codes) + len(lengths),), model.eos)
    valid = torch.ones(len(labels), dtype=torch.bool, device=device)
    valid[last] = False
    labels[valid] = codes[:, 0]
    first = F.cross_entropy(out["first_logits"].float(), labels, reduction="none")
    first_frames, eos = first[valid], first[~valid]
    first_hit = (out["first_logits"].argmax(-1) == labels)[valid]
    residual = torch.stack(
        [
            F.cross_entropy(logits.float(), codes[:, g + 1], reduction="none")
            for g, logits in enumerate(out["residual_logits"])
        ]
    ).mean(0)
    return first_frames.cpu(), residual.cpu(), eos.cpu(), first_hit.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-dataset", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--datasets", default="")
    args = parser.parse_args()
    device = torch.device("cuda")
    model = load_model(args.model, attn_implementation="sdpa").to(device).eval()
    dataset = FeatureDataset(
        REF_BUILD / "validation/manifest.json", SPECIALS, mask_reference=True, verify_integrity=False
    )
    rows = pick_rows(dataset, args.per_dataset, args.seed)
    if args.datasets:
        keep = set(args.datasets.split(","))
        rows = [r for r in rows if r[0] in keep]
    rng = np.random.default_rng(args.seed + 1)
    records = []
    start = time.time()
    for dataset_id in sorted({d for d, _ in rows}):
        indices = [i for d, i in rows if d == dataset_id]
        samples = dataset.__getitems__(indices)
        samples = [s for s in samples if s is not None]
        full, cross, other, speakers = lookup_embeddings(
            dataset_id, [s["id"] for s in samples], rng
        )
        for s, f, c, o, spk in zip(samples, full, cross, other, speakers):
            s["embeddings"] = {
                "full_self": f,
                "crop_self": s["speaker_embedding"],
                "cross_same_speaker": c,
                "other_speaker": o,
            }
            s["speaker_label"] = spk
        for b in range(0, len(samples), args.batch_size):
            chunk = samples[b : b + args.batch_size]
            results = {}
            for condition in CONDITIONS:
                rows_c = [
                    {**s, "speaker_embedding": s["embeddings"][condition]}
                    for s in chunk
                    if s["embeddings"][condition] is not None
                ]
                if not rows_c:
                    continue
                batch = collate(rows_c)
                first, residual, eos, hit = per_frame_ce(model, batch, device)
                offsets = np.cumsum([0, *[s["num_frames"] for s in rows_c]])
                for k, s in enumerate(rows_c):
                    sl = slice(offsets[k], offsets[k + 1])
                    results.setdefault(s["id"], {})[condition] = (
                        first[sl].numpy(),
                        residual[sl].numpy(),
                        float(eos[k]),
                        hit[sl].numpy(),
                    )
            for s in chunk:
                a, b_ = s["reference_interval"]
                T = s["num_frames"]
                inside = np.zeros(T, dtype=bool)
                inside[a:b_] = True
                rec = {
                    "dataset": dataset_id,
                    "id": s["id"],
                    "speaker": s["speaker_label"],
                    "frames": T,
                    "ref": [a, b_],
                    "norms": {
                        k: float(v.norm()) for k, v in s["embeddings"].items() if v is not None
                    },
                    "cos_to_full": {
                        k: float(F.cosine_similarity(v, s["embeddings"]["full_self"], dim=0))
                        for k, v in s["embeddings"].items()
                        if v is not None
                    },
                    "conditions": {},
                }
                for condition, (first, residual, eos, hit) in results[s["id"]].items():
                    pos = np.arange(T) / max(T - 1, 1)
                    rec["conditions"][condition] = {
                        "first_inside": float(first[inside].mean()),
                        "first_outside": float(first[~inside].mean()),
                        "first_before": float(first[:a].mean()) if a > 0 else None,
                        "first_after": float(first[b_:].mean()) if b_ < T else None,
                        "residual_inside": float(residual[inside].mean()),
                        "residual_outside": float(residual[~inside].mean()),
                        "first_all": float(first.mean()),
                        "residual_all": float(residual.mean()),
                        "acc_inside": float(hit[inside].mean()),
                        "acc_outside": float(hit[~inside].mean()),
                        "eos": eos,
                        "first_sum": float(first.sum()),
                        "residual_sum": float(residual.sum()),
                        "first_decile_sums": [
                            float(first[(pos >= d / 10) & (pos < (d + 1) / 10 + (d == 9))].sum())
                            for d in range(10)
                        ],
                        "decile_counts": [
                            int(((pos >= d / 10) & (pos < (d + 1) / 10 + (d == 9))).sum())
                            for d in range(10)
                        ],
                        "n_inside": int(inside.sum()),
                        "n_outside": int((~inside).sum()),
                    }
                records.append(rec)
        print(
            json.dumps({"dataset": dataset_id, "done": len(records), "elapsed": time.time() - start}),
            flush=True,
        )
    Path(args.out).write_text(
        json.dumps({"model": args.model, "seed": args.seed, "records": records})
    )


if __name__ == "__main__":
    main()
