"""Diagnostic only: compare Qwen3-TTS text embedding (2048) with Qwen3-0.6B-Base embedding (1024)."""

import glob
import json
import sys

import torch
from safetensors import safe_open

REPO = "/workspace/workspace/yanglin/LM-TTS-Training"
TTS = f"{REPO}/assets/base/Qwen3-TTS-12Hz-0.6B-Base"
BASE = f"{REPO}/assets/base/Qwen3-0.6B-Base"
torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def load(directory, name):
    for f in glob.glob(f"{directory}/*.safetensors"):
        with safe_open(f, "pt") as h:
            if name in h.keys():
                return h.get_tensor(name).float()
    raise KeyError(name)


def find_name(directory, pattern):
    for f in glob.glob(f"{directory}/*.safetensors"):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                if pattern in k and k.endswith("weight") and "projection" not in k:
                    return k


tts_name = find_name(TTS, "text_embedding")
tts = load(TTS, tts_name)
base = load(BASE, "model.embed_tokens.weight")
print("tts", tts_name, tuple(tts.shape), "| base", tuple(base.shape))
V = min(len(tts), len(base))
tts, base = tts[:V].to(dev), base[:V].to(dev)

# Rows that are actually used by the tokenizer (non-zero in both tables).
used = (tts.norm(dim=1) > 0) & (base.norm(dim=1) > 0)
print(f"rows non-zero in both: {int(used.sum())}/{V}")
idx = used.nonzero().squeeze(1)
perm = idx[torch.randperm(len(idx), device=dev)]
train, test = perm[:120000], perm[120000:140000]


def stats(x, name):
    n = x.norm(dim=1)
    xc = x - x.mean(0)
    s = torch.linalg.svdvals(xc[torch.randperm(len(xc), device=dev)[:40000]])
    p = (s**2) / (s**2).sum()
    eff = float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))
    k90 = int((p.cumsum(0) < 0.90).sum()) + 1
    k99 = int((p.cumsum(0) < 0.99).sum()) + 1
    print(
        f"{name}: norm mean {n.mean():.3f} std {n.std():.3f} | effective rank {eff:.0f} | "
        f"dims for 90% var {k90} | 99% var {k99} | top singular share {p[0]:.3f}"
    )


stats(tts[idx], "TTS 2048")
stats(base[idx], "Base 1024")


def r2_fit(x, y, label):
    # Ridge least squares x -> y, fit on train rows, evaluate on held-out rows.
    xm, ym = x[train].mean(0), y[train].mean(0)
    xt, yt = x[train] - xm, y[train] - ym
    lam = 1e-3 * xt.pow(2).sum() / len(xt)
    w = torch.linalg.solve(xt.T @ xt + lam * torch.eye(x.shape[1], device=dev), xt.T @ yt)
    pred = (x[test] - xm) @ w
    resid = (y[test] - ym - pred).pow(2).sum()
    total = (y[test] - ym).pow(2).sum()
    r2 = 1 - resid / total
    cos = torch.nn.functional.cosine_similarity(pred + ym, y[test], dim=1)
    print(f"{label}: held-out R2 {r2:.4f} | cosine(pred, true) median {cos.median():.4f}")


r2_fit(base, tts, "linear Base1024 -> TTS2048")
r2_fit(tts, base, "linear TTS2048 -> Base1024")

# Does the TTS table contain the Base table (e.g. a sub-block or a copy)?
corr_blocks = []
for start in range(0, 2048, 1024):
    block = tts[:, start : start + 1024]
    cos = torch.nn.functional.cosine_similarity(block[idx], base[idx], dim=1)
    corr_blocks.append(float(cos.abs().median()))
print("median |cos| of Base rows vs TTS column blocks [0:1024],[1024:2048]:", corr_blocks)

# Neighborhood structure: do the two tables agree on which tokens are similar?
q = perm[:3000]
tn = torch.nn.functional.normalize(tts[idx], dim=1)
bn = torch.nn.functional.normalize(base[idx], dim=1)
tq = torch.nn.functional.normalize(tts[q], dim=1)
bq = torch.nn.functional.normalize(base[q], dim=1)
k = 10
nt = (tq @ tn.T).topk(k + 1, dim=1).indices[:, 1:]
nb = (bq @ bn.T).topk(k + 1, dim=1).indices[:, 1:]
overlap = torch.tensor([len(set(a.tolist()) & set(b.tolist())) for a, b in zip(nt, nb)]).float() / k
print(f"top-{k} neighbour overlap between tables: mean {overlap.mean():.3f} (random ~{k / len(idx):.5f})")

# Pairwise-similarity agreement on random token pairs (Spearman-ish via Pearson on cosines).
a, b = perm[:20000], perm[20000:40000]
ct = torch.nn.functional.cosine_similarity(tts[a], tts[b], dim=1)
cb = torch.nn.functional.cosine_similarity(base[a], base[b], dim=1)
print(f"correlation of pairwise cosines (20k random pairs): {torch.corrcoef(torch.stack([ct, cb]))[0, 1]:.4f}")

# TTS special-token rows that the TTS model relies on.
specials = json.load(open(f"{TTS}/config.json"))
for key in ("tts_bos_token_id", "tts_eos_token_id", "tts_pad_token_id"):
    if key in specials:
        t = specials[key]
        print(
            f"{key}={t}: TTS norm {tts[t].norm():.3f}, Base norm {base[t].norm() if t < len(base) else float('nan'):.3f}"
        )
