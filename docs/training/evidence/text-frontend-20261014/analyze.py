"""Aggregate speaker_probe.py outputs into tables (frame-weighted means)."""

import json
import sys
from collections import defaultdict

import numpy as np

CONDITIONS = ("full_self", "crop_self", "cross_same_speaker", "other_speaker")
LABEL = {
    "full_self": "自身完整",
    "crop_self": "自身裁剪",
    "cross_same_speaker": "同说话人另一条",
    "other_speaker": "其他说话人",
}
UNRELIABLE = {"emilia", "emilia_yodas", "wenetspeech4tts", "mls_sidon", "libriheavy"}


def weighted(records, condition, key_sum, key_n):
    s = n = 0.0
    for r in records:
        c = r["conditions"].get(condition)
        if c is None:
            continue
        s += c[key_sum]
        n += c[key_n]
    return s / n if n else float("nan")


def region(records, condition, which):
    s = n = 0.0
    for r in records:
        c = r["conditions"].get(condition)
        if c is None:
            continue
        s += c[f"first_{which}"] * c[f"n_{which}"]
        n += c[f"n_{which}"]
    return s / n if n else float("nan")


def region_res(records, condition, which):
    s = n = 0.0
    for r in records:
        c = r["conditions"].get(condition)
        if c is None:
            continue
        s += c[f"residual_{which}"] * c[f"n_{which}"]
        n += c[f"n_{which}"]
    return s / n if n else float("nan")


def acc(records, condition, which):
    s = n = 0.0
    for r in records:
        c = r["conditions"].get(condition)
        if c is None:
            continue
        s += c[f"acc_{which}"] * c[f"n_{which}"]
        n += c[f"n_{which}"]
    return s / n if n else float("nan")


def before_after(records, condition):
    out = {}
    for which in ("before", "after"):
        vals, ws = [], []
        for r in records:
            c = r["conditions"].get(condition)
            if c is None or c[f"first_{which}"] is None:
                continue
            a, b = r["ref"]
            w = a if which == "before" else r["frames"] - b
            vals.append(c[f"first_{which}"] * w)
            ws.append(w)
        out[which] = sum(vals) / sum(ws) if ws else float("nan")
    return out


def deciles(records, condition):
    s = np.zeros(10)
    n = np.zeros(10)
    for r in records:
        c = r["conditions"].get(condition)
        if c is None:
            continue
        s += c["first_decile_sums"]
        n += c["decile_counts"]
    return s / n


def main(path):
    data = json.load(open(path))
    records = data["records"]
    print(f"# {data['model']}")
    print(f"records={len(records)} frames={sum(r['frames'] for r in records)}")
    n_cross = sum("cross_same_speaker" in r["conditions"] for r in records)
    print(f"with cross-utterance embedding: {n_cross}")
    print()
    print("## overall (frame-weighted)")
    print("| condition | first CE | residual CE | first acc | EOS CE |")
    for c in CONDITIONS:
        eos = np.mean([r["conditions"][c]["eos"] for r in records if c in r["conditions"]])
        print(
            f"| {LABEL[c]} | {weighted(records, c, 'first_sum', 'frames') if False else weighted_frames(records, c):.4f} "
            f"| {weighted_res(records, c):.4f} | {acc_all(records, c):.4f} | {eos:.4f} |"
        )
    print()
    print("## inside vs outside the reference interval (first codebook CE)")
    print("| condition | inside | outside | inside-outside | before | after | acc in | acc out |")
    for c in CONDITIONS:
        i, o = region(records, c, "inside"), region(records, c, "outside")
        ba = before_after(records, c)
        print(
            f"| {LABEL[c]} | {i:.4f} | {o:.4f} | {i - o:+.4f} | {ba['before']:.4f} | {ba['after']:.4f} "
            f"| {acc(records, c, 'inside'):.4f} | {acc(records, c, 'outside'):.4f} |"
        )
    print()
    print("## residual CE inside vs outside")
    print("| condition | inside | outside |")
    for c in CONDITIONS:
        print(f"| {LABEL[c]} | {region_res(records, c, 'inside'):.4f} | {region_res(records, c, 'outside'):.4f} |")
    print()
    print("## first CE by relative position decile")
    print("| condition | " + " | ".join(f"{d * 10}-{d * 10 + 10}%" for d in range(10)) + " |")
    for c in CONDITIONS:
        print(f"| {LABEL[c]} | " + " | ".join(f"{v:.3f}" for v in deciles(records, c)) + " |")
    print()
    print("## per dataset: first CE (all frames) and crop inside/outside")
    print("| dataset | n | full_self | cross | other | crop in | crop out | reliable spk |")
    by = defaultdict(list)
    for r in records:
        by[r["dataset"]].append(r)
    for d in sorted(by):
        rs = by[d]
        print(
            f"| {d} | {len(rs)} | {weighted_frames(rs, 'full_self'):.3f} | {weighted_frames(rs, 'cross_same_speaker'):.3f} "
            f"| {weighted_frames(rs, 'other_speaker'):.3f} | {region(rs, 'crop_self', 'inside'):.3f} "
            f"| {region(rs, 'crop_self', 'outside'):.3f} | {'no' if d in UNRELIABLE else 'yes'} |"
        )
    print()
    print("## embedding geometry")
    for c in CONDITIONS:
        norms = [r["norms"][c] for r in records if c in r["norms"]]
        cos = [r["cos_to_full"][c] for r in records if c in r["cos_to_full"]]
        print(f"- {LABEL[c]}: norm median {np.median(norms):.2f}, cos(to full self) median {np.median(cos):.4f}")
    print()
    print("## per-utterance: fraction where condition beats cross_same_speaker (first CE)")
    for c in ("full_self", "crop_self", "other_speaker"):
        wins = [
            r["conditions"][c]["first_all"] < r["conditions"]["cross_same_speaker"]["first_all"]
            for r in records
            if c in r["conditions"] and "cross_same_speaker" in r["conditions"]
        ]
        print(f"- {LABEL[c]} < 同说话人另一条: {np.mean(wins):.3f} (n={len(wins)})")


def weighted_frames(records, c):
    s = n = 0.0
    for r in records:
        x = r["conditions"].get(c)
        if x is None:
            continue
        s += x["first_sum"]
        n += r["frames"]
    return s / n if n else float("nan")


def weighted_res(records, c):
    s = n = 0.0
    for r in records:
        x = r["conditions"].get(c)
        if x is None:
            continue
        s += x["residual_sum"]
        n += r["frames"]
    return s / n if n else float("nan")


def acc_all(records, c):
    s = n = 0.0
    for r in records:
        x = r["conditions"].get(c)
        if x is None:
            continue
        s += x["acc_inside"] * x["n_inside"] + x["acc_outside"] * x["n_outside"]
        n += r["frames"]
    return s / n if n else float("nan")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
        print("\n---\n")
