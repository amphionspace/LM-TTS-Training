"""(text tokens + 2) / codec frames per training target, by dataset and language (3M-row uniform sample)."""
import json, sys
import numpy as np

build = sys.argv[1]  # e.g. data/builds/supervised-tts-20261019-emilia-v4-emilia2-reference/train
manifest = json.load(open(f"{build}/manifest.json"))
out = {}
for binding in manifest["bindings"]:
    index = np.load(f"{build}/{binding['sampling_index']}", mmap_mode="r")
    rng = np.random.default_rng(0)
    rows = index[np.sort(rng.choice(len(index), size=min(len(index), 3_000_000), replace=False))]
    ratio = (rows["tokens"] + 2) / rows["frames"]  # text row includes tts_bos / tts_eos
    names = {v: k for k, v in binding["languages"].items()}
    stats = lambda r: {"n": int(len(r)), "p50": round(float(np.median(r)), 3), "p95": round(float(np.quantile(r, .95)), 3),
                       "p99": round(float(np.quantile(r, .99)), 3), "over_0.5_pct": round(100 * float(np.mean(r > .5)), 3),
                       "over_1_pct": round(100 * float(np.mean(r > 1)), 4)}
    out[binding["dataset_id"]] = {"all": stats(ratio), **{names[l]: stats(ratio[rows["language"] == l]) for l in sorted(set(rows["language"].tolist()))}}
print(json.dumps(out, indent=1))
