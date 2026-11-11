"""Deterministic holdout: whole speakers for cross-reference bindings, rows otherwise."""

import hashlib
import json
from pathlib import Path

import numpy as np

from ..artifacts import digest, file_hash
from .build import INDEX_DTYPE, read_complete


def split_build(build_path, fraction=0.001, seed=42):
    if not 0 < fraction < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Holdout requires a fraction in (0, 1) and a nonnegative integer seed")
    build_path = Path(build_path).resolve()
    source = read_complete(build_path)
    identity = {
        "source_build_sha256": file_hash(build_path),
        "fraction": fraction,
        "seed": seed,
        "method": (
            "cross_reference_speakers_otherwise_uniform_rows_v2"
            if any("reference_index" in b for b in source["bindings"])
            else "fixed_snapshot_uniform_rows_per_dataset_v1"
        ),
    }
    manifests = {}
    for role, policy in (("train", "train_isolated"), ("validation", "heldout")):
        destination = build_path.parent / role
        destination.mkdir(exist_ok=False)
        manifests[role] = {
            **source,
            "evaluation": policy,
            "bindings": [],
            "split": identity,
            "split_sha256": digest(identity),
        }
    for binding in source["bindings"]:
        original = np.load(build_path.parent / binding["sampling_index"], mmap_mode="r")[
            : binding["ready_rows"]
        ]
        count = len(original)
        heldout = max(1, round(count * fraction))
        if heldout >= count:
            raise ValueError(f"Dataset {binding['dataset_id']} is too small to split")
        dataset_seed = int.from_bytes(
            hashlib.sha256(binding["dataset_id"].encode()).digest()[:8], "little"
        )
        rng = np.random.default_rng(np.random.SeedSequence([seed, dataset_seed]))
        references = None
        if "reference_index" in binding:
            with np.load(build_path.parent / binding["reference_index"]) as arrays:
                references = {k: arrays[k] for k in arrays.files}
            positions = np.searchsorted(references["rows"], original["row"])
            groups = references["group"][positions]
            speakers = np.unique(groups)
            n = max(1, round(len(speakers) * fraction))
            if n >= len(speakers):
                raise ValueError(f"Dataset {binding['dataset_id']} needs at least two speakers")
            heldout_speakers = rng.choice(speakers, size=n, replace=False)
            chosen = np.flatnonzero(np.isin(groups, heldout_speakers))
            heldout = len(chosen)
        else:
            chosen = np.sort(rng.choice(count, size=heldout, replace=False))
        outputs, cursors = {}, {"train": 0, "validation": 0}
        name = f"sampling-{binding['binding_slot']:03d}.npy"
        for role, length in (("train", count - heldout), ("validation", heldout)):
            outputs[role] = np.lib.format.open_memmap(
                build_path.parent / role / name, mode="w+", dtype=INDEX_DTYPE, shape=(length,)
            )
        for start in range(0, count, 1_000_000):
            part = original[start : start + 1_000_000]
            selected = (
                chosen[np.searchsorted(chosen, start) : np.searchsorted(chosen, start + len(part))]
                - start
            )
            mask = np.zeros(len(part), dtype=bool)
            mask[selected] = True
            for role, values in (("train", part[~mask]), ("validation", part[mask])):
                outputs[role][cursors[role] : cursors[role] + len(values)] = values
                cursors[role] += len(values)
        for role in outputs:
            outputs[role].flush()
            extra = {}
            if references is not None:
                selected = np.isin(references["group"], heldout_speakers)
                if role == "train":
                    selected = ~selected
                ref = {k: references[k][selected] for k in ("rows", "text", "long", "style")}
                _, group = np.unique(references["group"][selected], return_inverse=True)
                order = np.argsort(group, kind="stable")
                starts = np.searchsorted(group[order], np.arange(group.max() + 2))
                path = build_path.parent / role / binding["reference_index"]
                np.savez(path, **ref, group=group.astype(np.int32), order=order, starts=starts)
                extra = {"reference_sha256": file_hash(path), "speakers": len(starts) - 1}
            manifests[role]["bindings"].append(
                {
                    **binding,
                    **extra,
                    "sampling_index": name,
                    "sampling_sha256": file_hash(build_path.parent / role / name),
                    "ready_rows": cursors[role],
                    "coverage": {
                        "ready": cursors[role],
                        "excluded_for_split": count - cursors[role],
                    },
                }
            )
        print(json.dumps({"split_dataset": binding["dataset_id"], **cursors}), flush=True)
    for role, manifest in manifests.items():
        ready = sum(b["ready_rows"] for b in manifest["bindings"])
        manifest["coverage"] = {
            "ready": ready,
            "excluded_for_split": source["coverage"]["ready"] - ready,
            "dataset_unbound": source["coverage"].get("dataset_unbound", 0),
        }
        (build_path.parent / role / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2)
        )
    return manifests["train"]
