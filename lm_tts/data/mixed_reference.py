"""Mix two reference placements of the same targets into one build, sample by sample."""

import json
from pathlib import Path

import numpy as np

from ..artifacts import file_hash
from .build import open_snapshot, read_complete
from .reference import is_reference

ROLES = {"all": ".", "train": "train", "validation": "validation"}
SHARED = ("dataset_id", "release_id", "ready_rows", "sampling_sha256", "selection_sha256")
IDENTITY = ("manifest_sha256", "profile_id", "lance_version")


def compose_mixed_build(primary, alternate, output, fraction, seed):
    """Route a seeded `fraction` of targets to `alternate`'s reference features.

    Both builds must index the same rows with the same tokens and holdout and differ only in the
    speaker reference. Alternate rows keep their metadata and are offset by the merged row count,
    so every batch mixes both placements while the sampler and lengths stay unchanged.
    """
    primary, alternate, output = (Path(p).resolve() for p in (primary, alternate, output))
    if not 0 < fraction < 1 or type(seed) is not int:
        raise ValueError("Mixing needs a fraction in (0, 1) and an integer seed")
    sources = {}
    for role, folder in ROLES.items():
        a = read_complete(primary / folder / "manifest.json")
        b = read_complete(alternate / folder / "manifest.json")
        if len(a["bindings"]) != 1 or len(b["bindings"]) != 1:
            raise ValueError("Mixing supports single-binding builds")
        x, y = a["bindings"][0], b["bindings"][0]
        # Equal sampling hashes mean identical rows, lengths, tokens counts and holdout.
        if (
            {a["representation"], b["representation"]} != {"merged_indexed_references"}
            or any(a.get(k) != b.get(k) for k in ("tokenizer", "selection_sha256", "evaluation"))
            or any(x.get(k) != y.get(k) for k in SHARED)
            or any(x[kind][k] != y[kind][k] for kind in ("codec", "text") for k in IDENTITY)
            or a.get("languages") != b.get("languages")
            or not (is_reference(x) and is_reference(y))
            or x["merged_profile"]["speaker_mode"] == y["merged_profile"]["speaker_mode"]
        ):
            raise ValueError(f"Builds differ beyond the reference placement: {role}")
        sources[role] = (a, x, y, folder)
    total = sources["all"][1]["ready_rows"]
    for binding in sources["all"][1:3]:
        if open_snapshot(binding["merged"]).count_rows() != total:
            raise ValueError("Merged token cache does not cover the full build")
    # One assignment over all merged rows, so train and validation follow the same rule.
    chosen = np.random.default_rng(np.random.SeedSequence([seed, 7])).random(total) < fraction
    mix = {"fraction": fraction, "seed": seed, "method": "seeded_uniform_per_merged_row_v1"}
    output.mkdir(parents=True, exist_ok=False)
    manifests = {}
    for role, (manifest, x, y, folder) in sources.items():
        index = np.load(primary / folder / x["sampling_index"])
        selected = chosen[index["row"]]
        index["row"][selected] += total
        destination = output / folder
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / "sampling-000.npy", index)
        binding = {
            **x,
            "sampling_index": "sampling-000.npy",
            "sampling_sha256": file_hash(destination / "sampling-000.npy"),
            "alternate": {
                "offset": total,
                **{k: y[k] for k in ("merged", "merged_profile", "speaker", "speaker_profile")},
            },
            "reference_mix": {**mix, "rows": len(index), "alternate_rows": int(selected.sum())},
        }
        manifests[role] = {
            **manifest,
            "build_id": output.name,
            "reference_scope": "+".join(b["merged_profile"]["speaker_mode"] for b in (x, y)),
            "bindings": [binding],
            "reference_mix": {
                **mix,
                "primary_build_sha256": file_hash(primary / folder / "manifest.json"),
                "alternate_build_sha256": file_hash(alternate / folder / "manifest.json"),
            },
        }
        (destination / "manifest.json").write_text(json.dumps(manifests[role], indent=2))
    return manifests["train"]
