"""Reuse pinned tokenization and holdout membership for reference-feature subsets."""

import json
import multiprocessing
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ..artifacts import digest, file_hash
from .build import INDEX_DTYPE, feature_reference, open_snapshot, read_complete
from .merged import METADATA_COLUMNS, validate_row
from .reference import REFERENCE_COLUMNS


def aligned_cache(table, old_table, binding):
    """Match an ordered subset by ID; fail rather than silently reuse another sample."""

    def previous_rows():
        for batch in old_table.scanner(
            columns=["target_id", "audio_sha256", "text", "text_ids", "num_text_tokens"],
            batch_size=8192,
            scan_in_order=True,
        ).to_batches():
            tokens = batch.column("text_ids")
            lengths = batch.column("num_text_tokens")
            if (
                tokens.null_count
                or lengths.null_count
                or not pc.all(pc.greater(lengths, 0)).as_py()
                or not pc.all(pc.equal(pc.list_value_length(tokens), lengths)).as_py()
            ):
                raise ValueError("Invalid cached text tokens")
            metadata = batch.select(["target_id", "audio_sha256", "text"]).to_pylist()
            for index, row in enumerate(metadata):
                yield row, batch, index

    old = iter(enumerate(previous_rows()))
    for batch in table.scanner(
        columns=METADATA_COLUMNS + REFERENCE_COLUMNS,
        batch_size=8192,
        scan_in_order=True,
    ).to_batches():
        rows, locators, pieces = [], [], []
        for row in batch.to_pylist():
            validate_row(row, binding)
            for locator, (previous, cached, index) in old:
                if previous["target_id"] == row["target_id"]:
                    break
            else:
                raise ValueError("Reference rows must be an ordered subset of the token cache")
            if previous["text"] != row["text"] or previous["audio_sha256"] != row["audio_sha256"]:
                raise ValueError("Reference text/audio differs from the pinned token cache")
            rows.append(row)
            locators.append(locator)
            if not pieces or pieces[-1][0] is not cached:
                pieces.append((cached, []))
            pieces[-1][1].append(index)
        # Keep token arrays in Arrow; expanding billions of IDs into Python dominates build time.
        token_cache = pa.Table.from_batches(
            [
                cached.select(["text_ids", "num_text_tokens"]).take(pa.array(indices))
                for cached, indices in pieces
            ]
        )
        yield rows, np.asarray(locators), token_cache


def build_dataset(arguments):
    root, configured, old_binding, old_root, val_binding, val_root, selection_sha, output = (
        arguments
    )
    root, old_root, val_root, output = map(Path, (root, old_root, val_root, output))
    output.mkdir(parents=True, exist_ok=False)
    manifest, reference, table = feature_reference(root, configured["merged_manifest"], "merged")
    if (manifest["dataset_id"], manifest["release_id"]) != (
        old_binding["dataset_id"],
        old_binding["release_id"],
    ):
        raise ValueError("Reference dataset identity differs from token cache")
    filtering = manifest.get("reference_filtering", {})
    if (
        manifest["profile"].get("speaker_mode") != "codec_grid_reference"
        or digest(manifest["profile"]) != manifest["profile_id"]
        or manifest["selection"]["manifest_sha256"] != selection_sha
        or filtering.get("input_rows") != old_binding["ready_rows"]
        or filtering.get("retained") != manifest["rows"]
        or filtering["input_rows"] - sum(filtering["excluded_reasons"].values()) != manifest["rows"]
        or manifest["coverage"]["ok"] != manifest["rows"]
    ):
        raise ValueError("Invalid published reference subset")
    sources, profiles = {}, {}
    for kind in ("codec", "speaker", "text"):
        refs = [s for s in manifest["inputs"] if s["alias"] == kind]
        if len(refs) != 1:
            raise ValueError("Reference merge requires one source of each feature kind")
        source = refs[0]
        path = root / source["manifest_path"]
        m = read_complete(path, "speaker_embedding" if kind == "speaker" else kind)
        if (
            file_hash(path) != source["manifest_sha256"]
            or m["profile_id"] != source["profile_id"]
            or digest(m["profile"]) != source["profile_id"]
            or m["profile_id"] != manifest["profile"]["sources"][kind]
            or m["lance_version"] != source["lance_version"]
            or m["rows"] != filtering["input_rows"]
            or m["dataset_id"] != manifest["dataset_id"]
            or m["release_id"] != manifest["release_id"]
            or m["selection"]["manifest_sha256"] != selection_sha
        ):
            raise ValueError("Reference feature provenance changed")
        if kind in ("codec", "text") and any(
            source[k] != old_binding[kind][k]
            for k in ("manifest_sha256", "profile_id", "lance_version")
        ):
            raise ValueError("Text/codec sources differ; cannot reuse tokenization")
        sources[kind], profiles[kind] = source, m["profile"]
    policy = profiles["speaker"].get("reference", {})
    if (
        policy.get("version"),
        policy.get("codec_sample_rate"),
        policy.get("codec_hop_samples"),
    ) != ("codec-grid-reference-v1", 24000, 1920):
        raise ValueError("Unsupported reference timing profile")
    for b, parent in ((old_binding, old_root), (val_binding, val_root)):
        if file_hash(parent / b["sampling_index"]) != b["sampling_sha256"]:
            raise ValueError("Baseline sampling index changed")
        if file_hash(b["merged"]["manifest_path"]) != b["merged"]["manifest_sha256"]:
            raise ValueError("Baseline feature manifest changed")
    if old_binding["merged"] != val_binding["merged"]:
        raise ValueError("Baseline validation refers to a different token cache")
    old_table = open_snapshot(old_binding["merged"])
    if old_table.count_rows() != old_binding["ready_rows"]:
        raise ValueError("Baseline token cache row count changed")
    old_index = np.load(old_root / old_binding["sampling_index"], mmap_mode="r")
    old_val = np.load(val_root / val_binding["sampling_index"], mmap_mode="r")
    validation_rows = np.zeros(len(old_index), dtype=np.bool_)
    validation_rows[old_val["row"]] = True
    binding = {
        **old_binding,
        **sources,
        "source_merged": reference,
        "merged_profile": manifest["profile"],
        "speaker_profile": profiles["speaker"],
        "ready_rows": manifest["rows"],
        "selected_rows": manifest["rows"],
        "coverage": {"ready": manifest["rows"]},
        "reference_filtering": filtering,
    }
    slot = configured.get("binding_slot", 0)
    index_paths = {
        "all": output / "sampling-000.npy",
        **{
            role: output.parent / role / f"sampling-{slot:03d}.npy"
            for role in ("train", "validation")
        },
    }
    for path in index_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    # AFS mmap page faults make small writes prohibitively slow. Publish contiguous arrays once.
    with tempfile.TemporaryDirectory(prefix="lm-tts-reference-index-") as scratch:
        indices = {
            role: np.lib.format.open_memmap(
                Path(scratch) / f"{role}.npy",
                mode="w+",
                dtype=INDEX_DTYPE,
                shape=(len(old_val) if role == "validation" else manifest["rows"],),
            )
            for role in ("all", "train", "validation")
        }
        cursor = dict.fromkeys(indices, 0)
        schema = pa.schema([("text_ids", pa.list_(pa.int32())), ("num_text_tokens", pa.int32())])

        def columns():
            for rows, locators, cached in aligned_cache(table, old_table, binding):
                records = old_index[locators].copy()
                if (
                    not np.array_equal(records["row"], locators)
                    or not np.array_equal(
                        records["frames"], [r["codec_num_codec_frames"] for r in rows]
                    )
                    or not np.array_equal(records["tokens"], cached["num_text_tokens"].to_numpy())
                ):
                    raise ValueError("Baseline sampling metadata disagrees with matched row")
                records["row"] = np.arange(cursor["all"], cursor["all"] + len(rows))
                heldout = validation_rows[locators]
                for role, selected in (
                    ("all", records),
                    ("train", records[~heldout]),
                    ("validation", records[heldout]),
                ):
                    indices[role][cursor[role] : cursor[role] + len(selected)] = selected
                    cursor[role] += len(selected)
                if cursor["all"] % (8192 * 100) == 0 or cursor["all"] == manifest["rows"]:
                    print(
                        json.dumps(
                            {
                                "building_reference": manifest["dataset_id"],
                                **cursor,
                                "total": manifest["rows"],
                            }
                        ),
                        flush=True,
                    )
                yield from cached.cast(schema).to_batches()

        clone = table.shallow_clone(output / "merged-000.lance", reference["lance_version"])
        clone.add_columns(pa.RecordBatchReader.from_batches(schema, columns()), batch_size=8192)
        if cursor["all"] != manifest["rows"] or not cursor["train"] or not cursor["validation"]:
            raise ValueError("Incomplete reference build or empty train/validation partition")
        for role, index in indices.items():
            np.save(index_paths[role], index[: cursor[role]])
    result = {
        role: {
            **binding,
            "ready_rows": cursor[role],
            "coverage": {"ready": cursor[role]},
            "sampling_index": str(index_paths[role]),
            "sampling_sha256": file_hash(index_paths[role]),
            "merged": {
                **reference,
                "table_path": str(output / "merged-000.lance"),
                "lance_version": clone.version,
            },
        }
        for role in indices
    }
    return result


def create_reference_build(recipe, output, tokenizer_info, selection_sha):
    baseline_path = Path(recipe["reuse_build"]).resolve()
    baseline = read_complete(baseline_path)
    val_path = baseline_path.parent / "validation/manifest.json"
    val = read_complete(val_path)
    train_path = baseline_path.parent / "train/manifest.json"
    train = read_complete(train_path)
    if (
        baseline.get("representation") != "merged_indexed_references"
        or baseline["tokenizer"]["sha256"] != tokenizer_info["sha256"]
        or baseline["selection_sha256"] != selection_sha
        or val["evaluation"] != "heldout"
        or train["evaluation"] != "train_isolated"
        or val.get("split_sha256") != train.get("split_sha256")
        or val["split"]["source_build_sha256"] != file_hash(baseline_path)
    ):
        raise ValueError("Reuse requires the original token cache and its isolated holdout")
    configured = recipe["bindings"]
    old = {b["dataset_id"]: b for b in baseline["bindings"]}
    heldout = {b["dataset_id"]: b for b in val["bindings"]}
    if len(configured) != len(old) or {b["dataset_id"] for b in configured} != old.keys():
        raise ValueError("Reference experiment must cover each baseline dataset exactly once")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    arguments = [
        (
            recipe["root"],
            {**b, "binding_slot": slot},
            old[b["dataset_id"]],
            baseline_path.parent,
            heldout[b["dataset_id"]],
            val_path.parent,
            selection_sha,
            output / b["dataset_id"],
        )
        for slot, b in enumerate(configured)
    ]
    workers = recipe.get("workers", 4)
    if workers == 1:
        children = list(map(build_dataset, arguments))
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            children = list(pool.map(build_dataset, arguments))
    split = {
        "method": "preserve_baseline_target_membership_v1",
        "source_build_sha256": file_hash(baseline_path),
        "source_validation_sha256": file_hash(val_path),
        "source_training_sha256": file_hash(train_path),
    }
    manifests = {}
    for role, evaluation in (
        ("all", "no_holdout"),
        ("train", "train_isolated"),
        ("validation", "heldout"),
    ):
        destination = output if role == "all" else output / role
        destination.mkdir(exist_ok=True)
        bindings = [
            {
                **child[role],
                "binding_slot": i,
                "sampling_index": str(Path(child[role]["sampling_index"]).relative_to(destination)),
            }
            for i, child in enumerate(children)
        ]
        manifest = {
            **baseline,
            "build_id": output.name,
            "reference_scope": "codec_grid_reference",
            "model_protocol": {
                **baseline["model_protocol"],
                "loss_mask": "reference_codec_interval",
            },
            "bindings": bindings,
            "evaluation": evaluation,
            "split": split,
            "split_sha256": digest(split),
            "recipe_sha256": digest(recipe),
            "coverage": {"ready": sum(b["ready_rows"] for b in bindings), "dataset_unbound": 0},
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
        manifests[role] = manifest
    (output / "data_recipe.json").write_text(json.dumps(recipe, indent=2))
    for configured_binding, child in zip(configured, children, strict=True):
        directory = output / configured_binding["dataset_id"]
        child_recipe = {**recipe, "bindings": [configured_binding]}
        manifest = {
            **manifests["all"],
            "build_id": directory.name,
            "bindings": [{**child["all"], "binding_slot": 0, "sampling_index": "sampling-000.npy"}],
            "coverage": {
                "ready": child["all"]["ready_rows"],
                "dataset_unbound": baseline["selected_rows"] - old[directory.name]["ready_rows"],
            },
            "recipe_sha256": digest(child_recipe),
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (directory / "data_recipe.json").write_text(
            json.dumps(child_recipe, indent=2)
        )
    return manifests["train"]
