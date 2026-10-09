"""Bind published unified features once; never join the full corpus during training."""

import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
from transformers import AutoTokenizer

from ..artifacts import digest, file_hash
from ..config import read_yaml

INDEX_DTYPE = np.dtype(
    [
        ("row", "<i8"),
        ("frames", "<i4"),
        ("tokens", "<i4"),
        ("language", "<i2"),
        ("duration", "<f8"),
    ]
)
IDENTITY_COLUMNS = [
    "target_id",
    "parent_sample_id",
    "audio_sha256",
    "start_frame",
    "end_frame",
    "native_sample_rate",
]


def read_complete(path, kind=None):
    path = Path(path).resolve()
    if any(p.endswith(".incomplete") for p in path.parts):
        raise ValueError(f"Unpublished input: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"Incomplete manifest: {path}")
    if kind is not None and manifest.get("kind") != kind:
        raise ValueError(f"Expected {kind} manifest: {path}")
    return manifest


def open_snapshot(reference):
    version = reference["lance_version"]
    if type(version) is not int or version < 1:
        raise ValueError("A positive fixed Lance version is required")
    branch = reference.get("branch")
    if branch:
        return lance.dataset(reference["table_path"]).checkout_version((branch, version))
    return lance.dataset(reference["table_path"], version=version)


def feature_reference(root, path, kind):
    path = Path(path)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    manifest = read_complete(path, kind)
    release = next(
        (
            p
            for p in path.parents
            if p.name == manifest["release_id"]
            and p.parent.name == manifest["dataset_id"]
            and p.parent.parent.name == "datasets"
        ),
        None,
    )
    if release is None:
        raise ValueError("Feature manifest must live within datasets/<dataset>/<release>")
    reference = {
        "manifest_path": str(path),
        "manifest_sha256": file_hash(path),
        "table_path": str(release / manifest["table_path"]),
        "branch": None,
        "lance_version": manifest["lance_version"],
        "profile_id": manifest["profile_id"],
    }
    table = open_snapshot(reference)
    if table.count_rows() != manifest["rows"]:
        raise ValueError("Feature manifest row count disagrees with its snapshot")
    if any(f.metadata.deletion_file for f in table.get_fragments()):
        raise ValueError("Build inputs cannot contain deleted rows")
    return manifest, reference, table


def tokenizer_identity(directory):
    directory = Path(directory).resolve()
    files = {
        p.name: file_hash(p)
        for p in directory.iterdir()
        if p.is_file()
        and (
            p.name.startswith("tokenizer")
            or p.name in {"vocab.json", "merges.txt", "special_tokens_map.json", "config.json"}
        )
    }
    if not any(name.startswith("tokenizer") for name in files):
        raise ValueError("Tokenizer must be a local, fixed artifact")
    return {"path": str(directory), "files": files, "sha256": digest(files)}


def verify_tokenizer_compatibility(directory, expected):
    """Keep legacy build fingerprints while allowing different model-only dimensions."""
    source = tokenizer_identity(expected["path"])
    if source["sha256"] != expected["sha256"]:
        raise ValueError("Build tokenizer artifact changed")
    target = tokenizer_identity(directory)
    if target["sha256"] == source["sha256"]:
        return
    # Older builds included the whole model config in their tokenizer fingerprint.
    # Tokenizer bytes and every text protocol ID still have to match exactly.
    files = [
        {k: v for k, v in item["files"].items() if k != "config.json"} for item in (source, target)
    ]
    if files[0] != files[1]:
        raise ValueError("Assembled tokenizer files differ from the training build")
    configs = [
        json.loads((Path(item["path"]) / "config.json").read_text()) for item in (source, target)
    ]

    def protocol(value):
        talker = value.get("talker_config", {})
        return {
            "model_type": value.get("model_type"),
            "token_ids": {
                k: v for k, v in value.items() if k.endswith(("_token_id", "_token_ids"))
            },
            "input_protocol": talker.get("lm_tts_input_protocol"),
            "role_ids": talker.get("lm_tts_role_ids"),
            "pad_id": talker.get("lm_tts_pad_token_id"),
            "text_vocab_size": talker.get("text_vocab_size"),
        }

    if protocol(configs[0]) != protocol(configs[1]):
        raise ValueError("Assembled tokenizer protocol differs from the training build")


def create_build(recipe_path, output, *, paths=None):
    recipe = read_yaml(
        recipe_path,
        overrides={"paths": paths} if paths is not None else None,
        keys=(
            "root",
            "selection_manifest",
            "tokenizer",
            "reference_policy",
            "speaker_conditioning_mode",
            "evaluation",
            "bindings",
            "workers",
            "validation_fraction",
            "split_seed",
            "reuse_build",
        ),
    )
    root = Path(recipe["root"]).resolve()
    selection_path = root / recipe["selection_manifest"]
    selection = read_complete(selection_path)
    selection_hash = file_hash(selection_path)
    if (
        recipe["reference_policy"] != "self"
        or recipe["speaker_conditioning_mode"] != "frozen_embedding"
    ):
        raise ValueError("The first unified build uses self-reference and frozen_embedding")
    if recipe["evaluation"] != "no_holdout":
        raise ValueError(
            "Use an independently isolated evaluation build; this selection has no holdout"
        )
    tokenizer_info = tokenizer_identity(recipe["tokenizer"])
    if recipe.get("reuse_build"):
        from .reference_build import create_reference_build

        return create_reference_build(recipe, output, tokenizer_info, selection_hash)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_info["path"], fix_mistral_regex=False, local_files_only=True
    )
    if any("merged_manifest" in binding for binding in recipe["bindings"]):
        from .merged import bind_merged

        manifest = bind_merged(
            recipe, output, selection, selection_path, selection_hash, tokenizer_info, tokenizer
        )
    else:
        manifest = bind_features(
            recipe, output, selection, selection_path, selection_hash, tokenizer_info, tokenizer
        )
    if recipe.get("validation_fraction"):
        from .split import split_build

        return split_build(
            Path(output) / "manifest.json",
            recipe["validation_fraction"],
            recipe.get("split_seed", 42),
        )
    return manifest


def bind_features(
    recipe, output, selection, selection_path, selection_hash, tokenizer_info, tokenizer
):
    root, output = Path(recipe["root"]).resolve(), Path(output).resolve()
    if not recipe["bindings"]:
        raise ValueError("At least one explicit codec/speaker binding is required")
    if len({b["codec_manifest"] for b in recipe["bindings"]}) != len(recipe["bindings"]):
        raise ValueError("Duplicate codec bindings would repeat training samples")
    output.mkdir(parents=True, exist_ok=False)
    name = output.name
    bound, profiles, language_ids = [], {}, {}
    datasets = set()
    for slot, binding in enumerate(recipe["bindings"]):
        codec, codec_ref, codec_table = feature_reference(root, binding["codec_manifest"], "codec")
        speaker, speaker_ref, speaker_table = feature_reference(
            root, binding["speaker_manifest"], "speaker_embedding"
        )
        dataset = codec["dataset_id"]
        if binding.get("dataset_id", dataset) != dataset:
            raise ValueError("Configured dataset_id differs from the feature manifest")
        if dataset in datasets:
            raise ValueError("The first build requires one codec/speaker binding per dataset")
        datasets.add(dataset)
        if (dataset, codec["release_id"]) != (speaker["dataset_id"], speaker["release_id"]):
            raise ValueError("Codec and speaker bindings must use the same dataset/release")
        for kind, manifest in [("codec", codec), ("speaker", speaker)]:
            if manifest["target_kind"] != "sample":
                raise ValueError("Current unified build requires full samples, not views")
            if manifest["selection"]["manifest_sha256"] != selection_hash:
                raise ValueError("Feature was not produced from the requested selection")
            if kind in profiles and profiles[kind] != manifest["profile_id"]:
                raise ValueError("Mixed feature profiles in one build")
            profiles[kind] = manifest["profile_id"]
            if digest(manifest["profile"]) != manifest["profile_id"]:
                raise ValueError("Feature profile digest changed")
        source = next((s for s in selection["outputs"] if s["dataset_id"] == dataset), None)
        if source is None or source["selected_rows"] != codec["rows"]:
            raise ValueError("Codec must account for the complete selected dataset")
        source_ref = {**source, "table_path": str(root / source["table_path"])}
        selected_table = open_snapshot(source_ref)
        required = {"text", "language", "speaker_id", "text_revision", "num_codec_frames"}
        if not required <= set(codec_table.schema.names):
            raise ValueError("Codec lacks published selection text columns")
        if any(k in codec_table.schema.names for k in ["build_ready", "speaker_row", "text_ids"]):
            raise ValueError("Build must derive from an unmodified feature snapshot")
        index_path = output / f"sampling-{slot:03d}.npy"
        index = np.lib.format.open_memmap(
            index_path, mode="w+", dtype=INDEX_DTYPE, shape=(codec["rows"],)
        )
        reasons, ready = Counter(), 0
        with tempfile.TemporaryDirectory(prefix="tts-build-") as temporary:
            # Arrow consumes the reader on a Rust worker thread, serially.
            db = sqlite3.connect(str(Path(temporary) / "join.sqlite"), check_same_thread=False)
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            db.execute(
                "CREATE TABLE selected (id TEXT PRIMARY KEY, seen INTEGER NOT NULL DEFAULT 0)"
            )
            for batch in selected_table.scanner(
                columns=["sample_id"],
                filter="selection_reason = 0",
                scan_in_order=True,
                batch_size=8192,
            ).to_batches():
                db.executemany(
                    "INSERT INTO selected(id) VALUES (?)",
                    ((i,) for i in batch.column(0).to_pylist()),
                )
            if db.execute("SELECT COUNT(*) FROM selected").fetchone()[0] != codec["rows"]:
                raise ValueError("Selection snapshot count changed")
            db.execute(
                "CREATE TABLE speaker (id TEXT PRIMARY KEY, row INTEGER, identity TEXT, status TEXT)"
            )
            offset = 0
            for batch in speaker_table.scanner(
                columns=[*IDENTITY_COLUMNS, "status", "profile_id"],
                scan_in_order=True,
                batch_size=8192,
            ).to_batches():
                records = batch.to_pylist()
                if any(r["profile_id"] != speaker_ref["profile_id"] for r in records):
                    raise ValueError("Mixed speaker profiles")
                db.executemany(
                    "INSERT INTO speaker VALUES (?,?,?,?)",
                    (
                        (
                            r["target_id"],
                            offset + i,
                            digest([r[k] for k in IDENTITY_COLUMNS]),
                            r["status"],
                        )
                        for i, r in enumerate(records)
                    ),
                )
                offset += len(records)
            db.commit()
            schema = pa.schema(
                [
                    ("speaker_row", pa.int64()),
                    ("build_ready", pa.bool_()),
                    ("build_reason", pa.string()),
                    ("text_ids", pa.list_(pa.int32())),
                    ("num_text_tokens", pa.int32()),
                ]
            )
            pinned = open_snapshot(codec_ref)

            def columns():
                nonlocal ready
                row_offset = 0
                projection = [
                    *IDENTITY_COLUMNS,
                    "status",
                    "profile_id",
                    "text",
                    "language",
                    "num_codec_frames",
                ]
                for batch in pinned.scanner(
                    columns=projection, scan_in_order=True, batch_size=4096
                ).to_batches():
                    records = batch.to_pylist()
                    ids = [r["target_id"] for r in records]
                    placeholders = ",".join("?" for _ in ids)
                    speakers = {
                        r[0]: r[1:]
                        for r in db.execute(
                            f"SELECT id,row,identity,status FROM speaker WHERE id IN ({placeholders})",
                            ids,
                        )
                    }
                    seen = dict(
                        db.execute(
                            f"SELECT id,seen FROM selected WHERE id IN ({placeholders})", ids
                        )
                    )
                    if len(set(ids)) != len(ids) or any(i not in seen or seen[i] for i in ids):
                        raise ValueError("Extra or duplicate codec target outside selection")
                    db.executemany("UPDATE selected SET seen=1 WHERE id=?", ((i,) for i in ids))
                    tokens = tokenizer([r["text"] for r in records], add_special_tokens=False)[
                        "input_ids"
                    ]
                    rows = []
                    for i, (r, text_ids) in enumerate(zip(records, tokens, strict=True)):
                        if r["profile_id"] != codec_ref["profile_id"]:
                            raise ValueError("Mixed codec profiles")
                        spk = speakers.get(r["target_id"])
                        reason = (
                            "codec_failed"
                            if r["status"] != "ok"
                            else (
                                "speaker_missing"
                                if spk is None
                                else ("speaker_failed" if spk[2] != "ok" else "ready")
                            )
                        )
                        if spk is not None and spk[1] != digest([r[k] for k in IDENTITY_COLUMNS]):
                            raise ValueError(
                                "Codec/speaker audio identity or native interval differs"
                            )
                        if (
                            not text_ids
                            or r["num_codec_frames"] is None
                            or r["num_codec_frames"] < 1
                        ):
                            if reason == "ready":
                                raise ValueError("Ready sample has empty tokens or frames")
                        rows.append(
                            {
                                "speaker_row": spk[0] if reason == "ready" else None,
                                "build_ready": reason == "ready",
                                "build_reason": reason,
                                "text_ids": text_ids,
                                "num_text_tokens": len(text_ids),
                            }
                        )
                        reasons[reason] += 1
                        if reason == "ready":
                            language = r["language"]
                            if language not in language_ids:
                                language_ids[language] = len(language_ids)
                            duration = (r["end_frame"] - r["start_frame"]) / r["native_sample_rate"]
                            index[ready] = (
                                row_offset + i,
                                r["num_codec_frames"],
                                len(text_ids),
                                language_ids[language],
                                duration,
                            )
                            ready += 1
                    row_offset += len(records)
                    yield pa.RecordBatch.from_pylist(rows, schema=schema)

            # A local shallow clone references immutable source payload files.
            # Training metadata never changes the published feature dataset.
            build_table = codec_table.shallow_clone(
                output / f"codec-{slot:03d}.lance", codec_ref["lance_version"]
            )
            build_table.add_columns(
                pa.RecordBatchReader.from_batches(schema, columns()), batch_size=4096
            )
            if db.execute("SELECT COUNT(*) FROM selected WHERE seen=0").fetchone()[0]:
                raise ValueError("Codec misses selected targets")
            db.close()
        index.flush()
        index = None
        build_ref = {
            **codec_ref,
            "table_path": str(output / f"codec-{slot:03d}.lance"),
            "branch": None,
            "lance_version": build_table.version,
        }
        # Re-read added locators, independent of the generator callback order.
        observed = 0
        stored = np.load(index_path, mmap_mode="r")
        offset = 0
        for batch in build_table.scanner(
            columns=["build_ready", "speaker_row", "num_codec_frames", "num_text_tokens"],
            scan_in_order=True,
            batch_size=8192,
        ).to_batches():
            for i, r in enumerate(batch.to_pylist()):
                if r["build_ready"]:
                    item = stored[observed]
                    if (int(item["row"]), int(item["frames"]), int(item["tokens"])) != (
                        offset + i,
                        r["num_codec_frames"],
                        r["num_text_tokens"],
                    ) or r["speaker_row"] is None:
                        raise ValueError("Build table/index round-trip differs")
                    observed += 1
            offset += batch.num_rows
        if observed != ready or open_snapshot(codec_ref).count_rows() != codec["rows"]:
            raise ValueError("Build coverage or source snapshot changed")
        if ready == 0:
            raise ValueError("No trainable rows in binding")
        bound.append(
            {
                "binding_slot": slot,
                "dataset_id": dataset,
                "codec": build_ref,
                "source_codec": codec_ref,
                "speaker": speaker_ref,
                "speaker_profile": speaker["profile"],
                "sampling_index": index_path.name,
                "sampling_sha256": file_hash(index_path),
                "ready_rows": ready,
                "selected_rows": codec["rows"],
                "coverage": dict(reasons),
            }
        )
    unbound = [
        s for s in selection["outputs"] if s["dataset_id"] not in datasets and s["selected_rows"]
    ]
    coverage = Counter()
    for binding in bound:
        coverage.update(binding["coverage"])
    coverage["dataset_unbound"] = sum(s["selected_rows"] for s in unbound)
    selected_rows = sum(s["selected_rows"] for s in selection["outputs"])
    if sum(coverage.values()) != selected_rows:
        raise ValueError("Ready plus unready targets must account for the complete selection")
    manifest = {
        "contract_version": "v0.1",
        "status": "complete",
        "artifact_kind": "training_build",
        "representation": "indexed_references",
        "build_id": name,
        "reference_policy": "self",
        "reference_scope": "target",
        "reference_count": 1,
        "model_protocol": {
            "conditioning": "speaker_only",
            "requires_reference_text": False,
            "requires_reference_codec": False,
            "uses_speaker_conditioning": True,
        },
        "speaker_conditioning_mode": "frozen_embedding",
        "evaluation": recipe["evaluation"],
        "selection_manifest": str(selection_path),
        "selection_sha256": selection_hash,
        "tokenizer": tokenizer_info,
        "profiles": profiles,
        "languages": language_ids,
        "bindings": bound,
        "coverage": dict(coverage),
        "selected_rows": selected_rows,
        "unbound_sources": unbound,
        "recipe_sha256": digest(recipe),
    }
    (output / "data_recipe.json").write_text(json.dumps(recipe, ensure_ascii=False, indent=2))
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest
