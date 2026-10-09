"""Index published merged features without joining or rewriting source payloads."""

import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa

from ..artifacts import digest, file_hash
from .build import INDEX_DTYPE, feature_reference, read_complete
from .reference import is_reference, validate_reference

METADATA_COLUMNS = [
    "target_id",
    "dataset_id",
    "release_id",
    "audio_sha256",
    "text",
    "language",
    "speaker_id",
    "codec_parent_sample_id",
    "codec_profile_id",
    "codec_status",
    "codec_start_frame",
    "codec_end_frame",
    "codec_native_sample_rate",
    "codec_num_codec_frames",
    "speaker_parent_sample_id",
    "speaker_profile_id",
    "speaker_status",
    "speaker_start_frame",
    "speaker_end_frame",
    "speaker_native_sample_rate",
    "speaker_embedding_dim",
]


def validate_row(row, binding):
    if (row["dataset_id"], row["release_id"]) != (binding["dataset_id"], binding["release_id"]):
        raise ValueError("Merged row belongs to a different dataset/release")
    for kind in ("codec", "speaker"):
        if (
            row[f"{kind}_parent_sample_id"] != row["target_id"]
            or row[f"{kind}_status"] != "ok"
            or row[f"{kind}_profile_id"] != binding[kind]["profile_id"]
            or (row[f"{kind}_start_frame"] != 0 and (kind == "codec" or not is_reference(binding)))
        ):
            raise ValueError("Merged feature identity, status or profile differs")
    rate = row["codec_native_sample_rate"]
    codec_end, speaker_end = row["codec_end_frame"], row["speaker_end_frame"]
    if rate <= 0 or row["speaker_native_sample_rate"] != rate or min(codec_end, speaker_end) <= 0:
        raise ValueError("Invalid merged native audio range")
    profile = binding["merged_profile"]
    if is_reference(binding):
        validate_reference(row)
    # Published runs pin either the original two-frame tolerance or the later 20 ms policy.
    tolerance = (
        max(2, math.ceil(rate * profile["native_duration_tolerance_seconds"]))
        if "native_duration_tolerance_seconds" in profile
        else profile["native_frame_tolerance"]
    )
    if not is_reference(binding) and abs(codec_end - speaker_end) > tolerance:
        raise ValueError("Merged codec/speaker intervals exceed the published tolerance")
    if (
        not isinstance(row["text"], str)
        or not row["text"].strip()
        or row["codec_num_codec_frames"] < 1
    ):
        raise ValueError("Merged sample has empty text or codec frames")


def _build_dataset(arguments):
    from transformers import AutoTokenizer

    recipe, output, selection, selection_path, selection_hash, tokenizer_info = arguments
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_info["path"], local_files_only=True, fix_mistral_regex=False
    )
    return bind_merged(
        recipe, output, selection, selection_path, selection_hash, tokenizer_info, tokenizer
    )


def bind_merged(
    recipe, output, selection, selection_path, selection_hash, tokenizer_info, tokenizer
):
    root, output = Path(recipe["root"]).resolve(), Path(output).resolve()
    bindings = recipe["bindings"]
    if not bindings or any(set(b) != {"dataset_id", "merged_manifest"} for b in bindings):
        raise ValueError("Merged bindings require dataset_id and merged_manifest only")
    names = [b["dataset_id"] for b in bindings]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate merged datasets would repeat training samples")
    workers = recipe.get("workers", 1)
    if type(workers) is not int or workers < 1:
        raise ValueError("Data build workers must be a positive integer")
    output.mkdir(parents=True, exist_ok=False)
    if workers > 1 and len(bindings) > 1:
        arguments = [
            (
                {**recipe, "bindings": [binding], "workers": 1},
                output / binding["dataset_id"],
                selection,
                selection_path,
                selection_hash,
                tokenizer_info,
            )
            for binding in bindings
        ]
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            children = list(pool.map(_build_dataset, arguments))
        manifest = {
            **children[0],
            "build_id": output.name,
            "bindings": [],
            "recipe_sha256": digest(recipe),
        }
        manifest.pop("languages")
        for slot, child in enumerate(children):
            binding = dict(child["bindings"][0])
            binding.update(
                binding_slot=slot,
                languages=child["languages"],
                sampling_index=str(Path(binding["dataset_id"]) / binding["sampling_index"]),
            )
            manifest["bindings"].append(binding)
        unbound = [
            s for s in selection["outputs"] if s["dataset_id"] not in names and s["selected_rows"]
        ]
        manifest.update(
            unbound_sources=unbound,
            coverage={
                "ready": sum(b["ready_rows"] for b in manifest["bindings"]),
                "dataset_unbound": sum(s["selected_rows"] for s in unbound),
            },
        )
        (output / "data_recipe.json").write_text(json.dumps(recipe, ensure_ascii=False, indent=2))
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        return manifest
    bound, languages = [], {}
    for slot, configured in enumerate(bindings):
        manifest, reference, table = feature_reference(
            root, configured["merged_manifest"], "merged"
        )
        dataset = configured["dataset_id"]
        selected = next((s for s in selection["outputs"] if s["dataset_id"] == dataset), None)
        if (
            manifest["dataset_id"] != dataset
            or manifest["target_kind"] != "sample"
            or selected is None
            or selected["selected_rows"] != manifest["rows"]
            or manifest["selection"]["manifest_sha256"] != selection_hash
            or manifest["coverage"]["ok"] != manifest["rows"]
            or manifest["rows"] < 1
        ):
            raise ValueError("Merged manifest must fully cover its pinned selected dataset")
        if digest(manifest["profile"]) != manifest["profile_id"]:
            raise ValueError("Merged profile digest changed")
        sources, source_manifests = {}, {}
        for kind, feature_kind in (
            ("codec", "codec"),
            ("speaker", "speaker_embedding"),
            ("text", "text"),
        ):
            inputs = [i for i in manifest["inputs"] if i["alias"] == kind]
            if len(inputs) != 1:
                raise ValueError(f"Merged manifest requires exactly one {kind} source")
            source = dict(inputs[0])
            source_path = root / source["manifest_path"]
            source_manifest = read_complete(source_path, feature_kind)
            if (
                file_hash(source_path) != source["manifest_sha256"]
                or source_manifest["profile_id"] != manifest["profile"]["sources"][kind]
                or source_manifest["profile_id"] != source["profile_id"]
                or digest(source_manifest["profile"]) != source["profile_id"]
                or source_manifest["lance_version"] != source["lance_version"]
                or source_manifest["dataset_id"] != dataset
                or source_manifest["release_id"] != manifest["release_id"]
                or source_manifest["rows"] != manifest["rows"]
                or source_manifest["selection"]["manifest_sha256"] != selection_hash
            ):
                raise ValueError("Merged source identity, profile or selection changed")
            sources[kind], source_manifests[kind] = source, source_manifest
        binding = {
            "binding_slot": slot,
            "dataset_id": dataset,
            "release_id": manifest["release_id"],
            "source_merged": reference,
            "merged_profile": manifest["profile"],
            **sources,
            "speaker_profile": source_manifests["speaker"]["profile"],
            "ready_rows": manifest["rows"],
            "selected_rows": manifest["rows"],
            "coverage": {"ready": manifest["rows"]},
        }
        if not set(METADATA_COLUMNS + ["codec_codes", "speaker_embedding"]) <= set(
            table.schema.names
        ):
            raise ValueError("Merged table lacks required text, codec or speaker columns")
        if "text_ids" in table.schema.names:
            raise ValueError("Expected published features without training tokenization")
        index_path = output / f"sampling-{slot:03d}.npy"
        index = np.lib.format.open_memmap(
            index_path, mode="w+", dtype=INDEX_DTYPE, shape=(manifest["rows"],)
        )
        schema = pa.schema([("text_ids", pa.list_(pa.int32())), ("num_text_tokens", pa.int32())])
        observed = 0

        def columns():
            nonlocal observed
            # Project metadata only: codec arrays and embeddings stay in the immutable source.
            for batch in table.scanner(
                columns=[
                    "text",
                    "language",
                    "codec_num_codec_frames",
                    "codec_end_frame",
                    "codec_native_sample_rate",
                ],
                scan_in_order=True,
                batch_size=4096,
            ).to_batches():
                records = batch.to_pylist()
                if any(
                    not row["text"].strip() or row["codec_num_codec_frames"] < 1 for row in records
                ):
                    raise ValueError("Published merged table has empty text or codec frames")
                tokens = tokenizer([r["text"] for r in records], add_special_tokens=False)[
                    "input_ids"
                ]
                cached = []
                for row, ids in zip(records, tokens, strict=True):
                    if not ids:
                        raise ValueError("Merged text tokenized to an empty sequence")
                    language = row["language"]
                    if language not in languages:
                        languages[language] = len(languages)
                    index[observed] = (
                        observed,
                        row["codec_num_codec_frames"],
                        len(ids),
                        languages[language],
                        row["codec_end_frame"] / row["codec_native_sample_rate"],
                    )
                    observed += 1
                    cached.append({"text_ids": ids, "num_text_tokens": len(ids)})
                if observed % (4096 * 100) == 0 or observed == manifest["rows"]:
                    print(
                        json.dumps(
                            {
                                "building": dataset,
                                "indexed_rows": observed,
                                "rows": manifest["rows"],
                            }
                        ),
                        flush=True,
                    )
                yield pa.RecordBatch.from_pylist(cached, schema=schema)

        clone = table.shallow_clone(output / f"merged-{slot:03d}.lance", reference["lance_version"])
        clone.add_columns(pa.RecordBatchReader.from_batches(schema, columns()), batch_size=4096)
        if observed != manifest["rows"]:
            raise ValueError("Merged build coverage changed")
        index.flush()
        index = None
        binding.update(
            merged={
                **reference,
                "table_path": str(output / f"merged-{slot:03d}.lance"),
                "lance_version": clone.version,
            },
            sampling_index=index_path.name,
            sampling_sha256=file_hash(index_path),
        )
        bound.append(binding)
    unbound = [
        s for s in selection["outputs"] if s["dataset_id"] not in names and s["selected_rows"]
    ]
    manifest = {
        "contract_version": "v0.1",
        "status": "complete",
        "artifact_kind": "training_build",
        "representation": "merged_indexed_references",
        "build_id": output.name,
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
        "languages": languages,
        "bindings": bound,
        "coverage": {
            "ready": sum(b["ready_rows"] for b in bound),
            "dataset_unbound": sum(s["selected_rows"] for s in unbound),
        },
        "selected_rows": sum(s["selected_rows"] for s in selection["outputs"]),
        "unbound_sources": unbound,
        "recipe_sha256": digest(recipe),
    }
    (output / "data_recipe.json").write_text(json.dumps(recipe, ensure_ascii=False, indent=2))
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest
