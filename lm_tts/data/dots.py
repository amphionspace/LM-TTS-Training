"""dots.tts training builds, samples and batches over published dots_vae features.

A build pins one shallow clone of each dots_vae feature table and adds its training tokenization;
VAE posteriors and x-vectors stay in the immutable source payloads."""

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

from ..artifacts import digest, file_hash
from .build import INDEX_DTYPE, feature_reference, open_snapshot, read_complete
from .reader import MetadataColumn

FEATURE_KIND = "dots_vae"
LATENT_DIM, XVECTOR_DIM, PATCH_SIZE = 128, 512, 4
LATENT_RATE = 25  # 48 kHz / 1920-sample VAE hop
SPECIAL_TOKENS = {
    "start": "<|audio_gen_start|>",
    "span": "<|audio_gen_span|>",
    "end": "<|audio_gen_end|>",
    "text_end": "<|text_cond_end|>",
}
PROMPTS = {"text": "[文本]", "audio": "[文本对应语音]", "interleave": "[流式语音合成]"}
# Reference rules the feature pipeline publishes, each with its minimum span in patches; v2 lifts
# it to 3 s (capped at half the utterance) so the x-vector sees enough speech.
REFERENCE_RULES = {
    "dots-reference-v1:codec-grid-v1-on-patches": 4,
    "dots-reference-v2:min-3s-on-patches": 19,
}
TAG = re.compile(r"\[[A-Za-z]+(?:-[A-Za-z]+)*\]")


def reference_interval(target_id, patches, rule="dots-reference-v1:codec-grid-v1-on-patches"):
    """The half-open patch span whose audio alone yields the x-vector, or None. As in the codec
    reference policy: a uniform length in [max(0.5 s, 10%), 50%], then a uniform start; v2 also
    requires min(3 s, 50%). Without paired utterances the reference comes from the target itself,
    so its patches take no loss."""
    low, high = max(4, -(-patches // 10)), patches // 2
    if low > high:
        return None
    low = max(low, min(REFERENCE_RULES[rule], high))
    h = hashlib.sha256(f"{rule.partition(':')[0]}:{target_id}".encode()).digest()
    length = low + int.from_bytes(h[:8], "big") % (high - low + 1)
    start = int.from_bytes(h[8:16], "big") % (patches - length + 1)
    return start, start + length


class Layout:
    """The release's token layouts. Every input <|audio_gen_span|> carries one patch embedding;
    the position whose next token is a span predicts that patch.

    basic:      [文本] text [文本对应语音] <start> S*N <end> <eos>
    interleave: [流式语音合成] t1 S t2 S ... tn S <text_end> S ... S <end> <eos>
    """

    def __init__(self, tokenizer):
        self.ids = {
            key: tokenizer.convert_tokens_to_ids(token) for key, token in SPECIAL_TOKENS.items()
        }
        self.prompts = {
            key: tokenizer(text, add_special_tokens=False)["input_ids"]
            for key, text in PROMPTS.items()
        }
        self.eos = tokenizer.eos_token_id
        # Inputs drop the final EOS; the basic layout bounds both lengths for text <= N.
        self.overhead = len(self.prompts["text"]) + len(self.prompts["audio"]) + 2

    def sequence(self, text, patches, interleave):
        span, end = self.ids["span"], self.ids["end"]
        if not interleave:
            return [
                *self.prompts["text"],
                *text,
                *self.prompts["audio"],
                self.ids["start"],
                *[span] * patches,
                end,
                self.eos,
            ]
        sequence = list(self.prompts["interleave"])
        for token in text:
            sequence += [token, span]
        sequence += [self.ids["text_end"], *[span] * (patches - len(text)), end, self.eos]
        return sequence

    def to_dict(self):
        return {"ids": self.ids, "prompts": self.prompts, "eos": self.eos}


def utterance_key(text):
    """Text identity across re-encoded copies of one recording: tags removed, NFKC, lowercase,
    letters and digits only."""
    text = unicodedata.normalize("NFKC", TAG.sub("", text)).lower()
    return "".join(c for c in text if c.isalnum())


def is_duplicate(seen, text, seconds, tolerance=0.3, min_characters=5):
    """Same normalized text and a duration within `tolerance` of an earlier dataset's row. Texts
    shorter than `min_characters` (e.g. a lone tag) identify nothing and are left to the audio."""
    key = utterance_key(text)
    return len(key) >= min_characters and any(
        abs(seconds - other) <= tolerance for other in seen.get(key, ())
    )


def audio_fingerprint(latents, mean, std, segments=4):
    """Unit vector of the standardized VAE posterior mean averaged over equal time segments.
    Re-encodings of one recording keep it nearly unchanged; edge silence is already normalized."""
    z = (latents - mean) / std
    vector = np.concatenate([part.mean(0) for part in np.array_split(z, segments)])
    return vector / (np.linalg.norm(vector) + 1e-8)


def audio_matches(new_fp, new_seconds, old_fp, old_seconds, tolerance=0.3, chunk=4096):
    """Highest cosine similarity of each new row to an old row within `tolerance` seconds."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best = np.full(len(new_fp), -1.0, dtype=np.float32)
    if not len(old_fp) or not len(new_fp):
        return best
    order_old = np.argsort(old_seconds)
    old_sorted = old_seconds[order_old]
    old = torch.from_numpy(old_fp[order_old]).to(device, torch.float32)
    order_new = np.argsort(new_seconds)
    for start in range(0, len(order_new), chunk):
        rows = order_new[start : start + chunk]
        low = np.searchsorted(old_sorted, new_seconds[rows].min() - tolerance, side="left")
        high = np.searchsorted(old_sorted, new_seconds[rows].max() + tolerance, side="right")
        if low == high:
            continue
        query = torch.from_numpy(new_fp[rows]).to(device, torch.float32)
        gap = np.abs(new_seconds[rows][:, None] - old_sorted[None, low:high]) > tolerance
        similarity = (query @ old[low:high].T).masked_fill(torch.from_numpy(gap).to(device), -1.0)
        best[rows] = similarity.max(1).values.cpu().numpy()
    return best


def hours_subset(targets, durations, max_hours):
    """Positions (in original order) of a fixed pseudo-random subset holding at most max_hours.
    The order depends only on target IDs, so a larger cap keeps every row of a smaller one."""
    order = sorted(
        range(len(targets)),
        key=lambda i: hashlib.sha256(f"hours-cap-v1:{targets[i]}".encode()).digest(),
    )
    total = np.cumsum(durations[order]) / 3600
    return np.sort(np.asarray(order[: int(np.searchsorted(total, max_hours, side="right"))]))


def bind_dots(recipe, output, tokenizer_info, tokenizer):
    root, output = Path(recipe["root"]).resolve(), Path(output).resolve()
    bindings = recipe["bindings"]
    if not bindings or any(
        not {"dataset_id", "dots_manifest"}
        <= set(b)
        <= {
            "dataset_id",
            "dots_manifest",
            "text_kind",
            "max_hours",
            "dedup_against",
            "exclude_list",
        }
        for b in bindings
    ):
        raise ValueError(
            "dots bindings take dataset_id, dots_manifest and optionally text_kind, max_hours,"
            " dedup_against, exclude_list"
        )
    names = [b["dataset_id"] for b in bindings]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate dots datasets would repeat training samples")
    output.mkdir(parents=True, exist_ok=False)
    tags = set(tokenizer.get_added_vocab())
    utterances, fingerprints = {}, {}
    # Rows whose audio fingerprint matches an earlier binding at or above this cosine are dropped.
    audio_threshold = recipe.get("audio_dedup_threshold")
    if audio_threshold is not None:
        stats = json.loads((Path(tokenizer_info["path"]) / "latent_stats.json").read_text())
        latent_mean = np.asarray(stats["mean"], dtype=np.float32)
        latent_std = np.sqrt(np.asarray(stats["var"], dtype=np.float32))
    bound, languages, profiles = [], {}, set()
    for slot, configured in enumerate(bindings):
        manifest, reference, table = feature_reference(
            root, configured["dots_manifest"], FEATURE_KIND
        )
        dataset = configured["dataset_id"]
        if (
            manifest["dataset_id"] != dataset
            or digest(manifest["profile"]) != manifest["profile_id"]
        ):
            raise ValueError("dots manifest identity or profile digest differs")
        rule = manifest["profile"].get("reference_rule")
        if rule not in REFERENCE_RULES:
            raise ValueError("dots features lack reference x-vectors under a known rule")
        # A pinned text_kind (e.g. nv-canonical-v1) guards against reading another text revision.
        text_kind = configured.get("text_kind")
        if text_kind and manifest.get("text", {}).get("text_kind") != text_kind:
            raise ValueError(f"dots features carry another text revision: {dataset}")
        profiles.add(manifest["profile_id"])
        index_path = output / f"sampling-{slot:03d}.npy"
        index = np.lib.format.open_memmap(
            index_path, mode="w+", dtype=INDEX_DTYPE, shape=(manifest["rows"],)
        )
        schema = pa.schema([("text_ids", pa.list_(pa.int32())), ("num_text_tokens", pa.int32())])
        observed = ready = 0
        excluded = defaultdict(int)

        def columns():
            nonlocal observed, ready
            for batch in table.scanner(
                columns=[
                    "target_id",
                    "status",
                    "text",
                    "text_kind",
                    "language",
                    "num_latent_frames",
                    "processed_frames",
                    "reference_start_patch",
                    "reference_end_patch",
                    *(["vae_mean"] if audio_threshold is not None else []),
                ],
                scan_in_order=True,
                batch_size=4096,
            ).to_batches():
                latents = None
                if audio_threshold is not None:
                    latents = posteriors(batch.column("vae_mean"))
                    batch = batch.drop_columns(["vae_mean"])
                records = batch.to_pylist()
                tokens = tokenizer(
                    [r["text"] if r["status"] == "ok" else "" for r in records],
                    add_special_tokens=False,
                )["input_ids"]
                cached = []
                for position, (row, ids) in enumerate(zip(records, tokens, strict=True)):
                    frames = row["num_latent_frames"]
                    if row["status"] == "ok":
                        if text_kind and row["text_kind"] != text_kind:
                            raise ValueError(f"Row text_kind {row['text_kind']!r}: {dataset}")
                        # A tag the tokenizer does not hold would be read aloud as subwords.
                        unknown = set(TAG.findall(row["text"])) - tags
                        if unknown:
                            raise ValueError(
                                f"Tags outside the tokenizer {sorted(unknown)}: {dataset}"
                            )
                    reason = (
                        "status"
                        if row["status"] != "ok"
                        else "empty_text"
                        if not ids
                        else "latent_frames"
                        if not frames or frames % PATCH_SIZE
                        else "no_reference"
                        if row["reference_start_patch"] is None
                        else None
                    )
                    published = row["reference_start_patch"], row["reference_end_patch"]
                    if row["status"] == "ok" and reference_interval(
                        row["target_id"], frames // PATCH_SIZE, rule
                    ) != (None if published[0] is None else published):
                        raise ValueError(f"reference interval breaks {rule}: {dataset}")
                    seconds = row["processed_frames"] / 48000 if reason is None else None
                    if reason is None and row["target_id"] in listed:
                        reason = "listed_duplicate"
                    if reason is None and any(
                        is_duplicate(utterances[name], row["text"], seconds) for name in against
                    ):
                        reason = "duplicate"
                    if reason is None:
                        utterances[dataset].setdefault(utterance_key(row["text"]), []).append(
                            seconds
                        )
                        targets.append(row["target_id"])
                        if latents is not None:
                            prints.append(
                                audio_fingerprint(
                                    latents[position].numpy(), latent_mean, latent_std
                                )
                            )
                        # Upstream labels vary (zh-CN, en-US); the index keeps the base language.
                        base_language = row["language"].split("-")[0].lower()
                        language = languages.setdefault(base_language, len(languages))
                        index[ready] = (
                            observed,
                            frames // PATCH_SIZE,
                            len(ids),
                            language,
                            row["processed_frames"] / 48000,
                        )
                        ready += 1
                    else:
                        excluded[reason] += 1
                    observed += 1
                    cached.append({"text_ids": ids, "num_text_tokens": len(ids)})
                yield pa.RecordBatch.from_pylist(cached, schema=schema)

        targets, prints = [], []
        # Re-encoded copies of a recording share no ID or hash, only text and length; rows that
        # duplicate an earlier binding are excluded from the index, the tables stay unchanged.
        against = configured.get("dedup_against", [])
        if not set(against) <= set(utterances):
            raise ValueError(f"dedup_against must name earlier bindings: {dataset}")
        utterances[dataset] = {}
        # Externally matched duplicates: {"path": jsonl, "field": target_id key}, read only.
        listed = set()
        if "exclude_list" in configured:
            spec = configured["exclude_list"]
            with open(spec["path"]) as lines:
                listed = {json.loads(line)[spec["field"]] for line in lines}
        clone = table.shallow_clone(output / f"dots-{slot:03d}.lance", reference["lance_version"])
        clone.add_columns(pa.RecordBatchReader.from_batches(schema, columns()), batch_size=4096)
        if observed != manifest["rows"] or not ready:
            raise ValueError(f"dots build coverage changed or empty: {dataset}")
        kept = np.arange(ready)
        if audio_threshold is not None:
            prints = np.stack(prints).astype(np.float16)
            seconds = index["duration"][:ready].astype(np.float32)
            best = np.full(ready, -1.0, dtype=np.float32)
            for name in against:
                best = np.maximum(best, audio_matches(prints, seconds, *fingerprints[name]))
            similarity = {f">={t}": int((best >= t).sum()) for t in (0.9, 0.95, 0.97, 0.98, 0.99)}
            # Per-row best match, aligned with the pre-dedup index, for threshold audits.
            np.save(output / f"audio-similarity-{slot:03d}.npy", best)
            kept = kept[best < audio_threshold]
            excluded["audio_duplicate"] += ready - len(kept)
        if "max_hours" in configured:
            capped = hours_subset(
                [targets[i] for i in kept], index["duration"][kept], configured["max_hours"]
            )
            excluded["hours_cap"] += len(kept) - len(capped)
            kept = kept[capped]
        if len(kept) < ready:
            index[: len(kept)] = index[kept]
            ready = len(kept)
        if audio_threshold is not None:
            fingerprints[dataset] = (prints[kept], index["duration"][:ready].astype(np.float32))
        index.flush()
        index = None
        bound.append(
            {
                "binding_slot": slot,
                "dataset_id": dataset,
                "release_id": manifest["release_id"],
                "profile_id": manifest["profile_id"],
                "source": reference,
                "features": {
                    **reference,
                    "table_path": str(output / f"dots-{slot:03d}.lance"),
                    "lance_version": clone.version,
                },
                "sampling_index": index_path.name,
                "sampling_sha256": file_hash(index_path),
                "ready_rows": ready,
                "coverage": {"ready": ready, "excluded": dict(excluded)},
                **({"audio_similarity": similarity} if audio_threshold is not None else {}),
            }
        )
        print(json.dumps({"built": dataset, "ready": ready, "excluded": excluded}), flush=True)
    if len(profiles) != 1:
        raise ValueError("Mixed dots_vae profiles in one build")
    manifest = {
        "contract_version": "v0.1",
        "status": "complete",
        "artifact_kind": "training_build",
        "representation": FEATURE_KIND,
        "build_id": output.name,
        "evaluation": recipe["evaluation"],
        "tokenizer": tokenizer_info,
        "layout": Layout(tokenizer).to_dict(),
        "languages": languages,
        "bindings": bound,
        "coverage": {"ready": sum(b["ready_rows"] for b in bound)},
        "recipe_sha256": digest(recipe),
    }
    (output / "data_recipe.json").write_text(json.dumps(recipe, ensure_ascii=False, indent=2))
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def posteriors(column):
    """list<fixed_size_list<float32, 128>> -> per-row (T, 128) tensors without Python floats."""
    column = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    values = column.values.values.to_numpy(zero_copy_only=False).reshape(-1, LATENT_DIM)
    offsets = column.offsets.to_numpy()
    return [torch.from_numpy(values[a:b].copy()) for a, b in zip(offsets[:-1], offsets[1:])]


class DotsFeatureDataset:
    def __init__(self, build_path, layout, *, interleave_fraction=0.0, verify_integrity=True):
        self.path = Path(build_path).resolve()
        self.manifest = read_complete(self.path)
        if (
            self.manifest.get("artifact_kind") != "training_build"
            or self.manifest.get("representation") != FEATURE_KIND
        ):
            raise ValueError("Expected a published dots_vae training build")
        if self.manifest["layout"] != layout.to_dict():
            raise ValueError("Build token layout differs from the assembled tokenizer")
        if not 0 <= interleave_fraction <= 1:
            raise ValueError("interleave_fraction must be in [0, 1]")
        self.fingerprint = file_hash(self.path)
        self.layout, self.interleave_fraction = layout, interleave_fraction
        self.bindings = self.manifest["bindings"]
        self.indices = []
        for binding in self.bindings:
            path = self.path.parent / binding["sampling_index"]
            if verify_integrity:
                if file_hash(path) != binding["sampling_sha256"]:
                    raise ValueError("Sampling metadata changed")
                if (
                    file_hash(binding["source"]["manifest_path"])
                    != binding["source"]["manifest_sha256"]
                ):
                    raise ValueError("Feature manifest changed")
            self.indices.append(np.load(path, mmap_mode="r")[: binding["ready_rows"]])
        self.offsets = np.cumsum([0, *(len(index) for index in self.indices)])
        self.tables = {}
        self.frame_lengths = MetadataColumn(self, "frames")
        self.token_lengths = MetadataColumn(self, "tokens", extra=layout.overhead)

    def __len__(self):
        return int(self.offsets[-1])

    def validate_budgets(self, max_frames, max_tokens):
        for index in self.indices:
            if np.any(index["frames"] > max_frames) or np.any(
                index["tokens"] + index["frames"] + self.layout.overhead > max_tokens
            ):
                raise ValueError("Training build contains samples exceeding frame/token budgets")

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        slot = int(np.searchsorted(self.offsets, index, side="right") - 1)
        return slot, self.indices[slot][index - self.offsets[slot]]

    def __getstate__(self):
        # Workers reopen memory maps and Lance tables instead of pickling them.
        return {
            "path": self.path,
            "layout": self.layout,
            "interleave_fraction": self.interleave_fraction,
            "fingerprint": self.fingerprint,
        }

    def __setstate__(self, state):
        self.__init__(
            state["path"],
            state["layout"],
            interleave_fraction=state["interleave_fraction"],
            verify_integrity=False,
        )
        if self.fingerprint != state["fingerprint"]:
            raise ValueError("Training build changed while starting workers")

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        groups = defaultdict(list)
        for position, index in enumerate(indices):
            slot, row = self.locate(index)
            groups[slot].append((position, row))
        result = [None] * len(indices)
        for slot, group in groups.items():
            if slot not in self.tables:
                self.tables[slot] = open_snapshot(self.bindings[slot]["features"])
            rows = self.tables[slot].take(
                [int(row["row"]) for _, row in group],
                columns=[
                    "target_id",
                    "status",
                    "text_ids",
                    "vae_mean",
                    "vae_logstd",
                    "reference_start_patch",
                    "reference_end_patch",
                    "reference_xvector",
                ],
            )
            means, log_stds = posteriors(rows["vae_mean"]), posteriors(rows["vae_logstd"])
            xvectors = (
                rows["reference_xvector"].combine_chunks().values.to_numpy(zero_copy_only=False)
            )
            for i, (position, row) in enumerate(group):
                patches = int(row["frames"])
                if rows["status"][i].as_py() != "ok" or len(means[i]) != patches * PATCH_SIZE:
                    raise ValueError("Sampling index selected an unready or misaligned row")
                text = rows["text_ids"][i].as_py()
                target = rows["target_id"][i].as_py()
                digest_ = hashlib.sha256(f"interleave-v1:{target}".encode()).digest()
                chosen = int.from_bytes(digest_[:8], "little") < self.interleave_fraction * 2**64
                result[position] = {
                    "id": target,
                    "text_ids": text,
                    # Interleaving needs every text token placed before the speech runs out.
                    "interleave": chosen and len(text) <= patches,
                    "vae_mean": means[i],
                    "vae_logstd": log_stds[i],
                    "reference": (
                        rows["reference_start_patch"][i].as_py(),
                        rows["reference_end_patch"][i].as_py(),
                    ),
                    "xvector": torch.from_numpy(
                        xvectors[i * XVECTOR_DIM : (i + 1) * XVECTOR_DIM].copy()
                    ),
                    "duration": float(row["duration"]),
                }
        return result


class DotsCollator:
    """Packs utterances for DotsTTSModel. `frame_lengths` counts patches, the model's
    autoregressive steps, so the trainer's per-utterance accounting applies unchanged."""

    def __init__(self, layout):
        self.layout = layout

    def __call__(self, rows):
        skipped = sum(row is None for row in rows)
        rows = [row for row in rows if row is not None]
        if not rows:
            return {"skipped_samples": torch.tensor(skipped)}
        span = self.layout.ids["span"]
        inputs, spans, predictors, offset = [], [], [], 0
        for row in rows:
            sequence = torch.tensor(
                self.layout.sequence(
                    row["text_ids"], len(row["vae_mean"]) // PATCH_SIZE, row["interleave"]
                )
            )
            inputs.append(sequence[:-1])
            spans.append(offset + (sequence[:-1] == span).nonzero()[:, 0])
            predictors.append(offset + (sequence[1:] == span).nonzero()[:, 0])
            offset += len(sequence) - 1
        return {
            "input_ids": torch.cat(inputs),
            "sequence_lengths": torch.tensor([len(x) for x in inputs]),
            "span_positions": torch.cat(spans),
            "predictor_positions": torch.cat(predictors),
            "frame_lengths": torch.tensor([len(r["vae_mean"]) // PATCH_SIZE for r in rows]),
            "reference_intervals": torch.tensor([r["reference"] for r in rows]),
            "vae_mean": torch.cat([r["vae_mean"] for r in rows]),
            "vae_logstd": torch.cat([r["vae_logstd"] for r in rows]),
            "xvectors": torch.stack([r["xvector"] for r in rows]),
            "interleave": torch.tensor([r["interleave"] for r in rows]),
            "audio_seconds": torch.tensor([r["duration"] for r in rows], dtype=torch.float64),
            "skipped_samples": torch.tensor(skipped),
        }
