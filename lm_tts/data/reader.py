"""Batched fixed-snapshot feature reads with mmap sampling metadata."""

import warnings
from collections import defaultdict
from numbers import Integral
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

from ..artifacts import file_hash
from .build import open_snapshot, read_complete
from .reference import REFERENCE_COLUMNS, is_reference, validate_reference


class MetadataColumn:
    def __init__(self, dataset, name, extra=0):
        self.dataset, self.name, self.extra = dataset, name, extra

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        _, row = self.dataset.locate(index)
        value = row[self.name]
        if self.name == "tokens":
            value += row["frames"]
        return int(value) + self.extra


class FeatureDataset:
    def __init__(
        self,
        build_path,
        text_special_tokens,
        *,
        verify_integrity=True,
        use_speaker_embedding=True,
        mask_reference=False,
        text_vocab_size=None,
    ):
        self.path = Path(build_path).resolve()
        self.manifest = read_complete(self.path)
        if self.manifest.get("artifact_kind") != "training_build":
            raise ValueError("Expected a published training build")
        if self.manifest["speaker_conditioning_mode"] != "frozen_embedding":
            raise ValueError("FeatureDataset requires frozen_embedding")
        self.fingerprint = file_hash(self.path)
        self.text_special_tokens = text_special_tokens
        self.text_vocab_size = text_vocab_size
        if text_vocab_size is not None and (
            text_vocab_size < 1 or any(not 0 <= t < text_vocab_size for t in text_special_tokens)
        ):
            raise ValueError("Text special tokens must be inside the model vocabulary")
        self.use_speaker_embedding = use_speaker_embedding
        self.mask_reference = mask_reference
        self.bindings = self.manifest["bindings"]
        if mask_reference and not all(is_reference(b) for b in self.bindings):
            raise ValueError("Reference masking requires reference features for every binding")
        if (
            use_speaker_embedding
            and any(is_reference(b) for b in self.bindings)
            and not mask_reference
        ):
            raise ValueError("Reference speaker conditioning requires train.mask_reference=true")
        self.indices, lengths = [], []
        for binding in self.bindings:
            path = self.path.parent / binding["sampling_index"]
            if verify_integrity and file_hash(path) != binding["sampling_sha256"]:
                raise ValueError("Sampling metadata changed")
            self.indices.append(np.load(path, mmap_mode="r")[: binding["ready_rows"]])
            lengths.append(binding["ready_rows"])
            for kind in ("merged",) if "merged" in binding else ("codec", "speaker"):
                ref = binding[kind]
                if verify_integrity and file_hash(ref["manifest_path"]) != ref["manifest_sha256"]:
                    raise ValueError("Feature manifest changed")
        self.offsets = np.cumsum([0, *lengths])
        if not self.offsets[-1]:
            raise ValueError("Empty training build")
        self.tables = {}
        self.bad_rows_reported = 0
        self.frame_lengths = MetadataColumn(self, "frames")
        self.token_lengths = MetadataColumn(self, "tokens", extra=10)

    def __len__(self):
        return int(self.offsets[-1])

    def validate_budgets(self, max_frames, max_tokens):
        for index in self.indices:
            for start in range(0, len(index), 1000000):
                part = index[start : start + 1000000]
                if np.any(part["frames"] > max_frames) or np.any(
                    part["tokens"] + part["frames"] + 10 > max_tokens
                ):
                    raise ValueError(
                        "Training build contains samples exceeding frame/token budgets"
                    )

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        slot = int(np.searchsorted(self.offsets, index, side="right") - 1)
        return slot, self.indices[slot][index - self.offsets[slot]]

    def __getstate__(self):
        # Workers reopen memory maps; they must not pickle the corpus arrays.
        return {
            "path": self.path,
            "text_special_tokens": self.text_special_tokens,
            "fingerprint": self.fingerprint,
            "use_speaker_embedding": self.use_speaker_embedding,
            "mask_reference": self.mask_reference,
            "text_vocab_size": self.text_vocab_size,
        }

    def __setstate__(self, state):
        # Parent validated large indices. Recheck the small manifest in workers.
        self.__init__(
            state["path"],
            state["text_special_tokens"],
            verify_integrity=False,
            use_speaker_embedding=state["use_speaker_embedding"],
            mask_reference=state["mask_reference"],
            text_vocab_size=state["text_vocab_size"],
        )
        if self.fingerprint != state["fingerprint"]:
            raise ValueError("Training build changed while starting workers")

    def table(self, slot, kind):
        key = (slot, kind)
        if key not in self.tables:
            self.tables[key] = open_snapshot(self.bindings[slot][kind])
        return self.tables[key]

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        groups = defaultdict(list)
        for position, index in enumerate(indices):
            slot, row = self.locate(index)
            groups[slot].append((position, int(row["row"])))
        result = [None] * len(indices)
        errors = (ValueError, TypeError, OSError, RuntimeError, pa.ArrowException)
        for slot, group in groups.items():
            positions, locators = zip(*group)
            try:
                rows = self._read_rows(slot, locators)
            except errors:
                # Isolate a corrupt payload; one failed batched take must not discard its peers.
                rows = []
                for locator in locators:
                    try:
                        rows.extend(self._read_rows(slot, [locator]))
                    except errors as error:
                        self._report_bad_row(slot, locator, error)
                        rows.append(None)
            for position, locator, pair in zip(positions, locators, rows, strict=True):
                if pair is None:
                    continue
                try:
                    result[position] = self._sample(*pair, self.bindings[slot])
                except errors as error:
                    self._report_bad_row(slot, locator, error)
        return result

    def _report_bad_row(self, slot, locator, error):
        # Bound per-worker log volume; exact skipped counts are reported by the trainer.
        if self.bad_rows_reported < 5:
            warnings.warn(
                f"Skipping bad sample: dataset={self.bindings[slot]['dataset_id']} row={locator}: {error}",
                RuntimeWarning,
            )
        self.bad_rows_reported += 1

    def _read_rows(self, slot, locators):
        binding = self.bindings[slot]
        if "merged" in binding:
            from .merged import METADATA_COLUMNS

            rows = (
                self.table(slot, "merged")
                .take(
                    list(locators),
                    columns=[*METADATA_COLUMNS, "codec_codes", "text_ids"]
                    + (["speaker_embedding"] if self.use_speaker_embedding else [])
                    + (REFERENCE_COLUMNS if is_reference(binding) else []),
                )
                .to_pylist()
            )
            codec, speaker = [], []
            for row in rows:
                codec.append(
                    {
                        **row,
                        **{
                            k.removeprefix("codec_"): v
                            for k, v in row.items()
                            if k.startswith("codec_")
                        },
                    }
                )
                speaker.append(
                    {
                        **row,
                        **{
                            k.removeprefix("speaker_"): v
                            for k, v in row.items()
                            if k.startswith("speaker_")
                        },
                    }
                )
        else:
            codec = (
                self.table(slot, "codec")
                .take(
                    list(locators),
                    columns=[
                        "target_id",
                        "parent_sample_id",
                        "profile_id",
                        "audio_sha256",
                        "start_frame",
                        "end_frame",
                        "native_sample_rate",
                        "status",
                        "codes",
                        "num_codec_frames",
                        "text",
                        "language",
                        "speaker_id",
                        "text_ids",
                        "speaker_row",
                        "build_ready",
                    ],
                )
                .to_pylist()
            )
            speaker = (
                self.table(slot, "speaker")
                .take(
                    [r["speaker_row"] if r["speaker_row"] is not None else 0 for r in codec],
                    columns=[
                        "target_id",
                        "parent_sample_id",
                        "profile_id",
                        "audio_sha256",
                        "start_frame",
                        "end_frame",
                        "native_sample_rate",
                        "status",
                    ]
                    + (["embedding"] if self.use_speaker_embedding else []),
                )
                .to_pylist()
            )
        return list(zip(codec, speaker, strict=True))

    def _sample(self, c, s, binding):
        if "merged" in binding:
            from .merged import validate_row

            validate_row(c, binding)
            if self.use_speaker_embedding and len(s["embedding"]) != s["embedding_dim"]:
                raise ValueError("Merged speaker embedding dimension differs")
        elif not c["build_ready"] or c["speaker_row"] is None:
            raise ValueError("Sampling index selected an unready row")
        if not c["text_ids"] or not isinstance(c["text"], str) or not c["text"].strip():
            raise ValueError("Empty training text")
        if any(
            isinstance(token, bool)
            or not isinstance(token, Integral)
            or token < 0
            or (self.text_vocab_size is not None and token >= self.text_vocab_size)
            for token in c["text_ids"]
        ):
            raise ValueError("Invalid text token type or vocabulary range")
        keys = (
            "target_id",
            "parent_sample_id",
            "audio_sha256",
            "start_frame",
            "end_frame",
            "native_sample_rate",
        )
        if "merged" in binding:
            keys = tuple(k for k in keys if k != "end_frame")
        if is_reference(binding):
            keys = tuple(k for k in keys if k != "start_frame")
        if any(c[k] != s[k] for k in keys) or c["status"] != "ok" or s["status"] != "ok":
            raise ValueError("Cached speaker locator has the wrong identity/interval")
        if (
            c["profile_id"] != binding["codec"]["profile_id"]
            or s["profile_id"] != binding["speaker"]["profile_id"]
        ):
            raise ValueError("Feature profile differs from build binding")
        codes = torch.tensor(c["codes"], dtype=torch.long)
        if (
            codes.shape != (c["num_codec_frames"], 16)
            or not len(codes)
            or codes.min() < 0
            or codes.max() >= 2048
        ):
            raise ValueError("Invalid codec shape or vocabulary")
        bos, eos = self.text_special_tokens
        sample = {
            "id": c["target_id"],
            "text": c["text"],
            "language": c["language"],
            "speaker": c["speaker_id"],
            "text_ids": [bos, *c["text_ids"], eos],
            "codes": codes,
            "num_frames": len(codes),
            "duration": (c["end_frame"] - c["start_frame"]) / c["native_sample_rate"],
        }
        if self.use_speaker_embedding:
            vector = torch.tensor(s["embedding"], dtype=torch.float32)
            if vector.ndim != 1 or not torch.isfinite(vector).all() or vector.norm() == 0:
                raise ValueError("Invalid cached speaker embedding")
            sample["speaker_embedding"] = vector
        if self.mask_reference:
            sample["reference_interval"] = validate_reference(c)
        return sample
