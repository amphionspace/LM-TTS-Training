"""Batched fixed-snapshot feature reads with mmap sampling metadata."""

import hashlib
import math
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
        language_codec_ids=None,
        language_auto_fraction=0.0,
        streaming_fraction=0.0,
        streaming_max_text_ratio=None,
        streaming_pause_probability=0.0,
        streaming_max_pause_frames=0,
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
        # ISO language -> codec language ID (-1 = Auto); None keeps the Auto-only prefix.
        self.language_codec_ids = language_codec_ids
        self.language_auto_fraction = language_auto_fraction
        # A seeded share of text-leading samples uses the official streaming layout.
        self.streaming_fraction = streaming_fraction
        self.streaming_max_text_ratio = streaming_max_text_ratio
        if streaming_fraction and not 0 < (streaming_max_text_ratio or 0) <= 1:
            raise ValueError("Streaming needs streaming_max_text_ratio in (0, 1]")
        # Text that arrives late (an upstream LLM stalling) shows up as tts_pad inside the text.
        self.streaming_pause_probability = streaming_pause_probability
        self.streaming_max_pause_frames = streaming_max_pause_frames
        if streaming_pause_probability and streaming_max_pause_frames < 1:
            raise ValueError("Streaming pauses need streaming_max_pause_frames >= 1")
        self.prefix_tokens = 10 + (language_codec_ids is not None)
        self.bindings = self.manifest["bindings"]
        if language_codec_ids is not None:
            # Fail before training instead of skipping every row of an unmapped language.
            built = set(self.manifest.get("languages", {}))
            for binding in self.bindings:
                built |= set(binding.get("languages", {}))
            if not built or built - set(language_codec_ids):
                raise ValueError(
                    f"No codec language mapping for {sorted(built - set(language_codec_ids))}"
                )
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
            refs = [
                binding[k] for k in (("merged",) if "merged" in binding else ("codec", "speaker"))
            ]
            if "alternate" in binding:
                refs.append(binding["alternate"]["merged"])
            for ref in refs:
                if verify_integrity and file_hash(ref["manifest_path"]) != ref["manifest_sha256"]:
                    raise ValueError("Feature manifest changed")
        self.offsets = np.cumsum([0, *lengths])
        if not self.offsets[-1]:
            raise ValueError("Empty training build")
        self.tables = {}
        self.bad_rows_reported = 0
        self.frame_lengths = MetadataColumn(self, "frames")
        self.token_lengths = MetadataColumn(self, "tokens", extra=self.prefix_tokens)

    def __len__(self):
        return int(self.offsets[-1])

    def validate_budgets(self, max_frames, max_tokens):
        for index in self.indices:
            for start in range(0, len(index), 1000000):
                part = index[start : start + 1000000]
                if np.any(part["frames"] > max_frames) or np.any(
                    part["tokens"] + part["frames"] + self.prefix_tokens > max_tokens
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
            "language_codec_ids": self.language_codec_ids,
            "language_auto_fraction": self.language_auto_fraction,
            "streaming_fraction": self.streaming_fraction,
            "streaming_max_text_ratio": self.streaming_max_text_ratio,
            "streaming_pause_probability": self.streaming_pause_probability,
            "streaming_max_pause_frames": self.streaming_max_pause_frames,
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
            language_codec_ids=state["language_codec_ids"],
            language_auto_fraction=state["language_auto_fraction"],
            streaming_fraction=state["streaming_fraction"],
            streaming_max_text_ratio=state["streaming_max_text_ratio"],
            streaming_pause_probability=state["streaming_pause_probability"],
            streaming_max_pause_frames=state["streaming_max_pause_frames"],
        )
        if self.fingerprint != state["fingerprint"]:
            raise ValueError("Training build changed while starting workers")

    def table(self, slot, kind):
        key = (slot, kind)
        if key not in self.tables:
            binding = self.bindings[slot]
            reference = binding["alternate"]["merged"] if kind == "alternate" else binding[kind]
            self.tables[key] = open_snapshot(reference)
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
                    result[position] = self._sample(*pair, self._binding(slot, locator))
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

    def _binding(self, slot, row):
        # Mixed builds offset alternate-reference rows; they validate against that reference.
        binding = self.bindings[slot]
        alternate = binding.get("alternate")
        if alternate is not None and row >= alternate["offset"]:
            return {**binding, **alternate}
        return binding

    def _read_rows(self, slot, locators):
        binding = self.bindings[slot]
        alternate = binding.get("alternate")
        if alternate is None:
            return self._read_table(slot, "merged", binding, locators)
        offset, result = alternate["offset"], [None] * len(locators)
        for kind, shift in (("merged", 0), ("alternate", offset)):
            items = [(i, r - shift) for i, r in enumerate(locators) if (r >= offset) == bool(shift)]
            if items:
                positions, rows = zip(*items)
                for i, pair in zip(
                    positions, self._read_table(slot, kind, binding, rows), strict=True
                ):
                    result[i] = pair
        return result

    def _read_table(self, slot, kind, binding, locators):
        if "merged" in binding:
            from .merged import METADATA_COLUMNS

            rows = (
                self.table(slot, kind)
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
        if self.language_codec_ids is not None:
            sample["language_codec_id"] = self._language_codec_id(c["target_id"], c["language"])
        if self.streaming_fraction:
            # Text must lead speech: one text token per 12.5 Hz frame, with headroom below 1.
            eligible = len(sample["text_ids"]) <= self.streaming_max_text_ratio * len(codes)
            digest = hashlib.sha256(f"streaming-v1:{c['target_id']}".encode()).digest()
            chosen = int.from_bytes(digest[:8], "little") < self.streaming_fraction * 2**64
            sample["streaming_eligible"] = eligible
            sample["streaming"] = eligible and chosen
            sample["streaming_pause"] = (
                self._streaming_pause(digest, len(sample["text_ids"]), len(codes))
                if eligible
                else (0, 0)
            )
        return sample

    def _streaming_pause(self, digest, text_length, frames):
        """(position, length) of one tts_pad run inside text_ids[1:], or (0, 0).

        Only the text is delayed; the audio keeps talking. At an even speaking rate r tokens per
        frame, token k is spoken near frame k / r but arrives at frame k + length, so a pause at
        position j is safe only if j >= length * r / (1 - r). A 1.5x margin covers locally faster
        speech; pauses that cannot meet it shrink or are dropped, and the padded text still keeps
        the lead that eligibility requires.
        """
        draw = [int.from_bytes(digest[i : i + 8], "little") for i in (8, 16, 24)]
        streamed = text_length - 1  # text_ids[1:], eos included
        rate = text_length / frames
        if rate >= 1 or draw[0] >= self.streaming_pause_probability * 2**64:
            return 0, 0
        margin = 1.5 * rate / (1 - rate)
        longest = min(
            self.streaming_max_pause_frames,
            int(self.streaming_max_text_ratio * frames) - text_length,
            int((streamed - 1) / margin),  # the run must start before eos
        )
        if longest < 1:
            return 0, 0
        length = 1 + draw[2] % longest
        earliest = max(1, math.ceil(margin * length))
        return earliest + draw[1] % (streamed - earliest), length

    def _language_codec_id(self, target_id, language):
        if language not in self.language_codec_ids:
            raise ValueError(f"No codec language mapping for {language!r}")
        # A fixed hash of the target decides Auto, so resume, workers and validation agree.
        digest = hashlib.sha256(f"language-auto-v1:{target_id}".encode()).digest()
        if int.from_bytes(digest[:8], "little") < self.language_auto_fraction * 2**64:
            return -1
        return self.language_codec_ids[language]
