"""Batched fixed-snapshot feature reads with mmap sampling metadata."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..artifacts import file_hash
from .build import open_snapshot, read_complete


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
    def __init__(self, build_path, text_special_tokens, *, verify_integrity=True):
        self.path = Path(build_path).resolve()
        self.manifest = read_complete(self.path)
        if self.manifest.get("artifact_kind") != "training_build":
            raise ValueError("Expected a published training build")
        if self.manifest["speaker_conditioning_mode"] != "frozen_embedding":
            raise ValueError("FeatureDataset requires frozen_embedding")
        self.fingerprint = file_hash(self.path)
        self.text_special_tokens = text_special_tokens
        self.bindings = self.manifest["bindings"]
        self.indices, lengths = [], []
        for binding in self.bindings:
            path = self.path.parent / binding["sampling_index"]
            if verify_integrity and file_hash(path) != binding["sampling_sha256"]:
                raise ValueError("Sampling metadata changed")
            self.indices.append(np.load(path, mmap_mode="r")[: binding["ready_rows"]])
            lengths.append(binding["ready_rows"])
            for kind in ("codec", "speaker"):
                ref = binding[kind]
                if verify_integrity and file_hash(ref["manifest_path"]) != ref["manifest_sha256"]:
                    raise ValueError("Feature manifest changed")
        self.offsets = np.cumsum([0, *lengths])
        if not self.offsets[-1]:
            raise ValueError("Empty training build")
        self.tables = {}
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
        }

    def __setstate__(self, state):
        # Parent validated large indices. Recheck the small manifest in workers.
        self.__init__(state["path"], state["text_special_tokens"], verify_integrity=False)
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
        for slot, group in groups.items():
            positions, locators = zip(*group)
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
            if any(not r["build_ready"] or r["speaker_row"] is None for r in codec):
                raise ValueError("Sampling index selected an unready row")
            speaker = (
                self.table(slot, "speaker")
                .take(
                    [r["speaker_row"] for r in codec],
                    columns=[
                        "target_id",
                        "parent_sample_id",
                        "profile_id",
                        "audio_sha256",
                        "start_frame",
                        "end_frame",
                        "native_sample_rate",
                        "status",
                        "embedding",
                    ],
                )
                .to_pylist()
            )
            for position, c, s in zip(positions, codec, speaker, strict=True):
                keys = (
                    "target_id",
                    "parent_sample_id",
                    "audio_sha256",
                    "start_frame",
                    "end_frame",
                    "native_sample_rate",
                )
                if any(c[k] != s[k] for k in keys) or c["status"] != "ok" or s["status"] != "ok":
                    raise ValueError("Cached speaker locator has the wrong identity/interval")
                binding = self.bindings[slot]
                if (
                    c["profile_id"] != binding["codec"]["profile_id"]
                    or s["profile_id"] != binding["speaker"]["profile_id"]
                ):
                    raise ValueError("Feature profile differs from build binding")
                codes = torch.tensor(c["codes"], dtype=torch.long)
                vector = torch.tensor(s["embedding"], dtype=torch.float32)
                if (
                    codes.shape != (c["num_codec_frames"], 16)
                    or not len(codes)
                    or codes.min() < 0
                    or codes.max() >= 2048
                ):
                    raise ValueError("Invalid codec shape or vocabulary")
                if vector.ndim != 1 or not torch.isfinite(vector).all() or vector.norm() == 0:
                    raise ValueError("Invalid cached speaker embedding")
                bos, eos = self.text_special_tokens
                result[position] = {
                    "id": c["target_id"],
                    "text": c["text"],
                    "language": c["language"],
                    "speaker": c["speaker_id"],
                    "text_ids": [bos, *c["text_ids"], eos],
                    "codes": codes,
                    "speaker_embedding": vector,
                    "num_frames": len(codes),
                    "duration": (c["end_frame"] - c["start_frame"]) / c["native_sample_rate"],
                }
        return result
