import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch


class CodeDataset:
    def __init__(self, manifest):
        self.references = None
        self.path = Path(manifest).resolve()
        self.fingerprint = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"Empty manifest: {manifest}")

    def set_speaker_references(self, pool, seconds):
        if not 0.1 <= seconds <= 30:
            raise ValueError("speaker_reference_seconds must be between 0.1 and 30")
        by_speaker = {}
        for row in pool.rows:
            if row.get("speaker"):
                by_speaker.setdefault(row["speaker"], []).append(row)
        self.references = []
        for row in self.rows:
            candidates = [r for r in by_speaker.get(row.get("speaker"), [])
                          if r["id"] != row["id"] and r["audio"] != row["audio"]]
            if not candidates:
                raise ValueError(f"Need another training utterance with the same speaker for {row['id']}")
            index = int(hashlib.sha256(row["id"].encode()).hexdigest(), 16) % len(candidates)
            self.references.append(str(Path(candidates[index]["audio"]).resolve()))
        self.reference_seconds = seconds
        # Reference audio is part of the model input and therefore part of the
        # exact-resume identity, even though codec features were cached earlier.
        hashes = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                  for path in sorted(set(self.references))}
        self.reference_fingerprint = hashlib.sha256(json.dumps(
            {"assignments": self.references, "audio_sha256": hashes}, sort_keys=True).encode()).hexdigest()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        code_path = self.path.parent / row["codes"]
        if row.get("codes_sha256") and hashlib.sha256(code_path.read_bytes()).hexdigest() != row["codes_sha256"]:
            raise ValueError(f"Codec data changed: {code_path}")
        with np.load(code_path, allow_pickle=False) as f:
            codes = torch.from_numpy(f["codes"].astype(np.int64))
        if codes.ndim != 2 or codes.shape[1] != 16 or not len(codes):
            raise ValueError(f"Invalid codec shape for {row['id']}: {codes.shape}")
        if codes.min() < 0 or codes.max() >= 2048:
            raise ValueError(f"Invalid codec token for {row['id']}")
        item = {**row, "codes": codes}
        if self.references is not None:
            from .speaker import reference_mel
            item["reference_audio"] = self.references[index]
            item["speaker_mels"] = reference_mel(self.references[index], self.reference_seconds)
        return item


def collate(rows):
    b = len(rows)
    text_len = max(len(r["text_ids"]) for r in rows)
    frame_len = max(len(r["codes"]) for r in rows)
    ids = torch.zeros(b, text_len, dtype=torch.long)
    text_mask = torch.zeros_like(ids, dtype=torch.bool)
    codes = torch.zeros(b, frame_len, 16, dtype=torch.long)
    frame_mask = torch.zeros(b, frame_len, dtype=torch.bool)
    for i, row in enumerate(rows):
        n, t = len(row["text_ids"]), len(row["codes"])
        if n == 0:
            raise ValueError("Text token sequence cannot be empty")
        ids[i, :n] = torch.tensor(row["text_ids"])
        text_mask[i, :n] = True
        codes[i, :t] = row["codes"]
        frame_mask[i, :t] = True
    batch = dict(text_ids=ids, text_mask=text_mask, codes=codes, frame_mask=frame_mask)
    if "speaker_mels" in rows[0]:
        batch["speaker_mels"] = torch.stack([r["speaker_mels"] for r in rows])
    return batch


def train_batches(dataset, batch_size, world_size, rank, seed, epoch):
    rng = random.Random(seed + epoch)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    global_batch = batch_size * world_size
    # Drop a random tail before length sorting, so the longest examples are not
    # systematically excluded in small datasets.
    indices = indices[:len(indices) // global_batch * global_batch]
    bucket = max(batch_size * world_size * 32, 1)
    ordered = []
    for start in range(0, len(indices), bucket):
        ordered.extend(sorted(indices[start:start + bucket], key=lambda i: dataset.rows[i]["num_frames"] + len(dataset.rows[i]["text_ids"])))
    global_batch = batch_size * world_size
    batches = [ordered[i:i + global_batch] for i in range(0, len(ordered) - global_batch + 1, global_batch)]
    rng.shuffle(batches)
    if not batches:
        raise ValueError("Training set is smaller than one global microbatch")
    return [b[rank * batch_size:(rank + 1) * batch_size] for b in batches]


def val_batches(dataset, batch_size, world_size, rank):
    # Pad rank workloads with a duplicate marked weight=0, so all FSDP ranks call
    # forward equally often while every real validation utterance is counted once.
    global_batch = batch_size * world_size
    for start in range(0, math.ceil(len(dataset) / global_batch) * global_batch, global_batch):
        indices = list(range(start + rank * batch_size, min(start + (rank + 1) * batch_size, len(dataset))))
        yield (indices, True) if indices else ([0], False)
