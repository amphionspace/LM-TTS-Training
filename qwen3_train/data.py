import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch


class CodeDataset:
    def __init__(self, manifest):
        self.target_speaker = False
        self.path = Path(manifest).resolve()
        self.fingerprint = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"Empty manifest: {manifest}")

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
        if self.target_speaker:
            from .speaker import audio_mel
            item['speaker_mels'] = audio_mel(row)
        return item


def collate(rows):
    text_lengths = torch.tensor([len(r['text_ids']) for r in rows])
    if (text_lengths == 0).any():
        raise ValueError("Text token sequence cannot be empty")
    batch = dict(text_ids=torch.cat([torch.tensor(r['text_ids'], dtype=torch.long) for r in rows]),
                 codes=torch.cat([r['codes'] for r in rows]), text_lengths=text_lengths,
                 frame_lengths=torch.tensor([len(r['codes']) for r in rows]))
    if 'speaker_mels' in rows[0]:
        batch['speaker_lengths'] = torch.tensor([len(r['speaker_mels']) for r in rows])
        batch['speaker_mels'] = torch.cat([r['speaker_mels'] for r in rows])
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
