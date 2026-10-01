import math

import torch


def collate(rows):
    skipped = sum(row is None for row in rows)
    rows = [row for row in rows if row is not None]
    if not rows:
        return {"skipped_samples": torch.tensor(skipped)}
    text_lengths = torch.tensor([len(r["text_ids"]) for r in rows])
    if (text_lengths == 0).any():
        raise ValueError("Text token sequence cannot be empty")
    batch = dict(
        text_ids=torch.cat([torch.tensor(r["text_ids"], dtype=torch.long) for r in rows]),
        codes=torch.cat([r["codes"] for r in rows]),
        text_lengths=text_lengths,
        frame_lengths=torch.tensor([len(r["codes"]) for r in rows]),
    )
    if "speaker_mels" in rows[0]:
        batch["speaker_lengths"] = torch.tensor([len(r["speaker_mels"]) for r in rows])
        batch["speaker_mels"] = torch.cat([r["speaker_mels"] for r in rows])
    if "speaker_embedding" in rows[0]:
        batch["speaker_embeddings"] = torch.stack([r["speaker_embedding"] for r in rows])
    if "duration" in rows[0]:
        batch["audio_seconds"] = torch.tensor([r["duration"] for r in rows], dtype=torch.float64)
    batch["skipped_samples"] = torch.tensor(skipped)
    return batch


def val_batches(dataset, batch_size, world_size, rank):
    # Pad rank workloads with a duplicate marked weight=0, so all FSDP ranks call
    # forward equally often while every real validation utterance is counted once.
    global_batch = batch_size * world_size
    for start in range(0, math.ceil(len(dataset) / global_batch) * global_batch, global_batch):
        indices = list(
            range(start + rank * batch_size, min(start + (rank + 1) * batch_size, len(dataset)))
        )
        yield (indices, True) if indices else ([0], False)
