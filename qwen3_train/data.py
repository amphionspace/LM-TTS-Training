import hashlib
import json
import math
from array import array
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler


class CodeDataset:
    def __init__(self, manifest):
        self.target_speaker = False
        self.text_special_tokens = None
        self.path = Path(manifest).resolve()
        self.rows = []
        self._offsets = array('Q')
        checksum = hashlib.sha256()
        offset = 0
        with self.path.open('rb') as stream:
            for line in stream:
                checksum.update(line)
                if line.strip():
                    self._offsets.append(offset)
                    self.rows.append(json.loads(line))
                offset += len(line)
        self.fingerprint = checksum.hexdigest()
        if not self.rows:
            raise ValueError(f"Empty manifest: {manifest}")

    def __getstate__(self):
        # Spawned workers read indexed rows instead of copying the full manifest.
        return {**self.__dict__, 'rows': None}

    def __len__(self):
        return len(self._offsets)

    def __getitem__(self, index):
        if self.rows is None:
            with self.path.open('rb') as stream:
                stream.seek(self._offsets[index])
                row = json.loads(stream.readline())
        else:
            row = self.rows[index]
        code_path = self.path.parent / row["codes"]
        if row.get("codes_sha256") and hashlib.sha256(code_path.read_bytes()).hexdigest() != row["codes_sha256"]:
            raise ValueError(f"Codec data changed: {code_path}")
        with np.load(code_path, allow_pickle=False) as f:
            codes = torch.from_numpy(f["codes"].astype(np.int64))
        if codes.ndim != 2 or codes.shape[1] != 16 or not len(codes):
            raise ValueError(f"Invalid codec shape for {row['id']}: {codes.shape}")
        if len(codes) != row['num_frames']:
            raise ValueError(f"Codec frame count disagrees with manifest for {row['id']}")
        if codes.min() < 0 or codes.max() >= 2048:
            raise ValueError(f"Invalid codec token for {row['id']}")
        item = {**row, "codes": codes}
        if self.text_special_tokens is not None:
            bos, eos = self.text_special_tokens
            item['text_ids'] = [bos, *row['text_ids'], eos]
        if self.target_speaker:
            from .speaker import audio_mel
            item['speaker_mels'] = audio_mel(row)
        return item


class DistributedTokenBatchSampler(Sampler):
    """Pack shuffled samples under per-rank frame/token budgets with equal steps.

    All ranks construct the same plan. At most world_size - 1 shuffled tail
    samples are dropped when they cannot seed a nonempty batch on every rank.
    """

    def __init__(self, frame_lengths, token_lengths, max_frames, max_tokens, *, world_size, rank, seed,
                 languages=None, durations=None):
        self.costs = list(zip(frame_lengths, token_lengths, strict=True))
        if max_frames < 1 or max_tokens < 1 or world_size < 1 or not 0 <= rank < world_size:
            raise ValueError('Batch budgets and world_size must be positive; rank must be in range')
        if len(self.costs) < world_size:
            raise ValueError('Training set needs at least one sample per rank')
        for index, (frames, tokens) in enumerate(self.costs):
            if not 0 < frames <= max_frames or not 0 < tokens <= max_tokens:
                raise ValueError(f'Sample {index} exceeds batch budgets or has invalid lengths: '
                                 f'{frames} frames / {tokens} tokens; limits {max_frames} / {max_tokens}')
        self.max_frames, self.max_tokens = max_frames, max_tokens
        self.world_size, self.rank, self.seed = world_size, rank, seed
        self.languages, self.durations = languages, durations
        if languages is not None:
            if (len(languages) != len(self.costs) or durations is None or len(durations) != len(self.costs)
                    or set(languages) != {'en', 'zh'} or any(not math.isfinite(d) or d <= 0 for d in durations)):
                raise ValueError('Language balancing requires English/Chinese labels and positive audio durations for every sample')
        self.epoch = 0
        self.max_batches = None
        self._batches = None

    def set_epoch(self, epoch, *, max_batches=None):
        self.max_batches = max_batches
        if epoch != self.epoch:
            self.epoch = epoch
            self._batches = None

    def _plan(self):
        if self._batches is not None:
            return self._batches
        order = torch.randperm(len(self.costs), generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        if self.languages is not None:
            queues = {language: [i for i in order if self.languages[i] == language] for language in ('en', 'zh')}
            target = max(sum(self.durations[i] for i in queue) for queue in queues.values())
            elapsed = dict.fromkeys(queues, 0.)
            cursors = dict.fromkeys(queues, 0)
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = []
            # Cover the larger language once and repeat the smaller language to
            # equalize audio exposure without discarding its counterpart's data.
            while min(elapsed.values()) < target:
                language = min(elapsed, key=elapsed.get)
                queue = queues[language]
                if cursors[language] == len(queue):
                    queues[language] = queue = [queue[i] for i in torch.randperm(len(queue), generator=generator).tolist()]
                    cursors[language] = 0
                index = queue[cursors[language]]
                order.append(index)
                cursors[language] += 1
                elapsed[language] += self.durations[index]
        batches, cursor = [], 0
        while len(order) - cursor >= self.world_size:
            group = [[i] for i in order[cursor:cursor + self.world_size]]
            loads = [list(self.costs[batch[0]]) for batch in group]
            cursor += self.world_size
            while cursor < len(order):
                index = order[cursor]
                frames, tokens = self.costs[index]
                fits = [r for r, (f, t) in enumerate(loads)
                        if f + frames <= self.max_frames and t + tokens <= self.max_tokens]
                if not fits:
                    break
                rank = min(fits, key=lambda r: max(loads[r][0] / self.max_frames,
                                                   loads[r][1] / self.max_tokens))
                group[rank].append(index)
                loads[rank][0] += frames
                loads[rank][1] += tokens
                cursor += 1
            batches.append(group[self.rank])
        self._batches = batches
        return batches

    def __iter__(self):
        return iter(self._plan()[:self.max_batches])

    def __len__(self):
        length = len(self._plan())
        return length if self.max_batches is None else min(length, self.max_batches)


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


def val_batches(dataset, batch_size, world_size, rank):
    # Pad rank workloads with a duplicate marked weight=0, so all FSDP ranks call
    # forward equally often while every real validation utterance is counted once.
    global_batch = batch_size * world_size
    for start in range(0, math.ceil(len(dataset) / global_batch) * global_batch, global_batch):
        indices = list(range(start + rank * batch_size, min(start + (rank + 1) * batch_size, len(dataset))))
        yield (indices, True) if indices else ([0], False)
