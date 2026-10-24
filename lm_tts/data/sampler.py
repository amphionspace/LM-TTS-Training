"""Streaming, bounded-memory distributed token batching."""

import numpy as np

from .permutation import FeistelPermutation

# window: window-shuffled contiguous index ranges (optionally length-sorted), kept for old runs.
# feistel: a keyed uniform permutation of the whole epoch, so every batch is a uniform sample.
ORDERS = {"window", "feistel"}


class TokenBatchSampler:
    """Bounded window shuffling and streaming global-rank batch assignment.

    Every rank reconstructs the same sequence; only local batches are yielded.
    Recovery skips metadata only, never feature payloads or audio decoding.
    """

    def __init__(
        self,
        frames,
        tokens,
        max_frames,
        max_tokens,
        *,
        world_size,
        rank,
        seed,
        shuffle_window=65536,
        length_bucket_size=0,
        pad_to_longest=False,
        order="window",
    ):
        if len(frames) != len(tokens) or len(frames) < world_size:
            raise ValueError("Sampling metadata must have at least one sample per rank")
        if (
            min(max_frames, max_tokens, world_size, shuffle_window) < 1
            or not 0 <= rank < world_size
        ):
            raise ValueError("Invalid token budgets or distributed rank")
        self.frames, self.tokens = frames, tokens
        self.max_frames, self.max_tokens = max_frames, max_tokens
        self.world_size, self.rank, self.seed = world_size, rank, seed
        if type(length_bucket_size) is not int or length_bucket_size < 0:
            raise ValueError("length_bucket_size must be a nonnegative integer")
        self.length_bucket_size = length_bucket_size
        if order not in ORDERS or (order == "feistel" and length_bucket_size):
            raise ValueError("Sampler order must be window, or feistel without length buckets")
        self.order_kind = order
        self.pad_to_longest = pad_to_longest
        self.window = shuffle_window
        self.epoch, self.start_batch, self.start_sample, self.max_batches = 0, 0, None, None

    def set_epoch(self, epoch, *, start_batch=0, start_sample=None, max_batches=None):
        """start_sample, when known, is the epoch-order position where batch start_batch begins
        (the global count of samples consumed before it); resume then skips by position instead
        of replaying the packing of every earlier batch."""
        self.epoch, self.start_batch, self.max_batches = epoch, start_batch, max_batches
        self.start_sample = start_sample

    def order(self, skip=0):
        if self.order_kind == "feistel":
            permutation = FeistelPermutation(len(self.frames), [self.seed, self.epoch, 3])
            for start in range(skip, len(self.frames), 1 << 20):
                stop = min(len(self.frames), start + (1 << 20))
                yield from permutation(np.arange(start, stop)).tolist()
            return
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        windows = rng.permutation((len(self.frames) + self.window - 1) // self.window)
        for window in windows:
            start = int(window) * self.window
            size = min(self.window, len(self.frames) - start)
            if skip >= size:
                skip -= size
                continue
            # Seed by logical window identity, independent of worker count.
            local = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch, int(window), 1])
            )
            indices = start + local.permutation(size)
            if self.length_bucket_size:
                # Randomized local buckets reduce SDPA padding without changing epoch coverage.
                ordered = []
                for offset in range(0, len(indices), self.length_bucket_size):
                    bucket = indices[offset : offset + self.length_bucket_size]
                    ordered += sorted(map(int, bucket), key=lambda i: self.tokens[i])
                yield from ordered[skip:]
            else:
                yield from map(int, indices[skip:])
            skip = 0

    def __iter__(self):
        # Packing restarts its loads at every batch, so a batch depends only on where it starts.
        skip = self.start_sample is not None
        order = iter(self.order(self.start_sample if skip else 0))
        pending = next(order, None)
        batch_number, yielded = (self.start_batch if skip else 0), 0
        while pending is not None:
            group, loads = [], []
            for _ in range(self.world_size):
                if pending is None:
                    return
                f, t = int(self.frames[pending]), int(self.tokens[pending])
                if not 0 < f <= self.max_frames or not 0 < t <= self.max_tokens:
                    raise ValueError(f"Sample {pending} exceeds frame/token budgets: {f}/{t}")
                group.append([pending])
                loads.append([f, t, t])
                pending = next(order, None)
            while pending is not None:
                f, t = int(self.frames[pending]), int(self.tokens[pending])
                if not 0 < f <= self.max_frames or not 0 < t <= self.max_tokens:
                    raise ValueError(f"Sample {pending} exceeds frame/token budgets: {f}/{t}")
                fits = [
                    r
                    for r, (lf, lt, longest) in enumerate(loads)
                    if lf + f <= self.max_frames
                    and lt + t <= self.max_tokens
                    and (
                        not self.pad_to_longest
                        or (len(group[r]) + 1) * max(longest, t) <= self.max_tokens
                    )
                ]
                if not fits:
                    break
                rank = min(
                    fits,
                    key=lambda r: max(loads[r][0] / self.max_frames, loads[r][1] / self.max_tokens),
                )
                group[rank].append(pending)
                loads[rank][0] += f
                loads[rank][1] += t
                loads[rank][2] = max(loads[rank][2], t)
                pending = next(order, None)
            if batch_number >= self.start_batch:
                if self.max_batches is not None and yielded >= self.max_batches:
                    return
                yield group[self.rank]
                yielded += 1
            batch_number += 1
