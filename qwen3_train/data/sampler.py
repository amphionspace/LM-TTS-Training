"""Streaming, bounded-memory distributed token batching."""

import numpy as np


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
        self.window = shuffle_window
        self.epoch, self.start_batch, self.max_batches = 0, 0, None

    def set_epoch(self, epoch, *, start_batch=0, max_batches=None):
        self.epoch, self.start_batch, self.max_batches = epoch, start_batch, max_batches

    def order(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        windows = rng.permutation((len(self.frames) + self.window - 1) // self.window)
        for window in windows:
            start = int(window) * self.window
            # Seed by logical window identity, independent of worker count.
            local = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch, int(window), 1])
            )
            yield from (
                start + int(i)
                for i in local.permutation(min(self.window, len(self.frames) - start))
            )

    def __iter__(self):
        order = iter(self.order())
        pending = next(order, None)
        batch_number, yielded = 0, 0
        while pending is not None:
            group, loads = [], []
            for _ in range(self.world_size):
                if pending is None:
                    return
                f, t = int(self.frames[pending]), int(self.tokens[pending])
                if not 0 < f <= self.max_frames or not 0 < t <= self.max_tokens:
                    raise ValueError(f"Sample {pending} exceeds frame/token budgets: {f}/{t}")
                group.append([pending])
                loads.append([f, t])
                pending = next(order, None)
            while pending is not None:
                f, t = int(self.frames[pending]), int(self.tokens[pending])
                if not 0 < f <= self.max_frames or not 0 < t <= self.max_tokens:
                    raise ValueError(f"Sample {pending} exceeds frame/token budgets: {f}/{t}")
                fits = [
                    r
                    for r, (lf, lt) in enumerate(loads)
                    if lf + f <= self.max_frames and lt + t <= self.max_tokens
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
                pending = next(order, None)
            if batch_number >= self.start_batch:
                if self.max_batches is not None and yielded >= self.max_batches:
                    return
                yield group[self.rank]
                yielded += 1
            batch_number += 1
