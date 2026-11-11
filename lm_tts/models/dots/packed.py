"""Packed DiT positions and block-sparse prefix/history/own-chunk attention."""

import torch
from torch.nn.attention.flex_attention import BlockMask

CHUNK = 5  # One hidden state followed by four latent frames.
BLOCK = 128


def packed_dit_mask(lengths, device):
    positions, samples, generations = [], [], []
    total = 2 * CHUNK * sum(lengths)
    blocks = (total + BLOCK - 1) // BLOCK
    # Only block-level candidates: no total_tokens squared mask on CUDA.
    candidates = [set() for _ in range(blocks)]
    offset = 0
    for sample, patches in enumerate(lengths):
        half = CHUNK * patches
        positions.extend(list(range(half)) * 2)
        samples.extend([sample] * (2 * half))
        generations.extend([False] * half + [True] * half)
        for query in range(offset // BLOCK, (offset + 2 * half - 1) // BLOCK + 1):
            lo = max(query * BLOCK, offset) - offset
            hi = min((query + 1) * BLOCK, offset + 2 * half) - offset
            if lo < half:
                end = min(hi, half)
                candidates[query].update(range(offset // BLOCK, (offset + end - 1) // BLOCK + 1))
            if hi > half:
                first = max(lo - half, 0) // CHUNK
                last = (hi - half - 1) // CHUNK
                if last:
                    candidates[query].update(
                        range(offset // BLOCK, (offset + CHUNK * last - 1) // BLOCK + 1)
                    )
                candidates[query].update(
                    range(
                        (offset + half + CHUNK * first) // BLOCK,
                        (offset + half + CHUNK * (last + 1) - 1) // BLOCK + 1,
                    )
                )
        offset += 2 * half
    positions = torch.tensor(positions, device=device)
    sample_ids = torch.tensor(samples, device=device)
    generation = torch.tensor(generations, dtype=torch.bool, device=device)

    def allowed(b, h, q, k):
        # Edge tiles can extend beyond the sequence end.
        qi, ki = q.clamp(max=total - 1), k.clamp(max=total - 1)
        qpos, kpos = positions[qi], positions[ki]
        qgen, kgen = generation[qi], generation[ki]
        prefix = ~qgen & ~kgen & (kpos <= qpos)
        history = qgen & ~kgen & (kpos < CHUNK * (qpos // CHUNK))
        own = qgen & kgen & (kpos // CHUNK == qpos // CHUNK)
        return (
            (q < total)
            & (k < total)
            & (sample_ids[qi] == sample_ids[ki])
            & (prefix | history | own)
        )

    if torch.device(device).type == "cpu":
        # Small CPU model tests use SDPA; production CUDA uses sparse blocks.
        idx = torch.arange(total, device=device)
        return allowed(None, None, idx[:, None], idx[None])[None], positions, sample_ids
    counts = torch.tensor([len(row) for row in candidates], dtype=torch.int32)
    indices = torch.zeros((blocks, blocks), dtype=torch.int32)
    for row, keys in enumerate(candidates):
        indices[row, : len(keys)] = torch.tensor(sorted(keys), dtype=torch.int32)
    mask = BlockMask.from_kv_blocks(
        counts[None, None].to(device),
        indices[None, None].to(device),
        BLOCK_SIZE=BLOCK,
        mask_mod=allowed,
        seq_lengths=(total, total),
    )
    return mask, positions, sample_ids
