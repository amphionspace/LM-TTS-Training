"""Stateless keyed permutations of [0, n) for epoch-wide uniform sample order."""

import numpy as np

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


def _splitmix64(x):
    x = x + _GOLDEN
    x = (x ^ (x >> np.uint64(30))) * _MIX1
    x = (x ^ (x >> np.uint64(27))) * _MIX2
    return x ^ (x >> np.uint64(31))


class FeistelPermutation:
    """A bijection of [0, n), computed per position without storing the permutation.

    A balanced Feistel network permutes [0, 2^(2h)) ⊇ [0, n); cycle-walking re-applies it to
    values >= n, which keeps it a bijection of [0, n). Round keys come from
    SeedSequence.generate_state, and the round function is integer arithmetic, so the order
    does not depend on NumPy's random generator streams.
    """

    def __init__(self, n, key, rounds=6):
        if type(n) is not int or n < 1:
            raise ValueError("Permutation size must be a positive integer")
        bits = max(2, (n - 1).bit_length())
        self.half = (bits + 1) // 2
        self.n = n
        self.mask = np.uint64((1 << self.half) - 1)
        self.keys = np.random.SeedSequence(key).generate_state(rounds, dtype=np.uint64)

    def _encrypt(self, x):
        shift = np.uint64(self.half)
        left, right = x >> shift, x & self.mask
        for key in self.keys:
            left, right = right, left ^ (_splitmix64(right ^ key) & self.mask)
        return (left << shift) | right

    def __call__(self, positions):
        x = np.asarray(positions, dtype=np.uint64)
        if x.size and int(x.max()) >= self.n:
            raise ValueError("Permutation position out of range")
        x = self._encrypt(x)
        outside = x >= np.uint64(self.n)
        while outside.any():
            x[outside] = self._encrypt(x[outside])
            outside = x >= np.uint64(self.n)
        return x.astype(np.int64)
