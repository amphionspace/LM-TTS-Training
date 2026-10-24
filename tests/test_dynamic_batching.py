import unittest

from lm_tts.data.sampler import TokenBatchSampler


class DynamicBatchingTests(unittest.TestCase):
    def test_distributed_coverage_budgets_shuffle_and_resume(self):
        frames = [1, 17, 4, 13, 2, 11, 7] * 11
        tokens = [f + 10 + i % 5 for i, f in enumerate(frames)]
        samplers = [
            TokenBatchSampler(
                frames,
                tokens,
                40,
                100,
                world_size=4,
                rank=r,
                seed=42,
                shuffle_window=11,
                length_bucket_size=7,
                pad_to_longest=True,
            )
            for r in range(4)
        ]
        previous = None
        for epoch in (0, 1):
            for sampler in samplers:
                sampler.set_epoch(epoch)
            plans = [list(s) for s in samplers]
            self.assertEqual(len({len(p) for p in plans}), 1)
            seen = [i for plan in plans for batch in plan for i in batch]
            self.assertEqual(len(seen), len(set(seen)))
            self.assertLess(len(frames) - len(seen), 4)
            for plan in plans:
                for batch in plan:
                    self.assertLessEqual(sum(frames[i] for i in batch), 40)
                    self.assertLessEqual(sum(tokens[i] for i in batch), 100)
                    self.assertLessEqual(len(batch) * max(tokens[i] for i in batch), 100)
            if previous is not None:
                self.assertNotEqual(previous, plans)
            previous = plans
            samplers[2].set_epoch(epoch, start_batch=2, max_batches=3)
            self.assertEqual(list(samplers[2]), plans[2][2:5])

    def test_resume_by_sample_position_matches_replay(self):
        frames = [1, 17, 4, 13, 2, 11, 7, 39, 3] * 37
        tokens = [f + 10 + i % 5 for i, f in enumerate(frames)]
        for options in (
            dict(shuffle_window=11, length_bucket_size=7, pad_to_longest=True),
            dict(shuffle_window=64),
            dict(order="feistel"),
        ):
            samplers = [
                TokenBatchSampler(frames, tokens, 40, 100, world_size=4, rank=r, seed=7, **options)
                for r in range(4)
            ]
            for sampler in samplers:
                sampler.set_epoch(1)
            plans = [list(s) for s in samplers]
            consumed = 0
            # The tail that cannot fill every rank is dropped, as in a replayed epoch.
            for batch in range(len(plans[0]) + 1):
                for rank, sampler in enumerate(samplers):
                    sampler.set_epoch(1, start_batch=batch, start_sample=consumed, max_batches=3)
                    self.assertEqual(list(sampler), plans[rank][batch : batch + 3])
                if batch < len(plans[0]):
                    consumed += sum(len(plan[batch]) for plan in plans)

    def test_invalid_sample_fails_and_empty_rank_is_never_padded(self):
        sampler = TokenBatchSampler([41, 1], [50, 11], 40, 100, world_size=2, rank=0, seed=1)
        with self.assertRaises(ValueError):
            list(sampler)
        samplers = [
            TokenBatchSampler([1] * 5, [11] * 5, 1, 11, world_size=2, rank=r, seed=1)
            for r in range(2)
        ]
        self.assertEqual([len(list(s)) for s in samplers], [2, 2])


class UniformOrderTests(unittest.TestCase):
    def test_feistel_permutation_is_a_keyed_bijection(self):
        import numpy as np

        from lm_tts.data.permutation import FeistelPermutation

        for n in (1, 2, 3, 1000, 65537, 300000):
            order = FeistelPermutation(n, [42, 0, 3])(np.arange(n))
            self.assertTrue(np.array_equal(np.sort(order), np.arange(n)))
        a, b = (FeistelPermutation(65537, [42, e, 3])(np.arange(65537)) for e in (0, 1))
        self.assertFalse(np.array_equal(a, b))
        again = FeistelPermutation(65537, [42, 0, 3])(np.arange(65537))
        self.assertTrue(np.array_equal(a, again))

    def test_uniform_sampler_mixes_contiguous_datasets_in_every_batch(self):
        # Two "datasets" stored back to back, as bindings are concatenated in the build.
        frames = [5] * 3000 + [9] * 7000
        tokens = [f + 12 for f in frames]
        samplers = [
            TokenBatchSampler(
                frames,
                tokens,
                400,
                1000,
                world_size=4,
                rank=r,
                seed=42,
                order="feistel",
            )
            for r in range(4)
        ]
        plans = [list(s) for s in samplers]
        seen = [i for plan in plans for batch in plan for i in batch]
        self.assertEqual(len(seen), len(set(seen)))
        self.assertLess(len(frames) - len(seen), 4)
        steps = list(zip(*plans))
        shares = [sum(i < 3000 for b in step for i in b) / sum(map(len, step)) for step in steps]
        self.assertTrue(all(0.15 < s < 0.45 for s in shares))
        samplers[1].set_epoch(0, start_batch=3, max_batches=4)
        self.assertEqual(list(samplers[1]), plans[1][3:7])
        with self.assertRaises(ValueError):
            TokenBatchSampler(
                frames,
                tokens,
                400,
                1000,
                world_size=1,
                rank=0,
                seed=1,
                length_bucket_size=8,
                order="feistel",
            )
