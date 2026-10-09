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

    def test_invalid_sample_fails_and_empty_rank_is_never_padded(self):
        sampler = TokenBatchSampler([41, 1], [50, 11], 40, 100, world_size=2, rank=0, seed=1)
        with self.assertRaises(ValueError):
            list(sampler)
        samplers = [
            TokenBatchSampler([1] * 5, [11] * 5, 1, 11, world_size=2, rank=r, seed=1)
            for r in range(2)
        ]
        self.assertEqual([len(list(s)) for s in samplers], [2, 2])
