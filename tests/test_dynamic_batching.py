import unittest

from accelerate import skip_first_batches
from torch.utils.data import DataLoader

from qwen3_train.data import DistributedTokenBatchSampler


class DynamicBatchingTests(unittest.TestCase):
    def test_budgets_rank_alignment_coverage_and_resume(self):
        frames = [1, 17, 4, 13, 2, 11, 7] * 11
        tokens = [f + 2 + (i % 5) * 3 for i, f in enumerate(frames)]
        samplers = [DistributedTokenBatchSampler(frames, tokens, 40, 65,
                    world_size=4, rank=r, seed=42) for r in range(4)]
        previous = None
        for epoch in (0, 1):
            for sampler in samplers:
                sampler.set_epoch(epoch)
            plans = [list(s) for s in samplers]
            self.assertEqual(len({len(plan) for plan in plans}), 1)
            seen = []
            sizes = set()
            for plan in plans:
                for batch in plan:
                    self.assertTrue(batch)
                    self.assertLessEqual(sum(frames[i] for i in batch), 40)
                    self.assertLessEqual(sum(tokens[i] for i in batch), 65)
                    seen.extend(batch)
                    sizes.add(len(batch))
            self.assertEqual(len(seen), len(set(seen)))
            self.assertLess(len(frames) - len(seen), 4)
            self.assertGreater(len(sizes), 1)
            if previous is not None:
                self.assertNotEqual(plans, previous)
            previous = plans
            # Restoring the epoch + batch cursor must retain variable batch boundaries.
            restored = DistributedTokenBatchSampler(frames, tokens, 40, 65,
                        world_size=4, rank=2, seed=42)
            restored.set_epoch(epoch)
            loader = DataLoader(list(range(len(frames))), batch_sampler=restored)
            actual = [batch.tolist() for batch in skip_first_batches(loader, 2)]
            self.assertEqual(actual, plans[2][2:])
            self.assertEqual(len(restored), len(plans[2]))
            restored.set_epoch(epoch, max_batches=3)
            self.assertEqual(len(restored), 3)
            self.assertEqual([batch.tolist() for batch in skip_first_batches(loader, 2)], plans[2][2:3])
            restored.set_epoch(epoch)
            self.assertEqual(list(restored), plans[2])

    def test_exact_limit_and_small_distributed_tail(self):
        plans = [list(DistributedTokenBatchSampler([10] * 11, [20] * 11, 10, 20,
                      world_size=4, rank=r, seed=1)) for r in range(4)]
        self.assertEqual([len(plan) for plan in plans], [2] * 4)
        self.assertEqual(len({i for p in plans for b in p for i in b}), 8)
        with self.assertRaisesRegex(ValueError, 'Sample 0'):
            DistributedTokenBatchSampler([11], [20], 10, 20, world_size=1, rank=0, seed=1)
        with self.assertRaisesRegex(ValueError, 'Sample 0'):
            DistributedTokenBatchSampler([10], [21], 10, 20, world_size=1, rank=0, seed=1)
        with self.assertRaisesRegex(ValueError, 'at least one'):
            DistributedTokenBatchSampler([1], [2], 10, 20, world_size=2, rank=0, seed=1)


if __name__ == '__main__':
    unittest.main()
