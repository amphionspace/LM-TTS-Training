import unittest

from accelerate import skip_first_batches
from torch.utils.data import DataLoader

from qwen3_train.data import DistributedTokenBatchSampler


class DynamicBatchingTests(unittest.TestCase):
    def test_duration_balance_covers_both_languages_and_repeats_the_smaller_pool(self):
        languages = ['en'] * 12 + ['zh'] * 3
        durations = [1., 2., 3.] * 4 + [1., 2., 3.]
        sampler = DistributedTokenBatchSampler([int(d) for d in durations], [int(d) + 2 for d in durations],
            15, 30, world_size=1, rank=0, seed=42, languages=languages, durations=durations)
        previous = None
        for epoch in (0, 1):
            sampler.set_epoch(epoch)
            seen = [i for batch in sampler for i in batch]
            self.assertEqual(set(seen), set(range(len(languages))))
            self.assertEqual(sum(languages[i] == 'en' for i in seen), 12)
            self.assertGreater(sum(languages[i] == 'zh' for i in seen), 3)
            hours = {lang: sum(durations[i] for i in seen if languages[i] == lang) for lang in ('en', 'zh')}
            self.assertLessEqual(abs(hours['en'] - hours['zh']), max(durations))
            if previous is not None:
                self.assertNotEqual(previous, seen)
            previous = seen

    def test_duration_balance_keeps_distributed_batches_and_resume_aligned(self):
        frames = [1, 5, 3, 8, 2] * 20
        languages = ['en'] * 80 + ['zh'] * 20
        samplers = [DistributedTokenBatchSampler(frames, [f + 3 for f in frames], 25, 40,
            world_size=4, rank=rank, seed=7, languages=languages, durations=frames) for rank in range(4)]
        plans = [list(s) for s in samplers]
        self.assertEqual(len({len(p) for p in plans}), 1)
        for plan in plans:
            for batch in plan:
                self.assertLessEqual(sum(frames[i] for i in batch), 25)
                self.assertLessEqual(sum(frames[i] + 3 for i in batch), 40)
        restored = DistributedTokenBatchSampler(frames, [f + 3 for f in frames], 25, 40,
            world_size=4, rank=2, seed=7, languages=languages, durations=frames)
        restored.set_epoch(0, max_batches=8)
        actual = [b.tolist() for b in skip_first_batches(DataLoader(list(range(100)), batch_sampler=restored), 3)]
        self.assertEqual(actual, plans[2][3:8])

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
