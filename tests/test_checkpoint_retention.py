from pathlib import Path
import tempfile
import unittest

from qwen3_train.checkpoint import prune_checkpoints


class CheckpointRetentionTests(unittest.TestCase):
    def test_pruning_preserves_latest_and_incomplete_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in range(1, 5):
                checkpoint = root / f'step-{step:08d}'
                checkpoint.mkdir()
                (checkpoint / 'COMPLETE').write_text('ok')
            incomplete = root / 'step-00000005.incomplete'
            incomplete.mkdir()
            (incomplete / 'partial').write_text('unfinished')
            (root / 'latest').write_text('step-00000004\n')
            self.assertEqual(prune_checkpoints(root, 2), ['step-00000001', 'step-00000002'])
            self.assertTrue((root / 'step-00000003/COMPLETE').exists())
            self.assertTrue((root / 'step-00000004/COMPLETE').exists())
            self.assertTrue((incomplete / 'partial').exists())
            (root / 'latest').write_text('step-00000003\n')
            self.assertEqual(prune_checkpoints(root, 1), ['step-00000004'])
            self.assertTrue((root / 'step-00000003/COMPLETE').exists())

    def test_invalid_retention_or_latest_cannot_delete_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'step-00000001'
            checkpoint.mkdir()
            (checkpoint / 'COMPLETE').write_text('ok')
            (root / 'latest').write_text('missing\n')
            for keep in [0, -1, 2]:
                with self.assertRaises(ValueError):
                    prune_checkpoints(root, keep)
                self.assertTrue(checkpoint.exists())


if __name__ == '__main__':
    unittest.main()
