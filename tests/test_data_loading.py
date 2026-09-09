import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from qwen3_train.data import CodeDataset, DistributedTokenBatchSampler, collate


class DataLoadingTests(unittest.TestCase):
    def test_inaccurate_manifest_cannot_understate_batch_frame_cost(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.savez(root / 'codes.npz', codes=np.zeros((4, 16), dtype=np.int16))
            manifest = root / 'train.jsonl'
            manifest.write_text(json.dumps({'id': 'underreported', 'text_ids': [1],
                                            'codes': 'codes.npz', 'num_frames': 2}))
            with self.assertRaisesRegex(ValueError, 'frame count disagrees'):
                CodeDataset(manifest)[0]

    def test_spawned_workers_preserve_packed_audio_and_text_across_epochs(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            for index in range(8):
                codes = root / f'{index}.npz'
                np.savez(codes, codes=np.full((index + 1, 16), index, dtype=np.int16))
                audio = root / f'{index}.wav'
                waveform = np.sin(np.arange(12000 + index * 256) * 0.04).astype(np.float32) * 0.1
                sf.write(audio, waveform, 24000, subtype='FLOAT')
                rows.append({'id': f'样本-{index}', 'text_ids': [index + 10] * (index % 3 + 1),
                             'codes': codes.name, 'codes_sha256': hashlib.sha256(codes.read_bytes()).hexdigest(),
                             'audio': str(audio), 'num_frames': index + 1})
            manifest = root / 'train.jsonl'
            manifest.write_text('\n' + '\n\n'.join(json.dumps(row, ensure_ascii=False) for row in rows),
                                encoding='utf-8')
            dataset = CodeDataset(manifest)
            dataset.target_speaker = True
            dataset.text_special_tokens = (100, 101)
            self.assertEqual(dataset.fingerprint, hashlib.sha256(manifest.read_bytes()).hexdigest())
            sampler = DistributedTokenBatchSampler([r['num_frames'] for r in rows],
                [r['num_frames'] + len(r['text_ids']) + 2 for r in rows], 15, 24, world_size=2, rank=1, seed=42)
            loader = DataLoader(dataset, batch_sampler=sampler,
                                collate_fn=collate, num_workers=2, prefetch_factor=2,
                                persistent_workers=True, multiprocessing_context='spawn',
                                generator=torch.Generator().manual_seed(43))
            try:
                for epoch, limit in ((3, None), (4, None), (5, 1)):
                    sampler.set_epoch(epoch, max_batches=limit)
                    indices = list(sampler)
                    actual = list(loader)
                    expected = [collate([dataset[i] for i in batch]) for batch in indices]
                    self.assertEqual(len(actual), len(expected))
                    for batch, reference in zip(actual, expected):
                        self.assertEqual(batch.keys(), reference.keys())
                        for key in reference:
                            torch.testing.assert_close(batch[key], reference[key], rtol=0, atol=0)
                        for text in batch['text_ids'].split(batch['text_lengths'].tolist()):
                            self.assertEqual((text[0].item(), text[-1].item()), (100, 101))
            finally:
                del loader


if __name__ == '__main__':
    unittest.main()
