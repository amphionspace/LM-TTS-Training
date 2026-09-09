import json
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qwen3_train.metrics import normalize
from scripts.prepare_emilia_streaming import export, finalize, take_ready_batch


class StreamingPreparationTests(unittest.TestCase):
    def test_ready_batch_can_pass_a_slow_long_carrier_without_changing_batch_membership(self):
        slow, fast = Future(), Future()
        fast.set_result('decoded')
        blocked = [('slow-clip', slow, True)]
        ready = [('first-clip', fast, True), ('second-clip', fast, True)]
        pending = deque([blocked, ready])
        self.assertIs(take_ready_batch(pending), ready)
        self.assertEqual(list(pending), [blocked])
        slow.set_result('decoded')
        self.assertIs(take_ready_batch(pending), blocked)
        self.assertFalse(pending)

    def test_export_resume_preserves_published_chunks_and_unrestricted_durations(self):
        rows = [{'id': str(index), 'language': lang, 'duration': duration}
                for index, (lang, duration) in enumerate([('en', .5), ('zh', .5), ('en', 31.5), ('zh', 31.5)])]

        def interrupted(*args, **kwargs):
            yield from rows[:3]
            raise OSError('interrupted source')

        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(output=directory, source=directory, hours_per_language=32 / 3600,
                                   chunk_records=2, export_workers=2, resume_export=False,
                                   include_long_shorts=False, allow_shortfall=False)
            with patch('qwen3_train.sources.emilia_short', interrupted):
                with self.assertRaises(OSError):
                    export(args)
            published = (root / 'raw/000000.jsonl').read_bytes()
            args.resume_export = True
            with patch('qwen3_train.sources.emilia_short', return_value=iter(rows)):
                export(args)
            self.assertEqual((root / 'raw/000000.jsonl').read_bytes(), published)
            actual = [json.loads(line) for path in sorted((root / 'raw').glob('*.jsonl'))
                      for line in path.read_text().splitlines()]
            self.assertEqual(actual, rows)
            self.assertTrue((root / 'EXPORT_COMPLETE.json').exists())

    def test_long_supplement_deduplicates_ids_keeps_carriers_together_and_reports_shortfall(self):
        rows = [{'id': f'{carrier}-{i}', 'language': 'en', 'duration': 1,
                 'audio': {'archive': 'source.tar', 'offset': carrier}}
                for carrier in (0, 100) for i in range(3)]
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(output=directory, source=directory, hours_per_language=5000,
                                   chunk_records=2, export_workers=2, resume_export=False,
                                   include_long_shorts=True, allow_shortfall=True)
            with patch('qwen3_train.sources.emilia_short', return_value=iter([*rows, rows[0]])):
                export(args)
            root = Path(directory)
            chunks = [[json.loads(line) for line in p.read_text().splitlines()]
                      for p in sorted((root / 'raw').glob('*.jsonl'))]
            self.assertEqual(chunks, [rows[:3], rows[3:]])
            report = json.loads((root / 'EXPORT_COMPLETE.json').read_text())
            self.assertEqual(report['records'], {'en': 6})
            self.assertEqual(report['duplicate_ids_skipped'], 1)
            self.assertFalse(any(report['target_reached'].values()))

    def make_chunks(self, root):
        (root / 'prepared').mkdir()
        rows = []
        for language in ('en', 'zh'):
            for index, text in enumerate(['Shared text!', f'{language} unique one', f'{language} unique two']):
                rows.append({'id': f'{language}-{index}', 'language': language,
                             'speaker': language, 'text': text, 'text_ids': [1] * (300 if index == 2 else 3),
                             'source': {'type': 'short'},
                             'duration': 31.5 if index == 2 else .5, 'num_frames': 394 if index == 2 else 7})
        for index, chunk in enumerate((rows[::2], rows[1::2])):
            (root / 'prepared' / f'{index:06d}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in chunk))
        (root / 'EXPORT_COMPLETE.json').write_text(json.dumps({'chunks': 2, 'records': {'en': 3, 'zh': 3}}))
        return rows

    def test_global_split_excludes_cross_chunk_text_duplicates_and_keeps_long_rows(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original = self.make_chunks(root)
            finalize(SimpleNamespace(output=directory, val_count=1))
            train = [json.loads(line) for line in (root / 'train.jsonl').read_text().splitlines()]
            val = [json.loads(line) for line in (root / 'val.jsonl').read_text().splitlines()]
            self.assertEqual({r['id'] for r in train + val}, {r['id'] for r in original})
            self.assertEqual(len(train) + len(val), len(original))
            self.assertFalse({normalize(r['text']) for r in train} & {normalize(r['text']) for r in val})
            self.assertEqual({r['language'] for r in val}, {'en', 'zh'})
            for row in val:
                self.assertGreaterEqual(sum(r['speaker'] == row['speaker'] for r in train), 2)
            for row in train + val:
                self.assertEqual(row, next(r for r in original if r['id'] == row['id']))
            self.assertTrue((root / 'PREPARATION_COMPLETE').exists())

    def test_missing_chunk_does_not_publish_complete_manifests(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_chunks(root)
            (root / 'prepared/000001.jsonl').unlink()
            with self.assertRaises(FileNotFoundError):
                finalize(SimpleNamespace(output=directory, val_count=1))
            self.assertFalse((root / 'PREPARATION_COMPLETE').exists())
            self.assertFalse((root / 'train.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
