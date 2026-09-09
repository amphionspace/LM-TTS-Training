import copy
import io
import numpy as np
import soundfile as sf
import json
from pathlib import Path
import tempfile
import unittest

from qwen3_train.sources import emilia_short, short_record, materialize_audio
from scripts.rescore_english import english_metrics


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.meta = {'id': 'u1', 'type': 'short', 'recording_id': 'recording1',
            'frames': 44100, 'sample_rate': 44100,
            'short': [{'text': 'Hello.', 'language': 'en', 'speaker': 'recording1_spk1',
                       'rel_start_samples': 0, 'rel_end_samples': 44100}]}

    def test_top_level_type_wins_over_filename_and_nested_short_views(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'data'
            data.mkdir()
            archive = data / 'fixture.tar'
            lines = []
            with archive.open('wb') as stream:
                # Every item contains a valid short view. Only type=short is eligible.
                for i, kind in enumerate(['long', 'dialogue', 'short']):
                    meta = {**self.meta, 'id': str(i), 'type': kind}
                    name = f'looks_short_{i}'
                    for suffix, payload in [('m4a', b'audio'), ('json', json.dumps(meta).encode())]:
                        lines.append(f'{name}.{suffix}\t{stream.tell()}\t{len(payload)}\n')
                        stream.write(payload)
            archive.with_suffix('.tar.idx').write_text(''.join(lines))
            rows = list(emilia_short(directory))
            self.assertEqual([r['id'] for r in rows], ['emilia2:2'])
            locator = rows[0]['audio']
            with archive.open('rb') as stream:
                stream.seek(locator['offset'])
                self.assertEqual(stream.read(locator['size']), b'audio')

    def test_reject_partial_slice_disguised_as_standalone_short(self):
        meta = copy.deepcopy(self.meta)
        meta['short'][0]['rel_start_samples'] = 100
        with self.assertRaisesRegex(ValueError, 'cover its carrier'):
            short_record(meta, 'file.tar', 'audio.m4a', 512, 100)

    def test_submillisecond_tail_padding_and_large_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = io.BytesIO()
            sf.write(payload, np.linspace(-.1, .1, 989, dtype=np.float32), 44100, format='WAV', subtype='FLOAT')
            archive = root / 'fixture.tar'
            archive.write_bytes(payload.getvalue())
            record = {'id': 'tail-fixture', 'audio': {'kind': 'tar_member',
                'archive': str(archive), 'offset': 0, 'size': len(payload.getvalue()),
                'sample_rate': 44100, 'frames': 1000}}
            with self.assertWarnsRegex(UserWarning, '11 samples'):
                path = materialize_audio(record, root / 'padded.wav')
            waveform, rate = sf.read(path)
            self.assertEqual(rate, 24000)
            self.assertEqual(len(waveform), (1000 * 24000 + 44099) // 44100)
            self.assertFalse(path.with_suffix('.incomplete').exists())
            record['audio']['frames'] = 1100
            with self.assertRaisesRegex(ValueError, 'tail-fixture'):
                materialize_audio(record, root / 'invalid.wav')
            self.assertFalse((root / 'invalid.wav').exists())

    def test_english_numbers_do_not_hide_actual_number_errors(self):
        reference = 'or eight inches about one hundred forty-five pounds'
        self.assertEqual(english_metrics(reference, 'or 8 inches about 145 pounds')['wer'], 0)
        self.assertGreater(english_metrics(reference, 'or 8 inches about 146 pounds')['wer'], 0)


if __name__ == '__main__':
    unittest.main()
