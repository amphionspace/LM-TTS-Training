import copy
import io
import numpy as np
import soundfile as sf
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qwen3_train import sources
from qwen3_train.sources import emilia_short, short_record, materialize_audio, decode_emilia_audio, decode_emilia_group
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

    def test_long_short_view_preserves_original_id_and_rejects_invalid_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'data'
            data.mkdir()
            archive = data / 'fixture.tar'
            item = {**self.meta['short'][0], 'id': 'original-utterance', 'rel_start_samples': 1234,
                    'rel_end_samples': 55555}
            meta = {**self.meta, 'id': 'carrier_L0', 'type': 'long', 'frames': 44100 * 60,
                    'short': [item, {**item, 'id': 'invalid', 'rel_start_samples': -882}]}
            payload = json.dumps(meta).encode()
            dialogue = json.dumps({**meta, 'id': 'carrier_D0', 'type': 'dialogue',
                                   'short': [{**item, 'id': 'dialogue-utterance'}]}).encode()
            dialogue_offset = len(payload) + 5
            archive.write_bytes(payload + b'audio' + dialogue + b'audio')
            archive.with_suffix('.tar.idx').write_text(
                f'carrier.json\t0\t{len(payload)}\ncarrier.m4a\t{len(payload)}\t5\n'
                f'dialogue.json\t{dialogue_offset}\t{len(dialogue)}\n'
                f'dialogue.m4a\t{dialogue_offset + len(dialogue)}\t5\n')
            with self.assertWarnsRegex(UserWarning, 'Invalid long short annotation'):
                rows = list(emilia_short(directory, include_long_shorts=True))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['id'], 'emilia2:original-utterance')
            self.assertEqual(rows[0]['audio']['frames'], 44100 * 60)
            self.assertEqual(rows[0]['audio']['start_frame'], 1234)
            self.assertEqual(rows[0]['audio']['end_frame'], 55555)
            self.assertAlmostEqual(rows[0]['duration'], (55555 - 1234) / 44100)

    def test_grouped_decode_matches_native_rate_slices_with_one_carrier_decode(self):
        from scipy.signal import resample_poly
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = np.random.default_rng(42).uniform(-.5, .5, 88200).astype(np.float32)
            payload = io.BytesIO()
            sf.write(payload, samples, 44100, format='WAV', subtype='FLOAT')
            archive = root / 'carrier.tar'
            archive.write_bytes(payload.getvalue())
            records = [{'id': str(index), 'audio': {'kind': 'tar_member', 'archive': str(archive),
                        'offset': 0, 'size': len(payload.getvalue()), 'frames': len(samples),
                        'sample_rate': 44100, 'start_frame': start, 'end_frame': end}}
                       for index, (start, end) in enumerate([(0, 12345), (23457, 55555), (55555, 88200)])]
            with patch.object(sources, '_decode_emilia_carrier', wraps=sources._decode_emilia_carrier) as decode:
                actual = decode_emilia_group(records)
                self.assertEqual(decode.call_count, 1)
            for row in records:
                locator = row['audio']
                expected = resample_poly(samples[locator['start_frame']:locator['end_frame']], 80, 147)
                np.testing.assert_array_equal(actual[row['id']], expected)
                np.testing.assert_array_equal(decode_emilia_audio(row), expected)
            invalid = {**records[0], 'audio': {**records[0]['audio'], 'start_frame': -1}}
            with self.assertRaisesRegex(ValueError, 'Invalid audio slice'):
                decode_emilia_audio(invalid)

    def test_aac_random_access_preserves_timing_and_is_repeatable(self):
        import av
        from fractions import Fraction
        with tempfile.TemporaryDirectory() as directory:
            rate = 44100
            t = np.arange(rate * 6 + 13) / rate
            samples = (.2 * np.sin(2 * np.pi * (200 * t + 70 * t * t))
                       + .01 * np.random.default_rng(8).standard_normal(len(t))).astype(np.float32)
            payload = io.BytesIO()
            with av.open(payload, 'w', format='ipod') as output:
                stream = output.add_stream('aac', rate=rate)
                stream.layout = 'mono'
                for start in range(0, len(samples), 4096):
                    frame = av.AudioFrame.from_ndarray(samples[None, start:start + 4096], format='flt', layout='mono')
                    frame.sample_rate, frame.pts, frame.time_base = rate, start, Fraction(1, rate)
                    for packet in stream.encode(frame):
                        output.mux(packet)
                for packet in stream.encode(None):
                    output.mux(packet)
            archive = Path(directory) / 'audio.tar'
            # Nonzero tar offset and trailing bytes exercise the subfile boundary.
            archive.write_bytes(b'x' * 512 + payload.getvalue() + b'y' * 1024)
            base = {'kind': 'tar_member', 'archive': str(archive), 'offset': 512,
                    'size': len(payload.getvalue()), 'frames': len(samples), 'sample_rate': rate}
            full = sources._decode_emilia_carrier({'id': 'aac', 'audio': base})
            for start, end in [(0, 73111), (rate * 2 + 17, rate * 4 + 29), (rate * 5 + 9, len(samples))]:
                locator = {**base, 'start_frame': start, 'end_frame': end}
                record = {'id': 'aac', 'audio': locator}
                expected = sources._emilia_slice(full, locator)
                actual = decode_emilia_audio(record)
                self.assertEqual(len(actual), len(expected))
                np.testing.assert_array_equal(actual, decode_emilia_audio(record))
                if start == 0:
                    np.testing.assert_array_equal(actual, expected)
                else:
                    # AAC noise synthesis depends on decoder history. Compare
                    # nearby alignments so that noise cannot hide a sample shift.
                    errors = [np.mean((actual[8:-8] - expected[8 + lag:len(expected) - 8 + lag]) ** 2)
                              for lag in range(-4, 5)]
                    self.assertEqual(int(np.argmin(errors)), 4)
                    np.testing.assert_allclose(np.std(actual), np.std(expected), rtol=.01)

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
