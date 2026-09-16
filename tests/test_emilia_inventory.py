import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.inventory_emilia import scan
from scripts.export_emilia_full import main as export_full
from scripts.filter_emilia_dialogue import AudioQuality, overlaps_other_speaker


class EmiliaInventoryTests(unittest.TestCase):
    def test_overlapping_views_prefer_standalone_audio_and_reject_invalid_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'output'
            (output / 'inventory').mkdir(parents=True)
            for kind in ('short', 'long', 'dialogue'):
                (output / 'candidates' / kind).mkdir(parents=True)

            def item(uid, start, end, language='en'):
                return {'id': uid, 'rel_start_samples': start, 'rel_end_samples': end,
                        'language': language, 'speaker': 'speaker', 'text': uid}

            entries = [('short', 'same', 100, [item('same', 0, 100)]),
                       ('long', 'long-carrier', 400, [item('same', 100, 200), item('long-only', 200, 300, 'zh')]),
                       ('dialogue', 'dialogue-carrier', 500,
                        [item('same', 0, 100), item('long-only', 100, 200, 'zh'),
                         item('dialogue-only', 200, 400), item('out-of-bounds', 400, 600)])]
            index_lines = []
            archive = root / 'shard.tar'
            with archive.open('wb') as f:
                for kind, uid, frames, shorts in entries:
                    meta = {'id': uid, 'type': kind, 'frames': frames, 'sample_rate': 100,
                            'languages': ['en'], 'recording_id': 'recording', 'short': shorts}
                    payload = json.dumps(meta).encode()
                    offset = f.tell()
                    f.write(payload)
                    index_lines.append(f'{uid}.json\t{offset}\t{len(payload)}\n')
                    index_lines.append(f'{uid}.m4a\t{f.tell()}\t1\n')
                    f.write(b'x')
            index = root / 'shard.tar.idx'
            index.write_text(''.join(index_lines))
            report = scan((str(index), str(output)))
            self.assertEqual(report['invalid_short_views'], {'dialogue/en': 1})
            self.assertEqual(report['short_views']['dialogue/en'], 2)
            (output / 'INVENTORY_COMPLETE.json').write_text(json.dumps(report))
            accepted = output / 'dialogue-quality/kept'
            accepted.mkdir(parents=True)
            dialogue_path = next((output / 'candidates/dialogue').glob('*.jsonl'))
            dialogue = [json.loads(line) for line in dialogue_path.read_text().splitlines()]
            for row in dialogue:
                row['source']['quality'] = {'method': 'dnsmos_p835_actual_slice'}
            (accepted / dialogue_path.name).write_text(''.join(json.dumps(row) + '\n' for row in dialogue))
            # An unapproved source candidate must never enter the exported pool.
            with dialogue_path.open('a') as stream:
                stream.write(json.dumps({**dialogue[0], 'id': 'unapproved'}) + '\n')
            quality_source = Path(__file__).resolve().parents[1] / 'scripts/filter_emilia_dialogue.py'
            (output / 'DIALOGUE_QUALITY_COMPLETE.json').write_text(json.dumps({
                'recipe': {'source_sha256': hashlib.sha256(quality_source.read_bytes()).hexdigest()}}))
            asmr = {**dialogue[0], 'id': 'asmr-only',
                    'audio': {**dialogue[0]['audio'], 'archive': '/dataset/ASMR/asmr-00000.tar'},
                    'source': {'type': 'short', 'recording_id': 'asmr-recording'}}
            (output / 'candidates/short/asmr-00000.jsonl').write_text(json.dumps(asmr) + '\n')
            (output / 'candidates/short/other.jsonl').write_text(json.dumps({
                **asmr, 'id': 'same-recording-other-id', 'audio': dialogue[0]['audio']}) + '\n')
            with patch('sys.argv', ['export', '--output', str(output), '--chunk-records', '1']):
                export_full()
            rows = [json.loads(line) for p in sorted((output / 'raw').glob('*.jsonl'))
                    for line in p.read_text().splitlines()]
            self.assertEqual([(r['id'], r['source']['type']) for r in rows],
                             [('emilia2:same', 'short'), ('emilia2:long-only', 'long'),
                              ('emilia2:dialogue-only', 'dialogue')])
            self.assertNotIn('start_frame', rows[0]['audio'])
            self.assertEqual((rows[2]['audio']['start_frame'], rows[2]['audio']['end_frame']), (200, 400))
            result = json.loads((output / 'EXPORT_COMPLETE.json').read_text())
            self.assertEqual(result['records'], {'en': 2, 'zh': 1})
            self.assertEqual(result['duplicate_ids_skipped'], 3)
            self.assertEqual(result['excluded'], {'short/en/ASMR': 2})
            self.assertAlmostEqual(result['hours']['en'], 3 / 3600)
            self.assertAlmostEqual(result['hours']['zh'], 1 / 3600)

    def test_dialogue_overlap_uses_all_speakers_but_allows_touching_boundaries(self):
        row = {'speaker': 'emilia2:alice', 'audio': {'start_frame': 100, 'end_frame': 200}}
        def item(speaker, start, end):
            return {'speaker': speaker, 'rel_start_samples': start, 'rel_end_samples': end, 'language': 'ja'}
        self.assertFalse(overlaps_other_speaker(row, [item('alice', 50, 250), item('bob', 200, 300)]))
        self.assertTrue(overlaps_other_speaker(row, [item('bob', 199, 300)]))

    def test_quality_windows_include_late_audio(self):
        class Session:
            def run(self, _, inputs):
                return [np.repeat(inputs['input_1'].mean(), 3).reshape(1, 3)]
        scorer = AudioQuality.__new__(AudioQuality)
        scorer.session = Session()
        audio = np.zeros(20 * 16000, dtype=np.float32)
        before = scorer.score(audio)
        audio[17 * 16000:] = 1
        self.assertGreater(scorer.score(audio)['OVRL'], before['OVRL'])


if __name__ == '__main__':
    unittest.main()
