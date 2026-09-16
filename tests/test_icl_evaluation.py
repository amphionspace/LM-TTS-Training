from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf
import torch

from qwen3_train.data import CodeDataset, collate
from qwen3_train.metrics import content_metrics
from qwen3_train.train import evaluate_audio, generate_sample
from reference_model import ReferenceTTSModel
from qwen3_train.model import make_config


class IclEvaluationTests(unittest.TestCase):
    def test_paired_modes_use_reference_context_but_score_only_target_continuation(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder, ExitStack() as patches:
            root = Path(folder)
            audio = root / 'target.wav'
            sf.write(audio, np.zeros(24000, dtype=np.float32), 24000)
            reference_audio = root / 'reference.wav'
            sf.write(reference_audio, np.sin(np.arange(24000) * 0.04).astype(np.float32) * 0.1, 24000)
            np.savez(root / 'target.npz', codes=np.full((7, 16), 999, dtype=np.int16))
            np.savez(root / 'reference.npz', codes=np.full((3, 16), 77, dtype=np.int16))
            target = {'id': 'target', 'text': 'Say target.', 'text_ids': [4, 5],
                      'speaker': 'speaker', 'language': 'en', 'duration': 1,
                      'audio': str(audio), 'codes': 'target.npz', 'num_frames': 7}
            duplicate = {**target, 'id': 'duplicate', 'text': 'say TARGET!'}
            reference = {**target, 'id': 'reference', 'text': 'Reference words.',
                         'text_ids': [6, 7, 8], 'codes': 'reference.npz', 'num_frames': 3, 'audio': str(reference_audio)}
            for split, rows in [('train', [duplicate, reference]), ('val', [target])]:
                (root / f'{split}.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
            train, val = (CodeDataset(root / f'{split}.jsonl') for split in ('train', 'val'))
            for dataset in (train, val):
                dataset.target_speaker = True
                dataset.text_special_tokens = (18, 19)
            calls, decoded, scores = [], [], []

            class Model:
                config = SimpleNamespace(lm_tts_input_protocol='qwen3_non_streaming')
                speaker_encoder = object()

                def eval(self):
                    pass

                def train(self):
                    pass

                def __call__(self, batch, mode, suppress_eos=False):
                    calls.append({key: value.clone() for key, value in batch.items()})
                    return torch.full((1, 16), 100 + len(calls) % 3), torch.tensor([len(calls) % 3 == 0])

            codec = MagicMock()

            def decode(batch):
                codes = batch['audio_codes'][0].clone()
                decoded.append(codes)
                waveform = np.repeat(codes[:, 0].numpy().astype(np.float32) / 1000, 1920)
                return [waveform], 24000

            codec.decode.side_effect = decode
            scorer = MagicMock()

            def score(path, text, language):
                waveform, sr = sf.read(path)
                scores.append((Path(path), text, language, waveform, sr))
                return {'transcript': text, **content_metrics(text, text)}

            scorer.score.side_effect = score
            patches.enter_context(patch('qwen_tts.Qwen3TTSTokenizer.from_pretrained', return_value=codec))
            patches.enter_context(patch('qwen3_train.train.ASRScorer', return_value=scorer))
            patches.enter_context(patch('qwen3_train.sources.evaluation_audio', return_value=audio))
            patches.enter_context(patch('qwen3_train.train.dist.get_rank', return_value=0))
            patches.enter_context(patch('qwen3_train.train.dist.broadcast'))
            patches.enter_context(patch('qwen3_train.train.dist.barrier'))
            settings = {'codec': 'unused', 'num_samples': 1, 'max_frames': 4, 'asr': True,
                        'conditioning_modes': ['speaker_only', 'icl']}
            evaluate_audio(Model(), val, torch.device('cpu'), settings, root, 10, MagicMock(), train)
            self.assertEqual(calls[0]['codes'].shape, (0, 16))
            self.assertEqual(calls[0]['text_ids'].tolist(), [18, 4, 5, 19])
            self.assertEqual(calls[3]['text_ids'].tolist(), [18, 6, 7, 8, 4, 5, 19])
            torch.testing.assert_close(calls[3]['codes'], train[1]['codes'])
            for batch in (calls[0], calls[3]):
                torch.testing.assert_close(batch['speaker_mels'], train[1]['speaker_mels'])
                self.assertFalse(torch.equal(batch['speaker_mels'], val[0]['speaker_mels']))
            self.assertFalse(any((batch['codes'] == 999).any() for batch in calls))
            self.assertEqual([len(codes) for codes in decoded], [2, 5])
            torch.testing.assert_close(decoded[1][:3], train[1]['codes'])
            torch.testing.assert_close(decoded[0], decoded[1][3:])
            generated = [item for item in scores if item[0].name == 'generated.wav']
            self.assertEqual(len(generated), 2)
            for _, text, language, waveform, sr in generated:
                self.assertEqual((text, language, len(waveform), sr), ('Say target.', 'en', 3840, 24000))
            np.testing.assert_array_equal(generated[0][3], generated[1][3])
            for mode in settings['conditioning_modes']:
                output = root / 'evaluation' / mode / 'step-00000010'
                summary = json.loads((output / 'summary.json').read_text())
                result = json.loads((output / 'sample-00' / 'metrics.json').read_text())
                self.assertEqual(result['speaker_reference_id'], 'reference')
                self.assertEqual((result['frames'], result['eos_reached'], result['duration_seconds']), (2, True, 0.16))
                self.assertEqual((summary['conditioning'], summary['wer']), (mode, 0))

    def test_eos_is_suppressed_before_selection_for_two_new_frames(self):
        torch.set_num_threads(1)
        model = ReferenceTTSModel(make_config(tiny=True)).eval()

        class PreferEos(torch.nn.Module):
            def forward(self, hidden):
                logits = hidden.new_zeros(*hidden.shape[:-1], model.config.vocab_size)
                logits[..., model.eos] = 20
                logits[..., 7] = 10
                return logits

        model.talker.codec_head = PreferEos()
        with tempfile.TemporaryDirectory() as folder, ExitStack() as patches:
            root = Path(folder)
            audio = root / 'target.wav'
            sf.write(audio, np.zeros(24000, dtype=np.float32), 24000)
            target = {'id': 'target', 'text': 'Target.', 'text_ids': [18, 4, 19],
                      'codes': torch.full((7, 16), 999), 'duration': 1, 'audio': str(audio)}
            reference = {**target, 'id': 'reference', 'text': 'Reference.',
                         'text_ids': [18, 5, 19], 'codes': torch.full((5, 16), 77),
                         'speaker_mels': torch.zeros(32, 8)}
            decoded = []
            codec = MagicMock()

            def decode(batch):
                codes = batch['audio_codes'][0]
                decoded.append(codes.clone())
                return [np.zeros(len(codes) * 1920, dtype=np.float32)], 24000

            codec.decode.side_effect = decode
            patches.enter_context(patch('qwen3_train.sources.evaluation_audio', return_value=audio))
            patches.enter_context(patch('qwen3_train.train.dist.get_rank', return_value=0))
            patches.enter_context(patch('qwen3_train.train.dist.broadcast'))
            patches.enter_context(patch('qwen3_train.train.dist.barrier'))
            for conditioning in ['speaker_only', 'icl']:
                with self.subTest(conditioning=conditioning):
                    result = generate_sample(model, [target], torch.device('cpu'),
                                             {'max_frames': 4, 'asr': False}, root, 10, MagicMock(),
                                             codec=codec, reference=reference, conditioning=conditioning)
                    self.assertEqual((result['frames'], result['eos_reached'], result['truncated']),
                                     (2, True, False))
                    self.assertEqual(result['duration_seconds'], .16)
                    # EOS must be masked before argmax, rather than ignored after
                    # it selected the EOS placeholder/clamped audio code.
                    torch.testing.assert_close(decoded[-1][-2:, 0], torch.tensor([7, 7]))
                    self.assertEqual(len(decoded[-1]), 2 + (5 if conditioning == 'icl' else 0))
            from scripts.inspect_checkpoint import generate
            codes, stopped = generate(model, collate([target]), max_frames=4)
            self.assertTrue(stopped)
            np.testing.assert_array_equal(codes[:, 0], [7, 7])


if __name__ == '__main__':
    unittest.main()
