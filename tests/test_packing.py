import copy
import unittest
from unittest.mock import patch

import torch
from transformers.utils import is_flash_attn_2_available

from qwen3_train.data import collate
import test_qwen_protocol


class PackingTests(unittest.TestCase):
    def setUp(self):
        fixture = test_qwen_protocol.FrozenFrontendProtocolTests()
        fixture.setUp()
        self.model = fixture.model
        self.rows = [fixture.row,
                     {**fixture.row, 'text_ids': [18, 5, 19],
                      'codes': fixture.row['codes'][:1] + 1,
                      'speaker_mels': fixture.row['speaker_mels'][:19] + 1},
                     {**fixture.row, 'text_ids': [18, 6, 7, 8, 9, 10, 19],
                      'codes': fixture.row['codes'][:2] + 2,
                      'speaker_mels': fixture.row['speaker_mels'] - 1}]

    def test_packed_embeddings_match_individual_utterances(self):
        for protocol in ['qwen3_non_streaming', 'legacy_prefix']:
            with self.subTest(protocol=protocol):
                self.model.config.lm_tts_input_protocol = protocol
                # An empty audio history is needed when starting generation.
                rows = [*self.rows, {**self.rows[0], 'codes': self.rows[0]['codes'][:0]}]
                batch = collate(rows)
                inputs, audio_positions = self.model.input_embeddings(batch)
                expected_inputs, expected_audio, expected_positions = [], [], []
                for row in rows:
                    single, indices = self.model.input_embeddings(collate([row]))
                    embeds = single['inputs_embeds']
                    expected_inputs.append(embeds[0])
                    expected_audio.append(embeds[0, indices])
                    expected_positions.append(torch.arange(embeds.shape[1]))
                torch.testing.assert_close(inputs['inputs_embeds'][0], torch.cat(expected_inputs))
                torch.testing.assert_close(inputs['inputs_embeds'][0, audio_positions], torch.cat(expected_audio))
                torch.testing.assert_close(inputs['position_ids'][0], torch.cat(expected_positions))
                torch.testing.assert_close(inputs['cu_seq_lens_q'].diff().long(),
                                           torch.tensor([len(t) for t in expected_inputs]))
                self.assertEqual(batch['speaker_mels'].shape[0], sum(len(r['speaker_mels']) for r in rows))

    def test_packed_loss_and_gradients_match_individual_utterances(self):
        gradients, losses = [], []
        first_count = sum(len(r['codes']) + 1 for r in self.rows)
        residual_count = 15 * sum(len(r['codes']) for r in self.rows)
        for batches in [[self.rows], [[row] for row in self.rows]]:
            self.model.zero_grad(set_to_none=True)
            loss_sum = 0
            for rows in batches:
                out = self.model(collate(rows))
                loss = out['first_sum'] / first_count + .3 * out['residual_sum'] / residual_count
                loss.backward()
                loss_sum += loss.detach()
            losses.append(loss_sum)
            gradients.append(torch.cat([p.grad.flatten() for p in self.model.parameters() if p.requires_grad]))
        torch.testing.assert_close(losses[0], losses[1], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(gradients[0], gradients[1], atol=2e-6, rtol=2e-4)

    @unittest.skipUnless(torch.cuda.is_available() and is_flash_attn_2_available(), 'Requires CUDA and FA2')
    def test_flash_packed_loss_and_gradients_match_individual_utterances(self):
        model = self.model.bfloat16().cuda().train()
        model.config._attn_implementation = 'flash_attention_2'
        model.config.code_predictor_config._attn_implementation = 'flash_attention_2'
        with patch.dict('os.environ', {'FLASH_ATTENTION_DETERMINISTIC': '1'}):
            for protocol in ['qwen3_non_streaming', 'legacy_prefix']:
                with self.subTest(protocol=protocol):
                    model.config.lm_tts_input_protocol = protocol
                    outputs, gradients = [], []
                    first_count = sum(len(r['codes']) + 1 for r in self.rows)
                    residual_count = 15 * sum(len(r['codes']) for r in self.rows)
                    for batches in [[self.rows], [[row] for row in self.rows]]:
                        model.zero_grad(set_to_none=True)
                        loss_sum = 0
                        for rows in batches:
                            batch = {k: v.cuda() for k, v in collate(rows).items()}
                            batch['speaker_mels'] = batch['speaker_mels'].bfloat16()
                            out = model(batch)
                            loss = out['first_sum'] / first_count + .3 * out['residual_sum'] / residual_count
                            loss.backward()
                            loss_sum += loss.detach()
                        outputs.append(loss_sum)
                        gradients.append(torch.cat([p.grad.float().flatten() for p in model.parameters() if p.requires_grad]))
                    torch.testing.assert_close(outputs[1], outputs[0], rtol=2e-3, atol=1e-3)
                    relative_error = (gradients[1] - gradients[0]).norm() / gradients[0].norm()
                    self.assertLess(relative_error.item(), .03)

    @unittest.skipUnless(torch.cuda.is_available() and is_flash_attn_2_available(), 'Requires CUDA and FA2')
    @torch.no_grad()
    def test_packed_attention_isolates_samples_and_future_audio(self):
        model = self.model.bfloat16().cuda().eval()
        model.config._attn_implementation = 'flash_attention_2'
        model.config.code_predictor_config._attn_implementation = 'flash_attention_2'
        def hidden(rows):
            batch = {k: v.cuda() for k, v in collate(rows).items()}
            batch['speaker_mels'] = batch['speaker_mels'].bfloat16()
            return model.hidden(batch)
        original = hidden(self.rows)
        changed = copy.deepcopy(self.rows)
        changed[0]['text_ids'] = [token + 1 for token in changed[0]['text_ids']]
        changed[0]['codes'].fill_(73)
        changed[0]['speaker_mels'].add_(2)
        boundary = len(self.rows[0]['codes']) + 1
        torch.testing.assert_close(hidden(changed)[boundary:], original[boundary:], rtol=0, atol=0)
        changed = copy.deepcopy(self.rows)
        changed[0]['codes'][1:].fill_(73)
        torch.testing.assert_close(hidden(changed)[:2], original[:2], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
