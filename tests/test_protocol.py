import unittest

import torch
from torch.nn import functional as F
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig

from qwen3_train.data import collate
from qwen3_train.model import TTSModel, make_config


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(91)
        speaker = Qwen3TTSSpeakerEncoderConfig(
            enc_dim=64, mel_dim=8, enc_channels=[16, 16, 16, 16, 48])
        self.model = TTSModel(make_config(tiny=True), speaker).eval()
        self.row = {'text_ids': [3, 7, 11], 'codes': torch.randint(0, 2048, (3, 16)),
                    'speaker_mels': torch.randn(32, 8)}

    @torch.no_grad()
    def test_parallel_training_logits_match_autoregressive_generation(self):
        # Fill the target frame with the generated choices. Parallel teacher
        # forcing must give the same logits despite seeing future input tokens.
        heads = [self.model.talker.codec_head, *self.model.talker.code_predictor.lm_head]
        captures = [[] for _ in heads]
        handles = [head.register_forward_hook(
            lambda module, args, output, i=i: captures[i].append(output.detach().clone()))
            for i, head in enumerate(heads)]
        try:
            for t in range(3):
                prefix = {**self.row, 'codes': self.row['codes'][:t]}
                frame, _ = self.model(collate([prefix]), mode='next_frame')
                generated_logits = [values.pop() for values in captures]
                target = {**self.row, 'codes': self.row['codes'].clone()}
                target['codes'][t] = frame[0]
                self.model(collate([target]))
                training_logits = [values.pop() for values in captures]
                torch.testing.assert_close(training_logits[0][t:t + 1], generated_logits[0],
                                           atol=2e-6, rtol=2e-5)
                for g in range(1, 16):
                    torch.testing.assert_close(training_logits[g][t:t + 1], generated_logits[g],
                                               atol=2e-6, rtol=2e-5)
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def test_packed_loss_matches_prefix_targets_and_official_depth_logits(self):
        second = {**self.row, 'text_ids': [5], 'codes': self.row['codes'][:1],
                  'speaker_mels': self.row['speaker_mels'][:19] + 1}
        actual = self.model(collate([self.row, second]))
        first_sum = torch.zeros(())
        residual_sum = torch.zeros(())
        for row in [self.row, second]:
            for t in range(len(row['codes']) + 1):
                prefix = {**row, 'codes': row['codes'][:t]}
                h = self.model.hidden(collate([prefix]))[-1:]
                label = row['codes'][t, 0].item() if t < len(row['codes']) else self.model.eos
                first_sum += F.cross_entropy(self.model.talker.codec_head(h), torch.tensor([label]), reduction='sum')
                if t < len(row['codes']):
                    codes = row['codes'][t:t + 1]
                    logits, _ = self.model.talker.forward_sub_talker_finetune(codes, h)
                    residual_sum += F.cross_entropy(logits.flatten(0, 1), codes[:, 1:].flatten(), reduction='sum')
        torch.testing.assert_close(actual['first_sum'], first_sum, atol=2e-5, rtol=2e-6)
        torch.testing.assert_close(actual['residual_sum'], residual_sum, atol=1e-4, rtol=2e-6)


if __name__ == '__main__':
    unittest.main()
