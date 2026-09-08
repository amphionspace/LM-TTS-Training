import unittest
from unittest.mock import patch
import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from qwen3_train.data import collate
import test_protocol


class QwenProtocolTests(test_protocol.ProtocolTests):
    def setUp(self):
        super().setUp()
        cfg = self.model.config
        cfg.lm_tts_input_protocol = "qwen3_non_streaming"
        cfg.lm_tts_text_projection = "identity"
        cfg.lm_tts_pad_token_id = 20
        cfg.lm_tts_role_ids = [21, 22, 23]
        cfg.codec_nothink_id = 2151
        cfg.codec_think_bos_id = 2152
        cfg.codec_think_eos_id = 2153
        cfg.spk_id = {}
        cfg.spk_is_dialect = {}
        cfg.codec_language_id = {}
        self.model.talker.text_projection = torch.nn.Identity()
        self.row['text_ids'] = [18, 3, 7, 11, 19]

    @torch.no_grad()
    def test_embeddings_match_official_generate_and_icl(self):
        full = Qwen3TTSConfig(talker_config=self.model.config.to_dict(),
            speaker_encoder_config={'enc_dim': 64, 'mel_dim': 8, 'enc_channels': [16, 16, 16, 16, 48]},
            tts_bos_token_id=18, tts_eos_token_id=19, tts_pad_token_id=20, tts_model_type="base")
        official = Qwen3TTSForConditionalGeneration(full).eval()
        official.talker = self.model.talker
        official.speaker_encoder = self.model.speaker_encoder
        class Captured(Exception):
            pass
        captured = {}
        def capture(**kwargs):
            captured.update(kwargs)
            raise Captured()
        for text in ([3, 7, 11], [5]):
            for history_length in (0, 1, 3):
                row = {**self.row, 'text_ids': [18, *text, 19], 'codes': self.row['codes'][:history_length]}
                batch = collate([row])
                speaker = self.model.speaker_encoder(batch['speaker_mels'])[0]
                prompt = {'ref_spk_embedding': [speaker], 'x_vector_only_mode': [history_length == 0],
                          'icl_mode': [history_length > 0], 'ref_code': [row['codes']] if history_length else None}
                if history_length:
                    ids = [torch.tensor([[21, 22, 23, *text[1:], 24, 25, 21, 22, 23]])]
                    refs = [torch.tensor([[21, 22, 23, text[0], 24, 25]])]
                else:
                    ids = [torch.tensor([[21, 22, 23, *text, 24, 25, 21, 22, 23]])]
                    refs = None
                with patch.object(official.talker, 'generate', side_effect=capture):
                    with self.assertRaises(Captured):
                        official.generate(input_ids=ids, ref_ids=refs, voice_clone_prompt=prompt,
                                          languages=['Auto'], non_streaming_mode=True)
                actual, mask, _ = self.model.input_embeddings(batch)
                torch.testing.assert_close(actual, captured['inputs_embeds'], atol=2e-7, rtol=2e-6)
                torch.testing.assert_close(mask.long(), captured['attention_mask'])


class FrozenFrontendProtocolTests(QwenProtocolTests):
    def setUp(self):
        super().setUp()
        from qwen3_train.model import TTSModel
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
        config = self.model.config
        config.text_hidden_size = 128
        config.lm_tts_text_projection = 'mlp'
        config.lm_tts_freeze_text_frontend = True
        speaker = Qwen3TTSSpeakerEncoderConfig(enc_dim=64, mel_dim=8, enc_channels=[16, 16, 16, 16, 48])
        self.model = TTSModel(config, speaker).eval()

    def test_frozen_frontend_survives_optimizer_step(self):
        modules = [self.model.talker.model.text_embedding, self.model.talker.text_projection]
        before = [p.detach().clone() for module in modules for p in module.parameters()]
        out = self.model(collate([self.row]))
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=1e-3)
        (out['first_sum'] + out['residual_sum']).backward()
        self.assertTrue(all(p.grad is not None for p in self.model.parameters() if p.requires_grad))
        optimizer.step()
        for expected, parameter in zip(before, [p for module in modules for p in module.parameters()]):
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
            torch.testing.assert_close(parameter, expected, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
