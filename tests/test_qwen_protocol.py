import copy
import unittest
from unittest.mock import patch

import test_protocol
import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

from lm_tts.data.batch import collate
from lm_tts.models.protocol import frame_embeddings


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
        self.row["text_ids"] = [18, 3, 7, 11, 19]

    @torch.no_grad()
    def test_embeddings_match_official_generate_and_icl(self):
        self.check_official_prefix(["Auto"], None)

    @torch.no_grad()
    def test_language_ids_match_official_generate(self):
        cfg = self.model.config
        cfg.codec_think_id = 2154
        cfg.codec_language_id = {"english": 2050, "dutch": 2072}
        cfg.lm_tts_language_conditioning = "codec_language_id"
        for language, codec_id in (("English", 2050), ("Dutch", 2072), ("Auto", -1)):
            self.check_official_prefix([language], codec_id)
        with self.assertRaisesRegex(ValueError, "language_codec_ids"):
            self.model.input_embeddings(collate([self.row]))

    @torch.no_grad()
    def test_streaming_matches_official_generate_and_icl(self):
        self.model.config.lm_tts_input_layout = "mixed"
        self.check_official_prefix(["Auto"], None, streaming=True)

    @torch.no_grad()
    def test_streaming_language_ids_match_official_generate(self):
        cfg = self.model.config
        cfg.codec_think_id = 2154
        cfg.codec_language_id = {"english": 2050}
        cfg.lm_tts_language_conditioning = "codec_language_id"
        cfg.lm_tts_input_layout = "mixed"
        self.check_official_prefix(["English"], 2050, streaming=True)

    @torch.no_grad()
    def test_mixed_batch_keeps_non_streaming_rows_and_rejects_lagging_text(self):
        second = {**self.row, "text_ids": [18, 5, 19], "codes": self.row["codes"][:2]}
        plain = self.model.input_embeddings(collate([self.row, second]))[0]
        self.model.config.lm_tts_input_layout = "mixed"
        with self.assertRaisesRegex(ValueError, "streaming flags"):
            self.model.input_embeddings(collate([self.row]))
        batch = collate([self.row, second])
        batch["streaming"] = torch.tensor([False, True])
        inputs, audio = self.model.input_embeddings(batch)
        # The non-streaming row must be bit-identical to the plain layout of the same batch.
        end = plain["cu_seq_lens_q"][1].item()
        self.assertEqual(inputs["cu_seq_lens_q"][1].item(), end)
        torch.testing.assert_close(
            inputs["inputs_embeds"][0, :end], plain["inputs_embeds"][0, :end], atol=0, rtol=0
        )
        # Streaming keeps only bos of the 3 text tokens in the prefix.
        lengths = inputs["cu_seq_lens_q"].diff().tolist()
        self.assertEqual(lengths[1], plain["cu_seq_lens_q"].diff()[1].item() - (3 - 1))
        self.assertEqual(len(audio), (3 + 1) + (2 + 1))
        lagging = collate([{**self.row, "codes": self.row["codes"][:2]}])
        lagging["streaming"] = torch.tensor([True])
        with self.assertRaisesRegex(ValueError, "ends by the EOS"):
            self.model.input_embeddings(lagging)

    @torch.no_grad()
    def test_streaming_pause_inserts_tts_pad_between_streamed_tokens(self):
        self.model.config.lm_tts_input_layout = "mixed"
        talker, cfg = self.model.talker, self.model.config
        row = {**self.row, "text_ids": [18, 5, 19]}
        batch = collate([row])
        batch["streaming"] = torch.tensor([True])
        batch["streaming_pause"] = torch.tensor([[1, 2]])
        inputs, audio = self.model.input_embeddings(batch)
        text = talker.text_projection(talker.model.text_embedding(torch.tensor([5, 20, 20, 19])))
        cbos = talker.model.codec_embedding.weight[cfg.codec_bos_id : cfg.codec_bos_id + 1]
        frames = torch.cat([cbos, frame_embeddings(talker, 16, row["codes"])])
        hidden = inputs["inputs_embeds"].flatten(0, 1)
        torch.testing.assert_close(hidden[audio], frames + text, atol=2e-7, rtol=2e-6)
        batch["streaming_pause"] = torch.tensor([[1, 3]])
        with self.assertRaisesRegex(ValueError, "ends by the EOS"):
            self.model.input_embeddings(batch)

    def check_official_prefix(self, languages, codec_id, streaming=False):
        full = Qwen3TTSConfig(
            talker_config=self.model.config.to_dict(),
            speaker_encoder_config={
                "enc_dim": 64,
                "mel_dim": 8,
                "enc_channels": [16, 16, 16, 16, 48],
            },
            tts_bos_token_id=18,
            tts_eos_token_id=19,
            tts_pad_token_id=20,
            tts_model_type="base",
        )
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
                row = {
                    **self.row,
                    "text_ids": [18, *text, 19],
                    # Streaming compares the whole teacher-forced row with prefill + decode steps.
                    "codes": self.row["codes"] if streaming else self.row["codes"][:history_length],
                }
                batch = collate([row])
                if codec_id is not None:
                    batch["language_codec_ids"] = torch.tensor([codec_id])
                if streaming:
                    batch["streaming"] = torch.tensor([True])
                speaker = self.model.speaker_encoder(batch["speaker_mels"].unsqueeze(0))[0]
                prompt = {
                    "ref_spk_embedding": [speaker],
                    "x_vector_only_mode": [history_length == 0],
                    "icl_mode": [history_length > 0],
                    "ref_code": [row["codes"][:history_length]] if history_length else None,
                }
                if history_length:
                    ids = [torch.tensor([[21, 22, 23, *text[1:], 24, 25, 21, 22, 23]])]
                    refs = [torch.tensor([[21, 22, 23, text[0], 24, 25]])]
                else:
                    ids = [torch.tensor([[21, 22, 23, *text, 24, 25, 21, 22, 23]])]
                    refs = None
                with patch.object(official.talker, "generate", side_effect=capture):
                    with self.assertRaises(Captured):
                        official.generate(
                            input_ids=ids,
                            ref_ids=refs,
                            voice_clone_prompt=prompt,
                            languages=languages,
                            non_streaming_mode=not streaming,
                        )
                inputs, _ = self.model.input_embeddings(batch)
                expected = captured["inputs_embeds"]
                if streaming:
                    # Each decode step adds the next trailing text hidden (then tts_pad) to the
                    # embedding of the frame fed back, as the official talker forward does.
                    trailing, pad = captured["trailing_text_hidden"], captured["tts_pad_embed"]
                    steps = [
                        frame_embeddings(self.model.talker, 16, row["codes"][g : g + 1])
                        + (trailing[0, step : step + 1] if step < trailing.shape[1] else pad[0])
                        for step, g in enumerate(range(history_length, len(row["codes"])))
                    ]
                    expected = torch.cat([expected[0], *steps]).unsqueeze(0)
                torch.testing.assert_close(inputs["inputs_embeds"], expected, atol=2e-7, rtol=2e-6)
                torch.testing.assert_close(
                    inputs["position_ids"][0],
                    torch.arange(expected.shape[1]),
                )


class FrozenFrontendProtocolTests(QwenProtocolTests):
    def setUp(self):
        super().setUp()
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
        from reference_model import ReferenceTTSModel

        config = self.model.config
        config.text_hidden_size = 128
        config.lm_tts_text_projection = "mlp"
        config.lm_tts_freeze_text_frontend = True
        config.lm_tts_freeze_speaker_encoder = True
        speaker = Qwen3TTSSpeakerEncoderConfig(
            enc_dim=64, mel_dim=8, enc_channels=[16, 16, 16, 16, 48]
        )
        self.model = ReferenceTTSModel(config, speaker).eval()

    def test_frozen_frontend_survives_optimizer_step(self):
        self.model.train()
        self.assertFalse(self.model.speaker_encoder.training)
        modules = [
            self.model.talker.model.text_embedding,
            self.model.talker.text_projection,
            self.model.speaker_encoder,
        ]
        before = [p.detach().clone() for module in modules for p in module.parameters()]
        out = self.model(collate([self.row]))
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=1e-3
        )
        (out["first_sum"] + out["residual_sum"]).backward()
        self.assertTrue(all(p.grad is not None for p in self.model.parameters() if p.requires_grad))
        optimizer.step()
        for expected, parameter in zip(
            before, [p for module in modules for p in module.parameters()]
        ):
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
            torch.testing.assert_close(parameter, expected, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "Flash Attention requires CUDA")
    def test_flash_attention_preserves_variable_length_loss_and_gradients(self):
        from transformers.utils import is_flash_attn_2_available

        if not is_flash_attn_2_available():
            self.skipTest("Flash Attention 2 is not installed")
        flash = copy.deepcopy(self.model)
        flash.config._attn_implementation = "flash_attention_2"
        flash.config.code_predictor_config._attn_implementation = "sdpa"
        second = {
            **self.row,
            "text_ids": [18, 5, 19],
            "codes": self.row["codes"][:1],
            "speaker_mels": self.row["speaker_mels"][:19] + 1,
        }
        batch = {k: v.cuda() for k, v in collate([self.row, second]).items()}
        batch["speaker_mels"] = batch["speaker_mels"].bfloat16()
        results, gradients = [], []
        with patch.dict("os.environ", {"FLASH_ATTENTION_DETERMINISTIC": "1"}):
            for model in [self.model, flash, flash]:
                model.cuda().bfloat16().train()
                model.zero_grad(set_to_none=True)
                out = model(batch)
                loss = out["first_sum"] / out["first_count"] + 0.3 * out["residual_sum"] / (
                    15 * out["frame_count"]
                )
                loss.backward()
                results.append(torch.stack([out["first_sum"], out["residual_sum"]]).detach())
                gradients.append(
                    torch.cat(
                        [p.grad.float().flatten() for p in model.parameters() if p.requires_grad]
                    )
                )
        torch.testing.assert_close(results[1], results[0], rtol=2e-3, atol=1e-3)
        relative_gradient_error = (gradients[1] - gradients[0]).norm() / gradients[0].norm()
        self.assertLess(relative_gradient_error.item(), 0.03)
        torch.testing.assert_close(results[2], results[1], rtol=0, atol=0)
        torch.testing.assert_close(gradients[2], gradients[1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
