import tempfile
import copy
import json
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3Model
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from qwen3_train.assembly import initialize_model, vocabulary_plan, sha256, save_model
from qwen3_train.model import make_config, TTSModel
from reference_model import ReferenceTTSModel


class AssemblyTests(unittest.TestCase):
    def test_added_tokens_can_use_reserved_embedding_rows(self):
        plan = vocabulary_plan({"a": 0, "b": 1}, {"a": 0, "b": 1, "<tts>": 2}, 8)
        self.assertEqual(plan["physical_rows_added"], 0)
        self.assertEqual(plan["added_tokens"], {"<tts>": 2})

    def test_grow_embeddings_when_capacity_is_exhausted(self):
        plan = vocabulary_plan({"a": 0}, {"a": 0, "<tts>": 1}, 1)
        self.assertEqual(plan["target_embedding_rows"], 2)

    def test_reject_conflicting_shared_token_ids(self):
        with self.assertRaises(ValueError):
            vocabulary_plan({"a": 0}, {"a": 1}, 8)

    def test_weights_come_from_intended_sources(self):
        torch.set_num_threads(2)
        torch.manual_seed(18)
        talker = make_config(tiny=True)
        talker.spk_id = {}
        talker.codec_language_id = {}
        config = Qwen3TTSConfig(talker_config=talker.to_dict(), tts_model_type="base",
            speaker_encoder_config={"enc_dim": 64, "mel_dim": 8, "enc_channels": [16, 16, 16, 16, 48]})
        source = Qwen3Model(Qwen3Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=256,
            rope_theta=talker.rope_theta, rms_norm_eps=talker.rms_norm_eps))
        donor_model = Qwen3TTSForConditionalGeneration(config)
        speaker = donor_model.speaker_encoder
        with torch.no_grad():
            source.embed_tokens.weight[7].fill_(123)
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "base"
            donor = Path(folder) / "speaker"
            donor.mkdir()
            source.save_pretrained(base)
            save_file({k: v.contiguous() for k, v in donor_model.state_dict().items()}, donor / "model.safetensors")
            assembled = initialize_model(config, base, donor, {"<new_tts>": 7}, dtype=torch.float32)
            near = initialize_model(config, base, donor, {"<new_tts>": 7}, dtype=torch.float32,
                                    text_projection_init="near-identity")
            for key, tensor in assembled.state_dict().items():
                if key not in ("talker.text_projection.linear_fc1.weight", "talker.text_projection.linear_fc2.weight"):
                    torch.testing.assert_close(near.state_dict()[key], tensor, atol=0, rtol=0)
            text = near.talker.model.text_embedding.weight[10:40]
            projected = near.talker.text_projection(text)
            self.assertGreater(torch.nn.functional.cosine_similarity(text, projected).mean().item(), 0.999)
            self.assertLess(((projected - text).square().mean() / text.square().mean()).sqrt().item(), 0.04)
            direct = initialize_model(copy.deepcopy(config), base, donor, {"<new_tts>": 7}, dtype=torch.float32,
                                      text_projection_init="identity")
            self.assertIsInstance(direct.talker.text_projection, torch.nn.Identity)
            for key, tensor in direct.state_dict().items():
                torch.testing.assert_close(tensor, assembled.state_dict()[key], atol=0, rtol=0)
            direct_saved = Path(folder) / "direct"
            save_model(direct, direct_saved)
            (direct_saved / "ASSEMBLY_COMPLETE").write_text("ok")
            (direct_saved / "assembly_report.json").write_text(json.dumps({"artifact_sha256": {
                path.name: sha256(path) for path in direct_saved.iterdir() if path.suffix in [".json", ".safetensors"]}}))
            restored = TTSModel.from_assembled(direct_saved)
            self.assertIsInstance(restored.talker.text_projection, torch.nn.Identity)
            for key, tensor in direct.talker.state_dict().items():
                torch.testing.assert_close(restored.talker.state_dict()[key], tensor, atol=0, rtol=0)
            frozen_config = copy.deepcopy(config)
            frozen_config.talker_config.lm_tts_freeze_text_frontend = True
            frozen_config.talker_config.lm_tts_freeze_speaker_encoder = True
            frozen = initialize_model(frozen_config, base, donor, {"<new_tts>": 7}, dtype=torch.float32,
                                      text_projection_init="pretrained", text_initialization="qwen-tts")
            self.assertTrue(all(not p.requires_grad for p in frozen.speaker_encoder.parameters()))
            for key, tensor in donor_model.speaker_encoder.state_dict().items():
                torch.testing.assert_close(frozen.speaker_encoder.state_dict()[key], tensor, atol=0, rtol=0)
            for module in ['text_projection', 'model.text_embedding']:
                expected = donor_model.talker.get_submodule(module)
                actual = frozen.talker.get_submodule(module)
                for key, tensor in expected.state_dict().items():
                    torch.testing.assert_close(actual.state_dict()[key], tensor, atol=0, rtol=0)
                self.assertTrue(all(not p.requires_grad for p in actual.parameters()))
            saved = Path(folder) / "assembled"
            save_model(assembled, saved)
            (saved / "ASSEMBLY_COMPLETE").write_text("ok")
            (saved / "assembly_report.json").write_text(json.dumps({"artifact_sha256": {
                path.name: sha256(path) for path in saved.iterdir() if path.suffix in [".json", ".safetensors"]}}))
            training = TTSModel.from_assembled(saved)
            for key, tensor in assembled.talker.state_dict().items():
                torch.testing.assert_close(training.talker.state_dict()[key], tensor, atol=0, rtol=0)
            for key, tensor in assembled.speaker_encoder.state_dict().items():
                torch.testing.assert_close(training.speaker_encoder.state_dict()[key], tensor, atol=0, rtol=0)
            with (saved / "config.json").open("a") as f:
                f.write(" ")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                TTSModel.from_assembled(saved)
        for key, tensor in source.layers.state_dict().items():
            torch.testing.assert_close(assembled.talker.model.layers.state_dict()[key], tensor, atol=0, rtol=0)
        torch.testing.assert_close(assembled.talker.model.text_embedding.weight[:7], source.embed_tokens.weight[:7], atol=0, rtol=0)
        self.assertLess(assembled.talker.model.text_embedding.weight[7].abs().max().item(), 1)
        for key, tensor in speaker.state_dict().items():
            torch.testing.assert_close(assembled.speaker_encoder.state_dict()[key], tensor, atol=0, rtol=0)
        self.assertEqual(assembled.talker.text_projection.__class__.__name__, "Qwen3TTSTalkerResizeMLP")

    def test_deterministic_padding_preserves_values_and_gradients(self):
        import copy
        from qwen3_train.speaker import deterministic_speaker_padding
        original = torch.nn.Conv1d(3, 4, 3, dilation=2, padding="same", padding_mode="reflect").double()
        converted = copy.deepcopy(original)
        deterministic_speaker_padding(converted)
        a = torch.randn(2, 3, 17, dtype=torch.float64, requires_grad=True)
        b = a.detach().clone().requires_grad_(True)
        expected, actual = original(a), converted(b)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected.square().sum().backward()
        actual.square().sum().backward()
        torch.testing.assert_close(b.grad, a.grad, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(converted.weight.grad, original.weight.grad, rtol=1e-12, atol=1e-12)

    def test_speaker_changes_predictions_and_receives_updates(self):
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
        from qwen3_train.data import collate
        cfg = Qwen3TTSSpeakerEncoderConfig(enc_dim=64, mel_dim=8, enc_channels=[16, 16, 16, 16, 48])
        model = ReferenceTTSModel(make_config(tiny=True), cfg).eval()
        batch = collate([{"text_ids": [3, 5], "codes": torch.randint(0, 2048, (2, 16)),
                          "speaker_mels": torch.randn(32, 8)}])
        with torch.no_grad():
            before = model.hidden(batch)
            changed = {**batch, "speaker_mels": batch["speaker_mels"] + 2}
            self.assertGreater((model.hidden(changed) - before).abs().max().item(), 1e-6)
            changed = {**batch, "codes": (batch["codes"] + 1) % 2048}
            torch.testing.assert_close(model.hidden(changed)[:1], before[:1])
        initial = model.speaker_encoder.fc.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        out = model(batch)
        (out["first_sum"] + out["residual_sum"]).backward()
        self.assertEqual([name for name, p in model.named_parameters() if p.grad is None], [])
        grads = [p.grad for p in model.speaker_encoder.parameters()]
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
        optimizer.step()
        self.assertFalse(torch.equal(initial, model.speaker_encoder.fc.weight))


if __name__ == "__main__":
    unittest.main()
