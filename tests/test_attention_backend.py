import unittest

import torch

from lm_tts.data.batch import collate
from lm_tts.models.qwen import TTSModel, make_config
from lm_tts.objectives.tts import tts_loss
from lm_tts.training.precision import training_precision


class PrecisionTests(unittest.TestCase):
    def test_precision_policies_and_incompatible_backend(self):
        for precision, dtype, backend in [
            ("bf16", torch.bfloat16, "flash_attention_2"),
            ("fp32", torch.float32, "sdpa"),
        ]:
            policy, actual = training_precision({"precision": precision}, {})
            self.assertEqual(
                (policy.param_dtype, policy.reduce_dtype, actual), (dtype, torch.float32, backend)
            )
        with self.assertRaises(ValueError):
            training_precision({"precision": "fp16"}, {})
        with self.assertRaises(ValueError):
            training_precision({"precision": "fp32"}, {"attn_implementation": "flash_attention_2"})

    def test_fp32_isolates_samples_and_matches_individual_gradients(self):
        torch.set_num_threads(2)
        torch.manual_seed(12)
        config = make_config(tiny=True)
        config._attn_implementation = config.code_predictor_config._attn_implementation = "sdpa"
        model = TTSModel(config).float().eval()
        rows = [
            {"text_ids": [2, 4], "codes": torch.randint(0, 2048, (2, 16))},
            {"text_ids": [3, 5, 7, 9, 11], "codes": torch.randint(0, 2048, (5, 16))},
        ]
        batch = collate(rows)
        together = tts_loss(model(batch), batch, model.eos)
        (together["first_sum"] + 0.3 * together["residual_sum"]).backward()
        gradients = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        individual = []
        for row in rows:
            b = collate([row])
            out = tts_loss(model(b), b, model.eos)
            individual.append(out)
            (out["first_sum"] + 0.3 * out["residual_sum"]).backward()
        for key in ("first_sum", "residual_sum"):
            torch.testing.assert_close(
                together[key], sum(o[key] for o in individual), atol=1e-4, rtol=1e-5
            )
        for key, p in model.named_parameters():
            if key in gradients:
                torch.testing.assert_close(p.grad, gradients[key], atol=5e-5, rtol=1e-4)
        original = model.hidden(batch).detach()
        changed = {**rows[1], "text_ids": [42] * 5, "codes": (rows[1]["codes"] + 5) % 2048}
        torch.testing.assert_close(
            original[:3], model.hidden(collate([rows[0], changed]))[:3], atol=1e-6, rtol=1e-5
        )
