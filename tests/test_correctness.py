import unittest
import tempfile
import torch

from qwen3_train.data import collate, train_batches
from qwen3_train.metrics import content_metrics, aggregate_content
from qwen3_train.model import TTSModel, make_config


def row(length, text=(12, 34, 56)):
    return {"codes": torch.randint(0, 2048, (length, 16)), "text_ids": list(text)}


class Correctness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(123)
        cls.model = TTSModel(make_config(tiny=True)).eval()

    def test_text_backbone_mapping_preserves_hidden_states(self):
        from transformers import Qwen3Config, Qwen3Model
        cfg = make_config(tiny=True)
        source_cfg = Qwen3Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                vocab_size=256, rope_theta=cfg.rope_theta, rms_norm_eps=cfg.rms_norm_eps)
        source_cfg._attn_implementation = "sdpa"
        source = Qwen3Model(source_cfg).eval()
        target = TTSModel(cfg).eval()
        with tempfile.TemporaryDirectory() as folder:
            source.save_pretrained(folder)
            report = target.initialize_backbone(folder)
        self.assertEqual(report["loaded"], ["layers", "norm", "text_embedding"])
        ids = torch.tensor([[3, 5, 7, 9]])
        with torch.no_grad():
            expected = source(ids, use_cache=False).last_hidden_state
            actual = target.talker.model(inputs_embeds=target.talker.model.text_embedding(ids), use_cache=False).last_hidden_state
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_target_frame_cannot_leak_into_its_prediction(self):
        batch = collate([row(4)])
        with torch.no_grad():
            before = self.model.hidden(batch)
            changed = {k: v.clone() for k, v in batch.items()}
            changed["codes"][2] = (changed["codes"][2] + 71) % 2048
            after = self.model.hidden(changed)
        # BOS, frame_0 and frame_1 states predict frames_0..2 respectively.
        torch.testing.assert_close(before[:3], after[:3], atol=1e-6, rtol=1e-6)
        self.assertGreater((before[3] - after[3]).abs().max().item(), 1e-5)

    def test_packing_preserves_sum_of_utterance_losses(self):
        a, b = row(2, (5, 6)), row(4, (7, 8, 9, 10))
        with torch.no_grad():
            one, two = self.model(collate([a])), self.model(collate([b]))
            together = self.model(collate([a, b]))
        self.assertEqual(together["first_count"].item(), 8)  # 6 frames + 2 EOS
        self.assertEqual(together["frame_count"].item(), 6)
        for key in ["first_sum", "residual_sum"]:
            torch.testing.assert_close(together[key], one[key] + two[key], atol=1e-3, rtol=1e-5)

    def test_empty_audio_prefix_can_generate(self):
        batch = collate([row(1)])
        batch["codes"] = batch["codes"][:0]
        batch["frame_lengths"].zero_()
        with torch.no_grad():
            codes, stop = self.model(batch, mode="next_frame")
        self.assertEqual(codes.shape, (1, 16))
        self.assertTrue(((codes >= 0) & (codes < 2048)).all())
        self.assertEqual(stop.shape, (1,))

    def test_all_trainable_modules_receive_gradients(self):
        self.model.zero_grad(set_to_none=True)
        output = self.model(collate([row(2)]))
        (output["first_sum"] + output["residual_sum"]).backward()
        missing = [n for n, p in self.model.named_parameters() if p.requires_grad and p.grad is None]
        self.assertEqual(missing, [])

    def test_sampler_does_not_always_drop_longest_sample(self):
        class Dataset:
            rows = [{"num_frames": i + 1, "text_ids": [1]} for i in range(3)]
            def __len__(self):
                return len(self.rows)
        seen = set()
        for epoch in range(20):
            for rank in range(2):
                seen.update(i for batch in train_batches(Dataset(), 1, 2, rank, 42, epoch) for i in batch)
        self.assertEqual(seen, {0, 1, 2})

    def test_corpus_wer_weights_reference_lengths(self):
        scores = [content_metrics("one", "wrong"), content_metrics("a b c d e f g h i", "a b c d e f g h i")]
        self.assertAlmostEqual(aggregate_content(scores)["wer"], 0.1)

    def test_content_error_accounting(self):
        self.assertEqual(content_metrics("Hello, WORLD!", "hello world")["wer"], 0)
        missing = content_metrics("one two three", "one three")
        self.assertEqual(missing["deletions"], 1)
        self.assertAlmostEqual(missing["wer"], 1 / 3)
        self.assertEqual(content_metrics("one two", "one two two")["insertions"], 1)
        self.assertEqual(content_metrics("one two", "")["wer"], 1)
        self.assertEqual(content_metrics("cat", "bat")["substitutions"], 1)


if __name__ == "__main__":
    unittest.main()
