import unittest

import torch

from lm_tts.objectives.tts import loss_normalizers, tts_loss


class SelectedStatisticsTests(unittest.TestCase):
    def test_selected_utterances_recombine_and_match_a_subset_only_batch(self):
        torch.manual_seed(0)
        lengths, supervised = torch.tensor([5, 3, 7, 4]), torch.tensor([3, 3, 5, 4])
        frames = int(lengths.sum())
        codes = torch.randint(0, 30, (frames, 16))
        mask = torch.cat([torch.arange(n) >= n - s for n, s in zip(lengths, supervised)])
        predictions = {
            "first_logits": torch.randn(frames + len(lengths), 32),
            "residual_logits": [torch.randn(frames, 30) for _ in range(15)],
        }
        batch = dict(
            frame_lengths=lengths,
            codes=codes,
            supervised_frame_lengths=supervised,
            codec_loss_mask=mask,
        )
        chosen = torch.tensor([True, False, True, False])
        # The chosen utterances alone: frames 0-4 and 8-14, first-codebook rows 0-5 and 10-17.
        alone = dict(
            frame_lengths=lengths[chosen],
            codes=torch.cat([codes[0:5], codes[8:15]]),
            supervised_frame_lengths=supervised[chosen],
            codec_loss_mask=torch.cat([mask[0:5], mask[8:15]]),
        )
        alone_predictions = {
            "first_logits": torch.cat(
                [predictions["first_logits"][r] for r in (slice(0, 6), slice(10, 18))]
            ),
            "residual_logits": [
                torch.cat([r[0:5], r[8:15]]) for r in predictions["residual_logits"]
            ],
        }
        for reduction in ("token", "sample", "sqrt"):
            whole = tts_loss(predictions, batch, 31, reduction, chosen)
            rest = tts_loss(predictions, batch, 31, reduction, ~chosen)["selected"]
            subset = tts_loss(alone_predictions, alone, 31, reduction)
            for key, value in subset.items():
                torch.testing.assert_close(whole["selected"][key].float(), value.float())
                torch.testing.assert_close(
                    whole[key].float(), (whole["selected"][key] + rest[key]).float()
                )
            torch.testing.assert_close(
                loss_normalizers(supervised, reduction),
                loss_normalizers(supervised[chosen], reduction)
                + loss_normalizers(supervised[~chosen], reduction),
            )


if __name__ == "__main__":
    unittest.main()
