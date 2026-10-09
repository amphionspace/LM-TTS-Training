"""Sampling and prefix continuation must honor the caller's generation settings."""

import pytest
import torch

from lm_tts.inference.codec import validate_generation
from lm_tts.models.sampling import sample_token


def test_sampling_filters_and_repetition_penalty():
    logits = torch.tensor([[2.0, 3.0, 1.0]])
    assert sample_token(logits).item() == 1
    assert sample_token(logits, history=torch.tensor([[1]]), repetition_penalty=2).item() == 0
    for _ in range(10):
        assert sample_token(logits, do_sample=True, top_k=1).item() == 1
        assert sample_token(logits, do_sample=True, top_p=0.1).item() == 1
    assert torch.equal(logits, torch.tensor([[2.0, 3.0, 1.0]]))


@pytest.mark.parametrize(
    "settings",
    [
        {"temperature": 0},
        {"top_p": 0},
        {"top_k": -1},
        {"subtalker_temperature": 0},
        {"max_new_tokens": 0},
        {"min_new_tokens": 5, "max_new_tokens": 4},
        {"typo": True},
    ],
)
def test_invalid_generation_settings_fail_before_forward(settings):
    with pytest.raises(ValueError):
        validate_generation(settings)


def test_icl_preserves_text_token_boundary_and_trims_reference():
    from types import SimpleNamespace

    import numpy as np
    from test_lfm import tiny_config
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from lm_tts.inference.codec import CodecSynthesizer
    from lm_tts.models.lfm import LfmTTSModel

    synth = CodecSynthesizer.__new__(CodecSynthesizer)
    synth.raw = {"tts_bos_token_id": 18, "tts_eos_token_id": 19}
    synth.model = LfmTTSModel(tiny_config()).eval()
    synth.tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[UNK]": 1, "hello": 2, "world": 3, "helloworld": 4}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
    )

    class Codec:
        def encode(self, path):
            return SimpleNamespace(audio_codes=[torch.zeros(2, 16, dtype=torch.long)])

        def decode(self, batch):
            return [np.zeros(len(batch["audio_codes"][0]) * 1920)], 24000

    synth.codec = Codec()
    seen = []
    hook = synth.model.register_forward_pre_hook(
        lambda model, inputs: seen.append(inputs[0]["text_ids"].tolist())
    )
    wav, rate = synth.synthesize(
        "world", "reference.wav", "hello", max_new_tokens=1, min_new_tokens=1
    )
    hook.remove()
    assert seen == [[18, 2, 3, 19]]
    assert len(wav) == 1920 and rate == 24000
    with pytest.raises(ValueError, match="requires reference_text"):
        synth.synthesize("world", "reference.wav")
