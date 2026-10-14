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


def test_speaker_generation_matches_training_prefix_next_frame():
    from test_lfm import speaker_model

    from lm_tts.data.batch import collate
    from lm_tts.inference.codec import generate_codes

    torch.set_num_threads(2)
    model = speaker_model().eval()
    text = [18, 25, 27, 19]
    prefix = torch.arange(32).reshape(2, 16)
    speaker = torch.linspace(-1, 1, model.config.hidden_size)
    settings = dict(
        do_sample=True,
        top_k=10,
        temperature=0.9,
        subtalker_dosample=True,
        subtalker_top_k=10,
        subtalker_temperature=0.9,
        repetition_penalty=1.05,
    )
    torch.manual_seed(42)
    expected = prefix.clone()
    with torch.inference_mode():
        for _ in range(2):
            batch = collate([{"text_ids": text, "codes": expected, "speaker_embedding": speaker}])
            frame, stop = model(batch, mode="next_frame", suppress_eos=True, generation=settings)
            assert not stop.item()
            expected = torch.cat([expected, frame])
    torch.manual_seed(42)
    actual = generate_codes(
        model,
        text,
        prefix,
        speaker_embedding=speaker,
        max_new_tokens=2,
        min_new_tokens=2,
        **settings,
    )
    torch.testing.assert_close(actual, expected[2:], atol=0, rtol=0)
    with pytest.raises(ValueError, match="conditioning configuration"):
        generate_codes(model, text, prefix)
    for bad in (torch.zeros(64), torch.ones(63), torch.full((64,), float("nan"))):
        with pytest.raises(ValueError, match="finite nonzero vector"):
            generate_codes(model, text, prefix, speaker_embedding=bad)


def test_speaker_frontend_matches_official_extraction(tmp_path):
    from types import SimpleNamespace

    import librosa
    import numpy as np
    import soundfile as sf
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSForConditionalGeneration,
        Qwen3TTSSpeakerEncoder,
    )
    from test_lfm import SPEAKER

    from lm_tts.inference.codec import extract_speaker_embedding

    encoder = Qwen3TTSSpeakerEncoder(
        Qwen3TTSSpeakerEncoderConfig(**{**SPEAKER, "mel_dim": 128})
    ).eval()
    path = tmp_path / "reference.wav"
    t = np.arange(16000) / 16000
    sf.write(
        path, np.stack([np.sin(2 * np.pi * 230 * t), np.sin(2 * np.pi * 410 * t)], axis=1), 16000
    )
    audio, sr = librosa.load(path, sr=24000, mono=True)
    wrapper = SimpleNamespace(
        speaker_encoder=encoder, device=torch.device("cpu"), dtype=torch.float32
    )
    expected = Qwen3TTSForConditionalGeneration.extract_speaker_embedding(wrapper, audio, sr)
    actual = extract_speaker_embedding(encoder, path)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
