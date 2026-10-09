"""Correctness-first pure-codec generation using complete prefixes, without KV caching."""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ..models.loading import load_model


def validate_generation(settings):
    allowed = {
        "max_new_tokens",
        "min_new_tokens",
        "do_sample",
        "temperature",
        "top_k",
        "top_p",
        "repetition_penalty",
        "subtalker_dosample",
        "subtalker_temperature",
        "subtalker_top_k",
        "subtalker_top_p",
    }
    if set(settings) - allowed:
        raise ValueError(f"Unsupported generation settings: {sorted(set(settings) - allowed)}")
    maximum, minimum = settings.get("max_new_tokens", 2048), settings.get("min_new_tokens", 1)
    if (
        type(maximum) is not int
        or type(minimum) is not int
        or not 0 <= minimum <= maximum
        or maximum < 1
    ):
        raise ValueError("Require 0 <= min_new_tokens <= max_new_tokens and max_new_tokens > 0")
    for key in ("do_sample", "subtalker_dosample"):
        if key in settings and type(settings[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for prefix in ("", "subtalker_"):
        if (
            settings.get(prefix + "temperature", 1.0) <= 0
            or not 0 < settings.get(prefix + "top_p", 1.0) <= 1
        ):
            raise ValueError("Require positive temperature and 0 < top_p <= 1")
        top_k = settings.get(prefix + "top_k", 0)
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
    if settings.get("repetition_penalty", 1.0) <= 0:
        raise ValueError("repetition_penalty must be positive")


@torch.inference_mode()
def generate_codes(model, text_ids, reference_codes=None, **settings):
    validate_generation(settings)
    if getattr(model.config, "lm_tts_use_speaker_embedding", True):
        raise ValueError("Pure-codec generation requires a no-speaker model")
    device = next(model.parameters()).device
    text = torch.as_tensor(text_ids, device=device, dtype=torch.long)
    if (
        text.ndim != 1
        or not text.numel()
        or text.min() < 0
        or text.max() >= model.config.text_vocab_size
    ):
        raise ValueError("Text IDs must be a nonempty sequence within the model vocabulary")
    codes = (
        torch.empty((0, model.groups), dtype=torch.long, device=device)
        if reference_codes is None
        else torch.as_tensor(reference_codes, device=device, dtype=torch.long)
    )
    if (
        codes.ndim != 2
        or codes.shape[1] != model.groups
        or (codes.numel() and (codes.min() < 0 or codes.max() >= model.code_size))
    ):
        raise ValueError("Reference codec shape or vocabulary range is invalid")
    reference_frames = len(codes)
    batch = {"text_ids": text, "text_lengths": text.new_tensor([len(text)])}
    model.eval()
    for frame in range(settings.get("max_new_tokens", 2048)):
        batch.update(codes=codes, frame_lengths=text.new_tensor([len(codes)]))
        next_codes, stop = model(
            batch,
            mode="next_frame",
            suppress_eos=frame < settings.get("min_new_tokens", 1),
            generation=settings,
        )
        if stop.item():
            break
        codes = torch.cat([codes, next_codes])
    return codes[reference_frames:]


class CodecSynthesizer:
    def __init__(self, directory, device, dtype, backend):
        from qwen_tts import Qwen3TTSTokenizer

        directory = Path(directory)
        self.raw = json.loads((directory / "config.json").read_text())
        self.model = (
            load_model(directory, attn_implementation=backend).to(device=device, dtype=dtype).eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.codec = Qwen3TTSTokenizer.from_pretrained(
            str(directory / "speech_tokenizer"), device_map=device
        )

    @torch.inference_mode()
    def synthesize(self, text, reference_audio=None, reference_text=None, **settings):
        if reference_audio is not None and not (reference_text and reference_text.strip()):
            raise ValueError("Codec-prefix ICL requires reference_text matching reference_audio")
        if reference_audio is None and reference_text:
            raise ValueError("reference_text requires reference_audio")
        ids = [
            self.raw["tts_bos_token_id"],
            # Preserve the reference/target token boundary, as in official ICL.
            *self.tokenizer.encode(reference_text or "", add_special_tokens=False),
            *self.tokenizer.encode(text, add_special_tokens=False),
            self.raw["tts_eos_token_id"],
        ]
        prefix = self.codec.encode(str(reference_audio)).audio_codes[0] if reference_audio else None
        codes = generate_codes(self.model, ids, prefix, **settings)
        if not len(codes):
            raise ValueError("Model generated EOS before any audio frame")
        combined = codes if prefix is None else torch.cat([prefix.to(codes.device), codes])
        wavs, rate = self.codec.decode({"audio_codes": [combined]})
        # Decode the prefix too, preserving decoder context at the continuation boundary.
        trim = 0 if prefix is None else int(len(wavs[0]) * len(prefix) / len(combined))
        return wavs[0][trim:], rate
