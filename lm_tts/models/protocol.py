"""Qwen condition layouts (non-streaming, and the official input-streaming overlay) and teacher-forcing positions."""

import torch
from torch import nn

# Training languages (ISO 639-1) and their codec language names.
LANGUAGE_NAMES = {
    "zh": "chinese",
    "en": "english",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "es": "spanish",
    "ja": "japanese",
    "ko": "korean",
    "fr": "french",
    "ru": "russian",
    "nl": "dutch",
    "pl": "polish",
}


def language_codec_ids(config_language_ids):
    """Map training languages (ISO codes) to the assembled config's codec language IDs."""
    missing = sorted(set(LANGUAGE_NAMES.values()) - set(config_language_ids))
    if missing:
        raise ValueError(
            f"Assembly lacks codec language IDs for {missing}; see extra_codec_languages"
        )
    return {code: config_language_ids[name] for code, name in LANGUAGE_NAMES.items()}


def frame_embeddings(talker, groups, codes):
    value = talker.model.codec_embedding(codes[:, 0])
    for g in range(1, groups):
        value = value + talker.code_predictor.get_input_embeddings()[g - 1](codes[:, g])
    return value


def build_inputs(talker, speaker_encoder, config, batch):
    text = talker.text_projection(talker.model.text_embedding(batch["text_ids"]))
    frames = frame_embeddings(talker, config.num_code_groups, batch["codes"])
    embedding = talker.model.codec_embedding
    bos = embedding.weight[config.codec_bos_id : config.codec_bos_id + 1]
    speaker = None
    use_speaker = getattr(config, "lm_tts_use_speaker_embedding", True)
    if use_speaker and "speaker_embeddings" in batch:
        if speaker_encoder is None or any(p.requires_grad for p in speaker_encoder.parameters()):
            raise ValueError("Cached speaker embeddings require a frozen speaker encoder")
        speaker = batch["speaker_embeddings"].to(text.dtype)
        if speaker.shape != (len(batch["frame_lengths"]), config.hidden_size):
            raise ValueError("Cached speaker embedding dimension disagrees with the model")
    elif use_speaker and speaker_encoder is not None:
        lengths = batch["speaker_lengths"].tolist()
        mel_rows = batch["speaker_mels"].split(lengths)
        groups = {}
        for index, length in enumerate(lengths):
            groups.setdefault(length, []).append(index)
        speaker = text.new_zeros(len(lengths), config.hidden_size)
        # ECAPA pooling must not see padding or other utterances.
        for indices in groups.values():
            speaker[indices] = speaker_encoder(torch.stack([mel_rows[i] for i in indices])).to(
                text.dtype
            )
    protocol = getattr(config, "lm_tts_input_protocol", "legacy_prefix")
    mixed = getattr(config, "lm_tts_input_layout", "non_streaming") == "mixed"
    if mixed != ("streaming" in batch):
        raise ValueError(
            "A mixed-layout model needs per-sample streaming flags, and only it accepts them"
        )
    streaming = batch["streaming"].tolist() if mixed else [False] * len(batch["frame_lengths"])
    if mixed and protocol != "qwen3_non_streaming":
        raise ValueError(
            "Streaming follows the Qwen3 prefill; use the qwen3_non_streaming protocol"
        )
    # Streaming rows overlay raw text on [cbos, C0, ...]; keep the embeddings before any padding.
    raw_text, raw_frames, raw_bos = text, frames, bos
    if protocol == "qwen3_non_streaming":
        text_pad = talker.text_projection(
            talker.model.text_embedding(batch["text_ids"].new_tensor([config.lm_tts_pad_token_id]))
        )
        role = talker.text_projection(
            talker.model.text_embedding(batch["text_ids"].new_tensor(config.lm_tts_role_ids))
        )
        auto = [config.codec_nothink_id, config.codec_think_bos_id, config.codec_think_eos_id]
        if getattr(config, "lm_tts_language_conditioning", "auto") == "codec_language_id":
            if "language_codec_ids" not in batch:
                raise ValueError("A language-conditioned model needs language_codec_ids")
            # Official explicit-language prefill: think, think_bos, language, think_eos.
            controls = [
                embedding(
                    batch["text_ids"].new_tensor(
                        auto
                        if language < 0
                        else [config.codec_think_id, config.codec_think_bos_id, language]
                        + [config.codec_think_eos_id]
                    )
                )
                + text_pad
                for language in batch["language_codec_ids"].tolist()
            ]
        else:
            controls = [embedding(batch["text_ids"].new_tensor(auto)) + text_pad] * len(
                batch["frame_lengths"]
            )
        text = text + embedding.weight[config.codec_pad_id]
        frames = frames + text_pad
        bos = bos + text_pad
        if speaker is not None:
            speaker = speaker + text_pad
    elif protocol != "legacy_prefix":
        raise ValueError(f"Unknown input protocol: {protocol}")
    pieces, positions, audio_positions, offsets = [], [], [], [0]
    text_rows = text.split(batch["text_lengths"].tolist())
    frame_rows = frames.split(batch["frame_lengths"].tolist())
    raw_text_rows = raw_text.split(batch["text_lengths"].tolist())
    raw_frame_rows = raw_frames.split(batch["frame_lengths"].tolist())
    for index, (text_row, frame_row) in enumerate(zip(text_rows, frame_rows)):
        # Official streaming: [bos] stays on the last prefix position (codec pad), then
        # text_row[1:] (text, eos) is added position by position to [cbos, C0, ...], then tts_pad.
        prefix_text = text_row[:1] if streaming[index] else text_row
        prefix = (
            [role, controls[index], prefix_text]
            if protocol == "qwen3_non_streaming"
            else [prefix_text]
        )
        if speaker is not None:
            prefix.insert(2 if protocol == "qwen3_non_streaming" else 1, speaker[index : index + 1])
        prefix_length = sum(len(part) for part in prefix)
        length = prefix_length + 1 + len(frame_row)
        if streaming[index]:
            overlay = raw_text_rows[index][1:]
            if "streaming_pause" in batch:
                # Late text: a run of tts_pad between streamed tokens, as an upstream stall gives.
                at, pause = batch["streaming_pause"][index].tolist()
                if pause:
                    overlay = torch.cat([overlay[:at], text_pad.expand(pause, -1), overlay[at:]])
            if len(overlay) > len(frame_row) + 1:
                raise ValueError("Streaming needs text that ends by the EOS position")
            overlay = torch.cat([overlay, text_pad.expand(len(frame_row) + 1 - len(overlay), -1)])
            pieces.extend([*prefix, torch.cat([raw_bos, raw_frame_rows[index]]) + overlay])
        else:
            pieces.extend([*prefix, bos, frame_row])
        positions.append(torch.arange(length, device=text.device))
        audio_positions.append(
            torch.arange(offsets[-1] + prefix_length, offsets[-1] + length, device=text.device)
        )
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=text.device)
    max_length = max(len(p) for p in positions)
    inputs = dict(
        inputs_embeds=torch.cat(pieces).unsqueeze(0),
        position_ids=torch.cat(positions).unsqueeze(0),
        cu_seq_lens_q=cu_seqlens,
        cu_seq_lens_k=cu_seqlens,
        max_length_q=max_length,
        max_length_k=max_length,
    )
    if config._attn_implementation == "sdpa":
        # Separate batch rows isolate utterances; SDPA accepts native FP32.
        sequences = inputs["inputs_embeds"][0].split([len(p) for p in positions])
        padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        position_ids = nn.utils.rnn.pad_sequence(positions, batch_first=True)
        mask = (
            torch.arange(max_length, device=text.device)[None, :]
            < torch.tensor([len(p) for p in positions], device=text.device)[:, None]
        )
        audio_positions = [p - offsets[i] + i * max_length for i, p in enumerate(audio_positions)]
        inputs = dict(inputs_embeds=padded, position_ids=position_ids, attention_mask=mask)
    return inputs, torch.cat(audio_positions)
