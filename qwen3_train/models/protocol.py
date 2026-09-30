"""Qwen non-streaming condition layout and teacher-forcing positions."""

import torch
from torch import nn


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
    if "speaker_embeddings" in batch:
        if speaker_encoder is None or any(p.requires_grad for p in speaker_encoder.parameters()):
            raise ValueError("Cached speaker embeddings require a frozen speaker encoder")
        speaker = batch["speaker_embeddings"].to(text.dtype)
        if speaker.shape != (len(batch["frame_lengths"]), config.hidden_size):
            raise ValueError("Cached speaker embedding dimension disagrees with the model")
    elif speaker_encoder is not None:
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
    if protocol == "qwen3_non_streaming":
        text_pad = talker.text_projection(
            talker.model.text_embedding(batch["text_ids"].new_tensor([config.lm_tts_pad_token_id]))
        )
        role = talker.text_projection(
            talker.model.text_embedding(batch["text_ids"].new_tensor(config.lm_tts_role_ids))
        )
        controls = (
            embedding(
                batch["text_ids"].new_tensor(
                    [config.codec_nothink_id, config.codec_think_bos_id, config.codec_think_eos_id]
                )
            )
            + text_pad
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
    for index, (text_row, frame_row) in enumerate(zip(text_rows, frame_rows)):
        prefix = [role, controls, text_row] if protocol == "qwen3_non_streaming" else [text_row]
        if speaker is not None:
            prefix.insert(2 if protocol == "qwen3_non_streaming" else 1, speaker[index : index + 1])
        prefix_length = sum(len(part) for part in prefix)
        length = prefix_length + 1 + len(frame_row)
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
