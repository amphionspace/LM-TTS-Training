"""One generated frame from a training-protocol prefix, built on the model's public training
interfaces (input_embeddings, hidden, codec_head and the code predictor)."""

import torch

from .sampling import sample_token


def cached_hidden(model, batch, cache):
    """Hidden state predicting the next frame of one generation row, running only the positions
    past `cache`. Valid because every row of a growing frame prefix extends the previous one."""
    inputs, audio_positions = model.input_embeddings(batch)
    embeds, positions = inputs["inputs_embeds"], inputs["position_ids"]
    start, length = cache.get_seq_length(), embeds.shape[1]
    if embeds.shape[0] != 1 or audio_positions[-1].item() != length - 1 or start >= length:
        raise ValueError("Cached generation needs one row that ends at its next predictor")
    outputs = model.talker.model(
        inputs_embeds=embeds[:, start:],
        position_ids=positions[:, start:],
        past_key_values=cache,
        use_cache=True,
        cache_position=torch.arange(start, length, device=embeds.device),
    )
    return outputs.last_hidden_state[:, -1]


def next_frame(model, batch, *, suppress_eos=False, generation=None, cache=None):
    """Sample the next frame of every row; generation uses one unpadded sample on every rank."""
    if cache is None:
        hidden = model.hidden(batch)
        h = hidden[(batch["frame_lengths"] + 1).cumsum(0) - 1]
    else:
        h = cached_hidden(model, batch, cache)
    logits = model.talker.codec_head(h).float()
    allowed = torch.cat(
        [logits[:, : model.code_size], logits[:, model.eos : model.eos + 1]], dim=-1
    )
    if suppress_eos:
        allowed[:, -1] = -torch.inf
    settings = generation or {}
    first = sample_token(
        allowed,
        history=batch.get("history", batch["codes"][:, 0]).unsqueeze(0) if len(h) == 1 else None,
        **{
            key: settings[key]
            for key in ("do_sample", "temperature", "top_k", "top_p", "repetition_penalty")
            if key in settings
        },
    )
    stop = first == model.code_size
    codes = [first.clamp_max(model.code_size - 1)]
    inputs = [h.unsqueeze(1), model.talker.model.codec_embedding(codes[0]).unsqueeze(1)]
    predictor = model.talker.code_predictor
    for g in range(1, model.groups):
        out = predictor.model(
            inputs_embeds=predictor.small_to_mtp_projection(torch.cat(inputs, dim=1)),
            use_cache=False,
        )
        code = sample_token(
            predictor.lm_head[g - 1](out.last_hidden_state[:, -1]),
            do_sample=settings.get("subtalker_dosample", False),
            temperature=settings.get("subtalker_temperature", 1.0),
            top_k=settings.get("subtalker_top_k", 0),
            top_p=settings.get("subtalker_top_p", 1.0),
        )
        codes.append(code)
        if g < model.groups - 1:
            inputs.append(predictor.get_input_embeddings()[g - 1](code).unsqueeze(1))
    return torch.stack(codes, dim=-1), stop
