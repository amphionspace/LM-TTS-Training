"""Codec token selection shared by full-prefix generation backends."""

import torch


def sample_token(
    logits,
    *,
    do_sample=False,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    repetition_penalty=1.0,
    history=None,
):
    scores = logits.float().clone()
    if history is not None and history.numel() and repetition_penalty != 1:
        previous = scores.gather(1, history)
        scores.scatter_(
            1,
            history,
            torch.where(previous < 0, previous * repetition_penalty, previous / repetition_penalty),
        )
    if not do_sample:
        return scores.argmax(-1)
    scores /= temperature
    if top_k:
        cutoff = scores.topk(min(top_k, scores.shape[-1]), dim=-1).values[:, -1:]
        scores.masked_fill_(scores < cutoff, -torch.inf)
    if top_p < 1:
        ordered, indices = scores.sort(descending=True, dim=-1)
        remove = ordered.softmax(-1).cumsum(-1) > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        scores.masked_fill_(torch.zeros_like(remove).scatter(1, indices, remove), -torch.inf)
    return torch.multinomial(scores.softmax(-1), 1).squeeze(-1)
