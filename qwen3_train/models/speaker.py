"""Deterministic Qwen speaker encoder padding."""

import torch


def _reflect_input(module, args):
    value = args[0]
    left, right = module._reversed_padding_repeated_twice
    if left or right:
        size = value.shape[-1]
        indices = torch.arange(-left, size + right, device=value.device)
        indices = torch.where(indices < 0, -indices, indices)
        indices = torch.where(indices >= size, 2 * size - 2 - indices, indices)
        value = value.index_select(-1, indices)
    return (value,)


def deterministic_speaker_padding(encoder):
    # CUDA reflection_pad1d backward uses atomic additions. Index selection has
    # a deterministic backward and preserves the official reflection samples.
    for module in encoder.modules():
        if isinstance(module, torch.nn.Conv1d) and module.padding_mode == "reflect":
            module.register_forward_pre_hook(_reflect_input)
            module.padding_mode = "zeros"
            module.padding = (0,)
