"""Function-preserving initialization of the Qwen-shaped LFM text frontend."""

import torch


@torch.no_grad()
def initialize_paired_frontend(embedding, projection, native_weight):
    width = native_weight.shape[1]
    embedding.weight[:, :width].copy_(native_weight)
    embedding.weight[:, width:].copy_(-native_weight)
    # SiLU(x) - SiLU(-x) = x. Both halves remain independent trainable parameters.
    projection.linear_fc1.weight.copy_(torch.eye(2 * width, device=native_weight.device))
    projection.linear_fc1.bias.zero_()
    identity = torch.eye(width, device=native_weight.device)
    projection.linear_fc2.weight.copy_(torch.cat([identity, -identity], dim=1))
    projection.linear_fc2.bias.zero_()
