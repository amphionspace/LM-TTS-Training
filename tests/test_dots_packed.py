"""Packed DiT must preserve the padded implementation's outputs and gradients."""

import copy

import pytest
import torch
from dots_padded_reference import padded_velocity
from test_dots import tiny_model

from lm_tts.models.dots.modules import DiT


@pytest.mark.parametrize("checked", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_packed_velocity_matches_padded_outputs_and_gradients(device, checked):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required for FlexAttention kernel")
    torch.manual_seed(901)
    model = tiny_model()
    model.velocity_field_predictor = DiT(32, 128, {**model.config["DiT"], "num_heads": 2})
    model = model.to(device).train()
    reference = copy.deepcopy(model)
    model.checkpointing = reference.checkpointing = checked
    # Includes 128-token tile boundaries and tiles shared by multiple utterances.
    lengths = torch.tensor([1, 29, 2, 17], device=device)
    n = int(lengths.sum())
    inputs = [
        torch.randn(n, 32, device=device),
        torch.randn(4 * n, 128, device=device),
        torch.randn(4 * n, 128, device=device),
        torch.rand(4, device=device),
        torch.randn(4, 32, device=device),
    ]
    results, grads = [], []
    for current, method in [(reference, padded_velocity), (model, type(model).velocity)]:
        args = [value.clone().requires_grad_() for value in inputs]
        output = method(current, *args, lengths)
        output.square().mean().backward()
        results.append(output.detach())
        grads.append([arg.grad for arg in args] + [p.grad for p in current.parameters()])
    torch.testing.assert_close(results[0], results[1], atol=3e-5, rtol=3e-4)
    for expected, actual in zip(*grads):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=5e-4)
