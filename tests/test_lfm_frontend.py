"""The wider text frontend must preserve initialization while remaining trainable."""

import pytest
import test_lfm as lfm_tests
import torch

from lm_tts.data.batch import collate
from lm_tts.models.lfm import LfmTTSModel
from lm_tts.training.optimizer import parameter_groups


@pytest.fixture
def wide_model():
    torch.set_num_threads(2)
    torch.manual_seed(71)
    return LfmTTSModel(lfm_tests.tiny_config("qwen-mlp")).eval()


def test_paired_initialization_preserves_model_outputs_and_audio_rng(wide_model):
    torch.manual_seed(71)
    native = LfmTTSModel(lfm_tests.tiny_config()).eval()
    for name, tensor in native.state_dict().items():
        if name != "talker.model.embed_tokens.weight":
            torch.testing.assert_close(wide_model.state_dict()[name], tensor, atol=0, rtol=0)
    weight = wide_model.talker.model.embed_tokens.weight
    width = native.config.hidden_size
    torch.testing.assert_close(
        weight[:, :width], native.talker.model.embed_tokens.weight, atol=0, rtol=0
    )
    torch.testing.assert_close(weight[:, width:], -weight[:, :width], atol=0, rtol=0)
    with torch.no_grad():
        for dtype, tolerance in [(torch.float32, 1e-6), (torch.bfloat16, 0.01)]:
            wide_model.to(dtype)
            actual = wide_model.talker.text_projection(wide_model.talker.model.embed_tokens.weight)
            expected = native.talker.model.embed_tokens.weight.to(dtype)
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        # Restore exact FP32 weights after the BF16 check.
        torch.manual_seed(71)
        restored = LfmTTSModel(lfm_tests.tiny_config("qwen-mlp")).eval()
        batch = collate(lfm_tests.rows())
        expected, actual = native(batch), restored(batch)
        torch.testing.assert_close(
            actual["first_logits"], expected["first_logits"], atol=2e-6, rtol=2e-5
        )
        for left, right in zip(actual["residual_logits"], expected["residual_logits"]):
            torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)


def test_wide_frontend_receives_gradients_and_uses_intended_lr(wide_model):
    settings = {"lr": 3e-4, "backbone_lr": 1e-4, "text_embedding_lr_group": "backbone"}
    groups = parameter_groups(wide_model, settings)
    embedding = wide_model.talker.model.embed_tokens.weight
    projector = wide_model.talker.text_projection
    assert id(embedding) in {id(p) for p in groups[0]["params"]}
    assert all(id(p) in {id(v) for v in groups[1]["params"]} for p in projector.parameters())
    lfm_tests.objective(wide_model, collate(lfm_tests.rows())).backward()
    for gradient in (
        embedding.grad[:, :64],
        embedding.grad[:, 64:],
        *(p.grad for p in projector.parameters()),
    ):
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    before = {name: p.detach().clone() for name, p in projector.named_parameters()}
    torch.optim.AdamW(groups).step()
    assert all(not torch.equal(before[name], p) for name, p in projector.named_parameters())
    assert not torch.equal(embedding[:, 64:], -embedding[:, :64])


@pytest.mark.parametrize("packed", [False, True])
def test_wide_frontend_packing(wide_model, packed):
    lfm_tests.test_batch_matches_individual_loss_and_gradients(wide_model, packed)


def test_wide_frontend_checkpointing_and_generation(wide_model):
    lfm_tests.test_checkpointing_and_state_roundtrip(wide_model)
    lfm_tests.test_teacher_forcing_matches_generated_frame(wide_model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA and FA2")
def test_wide_frontend_flash_packing(wide_model):
    lfm_tests.test_bf16_flash_packing(wide_model)
