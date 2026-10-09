"""LFM adaptation must preserve pretrained behavior and packed sample boundaries."""

import copy
import json

import pytest
import torch
from transformers import Lfm2Config, Lfm2Model

from lm_tts.data.batch import collate
from lm_tts.models.lfm import LfmTTSModel
from lm_tts.models.qwen import make_config
from lm_tts.objectives.tts import tts_loss
from lm_tts.training.optimizer import parameter_groups


def tiny_config():
    config = make_config(tiny=True)
    base = Lfm2Config(
        hidden_size=64,
        intermediate_size=128,
        block_ff_dim=128,
        block_auto_adjust_ff_dim=False,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        layer_types=["conv", "full_attention", "conv"],
        conv_L_cache=3,
    )
    config.lm_tts_lfm_config = base.to_dict()
    config.lm_tts_use_speaker_embedding = False
    config.lm_tts_input_protocol = "qwen3_non_streaming"
    config.lm_tts_role_ids = [6, 9]
    config.lm_tts_pad_token_id = 18
    config.codec_nothink_id = 21
    config.codec_think_bos_id = 22
    config.codec_think_eos_id = 23
    return config


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(71)
    return LfmTTSModel(tiny_config()).eval()


def rows():
    return [
        {"text_ids": [18, 25, 19], "codes": torch.arange(48).reshape(3, 16)},
        {"text_ids": [18, 27, 29, 30, 19], "codes": torch.arange(16).reshape(1, 16)},
    ]


def objective(model, batch):
    loss = tts_loss(model(batch), batch, model.eos)
    return loss["first_sum"] / 6 + 0.3 * loss["residual_sum"] / 60


def test_backbone_matches_native_weights_outputs_and_gradients(model):
    adapted = model.talker.model
    native = Lfm2Model(copy.deepcopy(adapted.config)).eval()
    state = {k: v for k, v in adapted.state_dict().items() if k != "codec_embedding.weight"}
    native.load_state_dict(state, strict=True)
    ids = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]])
    outputs = []
    for backbone in (native, adapted):
        output = backbone(input_ids=ids, use_cache=False).last_hidden_state
        outputs.append(output)
        output.square().sum().backward()
    torch.testing.assert_close(*outputs, atol=2e-6, rtol=2e-5)
    for name, parameter in native.named_parameters():
        torch.testing.assert_close(
            parameter.grad, dict(adapted.named_parameters())[name].grad, atol=3e-5, rtol=2e-3
        )


@pytest.mark.parametrize("packed", [False, True])
def test_batch_matches_individual_loss_and_gradients(model, packed):
    if packed:
        # Exercise the real packed convolution and SDPA's position-reset mask on CPU.
        # Only the protocol layout changes; the actual attention backend stays SDPA.
        model.config._attn_implementation = "flash_attention_2"
        model.config.code_predictor_config._attn_implementation = "sdpa"
    losses, gradients = [], []
    for batches in ([rows()], [[r] for r in rows()]):
        model.zero_grad(set_to_none=True)
        total = 0
        for rs in batches:
            loss = objective(model, collate(rs))
            loss.backward()
            total += loss.detach()
        losses.append(total)
        gradients.append({n: p.grad.clone() for n, p in model.named_parameters()})
    torch.testing.assert_close(*losses, atol=2e-6, rtol=2e-6)
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=2e-6, rtol=2e-3)


def test_packing_isolates_other_samples_and_future_audio(model):
    model.config._attn_implementation = "flash_attention_2"
    model.config.code_predictor_config._attn_implementation = "sdpa"
    original = rows()
    with torch.no_grad():
        expected = model.hidden(collate(original))
        changed = copy.deepcopy(original)
        changed[0]["codes"].fill_(80)
        changed[0]["text_ids"] = [35, 36, 37]
        torch.testing.assert_close(model.hidden(collate(changed))[4:], expected[4:], rtol=0, atol=0)
        changed = copy.deepcopy(original)
        changed[0]["codes"][1:].fill_(90)
        torch.testing.assert_close(model.hidden(collate(changed))[:2], expected[:2], rtol=0, atol=0)


def test_checkpointing_and_state_roundtrip(model):
    model.train()
    expected = objective(model, collate(rows()))
    expected.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters()}
    other = LfmTTSModel(tiny_config()).train()
    other.load_state_dict(model.state_dict(), strict=True)
    for module in (other.talker.model, other.talker.code_predictor.model):
        module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    actual = objective(other, collate(rows()))
    actual.backward()
    torch.testing.assert_close(actual, expected)
    for name, p in other.named_parameters():
        torch.testing.assert_close(p.grad, grads[name])


def test_pretrained_lr_and_no_duplicate_embedding(model):
    groups = parameter_groups(
        model, {"lr": 3e-4, "backbone_lr": 1e-4, "text_embedding_lr_group": "backbone"}
    )
    ids = {id(p) for p in groups[0]["params"]}
    assert id(model.talker.model.text_embedding.weight) in ids
    assert id(model.talker.model.embedding_norm.weight) in ids
    assert id(model.talker.model.codec_embedding.weight) not in ids
    assert not any("text_embedding" in key for key in model.state_dict())
    assert model.speaker_encoder is None


@pytest.mark.parametrize("capacity", [8, 16])
def test_offline_assembly_preserves_vocab_and_reloads(tmp_path, capacity):
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import AutoTokenizer, Lfm2ForCausalLM, PreTrainedTokenizerFast

    from lm_tts.models.assembly.lfm import assemble
    from lm_tts.models.loading import load_model

    base_config = Lfm2Config.from_dict(tiny_config().lm_tts_lfm_config)
    base_config.vocab_size = capacity
    base_config._attn_implementation = "sdpa"
    source = Lfm2ForCausalLM(base_config)
    backbone = tmp_path / "base"
    source.save_pretrained(backbone)
    vocab = {
        s: i
        for i, s in enumerate(
            [
                "<|pad|>",
                "<|startoftext|>",
                "<|im_end|>",
                "<|im_start|>",
                "[UNK]",
                "hello",
                "world",
                "test",
            ]
        )
    }
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
        bos_token="<|startoftext|>",
        eos_token="<|im_end|>",
        pad_token="<|pad|>",
    )
    tokenizer.save_pretrained(backbone)
    template = tmp_path / "template"
    template.mkdir()
    talker = tiny_config().to_dict()
    talker["code_predictor_config"]["num_hidden_layers"] = 5
    talker["code_predictor_config"]["layer_types"] = ["full_attention"] * 5
    (template / "config.json").write_text(
        json.dumps(
            {
                "tokenizer_type": "qwen3_tts_tokenizer_12hz",
                "talker_config": talker,
            }
        )
    )
    codec = tmp_path / "codec"
    codec.mkdir()
    (codec / "config.json").write_text("{}")
    save_file({"fixture": torch.ones(1)}, codec / "model.safetensors")
    output = tmp_path / "assembled"
    report = assemble(backbone, template, codec, output, dtype=torch.bfloat16)
    loaded = load_model(output, attn_implementation="sdpa")
    actual = loaded.talker.model.embed_tokens.weight[: len(vocab)]
    torch.testing.assert_close(
        actual, source.model.embed_tokens.weight[: len(vocab)].bfloat16().float(), rtol=0, atol=0
    )
    reloaded_tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
    assert all(reloaded_tokenizer.get_vocab()[s] == i for s, i in vocab.items())
    assert loaded.config.text_vocab_size == max(capacity, len(vocab) + 3)
    assert report["verification"]["exact_reload"]
    with pytest.raises(ValueError, match="pure codec"):
        config = tiny_config()
        config.lm_tts_use_speaker_embedding = True
        LfmTTSModel(config)


@torch.no_grad()
def test_teacher_forcing_matches_generated_frame(model):
    row = rows()[0]
    captures = []
    handle = model.talker.code_predictor.lm_head[-1].register_forward_hook(
        lambda module, args, out: captures.append(out.clone())
    )
    try:
        frame, _ = model(collate([{**row, "codes": row["codes"][:0]}]), mode="next_frame")
        generated = captures.pop()
        codes = row["codes"].clone()
        codes[0] = frame[0]
        model(collate([{**row, "codes": codes}]))
        torch.testing.assert_close(captures.pop()[:1], generated, atol=2e-6, rtol=2e-5)
    finally:
        handle.remove()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA and FA2")
def test_bf16_flash_packing(model):
    model = model.cuda().bfloat16()
    model.config._attn_implementation = "flash_attention_2"
    model.config.code_predictor_config._attn_implementation = "sdpa"
    model.talker.model.config._attn_implementation = "flash_attention_2"
    model.config.code_predictor_config._attn_implementation = "sdpa"
    values, grads = [], []
    for batches in ([rows()], [[r] for r in rows()]):
        model.zero_grad(set_to_none=True)
        total = 0
        for rs in batches:
            batch = {k: v.cuda() for k, v in collate(rs).items()}
            loss = objective(model, batch)
            loss.backward()
            total += loss.detach()
        values.append(total)
        grads.append(torch.cat([p.grad.float().flatten() for p in model.parameters()]))
    torch.testing.assert_close(*values, rtol=2e-3, atol=1e-3)
    assert (grads[0] - grads[1]).norm() / grads[0].norm() < 0.03
