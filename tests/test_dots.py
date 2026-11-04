"""dots.tts layouts, packing isolation and causality on a tiny random model."""

import torch
from transformers import Qwen2Config

from lm_tts.data.dots import (
    PATCH_SIZE,
    DotsCollator,
    is_duplicate,
    reference_interval,
    utterance_key,
)
from lm_tts.models.dots import DotsTTSModel
from lm_tts.models.lora import lora_parameters
from lm_tts.objectives.flow import eos_weights, flow_loss

SPAN, START, END, TEXT_END, EOS = 90, 91, 92, 93, 94


class Layout:
    """The real layout logic over a tiny fixed vocabulary."""

    from lm_tts.data.dots import Layout as _Real

    ids = {"start": START, "span": SPAN, "end": END, "text_end": TEXT_END}
    prompts = {"text": [1, 2], "audio": [3, 4, 5], "interleave": [6, 7]}
    eos = EOS
    sequence = _Real.sequence


def official_interleave(text, patches):
    """The release's tokenizing.py loop: one text token, then one audio token, until both end."""
    full, audio = [6, 7], [SPAN] * patches + [END]
    t = a = 0
    placed = False
    while t < len(text) or a < len(audio):
        if t < len(text):
            full.append(text[t])
            t += 1
        elif not placed:
            full.append(TEXT_END)
            placed = True
        if a < len(audio):
            full.append(audio[a])
            a += 1
    return full + [EOS]


def test_interleave_matches_release_whenever_text_fits():
    for text_length, patches in [(1, 1), (3, 3), (2, 5), (4, 9)]:
        text = list(range(20, 20 + text_length))
        assert Layout().sequence(text, patches, True) == official_interleave(text, patches)


def test_every_patch_has_one_input_span_and_one_predictor():
    rows = [
        {
            "text_ids": [20, 21, 22],
            "vae_mean": torch.zeros(4 * PATCH_SIZE, 128),
            "vae_logstd": torch.zeros(4 * PATCH_SIZE, 128),
            "xvector": torch.zeros(512),
            "reference": (0, 0),
            "interleave": interleave,
            "duration": 1.0,
        }
        for interleave in (False, True)
    ]
    batch = DotsCollator(Layout())(rows)
    ids = batch["input_ids"]
    assert len(batch["span_positions"]) == len(batch["predictor_positions"]) == 8
    assert (ids[batch["span_positions"]] == SPAN).all()
    # basic: <start> predicts the first patch; interleave: the first text token does.
    first = batch["predictor_positions"][[0, 4]]
    assert ids[first[0]] == START and ids[first[1]] == 20


def test_eos_weights_put_half_on_the_last_patch_and_sum_to_one():
    targets, weights = eos_weights(torch.tensor([1, 3, 5]), torch.tensor([[0, 0], [0, 0], [1, 3]]))
    sums = torch.stack([w.sum() for w in weights.split([1, 3, 5])])
    torch.testing.assert_close(sums, torch.ones(3))
    assert targets.tolist() == [1, 0, 0, 1, 0, 0, 0, 0, 1]
    assert weights[3].item() == 0.5 and weights[8].item() == 0.5
    assert weights[5:7].tolist() == [0.0, 0.0] and weights[4].item() == 0.25


def test_reference_span_follows_the_codec_policy_and_takes_no_loss():
    starts = set()
    for patches in range(1, 400):
        interval = reference_interval(f"utt-{patches}", patches)
        if patches < 8:
            assert interval is None
            continue
        start, end = interval
        assert max(4, -(-patches // 10)) <= end - start <= patches // 2 and end <= patches
        starts.add(start > 0)
    assert starts == {True, False}
    lengths, reference = torch.tensor([10, 20]), torch.tensor([[3, 7], [14, 20]])
    batch = {"frame_lengths": lengths, "reference_intervals": reference}
    predictions = {
        "velocity": torch.randn(30, PATCH_SIZE, 128),
        "target": torch.randn(30, PATCH_SIZE, 128),
        "eos_logits": torch.randn(30, 2),
    }
    before = flow_loss(predictions, batch)
    for span in (slice(3, 7), slice(24, 29)):
        predictions["velocity"][span] += 5.0
        predictions["eos_logits"][span] += 5.0
    after = flow_loss(predictions, batch)
    assert after["patch_count"].item() == 20
    torch.testing.assert_close(after["fm_sum"], before["fm_sum"])
    torch.testing.assert_close(after["eos_sum"], before["eos_sum"])
    # A reference that reaches the end still keeps the utterance's EOS positive.
    predictions["eos_logits"][29, 1] += 5.0
    assert not torch.allclose(flow_loss(predictions, batch)["eos_sum"], after["eos_sum"])


def tiny_model():
    torch.manual_seed(0)
    llm = Qwen2Config(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    llm._attn_implementation = "sdpa"
    block = dict(num_layers=2, num_heads=4, hidden_size=32, ffn_hidden_size=64)
    config = dict(
        latent_dim=128,
        patch_size=PATCH_SIZE,
        cfg_droprate=0.2,
        xvec_drop_rate=0.2,
        campplus_embedding_size=512,
        PatchEncoder=block,
        DiT={**block, "qk_norm": True, "rotary_theta": 10000.0},
    )
    model = DotsTTSModel(config, llm, [0.1] * 128, [2.0] * 128).eval()
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    return model


def batch_of(rows):
    batch = DotsCollator(Layout())(rows)
    batch["input_ids"] = batch["input_ids"] % 100
    return batch


def row(patches, seed, text=(20, 21)):
    generator = torch.Generator().manual_seed(seed)
    frames = patches * PATCH_SIZE
    return {
        "text_ids": list(text),
        "vae_mean": torch.randn(frames, 128, generator=generator),
        "vae_logstd": torch.full((frames, 128), -20.0),
        "xvector": torch.randn(512, generator=generator),
        "reference": (0, 0),
        "interleave": False,
        "duration": 1.0,
    }


def predict(model, batch):
    torch.manual_seed(5)
    with torch.no_grad():
        return model(batch)


def test_packing_and_padding_do_not_mix_utterances():
    model = tiny_model()
    short, long = row(2, 1), row(5, 2)
    alone = predict(model, batch_of([short]))
    # Per-utterance draws differ with batch position, so compare the deterministic parts.
    packed = predict(model, batch_of([short, long]))
    torch.testing.assert_close(packed["eos_logits"][:2], alone["eos_logits"])
    model.velocity_field_predictor.time_embedder.mlp[0].weight.data.zero_()
    model.velocity_field_predictor.time_embedder.mlp[0].bias.data.zero_()

    def velocity(rows, noise):
        batch = batch_of(rows)
        lengths = batch["frame_lengths"]
        with torch.no_grad():
            latents = batch["vae_mean"]
            hidden = model.hidden(batch, model.encode_patches(latents, lengths))
            clean = (latents - model.latent_mean) / model.latent_std
            return model.velocity(
                model.hidden_proj(hidden),
                clean,
                noise[: len(clean)],
                torch.zeros(len(lengths)),
                model.xvec_proj(batch["xvectors"]),
                lengths,
            )

    noise = torch.randn(7 * PATCH_SIZE, 128)
    torch.testing.assert_close(
        velocity([short, long], noise)[:2], velocity([short], noise), atol=1e-5, rtol=1e-5
    )


def test_later_latents_cannot_change_earlier_predictions():
    model = tiny_model()
    original, changed = row(4, 3), row(4, 3)
    changed["vae_mean"][3 * PATCH_SIZE :] += 3.0
    a, b = predict(model, batch_of([original])), predict(model, batch_of([changed]))
    # Patch k is denoised from patches before k and its own noisy input only.
    torch.testing.assert_close(a["velocity"][:3], b["velocity"][:3])
    torch.testing.assert_close(a["eos_logits"][:3], b["eos_logits"][:3])
    assert not torch.allclose(a["velocity"][3], b["velocity"][3])


ADAPTED = {
    "lr": 1e-5,
    "adapter_lr": 1e-4,
    "embedding_rows": "added",
    "lora": {
        "modules": ["llm"],
        "targets": ["q_proj", "v_proj", "down_proj"],
        "rank": 4,
        "alpha": 8,
        "dropout": 0.0,
    },
}


def adapted_model():
    model = tiny_model()
    model.pretrained_rows = 90  # The last ten rows stand in for the tag tokens added at assembly.
    return model


def test_adapted_model_starts_pretrained_and_trains_only_new_parameters():
    model, batch = adapted_model(), batch_of([row(3, 4, text=(20, 95))])
    before = predict(model, batch)
    model.adapt(ADAPTED)
    after = predict(model, batch)
    for key in ("velocity", "eos_logits"):
        torch.testing.assert_close(after[key], before[key], atol=0, rtol=0)
    model.train()
    flow_loss(model(batch), batch)["fm_sum"].backward()
    llm = model.llm.model
    assert llm.embed_tokens.pretrained.grad is None
    assert llm.embed_tokens.added.grad[95 - 90].abs().sum() > 0
    assert llm.layers[0].self_attn.q_proj.base.weight.grad is None
    assert llm.layers[0].self_attn.q_proj.lora_B.grad.abs().sum() > 0
    pretrained, new = model.parameter_groups(ADAPTED)
    assert (pretrained["lr"], new["lr"], new["weight_decay"]) == (1e-5, 1e-4, 0.0)
    assert new["params"] == [*lora_parameters(model), llm.embed_tokens.added]
    assert all(p.requires_grad for p in pretrained["params"])


def test_merged_export_matches_the_adapted_model_under_the_original_names():
    model = adapted_model()
    names = set(model.state_dict())
    model.adapt(ADAPTED)
    with torch.no_grad():
        for parameter in model.parameter_groups(ADAPTED)[1]["params"]:
            parameter.normal_(std=0.2)
    batch = batch_of([row(3, 5, text=(20, 95))])
    adapted = predict(model, batch)
    # Export rebuilds the adapters on a fresh model, loads the checkpoint, then merges.
    restored = adapted_model()
    restored.adapt(ADAPTED)
    restored.load_state_dict(model.state_dict())
    restored.merge_adapters()
    assert set(restored.state_dict()) == names
    merged = predict(restored, batch)
    for key in ("velocity", "eos_logits"):
        torch.testing.assert_close(merged[key], adapted[key], atol=1e-5, rtol=1e-5)


def test_duplicates_match_text_and_length_not_tags_or_punctuation():
    seen = {}
    seen.setdefault(utterance_key("[Laughter]你好，美丽的世界！"), []).append(3.20)
    assert is_duplicate(seen, "你好美丽的世界", 3.45)
    assert not is_duplicate(seen, "你好美丽的世界", 3.60)
    assert not is_duplicate(seen, "你好，美丽的地球", 3.20)
    assert utterance_key("Hello, World [Sniff]") == "helloworld"
    # A lone tag leaves no text to compare.
    seen.setdefault(utterance_key("[Cough]"), []).append(1.41)
    assert not is_duplicate(seen, "[Breathing]", 1.41)
