"""Diagnostic only: where do the Qwen3-TTS text embedding / projector / Talker weights come from?"""

import glob

import torch
from safetensors import safe_open

M = "/workspace/model"
A = "/workspace/workspace/yanglin/LM-TTS-Training/assets/base"
dev = "cuda" if torch.cuda.is_available() else "cpu"


def load(directory, name):
    for f in sorted(glob.glob(f"{directory}/*.safetensors")):
        with safe_open(f, "pt") as h:
            if name in h.keys():
                return h.get_tensor(name).float().to(dev)
    raise KeyError(f"{directory}: {name}")


def compare_rows(x, y, label):
    if x.shape != y.shape:
        print(f"{label}: shape mismatch {tuple(x.shape)} vs {tuple(y.shape)}")
        return
    cos = torch.nn.functional.cosine_similarity(x, y, dim=1)
    exact = (x == y).all(dim=1).float().mean()
    print(
        f"{label}: exact-equal rows {exact:.4f} | row cosine median {cos.median():.4f} "
        f"mean {cos.mean():.4f} | rows cos>0.99 {(cos > 0.99).float().mean():.4f} | "
        f"rows cos<0.5 {(cos < 0.5).float().mean():.4f} | max |diff| {(x - y).abs().max():.4g}"
    )


def compare_flat(x, y, label):
    if x.shape != y.shape:
        print(f"{label}: shape mismatch {tuple(x.shape)} vs {tuple(y.shape)}")
        return
    cos = torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0)
    print(f"{label}: flat cosine {cos:.4f} | exact {bool((x == y).all())} | max |diff| {(x - y).abs().max():.4g}")


tts06 = f"{A}/Qwen3-TTS-12Hz-0.6B-Base"
tts17 = f"{M}/Qwen3-TTS-12Hz-1.7B-Base"
tts17cv = f"{M}/Qwen3-TTS-12Hz-1.7B-CustomVoice"
asr17 = f"{M}/Qwen3-ASR-1.7B"
asr06 = f"{M}/Qwen3-ASR-0.6B"
base06 = f"{A}/Qwen3-0.6B-Base"
inst06 = f"{M}/Qwen3-0.6B"

TE = "talker.model.text_embedding.weight"
print("### text embedding tables")
e06 = load(tts06, TE)
e17 = load(tts17, TE)
e17cv = load(tts17cv, TE)
a17 = load(asr17, "thinker.model.embed_tokens.weight")
compare_rows(e06, e17, "TTS-0.6B text_embedding vs TTS-1.7B text_embedding")
compare_rows(e17, e17cv, "TTS-1.7B-Base vs TTS-1.7B-CustomVoice text_embedding")
compare_rows(e06, a17, "TTS-0.6B text_embedding vs Qwen3-ASR-1.7B embed_tokens (Qwen3-1.7B family, 2048)")
compare_rows(e17, a17, "TTS-1.7B text_embedding vs Qwen3-ASR-1.7B embed_tokens")
compare_rows(load(asr17, "thinker.lm_head.weight"), a17, "ASR-1.7B lm_head vs its embed_tokens (tied?)")

b06 = load(base06, "model.embed_tokens.weight")
compare_rows(b06, load(inst06, "model.embed_tokens.weight"), "Qwen3-0.6B-Base vs Qwen3-0.6B(instruct) embed_tokens")
compare_rows(b06, load(asr06, "thinker.model.embed_tokens.weight"), "Qwen3-0.6B-Base vs Qwen3-ASR-0.6B embed_tokens")
print("  (the last two show how much a Qwen3 fine-tune moves embedding rows; use as a yardstick)")

print()
print("### text projector")
for name in ("linear_fc1.weight", "linear_fc1.bias", "linear_fc2.weight", "linear_fc2.bias"):
    compare_flat(load(tts06, f"talker.text_projection.{name}"), load(tts17, f"talker.text_projection.{name}"), f"TTS-0.6B vs TTS-1.7B text_projection.{name}")
w1 = load(tts06, "talker.text_projection.linear_fc1.weight")
print(f"TTS-0.6B fc1 (2048x2048): flat cosine to identity {torch.nn.functional.cosine_similarity(w1.flatten(), torch.eye(2048, device=dev).flatten(), dim=0):.4f}, weight std {w1.std():.4f}")

print()
print("### Talker backbone vs text LMs (layer 0 / layer 14 q_proj, and final norm)")
for layer in (0, 14, 27):
    compare_flat(load(tts06, f"talker.model.layers.{layer}.self_attn.q_proj.weight"), load(base06, f"model.layers.{layer}.self_attn.q_proj.weight"), f"TTS-0.6B talker layer{layer} q_proj vs Qwen3-0.6B-Base")
    compare_flat(load(tts06, f"talker.model.layers.{layer}.mlp.down_proj.weight"), load(base06, f"model.layers.{layer}.mlp.down_proj.weight"), f"TTS-0.6B talker layer{layer} down_proj vs Qwen3-0.6B-Base")
    compare_flat(load(tts17, f"talker.model.layers.{layer}.self_attn.q_proj.weight"), load(asr17, f"thinker.model.layers.{layer}.self_attn.q_proj.weight"), f"TTS-1.7B talker layer{layer} q_proj vs Qwen3-ASR-1.7B")
compare_flat(load(asr06, "thinker.model.layers.0.self_attn.q_proj.weight"), load(base06, "model.layers.0.self_attn.q_proj.weight"), "Qwen3-ASR-0.6B layer0 q_proj vs Qwen3-0.6B-Base (yardstick for a fine-tune)")

print()
print("### codec embeddings shared between TTS sizes?")
compare_rows(load(tts06, "talker.model.codec_embedding.weight"), load(tts17, "talker.model.codec_embedding.weight"), "TTS-0.6B vs TTS-1.7B talker codec_embedding")
