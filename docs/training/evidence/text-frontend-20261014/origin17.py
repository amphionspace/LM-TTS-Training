import glob, torch
from safetensors import safe_open
def load(d, key):
    for f in sorted(glob.glob(f"{d}/*.safetensors")):
        with safe_open(f, "pt") as h:
            if key in h.keys(): return h.get_tensor(key).float()
    raise KeyError(key)
M="/workspace/model"
tts = load(f"{M}/Qwen3-TTS-12Hz-1.7B-Base", "talker.model.text_embedding.weight")
asr = load(f"{M}/Qwen3-ASR-1.7B", "thinker.model.embed_tokens.weight")
n = min(len(tts), len(asr), 151643)
cos = torch.nn.functional.cosine_similarity(tts[:n], asr[:n], dim=1)
q = torch.tensor([0.5, 0.9, 0.99, 0.999, 1.0])
print("rows", n, "cos quantiles", [round(float(x),4) for x in cos.quantile(q[:-1])], "max", round(float(cos.max()),4))
print("frac cos>0.1", float((cos>0.1).float().mean()), "frac cos>0.3", float((cos>0.3).float().mean()))
# random-direction yardstick for 2048-dim vectors: std ~ 1/sqrt(2048)
print("1/sqrt(2048) =", round(2048**-0.5, 4))
# rows the TTS model likely never trained (tiny norm change heuristic not available) -> check by ASR norm decile
norms = asr[:n].norm(dim=1)
for lo, hi in [(0,0.1),(0.45,0.55),(0.9,1.0)]:
    a, b = norms.quantile(lo), norms.quantile(hi)
    m = (norms>=a)&(norms<=b)
    print(f"ASR-norm quantile {lo}-{hi}: median cos {float(cos[m].median()):.4f}")
