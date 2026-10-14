import glob, torch
from safetensors import safe_open
F = torch.nn.functional
def load(d, key):
    for f in sorted(glob.glob(f"{d}/*.safetensors")):
        with safe_open(f, "pt") as h:
            if key in h.keys(): return h.get_tensor(key).float()
M = "/workspace/model"
tts = load(f"{M}/Qwen3-TTS-12Hz-1.7B-Base", "talker.model.text_embedding.weight")[:151643].cuda()
base = load(f"{M}/Qwen3-0.6B-Base", "model.embed_tokens.weight")[:151643].cuda()
def knn(X, k=10):
    X = F.normalize(X - X.mean(0), dim=1); out = []
    for c in X.split(4096): out.append((c @ X.T).topk(k + 1, dim=1).indices[:, 1:])
    return torch.cat(out)
a, b = knn(tts), knn(base)
ov = (a.unsqueeze(2) == b.unsqueeze(1)).any(2).float().mean(1)
tn, bn = tts.norm(dim=1), base.norm(dim=1)
print("token-id decile | knn overlap TTS vs Qwen3-0.6B-Base | TTS row norm | Base row norm")
for i in range(10):
    s = slice(i * 15164, (i + 1) * 15164)
    print(f"{i*15164:>6}-{(i+1)*15164:<6} {ov[s].mean():.3f}  {tn[s].mean():.3f}  {bn[s].mean():.3f}")
# tokens that are pure non-speech symbols / code-ish: check a few known ids
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(f"{M}/Qwen3-0.6B-Base")
groups = {"code-ish": ["def", "):", "();", "=>", "#include", "</div>", "self.", "__init__"],
          "common en": ["the", " the", " and", " of", " is", " hello", " world"],
          "common zh": ["的", "我们", "中国", "时间", "你好"]}
for g, ws in groups.items():
    ids = [tok.convert_tokens_to_ids(tok.tokenize(w))[0] for w in ws]
    print(g, "overlap", round(float(ov[ids].mean()), 3), "TTS norm", round(float(tn[ids].mean()), 3))
