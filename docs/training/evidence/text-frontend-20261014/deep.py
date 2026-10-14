import json, torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from lfm_text_embedding import ASM, QWEN, RUN, dev
F = torch.nn.functional
cfg = json.load(open(ASM.parent/"config.json"))
def keys(path, pred):
    with safe_open(str(path),"pt") as f: return {k:f.get_tensor(k).float() for k in f.keys() if pred(k)}
pred = lambda k: "embed_tokens" in k or "text_projection" in k or "codec_embedding" in k
init = keys(ASM, pred)
def ckpt(step):
    st = {k: torch.empty_like(v) for k,v in init.items()}
    dcp.load({"model": st}, checkpoint_id=RUN/f"checkpoints/step-{step:08d}/distributed"); return st
def analyze(name, E, fc1w, fc1b, fc2w, fc2b, rows):
    x = E[rows].to(dev); h = x@fc1w.to(dev).T+fc1b.to(dev); a = F.silu(h); y = a@fc2w.to(dev).T+fc2b.to(dev)
    # linear surrogate: replace SiLU by its tangent at 0 (x/2) -> how nonlinear is the MLP in practice
    ylin = (h/2)@fc2w.to(dev).T+fc2b.to(dev)
    # best linear fit from input to output (least squares, centered)
    X = torch.cat([x, torch.ones(len(x),1,device=dev)],1).double(); Y=y.double()
    W = torch.linalg.lstsq(X, Y).solution; r2 = 1-((X@W-Y)**2).sum()/((Y-Y.mean(0))**2).sum()
    print(f"{name}: |h| rms {h.pow(2).mean().sqrt():.3f}  frac|h|>1 {(h.abs()>1).float().mean():.4f}  "
          f"out norm {y.norm(dim=1).mean():.3f}  rel.err tangent-linear {((y-ylin).norm()/y.norm()):.3f}  best-linear R2 {float(r2):.4f}")
    return y
cp = cfg.get("codec_pad_id") or cfg["talker_config"]["codec_pad_id"] if "talker_config" in cfg else None
print("codec_pad_id", cp)
for step in [0, 1000, 6000]:
    st = init if step==0 else ckpt(step)
    E = st["talker.model.embed_tokens.weight"]
    rows = torch.arange(0, 64402)
    p = "talker.text_projection."
    y = analyze(f"LFM step{step}", E, st[p+"linear_fc1.weight"], st[p+"linear_fc1.bias"], st[p+"linear_fc2.weight"], st[p+"linear_fc2.bias"], rows)
    ce = st["talker.model.codec_embedding.weight"]
    cps = [st[f"talker.code_predictor.model.codec_embedding.{i}.weight"] for i in range(15)]
    print(f"   codec emb (cb0) row norm {ce[:2048].norm(dim=1).mean():.3f}; residual cb emb row norm mean {torch.stack([c.norm(dim=1).mean() for c in cps]).mean():.3f}")
q = keys(QWEN, lambda k: "text_embedding" in k or "text_projection" in k or "talker.model.codec_embedding" in k)
p="talker.text_projection."
analyze("QwenTTS", q["talker.model.text_embedding.weight"], q[p+"linear_fc1.weight"], q[p+"linear_fc1.bias"], q[p+"linear_fc2.weight"], q[p+"linear_fc2.bias"], torch.arange(0,151643))
print(f"   QwenTTS codec emb row norm {q['talker.model.codec_embedding.weight'][:2048].norm(dim=1).mean():.3f}")
