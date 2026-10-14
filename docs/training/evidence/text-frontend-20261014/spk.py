import json, sys, torch, random
from pathlib import Path
from safetensors import safe_open
sys.path.insert(0, "/workspace/workspace/yanglin/LM-TTS-Training")
from lm_tts.data.reader import FeatureDataset
F = torch.nn.functional
B = Path("/workspace/workspace/yanglin/LM-TTS-Training/data/builds")
M = Path("/workspace/workspace/yanglin/UltraEval-Audio/init_model")
A = Path("/workspace/workspace/yanglin/LM-TTS-Training/assets/assembled")
def spk(build, toks, n=400, mask=False):
    ds = FeatureDataset(B/build/"validation"/"manifest.json", toks, verify_integrity=False, mask_reference=mask)
    random.seed(0); idx = random.sample(range(len(ds)), n)
    return torch.stack([torch.as_tensor(ds[i]["speaker_embedding"]).float() for i in idx])
def frontend(path, emb_key):
    with safe_open(str(path/"model.safetensors"), "pt") as f:
        g = lambda k: f.get_tensor(k).float()
        p = "talker.text_projection."
        W = {k: g(p+k) for k in ["linear_fc1.weight","linear_fc1.bias","linear_fc2.weight","linear_fc2.bias"]}
        E = f.get_slice(emb_key); codec = g("talker.model.codec_embedding.weight")
        cps = [g(f"talker.code_predictor.model.codec_embedding.{i}.weight") for i in range(15)]
    cfg = json.load(open(path/"config.json")); tc = cfg.get("talker_config", cfg)
    pad = tc["lm_tts_pad_token_id"] if "lm_tts_pad_token_id" in tc else cfg["tts_pad_token_id"]
    proj = lambda x: F.silu(x@W["linear_fc1.weight"].T+W["linear_fc1.bias"])@W["linear_fc2.weight"].T+W["linear_fc2.bias"]
    t = proj(E[pad:pad+1].float())[0]
    torch.manual_seed(0)
    frames = codec[torch.randint(0,2048,(400,))] + sum(c[torch.randint(0,2048,(400,))] for c in cps)
    return t, frames
def report(name, s, t, frames):
    slot = s + t
    sc = s - s.mean(0)
    print(f"{name:34s} |spk| {s.norm(dim=1).mean():6.2f}  |text_pad| {t.norm():6.2f}  "
          f"speaker-specific part |s-mean| {sc.norm(dim=1).mean():5.2f}  cos(slot,text_pad) {F.cosine_similarity(slot,t[None]).mean():.3f}  "
          f"|audio frame| {(frames+t).norm(dim=1).mean():5.2f}")
all16 = spk("supervised-tts-20260929-merged-20261001-all16", [151672,151673])
lfm = spk("supervised-tts-20261012-emilia2-reference-lfm2.5-230m", [64403,64404], mask=True)
print("speaker emb dim", all16.shape, lfm.shape)
for name, path, key, s in [
    ("frozen-cond assembled(init)", A/"qwen3-tts-frozen-conditioning", "talker.model.text_embedding.weight", all16),
    ("frozen-cond 17500", M/"all16-frozen-conditioning-step-00017500", "talker.model.text_embedding.weight", all16),
    ("textbase assembled(init)", A/"qwen3-tts-text-base-trainable-random-projector", "talker.model.text_embedding.weight", all16),
    ("textbase 17500", M/"all16-textbase-step-00017500", "talker.model.text_embedding.weight", all16),
    ("textbase 25000", M/"all16-textbase-step-00025000", "talker.model.text_embedding.weight", all16),
    ("LFM assembled(init)", A/"lfm2.5-230m-base-text2048-spk", "talker.model.embed_tokens.weight", lfm),
    ("LFM 6000", M/"emilia2-lfm230m-text2048-spk-refmask-step-00006000", "talker.model.embed_tokens.weight", lfm),
]:
    try:
        t, fr = frontend(path, key); report(name, s, t, fr)
    except Exception as e: print(name, "ERR", repr(e)[:200])
