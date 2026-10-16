"""Per-frame teacher-forced CE of an exported model on a reference build's validation split."""
import argparse, json, random
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
from lm_tts.data.batch import collate
from lm_tts.data.reader import FeatureDataset
from lm_tts.models.loading import load_model

p = argparse.ArgumentParser()
p.add_argument("--model"); p.add_argument("--build"); p.add_argument("--out")
p.add_argument("--samples", type=int, default=3000); p.add_argument("--batch", type=int, default=16)
a = p.parse_args()
cfg = json.loads((Path(a.model) / "config.json").read_text())
specials = (cfg["tts_bos_token_id"], cfg["tts_eos_token_id"])
ds = FeatureDataset(a.build, specials, mask_reference=True, verify_integrity=False,
                    text_vocab_size=cfg["talker_config"]["text_vocab_size"])
model = load_model(a.model, attn_implementation="flash_attention_2").to("cuda", torch.bfloat16).eval()
random.seed(0); idx = sorted(random.sample(range(len(ds)), a.samples))
rows = []
with torch.no_grad():
    for s in range(0, len(idx), a.batch):
        items = [ds[i] for i in idx[s:s + a.batch]]
        batch = collate(items)
        batch.pop("codec_loss_mask"); batch.pop("supervised_frame_lengths")
        batch = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch["speaker_embeddings"] = batch["speaker_embeddings"].to(torch.bfloat16)
        out = model(batch)
        lengths, codes = batch["frame_lengths"], batch["codes"]
        last = (lengths + 1).cumsum(0) - 1
        labels = codes.new_full((len(codes) + len(lengths),), model.eos)
        valid = torch.ones(len(labels), dtype=torch.bool, device=codes.device); valid[last] = False
        labels[valid] = codes[:, 0]
        first = F.cross_entropy(out["first_logits"].float(), labels, reduction="none")
        resid = torch.stack([F.cross_entropy(l.float(), codes[:, g + 1], reduction="none")
                             for g, l in enumerate(out["residual_logits"])]).mean(0)
        first_frames, eos = first[valid].cpu().numpy(), first[last].cpu().numpy()
        resid = resid.cpu().numpy(); off = 0
        for k, item in enumerate(items):
            T = item["num_frames"]; ra, rb = item["reference_interval"]
            for t in range(T):
                rows.append((s + k, t, T, ra, rb, first_frames[off + t], resid[off + t], -1.0))
            rows.append((s + k, T, T, ra, rb, eos[k], np.nan, 1.0))  # EOS row
            off += T
        if s % (a.batch * 20) == 0: print(s, len(rows), flush=True)
np.save(a.out, np.array(rows, dtype=np.float64))
print("done", len(rows))
