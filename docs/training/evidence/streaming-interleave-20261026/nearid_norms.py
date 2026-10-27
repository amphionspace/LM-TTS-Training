"""Text/codec input norms across DCP checkpoints (no export needed): projected text tokens on real
validation rows, special tokens, 16-codebook frame sums, text/frame ratio, raw text-row drift and
projector deviation from its near-identity init."""
import json, sys
from pathlib import Path
import torch, torch.nn.functional as F
import torch.distributed.checkpoint as dcp
from lm_tts.data.reader import FeatureDataset

assembled, val, ckpts = Path(sys.argv[1]), sys.argv[2], list(map(Path, sys.argv[3:]))
full = json.loads((assembled / "config.json").read_text())
ds = FeatureDataset(val, (full["tts_bos_token_id"], full["tts_eos_token_id"]), verify_integrity=False,
                    mask_reference=True, text_vocab_size=full["talker_config"]["text_vocab_size"])
rows = [r for r in (ds[i] for i in range(200)) if r is not None][:150]
text_ids = torch.cat([torch.as_tensor(r["text_ids"][1:-1]) for r in rows])
codes = torch.cat([torch.as_tensor(r["codes"]) for r in rows])
specials = torch.tensor([full["tts_pad_token_id"], full["tts_bos_token_id"], full["tts_eos_token_id"]])
used = torch.unique(text_ids)
P = "model.talker."
keys = [P + "model.text_embedding.weight", P + "model.codec_embedding.weight"] + \
       [P + f"code_predictor.model.codec_embedding.{g}.weight" for g in range(15)]
proj_keys = [P + f"text_projection.linear_fc{i}.{w}" for i in (1, 2) for w in ("weight", "bias")]
first_text = None
for c in ckpts:
    md = dcp.FileSystemReader(c / "distributed").read_metadata()
    want = [k for k in keys + proj_keys if k in md.state_dict_metadata]
    st = {k: torch.empty(md.state_dict_metadata[k].size, dtype=torch.float32) for k in want}
    dcp.load(st, checkpoint_id=c / "distributed", no_dist=True)
    E = st[keys[0]]
    def project(x):
        if proj_keys[0] not in st:
            return x
        h = F.silu(x @ st[proj_keys[0]].T + st[proj_keys[1]])
        return h @ st[proj_keys[2]].T + st[proj_keys[3]]
    text = project(E[text_ids]); sp = project(E[specials])
    frame = st[keys[1]][codes[:, 0]] + sum(st[keys[2 + g]][codes[:, g + 1]] for g in range(15))
    med = lambda x: round(float(x.norm(dim=-1).median()), 3)
    row = {"ckpt": c.parent.parent.name[:40] + "/" + c.name, "text_proj": med(text), "tts_pad,bos,eos": [round(float(v), 3) for v in sp.norm(dim=-1)],
           "frame16": med(frame), "c0": med(st[keys[1]][codes[:, 0]]), "text/frame": round(med(text) / med(frame), 3),
           "raw_text_rows_used": med(E[used])}
    if first_text is None:
        first_text = E[used].clone()
    row["raw_text_cos_to_first"] = round(float(F.cosine_similarity(E[used], first_text).median()), 3)
    if proj_keys[0] in st and st[proj_keys[2]].shape[0] == st[proj_keys[2]].shape[1]:
        I = torch.eye(st[proj_keys[0]].shape[0])
        row["fc1_dev"] = round(float((st[proj_keys[0]] - I).norm() / I.norm()), 3)
        row["fc2_dev"] = round(float((st[proj_keys[2]] - 2 * I).norm() / (2 * I).norm()), 3)
    print(json.dumps(row), flush=True)
