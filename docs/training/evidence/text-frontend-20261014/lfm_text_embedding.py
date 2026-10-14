"""Section 5.2-style geometry of the LFM text2048 frontend: init vs trained checkpoints."""
import json
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open

ROOT = Path("/workspace/workspace/yanglin/LM-TTS-Training")
ASM = ROOT / "assets/assembled/lfm2.5-230m-base-text2048-spk/model.safetensors"
QWEN = ROOT / "assets/base/Qwen3-TTS-12Hz-0.6B-Base/model.safetensors"
RUN = Path("/workspace/workspace/yanglin/LM-TTS-Training-Runs/"
           "supervised-tts-20261012-emilia2-lfm230m-text2048-spk-refmask-bf16-8gpu-acc2-lr3e-4-bblr1e-4-ep3-wsd")
KEYS = ["talker.model.embed_tokens.weight", "talker.model.embedding_norm.weight",
        "talker.text_projection.linear_fc1.weight", "talker.text_projection.linear_fc1.bias",
        "talker.text_projection.linear_fc2.weight", "talker.text_projection.linear_fc2.bias"]
dev = "cuda"


def load_st(path, keys):
    with safe_open(str(path), "pt") as f:
        return {k: f.get_tensor(k).float() for k in keys}


def load_ckpt(step):
    init = load_st(ASM, KEYS)
    state = {k: torch.empty_like(v) for k, v in init.items()}
    dcp.load({"model": state}, checkpoint_id=RUN / f"checkpoints/step-{step:08d}/distributed")
    return state


def geometry(W):
    W = W.to(dev).double()
    s = torch.linalg.svdvals(W - W.mean(0))
    var = s**2 / (s**2).sum()
    p = s / s.sum()
    cum = var.cumsum(0)
    return {
        "eff_rank_s": float(torch.exp(-(p * p.clamp_min(1e-300).log()).sum())),
        "eff_rank_s2": float(torch.exp(-(var * var.clamp_min(1e-300).log()).sum())),
        "dim90": int((cum < 0.90).sum()) + 1,
        "dim99": int((cum < 0.99).sum()) + 1,
        "norm_mean": float(W.norm(dim=1).mean()),
    }


def project(state, E):
    fc1w, fc1b = state[KEYS[2]].to(dev), state[KEYS[3]].to(dev)
    fc2w, fc2b = state[KEYS[4]].to(dev), state[KEYS[5]].to(dev)
    out = []
    for chunk in E.to(dev).split(8192):
        out.append(torch.nn.functional.silu(chunk @ fc1w.T + fc1b) @ fc2w.T + fc2b)
    return torch.cat(out)


def knn(X, k=10):
    X = torch.nn.functional.normalize(X.to(dev), dim=1)
    idx = []
    for chunk in X.split(4096):
        sim = chunk @ X.T
        idx.append(sim.topk(k + 1, dim=1).indices[:, 1:])
    return torch.cat(idx)


def overlap(a, b):
    k = a.shape[1]
    hits = (a.unsqueeze(2) == b.unsqueeze(1)).any(2).sum(1)
    return float(hits.float().mean() / k)


def main():
    steps = [int(s) for s in sys.argv[1:]]
    init = load_st(ASM, KEYS)
    E0 = init[KEYS[0]]
    width = E0.shape[1] // 2
    report = {}
    qwen = load_st(QWEN, ["talker.model.text_embedding.weight"])["talker.model.text_embedding.weight"]
    report["qwen_tts_2048_all_rows"] = geometry(qwen)

    P0 = project(init, E0)
    for step in steps:
        st = load_ckpt(step)
        E = st[KEYS[0]]
        # Rows never seen only decay uniformly (AdamW); touched rows deviate from the global scale.
        cos = torch.nn.functional.cosine_similarity(E.to(dev), E0.to(dev), dim=1)
        touched = (cos < 1 - 1e-5).cpu()
        Et, E0t = E[touched], E0[touched]
        A, B = Et[:, :width], Et[:, width:]
        diff, sym = (A - B) / 2, (A + B) / 2
        P = project(st, E)
        Pt, P0t = P[touched.to(dev)], P0[touched.to(dev)]
        # Linear surrogate of the trained projector: how far from the identity path it moved.
        fc1, fc2 = st[KEYS[2]], st[KEYS[4]]
        eye1 = torch.eye(2 * width)
        fc2_0 = torch.cat([torch.eye(width), -torch.eye(width)], 1)
        nn_init, nn_now = knn(E0t[:, :width]), knn(Et)
        nn_p0, nn_p = knn(P0t), knn(Pt)
        drift = torch.nn.functional.cosine_similarity(Pt, P0t, dim=1)
        report[f"step{step}"] = {
            "touched_rows": int(touched.sum()),
            "row_cos_vs_init_touched_median": float(cos[touched.to(dev)].median()),
            "row_cos_vs_init_touched_p10": float(cos[touched.to(dev)].quantile(0.10)),
            "embed_init_touched": geometry(E0t),
            "embed_now_touched": geometry(Et),
            "embed_now_all": geometry(E),
            "diff_component(A-B)/2": geometry(diff),
            "sym_component(A+B)/2": geometry(sym),
            "sym_over_diff_energy": float((sym.norm() / diff.norm()) ** 2),
            "fc1_rel_dev_from_identity": float((fc1 - eye1).norm() / eye1.norm()),
            "fc2_rel_dev_from_pair": float((fc2 - fc2_0).norm() / fc2_0.norm()),
            "proj_out_init_touched": geometry(P0t),
            "proj_out_now_touched": geometry(Pt),
            "proj_out_cos_vs_init_median": float(drift.median()),
            "proj_out_cos_vs_init_p10": float(drift.quantile(0.10)),
            "knn10_overlap_embed_vs_init": overlap(nn_now, nn_init),
            "knn10_overlap_projout_vs_init": overlap(nn_p, nn_p0),
        }
        print(json.dumps(report[f"step{step}"], indent=1), flush=True)
    print(json.dumps(report["qwen_tts_2048_all_rows"]))
    out = Path(__file__).with_name("lfm_text_embedding.json")
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
