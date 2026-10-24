"""R^2 vs codec RVQ codebooks and median row norm, for both the LM input embeddings and the output
heads (talker.codec_head, code_predictor.lm_head.*), plus the mean row cosine between the two.
Arguments are exported model dirs (safetensors) or DCP checkpoint dirs (no repo code)."""
import json, sys, torch
from pathlib import Path
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
t = safe_open('/workspace/workspace/yanglin/LM-TTS-Training/assets/base/Qwen3-TTS-Tokenizer-12Hz/model.safetensors', 'pt')
def cb(p):
    return t.get_tensor(p+'embedding_sum').double() / t.get_tensor(p+'cluster_usage').double().clamp(min=1e-5)[:, None]
C = [cb('decoder.quantizer.rvq_first.vq.layers.0._codebook.')] + [cb(f'decoder.quantizer.rvq_rest.vq.layers.{g}._codebook.') for g in range(15)]
def r2(X, E):
    X1 = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1); W = torch.linalg.lstsq(X1, E).solution; R = E - X1 @ W
    return float(1 - (R**2).sum() / ((E - E.mean(0))**2).sum())
inputs = ['talker.model.codec_embedding.weight'] + [f'talker.code_predictor.model.codec_embedding.{g}.weight' for g in range(15)]
heads = ['talker.codec_head.weight'] + [f'talker.code_predictor.lm_head.{g}.weight' for g in range(15)]
def load(path):
    if (path / 'distributed').is_dir():
        md = dcp.FileSystemReader(path / 'distributed').read_metadata()
        state = {'model.' + k: torch.empty(md.state_dict_metadata['model.' + k].size, dtype=torch.float32) for k in inputs + heads}
        dcp.load(state, checkpoint_id=path / 'distributed', no_dist=True)
        return {k[len('model.'):]: v for k, v in state.items()}
    out = {}
    for f in sorted(path.glob('*.safetensors')):
        h = safe_open(f, 'pt'); out.update({k: h.get_tensor(k) for k in h.keys() if k in inputs + heads})
    return out
for path in map(Path, sys.argv[1:]):
    w = load(path); row = {'model': str(path)}
    for name, keys in (('input', inputs), ('head', heads)):
        mats = [w[k].double()[:2048] for k in keys]
        row[name + '_r2'] = [round(r2(C[g], m), 4) for g, m in enumerate(mats)]
        row[name + '_norm'] = [round(float(m.norm(dim=1).median()), 3) for m in mats]
    row['input_head_cos'] = [round(float(torch.nn.functional.cosine_similarity(w[a].double()[:2048], w[b].double()[:2048], dim=1).mean()), 3) for a, b in zip(inputs, heads)]
    print(json.dumps(row), flush=True)
