"""Where does codec input-embedding norm growth come from? Per checkpoint: median row norm of the 16 input
tables, direction stability (row cosine to the final checkpoint and to step 2.5k), norm vs codec usage, and
the first-layer / final RMSNorm gains."""
import json, sys, torch
from pathlib import Path
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
tok = safe_open('/workspace/workspace/yanglin/LM-TTS-Training/assets/base/Qwen3-TTS-Tokenizer-12Hz/model.safetensors', 'pt')
usage = [tok.get_tensor('decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage')] + \
        [tok.get_tensor(f'decoder.quantizer.rvq_rest.vq.layers.{g}._codebook.cluster_usage') for g in range(15)]
inputs = ['talker.model.codec_embedding.weight'] + [f'talker.code_predictor.model.codec_embedding.{g}.weight' for g in range(15)]
extra = ['talker.model.layers.0.input_layernorm.weight', 'talker.model.norm.weight']
def load(path):
    md = dcp.FileSystemReader(path / 'distributed').read_metadata()
    state = {'model.' + k: torch.empty(md.state_dict_metadata['model.' + k].size, dtype=torch.float32) for k in inputs + extra}
    dcp.load(state, checkpoint_id=path / 'distributed', no_dist=True)
    return {k[6:]: v for k, v in state.items()}
paths = list(map(Path, sys.argv[1:]))
W = {p: load(p) for p in paths}
final, early = W[paths[-1]], W[paths[0]]
def spearman(a, b):
    ra, rb = a.argsort().argsort().double(), b.argsort().argsort().double()
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])
for p in paths:
    w = W[p]; row = {'ckpt': p.parent.parent.name[-30:] + '/' + p.name}
    for name, sl in (('c0', [0]), ('rest', range(1, 16))):
        norms = [w[inputs[g]][:2048].norm(dim=-1) for g in sl]
        row[name + '_norm'] = round(float(torch.cat(norms).median()), 3)
        row[name + '_cos_final'] = round(float(torch.cat([torch.cosine_similarity(w[inputs[g]][:2048], final[inputs[g]][:2048]) for g in sl]).median()), 3)
        row[name + '_cos_early'] = round(float(torch.cat([torch.cosine_similarity(w[inputs[g]][:2048], early[inputs[g]][:2048]) for g in sl]).median()), 3)
        row[name + '_norm_vs_usage'] = round(sum(spearman(w[inputs[g]][:2048].norm(dim=-1), usage[g]) for g in sl) / len(sl), 3)
    row['layer0_in_gain'] = round(float(w[extra[0]].abs().mean()), 3)
    row['final_norm_gain'] = round(float(w[extra[1]].abs().mean()), 3)
    print(json.dumps(row), flush=True)
