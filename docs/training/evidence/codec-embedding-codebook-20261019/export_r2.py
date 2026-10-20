"""R^2 of exported/official codec embeddings against the codec RVQ codebooks (affine fit, plus a shuffled-row baseline)."""
import json, sys, torch
from safetensors import safe_open
ROOT = '/workspace/workspace/yanglin'
t = safe_open(f'{ROOT}/LM-TTS-Training/assets/base/Qwen3-TTS-Tokenizer-12Hz/model.safetensors', 'pt')
def cb(p):
    return t.get_tensor(p+'embedding_sum').double() / t.get_tensor(p+'cluster_usage').double().clamp(min=1e-5)[:, None]
C = [cb('decoder.quantizer.rvq_first.vq.layers.0._codebook.')] + [cb(f'decoder.quantizer.rvq_rest.vq.layers.{g}._codebook.') for g in range(15)]
def r2(X, E):
    X1 = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1); W = torch.linalg.lstsq(X1, E).solution; R = E - X1 @ W
    return float(1 - (R**2).sum() / ((E - E.mean(0))**2).sum())
models = {
    'official': f'{ROOT}/LM-TTS-Training/assets/base/Qwen3-TTS-12Hz-0.6B-Base/model.safetensors',
    'refmask-10000': f'{ROOT}/UltraEval-Audio/init_model/emilia2-refmask-step-00010000/model.safetensors',
    'refmask-34257': f'{ROOT}/UltraEval-Audio/init_model/emilia2-refmask-step-00034257/model.safetensors',
    'prefixref-30000': f'{ROOT}/UltraEval-Audio/init_model/emilia2-prefixref-step-00030000/model.safetensors',
}
torch.manual_seed(0); shuffle = torch.randperm(2048)
for name, path in models.items():
    f = safe_open(path, 'pt'); k0 = next(k for k in f.keys() if k.endswith('talker.model.codec_embedding.weight'))
    pre = k0[:-len('talker.model.codec_embedding.weight')]
    tables = [f.get_tensor(k0 if g == 0 else f'{pre}talker.code_predictor.model.codec_embedding.{g-1}.weight').double()[:2048] for g in range(16)]
    print(json.dumps({'model': name, 'r2': [round(r2(C[g], E), 4) for g, E in enumerate(tables)],
                      'shuffled_baseline_cb6': round(r2(C[5][shuffle], tables[5]), 4)}), flush=True)
