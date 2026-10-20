import json, torch, sys
from safetensors import safe_open
A='/workspace/workspace/yanglin/LM-TTS-Training/assets/assembled/qwen3-tts-frozen-conditioning'
B='/workspace/workspace/yanglin/LM-TTS-Training/assets/assembled/qwen3-tts-frozen-conditioning-codebook-codec'
def tensors(d):
    m=json.load(open(d+'/model.safetensors.index.json'))['weight_map']; files={}
    for k,f in m.items(): files.setdefault(f,[]).append(k)
    out={}
    for f,ks in files.items():
        h=safe_open(f'{d}/{f}','pt')
        for k in ks: out[k]=h.get_tensor(k)
    return out
a,b=tensors(A),tensors(B)
assert set(a)==set(b), 'key sets differ'
diff=[k for k in a if not torch.equal(a[k],b[k])]
expect={f'talker.code_predictor.model.codec_embedding.{g}.weight' for g in range(15)}|{'talker.model.codec_embedding.weight'}
print('tensors',len(a),'different',len(diff),'== expected 16 tables:',set(diff)==expect)
k='talker.model.codec_embedding.weight'; print('cb1 special rows 2048: identical', torch.equal(a[k][2048:],b[k][2048:]), 'code rows changed', not torch.equal(a[k][:2048],b[k][:2048]))
C0=t0=None
t=safe_open('/workspace/workspace/yanglin/LM-TTS-Training/assets/base/Qwen3-TTS-Tokenizer-12Hz/model.safetensors','pt')
def r2(X,E):
    X1=torch.cat([X,torch.ones(len(X),1,dtype=X.dtype)],1); W=torch.linalg.lstsq(X1,E).solution; R=E-X1@W
    return float(1-(R**2).sum()/((E-E.mean(0))**2).sum())
rs=[];stds=[]
for g in range(-1,15):
    p=f'decoder.quantizer.rvq_rest.vq.layers.{g}._codebook.' if g>=0 else 'decoder.quantizer.rvq_first.vq.layers.0._codebook.'
    C=t.get_tensor(p+'embedding_sum').double()/t.get_tensor(p+'cluster_usage').double().clamp(min=1e-5)[:,None]
    E=(b[f'talker.code_predictor.model.codec_embedding.{g}.weight'] if g>=0 else b['talker.model.codec_embedding.weight'][:2048]).double(); rs.append(r2(C,E)); stds.append(E.std().item())
print('R2 init min %.6f; std %.5f..%.5f; dtype %s'%(min(rs),min(stds),max(stds),b['talker.code_predictor.model.codec_embedding.0.weight'].dtype))
ra=json.load(open(A+'/assembly_report.json')); rb=json.load(open(B+'/assembly_report.json'))
print('report diff keys:',sorted(k for k in set(ra)|set(rb) if ra.get(k)!=rb.get(k)))
