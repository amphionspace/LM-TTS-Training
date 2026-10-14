import torch
from lfm_text_embedding import load_st, QWEN, ROOT, project, dev
def g(W, center):
    W=W.to(dev).double(); W = W - W.mean(0) if center else W
    s=torch.linalg.svdvals(W); v=s**2/(s**2).sum(); c=v.cumsum(0)
    return round(float(torch.exp(-(v*v.log()).sum()))), int((c<.9).sum())+1, int((c<.99).sum())+1
q=load_st(QWEN,["talker.model.text_embedding.weight","talker.text_projection.linear_fc1.weight","talker.text_projection.linear_fc1.bias","talker.text_projection.linear_fc2.weight","talker.text_projection.linear_fc2.bias"])
E=q["talker.model.text_embedding.weight"]
b=load_st(ROOT/"assets/base/Qwen3-0.6B-Base/model.safetensors",["model.embed_tokens.weight"])["model.embed_tokens.weight"]
for c in (False,True): print("center",c,"qwenTTS",g(E,c),"base1024",g(b,c))
st={"talker.text_projection."+k: q["talker.text_projection."+k] for k in ["linear_fc1.weight","linear_fc1.bias","linear_fc2.weight","linear_fc2.bias"]}
import lfm_text_embedding as m
st={m.KEYS[2]:q[m.KEYS[2]],m.KEYS[3]:q[m.KEYS[3]],m.KEYS[4]:q[m.KEYS[4]],m.KEYS[5]:q[m.KEYS[5]]}
P=project(st,E[:151643])
print("qwenTTS proj out (regular vocab rows) centered", g(P,True), "norm", float(P.norm(dim=1).mean()))
