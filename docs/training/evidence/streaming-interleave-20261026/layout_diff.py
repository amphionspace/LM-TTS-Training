"""Compare the training streaming layout (lm_tts build_inputs) with qwen_tts speaker-only streaming inference, position by position."""
import sys
sys.path.insert(0, "/workspace/workspace/yanglin/LM-TTS-Training")
import torch
from qwen_tts import Qwen3TTSModel
from lm_tts.models.protocol import build_inputs

model_dir, ref = sys.argv[1], sys.argv[2]
text = "Get the trust fund to the bank early."
m = Qwen3TTSModel.from_pretrained(model_dir, device_map="cuda:0", dtype=torch.float32, attn_implementation="sdpa")
talker = m.model.talker
captured = {}
def fake_generate(**kw):
    captured.update(kw); raise StopIteration
talker.generate = fake_generate
prompt = m.create_voice_clone_prompt(ref_audio=ref, x_vector_only_mode=True)
try:
    m.generate_voice_clone(text=[text], language=["Auto"], voice_clone_prompt=prompt, non_streaming_mode=False, x_vector_only_mode=True)
except StopIteration:
    pass
prefill, trailing, pad = captured["inputs_embeds"][0], captured["trailing_text_hidden"][0], captured["tts_pad_embed"][0, 0]
ids = m.processor(text=m._build_assistant_text(text), return_tensors="pt", padding=True)["input_ids"][0] if hasattr(m, "_build_assistant_text") else None
cfg = m.model.config
tok = m.processor.tokenizer if hasattr(m.processor, "tokenizer") else m.processor
full = tok(f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt")["input_ids"][0]
text_ids = full[3:-5].tolist()
N = 40
codes = torch.randint(0, 2048, (N, 16), device="cuda:0")
spk = prompt[0].ref_spk_embedding.view(1, -1).to("cuda:0", torch.float32)
batch = dict(text_ids=torch.tensor([cfg.tts_bos_token_id, *text_ids, cfg.tts_eos_token_id], device="cuda:0"),
             text_lengths=torch.tensor([len(text_ids) + 2]), codes=codes, frame_lengths=torch.tensor([N]),
             speaker_embeddings=spk, streaming=torch.tensor([True]), language_codec_ids=torch.tensor([-1]))
tc = talker.config
enc = m.model.speaker_encoder.requires_grad_(False)
with torch.no_grad():
    inputs, audio_pos = build_inputs(talker, enc, tc, batch)
train = inputs["inputs_embeds"][0]
# Inference teacher-forced sequence: prefill, then frame i embedding + trailing[i] (or pad).
def frame(c):
    v = talker.get_input_embeddings()(c[:1])[0]
    for g in range(1, 16):
        v = v + talker.code_predictor.get_input_embeddings()[g - 1](c[g:g + 1])[0]
    return v
gen = [frame(codes[i]) + (trailing[i] if i < len(trailing) else pad) for i in range(N)]
with torch.no_grad():
    gen = [frame(codes[i]) + (trailing[i] if i < len(trailing) else pad) for i in range(N)]
infer = torch.cat([prefill, torch.stack(gen)])
print("lengths train", len(train), "infer", len(infer), "prefill", len(prefill), "audio_pos start", audio_pos[0].item())
L = min(len(train), len(infer))
d = (train[:L] - infer[:L]).abs().max(-1).values
for i, v in enumerate(d.tolist()):
    print(i, round(v, 5))
