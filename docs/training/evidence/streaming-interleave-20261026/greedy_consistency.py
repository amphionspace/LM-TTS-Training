"""Greedy streaming generation via qwen_tts, then teacher-force the generated codes through the training layout:
argmax at each position must reproduce the generated first codebook if generation and training agree."""
import sys
sys.path.insert(0, "/workspace/workspace/yanglin/LM-TTS-Training")
import torch
from qwen_tts import Qwen3TTSModel
from lm_tts.models.protocol import build_inputs

model_dir, ref, mode = sys.argv[1], sys.argv[2], sys.argv[3]
text = "简单地说，这相当于惠普把消费领域市场拱手相让了。"
m = Qwen3TTSModel.from_pretrained(model_dir, device_map="cuda:0", dtype=torch.float32, attn_implementation="sdpa")
talker, cfg = m.model.talker, m.model.config
tcfg = talker.config
if getattr(tcfg, "lm_tts_input_layout", None) != "mixed":
    tcfg.lm_tts_input_layout = "mixed"; tcfg.lm_tts_input_protocol = "qwen3_non_streaming"
    tcfg.lm_tts_pad_token_id = cfg.tts_pad_token_id; tcfg.lm_tts_role_ids = [151644, 77091, 198]
tok = m.processor.tokenizer if hasattr(m.processor, "tokenizer") else m.processor
prompt = m.create_voice_clone_prompt(ref_audio=ref, x_vector_only_mode=True)
captured = {}
orig = m.model.speech_tokenizer.decode
def grab(encoded):
    captured["codes"] = encoded[0]["audio_codes"] if isinstance(encoded[0], dict) else encoded[0]
    return orig(encoded)
m.model.speech_tokenizer.decode = grab
streaming = mode == "stream"
wavs, sr = m.generate_voice_clone(text=[text], language=["Auto"], voice_clone_prompt=prompt, non_streaming_mode=not streaming,
                                  x_vector_only_mode=True, do_sample=False, repetition_penalty=1.0, subtalker_dosample=False, max_new_tokens=400)
codes = captured["codes"]
codes = (codes if torch.is_tensor(codes) else torch.as_tensor(codes)).to("cuda:0").long()
print("generated frames", codes.shape, "seconds", len(wavs[0]) / sr)
ids = tok(f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt")["input_ids"][0][3:-5].tolist()
batch = dict(text_ids=torch.tensor([cfg.tts_bos_token_id, *ids, cfg.tts_eos_token_id], device="cuda:0"),
             text_lengths=torch.tensor([len(ids) + 2]), codes=codes, frame_lengths=torch.tensor([len(codes)]),
             speaker_embeddings=prompt[0].ref_spk_embedding.view(1, -1).to("cuda:0", torch.float32),
             streaming=torch.tensor([streaming]), language_codec_ids=torch.tensor([-1]))
with torch.no_grad():
    inputs, pos = build_inputs(talker, m.model.speaker_encoder.requires_grad_(False), tcfg, batch)
    h = talker.model(inputs_embeds=inputs["inputs_embeds"], position_ids=inputs["position_ids"],
                     attention_mask=inputs["attention_mask"]).last_hidden_state[0]
    pred = talker.codec_head(h[pos]).argmax(-1)
target = torch.cat([codes[:, 0], codes.new_tensor([tcfg.codec_eos_token_id])])
agree = (pred == target).float()
print(mode, "argmax agreement", round(agree.mean().item(), 3), "first mismatches", (agree == 0).nonzero().flatten()[:10].tolist())
import soundfile as sf; sf.write(sys.argv[4], wavs[0], sr)
