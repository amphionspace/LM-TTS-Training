"""Teacher-forced first-codebook CE of an export on real utterances, streaming vs non-streaming (training layout)."""
import json, sys
sys.path.insert(0, "/workspace/workspace/yanglin/LM-TTS-Training")
import torch, torch.nn.functional as F
from qwen_tts import Qwen3TTSModel
from lm_tts.models.protocol import build_inputs

model_dir, manifest = sys.argv[1], sys.argv[2]
m = Qwen3TTSModel.from_pretrained(model_dir, device_map="cuda:0", dtype=torch.float32, attn_implementation="sdpa")
talker, cfg = m.model.talker, m.model.config
tok = m.processor.tokenizer if hasattr(m.processor, "tokenizer") else m.processor
enc = m.model.speaker_encoder.requires_grad_(False)
tcfg = talker.config
if getattr(tcfg, "lm_tts_input_layout", None) != "mixed":
    # Evaluate a non-mixed model with the same official layouts.
    tcfg.lm_tts_input_layout = "mixed"; tcfg.lm_tts_input_protocol = "qwen3_non_streaming"
    tcfg.lm_tts_pad_token_id = cfg.tts_pad_token_id; tcfg.lm_tts_role_ids = [151644, 77091, 198]
rows = [json.loads(l) for l in open(manifest)][:int(sys.argv[3])]
tot = {True: [0, 0], False: [0, 0]}
for r in rows:
    p = m.create_voice_clone_prompt(ref_audio=r["prompt_audio"], ref_text=r["prompt_text"], x_vector_only_mode=False)[0]
    codes = p.ref_code.to("cuda:0")
    ids = tok(f"<|im_start|>assistant\n{r['prompt_text']}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt")["input_ids"][0][3:-5].tolist()
    if len(ids) > 0.67 * len(codes):
        continue
    for s in (True, False):
        batch = dict(text_ids=torch.tensor([cfg.tts_bos_token_id, *ids, cfg.tts_eos_token_id], device="cuda:0"),
                     text_lengths=torch.tensor([len(ids) + 2]), codes=codes, frame_lengths=torch.tensor([len(codes)]),
                     speaker_embeddings=p.ref_spk_embedding.view(1, -1).to("cuda:0", torch.float32),
                     streaming=torch.tensor([s]), language_codec_ids=torch.tensor([-1]))
        with torch.no_grad():
            inputs, pos = build_inputs(talker, enc, talker.config, batch)
            h = talker.model(inputs_embeds=inputs["inputs_embeds"], position_ids=inputs["position_ids"],
                             attention_mask=inputs["attention_mask"]).last_hidden_state[0]
            logits = talker.codec_head(h[pos])
        target = torch.cat([codes[:, 0], codes.new_tensor([talker.config.codec_eos_token_id])])
        ce = F.cross_entropy(logits.float(), target, reduction="sum").item()
        tot[s][0] += ce; tot[s][1] += len(target)
print(model_dir.split("/")[-1], "n", len(rows), "streaming", round(tot[True][0] / tot[True][1], 3), "non_streaming", round(tot[False][0] / tot[False][1], 3))
