"""How loud is the streamed text next to the frame it is added to? Per model: norms of projected text
tokens, tts_pad, the 16-codebook frame sum, codec_bos/pad, on real validation rows."""
import json, sys
from pathlib import Path
import torch
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from lm_tts.data.reader import FeatureDataset
from lm_tts.models.protocol import frame_embeddings, language_codec_ids

val, models = sys.argv[1], sys.argv[2:]
first = json.loads((Path(models[-1]) / "config.json").read_text())
ds = FeatureDataset(val, (first["tts_bos_token_id"], first["tts_eos_token_id"]), verify_integrity=False, mask_reference=True,
    text_vocab_size=first["talker_config"]["text_vocab_size"],
    language_codec_ids=language_codec_ids(first["talker_config"]["codec_language_id"]))
rows = [r for r in (ds[i] for i in range(200)) if r is not None][:150]
for path in models:
    talker = Qwen3TTSForConditionalGeneration.from_pretrained(path, dtype=torch.float32).talker.eval()
    cfg = talker.config
    with torch.no_grad():
        project = lambda ids: talker.text_projection(talker.model.text_embedding(torch.as_tensor(ids)))
        text = torch.cat([project(r["text_ids"][1:-1]) for r in rows])
        frames = torch.cat([frame_embeddings(talker, cfg.num_code_groups, torch.as_tensor(r["codes"])) for r in rows])
        c0 = torch.cat([talker.model.codec_embedding(torch.as_tensor(r["codes"])[:, 0]) for r in rows])
        specials = project([first["tts_pad_token_id"], first["tts_bos_token_id"], first["tts_eos_token_id"]])
        codec = talker.model.codec_embedding.weight
        n = lambda x: round(float(x.norm(dim=-1).median()), 3)
        print(json.dumps({"model": Path(path).name, "text_p50": n(text), "frame16_p50": n(frames), "c0_p50": n(c0),
            "text/frame": round(n(text) / n(frames), 3),
            "tts_pad,bos,eos": [round(float(v), 3) for v in specials.norm(dim=-1)],
            "codec_bos": round(float(codec[cfg.codec_bos_id].norm()), 3), "codec_pad": round(float(codec[cfg.codec_pad_id].norm()), 3)}), flush=True)
