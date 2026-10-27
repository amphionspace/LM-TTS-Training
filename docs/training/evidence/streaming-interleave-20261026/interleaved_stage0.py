"""Stage-0 check on real data: interleaved eligibility rates, and the training forward + split
validation through flash attention on a real checkpoint (overlay-trained weights, so only the
plumbing and finiteness are meaningful)."""
import json, os, sys
from pathlib import Path
import torch, torch.distributed as dist
import torch.distributed.checkpoint as dcp
import lm_tts.evaluation.validation as V
from lm_tts.data.reader import FeatureDataset
from lm_tts.evaluation.validation import validate
from lm_tts.models.loading import load_model
from lm_tts.models.protocol import language_codec_ids
from lm_tts.training.config import load_config
from lm_tts.training.engine import STREAMING_KEYS

config_path, checkpoint = sys.argv[1], Path(sys.argv[2])
os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29733")
dist.init_process_group("nccl", rank=0, world_size=1)
config = load_config(config_path); settings = config["train"]
assembled = Path(config["model"]["assembled_model"])
full = json.loads((assembled / "config.json").read_text())
options = dict(language_codec_ids=language_codec_ids(full["talker_config"]["codec_language_id"]),
               language_auto_fraction=settings["language_auto_fraction"],
               **{k: settings[k] for k in STREAMING_KEYS if k in settings})
for name, build in (("val", config["data"]["val_build"]), ("train", config["data"]["build"])):
    ds = FeatureDataset(build, (full["tts_bos_token_id"], full["tts_eos_token_id"]), verify_integrity=False,
                        use_speaker_embedding=True, mask_reference=True,
                        text_vocab_size=full["talker_config"]["text_vocab_size"], **options)
    step = max(1, len(ds) // 2000)
    rows = [r for r in (ds[i] for i in range(0, len(ds), step)) if r is not None]
    print(name, len(rows), "eligible", round(sum(r["streaming_eligible"] for r in rows) / len(rows), 3),
          "streaming", round(sum(r["streaming"] for r in rows) / len(rows), 3), flush=True)
model = load_model(assembled, load_weights=False, attn_implementation="flash_attention_2", use_speaker_embedding=True)
model.config.lm_tts_language_conditioning = "codec_language_id"
model.config.lm_tts_input_layout = "mixed"
model.config.lm_tts_streaming_layout = "interleaved"
model.config.lm_tts_streaming_chunk = settings["streaming_chunk"]
state = {"model." + k: v for k, v in model.state_dict().items()}
dcp.load(state, checkpoint_id=checkpoint / "distributed", no_dist=True)
model.load_state_dict({k[6:]: v for k, v in state.items()})
model = model.cuda().to(torch.bfloat16).eval()
plan = V.val_batches
V.val_batches = lambda *a, **k: list(plan(*a, **k))[:10]
metrics = validate(model, ds if False else FeatureDataset(config["data"]["val_build"], (full["tts_bos_token_id"], full["tts_eos_token_id"]),
                   verify_integrity=False, use_speaker_embedding=True, mask_reference=True,
                   text_vocab_size=full["talker_config"]["text_vocab_size"], **options),
                   32, torch.device("cuda"), settings["loss_reduction"], settings["residual_weight"])
print(json.dumps({k: round(v, 4) for k, v in metrics.items() if k.endswith(("first_ce", "residual_ce", "loss"))}))
dist.destroy_process_group()
