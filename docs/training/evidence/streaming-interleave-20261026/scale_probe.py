"""Zero-training probe: scale the audio-code rows of all 16 codec input tables by s and rerun the
training validation (same reader, layouts and loss) on one GPU. Does not touch the run directory."""
import json, os, sys
from pathlib import Path
import torch, torch.distributed as dist
import torch.distributed.checkpoint as dcp
from lm_tts.data.reader import FeatureDataset
import lm_tts.evaluation.validation as V
from lm_tts.evaluation.validation import validate
BATCHES = int(sys.argv[4]) if len(sys.argv) > 4 else 0
if BATCHES:
    _plan = V.val_batches
    V.val_batches = lambda *a, **k: list(_plan(*a, **k))[:BATCHES]  # fixed leading subset, same for every scale
from lm_tts.models.loading import load_model
from lm_tts.models.protocol import language_codec_ids
from lm_tts.training.engine import STREAMING_KEYS

run, step, scales = Path(sys.argv[1]), int(sys.argv[2]), [float(s) for s in sys.argv[3].split(",")]
os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=os.environ.get("MASTER_PORT", "29731"))
dist.init_process_group("nccl", rank=0, world_size=1)
config = json.loads((run / "config.json").read_text())
settings, assembled = config["train"], Path(config["model"]["assembled_model"])
full = json.loads((assembled / "config.json").read_text())
options = dict(language_codec_ids=language_codec_ids(full["talker_config"]["codec_language_id"]),
               language_auto_fraction=settings.get("language_auto_fraction", 0.0))
if settings.get("streaming_fraction"):
    options.update({k: settings[k] for k in STREAMING_KEYS})
val = FeatureDataset(config["data"]["val_build"], (full["tts_bos_token_id"], full["tts_eos_token_id"]),
                     verify_integrity=False, use_speaker_embedding=True, mask_reference=settings.get("mask_reference", False),
                     text_vocab_size=full["talker_config"]["text_vocab_size"], **options)
model = load_model(assembled, load_weights=False, attn_implementation="sdpa", use_speaker_embedding=True)
model.config.lm_tts_language_conditioning = config["model"].get("language_conditioning", "auto")
if settings.get("streaming_fraction"):
    model.config.lm_tts_input_layout = "mixed"
state = {"model." + k: v for k, v in model.state_dict().items()}
dcp.load(state, checkpoint_id=run / f"checkpoints/step-{step:08d}/distributed", no_dist=True)
model.load_state_dict({k[6:]: v for k, v in state.items()})
model = model.cuda().eval()
tables = [model.talker.model.codec_embedding.weight] + [e.weight for e in model.talker.code_predictor.model.codec_embedding]
original = [t.detach().clone() for t in tables]
for s in scales:
    with torch.no_grad():
        for t, o in zip(tables, original):
            t.copy_(o); t[:2048].mul_(s)  # audio-code rows only; control/language rows untouched
    m = validate(model, val, config["eval"]["batch_size"], torch.device("cuda"), settings["loss_reduction"], settings["residual_weight"])
    keep = {k: round(v, 4) for k, v in m.items() if k.endswith(("first_ce", "residual_ce", "loss")) and not k.startswith("codebook")}
    print(json.dumps({"step": step, "scale": s, **keep}), flush=True)
dist.destroy_process_group()
