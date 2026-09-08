"""Score an audio file against known text; useful for codec and ASR baselines."""
import argparse
import json
from pathlib import Path
from qwen3_train.metrics import ASRScorer

p = argparse.ArgumentParser()
p.add_argument("--audio", required=True)
p.add_argument("--text", required=True)
p.add_argument("--asr-model", default="small.en")
p.add_argument("--output", required=True)
args = p.parse_args()
result = ASRScorer(args.asr_model).score(args.audio, args.text)
Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(json.dumps(result, indent=2, ensure_ascii=False))
