import json
from pathlib import Path
from qwen3_train.metrics import ASRScorer

root = Path('data/ljspeech-test')
row = json.loads((root / 'val.jsonl').read_text().splitlines()[0])
scorer = ASRScorer('small.en')
result = {"id": row['id'], "reference_text": row['text'],
          "original": scorer.score(row['audio'], row['text']),
          "codec_reconstruction": scorer.score(root / 'codec-reconstruction.wav', row['text'])}
(root / 'asr-baseline.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
