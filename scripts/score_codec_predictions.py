"""Decode checkpoint inspection outputs and compare them with original audio."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSTokenizer
from qwen3_train.metrics import ASRScorer, content_metrics

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--diagnostics', required=True)
    p.add_argument('--codec', required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    root = Path(args.diagnostics)
    results = json.loads((root / 'summary.json').read_text())
    codec = Qwen3TTSTokenizer.from_pretrained(args.codec, device_map=args.device)
    codec.model.eval().requires_grad_(False)
    for row in results:
        row['decoded'] = {}
        with np.load(root / f'{row["split"]}-codes.npz') as codes:
            for kind in codes.files:
                if not len(codes[kind]):
                    row['decoded'][kind] = None
                    continue
                audio, sr = codec.decode({'audio_codes': [torch.from_numpy(codes[kind].astype(np.int64))]})
                path = root / f'{row["split"]}-{kind}.wav'
                sf.write(path, audio[0], sr)
                row['decoded'][kind] = str(path)
    del codec
    torch.cuda.empty_cache()
    print('GPU codec decoding complete', flush=True)
    scorer = ASRScorer('small.en')
    for row in results:
        row['asr'] = {kind: scorer.score(path, row.get('generation_texts', {}).get(kind, row['text'])) if path else content_metrics(row.get('generation_texts', {}).get(kind, row['text']), '')
                      for kind, path in {'original': row['audio'], **row['decoded']}.items()}
        print(json.dumps({'split': row['split'], 'id': row['id'],
                          'wer': {kind: score['wer'] for kind, score in row['asr'].items()}}), flush=True)
    (root / 'scored.json').write_text(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
