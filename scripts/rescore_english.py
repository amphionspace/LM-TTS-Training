"""Add English-normalized scores to saved transcripts without rerunning ASR."""
import argparse
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from qwen3_train.metrics import english_metrics, aggregate_content


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run')
    args = p.parse_args()
    root = Path(args.run)
    writer = SummaryWriter(str(root / 'tensorboard-english'))
    for tag, folder in [('eval', root), ('train_eval', root / 'train-evaluation')]:
        for step in sorted((folder / 'evaluation').glob('step-*')):
            rows = []
            for path in sorted(step.rglob('metrics.json')):
                row = json.loads(path.read_text())
                scored = {'id': row['id'], 'text': row['text']}
                for key in ['content', 'reference_asr']:
                    if key in row:
                        scored[key] = english_metrics(row['text'], row[key].get('transcript', ''))
                if 'content' in scored:
                    rows.append(scored)
            if not rows:
                continue
            summary = aggregate_content([r['content'] for r in rows])
            report = {'normalizer': 'transformers Whisper EnglishTextNormalizer; empty spelling map',
                      'summary': summary, 'samples': rows}
            (step / 'english-scores.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
            for key in ['wer', 'cer']:
                writer.add_scalar(f'{tag}/english_{key}', summary[key], int(step.name.split('-')[1]))
            print(json.dumps({'tag': tag, 'step': step.name, **summary}))
    writer.close()


if __name__ == '__main__':
    main()
