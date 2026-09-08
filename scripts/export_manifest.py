"""Export source metadata into the common raw manifest (no codec work)."""
import argparse
import json
from pathlib import Path
from qwen3_train.sources import emilia_short, ljspeech


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['emilia-short', 'ljspeech'], required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int, default=128)
    p.add_argument('--max-shards', type=int, default=0)
    p.add_argument('--language')
    p.add_argument('--target-hours', type=float, default=0)
    p.add_argument('--min-seconds', type=float, default=1)
    p.add_argument('--max-seconds', type=float, default=10)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists() or args.limit < 1 or args.max_shards < 0:
        p.error('Use a new output and a positive pilot limit')
    output.parent.mkdir(parents=True, exist_ok=True)
    records = emilia_short(args.source, args.max_shards) if args.dataset == 'emilia-short' else ljspeech(args.source)
    ids, seconds, speakers = set(), 0., set()
    temporary = output.with_suffix('.incomplete')
    with temporary.open('w') as stream:
        for record in records:
            if args.language and record['language'] != args.language:
                continue
            if not args.min_seconds <= record['duration'] <= args.max_seconds:
                continue
            if record['id'] in ids:
                raise ValueError(f"Duplicate utterance: {record['id']}")
            ids.add(record['id'])
            speakers.add(record['speaker'])
            seconds += record['duration']
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            if len(ids) % 1000 == 0:
                print(json.dumps({"records": len(ids), "hours": seconds / 3600}), flush=True)
            if len(ids) >= args.limit or (args.target_hours and seconds >= args.target_hours * 3600):
                break
    if not ids:
        raise ValueError('No matching utterances')
    temporary.replace(output)
    output.with_suffix(".recipe.json").write_text(json.dumps(vars(args), indent=2))
    print(json.dumps({'records': len(ids), 'speakers': len(speakers), 'hours': seconds / 3600,
                      'output': str(output), 'dataset': args.dataset}))


if __name__ == '__main__':
    main()
