"""Deduplicate all inventoried English/Chinese short views into codec work chunks."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from scripts.prepare_emilia_streaming import publish_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk-records', type=int, default=8192)
    args = parser.parse_args()
    root = args.output.resolve()
    inventory = root / 'INVENTORY_COMPLETE.json'
    assert inventory.is_file(), 'Complete the source inventory first'
    raw = root / 'raw'
    raw.mkdir(exist_ok=True)
    excluded_recordings = set()
    for kind in ('short', 'long', 'dialogue'):
        for path in (root / 'candidates' / kind).glob('asmr*.jsonl'):
            with path.open() as stream:
                excluded_recordings.update(json.loads(line)['source']['recording_id'] for line in stream)
    quality_source_sha256 = hashlib.sha256(Path(__file__).with_name('filter_emilia_dialogue.py').read_bytes()).hexdigest()
    recipe = {'source_type': 'short_long_quality_filtered_dialogue_short', 'languages': ['en', 'zh'],
              'selection': 'unique_id_prefer_short_then_long_then_dialogue',
              'excluded_source': 'ASMR directory and all its recording IDs',
              'excluded_recordings': len(excluded_recordings),
              'excluded_recordings_sha256': hashlib.sha256('\n'.join(sorted(excluded_recordings)).encode()).hexdigest(),
              'dialogue_quality_source_sha256': quality_source_sha256,
              'hours_per_language': None, 'chunk_records': args.chunk_records,
              'inventory_sha256': hashlib.sha256(inventory.read_bytes()).hexdigest()}
    if (root / 'export-recipe.json').exists():
        assert json.loads((root / 'export-recipe.json').read_text()) == recipe
    publish_json(root / 'export-recipe.json', recipe)
    if (root / 'EXPORT_COMPLETE.json').exists():
        print('Full export already complete', flush=True)
        return
    ids = set()
    speakers = {'en': set(), 'zh': set()}
    counts, hours, source_counts, source_hours, duplicates = (Counter() for _ in range(5))
    exclusions = Counter()
    maxima = Counter()
    minimum_seconds = float('inf')
    pending = []
    last_carrier = None
    chunks = 0

    def flush():
        nonlocal chunks
        if not pending:
            return
        target = raw / f'{chunks:06d}.jsonl'
        payload = ''.join(pending).encode()
        if target.exists():
            assert target.read_bytes() == payload, f'Exported chunk changed: {target}'
        else:
            temporary = target.with_suffix('.incomplete')
            temporary.write_bytes(payload)
            temporary.replace(target)
        chunks += 1
        pending.clear()
        publish_json(root / 'export-status.json', {'chunks': chunks, 'records': dict(counts), 'hours': dict(hours)})

    for kind in ('short', 'long', 'dialogue'):
        if kind == 'dialogue':
            print('Waiting for actual-audio dialogue quality filtering', flush=True)
            while not (root / 'DIALOGUE_QUALITY_COMPLETE.json').exists():
                time.sleep(5)
            quality = json.loads((root / 'DIALOGUE_QUALITY_COMPLETE.json').read_text())
            assert quality['recipe']['source_sha256'] == quality_source_sha256
            folder = root / 'dialogue-quality/kept'
        else:
            folder = root / 'candidates' / kind
        for path in sorted(folder.glob('*.jsonl')):
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    language = row['language']
                    if Path(row['audio']['archive']).parent.name == 'ASMR' or row['source']['recording_id'] in excluded_recordings:
                        exclusions[f'{kind}/{language}/ASMR'] += 1
                        continue
                    if kind == 'dialogue':
                        assert row['source']['quality']['method'] == 'dnsmos_p835_actual_slice'
                    if row['id'] in ids:
                        duplicates[f'{kind}/{language}'] += 1
                        continue
                    ids.add(row['id'])
                    carrier = row['audio']['archive'], row['audio']['offset']
                    if len(pending) >= args.chunk_records and carrier != last_carrier:
                        flush()
                    last_carrier = carrier
                    pending.append(line)
                    counts[language] += 1
                    hours[language] += row['duration'] / 3600
                    source_counts[f'{kind}/{language}'] += 1
                    source_hours[f'{kind}/{language}'] += row['duration'] / 3600
                    speakers[language].add(row['speaker'])
                    maxima[f'{language}_seconds'] = max(maxima[f'{language}_seconds'], row['duration'])
                    minimum_seconds = min(minimum_seconds, row['duration'])
                    maxima['text_characters'] = max(maxima['text_characters'], len(row['text']))
                    maxima['carrier_seconds'] = max(maxima['carrier_seconds'], row['audio']['frames'] / row['audio']['sample_rate'])
        print(json.dumps({'kind': kind, 'chunks': chunks, 'hours': dict(hours)}), flush=True)
    flush()
    report = {'chunks': chunks, 'records': dict(counts), 'hours': dict(hours),
              'source_records': dict(source_counts), 'source_hours': dict(source_hours),
              'speakers': {language: len(values) for language, values in speakers.items()},
              'maximum': dict(maxima), 'minimum_seconds': minimum_seconds,
              'duplicate_ids_skipped': sum(duplicates.values()),
              'duplicates_by_source': dict(duplicates), 'target_reached': dict.fromkeys(counts, True),
              'excluded': dict(exclusions), 'dialogue_quality': quality,
              'selection': recipe['selection']}
    publish_json(root / 'EXPORT_COMPLETE.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
