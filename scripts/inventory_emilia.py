"""Inventory current Emilia carriers and export their English/Chinese short views."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import json
from pathlib import Path
import time


def scan(task):
    index, output = map(Path, task)
    report_path = output / 'inventory' / f'{index.stem}.json'
    if report_path.exists():
        return json.loads(report_path.read_text())
    members = {}
    for line in index.read_text().splitlines():
        name, offset, size = line.split('\t')
        assert name not in members
        members[name] = (int(offset), int(size))
    archive = index.with_suffix('')
    carriers, carrier_hours, shorts, short_hours, invalid = (Counter() for _ in range(5))
    maximum = Counter()
    with ExitStack() as stack:
        files = {kind: stack.enter_context((output / 'candidates' / kind / f'{index.stem}.incomplete').open('w'))
                 for kind in ('short', 'long', 'dialogue')}
        audio_file = stack.enter_context(archive.open('rb'))
        for name, (offset, size) in members.items():
            if not name.endswith('.json'):
                continue
            audio_file.seek(offset)
            meta = json.loads(audio_file.read(size))
            kind = meta['type']
            assert kind in files, kind
            languages = meta['languages']
            language_group = languages[0] if len(languages) == 1 else 'mixed'
            key = f'{kind}/{language_group}'
            carriers[key] += 1
            carrier_hours[key] += meta['frames'] / meta['sample_rate'] / 3600
            audio_name = name[:-5] + '.m4a'
            audio_offset, audio_size = members[audio_name]
            for item in meta['short']:
                language = item.get('language', 'unknown')
                key = f'{kind}/{language}'
                start, end = item['rel_start_samples'], item['rel_end_samples']
                if not 0 <= start < end <= meta['frames'] or not item.get('speaker') or not item.get('text', '').strip():
                    invalid[key] += 1
                    continue
                duration = (end - start) / meta['sample_rate']
                shorts[key] += 1
                short_hours[key] += duration / 3600
                maximum[key] = max(maximum[key], duration)
                if language not in ('en', 'zh'):
                    continue
                locator = {'kind': 'tar_member', 'archive': str(archive), 'member': audio_name,
                           'offset': audio_offset, 'size': audio_size, 'frames': meta['frames'],
                           'sample_rate': meta['sample_rate']}
                source = {'dataset': 'emilia2', 'type': kind, 'recording_id': meta['recording_id'],
                          'dnsmos': item.get('dnsmos')}
                if kind == 'short':
                    assert len(meta['short']) == 1 and start == 0 and end == meta['frames']
                    uid = meta['id']
                else:
                    locator.update(start_frame=start, end_frame=end)
                    source.update(view='short', carrier_id=meta['id'])
                    uid = item['id']
                row = {'schema_version': 1, 'id': 'emilia2:' + uid, 'text': item['text'].strip(),
                       'speaker': 'emilia2:' + item['speaker'], 'language': language,
                       'duration': duration, 'audio': locator, 'source': source}
                files[kind].write(json.dumps(row, ensure_ascii=False) + '\n')
    for kind in files:
        path = output / 'candidates' / kind / f'{index.stem}.incomplete'
        path.replace(path.with_suffix('.jsonl'))
    report = {'archive': str(archive), 'bytes': archive.stat().st_size,
              'carriers': dict(carriers), 'carrier_hours': dict(carrier_hours),
              'short_views': dict(shorts), 'short_view_hours': dict(short_hours),
              'invalid_short_views': dict(invalid), 'maximum_short_seconds': dict(maximum)}
    temporary = report_path.with_suffix('.incomplete')
    temporary.write_text(json.dumps(report))
    temporary.replace(report_path)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_m4a'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    folders = [args.source.resolve() / name for name in ('data', 'ASMR')]
    indices = sorted(path for folder in folders for path in folder.glob('*.tar.idx'))
    archives = {path for folder in folders for path in folder.glob('*.tar')}
    assert archives == {p.with_suffix('') for p in indices}, 'Tar/index coverage differs'
    output = args.output.resolve()
    (output / 'inventory').mkdir(parents=True, exist_ok=True)
    for kind in ('short', 'long', 'dialogue'):
        (output / 'candidates' / kind).mkdir(parents=True, exist_ok=True)
    counters = {key: Counter() for key in ('carriers', 'carrier_hours', 'short_views', 'short_view_hours', 'invalid_short_views')}
    maximum = Counter()
    total_bytes = 0
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(scan, (str(path), str(output))) for path in indices]
        for count, future in enumerate(as_completed(futures), 1):
            report = future.result()
            total_bytes += report['bytes']
            for key, counter in counters.items():
                counter.update(report[key])
            for key, value in report['maximum_short_seconds'].items():
                maximum[key] = max(maximum[key], value)
            if count % 50 == 0 or count == len(indices):
                status = {'scanned': count, 'total_shards': len(indices), 'elapsed_seconds': time.monotonic() - started,
                          'bytes': total_bytes, **{k: dict(v) for k, v in counters.items()},
                          'maximum_short_seconds': dict(maximum)}
                path = output / 'inventory-status.incomplete'
                path.write_text(json.dumps(status, indent=2))
                path.replace(output / 'inventory-status.json')
                print(json.dumps({'scanned': count, 'total': len(indices), 'seconds': status['elapsed_seconds']}), flush=True)
    (output / 'INVENTORY_COMPLETE.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
