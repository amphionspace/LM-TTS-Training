"""Prepare Emilia utterances with overlapping export and codec workers."""
import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import ExitStack
import hashlib
import heapq
import io
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import time


def publish_json(path, value):
    temporary = path.with_suffix('.incomplete')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def take_ready_batch(pending):
    while True:
        for index, batch in enumerate(pending):
            if all(future.done() for _, future, _ in batch):
                del pending[index]
                return batch
        wait({future for batch in pending for _, future, _ in batch if not future.done()},
             return_when=FIRST_COMPLETED)


def export(args):
    from qwen3_train.sources import emilia_short

    root = Path(args.output)
    raw = root / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    existing = sorted(raw.glob('*.jsonl'))
    if existing and not args.resume_export:
        raise ValueError('Raw chunks already exist; do not restart a live export')
    recipe = {'source': str(Path(args.source).resolve()),
              'hours_per_language': args.hours_per_language, 'languages': ['en', 'zh'],
              'source_type': 'short_and_long_short' if args.include_long_shorts else 'short',
              'min_seconds': None, 'max_seconds': None, 'allow_shortfall': args.allow_shortfall,
              'chunk_records': args.chunk_records,
              'selection': 'existing_prefix_then_sorted_shards' if args.include_long_shorts else 'sorted_shard_order'}
    recipe_path = root / 'export-recipe.json'
    if recipe_path.exists() and json.loads(recipe_path.read_text()) != recipe:
        raise ValueError('Export recipe changed; use a new output directory')
    publish_json(recipe_path, recipe)
    seconds = Counter()
    counts = Counter()
    ids = set()
    pending = []
    chunks = len(existing)
    started = time.monotonic()
    duplicates = 0

    def previous_rows():
        for index, path in enumerate(existing):
            if path.name != f'{index:06d}.jsonl':
                raise ValueError('Raw chunk sequence has a gap')
            with path.open() as stream:
                for line in stream:
                    yield json.loads(line)

    if args.include_long_shorts:
        for row in previous_rows():
            if row['id'] in ids:
                raise ValueError(f'Duplicate existing ID: {row["id"]}')
            ids.add(row['id'])
            counts[row['language']] += 1
            seconds[row['language']] += row['duration']
    previous = iter(()) if args.include_long_shorts else iter(previous_rows())
    expected = next(previous, None)

    def flush():
        nonlocal chunks
        if not pending:
            return
        destination = raw / f'{chunks:06d}.jsonl'
        temporary = destination.with_suffix('.incomplete')
        with temporary.open('w') as stream:
            for row in pending:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        temporary.replace(destination)
        chunks += 1
        pending.clear()
        status = {'chunks': chunks, 'records': dict(counts),
                  'hours': {k: v / 3600 for k, v in seconds.items()},
                  'duplicate_ids_skipped': duplicates,
                  'elapsed_seconds': time.monotonic() - started}
        publish_json(root / 'export-status.json', status)
        print(json.dumps(status), flush=True)

    records = emilia_short(args.source, workers=args.export_workers, include_long_shorts=args.include_long_shorts)
    if all(seconds[k] >= args.hours_per_language * 3600 for k in ('en', 'zh')):
        records = ()
    for row in records:
        language = row['language']
        if language not in ('en', 'zh') or seconds[language] >= args.hours_per_language * 3600:
            continue
        if row['id'] in ids:
            if args.include_long_shorts:
                duplicates += 1
                continue
            raise ValueError(f'Duplicate source ID: {row["id"]}')
        if args.include_long_shorts and len(pending) >= args.chunk_records:
            last, current = pending[-1]['audio'], row['audio']
            if (last['archive'], last['offset']) != (current['archive'], current['offset']):
                flush()
        ids.add(row['id'])
        counts[language] += 1
        seconds[language] += row['duration']
        if expected is not None:
            if row != expected:
                raise ValueError(f'Exported prefix changed at {row["id"]}')
            expected = next(previous, None)
        else:
            pending.append(row)
        if not args.include_long_shorts and len(pending) >= args.chunk_records:
            flush()
        if all(seconds[k] >= args.hours_per_language * 3600 for k in ('en', 'zh')):
            break
    flush()
    if expected is not None:
        raise ValueError('Source ended before the existing exported prefix')
    reached = {k: seconds[k] >= args.hours_per_language * 3600 for k in ('en', 'zh')}
    if not all(reached.values()) and not args.allow_shortfall:
        raise ValueError(f'Insufficient independent shorts: {dict(seconds)} seconds')
    publish_json(root / 'EXPORT_COMPLETE.json', {'chunks': chunks, 'records': dict(counts),
                 'target_reached': reached, 'duplicate_ids_skipped': duplicates,
                 'hours': {k: v / 3600 for k, v in seconds.items()}})


def encode(args):
    import numpy as np
    import torch
    from qwen_tts import Qwen3TTSTokenizer
    from transformers import AutoTokenizer
    from qwen3_train.sources import decode_emilia_audio, decode_emilia_group

    torch.set_num_threads(2)
    torch.cuda.set_device(args.rank)
    root = Path(args.output).resolve()
    prepared = root / 'prepared'
    prepared.mkdir(exist_ok=True)
    for prefix in range(256):
        (root / 'codes' / f'{prefix:02x}').mkdir(parents=True, exist_ok=True)
    codec_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path(args.codec).glob('*'))
                    if p.suffix == '.safetensors' or p.name == 'config.json'}
    policy = 'trim_aac_padding_1023_or_pad_tail_up_to_1ms'
    cache = {}
    for directory in args.reuse_codes_from:
        directory = Path(directory).resolve()
        recipe_path = directory / 'recipe.json'
        if not recipe_path.exists():
            recipe_path = directory / 'codec-recipe-rank-0.json'
        recipe = json.loads(recipe_path.read_text())
        if recipe['codec_sha256'] != codec_hashes or recipe['audio_length_policy'] != policy or recipe['batch_size'] != 16:
            raise ValueError(f'Incompatible codec cache: {directory}')
        for split in ('train', 'val'):
            with (directory / f'{split}.jsonl').open() as stream:
                for line in stream:
                    row = json.loads(line)
                    cache[row['id']] = (str(directory / row['codes']), row['codes_sha256'],
                                        row['num_frames'], row['audio_source'])
    print(json.dumps({'rank': args.rank, 'cache_records': len(cache)}), flush=True)
    recipe = {'codec_sha256': codec_hashes, 'audio_length_policy': policy, 'batch_size': 16,
              'audio_slice_policy': 'native_rate_slice_then_resample_24k',
              'dtype': 'float32', 'batch_order': 'duration_sorted_within_raw_chunk',
              'speaker_conditioning': 'full_target_audio', 'max_text_tokens': None,
              'tokenizer_sha256': hashlib.sha256((Path(args.tokenizer) / 'tokenizer.json').read_bytes()).hexdigest(),
              'reuse_codes_from': args.reuse_codes_from}
    recipe_path = root / f'codec-recipe-rank-{args.rank}.json'
    if recipe_path.exists() and json.loads(recipe_path.read_text()) != recipe:
        raise ValueError('Codec recipe changed; use a new output directory')
    publish_json(recipe_path, recipe)
    codec = Qwen3TTSTokenizer.from_pretrained(args.codec, device_map=f'cuda:{args.rank}')
    codec.model.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, fix_mistral_regex=False)
    counts = Counter()
    seconds = Counter()
    work_seconds = Counter()
    timings = Counter()
    started = time.monotonic()

    def load(row):
        entry = cache.get(row['id'])
        if entry is not None and row['audio'] == entry[3]:
            path, checksum, frames, locator = entry
            payload = Path(path).read_bytes()
            if hashlib.sha256(payload).hexdigest() != checksum:
                raise ValueError(f'Cached codes changed: {path}')
            with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
                codes = saved['codes']
                if codes.shape != (frames, 16) or not frames or codes.min() < 0 or codes.max() >= 2048:
                    raise ValueError(f'Invalid cached codes: {path}')
            return row, None, (path, checksum, frames)
        waveform = decoders.submit(decode_emilia_audio, row).result() if decoders else decode_emilia_audio(row)
        return row, waveform, None

    def save(row, codes):
        if codes.ndim != 2 or codes.shape[1] != 16 or not len(codes) or codes.max() >= 2048:
            raise ValueError(f'Invalid codec output: {row["id"]}')
        stem = hashlib.sha256(row['id'].encode()).hexdigest()
        destination = root / 'codes' / stem[:2] / f'{stem}.npz'
        buffer = io.BytesIO()
        np.savez(buffer, codes=codes)
        payload = buffer.getvalue()
        temporary = destination.with_suffix('.incomplete')
        temporary.write_bytes(payload)
        temporary.replace(destination)
        return str(destination), hashlib.sha256(payload).hexdigest(), len(codes)

    def prepared_row(row, feature):
        path, checksum, frames = feature
        locator = row['audio']
        stem = hashlib.sha256(row['id'].encode()).hexdigest()
        samples = locator.get('end_frame', locator['frames']) - locator.get('start_frame', 0)
        duration = ((samples * 24000 + locator['sample_rate'] - 1) // locator['sample_rate']) / 24000
        return {**row, 'audio': str(root / 'audio' / f'{stem}.wav'), 'audio_source': locator,
                'duration': duration, 'codes': path, 'codes_sha256': checksum, 'num_frames': frames}

    chunk_index = args.rank
    with ExitStack() as stack:
        decoders = stack.enter_context(ProcessPoolExecutor(max_workers=args.decode_processes,
                    mp_context=multiprocessing.get_context('spawn'))) if args.decode_processes else None
        readers = stack.enter_context(ThreadPoolExecutor(max_workers=args.decode_workers))
        writers = stack.enter_context(ThreadPoolExecutor(max_workers=args.write_workers))
        while True:
            source = root / 'raw' / f'{chunk_index:06d}.jsonl'
            destination = prepared / source.name
            if not source.exists():
                if (root / 'export-error.json').exists():
                    raise RuntimeError((root / 'export-error.json').read_text())
                if (root / 'EXPORT_COMPLETE.json').exists():
                    break
                time.sleep(2)
                continue
            if destination.exists():
                chunk_index += args.world_size
                continue
            with source.open() as stream:
                rows = [json.loads(line) for line in stream]
            tokenized = tokenizer([r['text'] for r in rows], add_special_tokens=False)['input_ids']
            for row, text_ids in zip(rows, tokenized):
                if not text_ids:
                    raise ValueError(f'Empty tokenized text: {row["id"]}')
                row['text_ids'] = text_ids
            groups, group_futures = {}, {}
            cached_ids = {row['id'] for row in rows if row['id'] in cache and row['audio'] == cache[row['id']][3]}
            for row in rows:
                if 'start_frame' in row['audio'] and row['id'] not in cached_ids:
                    key = row['audio']['archive'], row['audio']['offset']
                    groups.setdefault(key, []).append(row)
            ordered = sorted(rows, key=lambda r: r['duration'])
            # Submit each carrier once, in the order its first clip is needed.
            # Reader threads must not occupy all slots waiting on the same long.
            if decoders:
                for row in ordered:
                    if 'start_frame' in row['audio'] and row['id'] not in cached_ids:
                        key = row['audio']['archive'], row['audio']['offset']
                        if key not in group_futures:
                            group_futures[key] = decoders.submit(decode_emilia_group, groups[key])
            batches = iter(ordered[i:i + 16] for i in range(0, len(ordered), 16))
            pending = deque()
            write_pending = deque()
            result = {}

            def submit():
                batch = next(batches, None)
                if batch is not None:
                    futures = []
                    for row in batch:
                        grouped = bool(decoders and 'start_frame' in row['audio'] and row['id'] not in cached_ids)
                        future = group_futures[(row['audio']['archive'], row['audio']['offset'])] if grouped else readers.submit(load, row)
                        futures.append((row, future, grouped))
                    pending.append(futures)

            def drain():
                row, future = write_pending.popleft()
                result[row['id']] = prepared_row(row, future.result())

            # Long carriers can span hours. Other ready batches can run while
            # one carrier is still decoding; the membership of each batch stays fixed.
            for _ in range((len(ordered) + 15) // 16 if groups else args.prefetch_batches):
                submit()
            while pending:
                tick = time.monotonic()
                ready = take_ready_batch(pending)
                batch = [(row, future.result()[row['id']], None) if grouped else future.result()
                         for row, future, grouped in ready]
                timings['decode_wait_seconds'] += time.monotonic() - tick
                submit()
                uncached = [(row, waveform) for row, waveform, feature in batch if feature is None]
                for row, _, feature in batch:
                    if feature is not None:
                        result[row['id']] = prepared_row(row, feature)
                        counts['reused'] += 1
                        work_seconds['reused'] += row['duration']
                if uncached:
                    tick = time.monotonic()
                    with torch.inference_mode():
                        encoded = codec.encode([waveform for _, waveform in uncached], sr=24000).audio_codes
                    if len(encoded) != len(uncached):
                        raise ValueError('Codec batch length mismatch')
                    encoded = [code.cpu().numpy().astype(np.uint16) for code in encoded]
                    timings['codec_seconds'] += time.monotonic() - tick
                    for (row, _), codes in zip(uncached, encoded):
                        write_pending.append((row, writers.submit(save, row, codes)))
                        counts['encoded'] += 1
                        work_seconds['encoded'] += row['duration']
                tick = time.monotonic()
                while len(write_pending) > 16 * args.prefetch_batches:
                    drain()
                timings['write_wait_seconds'] += time.monotonic() - tick
            while write_pending:
                drain()
            temporary = destination.with_suffix('.incomplete')
            with temporary.open('w') as stream:
                for raw_row in rows:
                    row = result[raw_row['id']]
                    seconds[row['language']] += row['duration']
                    counts[row['language']] += 1
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            temporary.replace(destination)
            counts['grouped_carriers_decoded'] += len(group_futures)
            status = {'rank': args.rank, 'last_chunk': chunk_index, 'counts': dict(counts),
                      'hours': {k: v / 3600 for k, v in seconds.items()}, 'timings': dict(timings),
                      'work_hours': {k: v / 3600 for k, v in work_seconds.items()},
                      'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                      'elapsed_seconds': time.monotonic() - started}
            publish_json(root / f'worker-{args.rank}-status.json', status)
            print(json.dumps(status), flush=True)
            chunk_index += args.world_size
    publish_json(root / f'WORKER_{args.rank}_COMPLETE.json', {'counts': dict(counts)})


def finalize(args):
    from qwen3_train.metrics import normalize

    root = Path(args.output).resolve()
    exported = json.loads((root / 'EXPORT_COMPLETE.json').read_text())
    paths = [root / 'prepared' / f'{i:06d}.jsonl' for i in range(exported['chunks'])]
    text_counts, speaker_counts, counts = Counter(), Counter(), Counter()
    source_counts = Counter()
    ids = set()
    for path in paths:
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['id'] in ids:
                    raise ValueError(f'Duplicate prepared ID: {row["id"]}')
                ids.add(row['id'])
                text_counts[normalize(row['text'])] += 1
                speaker_counts[row['speaker']] += 1
                counts[row['language']] += 1
                source_counts[row['source']['type']] += 1
    if dict(counts) != exported['records']:
        raise ValueError('Prepared record counts do not match exported records')
    del ids
    candidates = {}
    for path in paths:
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if text_counts[normalize(row['text'])] != 1 or speaker_counts[row['speaker']] < 3:
                    continue
                priority = hashlib.sha256(('42:' + row['id']).encode()).hexdigest()
                key = row['language'], row['speaker']
                item = priority, row['id']
                if key not in candidates or item < candidates[key]:
                    candidates[key] = item
    val_ids = set()
    val_speakers = set()
    for language in ('en', 'zh'):
        chosen = heapq.nsmallest(args.val_count, ((*v, speaker) for (lang, speaker), v in candidates.items()
                                if lang == language and speaker not in val_speakers))
        if len(chosen) != args.val_count:
            raise ValueError(f'Insufficient validation speakers: {language}')
        val_ids.update(uid for _, uid, _ in chosen)
        val_speakers.update(speaker for _, _, speaker in chosen)
    del text_counts, speaker_counts, candidates
    stats = {'train': Counter(), 'val': Counter()}
    hours = {'train': Counter(), 'val': Counter()}
    maximum = Counter()
    checksums = {split: hashlib.sha256() for split in stats}
    with ExitStack() as stack:
        streams = {split: stack.enter_context((root / f'{split}.incomplete').open('wb')) for split in stats}
        for path in paths:
            with path.open() as source:
                for line in source:
                    row = json.loads(line)
                    split = 'val' if row['id'] in val_ids else 'train'
                    stats[split][row['language']] += 1
                    hours[split][row['language']] += row['duration'] / 3600
                    maximum['text_tokens'] = max(maximum['text_tokens'], len(row['text_ids']))
                    maximum['duration'] = max(maximum['duration'], row['duration'])
                    maximum['codec_frames'] = max(maximum['codec_frames'], row['num_frames'])
                    payload = line.encode()
                    streams[split].write(payload)
                    checksums[split].update(payload)
    for split in stats:
        (root / f'{split}.incomplete').replace(root / f'{split}.jsonl')
    report = {'train': sum(stats['train'].values()), 'val': sum(stats['val'].values()),
              'target_reached': exported.get('target_reached'),
              'duplicate_ids_skipped': exported.get('duplicate_ids_skipped', 0),
              'counts': {k: dict(v) for k, v in stats.items()},
              'source_records': dict(source_counts),
              'hours_by_split': {k: dict(v) for k, v in hours.items()},
              'hours': sum(sum(v.values()) for v in hours.values()),
              'train_hours': sum(hours['train'].values()), 'maximum': dict(maximum),
              'validation_text_disjoint': True, 'validation_speakers_disjoint': False,
              'validation_selection': 'seed_42_hash_one_per_speaker_with_unique_text',
              'manifest_sha256': {k: v.hexdigest() for k, v in checksums.items()}}
    publish_json(root / 'preparation.json', report)
    (root / 'PREPARATION_COMPLETE').write_text('ok\n')
    print(json.dumps(report), flush=True)


def supervise(args):
    root = Path(args.output).resolve()
    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    children = []
    with ExitStack() as stack:
        for rank in range(args.world_size):
            command = [sys.executable, '-u', __file__, '--stage', 'encode', '--output', str(root),
                       '--rank', str(rank), '--world-size', str(args.world_size),
                       '--codec', args.codec, '--tokenizer', args.tokenizer,
                       '--decode-workers', str(args.decode_workers), '--write-workers', str(args.write_workers),
                       '--decode-processes', str(args.decode_processes),
                       '--prefetch-batches', str(args.prefetch_batches), '--reuse-codes-from', *args.reuse_codes_from]
            log = stack.enter_context((run / f'worker-{rank}.log').open('a'))
            children.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        publish_json(run / 'pipeline-status.json', {'stage': 'encoding', 'worker_pids': [p.pid for p in children]})
        try:
            while any(p.poll() is None for p in children):
                failed = [p for p in children if p.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError(f'Codec workers failed: {[p.pid for p in failed]}')
                time.sleep(5)
            if any(p.returncode for p in children):
                raise RuntimeError('Codec worker failed')
            publish_json(run / 'pipeline-status.json', {'stage': 'finalizing'})
            finalize(args)
            publish_json(run / 'pipeline-status.json', {'stage': 'complete', 'output': str(root)})
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                child.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['export', 'encode', 'finalize', 'supervise'], required=True)
    parser.add_argument('--source', default='/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_m4a')
    parser.add_argument('--output', required=True)
    parser.add_argument('--hours-per-language', type=float, default=5000)
    parser.add_argument('--chunk-records', type=int, default=8192)
    parser.add_argument('--export-workers', type=int, default=8)
    parser.add_argument('--resume-export', action='store_true')
    parser.add_argument('--include-long-shorts', action='store_true')
    parser.add_argument('--allow-shortfall', action='store_true')
    parser.add_argument('--codec', default='pretrained/Qwen3-TTS-Tokenizer-12Hz')
    parser.add_argument('--tokenizer', default='pretrained/assembled-qwen3-tts-frozen-conditioning')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=2)
    parser.add_argument('--decode-workers', type=int, default=12)
    parser.add_argument('--decode-processes', type=int, default=12)
    parser.add_argument('--write-workers', type=int, default=8)
    parser.add_argument('--prefetch-batches', type=int, default=8)
    parser.add_argument('--val-count', type=int, default=256)
    parser.add_argument('--run', default='runs/emilia-en-zh-10000h-data')
    parser.add_argument('--reuse-codes-from', nargs='*', default=[
        '/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_prepared/LM-TTS-Training/emilia-short-en-1000h',
        '/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_prepared/LM-TTS-Training/emilia-short-en-zh-1000h/part-1'])
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    try:
        {'export': export, 'encode': encode, 'finalize': finalize, 'supervise': supervise}[args.stage](args)
    except Exception as error:
        publish_json(root / f'{args.stage}-error-{args.rank}.json', {'error': repr(error)})
        if args.stage == 'export':
            publish_json(root / 'export-error.json', {'error': repr(error)})
        if args.stage == 'supervise':
            publish_json(Path(args.run) / 'pipeline-status.json', {'stage': 'failed', 'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
