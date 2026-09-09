"""Materialize a pilot raw manifest and cache frozen codec features for training."""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import random
import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
from qwen_tts import Qwen3TTSTokenizer
from transformers import AutoTokenizer
from qwen3_train.sources import materialize_audio, decode_emilia_audio, write_prepared_audio
from qwen3_train.assembly import sha256
from qwen3_train.metrics import normalize


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--codec', default='pretrained/Qwen3-TTS-Tokenizer-12Hz')
    p.add_argument('--tokenizer', default='pretrained/assembled-qwen3-tts-frozen-conditioning')
    p.add_argument('--device', default='cpu')
    p.add_argument('--secondary-devices', nargs='+', default=[], help='Additional codec devices, each processing disjoint batches')
    p.add_argument('--val-count', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--decode-processes', type=int, default=0, help='CPU decode processes; zero uses the I/O threads')
    p.add_argument('--keep-audio', action='store_true')
    p.add_argument('--max-text-tokens', type=int, default=256)
    p.add_argument('--reuse-codes-from', help='Prepared directory with matching codec and audio recipe')
    args = p.parse_args()
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world > 1:
        dist.init_process_group('gloo')
    torch.set_num_threads(4)
    root = Path(args.output).resolve()
    if args.val_count < 1 or args.batch_size < 1 or args.workers < 1 or args.decode_processes < 0:
        p.error('Counts must be positive')
    recipe = {**{k: v for k, v in vars(args).items() if k not in ['secondary_devices', 'decode_processes', 'workers']}, 'format_version': 4, 'validation_text_disjoint': True,
              'speaker_conditioning': 'full_target_audio',
              'audio_length_policy': 'trim_aac_padding_1023_or_pad_tail_up_to_1ms',
              'raw_manifest_sha256': sha256(Path(args.manifest)),
              'codec_sha256': {path.name: sha256(path) for path in sorted(Path(args.codec).glob('*'))
                                if path.suffix == '.safetensors' or path.name == 'config.json'},
              'tokenizer_sha256': sha256(Path(args.tokenizer) / 'tokenizer.json')}
    if rank == 0 and root.exists():
        existing = json.loads((root / 'recipe.json').read_text()) if (root / 'recipe.json').exists() else None
        if existing is not None:
            existing.pop('workers', None)
        if existing != recipe:
            p.error('Output recipe differs; use a new directory')
    reuse = Path(args.reuse_codes_from) if args.reuse_codes_from else None
    if reuse:
        cached_recipe = json.loads((reuse / 'recipe.json').read_text())
        for key in ['codec_sha256', 'audio_length_policy', 'batch_size']:
            if cached_recipe[key] != recipe[key]:
                p.error(f'Cannot reuse codec cache with different {key}')
    if world > 1:
        dist.barrier()
    records = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    if any(row['language'] not in ('en', 'zh') for row in records):
        raise ValueError('Pretraining only accepts English and Chinese')
    if any(row['schema_version'] != 1 for row in records):
        raise ValueError('Unsupported raw manifest schema')
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate IDs in raw manifest')
    for row in records:
        if row['source']['dataset'] == 'emilia2' and row['source']['type'] != 'short':
            raise ValueError('Only top-level Emilia short records are allowed')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, fix_mistral_regex=False)
    for record in records:
        record['text_ids'] = tokenizer.encode(record['text'], add_special_tokens=False)
    rejected_text = sum(not 0 < len(r['text_ids']) <= args.max_text_tokens for r in records)
    records = [r for r in records if 0 < len(r['text_ids']) <= args.max_text_tokens]
    counts = Counter(r['speaker'] for r in records)
    eligible = records
    discarded = 0
    random.Random(args.seed).shuffle(eligible)
    text_counts = Counter(normalize(r['text']) for r in eligible)
    val_ids = set()
    for row in eligible:
        if len(val_ids) < args.val_count and counts[row['speaker']] > 2 and text_counts[normalize(row['text'])] == 1:
            val_ids.add(row['id'])
            counts[row['speaker']] -= 1
    if len(val_ids) != args.val_count:
        raise ValueError('Not enough utterances to hold out validation and retain two training recordings per speaker')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'codes').mkdir(exist_ok=True)
    if rank == 0:
        (root / 'recipe.json').write_text(json.dumps(recipe, indent=2))
    devices = [f'cuda:{int(os.environ["LOCAL_RANK"])}'] if world > 1 else [args.device, *args.secondary_devices]
    if len(set(devices)) != len(devices):
        p.error('Codec devices must be distinct')
    codecs = [Qwen3TTSTokenizer.from_pretrained(args.codec, device_map=device) for device in devices]
    for codec in codecs:
        codec.model.eval().requires_grad_(False)
    rows = []
    total_seconds = 0.
    def prepare_audio(record):
        stem = hashlib.sha256(record['id'].encode()).hexdigest()
        destination = root / 'audio' / f'{stem}.wav'
        code_file = root / 'codes' / f'{stem}.npz'
        if reuse and not code_file.exists() and (reuse / 'codes' / code_file.name).exists():
            os.link(reuse / 'codes' / code_file.name, code_file)
        cached = code_file.exists()
        if record['audio']['kind'] == 'tar_member':
            keep = args.keep_audio
            if cached and (not keep or destination.exists()):
                return destination, code_file, None
            if destination.exists():
                waveform, rate = sf.read(destination, dtype='float32')
                if rate != 24000:
                    raise ValueError(f'Prepared audio sample rate changed: {destination}')
            else:
                waveform = decoder_executor.submit(decode_emilia_audio, record).result() if decoder_executor else decode_emilia_audio(record)
                if keep:
                    write_prepared_audio(waveform, destination)
            return destination, code_file, None if cached else waveform
        audio = materialize_audio(record, destination)
        waveform = None if cached else codecs[0].load_audio(str(audio), target_sr=24000)
        return audio, code_file, waveform
    with ExitStack() as stack:
        decoder_executor = stack.enter_context(ProcessPoolExecutor(max_workers=args.decode_processes,
            mp_context=multiprocessing.get_context('spawn'))) if args.decode_processes else None
        workers = stack.enter_context(ThreadPoolExecutor(max_workers=args.workers * len(codecs)))
        encoders = [stack.enter_context(ThreadPoolExecutor(max_workers=1)) for _ in codecs]
        def process_batch(batch, codec):
            if codec.device.type == 'cuda':
                torch.cuda.set_device(codec.device)
            batch_rows = []
            paths = list(workers.map(prepare_audio, batch))
            pending = [i for i, (_, _, waveform) in enumerate(paths) if waveform is not None]
            features = {}
            if pending:
                with torch.inference_mode():
                    encoded = codec.encode([paths[i][2] for i in pending], sr=24000).audio_codes
                if len(encoded) != len(pending):
                    raise ValueError('Codec batch length mismatch')
                for index, tensor in zip(pending, encoded):
                    codes = tensor.cpu().numpy().astype(np.uint16)
                    code_file = paths[index][1]
                    if codes.ndim != 2 or codes.shape[1] != 16 or not len(codes) or codes.max() >= 2048:
                        raise ValueError(f"Invalid codec output: {batch[index]['id']}")
                    buffer = io.BytesIO()
                    np.savez(buffer, codes=codes)
                    payload = buffer.getvalue()
                    code_file.with_suffix('.incomplete').write_bytes(payload)
                    code_file.with_suffix('.incomplete').replace(code_file)
                    features[index] = (len(codes), hashlib.sha256(payload).hexdigest())
            for index, (record, (audio, code_file, _)) in enumerate(zip(batch, paths)):
                if index in features:
                    frames, checksum = features[index]
                else:
                    payload = code_file.read_bytes()
                    with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
                        codes = saved['codes']
                        if codes.ndim != 2 or codes.shape[1] != 16 or not len(codes) or codes.max() >= 2048:
                            raise ValueError(f'Invalid cached codes: {code_file}')
                        frames = len(codes)
                    checksum = hashlib.sha256(payload).hexdigest()
                locator = record['audio']
                audio_duration = ((locator['frames'] * 24000 + locator['sample_rate'] - 1) // locator['sample_rate']) / 24000 if locator['kind'] == 'tar_member' else sf.info(audio).duration
                batch_rows.append({**record, 'audio': str(audio), 'audio_source': record['audio'],
                    'duration': audio_duration, 'text_ids': record['text_ids'],
                    'codes': str(code_file.relative_to(root)), 'num_frames': frames,
                    'codes_sha256': checksum})
                if audio.exists() and not args.keep_audio and record['audio']['kind'] == 'tar_member':
                    audio.unlink()
            return batch_rows
        starts = iter(range(rank * args.batch_size, len(eligible), world * args.batch_size))
        pending_batches = deque()
        def submit(slot):
            start = next(starts, None)
            if start is not None:
                future = encoders[slot].submit(process_batch, eligible[start:start + args.batch_size], codecs[slot])
                pending_batches.append((slot, future))
        for slot in range(len(codecs)):
            submit(slot)
        while pending_batches:
            slot, future = pending_batches.popleft()
            batch_rows = future.result()
            rows.extend(batch_rows)
            total_seconds += sum(row['duration'] for row in batch_rows)
            print(json.dumps({'rank': rank, 'prepared': len(rows), 'total': len(eligible),
                              'hours': total_seconds / 3600}), flush=True)
            submit(slot)
    if world > 1:
        shard = root / f'rows-rank-{rank}.jsonl'
        shard.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        dist.barrier()
        if rank != 0:
            dist.destroy_process_group()
            return
        by_id = {}
        for slot in range(world):
            for line in (root / f'rows-rank-{slot}.jsonl').read_text().splitlines():
                row = json.loads(line)
                by_id[row['id']] = row
        rows = [by_id[row['id']] for row in eligible]
    for split in ['train', 'val']:
        subset = [r for r in rows if (r['id'] in val_ids) == (split == 'val')]
        temporary = root / f'{split}.incomplete'
        temporary.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in subset))
        temporary.replace(root / f'{split}.jsonl')
    report = {**vars(args), 'raw_manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
              'train': len(rows) - len(val_ids), 'val': len(val_ids),
              'rejected_text_records': rejected_text, 'validation_text_disjoint': True,
              'train_hours': sum(r['duration'] for r in rows if r['id'] not in val_ids) / 3600, 'discarded_singleton_speakers_records': discarded,
              'hours': sum(r['duration'] for r in rows) / 3600}
    (root / 'preparation.json').write_text(json.dumps(report, indent=2))
    (root / 'PREPARATION_COMPLETE').write_text('ok\n')
    if world > 1:
        for slot in range(world):
            (root / f'rows-rank-{slot}.jsonl').unlink()
        dist.destroy_process_group()
    print(json.dumps(report))


if __name__ == '__main__':
    main()
