"""Materialize a pilot raw manifest and cache frozen codec features for training."""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import random
import numpy as np
import soundfile as sf
import torch
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
    p.add_argument('--secondary-device', help='Optional second codec device; each device processes disjoint batches')
    p.add_argument('--val-count', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--decode-processes', type=int, default=0, help='CPU decode processes; zero uses the I/O threads')
    p.add_argument('--keep-audio', action='store_true')
    p.add_argument('--max-text-tokens', type=int, default=256)
    args = p.parse_args()
    torch.set_num_threads(4)
    root = Path(args.output).resolve()
    if args.val_count < 1 or args.batch_size < 1 or args.workers < 1 or args.decode_processes < 0:
        p.error('Counts must be positive')
    recipe = {**{k: v for k, v in vars(args).items() if k not in ['secondary_device', 'decode_processes']}, 'format_version': 3, 'validation_text_disjoint': True,
              'audio_length_policy': 'trim_aac_padding_1023_or_pad_tail_up_to_1ms',
              'raw_manifest_sha256': sha256(Path(args.manifest)),
              'codec_sha256': {path.name: sha256(path) for path in sorted(Path(args.codec).glob('*'))
                                if path.suffix == '.safetensors' or path.name == 'config.json'},
              'tokenizer_sha256': sha256(Path(args.tokenizer) / 'tokenizer.json')}
    if root.exists():
        existing = json.loads((root / 'recipe.json').read_text()) if (root / 'recipe.json').exists() else None
        previous = {key: value for key, value in recipe.items() if key != 'audio_length_policy'}
        previous['format_version'] = 2
        if existing != recipe and existing != previous:
            p.error('Output recipe differs; use a new directory')
        if existing == previous:
            # Every sample accepted by v2 has identical PCM under v3; newly accepted
            # sub-ms-short files could not have produced v2 codec caches.
            (root / 'recipe-v2.json').write_text(json.dumps(existing, indent=2))
            print('Upgrading audio length policy; existing accepted codec caches are unchanged', flush=True)
    records = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
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
    eligible = [r for r in records if counts[r['speaker']] >= 2]
    discarded = len(records) - len(eligible)
    random.Random(args.seed).shuffle(eligible)
    text_counts = Counter(normalize(r['text']) for r in eligible)
    val_ids = set()
    for row in eligible:
        if len(val_ids) < args.val_count and counts[row['speaker']] > 2 and text_counts[normalize(row['text'])] == 1:
            val_ids.add(row['id'])
            counts[row['speaker']] -= 1
    if len(val_ids) != args.val_count:
        raise ValueError('Not enough utterances to hold out validation and retain two training recordings per speaker')
    anchors = {}
    for row in eligible:
        if row['id'] not in val_ids and len(anchors.get(row['speaker'], [])) < 2:
            anchors.setdefault(row['speaker'], []).append(row['id'])
    retained = {uid for values in anchors.values() for uid in values}
    root.mkdir(parents=True, exist_ok=True)
    (root / 'codes').mkdir(exist_ok=True)
    (root / 'recipe.json').write_text(json.dumps(recipe, indent=2))
    devices = [args.device] + ([args.secondary_device] if args.secondary_device else [])
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
        cached = code_file.exists()
        if record['audio']['kind'] == 'tar_member':
            keep = args.keep_audio or record['id'] in retained
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
        workers = stack.enter_context(ThreadPoolExecutor(max_workers=args.workers))
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
                reference_id = next(uid for uid in anchors[record['speaker']] if uid != record['id'])
                locator = record['audio']
                audio_duration = ((locator['frames'] * 24000 + locator['sample_rate'] - 1) // locator['sample_rate']) / 24000 if locator['kind'] == 'tar_member' else sf.info(audio).duration
                batch_rows.append({**record, 'audio': str(audio), 'audio_source': record['audio'],
                    'speaker_reference_id': reference_id,
                    'duration': audio_duration, 'text_ids': record['text_ids'],
                    'codes': str(code_file.relative_to(root)), 'num_frames': frames,
                    'codes_sha256': checksum})
                if audio.exists() and not args.keep_audio and record['id'] not in retained and record['audio']['kind'] == 'tar_member':
                    audio.unlink()
            return batch_rows
        starts = iter(range(0, len(eligible), args.batch_size))
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
            print(json.dumps({'prepared': len(rows), 'total': len(eligible),
                              'hours': total_seconds / 3600}), flush=True)
            submit(slot)
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
    print(json.dumps(report))


if __name__ == '__main__':
    main()
