"""Resumable Seed-TTS synthesis with the native Qwen3-TTS inference API."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from safetensors import safe_open
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('token', 'sqrt', 'sample'), required=True)
    parser.add_argument('--mode', choices=('xvec_only', 'icl_xvec'), required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--ultraeval', type=Path, default=Path('/119010446/UltraEval-Audio'))
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0, help='Per-language smoke limit')
    parser.add_argument('--output-subdir', default='results')
    parser.add_argument('--verify-weights', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.run_dir.resolve()
    reference_validation = json.loads((root / 'input_validation.json').read_text())
    model_name = f'{args.model}-{args.mode}'
    output = root / args.output_subdir / model_name
    output.mkdir(parents=True, exist_ok=True)
    journal = output / f'inference-{args.shard:02d}.jsonl'
    done = set()
    if journal.exists():
        for line in journal.read_text().splitlines():
            row = json.loads(line)
            if row.get('status') == 'ok' and Path(row['audio']).is_file():
                done.add((row['dataset'], row['index']))
    batches = []
    metadata = json.loads((args.ultraeval / 'raw_data/voice_clone_manifests/manifest_metadata.json').read_text())
    for language in ('en', 'zh'):
        dataset = f'seed_tts_eval_{language}'
        manifest = args.ultraeval / f'raw_data/voice_clone_manifests/{dataset}.jsonl'
        raw = manifest.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == metadata[dataset]['sha256']
        rows = [json.loads(line) for line in raw.decode().splitlines()]
        assert len(rows) == metadata[dataset]['count']
        if args.limit:
            rows = rows[:args.limit]
        rows = [row for row in rows if row['index'] % args.num_shards == args.shard]
        batches.extend(rows[start:start + args.batch_size] for start in range(0, len(rows), args.batch_size))
    batches = [batch for batch in batches if any((r['dataset'], r['index']) not in done for r in batch)]
    print(f'{model_name} shard={args.shard} batches={len(batches)} gpu={os.getenv("CUDA_VISIBLE_DEVICES")}', flush=True)
    if not batches:
        return
    model_path = root / 'exports' / args.model
    model = Qwen3TTSModel.from_pretrained(str(model_path), device_map='cuda:0',
                                        dtype=torch.bfloat16, attn_implementation='flash_attention_2')
    with safe_open(model_path / 'model.safetensors', framework='pt', device='cpu') as weights:
        state = model.model.state_dict()
        assert set(state) == set(weights.keys()), 'Native/export state keys differ'
        for key, value in state.items():
            assert tuple(value.shape) == tuple(weights.get_slice(key).get_shape()), key
            if args.verify_weights:
                assert torch.equal(value.cpu(), weights.get_tensor(key)), key
    params = dict(max_new_tokens=2048, do_sample=True, top_k=50, top_p=1.0,
                  temperature=0.9, repetition_penalty=1.05, subtalker_dosample=True,
                  subtalker_top_k=50, subtalker_top_p=1.0, subtalker_temperature=0.9)
    (output / f'worker-{args.shard:02d}.json').write_text(json.dumps({
        **vars(args), 'run_dir': str(root), 'ultraeval': str(args.ultraeval),
        'model_sha256': json.loads((model_path / 'export.json').read_text())['weights_sha256'],
        'generation': params, 'language': 'Auto', 'non_streaming_mode': True,
        'attention': 'flash_attention_2', 'dtype': 'bfloat16', 'weights_verified': args.verify_weights,
    }, indent=2))
    observed = {}
    original_speaker = model.model.generate_speaker_prompt
    original_icl = model.model.generate_icl_prompt
    original_talker = model.model.talker.generate

    def speaker_prompt(prompt):
        result = original_speaker(prompt)
        observed['speaker_embedding'] = result is not None
        return result

    def icl_prompt(*positional, **keywords):
        observed['icl_calls'] += 1
        return original_icl(*positional, **keywords)

    def talker_generate(**kwargs):
        result = original_talker(**kwargs)
        eos = model.model.config.talker_config.codec_eos_token_id
        observed['eos_reached'] = (result.sequences == eos).any(dim=1).tolist()
        return result

    model.model.generate_speaker_prompt = speaker_prompt
    model.model.generate_icl_prompt = icl_prompt
    model.model.talker.generate = talker_generate
    with journal.open('a', buffering=1) as handle:
        for batch in batches:
            started = time.monotonic()
            seed = 42 + batch[0]['index'] + (100000 if batch[0]['language'] == 'zh' else 0)
            try:
                reference_hashes = [hashlib.sha256(Path(row['prompt_audio']).read_bytes()).hexdigest() for row in batch]
                for row, digest in zip(batch, reference_hashes):
                    assert digest == reference_validation[row['language']]['reference_wav_sha256'][str(row['index'])]
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                observed.clear()
                observed['icl_calls'] = 0
                wavs, sample_rate = model.generate_voice_clone(
                    text=[r['text'] for r in batch], language='Auto', non_streaming_mode=True,
                    ref_audio=[r['prompt_audio'] for r in batch],
                    ref_text=None if args.mode == 'xvec_only' else [r['prompt_text'] for r in batch],
                    x_vector_only_mode=args.mode == 'xvec_only', **params)
                assert observed['speaker_embedding']
                assert observed['icl_calls'] == (0 if args.mode == 'xvec_only' else len(batch))
                assert len(wavs) == len(batch)
                for index, (row, wav) in enumerate(zip(batch, wavs)):
                    if (row['dataset'], row['index']) in done:
                        continue
                    wav = np.asarray(wav, dtype=np.float32)
                    assert wav.ndim == 1 and len(wav) and np.isfinite(wav).all()
                    audio = output / row['dataset'] / f'{row["index"]:06d}.wav'
                    audio.parent.mkdir(parents=True, exist_ok=True)
                    temporary = audio.with_suffix('.tmp.wav')
                    sf.write(temporary, wav, sample_rate, subtype='PCM_16')
                    temporary.replace(audio)
                    result = {**row, 'model': model_name, 'status': 'ok', 'audio': str(audio),
                              'seed': seed, 'sample_rate': sample_rate, 'duration': len(wav) / sample_rate,
                              'batch_size': len(batch), 'batch_elapsed': time.monotonic() - started,
                              'eos_reached': observed['eos_reached'][index],
                              'reference_wav_sha256': reference_hashes[index],
                              'speaker_embedding': True, 'icl': args.mode == 'icl_xvec'}
                    handle.write(json.dumps(result, ensure_ascii=False) + '\n')
                print(f'{model_name} {batch[0]["dataset"]} indices={[r["index"] for r in batch]} seconds={time.monotonic()-started:.1f}', flush=True)
            except Exception:
                handle.write(json.dumps({'dataset': batch[0]['dataset'], 'index': batch[0]['index'],
                                         'status': 'error', 'error': traceback.format_exc()}) + '\n')
                raise


if __name__ == '__main__':
    main()
