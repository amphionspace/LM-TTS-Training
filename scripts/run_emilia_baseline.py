"""Prepare English/Chinese data, validate memory and start frozen-text pretraining."""
import argparse
import fcntl
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/emilia-pretrain.yaml')
    p.add_argument('--nproc-per-node', type=int, default=4)
    p.add_argument('--manifest', nargs='+', default=[
        '/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_prepared/LM-TTS-Training/emilia-short-en-500h.raw.jsonl',
        '/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_prepared/LM-TTS-Training/emilia-short-zh-500h.raw.jsonl'])
    p.add_argument('--reuse-codes-from', default='/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_prepared/LM-TTS-Training/emilia-short-en-1000h')
    p.add_argument('--export-pid', type=int, help='An already running exporter to wait for; never launches a duplicate')
    args = p.parse_args()
    if args.nproc_per_node < 1:
        p.error('--nproc-per-node must be positive')
    os.chdir(ROOT)
    config = yaml.safe_load(Path(args.config).read_text())
    output = Path(config['train']['output'])
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (output / 'pipeline.pid').write_text(str(os.getpid()))
    resuming = (output / 'checkpoints/latest').exists()
    if resuming:
        config = yaml.safe_load((output / 'baseline-config.yaml').read_text())
        checkpoint = output / 'checkpoints' / (output / 'checkpoints/latest').read_text().strip()
        metadata = json.loads((checkpoint / 'metadata.json').read_text())
        if metadata['world_size'] != args.nproc_per_node:
            p.error(f"Resume requires --nproc-per-node {metadata['world_size']}")
    python = str(ROOT / '.venv/bin/python')
    env = {**os.environ, 'PYTHONPATH': str(ROOT), 'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false',
           'NPROC_PER_NODE': str(args.nproc_per_node)}
    status_path = output / 'pipeline-status.json'
    def status(stage, **extra):
        record = {'stage': stage, 'updated_at': datetime.now(timezone.utc).isoformat(), **extra}
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(status_path)
        print(json.dumps(record), flush=True)
    def run(stage, command):
        status(stage, command=command)
        with (output / f'{stage}.log').open('a') as log:
            return subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    def checked(stage, command):
        code = run(stage, command)
        if code:
            raise RuntimeError(f'{stage} failed with exit code {code}; see {output / (stage + ".log")}')
    try:
        preparation = Path(config['data']['train']).resolve().parent
        reports, parts = [], []
        for index, name in enumerate(args.manifest):
            manifest = Path(name)
            if not manifest.exists():
                if args.export_pid is None:
                    raise ValueError('Export the raw manifest first, or pass its running --export-pid')
                status('waiting-for-export', export_pid=args.export_pid, manifest=str(manifest))
                while not manifest.exists():
                    try:
                        command = Path(f'/proc/{args.export_pid}/cmdline').read_bytes().split(b'\0')
                    except FileNotFoundError:
                        command = []
                    if b'scripts/export_manifest.py' not in command:
                        if manifest.exists():
                            break
                        raise RuntimeError('Exporter exited without publishing its manifest')
                    time.sleep(15)
            part = preparation / f'part-{index}'
            if not (part / 'PREPARATION_COMPLETE').exists():
                checked(f'prepare-{index}', [str(ROOT / '.venv/bin/torchrun'), '--standalone',
                    f'--nproc_per_node={args.nproc_per_node}', 'scripts/prepare_manifest.py',
                    '--manifest', str(manifest), '--output', str(part),
                    '--tokenizer', config['model']['assembled_model'], '--codec', config['eval']['codec'],
                    '--device', 'cuda:0', '--batch-size', '16', '--workers', '16',
                    '--decode-processes', '16', '--val-count', str(512 // len(args.manifest)),
                    '--reuse-codes-from', args.reuse_codes_from])
            report = json.loads((part / 'preparation.json').read_text())
            if report['raw_manifest_sha256'] != hashlib.sha256(manifest.read_bytes()).hexdigest():
                raise ValueError(f'Prepared manifest changed: {manifest}')
            reports.append(report)
            parts.append(part)
        status('merge-manifests')
        from qwen3_train.metrics import normalize
        texts, ids = {}, set()
        for split in ['train', 'val']:
            split_texts = set()
            temporary = preparation / f'{split}.incomplete'
            with temporary.open('w') as target:
                for part in parts:
                    with (part / f'{split}.jsonl').open() as source:
                        for line in source:
                            row = json.loads(line)
                            if row['id'] in ids:
                                raise ValueError(f'Duplicate prepared ID: {row["id"]}')
                            ids.add(row['id'])
                            split_texts.add(normalize(row['text']))
                            row['codes'] = str(part / row['codes'])
                            target.write(json.dumps(row, ensure_ascii=False) + '\n')
            texts[split] = split_texts
            temporary.replace(preparation / f'{split}.jsonl')
        if texts['train'] & texts['val']:
            raise ValueError('Combined train/validation text leakage')
        report = {'parts': reports, **{key: sum(r[key] for r in reports)
                  for key in ['train', 'val', 'hours', 'train_hours']}}
        (preparation / 'preparation.json').write_text(json.dumps(report, indent=2))
        (preparation / 'PREPARATION_COMPLETE').write_text('ok\n')
        del texts, ids
        if config['model'].get('activation_checkpointing'):
            raise ValueError('This memory probe measures activation_checkpointing=false')
        budgets = (config['train']['max_batch_frames'], config['train']['max_batch_tokens'])
        candidates = [budgets]
        if not resuming:
            candidates += [(int(budgets[0] * scale), int(budgets[1] * scale)) for scale in [.8, .6, .4]]
        for frames, tokens in candidates:
            stage = f'memory-frames{frames}-tokens{tokens}'
            command = [str(ROOT / '.venv/bin/torchrun'), '--standalone', f'--nproc_per_node={args.nproc_per_node}',
                'scripts/probe_batch_memory.py', '--assembled', config['model']['assembled_model'],
                '--manifest', config['data']['train'], '--max-batch-frames', str(frames),
                '--max-batch-tokens', str(tokens),
                '--attn-implementation', config['model'].get('attn_implementation', 'sdpa')]
            code = run(stage, command)
            if code == 0:
                config['train']['max_batch_frames'] = frames
                config['train']['max_batch_tokens'] = tokens
                break
            if 'out of memory' not in (output / f'{stage}.log').read_text().lower():
                raise RuntimeError('Memory probe failed for a reason other than CUDA OOM')
        else:
            raise RuntimeError('All memory probes failed; no training was started')
        effective = output / 'baseline-config.yaml'
        effective.write_text(yaml.safe_dump(config, sort_keys=False))
        status('ready-to-train', preparation=report, world_size=args.nproc_per_node,
               max_batch_frames=config['train']['max_batch_frames'],
               max_batch_tokens=config['train']['max_batch_tokens'], accumulation=config['train']['accumulation'])
        command = ['bash', 'scripts/run_train.sh', '--config', str(effective)]
        if (output / 'checkpoints/latest').exists():
            command += ['--resume', 'latest']
        checked('train', command)
        checkpoint = output / 'checkpoints' / (output / 'checkpoints/latest').read_text().strip()
        checked('frozen-weights', [python, 'scripts/check_frozen_frontend.py', '--assembled',
            config['model']['assembled_model'], '--checkpoint', str(checkpoint), '--include-speaker'])
        status('complete', checkpoint=str(checkpoint), preparation=report)
    except Exception as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
