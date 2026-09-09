"""Complete preparation, memory validation and training of the frozen-text baseline."""
import argparse
import fcntl
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
    p.add_argument('--config', default='configs/emilia-baseline.yaml')
    p.add_argument('--manifest', default='/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_prepared/LM-TTS-Training/emilia-short-en-1000h.raw.jsonl')
    p.add_argument('--export-pid', type=int, help='An already running exporter to wait for; never launches a duplicate')
    args = p.parse_args()
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
    python = str(ROOT / '.venv/bin/python')
    env = {**os.environ, 'PYTHONPATH': str(ROOT), 'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false'}
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
        manifest = Path(args.manifest)
        if not manifest.exists():
            if args.export_pid is None:
                p.error('Export the raw manifest first, or pass its running --export-pid')
            status('waiting-for-export', export_pid=args.export_pid)
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
        preparation = Path(config['data']['train']).parent
        checked('prepare', [python, 'scripts/prepare_manifest.py', '--manifest', str(manifest),
            '--output', str(preparation), '--tokenizer', config['model']['assembled_model'],
            '--codec', config['eval']['codec'], '--device', 'cuda:0', '--secondary-device', 'cuda:1', '--batch-size', '16',
            '--workers', '8', '--decode-processes', '8', '--val-count', '512'])
        report = json.loads((preparation / 'preparation.json').read_text())
        if not (preparation / 'PREPARATION_COMPLETE').exists():
            raise RuntimeError('Preparation completion marker missing')
        if config['model'].get('activation_checkpointing'):
            raise ValueError('This memory probe measures activation_checkpointing=false')
        candidates = [config['train']['batch_size']] if resuming else [48, 32]
        for batch_size in candidates:
            stage = f'memory-batch{batch_size}'
            command = [str(ROOT / '.venv/bin/torchrun'), '--standalone', '--nproc_per_node=2',
                'scripts/probe_batch_memory.py', '--assembled', config['model']['assembled_model'],
                '--manifest', config['data']['train'], '--batch-size', str(batch_size)]
            code = run(stage, command)
            if code == 0:
                config['train']['batch_size'] = batch_size
                break
            if 'out of memory' not in (output / f'{stage}.log').read_text().lower():
                raise RuntimeError('Memory probe failed for a reason other than CUDA OOM')
        else:
            raise RuntimeError('Both memory probes failed; no training was started')
        effective = output / 'baseline-config.yaml'
        effective.write_text(yaml.safe_dump(config, sort_keys=False))
        status('ready-to-train', preparation=report, per_gpu_batch=config['train']['batch_size'],
               global_batch=2 * config['train']['batch_size'] * config['train']['accumulation'])
        command = ['bash', 'scripts/run_train.sh', '--config', str(effective)]
        if (output / 'checkpoints/latest').exists():
            command += ['--resume', 'latest']
        checked('train', command)
        checkpoint = output / 'checkpoints' / (output / 'checkpoints/latest').read_text().strip()
        checked('frozen-weights', [python, 'scripts/check_frozen_frontend.py', '--assembled',
            config['model']['assembled_model'], '--checkpoint', str(checkpoint), '--include-speaker'])
        checked('english-scores', [python, 'scripts/rescore_english.py', str(output)])
        status('complete', checkpoint=str(checkpoint), preparation=report)
    except Exception as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
