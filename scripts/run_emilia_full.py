"""Prepare the full Emilia pool, then train a fresh token-loss model."""
import argparse
from array import array
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=Path('runs/emilia-full-token-scratch'))
    parser.add_argument('--data-dir', type=Path, default=Path(
        '/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_prepared/LM-TTS-Training/emilia-full-en-zh'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run, data = args.run_dir.resolve(), args.data_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    logs = run / 'data-preparation'
    logs.mkdir(exist_ok=True)
    lock = (run / 'pipeline.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = run / 'pipeline-status.json'
    if previous.exists():
        history = run / 'pipeline-history'
        history.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        (history / f'{stamp}.json').write_bytes(previous.read_bytes())
    state = {'status': 'running', 'pid': os.getpid(),
             'start_ticks': Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
             'started_at': datetime.now(timezone.utc).isoformat(),
             'initialization': 'fresh assembled model; no 19000-step checkpoint', 'stages': []}
    env = {**os.environ, 'PYTHONPATH': str(root), 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
           'OPENBLAS_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false', 'NPROC_PER_NODE': '4',
           'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}

    def save(path, value):
        temporary = path.with_suffix('.incomplete')
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(path)

    def stop(*_):
        raise SystemExit(128 + signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)

    def stage(name, commands):
        state['phase'] = name
        record = {'name': name, 'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(), 'processes': []}
        state['stages'].append(record)
        processes, handles = [], []
        try:
            for index, command in enumerate(commands):
                path = run / 'train.log' if name == 'training' else logs / f'{name}-{index}.log'
                handle = path.open('a')
                handles.append(handle)
                process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(process)
                identity = {'pid': process.pid, 'start_ticks': Path(f'/proc/{process.pid}/stat').read_text().split()[21],
                            'command': command, 'log': str(path), 'started_at': record['started_at']}
                record['processes'].append(identity)
                if name == 'training':
                    save(run / 'training-process.json', identity)
            while True:
                codes = [process.poll() for process in processes]
                record['returncodes'] = codes
                save(run / 'pipeline-status.json', state)
                if any(code not in (None, 0) for code in codes):
                    raise RuntimeError(f'{name} failed: {codes}')
                if all(code == 0 for code in codes):
                    break
                time.sleep(10)
            record['status'] = 'complete'
        except BaseException:
            record['status'] = 'failed'
            raise
        finally:
            for process in processes:
                if record['status'] != 'complete' or process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for process in processes:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            for handle in handles:
                handle.close()
            record['finished_at'] = datetime.now(timezone.utc).isoformat()
            if name == 'training' and processes:
                save(run / 'training-exit.json', {'pid': processes[0].pid, 'exit_code': processes[0].returncode,
                                               'finished_at': record['finished_at']})
            save(run / 'pipeline-status.json', state)

    try:
        source = Path('/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_m4a')
        expected_shards = sum(len(list((source / folder).glob('*.tar.idx'))) for folder in ('data', 'ASMR'))
        inventory_path = data / 'INVENTORY_COMPLETE.json'
        if not inventory_path.exists() or json.loads(inventory_path.read_text())['total_shards'] != expected_shards:
            stage('inventory', [[sys.executable, '-u', str(root / 'scripts/inventory_emilia.py'),
                                 '--output', str(data), '--workers', '32']])
        if not (data / 'PREPARATION_COMPLETE').exists():
            stage('prepare', [
                [sys.executable, '-u', str(root / 'scripts/filter_emilia_dialogue.py'),
                 '--output', str(data), '--workers', '32'],
                [sys.executable, '-u', str(root / 'scripts/export_emilia_full.py'), '--output', str(data)],
                [sys.executable, '-u', str(root / 'scripts/prepare_emilia_streaming.py'), '--stage', 'supervise',
                 '--output', str(data), '--run', str(logs / 'codec'), '--world-size', '4',
                 '--decode-processes', '8', '--decode-workers', '8', '--write-workers', '8',
                 '--reuse-codes-from', str(data.parent / 'emilia-short-en-zh-10000h')],
            ])
        from qwen3_train.data import DistributedTokenBatchSampler
        import torch
        state['phase'] = 'planning'
        save(run / 'pipeline-status.json', state)
        torch.set_num_threads(4)
        config = yaml.safe_load((root / 'configs/emilia-10kh-pretrain.yaml').read_text())
        config['data'] = {'train': str(data / 'train.jsonl'), 'val': str(data / 'val.jsonl'), 'balance_languages': True}
        config['train'].update(output=str(run), loss_reduction='token')
        config['eval']['audio_every'] = 2000
        full = json.loads((root / config['model']['assembled_model'] / 'config.json').read_text())
        prefix_tokens = 4 + len(full['talker_config']['lm_tts_role_ids']) + 3
        frames, tokens, durations = array('I'), array('I'), array('d')
        languages = []
        hours = Counter()
        checksum = hashlib.sha256()
        longest_audio = longest_text = None
        with (data / 'train.jsonl').open('rb') as stream:
            for line in stream:
                checksum.update(line)
                row = json.loads(line)
                frames.append(row['num_frames'])
                tokens.append(row['num_frames'] + len(row['text_ids']) + prefix_tokens)
                durations.append(row['duration'])
                language = 'en' if row['language'] == 'en' else 'zh'
                assert row['language'] == language
                languages.append(language)
                hours[language] += row['duration'] / 3600
                if longest_audio is None or row['num_frames'] > longest_audio['num_frames']:
                    longest_audio = row
                if longest_text is None or len(row['text_ids']) > len(longest_text['text_ids']):
                    longest_text = row
        preparation = json.loads((data / 'preparation.json').read_text())
        assert checksum.hexdigest() == preparation['manifest_sha256']['train']
        sampler = DistributedTokenBatchSampler(frames, tokens, config['train']['max_batch_frames'],
            config['train']['max_batch_tokens'], world_size=4, rank=0, seed=config['seed'],
            languages=languages, durations=durations)
        steps = len(sampler)
        config['train'].update(max_steps=steps, schedule_steps=steps)
        config_path = root / 'configs/emilia-full-token-pretrain.yaml'
        if (run / 'checkpoints/latest').exists():
            assert yaml.safe_load(config_path.read_text()) == config, 'Configuration changed before resume'
        else:
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        plan = {'balanced_epochs': 1, 'steps': steps, 'unique_training_rows': len(frames),
                'unique_training_hours': dict(hours), 'target_hours_per_language': max(hours.values()),
                'manifest_sha256': checksum.hexdigest(), 'config': str(config_path)}
        save(run / 'training-plan.json', plan)
        capacity_manifest = run / 'preflight/full-capacity.jsonl'
        capacity_manifest.parent.mkdir(exist_ok=True)
        capacity_manifest.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in (longest_audio, longest_text)))
        del sampler, frames, tokens, durations, languages
        stage('capacity', [
            [str(root / '.venv/bin/torchrun'), '--standalone', '--nproc_per_node=4',
             str(root / 'scripts/probe_batch_memory.py'), '--assembled', config['model']['assembled_model'],
             '--manifest', str(capacity_manifest), '--max-batch-frames', str(config['train']['max_batch_frames']),
             '--max-batch-tokens', str(config['train']['max_batch_tokens'])],
        ])
        command = ['bash', 'scripts/run_train.sh', '--config', str(config_path)]
        if (run / 'checkpoints/latest').exists():
            command += ['--resume', 'latest']
        elif (run / 'training-process.json').exists():
            raise RuntimeError('Previous full training has no checkpoint; inspect before restarting')
        stage('training', [command])
        checkpoint = run / 'checkpoints' / f'step-{steps:08d}'
        assert (checkpoint / 'COMPLETE').is_file()
        stage('frozen-check', [[sys.executable, str(root / 'scripts/check_frozen_frontend.py'),
            '--assembled', config['model']['assembled_model'], '--checkpoint', str(checkpoint), '--include-speaker']])
        state.update(status='complete', checkpoint=str(checkpoint))
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['finished_at'] = datetime.now(timezone.utc).isoformat()
        save(run / 'pipeline-status.json', state)


if __name__ == '__main__':
    main()
