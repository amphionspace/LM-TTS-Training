"""Run the six Seed-TTS evaluations and UltraEval scoring across four GPUs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ultra = Path('/119010446/UltraEval-Audio')
    run = args.run_dir.resolve()
    (run / 'logs').mkdir(parents=True, exist_ok=True)
    (run / 'results').mkdir(exist_ok=True)
    lock = (run / '.pipeline.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def stop(*_):
        raise SystemExit(128 + signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    status = {'status': 'running', 'started': time.time(), 'pid': os.getpid(), 'stages': []}
    env = {**os.environ, 'PYTHONPATH': str(root), 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
           'TOKENIZERS_PARALLELISM': 'false', 'HF_HOME': str(ultra / '.cache/huggingface'),
           'TORCH_HOME': str(ultra / '.cache/torch'), 'MODELSCOPE_CACHE': str(ultra / '.cache/modelscope')}

    def save():
        temporary = run / 'pipeline_status.tmp.json'
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(run / 'pipeline_status.json')

    def stage(name, commands):
        record = {'name': name, 'started': time.time(), 'status': 'running', 'processes': []}
        status['stages'].append(record)
        processes, handles = [], []
        try:
            for index, (command, gpu) in enumerate(commands):
                log = run / 'logs' / f'{name}-{index:02d}.log'
                handle = log.open('a')
                handles.append(handle)
                working_directory = ultra if name == 'score' else root
                process = subprocess.Popen(command, cwd=working_directory,
                    env={**env, 'PYTHONPATH': str(working_directory), 'CUDA_VISIBLE_DEVICES': str(gpu)},
                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(process)
                record['processes'].append({'pid': process.pid, 'gpu': gpu, 'command': command, 'log': str(log)})
            while True:
                codes = [p.poll() for p in processes]
                record['returncodes'] = codes
                save()
                if any(code not in (None, 0) for code in codes):
                    raise RuntimeError(f'{name} failed: {codes}')
                if all(code == 0 for code in codes):
                    break
                time.sleep(10)
            record.update(status='complete', seconds=time.time() - record['started'])
        finally:
            for process in processes:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            for process in processes:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            for handle in handles:
                handle.close()
            save()

    try:
        for model in ('token', 'sqrt', 'sample'):
            for mode in ('xvec_only', 'icl_xvec'):
                stage(f'{model}-{mode}', [([sys.executable, '-u', str(root / 'scripts/seed_eval_infer.py'),
                    '--model', model, '--mode', mode, '--run-dir', str(run), '--shard', str(shard),
                    '--num-shards', str(args.workers), '--batch-size', str(args.batch_size)], shard % 4)
                    for shard in range(args.workers)])
        stage('score', [([str(ultra / 'envs/runner/bin/python'), '-u', str(ultra / 'scripts/voice_clone_score.py'),
            '--group', 'seed', '--language', language, '--run-dir', str(run / 'results'),
            '--workers', '4', '--shard', str(shard)], shard)
            for language in ('en', 'zh') for shard in range(4)])
        subprocess.run([sys.executable, str(root / 'scripts/summarize_seed_eval.py'), '--run-dir', str(run)],
                       cwd=root, env=env, check=True)
        status['status'] = 'complete'
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        status['finished'] = time.time()
        save()


if __name__ == '__main__':
    main()
