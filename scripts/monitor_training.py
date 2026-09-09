"""Run the user-authorized Codex training review every thirty minutes."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def snapshot(run, previous_step):
    process_path = run / 'training-process.json'
    process = json.loads(process_path.read_text()) if process_path.exists() else None
    alive = False
    if process:
        stat = Path(f"/proc/{process['pid']}/stat")
        try:
            fields = stat.read_text().split()
            alive = fields[2] != 'Z' and fields[21] == process['start_ticks']
        except FileNotFoundError:
            pass
    records = []
    log = run / 'train.log'
    if log.exists():
        with log.open('rb') as stream:
            start = max(0, log.stat().st_size - 4 * 1024 * 1024)
            stream.seek(start)
            if start:
                stream.readline()
            tail = stream.read().decode(errors='replace')
        for line in tail.splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and 'train' in record:
                records.append(record)
    else:
        tail = ''
    recent = [r for r in records if r['step'] > previous_step]
    evaluation = {}
    for mode in ['speaker_only', 'icl']:
        paths = sorted((run / 'evaluation' / mode).glob('step-*/summary.json'))
        evaluation[mode] = [{'path': str(p), 'summary': json.loads(p.read_text())} for p in paths[-2:]]
    latest = run / 'checkpoints/latest'
    checkpoint = None
    if latest.exists():
        path = run / 'checkpoints' / latest.read_text().strip()
        checkpoint = {'path': str(path), 'complete': (path / 'COMPLETE').exists(),
                      'progress': json.loads((path / 'metadata.json').read_text())['progress']}
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu',
                          '--format=csv'], capture_output=True, text=True, timeout=20)
    exit_path = run / 'training-exit.json'
    return {'time': datetime.now(timezone.utc).isoformat(), 'process': process, 'process_alive': alive,
            'training_exit': json.loads(exit_path.read_text()) if exit_path.exists() else None,
            'last_step': records[-1]['step'] if records else 0, 'previous_check_step': previous_step,
            'new_logged_steps': len(recent), 'recent_logged_metrics': recent[-60:],
            'log_modified_at': datetime.fromtimestamp(log.stat().st_mtime, timezone.utc).isoformat() if log.exists() else None,
            'error_lines': [line for line in tail.splitlines() if any(word in line for word in
                            ['Traceback', 'OutOfMemory', 'out of memory', 'Non-finite', 'Aborted', 'ChildFailedError'])][-20:],
            'checkpoint': checkpoint, 'evaluation': evaluation, 'gpu': gpu.stdout or gpu.stderr,
            'disk_free_gib': shutil.disk_usage(run).free / 2**30}


def collect_snapshot(run, previous_step):
    try:
        return snapshot(run, previous_step)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {'time': datetime.now(timezone.utc).isoformat(), 'last_step': previous_step,
                'snapshot_error': f'{type(error).__name__}: {error}',
                'process': None, 'training_exit': None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--interval-seconds', type=int, default=1800)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval_seconds < 1:
        parser.error('interval must be positive')
    run = Path(args.run_dir).resolve()
    folder = run / 'supervision'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'monitor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (folder / 'monitor.pid').write_text(str(os.getpid()))
    state_path = folder / 'status.json'
    previous = json.loads(state_path.read_text()).get('last_step', 0) if state_path.exists() else 0
    session_path = folder / 'session-id'
    session_id = session_path.read_text().strip() if session_path.exists() else None
    deadline = time.monotonic()
    while not (folder / 'STOP').exists():
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        data = collect_snapshot(run, previous)
        record = folder / f'{stamp}.json'
        record.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        prompt = (run / 'supervision-prompt.md').read_text().replace('{{SNAPSHOT}}', str(record))
        command = ['codex', '-a', 'never', 'exec', '--disable', 'apps',
                   '-s', 'danger-full-access', '--color', 'never', '-C', str(ROOT),
                   '--json', '-o', str(folder / f'{stamp}.md')]
        command += ['resume', session_id, '-'] if session_id else ['-']
        log_path = folder / f'{stamp}.codex.log'
        with log_path.open('w') as log:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, text=True, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                child.communicate(prompt, timeout=1500)
                returncode = child.returncode
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.communicate()
                returncode = 124
        for line in log_path.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('type') == 'thread.started':
                actual_id = event['thread_id']
                if session_id and session_id != actual_id:
                    raise RuntimeError('Codex resumed a different session')
                session_id = actual_id
                session_path.write_text(session_id + '\n')
        previous = data['last_step']
        finished = collect_snapshot(run, previous)
        completion = finished['training_exit']
        verification_path = run / 'final-verification.json'
        verification = json.loads(verification_path.read_text()) if verification_path.exists() else {}
        complete = bool(returncode == 0 and completion and finished['process'] and
                        completion['pid'] == finished['process']['pid'] and completion['exit_code'] == 0
                        and verification.get('passed') is True
                        and verification.get('pid') == finished['process']['pid'])
        status = {'last_check': data['time'], 'last_step': previous, 'review_exit_code': returncode,
                  'latest_snapshot': str(record), 'latest_review': str(folder / f'{stamp}.md'),
                  'status': 'completed' if complete else ('monitoring' if returncode == 0 else 'review_failed'),
                  'interval_seconds': args.interval_seconds, 'session_id': session_id}
        state_path.write_text(json.dumps(status, indent=2))
        print(json.dumps(status), flush=True)
        if args.once or complete:
            break
        deadline += args.interval_seconds
        while deadline <= time.monotonic():
            deadline += args.interval_seconds
        while time.monotonic() < deadline and not (folder / 'STOP').exists():
            time.sleep(min(30, deadline - time.monotonic()))


if __name__ == '__main__':
    main()
