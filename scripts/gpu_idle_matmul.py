"""Run user-requested matrix work on idle GPUs, yielding to the training pipeline."""
import argparse
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time

import torch


def worker(device, run):
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    left = torch.randn((4096, 4096), device=f'cuda:{device}', dtype=torch.float16)
    right = torch.randn_like(left)
    output = torch.empty_like(left)
    folder = run / 'gpu-idle-matmul'
    while not (folder / 'STOP').exists():
        pipeline_path = run / 'pipeline-status.json'
        pipeline = json.loads(pipeline_path.read_text()) if pipeline_path.exists() else {}
        utilization = int(subprocess.check_output([
            'nvidia-smi', '-i', str(device), '--query-gpu=utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip())
        active = utilization < 20 and not (pipeline.get('status') == 'running' and
                                           pipeline.get('phase') in ('capacity', 'training'))
        if active:
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                torch.mm(left, right, out=output)
                torch.cuda.synchronize(device)
        status = {'pid': os.getpid(), 'gpu': device, 'time': time.time(),
                  'observed_utilization': utilization, 'matrix_work': active}
        temporary = folder / f'gpu-{device}.incomplete'
        temporary.write_text(json.dumps(status) + '\n')
        temporary.replace(folder / f'gpu-{device}.json')
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    folder = run / 'gpu-idle-matmul'
    folder.mkdir(exist_ok=True)
    lock = (folder / 'lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    context = multiprocessing.get_context('spawn')
    children = [context.Process(target=worker, args=(device, run)) for device in range(torch.cuda.device_count())]
    assert children, 'No visible CUDA devices'
    try:
        for child in children:
            child.start()
        while any(child.is_alive() for child in children):
            if any(child.exitcode not in (None, 0) for child in children):
                raise RuntimeError('GPU matrix worker failed')
            time.sleep(2)
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
        for child in children:
            child.join()


if __name__ == '__main__':
    main()
