"""Record only runtime evidence needed to diagnose an ACP validation job."""

import argparse
import json
import os
import socket
import subprocess
from importlib.metadata import version
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = [
        "SENSECORE_JOB_NAME",
        "SENSECORE_PYTORCH_NODE_RANK",
        "SENSECORE_PYTORCH_NNODES",
        "SENSECORE_ACCELERATE_DEVICE_COUNT",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NCCL_IB_GID_INDEX",
        "NCCL_IB_TC",
        "NCCL_IB_QPS_PER_CONNECTION",
        "NCCL_SOCKET_IFNAME",
    ]
    result = {
        "hostname": socket.gethostname(),
        "environment": {key: os.environ.get(key) for key in names},
        "versions": {
            key: version(key)
            for key in ["torch", "flash-attn", "transformers", "pylance", "qwen-tts"]
        },
        "gpu_count": torch.cuda.device_count(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_inventory": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
            text=True,
        ),
        "shared_memory": subprocess.check_output(["df", "-B1", "/dev/shm"], text=True),
        "mount": subprocess.check_output(["findmnt", "-T", str(args.output.parent)], text=True),
    }
    if not result["cuda_available"] or result["gpu_count"] != int(
        os.environ["SENSECORE_ACCELERATE_DEVICE_COUNT"]
    ):
        raise RuntimeError("ACP GPU exposure differs from the requested worker resources")
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
