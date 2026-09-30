"""Create a pinned indexed-reference training build from published features."""

import argparse
import json
from pathlib import Path

from qwen3_train.config import read_yaml
from qwen3_train.data.build import create_build

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train-bf16.yaml")
    args = parser.parse_args()
    config = read_yaml(args.config)
    manifest = create_build(config["data"]["config"], Path(config["data"]["build"]).parent)
    print(
        json.dumps(
            {
                "build_id": manifest["build_id"],
                "ready_rows": sum(b["ready_rows"] for b in manifest["bindings"]),
            }
        )
    )
