"""Initialize local SCO credentials without echoing secrets or their arguments."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from qwen3_train.config import read_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs/base.yaml"
    )
    args = parser.parse_args()
    config = read_yaml(args.config, keys=("acp",))["acp"]
    downloads = Path(config["home"])
    credentials = json.loads((downloads / ".credentials.json").read_text())
    env = {
        **os.environ,
        "SCO_HOME": str(downloads / ".sco"),
        "SCO_DATA_HOME": str(downloads / ".data"),
        "SCO_CONFIG": str(downloads / ".config"),
    }
    command = [
        str(downloads / ".sco/bin/sco"),
        "init",
        "--profile",
        "default",
        "--region",
        config["region"],
        "--zone",
        config["zone"],
        "--language",
        "zh-CN",
        "--access-key-id",
        credentials["access_key_id"],
        "--access-key-secret",
        credentials["access_key_secret"],
    ]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    output = result.stdout + result.stderr
    for value in credentials.values():
        output = output.replace(value, "[REDACTED]")
    for path in (downloads / ".config").rglob("*"):
        if path.is_file():
            path.chmod(0o600)
    print(output)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
