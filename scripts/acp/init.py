"""Initialize local SCO credentials without echoing secrets or their arguments."""

import json
import os
import subprocess
from pathlib import Path

from qwen3_train.config import read_yaml

root = Path(__file__).resolve().parent
downloads = root.parents[2] / "acp"
config = read_yaml(root.parents[1] / "configs/base.yaml", keys=("acp",))["acp"]
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
