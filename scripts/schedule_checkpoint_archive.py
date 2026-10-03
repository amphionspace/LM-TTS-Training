"""Archive complete checkpoints hourly, independently of the Codex supervisor."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def write_status(folder, status):
    temporary = folder / "status.tmp"
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(folder / "status.json")


def run_check(run, folder, *, interval, timeout):
    started = datetime.now(timezone.utc)
    log_path = folder / f"{started.strftime('%Y%m%dT%H%M%S%fZ')}.log"
    status = {
        "pid": os.getpid(),
        "run": str(run),
        "started_at": started.isoformat(),
        "state": "archiving",
        "log": str(log_path),
        "interval_seconds": interval,
    }
    write_status(folder, status)
    with log_path.open("w") as log:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "scripts.archive_checkpoints", "--run", str(run)],
                cwd=PROJECT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            code = result.returncode
        except subprocess.TimeoutExpired:
            log.write(f"\nArchive check exceeded {timeout} seconds; retry next hour.\n")
            code = 124
        except OSError as error:
            log.write(f"\nArchive process could not start: {error}\n")
            code = 1
    next_check = started.timestamp() + interval
    while next_check <= time.time():
        next_check += interval
    status.update(
        state="waiting" if code == 0 else "archive_failed",
        exit_code=code,
        finished_at=datetime.now(timezone.utc).isoformat(),
        next_check_unix=next_check,
    )
    write_status(folder, status)
    with (folder / "history.jsonl").open("a") as history:
        history.write(json.dumps(status) + "\n")
    print(json.dumps(status), flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--interval", type=int, default=3600)
    parser.add_argument("--timeout", type=int, default=3000)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not 0 < args.timeout < args.interval:
        parser.error("Require 0 < timeout < interval")
    run = args.run.resolve()
    if not (run / "signature.json").is_file():
        parser.error("Run must contain signature.json")
    folder = run / "archive-service"
    folder.mkdir(exist_ok=True)
    with (folder / "scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Restart checks immediately, so downtime cannot defer an unarchived checkpoint.
        while not (folder / "STOP").exists():
            status = run_check(run, folder, interval=args.interval, timeout=args.timeout)
            if args.once:
                return status["exit_code"]
            deadline = time.monotonic() + max(0, status["next_check_unix"] - time.time())
            while time.monotonic() < deadline:
                if (folder / "STOP").exists():
                    return 0
                time.sleep(min(30, max(0, deadline - time.monotonic())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
