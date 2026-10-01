"""Wake a persistent Codex reviewer; training decisions belong to the agent."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--instructions", required=True, type=Path)
    parser.add_argument("--interval", type=int, default=3600)
    parser.add_argument("--timeout", type=int, default=3000)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not 0 < args.timeout < args.interval:
        parser.error("Require 0 < timeout < interval")
    run = args.run.resolve()
    instructions = args.instructions.resolve()
    instructions.read_text()
    folder = run / "supervision"
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / "scheduler.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(folder / "scheduler-process.json", {"pid": os.getpid(), "run": str(run)})
    session_path = folder / "session-id"
    deadline = time.monotonic()
    while not (folder / "STOP").exists() and not (folder / "COMPLETE").exists():
        # Preserve the next scheduled wake-up across a launcher restart.
        previous = folder / "scheduler-status.json"
        if previous.exists() and not args.once:
            next_check = json.loads(previous.read_text()).get("next_check_unix", 0)
            deadline = max(deadline, time.monotonic() + next_check - time.time())
        while time.monotonic() < deadline:
            if (folder / "STOP").exists() or (folder / "COMPLETE").exists():
                return
            time.sleep(min(30, deadline - time.monotonic()))
        started = datetime.now(timezone.utc)
        stamp = started.strftime("%Y%m%dT%H%M%SZ")
        session = session_path.read_text().strip() if session_path.exists() else None
        prompt = (
            "Perform ONE authorized training inspection now, then finish this turn. "
            "You are the persistent hourly supervisor agent. Read the current instructions "
            f"at {instructions} and their linked supervision record before acting. "
            f"Run directory: {run}. UTC inspection start: {started.isoformat()}. "
            "Append evidence and actions to the designated Chinese docs record. "
            "Do not sleep until the next hour or launch another supervisor. "
            "The scheduler does not diagnose or restart training; you make those decisions."
        )
        command = [
            "codex",
            "exec",
            "--approve-for-me",
            "--disable",
            "apps",
            "--add-dir",
            str(run),
            "--add-dir",
            str(PROJECT.parent / "acp"),
            "-C",
            str(PROJECT),
            "--json",
            "-o",
            str(folder / f"{stamp}.md"),
        ]
        command += ["resume", session, "-"] if session else ["-"]
        status = {
            "started_at": started.isoformat(),
            "session_id": session,
            "interval_seconds": args.interval,
            "state": "reviewing",
            "latest_log": str(folder / f"{stamp}.jsonl"),
        }
        write_json(folder / "scheduler-status.json", status)
        with (folder / f"{stamp}.jsonl").open("w") as log:
            child = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=PROJECT,
                start_new_session=True,
            )
            status["review_pid"] = child.pid
            write_json(folder / "scheduler-status.json", status)
            try:
                child.communicate(prompt, timeout=args.timeout)
                code = child.returncode
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.communicate(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.communicate()
                code = 124
        with (folder / f"{stamp}.jsonl").open() as log:
            for line in log:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("type") == "thread.started":
                    actual = event["thread_id"]
                    if session and session != actual:
                        raise RuntimeError("Codex resumed a different session")
                    session = actual
                    session_path.write_text(session + "\n")
        next_check = started.timestamp() + args.interval
        while next_check <= time.time():
            next_check += args.interval
        status.update(
            session_id=session,
            review_exit_code=code,
            finished_at=datetime.now(timezone.utc).isoformat(),
            next_check_unix=next_check,
            state="waiting" if code == 0 else "review_failed",
        )
        write_json(folder / "scheduler-status.json", status)
        print(json.dumps(status), flush=True)
        if args.once:
            return
        deadline = time.monotonic() + next_check - time.time()


if __name__ == "__main__":
    main()
