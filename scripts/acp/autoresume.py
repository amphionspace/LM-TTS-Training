"""Resume listed runs from their latest checkpoint when their ACP job stops and nodes are free.

A run is resumed only when its newest job is FAILED, SUSPENDED or STOPPED, it has a completed
checkpoint, it has not finished, and its run directory has no NO_AUTORESUME file. Each stopped
job is handled at most once. Two fast failures in a row (less than --fast-fail-minutes of
runtime each) stop resuming that run, since a deterministic crash would only loop. The platform
exposes no free-node query, so "free" means the workspace's active jobs plus this run fit within
--capacity nodes. Resubmission snapshots this checkout, so a dirty lm_tts/scripts/configs tree
blocks it. Run detached: setsid nohup python -m scripts.acp.autoresume --capacity N CONFIG... &
"""

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

from lm_tts.config import read_yaml
from lm_tts.training.config import load_config

from .api import jobs_url, request

PROJECT = Path(__file__).resolve().parents[2]
STOPPED = {"FAILED", "SUSPENDED", "STOPPED"}
INACTIVE = STOPPED | {"SUCCEEDED", "DELETED"}


def log(event, **fields):
    stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    print(json.dumps({"time": stamp, "event": event, **fields}, ensure_ascii=False), flush=True)


def parse_time(value):
    if not value or value.startswith("0001-"):
        return None
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))


def runtime_seconds(job):
    start = parse_time(job.get("start_time"))
    end = parse_time(job.get("complete_time")) or parse_time(job.get("suspend_time"))
    return (end - start).total_seconds() if start and end else 0.0


def list_jobs(platform):
    jobs, token = [], ""
    while True:
        url = jobs_url(platform) + "?page_size=100" + (f"&page_token={token}" if token else "")
        page = request("GET", url, home=platform["home"])
        jobs += page.get("training_jobs", [])
        token = page.get("next_page_token")
        if not token or len(jobs) >= 300:
            return jobs


def nodes(job):
    return sum(role.get("total_replicas", 0) for role in job.get("roles", []))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path, help="Experiment YAMLs to keep running")
    parser.add_argument("--capacity", type=int, required=True, help="Workspace node budget")
    parser.add_argument("--interval", type=int, default=1800, help="Seconds between checks")
    parser.add_argument("--fast-fail-minutes", type=float, default=20)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    runs = []
    for path in args.configs:
        training = load_config(path)
        platform = read_yaml(path, keys=("acp",))["acp"]
        runs.append(
            {
                "config": path.resolve(),
                "name": training["train"]["run_name"],
                "dir": Path(training["train"]["output"]),
                "platform": platform,
                "python": training["paths"]["python"],
            }
        )
    handled, fast_fails = {}, {}
    log("started", capacity=args.capacity, runs=[r["name"] for r in runs])
    while True:
        try:
            jobs = list_jobs(runs[0]["platform"])
        except Exception as error:  # Transient API errors must not end supervision.
            log("list_failed", error=str(error))
            jobs = None
        for run in runs if jobs is not None else []:
            name, directory = run["name"], run["dir"]
            own = sorted(
                (j for j in jobs if j.get("display_name") == name),
                key=lambda j: j.get("create_time", ""),
            )
            if not own:
                continue
            job = own[-1]
            state, job_id = job.get("state"), job["name"]
            if state not in STOPPED or handled.get(name) == job_id:
                continue
            handled[name] = job_id
            reason = None
            if (directory / "completion.json").exists():
                reason = "run finished"
            elif (directory / "NO_AUTORESUME").exists():
                reason = "NO_AUTORESUME present"
            elif not (directory / "checkpoints/latest").exists():
                reason = "no checkpoint to resume"
            if reason:
                log("skip", run=name, job=job_id, state=state, reason=reason)
                continue
            ran = runtime_seconds(job)
            fast = state == "FAILED" and ran < args.fast_fail_minutes * 60
            fast_fails[name] = fast_fails.get(name, 0) + 1 if fast else 0
            if fast_fails[name] >= 2:
                log("give_up", run=name, job=job_id, reason="two fast failures in a row")
                continue
            dirty = subprocess.run(
                ["git", "status", "--porcelain", "--", "lm_tts", "scripts", "configs"],
                cwd=PROJECT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            if dirty:
                handled.pop(name)  # Retry once the tree is clean.
                log("blocked", run=name, job=job_id, reason="uncommitted source changes")
                continue
            used = sum(nodes(j) for j in jobs if j.get("state") not in INACTIVE)
            need = run["platform"]["nodes"]
            if used + need > args.capacity:
                handled.pop(name)  # Retry when nodes free up.
                log("waiting", run=name, job=job_id, used=used, need=need)
                continue
            latest = (directory / "checkpoints/latest").read_text().strip()
            log("resubmit", run=name, job=job_id, state=state, ran_seconds=ran, checkpoint=latest)
            result = subprocess.run(
                [run["python"], "-m", "scripts.acp.submit", "--config", str(run["config"])]
                + ["--resume", "latest", "--submit"],
                cwd=PROJECT,
                capture_output=True,
                text=True,
            )
            new = next(
                (
                    line.split('"')[3]
                    for line in result.stdout.splitlines()
                    if '"name": "pt-' in line
                ),
                None,
            )
            log(
                "submitted" if result.returncode == 0 else "submit_failed",
                run=name,
                job=new,
                stderr=result.stderr[-500:] if result.returncode else "",
            )
            if new:
                jobs.append(
                    {
                        "display_name": name,
                        "name": new,
                        "state": "INIT",
                        "roles": [{"total_replicas": need}],
                        "create_time": "9999",
                    }
                )
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
