"""Platform-independent training entry point."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .training.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Completed checkpoint directory, or 'latest'")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    for key, value in config.get("environment", {}).items():
        os.environ.setdefault(key, str(value))
        config["environment"][key] = os.environ[key]
    if args.eval_only and not config["data"].get("val_build"):
        parser.error("--eval-only requires an independently isolated validation build")
    if args.max_steps is not None:
        if args.max_steps < 1 or (
            config["train"].get("schedule_steps") is not None
            and args.max_steps > config["train"]["schedule_steps"]
        ):
            parser.error("--max-steps must be positive and within schedule_steps")
        config["train"]["max_steps"] = args.max_steps
    if args.output:
        config["train"]["output"] = args.output
        if "run_name" in config["train"]:
            destination = Path(args.output).resolve()
            config["train"]["run_name"] = destination.name
            config["train"]["runs_root"] = str(destination.parent)
    from .training.engine import run

    run(config, resume=args.resume, eval_only=args.eval_only)


if __name__ == "__main__":
    main()
