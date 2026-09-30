"""Open a run dashboard, optionally restoring curves from its metrics journal."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from qwen3_train.config import read_yaml
from qwen3_train.evaluation.telemetry import write_evaluation
from qwen3_train.training.telemetry import setup_dashboard, write_training, write_validation


def rebuild(run, destination=None):
    """Replace restored events from current journals, preserving live training events."""
    from torch.utils.tensorboard import SummaryWriter

    directory = destination or run / "tensorboard/restored"
    directory.mkdir(parents=True, exist_ok=True)
    previous = list(directory.glob("events.out.tfevents.*"))
    records = {}
    journal = run / "metrics.jsonl"
    if journal.exists():
        with journal.open() as stream:
            for line in stream:
                record = json.loads(line)
                for kind in ("train", "val"):
                    if kind in record:
                        # A resumed run can log the same step again; its last value wins.
                        records[(record["step"], kind)] = record[kind]
    groups = ["optimization", "performance", "batch"]
    # Publish only after successful replay, so missing audio cannot destroy a dashboard.
    with tempfile.TemporaryDirectory(prefix=".rebuild-", dir=directory.parent) as temporary:
        with SummaryWriter(temporary) as writer:
            setup_dashboard(writer, groups)
            for (step, kind), metrics in sorted(records.items()):
                if kind == "train":
                    write_training(writer, metrics, step, groups)
                else:
                    write_validation(writer, metrics, step)
            for report in sorted(run.glob("**/report.json")):
                samples = report.with_name("samples.json")
                if not samples.exists():
                    continue
                result = json.loads(report.read_text())
                if "identity" in result and result["identity"].get("step") is not None:
                    config = result["identity"].get("config", {})
                    write_evaluation(
                        writer,
                        result,
                        json.loads(samples.read_text()),
                        result["identity"]["step"],
                        config.get("tensorboard", {}),
                        max_audio_seconds=config.get("max_audio_seconds", 180),
                    )
        for event in Path(temporary).glob("events.out.tfevents.*"):
            event.replace(directory / event.name)
        for event in previous:
            event.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run name, or an explicit directory")
    parser.add_argument("--config", default="configs/train-bf16.yaml")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild restored curves and evaluation previews from current journals",
    )
    parser.add_argument("--rebuild-only", action="store_true")
    parser.add_argument("--port", type=int, default=6006)
    args = parser.parse_args()
    run = Path(args.run)
    if not run.is_absolute() and len(run.parts) == 1:
        run = Path(read_yaml(args.config)["train"]["runs_root"]) / run
    if not run.is_dir():
        parser.error(f"Run does not exist: {run}")
    if args.rebuild or args.rebuild_only:
        for journal in sorted(run.glob("**/metrics.jsonl")):
            if journal.parent != run:
                rebuild(
                    journal.parent, run / "tensorboard/restored" / journal.parent.relative_to(run)
                )
        rebuild(run)
    if not args.rebuild_only:
        bypass = ",".join(
            filter(
                None,
                [
                    os.environ.get("no_proxy", os.environ.get("NO_PROXY", "")),
                    "localhost",
                    "127.0.0.1",
                    "::1",
                ],
            )
        )
        os.environ["no_proxy"] = os.environ["NO_PROXY"] = bypass
        from tensorboard import program

        dashboard = program.TensorBoard()
        dashboard.configure(
            argv=[
                "tensorboard",
                "--logdir",
                str(run / "tensorboard"),
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
            ]
        )
        print(dashboard.launch(), flush=True)
        import threading

        threading.Event().wait()


if __name__ == "__main__":
    main()
