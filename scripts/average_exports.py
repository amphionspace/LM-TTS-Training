"""Create an FP32 equal-weight average of compatible flat model exports."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from lm_tts.artifacts import file_hash


def average_states(*states):
    if len(states) < 2:
        raise ValueError("At least two source models are required")
    left, *others = states
    if any(left.keys() != right.keys() for right in others):
        raise ValueError("Source tensor keys differ")
    averaged = {}
    for key, a in left.items():
        tensors = [a, *(right[key] for right in others)]
        if any(a.shape != b.shape or a.dtype != b.dtype for b in tensors[1:]):
            raise ValueError(f"Source tensor shape or dtype differs: {key}")
        identical = all(torch.equal(a, b) for b in tensors[1:])
        if a.is_floating_point():
            if any(not torch.isfinite(b).all() for b in tensors):
                raise ValueError(f"Non-finite source tensor: {key}")
            # Preserve shared frozen parameters exactly, including for thirds.
            if identical:
                averaged[key] = a.float().clone().contiguous()
            else:
                weight = 1.0 / len(tensors)
                value = a.float().mul(weight)
                for b in tensors[1:]:
                    value.add_(b.float(), alpha=weight)
                averaged[key] = value.contiguous()
        else:
            if not identical:
                raise ValueError(f"Non-floating source tensor differs: {key}")
            averaged[key] = a.contiguous()
    return averaged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.sources) < 2:
        parser.error("--sources requires at least two model directories")
    torch.set_num_threads(4)
    reports = []
    for source in args.sources:
        report = json.loads((source / "export.json").read_text())
        for name, expected in report["artifact_sha256"].items():
            if file_hash(source / name) != expected:
                raise ValueError(f"Source artifact changed: {source / name}")
        if report["weights_sha256"] != report["artifact_sha256"]["model.safetensors"]:
            raise ValueError(f"Source weights identity differs: {source}")
        reports.append(report)
    artifacts = [
        {k: v for k, v in r["artifact_sha256"].items() if k != "model.safetensors"} for r in reports
    ]
    if any(artifacts[0] != artifact for artifact in artifacts[1:]):
        raise ValueError("Source configuration or tokenizer artifacts differ")
    if any(reports[0]["use_speaker_embedding"] != r["use_speaker_embedding"] for r in reports[1:]):
        raise ValueError("Source speaker conditioning differs")
    states = [load_file(str(source / "model.safetensors")) for source in args.sources]
    state = average_states(*states)
    args.output.mkdir(parents=True, exist_ok=False)
    save_file(state, str(args.output / "model.safetensors"), metadata={"format": "pt"})
    for name in artifacts[0]:
        shutil.copy2(args.sources[0] / name, args.output / name)
    (args.output / "speech_tokenizer").symlink_to(
        (args.sources[0] / "speech_tokenizer").resolve(), target_is_directory=True
    )
    report = {
        "operation": "equal_weight_average",
        "precision": "fp32",
        "use_speaker_embedding": reports[0]["use_speaker_embedding"],
        "sources": [
            {"path": str(p.resolve()), "weight": 1.0 / len(reports), "export": r}
            for p, r in zip(args.sources, reports, strict=True)
        ],
        "weights_sha256": file_hash(args.output / "model.safetensors"),
        "tensor_count": len(state),
        "artifact_sha256": {
            p.name: file_hash(p) for p in sorted(args.output.iterdir()) if p.is_file()
        },
    }
    (args.output / "export.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "sources"}), flush=True)


if __name__ == "__main__":
    main()
