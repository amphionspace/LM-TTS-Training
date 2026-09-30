"""Accept the ACP run only when node, training, resume and evaluation evidence agree."""

import json
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main(root):
    nodes = [json.loads(path.read_text()) for path in sorted(root.glob("node-*/environment.json"))]
    expected_nodes = int(nodes[0]["environment"]["SENSECORE_PYTORCH_NNODES"])
    assert len(nodes) == expected_nodes and expected_nodes > 1
    assert len({n["hostname"] for n in nodes}) == expected_nodes
    uuids = [
        line.split(",")[-1].strip() for node in nodes for line in node["gpu_inventory"].splitlines()
    ]
    world = sum(node["gpu_count"] for node in nodes)
    assert len(uuids) == world == len(set(uuids))
    reports = {}
    for precision in ("bf16", "fp32"):
        report = json.loads((root / precision / "comparison.json").read_text())
        assert report["status"] == "passed" and report["max_resume_weight_difference"] == 0
        assert report["world_size"] == world
        assert report["frozen_modules_unchanged"] and len(report["updated_modules"]) == 4
        for mode in ("continuous", "resumed"):
            metadata = report["checkpoint_metadata"][mode]
            assert metadata["world_size"] == world and metadata["progress"]["step"] == 4
            events = EventAccumulator(str(root / precision / mode / "tensorboard"))
            events.Reload()
            assert {
                "train/loss",
                "train/audio_seconds_per_second",
            } <= set(events.Tags()["scalars"])
        reports[precision] = {k: v for k, v in report.items() if k != "checkpoint_metadata"}
    evaluation = json.loads((root / "scores/report.json").read_text())
    for name in ("wer", "cer", "dnsmos_ovrl", "speaker_similarity"):
        assert name in evaluation["overall"]
    result = {
        "status": "passed",
        "world_size": world,
        "nodes": expected_nodes,
        "training": reports,
        "evaluation": evaluation["overall"],
        "hostnames": [n["hostname"] for n in nodes],
    }
    (root / "result.json").write_text(json.dumps(result, indent=2))
    (root / "COMPLETE").write_text("ok\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
