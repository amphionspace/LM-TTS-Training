"""Compare tiny integration checkpoints after uninterrupted/resumed training."""
import argparse
import json
import tempfile
from pathlib import Path

import torch
import numpy as np
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def main():
    p = argparse.ArgumentParser()
    p.add_argument("left")
    p.add_argument("right")
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, source in enumerate([args.left, args.right]):
            target = Path(tmp) / f"{i}.pt"
            dcp_to_torch_save(Path(source) / "distributed", target)
            paths.append(torch.load(target, map_location="cpu", weights_only=False))
        tensors = 0
        def compare(a, b, name):
            nonlocal tensors
            if isinstance(a, torch.Tensor):
                torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
                tensors += 1
            elif isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b)
            elif isinstance(a, dict):
                assert a.keys() == b.keys(), name
                for k in a:
                    compare(a[k], b[k], name + "." + str(k))
            elif isinstance(a, (list, tuple)):
                assert len(a) == len(b), name
                for i, (x, y) in enumerate(zip(a, b)):
                    compare(x, y, name + f"[{i}]")
            else:
                assert a == b, (name, a, b)
        compare(paths[0], paths[1], "checkpoint")
        lm = json.loads((Path(args.left) / "metadata.json").read_text())
        rm = json.loads((Path(args.right) / "metadata.json").read_text())
        assert lm == rm
        for rank in range(lm["world_size"]):
            left_rng = torch.load(Path(args.left) / f"rng-{rank}.pt", weights_only=False)
            right_rng = torch.load(Path(args.right) / f"rng-{rank}.pt", weights_only=False)
            compare(left_rng, right_rng, f"rng-{rank}")
        print(json.dumps({"status": "passed", "exact_tensor_matches": tensors,
                          "progress": lm["progress"]}, indent=2))


if __name__ == "__main__":
    main()
