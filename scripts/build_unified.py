"""Create a pinned indexed-reference training build from published features."""

import argparse
import json

from qwen3_train.data.build import create_build

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = create_build(args.recipe, args.output)
    print(
        json.dumps(
            {
                "build_id": manifest["build_id"],
                "ready_rows": sum(b["ready_rows"] for b in manifest["bindings"]),
            }
        )
    )
