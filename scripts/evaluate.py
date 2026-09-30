"""Score generated audio and append only key metrics to TensorBoard."""

import argparse
import json

from qwen3_train.evaluation.report import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tensorboard")
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                args.config, args.pairs, args.output, tensorboard=args.tensorboard, step=args.step
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
