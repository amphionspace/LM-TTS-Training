#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"
export OMP_NUM_THREADS=2
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/torchrun --standalone --nproc_per_node=2 tests/check_distributed_equivalence.py
.venv/bin/python scripts/make_smoke_fixture.py
mkdir -p runs
VERIFY_DIR="$(mktemp -d "$PROJECT_DIR/runs/verification.XXXXXX")"
bash scripts/run_train.sh --config configs/smoke.yaml --output "$VERIFY_DIR/continuous"
bash scripts/run_train.sh --config configs/smoke.yaml --output "$VERIFY_DIR/interrupted" --max-steps 1
bash scripts/run_train.sh --config configs/smoke.yaml --output "$VERIFY_DIR/interrupted" --resume latest
.venv/bin/python scripts/compare_checkpoints.py "$VERIFY_DIR/continuous/checkpoints/step-00000003" "$VERIFY_DIR/interrupted/checkpoints/step-00000003"
