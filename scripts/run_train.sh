#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "$PROJECT_DIR/.venv/bin/torchrun" --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" -m qwen3_train.train "$@"
