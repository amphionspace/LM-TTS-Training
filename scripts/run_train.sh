#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Variable packed batches can fragment cached CUDA allocations after resume.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec "$PROJECT_DIR/.venv/bin/torchrun" --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" -m qwen3_train.train "$@"
