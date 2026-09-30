#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/../envs/lm-tts/bin/python}"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/../.cache/huggingface}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$PROJECT_DIR/../.cache/numba}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Variable packed batches can fragment cached CUDA allocations after resume.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
if (( NNODES == 1 )); then
    LAUNCH_ARGS=(--standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE")
else
    : "${MASTER_ADDR:?Multi-node training requires MASTER_ADDR}"
    : "${NODE_RANK:?Multi-node training requires NODE_RANK}"
    LAUNCH_ARGS=(--nnodes="$NNODES" --nproc_per_node="$NPROC_PER_NODE"
                 --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR"
                 --master_port="${MASTER_PORT:-29500}")
fi
exec "$PYTHON_BIN" -m torch.distributed.run "${LAUNCH_ARGS[@]}" -m qwen3_train.train "$@"
