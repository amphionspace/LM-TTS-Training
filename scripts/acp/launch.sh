#!/usr/bin/env bash
set -euo pipefail
# Translate platform variables here; the training repository only sees torchrun conventions.
export NNODES="${NNODES:-${SENSECORE_PYTORCH_NNODES:?ACP did not provide node count}}"
export NODE_RANK="${NODE_RANK:-${SENSECORE_PYTORCH_NODE_RANK:?ACP did not provide node rank}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:?ACP did not provide GPU count}}"
: "${MASTER_ADDR:?ACP did not provide MASTER_ADDR}"
: "${MASTER_PORT:?ACP did not provide MASTER_PORT}"
export PYTHON_BIN="${PYTHON_BIN:-/workspace/workspace/yanglin/envs/lm-tts/bin/python}"
PROJECT_DIR="${PROJECT_DIR:-/workspace/workspace/yanglin/LM-TTS-Training}"
exec bash "$PROJECT_DIR/scripts/run_train.sh" "$@"
