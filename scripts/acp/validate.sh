#!/usr/bin/env bash
set -euo pipefail
VALIDATION_ROOT="${1:?Provide the prepared validation directory}"
ACP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-/workspace/workspace/yanglin/LM-TTS-Training}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/workspace/yanglin/envs/lm-tts/bin/python}"
NODE_RANK="${SENSECORE_PYTORCH_NODE_RANK:?ACP node rank missing}"
NNODES="${SENSECORE_PYTORCH_NNODES:?ACP node count missing}"
NPROC_PER_NODE="${SENSECORE_ACCELERATE_DEVICE_COUNT:?ACP GPU count missing}"
export PROJECT_DIR PYTHON_BIN NODE_RANK NNODES NPROC_PER_NODE
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/tests${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$PROJECT_DIR/../.cache/numba}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/../.cache/huggingface}"
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
cd "$PROJECT_DIR"
NODE_OUTPUT="$VALIDATION_ROOT/node-$NODE_RANK"
mkdir -p "$NODE_OUTPUT"
"$PYTHON_BIN" "$ACP_DIR/validate_environment.py" --output "$NODE_OUTPUT/environment.json"
ORIGINAL_MASTER_PORT="${MASTER_PORT:?ACP master port missing}"
STAGE=0
for PRECISION in bf16 fp32; do
    ROOT="$VALIDATION_ROOT/$PRECISION"
    for MODE in continuous resumed resumed-final; do
        STAGE=$(( STAGE + 1 ))
        export MASTER_PORT=$(( ORIGINAL_MASTER_PORT + STAGE ))
        STEPS=4
        CONFIG="$ROOT/continuous.yaml"
        EXTRA_ARGS=()
        if [[ "$MODE" == resumed ]]; then
            CONFIG="$ROOT/resumed.yaml"
            STEPS=2
        elif [[ "$MODE" == resumed-final ]]; then
            CONFIG="$ROOT/resumed.yaml"
            EXTRA_ARGS=(--resume latest)
        fi
        timeout --signal=TERM --kill-after=30s 300s bash "$ACP_DIR/launch.sh" \
            --config "$CONFIG" --max-steps "$STEPS" "${EXTRA_ARGS[@]}" \
            > "$NODE_OUTPUT/$PRECISION-$MODE.log" 2>&1
        printf 'completed %s %s on node %s\n' "$PRECISION" "$MODE" "$NODE_RANK"
    done
done
if (( NODE_RANK == 0 )); then
    for PRECISION in bf16 fp32; do
        "$PYTHON_BIN" tests/check_training_resume.py --compare-only \
            --output "$VALIDATION_ROOT/$PRECISION" --precision "$PRECISION" \
            --world-size "$(( NNODES * NPROC_PER_NODE ))" \
            > "$NODE_OUTPUT/$PRECISION-comparison.log" 2>&1
    done
    "$PYTHON_BIN" -m scripts.evaluate --config configs/evaluation.yaml \
        --pairs artifacts/evaluation-smoke/pairs.json --output "$VALIDATION_ROOT/scores" \
        --tensorboard "$VALIDATION_ROOT/tensorboard/evaluation" --step 0 \
        > "$NODE_OUTPUT/evaluation.log" 2>&1
    "$PYTHON_BIN" "$ACP_DIR/verify_validation.py" "$VALIDATION_ROOT"
else
    for _ in $(seq 1 180); do
        if [[ -f "$VALIDATION_ROOT/COMPLETE" ]]; then
            break
        fi
        sleep 2
    done
    [[ -f "$VALIDATION_ROOT/COMPLETE" ]]
fi
