#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-$PROJECT_DIR/../envs/lm-tts}"
CONDA_BIN="${CONDA_BIN:-conda}"
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    "$CONDA_BIN" create --yes --prefix "$ENV_PREFIX" python=3.11 pip
fi
"$ENV_PREFIX/bin/python" -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
"$ENV_PREFIX/bin/python" -m pip install -r "$PROJECT_DIR/requirements.lock.txt"
"$ENV_PREFIX/bin/python" -m pip check
