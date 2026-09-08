#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PIP_CONFIG_FILE=/dev/null
export PIP_EXTRA_INDEX_URL=''
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_DISABLE_PIP_VERSION_CHECK=1
if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    "$PYTHON_BIN" -m venv --without-pip "$PROJECT_DIR/.venv"
fi
if ! "$PROJECT_DIR/.venv/bin/python" -m pip --version >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip --python "$PROJECT_DIR/.venv/bin/python" install --no-cache-dir pip==25.2
fi
"$PROJECT_DIR/.venv/bin/python" -m pip install --no-cache-dir pip==25.2 setuptools==80.9.0 wheel==0.45.1
"$PROJECT_DIR/.venv/bin/python" -m pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
"$PROJECT_DIR/.venv/bin/python" -m pip install --no-cache-dir -r "$PROJECT_DIR/requirements.lock.txt"
"$PROJECT_DIR/.venv/bin/python" -m pip check
