#!/usr/bin/env bash
# Run on the workstation, from the project root. Never modifies other environments.
set -euo pipefail
ROOT=$(pwd -P)
test -f "$ROOT/pyproject.toml" || { echo "Run from the project root" >&2; exit 2; }
BASE=${1:-python3}
"$BASE" -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ required"'
"$BASE" -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip
"$ROOT/.venv/bin/python" -m pip install torch --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
"$ROOT/.venv/bin/python" -m pip install -e '.[dev]'
