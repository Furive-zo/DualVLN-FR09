#!/usr/bin/env bash
set -euo pipefail
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
UI_HOST="${UI_HOST:-127.0.0.1}"
UI_PORT="${UI_PORT:-5803}"
export INFERENCE_URL="${INFERENCE_URL:-http://127.0.0.1:5801/infer}"
exec "$PYTHON_BIN" -m gunicorn \
  --chdir "$BUNDLE_DIR/runtime" \
  --workers 1 --threads 4 --timeout 320 \
  --bind "$UI_HOST:$UI_PORT" ui_wsgi:app
