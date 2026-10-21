#!/usr/bin/env bash
set -euo pipefail
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5801}"
export DEVICE="${DEVICE:-cuda:0}"
exec "$PYTHON_BIN" -m gunicorn \
  --chdir "$BUNDLE_DIR/runtime" \
  --workers 1 --threads 1 --timeout 300 --graceful-timeout 30 \
  --bind "$HOST:$PORT" wsgi:app
