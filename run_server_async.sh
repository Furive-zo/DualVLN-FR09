#!/usr/bin/env bash
# Async S2/S1: S2 goal worker on S2_DEVICE, S1 control path on S1_DEVICE.
set -euo pipefail
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5801}"
export ASYNC_S2=1
export S2_DEVICE="${S2_DEVICE:-cuda:0}"
export S1_DEVICE="${S1_DEVICE:-cuda:1}"
# Goal age is quantized by the 2 Hz frame period (1.0/1.5/2.0 s with S2 ~0.8 s); 2.25 s keeps
# the cap off a boundary and inside the offline-evaluated lag range (<= 2.0 s).
export MAX_GOAL_AGE_S="${MAX_GOAL_AGE_S:-2.25}"
export S1_GPU_PREPROCESS="${S1_GPU_PREPROCESS:-1}"
exec "$PYTHON_BIN" -m gunicorn \
  --chdir "$BUNDLE_DIR/runtime" \
  --workers 1 --threads 1 --timeout 300 --graceful-timeout 30 \
  --bind "$HOST:$PORT" wsgi:app
