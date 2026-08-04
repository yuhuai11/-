#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
MODE="${1:-preflight}"
CONFIG="${G7_R0_CONFIG:-configs/g7_idmt_r0.yaml}"

cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:src:artifacts/g7_panns/python:artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch"

"$PYTHON_BIN" -u -m dads_crnn.evaluate_g7_idmt_r0 \
  "$MODE" \
  --root "$ROOT" \
  --config "$CONFIG"
