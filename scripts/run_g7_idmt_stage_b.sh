#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
MODE="${1:-all}"

cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:src"

"$PYTHON_BIN" -u -m dads_crnn.prepare_g7_idmt_stage_b \
  "$MODE" \
  --root "$ROOT" \
  --config configs/g7_idmt_stage_b.yaml

"$PYTHON_BIN" -m unittest \
  tests.test_audio_float \
  tests.test_audio_24bit \
  tests.test_audio_identity
