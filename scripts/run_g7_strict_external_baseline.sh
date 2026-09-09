#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user1/JJZ/ABDDV-CRNN"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
LOG="logs/g7_strict_external_baseline.log"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/artifacts/g7_panns/python:$ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "${1:-status}" in
  preflight)
    "$PYTHON_BIN" -u -m dads_crnn.evaluate_g7_strict_external_baseline \
      --preflight-only
    ;;
  evaluate)
    "$PYTHON_BIN" -u -m dads_crnn.evaluate_g7_strict_external_baseline \
      --device cuda --batch-size 256
    ;;
  aggregate-one-second)
    "$PYTHON_BIN" -u -m dads_crnn.evaluate_g7_strict_multiscale
    ;;
  status)
    pgrep -af "dads_crnn.evaluate_g7_strict_external_baseline" || true
    nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader || true
    tail -n 40 "$LOG" 2>/dev/null || true
    ;;
  *)
    echo "Usage: $0 {preflight|evaluate|aggregate-one-second|status}" >&2
    exit 2
    ;;
esac
