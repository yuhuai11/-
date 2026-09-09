#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user1/JJZ/ABDDV-CRNN"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
CONFIG="configs/g7_strict_retrain_v1.yaml"
MANIFEST="artifacts/g7_leakage_fixed_v2/data/manifest.csv"
LOG="logs/g7_strict_retrain_v1.log"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/artifacts/g7_panns/python:$ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "${1:-status}" in
  validate)
    bash scripts/run_g7_leakage_fixed.sh validate-full
    ;;
  preflight)
    bash scripts/run_g7_leakage_fixed.sh validate-full
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config "$CONFIG" --manifest "$MANIFEST" \
      --seeds 42 --preflight-only
    ;;
  train)
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config "$CONFIG" --manifest "$MANIFEST" \
      --seeds 42 43 44
    ;;
  resume)
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config "$CONFIG" --manifest "$MANIFEST" \
      --seeds 42 43 44 --resume
    ;;
  audit-near-duplicates)
    "$PYTHON_BIN" -u -m dads_crnn.audit_dads_near_duplicates
    ;;
  status)
    pgrep -af "dads_crnn.train_panns.*g7_strict_retrain_v1.yaml" || true
    nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader || true
    tail -n 30 "$LOG" 2>/dev/null || true
    ;;
  *)
    echo "Usage: $0 {validate|preflight|train|resume|audit-near-duplicates|status}" >&2
    exit 2
    ;;
esac
