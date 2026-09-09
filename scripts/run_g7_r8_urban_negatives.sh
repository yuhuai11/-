#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user1/JJZ/ABDDV-CRNN"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
CONFIG="configs/g7_r8_urban_negatives.yaml"
MANIFEST="artifacts/g7_r8_urban_negatives/data/manifest.csv"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/artifacts/g7_panns/python:$ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "${1:-status}" in
  prepare)
    "$PYTHON_BIN" -u -m dads_crnn.prepare_g7_r8_urban_negatives
    ;;
  preflight)
    "$PYTHON_BIN" -u -m dads_crnn.audit_g7_r8_urban_negatives
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config "$CONFIG" --manifest "$MANIFEST" --seeds 42 --preflight-only
    ;;
  train)
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config "$CONFIG" --manifest "$MANIFEST" --seeds 42
    ;;
  resume)
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config "$CONFIG" --manifest "$MANIFEST" --seeds 42 --resume
    ;;
  external)
    "$PYTHON_BIN" -u -m dads_crnn.evaluate_g7_r8_external \
      --device cuda --batch-size 256
    ;;
  status)
    pgrep -af "dads_crnn.train_panns.*g7_r8_urban_negatives.yaml" || true
    nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader || true
    ;;
  *)
    echo "Usage: $0 {prepare|preflight|train|resume|external|status}" >&2
    exit 2
    ;;
esac
