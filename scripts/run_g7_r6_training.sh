#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user1/JJZ/ABDDV-CRNN"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
CONFIG="configs/g7_r6_dronenoise_control.yaml"
MANIFEST="artifacts/g7_r6_reusable_multicorpus/fit_manifest.csv"
LOG="logs/g7_r6_dronenoise_control_seed42.log"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/artifacts/g7_panns/python:$ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "${1:-status}" in
  prepare)
    bash scripts/run_g7_r6_protocol.sh prepare
    "$PYTHON_BIN" -u -m dads_crnn.audit_g7_r6_training
    ;;
  preflight)
    test -s "$MANIFEST"
    "$PYTHON_BIN" -u -m dads_crnn.audit_g7_r6_training
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config "$CONFIG" --manifest "$MANIFEST" --seeds 42 --preflight-only
    ;;
  train)
    test -s "$MANIFEST"
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config "$CONFIG" --manifest "$MANIFEST" --seeds 42
    ;;
  resume)
    test -s "$MANIFEST"
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config "$CONFIG" --manifest "$MANIFEST" --seeds 42 --resume
    ;;
  evaluate-dronenoise)
    "$PYTHON_BIN" -u -m dads_crnn.evaluate_g7_r6_dronenoise
    ;;
  evaluate-external)
    "$PYTHON_BIN" -u -m dads_crnn.evaluate_g7_r6_external_suite
    ;;
  status)
    pgrep -af "dads_crnn.train_panns.*g7_r6_dronenoise_control.yaml" || true
    nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader || true
    tail -n 30 "$LOG" 2>/dev/null || true
    ;;
  *)
    echo "Usage: $0 {prepare|preflight|train|resume|evaluate-dronenoise|evaluate-external|status}" >&2
    exit 2
    ;;
esac
