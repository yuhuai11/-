#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
CONFIG="configs/g15_p4_counterfactual_tune.yaml"
OUTPUT="artifacts/g15_constrained_adaptation/p4_counterfactual_tune_diagnostic"
LOG="logs/g15_p4_counterfactual_tune.log"

cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src:$ROOT/artifacts/g7_panns/python:$ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mode="${1:-preflight}"
case "$mode" in
  preflight)
    "$PYTHON_BIN" -u -m dads_crnn.evaluate_g14_counterfactual \
      --config "$CONFIG" \
      --mode preflight
    ;;
  run)
    if [[ ! -s "$OUTPUT/preflight.json" ]]; then
      echo "G15 P4 preflight is missing; run '$0 preflight' first" >&2
      exit 1
    fi
    if [[ -s "$OUTPUT/evaluation.json" ]]; then
      echo "Refusing to overwrite completed G15 P4 diagnostic" >&2
      exit 1
    fi
    "$PYTHON_BIN" -u -m dads_crnn.evaluate_g14_counterfactual \
      --config "$CONFIG" \
      --mode run
    ;;
  status)
    pgrep -af "dads_crnn.evaluate_g14_counterfactual.*g15_p4_counterfactual_tune.yaml.*--mode run" || true
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
    tail -n 30 "$LOG" 2>/dev/null || true
    ;;
  *)
    echo "Usage: $0 {preflight|run|status}" >&2
    exit 2
    ;;
esac
