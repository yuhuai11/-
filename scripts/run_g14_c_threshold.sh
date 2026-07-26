#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
CONFIG="configs/g14_c_tune_only_threshold.yaml"

cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src:$ROOT/artifacts/g7_panns/python:$ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mode="${1:-preflight}"
case "$mode" in
  preflight)
    "$PYTHON_BIN" -u -m dads_crnn.evaluate_g14_threshold \
      --config "$CONFIG" \
      --mode "$mode"
    ;;
  fit)
    if [[ -s artifacts/g14_domain_generalization/threshold_recalibration/calibration.json ]]; then
      echo "Refusing to overwrite the sealed G14-C calibration." >&2
      exit 1
    fi
    "$PYTHON_BIN" -u -m dads_crnn.evaluate_g14_threshold \
      --config "$CONFIG" \
      --mode fit
    ;;
  evaluate)
    if [[ ! -s artifacts/g14_domain_generalization/threshold_recalibration/calibration.json ]]; then
      echo "Run and inspect the tune-only fit before evaluation." >&2
      exit 1
    fi
    if [[ -s artifacts/g14_domain_generalization/threshold_recalibration/evaluation.json ]]; then
      echo "Refusing to overwrite the completed G14-C evaluation." >&2
      exit 1
    fi
    "$PYTHON_BIN" -u -m dads_crnn.evaluate_g14_threshold \
      --config "$CONFIG" \
      --mode evaluate
    ;;
  status)
    pgrep -af "dads_crnn.evaluate_g14_threshold" || true
    find artifacts/g14_domain_generalization/threshold_recalibration \
      -maxdepth 2 -type f -printf '%p %s bytes\n' 2>/dev/null | sort
    ;;
  *)
    echo "Usage: $0 {preflight|fit|evaluate|status}" >&2
    exit 2
    ;;
esac
