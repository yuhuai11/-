#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
CONFIG="configs/g14_b_regression.yaml"

cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src:$ROOT/artifacts/g7_panns/python:$ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mode="${1:-preflight}"
case "$mode" in
  preflight)
    "$PYTHON_BIN" -u -m dads_crnn.evaluate_g14_regression \
      --config "$CONFIG" \
      --mode preflight
    ;;
  run)
    output="artifacts/g14_domain_generalization/regression/summary.json"
    if [[ -s "$output" ]]; then
      echo "Refusing to overwrite completed G14-B result: $output" >&2
      exit 1
    fi
    "$PYTHON_BIN" -u -m dads_crnn.evaluate_g14_regression \
      --config "$CONFIG" \
      --mode run
    ;;
  status)
    pgrep -af "dads_crnn.evaluate_g14_regression.*--mode run" || true
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
    if [[ -s artifacts/g14_domain_generalization/regression/summary.json ]]; then
      echo "G14-B summary is complete."
    else
      find artifacts/g14_domain_generalization/regression \
        -maxdepth 2 -type f -printf '%p %s bytes\n' 2>/dev/null | sort
    fi
    ;;
  *)
    echo "Usage: $0 {preflight|run|status}" >&2
    exit 2
    ;;
esac
