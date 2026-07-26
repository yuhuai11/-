#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
CONFIG="configs/g14_d_counterfactual_evaluation.yaml"
OUTPUT="artifacts/g14_domain_generalization/counterfactual_evaluation/evaluation.json"

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
    if [[ -s "$OUTPUT" ]]; then
      echo "Refusing to overwrite completed G14-D evaluation: $OUTPUT" >&2
      exit 1
    fi
    "$PYTHON_BIN" -u -m dads_crnn.evaluate_g14_counterfactual \
      --config "$CONFIG" \
      --mode run
    ;;
  status)
    pgrep -af "dads_crnn.evaluate_g14_counterfactual.*--mode run" || true
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
    find artifacts/g14_domain_generalization/counterfactual_evaluation \
      -maxdepth 1 -type f -printf '%f %s bytes\n' 2>/dev/null | sort
    ;;
  *)
    echo "Usage: $0 {preflight|run|status}" >&2
    exit 2
    ;;
esac
