#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
MODE="${1:-}"
CONFIG="$ROOT/configs/g13_external_confirmation_eval.yaml"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found: $PYTHON" >&2
  exit 1
fi

mkdir -p "$ROOT/logs"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/artifacts/g7_panns/python:$ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"

case "$MODE" in
  freeze)
    exec "$PYTHON" -u -m dads_crnn.evaluate_g13_confirmation freeze --config "$CONFIG"
    ;;
  preflight)
    exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      "$PYTHON" -u -m dads_crnn.evaluate_g13_confirmation preflight --config "$CONFIG"
    ;;
  evaluate)
    if [[ "${G13_CONFIRM_ONCE:-}" != "YES" ]]; then
      echo "Refusing to consume G13. Set G13_CONFIRM_ONCE=YES for the single final run." >&2
      exit 2
    fi
    exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      "$PYTHON" -u -m dads_crnn.evaluate_g13_confirmation evaluate \
      --config "$CONFIG" \
      --unlock-final-evaluation RUN_G13_EXTERNAL_CONFIRMATION_ONCE
    ;;
  *)
    echo "Usage: $0 {freeze|preflight|evaluate}" >&2
    exit 2
    ;;
esac
