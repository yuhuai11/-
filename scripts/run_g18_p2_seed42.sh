#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
MODE="${1:-train}"

cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src:$ROOT/artifacts/g7_panns/python:$ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "$MODE" in
  train)
    exec "$PYTHON_BIN" -u -m dads_crnn.train_g18_model_id \
      --config configs/g18_model_id_seed42.yaml
    ;;
  resume)
    exec "$PYTHON_BIN" -u -m dads_crnn.train_g18_model_id \
      --config configs/g18_model_id_seed42.yaml \
      --resume
    ;;
  status)
    tail -n 40 logs/g18_p2_seed42.log
    ;;
  *)
    echo "Usage: $0 {train|resume|status}" >&2
    exit 2
    ;;
esac
