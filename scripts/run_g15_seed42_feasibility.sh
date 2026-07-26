#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"

cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src:$ROOT/artifacts/g7_panns/python:$ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mode="${1:-preflight}"
case "$mode" in
  preflight)
    "$PYTHON_BIN" -u -m dads_crnn.train_g15_constrained \
      --config configs/g15_seed42_feasibility.yaml \
      --mode preflight
    ;;
  run)
    if [[ -s artifacts/g15_constrained_adaptation/p3_seed42/seed_42/summary.json ]]; then
      echo "Refusing to overwrite completed G15 seed42 run" >&2
      exit 1
    fi
    "$PYTHON_BIN" -u -m dads_crnn.train_g15_constrained \
      --config configs/g15_seed42_feasibility.yaml \
      --mode run \
      --resume
    ;;
  status)
    pgrep -af "dads_crnn.train_g15_constrained.*--mode run" || true
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
    tail -n 20 logs/g15_seed42_feasibility.log 2>/dev/null || true
    ;;
  *)
    echo "Usage: $0 {preflight|run|status}" >&2
    exit 2
    ;;
esac
