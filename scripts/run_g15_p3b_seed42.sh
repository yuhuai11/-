#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
CONFIG="configs/g15_seed42_feasibility_p3b.yaml"
OUTPUT="artifacts/g15_constrained_adaptation/p3b_seed42"
PREFLIGHT="$OUTPUT/preflight/report.json"
LOG="logs/g15_p3b_seed42.log"

cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src:$ROOT/artifacts/g7_panns/python:$ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mode="${1:-preflight}"
case "$mode" in
  preflight)
    "$PYTHON_BIN" -u -m dads_crnn.train_g15_constrained \
      --config "$CONFIG" \
      --mode preflight
    ;;
  run)
    if [[ ! -s "$PREFLIGHT" ]]; then
      echo "G15 P3b preflight report is missing; run '$0 preflight' first" >&2
      exit 1
    fi
    "$PYTHON_BIN" -c \
      'import json,sys; r=json.load(open(sys.argv[1])); assert r.get("passed") is True and r.get("protocol") == "g15_p3b_seed42_safe_improvement_early_stopping_v2" and r.get("formal_training_started") is False and r.get("checkpoint_written") is False' \
      "$PREFLIGHT"
    if [[ -s "$OUTPUT/seed_42/summary.json" ]]; then
      echo "Refusing to overwrite completed G15 P3b seed42 run" >&2
      exit 1
    fi
    "$PYTHON_BIN" -u -m dads_crnn.train_g15_constrained \
      --config "$CONFIG" \
      --mode run \
      --resume
    ;;
  status)
    pgrep -af "dads_crnn.train_g15_constrained.*g15_seed42_feasibility_p3b.yaml.*--mode run" || true
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
    tail -n 30 "$LOG" 2>/dev/null || true
    ;;
  *)
    echo "Usage: $0 {preflight|run|status}" >&2
    exit 2
    ;;
esac
