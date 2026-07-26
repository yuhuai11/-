#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
MODE="${1:-status}"
CONFIG="configs/g18_final_holdout.yaml"
OUTPUT="artifacts/g18_model_identification/p6_final_holdout_once"

cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src:$ROOT/artifacts/g7_panns/python:$ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "$MODE" in
  preflight)
    exec "$PYTHON_BIN" -u -m dads_crnn.evaluate_g18_final_holdout \
      --config "$CONFIG" \
      --preflight-only
    ;;
  run)
    if [[ ! -f artifacts/g18_model_identification/p6_preflight/report.json ]]; then
      echo "G18 P6 preflight is missing; run '$0 preflight' first" >&2
      exit 1
    fi
    if [[ -e "$OUTPUT/FORMAL_RUN_STARTED.json" || -e "$OUTPUT/summary.json" ]]; then
      echo "G18 P6 was already started; fresh rerun is forbidden" >&2
      exit 1
    fi
    exec "$PYTHON_BIN" -u -m dads_crnn.evaluate_g18_final_holdout \
      --config "$CONFIG"
    ;;
  resume)
    if [[ -f "$OUTPUT/summary.json" ]]; then
      echo "G18 P6 final result already exists; Holdout is closed" >&2
      exit 1
    fi
    if [[ ! -f "$OUTPUT/FORMAL_RUN_STARTED.json" ]]; then
      echo "G18 P6 formal-run marker is missing; use '$0 run'" >&2
      exit 1
    fi
    exec "$PYTHON_BIN" -u -m dads_crnn.evaluate_g18_final_holdout \
      --config "$CONFIG" \
      --resume
    ;;
  status)
    if [[ -f "$OUTPUT/summary.json" ]]; then
      echo "G18 P6 final Holdout complete and closed"
      tail -n 100 logs/g18_p6_final_holdout.log
    elif [[ -f "$OUTPUT/FORMAL_RUN_STARTED.json" ]]; then
      echo "G18 P6 formal Holdout run has started but is incomplete"
      tail -n 100 logs/g18_p6_final_holdout.log
    else
      echo "G18 P6 formal Holdout has not started"
    fi
    ;;
  *)
    echo "Usage: $0 {preflight|run|resume|status}" >&2
    exit 2
    ;;
esac
