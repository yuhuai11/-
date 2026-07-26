#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
MODE="${1:-run}"
SEEDS=(43 44)
CONFIG="configs/g18_multiseed_outlier_exposure.yaml"
OUTPUT="artifacts/g18_model_identification/p5_multiseed_outlier_exposure"

cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src:$ROOT/artifacts/g7_panns/python:$ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

run_seed() {
  local seed="$1"
  "$PYTHON_BIN" -u -m dads_crnn.probe_g18_multiseed_outlier_exposure \
    --config "$CONFIG" \
    --seed "$seed"
}

case "$MODE" in
  preflight)
    exec "$PYTHON_BIN" -u -m dads_crnn.probe_g18_multiseed_outlier_exposure \
      --config "$CONFIG" \
      --preflight-only
    ;;
  run)
    if [[ ! -f artifacts/g18_model_identification/p5_preflight/report.json ]]; then
      echo "G18 P5 preflight report is missing; run '$0 preflight' first" >&2
      exit 1
    fi
    for seed in "${SEEDS[@]}"; do
      if [[ -e "$OUTPUT/seed_${seed}" ]]; then
        echo "Refusing fresh P5 run because output exists: $OUTPUT/seed_${seed}" >&2
        exit 1
      fi
    done
    for seed in "${SEEDS[@]}"; do
      run_seed "$seed"
    done
    "$PYTHON_BIN" -u -m dads_crnn.probe_g18_multiseed_outlier_exposure \
      --config "$CONFIG" \
      --aggregate
    ;;
  resume)
    if [[ ! -f artifacts/g18_model_identification/p5_preflight/report.json ]]; then
      echo "G18 P5 preflight report is missing; run '$0 preflight' first" >&2
      exit 1
    fi
    for seed in "${SEEDS[@]}"; do
      if [[ -f "$OUTPUT/seed_${seed}/summary.json" ]]; then
        echo "G18 P5 seed ${seed} already complete; skipping"
      else
        run_seed "$seed"
      fi
    done
    if [[ ! -f "$OUTPUT/summary.json" ]]; then
      "$PYTHON_BIN" -u -m dads_crnn.probe_g18_multiseed_outlier_exposure \
        --config "$CONFIG" \
        --aggregate
    else
      echo "G18 P5 aggregate summary already exists; skipping"
    fi
    ;;
  aggregate)
    exec "$PYTHON_BIN" -u -m dads_crnn.probe_g18_multiseed_outlier_exposure \
      --config "$CONFIG" \
      --aggregate
    ;;
  status)
    tail -n 100 logs/g18_p5_multiseed_oe.log
    ;;
  *)
    echo "Usage: $0 {preflight|run|resume|aggregate|status}" >&2
    exit 2
    ;;
esac
