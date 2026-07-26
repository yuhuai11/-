#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
MODE="${1:-train}"
SEEDS=(43 44)

cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src:$ROOT/artifacts/g7_panns/python:$ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

validate_preflight() {
  "$PYTHON_BIN" -m dads_crnn.preflight_g18_multiseed --validate-existing
}

run_fresh() {
  local seed="$1"
  "$PYTHON_BIN" -u -m dads_crnn.train_g18_model_id \
    --config "configs/g18_model_id_seed${seed}.yaml"
}

resume_or_start() {
  local seed="$1"
  local directory="artifacts/g18_model_identification/p4_multiseed/seed_${seed}"
  if [[ -f "$directory/summary.json" ]]; then
    echo "G18 seed ${seed} already complete; skipping"
  elif [[ -f "$directory/latest.pt" ]]; then
    "$PYTHON_BIN" -u -m dads_crnn.train_g18_model_id \
      --config "configs/g18_model_id_seed${seed}.yaml" \
      --resume
  else
    run_fresh "$seed"
  fi
}

case "$MODE" in
  preflight)
    exec "$PYTHON_BIN" -u -m dads_crnn.preflight_g18_multiseed
    ;;
  train)
    validate_preflight
    for seed in "${SEEDS[@]}"; do
      directory="artifacts/g18_model_identification/p4_multiseed/seed_${seed}"
      if [[ -e "$directory" ]]; then
        echo "Refusing fresh training because output exists: $directory" >&2
        exit 1
      fi
    done
    for seed in "${SEEDS[@]}"; do
      run_fresh "$seed"
    done
    ;;
  resume)
    validate_preflight
    for seed in "${SEEDS[@]}"; do
      resume_or_start "$seed"
    done
    ;;
  status)
    tail -n 100 logs/g18_p4_multiseed.log
    ;;
  *)
    echo "Usage: $0 {preflight|train|resume|status}" >&2
    exit 2
    ;;
esac
