#!/usr/bin/env bash
set -euo pipefail

cd /home/user1/JJZ/ABDDV-CRNN

PYTHON=/usr/local/anaconda3/envs/dads-crnn/bin/python
CONFIG=configs/g9_panns_cnn14_16k_hn_bce.yaml
MANIFEST=artifacts/g9_hard_negatives/manifests/dads_g9_hn_seed42.csv
AUDIT=artifacts/g9_hard_negatives/audit.json
RUN_ROOT=archive/historical_models/failed_candidates/artifacts_g9_panns_hn_bce/runs
TARGET_SEEDS=(43 44)

export PYTHONPATH="src:artifacts/g7_panns/python:artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

usage() {
  echo "Usage: $0 {preflight|train|resume|summarize}" >&2
}

require_inputs() {
  local path
  for path in "$CONFIG" "$MANIFEST" "$AUDIT" "$RUN_ROOT/seed_42/metrics.json"; do
    if [[ ! -s "$path" ]]; then
      echo "Missing or empty frozen G9 input: $path" >&2
      exit 1
    fi
  done
}

is_complete() {
  local seed=$1
  local run_dir="$RUN_ROOT/seed_$seed"
  [[ -s "$run_dir/best.pt" && -s "$run_dir/last.pt" && -s "$run_dir/metrics.json" \
     && -s "$run_dir/val_probabilities.npy" && -s "$run_dir/test_probabilities.npy" ]]
}

summarize() {
  "$PYTHON" -u -m dads_crnn.summarize_panns_runs \
    --run-root "$RUN_ROOT" \
    --expected-seeds 42 43 44
}

mode=${1:-}
require_inputs
case "$mode" in
  preflight)
    "$PYTHON" -u -m dads_crnn.train_panns \
      --config "$CONFIG" \
      --manifest "$MANIFEST" \
      --seeds 43 \
      --preflight-only
    ;;
  train)
    for seed in "${TARGET_SEEDS[@]}"; do
      run_dir="$RUN_ROOT/seed_$seed"
      if [[ -d "$run_dir" && -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Refusing to overwrite existing seed $seed run: $run_dir" >&2
        echo "Use '$0 resume' only after inspecting its checkpoint." >&2
        exit 1
      fi
    done
    "$PYTHON" -u -m dads_crnn.train_panns \
      --config "$CONFIG" \
      --manifest "$MANIFEST" \
      --seeds "${TARGET_SEEDS[@]}"
    summarize
    ;;
  resume)
    pending=()
    for seed in "${TARGET_SEEDS[@]}"; do
      run_dir="$RUN_ROOT/seed_$seed"
      if is_complete "$seed"; then
        echo "Seed $seed is already complete; skipping."
      elif [[ -d "$run_dir" && -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" \
              && ! -s "$run_dir/last.pt" ]]; then
        echo "Cannot resume seed $seed without $run_dir/last.pt" >&2
        exit 1
      else
        pending+=("$seed")
      fi
    done
    if (( ${#pending[@]} )); then
      "$PYTHON" -u -m dads_crnn.train_panns \
        --config "$CONFIG" \
        --manifest "$MANIFEST" \
        --seeds "${pending[@]}" \
        --resume
    fi
    summarize
    ;;
  summarize)
    summarize
    ;;
  *)
    usage
    exit 2
    ;;
esac
