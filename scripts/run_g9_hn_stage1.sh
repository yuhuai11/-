#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user1/JJZ/ABDDV-CRNN
PYTHON=/usr/local/anaconda3/envs/dads-crnn/bin/python
DATA_CONFIG=configs/g9_hard_negative_data.yaml
TRAIN_CONFIG=configs/g9_panns_cnn14_16k_hn_bce.yaml
DADS_MANIFEST=artifacts_full/manifests/dads_all_seed42.csv
G9_ROOT=artifacts/g9_hard_negatives
G9_MANIFEST="$G9_ROOT/manifests/dads_g9_hn_seed42.csv"
G9_AUDIT="$G9_ROOT/audit.json"
RUN_DIR=archive/historical_models/failed_candidates/artifacts_g9_panns_hn_bce/runs/seed_42

cd "$ROOT"
mkdir -p logs

export PYTHONPATH="src:artifacts/g7_panns/python:artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

usage() {
  echo "Usage: $0 {prepare|preflight|train}" >&2
  echo "  prepare   audit and build the source-disjoint hard-negative manifests" >&2
  echo "  preflight verify hashes and exercise one real G9 GPU batch" >&2
  echo "  train     start a fresh seed-42 G9-HN-BCE run (never resumes implicitly)" >&2
}

require_prepared_inputs() {
  local path
  for path in \
    "$G9_AUDIT" \
    "$G9_ROOT/manifests/hn_train.csv" \
    "$G9_ROOT/manifests/hn_guard.csv" \
    "$G9_MANIFEST"; do
    if [[ ! -s "$path" ]]; then
      echo "Missing or empty prepared G9 artifact: $path" >&2
      echo "Run '$0 prepare' and inspect its audit before continuing." >&2
      exit 1
    fi
  done
}

mode="${1:-}"
case "$mode" in
  prepare)
    "$PYTHON" -u -m dads_crnn.prepare_g9_hard_negatives \
      --config "$DATA_CONFIG" \
      --dads-manifest "$DADS_MANIFEST"
    require_prepared_inputs
    echo "G9 hard-negative preparation completed. Review $G9_AUDIT before preflight."
    ;;
  preflight)
    require_prepared_inputs
    "$PYTHON" -u -m dads_crnn.train_panns \
      --config "$TRAIN_CONFIG" \
      --manifest "$G9_MANIFEST" \
      --seeds 42 \
      --preflight-only
    ;;
  train)
    require_prepared_inputs
    if [[ -d "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Refusing to overwrite or implicitly resume the existing G9 run: $RUN_DIR" >&2
      echo "A deliberate resume must use a G9 checkpoint whose manifest/audit identity matches." >&2
      exit 1
    fi
    "$PYTHON" -u -m dads_crnn.train_panns \
      --config "$TRAIN_CONFIG" \
      --manifest "$G9_MANIFEST" \
      --seeds 42
    ;;
  *)
    usage
    exit 2
    ;;
esac
