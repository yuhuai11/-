#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user1/JJZ/ABDDV-CRNN"
PYTHON="/usr/local/anaconda3/envs/dads-crnn/bin/python"
CONFIG="configs/g7_leakage_fixed_halfsec.yaml"
MANIFEST="artifacts/g7_leakage_fixed_v2/data/manifest.csv"
AUDIT="artifacts/g7_leakage_fixed_v2/data/audit.json"
MODE="${1:-}"
SEEDS=("${@:2}")
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  if [[ "$MODE" == "summarize" ]]; then
    SEEDS=(42 43 44)
  else
    SEEDS=(42)
  fi
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/artifacts/g7_panns/python:$ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "$MODE" in
  prepare)
    "$PYTHON" -u -m dads_crnn.prepare_dads_leakage_fixed prepare \
      --root "$ROOT" \
      --source-registry artifacts/dads_dedup_v2/source_registry.csv \
      --dedup-audit artifacts/dads_dedup_v2/audit.json \
      --output-dir artifacts/g7_leakage_fixed_v2/data \
      --seed 42
    ;;
  validate)
    "$PYTHON" -u -m dads_crnn.prepare_dads_leakage_fixed validate \
      --manifest "$MANIFEST" \
      --audit "$AUDIT"
    ;;
  validate-full)
    "$PYTHON" -u -m dads_crnn.prepare_dads_leakage_fixed validate \
      --manifest "$MANIFEST" \
      --audit "$AUDIT" \
      --verify-cache-file
    ;;
  reseal)
    "$PYTHON" -u -m dads_crnn.prepare_dads_leakage_fixed reseal \
      --manifest "$MANIFEST" \
      --audit "$AUDIT"
    ;;
  preflight)
    "$PYTHON" -u -m dads_crnn.train_panns \
      --config "$CONFIG" \
      --manifest "$MANIFEST" \
      --seeds "${SEEDS[@]}" \
      --preflight-only
    ;;
  train)
    "$PYTHON" -u -m dads_crnn.train_panns \
      --config "$CONFIG" \
      --manifest "$MANIFEST" \
      --seeds "${SEEDS[@]}"
    ;;
  resume)
    "$PYTHON" -u -m dads_crnn.train_panns \
      --config "$CONFIG" \
      --manifest "$MANIFEST" \
      --seeds "${SEEDS[@]}" \
      --resume
    ;;
  summarize)
    "$PYTHON" -u -m dads_crnn.summarize_g7_leakage_fixed_multiseed \
      --run-root artifacts/g7_leakage_fixed_v2/runs \
      --expected-seeds "${SEEDS[@]}"
    ;;
  *)
    echo "Usage: bash scripts/run_g7_leakage_fixed.sh {prepare|validate|validate-full|reseal|preflight|train|resume|summarize} [seeds...]" >&2
    exit 2
    ;;
esac
