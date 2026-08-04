#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user1/JJZ/ABDDV-CRNN"
PYTHON="/usr/local/anaconda3/envs/dads-crnn/bin/python"
CONFIG="configs/g7_r1_structured_mic.yaml"
PROTOCOL="configs/g7_r1_protocol.yaml"
MANIFEST="artifacts_full/manifests/dads_all_seed42.csv"
MODE="${1:-}"

cd "$ROOT"
export PYTHONPATH="src:artifacts/g7_panns/python:artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "$MODE" in
  freeze)
    "$PYTHON" -u -m dads_crnn.audit_g7_r1 freeze --protocol "$PROTOCOL"
    ;;
  verify)
    "$PYTHON" -u -m dads_crnn.audit_g7_r1 verify --protocol "$PROTOCOL"
    ;;
  preflight)
    "$PYTHON" -u -m dads_crnn.audit_g7_r1 verify --protocol "$PROTOCOL"
    "$PYTHON" -u -m dads_crnn.train_panns \
      --config "$CONFIG" \
      --manifest "$MANIFEST" \
      --seeds 42 \
      --preflight-only
    ;;
  train)
    "$PYTHON" -u -m dads_crnn.audit_g7_r1 verify --protocol "$PROTOCOL"
    "$PYTHON" -u -m dads_crnn.train_panns \
      --config "$CONFIG" \
      --manifest "$MANIFEST" \
      --seeds 42
    ;;
  resume)
    "$PYTHON" -u -m dads_crnn.audit_g7_r1 verify --protocol "$PROTOCOL"
    "$PYTHON" -u -m dads_crnn.train_panns \
      --config "$CONFIG" \
      --manifest "$MANIFEST" \
      --seeds 42 \
      --resume
    ;;
  guard)
    "$PYTHON" -u -m dads_crnn.audit_g7_r1 verify --protocol "$PROTOCOL"
    "$PYTHON" -u -m dads_crnn.evaluate_g7_r1_guards --protocol "$PROTOCOL"
    ;;
  idmt-freeze)
    "$PYTHON" -u -m dads_crnn.evaluate_g7_r1_idmt freeze --protocol "$PROTOCOL"
    ;;
  idmt-preflight)
    "$PYTHON" -u -m dads_crnn.evaluate_g7_r1_idmt preflight --protocol "$PROTOCOL"
    ;;
  idmt-evaluate)
    "$PYTHON" -u -m dads_crnn.evaluate_g7_r1_idmt evaluate --protocol "$PROTOCOL"
    ;;
  *)
    echo "Usage: bash scripts/run_g7_r1.sh {freeze|verify|preflight|train|resume|guard|idmt-freeze|idmt-preflight|idmt-evaluate}" >&2
    exit 2
    ;;
esac
