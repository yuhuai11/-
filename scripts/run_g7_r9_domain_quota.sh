#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user1/JJZ/ABDDV-CRNN"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
MANIFEST="artifacts/g7_r8_urban_negatives/data/manifest.csv"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/artifacts/g7_panns/python:$ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

resolve_arm() {
  case "${1:-}" in
    control_00) CONFIG="configs/g7_r9_domain_quota_control.yaml" ;;
    tau_05) CONFIG="configs/g7_r9_domain_quota_tau05.yaml" ;;
    tau_10) CONFIG="configs/g7_r9_domain_quota_tau10.yaml" ;;
    *) echo "Arm must be one of: control_00, tau_05, tau_10" >&2; exit 2 ;;
  esac
}

case "${1:-status}" in
  audit)
    "$PYTHON_BIN" -u -m dads_crnn.audit_g7_r9_domain_quota
    ;;
  preflight)
    resolve_arm "${2:-}"
    "$PYTHON_BIN" -u -m dads_crnn.audit_g7_r9_domain_quota
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config "$CONFIG" --manifest "$MANIFEST" --seeds 42 --preflight-only
    ;;
  train)
    resolve_arm "${2:-}"
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config "$CONFIG" --manifest "$MANIFEST" --seeds 42
    ;;
  resume)
    resolve_arm "${2:-}"
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config "$CONFIG" --manifest "$MANIFEST" --seeds 42 --resume
    ;;
  external)
    resolve_arm "${2:-}"
    "$PYTHON_BIN" -u -m dads_crnn.evaluate_g7_r9_external \
      --arm "${2:-}" --device cuda --batch-size 256
    ;;
  status)
    pgrep -af "dads_crnn.train_panns.*g7_r9_domain_quota" || true
    nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader || true
    ;;
  *)
    echo "Usage: $0 {audit|preflight|train|resume|external|status} [control_00|tau_05|tau_10]" >&2
    exit 2
    ;;
esac
