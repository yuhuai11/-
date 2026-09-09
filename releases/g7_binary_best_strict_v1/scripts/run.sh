#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"

cd "$BUNDLE_ROOT"
export PYTHONPATH="$BUNDLE_ROOT/src:$BUNDLE_ROOT/artifacts/g7_panns/python:$BUNDLE_ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "${1:-verify-fast}" in
  verify-fast)
    "$PYTHON_BIN" tools/verify_bundle.py --fast
    ;;
  verify)
    "$PYTHON_BIN" tools/verify_bundle.py
    ;;
  infer)
    shift
    "$PYTHON_BIN" tools/infer_wav.py "$@"
    ;;
  preflight)
    PYTHONWARNINGS=default "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config configs/g7_strict_retrain_v1.yaml \
      --manifest artifacts/g7_leakage_fixed_v2/data/manifest.csv \
      --seeds 42 --preflight-only
    ;;
  train)
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config configs/g7_strict_retrain_v1.yaml \
      --manifest artifacts/g7_leakage_fixed_v2/data/manifest.csv \
      --seeds 42 43 44
    ;;
  resume)
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config configs/g7_strict_retrain_v1.yaml \
      --manifest artifacts/g7_leakage_fixed_v2/data/manifest.csv \
      --seeds 42 43 44 --resume
    ;;
  external)
    "$PYTHON_BIN" -u -m dads_crnn.evaluate_g7_strict_external_baseline \
      --device cuda --batch-size 256
    ;;
  aggregate-one-second)
    "$PYTHON_BIN" -u -m dads_crnn.evaluate_g7_strict_multiscale
    ;;
  test)
    "$PYTHON_BIN" -m pytest -q tests
    ;;
  *)
    echo "Usage: $0 {verify-fast|verify|infer|preflight|train|resume|external|aggregate-one-second|test}" >&2
    exit 2
    ;;
esac
