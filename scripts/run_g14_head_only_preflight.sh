#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/artifacts/g7_panns/python:$ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$PYTHON_BIN" -u -m dads_crnn.preflight_g14_head_only \
  --root "$ROOT" \
  --config configs/g14_panns_head_only.yaml \
  --manifest artifacts/g14_domain_generalization/segment_cache/g14_segment_manifest.csv \
  --sampler-audit artifacts/g14_domain_generalization/sampler_preflight/sampler_preflight_audit.json

"$PYTHON_BIN" -m unittest tests.test_g14_head_only
