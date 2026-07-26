#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"

cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:src"

"$PYTHON_BIN" -u -m dads_crnn.prepare_dads_dedup_v2 \
  --root "$ROOT" \
  --manifest artifacts_full/manifests/dads_all_seed42.csv \
  --output-dir artifacts/dads_dedup_v2

"$PYTHON_BIN" -m unittest tests.test_dads_dedup_v2
