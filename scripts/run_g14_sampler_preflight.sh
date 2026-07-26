#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"

cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:src"

"$PYTHON_BIN" -u -m dads_crnn.audit_g14_sampler_preflight \
  --root "$ROOT" \
  --config configs/g14_sampler_preflight.yaml

"$PYTHON_BIN" -m unittest tests.test_g14_balanced_batch_sampler
