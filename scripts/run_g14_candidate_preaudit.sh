#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"

cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:src"

"$PYTHON_BIN" -u -m dads_crnn.prepare_g14_candidate_registry \
  --root "$ROOT" \
  --config configs/g14_domain_generalization_intake.yaml

"$PYTHON_BIN" -m unittest tests.test_g14_candidate_registry
