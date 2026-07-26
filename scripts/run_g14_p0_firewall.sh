#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"

cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:src"

"$PYTHON_BIN" -u -m dads_crnn.prepare_g14_firewall \
  --root "$ROOT" \
  --output-dir artifacts/g14_domain_generalization/p0_firewall

"$PYTHON_BIN" -m unittest \
  tests.test_data_firewall \
  tests.test_beats_probe \
  tests.test_g7_panns \
  tests.test_g8_source_robust_calibration \
  tests.test_g9_hard_negatives
