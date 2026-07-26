#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -u -m dads_crnn.audit_g14_nasa_license \
  --config configs/g14_domain_generalization_intake.yaml \
  --evidence configs/g14_nasa_license_evidence.yaml \
  --root "$ROOT"

"$PYTHON_BIN" -m unittest tests.test_g14_nasa_license
