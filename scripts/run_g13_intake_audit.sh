#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
export PYTHONPATH="${PYTHONPATH:-src}"

exec "$PYTHON_BIN" -u -m dads_crnn.prepare_g13_confirmation \
  --config configs/g13_external_confirmation_intake.yaml
