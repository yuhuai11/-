#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "fit" && "$MODE" != "evaluate" ]]; then
  echo "usage: $0 {fit|evaluate}" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
export PYTHONPATH="${PYTHONPATH:-src}"

exec "$PYTHON_BIN" -u -m dads_crnn.evaluate_g12_strict "$MODE" \
  --config configs/g12_strict_source_conformal.yaml
