#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
CONFIG="configs/g22_harmonic_fusion_multiseed.yaml"
OUTPUT="artifacts/g22_harmonic_fusion_multiseed/summary.json"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f "$OUTPUT" ]]; then
  "$PYTHON_BIN" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); print("G22 multiseed complete:", d["decision"]); print(json.dumps(d["aggregate_metrics"], ensure_ascii=False, indent=2))' \
    "$OUTPUT"
else
  exec "$PYTHON_BIN" -u -m dads_crnn.evaluate_g22_harmonic_multiseed \
    --config "$CONFIG"
fi
