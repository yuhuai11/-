#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
CONFIG="configs/g22_harmonic_fusion_probe.yaml"
OUTPUT="artifacts/g22_harmonic_fusion_probe/summary.json"

cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f "$OUTPUT" ]]; then
  "$PYTHON_BIN" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); print("G22 complete:", d["decision"]); print(json.dumps(d["primary_fusion_metrics"], ensure_ascii=False, indent=2))' \
    "$OUTPUT"
else
  exec "$PYTHON_BIN" -u -m dads_crnn.probe_g22_harmonic_fusion \
    --config "$CONFIG"
fi
