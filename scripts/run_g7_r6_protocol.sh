#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user1/JJZ/ABDDV-CRNN"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

case "${1:-prepare}" in
  dronenoise-cache)
    "$PYTHON_BIN" -u -m dads_crnn.prepare_g7_r6_dronenoise_cache
    ;;
  dronenoise-split)
    "$PYTHON_BIN" -u -m dads_crnn.prepare_g7_r6_dronenoise_splits
    ;;
  prepare)
    "$PYTHON_BIN" -u -m dads_crnn.prepare_g7_r6_protocol
    ;;
  audit)
    "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
p = Path("artifacts/g7_r6_reusable_multicorpus/protocol_audit.json")
d = json.loads(p.read_text(encoding="utf-8"))
assert d["passed"] is True
assert d["reusable_benchmark"]["repeat_evaluation_allowed"] is True
assert d["reusable_benchmark"]["independent_final_claim_allowed"] is False
assert all(v == 0 for checks in d["overlap_checks"].values() for v in checks.values())
print(json.dumps(d, ensure_ascii=False, indent=2))
PY
    ;;
  *)
    echo "Usage: $0 {dronenoise-split|dronenoise-cache|prepare|audit}" >&2
    exit 2
    ;;
esac
