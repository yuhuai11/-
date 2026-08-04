#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
MODE="${1:-status}"
CONFIG="configs/g20_closed_set_multiscale.yaml"
PREFLIGHT="artifacts/g20_closed_set_multiscale/p0_preflight/report.json"
OUTPUT="artifacts/g20_closed_set_multiscale/p1_training"

cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "$MODE" in
  preflight)
    exec "$PYTHON_BIN" -u -m dads_crnn.train_g20_closed_set \
      --config "$CONFIG" --preflight-only
    ;;
  train)
    [[ -f "$PREFLIGHT" ]] || {
      echo "G20 preflight is missing; run '$0 preflight' first" >&2
      exit 1
    }
    [[ ! -e "$OUTPUT/summary.json" ]] || {
      echo "G20 training is already complete" >&2
      exit 1
    }
    [[ ! -e "$OUTPUT/latest.pt" && ! -e "$OUTPUT/best.pt" ]] || {
      echo "Interrupted G20 artifacts exist; use '$0 resume'" >&2
      exit 1
    }
    exec "$PYTHON_BIN" -u -m dads_crnn.train_g20_closed_set --config "$CONFIG"
    ;;
  resume)
    [[ ! -e "$OUTPUT/summary.json" ]] || {
      echo "G20 training is already complete" >&2
      exit 1
    }
    [[ -f "$OUTPUT/latest.pt" ]] || {
      echo "G20 latest checkpoint is missing; use '$0 train'" >&2
      exit 1
    }
    exec "$PYTHON_BIN" -u -m dads_crnn.train_g20_closed_set \
      --config "$CONFIG" --resume
    ;;
  status)
    if [[ -f "$OUTPUT/summary.json" ]]; then
      "$PYTHON_BIN" -c \
        'import json,sys; d=json.load(open(sys.argv[1])); print("G20 training complete:", d["decision"]); print(json.dumps(d["best_metrics"], ensure_ascii=False, indent=2))' \
        "$OUTPUT/summary.json"
    elif [[ -f "$OUTPUT/latest.pt" ]]; then
      echo "G20 training was interrupted; resume is available"
      tail -n 60 logs/g20_closed_set.log 2>/dev/null || true
    elif [[ -f "$PREFLIGHT" ]]; then
      echo "G20 preflight exists; training is pending"
    else
      echo "G20 preflight has not completed"
    fi
    ;;
  *)
    echo "Usage: $0 {preflight|train|resume|status}" >&2
    exit 2
    ;;
esac
