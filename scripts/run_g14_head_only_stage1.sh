#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
CONFIG="configs/g14_panns_head_only.yaml"
MANIFEST="artifacts/g14_domain_generalization/segment_cache/g14_segment_manifest.csv"
PREFLIGHT="artifacts/g14_domain_generalization/model_preflight/g14_head_only_preflight.json"
RUN_ROOT="artifacts/g14_domain_generalization/runs/head_only"
SEEDS=(42 43 44)

cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src:$ROOT/artifacts/g7_panns/python:$ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

usage() {
  echo "Usage: $0 {preflight|train|resume|status}" >&2
}

require_stage1_gate() {
  local path
  for path in "$CONFIG" "$MANIFEST" "$PREFLIGHT"; do
    if [[ ! -s "$path" ]]; then
      echo "Missing or empty G14-A input: $path" >&2
      exit 1
    fi
  done
  "$PYTHON_BIN" -c '
import json
from pathlib import Path
report = json.loads(Path("'"$PREFLIGHT"'").read_text(encoding="utf-8"))
required = {
    "passed": True,
    "protocol": "g14_panns_head_only_preflight_v1",
    "ready_for_stage1_training": True,
    "checkpoint_written": False,
    "formal_training_started": False,
}
for key, expected in required.items():
    if report.get(key) != expected:
        raise SystemExit(f"G14-A preflight gate failed: {key}={report.get(key)!r}")
if report.get("g7_checkpoint_sha256") != "d357f194105ec27a61838e0f4c2e7575932e2dd861e529ef970b1fb467954cdc":
    raise SystemExit("G14-A preflight gate failed: G7 checkpoint identity changed")
'
}

is_complete() {
  local seed=$1
  local run_dir="$RUN_ROOT/seed_$seed"
  [[ -s "$run_dir/best.pt" && -s "$run_dir/last.pt" &&
     -s "$run_dir/metrics.json" && -s "$run_dir/history.csv" ]]
}

mode="${1:-}"
case "$mode" in
  preflight)
    bash scripts/run_g14_head_only_preflight.sh
    ;;
  train)
    require_stage1_gate
    for seed in "${SEEDS[@]}"; do
      run_dir="$RUN_ROOT/seed_$seed"
      if [[ -d "$run_dir" && -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Refusing to overwrite existing G14-A seed $seed: $run_dir" >&2
        echo "Inspect it first; use '$0 resume' only for a deliberate continuation." >&2
        exit 1
      fi
    done
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config "$CONFIG" \
      --manifest "$MANIFEST" \
      --seeds "${SEEDS[@]}"
    ;;
  resume)
    require_stage1_gate
    pending=()
    for seed in "${SEEDS[@]}"; do
      run_dir="$RUN_ROOT/seed_$seed"
      if is_complete "$seed"; then
        echo "G14-A seed $seed is complete; skipping."
      elif [[ -d "$run_dir" && -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" &&
              ! -s "$run_dir/last.pt" ]]; then
        echo "Cannot resume seed $seed without $run_dir/last.pt" >&2
        exit 1
      else
        pending+=("$seed")
      fi
    done
    if (( ${#pending[@]} )); then
      "$PYTHON_BIN" -u -m dads_crnn.train_panns \
        --config "$CONFIG" \
        --manifest "$MANIFEST" \
        --seeds "${pending[@]}" \
        --resume
    else
      echo "All G14-A seeds are already complete."
    fi
    ;;
  status)
    pgrep -af "dads_crnn.train_panns.*g14_panns_head_only" || true
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
    for seed in "${SEEDS[@]}"; do
      history="$RUN_ROOT/seed_$seed/history.csv"
      if [[ -s "$history" ]]; then
        echo "seed_$seed:"
        tail -n 2 "$history"
      fi
    done
    ;;
  *)
    usage
    exit 2
    ;;
esac
