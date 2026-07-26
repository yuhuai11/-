#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
CONFIG="configs/g16_seed42_multisnr.yaml"
OUTPUT="artifacts/g16_multisnr_constrained/p2_seed42"
LOG="logs/g16_p2_seed42.log"

cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src:$ROOT/artifacts/g7_panns/python:$ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mode="${1:-preflight}"
case "$mode" in
  preflight)
    "$PYTHON_BIN" -u -m dads_crnn.train_g16_multisnr \
      --config "$CONFIG" --mode preflight
    ;;
  run)
    if [[ ! -s "$OUTPUT/preflight/report.json" ]]; then
      echo "G16 P2 selection preflight is missing" >&2
      exit 1
    fi
    "$PYTHON_BIN" -c \
      'import hashlib,json,sys; r=json.load(open(sys.argv[1])); h=hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest(); assert r.get("passed") is True and r.get("protocol") == "g16_p2_seed42_multisnr_constrained_head_v1" and r.get("ready_for_formal_seed42") is True and r.get("formal_training_started") is False and r.get("checkpoint_written") is False and r.get("inputs",{}).get("config") == h' \
      "$OUTPUT/preflight/report.json" "$CONFIG"
    if [[ -s "$OUTPUT/seed_42/summary.json" ]]; then
      echo "Refusing to overwrite completed G16 seed42 run" >&2
      exit 1
    fi
    "$PYTHON_BIN" -u -m dads_crnn.train_g16_multisnr \
      --config "$CONFIG" --mode run --resume
    ;;
  status)
    pgrep -af "dads_crnn.train_g16_multisnr.*--mode run" || true
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
    tail -n 30 "$LOG" 2>/dev/null || true
    ;;
  *)
    echo "Usage: $0 {preflight|run|status}" >&2
    exit 2
    ;;
esac
