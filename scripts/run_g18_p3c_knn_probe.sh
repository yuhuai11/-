#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
MODE="${1:-probe}"

cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src:$ROOT/artifacts/g7_panns/python:$ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "$MODE" in
  preflight)
    exec "$PYTHON_BIN" -u -m dads_crnn.probe_g18_knn \
      --config configs/g18_knn_probe.yaml \
      --preflight-only
    ;;
  probe)
    if [[ ! -f artifacts/g18_model_identification/p3c_preflight/report.json ]]; then
      echo "G18 P3c preflight report is missing; run '$0 preflight' first" >&2
      exit 1
    fi
    if [[ -f artifacts/g18_model_identification/p3c_knn_probe/summary.json ]]; then
      echo "Refusing to overwrite completed G18 P3c probe" >&2
      exit 1
    fi
    exec "$PYTHON_BIN" -u -m dads_crnn.probe_g18_knn \
      --config configs/g18_knn_probe.yaml
    ;;
  status)
    tail -n 80 logs/g18_p3c_knn_probe.log
    ;;
  *)
    echo "Usage: $0 {preflight|probe|status}" >&2
    exit 2
    ;;
esac
