#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user1/JJZ/ABDDV-CRNN"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/artifacts/g7_panns/python:$ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "${1:-prepare}" in
  prepare)
    "$PYTHON_BIN" -u -m dads_crnn.prepare_g7_r5_splits
    ;;
  audit)
    "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
p=Path("artifacts/g7_r5_train_val_test/split_audit.json")
d=json.loads(p.read_text())
assert d["passed"] is True
assert d["locked_external_test"]["status"] == "locked_unconsumed"
assert d["locked_external_test"]["audio_payload_read_during_arrangement"] is False
print(json.dumps(d, ensure_ascii=False, indent=2))
PY
    ;;
  freeze)
    "$PYTHON_BIN" -u -m dads_crnn.freeze_g7_r5_test \
      --config configs/g7_r5_locked_test_experiment.yaml \
      --device cuda --batch-size 256
    ;;
  test-once)
    "$PYTHON_BIN" -u -m dads_crnn.evaluate_g7_r5_locked_test \
      --config configs/g7_r5_locked_test_experiment.yaml \
      --device cuda --batch-size 256
    ;;
  *)
    echo "Usage: $0 {prepare|audit|freeze|test-once}" >&2
    exit 2
    ;;
esac
