#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
MODE="${1:-status}"
CONFIG="configs/g19_supcon_attention.yaml"
PREFLIGHT_REPORT="artifacts/g19_supcon_attention/p0_preflight/report.json"
REPRESENTATION_CHECKPOINT="artifacts/g19_supcon_attention/p1_representation/best.pt"
LATEST_CHECKPOINT="artifacts/g19_supcon_attention/p1_representation/latest.pt"
TRAIN_HISTORY="artifacts/g19_supcon_attention/p1_representation/history.csv"
TRAIN_SUMMARY="artifacts/g19_supcon_attention/p1_representation/summary.json"
CALIBRATION_SUMMARY="artifacts/g19_supcon_attention/p2_class_conditional_open_set/summary.json"
CANDIDATE_BOUNDARY="artifacts/g19_supcon_attention/p2_class_conditional_open_set/candidate_class_conditional_boundary_v1.npz"
CALIBRATION_PREDICTIONS="artifacts/g19_supcon_attention/p2_class_conditional_open_set/recording_predictions.csv"
SCORE_PROBE_SUMMARY="artifacts/g19_supcon_attention/p3_known_only_score_baselines/summary.json"

cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src:$ROOT/artifacts/g7_panns/python:$ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "$MODE" in
  preflight)
    exec "$PYTHON_BIN" -u -m dads_crnn.train_g19_representation \
      --config "$CONFIG" \
      --preflight-only
    ;;
  train)
    if [[ ! -f "$PREFLIGHT_REPORT" ]]; then
      echo "G19 preflight report is missing; run '$0 preflight' first" >&2
      exit 1
    fi
    if [[ -e "$TRAIN_SUMMARY" ]]; then
      echo "Refusing to overwrite completed G19 representation training" >&2
      exit 1
    fi
    if [[ -e "$LATEST_CHECKPOINT" || -e "$REPRESENTATION_CHECKPOINT" || -e "$TRAIN_HISTORY" ]]; then
      echo "Interrupted G19 training artifacts exist; use '$0 resume'" >&2
      exit 1
    fi
    exec "$PYTHON_BIN" -u -m dads_crnn.train_g19_representation \
      --config "$CONFIG"
    ;;
  resume)
    if [[ -e "$TRAIN_SUMMARY" ]]; then
      echo "G19 representation training is already complete" >&2
      exit 1
    fi
    if [[ ! -f "$LATEST_CHECKPOINT" ]]; then
      echo "G19 latest checkpoint is missing; use '$0 train'" >&2
      exit 1
    fi
    exec "$PYTHON_BIN" -u -m dads_crnn.train_g19_representation \
      --config "$CONFIG" \
      --resume
    ;;
  calibrate)
    if [[ ! -f "$REPRESENTATION_CHECKPOINT" ]]; then
      echo "G19 representation checkpoint is missing; run '$0 train' first" >&2
      exit 1
    fi
    if [[ -e "$CALIBRATION_SUMMARY" ]]; then
      echo "Refusing to overwrite completed G19 open-set calibration" >&2
      exit 1
    fi
    if [[ -e "$CANDIDATE_BOUNDARY" || -e "$CALIBRATION_PREDICTIONS" ]]; then
      echo "Partial G19 calibration artifacts exist; inspect them before retrying" >&2
      exit 1
    fi
    exec "$PYTHON_BIN" -u -m dads_crnn.probe_g19_class_conditional \
      --config "$CONFIG"
    ;;
  score-baselines)
    if [[ ! -f "$REPRESENTATION_CHECKPOINT" ]]; then
      echo "G19 representation checkpoint is missing; run '$0 train' first" >&2
      exit 1
    fi
    if [[ -e "$SCORE_PROBE_SUMMARY" ]]; then
      echo "Refusing to overwrite completed G19 score probe" >&2
      exit 1
    fi
    exec "$PYTHON_BIN" -u -m dads_crnn.probe_g19_score_baselines \
      --config "$CONFIG" \
      --root "$ROOT"
    ;;
  status)
    if [[ -f "$CALIBRATION_SUMMARY" ]]; then
      DECISION="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["decision"])' "$CALIBRATION_SUMMARY")"
      echo "G19 calibration execution is complete: $DECISION"
    elif [[ -e "$CANDIDATE_BOUNDARY" || -e "$CALIBRATION_PREDICTIONS" ]]; then
      echo "G19 calibration was interrupted; partial artifacts require inspection"
    elif [[ -f "$TRAIN_SUMMARY" ]]; then
      DECISION="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["decision"])' "$TRAIN_SUMMARY")"
      echo "G19 representation training is complete: $DECISION"
    elif [[ -f "$LATEST_CHECKPOINT" ]]; then
      echo "G19 representation training was interrupted; resume is available"
    elif [[ -f "$PREFLIGHT_REPORT" ]]; then
      if "$PYTHON_BIN" -c 'import sys; from pathlib import Path; from dads_crnn.config import load_config; from dads_crnn.train_g19_representation import _require_preflight,_verify_inputs; root=Path(sys.argv[1]).resolve(); config_path=(root / sys.argv[2]).resolve(); config=load_config(config_path); paths,observed,_=_verify_inputs(config,root); _require_preflight(config,config_path,root,observed,paths)' "$ROOT" "$CONFIG" >/dev/null 2>&1; then
        echo "G19 preflight is current; representation training is pending"
      else
        echo "G19 preflight report is stale; rerun '$0 preflight'"
      fi
    else
      echo "G19 preflight has not completed"
    fi
    if [[ -f logs/g19_supcon_attention.log ]]; then
      tail -n 80 logs/g19_supcon_attention.log
    fi
    ;;
  *)
    echo "Usage: $0 {preflight|train|resume|calibrate|score-baselines|status}" >&2
    exit 2
    ;;
esac
