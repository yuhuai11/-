#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user1/JJZ/ABDDV-CRNN"
PYTHON="/usr/local/anaconda3/envs/dads-crnn/bin/python"
MANIFEST="artifacts/g7_leakage_fixed_v2/data/manifest.csv"
AUDIT="artifacts/g7_leakage_fixed_v2/data/audit.json"
VAL_OOD_ROOT="artifacts/g7_r2_generalization/val_ood"
LOW_FPR_ROOT="artifacts/g7_r2_generalization/low_fpr"
RECORDING_ROOT="artifacts/g7_r3_recording_aggregation"
MODE="${1:-}"
CANDIDATE="${2:-}"
SEEDS=("${@:3}")
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  SEEDS=(42)
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/artifacts/g7_panns/python:$ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

candidate_config() {
  case "$1" in
    pt_control) echo "configs/g7_r2_pt_control.yaml" ;;
    pt_mic) echo "configs/g7_r2_pt_mic.yaml" ;;
    pt_mic_bg) echo "configs/g7_r2_pt_mic_bg.yaml" ;;
    pt_mic_bg_freq) echo "configs/g7_r2_pt_mic_bg_freq.yaml" ;;
    *)
      echo "Unknown candidate: $1" >&2
      echo "Candidates: pt_control, pt_mic, pt_mic_bg, pt_mic_bg_freq" >&2
      return 2
      ;;
  esac
}

candidate_run_root() {
  case "$1" in
    pt_control) echo "artifacts/g7_r2_generalization/ablations/pt_control/runs" ;;
    pt_mic) echo "artifacts/g7_r2_generalization/ablations/pt_mic/runs" ;;
    pt_mic_bg) echo "artifacts/g7_r2_generalization/ablations/pt_mic_bg/runs" ;;
    pt_mic_bg_freq) echo "artifacts/g7_r2_generalization/ablations/pt_mic_bg_freq/runs" ;;
    *) return 2 ;;
  esac
}

run_preflight() {
  local name="$1"
  local config
  config="$(candidate_config "$name")"
  "$PYTHON" -u -m dads_crnn.train_panns \
    --config "$config" \
    --manifest "$MANIFEST" \
    --seeds 42 \
    --preflight-only
}

case "$MODE" in
  validate-data)
    "$PYTHON" -u -m dads_crnn.prepare_dads_leakage_fixed validate \
      --manifest "$MANIFEST" \
      --audit "$AUDIT"
    ;;
  paper-checksum)
    sha256sum --check docs/references/papers/reuter_2026/CHECKSUMS.sha256
    ;;
  preflight-all)
    for name in pt_control pt_mic pt_mic_bg pt_mic_bg_freq; do
      echo "=== preflight: $name ==="
      run_preflight "$name"
    done
    ;;
  preflight)
    run_preflight "$CANDIDATE"
    ;;
  train|resume)
    CONFIG="$(candidate_config "$CANDIDATE")"
    EXTRA=()
    if [[ "$MODE" == "resume" ]]; then
      EXTRA=(--resume)
    fi
    "$PYTHON" -u -m dads_crnn.train_panns \
      --config "$CONFIG" \
      --manifest "$MANIFEST" \
      --seeds "${SEEDS[@]}" \
      "${EXTRA[@]}"
    ;;
  calibrate)
    RUN_ROOT="$(candidate_run_root "$CANDIDATE")"
    for seed in "${SEEDS[@]}"; do
      "$PYTHON" -u -m dads_crnn.calibrate_ood \
        --checkpoint "$RUN_ROOT/seed_${seed}/best.pt" \
        --tune-manifest artifacts/val_ood/manifests/val_ood_tune_manifest.csv \
        --holdout-manifest artifacts/val_ood/manifests/val_ood_holdout_manifest.csv \
        --output-dir "$VAL_OOD_ROOT" \
        --experiment "$CANDIDATE" \
        --batch-size 256 \
        --num-workers 0 \
        --device cuda
    done
    ;;
  low-fpr)
    for seed in "${SEEDS[@]}"; do
      "$PYTHON" -u -m dads_crnn.evaluate_low_fpr \
        --tune-predictions "$VAL_OOD_ROOT/predictions/tune/$CANDIDATE/seed_${seed}/predictions.csv" \
        --holdout-predictions "$VAL_OOD_ROOT/predictions/holdout/$CANDIDATE/seed_${seed}/predictions.csv" \
        --output-dir "$LOW_FPR_ROOT/$CANDIDATE/seed_${seed}" \
        --experiment "$CANDIDATE" \
        --target-fprs 0.01 0.05 \
        --bootstrap-samples 2000 \
        --bootstrap-seed 20260801
    done
    ;;
  recording-aggregate)
    RUN_ROOT="$(candidate_run_root "$CANDIDATE")"
    for seed in "${SEEDS[@]}"; do
      "$PYTHON" -u -m dads_crnn.evaluate_g7_recording_aggregation \
        --checkpoint "$RUN_ROOT/seed_${seed}/best.pt" \
        --tune-manifest artifacts/val_ood/manifests/val_ood_tune_manifest.csv \
        --holdout-manifest artifacts/val_ood/manifests/val_ood_holdout_manifest.csv \
        --protocol configs/g7_r3_recording_aggregation.yaml \
        --output-dir "$RECORDING_ROOT/$CANDIDATE/seed_${seed}" \
        --experiment "$CANDIDATE" \
        --batch-size 256 \
        --num-workers 0 \
        --device cuda
    done
    ;;
  recording-cap)
    RUN_ROOT="$(candidate_run_root "$CANDIDATE")"
    for seed in "${SEEDS[@]}"; do
      "$PYTHON" -u -m dads_crnn.evaluate_g7_recording_aggregation \
        --checkpoint "$RUN_ROOT/seed_${seed}/best.pt" \
        --tune-manifest artifacts/val_ood/manifests/val_ood_tune_manifest.csv \
        --holdout-manifest artifacts/val_ood/manifests/val_ood_holdout_manifest.csv \
        --protocol configs/g7_r3b_constrained_autopool.yaml \
        --output-dir "$RECORDING_ROOT/${CANDIDATE}_cap/seed_${seed}" \
        --experiment "${CANDIDATE}_cap" \
        --batch-size 256 \
        --num-workers 0 \
        --device cuda
    done
    ;;
  recording-cap-confirm)
    RUN_ROOT="$(candidate_run_root "$CANDIDATE")"
    for seed in "${SEEDS[@]}"; do
      "$PYTHON" -u -m dads_crnn.evaluate_g7_recording_aggregation \
        --checkpoint "$RUN_ROOT/seed_${seed}/best.pt" \
        --tune-manifest artifacts/val_ood/manifests/val_ood_tune_manifest.csv \
        --holdout-manifest artifacts/val_ood/manifests/val_ood_holdout_manifest.csv \
        --protocol configs/g7_r3b_cap065_confirmation.yaml \
        --output-dir "$RECORDING_ROOT/${CANDIDATE}_cap065_confirm/seed_${seed}" \
        --experiment "${CANDIDATE}_cap065_confirm" \
        --batch-size 256 \
        --num-workers 0 \
        --device cuda
    done
    ;;
  *)
    echo "Usage:" >&2
    echo "  bash scripts/run_g7_r2_generalization.sh validate-data" >&2
    echo "  bash scripts/run_g7_r2_generalization.sh paper-checksum" >&2
    echo "  bash scripts/run_g7_r2_generalization.sh preflight-all" >&2
    echo "  bash scripts/run_g7_r2_generalization.sh {preflight|train|resume|calibrate|low-fpr|recording-aggregate|recording-cap|recording-cap-confirm} CANDIDATE [seeds...]" >&2
    exit 2
    ;;
esac
