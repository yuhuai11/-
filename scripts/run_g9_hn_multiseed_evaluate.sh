#!/usr/bin/env bash
set -euo pipefail

cd /home/user1/JJZ/ABDDV-CRNN

PYTHON=/usr/local/anaconda3/envs/dads-crnn/bin/python
BASELINE=artifacts_g7_panns_pt/runs/seed_42/best.pt
RUN_ROOT=archive/historical_models/failed_candidates/artifacts_g9_panns_hn_bce/runs
SEEDS=(43 44)

export PYTHONPATH="src:artifacts/g7_panns/python:artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mode=${1:-all}
if [[ "$mode" != "guard" && "$mode" != "ood" && "$mode" != "all" ]]; then
  echo "Usage: $0 [guard|ood|all]" >&2
  exit 2
fi

for seed in "${SEEDS[@]}"; do
  candidate="$RUN_ROOT/seed_$seed/best.pt"
  if [[ ! -s "$candidate" ]]; then
    echo "Missing completed G9 seed $seed checkpoint: $candidate" >&2
    exit 1
  fi

  if [[ "$mode" == "guard" || "$mode" == "all" ]]; then
    "$PYTHON" -u -m dads_crnn.evaluate_g9_guard \
      --baseline-checkpoint "$BASELINE" \
      --candidate-checkpoint "$candidate" \
      --manifest artifacts/g9_hard_negatives/manifests/hn_guard.csv \
      --audit artifacts/g9_hard_negatives/audit.json \
      --output-dir "artifacts/g9_hard_negatives/evaluation/seed_$seed" \
      --thresholds 0.40 0.50 0.65 \
      --batch-size 128 \
      --num-workers 0 \
      --device auto
  fi

  if [[ "$mode" == "ood" || "$mode" == "all" ]]; then
    "$PYTHON" -u -m dads_crnn.calibrate_ood \
      --checkpoint "$candidate" \
      --tune-manifest artifacts/val_ood/manifests/val_ood_tune_manifest.csv \
      --holdout-manifest artifacts/val_ood/manifests/val_ood_holdout_manifest.csv \
      --output-dir artifacts/val_ood \
      --experiment g9_panns_hn \
      --batch-size 128 \
      --num-workers 0 \
      --device auto

    "$PYTHON" -u -m dads_crnn.evaluate_low_fpr \
      --tune-predictions "artifacts/val_ood/predictions/tune/g9_panns_hn/seed_$seed/predictions.csv" \
      --holdout-predictions "artifacts/val_ood/predictions/holdout/g9_panns_hn/seed_$seed/predictions.csv" \
      --output-dir "artifacts/g9_low_fpr_candidate/g9_panns_hn_seed$seed" \
      --experiment "g9_panns_hn_seed$seed" \
      --target-fprs 0.01 0.05 \
      --bootstrap-samples 2000 \
      --bootstrap-seed 20260720
  fi
done
