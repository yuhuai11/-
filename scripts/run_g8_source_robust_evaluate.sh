#!/usr/bin/env bash
set -euo pipefail

cd /home/user1/JJZ/ABDDV-CRNN

export PYTHONPATH="src"

/usr/local/anaconda3/envs/dads-crnn/bin/python -u -m dads_crnn.evaluate_source_robust_fpr evaluate \
  --tune-predictions artifacts/val_ood/predictions/tune/g7_panns_pt/seed_42/predictions.csv \
  --tune-manifest artifacts/val_ood/manifests/val_ood_tune_manifest.csv \
  --holdout-predictions artifacts/val_ood/predictions/holdout/g7_panns_pt/seed_42/predictions.csv \
  --holdout-manifest artifacts/val_ood/manifests/val_ood_holdout_manifest.csv \
  --calibration artifacts/g8_source_robust/frozen_thresholds/g7_panns_pt_seed42.json \
  --reference-metrics artifacts/g7_low_fpr_candidate/g7_panns_pt_seed42/metrics.json \
  --baseline-metrics artifacts/g7_low_fpr_baseline/g2/metrics.json \
  --output-dir artifacts/g8_source_robust/evaluation/g7_panns_pt_seed42 \
  --experiment g8_source_robust_g7_panns_pt_seed42 \
  --bootstrap-samples 2000 \
  --bootstrap-seed 20260720
