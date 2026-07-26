#!/usr/bin/env bash
set -euo pipefail

cd /home/user1/JJZ/ABDDV-CRNN

export PYTHONPATH="src"

/usr/local/anaconda3/envs/dads-crnn/bin/python -u -m dads_crnn.evaluate_source_robust_fpr fit \
  --tune-predictions artifacts/val_ood/predictions/tune/g7_panns_pt/seed_42/predictions.csv \
  --tune-manifest artifacts/val_ood/manifests/val_ood_tune_manifest.csv \
  --output artifacts/g8_source_robust/frozen_thresholds/g7_panns_pt_seed42.json \
  --experiment g8_source_robust_g7_panns_pt_seed42 \
  --target-fprs 0.01 0.05 \
  --source-miscoverage-delta 0.05 \
  --holdout-fpr-caps 0.02 0.075 \
  --minimum-holdout-tprs 0.10 0.30 \
  --minimum-tpr-retention 0.80 \
  --protocol-document docs/G8分源稳健阈值校准方案.md
