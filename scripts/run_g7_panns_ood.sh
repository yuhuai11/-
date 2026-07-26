#!/usr/bin/env bash
set -euo pipefail

cd /home/user1/JJZ/ABDDV-CRNN

export PYTHONPATH="src:artifacts/g7_panns/python:artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch"
export CUDA_VISIBLE_DEVICES=0

/usr/local/anaconda3/envs/dads-crnn/bin/python -u -m dads_crnn.calibrate_ood \
  --checkpoint artifacts_g7_panns_pt/runs/seed_42/best.pt \
  --tune-manifest artifacts/val_ood/manifests/val_ood_tune_manifest.csv \
  --holdout-manifest artifacts/val_ood/manifests/val_ood_holdout_manifest.csv \
  --output-dir artifacts/val_ood \
  --experiment g7_panns_pt \
  --batch-size 128 \
  --num-workers 0 \
  --device auto

/usr/local/anaconda3/envs/dads-crnn/bin/python -u -m dads_crnn.evaluate_low_fpr \
  --tune-predictions artifacts/val_ood/predictions/tune/g7_panns_pt/seed_42/predictions.csv \
  --holdout-predictions artifacts/val_ood/predictions/holdout/g7_panns_pt/seed_42/predictions.csv \
  --output-dir artifacts/g7_low_fpr_candidate/g7_panns_pt_seed42 \
  --experiment g7_panns_pt_seed42 \
  --target-fprs 0.01 0.05 \
  --bootstrap-samples 2000 \
  --bootstrap-seed 20260718
