#!/usr/bin/env bash
set -euo pipefail

PYTHON=/usr/local/anaconda3/envs/dads-crnn/bin/python
export PYTHONPATH=artifacts/p1_beats_probe/python:src
export CUDA_VISIBLE_DEVICES=0

"$PYTHON" -u -m dads_crnn.prepare_beats_probe

for split in train val test; do
  "$PYTHON" -u -m dads_crnn.extract_beats_embeddings \
    --manifest artifacts/p1_beats_probe/dads_selected_segments.csv \
    --split "$split" \
    --output-dir "artifacts/p1_beats_probe/embeddings/dads_${split}" \
    --batch-size 32 \
    --device auto
done

"$PYTHON" -u -m dads_crnn.extract_beats_embeddings \
  --manifest artifacts/val_ood/manifests/val_ood_tune_manifest.csv \
  --output-dir artifacts/p1_beats_probe/embeddings/val_ood_tune \
  --batch-size 32 \
  --device auto

"$PYTHON" -u -m dads_crnn.extract_beats_embeddings \
  --manifest artifacts/val_ood/manifests/val_ood_holdout_manifest.csv \
  --output-dir artifacts/p1_beats_probe/embeddings/val_ood_holdout \
  --batch-size 32 \
  --device auto

"$PYTHON" -u -m dads_crnn.train_beats_probe \
  --train-dir artifacts/p1_beats_probe/embeddings/dads_train \
  --val-dir artifacts/p1_beats_probe/embeddings/dads_val \
  --test-dir artifacts/p1_beats_probe/embeddings/dads_test \
  --output-dir artifacts/p1_beats_probe/linear

"$PYTHON" -u -m dads_crnn.evaluate_beats_probe \
  --model artifacts/p1_beats_probe/linear/linear_probe.joblib \
  --dads-metrics artifacts/p1_beats_probe/linear/metrics.json \
  --tune-dir artifacts/p1_beats_probe/embeddings/val_ood_tune \
  --holdout-dir artifacts/p1_beats_probe/embeddings/val_ood_holdout \
  --output-dir artifacts/p1_beats_probe/evaluation
