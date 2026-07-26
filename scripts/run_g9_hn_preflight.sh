#!/usr/bin/env bash
set -euo pipefail

cd /home/user1/JJZ/ABDDV-CRNN

export PYTHONPATH="src:artifacts/g7_panns/python:artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch"
export CUDA_VISIBLE_DEVICES=0

/usr/local/anaconda3/envs/dads-crnn/bin/python -u -m dads_crnn.train_panns \
  --config configs/g9_panns_cnn14_16k_hn_bce.yaml \
  --manifest artifacts/g9_hard_negatives/manifests/dads_g9_hn_seed42.csv \
  --seeds 42 \
  --preflight-only
