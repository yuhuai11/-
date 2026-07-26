#!/usr/bin/env bash
set -euo pipefail

cd /home/user1/JJZ/ABDDV-CRNN

export PYTHONPATH="src:artifacts/g7_panns/python:artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch"
export CUDA_VISIBLE_DEVICES=0

/usr/local/anaconda3/envs/dads-crnn/bin/python -u -m dads_crnn.train_panns \
  --config configs/g7_panns_cnn14_16k_scratch.yaml \
  --manifest artifacts_full/manifests/dads_all_seed42.csv \
  --seeds 42 \
  --resume
