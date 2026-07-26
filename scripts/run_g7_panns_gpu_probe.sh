#!/usr/bin/env bash
set -euo pipefail

cd /home/user1/JJZ/ABDDV-CRNN

PYTHONPATH="artifacts/g7_panns/python:artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch" \
  /usr/local/anaconda3/envs/dads-crnn/bin/python -u scripts/probe_g7_panns_gpu.py \
  --batch-sizes 16 32 64 128 256 \
  --output artifacts/g7_panns/gpu_memory_audit.json
