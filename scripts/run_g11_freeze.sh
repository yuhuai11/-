#!/usr/bin/env bash
set -euo pipefail
cd /home/user1/JJZ/ABDDV-CRNN
export PYTHONPATH="src:artifacts/g7_panns/python:artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch"
/usr/local/anaconda3/envs/dads-crnn/bin/python -u -m dads_crnn.evaluate_g11_final freeze \
  --config configs/g11_final_dual_mode.yaml
