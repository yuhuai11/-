#!/usr/bin/env bash
set -euo pipefail
cd /home/user1/JJZ/ABDDV-CRNN
export PYTHONPATH=src
/usr/local/anaconda3/envs/dads-crnn/bin/python -u -m dads_crnn.evaluate_g10_ensemble fit \
  --config configs/g10_ensemble_source_robust.yaml
