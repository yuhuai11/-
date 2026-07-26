#!/usr/bin/env bash
set -euo pipefail

cd /home/user1/JJZ/ABDDV-CRNN

export PYTHONPATH="src"

/usr/local/anaconda3/envs/dads-crnn/bin/python -u -m dads_crnn.prepare_g9_hard_negatives \
  --config configs/g9_hard_negative_data.yaml \
  --dads-manifest artifacts_full/manifests/dads_all_seed42.csv
