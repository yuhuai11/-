#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user1/JJZ/ABDDV-CRNN"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
CONFIG="configs/g7_r4a_source_balanced_fc1.yaml"
R4B_CONFIG="configs/g7_r4b_lowsnr_curriculum.yaml"
R4C_CONFIG="configs/g7_r4c_lowsnr_conservative.yaml"
MANIFEST="artifacts/g7_r4_cross_dataset/p0_manifest/manifest.csv"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/artifacts/g7_panns/python:$ROOT/artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mode="${1:-status}"
shift || true
case "$mode" in
  prepare)
    "$PYTHON_BIN" -u -m dads_crnn.prepare_g7_r4
    ;;
  preflight)
    test -s "$MANIFEST"
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config "$CONFIG" --manifest "$MANIFEST" --seeds 42 --preflight-only
    ;;
  train)
    test -s "$MANIFEST"
    if [[ $# -eq 0 ]]; then
      seeds=(42)
    else
      seeds=("$@")
    fi
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config "$CONFIG" --manifest "$MANIFEST" --seeds "${seeds[@]}"
    ;;
  resume)
    test -s "$MANIFEST"
    if [[ $# -eq 0 ]]; then
      seeds=(42)
    else
      seeds=("$@")
    fi
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config "$CONFIG" --manifest "$MANIFEST" --seeds "${seeds[@]}" --resume
    ;;
  status)
    pgrep -af "dads_crnn.train_panns.*g7_r4a_source_balanced_fc1.yaml" || true
    nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader || true
    tail -n 30 logs/g7_r4a_seed42.log 2>/dev/null || true
    ;;
  evaluate-development)
    "$PYTHON_BIN" -u -m dads_crnn.evaluate_g7_r4_development \
      --baseline-checkpoint artifacts/g7_r2_generalization/ablations/pt_mic_bg_freq/runs/seed_42/best.pt \
      --candidate-checkpoint artifacts/g7_r4_cross_dataset/r4a_source_balanced_fc1/runs/seed_42/best.pt \
      --manifest "$MANIFEST" \
      --pair-manifest artifacts/g14_domain_generalization/counterfactual_pairs/tune_pairs.csv \
      --output-dir artifacts/g7_r4_cross_dataset/r4a_source_balanced_fc1/development_evaluation \
      --device cuda \
      --batch-size 256
    ;;
  evaluate-r4b)
    "$PYTHON_BIN" -u -m dads_crnn.evaluate_g7_r4_development \
      --baseline-checkpoint artifacts/g7_r2_generalization/ablations/pt_mic_bg_freq/runs/seed_42/best.pt \
      --candidate-checkpoint artifacts/g7_r4_cross_dataset/r4b_lowsnr_curriculum/runs/seed_42/best.pt \
      --manifest "$MANIFEST" \
      --pair-manifest artifacts/g14_domain_generalization/counterfactual_pairs/tune_pairs.csv \
      --output-dir artifacts/g7_r4_cross_dataset/r4b_lowsnr_curriculum/development_evaluation \
      --device cuda \
      --batch-size 256
    ;;
  audit-r4b)
    "$PYTHON_BIN" -u -m dads_crnn.audit_g7_r4b
    ;;
  preflight-r4b)
    test -s "$MANIFEST"
    "$PYTHON_BIN" -u -m dads_crnn.audit_g7_r4b
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config "$R4B_CONFIG" --manifest "$MANIFEST" --seeds 42 --preflight-only
    ;;
  train-r4b)
    test -s "$MANIFEST"
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config "$R4B_CONFIG" --manifest "$MANIFEST" --seeds 42
    ;;
  resume-r4b)
    test -s "$MANIFEST"
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config "$R4B_CONFIG" --manifest "$MANIFEST" --seeds 42 --resume
    ;;
  audit-r4c)
    "$PYTHON_BIN" -u -m dads_crnn.audit_g7_r4c
    ;;
  preflight-r4c)
    test -s "$MANIFEST"
    "$PYTHON_BIN" -u -m dads_crnn.audit_g7_r4c
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config "$R4C_CONFIG" --manifest "$MANIFEST" --seeds 42 --preflight-only
    ;;
  train-r4c)
    test -s "$MANIFEST"
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config "$R4C_CONFIG" --manifest "$MANIFEST" --seeds 42
    ;;
  confirm-r4c)
    test -s "$MANIFEST"
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config "$R4C_CONFIG" --manifest "$MANIFEST" --seeds 43 44
    ;;
  resume-r4c)
    test -s "$MANIFEST"
    "$PYTHON_BIN" -u -m dads_crnn.train_panns \
      --config "$R4C_CONFIG" --manifest "$MANIFEST" --seeds 42 --resume
    ;;
  evaluate-r4c)
    "$PYTHON_BIN" -u -m dads_crnn.evaluate_g7_r4_development \
      --baseline-checkpoint artifacts/g7_r2_generalization/ablations/pt_mic_bg_freq/runs/seed_42/best.pt \
      --candidate-checkpoint artifacts/g7_r4_cross_dataset/r4c_lowsnr_conservative/runs/seed_42/best.pt \
      --manifest "$MANIFEST" \
      --pair-manifest artifacts/g14_domain_generalization/counterfactual_pairs/tune_pairs.csv \
      --output-dir artifacts/g7_r4_cross_dataset/r4c_lowsnr_conservative/development_evaluation \
      --device cuda \
      --batch-size 256
    ;;
  evaluate-r4c-confirm)
    for seed in 43 44; do
      "$PYTHON_BIN" -u -m dads_crnn.evaluate_g7_r4_development \
        --baseline-checkpoint artifacts/g7_r2_generalization/ablations/pt_mic_bg_freq/runs/seed_42/best.pt \
        --candidate-checkpoint "artifacts/g7_r4_cross_dataset/r4c_lowsnr_conservative/runs/seed_${seed}/best.pt" \
        --manifest "$MANIFEST" \
        --pair-manifest artifacts/g14_domain_generalization/counterfactual_pairs/tune_pairs.csv \
        --output-dir "artifacts/g7_r4_cross_dataset/r4c_lowsnr_conservative/development_evaluation_seed_${seed}" \
        --device cuda \
        --batch-size 256
    done
    ;;
  summarize-r4c)
    "$PYTHON_BIN" -u -m dads_crnn.summarize_g7_r4c
    ;;
  *)
    echo "Usage: $0 {prepare|preflight|train|resume|evaluate-development|evaluate-r4b|evaluate-r4c|evaluate-r4c-confirm|summarize-r4c|audit-r4b|preflight-r4b|train-r4b|resume-r4b|audit-r4c|preflight-r4c|train-r4c|resume-r4c|confirm-r4c|status} [seeds...]" >&2
    exit 2
    ;;
esac
