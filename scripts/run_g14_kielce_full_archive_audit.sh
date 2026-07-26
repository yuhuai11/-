#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"

cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:src"

"$PYTHON_BIN" -u -m dads_crnn.audit_g14_kielce_full_archive \
  --root "$ROOT" \
  --manifest configs/g14_kielce_17_uav_md5.txt \
  --output artifacts/g14_domain_generalization/intake/kielce_full_archive_audit.json
