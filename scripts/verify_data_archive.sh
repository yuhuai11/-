#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

sha256sum --check archive/data_versions/METADATA_CHECKSUMS.sha256

status_files=(
  date/DADS/DATA_STATUS.md
  artifacts_full/DATA_STATUS.md
  artifacts/dads_dedup_v2/DATA_STATUS.md
  artifacts/dads_segment_guarded_v3/DATA_STATUS.md
  artifacts/g7_leakage_fixed_v2/DATA_STATUS.md
  artifacts/val_ood/DATA_STATUS.md
  artifacts/external_evaluation/DATA_STATUS.md
  artifacts/g11_final_dual_mode/DATA_STATUS.md
  data/external_confirmation_v2/DATA_STATUS.md
  artifacts/g13_external_confirmation/DATA_STATUS.md
  data/g14_domain_generalization/DATA_STATUS.md
  artifacts/g14_domain_generalization/DATA_STATUS.md
  data/g7_cross_domain/DATA_STATUS.md
  artifacts/g18_model_identification/DATA_STATUS.md
)

for status_file in "${status_files[@]}"; do
  test -f "$status_file" || {
    echo "缺少数据状态文件：$status_file" >&2
    exit 1
  }
done

unexpected_final_files="$(find data/future_final_holdout -type f ! -name README.md -print)"
if [[ -n "$unexpected_final_files" ]]; then
  echo "未来最终测试集入口出现未登记文件：" >&2
  echo "$unexpected_final_files" >&2
  exit 1
fi

echo "数据封存登记校验通过；未来最终测试集入口仍为空。"
