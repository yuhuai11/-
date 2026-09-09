from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .data_firewall import file_sha256


PROTOCOL = "g7_r4b_single_variable_lowsnr_audit_v1"


def audit(baseline_path: Path, candidate_path: Path, output: Path) -> dict:
    baseline = load_config(baseline_path)
    candidate = load_config(candidate_path)
    baseline_compare = copy.deepcopy(baseline)
    candidate_compare = copy.deepcopy(candidate)
    for value in (baseline_compare, candidate_compare):
        value.pop("protocol", None)
        value.pop("output_dir", None)
    baseline_aug = baseline_compare["train"]["augmentation"]
    candidate_aug = candidate_compare["train"]["augmentation"]
    baseline_snr = baseline_aug.pop("mix_snr_uniform_db")
    candidate_levels = candidate_aug.pop("mix_snr_db")
    candidate_weights = candidate_aug.pop("mix_snr_weights")
    unchanged = baseline_compare == candidate_compare
    weights = np.asarray(candidate_weights, dtype=np.float64)
    levels = np.asarray(candidate_levels, dtype=np.float64)
    low_weight = float(weights[np.isin(levels, [-15.0, -10.0])].sum())
    mix_probability = float(candidate["train"]["augmentation"]["positive_mix_probability"])
    report = {
        "passed": bool(
            unchanged
            and baseline_snr == [10.0, 20.0]
            and candidate_levels == [-15.0, -10.0, 10.0, 15.0, 20.0]
            and np.isclose(weights.sum(), 1.0)
            and np.isclose(low_weight, 0.30)
        ),
        "protocol": PROTOCOL,
        "single_variable": "positive_background_mix_snr_distribution",
        "all_other_config_fields_unchanged": unchanged,
        "baseline_uniform_snr_db": baseline_snr,
        "candidate_snr_db": candidate_levels,
        "candidate_snr_weights": candidate_weights,
        "low_snr_weight_within_mixed_positives": low_weight,
        "expected_low_snr_fraction_of_all_positive_views": mix_probability * low_weight,
        "inputs": {
            "baseline_config": {"path": str(baseline_path), "sha256": file_sha256(baseline_path)},
            "candidate_config": {"path": str(candidate_path), "sha256": file_sha256(candidate_path)},
        },
        "locked_datasets_read": [],
        "formal_training_started": False,
    }
    if not report["passed"]:
        raise ValueError(f"G7-R4B is not a single-variable SNR ablation: {report}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit G7-R4B single-variable delta")
    parser.add_argument("--baseline", type=Path, default=Path("configs/g7_r4a_source_balanced_fc1.yaml"))
    parser.add_argument("--candidate", type=Path, default=Path("configs/g7_r4b_lowsnr_curriculum.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/g7_r4_cross_dataset/r4b_lowsnr_curriculum/preflight/config_delta_audit.json"))
    args = parser.parse_args()
    print(json.dumps(audit(args.baseline, args.candidate, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
