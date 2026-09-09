from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .data_firewall import file_sha256


PROTOCOL = "g7_r4c_single_variable_lowsnr_dose_audit_v1"


def audit(r4b_path: Path, r4c_path: Path, output: Path) -> dict:
    r4b = load_config(r4b_path)
    r4c = load_config(r4c_path)
    left = copy.deepcopy(r4b)
    right = copy.deepcopy(r4c)
    for value in (left, right):
        value.pop("protocol", None)
        value.pop("output_dir", None)
    b_weights = left["train"]["augmentation"].pop("mix_snr_weights")
    c_weights = right["train"]["augmentation"].pop("mix_snr_weights")
    unchanged = left == right
    levels = np.asarray(r4c["train"]["augmentation"]["mix_snr_db"], dtype=np.float64)
    weights = np.asarray(c_weights, dtype=np.float64)
    low_weight = float(weights[np.isin(levels, [-15.0, -10.0])].sum())
    positive_mix = float(r4c["train"]["augmentation"]["positive_mix_probability"])
    report = {
        "passed": bool(
            unchanged
            and b_weights == [0.15, 0.15, 0.233333, 0.233333, 0.233334]
            and c_weights == [0.075, 0.075, 0.283333, 0.283333, 0.283334]
            and np.isclose(weights.sum(), 1.0)
            and np.isclose(low_weight, 0.15)
        ),
        "protocol": PROTOCOL,
        "single_variable": "low_snr_mixture_weight",
        "all_other_config_fields_unchanged": unchanged,
        "r4b_weights": b_weights,
        "r4c_weights": c_weights,
        "low_snr_weight_within_mixed_positives": low_weight,
        "expected_low_snr_fraction_of_all_positive_views": positive_mix * low_weight,
        "inputs": {
            "r4b_config": {"path": str(r4b_path), "sha256": file_sha256(r4b_path)},
            "r4c_config": {"path": str(r4c_path), "sha256": file_sha256(r4c_path)},
        },
        "locked_datasets_read": [],
        "formal_training_started": False,
    }
    if not report["passed"]:
        raise ValueError(f"G7-R4C is not the preregistered single-dose ablation: {report}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit G7-R4C low-SNR dose delta")
    parser.add_argument("--r4b", type=Path, default=Path("configs/g7_r4b_lowsnr_curriculum.yaml"))
    parser.add_argument("--r4c", type=Path, default=Path("configs/g7_r4c_lowsnr_conservative.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/g7_r4_cross_dataset/r4c_lowsnr_conservative/preflight/config_delta_audit.json"))
    args = parser.parse_args()
    print(json.dumps(audit(args.r4b, args.r4c, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
