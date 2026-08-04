from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .data_firewall import file_sha256


PROTOCOL = "g7_r4c_three_seed_development_confirmation_v1"


METRICS: dict[str, Callable[[dict], float]] = {
    "dads_f1": lambda c: c["domains"]["dads_halfsec"]["threshold_0_5"]["f1"],
    "kielce_recall": lambda c: c["domains"]["kielce_17_uav"]["recall_at_0_5"],
    "kielce_source_macro_recall": lambda c: c["domains"]["kielce_17_uav"]["source_macro_recall_at_0_5"],
    "kielce_worst_source_recall": lambda c: c["domains"]["kielce_17_uav"]["worst_source_recall_at_0_5"],
    "tau_fpr": lambda c: c["domains"]["tau_urban_2022"]["fpr_at_0_5"],
    "cross_roc_auc": lambda c: c["low_fpr"]["roc_auc"],
    "cross_pr_auc": lambda c: c["low_fpr"]["pr_auc"],
    "cross_standardized_pauc": lambda c: c["low_fpr"]["standardized_pauc_fpr_le_0_05"],
    "cross_tpr_at_fpr_0_01": lambda c: c["low_fpr"]["operating_points"]["0.01"]["kielce_tpr"],
    "cross_tpr_at_fpr_0_05": lambda c: c["low_fpr"]["operating_points"]["0.05"]["kielce_tpr"],
    "minus15_tpr_at_0_5": lambda c: c["low_snr"]["by_snr"]["-15.0"]["positive_tpr_at_0_5"],
    "minus10_tpr_at_0_5": lambda c: c["low_snr"]["by_snr"]["-10.0"]["positive_tpr_at_0_5"],
}


def summarize(paths: dict[int, Path], output: Path) -> dict:
    reports = {seed: json.loads(path.read_text(encoding="utf-8")) for seed, path in paths.items()}
    for seed, report in reports.items():
        if report.get("external_dev_holdout_read") is not False or report.get("locked_datasets_read") != []:
            raise ValueError(f"Seed {seed} confirmation report crossed the development firewall")
    values = {
        name: {str(seed): float(extract(report["candidate"])) for seed, report in reports.items()}
        for name, extract in METRICS.items()
    }
    statistics = {}
    for name, by_seed in values.items():
        array = np.asarray(list(by_seed.values()), dtype=np.float64)
        statistics[name] = {
            "by_seed": by_seed,
            "mean": float(array.mean()),
            "sample_std": float(array.std(ddof=1)),
            "minimum": float(array.min()),
            "maximum": float(array.max()),
        }
    seed_checks = {
        str(seed): {
            "eligible": bool(report["selection"]["eligible"]),
            "checks": report["selection"]["checks"],
            "deltas": report["selection"]["deltas"],
        }
        for seed, report in reports.items()
    }
    all_seeds_eligible = all(value["eligible"] for value in seed_checks.values())
    result = {
        "passed": True,
        "protocol": PROTOCOL,
        "decision": (
            "freeze_candidate_for_locked_confirmation"
            if all_seeds_eligible
            else "terminate_r4c_not_stable_across_seeds"
        ),
        "promotion_rule": "every_seed_must_pass_every_preregistered_development_gate",
        "all_seeds_eligible": all_seeds_eligible,
        "seed_checks": seed_checks,
        "statistics": statistics,
        "inputs": {
            str(seed): {"path": str(path), "sha256": file_sha256(path)}
            for seed, path in paths.items()
        },
        "locked_datasets_read": [],
        "external_dev_holdout_read": False,
        "further_weight_search_on_same_development_set_allowed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize G7-R4C three-seed confirmation")
    parser.add_argument("--output", type=Path, default=Path("artifacts/g7_r4_cross_dataset/r4c_lowsnr_conservative/confirmation_summary.json"))
    args = parser.parse_args()
    root = Path("artifacts/g7_r4_cross_dataset/r4c_lowsnr_conservative")
    paths = {
        42: root / "development_evaluation" / "summary.json",
        43: root / "development_evaluation_seed_43" / "summary.json",
        44: root / "development_evaluation_seed_44" / "summary.json",
    }
    print(json.dumps(summarize(paths, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
