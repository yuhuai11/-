from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_dirs, load_config
from .data_firewall import file_sha256


PROTOCOL = "g15_p4_tune_counterfactual_analysis_v1"


def paired_bootstrap_ci(
    values: np.ndarray, *, samples: int, seed: int
) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("P4 bootstrap values must be a finite nonempty vector")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = rng.integers(0, len(values), size=len(values))
        estimates[index] = values[draw].mean()
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def _paired_metric(
    delta: np.ndarray, *, samples: int, seed: int
) -> dict[str, Any]:
    delta = np.asarray(delta, dtype=np.float64)
    return {
        "delta": float(delta.mean()),
        "paired_bootstrap_95": paired_bootstrap_ci(
            delta, samples=samples, seed=seed
        ),
    }


def analyze(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported G15 P4 analysis protocol")
    verified = {}
    for name, spec in config["inputs"].items():
        path = Path(spec["path"])
        observed = file_sha256(path)
        if observed != str(spec["sha256"]):
            raise ValueError(f"G15 P4 input SHA256 mismatch: {name}")
        verified[name] = observed

    evaluation = json.loads(
        Path(config["inputs"]["evaluation"]["path"]).read_text(encoding="utf-8")
    )
    if (
        evaluation.get("passed") is not True
        or set(evaluation.get("results", {})) != {"tune"}
        or evaluation.get("models_remain_ineligible_for_promotion") is not True
        or evaluation.get("locked_datasets_read") != []
    ):
        raise ValueError("G15 P4 evaluation is not a tune-only diagnostic")

    rows = pd.read_csv(config["inputs"]["predictions"]["path"], low_memory=False)
    expected = config["expected"]
    if len(rows) != int(expected["rows"]):
        raise ValueError("Unexpected G15 P4 prediction row count")
    snrs = [float(value) for value in expected["target_snr_db"]]
    if sorted(rows["target_snr_db"].astype(float).unique()) != sorted(snrs):
        raise ValueError("Unexpected G15 P4 SNR grid")
    if rows["base_pair_id"].nunique() != int(expected["base_pairs"]):
        raise ValueError("Unexpected G15 P4 base-pair count")
    if set(rows["split"].astype(str)) != {"tune"}:
        raise ValueError("Non-tune rows entered G15 P4 analysis")

    baseline = str(config["models"]["baseline"])
    candidate = str(config["models"]["candidate"])
    required = {
        f"{baseline}_negative_probability",
        f"{baseline}_positive_probability",
        f"{candidate}_negative_probability",
        f"{candidate}_positive_probability",
    }
    if not required.issubset(rows.columns):
        raise ValueError("G15 P4 prediction columns are incomplete")
    samples = int(config["statistics"]["bootstrap_samples"])
    seed = int(config["statistics"]["bootstrap_seed"])
    threshold = float(expected["threshold"])

    by_snr = {}
    for offset, (snr, group) in enumerate(
        rows.groupby("target_snr_db", sort=True)
    ):
        base_negative = group[f"{baseline}_negative_probability"].to_numpy()
        base_positive = group[f"{baseline}_positive_probability"].to_numpy()
        candidate_negative = group[f"{candidate}_negative_probability"].to_numpy()
        candidate_positive = group[f"{candidate}_positive_probability"].to_numpy()
        base_lift = base_positive - base_negative
        candidate_lift = candidate_positive - candidate_negative
        base_ordered = base_lift > 0
        candidate_ordered = candidate_lift > 0
        local_seed = seed + offset * 100
        by_snr[str(float(snr))] = {
            "pairs": int(len(group)),
            "mean_lift": _paired_metric(
                candidate_lift - base_lift,
                samples=samples,
                seed=local_seed,
            ),
            "ordering_accuracy": _paired_metric(
                candidate_ordered.astype(float) - base_ordered.astype(float),
                samples=samples,
                seed=local_seed + 1,
            ),
            "ordering_discordance": {
                "improved": int((~base_ordered & candidate_ordered).sum()),
                "worsened": int((base_ordered & ~candidate_ordered).sum()),
            },
            "negative_fpr": _paired_metric(
                (candidate_negative >= threshold).astype(float)
                - (base_negative >= threshold).astype(float),
                samples=samples,
                seed=local_seed + 2,
            ),
            "positive_tpr": _paired_metric(
                (candidate_positive >= threshold).astype(float)
                - (base_positive >= threshold).astype(float),
                samples=samples,
                seed=local_seed + 3,
            ),
        }

    monotonic = {}
    for model in (baseline, candidate):
        pivot = rows.pivot(
            index="base_pair_id",
            columns="target_snr_db",
            values=f"{model}_positive_probability",
        ).sort_index(axis=1)
        monotonic[model] = np.all(
            np.diff(pivot.to_numpy(), axis=1) >= -1.0e-6, axis=1
        )
    monotonic_delta = monotonic[candidate].astype(float) - monotonic[
        baseline
    ].astype(float)
    monotonic_summary = {
        "baseline": float(monotonic[baseline].mean()),
        "candidate": float(monotonic[candidate].mean()),
        **_paired_metric(monotonic_delta, samples=samples, seed=seed + 1000),
        "discordance": {
            "improved": int((~monotonic[baseline] & monotonic[candidate]).sum()),
            "worsened": int((monotonic[baseline] & ~monotonic[candidate]).sum()),
        },
    }

    all_lift_positive = all(
        item["mean_lift"]["paired_bootstrap_95"][0] > 0
        for item in by_snr.values()
    )
    all_ordering_positive = all(
        item["ordering_accuracy"]["paired_bootstrap_95"][0] > 0
        for item in by_snr.values()
    )
    background_fpr_reduced = all(
        item["negative_fpr"]["paired_bootstrap_95"][1] < 0
        for item in by_snr.values()
    )
    monotonicity_improved = monotonic_summary["paired_bootstrap_95"][0] > 0
    mechanism_supported = bool(
        all_lift_positive
        and all_ordering_positive
        and background_fpr_reduced
        and monotonicity_improved
    )
    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "scope": "tune_only_post_termination_mechanistic_diagnostic",
        "candidate_remains_ineligible_for_promotion": True,
        "by_snr": by_snr,
        "snr_monotonicity": monotonic_summary,
        "checks": {
            "all_snr_mean_lift_delta_ci_above_zero": all_lift_positive,
            "all_snr_ordering_delta_ci_above_zero": all_ordering_positive,
            "all_snr_background_fpr_delta_ci_below_zero": background_fpr_reduced,
            "monotonicity_delta_ci_above_zero": monotonicity_improved,
        },
        "mechanism_improvement_supported": mechanism_supported,
        "fixed_threshold_tradeoff_present": any(
            item["positive_tpr"]["delta"] < 0 for item in by_snr.values()
        ),
        "recommendation": (
            "preregister_multisnr_constrained_head_training"
            if mechanism_supported
            else "evaluate_limited_fc1_unfreezing"
        ),
        "verified_inputs": verified,
        "locked_datasets_read": [],
        "dev_holdout_read": False,
        "training_started": False,
    }
    output = Path(config["output"])
    ensure_dirs(output.parent)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze G15 P4 tune diagnostic")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g15_p4_analysis.yaml"),
    )
    args = parser.parse_args()
    report = analyze(load_config(args.config))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
