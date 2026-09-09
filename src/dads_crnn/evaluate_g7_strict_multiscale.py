from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluate_g7_r6_external_suite import _report
from .evaluate_g7_strict_external_baseline import DEFAULT_MODELS, PROTOCOL as P0_PROTOCOL, _datasets
from .evaluate_low_fpr import threshold_at_target_fpr


PROTOCOL = "g7_strict_nonoverlap_one_second_aggregation_v1"


def aggregate_nonoverlap(
    rows: pd.DataFrame,
    scores: np.ndarray,
    *,
    windows_per_decision: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    if windows_per_decision < 1:
        raise ValueError("windows_per_decision must be positive")
    if scores.shape != (len(rows),) or not np.isfinite(scores).all():
        raise ValueError("Rows and scores are not aligned finite arrays")
    frame = rows.reset_index(drop=True).copy()
    frame["probability"] = scores
    frame["original_order"] = np.arange(len(frame), dtype=np.int64)
    output_rows: list[dict[str, Any]] = []
    output_scores: list[float] = []
    dropped = 0
    for recording_group, group in frame.groupby("recording_group", sort=False):
        if group["label"].nunique() != 1:
            raise ValueError(f"Conflicting labels in recording group: {recording_group}")
        group = group.sort_values("original_order")
        complete = len(group) // windows_per_decision
        dropped += len(group) - complete * windows_per_decision
        for window_index in range(complete):
            selected = group.iloc[
                window_index * windows_per_decision : (window_index + 1) * windows_per_decision
            ]
            first = selected.iloc[0]
            output_rows.append(
                {
                    "label": int(first["label"]),
                    "recording_group": f"{recording_group}:window_{window_index}",
                    "parent_recording_group": str(recording_group),
                    "source_group": str(first["source_group"]),
                    "dataset_origin": str(first["dataset_origin"]),
                    "subtype": str(first.get("subtype", "")),
                }
            )
            output_scores.append(float(selected["probability"].mean()))
    return (
        pd.DataFrame(output_rows),
        np.asarray(output_scores, dtype=np.float64),
        {
            "input_halfsecond_views": int(len(frame)),
            "aggregated_decisions": int(len(output_rows)),
            "dropped_incomplete_halfsecond_views": int(dropped),
        },
    )


def _add_false_alarms_per_hour(report: dict[str, Any], duration_seconds: float) -> None:
    decisions_per_hour = 3600.0 / duration_seconds
    for unit in (
        "segment_operating_points",
        "recording_mean_operating_points",
        "source_group_mean_operating_points",
    ):
        values = report.get(unit)
        if not values:
            continue
        for metrics in values.values():
            fpr = metrics.get("false_positive_rate")
            if fpr is not None:
                metrics["nominal_false_positive_decisions_per_hour"] = float(
                    fpr * decisions_per_hour
                )


def evaluate(p0_dir: Path, output_dir: Path) -> dict[str, Any]:
    p0_path = p0_dir / "metrics.json"
    p0 = json.loads(p0_path.read_text(encoding="utf-8"))
    if p0.get("protocol") != P0_PROTOCOL or not p0.get("evaluation_completed"):
        raise ValueError("P1 requires a completed frozen P0 external baseline")
    datasets = _datasets()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_reports: dict[str, Any] = {}

    for model_name in DEFAULT_MODELS:
        aggregated: dict[str, tuple[pd.DataFrame, np.ndarray, dict[str, int]]] = {}
        for dataset_name, dataset in datasets.items():
            probability_path = p0_dir / f"{model_name}_{dataset_name}_probabilities.npy"
            scores = np.load(probability_path)
            aggregated[dataset_name] = aggregate_nonoverlap(
                dataset.rows, scores, windows_per_decision=2
            )

        calibration_scores = aggregated["calibration"][1]
        thresholds = {"fixed_0_5": 0.5}
        calibration = {}
        for target in (0.01, 0.05):
            item = threshold_at_target_fpr(calibration_scores, target)
            key = f"calibrated_fpr_{target:.2f}"
            thresholds[key] = float(item["threshold"])
            calibration[key] = item

        datasets_report = {}
        for dataset_name, (rows, scores, audit) in aggregated.items():
            if dataset_name == "calibration":
                continue
            item = _report(rows, scores, thresholds)
            _add_false_alarms_per_hour(item, 1.0)
            item["aggregation_audit"] = audit
            datasets_report[dataset_name] = item
        model_reports[model_name] = {
            "checkpoint": p0["models"][model_name]["checkpoint"],
            "checkpoint_sha256": p0["models"][model_name]["checkpoint_sha256"],
            "thresholds": thresholds,
            "calibration": calibration,
            "datasets": datasets_report,
        }

    report = {
        "evaluation_completed": True,
        "protocol": PROTOCOL,
        "parent_protocol": P0_PROTOCOL,
        "benchmark_status": "consumed_reusable_development_benchmark",
        "independent_final_claim_allowed": False,
        "aggregation": {
            "model_input_seconds": 0.5,
            "decision_seconds": 1.0,
            "method": "nonoverlapping_mean_probability_over_two_consecutive_native_views",
            "thresholds_recalibrated_for_decision_duration": True,
            "two_second_evaluation_performed": False,
            "two_second_reason": (
                "The frozen calibration recordings contain only two 0.5-second views each; "
                "a valid independent 2-second calibration unit cannot be constructed."
            ),
        },
        "models": model_reports,
        "p0_metrics_path": str(p0_path),
    }
    output_path = output_dir / "metrics.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen G7 models at a 1-second decision duration")
    parser.add_argument(
        "--p0-dir",
        type=Path,
        default=Path("artifacts/g7_strict_retrain_v1/external_baseline"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/g7_strict_retrain_v1/multiscale_aggregation"),
    )
    args = parser.parse_args()
    report = evaluate(args.p0_dir, args.output_dir)
    print(
        json.dumps(
            {
                "evaluation_completed": report["evaluation_completed"],
                "protocol": report["protocol"],
                "output": str(args.output_dir / "metrics.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
