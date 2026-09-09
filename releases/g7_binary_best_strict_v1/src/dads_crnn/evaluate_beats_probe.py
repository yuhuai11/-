from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .calibrate_ood import (
    _prediction_frame,
    fit_temperature,
    probabilities_from_logits,
    probability_metrics,
    search_threshold,
)
from .config import ensure_dirs
from .evaluate_external import subgroup_metrics
from .prepare_beats_probe import reject_locked_path, sha256


IDENTITY_COLUMNS = (
    "path",
    "sha256",
    "label",
    "source_group",
    "uav_source",
    "background_source",
    "condition",
    "ood_split",
)


def _load_bundle(directory: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    embeddings = np.load(directory / "embeddings.npy", mmap_mode="r")
    labels = np.load(directory / "labels.npy").astype(np.int64)
    rows = pd.read_csv(directory / "metadata.csv")
    if (
        embeddings.shape != (len(rows), 768)
        or labels.shape != (len(rows),)
        or not np.array_equal(labels, rows["label"].to_numpy(dtype=np.int64))
    ):
        raise ValueError(f"Invalid external embedding bundle: {directory}")
    return embeddings, labels, rows


def source_macro_metric(
    rows: pd.DataFrame,
    predictions: np.ndarray,
    *,
    label: int,
    group_field: str,
) -> dict[str, Any]:
    labels = rows["label"].to_numpy(dtype=np.int64)
    selected = labels == label
    frame = rows.loc[selected, [group_field]].copy()
    frame["correct"] = predictions[selected] == label
    frame = frame.loc[frame[group_field].fillna("").astype(str) != ""]
    if frame.empty:
        raise ValueError(f"No groups for {group_field} label={label}")
    values = frame.groupby(group_field, sort=True)["correct"].mean()
    return {
        "group_field": group_field,
        "label": label,
        "groups": int(len(values)),
        "macro": float(values.mean()),
        "minimum": float(values.min()),
    }


def _identity(rows: pd.DataFrame, baseline: pd.DataFrame, split: str) -> None:
    if not all(column in rows.columns and column in baseline.columns for column in IDENTITY_COLUMNS):
        raise ValueError(f"Missing identity columns for {split}")
    left = rows[list(IDENTITY_COLUMNS)].fillna("").astype(str).reset_index(drop=True)
    right = baseline[list(IDENTITY_COLUMNS)].fillna("").astype(str).reset_index(drop=True)
    if not left.equals(right):
        raise ValueError(f"G2/P1 sample identity mismatch for {split}")


def evaluate_probe(
    model_path: Path,
    dads_metrics_path: Path,
    tune_dir: Path,
    holdout_dir: Path,
    baseline_calibration_path: Path,
    baseline_tune_predictions: Path,
    baseline_holdout_predictions: Path,
    output_dir: Path,
) -> dict[str, Any]:
    for path in (tune_dir, holdout_dir, baseline_tune_predictions, baseline_holdout_predictions):
        reject_locked_path(Path(path))
    model = joblib.load(model_path)
    tune_x, tune_labels, tune_rows = _load_bundle(tune_dir)
    holdout_x, holdout_labels, holdout_rows = _load_bundle(holdout_dir)
    baseline_tune = pd.read_csv(baseline_tune_predictions)
    baseline_holdout = pd.read_csv(baseline_holdout_predictions)
    _identity(tune_rows, baseline_tune, "tune")
    _identity(holdout_rows, baseline_holdout, "holdout")
    tune_logits = np.asarray(model.decision_function(tune_x), dtype=np.float64)
    holdout_logits = np.asarray(model.decision_function(holdout_x), dtype=np.float64)
    temperature = fit_temperature(tune_labels, tune_logits)
    tune_probability = probabilities_from_logits(tune_logits, temperature)
    holdout_probability = probabilities_from_logits(holdout_logits, temperature)
    threshold, feasible, search = search_threshold(
        tune_labels,
        tune_probability,
        target_recall=0.80,
        target_specificity=0.90,
    )
    tune_metrics = probability_metrics(tune_labels, tune_probability, threshold, ece_bins=15)
    holdout_metrics = probability_metrics(
        holdout_labels, holdout_probability, threshold, ece_bins=15
    )
    holdout_predictions = (holdout_probability >= threshold).astype(np.int64)
    candidate_macro_recall = source_macro_metric(
        holdout_rows, holdout_predictions, label=1, group_field="uav_source"
    )
    candidate_macro_specificity = source_macro_metric(
        holdout_rows, holdout_predictions, label=0, group_field="background_source"
    )
    baseline_selected = baseline_holdout["selected_prediction"].to_numpy(dtype=np.int64)
    baseline_macro_recall = source_macro_metric(
        baseline_holdout, baseline_selected, label=1, group_field="uav_source"
    )
    baseline_macro_specificity = source_macro_metric(
        baseline_holdout, baseline_selected, label=0, group_field="background_source"
    )
    baseline_calibration = json.loads(baseline_calibration_path.read_text(encoding="utf-8"))
    baseline_metrics = baseline_calibration["holdout_metrics"]["temperature_selected"]
    dads_metrics = json.loads(dads_metrics_path.read_text(encoding="utf-8"))
    dads_test = dads_metrics["splits"]["test"]["metrics"]
    checks = [
        {"name": "dads_source_test_auc", "value": dads_test["auc"], "minimum": 0.98},
        {"name": "dads_source_test_f1", "value": dads_test["f1"], "minimum": 0.95},
        {
            "name": "val_ood_holdout_auc",
            "value": holdout_metrics["auc"],
            "minimum": float(baseline_metrics["auc"]) + 0.03,
        },
        {
            "name": "val_ood_holdout_f1",
            "value": holdout_metrics["f1"],
            "minimum": float(baseline_metrics["f1"]) - 0.01,
        },
        {
            "name": "uav_source_macro_recall",
            "value": candidate_macro_recall["macro"],
            "minimum": baseline_macro_recall["macro"] - 0.02,
        },
        {
            "name": "background_source_macro_specificity",
            "value": candidate_macro_specificity["macro"],
            "minimum": baseline_macro_specificity["macro"] - 0.02,
        },
    ]
    for check in checks:
        check["passed"] = bool(float(check["value"]) >= float(check["minimum"]))
    ensure_dirs(output_dir)
    for split, rows, logits, probability in (
        ("tune", tune_rows, tune_logits, tune_probability),
        ("holdout", holdout_rows, holdout_logits, holdout_probability),
    ):
        prediction = _prediction_frame(
            rows,
            logits,
            probabilities_from_logits(logits),
            probability,
            threshold,
        )
        ensure_dirs(output_dir / "predictions" / split)
        prediction.to_csv(output_dir / "predictions" / split / "predictions.csv", index=False)
    pd.DataFrame(search).to_csv(output_dir / "threshold_search.csv", index=False)
    report = {
        "passed": all(check["passed"] for check in checks),
        "decision": "proceed_to_beats_finetune" if all(check["passed"] for check in checks) else "stop_p1",
        "temperature": temperature,
        "selected_threshold": threshold,
        "constraints_feasible_on_tune": feasible,
        "tune_metrics": tune_metrics,
        "holdout_metrics": holdout_metrics,
        "holdout_subgroups": subgroup_metrics(holdout_rows, holdout_probability, threshold),
        "source_macro": {
            "baseline_recall": baseline_macro_recall,
            "candidate_recall": candidate_macro_recall,
            "baseline_specificity": baseline_macro_specificity,
            "candidate_specificity": candidate_macro_specificity,
        },
        "checks": checks,
        "inputs": {
            "model_sha256": sha256(model_path),
            "dads_metrics_sha256": sha256(dads_metrics_path),
            "tune_audit_sha256": sha256(tune_dir / "audit.json"),
            "holdout_audit_sha256": sha256(holdout_dir / "audit.json"),
        },
        "locked_datasets_read": [],
    }
    (output_dir / "gate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen BEATs probe on val_ood")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dads-metrics", type=Path, required=True)
    parser.add_argument("--tune-dir", type=Path, required=True)
    parser.add_argument("--holdout-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-calibration",
        type=Path,
        default=Path("artifacts/val_ood/calibration/crnn_full_augmented_g2/seed_42/calibration.json"),
    )
    parser.add_argument(
        "--baseline-tune-predictions",
        type=Path,
        default=Path("artifacts/val_ood/predictions/tune/crnn_full_augmented_g2/seed_42/predictions.csv"),
    )
    parser.add_argument(
        "--baseline-holdout-predictions",
        type=Path,
        default=Path("artifacts/val_ood/predictions/holdout/crnn_full_augmented_g2/seed_42/predictions.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/p1_beats_probe/evaluation")
    )
    args = parser.parse_args()
    result = evaluate_probe(
        args.model,
        args.dads_metrics,
        args.tune_dir,
        args.holdout_dir,
        args.baseline_calibration,
        args.baseline_tune_predictions,
        args.baseline_holdout_predictions,
        args.output_dir,
    )
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("P1 frozen BEATs probe did not pass its continuation gate")


if __name__ == "__main__":
    main()
