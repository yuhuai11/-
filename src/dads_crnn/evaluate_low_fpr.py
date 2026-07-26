from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import ensure_dirs
from .data_firewall import audit_csv_rows
from .evaluate_external import _fast_threshold_metrics
from .prepare_beats_probe import reject_locked_path


REQUIRED_COLUMNS = {
    "path",
    "sha256",
    "label",
    "uav_source",
    "background_source",
    "condition",
    "ood_split",
    "calibrated_probability",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_predictions(path: Path, expected_split: str) -> pd.DataFrame:
    reject_locked_path(path)
    audit_csv_rows(path)
    rows = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(rows.columns))
    if missing:
        raise ValueError(f"Missing prediction columns in {path}: {missing}")
    if rows.empty:
        raise ValueError(f"Empty prediction file: {path}")
    labels = rows["label"].to_numpy(dtype=np.int64)
    probabilities = rows["calibrated_probability"].to_numpy(dtype=np.float64)
    if not set(np.unique(labels)).issubset({0, 1}) or len(np.unique(labels)) != 2:
        raise ValueError(f"Predictions must contain both binary labels: {path}")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError(f"Invalid calibrated probabilities: {path}")
    observed_splits = set(rows["ood_split"].astype(str).unique())
    if observed_splits != {expected_split}:
        raise ValueError(
            f"Expected only ood_split={expected_split}, observed {sorted(observed_splits)}"
        )
    if rows["sha256"].astype(str).duplicated().any():
        raise ValueError(f"Duplicate sample hashes in {path}")
    return rows


def threshold_at_target_fpr(negative_probabilities: np.ndarray, target_fpr: float) -> dict[str, Any]:
    negatives = np.asarray(negative_probabilities, dtype=np.float64)
    if negatives.ndim != 1 or negatives.size == 0 or not np.isfinite(negatives).all():
        raise ValueError("A non-empty finite 1D negative score array is required")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be strictly between zero and one")
    allowed_false_positives = int(np.floor(target_fpr * negatives.size))
    descending = np.sort(negatives)[::-1]
    if allowed_false_positives == 0:
        threshold = float(np.nextafter(descending[0], np.inf))
    else:
        first_disallowed = descending[allowed_false_positives]
        threshold = float(np.nextafter(first_disallowed, np.inf))
    actual_false_positives = int(np.sum(negatives >= threshold))
    if actual_false_positives > allowed_false_positives:
        raise RuntimeError("Conservative FPR calibration exceeded its false-positive budget")
    return {
        "target_fpr": float(target_fpr),
        "threshold": threshold,
        "negative_samples": int(negatives.size),
        "allowed_false_positives": allowed_false_positives,
        "actual_false_positives": actual_false_positives,
        "empirical_fpr": float(actual_false_positives / negatives.size),
    }


def clopper_pearson(successes: int, total: int, confidence: float = 0.95) -> dict[str, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Invalid binomial counts")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one")
    alpha = 1.0 - confidence
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, total - successes + 1))
    high = (
        1.0
        if successes == total
        else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, total - successes))
    )
    return {"low": low, "high": high, "confidence": float(confidence)}


def source_macro(
    rows: pd.DataFrame,
    predictions: np.ndarray,
    *,
    label: int,
    group_field: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    labels = rows["label"].to_numpy(dtype=np.int64)
    selected = labels == label
    frame = rows.loc[selected, [group_field]].copy()
    frame["correct"] = predictions[selected] == label
    frame = frame.loc[frame[group_field].fillna("").astype(str) != ""]
    values = frame.groupby(group_field, sort=True)["correct"].mean().to_numpy(dtype=np.float64)
    if values.size == 0:
        raise ValueError(f"No source groups for {group_field} label={label}")
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        bootstrap[index] = rng.choice(values, size=values.size, replace=True).mean()
    macro_correct = float(values.mean())
    return {
        "group_field": group_field,
        "groups": int(values.size),
        "macro_correct": macro_correct,
        "macro_error": float(1.0 - macro_correct),
        "minimum_correct": float(values.min()),
        "bootstrap_95_ci": {
            "low": float(np.quantile(bootstrap, 0.025)),
            "high": float(np.quantile(bootstrap, 0.975)),
        },
    }


def ranking_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "partial_auc_fpr_0_05_standardized": float(
            roc_auc_score(labels, probabilities, max_fpr=0.05)
        ),
    }


def operating_point(
    rows: pd.DataFrame,
    threshold: float,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    labels = rows["label"].to_numpy(dtype=np.int64)
    probabilities = rows["calibrated_probability"].to_numpy(dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    positive = labels == 1
    negative = labels == 0
    tp = int(np.sum((predictions == 1) & positive))
    fp = int(np.sum((predictions == 1) & negative))
    metrics = _fast_threshold_metrics(labels, probabilities, threshold)
    metrics.update(
        {
            "threshold": float(threshold),
            "positive_samples": int(positive.sum()),
            "negative_samples": int(negative.sum()),
            "true_positives": tp,
            "false_positives": fp,
            "fpr": float(fp / negative.sum()),
            "tpr_clopper_pearson_95_ci": clopper_pearson(tp, int(positive.sum())),
            "fpr_clopper_pearson_95_ci": clopper_pearson(fp, int(negative.sum())),
            "uav_source_macro": source_macro(
                rows,
                predictions,
                label=1,
                group_field="uav_source",
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            ),
            "background_source_macro": source_macro(
                rows,
                predictions,
                label=0,
                group_field="background_source",
                bootstrap_samples=bootstrap_samples,
                seed=seed + 1,
            ),
        }
    )
    return metrics


def evaluate_low_fpr(
    tune_predictions: Path,
    holdout_predictions: Path,
    output_dir: Path,
    *,
    experiment: str,
    target_fprs: list[float],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    tune = validate_predictions(tune_predictions, "tune")
    holdout = validate_predictions(holdout_predictions, "holdout")
    if set(tune["sha256"].astype(str)) & set(holdout["sha256"].astype(str)):
        raise ValueError("Tune and holdout sample hashes overlap")
    tune_labels = tune["label"].to_numpy(dtype=np.int64)
    tune_probabilities = tune["calibrated_probability"].to_numpy(dtype=np.float64)
    holdout_labels = holdout["label"].to_numpy(dtype=np.int64)
    holdout_probabilities = holdout["calibrated_probability"].to_numpy(dtype=np.float64)
    calibrations = [
        threshold_at_target_fpr(tune_probabilities[tune_labels == 0], target)
        for target in sorted(set(target_fprs))
    ]
    operating_points = []
    for index, calibration in enumerate(calibrations):
        threshold = float(calibration["threshold"])
        operating_points.append(
            {
                "target_fpr": float(calibration["target_fpr"]),
                "calibration": calibration,
                "tune": operating_point(
                    tune,
                    threshold,
                    bootstrap_samples=bootstrap_samples,
                    seed=bootstrap_seed + index * 10,
                ),
                "holdout": operating_point(
                    holdout,
                    threshold,
                    bootstrap_samples=bootstrap_samples,
                    seed=bootstrap_seed + index * 10 + 2,
                ),
            }
        )
    report = {
        "experiment": experiment,
        "protocol": {
            "threshold_selection": "negative_only_conservative_empirical_quantile",
            "target_fprs": [float(value) for value in sorted(set(target_fprs))],
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_seed": int(bootstrap_seed),
            "holdout_used_for_threshold_selection": False,
        },
        "ranking_metrics": {
            "tune": ranking_metrics(tune_labels, tune_probabilities),
            "holdout": ranking_metrics(holdout_labels, holdout_probabilities),
        },
        "operating_points": operating_points,
        "inputs": {
            "tune_predictions": tune_predictions.as_posix(),
            "tune_sha256": sha256(tune_predictions),
            "holdout_predictions": holdout_predictions.as_posix(),
            "holdout_sha256": sha256(holdout_predictions),
        },
        "locked_datasets_read": [],
    }
    ensure_dirs(output_dir)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "target_fpr": item["target_fpr"],
                "threshold": item["calibration"]["threshold"],
                "tune_fpr": item["tune"]["fpr"],
                "tune_tpr": item["tune"]["recall"],
                "holdout_fpr": item["holdout"]["fpr"],
                "holdout_tpr": item["holdout"]["recall"],
                "holdout_precision": item["holdout"]["precision"],
                "holdout_f1": item["holdout"]["f1"],
                "holdout_uav_source_macro_tpr": item["holdout"]["uav_source_macro"][
                    "macro_correct"
                ],
                "holdout_background_source_macro_fpr": item["holdout"][
                    "background_source_macro"
                ]["macro_error"],
            }
            for item in operating_points
        ]
    ).to_csv(output_dir / "operating_points.csv", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen predictions at target FPRs")
    parser.add_argument("--tune-predictions", type=Path, required=True)
    parser.add_argument("--holdout-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--target-fprs", type=float, nargs="+", default=[0.01, 0.05])
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    result = evaluate_low_fpr(
        args.tune_predictions,
        args.holdout_predictions,
        args.output_dir,
        experiment=args.experiment,
        target_fprs=args.target_fprs,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
