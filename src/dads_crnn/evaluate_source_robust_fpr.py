from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_dirs
from .data_firewall import (
    LOCKED_COMPACT_TOKENS,
    audit_csv_rows,
    compact_token,
    reject_locked_path,
    reject_locked_value,
)
from .evaluate_low_fpr import operating_point, ranking_metrics, threshold_at_target_fpr


ALGORITHM = "g8_source_block_order_statistic_v1"
IDENTITY_COLUMNS = (
    "dataset",
    "path",
    "sha256",
    "label",
    "source_group",
    "uav_source",
    "background_source",
    "condition",
    "ood_split",
)
SCORE_COLUMN = "calibrated_probability"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compact_path(value: str | Path) -> str:
    return compact_token(value)


def reject_locked_input_path(path: Path) -> None:
    """Reject a locked final-test path before any file is opened."""
    reject_locked_path(path, context="source-robust calibration input")


def _reject_locked_rows(frame: pd.DataFrame, source: Path) -> None:
    if "dataset" not in frame.columns:
        raise ValueError(f"Missing dataset column in {source}")
    datasets = frame["dataset"].fillna("").astype(str).str.strip().str.lower()
    if set(datasets.unique()) != {"val_ood"}:
        raise ValueError(f"G8 accepts only dataset=val_ood rows: {source}")
    for column in ("path", "uav_source", "background_source", "source_group"):
        if column not in frame.columns:
            continue
        for value in frame[column].dropna().astype(str):
            try:
                reject_locked_value(value, context=f"{source}:{column}")
            except ValueError as error:
                raise ValueError(
                    f"Locked final-test row in {source}: {column}={value}"
                ) from error


def _strict_binary_labels(values: pd.Series, source: Path) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError(f"Labels must be exactly binary integers in {source}")
    return numeric.astype(np.int64)


def _validate_hashes(values: pd.Series, source: Path) -> pd.Series:
    hashes = values.fillna("").astype(str).str.strip().str.lower()
    if hashes.duplicated().any():
        raise ValueError(f"Duplicate sample hashes in {source}")
    if not hashes.map(lambda value: bool(re.fullmatch(r"[0-9a-f]{64}", value))).all():
        raise ValueError(f"Malformed SHA256 value in {source}")
    return hashes


def _read_bound_predictions(
    predictions_path: Path,
    manifest_path: Path,
    expected_split: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # Audit every external path before the first read, including symlink targets.
    reject_locked_input_path(predictions_path)
    reject_locked_input_path(manifest_path)
    audit_csv_rows(predictions_path)
    audit_csv_rows(manifest_path)
    predictions = pd.read_csv(predictions_path)
    manifest = pd.read_csv(manifest_path)
    if predictions.empty or manifest.empty:
        raise ValueError("G8 inputs must be non-empty")
    required_predictions = set(IDENTITY_COLUMNS) | {SCORE_COLUMN}
    required_manifest = set(IDENTITY_COLUMNS)
    missing_predictions = sorted(required_predictions - set(predictions.columns))
    missing_manifest = sorted(required_manifest - set(manifest.columns))
    if missing_predictions:
        raise ValueError(f"Missing prediction columns in {predictions_path}: {missing_predictions}")
    if missing_manifest:
        raise ValueError(f"Missing manifest columns in {manifest_path}: {missing_manifest}")
    _reject_locked_rows(predictions, predictions_path)
    _reject_locked_rows(manifest, manifest_path)
    prediction_splits = set(predictions["ood_split"].fillna("").astype(str).str.strip())
    manifest_splits = set(manifest["ood_split"].fillna("").astype(str).str.strip())
    if prediction_splits != {expected_split} or manifest_splits != {expected_split}:
        raise ValueError(f"Expected only ood_split={expected_split}")
    predictions = predictions.copy()
    manifest = manifest.copy()
    predictions["label"] = _strict_binary_labels(predictions["label"], predictions_path)
    manifest["label"] = _strict_binary_labels(manifest["label"], manifest_path)
    predictions["sha256"] = _validate_hashes(predictions["sha256"], predictions_path)
    manifest["sha256"] = _validate_hashes(manifest["sha256"], manifest_path)
    probabilities = pd.to_numeric(predictions[SCORE_COLUMN], errors="raise").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError(f"Invalid {SCORE_COLUMN} values in {predictions_path}")
    predictions[SCORE_COLUMN] = probabilities
    prediction_index = predictions.set_index("sha256", drop=False).sort_index()
    manifest_index = manifest.set_index("sha256", drop=False).sort_index()
    if not prediction_index.index.equals(manifest_index.index):
        raise ValueError("Prediction rows do not match the canonical manifest SHA256 set")
    for column in IDENTITY_COLUMNS:
        if column in {"sha256", "label"}:
            left = prediction_index[column].to_numpy()
            right = manifest_index[column].to_numpy()
        else:
            left = prediction_index[column].fillna("").astype(str).to_numpy()
            right = manifest_index[column].fillna("").astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(f"Prediction/manifest identity mismatch in column {column}")
    audit = {
        "predictions": {
            "path": predictions_path.resolve().as_posix(),
            "sha256": sha256(predictions_path),
            "rows": int(len(predictions)),
        },
        "manifest": {
            "path": manifest_path.resolve().as_posix(),
            "sha256": sha256(manifest_path),
            "rows": int(len(manifest)),
        },
        "identity_columns": list(IDENTITY_COLUMNS),
        "identity_match": True,
    }
    return prediction_index.reset_index(drop=True), audit


def _validate_target_fprs(target_fprs: list[float]) -> list[float]:
    if not target_fprs:
        raise ValueError("At least one target FPR is required")
    values = np.asarray(target_fprs, dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values <= 0.0) | (values >= 1.0)):
        raise ValueError("Every target FPR must be finite and strictly between zero and one")
    return sorted(set(float(value) for value in values))


def _local_source_threshold(scores: np.ndarray, target_fpr: float) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Each background source requires finite scores")
    allowed = int(math.floor(target_fpr * values.size))
    descending = np.sort(values)[::-1]
    boundary = descending[allowed]
    threshold = float(np.nextafter(boundary, np.inf))
    actual = int(np.sum(values >= threshold))
    if actual > allowed:
        raise RuntimeError("Source-local threshold exceeded its empirical FPR budget")
    return {
        "samples": int(values.size),
        "allowed_false_positives": allowed,
        "actual_false_positives": actual,
        "empirical_fpr": float(actual / values.size),
        "boundary_score": float(boundary),
        "local_threshold": threshold,
    }


def source_block_threshold(
    tune_rows: pd.DataFrame,
    target_fpr: float,
    *,
    source_miscoverage_delta: float,
) -> dict[str, Any]:
    targets = _validate_target_fprs([target_fpr])
    target = targets[0]
    if not math.isfinite(source_miscoverage_delta) or not 0.0 < source_miscoverage_delta < 1.0:
        raise ValueError("source_miscoverage_delta must be strictly between zero and one")
    labels = _strict_binary_labels(tune_rows["label"], Path("<in-memory tune rows>"))
    negative = tune_rows.loc[labels == 0].copy()
    if negative.empty:
        raise ValueError("G8 calibration requires tune negatives")
    groups = negative["background_source"].fillna("").astype(str).str.strip()
    if (groups == "").any():
        raise ValueError("Every tune negative requires a non-empty background_source")
    negative["background_source"] = groups
    probabilities = pd.to_numeric(negative[SCORE_COLUMN], errors="raise").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError(f"Invalid {SCORE_COLUMN} values")
    negative[SCORE_COLUMN] = probabilities
    source_rows = []
    for source, rows in negative.groupby("background_source", sort=True):
        local = _local_source_threshold(rows[SCORE_COLUMN].to_numpy(dtype=np.float64), target)
        source_rows.append(
            {
                "source": str(source),
                "source_sha256": hashlib.sha256(str(source).encode("utf-8")).hexdigest(),
                **local,
            }
        )
    source_count = len(source_rows)
    rank = int(math.ceil((source_count + 1) * (1.0 - source_miscoverage_delta)))
    if rank > source_count:
        raise ValueError(
            "insufficient_source_groups: requested source coverage cannot be certified"
        )
    ordered = sorted(source_rows, key=lambda row: (row["local_threshold"], row["source"]))
    threshold = float(ordered[rank - 1]["local_threshold"])
    for item in source_rows:
        source_scores = negative.loc[
            negative["background_source"] == item["source"], SCORE_COLUMN
        ].to_numpy(dtype=np.float64)
        item["global_false_positives"] = int(np.sum(source_scores >= threshold))
        item["global_empirical_fpr"] = float(item["global_false_positives"] / len(source_scores))
        if item["global_false_positives"] > item["allowed_false_positives"]:
            raise RuntimeError("Global source-block threshold violated a local FPR budget")
    pooled_scores = negative[SCORE_COLUMN].to_numpy(dtype=np.float64)
    pooled = threshold_at_target_fpr(pooled_scores, target)
    rates = np.asarray([item["global_empirical_fpr"] for item in source_rows])
    counts = np.asarray([item["samples"] for item in source_rows])
    active = [item["source_sha256"] for item in source_rows if item["local_threshold"] == threshold]
    leave_one_out = []
    for excluded in range(source_count):
        remaining = [row["local_threshold"] for index, row in enumerate(ordered) if index != excluded]
        remaining_rank = int(
            math.ceil((len(remaining) + 1) * (1.0 - source_miscoverage_delta))
        )
        if remaining_rank <= len(remaining):
            leave_one_out.append(float(sorted(remaining)[remaining_rank - 1]))
    return {
        "target_fpr": target,
        "threshold": threshold,
        "source_miscoverage_delta": float(source_miscoverage_delta),
        "background_sources": source_count,
        "order_statistic_rank": rank,
        "source_coverage_lower_bound": float(rank / (source_count + 1)),
        "negative_samples": int(len(negative)),
        "source_sample_counts": {
            "minimum": int(counts.min()),
            "median": float(np.median(counts)),
            "maximum": int(counts.max()),
        },
        "pooled_threshold": float(pooled["threshold"]),
        "threshold_margin_over_pooled": float(threshold - float(pooled["threshold"])),
        "tune_pooled_fpr": float(np.sum(pooled_scores >= threshold) / len(pooled_scores)),
        "tune_source_macro_fpr": float(rates.mean()),
        "tune_source_p90_fpr": float(np.quantile(rates, 0.90)),
        "tune_source_p95_fpr": float(np.quantile(rates, 0.95)),
        "tune_worst_source_fpr": float(rates.max()),
        "active_source_hashes": active,
        "leave_one_source_out_threshold": {
            "minimum": float(min(leave_one_out)) if leave_one_out else None,
            "median": float(np.median(leave_one_out)) if leave_one_out else None,
            "maximum": float(max(leave_one_out)) if leave_one_out else None,
        },
        "per_source": sorted(source_rows, key=lambda row: row["source_sha256"]),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any], *, refuse_overwrite: bool) -> None:
    if refuse_overwrite and path.exists():
        raise FileExistsError(f"Frozen artifact already exists: {path}")
    ensure_dirs(path.parent)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def fit_source_robust_calibration(
    tune_predictions: Path,
    tune_manifest: Path,
    output_path: Path,
    *,
    experiment: str,
    target_fprs: list[float],
    source_miscoverage_delta: float,
    holdout_fpr_caps: list[float],
    minimum_holdout_tprs: list[float],
    minimum_tpr_retention: float,
    protocol_document: Path | None = None,
) -> dict[str, Any]:
    reject_locked_input_path(tune_predictions)
    reject_locked_input_path(tune_manifest)
    if protocol_document is not None:
        reject_locked_input_path(protocol_document)
    tune, audit = _read_bound_predictions(tune_predictions, tune_manifest, "tune")
    targets = _validate_target_fprs(target_fprs)
    if len(holdout_fpr_caps) != len(targets) or len(minimum_holdout_tprs) != len(targets):
        raise ValueError("Every target FPR requires one frozen FPR cap and minimum TPR")
    if not 0.0 < minimum_tpr_retention <= 1.0:
        raise ValueError("minimum_tpr_retention must be in (0, 1]")
    caps = [float(value) for value in holdout_fpr_caps]
    recalls = [float(value) for value in minimum_holdout_tprs]
    if not np.isfinite(caps).all() or not np.isfinite(recalls).all():
        raise ValueError("Frozen guardrails must be finite")
    calibrations = [
        source_block_threshold(
            tune,
            target,
            source_miscoverage_delta=source_miscoverage_delta,
        )
        for target in targets
    ]
    thresholds = [float(item["threshold"]) for item in calibrations]
    if any(left < right for left, right in zip(thresholds, thresholds[1:])):
        raise RuntimeError("Stricter target FPR unexpectedly produced a lower threshold")
    report = {
        "algorithm": ALGORITHM,
        "experiment": experiment,
        "protocol": {
            "score_column": SCORE_COLUMN,
            "group_column": "background_source",
            "comparison": "probability >= threshold",
            "tie_policy": "nextafter_boundary_toward_positive_infinity",
            "threshold_selection_labels": [0],
            "positive_rows_used_for_threshold_selection": False,
            "holdout_used_for_threshold_selection": False,
            "source_miscoverage_delta": float(source_miscoverage_delta),
            "exchangeability_assumption": "future background source blocks exchangeable with tune source blocks",
            "population_fpr_not_certified": True,
            "development_only": True,
        },
        "promotion_guardrails": {
            "target_fprs": targets,
            "holdout_fpr_caps": caps,
            "minimum_holdout_tprs": recalls,
            "minimum_tpr_retention": float(minimum_tpr_retention),
            "maximum_background_macro_fpr_reverse_change": 0.02,
        },
        "thresholds": calibrations,
        "implementation": {
            "path": Path(__file__).resolve().as_posix(),
            "sha256": sha256(Path(__file__)),
            "protocol_document": (
                {
                    "path": protocol_document.resolve().as_posix(),
                    "sha256": sha256(protocol_document),
                }
                if protocol_document is not None
                else None
            ),
        },
        "input_audit": audit,
        "opened_inputs": [
            audit["predictions"]["path"],
            audit["manifest"]["path"],
            *([protocol_document.resolve().as_posix()] if protocol_document is not None else []),
        ],
        "locked_datasets_read": [],
    }
    _atomic_write_json(output_path, report, refuse_overwrite=True)
    return report


def _load_json(path: Path) -> dict[str, Any]:
    reject_locked_input_path(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _points_by_target(report: dict[str, Any]) -> dict[float, dict[str, Any]]:
    return {float(item["target_fpr"]): item for item in report["operating_points"]}


def evaluate_frozen_source_robust(
    tune_predictions: Path,
    tune_manifest: Path,
    holdout_predictions: Path,
    holdout_manifest: Path,
    calibration_path: Path,
    reference_metrics_path: Path,
    baseline_metrics_path: Path,
    output_dir: Path,
    *,
    experiment: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    all_inputs = (
        tune_predictions,
        tune_manifest,
        holdout_predictions,
        holdout_manifest,
        calibration_path,
        reference_metrics_path,
        baseline_metrics_path,
    )
    for path in all_inputs:
        reject_locked_input_path(path)
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    calibration = _load_json(calibration_path)
    reference = _load_json(reference_metrics_path)
    baseline = _load_json(baseline_metrics_path)
    if calibration.get("algorithm") != ALGORITHM:
        raise ValueError("Unexpected frozen G8 algorithm")
    if calibration.get("protocol", {}).get("holdout_used_for_threshold_selection") is not False:
        raise ValueError("Frozen calibration does not prove holdout isolation")
    tune, tune_audit = _read_bound_predictions(tune_predictions, tune_manifest, "tune")
    holdout, holdout_audit = _read_bound_predictions(
        holdout_predictions, holdout_manifest, "holdout"
    )
    if tune_audit["predictions"]["sha256"] != calibration["input_audit"]["predictions"]["sha256"]:
        raise ValueError("Tune prediction SHA256 differs from the frozen calibration")
    if tune_audit["manifest"]["sha256"] != calibration["input_audit"]["manifest"]["sha256"]:
        raise ValueError("Tune manifest SHA256 differs from the frozen calibration")
    if set(tune["sha256"]) & set(holdout["sha256"]):
        raise ValueError("Tune and holdout sample SHA256 values overlap")
    tune_background = set(tune.loc[tune["label"] == 0, "background_source"].astype(str))
    holdout_background = set(
        holdout.loc[holdout["label"] == 0, "background_source"].astype(str)
    )
    tune_uav = set(tune.loc[tune["label"] == 1, "uav_source"].astype(str))
    holdout_uav = set(holdout.loc[holdout["label"] == 1, "uav_source"].astype(str))
    if tune_background & holdout_background or tune_uav & holdout_uav:
        raise ValueError("Tune and holdout source identities overlap")
    reference_inputs = reference.get("inputs", {})
    if reference_inputs.get("tune_sha256") != tune_audit["predictions"]["sha256"]:
        raise ValueError("Reference metrics are not bound to the G8 tune predictions")
    if reference_inputs.get("holdout_sha256") != holdout_audit["predictions"]["sha256"]:
        raise ValueError("Reference metrics are not bound to the G8 holdout predictions")
    tune_labels = tune["label"].to_numpy(dtype=np.int64)
    holdout_labels = holdout["label"].to_numpy(dtype=np.int64)
    tune_probabilities = tune[SCORE_COLUMN].to_numpy(dtype=np.float64)
    holdout_probabilities = holdout[SCORE_COLUMN].to_numpy(dtype=np.float64)
    guardrails = calibration["promotion_guardrails"]
    reference_points = _points_by_target(reference)
    baseline_points = _points_by_target(baseline)
    operating_points = []
    checks = []
    for index, frozen in enumerate(calibration["thresholds"]):
        target = float(frozen["target_fpr"])
        threshold = float(frozen["threshold"])
        tune_point = operating_point(
            tune,
            threshold,
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed + index * 10,
        )
        holdout_point = operating_point(
            holdout,
            threshold,
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed + index * 10 + 2,
        )
        reference_point = reference_points[target]["holdout"]
        baseline_point = baseline_points[target]["holdout"]
        position = guardrails["target_fprs"].index(target)
        fpr_cap = float(guardrails["holdout_fpr_caps"][position])
        minimum_tpr = float(guardrails["minimum_holdout_tprs"][position])
        retention = float(holdout_point["recall"] / reference_point["recall"])
        point_checks = {
            "holdout_fpr_cap": {
                "value": float(holdout_point["fpr"]),
                "maximum": fpr_cap,
                "passed": bool(holdout_point["fpr"] <= fpr_cap),
            },
            "minimum_holdout_tpr": {
                "value": float(holdout_point["recall"]),
                "minimum": minimum_tpr,
                "passed": bool(holdout_point["recall"] >= minimum_tpr),
            },
            "minimum_reference_tpr_retention": {
                "value": retention,
                "minimum": float(guardrails["minimum_tpr_retention"]),
                "passed": bool(retention >= float(guardrails["minimum_tpr_retention"])),
            },
            "fpr_nonincrease": {
                "value": float(holdout_point["fpr"]),
                "maximum": float(reference_point["fpr"]),
                "passed": bool(holdout_point["fpr"] <= reference_point["fpr"]),
            },
            "background_macro_guard": {
                "value": float(holdout_point["background_source_macro"]["macro_error"]),
                "maximum": float(
                    baseline_point["background_source_macro"]["macro_error"]
                    + guardrails["maximum_background_macro_fpr_reverse_change"]
                ),
                "passed": bool(
                    holdout_point["background_source_macro"]["macro_error"]
                    <= baseline_point["background_source_macro"]["macro_error"]
                    + guardrails["maximum_background_macro_fpr_reverse_change"]
                ),
            },
        }
        checks.append(
            {
                "target_fpr": target,
                "passed": all(item["passed"] for item in point_checks.values()),
                "checks": point_checks,
            }
        )
        operating_points.append(
            {
                "target_fpr": target,
                "frozen_calibration": frozen,
                "tune": tune_point,
                "holdout": holdout_point,
                "reference_holdout": reference_point,
                "baseline_holdout": baseline_point,
            }
        )
    ranks = {
        "tune": ranking_metrics(tune_labels, tune_probabilities),
        "holdout": ranking_metrics(holdout_labels, holdout_probabilities),
    }
    rank_consistency = {
        split: {
            metric: {
                "value": float(ranks[split][metric]),
                "reference": float(reference["ranking_metrics"][split][metric]),
                "passed": bool(
                    abs(ranks[split][metric] - reference["ranking_metrics"][split][metric])
                    <= 1e-12
                ),
            }
            for metric in ranks[split]
        }
        for split in ("tune", "holdout")
    }
    consistency_passed = all(
        item["passed"] for split in rank_consistency.values() for item in split.values()
    )
    passed = bool(consistency_passed and all(item["passed"] for item in checks))
    report = {
        "experiment": experiment,
        "algorithm": ALGORITHM,
        "decision": "pass_development_gate" if passed else "do_not_promote",
        "development_only": True,
        "protocol": {
            "thresholds_frozen_before_holdout_read": True,
            "holdout_used_for_threshold_selection": False,
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_seed": int(bootstrap_seed),
        },
        "ranking_metrics": ranks,
        "ranking_consistency": rank_consistency,
        "operating_points": operating_points,
        "gate": {
            "passed": passed,
            "ranking_consistency_passed": consistency_passed,
            "operating_point_checks": checks,
        },
        "source_isolation": {
            "sample_sha256_overlap": 0,
            "background_source_overlap": 0,
            "uav_source_overlap": 0,
        },
        "inputs": {
            "calibration": calibration_path.resolve().as_posix(),
            "calibration_sha256": sha256(calibration_path),
            "reference_metrics": reference_metrics_path.resolve().as_posix(),
            "reference_metrics_sha256": sha256(reference_metrics_path),
            "baseline_metrics": baseline_metrics_path.resolve().as_posix(),
            "baseline_metrics_sha256": sha256(baseline_metrics_path),
            "tune": tune_audit,
            "holdout": holdout_audit,
        },
        "opened_inputs": [path.resolve().as_posix() for path in all_inputs],
        "locked_datasets_read": [],
    }
    metrics_path = output_dir / "metrics.json"
    points_path = output_dir / "operating_points.csv"
    if metrics_path.exists() or points_path.exists():
        raise FileExistsError(f"G8 output already exists: {output_dir}")
    ensure_dirs(output_dir)
    rows = [
        {
            "target_fpr": item["target_fpr"],
            "threshold": item["frozen_calibration"]["threshold"],
            "tune_fpr": item["tune"]["fpr"],
            "tune_tpr": item["tune"]["recall"],
            "holdout_fpr": item["holdout"]["fpr"],
            "holdout_tpr": item["holdout"]["recall"],
            "holdout_precision": item["holdout"]["precision"],
            "holdout_f1": item["holdout"]["f1"],
            "reference_holdout_fpr": item["reference_holdout"]["fpr"],
            "reference_holdout_tpr": item["reference_holdout"]["recall"],
            "passed": checks[index]["passed"],
        }
        for index, item in enumerate(operating_points)
    ]
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_dir, prefix=".operating_points.", delete=False
    ) as handle:
        temporary_csv = Path(handle.name)
        pd.DataFrame(rows).to_csv(handle, index=False)
    os.replace(temporary_csv, points_path)
    _atomic_write_json(metrics_path, report, refuse_overwrite=True)
    return report


def _fit_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("fit", help="Fit and freeze G8 thresholds from tune only")
    parser.add_argument("--tune-predictions", type=Path, required=True)
    parser.add_argument("--tune-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--target-fprs", type=float, nargs="+", default=[0.01, 0.05])
    parser.add_argument("--source-miscoverage-delta", type=float, default=0.05)
    parser.add_argument("--holdout-fpr-caps", type=float, nargs="+", default=[0.02, 0.075])
    parser.add_argument("--minimum-holdout-tprs", type=float, nargs="+", default=[0.10, 0.30])
    parser.add_argument("--minimum-tpr-retention", type=float, default=0.80)
    parser.add_argument("--protocol-document", type=Path)


def _evaluate_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("evaluate", help="Apply frozen G8 thresholds to holdout")
    parser.add_argument("--tune-predictions", type=Path, required=True)
    parser.add_argument("--tune-manifest", type=Path, required=True)
    parser.add_argument("--holdout-predictions", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--reference-metrics", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260720)


def main() -> None:
    parser = argparse.ArgumentParser(description="G8 source-block robust FPR calibration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _fit_parser(subparsers)
    _evaluate_parser(subparsers)
    args = parser.parse_args()
    if args.command == "fit":
        result = fit_source_robust_calibration(
            args.tune_predictions,
            args.tune_manifest,
            args.output,
            experiment=args.experiment,
            target_fprs=args.target_fprs,
            source_miscoverage_delta=args.source_miscoverage_delta,
            holdout_fpr_caps=args.holdout_fpr_caps,
            minimum_holdout_tprs=args.minimum_holdout_tprs,
            minimum_tpr_retention=args.minimum_tpr_retention,
            protocol_document=args.protocol_document,
        )
    else:
        result = evaluate_frozen_source_robust(
            args.tune_predictions,
            args.tune_manifest,
            args.holdout_predictions,
            args.holdout_manifest,
            args.calibration,
            args.reference_metrics,
            args.baseline_metrics,
            args.output_dir,
            experiment=args.experiment,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
