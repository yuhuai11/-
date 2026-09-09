from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calibrate_ood import fit_temperature, probabilities_from_logits
from .config import ensure_dirs, load_config
from .evaluate_low_fpr import operating_point, ranking_metrics, threshold_at_target_fpr
from .evaluate_source_robust_fpr import (
    _read_bound_predictions,
    reject_locked_input_path,
    sha256,
)


ALGORITHM = "g10_logit_ensemble_source_quantile_v1"


def _atomic_json(path: Path, payload: dict[str, Any], *, refuse_overwrite: bool) -> None:
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


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("algorithm") != ALGORITHM:
        raise ValueError(f"G10 algorithm must be frozen as {ALGORITHM}")
    seeds = [int(value) for value in config.get("seeds", [])]
    if seeds != [42, 43, 44]:
        raise ValueError("G10 requires the frozen ordered seeds [42, 43, 44]")
    calibration = config.get("calibration", {})
    if [float(value) for value in calibration.get("target_fprs", [])] != [0.01, 0.05]:
        raise ValueError("G10 target FPRs are frozen as [0.01, 0.05]")
    if [float(value) for value in calibration.get("source_coverage", [])] != [0.90, 0.75]:
        raise ValueError("G10 source coverage is frozen as [0.90, 0.75]")
    if float(calibration.get("minimum_tune_tpr_retention", -1)) != 0.90:
        raise ValueError("G10 minimum tune TPR retention is frozen as 0.90")
    if calibration.get("group_column") != "background_source":
        raise ValueError("G10 group column is frozen as background_source")
    if calibration.get("ensemble") != "arithmetic_mean_raw_logit":
        raise ValueError("G10 requires arithmetic mean raw-logit ensembling")


def _ensemble_predictions(
    prediction_paths: list[Path],
    manifest_path: Path,
    split: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if len(prediction_paths) != 3:
        raise ValueError("G10 requires exactly three prediction files")
    frames = []
    audits = []
    for path in prediction_paths:
        rows, audit = _read_bound_predictions(path, manifest_path, split)
        if "logit" not in rows.columns:
            raise ValueError(f"Missing raw logit column in {path}")
        logits = pd.to_numeric(rows["logit"], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(logits).all():
            raise ValueError(f"Non-finite raw logits in {path}")
        rows = rows.copy()
        rows["logit"] = logits
        frames.append(rows)
        audits.append(audit)
    reference_hashes = frames[0]["sha256"].astype(str).to_numpy()
    for rows in frames[1:]:
        if not np.array_equal(reference_hashes, rows["sha256"].astype(str).to_numpy()):
            raise ValueError("G10 seed predictions are not sample-aligned")
    output = frames[0].copy()
    output["ensemble_logit"] = np.mean(
        [rows["logit"].to_numpy(dtype=np.float64) for rows in frames], axis=0
    )
    return output, audits


def source_quantile_threshold(
    rows: pd.DataFrame,
    target_fpr: float,
    source_coverage: float,
) -> dict[str, Any]:
    negative = rows.loc[rows["label"].astype(int) == 0].copy()
    groups = negative["background_source"].fillna("").astype(str).str.strip()
    if negative.empty or (groups == "").any():
        raise ValueError("Every G10 tune negative requires a background source")
    negative["background_source"] = groups
    pooled = threshold_at_target_fpr(
        negative["calibrated_probability"].to_numpy(dtype=np.float64), target_fpr
    )
    local_rows = []
    for source, group in negative.groupby("background_source", sort=True):
        scores = np.sort(group["calibrated_probability"].to_numpy(dtype=np.float64))[::-1]
        allowed = int(math.floor(target_fpr * len(scores)))
        threshold = float(np.nextafter(scores[allowed], np.inf))
        local_rows.append(
            {
                "source_sha256": hashlib.sha256(str(source).encode("utf-8")).hexdigest(),
                "samples": int(len(scores)),
                "allowed_false_positives": allowed,
                "local_threshold": threshold,
            }
        )
    ordered = sorted(local_rows, key=lambda item: item["local_threshold"])
    rank = int(math.ceil((len(ordered) + 1) * source_coverage))
    if rank > len(ordered):
        raise ValueError("Requested G10 source coverage is not estimable")
    source_threshold = float(ordered[rank - 1]["local_threshold"])
    threshold = max(float(pooled["threshold"]), source_threshold)
    rates = []
    for source, group in negative.groupby("background_source", sort=True):
        del source
        rates.append(float((group["calibrated_probability"] >= threshold).mean()))
    return {
        "target_fpr": float(target_fpr),
        "threshold": threshold,
        "pooled_threshold": float(pooled["threshold"]),
        "source_threshold": source_threshold,
        "source_coverage_target": float(source_coverage),
        "source_coverage_order_statistic": float(rank / (len(ordered) + 1)),
        "order_statistic_rank": rank,
        "background_sources": int(len(ordered)),
        "tune_pooled_fpr": float(
            (negative["calibrated_probability"] >= threshold).mean()
        ),
        "tune_source_macro_fpr": float(np.mean(rates)),
        "tune_source_p90_fpr": float(np.quantile(rates, 0.90)),
        "tune_worst_source_fpr": float(np.max(rates)),
        "local_thresholds": ordered,
    }


def fit(config_path: Path) -> dict[str, Any]:
    reject_locked_input_path(config_path)
    config = load_config(config_path)
    _validate_config(config)
    inputs = config["inputs"]
    tune_manifest = Path(inputs["tune_manifest"])
    tune_paths = [Path(value) for value in inputs["tune_predictions"]]
    protocol_document = Path(config["protocol_document"])
    for path in (*tune_paths, tune_manifest, protocol_document):
        reject_locked_input_path(path)
    rows, audits = _ensemble_predictions(tune_paths, tune_manifest, "tune")
    labels = rows["label"].to_numpy(dtype=np.int64)
    temperature = fit_temperature(labels, rows["ensemble_logit"].to_numpy(dtype=np.float64))
    rows["calibrated_probability"] = probabilities_from_logits(
        rows["ensemble_logit"].to_numpy(dtype=np.float64), temperature
    )
    settings = config["calibration"]
    thresholds = []
    for target, coverage in zip(
        settings["target_fprs"], settings["source_coverage"], strict=True
    ):
        item = source_quantile_threshold(rows, float(target), float(coverage))
        pooled_threshold = float(item["pooled_threshold"])
        positive = rows["label"].astype(int) == 1
        pooled_tpr = float(
            (rows.loc[positive, "calibrated_probability"] >= pooled_threshold).mean()
        )
        selected_tpr = float(
            (rows.loc[positive, "calibrated_probability"] >= item["threshold"]).mean()
        )
        retention = selected_tpr / pooled_tpr
        item.update(
            {
                "tune_pooled_tpr": pooled_tpr,
                "tune_selected_tpr": selected_tpr,
                "tune_tpr_retention": retention,
                "minimum_tune_tpr_retention": float(
                    settings["minimum_tune_tpr_retention"]
                ),
            }
        )
        if retention < float(settings["minimum_tune_tpr_retention"]):
            raise RuntimeError(f"G10 target {target} failed the frozen tune TPR retention gate")
        thresholds.append(item)
    output = Path(config["outputs"]["frozen_calibration"])
    report = {
        "algorithm": ALGORITHM,
        "protocol": {
            "seeds": [42, 43, 44],
            "ensemble": "arithmetic_mean_raw_logit",
            "temperature_fit_split": "tune",
            "threshold_fit_split": "tune",
            "threshold_selection_labels": [0],
            "positive_rows_used_for_threshold_selection": False,
            "positive_rows_used_for_feasibility_only": True,
            "holdout_used_for_fitting": False,
            "locked_final_tests_used": False,
        },
        "temperature": float(temperature),
        "thresholds": thresholds,
        "inputs": {
            "config": {"path": config_path.as_posix(), "sha256": sha256(config_path)},
            "protocol_document": {
                "path": protocol_document.as_posix(),
                "sha256": sha256(protocol_document),
            },
            "tune_manifest": {"path": tune_manifest.as_posix(), "sha256": sha256(tune_manifest)},
            "tune_predictions": audits,
        },
        "implementation": {
            "path": Path(__file__).resolve().as_posix(),
            "sha256": sha256(Path(__file__)),
        },
        "locked_datasets_read": [],
    }
    _atomic_json(output, report, refuse_overwrite=True)
    print(json.dumps({"temperature": temperature, "thresholds": thresholds}, indent=2))
    return report


def evaluate(config_path: Path) -> dict[str, Any]:
    reject_locked_input_path(config_path)
    config = load_config(config_path)
    _validate_config(config)
    calibration_path = Path(config["outputs"]["frozen_calibration"])
    reject_locked_input_path(calibration_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("algorithm") != ALGORITHM:
        raise ValueError("Unexpected G10 frozen calibration algorithm")
    if calibration.get("protocol", {}).get("holdout_used_for_fitting") is not False:
        raise ValueError("G10 calibration does not prove holdout isolation")
    if calibration["inputs"]["config"]["sha256"] != sha256(config_path):
        raise ValueError("G10 config changed after calibration was frozen")
    protocol_document = Path(config["protocol_document"])
    if calibration["inputs"]["protocol_document"]["sha256"] != sha256(
        protocol_document
    ):
        raise ValueError("G10 protocol document changed after calibration was frozen")

    inputs = config["inputs"]
    tune_manifest = Path(inputs["tune_manifest"])
    holdout_manifest = Path(inputs["holdout_manifest"])
    tune_paths = [Path(value) for value in inputs["tune_predictions"]]
    holdout_paths = [Path(value) for value in inputs["holdout_predictions"]]
    baseline_path = Path(inputs["baseline_metrics"])
    for path in (*tune_paths, *holdout_paths, tune_manifest, holdout_manifest, baseline_path):
        reject_locked_input_path(path)
    tune, tune_audits = _ensemble_predictions(tune_paths, tune_manifest, "tune")
    holdout, holdout_audits = _ensemble_predictions(
        holdout_paths, holdout_manifest, "holdout"
    )
    expected_tune_hashes = [
        item["predictions"]["sha256"] for item in calibration["inputs"]["tune_predictions"]
    ]
    actual_tune_hashes = [item["predictions"]["sha256"] for item in tune_audits]
    if actual_tune_hashes != expected_tune_hashes:
        raise ValueError("G10 tune predictions changed after calibration was frozen")
    if set(tune["sha256"].astype(str)) & set(holdout["sha256"].astype(str)):
        raise ValueError("G10 tune and holdout hashes overlap")
    temperature = float(calibration["temperature"])
    for rows in (tune, holdout):
        rows["calibrated_probability"] = probabilities_from_logits(
            rows["ensemble_logit"].to_numpy(dtype=np.float64), temperature
        )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_points = {
        float(item["target_fpr"]): item for item in baseline["operating_points"]
    }
    settings = config["evaluation"]
    points = []
    all_checks = []
    for index, frozen in enumerate(calibration["thresholds"]):
        target = float(frozen["target_fpr"])
        threshold = float(frozen["threshold"])
        tune_metrics = operating_point(
            tune,
            threshold,
            bootstrap_samples=int(settings["bootstrap_samples"]),
            seed=int(settings["bootstrap_seed"]) + index * 10,
        )
        holdout_metrics = operating_point(
            holdout,
            threshold,
            bootstrap_samples=int(settings["bootstrap_samples"]),
            seed=int(settings["bootstrap_seed"]) + index * 10 + 1,
        )
        reference = baseline_points[target]["holdout"]
        checks = [
            {"name": "holdout_fpr", "value": holdout_metrics["fpr"], "maximum": reference["fpr"], "passed": holdout_metrics["fpr"] <= reference["fpr"]},
            {"name": "holdout_tpr", "value": holdout_metrics["recall"], "minimum": reference["recall"], "passed": holdout_metrics["recall"] >= reference["recall"]},
            {"name": "holdout_f1", "value": holdout_metrics["f1"], "minimum": reference["f1"], "passed": holdout_metrics["f1"] >= reference["f1"]},
            {"name": "uav_source_macro_tpr", "value": holdout_metrics["uav_source_macro"]["macro_correct"], "minimum": reference["uav_source_macro"]["macro_correct"], "passed": holdout_metrics["uav_source_macro"]["macro_correct"] >= reference["uav_source_macro"]["macro_correct"]},
            {"name": "background_source_macro_fpr", "value": holdout_metrics["background_source_macro"]["macro_error"], "maximum": reference["background_source_macro"]["macro_error"], "passed": holdout_metrics["background_source_macro"]["macro_error"] <= reference["background_source_macro"]["macro_error"]},
        ]
        all_checks.extend(checks)
        points.append(
            {
                "target_fpr": target,
                "frozen_calibration": frozen,
                "tune": tune_metrics,
                "holdout": holdout_metrics,
                "baseline_holdout": reference,
                "checks": checks,
            }
        )
    ensemble_ranking = {
        "tune": ranking_metrics(
            tune["label"].to_numpy(dtype=np.int64),
            tune["calibrated_probability"].to_numpy(dtype=np.float64),
        ),
        "holdout": ranking_metrics(
            holdout["label"].to_numpy(dtype=np.int64),
            holdout["calibrated_probability"].to_numpy(dtype=np.float64),
        ),
    }
    baseline_ranking = baseline["ranking_metrics"]["holdout"]
    ranking_checks = [
        {
            "name": name,
            "value": ensemble_ranking["holdout"][name],
            "minimum": baseline_ranking[name],
            "passed": ensemble_ranking["holdout"][name] >= baseline_ranking[name],
        }
        for name in ("roc_auc", "pr_auc", "partial_auc_fpr_0_05_standardized")
    ]
    decision = "promote" if all(item["passed"] for item in (*ranking_checks, *all_checks)) else "do_not_promote"
    output_dir = Path(config["outputs"]["evaluation_dir"])
    if (output_dir / "metrics.json").exists():
        raise FileExistsError(f"Frozen G10 evaluation already exists: {output_dir}")
    report = {
        "algorithm": ALGORITHM,
        "decision": decision,
        "ranking_metrics": ensemble_ranking,
        "ranking_checks": ranking_checks,
        "operating_points": points,
        "inputs": {
            "calibration": {"path": calibration_path.as_posix(), "sha256": sha256(calibration_path)},
            "holdout_manifest": {"path": holdout_manifest.as_posix(), "sha256": sha256(holdout_manifest)},
            "holdout_predictions": holdout_audits,
            "baseline_metrics": {"path": baseline_path.as_posix(), "sha256": sha256(baseline_path)},
        },
        "locked_datasets_read": [],
    }
    ensure_dirs(output_dir)
    tune.to_csv(output_dir / "tune_predictions.csv", index=False)
    holdout.to_csv(output_dir / "holdout_predictions.csv", index=False)
    _atomic_json(output_dir / "metrics.json", report, refuse_overwrite=True)
    print(json.dumps({"decision": decision, "ranking_checks": ranking_checks, "operating_points": points}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="G10 frozen ensemble source-robust calibration")
    parser.add_argument("mode", choices=("fit", "evaluate"))
    parser.add_argument("--config", type=Path, default=Path("configs/g10_ensemble_source_robust.yaml"))
    args = parser.parse_args()
    fit(args.config) if args.mode == "fit" else evaluate(args.config)


if __name__ == "__main__":
    main()
