from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .calibrate_ood import probabilities_from_logits
from .config import ensure_dirs, load_config
from .evaluate_g10_ensemble import (
    _atomic_json,
    _ensemble_predictions,
    source_quantile_threshold,
)
from .evaluate_low_fpr import operating_point, ranking_metrics
from .evaluate_source_robust_fpr import reject_locked_input_path, sha256


ALGORITHM = "g12_strict_source_conformal_v1"
G10_ALGORITHM = "g10_logit_ensemble_source_quantile_v1"


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("algorithm") != ALGORITHM:
        raise ValueError(f"G12 algorithm must be {ALGORITHM}")
    if [int(value) for value in config.get("seeds", [])] != [42, 43, 44]:
        raise ValueError("G12 requires the frozen ordered seeds [42, 43, 44]")
    settings = config.get("calibration", {})
    expected = {
        "target_fpr": 0.01,
        "source_coverage": 0.95,
        "minimum_tune_tpr_retention": 0.80,
        "group_column": "background_source",
        "threshold_selection": "maximum_estimable_source_order_statistic",
        "balanced_mode_policy": "retain_g7",
    }
    for key, value in expected.items():
        actual = settings.get(key)
        if isinstance(value, float):
            if not math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"G12 {key} must be frozen as {value}")
        elif actual != value:
            raise ValueError(f"G12 {key} must be frozen as {value}")


def _load_g10_temperature(path: Path) -> tuple[float, dict[str, Any]]:
    reject_locked_input_path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("algorithm") != G10_ALGORITHM:
        raise ValueError("G12 requires the frozen G10 ensemble calibration")
    if report.get("protocol", {}).get("locked_final_tests_used") is not False:
        raise ValueError("G10 temperature provenance does not prove final-test isolation")
    temperature = float(report["temperature"])
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Invalid frozen G10 temperature")
    return temperature, report


def leave_one_source_out_audit(rows, target_fpr: float) -> dict[str, Any]:
    negative = rows.loc[rows["label"].astype(int) == 0].copy()
    groups = negative["background_source"].fillna("").astype(str).str.strip()
    if negative.empty or (groups == "").any():
        raise ValueError("Every G12 tune negative requires a background source")
    negative["background_source"] = groups
    source_names = sorted(negative["background_source"].unique())
    audits = []
    for held_source in source_names:
        fit_rows = negative.loc[negative["background_source"] != held_source]
        held = negative.loc[negative["background_source"] == held_source]
        remaining_sources = int(fit_rows["background_source"].nunique())
        maximum_estimable_coverage = remaining_sources / (remaining_sources + 1)
        fitted = source_quantile_threshold(
            fit_rows, target_fpr, maximum_estimable_coverage
        )
        scores = held["calibrated_probability"].to_numpy(dtype=np.float64)
        false_positives = int(np.sum(scores >= float(fitted["threshold"])))
        audits.append(
            {
                "held_source_sha256": sha256_text(held_source),
                "samples": int(len(scores)),
                "threshold": float(fitted["threshold"]),
                "false_positives": false_positives,
                "fpr": float(false_positives / len(scores)),
            }
        )
    rates = np.asarray([item["fpr"] for item in audits], dtype=np.float64)
    return {
        "sources": len(audits),
        "source_identifiers_hashed": True,
        "passing_sources_at_target_fpr": int(np.sum(rates <= target_fpr)),
        "passing_fraction": float(np.mean(rates <= target_fpr)),
        "macro_fpr": float(rates.mean()),
        "worst_fpr": float(rates.max()),
        "per_source": audits,
    }


def sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fit(config_path: Path) -> dict[str, Any]:
    reject_locked_input_path(config_path)
    config = load_config(config_path)
    _validate_config(config)
    inputs = config["inputs"]
    tune_manifest = Path(inputs["tune_manifest"])
    tune_paths = [Path(value) for value in inputs["tune_predictions"]]
    g10_path = Path(inputs["g10_calibration"])
    protocol_document = Path(config["protocol_document"])
    for path in (*tune_paths, tune_manifest, g10_path, protocol_document):
        reject_locked_input_path(path)

    temperature, _ = _load_g10_temperature(g10_path)
    rows, audits = _ensemble_predictions(tune_paths, tune_manifest, "tune")
    rows["calibrated_probability"] = probabilities_from_logits(
        rows["ensemble_logit"].to_numpy(dtype=np.float64), temperature
    )
    settings = config["calibration"]
    selected = source_quantile_threshold(
        rows, float(settings["target_fpr"]), float(settings["source_coverage"])
    )
    if selected["order_statistic_rank"] != selected["background_sources"]:
        raise RuntimeError("G12 must select the maximum estimable source order statistic")
    positive = rows["label"].astype(int) == 1
    pooled_tpr = float(
        (rows.loc[positive, "calibrated_probability"] >= selected["pooled_threshold"]).mean()
    )
    selected_tpr = float(
        (rows.loc[positive, "calibrated_probability"] >= selected["threshold"]).mean()
    )
    retention = selected_tpr / pooled_tpr
    selected.update(
        {
            "tune_pooled_tpr": pooled_tpr,
            "tune_selected_tpr": selected_tpr,
            "tune_tpr_retention": retention,
            "minimum_tune_tpr_retention": float(settings["minimum_tune_tpr_retention"]),
        }
    )
    if retention < float(settings["minimum_tune_tpr_retention"]):
        raise RuntimeError("G12 failed its frozen tune TPR-retention floor")

    report = {
        "algorithm": ALGORITHM,
        "protocol": {
            "mode": "strict_only",
            "balanced_mode_policy": "retain_g7",
            "temperature_reused_from": G10_ALGORITHM,
            "threshold_fit_split": "tune",
            "threshold_selection_labels": [0],
            "holdout_used_for_fitting": False,
            "locked_final_tests_used": False,
        },
        "temperature": temperature,
        "strict_calibration": selected,
        "leave_one_source_out_audit": leave_one_source_out_audit(
            rows, float(settings["target_fpr"])
        ),
        "inputs": {
            "config": {"path": config_path.as_posix(), "sha256": sha256(config_path)},
            "protocol_document": {
                "path": protocol_document.as_posix(),
                "sha256": sha256(protocol_document),
            },
            "g10_calibration": {"path": g10_path.as_posix(), "sha256": sha256(g10_path)},
            "tune_manifest": {"path": tune_manifest.as_posix(), "sha256": sha256(tune_manifest)},
            "tune_predictions": audits,
        },
        "implementation": {"path": Path(__file__).resolve().as_posix(), "sha256": sha256(Path(__file__))},
        "locked_datasets_read": [],
    }
    output = Path(config["outputs"]["frozen_calibration"])
    _atomic_json(output, report, refuse_overwrite=True)
    print(json.dumps({"threshold": selected["threshold"], "tune_tpr_retention": retention}, indent=2))
    return report


def evaluate(config_path: Path) -> dict[str, Any]:
    reject_locked_input_path(config_path)
    config = load_config(config_path)
    _validate_config(config)
    inputs = config["inputs"]
    calibration_path = Path(config["outputs"]["frozen_calibration"])
    reject_locked_input_path(calibration_path)
    frozen = json.loads(calibration_path.read_text(encoding="utf-8"))
    if frozen.get("algorithm") != ALGORITHM:
        raise ValueError("Unexpected G12 calibration algorithm")
    if frozen["inputs"]["config"]["sha256"] != sha256(config_path):
        raise ValueError("G12 config changed after calibration was frozen")
    protocol_document = Path(config["protocol_document"])
    if frozen["inputs"]["protocol_document"]["sha256"] != sha256(protocol_document):
        raise ValueError("G12 protocol document changed after calibration was frozen")
    if frozen["protocol"]["holdout_used_for_fitting"] is not False:
        raise ValueError("G12 calibration does not prove holdout isolation")

    holdout_manifest = Path(inputs["holdout_manifest"])
    holdout_paths = [Path(value) for value in inputs["holdout_predictions"]]
    baseline_path = Path(inputs["baseline_metrics"])
    for path in (*holdout_paths, holdout_manifest, baseline_path):
        reject_locked_input_path(path)
    holdout, holdout_audits = _ensemble_predictions(holdout_paths, holdout_manifest, "holdout")
    holdout["calibrated_probability"] = probabilities_from_logits(
        holdout["ensemble_logit"].to_numpy(dtype=np.float64), float(frozen["temperature"])
    )
    threshold = float(frozen["strict_calibration"]["threshold"])
    settings = config["evaluation"]
    candidate = operating_point(
        holdout,
        threshold,
        bootstrap_samples=int(settings["bootstrap_samples"]),
        seed=int(settings["bootstrap_seed"]),
    )
    baseline_report = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_point = next(
        item["holdout"]
        for item in baseline_report["operating_points"]
        if math.isclose(float(item["target_fpr"]), 0.01, abs_tol=1e-12)
    )
    checks = [
        {"name": "holdout_fpr", "value": candidate["fpr"], "maximum": baseline_point["fpr"], "passed": candidate["fpr"] <= baseline_point["fpr"]},
        {"name": "holdout_tpr", "value": candidate["recall"], "minimum": baseline_point["recall"], "passed": candidate["recall"] >= baseline_point["recall"]},
        {"name": "holdout_f1", "value": candidate["f1"], "minimum": baseline_point["f1"], "passed": candidate["f1"] >= baseline_point["f1"]},
        {"name": "uav_source_macro_tpr", "value": candidate["uav_source_macro"]["macro_correct"], "minimum": baseline_point["uav_source_macro"]["macro_correct"], "passed": candidate["uav_source_macro"]["macro_correct"] >= baseline_point["uav_source_macro"]["macro_correct"]},
        {"name": "background_source_macro_fpr", "value": candidate["background_source_macro"]["macro_error"], "maximum": baseline_point["background_source_macro"]["macro_error"], "passed": candidate["background_source_macro"]["macro_error"] <= baseline_point["background_source_macro"]["macro_error"]},
    ]
    candidate_ranking = ranking_metrics(
        holdout["label"].to_numpy(dtype=np.int64),
        holdout["calibrated_probability"].to_numpy(dtype=np.float64),
    )
    baseline_ranking = baseline_report["ranking_metrics"]["holdout"]
    ranking_checks = [
        {"name": name, "value": candidate_ranking[name], "minimum": baseline_ranking[name], "passed": candidate_ranking[name] >= baseline_ranking[name]}
        for name in ("roc_auc", "pr_auc", "partial_auc_fpr_0_05_standardized")
    ]
    passed = all(item["passed"] for item in (*checks, *ranking_checks))
    decision = "strict_candidate_ready_for_new_external_confirmation" if passed else "retain_g7"
    report = {
        "algorithm": ALGORITHM,
        "decision": decision,
        "balanced_mode_policy": "retain_g7",
        "strict": {"threshold": threshold, "candidate_holdout": candidate, "baseline_holdout": baseline_point, "checks": checks},
        "ranking": {"candidate_holdout": candidate_ranking, "baseline_holdout": baseline_ranking, "checks": ranking_checks},
        "inputs": {
            "calibration": {"path": calibration_path.as_posix(), "sha256": sha256(calibration_path)},
            "holdout_manifest": {"path": holdout_manifest.as_posix(), "sha256": sha256(holdout_manifest)},
            "holdout_predictions": holdout_audits,
            "baseline_metrics": {"path": baseline_path.as_posix(), "sha256": sha256(baseline_path)},
        },
        "locked_datasets_read": [],
    }
    output_dir = Path(config["outputs"]["evaluation_dir"])
    if (output_dir / "metrics.json").exists():
        raise FileExistsError(f"Frozen G12 evaluation already exists: {output_dir}")
    ensure_dirs(output_dir)
    holdout.to_csv(output_dir / "holdout_predictions.csv", index=False)
    _atomic_json(output_dir / "metrics.json", report, refuse_overwrite=True)
    print(json.dumps({"decision": decision, "strict_checks": checks, "ranking_checks": ranking_checks}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="G12 strict source-conformal calibration")
    parser.add_argument("mode", choices=("fit", "evaluate"))
    parser.add_argument("--config", type=Path, default=Path("configs/g12_strict_source_conformal.yaml"))
    args = parser.parse_args()
    fit(args.config) if args.mode == "fit" else evaluate(args.config)


if __name__ == "__main__":
    main()
