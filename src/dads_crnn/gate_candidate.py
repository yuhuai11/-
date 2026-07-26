from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .calibrate_ood import (
    fit_temperature,
    probabilities_from_logits,
    search_threshold,
    verify_manifest_audio_hashes,
)
from .config import ensure_dirs, load_config
from .evaluate_external import subgroup_metrics
from .metrics import binary_metrics


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {source}")
    return value


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_metrics_metadata(result: dict[str, Any], spec: dict[str, Any], seed: int) -> None:
    if int(result.get("seed", -1)) != seed:
        raise ValueError(f"Metrics seed mismatch for {spec['name']}")
    if str(result.get("model_type")) != str(spec["model_type"]):
        raise ValueError(f"Metrics model_type mismatch for {spec['name']}")
    if str(result.get("feature_type")) != str(spec["feature_type"]):
        raise ValueError(f"Metrics feature_type mismatch for {spec['name']}")
    if "temporal_pooling" in spec:
        observed = result.get("temporal_pooling", "mean")
        if str(observed) != str(spec["temporal_pooling"]):
            raise ValueError(f"Metrics temporal_pooling mismatch for {spec['name']}")
    if spec.get("require_training_artifact_hashes"):
        if result.get("checkpoint_sha256") != _sha256(spec["checkpoint"]):
            raise ValueError(f"Metrics checkpoint hash mismatch for {spec['name']}")
        recorded = result.get("prediction_sha256")
        if not isinstance(recorded, dict):
            raise ValueError(f"Metrics prediction hashes missing for {spec['name']}")
        for name in (
            "val_probabilities",
            "val_labels",
            "test_probabilities",
            "test_labels",
        ):
            if recorded.get(name) != _sha256(spec[name]):
                raise ValueError(f"Metrics {name} hash mismatch for {spec['name']}")


def _validate_calibration_metadata(
    result: dict[str, Any],
    spec: dict[str, Any],
    seed: int,
    calibration_targets: dict[str, Any] | None = None,
    calibration_manifests: dict[str, Any] | None = None,
) -> None:
    if result.get("experiment") != spec["name"]:
        raise ValueError(f"Calibration experiment mismatch for {spec['name']}")
    if int(result.get("seed", -1)) != seed:
        raise ValueError(f"Calibration seed mismatch for {spec['name']}")
    if str(result.get("model_type")) != str(spec["model_type"]):
        raise ValueError(f"Calibration model_type mismatch for {spec['name']}")
    if str(result.get("feature_type")) != str(spec["feature_type"]):
        raise ValueError(f"Calibration feature_type mismatch for {spec['name']}")
    if "temporal_pooling" in spec:
        observed = result.get("temporal_pooling", "mean")
        if str(observed) != str(spec["temporal_pooling"]):
            raise ValueError(f"Calibration temporal_pooling mismatch for {spec['name']}")
    if Path(str(result.get("checkpoint"))).as_posix() != Path(spec["checkpoint"]).as_posix():
        raise ValueError(f"Calibration checkpoint mismatch for {spec['name']}")
    if spec.get("require_checkpoint_sha256"):
        if result.get("checkpoint_sha256") != _sha256(spec["checkpoint"]):
            raise ValueError(f"Calibration checkpoint hash mismatch for {spec['name']}")
    if calibration_targets is not None:
        for metric in ("recall", "specificity"):
            key = f"target_{metric}"
            if abs(_finite(result.get(key)) - _finite(calibration_targets[metric])) > 1e-12:
                raise ValueError(f"Calibration {key} mismatch for {spec['name']}")
    if spec.get("require_manifest_sha256"):
        if calibration_manifests is None:
            raise ValueError("Calibration manifests are required but not configured")
        for split in ("tune", "holdout"):
            path = Path(calibration_manifests[split])
            if Path(str(result.get(f"{split}_manifest"))).as_posix() != path.as_posix():
                raise ValueError(f"Calibration {split} manifest mismatch for {spec['name']}")
            if result.get(f"{split}_manifest_sha256") != _sha256(path):
                raise ValueError(
                    f"Calibration {split} manifest hash mismatch for {spec['name']}"
                )
    if spec.get("require_audio_hash_audit"):
        if calibration_manifests is None:
            raise ValueError("Audio hash audit manifests are not configured")
        recorded = result.get("audio_hash_audit")
        if not isinstance(recorded, dict):
            raise ValueError(f"Calibration audio hash audit missing for {spec['name']}")
        for split in ("tune", "holdout"):
            current = verify_manifest_audio_hashes(Path(calibration_manifests[split]))
            if recorded.get(split) != current:
                raise ValueError(
                    f"Calibration audio hash audit mismatch for {spec['name']} {split}"
                )


def _validate_audit(config: dict[str, Any]) -> dict[str, Any]:
    audit_cfg = config["audit"]
    audit = _load_json(audit_cfg["path"])
    if not audit.get("passed") or audit.get("mode") != "verify":
        raise ValueError("Candidate delta audit is absent, failed, or not in verify mode")
    if audit.get("baseline") != audit_cfg["baseline_config"]:
        raise ValueError("Delta audit baseline config mismatch")
    if audit.get("candidate") != audit_cfg["candidate_config"]:
        raise ValueError("Delta audit candidate config mismatch")
    expected = {
        "baseline_config_sha256": _sha256(audit_cfg["baseline_config"]),
        "candidate_config_sha256": _sha256(audit_cfg["candidate_config"]),
        "guardrails_sha256": _sha256(config["_config_path"]),
    }
    for key, value in expected.items():
        if audit.get(key) != value:
            raise ValueError(f"Delta audit hash mismatch: {key}")
    protected_inventory_summary: dict[str, Any] = {}
    if audit_cfg.get("require_protected_inventory_hashes"):
        recorded_inventory = audit.get("protected_inventory")
        if not isinstance(recorded_inventory, dict) or not recorded_inventory:
            raise ValueError("Delta audit lacks protected inventory")
        inventory_path = audit_cfg.get("baseline_inventory")
        if not inventory_path:
            raise ValueError("Protected baseline inventory path is not configured")
        saved_inventory = _load_json(inventory_path)
        if saved_inventory != recorded_inventory:
            raise ValueError("Delta audit and baseline inventory disagree")
        for name, entry in recorded_inventory.items():
            if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
                raise ValueError(f"Malformed protected inventory entry: {name}")
            if _sha256(entry["path"]) != entry["sha256"]:
                raise ValueError(f"Protected artifact hash mismatch: {name}")
        protected_inventory_summary = {
            "path": Path(inventory_path).as_posix(),
            "entries": len(recorded_inventory),
            "sha256": _sha256(inventory_path),
        }
    verified_artifacts: dict[str, Any] = {}
    required_artifacts = audit_cfg.get("require_candidate_artifact_hashes", [])
    if required_artifacts is True:
        required_artifacts = ["checkpoint", "metrics"]
    if required_artifacts:
        if not isinstance(required_artifacts, list):
            raise ValueError("require_candidate_artifact_hashes must be a list")
        recorded = audit.get("candidate_artifacts")
        if not isinstance(recorded, dict):
            raise ValueError("Delta audit lacks candidate artifact hashes")
        for name in required_artifacts:
            entry = recorded.get(name)
            expected_path = Path(config["candidate"][name]).as_posix()
            if not isinstance(entry, dict) or Path(str(entry.get("path"))).as_posix() != expected_path:
                raise ValueError(f"Delta audit candidate {name} path mismatch")
            current_hash = _sha256(expected_path)
            if entry.get("sha256") != current_hash:
                raise ValueError(f"Delta audit candidate {name} hash mismatch")
            verified_artifacts[name] = {"path": expected_path, "sha256": current_hash}
    return {
        "path": audit_cfg["path"],
        **expected,
        "protected_inventory": protected_inventory_summary,
        "candidate_artifacts": verified_artifacts,
    }


def _validate_prediction_identity(config: dict[str, Any]) -> dict[str, Any]:
    identity_columns = [
        "path",
        "sha256",
        "label",
        "source_group",
        "uav_source",
        "background_source",
        "condition",
        "ood_split",
    ]
    result = {}
    for split in ("tune", "holdout"):
        baseline = pd.read_csv(config["baseline"]["predictions"][split])
        candidate = pd.read_csv(config["candidate"]["predictions"][split])
        if not all(column in baseline.columns and column in candidate.columns for column in identity_columns):
            raise ValueError(f"Missing prediction identity column for {split}")
        left = baseline[identity_columns].reset_index(drop=True)
        right = candidate[identity_columns].reset_index(drop=True)
        if not left.equals(right):
            raise ValueError(f"Prediction sample identity/order mismatch for {split}")
        identity_hash = hashlib.sha256(
            left.to_csv(index=False, lineterminator="\n").encode("utf-8")
        ).hexdigest()
        result[split] = {"samples": len(left), "identity_sha256": identity_hash}
    return result


def _compare_metric_rows(
    actual: dict[str, Any], expected: dict[str, Any], *, context: str
) -> None:
    for metric in ("f1", "balanced_accuracy", "recall", "specificity", "auc"):
        if abs(_finite(actual[metric]) - _finite(expected[metric])) > 1e-10:
            raise ValueError(f"Prediction/calibration {metric} mismatch for {context}")


def _compare_subgroups(
    actual: list[dict[str, Any]], expected: list[dict[str, Any]], *, context: str
) -> None:
    key = lambda row: (str(row["group_field"]), str(row["group"]))
    actual_by_key = {key(row): row for row in actual}
    expected_by_key = {key(row): row for row in expected}
    if actual_by_key.keys() != expected_by_key.keys():
        raise ValueError(f"Prediction/calibration subgroup identity mismatch for {context}")
    for subgroup_key, actual_row in actual_by_key.items():
        expected_row = expected_by_key[subgroup_key]
        for field in ("samples", "positives", "negatives", "false_positives", "false_negatives"):
            if int(actual_row[field]) != int(expected_row[field]):
                raise ValueError(
                    f"Prediction/calibration subgroup {field} mismatch for {context} {subgroup_key}"
                )
        for field in ("recall", "specificity"):
            left = actual_row[field]
            right = expected_row[field]
            if left is None or right is None:
                if left is not None or right is not None:
                    raise ValueError(
                        f"Prediction/calibration subgroup {field} mismatch for {context} {subgroup_key}"
                    )
            elif abs(_finite(left) - _finite(right)) > 1e-10:
                raise ValueError(
                    f"Prediction/calibration subgroup {field} mismatch for {context} {subgroup_key}"
                )


def _validate_prediction_content(
    config: dict[str, Any], calibrations: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for role in ("baseline", "candidate"):
        calibration = calibrations[role]
        threshold = _finite(calibration["selected_threshold"])
        temperature = _finite(calibration["temperature"])
        if temperature <= 0.0:
            raise ValueError(f"Non-positive calibration temperature for {role}")
        role_summary: dict[str, Any] = {}
        prediction_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for split in ("tune", "holdout"):
            frame = pd.read_csv(config[role]["predictions"][split])
            required = {
                "label",
                "source_group",
                "condition",
                "logit",
                "raw_probability",
                "calibrated_probability",
                "selected_prediction",
            }
            if not required.issubset(frame.columns):
                raise ValueError(f"Missing calibrated prediction column for {role} {split}")
            labels = frame["label"].to_numpy(dtype=np.int64)
            logits = frame["logit"].to_numpy(dtype=np.float64)
            raw_probabilities = frame["raw_probability"].to_numpy(dtype=np.float64)
            probabilities = frame["calibrated_probability"].to_numpy(dtype=np.float64)
            selected = frame["selected_prediction"].to_numpy(dtype=np.int64)
            if (
                not np.isfinite(logits).all()
                or not np.isfinite(raw_probabilities).all()
                or not np.isfinite(probabilities).all()
                or not ((raw_probabilities >= 0.0) & (raw_probabilities <= 1.0)).all()
                or not ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
                or not np.isin(labels, [0, 1]).all()
                or not np.isin(selected, [0, 1]).all()
            ):
                raise ValueError(f"Invalid calibrated predictions for {role} {split}")
            recomputed_raw = probabilities_from_logits(logits)
            recomputed_calibrated = probabilities_from_logits(logits, temperature)
            strict_probability_chain = bool(
                config[role].get("require_prediction_sha256")
            )
            probability_rtol = 1e-10 if strict_probability_chain else 1e-6
            probability_atol = 1e-12 if strict_probability_chain else 2e-8
            if not np.allclose(
                raw_probabilities,
                recomputed_raw,
                rtol=probability_rtol,
                atol=probability_atol,
            ):
                raise ValueError(f"Raw probability/logit mismatch for {role} {split}")
            if not np.allclose(
                probabilities,
                recomputed_calibrated,
                rtol=probability_rtol,
                atol=probability_atol,
            ):
                raise ValueError(
                    f"Calibrated probability/logit mismatch for {role} {split}"
                )
            if config[role].get("require_prediction_sha256"):
                prediction_hashes = calibration.get("prediction_sha256")
                if not isinstance(prediction_hashes, dict) or prediction_hashes.get(
                    split
                ) != _sha256(config[role]["predictions"][split]):
                    raise ValueError(f"Prediction hash mismatch for {role} {split}")
            expected_selected = (probabilities >= threshold).astype(np.int64)
            disagreements = selected != expected_selected
            boundary_tolerance = 1e-12
            if np.any(disagreements & (np.abs(probabilities - threshold) > boundary_tolerance)):
                raise ValueError(f"Selected prediction mismatch for {role} {split}")
            actual_metrics = _fast_metrics_from_predictions(
                labels, selected.astype(bool), probabilities
            )
            expected_metrics = calibration[f"{split}_metrics"]["temperature_selected"]
            _compare_metric_rows(actual_metrics, expected_metrics, context=f"{role} {split}")
            if split == "holdout":
                subgroup_probabilities = probabilities.copy()
                below_selected = disagreements & (selected == 1)
                above_rejected = disagreements & (selected == 0)
                subgroup_probabilities[below_selected] = threshold
                subgroup_probabilities[above_rejected] = np.nextafter(
                    threshold, -np.inf
                )
                actual_subgroups = subgroup_metrics(
                    frame, subgroup_probabilities, threshold
                )
                _compare_subgroups(
                    actual_subgroups,
                    calibration["holdout_subgroups"],
                    context=role,
                )
            role_summary[split] = {
                "samples": int(len(frame)),
                "threshold": threshold,
                "metrics_match": True,
            }
            prediction_data[split] = (labels, logits, probabilities)
        tune_labels, tune_logits, tune_probabilities = prediction_data["tune"]
        fitted_temperature = fit_temperature(tune_labels, tune_logits)
        if not math.isclose(
            fitted_temperature, temperature, rel_tol=1e-7, abs_tol=1e-10
        ):
            raise ValueError(f"Refitted temperature mismatch for {role}")
        refitted_threshold, feasible, _ = search_threshold(
            tune_labels,
            tune_probabilities,
            target_recall=_finite(calibration["target_recall"]),
            target_specificity=_finite(calibration["target_specificity"]),
        )
        if not math.isclose(refitted_threshold, threshold, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Refitted threshold mismatch for {role}")
        if bool(feasible) != bool(calibration["constraints_feasible_on_tune"]):
            raise ValueError(f"Tune feasibility mismatch for {role}")
        role_summary["calibration_chain"] = {
            "temperature": temperature,
            "refitted_temperature": fitted_temperature,
            "selected_threshold": threshold,
            "refitted_threshold": refitted_threshold,
            "constraints_feasible": bool(feasible),
        }
        summary[role] = role_summary
    return summary


def _fast_metrics_from_predictions(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    if weights is None:
        weights = np.ones(labels.shape[0], dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != labels.shape or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid metric weights")
    positive = labels == 1
    negative = ~positive
    tp = float(np.sum(weights[predictions & positive]))
    tn = float(np.sum(weights[~predictions & negative]))
    fp = float(np.sum(weights[predictions & negative]))
    fn = float(np.sum(weights[~predictions & positive]))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": float(f1),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "recall": float(recall),
        "specificity": float(specificity),
        "auc": float(roc_auc_score(labels, probabilities, sample_weight=weights)),
    }


def paired_multiway_bootstrap(config: dict[str, Any]) -> dict[str, Any]:
    settings = config["guardrails"]["external_promotion"].get(
        "paired_multiway_bootstrap"
    )
    if not settings:
        return {"passed": True, "checks": [], "improvement_checks": []}
    baseline = pd.read_csv(config["baseline"]["predictions"]["holdout"])
    candidate = pd.read_csv(config["candidate"]["predictions"]["holdout"])
    labels = baseline["label"].to_numpy(dtype=np.int64)
    if not np.array_equal(labels, candidate["label"].to_numpy(dtype=np.int64)):
        raise ValueError("Paired bootstrap label mismatch")
    cluster_fields = [str(field) for field in settings["cluster_fields"]]
    if not cluster_fields:
        raise ValueError("Paired multiway bootstrap requires cluster fields")
    cluster_values: dict[str, np.ndarray] = {}
    for field in cluster_fields:
        if field not in baseline.columns or field not in candidate.columns:
            raise ValueError(f"Missing paired bootstrap cluster field: {field}")
        left = baseline[field].fillna("").astype(str).to_numpy()
        right = candidate[field].fillna("").astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(f"Paired bootstrap cluster identity mismatch: {field}")
        cluster_values[field] = left
    baseline_probabilities = baseline["calibrated_probability"].to_numpy(dtype=np.float64)
    candidate_probabilities = candidate["calibrated_probability"].to_numpy(dtype=np.float64)
    baseline_predictions = baseline["selected_prediction"].to_numpy(dtype=np.int64).astype(bool)
    candidate_predictions = candidate["selected_prediction"].to_numpy(dtype=np.int64).astype(bool)
    if str(settings.get("method")) != "multiway_bayesian":
        raise ValueError("Only multiway_bayesian bootstrap is supported")
    levels = {
        field: np.unique(values[values != ""])
        for field, values in cluster_values.items()
    }
    if any(len(field_levels) == 0 for field_levels in levels.values()):
        raise ValueError("A paired bootstrap cluster field has no observed levels")
    metric_names = list(settings["noninferiority_margins"])
    deltas = {metric: [] for metric in metric_names}
    rng = np.random.default_rng(int(settings["seed"]))
    samples = int(settings["samples"])
    if samples <= 0:
        raise ValueError("Paired bootstrap samples must be positive")
    for _ in range(samples):
        weights = np.ones(labels.shape[0], dtype=np.float64)
        for field in cluster_fields:
            field_values = cluster_values[field]
            field_levels = levels[field]
            multipliers = rng.exponential(scale=1.0, size=len(field_levels))
            index = {level: position for position, level in enumerate(field_levels)}
            observed = field_values != ""
            factors = np.ones(labels.shape[0], dtype=np.float64)
            factors[observed] = np.asarray(
                [multipliers[index[value]] for value in field_values[observed]],
                dtype=np.float64,
            )
            weights *= factors
        for label in (0, 1):
            mask = labels == label
            total = float(weights[mask].sum())
            if total <= 0.0:
                raise ValueError(f"Degenerate bootstrap weights for label {label}")
            weights[mask] *= float(mask.sum()) / total
        baseline_metrics = _fast_metrics_from_predictions(
            labels,
            baseline_predictions,
            baseline_probabilities,
            weights,
        )
        candidate_metrics = _fast_metrics_from_predictions(
            labels,
            candidate_predictions,
            candidate_probabilities,
            weights,
        )
        for metric in metric_names:
            deltas[metric].append(candidate_metrics[metric] - baseline_metrics[metric])
    confidence = _finite(settings["confidence"])
    if not 0.5 < confidence < 1.0:
        raise ValueError("Paired bootstrap confidence must be between 0.5 and 1")
    checks = []
    for metric, margin_value in settings["noninferiority_margins"].items():
        values = np.asarray(deltas[metric], dtype=np.float64)
        lower = float(np.quantile(values, 1.0 - confidence))
        upper = float(np.quantile(values, confidence))
        margin = _finite(margin_value)
        checks.append(
            {
                "metric": metric,
                "lower": lower,
                "upper": upper,
                "noninferiority_margin": -margin,
                "passed": lower >= -margin,
            }
        )
    required_improvement = _finite(settings["improvement_lower_bound"])
    improvement_checks = []
    for metric in settings["improvement_metrics"]:
        values = np.asarray(deltas[str(metric)], dtype=np.float64)
        lower = float(np.quantile(values, 1.0 - confidence))
        improvement_checks.append(
            {
                "metric": str(metric),
                "lower": lower,
                "required_lower_bound": required_improvement,
                "passed": lower > required_improvement,
            }
        )
    return {
        "passed": all(check["passed"] for check in checks)
        and any(check["passed"] for check in improvement_checks),
        "method": "multiway_bayesian",
        "cluster_fields": cluster_fields,
        "cluster_levels": {field: int(len(values)) for field, values in levels.items()},
        "samples": samples,
        "confidence": confidence,
        "checks": checks,
        "improvement_checks": improvement_checks,
    }


def supported_meaningful_improvement_metrics(
    external: dict[str, Any], bootstrap: dict[str, Any]
) -> list[str]:
    point_supported = {
        str(check["metric"])
        for check in external.get("minimum_improvement_any", [])
        if check.get("passed")
    }
    statistically_supported = {
        str(check["metric"])
        for check in bootstrap.get("improvement_checks", [])
        if check.get("passed")
    }
    return sorted(point_supported & statistically_supported)


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite metric: {value!r}")
    return number


def _threshold_row(result: dict[str, Any], split: str, threshold: float) -> dict[str, Any]:
    key = "val_threshold_metrics" if split == "val" else "threshold_metrics"
    rows = result.get(key, [])
    matches = [row for row in rows if abs(_finite(row["threshold"]) - threshold) < 1e-9]
    if len(matches) != 1:
        raise ValueError(f"Expected one {split} row at threshold {threshold}, got {len(matches)}")
    return matches[0]


def assess_internal(
    baseline: dict[str, Any], candidate: dict[str, Any], guardrails: dict[str, Any]
) -> dict[str, Any]:
    threshold = _finite(guardrails["threshold"])
    checks = []
    for rule in guardrails["internal_screen"]:
        split = str(rule["split"])
        metric = str(rule["metric"])
        baseline_value = _finite(_threshold_row(baseline, split, threshold)[metric])
        candidate_value = _finite(_threshold_row(candidate, split, threshold)[metric])
        required = baseline_value - _finite(rule["max_drop"]) if "max_drop" in rule else _finite(rule["floor"])
        checks.append(
            {
                "split": split,
                "metric": metric,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": candidate_value - baseline_value,
                "required_minimum": required,
                "passed": candidate_value >= required,
            }
        )
    return {"passed": all(check["passed"] for check in checks), "threshold": threshold, "checks": checks}


def _internal_split(
    config: dict[str, Any], spec: dict[str, Any], split: str
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows = pd.read_csv(config["dads_manifest"])
    rows = rows.loc[rows["split"] == split].reset_index(drop=True)
    labels = np.load(spec[f"{split}_labels"]).astype(np.int64)
    probabilities = np.load(spec[f"{split}_probabilities"]).astype(np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1 or labels.shape != probabilities.shape:
        raise ValueError(f"Invalid {split} prediction array shape for {spec['name']}")
    if len(rows) != labels.size:
        raise ValueError(f"Manifest/prediction length mismatch for {spec['name']} {split}")
    manifest_labels = rows["label"].to_numpy(dtype=np.int64)
    if not np.array_equal(labels, manifest_labels):
        raise ValueError(f"Manifest/prediction labels mismatch for {spec['name']} {split}")
    if not np.isfinite(probabilities).all() or not (
        ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    ):
        raise ValueError(f"Non-finite probabilities for {spec['name']} {split}")
    return rows, labels, probabilities


def _validate_internal_prediction_content(
    config: dict[str, Any], metrics_by_role: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if "dads_manifest" not in config:
        return {}
    summary: dict[str, Any] = {}
    for role in ("baseline", "candidate"):
        result = metrics_by_role[role]
        role_summary: dict[str, Any] = {}
        for split in ("val", "test"):
            _, labels, probabilities = _internal_split(config, config[role], split)
            key = "val_threshold_metrics" if split == "val" else "threshold_metrics"
            expected_rows = result.get(key, [])
            if not expected_rows:
                raise ValueError(f"Missing {split} threshold metrics for {config[role]['name']}")
            thresholds = []
            for expected in expected_rows:
                threshold = _finite(expected["threshold"])
                thresholds.append(threshold)
                actual = binary_metrics(labels, probabilities, threshold)
                _compare_metric_rows(
                    actual,
                    expected,
                    context=f"{config[role]['name']} internal {split} threshold={threshold}",
                )
                for count in ("tn", "fp", "fn", "tp"):
                    if int(actual[count]) != int(expected[count]):
                        raise ValueError(
                            f"Prediction/metrics {count} mismatch for "
                            f"{config[role]['name']} {split} threshold={threshold}"
                        )
            if len(set(thresholds)) != len(thresholds):
                raise ValueError(f"Duplicate threshold metrics for {config[role]['name']} {split}")
            role_summary[split] = {
                "samples": int(labels.size),
                "thresholds": thresholds,
                "metrics_match": True,
            }
        summary[role] = role_summary
    return summary


def _conditional_rate(
    rows: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    group_field: str,
    group: str,
    metric: str,
    threshold: float,
) -> tuple[float, int]:
    if group_field not in rows.columns:
        raise ValueError(f"Missing internal subgroup column: {group_field}")
    group_mask = rows[group_field].astype(str).to_numpy() == group
    class_label = 1 if metric == "recall" else 0 if metric == "specificity" else None
    if class_label is None:
        raise ValueError(f"Unsupported internal subgroup metric: {metric}")
    mask = group_mask & (labels == class_label)
    if not mask.any():
        raise ValueError(f"Empty internal subgroup: {group_field}={group}, metric={metric}")
    predictions = probabilities[mask] >= threshold
    value = predictions.mean() if class_label == 1 else (~predictions).mean()
    return float(value), int(mask.sum())


def assess_internal_subgroups(config: dict[str, Any]) -> dict[str, Any]:
    rules = config["guardrails"].get("internal_subgroup_checks", [])
    if not rules:
        return {"passed": True, "checks": []}
    threshold = _finite(config["guardrails"]["threshold"])
    cache: dict[tuple[str, str], tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    checks = []
    for rule in rules:
        split = str(rule["split"])
        field = str(rule["group_field"])
        group = str(rule["group"])
        metric = str(rule["metric"])
        for role in ("baseline", "candidate"):
            key = (role, split)
            if key not in cache:
                cache[key] = _internal_split(config, config[role], split)
        baseline_value, baseline_samples = _conditional_rate(
            *cache[("baseline", split)],
            group_field=field,
            group=group,
            metric=metric,
            threshold=threshold,
        )
        candidate_value, candidate_samples = _conditional_rate(
            *cache[("candidate", split)],
            group_field=field,
            group=group,
            metric=metric,
            threshold=threshold,
        )
        if baseline_samples != candidate_samples:
            raise ValueError(f"Internal subgroup sample mismatch for {split} {field}={group}")
        required = baseline_value - _finite(rule["max_drop"])
        checks.append(
            {
                "split": split,
                "group_field": field,
                "group": group,
                "metric": metric,
                "samples": baseline_samples,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": candidate_value - baseline_value,
                "required_minimum": required,
                "passed": candidate_value >= required,
            }
        )
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def _subgroup(result: dict[str, Any], field: str, group: str) -> dict[str, Any]:
    matches = [
        row
        for row in result.get("holdout_subgroups", [])
        if str(row.get("group_field")) == field and str(row.get("group")) == group
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one subgroup {field}={group}, got {len(matches)}")
    return matches[0]


def _macro_subgroup_metric(result: dict[str, Any], field: str, metric: str) -> tuple[float, int]:
    values = []
    for row in result.get("holdout_subgroups", []):
        if str(row.get("group_field")) != field or row.get(metric) is None:
            continue
        values.append(_finite(row[metric]))
    if not values:
        raise ValueError(f"No finite subgroup values for {field}.{metric}")
    return float(sum(values) / len(values)), len(values)


def assess_external(
    baseline: dict[str, Any], candidate: dict[str, Any], guardrails: dict[str, Any]
) -> dict[str, Any]:
    settings = guardrails["external_promotion"]
    scenario = str(settings["scenario"])
    baseline_metrics = baseline["holdout_metrics"][scenario]
    candidate_metrics = candidate["holdout_metrics"][scenario]
    checks = []
    for rule in settings["checks"]:
        metric = str(rule["metric"])
        baseline_value = _finite(baseline_metrics[metric])
        candidate_value = _finite(candidate_metrics[metric])
        required = baseline_value - _finite(rule["max_drop"])
        checks.append(
            {
                "metric": metric,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": candidate_value - baseline_value,
                "required_minimum": required,
                "passed": candidate_value >= required,
            }
        )

    improvements = []
    for rule in settings["minimum_improvement"]["any"]:
        metric = str(rule["metric"])
        baseline_value = _finite(baseline_metrics[metric])
        candidate_value = _finite(candidate_metrics[metric])
        required_delta = _finite(rule["delta"])
        improvements.append(
            {
                "metric": metric,
                "delta": candidate_value - baseline_value,
                "required_delta": required_delta,
                "passed": candidate_value - baseline_value >= required_delta,
            }
        )

    subgroup_checks = []
    for rule in settings.get("subgroup_checks", []):
        field = str(rule["group_field"])
        group = str(rule["group"])
        metric = str(rule["metric"])
        baseline_value = _finite(_subgroup(baseline, field, group)[metric])
        candidate_value = _finite(_subgroup(candidate, field, group)[metric])
        if "min_delta" in rule:
            required_delta = _finite(rule["min_delta"])
        elif "max_drop" in rule:
            required_delta = -_finite(rule["max_drop"])
        else:
            raise ValueError("Subgroup rule requires min_delta or max_drop")
        subgroup_checks.append(
            {
                "group_field": field,
                "group": group,
                "metric": metric,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": candidate_value - baseline_value,
                "required_delta": required_delta,
                "passed": candidate_value - baseline_value >= required_delta,
            }
        )
    aggregate_subgroup_checks = []
    for rule in settings.get("aggregate_subgroup_checks", []):
        field = str(rule["group_field"])
        metric = str(rule["metric"])
        aggregation = str(rule.get("aggregation", "macro"))
        if aggregation != "macro":
            raise ValueError(f"Unsupported subgroup aggregation: {aggregation}")
        baseline_value, baseline_groups = _macro_subgroup_metric(baseline, field, metric)
        candidate_value, candidate_groups = _macro_subgroup_metric(candidate, field, metric)
        if baseline_groups != candidate_groups:
            raise ValueError(f"Subgroup count mismatch for {field}.{metric}")
        required_delta = -_finite(rule["max_drop"])
        aggregate_subgroup_checks.append(
            {
                "group_field": field,
                "metric": metric,
                "aggregation": aggregation,
                "groups": baseline_groups,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": candidate_value - baseline_value,
                "required_delta": required_delta,
                "passed": candidate_value - baseline_value >= required_delta,
            }
        )
    pareto_passed = all(check["passed"] for check in checks)
    improvement_passed = any(check["passed"] for check in improvements)
    subgroups_passed = all(check["passed"] for check in subgroup_checks) and all(
        check["passed"] for check in aggregate_subgroup_checks
    )
    targets = guardrails["deployment_targets"]
    deployment_ready = bool(
        candidate.get("constraints_feasible_on_tune", False)
        and _finite(candidate_metrics["recall"]) >= _finite(targets["recall"])
        and _finite(candidate_metrics["specificity"]) >= _finite(targets["specificity"])
    )
    deployment_required = bool(settings.get("require_deployment_ready", False))
    return {
        "passed": (
            pareto_passed
            and improvement_passed
            and subgroups_passed
            and (deployment_ready or not deployment_required)
        ),
        "scenario": scenario,
        "pareto_noninferiority_passed": pareto_passed,
        "minimum_improvement_passed": improvement_passed,
        "subgroup_checks_passed": subgroups_passed,
        "checks": checks,
        "minimum_improvement_any": improvements,
        "subgroup_checks": subgroup_checks,
        "aggregate_subgroup_checks": aggregate_subgroup_checks,
        "deployment_ready": deployment_ready,
        "deployment_required_for_promotion": deployment_required,
    }


def _markdown(report: dict[str, Any]) -> str:
    baseline = report["baseline"]
    candidate = report["candidate"]
    lines = [
        f"# {candidate}候选晋级判定",
        "",
        f"- 阶段：`{report['stage']}`",
        f"- 判定：`{report['decision']}`",
        f"- 总体通过：`{report['passed']}`",
        "",
        "## DADS内部保护",
        "",
        f"| Split | 指标 | {baseline} | {candidate} | 差值 | 最低要求 | 通过 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for check in report.get("internal", {}).get("checks", []):
        lines.append(
            f"| {check['split']} | {check['metric']} | {check['baseline']:.5f} | "
            f"{check['candidate']:.5f} | {check['delta']:+.5f} | "
            f"{check['required_minimum']:.5f} | {check['passed']} |"
        )
    subgroup_checks = report.get("internal", {}).get("subgroup_checks", [])
    if subgroup_checks:
        lines += [
            "",
            "## DADS片段类型保护",
            "",
            f"| Split | 分组 | 指标 | 样本 | {baseline} | {candidate} | 差值 | 通过 |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for check in subgroup_checks:
            lines.append(
                f"| {check['split']} | {check['group_field']}={check['group']} | "
                f"{check['metric']} | {check['samples']} | {check['baseline']:.5f} | "
                f"{check['candidate']:.5f} | {check['delta']:+.5f} | "
                f"{check['passed']} |"
            )
    if "external" in report:
        lines += [
            "",
            "## val_ood Holdout严格晋级",
            "",
            f"| 指标 | {baseline} | {candidate} | 差值 | 通过 |",
            "|---|---:|---:|---:|---:|",
        ]
        for check in report["external"]["checks"]:
            lines.append(
                f"| {check['metric']} | {check['baseline']:.5f} | {check['candidate']:.5f} | "
                f"{check['delta']:+.5f} | {check['passed']} |"
            )
        lines += [
            "",
            "### 点估计实质提升",
            "",
            "| 指标 | 差值 | 所需差值 | 通过 |",
            "|---|---:|---:|---:|",
        ]
        for check in report["external"].get("minimum_improvement_any", []):
            lines.append(
                f"| {check['metric']} | {check['delta']:+.5f} | "
                f"{check['required_delta']:+.5f} | {check['passed']} |"
            )
        external_subgroups = report["external"].get("subgroup_checks", [])
        if external_subgroups:
            lines += [
                "",
                "### 条件级保护",
                "",
                "| 分组 | 指标 | 差值 | 最低差值 | 通过 |",
                "|---|---|---:|---:|---:|",
            ]
            for check in external_subgroups:
                lines.append(
                    f"| {check['group_field']}={check['group']} | {check['metric']} | "
                    f"{check['delta']:+.5f} | {check['required_delta']:+.5f} | "
                    f"{check['passed']} |"
                )
        aggregate_checks = report["external"].get("aggregate_subgroup_checks", [])
        if aggregate_checks:
            lines += [
                "",
                "### 来源宏平均保护",
                "",
                "| 分组字段 | 指标 | 组数 | 差值 | 最低差值 | 通过 |",
                "|---|---|---:|---:|---:|---:|",
            ]
            for check in aggregate_checks:
                lines.append(
                    f"| {check['group_field']} | {check['metric']} | {check['groups']} | "
                    f"{check['delta']:+.5f} | {check['required_delta']:+.5f} | "
                    f"{check['passed']} |"
                )
        bootstrap = report["external"].get("paired_multiway_bootstrap")
        if bootstrap:
            lines += [
                "",
                "## UAV与背景来源双向配对Bootstrap",
                "",
                "| 指标 | 单侧下界 | 非劣界 | 通过 |",
                "|---|---:|---:|---:|",
            ]
            for check in bootstrap["checks"]:
                lines.append(
                    f"| {check['metric']} | {check['lower']:+.5f} | "
                    f"{check['noninferiority_margin']:+.5f} | {check['passed']} |"
                )
            lines += [
                "",
                "| 提升指标 | 单侧下界 | 要求 | 通过 |",
                "|---|---:|---:|---:|",
            ]
            for check in bootstrap["improvement_checks"]:
                lines.append(
                    f"| {check['metric']} | {check['lower']:+.5f} | "
                    f"> {check['required_lower_bound']:+.5f} | {check['passed']} |"
                )
            lines += [
                "",
                "点估计与统计支持同时通过的指标："
                f"`{bootstrap.get('supported_meaningful_improvement_metrics', [])}`",
            ]
        lines += [
            "",
            f"部署门槛通过：`{report['external']['deployment_ready']}`",
            "部署门槛是否为晋级硬条件："
            f"`{report['external']['deployment_required_for_promotion']}`",
        ]
    if report.get("errors"):
        lines += ["", "## 错误", ""] + [f"- {error}" for error in report["errors"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed champion/challenger experiment gate")
    parser.add_argument("--config", default="configs/g4_low_snr_guardrails.yaml")
    parser.add_argument("--stage", choices=("internal", "final"), required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    config["_config_path"] = args.config
    output = args.output or Path(config["output"])
    report: dict[str, Any] = {
        "stage": args.stage,
        "baseline": config["baseline"]["name"],
        "candidate": config["candidate"]["name"],
        "passed": False,
        "decision": config.get("keep_decision", "keep_baseline"),
        "errors": [],
    }
    try:
        baseline_metrics = _load_json(config["baseline"]["metrics"])
        candidate_metrics = _load_json(config["candidate"]["metrics"])
        seed = int(config.get("screening_seed", 42))
        _validate_metrics_metadata(baseline_metrics, config["baseline"], seed)
        _validate_metrics_metadata(candidate_metrics, config["candidate"], seed)
        report["internal_prediction_content"] = _validate_internal_prediction_content(
            config, {"baseline": baseline_metrics, "candidate": candidate_metrics}
        )
        report["provenance_audit"] = _validate_audit(config)
        report["internal"] = assess_internal(
            baseline_metrics, candidate_metrics, config["guardrails"]
        )
        internal_subgroups = assess_internal_subgroups(config)
        report["internal"]["subgroup_checks"] = internal_subgroups["checks"]
        report["internal"]["subgroup_checks_passed"] = internal_subgroups["passed"]
        report["internal"]["passed"] = bool(
            report["internal"]["passed"] and internal_subgroups["passed"]
        )
        if args.stage == "internal":
            report["passed"] = report["internal"]["passed"]
            report["decision"] = (
                "proceed_to_val_ood"
                if report["passed"]
                else config.get("keep_decision", "keep_baseline")
            )
        else:
            baseline_calibration = _load_json(config["baseline"]["calibration"])
            candidate_calibration = _load_json(config["candidate"]["calibration"])
            calibration_targets = config.get("calibration_targets")
            calibration_manifests = config.get("calibration_manifests")
            _validate_calibration_metadata(
                baseline_calibration,
                config["baseline"],
                seed,
                calibration_targets,
                calibration_manifests,
            )
            _validate_calibration_metadata(
                candidate_calibration,
                config["candidate"],
                seed,
                calibration_targets,
                calibration_manifests,
            )
            report["prediction_identity"] = _validate_prediction_identity(config)
            report["prediction_content"] = _validate_prediction_content(
                config,
                {"baseline": baseline_calibration, "candidate": candidate_calibration},
            )
            report["external"] = assess_external(
                baseline_calibration, candidate_calibration, config["guardrails"]
            )
            bootstrap_settings = config["guardrails"]["external_promotion"].get(
                "paired_multiway_bootstrap"
            )
            if bootstrap_settings:
                bootstrap = paired_multiway_bootstrap(config)
                supported_metrics = supported_meaningful_improvement_metrics(
                    report["external"], bootstrap
                )
                bootstrap["supported_meaningful_improvement_metrics"] = (
                    supported_metrics
                )
                bootstrap["same_metric_improvement_supported"] = bool(
                    supported_metrics
                )
                report["external"]["paired_multiway_bootstrap"] = bootstrap
                report["external"]["passed"] = bool(
                    report["external"]["passed"]
                    and bootstrap["passed"]
                    and supported_metrics
                )
            report["passed"] = report["internal"]["passed"] and report["external"]["passed"]
            report["decision"] = (
                config.get("promote_decision", "promote_candidate")
                if report["passed"]
                else config.get("keep_decision", "keep_baseline")
            )
    except Exception as error:  # Fail closed while still leaving an auditable report.
        report["errors"].append(f"{type(error).__name__}: {error}")
        report["passed"] = False
        report["decision"] = config.get("keep_decision", "keep_baseline")

    ensure_dirs(output.parent)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("Candidate rejected; G2 remains the champion")


if __name__ == "__main__":
    main()
