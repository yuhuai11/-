from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .audit_g7_r1 import ALGORITHM, build_audit
from .calibrate_ood import (
    _predict_logits,
    fit_temperature,
    probabilities_from_logits,
    verify_manifest_audio_hashes,
)
from .config import load_config
from .data_firewall import file_sha256
from .evaluate_g9_guard import (
    negative_metrics,
    predict_guard,
    validate_guard_manifest,
)
from .evaluate_low_fpr import threshold_at_target_fpr
from .metrics import binary_metrics


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _threshold_row(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if np.isclose(float(row["threshold"]), threshold, rtol=0.0, atol=1e-12)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one metric row at threshold={threshold}")
    return matches[0]


def _verify_frozen(root: Path, protocol_path: Path) -> tuple[dict, dict, dict]:
    protocol = load_config(protocol_path)
    if protocol.get("algorithm") != ALGORITHM:
        raise ValueError(f"Unexpected R1 protocol: {protocol.get('algorithm')}")
    outputs = {
        name: _resolve(root, value)
        for name, value in protocol["outputs"].items()
    }
    frozen_path = outputs["frozen_protocol"]
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    observed = build_audit(root, protocol_path)
    if frozen.get("status") != "frozen_before_training":
        raise ValueError("R1 was not frozen before training")
    for name in ("inputs", "implementation_sha256", "config_differences", "promotion_gates"):
        if frozen.get(name) != observed.get(name):
            raise ValueError(f"R1 frozen protocol changed: {name}")
    return protocol, frozen, outputs


def _checkpoint_and_metrics(
    root: Path,
    protocol: dict,
    frozen: dict,
) -> tuple[Path, dict, dict]:
    candidate_config_path = _resolve(
        root, protocol["inputs"]["candidate_config"]
    )
    candidate_config = load_config(candidate_config_path)
    run_dir = _resolve(root, candidate_config["output_dir"]) / "seed_42"
    checkpoint_path = run_dir / "best.pt"
    metrics_path = run_dir / "metrics.json"
    if not checkpoint_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError("R1 seed-42 training artifacts are incomplete")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(checkpoint.get("seed", -1)) != 42:
        raise ValueError("R1 candidate checkpoint seed changed")
    if checkpoint.get("config") != candidate_config:
        raise ValueError("R1 checkpoint does not embed the frozen candidate config")
    if metrics.get("checkpoint_sha256") != file_sha256(checkpoint_path):
        raise ValueError("R1 metrics/checkpoint identity mismatch")
    training_inputs = checkpoint.get("training_inputs", {})
    if int(metrics.get("seed", -1)) != 42:
        raise ValueError("R1 training metrics seed changed")
    if metrics.get("training_inputs") != training_inputs:
        raise ValueError("R1 metrics/checkpoint training-input identity mismatch")
    expected_manifest = frozen["training_manifest"]
    if (
        training_inputs.get("manifest_sha256") != expected_manifest["sha256"]
        or int(training_inputs.get("manifest_rows", -1)) != int(expected_manifest["rows"])
    ):
        raise ValueError("R1 checkpoint is not bound to the frozen DADS manifest")
    if metrics.get("locked_datasets_read") != []:
        raise ValueError("R1 training metrics report locked dataset access")
    return checkpoint_path, checkpoint, metrics


def _verified_internal_metrics(
    root: Path,
    protocol: dict,
    checkpoint_path: Path,
    metrics: dict,
) -> dict[str, dict[str, Any]]:
    manifest_path = _resolve(root, protocol["inputs"]["training_manifest"])
    manifest = pd.read_csv(manifest_path, low_memory=False)
    output: dict[str, dict[str, Any]] = {}
    for split, prefix, metric_key in (
        ("val", "val", "val_threshold_metrics"),
        ("test", "test", "threshold_metrics"),
    ):
        label_path = checkpoint_path.parent / f"{prefix}_labels.npy"
        probability_path = checkpoint_path.parent / f"{prefix}_probabilities.npy"
        expected_hashes = metrics.get("prediction_sha256", {})
        for name, path in (
            (f"{prefix}_labels", label_path),
            (f"{prefix}_probabilities", probability_path),
        ):
            if expected_hashes.get(name) != file_sha256(path):
                raise ValueError(f"R1 stored prediction identity changed: {name}")
        labels = np.load(label_path, allow_pickle=False).astype(np.int64)
        probabilities = np.load(probability_path, allow_pickle=False).astype(
            np.float64
        )
        expected_labels = (
            manifest.loc[manifest["split"].astype(str).eq(split), "label"]
            .to_numpy(dtype=np.int64)
        )
        if not np.array_equal(labels, expected_labels):
            raise ValueError(f"R1 {split} labels do not align with frozen DADS manifest")
        if probabilities.shape != labels.shape or not np.isfinite(
            probabilities
        ).all():
            raise ValueError(f"R1 {split} probabilities are invalid")
        recomputed = binary_metrics(labels, probabilities, 0.5)
        stored = _threshold_row(metrics[metric_key], 0.5)
        for name in (
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "auc",
        ):
            if not np.isclose(
                float(recomputed[name]),
                float(stored[name]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(f"R1 {split} metric drift for {name}")
        output[split] = recomputed
    return output


def _internal_checks(
    internal: dict[str, dict[str, Any]], gates: dict
) -> tuple[dict, list[dict]]:
    threshold = 0.5
    validation = internal["val"]
    test = internal["test"]
    test_gate = gates["dads_test_at_probability_0_5"]
    val_gate = gates["dads_validation_at_probability_0_5"]
    checks = [
        {
            "name": "dads_test_f1",
            "value": float(test["f1"]),
            "minimum": float(test_gate["minimum_f1"]),
            "passed": float(test["f1"]) >= float(test_gate["minimum_f1"]),
        },
        {
            "name": "dads_test_roc_auc",
            "value": float(test["auc"]),
            "minimum": float(test_gate["minimum_roc_auc"]),
            "passed": float(test["auc"]) >= float(test_gate["minimum_roc_auc"]),
        },
        {
            "name": "dads_test_recall",
            "value": float(test["recall"]),
            "minimum": float(test_gate["minimum_recall"]),
            "passed": float(test["recall"]) >= float(test_gate["minimum_recall"]),
        },
        {
            "name": "dads_test_specificity",
            "value": float(test["specificity"]),
            "minimum": float(test_gate["minimum_specificity"]),
            "passed": float(test["specificity"])
            >= float(test_gate["minimum_specificity"]),
        },
        {
            "name": "dads_validation_f1",
            "value": float(validation["f1"]),
            "minimum": float(val_gate["minimum_f1"]),
            "passed": float(validation["f1"]) >= float(val_gate["minimum_f1"]),
        },
        {
            "name": "dads_validation_roc_auc",
            "value": float(validation["auc"]),
            "minimum": float(val_gate["minimum_roc_auc"]),
            "passed": float(validation["auc"])
            >= float(val_gate["minimum_roc_auc"]),
        },
    ]
    return {"validation": validation, "test": test}, checks


def _baseline_val_ood(
    root: Path,
    protocol: dict,
    manifest: pd.DataFrame,
    *,
    temperature: float,
    threshold: float,
) -> dict[str, Any]:
    path = _resolve(root, protocol["inputs"]["baseline_val_ood_holdout_predictions"])
    rows = pd.read_csv(path, low_memory=False)
    if len(rows) != len(manifest):
        raise ValueError("Baseline val_ood holdout prediction row count changed")
    for column in ("sha256", "label", "logit"):
        if column not in rows:
            raise ValueError(f"Baseline val_ood predictions lack {column}")
    if not np.array_equal(
        rows["sha256"].astype(str).to_numpy(),
        manifest["sha256"].astype(str).to_numpy(),
    ):
        raise ValueError("Baseline val_ood predictions do not align by SHA256")
    logits = rows["logit"].to_numpy(dtype=np.float64)
    probabilities = probabilities_from_logits(logits, temperature)
    return binary_metrics(
        manifest["label"].to_numpy(dtype=np.int64), probabilities, threshold
    )


def _verify_guard_cache(rows: pd.DataFrame, root: Path) -> dict[str, Any]:
    if "cache_sha256" not in rows or "cache_path" not in rows:
        raise ValueError("G9 guard lacks cache identity columns")
    identities = rows[["cache_path", "cache_sha256"]].astype(str)
    if identities.groupby("cache_path")["cache_sha256"].nunique().gt(1).any():
        raise ValueError("A G9 guard cache path has multiple declared hashes")
    identities = identities.drop_duplicates().reset_index(drop=True)
    checked = 0
    for row in identities.itertuples(index=False):
        path = _resolve(root, row.cache_path)
        expected = str(row.cache_sha256).strip().lower()
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"G9 guard cache SHA256 mismatch: {path}")
        checked += 1
    return {"files": checked, "verified": True}


def evaluate(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, frozen, outputs = _verify_frozen(root, protocol_path)
    output_path = outputs["guard_evaluation"]
    if output_path.exists():
        raise FileExistsError(f"R1 guard evaluation already exists: {output_path}")
    checkpoint_path, checkpoint, training_metrics = _checkpoint_and_metrics(
        root, protocol, frozen
    )
    gates = protocol["promotion_gates"]
    recomputed_internal = _verified_internal_metrics(
        root, protocol, checkpoint_path, training_metrics
    )
    internal, checks = _internal_checks(recomputed_internal, gates)
    if not all(item["passed"] for item in checks):
        report = {
            "algorithm": ALGORITHM,
            "status": "stopped_after_internal_regression_gate",
            "decision": "reject_r1_seed42_without_external_audio_inference",
            "checkpoint": {
                "path": checkpoint_path.relative_to(root).as_posix(),
                "sha256": file_sha256(checkpoint_path),
            },
            "internal": internal,
            "checks": checks,
            "gate_passed": False,
            "val_ood_audio_read": False,
            "g9_guard_audio_read": False,
            "idmt_calibration_audio_read": False,
            "idmt_development_audio_read": False,
            "idmt_final_holdout_audio_read": False,
            "locked_datasets_read": [],
        }
        _atomic_json(output_path, report)
        return report

    score_protocol = protocol["contract"]["score_protocol"]
    baseline_mapping = score_protocol["baseline_g7_mapping"]
    baseline_temperature = float(baseline_mapping["temperature"])
    baseline_strict_threshold = float(baseline_mapping["strict_threshold"])
    baseline_sensitivity_threshold = float(
        baseline_mapping["sensitivity_threshold"]
    )
    tune_manifest_path = _resolve(
        root, protocol["inputs"]["val_ood_tune_manifest"]
    )
    holdout_manifest_path = _resolve(
        root, protocol["inputs"]["val_ood_holdout_manifest"]
    )
    tune_audio_audit = verify_manifest_audio_hashes(tune_manifest_path)
    holdout_audio_audit = verify_manifest_audio_hashes(holdout_manifest_path)
    tune_manifest = pd.read_csv(tune_manifest_path, low_memory=False)
    holdout_manifest = pd.read_csv(holdout_manifest_path, low_memory=False)
    if set(tune_manifest["sha256"].astype(str)) & set(
        holdout_manifest["sha256"].astype(str)
    ):
        raise ValueError("val_ood Tune/Holdout audio hashes overlap")
    if set(tune_manifest["source_group"].astype(str)) & set(
        holdout_manifest["source_group"].astype(str)
    ):
        raise ValueError("val_ood Tune/Holdout source groups overlap")
    baseline_holdout = _baseline_val_ood(
        root,
        protocol,
        holdout_manifest,
        temperature=baseline_temperature,
        threshold=baseline_strict_threshold,
    )
    candidate_tune_rows, candidate_tune_logits, _, tune_seed = _predict_logits(
        checkpoint_path,
        tune_manifest_path,
        batch_size=128,
        num_workers=0,
        device_name="cuda",
    )
    candidate_rows, candidate_holdout_logits, _, candidate_seed = _predict_logits(
        checkpoint_path,
        holdout_manifest_path,
        batch_size=128,
        num_workers=0,
        device_name="cuda",
    )
    if tune_seed != 42 or candidate_seed != 42:
        raise ValueError("R1 seed changed during val_ood inference")
    if not np.array_equal(
        candidate_tune_rows["sha256"].astype(str).to_numpy(),
        tune_manifest["sha256"].astype(str).to_numpy(),
    ):
        raise ValueError("R1 val_ood Tune predictions do not align by SHA256")
    if not np.array_equal(
        candidate_rows["sha256"].astype(str).to_numpy(),
        holdout_manifest["sha256"].astype(str).to_numpy(),
    ):
        raise ValueError("R1 val_ood predictions do not align by SHA256")
    tune_labels = tune_manifest["label"].to_numpy(dtype=np.int64)
    candidate_temperature = fit_temperature(tune_labels, candidate_tune_logits)
    candidate_tune_probability = probabilities_from_logits(
        candidate_tune_logits, candidate_temperature
    )
    candidate_holdout_probability = probabilities_from_logits(
        candidate_holdout_logits, candidate_temperature
    )
    calibrations = {
        float(target): threshold_at_target_fpr(
            candidate_tune_probability[tune_labels == 0], float(target)
        )
        for target in score_protocol["target_fprs"]
    }
    candidate_strict_threshold = float(calibrations[0.01]["threshold"])
    candidate_sensitivity_threshold = float(calibrations[0.05]["threshold"])
    candidate_holdout = binary_metrics(
        holdout_manifest["label"].to_numpy(dtype=np.int64),
        candidate_holdout_probability,
        candidate_strict_threshold,
    )
    candidate_holdout_sensitivity = binary_metrics(
        holdout_manifest["label"].to_numpy(dtype=np.int64),
        candidate_holdout_probability,
        candidate_sensitivity_threshold,
    )
    candidate_fixed_mapping_probability = probabilities_from_logits(
        candidate_holdout_logits, baseline_temperature
    )
    candidate_fixed_mapping = binary_metrics(
        holdout_manifest["label"].to_numpy(dtype=np.int64),
        candidate_fixed_mapping_probability,
        baseline_strict_threshold,
    )

    guard_manifest_path = _resolve(root, protocol["inputs"]["g9_guard_manifest"])
    guard_audit_path = _resolve(root, protocol["inputs"]["g9_guard_audit"])
    guard_rows, _ = validate_guard_manifest(guard_manifest_path, guard_audit_path)
    guard_cache_audit = _verify_guard_cache(guard_rows, root)
    baseline_guard_rows = pd.read_csv(
        _resolve(root, protocol["inputs"]["baseline_g9_guard_predictions"]),
        low_memory=False,
    )
    if not np.array_equal(
        baseline_guard_rows["sha256"].astype(str).to_numpy(),
        guard_rows["sha256"].astype(str).to_numpy(),
    ):
        raise ValueError("Baseline G9 guard predictions do not align by SHA256")
    baseline_guard = negative_metrics(
        guard_rows,
        baseline_guard_rows["g7_probability"].to_numpy(dtype=np.float64),
        0.5,
    )
    candidate_guard_probability, guard_checkpoint = predict_guard(
        checkpoint_path,
        guard_manifest_path,
        batch_size=128,
        num_workers=0,
        device_name="cuda",
    )
    if int(guard_checkpoint.get("seed", -1)) != 42:
        raise ValueError("R1 seed changed during G9 guard inference")
    candidate_guard = negative_metrics(
        guard_rows, candidate_guard_probability, 0.5
    )

    val_gate = gates[
        "val_ood_holdout_at_candidate_tune_calibrated_1pct_fpr"
    ]
    g9_gate = gates["g9_mechanical_guard_at_probability_0_5"]
    checks.extend(
        [
            {
                "name": "val_ood_holdout_tpr_strict",
                "value": float(candidate_holdout["recall"]),
                "minimum": float(val_gate["minimum_tpr_at_strict_threshold"]),
                "passed": float(candidate_holdout["recall"])
                >= float(val_gate["minimum_tpr_at_strict_threshold"]),
            },
            {
                "name": "val_ood_holdout_fpr_strict",
                "value": float(candidate_holdout["false_positive_rate"]),
                "maximum": float(val_gate["maximum_fpr_at_strict_threshold"]),
                "passed": float(candidate_holdout["false_positive_rate"])
                <= float(val_gate["maximum_fpr_at_strict_threshold"]),
            },
            {
                "name": "val_ood_holdout_roc_auc",
                "value": float(candidate_holdout["auc"]),
                "minimum": float(val_gate["minimum_roc_auc"]),
                "passed": float(candidate_holdout["auc"])
                >= float(val_gate["minimum_roc_auc"]),
            },
            {
                "name": "g9_mechanical_guard_segment_fpr",
                "value": float(candidate_guard["segment_false_positive_rate"]),
                "maximum": float(g9_gate["maximum_segment_fpr"]),
                "passed": float(candidate_guard["segment_false_positive_rate"])
                <= float(g9_gate["maximum_segment_fpr"]),
            },
        ]
    )
    gate_passed = all(item["passed"] for item in checks)
    prediction_dir = output_path.parent / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    tune_predictions = tune_manifest.copy()
    tune_predictions["r1_logit"] = np.asarray(
        candidate_tune_logits, dtype=np.float64
    )
    tune_predictions["r1_calibrated_probability"] = candidate_tune_probability
    tune_path = prediction_dir / "val_ood_tune.csv"
    tune_predictions.to_csv(tune_path, index=False)
    val_predictions = holdout_manifest.copy()
    val_predictions["r1_logit"] = np.asarray(
        candidate_holdout_logits, dtype=np.float64
    )
    val_predictions["r1_candidate_calibrated_probability"] = (
        candidate_holdout_probability
    )
    val_predictions["r1_fixed_g7_mapping_probability"] = (
        candidate_fixed_mapping_probability
    )
    val_path = prediction_dir / "val_ood_holdout.csv"
    val_predictions.to_csv(val_path, index=False)
    guard_predictions = guard_rows.copy()
    guard_predictions["r1_probability"] = candidate_guard_probability
    guard_path = prediction_dir / "g9_guard.csv"
    guard_predictions.to_csv(guard_path, index=False)

    report = {
        "algorithm": ALGORITHM,
        "status": "guard_evaluation_complete",
        "decision": (
            "advance_to_idmt_development"
            if gate_passed
            else "reject_r1_seed42_before_idmt_development"
        ),
        "checkpoint": {
            "path": checkpoint_path.relative_to(root).as_posix(),
            "sha256": file_sha256(checkpoint_path),
            "seed": int(checkpoint["seed"]),
        },
        "score_protocol": score_protocol,
        "candidate_calibration": {
            "temperature": candidate_temperature,
            "strict_target_fpr_1": calibrations[0.01],
            "sensitivity_target_fpr_5": calibrations[0.05],
            "holdout_used_for_temperature_or_threshold": False,
            "idmt_used_for_temperature_or_threshold": False,
        },
        "internal": internal,
        "val_ood_holdout": {
            "baseline_g7": baseline_holdout,
            "candidate_r1_strict": candidate_holdout,
            "candidate_r1_sensitivity": candidate_holdout_sensitivity,
            "candidate_fixed_g7_mapping_diagnostic": candidate_fixed_mapping,
        },
        "g9_mechanical_guard": {
            "baseline_g7": baseline_guard,
            "candidate_r1": candidate_guard,
        },
        "checks": checks,
        "gate_passed": gate_passed,
        "predictions": {
            "val_ood_tune": {
                "path": tune_path.relative_to(root).as_posix(),
                "sha256": file_sha256(tune_path),
            },
            "val_ood_holdout": {
                "path": val_path.relative_to(root).as_posix(),
                "sha256": file_sha256(val_path),
            },
            "g9_guard": {
                "path": guard_path.relative_to(root).as_posix(),
                "sha256": file_sha256(guard_path),
            },
        },
        "audio_identity_audit": {
            "val_ood_tune": tune_audio_audit,
            "val_ood_holdout": holdout_audio_audit,
            "g9_guard_cache": guard_cache_audit,
        },
        "val_ood_audio_read": True,
        "g9_guard_audio_read": True,
        "idmt_calibration_audio_read": False,
        "idmt_development_audio_read": False,
        "idmt_final_holdout_audio_read": False,
        "locked_datasets_read": [],
    }
    _atomic_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pre-IDMT gates for G7-R1")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/g7_r1_protocol.yaml"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = _resolve(root, args.protocol).resolve(strict=True)
    report = evaluate(root, protocol_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
