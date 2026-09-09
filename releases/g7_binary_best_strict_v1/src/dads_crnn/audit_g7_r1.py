from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from .config import load_config
from .data_firewall import (
    LOCKED_COMPACT_TOKENS,
    PATH_LIKE_COLUMNS,
    audit_csv_rows,
    compact_token,
    file_sha256,
    reject_locked_path,
    reject_locked_value,
)


ALGORITHM = "g7_r1_structured_microphone_single_variable_v1"
IMPLEMENTATION_PATHS = (
    "src/dads_crnn/audit_g7_r1.py",
    "src/dads_crnn/audio.py",
    "src/dads_crnn/augmentation.py",
    "src/dads_crnn/calibrate_ood.py",
    "src/dads_crnn/data_firewall.py",
    "src/dads_crnn/dataset.py",
    "src/dads_crnn/evaluate_g7_idmt_r0.py",
    "src/dads_crnn/evaluate_g9_guard.py",
    "src/dads_crnn/evaluate_low_fpr.py",
    "src/dads_crnn/metrics.py",
    "src/dads_crnn/panns.py",
    "src/dads_crnn/train.py",
    "src/dads_crnn/train_panns.py",
    "src/dads_crnn/evaluate_g7_r1_guards.py",
    "src/dads_crnn/evaluate_g7_r1_idmt.py",
)
R1_LOCKED_COMPACT_TOKENS = tuple(
    dict.fromkeys(
        (
            *LOCKED_COMPACT_TOKENS,
            "hohenwarte",
            "finalholdout",
            "final_holdout",
        )
    )
)
EXPECTED_ALLOWED_CONFIG_DIFFERENCES = (
    "output_dir",
    "train.augmentation.frequency_response_mode",
    "train.augmentation.microphone_highpass_hz",
    "train.augmentation.microphone_lowpass_hz",
    "train.augmentation.microphone_peaking_center_hz",
    "train.augmentation.microphone_peaking_gain_db",
    "train.augmentation.microphone_peaking_q",
    "train.augmentation.microphone_peaking_sections",
)
EXPECTED_TRAINING_MANIFEST_PATH = "artifacts_full/manifests/dads_all_seed42.csv"
EXPECTED_TRAINING_MANIFEST_SHA256 = (
    "be1a6293b90a15208c75d4f2aa8cfe4efefc68c8cc382ca57f9c637e08f09c45"
)
EXPECTED_STRUCTURED_RANGES: dict[str, list[float] | list[int]] = {
    "microphone_highpass_hz": [60.0, 120.0],
    "microphone_lowpass_hz": [6000.0, 8000.0],
    "microphone_peaking_sections": [2, 3],
    "microphone_peaking_center_hz": [120.0, 6000.0],
    "microphone_peaking_q": [0.5, 2.0],
    "microphone_peaking_gain_db": [-4.0, 4.0],
}
EXPECTED_SCORE_PROTOCOL = {
    "primary_comparison": "per_model_same_tune_calibration",
    "temperature_fit": "tune_all_labels_nll",
    "threshold_selection": "negative_only_conservative_empirical_quantile",
    "target_fprs": [0.01, 0.05],
    "calibration_manifest_role": "val_ood_tune",
    "holdout_used": False,
    "idmt_used": False,
    "baseline_g7_mapping": {
        "temperature": 2.258312527893372,
        "strict_threshold": 0.7236399840238329,
        "sensitivity_threshold": 0.5085329108689571,
    },
    "fixed_g7_mapping_role": "deployment_compatibility_diagnostic_only",
}
EXPECTED_PROMOTION_GATES = {
    "dads_test_at_probability_0_5": {
        "minimum_f1": 0.9949923708920188,
        "minimum_roc_auc": 0.9993540433796199,
        "minimum_recall": 0.9955672826126887,
        "minimum_specificity": 0.9913047670058918,
    },
    "dads_validation_at_probability_0_5": {
        "minimum_f1": 0.9949094449506047,
        "minimum_roc_auc": 0.9993491688096940,
    },
    "val_ood_holdout_at_candidate_tune_calibrated_1pct_fpr": {
        "minimum_tpr_at_strict_threshold": 0.2169863013698630,
        "maximum_fpr_at_strict_threshold": 0.0426850605652759,
        "minimum_roc_auc": 0.7816167333468538,
    },
    "g9_mechanical_guard_at_probability_0_5": {
        "maximum_segment_fpr": 0.0725,
    },
    "idmt_development_at_candidate_tune_calibrated_operating_points": {
        "maximum_overall_fpr_at_strict_threshold": 0.0567893409664009,
        "maximum_me_vehicle_fpr_at_strict_threshold": 0.18804331013147717,
        "maximum_se_fpr_at_strict_threshold": 0.0243592365371507,
        "maximum_background_only_fpr_at_strict_threshold": 0.0118691250903832,
        "maximum_overall_fpr_at_sensitivity_threshold": 0.1746745723437607,
        "paired_session_bootstrap_samples": 2000,
        "paired_session_bootstrap_seed": 20260731,
        "require_one_sided_95pct_upper_fpr_delta_below_zero": True,
    },
    "decision": {
        "advance_to_seeds_43_44_only_if_all_gates_pass": True,
        "development_results_are_not_final_unbiased_claims": True,
    },
}


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _reject_r1_locked_path(path: Path, *, context: str) -> None:
    reject_locked_path(
        path,
        context=context,
        tokens=R1_LOCKED_COMPACT_TOKENS,
    )


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


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    output: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        output.update(_flatten(item, path))
    return output


def config_differences(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    baseline_flat = _flatten(baseline)
    candidate_flat = _flatten(candidate)
    return sorted(
        key
        for key in set(baseline_flat) | set(candidate_flat)
        if baseline_flat.get(key) != candidate_flat.get(key)
    )


def _numeric_pair(value: object, *, name: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) for item in value)
    ):
        raise ValueError(f"R1 {name} must be a two-value numeric list")
    try:
        pair = [float(item) for item in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"R1 {name} must be a two-value numeric list") from error
    if not all(math.isfinite(item) for item in pair) or pair[0] > pair[1]:
        raise ValueError(f"R1 {name} must be a finite increasing pair")
    return pair


def _validate_structured_ranges(
    protocol: dict[str, Any], candidate_aug: dict[str, Any]
) -> dict[str, list[float] | list[int]]:
    observed: dict[str, list[float] | list[int]] = {}
    for name, expected in EXPECTED_STRUCTURED_RANGES.items():
        pair = _numeric_pair(candidate_aug.get(name), name=name)
        if name == "microphone_peaking_sections":
            if any(not item.is_integer() for item in pair):
                raise ValueError("R1 microphone_peaking_sections must contain integers")
            normalized: list[float] | list[int] = [int(item) for item in pair]
        else:
            normalized = pair
        if normalized != expected:
            raise ValueError(
                f"R1 structured range changed for {name}: "
                f"observed={normalized}, expected={expected}"
            )
        observed[name] = normalized

    reference = protocol["contract"].get("reference_method", {})
    inherited = reference.get("inherited_ranges", {})
    project = reference.get("project_specific_preregistered_choices", {})
    reference_ranges = {
        "microphone_highpass_hz": inherited.get("highpass_hz"),
        "microphone_lowpass_hz": inherited.get("lowpass_hz"),
        "microphone_peaking_sections": inherited.get("peaking_sections"),
        "microphone_peaking_gain_db": inherited.get("peaking_gain_db"),
        "microphone_peaking_center_hz": project.get("peaking_center_hz"),
        "microphone_peaking_q": project.get("peaking_q"),
    }
    if reference_ranges != EXPECTED_STRUCTURED_RANGES:
        raise ValueError(
            "R1 protocol reference ranges differ from the frozen structured ranges"
        )
    if project.get("peaking_center_distribution") != "log_uniform":
        raise ValueError("R1 peaking center distribution must remain log_uniform")
    return observed


def _validate_promotion_gates(protocol: dict[str, Any]) -> dict[str, Any]:
    gates = protocol.get("promotion_gates")
    if not isinstance(gates, dict) or gates != EXPECTED_PROMOTION_GATES:
        raise ValueError(
            "R1 promotion gates are missing or differ from the preregistered gates"
        )
    return gates


def _validate_score_protocol(
    protocol: dict[str, Any],
    low_fpr_metrics: dict[str, Any],
    r0_metrics: dict[str, Any],
) -> dict[str, Any]:
    score_protocol = protocol.get("contract", {}).get("score_protocol")
    if score_protocol != EXPECTED_SCORE_PROTOCOL:
        raise ValueError("R1 score protocol differs from the preregistered policy")

    by_target = {
        float(item["target_fpr"]): float(item["calibration"]["threshold"])
        for item in low_fpr_metrics.get("operating_points", [])
    }
    baseline_mapping = EXPECTED_SCORE_PROTOCOL["baseline_g7_mapping"]
    if (
        by_target.get(0.01) != baseline_mapping["strict_threshold"]
        or by_target.get(0.05) != baseline_mapping["sensitivity_threshold"]
    ):
        raise ValueError(
            "R1 diagnostic G7 thresholds do not match the frozen low-FPR source"
        )
    r0_protocol = r0_metrics.get("protocol", {})
    if (
        float(r0_protocol.get("temperature", float("nan")))
        != baseline_mapping["temperature"]
    ):
        raise ValueError(
            "R1 diagnostic G7 temperature does not match the formal GPU R0 result"
        )
    return score_protocol


def _validate_single_variable_contract(
    protocol: dict[str, Any],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    contract = protocol.get("contract", {})
    observed_differences = config_differences(baseline, candidate)
    allowed_differences = sorted(
        str(value) for value in contract.get("allowed_config_differences", [])
    )
    if tuple(allowed_differences) != EXPECTED_ALLOWED_CONFIG_DIFFERENCES:
        raise ValueError("R1 allowed config differences changed")
    if observed_differences != allowed_differences:
        raise ValueError(
            "R1 is not a single-variable configuration delta: "
            f"observed={observed_differences}, allowed={allowed_differences}"
        )

    baseline_aug = baseline["train"]["augmentation"]
    candidate_aug = candidate["train"]["augmentation"]
    expected_probability = float(
        contract["unchanged_frequency_response_probability"]
    )
    if expected_probability != 0.50:
        raise ValueError("R1 contract must freeze frequency-response probability at 0.50")
    if (
        float(baseline_aug["frequency_response_probability"])
        != expected_probability
        or float(candidate_aug["frequency_response_probability"])
        != expected_probability
    ):
        raise ValueError("R1 frequency-response probability changed")
    if candidate_aug.get("frequency_response_mode") != "structured_microphone_v1":
        raise ValueError("R1 must select structured_microphone_v1")

    protocol_seeds = [int(value) for value in contract.get("training_seeds", [])]
    baseline_seeds = [int(value) for value in baseline["train"]["seeds"]]
    candidate_seeds = [int(value) for value in candidate["train"]["seeds"]]
    if protocol_seeds != [42] or baseline_seeds != [42] or candidate_seeds != [42]:
        raise ValueError("R1 pilot and its G7 baseline must use only seed 42")
    if contract.get("training_data_role") != "historical_dads_only":
        raise ValueError("R1 training data role must remain historical_dads_only")
    for name in (
        "idmt_calibration_audio_allowed",
        "idmt_development_audio_allowed_before_training_complete",
        "idmt_final_holdout_allowed",
    ):
        if contract.get(name) is not False:
            raise ValueError(f"R1 firewall contract changed: {name}")
    if int(contract.get("idmt_training_rows", -1)) != 0:
        raise ValueError("R1 must not train on IDMT rows")
    if Path(candidate["output_dir"]).as_posix() == Path(
        baseline["output_dir"]
    ).as_posix():
        raise ValueError("R1 must not overwrite the G7 baseline run")

    structured_ranges = _validate_structured_ranges(protocol, candidate_aug)
    promotion_gates = _validate_promotion_gates(protocol)
    return {
        "observed_differences": observed_differences,
        "allowed_differences": allowed_differences,
        "frequency_response_probability": expected_probability,
        "training_seeds": candidate_seeds,
        "structured_ranges": structured_ranges,
        "promotion_gates": promotion_gates,
    }


def _audit_r1_manifest_paths(root: Path, manifest: Path) -> dict[str, Any]:
    _reject_r1_locked_path(manifest, context="R1 training manifest")
    _reject_r1_locked_path(root, context="R1 workspace root")
    checked_parents: set[str] = set()
    symlink_values_checked = 0
    path_values_checked = 0
    path_columns: list[str] = []
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"R1 training manifest lacks a CSV header: {manifest}")
        path_columns = [
            column
            for column in reader.fieldnames
            if column.lower() in PATH_LIKE_COLUMNS or "path" in column.lower()
        ]
        for row_number, row in enumerate(reader, start=2):
            for column in path_columns:
                value = str(row.get(column, "")).strip()
                if not value:
                    continue
                path_values_checked += 1
                reject_locked_value(
                    value,
                    context=f"{manifest}: row {row_number}, column {column}",
                    tokens=R1_LOCKED_COMPACT_TOKENS,
                )
                if "://" in value:
                    continue
                raw_path = Path(value)
                candidate = raw_path if raw_path.is_absolute() else root / raw_path
                parent_key = candidate.parent.as_posix()
                if parent_key not in checked_parents:
                    _reject_r1_locked_path(
                        candidate.parent,
                        context=(
                            f"R1 manifest path root at row {row_number}, "
                            f"column {column}"
                        ),
                    )
                    checked_parents.add(parent_key)
                if candidate.is_symlink():
                    _reject_r1_locked_path(
                        candidate,
                        context=(
                            f"R1 manifest symlink at row {row_number}, "
                            f"column {column}"
                        ),
                    )
                    symlink_values_checked += 1
    return {
        "custom_compact_tokens": sorted(
            {compact_token(value) for value in R1_LOCKED_COMPACT_TOKENS}
        ),
        "path_columns": path_columns,
        "path_values_checked": path_values_checked,
        "resolved_path_parents_checked": len(checked_parents),
        "symlink_values_checked": symlink_values_checked,
    }


def _validate_training_manifest_identity(root: Path, manifest: Path) -> str:
    try:
        relative = manifest.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("R1 training manifest must remain inside the workspace") from error
    if relative != EXPECTED_TRAINING_MANIFEST_PATH:
        raise ValueError(
            "R1 training manifest path changed: "
            f"observed={relative}, expected={EXPECTED_TRAINING_MANIFEST_PATH}"
        )
    observed_sha256 = file_sha256(manifest)
    if observed_sha256 != EXPECTED_TRAINING_MANIFEST_SHA256:
        raise ValueError(
            "R1 training manifest SHA256 changed: "
            f"observed={observed_sha256}, "
            f"expected={EXPECTED_TRAINING_MANIFEST_SHA256}"
        )
    return observed_sha256


def _load_protocol(root: Path, protocol_path: Path):
    _reject_r1_locked_path(protocol_path, context="R1 protocol config")
    protocol = load_config(protocol_path)
    if protocol.get("algorithm") != ALGORITHM:
        raise ValueError(f"R1 algorithm must be {ALGORITHM}")
    inputs = {
        name: _resolve(root, value)
        for name, value in protocol["inputs"].items()
    }
    outputs = {
        name: _resolve(root, value)
        for name, value in protocol["outputs"].items()
    }
    for name, path in inputs.items():
        _reject_r1_locked_path(path, context=f"R1 input {name}")
    for name, path in outputs.items():
        _reject_r1_locked_path(path, context=f"R1 output {name}")
    return protocol, inputs, outputs


def build_audit(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, inputs, outputs = _load_protocol(root, protocol_path)
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing R1 input {name}: {path}")

    manifest = inputs["training_manifest"]
    manifest_firewall = _audit_r1_manifest_paths(root, manifest)
    manifest_sha256 = _validate_training_manifest_identity(root, manifest)
    manifest_rows = audit_csv_rows(manifest, required_columns=("split", "label"))

    baseline = load_config(inputs["baseline_config"])
    candidate = load_config(inputs["candidate_config"])
    contract_audit = _validate_single_variable_contract(
        protocol, baseline, candidate
    )
    low_fpr_metrics = json.loads(
        inputs["baseline_low_fpr_metrics"].read_text(encoding="utf-8")
    )
    r0_metrics = json.loads(inputs["r0_idmt_metrics"].read_text(encoding="utf-8"))
    score_protocol = _validate_score_protocol(
        protocol, low_fpr_metrics, r0_metrics
    )

    implementation = {
        value: file_sha256(root / value) for value in IMPLEMENTATION_PATHS
    }
    identity_paths = {
        "protocol_config": protocol_path,
        **inputs,
    }
    identity = {
        name: {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
        }
        for name, path in identity_paths.items()
    }
    report = {
        "algorithm": ALGORITHM,
        "passed": True,
        "status": "audited_before_training",
        "experimental_factor": protocol["contract"]["experimental_factor"],
        "config_differences": contract_audit["observed_differences"],
        "allowed_config_differences": contract_audit["allowed_differences"],
        "frequency_response_probability": contract_audit[
            "frequency_response_probability"
        ],
        "training_seeds": contract_audit["training_seeds"],
        "structured_microphone_ranges": contract_audit["structured_ranges"],
        "score_protocol": score_protocol,
        "training_manifest": {
            "path": manifest.relative_to(root).as_posix(),
            "sha256": manifest_sha256,
            "rows": manifest_rows,
        },
        "training_manifest_firewall": manifest_firewall,
        "inputs": identity,
        "implementation_sha256": implementation,
        "promotion_gates": contract_audit["promotion_gates"],
        "locked_datasets_read": [],
        "idmt_training_rows": 0,
        "idmt_calibration_audio_read": False,
        "idmt_development_audio_read": False,
        "idmt_final_holdout_audio_read": False,
        "model_training_started": False,
    }
    return report


def freeze(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, _, outputs = _load_protocol(root, protocol_path)
    frozen_path = outputs["frozen_protocol"]
    if frozen_path.exists():
        raise FileExistsError(f"R1 protocol is already frozen: {frozen_path}")
    report = build_audit(root, protocol_path)
    report["status"] = "frozen_before_training"
    report["protocol_contract"] = protocol["contract"]
    _atomic_json(frozen_path, report)
    _atomic_json(outputs["audit"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def verify(root: Path, protocol_path: Path) -> dict[str, Any]:
    _, _, outputs = _load_protocol(root, protocol_path)
    frozen_path = outputs["frozen_protocol"]
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    observed = build_audit(root, protocol_path)
    if frozen.get("algorithm") != ALGORITHM or frozen.get("status") != "frozen_before_training":
        raise ValueError("R1 frozen protocol is invalid")
    for name in (
        "inputs",
        "implementation_sha256",
        "config_differences",
        "allowed_config_differences",
        "frequency_response_probability",
        "training_seeds",
        "structured_microphone_ranges",
        "score_protocol",
        "training_manifest_firewall",
        "promotion_gates",
    ):
        if frozen.get(name) != observed.get(name):
            raise ValueError(f"R1 frozen field changed after freeze: {name}")
    report = {
        "algorithm": ALGORITHM,
        "passed": True,
        "status": "verified_against_frozen_protocol",
        "frozen_protocol": {
            "path": frozen_path.relative_to(root).as_posix(),
            "sha256": file_sha256(frozen_path),
        },
        "locked_datasets_read": [],
        "idmt_calibration_audio_read": False,
        "idmt_development_audio_read": False,
        "idmt_final_holdout_audio_read": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze or verify the G7-R1 experiment")
    parser.add_argument("mode", choices=("freeze", "verify"))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/g7_r1_protocol.yaml"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = _resolve(root, args.protocol).resolve(strict=True)
    if args.mode == "freeze":
        freeze(root, protocol_path)
    else:
        verify(root, protocol_path)


if __name__ == "__main__":
    main()
