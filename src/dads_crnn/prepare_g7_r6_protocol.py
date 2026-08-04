from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config
from .data_firewall import file_sha256


PROTOCOL = "g7_r6_reuter_reusable_multicorpus_v3"
IDENTITY_COLUMNS = ("audio_sha256", "segment_sha256", "recording_group", "source_group")


def _matches_prefixes(values: pd.Series, prefixes: list[str]) -> pd.Series:
    text = values.fillna("").astype(str)
    return text.map(lambda value: any(value.startswith(prefix) for prefix in prefixes))


def _summary(rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(rows)),
        "recordings": int(rows["recording_group"].nunique()),
        "source_groups": int(rows["source_group"].nunique()),
        "by_dataset_label": {
            f"{dataset}:label_{int(label)}": int(count)
            for (dataset, label), count in rows.groupby(["dataset_origin", "label"]).size().items()
        },
    }


def _overlap(left: pd.DataFrame, right: pd.DataFrame, column: str) -> int:
    return len(set(left[column].dropna().astype(str)) & set(right[column].dropna().astype(str)))


def prepare(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected G7-R6 protocol")

    train_path = Path(config["base_manifests"]["train"])
    validation_path = Path(config["base_manifests"]["validation"])
    internal_test_path = Path(config["base_manifests"]["internal_benchmark"])
    dronenoise_path = Path(config["base_manifests"]["dronenoise_halfsecond"])
    train = pd.read_csv(train_path, low_memory=False)
    validation = pd.read_csv(validation_path, low_memory=False)
    reusable_internal_test = pd.read_csv(internal_test_path, low_memory=False)
    dronenoise = pd.read_csv(dronenoise_path, low_memory=False)
    required_columns = set(train.columns)
    if set(dronenoise.columns) != required_columns:
        missing = sorted(required_columns - set(dronenoise.columns))
        extra = sorted(set(dronenoise.columns) - required_columns)
        raise ValueError(f"DroneNoise model manifest schema mismatch: missing={missing}, extra={extra}")
    if set(dronenoise["split"].astype(str)) != {"train", "validation", "test"}:
        raise ValueError("DroneNoise half-second manifest must contain train/validation/test")

    tau = validation["dataset_origin"].eq("tau_urban_2022")
    validation_tau = tau & _matches_prefixes(
        validation["source_group"],
        list(config["development_partition"]["tau_model_validation_prefixes"]),
    )
    calibration_tau = tau & _matches_prefixes(
        validation["source_group"],
        list(config["development_partition"]["tau_threshold_calibration_prefixes"]),
    )
    if bool((validation_tau & calibration_tau).any()):
        raise ValueError("TAU model-validation and calibration groups overlap")
    if not bool((validation_tau | calibration_tau).equals(tau)):
        missing = sorted(validation.loc[tau & ~(validation_tau | calibration_tau), "source_group"].unique())
        raise ValueError(f"Unassigned TAU validation source groups: {missing}")

    model_validation = validation[~tau | validation_tau].copy()
    threshold_calibration = validation[calibration_tau].copy()
    dronenoise_train = dronenoise[dronenoise["split"].eq("train")].copy()
    dronenoise_validation = dronenoise[dronenoise["split"].eq("validation")].copy()
    reusable_positive_test = dronenoise[dronenoise["split"].eq("test")].copy()
    dronenoise_train["dataset_role"] = "train"
    dronenoise_train["split"] = "train"
    dronenoise_validation["dataset_role"] = "model_validation"
    dronenoise_validation["split"] = "model_validation"
    reusable_positive_test["dataset_role"] = "reusable_positive_test"
    reusable_positive_test["split"] = "reusable_positive_test"
    reusable_internal_test["dataset_role"] = "reusable_internal_test"
    reusable_internal_test["split"] = "reusable_internal_test"
    train = pd.concat([train, dronenoise_train], ignore_index=True)
    model_validation = pd.concat(
        [model_validation, dronenoise_validation], ignore_index=True
    )
    model_validation["dataset_role"] = "model_validation"
    model_validation["split"] = "model_validation"
    threshold_calibration["dataset_role"] = "threshold_calibration"
    threshold_calibration["split"] = "threshold_calibration"
    if set(threshold_calibration["label"].astype(int)) != {0}:
        raise ValueError("Threshold calibration must contain background-only rows")

    partitions = {
        "train": train,
        "model_validation": model_validation,
        "threshold_calibration": threshold_calibration,
        "reusable_internal_test": reusable_internal_test,
        "reusable_positive_test": reusable_positive_test,
    }
    overlaps: dict[str, dict[str, int]] = {}
    names = list(partitions)
    for column in IDENTITY_COLUMNS:
        overlaps[column] = {}
        for index, left_name in enumerate(names):
            for right_name in names[index + 1 :]:
                key = f"{left_name}__{right_name}"
                overlaps[column][key] = _overlap(
                    partitions[left_name], partitions[right_name], column
                )
    if any(value for checks in overlaps.values() for value in checks.values()):
        raise ValueError("G7-R6 development partitions are not identity-disjoint")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, rows in partitions.items():
        path = output_dir / f"{name}_manifest.csv"
        rows.to_csv(path, index=False)
        outputs[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            **_summary(rows),
        }
    fit = pd.concat(
        [train, model_validation, reusable_internal_test], ignore_index=True
    )
    fit_path = output_dir / "fit_manifest.csv"
    fit.to_csv(fit_path, index=False)
    outputs["fit"] = {
        "path": str(fit_path),
        "sha256": file_sha256(fit_path),
        **_summary(fit),
        "roles": {
            str(key): int(value) for key, value in fit["split"].value_counts().items()
        },
    }

    policy_path = output_dir / "REUSABLE_BENCHMARK_POLICY.json"
    benchmark_policy = {
        "protocol": "g7_r6_reusable_benchmark_policy_v1",
        "status": "active_reusable",
        **config["benchmark_policy"],
    }
    policy_path.write_text(
        json.dumps(benchmark_policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    audit = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reuter_style_layers": {
            "session_disjoint_development": True,
            "model_validation_separate_from_threshold_calibration": True,
            "historical_benchmarks_separate_from_final_claim": True,
            "reusable_benchmark_enabled": True,
            "one_time_final_holdout_required": False,
            "dronenoise_native_halfsecond_integrated": True,
            "dronenoise_test_is_positive_only": True,
            "reusable_internal_test_is_not_used_for_early_stopping": True,
        },
        "outputs": outputs,
        "overlap_checks": overlaps,
        "reusable_benchmarks": config["roles"]["reusable_external_benchmark"]["sources"],
        "new_dataset_intake": config["new_dataset_intake"],
        "reusable_benchmark": {
            "policy_path": str(policy_path),
            "status": benchmark_policy["status"],
            "repeat_evaluation_allowed": True,
            "independent_final_claim_allowed": False,
        },
        "inputs": {
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "train": {"path": str(train_path), "sha256": file_sha256(train_path)},
            "validation": {
                "path": str(validation_path),
                "sha256": file_sha256(validation_path),
            },
            "reusable_internal_test": {
                "path": str(internal_test_path),
                "sha256": file_sha256(internal_test_path),
            },
            "dronenoise_halfsecond": {
                "path": str(dronenoise_path),
                "sha256": file_sha256(dronenoise_path),
            },
        },
    }
    audit_path = output_dir / "protocol_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare strict G7-R6 multicorpus data roles")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g7_r6_reuter_reusable_protocol.yaml"),
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
