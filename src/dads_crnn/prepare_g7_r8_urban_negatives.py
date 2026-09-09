from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data_firewall import file_sha256


PROTOCOL = "g7_r8_tau_train_urban_negative_replay_manifest_v1"
BASE_POS_WEIGHT = 155685 / 136368


def build(dads_path: Path, development_path: Path, validity_path: Path, output_dir: Path) -> dict:
    dads = pd.read_csv(dads_path, low_memory=False)
    development = pd.read_csv(development_path, low_memory=False)
    validity = json.loads(validity_path.read_text(encoding="utf-8"))
    identity = validity.get("model_input_identity", {})
    if validity.get("passed") is not True:
        raise ValueError("The bound reusable-benchmark identity audit did not pass")
    if identity.get("train_vs_model_validation_overlap") != 0:
        raise ValueError("Development train/model-validation identity overlap detected")
    if any(identity.get("development_vs_non_development_overlap", {}).values()):
        raise ValueError("Development/non-development model-input identity overlap detected")

    required_dads = {"split", "label", "cache_path", "cache_index", "recording_group"}
    if missing := required_dads.difference(dads.columns):
        raise ValueError(f"DADS manifest lacks columns: {sorted(missing)}")
    required_development = {
        "split", "dataset_origin", "label", "cache_path", "cache_index",
        "cache_start_sample", "cache_end_sample", "recording_group", "source_group",
        "audio_sha256", "segment_sha256",
    }
    if missing := required_development.difference(development.columns):
        raise ValueError(f"Development manifest lacks columns: {sorted(missing)}")

    dads = dads.copy()
    dads["dataset_origin"] = "dads_halfsec"
    dads["source_group"] = dads["recording_group"].astype(str)
    dads["domain_bucket"] = np.where(
        dads["label"].astype(int).eq(1), "positive:dads", "negative:dads"
    )
    dads["background_mix_eligible"] = dads["label"].astype(int).eq(0)

    tau = development[
        development["split"].eq("train")
        & development["dataset_origin"].eq("tau_urban_2022")
    ].copy()
    if tau.empty or not bool(tau["label"].astype(int).eq(0).all()):
        raise ValueError("TAU training replay must be a non-empty pure-negative partition")
    tau["split"] = "train"
    tau["domain_bucket"] = "negative:tau_urban_train"
    # Keep the experiment interpretable: TAU is replayed as a direct negative
    # but does not alter R7's positive/background mixing distribution.
    tau["background_mix_eligible"] = False

    combined = pd.concat([dads, tau], ignore_index=True, sort=False)
    required_output = [
        "split", "label", "cache_path", "recording_group", "source_group",
        "dataset_origin", "domain_bucket", "background_mix_eligible",
    ]
    if combined[required_output].isna().any().any():
        raise ValueError("R8 manifest contains null required values")
    if set(combined["split"].astype(str)) != {"train", "val", "test"}:
        raise ValueError("R8 must preserve train/val/test roles")
    tau_mask = combined["dataset_origin"].eq("tau_urban_2022")
    if set(combined.loc[tau_mask, "split"].astype(str)) != {"train"}:
        raise ValueError("Non-training TAU data entered R8")
    if bool(combined.loc[tau_mask, "background_mix_eligible"].any()):
        raise ValueError("TAU replay entered the positive background mixer")

    original_val = dads[dads["split"].eq("val")]
    original_test = dads[dads["split"].eq("test")]
    candidate_val = combined[combined["split"].eq("val")]
    candidate_test = combined[combined["split"].eq("test")]
    val_identity = original_val["segment_float32_sha256"].tolist() == candidate_val[
        "segment_float32_sha256"
    ].tolist()
    test_identity = original_test["segment_float32_sha256"].tolist() == candidate_test[
        "segment_float32_sha256"
    ].tolist()
    if not (val_identity and test_identity):
        raise ValueError("DADS validation/test identities changed")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    combined.to_csv(manifest_path, index=False)
    train = combined[combined["split"].eq("train")]
    dads_train = train[train["dataset_origin"].eq("dads_halfsec")]
    observed_pos_weight = float(
        dads_train["label"].eq(0).sum() / dads_train["label"].eq(1).sum()
    )
    if not np.isclose(observed_pos_weight, BASE_POS_WEIGHT, rtol=0, atol=1e-12):
        raise ValueError("R7 baseline positive-class weight identity changed")

    audit = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experimental_variable": "direct_replay_of_source_isolated_tau_training_negatives",
        "inputs": {
            "dads": {"path": str(dads_path), "sha256": file_sha256(dads_path)},
            "development": {
                "path": str(development_path), "sha256": file_sha256(development_path)
            },
            "bound_model_input_identity_audit": {
                "path": str(validity_path), "sha256": file_sha256(validity_path)
            },
        },
        "output": {
            "manifest": {
                "path": str(manifest_path), "sha256": file_sha256(manifest_path),
                "rows": int(len(combined)),
            }
        },
        "counts": {
            "by_split": {str(k): int(v) for k, v in combined["split"].value_counts().items()},
            "train_by_origin_and_label": {
                f"{origin}:label_{int(label)}": int(len(rows))
                for (origin, label), rows in train.groupby(["dataset_origin", "label"])
            },
            "tau_train_rows": int(len(tau)),
            "tau_train_recordings": int(tau["recording_group"].nunique()),
            "tau_train_source_groups": int(tau["source_group"].nunique()),
        },
        "controls": {
            "natural_sampling_without_replacement": True,
            "frequency_mixstyle_preserved": True,
            "tau_background_mix_eligible": False,
            "dads_validation_identity_preserved": bool(val_identity),
            "dads_test_identity_preserved": bool(test_identity),
            "fixed_pos_weight": observed_pos_weight,
            "formal_training_started": False,
        },
        "firewall": {
            "idmt_training_rows": 0,
            "g13_training_rows": 0,
            "esc50_guard_training_rows": 0,
            "kielce_tau_holdout_training_rows": 0,
            "threshold_calibration_training_rows": 0,
            "benchmark_audio_read_during_build": False,
            "metadata_identity_audit_reused": True,
            "development_vs_non_development_model_input_overlap": identity[
                "development_vs_non_development_overlap"
            ],
        },
        "locked_datasets_read": [],
        "limitations": [
            "All external benchmarks are consumed reusable development benchmarks.",
            "The bound exact-identity audit does not exclude acoustic near-duplicates.",
            "TAU replay tests urban-negative robustness, not IDMT-specific adaptation.",
        ],
    }
    audit_path = output_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare leakage-controlled G7-R8 urban negatives")
    parser.add_argument("--dads", type=Path, default=Path("artifacts/g7_leakage_fixed_v2/data/manifest.csv"))
    parser.add_argument("--development", type=Path, default=Path("artifacts/g7_r6_reusable_multicorpus/fit_manifest.csv"))
    parser.add_argument("--validity", type=Path, default=Path("artifacts/g7_r6_dronenoise_control/external_suite/validity_audit.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/g7_r8_urban_negatives/data"))
    args = parser.parse_args()
    print(json.dumps(build(args.dads, args.development, args.validity, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
