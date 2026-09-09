from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data_firewall import file_sha256


PROTOCOL = "g7_r5_explicit_train_validation_test_roles_v1"
TARGET_SAMPLES = 8000
OUTPUT_COLUMNS = [
    "dataset_role",
    "split",
    "dataset_origin",
    "label",
    "cache_path",
    "cache_index",
    "cache_start_sample",
    "cache_end_sample",
    "source_path",
    "recording_group",
    "source_group",
    "domain_bucket",
    "background_mix_eligible",
    "audio_sha256",
    "segment_sha256",
    "half_index",
    "uav_subtype",
    "uav_novelty",
    "device",
    "scene",
    "license",
]


def _empty_string(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame:
        return frame[column].fillna("").astype(str)
    return pd.Series("", index=frame.index, dtype=object)


def _dads(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, low_memory=False)
    role_map = {"train": "train", "val": "validation", "test": "internal_test"}
    output = pd.DataFrame(index=source.index)
    output["dataset_role"] = source["split"].map(role_map)
    output["split"] = output["dataset_role"]
    output["dataset_origin"] = "dads_halfsec"
    output["label"] = source["label"].astype(int)
    output["cache_path"] = source["cache_path"].astype(str)
    output["cache_index"] = source["cache_index"].astype(int)
    output["cache_start_sample"] = 0
    output["cache_end_sample"] = TARGET_SAMPLES
    output["source_path"] = source["source_path"].astype(str)
    output["recording_group"] = source["recording_group"].astype(str)
    output["source_group"] = source["recording_group"].astype(str)
    output["domain_bucket"] = np.where(output["label"].eq(1), "positive:dads", "negative:dads")
    output["background_mix_eligible"] = output["label"].eq(0)
    output["audio_sha256"] = source["raw_audio_sha256"].astype(str)
    output["segment_sha256"] = source["segment_float32_sha256"].astype(str)
    output["half_index"] = source["segment_index"].astype(int)
    output["uav_subtype"] = np.where(output["label"].eq(1), "DADS_unspecified", "")
    output["uav_novelty"] = np.where(output["label"].eq(1), "unspecified", "not_applicable")
    output["device"] = "unavailable"
    output["scene"] = "unavailable"
    output["license"] = "source_dataset_terms"
    return output[OUTPUT_COLUMNS]


def _external(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, low_memory=False)
    role_map = {
        "train": "train",
        "tune": "validation",
        "dev_holdout": "locked_external_test",
    }
    source = source.copy()
    source["dataset_role"] = source["split"].map(role_map)
    development_types = set(
        source[
            source["dataset_role"].isin(["train", "validation"])
            & source["label"].astype(int).eq(1)
        ]["subtype"].dropna().astype(str)
    )
    halves = []
    for half, start in enumerate((0, TARGET_SAMPLES)):
        output = pd.DataFrame(index=source.index)
        output["dataset_role"] = source["dataset_role"]
        output["split"] = output["dataset_role"]
        output["dataset_origin"] = source["dataset"].astype(str)
        output["label"] = source["label"].astype(int)
        output["cache_path"] = source["cache_path"].astype(str)
        output["cache_index"] = source["cache_index"].astype(int)
        output["cache_start_sample"] = start
        output["cache_end_sample"] = start + TARGET_SAMPLES
        output["source_path"] = source["dataset"].astype(str) + ":" + source["audio_sha256"].astype(str)
        output["recording_group"] = output["source_path"]
        output["source_group"] = source["source_group"].astype(str)
        positive = output["label"].eq(1)
        subtype = _empty_string(source, "subtype")
        output["domain_bucket"] = np.where(
            positive,
            "positive:kielce:" + subtype,
            "negative:tau:" + subtype,
        )
        output["background_mix_eligible"] = ~positive
        output["audio_sha256"] = source["audio_sha256"].astype(str)
        output["segment_sha256"] = source["segment_sha256"].astype(str)
        output["half_index"] = half
        output["uav_subtype"] = np.where(positive, subtype, "")
        output["uav_novelty"] = np.where(
            ~positive,
            "not_applicable",
            np.where(subtype.isin(development_types), "seen_model_type", "unseen_model_type"),
        )
        output["device"] = _empty_string(source, "device")
        output["scene"] = _empty_string(source, "scene_label")
        output["license"] = _empty_string(source, "license")
        halves.append(output[OUTPUT_COLUMNS])
    return pd.concat(halves, ignore_index=True)


def _g9(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, low_memory=False)
    halves = []
    for half, start in enumerate((0, TARGET_SAMPLES)):
        output = pd.DataFrame(index=source.index)
        output["dataset_role"] = "train"
        output["split"] = "train"
        output["dataset_origin"] = "g9_mechanical_hard_negative"
        output["label"] = 0
        output["cache_path"] = source["cache_path"].astype(str)
        output["cache_index"] = np.nan
        output["cache_start_sample"] = start
        output["cache_end_sample"] = start + TARGET_SAMPLES
        output["source_path"] = source["recording_group"].astype(str)
        output["recording_group"] = source["recording_group"].astype(str)
        output["source_group"] = source["source_group"].astype(str)
        output["domain_bucket"] = "negative:g9:" + source["hard_negative_class"].astype(str)
        output["background_mix_eligible"] = False
        output["audio_sha256"] = source["audio_sha256"].astype(str)
        output["segment_sha256"] = source["segment_sha256"].astype(str)
        output["half_index"] = half
        output["uav_subtype"] = ""
        output["uav_novelty"] = "not_applicable"
        output["device"] = "ESC-50"
        output["scene"] = source["hard_negative_class"].astype(str)
        output["license"] = "source_dataset_terms"
        halves.append(output[OUTPUT_COLUMNS])
    return pd.concat(halves, ignore_index=True)


def _role_overlap(frames: dict[str, pd.DataFrame], column: str) -> dict[str, int]:
    keys = list(frames)
    return {
        f"{left}__{right}": len(
            set(frames[left][column].dropna().astype(str))
            & set(frames[right][column].dropna().astype(str))
        )
        for index, left in enumerate(keys)
        for right in keys[index + 1 :]
    }


def prepare(dads_path: Path, external_path: Path, g9_path: Path, output_dir: Path) -> dict:
    combined = pd.concat(
        [_dads(dads_path), _external(external_path), _g9(g9_path)],
        ignore_index=True,
    )
    if combined[OUTPUT_COLUMNS[:-1]].isna().any().any():
        nullable_cache = combined["dataset_origin"].eq("g9_mechanical_hard_negative")
        invalid = combined.drop(columns=["cache_index"]).isna().any().any()
        if invalid or combined.loc[~nullable_cache, "cache_index"].isna().any():
            raise ValueError("R5 role registry contains unexpected null values")
    roles = ["train", "validation", "internal_test", "locked_external_test"]
    frames = {
        role: combined[combined["dataset_role"].eq(role)].reset_index(drop=True)
        for role in roles
    }
    if any(frame.empty for frame in frames.values()):
        raise ValueError("Every R5 dataset role must be non-empty")
    overlap = {
        column: _role_overlap(frames, column)
        for column in ("audio_sha256", "recording_group", "source_group", "segment_sha256")
    }
    if any(value for checks in overlap.values() for value in checks.values()):
        raise ValueError(f"R5 role leakage detected: {overlap}")
    for role, frame in frames.items():
        conflicts = frame.groupby("recording_group")["label"].nunique()
        if bool((conflicts > 1).any()):
            raise ValueError(f"Cross-label recording conflict in role={role}")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "train_manifest.csv",
        "validation": output_dir / "validation_manifest.csv",
        "internal_test": output_dir / "internal_test_manifest.csv",
        # The lexical 'unseen' token is intentionally rejected by existing
        # development firewall helpers.
        "locked_external_test": output_dir / "locked_unseen_external_test" / "manifest.csv",
    }
    for role, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[role].to_csv(path, index=False)

    counts = {
        role: {
            "rows": len(frame),
            "recordings": int(frame["recording_group"].nunique()),
            "source_groups": int(frame["source_group"].nunique()),
            "by_dataset_label": {
                f"{dataset}:label_{label}": int(count)
                for (dataset, label), count in frame.groupby(["dataset_origin", "label"]).size().items()
            },
            "uav_types": sorted(frame.loc[frame["label"].eq(1), "uav_subtype"].unique().tolist()),
        }
        for role, frame in frames.items()
    }
    test = frames["locked_external_test"]
    novelty_counts = {
        str(key): int(value)
        for key, value in test[test["label"].eq(1)]["uav_novelty"].value_counts().items()
    }
    audit = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_rate": 16000,
        "clip_seconds": 0.5,
        "split_unit": "recording_and_source_group",
        "counts": counts,
        "locked_test_positive_novelty_rows": novelty_counts,
        "overlap_checks": overlap,
        "inputs": {
            "dads": {"path": str(dads_path), "sha256": file_sha256(dads_path)},
            "external": {"path": str(external_path), "sha256": file_sha256(external_path)},
            "g9": {"path": str(g9_path), "sha256": file_sha256(g9_path)},
        },
        "outputs": {
            role: {"path": str(path), "sha256": file_sha256(path), "rows": len(frames[role])}
            for role, path in paths.items()
        },
        "locked_external_test": {
            "status": "locked_unconsumed",
            "audio_payload_read_during_arrangement": False,
            "model_inference_run": False,
            "threshold_selection_allowed": False,
            "candidate_selection_allowed": False,
            "read_once_after_freeze": True,
        },
        "internal_test": {
            "status": "previously_consumed_internal_regression",
            "fresh_final_claim_allowed": False,
        },
    }
    audit_path = output_dir / "split_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lock = {
        "protocol": "g7_r5_locked_external_test_access_v1",
        "status": "locked_unconsumed",
        "manifest_path": str(paths["locked_external_test"]),
        "manifest_sha256": file_sha256(paths["locked_external_test"]),
        "consumed": False,
        "authorized_candidate": None,
        "authorized_thresholds": None,
        "authorized_aggregation": None,
        "access_log": [],
    }
    (paths["locked_external_test"].parent / "ACCESS_LOCK.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Arrange G7 datasets as Train/Validation/Test")
    parser.add_argument("--dads", type=Path, default=Path("artifacts/g7_leakage_fixed_v2/data/manifest.csv"))
    parser.add_argument("--external", type=Path, default=Path("artifacts/g14_domain_generalization/segment_cache/g14_segment_manifest.csv"))
    parser.add_argument("--g9", type=Path, default=Path("artifacts/g15_constrained_adaptation/p0_registry/g9_mechanical_hard_negative.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/g7_r5_train_val_test"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.dads, args.external, args.g9, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
