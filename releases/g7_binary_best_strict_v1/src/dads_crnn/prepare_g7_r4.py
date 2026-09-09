from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data_firewall import file_sha256


PROTOCOL = "g7_r4_multidomain_halfsec_manifest_v1"
TARGET_SAMPLES = 8000


def _sample(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if len(frame) <= count:
        return frame.copy()
    return frame.sample(n=count, random_state=seed).copy()


def _dads_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {"split", "label", "cache_path", "cache_index", "recording_group"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"DADS manifest lacks columns: {sorted(missing)}")
    frame = frame.copy()
    frame["dataset_origin"] = "dads_halfsec"
    frame["domain_bucket"] = np.where(
        frame["label"].astype(int).eq(1), "positive:dads", "negative:dads"
    )
    frame["cache_start_sample"] = 0
    frame["cache_end_sample"] = TARGET_SAMPLES
    frame["background_mix_eligible"] = frame["label"].astype(int).eq(0)
    return frame


def _external_half_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame = frame[frame["split"].isin(["train", "tune"])].copy()
    frame["split"] = frame["split"].replace({"tune": "val"})
    frame["dataset_origin"] = frame["dataset"].astype(str)
    frame["source_path"] = (
        frame["dataset"].astype(str) + ":" + frame["audio_sha256"].astype(str)
    )
    frame["recording_group"] = frame["source_path"]
    positive = frame["label"].astype(int).eq(1)
    frame["domain_bucket"] = np.where(
        positive,
        "positive:kielce:" + frame["subtype"].fillna("unknown").astype(str),
        "negative:tau:" + frame["subtype"].fillna("unknown").astype(str),
    )
    frame["background_mix_eligible"] = ~positive
    halves = []
    for half, start in enumerate((0, TARGET_SAMPLES)):
        view = frame.copy()
        view["half_index"] = half
        view["cache_start_sample"] = start
        view["cache_end_sample"] = start + TARGET_SAMPLES
        view["segment_index"] = view["segment_index"].astype(int) * 2 + half
        halves.append(view)
    return pd.concat(halves, ignore_index=True)


def _g9_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False).copy()
    # The G15 registry uses -1 as a sentinel for individual .npy files, while
    # DADSDataset intentionally treats a non-empty cache_index as a memmap row.
    # Normalize the sentinel at the adapter boundary.
    frame["cache_index"] = np.nan
    frame["split"] = "train"
    frame["dataset_origin"] = "g9_mechanical_hard_negative"
    frame["source_path"] = frame["recording_group"].astype(str)
    frame["domain_bucket"] = (
        "negative:g9:" + frame["hard_negative_class"].astype(str)
    )
    frame["background_mix_eligible"] = False
    halves = []
    for half, start in enumerate((0, TARGET_SAMPLES)):
        view = frame.copy()
        view["half_index"] = half
        view["cache_start_sample"] = start
        view["cache_end_sample"] = start + TARGET_SAMPLES
        halves.append(view)
    return pd.concat(halves, ignore_index=True)


def build(
    dads_path: Path,
    external_path: Path,
    g9_path: Path,
    output_dir: Path,
) -> dict:
    dads = _dads_rows(dads_path)
    external = _external_half_rows(external_path)
    g9 = _g9_rows(g9_path)

    # Validation is deliberately bounded and balanced so every training epoch
    # remains tractable.  The locked external dev_holdout is never admitted.
    train = pd.concat(
        [dads[dads["split"].eq("train")], external[external["split"].eq("train")], g9],
        ignore_index=True,
        sort=False,
    )
    dads_val = pd.concat(
        [
            _sample(dads[(dads["split"].eq("val")) & dads["label"].eq(label)], 4096, 420 + label)
            for label in (0, 1)
        ],
        ignore_index=True,
    )
    ext_val = pd.concat(
        [
            _sample(external[(external["split"].eq("val")) & external["label"].eq(label)], 4096, 430 + label)
            for label in (0, 1)
        ],
        ignore_index=True,
    )
    val = pd.concat([dads_val, ext_val], ignore_index=True, sort=False)
    dads_test = pd.concat(
        [
            _sample(dads[(dads["split"].eq("test")) & dads["label"].eq(label)], 4096, 440 + label)
            for label in (0, 1)
        ],
        ignore_index=True,
    )
    dads_test["split"] = "test"
    val["split"] = "val"
    train["split"] = "train"
    combined = pd.concat([train, val, dads_test], ignore_index=True, sort=False)

    required = [
        "split", "label", "cache_path", "cache_start_sample",
        "cache_end_sample", "source_path", "recording_group", "domain_bucket",
        "dataset_origin", "background_mix_eligible",
    ]
    if combined[required].isna().any().any():
        raise ValueError("G7-R4 manifest contains null required values")
    indexed = ~combined["dataset_origin"].eq("g9_mechanical_hard_negative")
    if combined.loc[indexed, "cache_index"].isna().any():
        raise ValueError("Indexed G7-R4 caches contain null cache_index values")
    if set(combined["split"]) != {"train", "val", "test"}:
        raise ValueError("G7-R4 manifest split roles changed")
    if "dev_holdout" in set(combined["split"].astype(str)):
        raise ValueError("Locked external dev_holdout entered G7-R4 development")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    combined.to_csv(manifest_path, index=False)
    counts = {
        split: {
            dataset: {str(k): int(v) for k, v in rows["label"].value_counts().sort_index().items()}
            for dataset, rows in split_rows.groupby("dataset_origin", dropna=False)
        }
        for split, split_rows in combined.groupby("split")
    }
    audit = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_rate": 16000,
        "clip_seconds": 0.5,
        "target_samples": TARGET_SAMPLES,
        "inputs": {
            "dads": {"path": str(dads_path), "sha256": file_sha256(dads_path)},
            "external": {"path": str(external_path), "sha256": file_sha256(external_path)},
            "g9": {"path": str(g9_path), "sha256": file_sha256(g9_path)},
        },
        "output": {
            "manifest": {"path": str(manifest_path), "sha256": file_sha256(manifest_path), "rows": len(combined)}
        },
        "counts": counts,
        "train_domain_buckets": {
            str(k): int(v) for k, v in train["domain_bucket"].value_counts().sort_index().items()
        },
        "external_one_second_policy": "two_nonoverlapping_half_second_views",
        "locked_datasets_read": [],
        "external_dev_holdout_read": False,
        "formal_training_started": False,
        "limitations": [
            "Kielce and TAU tune are consumed development data, not fresh final test data.",
            "The DADS test role is an already-consumed internal regression split.",
            "Fresh cross-dataset claims require the still-locked dev_holdout or a new dataset after candidate selection.",
        ],
    }
    audit_path = output_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare G7-R4 multidomain half-second manifest")
    parser.add_argument("--dads", type=Path, default=Path("artifacts/g7_leakage_fixed_v2/data/manifest.csv"))
    parser.add_argument("--external", type=Path, default=Path("artifacts/g14_domain_generalization/segment_cache/g14_segment_manifest.csv"))
    parser.add_argument("--g9", type=Path, default=Path("artifacts/g15_constrained_adaptation/p0_registry/g9_mechanical_hard_negative.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/g7_r4_cross_dataset/p0_manifest"))
    args = parser.parse_args()
    print(json.dumps(build(args.dads, args.external, args.g9, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
