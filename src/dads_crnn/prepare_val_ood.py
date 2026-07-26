from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ensure_dirs, load_config
from .external_data import build_external_manifest
from .prepare_external_manifests import dads_audio_hashes


def _attach_source_metadata(manifest: pd.DataFrame, metadata_path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    metadata = metadata[metadata["split"] == "val_ood"].copy()
    metadata["filename"] = metadata["filename"].map(lambda value: Path(str(value)).name)
    if metadata["filename"].duplicated().any():
        raise ValueError("val_ood metadata contains duplicate filenames")
    keep = [
        "filename",
        "class",
        "uav_family",
        "uav_group",
        "uav_source",
        "background_source",
        "target_snr_db",
        "achieved_snr_db",
        "kind",
        "target_rms_dbfs",
        "achieved_rms_dbfs",
    ]
    merged = manifest.drop(columns=["source_group", "condition"]).merge(
        metadata[keep], on="filename", how="left", validate="one_to_one"
    )
    if merged["kind"].isna().any():
        missing = merged.loc[merged["kind"].isna(), "filename"].head().tolist()
        raise ValueError(f"Missing source metadata for val_ood files: {missing}")
    expected_labels = merged["class"].map({"Background": 0, "UAV": 1})
    if expected_labels.isna().any() or not expected_labels.astype(int).equals(merged["label"].astype(int)):
        raise ValueError("Directory labels and val_ood metadata classes disagree")
    merged["source_group"] = np.where(
        merged["label"] == 1,
        merged["uav_group"],
        merged["background_source"].map(lambda value: Path(str(value)).stem),
    )
    merged["condition"] = np.where(
        merged["kind"] == "uav_background_mix",
        merged["target_snr_db"].map(lambda value: f"snr_{value:+.0f}_db"),
        merged["kind"],
    )
    return merged.sort_values(["label", "source_group", "path"]).reset_index(drop=True)


def _source_disjoint_split(
    manifest: pd.DataFrame, *, tune_fraction: float, seed: int
) -> pd.DataFrame:
    if not 0.0 < tune_fraction < 1.0:
        raise ValueError("tune_fraction must be between zero and one")
    rng = np.random.default_rng(seed)
    result = manifest.copy()
    result["ood_split"] = ""
    for label in (0, 1):
        label_rows = result[result["label"] == label]
        counts = label_rows.groupby("source_group").size().to_dict()
        groups = list(counts)
        rng.shuffle(groups)
        target = len(label_rows) * tune_fraction
        tune_groups: set[str] = set()
        tune_count = 0
        for group in sorted(groups, key=lambda item: counts[item], reverse=True):
            if abs((tune_count + counts[group]) - target) <= abs(tune_count - target):
                tune_groups.add(group)
                tune_count += counts[group]
        mask = (result["label"] == label) & result["source_group"].isin(tune_groups)
        result.loc[mask, "ood_split"] = "tune"
        result.loc[(result["label"] == label) & ~mask, "ood_split"] = "holdout"
    return result


def _hash_audit(
    manifest: pd.DataFrame,
    dads_hashes: set[str],
    final_manifests: list[Path],
) -> dict:
    final_hashes: set[str] = set()
    missing_final_manifests = []
    for path in final_manifests:
        if not path.exists():
            missing_final_manifests.append(path.as_posix())
            continue
        rows = pd.read_csv(path, usecols=["sha256"])
        final_hashes.update(rows["sha256"].dropna().astype(str))
    grouped = manifest.groupby("sha256").agg(rows=("sha256", "size"), labels=("label", "nunique"))
    return {
        "unique_hashes": int(manifest["sha256"].nunique()),
        "duplicate_extra_rows": int(len(manifest) - manifest["sha256"].nunique()),
        "cross_label_hash_groups": int((grouped["labels"] > 1).sum()),
        "exact_dads_overlaps": int(manifest["sha256"].isin(dads_hashes).sum()),
        "exact_final_test_overlaps": int(manifest["sha256"].isin(final_hashes).sum()),
        "missing_final_manifests": missing_final_manifests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and audit a source-disjoint OOD validation set")
    parser.add_argument("--config", default="configs/val_ood.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    audit_cfg = config["audit"]
    output_dir = Path(config["output_dir"])
    manifest_dir = output_dir / "manifests"
    ensure_dirs(manifest_dir)

    manifest = build_external_manifest(
        str(dataset_cfg["name"]),
        Path(dataset_cfg["root"]),
        {str(key): int(value) for key, value in dataset_cfg["labels"].items()},
        hash_files=True,
    )
    manifest = _attach_source_metadata(manifest, Path(dataset_cfg["metadata"]))
    manifest = _source_disjoint_split(
        manifest,
        tune_fraction=float(config["split"]["tune_fraction"]),
        seed=int(config["split"]["seed"]),
    )

    print("Hashing DADS audio for an independent overlap audit...", flush=True)
    dads_hashes = dads_audio_hashes(Path(audit_cfg["dads_parquet_dir"]))
    hash_audit = _hash_audit(
        manifest,
        dads_hashes,
        [Path(value) for value in audit_cfg["final_test_manifests"]],
    )
    source_text = "\n".join(
        manifest[["uav_source", "background_source"]].fillna("").astype(str).to_numpy().ravel()
    ).lower()
    forbidden_matches = [
        token for token in audit_cfg["forbidden_source_tokens"] if str(token).lower() in source_text
    ]
    split_counts = (
        manifest.groupby(["ood_split", "label"]).size().unstack(fill_value=0).sort_index()
    )
    tune_groups = set(manifest.loc[manifest["ood_split"] == "tune", "source_group"])
    holdout_groups = set(manifest.loc[manifest["ood_split"] == "holdout", "source_group"])
    report = {
        "passed": bool(
            not hash_audit["missing_final_manifests"]
            and hash_audit["cross_label_hash_groups"] == 0
            and hash_audit["exact_dads_overlaps"] == 0
            and hash_audit["exact_final_test_overlaps"] == 0
            and not forbidden_matches
            and not (tune_groups & holdout_groups)
        ),
        "samples": int(len(manifest)),
        "label_counts": {str(k): int(v) for k, v in manifest["label"].value_counts().sort_index().items()},
        "split_counts": {
            split: {str(label): int(value) for label, value in row.items()}
            for split, row in split_counts.to_dict(orient="index").items()
        },
        "uav_recording_groups": int(manifest.loc[manifest["label"] == 1, "source_group"].nunique()),
        "background_sources": int(manifest.loc[manifest["label"] == 0, "source_group"].nunique()),
        "source_group_overlap_between_splits": int(len(tune_groups & holdout_groups)),
        "forbidden_source_matches": forbidden_matches,
        **hash_audit,
    }
    manifest.to_csv(manifest_dir / "val_ood_manifest.csv", index=False)
    manifest[manifest["ood_split"] == "tune"].to_csv(
        manifest_dir / "val_ood_tune_manifest.csv", index=False
    )
    manifest[manifest["ood_split"] == "holdout"].to_csv(
        manifest_dir / "val_ood_holdout_manifest.csv", index=False
    )
    (manifest_dir / "val_ood_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("val_ood audit failed; inspect val_ood_audit.json")


if __name__ == "__main__":
    main()
