from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from .audio import decode_wav_bytes
from .config import ensure_dirs, load_config
from .prepare_external_manifests import dads_audio_hashes


ALGORITHM = "g13_external_confirmation_intake_v1"
REGISTRY_COLUMNS = {
    "source_group",
    "label",
    "acquisition_id",
    "provenance",
    "independent_from_existing",
    "license",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def validate_registry(path: Path, labels: set[int], require_new: bool) -> pd.DataFrame:
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted(REGISTRY_COLUMNS - set(rows.columns))
    if missing:
        raise ValueError(f"G13 source registry is missing columns: {missing}")
    rows = rows[list(sorted(REGISTRY_COLUMNS))].copy()
    rows["source_group"] = rows["source_group"].str.strip()
    rows["label"] = pd.to_numeric(rows["label"], errors="raise").astype(int)
    if rows.empty or (rows["source_group"] == "").any():
        raise ValueError("G13 source registry requires non-empty source groups")
    if set(rows["label"]) - labels:
        raise ValueError("G13 source registry contains an unsupported label")
    if rows["source_group"].duplicated().any():
        raise ValueError("G13 source groups must be globally unique")
    for field in ("acquisition_id", "provenance", "license"):
        if (rows[field].str.strip() == "").any():
            raise ValueError(f"G13 source registry requires non-empty {field}")
    if require_new and not rows["independent_from_existing"].map(_truthy).all():
        raise ValueError("Every G13 source must be declared independent from existing data")
    return rows.sort_values("source_group").reset_index(drop=True)


def build_manifest(
    root: Path,
    dataset_name: str,
    label_folders: dict[str, int],
    registry: pd.DataFrame,
) -> pd.DataFrame:
    registry_labels = dict(zip(registry["source_group"], registry["label"], strict=True))
    rows: list[dict[str, Any]] = []
    for folder, label in label_folders.items():
        class_root = root / folder
        if not class_root.is_dir():
            raise FileNotFoundError(f"Missing G13 label directory: {class_root}")
        for path in sorted(class_root.rglob("*.wav")):
            relative = path.relative_to(class_root)
            if len(relative.parts) < 2:
                raise ValueError(
                    f"G13 WAV must be stored under <label>/<source_group>/: {path}"
                )
            source_group = relative.parts[0]
            if source_group not in registry_labels:
                raise ValueError(f"Unregistered G13 source group: {source_group}")
            if int(registry_labels[source_group]) != int(label):
                raise ValueError(f"G13 directory/registry label mismatch: {source_group}")
            condition = relative.parts[1] if len(relative.parts) >= 3 else "unspecified"
            wav_bytes = path.read_bytes()
            audio, sample_rate = decode_wav_bytes(wav_bytes)
            rows.append(
                {
                    "dataset": dataset_name,
                    "path": path.resolve().as_posix(),
                    "filename": path.name,
                    "label": int(label),
                    "source_group": source_group,
                    "condition": condition,
                    "sample_rate": int(sample_rate),
                    "samples": int(audio.size),
                    "duration_seconds": float(audio.size / sample_rate),
                    "sha256": hashlib.sha256(wav_bytes).hexdigest(),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No G13 WAV files found under {root}")
    manifest = pd.DataFrame(rows).sort_values(["label", "source_group", "path"])
    observed_sources = set(manifest["source_group"])
    registered_sources = set(registry["source_group"])
    if observed_sources != registered_sources:
        missing = sorted(registered_sources - observed_sources)
        extra = sorted(observed_sources - registered_sources)
        raise ValueError(f"G13 registry/audio source mismatch; missing={missing}, extra={extra}")
    return manifest.reset_index(drop=True)


def reference_hashes(paths: list[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    hashes: set[str] = set()
    audits = []
    for path in paths:
        frame = pd.read_csv(path, usecols=lambda column: column in {"sha256", "cache_sha256"}, low_memory=False)
        if frame.empty and not set(frame.columns):
            raise ValueError(f"Reference manifest has no hash column: {path}")
        local: set[str] = set()
        for column in frame.columns:
            local.update(
                value
                for value in frame[column].fillna("").astype(str).str.strip()
                if len(value) == 64
            )
        hashes.update(local)
        audits.append(
            {
                "path": path.as_posix(),
                "sha256": sha256(path),
                "hashes": len(local),
                "read_for_hash_overlap_only": True,
            }
        )
    return hashes, audits


def audit_manifest(
    manifest: pd.DataFrame,
    registry: pd.DataFrame,
    requirements: dict[str, Any],
    known_hashes: set[str],
    dads_hashes: set[str],
) -> dict[str, Any]:
    label_counts = manifest["label"].value_counts().sort_index()
    source_counts = manifest.groupby(["label", "source_group"]).size()
    duplicate_rows = int(manifest["sha256"].duplicated(keep=False).sum())
    known_overlaps = int(manifest["sha256"].isin(known_hashes).sum())
    dads_overlaps = int(manifest["sha256"].isin(dads_hashes).sum())
    checks = [
        {
            "name": "minimum_samples_per_label",
            "value": int(label_counts.min()),
            "minimum": int(requirements["minimum_samples_per_label"]),
            "passed": int(label_counts.min()) >= int(requirements["minimum_samples_per_label"]),
        },
        {
            "name": "minimum_source_groups_per_label",
            "value": int(manifest.groupby("label")["source_group"].nunique().min()),
            "minimum": int(requirements["minimum_source_groups_per_label"]),
            "passed": int(manifest.groupby("label")["source_group"].nunique().min())
            >= int(requirements["minimum_source_groups_per_label"]),
        },
        {
            "name": "minimum_samples_per_source_group",
            "value": int(source_counts.min()),
            "minimum": int(requirements["minimum_samples_per_source_group"]),
            "passed": int(source_counts.min()) >= int(requirements["minimum_samples_per_source_group"]),
        },
        {
            "name": "duration_min",
            "value": float(manifest["duration_seconds"].min()),
            "minimum": float(requirements["duration_seconds_min"]),
            "passed": float(manifest["duration_seconds"].min())
            >= float(requirements["duration_seconds_min"]),
        },
        {
            "name": "duration_max",
            "value": float(manifest["duration_seconds"].max()),
            "maximum": float(requirements["duration_seconds_max"]),
            "passed": float(manifest["duration_seconds"].max())
            <= float(requirements["duration_seconds_max"]),
        },
        {"name": "unique_audio_sha256", "value": duplicate_rows, "maximum": 0, "passed": duplicate_rows == 0},
        {"name": "known_manifest_overlap", "value": known_overlaps, "maximum": 0, "passed": known_overlaps == 0},
        {"name": "dads_audio_overlap", "value": dads_overlaps, "maximum": 0, "passed": dads_overlaps == 0},
    ]
    return {
        "passed": all(item["passed"] for item in checks),
        "samples": int(len(manifest)),
        "unique_sha256": int(manifest["sha256"].nunique()),
        "label_counts": {str(key): int(value) for key, value in label_counts.items()},
        "source_groups_by_label": {
            str(key): int(value)
            for key, value in manifest.groupby("label")["source_group"].nunique().items()
        },
        "sample_rates": sorted(int(value) for value in manifest["sample_rate"].unique()),
        "registry_sources": int(len(registry)),
        "checks": checks,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dirs(path.parent)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and audit a new G13 external confirmation set")
    parser.add_argument("--config", type=Path, default=Path("configs/g13_external_confirmation_intake.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("algorithm") != ALGORITHM:
        raise ValueError(f"G13 algorithm must be {ALGORITHM}")
    dataset = config["dataset"]
    labels = {str(key): int(value) for key, value in dataset["labels"].items()}
    if sorted(labels.values()) != [0, 1]:
        raise ValueError("G13 requires exactly binary labels 0 and 1")
    root = Path(dataset["root"])
    registry_path = Path(dataset["source_registry"])
    registry = validate_registry(
        registry_path,
        set(labels.values()),
        bool(config["requirements"]["require_new_source_declaration"]),
    )
    manifest = build_manifest(root, str(dataset["name"]), labels, registry)
    reference_paths = [Path(value) for value in config["overlap_audit"]["reference_manifests"]]
    known_hashes, reference_audits = reference_hashes(reference_paths)
    parquet_dir = Path(config["overlap_audit"]["dads_parquet_dir"])
    dads_hash_set = dads_audio_hashes(parquet_dir)
    summary = audit_manifest(
        manifest, registry, config["requirements"], known_hashes, dads_hash_set
    )
    manifest_path = Path(config["outputs"]["manifest"])
    audit_path = Path(config["outputs"]["audit"])
    if manifest_path.exists() or audit_path.exists():
        raise FileExistsError("G13 intake artifacts already exist; refusing overwrite")
    ensure_dirs(manifest_path.parent)
    manifest.to_csv(manifest_path, index=False)
    report = {
        "algorithm": ALGORITHM,
        **summary,
        "inputs": {
            "config": {"path": args.config.as_posix(), "sha256": sha256(args.config)},
            "protocol_document": {
                "path": str(config["protocol_document"]),
                "sha256": sha256(Path(config["protocol_document"])),
            },
            "source_registry": {"path": registry_path.as_posix(), "sha256": sha256(registry_path)},
            "dads_parquet_dir": parquet_dir.as_posix(),
            "reference_manifests": reference_audits,
        },
        "outputs": {"manifest": manifest_path.as_posix()},
        "historical_final_manifests_read_for_hash_overlap_only": [
            item["path"]
            for item in reference_audits
            if "unseen_manifest" in item["path"] or "real_world_manifest" in item["path"]
        ],
        "model_predictions_read": False,
    }
    _atomic_json(audit_path, report)
    print(json.dumps({"passed": report["passed"], "samples": report["samples"], "checks": report["checks"]}, indent=2))
    if not report["passed"]:
        raise RuntimeError("G13 intake audit failed")


if __name__ == "__main__":
    main()
