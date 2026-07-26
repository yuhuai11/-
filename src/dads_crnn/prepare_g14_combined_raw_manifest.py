from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .data_firewall import file_sha256, reject_locked_path


PROTOCOL = "g14_combined_raw_manifest_v1"
SPLITS = ("train", "tune", "dev_holdout")
SPLIT_ORDER = {value: index for index, value in enumerate(SPLITS)}


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    reject_locked_path(path, context="G14 combined raw manifest")
    return path.resolve(strict=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty input manifest: {path}")
    return rows


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _verify_audit(path: Path, manifest_path: Path, expected_protocol: str) -> dict[str, Any]:
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True or audit.get("protocol") != expected_protocol:
        raise ValueError(f"Input audit is not passed: {path}")
    output = audit.get("outputs", {}).get("manifest", {})
    if output.get("sha256") != file_sha256(manifest_path):
        raise ValueError(f"Input manifest no longer matches its audit: {manifest_path}")
    return audit


def _normalize_positive(row: dict[str, str]) -> dict[str, Any]:
    return {
        "dataset": row["dataset"],
        "archive_path": row["archive_path"],
        "archive_member": row["archive_member"],
        "label": 1,
        "split": row["split"],
        "source_group": row["source_group"],
        "audio_sha256": row["audio_sha256"],
        "uncompressed_bytes": int(row["uncompressed_bytes"]),
        "domain_type": "uav",
        "domain": row["location"],
        "subtype": row["vehicle"],
        "device": "OLYMPUS_LS11",
        "acquisition_date": row["acquisition_date"],
        "rotor_layout": row["rotor_layout"],
        "height_m": row["height_m"],
        "distance_m": row["distance_m"],
        "scene_label": "",
        "license": row["license"],
        "sampling_unit": "raw_recording",
    }


def _normalize_background(row: dict[str, str]) -> dict[str, Any]:
    return {
        "dataset": row["dataset"],
        "archive_path": row["archive_path"],
        "archive_member": row["archive_member"],
        "label": 0,
        "split": row["split"],
        "source_group": row["source_group"],
        "audio_sha256": row["audio_sha256"],
        "uncompressed_bytes": int(row["uncompressed_bytes"]),
        "domain_type": "background",
        "domain": row["city"],
        "subtype": row["scene_label"],
        "device": row["device"],
        "acquisition_date": "",
        "rotor_layout": "",
        "height_m": "",
        "distance_m": "",
        "scene_label": row["scene_label"],
        "license": row["license"],
        "sampling_unit": "raw_recording",
    }


def prepare(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected combined raw-manifest protocol")
    positive_path = _resolve(root, config["positive_manifest"])
    positive_audit_path = _resolve(root, config["positive_audit"])
    background_path = _resolve(root, config["background_manifest"])
    background_audit_path = _resolve(root, config["background_audit"])
    positive_audit = _verify_audit(
        positive_audit_path, positive_path, "g14_kielce_grouped_manifest_v1"
    )
    background_audit = _verify_audit(
        background_audit_path, background_path, "g14_tau_grouped_manifest_v1"
    )
    positives = [_normalize_positive(row) for row in _read_csv(positive_path)]
    backgrounds = [_normalize_background(row) for row in _read_csv(background_path)]
    expected = config["expected"]
    if len(positives) != int(expected["positive_rows"]):
        raise ValueError("Positive row-count contract mismatch")
    if len(backgrounds) != int(expected["background_rows"]):
        raise ValueError("Background row-count contract mismatch")
    rows = positives + backgrounds
    if len(rows) != int(expected["total_rows"]):
        raise ValueError("Combined row-count contract mismatch")

    hashes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        hashes[row["audio_sha256"]].append(row)
    duplicates = {key: value for key, value in hashes.items() if len(value) > 1}
    if duplicates:
        raise ValueError(f"Combined raw manifest has duplicate hashes: {len(duplicates)}")

    source_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source_splits[row["source_group"]].add(row["split"])
    crossing_sources = {
        source: splits for source, splits in source_splits.items() if len(splits) > 1
    }
    if crossing_sources:
        raise ValueError(f"Source groups cross splits: {len(crossing_sources)}")
    observed_splits = sorted({row["split"] for row in rows}, key=SPLIT_ORDER.get)
    if observed_splits != list(config["expected"]["splits"]):
        raise ValueError(f"Combined split contract mismatch: {observed_splits}")

    rows.sort(
        key=lambda row: (
            SPLIT_ORDER[row["split"]],
            -int(row["label"]),
            row["dataset"],
            row["source_group"],
            row["archive_member"],
        )
    )
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    manifest_path = output_dir / "g14_combined_raw_manifest.csv"
    _atomic_csv(manifest_path, rows)

    sample_counts = Counter((row["split"], int(row["label"])) for row in rows)
    source_counts = Counter(
        (next(iter(splits)), int(next(row["label"] for row in rows if row["source_group"] == source)))
        for source, splits in source_splits.items()
    )
    split_summary = {}
    for split in SPLITS:
        positive = sample_counts[(split, 1)]
        background = sample_counts[(split, 0)]
        split_summary[split] = {
            "positive_raw_recordings": positive,
            "background_raw_recordings": background,
            "positive_source_groups": source_counts[(split, 1)],
            "background_source_groups": source_counts[(split, 0)],
            "raw_background_to_positive_ratio": background / positive,
        }

    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "unique_audio_sha256": len(hashes),
        "exact_duplicate_hash_groups": len(duplicates),
        "source_groups": len(source_splits),
        "source_group_overlap_across_splits": len(crossing_sources),
        "split_summary": split_summary,
        "ready_for_controlled_decode": True,
        "ready_for_training": False,
        "training_blockers": [
            "zip_member_decode_cache_not_prepared",
            "fixed_length_segmentation_not_prepared",
            "source_balanced_sampler_not_configured",
        ],
        "model_inference_run": False,
        "training_started": False,
        "audio_extracted_to_disk": False,
        "inputs": {
            "config_sha256": file_sha256(config_path),
            "positive_manifest_sha256": file_sha256(positive_path),
            "positive_audit_sha256": file_sha256(positive_audit_path),
            "background_manifest_sha256": file_sha256(background_path),
            "background_audit_sha256": file_sha256(background_audit_path),
            "positive_protocol": positive_audit["protocol"],
            "background_protocol": background_audit["protocol"],
        },
        "outputs": {
            "manifest": {
                "path": manifest_path.relative_to(root).as_posix(),
                "sha256": file_sha256(manifest_path),
                "rows": len(rows),
            }
        },
    }
    _atomic_json(output_dir / "g14_combined_raw_audit.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge the isolated G14 UAV and background raw manifests."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g14_combined_raw_manifest.yaml")
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    prepare(args.config, args.root)


if __name__ == "__main__":
    main()
