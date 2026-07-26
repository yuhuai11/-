from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .config import load_config
from .data_firewall import file_sha256, reject_locked_path


PROTOCOL = "g14_kielce_grouped_manifest_v1"
ARCHIVE_PATTERN = re.compile(
    r"^(?P<rotor>X[46])_(?P<drone>D\d+)_(?P<vehicle>.+)\.zip$", re.IGNORECASE
)
MEMBER_PATTERN = re.compile(
    r"_(?P<height>\d+)m_(?P<distance>\d+)m_"
    r"(?P<date>\d{6})_+(?P<measurement>\d+)(?:_[^.]+)?\.wav$",
    re.IGNORECASE,
)
SPLIT_ORDER = {"train": 0, "tune": 1, "dev_holdout": 2}


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
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


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
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


def _resolve(root: Path, value: object, *, context: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    reject_locked_path(path, context=context)
    return path.resolve(strict=True)


def _resolve_locked_metadata(root: Path, value: object) -> Path:
    """Resolve an explicitly allowed archival registry without opening locked audio."""
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Archival metadata escapes project root: {resolved}")
    return resolved


def _read_hashes(path: Path) -> set[str]:
    values = {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if any(len(value) != 64 for value in values):
        raise ValueError(f"Malformed SHA256 registry: {path}")
    return values


def _read_dads_hashes(path: Path) -> set[str]:
    hashes = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "raw_audio_sha256" not in reader.fieldnames:
            raise ValueError("DADS registry lacks raw_audio_sha256")
        for row in reader:
            value = str(row["raw_audio_sha256"]).strip().lower()
            if value:
                hashes.add(value)
    return hashes


def _stream_sha256(handle: Any) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _load_full_audit(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "passed": True,
        "official_files_verified": int(expected["official_files"]),
        "zip_archives_verified": int(expected["archives"]),
        "eligible_primary_uav_wavs": int(expected["eligible_primary_uav_wavs"]),
        "integrity_passed": True,
        "zip_crc_passed": True,
        "zip_paths_safe": True,
    }
    mismatches = {
        key: {"expected": value, "observed": report.get(key)}
        for key, value in required.items()
        if report.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Full archive audit contract mismatch: {mismatches}")
    return report


def _split_sets(rows: Iterable[dict[str, Any]], field: str) -> dict[str, set[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        values[str(row["split"])].add(str(row[field]))
    return values


def prepare(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected Kielce manifest protocol")

    archive_dir = _resolve(root, config["archive_dir"], context="Kielce archive")
    audit_path = _resolve(root, config["full_archive_audit"], context="Kielce audit")
    g13_path = _resolve_locked_metadata(root, config["g13_consumed_hashes"])
    dads_path = _resolve(root, config["dads_raw_registry"], context="DADS registry")
    official_path = _resolve(
        root, config["official_md5_manifest"], context="Kielce MD5 manifest"
    )
    full_audit = _load_full_audit(audit_path, config["expected"])
    g13_hashes = _read_hashes(g13_path)
    dads_hashes = _read_dads_hashes(dads_path)
    speech_token = str(config["speech_token"]).lower()
    date_domains = config["date_domains"]

    archive_paths = sorted(archive_dir.glob("*.zip"))
    if len(archive_paths) != int(config["expected"]["archives"]):
        raise ValueError(f"Expected 17 ZIP archives, found {len(archive_paths)}")

    rows: list[dict[str, Any]] = []
    excluded_speech: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for archive_path in archive_paths:
        archive_match = ARCHIVE_PATTERN.match(archive_path.name)
        if not archive_match:
            raise ValueError(f"Unrecognized archive name: {archive_path.name}")
        archive_fields = archive_match.groupdict()
        drone_id = archive_fields["drone"].upper()
        rotor_layout = archive_fields["rotor"].upper()
        vehicle = archive_fields["vehicle"]
        member_dates = set()
        primary_in_archive = 0
        speech_in_archive = 0
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".wav"):
                    continue
                member = PurePosixPath(info.filename)
                if member.is_absolute() or ".." in member.parts:
                    raise ValueError(f"Unsafe member path: {info.filename}")
                match = MEMBER_PATTERN.search(member.name)
                if not match:
                    raise ValueError(f"Cannot parse WAV member: {info.filename}")
                fields = match.groupdict()
                date_code = fields["date"]
                member_dates.add(date_code)
                if date_code not in date_domains:
                    raise ValueError(f"Unregistered acquisition date: {date_code}")
                domain = date_domains[date_code]
                if speech_token in member.name.lower():
                    speech_in_archive += 1
                    excluded_speech.append(
                        {
                            "dataset": config["dataset"],
                            "archive_path": archive_path.relative_to(root).as_posix(),
                            "archive_member": info.filename,
                            "drone_id": drone_id,
                            "acquisition_date": domain["date_iso"],
                            "reason": "contains_speech",
                        }
                    )
                    continue
                primary_in_archive += 1
                with archive.open(info) as stream:
                    audio_sha256 = _stream_sha256(stream)
                rows.append(
                    {
                        "dataset": config["dataset"],
                        "archive_path": archive_path.relative_to(root).as_posix(),
                        "archive_member": info.filename,
                        "label": 1,
                        "split": domain["split"],
                        "source_group": (
                            f"{config['dataset']}:{drone_id}:{domain['date_iso']}"
                        ),
                        "drone_id": drone_id,
                        "vehicle": vehicle,
                        "rotor_layout": rotor_layout,
                        "acquisition_date": domain["date_iso"],
                        "location": domain["location"],
                        "height_m": int(fields["height"]),
                        "distance_m": int(fields["distance"]),
                        "measurement_id": int(fields["measurement"]),
                        "audio_sha256": audio_sha256,
                        "uncompressed_bytes": info.file_size,
                        "license": config["license"],
                    }
                )
        if len(member_dates) != 1:
            raise ValueError(f"Archive crosses acquisition dates: {archive_path.name}")
        date_code = next(iter(member_dates))
        domain = date_domains[date_code]
        source_rows.append(
            {
                "dataset": config["dataset"],
                "source_group": f"{config['dataset']}:{drone_id}:{domain['date_iso']}",
                "split": domain["split"],
                "drone_id": drone_id,
                "vehicle": vehicle,
                "rotor_layout": rotor_layout,
                "acquisition_date": domain["date_iso"],
                "location": domain["location"],
                "primary_uav_wavs": primary_in_archive,
                "speech_wavs_excluded": speech_in_archive,
                "archive_path": archive_path.relative_to(root).as_posix(),
            }
        )

    rows.sort(
        key=lambda row: (
            SPLIT_ORDER[row["split"]],
            int(str(row["drone_id"])[1:]),
            row["height_m"],
            row["distance_m"],
            row["measurement_id"],
            row["archive_member"],
        )
    )
    source_rows.sort(
        key=lambda row: (SPLIT_ORDER[row["split"]], int(str(row["drone_id"])[1:]))
    )
    excluded_speech.sort(
        key=lambda row: (int(str(row["drone_id"])[1:]), row["archive_member"])
    )

    expected_wavs = int(config["expected"]["eligible_primary_uav_wavs"])
    if len(rows) != expected_wavs:
        raise ValueError(f"Expected {expected_wavs} primary WAVs, found {len(rows)}")
    hash_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        hash_groups[str(row["audio_sha256"])].append(row)
    duplicates = {
        digest: values for digest, values in hash_groups.items() if len(values) > 1
    }
    g13_overlap = sorted(set(hash_groups) & g13_hashes)
    dads_overlap = sorted(set(hash_groups) & dads_hashes)
    if duplicates:
        raise ValueError(f"Exact duplicate Kielce WAV groups found: {len(duplicates)}")
    if g13_overlap:
        raise ValueError(f"Kielce/G13 exact audio overlap found: {len(g13_overlap)}")
    if dads_overlap:
        raise ValueError(f"Kielce/DADS exact audio overlap found: {len(dads_overlap)}")

    source_sets = _split_sets(rows, "source_group")
    date_sets = _split_sets(rows, "acquisition_date")
    location_sets = _split_sets(rows, "location")
    split_pairs = (("train", "tune"), ("train", "dev_holdout"), ("tune", "dev_holdout"))
    if any(source_sets[left] & source_sets[right] for left, right in split_pairs):
        raise ValueError("Source group overlap across splits")
    if any(date_sets[left] & date_sets[right] for left, right in split_pairs):
        raise ValueError("Acquisition date overlap across splits")
    if location_sets["dev_holdout"] & (
        location_sets["train"] | location_sets["tune"]
    ):
        raise ValueError("Development holdout location is not unseen")
    minimum_train = int(config["expected"]["train_source_groups_minimum"])
    if len(source_sets["train"]) < minimum_train:
        raise ValueError("Training UAV source-group minimum is not met")

    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    manifest_path = output_dir / "kielce_primary_uav_manifest.csv"
    source_path = output_dir / "kielce_source_registry.csv"
    speech_path = output_dir / "kielce_excluded_speech.csv"
    _atomic_write_csv(manifest_path, rows)
    _atomic_write_csv(source_path, source_rows)
    _atomic_write_csv(speech_path, excluded_speech)

    split_samples = Counter(str(row["split"]) for row in rows)
    split_sources = {
        split: len(source_sets[split]) for split in ("train", "tune", "dev_holdout")
    }
    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "eligible_primary_uav_wavs": len(rows),
        "speech_wavs_excluded": len(excluded_speech),
        "unique_audio_sha256": len(hash_groups),
        "exact_duplicate_hash_groups": len(duplicates),
        "g13_hashes_checked": len(g13_hashes),
        "exact_g13_audio_overlaps": len(g13_overlap),
        "dads_raw_hashes_checked": len(dads_hashes),
        "exact_dads_raw_audio_overlaps": len(dads_overlap),
        "split_samples": dict(sorted(split_samples.items())),
        "split_source_groups": split_sources,
        "split_dates": {
            split: sorted(date_sets[split])
            for split in ("train", "tune", "dev_holdout")
        },
        "split_locations": {
            split: sorted(location_sets[split])
            for split in ("train", "tune", "dev_holdout")
        },
        "source_group_overlap_across_splits": 0,
        "acquisition_date_overlap_across_splits": 0,
        "dev_holdout_location_unseen": True,
        "train_tune_location_overlap": sorted(
            location_sets["train"] & location_sets["tune"]
        ),
        "ready_for_background_merge": True,
        "ready_for_training": False,
        "reason_training_blocked": "tau_background_grouped_manifest_not_merged",
        "model_inference_run": False,
        "training_started": False,
        "audio_extracted_to_disk": False,
        "locked_dataset_audio_read": False,
        "inputs": {
            "config_sha256": file_sha256(config_path),
            "official_md5_manifest_sha256": file_sha256(official_path),
            "full_archive_audit_sha256": file_sha256(audit_path),
            "g13_hash_registry_sha256": file_sha256(g13_path),
            "dads_raw_registry_sha256": file_sha256(dads_path),
            "bound_full_archive_protocol": full_audit["protocol"],
        },
        "outputs": {
            "manifest": {
                "path": manifest_path.relative_to(root).as_posix(),
                "sha256": file_sha256(manifest_path),
                "rows": len(rows),
            },
            "source_registry": {
                "path": source_path.relative_to(root).as_posix(),
                "sha256": file_sha256(source_path),
                "rows": len(source_rows),
            },
            "excluded_speech": {
                "path": speech_path.relative_to(root).as_posix(),
                "sha256": file_sha256(speech_path),
                "rows": len(excluded_speech),
            },
        },
    }
    _atomic_write_json(output_dir / "kielce_manifest_audit.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a grouped Kielce UAV manifest and exact-audio firewall."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g14_kielce_manifest.yaml")
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    prepare(args.config, args.root)


if __name__ == "__main__":
    main()
