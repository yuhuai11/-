from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .config import load_config
from .data_firewall import file_sha256, reject_locked_path


PROTOCOL = "g14_tau_grouped_manifest_v1"
SPLITS = ("train", "tune", "dev_holdout")
SPLIT_ORDER = {value: index for index, value in enumerate(SPLITS)}


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Empty CSV output: {path}")
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


def _resolve(root: Path, value: object, *, allow_locked_metadata: bool = False) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path escapes project root: {resolved}")
    if not allow_locked_metadata:
        reject_locked_path(resolved, context="G14 TAU grouped manifest")
    return resolved


def _hash_registry(path: Path, column: str | None = None) -> set[str]:
    if column is None:
        values = {
            line.strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    else:
        values = set()
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or column not in reader.fieldnames:
                raise ValueError(f"{path} lacks {column}")
            for row in reader:
                value = str(row[column]).strip().lower()
                if value:
                    values.add(value)
    if any(len(value) != 64 for value in values):
        raise ValueError(f"Malformed SHA256 registry: {path}")
    return values


def _sha256_stream(handle: Any) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def prepare(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected TAU grouped-manifest protocol")

    registry_path = _resolve(root, config["intake_registry"])
    audit_path = _resolve(root, config["intake_audit"])
    g13_path = _resolve(
        root, config["g13_consumed_hashes"], allow_locked_metadata=True
    )
    dads_path = _resolve(root, config["dads_raw_registry"])
    kielce_path = _resolve(root, config["kielce_manifest"])
    intake_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = config["expected"]
    if not (
        intake_audit.get("passed") is True
        and intake_audit.get("integrity_passed") is True
        and intake_audit.get("coverage_passed") is True
        and intake_audit.get("samples") == int(expected["samples"])
        and intake_audit.get("source_groups") == int(expected["source_groups"])
    ):
        raise ValueError("TAU intake audit contract mismatch")

    registry = _load_rows(registry_path)
    if len(registry) != int(expected["samples"]):
        raise ValueError("TAU intake registry row count mismatch")
    by_filename = {row["filename"]: row for row in registry}
    if len(by_filename) != len(registry):
        raise ValueError("TAU intake registry has duplicate filenames")

    archive_members: dict[str, tuple[Path, str, int]] = {}
    for archive_settings in config["archives"]:
        archive_path = _resolve(root, archive_settings["path"])
        count = 0
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".wav"):
                    continue
                member = PurePosixPath(info.filename)
                if member.is_absolute() or ".." in member.parts:
                    raise ValueError(f"Unsafe TAU ZIP member: {info.filename}")
                if "audio/" not in info.filename:
                    raise ValueError(f"TAU member lacks audio prefix: {info.filename}")
                filename = info.filename[info.filename.index("audio/") :]
                if filename in archive_members:
                    raise ValueError(f"TAU member occurs in multiple parts: {filename}")
                archive_members[filename] = (archive_path, info.filename, info.file_size)
                count += 1
        if count != int(archive_settings["expected_wavs"]):
            raise ValueError(
                f"TAU part {archive_settings['part']} WAV count mismatch: {count}"
            )
    if set(archive_members) != set(by_filename):
        raise ValueError("TAU registry and archive members do not match exactly")

    g13_hashes = _hash_registry(g13_path)
    dads_hashes = _hash_registry(dads_path, "raw_audio_sha256")
    kielce_hashes = _hash_registry(kielce_path, "audio_sha256")
    city_splits = {str(key): str(value) for key, value in config["city_splits"].items()}

    rows = []
    archive_handles: dict[Path, zipfile.ZipFile] = {}
    try:
        for filename in sorted(by_filename):
            source = by_filename[filename]
            city = str(source["city"])
            if city not in city_splits:
                raise ValueError(f"TAU city lacks split assignment: {city}")
            split = city_splits[city]
            archive_path, archive_member, uncompressed_bytes = archive_members[filename]
            if archive_path not in archive_handles:
                archive_handles[archive_path] = zipfile.ZipFile(archive_path)
            with archive_handles[archive_path].open(archive_member) as stream:
                audio_sha256 = _sha256_stream(stream)
            rows.append(
                {
                    "dataset": config["dataset"],
                    "archive_path": archive_path.relative_to(root).as_posix(),
                    "archive_member": archive_member,
                    "label": 0,
                    "split": split,
                    "source_group": source["source_group"],
                    "city": city,
                    "scene_label": source["scene_label"],
                    "identifier": source["identifier"],
                    "device": source["device"],
                    "audio_sha256": audio_sha256,
                    "uncompressed_bytes": uncompressed_bytes,
                    "license": config["license"],
                }
            )
    finally:
        for handle in archive_handles.values():
            handle.close()

    rows.sort(
        key=lambda row: (
            SPLIT_ORDER[row["split"]],
            row["city"],
            row["source_group"],
            row["archive_member"],
        )
    )
    groups_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups_by_hash[row["audio_sha256"]].append(row)
    duplicates = {
        digest: values for digest, values in groups_by_hash.items() if len(values) > 1
    }
    g13_overlap = sorted(set(groups_by_hash) & g13_hashes)
    dads_overlap = sorted(set(groups_by_hash) & dads_hashes)
    kielce_overlap = sorted(set(groups_by_hash) & kielce_hashes)
    if duplicates:
        raise ValueError(f"Exact duplicate TAU WAV groups found: {len(duplicates)}")
    if g13_overlap or dads_overlap or kielce_overlap:
        raise ValueError(
            "TAU exact-audio firewall failed: "
            f"G13={len(g13_overlap)}, DADS={len(dads_overlap)}, "
            f"Kielce={len(kielce_overlap)}"
        )

    split_sources: dict[str, set[str]] = defaultdict(set)
    split_cities: dict[str, set[str]] = defaultdict(set)
    split_scenes: dict[str, set[str]] = defaultdict(set)
    split_devices: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = row["split"]
        split_sources[split].add(row["source_group"])
        split_cities[split].add(row["city"])
        split_scenes[split].add(row["scene_label"])
        split_devices[split].add(row["device"])
    for left, right in (
        ("train", "tune"),
        ("train", "dev_holdout"),
        ("tune", "dev_holdout"),
    ):
        if split_sources[left] & split_sources[right]:
            raise ValueError("TAU source group crosses splits")
        if split_cities[left] & split_cities[right]:
            raise ValueError("TAU city crosses splits")
    for split in SPLITS:
        minimum = int(expected[f"{split}_source_groups_minimum"])
        if len(split_sources[split]) < minimum:
            raise ValueError(f"TAU {split} source-group minimum is not met")
        required_scenes = set(expected[f"{split}_scenes"])
        if split_scenes[split] != required_scenes:
            raise ValueError(
                f"TAU {split} scene contract mismatch: {sorted(split_scenes[split])}"
            )
        if len(split_devices[split]) != int(expected["unique_devices"]):
            raise ValueError(f"TAU {split} does not cover every device")

    source_rows = []
    for (split, source_group), values in sorted(
        (
            (key, values)
            for key, values in _group_rows(rows, ("split", "source_group")).items()
        ),
        key=lambda item: (SPLIT_ORDER[item[0][0]], item[0][1]),
    ):
        source_rows.append(
            {
                "dataset": config["dataset"],
                "source_group": source_group,
                "split": split,
                "city": values[0]["city"],
                "scenes": "|".join(sorted({row["scene_label"] for row in values})),
                "devices": "|".join(sorted({row["device"] for row in values})),
                "samples": len(values),
            }
        )

    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    manifest_path = output_dir / "tau_background_manifest.csv"
    source_path = output_dir / "tau_source_registry.csv"
    _atomic_csv(manifest_path, rows)
    _atomic_csv(source_path, source_rows)
    sample_counts = Counter(row["split"] for row in rows)
    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "samples": len(rows),
        "unique_audio_sha256": len(groups_by_hash),
        "exact_duplicate_hash_groups": len(duplicates),
        "exact_g13_audio_overlaps": len(g13_overlap),
        "exact_dads_raw_audio_overlaps": len(dads_overlap),
        "exact_kielce_audio_overlaps": len(kielce_overlap),
        "split_samples": dict(sorted(sample_counts.items())),
        "split_source_groups": {
            split: len(split_sources[split]) for split in SPLITS
        },
        "split_cities": {split: sorted(split_cities[split]) for split in SPLITS},
        "split_scenes": {split: sorted(split_scenes[split]) for split in SPLITS},
        "split_devices": {split: sorted(split_devices[split]) for split in SPLITS},
        "source_group_overlap_across_splits": 0,
        "city_overlap_across_splits": 0,
        "ready_for_kielce_merge": True,
        "ready_for_training": False,
        "reason_training_blocked": "positive_and_background_manifests_not_merged",
        "model_inference_run": False,
        "training_started": False,
        "audio_extracted_to_disk": False,
        "locked_dataset_audio_read": False,
        "inputs": {
            "config_sha256": file_sha256(config_path),
            "intake_registry_sha256": file_sha256(registry_path),
            "intake_audit_sha256": file_sha256(audit_path),
            "g13_hash_registry_sha256": file_sha256(g13_path),
            "dads_raw_registry_sha256": file_sha256(dads_path),
            "kielce_manifest_sha256": file_sha256(kielce_path),
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
        },
    }
    _atomic_json(output_dir / "tau_manifest_audit.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def _group_rows(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a city-isolated TAU background manifest."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g14_tau_grouped_manifest.yaml")
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    prepare(args.config, args.root)


if __name__ == "__main__":
    main()
