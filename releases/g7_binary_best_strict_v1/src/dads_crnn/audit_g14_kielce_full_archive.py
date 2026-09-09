from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .data_firewall import file_sha256, reject_locked_path


PROTOCOL = "g14_kielce_17_uav_full_archive_audit_v1"
SPEECH_TOKEN = "sekwencja"
DRONE_PATTERN = re.compile(r"_(D\d+)_", re.IGNORECASE)
DATE_PATTERN = re.compile(r"20\d{2}[-_.]?\d{2}[-_.]?\d{2}|(?<!\d)\d{6}(?!\d)")
GEOMETRY_PATTERN = re.compile(r"_(\d+)m_(\d+)m_", re.IGNORECASE)


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_expected(path: Path, root: Path) -> dict[Path, str]:
    expected: dict[Path, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            digest, relative = line.split(maxsplit=1)
        except ValueError as error:
            raise ValueError(f"Malformed MD5 manifest line {line_number}") from error
        candidate = (root / relative.strip()).resolve(strict=False)
        reject_locked_path(candidate, context="G14 Kielce official archive")
        expected[candidate] = digest.lower()
    if len(expected) != 19:
        raise ValueError(f"Expected 19 official files, found {len(expected)}")
    return expected


def _safe_members(handle: zipfile.ZipFile, archive_name: str) -> list[zipfile.ZipInfo]:
    members = []
    for info in handle.infolist():
        member = PurePosixPath(info.filename)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"Unsafe ZIP member in {archive_name}: {info.filename}")
        if not info.is_dir():
            members.append(info)
    bad_member = handle.testzip()
    if bad_member is not None:
        raise ValueError(f"ZIP CRC failure in {archive_name}: {bad_member}")
    return members


def audit(root: Path, manifest_path: Path, output_path: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    expected = _load_expected(manifest_path, root)

    file_checks = []
    for path, expected_md5 in expected.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing official file: {path}")
        observed_md5 = _md5(path)
        if observed_md5 != expected_md5:
            raise ValueError(f"MD5 mismatch: {path.name}")
        file_checks.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "md5": observed_md5,
                "passed": True,
            }
        )

    archives = sorted(path for path in expected if path.suffix.lower() == ".zip")
    if len(archives) != 17:
        raise ValueError(f"Expected 17 UAV ZIP archives, found {len(archives)}")

    formats: Counter[str] = Counter()
    dates: Counter[str] = Counter()
    geometry: Counter[str] = Counter()
    archive_rows = []
    total_wavs = 0
    total_speech = 0
    total_primary = 0
    drone_ids = set()
    for archive in archives:
        drone_match = DRONE_PATTERN.search(f"_{archive.stem}_")
        if not drone_match:
            raise ValueError(f"Cannot parse drone ID: {archive.name}")
        drone_id = drone_match.group(1).upper()
        drone_ids.add(drone_id)
        with zipfile.ZipFile(archive) as handle:
            members = _safe_members(handle, archive.name)

        wavs = [member for member in members if member.filename.lower().endswith(".wav")]
        speech = [
            member for member in wavs if SPEECH_TOKEN in member.filename.lower()
        ]
        primary = [
            member for member in wavs if SPEECH_TOKEN not in member.filename.lower()
        ]
        parsed_geometry = 0
        parsed_dates = set()
        for member in members:
            suffix = PurePosixPath(member.filename).suffix.lower() or "[no_suffix]"
            formats[suffix] += 1
        for member in primary:
            match = GEOMETRY_PATTERN.search(PurePosixPath(member.filename).name)
            if match:
                parsed_geometry += 1
                geometry[f"{match.group(1)}m_height_{match.group(2)}m_distance"] += 1
            date_match = DATE_PATTERN.search(PurePosixPath(member.filename).name)
            if date_match:
                parsed_dates.add(date_match.group(0))
                dates[date_match.group(0)] += 1

        if parsed_geometry != len(primary) or not parsed_dates:
            raise ValueError(f"Filename metadata parse failure in {archive.name}")
        archive_rows.append(
            {
                "archive": archive.name,
                "drone_id": drone_id,
                "members": len(members),
                "wav_files": len(wavs),
                "speech_wavs_excluded": len(speech),
                "eligible_primary_uav_wavs": len(primary),
                "primary_wav_count_deviation_from_45": len(primary) - 45,
                "acquisition_dates": sorted(parsed_dates),
                "geometry_parsed": parsed_geometry,
                "crc_passed": True,
                "safe_paths": True,
            }
        )
        total_wavs += len(wavs)
        total_speech += len(speech)
        total_primary += len(primary)

    expected_ids = {f"D{value}" for value in range(1, 18)}
    if drone_ids != expected_ids:
        raise ValueError(
            f"Physical UAV IDs are incomplete: missing={sorted(expected_ids - drone_ids)}"
        )
    if total_primary != 765:
        raise ValueError(f"Expected 765 primary UAV WAVs, found {total_primary}")

    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_files_verified": len(file_checks),
        "zip_archives_verified": len(archives),
        "physical_uav_ids": sorted(drone_ids, key=lambda value: int(value[1:])),
        "wav_files": total_wavs,
        "eligible_primary_uav_wavs": total_primary,
        "speech_wavs_excluded": total_speech,
        "acquisition_dates": dict(sorted(dates.items())),
        "geometry": dict(sorted(geometry.items())),
        "archive_formats": dict(sorted(formats.items())),
        "archives": archive_rows,
        "file_checks": sorted(file_checks, key=lambda row: row["name"]),
        "integrity_passed": True,
        "zip_crc_passed": True,
        "zip_paths_safe": True,
        "ready_for_manifest_preparation": True,
        "ready_for_training": False,
        "reason_training_blocked": (
            "group_split_and_consumed_audio_hash_firewall_not_completed"
        ),
        "model_inference_run": False,
        "training_started": False,
        "locked_dataset_audio_read": False,
        "inputs": {
            "md5_manifest": manifest_path.relative_to(root).as_posix(),
            "md5_manifest_sha256": file_sha256(manifest_path),
        },
    }
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "passed",
                    "official_files_verified",
                    "zip_archives_verified",
                    "physical_uav_ids",
                    "wav_files",
                    "eligible_primary_uav_wavs",
                    "speech_wavs_excluded",
                    "integrity_passed",
                    "zip_crc_passed",
                    "zip_paths_safe",
                    "ready_for_manifest_preparation",
                    "ready_for_training",
                    "reason_training_blocked",
                    "model_inference_run",
                    "training_started",
                )
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit all 19 official Kielce 17-UAV files without extraction."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/g14_kielce_17_uav_md5.txt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/g14_domain_generalization/intake/kielce_full_archive_audit.json"
        ),
    )
    args = parser.parse_args()
    audit(args.root, args.manifest, args.output)


if __name__ == "__main__":
    main()
