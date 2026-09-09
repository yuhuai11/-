from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .config import load_config
from .data_firewall import file_sha256, reject_locked_path


PROTOCOL = "g14_kielce_17_uav_pilot_audit_v1"
WAV_PATTERN = re.compile(
    r"(?P<drone>D\d+).*?(?P<height>\d+)[mM].*?(?P<distance>\d+)[mM]",
    re.IGNORECASE,
)
DATE_PATTERN = re.compile(r"20\d{2}[-_.]?\d{2}[-_.]?\d{2}|(?<!\d)\d{6}(?!\d)")


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    reject_locked_path(path, context="G14 Kielce 17-UAV pilot")
    return path.resolve(strict=True)


def _safe_files(handle: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    bad_member = handle.testzip()
    if bad_member is not None:
        raise ValueError(f"Kielce pilot ZIP CRC failure: {bad_member}")
    files = []
    for info in handle.infolist():
        member = PurePosixPath(info.filename)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"Unsafe Kielce pilot ZIP member: {info.filename}")
        if not info.is_dir():
            files.append(info)
    return files


def _device(name: str) -> str:
    upper = name.upper()
    if "NORSONIC" in upper:
        return "NORSONIC_140"
    if "OLYMPUS" in upper:
        return "OLYMPUS_LS11"
    return "unknown"


def audit(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    settings = config["sources"]["kielce_17_uav"]
    pilot = settings["pilot_audio"]
    metadata_path = _resolve(root, settings["metadata_path"])
    archive_path = _resolve(root, pilot["path"])

    checks = (
        (
            metadata_path,
            int(settings["metadata_expected_bytes"]),
            str(settings["metadata_expected_md5"]).lower(),
            "description",
        ),
        (
            archive_path,
            int(pilot["expected_bytes"]),
            str(pilot["expected_md5"]).lower(),
            "pilot archive",
        ),
    )
    for path, expected_bytes, expected_md5, label in checks:
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"Kielce {label} byte count mismatch")
        if _md5(path) != expected_md5:
            raise ValueError(f"Kielce {label} MD5 mismatch")

    with zipfile.ZipFile(archive_path) as handle:
        files = _safe_files(handle)

    wav_files = [info for info in files if info.filename.lower().endswith(".wav")]
    if not wav_files:
        raise ValueError("Kielce pilot contains no WAV files")

    rows = []
    speech_token = str(settings["full_archive"]["speech_token"]).lower()
    for info in wav_files:
        path = PurePosixPath(info.filename)
        name = path.name
        match = WAV_PATTERN.search(name)
        date_match = DATE_PATTERN.search(name)
        device = _device(info.filename)
        rows.append(
            {
                "dataset": "kielce_17_uav",
                "archive_member": info.filename,
                "label": 1,
                "drone_id": str(pilot["drone_id"]),
                "vehicle": str(pilot["vehicle"]),
                "rotor_layout": str(pilot["rotor_layout"]),
                "device": device,
                "height_m": match.group("height") if match else "",
                "distance_m": match.group("distance") if match else "",
                "acquisition_date": date_match.group(0) if date_match else "",
                "contains_speech": speech_token in name.lower(),
                "eligible_primary_uav": speech_token not in name.lower(),
                "uncompressed_bytes": info.file_size,
                "source_group": (
                    f"kielce_17_uav:{pilot['drone_id']}:"
                    f"{date_match.group(0) if date_match else 'date_unknown'}"
                ),
            }
        )

    devices = Counter(row["device"] for row in rows)
    archive_devices = Counter(_device(info.filename) for info in files)
    archive_formats = Counter(
        PurePosixPath(info.filename).suffix.lower() or "[no_suffix]" for info in files
    )
    eligible = [row for row in rows if row["eligible_primary_uav"]]
    speech = [row for row in rows if row["contains_speech"]]
    unknown_device = devices.get("unknown", 0)
    if not eligible:
        raise ValueError("Kielce pilot has no speech-free UAV WAV files")
    if unknown_device:
        raise ValueError(f"Kielce pilot has {unknown_device} WAV files with unknown recorder")

    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / "kielce_pilot_registry.csv"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=output_dir, delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(registry_path)

    parsed_geometry = sum(
        bool(row["height_m"]) and bool(row["distance_m"]) for row in eligible
    )
    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "integrity_passed": True,
        "license": settings["license"]["expression"],
        "license_training_allowed": settings["license"]["training_allowed"] is True,
        "archive_members": len(files),
        "wav_files": len(wav_files),
        "eligible_primary_uav_wavs": len(eligible),
        "speech_wavs_excluded": len(speech),
        "wav_recorders": dict(sorted(devices.items())),
        "archive_recorders": dict(sorted(archive_devices.items())),
        "archive_formats": dict(sorted(archive_formats.items())),
        "directly_trainable_waveform_systems": ["OLYMPUS_LS11_WAV"],
        "norsonic_requires_conversion_or_feature_level_policy": True,
        "geometry_parsed": parsed_geometry,
        "geometry_parse_rate": parsed_geometry / len(eligible),
        "source_groups": len({row["source_group"] for row in eligible}),
        "structure_preflight_passed": True,
        "coverage_preflight_passed": False,
        "ready_for_full_download": True,
        "ready_for_training": False,
        "reason_training_blocked": "pilot_only_full_17_uav_archive_not_downloaded_or_split",
        "outputs": {
            "registry": {
                "path": registry_path.relative_to(root).as_posix(),
                "sha256": file_sha256(registry_path),
                "rows": len(rows),
            }
        },
        "inputs": {
            "description_sha256": file_sha256(metadata_path),
            "pilot_archive_sha256": file_sha256(archive_path),
        },
        "model_inference_run": False,
        "training_started": False,
        "locked_dataset_audio_read": False,
    }
    audit_path = output_dir / "kielce_pilot_audit.json"
    audit_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the minimal Kielce 17-UAV G14 pilot without extracting audio."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g14_domain_generalization_intake.yaml"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    audit(args.config, args.root)


if __name__ == "__main__":
    main()
