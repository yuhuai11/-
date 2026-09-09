from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .config import load_config
from .data_firewall import file_sha256, reject_locked_path


PROTOCOL = "g14_nasa_suas_archive_audit_v1"
MAT_PATTERN = re.compile(
    r"^data/(?P<vehicle>cub|edge|hex|phantom|y6)_"
    r"(?P<maneuver>flyover|hover)_(?P<flight>[0-9]+)\.mat$"
)
ACQUISITION_GROUP = {
    "edge": "virginia_beach_2014",
    "phantom": "virginia_beach_2014",
    "y6": "virginia_beach_2014",
    "cub": "ap_hill_2015",
    "hex": "ap_hill_2015",
}


def parse_member(
    name: str,
    *,
    positive_tokens: set[str],
    negative_tokens: set[str],
) -> dict[str, Any] | None:
    match = MAT_PATTERN.fullmatch(name)
    if match is None:
        return None
    vehicle = match.group("vehicle")
    if vehicle in positive_tokens:
        label = 1
        role = "uav"
    elif vehicle in negative_tokens:
        label = 0
        role = "fixed_wing_hard_negative"
    else:
        raise ValueError(f"Unregistered NASA vehicle token: {vehicle}")
    maneuver = match.group("maneuver")
    flight = int(match.group("flight"))
    return {
        "dataset": "nasa_suas",
        "archive_member": name,
        "vehicle": vehicle,
        "maneuver": maneuver,
        "flight": flight,
        "label": label,
        "role": role,
        "acquisition_group": ACQUISITION_GROUP[vehicle],
        "source_group": f"nasa_suas:{vehicle}:{maneuver}:{flight:03d}",
    }


def audit(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    settings = config["sources"]["nasa_suas"]
    archive = Path(settings["archive_path"])
    if not archive.is_absolute():
        archive = root / archive
    reject_locked_path(archive, context="G14 NASA archive")
    archive = archive.resolve(strict=True)
    expected_bytes = int(settings["expected_bytes"])
    if archive.stat().st_size != expected_bytes:
        raise ValueError(
            f"NASA archive size mismatch: {archive.stat().st_size} != {expected_bytes}"
        )
    observed_sha256 = file_sha256(archive)
    if observed_sha256 != str(settings["expected_sha256"]).lower():
        raise ValueError("NASA archive SHA256 does not match the frozen intake config")

    positive = {str(value).lower() for value in settings["positive_vehicle_tokens"]}
    negative = {str(value).lower() for value in settings["negative_vehicle_tokens"]}
    if positive & negative:
        raise ValueError("NASA positive and negative vehicle sets overlap")

    records = []
    metadata_members = []
    with zipfile.ZipFile(archive) as handle:
        bad_member = handle.testzip()
        if bad_member is not None:
            raise ValueError(f"NASA ZIP CRC failure: {bad_member}")
        for info in handle.infolist():
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe NASA archive member: {info.filename}")
            parsed = parse_member(
                info.filename,
                positive_tokens=positive,
                negative_tokens=negative,
            )
            if parsed is None:
                if not info.is_dir():
                    metadata_members.append(info.filename)
                continue
            parsed["uncompressed_bytes"] = int(info.file_size)
            parsed["compressed_bytes"] = int(info.compress_size)
            parsed["crc32"] = f"{info.CRC:08x}"
            records.append(parsed)

    if not records:
        raise ValueError("NASA archive contains no registered MAT flight files")
    source_groups = [record["source_group"] for record in records]
    if len(set(source_groups)) != len(source_groups):
        raise ValueError("NASA archive contains duplicate vehicle/flight source groups")
    label_counts = Counter(int(record["label"]) for record in records)
    minimum = int(config["minimum"]["source_groups_per_label"])
    label_minimum_passed = all(label_counts[label] >= minimum for label in (0, 1))
    if not label_minimum_passed:
        raise ValueError(
            f"NASA source groups do not meet the minimum {minimum}: {dict(label_counts)}"
        )

    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "nasa_archive_inventory.csv"
    fieldnames = list(records[0])
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: row["source_group"]))

    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": {
            "path": archive.relative_to(root).as_posix(),
            "bytes": archive.stat().st_size,
            "sha256": observed_sha256,
            "zip_crc_tested": True,
        },
        "flight_files": len(records),
        "source_groups": len(set(source_groups)),
        "source_groups_by_label": {
            str(label): int(label_counts[label]) for label in (0, 1)
        },
        "source_groups_by_vehicle": dict(
            sorted(Counter(record["vehicle"] for record in records).items())
        ),
        "source_groups_by_acquisition": dict(
            sorted(Counter(record["acquisition_group"] for record in records).items())
        ),
        "metadata_members": sorted(metadata_members),
        "minimum_source_groups_per_label": minimum,
        "minimum_source_groups_passed": label_minimum_passed,
        "license_status": settings["status"],
        "ready_for_extraction": True,
        "ready_for_training": False,
        "training_blockers": [
            "NASA training license/use terms have not been explicitly archived",
            "cross-dataset SHA256 deduplication has not been performed",
            "source-disjoint train/tune/holdout split has not been frozen",
        ],
        "outputs": {
            "inventory": {
                "path": inventory_path.relative_to(root).as_posix(),
                "sha256": file_sha256(inventory_path),
                "rows": len(records),
            }
        },
        "model_inference_run": False,
        "training_started": False,
        "locked_dataset_audio_read": False,
    }
    audit_path = output_dir / "nasa_archive_audit.json"
    audit_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "archive_sha256": observed_sha256,
                "flight_files": report["flight_files"],
                "source_groups_by_label": report["source_groups_by_label"],
                "source_groups_by_vehicle": report["source_groups_by_vehicle"],
                "minimum_source_groups_passed": label_minimum_passed,
                "ready_for_extraction": True,
                "ready_for_training": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the downloaded G14 NASA sUAS archive.")
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
