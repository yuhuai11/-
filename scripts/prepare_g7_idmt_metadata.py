#!/usr/bin/env python3
"""Create the frozen G7 IDMT-TRAFFIC intake manifest without reading audio."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROLE_BY_LOCATION = {
    "Fraunhofer-IDMT": "calibration",
    "Langewiesener-Strasse": "development_test",
    "Schleusinger-Allee": "development_test",
    "Hohenwarte": "final_holdout",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_filename(filename: str) -> dict[str, str | int | bool]:
    stem = filename.removesuffix(".wav")
    parts = stem.split("_")
    is_background = "-BG" in stem
    if is_background:
        date_time, location, speed, sample_pos, microphone, channel = parts
        daytime = weather = vehicle = direction = ""
    else:
        (
            date_time,
            location,
            speed,
            sample_pos,
            daytime,
            weather,
            vehicle_direction,
            microphone,
            channel,
        ) = parts
        vehicle, direction = vehicle_direction

    channel = channel.removesuffix("-BG")
    speed = speed.replace("unknownKmh", "UNK").replace("Kmh", "")
    return {
        "dataset_name": "IDMT-TRAFFIC",
        "recording_id": stem,
        "archive_member": f"IDMT_Traffic/audio/{filename}",
        "session_id": f"{date_time}_{location}",
        "date_time": date_time,
        "location_id": location,
        "speed_kmh": speed,
        "sample_position": sample_pos,
        "microphone_id": microphone,
        "channels": channel,
        "traffic_content": "background_only" if is_background else "vehicle_passing",
        "daytime": daytime,
        "weather": weather,
        "vehicle": vehicle,
        "direction": direction,
        "duration_seconds": 2,
        "drone_label": 0,
        "intended_role": ROLE_BY_LOCATION[location],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file-list",
        type=Path,
        default=Path(
            "data/g7_cross_domain/idmt_traffic_metadata/"
            "IDMT_Traffic/annotation/idmt_traffic_all.txt"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "artifacts/g7_improvement/stage_a/idmt_traffic_manifest.csv"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "artifacts/g7_improvement/stage_a/idmt_traffic_split_summary.json"
        ),
    )
    args = parser.parse_args()

    filenames = [
        line.strip()
        for line in args.file_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [parse_filename(filename) for filename in filenames]
    if len(rows) != 17_506:
        raise ValueError(f"Expected 17506 files, observed {len(rows)}")
    if len({str(row["archive_member"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate archive member in IDMT file list")

    sessions_by_role: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        sessions_by_role[str(row["intended_role"])].add(str(row["session_id"]))
    roles = sorted(sessions_by_role)
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            overlap = sessions_by_role[left] & sessions_by_role[right]
            if overlap:
                raise ValueError(f"Session leakage between {left} and {right}: {overlap}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    role_counts = Counter(str(row["intended_role"]) for row in rows)
    location_counts = Counter(str(row["location_id"]) for row in rows)
    content_counts = Counter(str(row["traffic_content"]) for row in rows)
    summary = {
        "dataset_name": "IDMT-TRAFFIC",
        "created_without_audio_inference": True,
        "split_unit": "recording location; sessions never cross roles",
        "role_policy": ROLE_BY_LOCATION,
        "rows": len(rows),
        "duration_hours": len(rows) * 2 / 3600,
        "counts_by_role": dict(sorted(role_counts.items())),
        "counts_by_location": dict(sorted(location_counts.items())),
        "counts_by_traffic_content": dict(sorted(content_counts.items())),
        "sessions_by_role": {
            role: len(sessions) for role, sessions in sorted(sessions_by_role.items())
        },
        "source_file_list": {
            "path": str(args.file_list),
            "sha256": sha256(args.file_list),
        },
        "manifest": {
            "path": str(args.manifest),
            "sha256": sha256(args.manifest),
        },
        "holdout_policy": (
            "Hohenwarte is locked. Do not run model inference or inspect prediction "
            "metrics until the final protocol is approved."
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
