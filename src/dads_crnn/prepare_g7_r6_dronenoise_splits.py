from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data_firewall import file_sha256


PROTOCOL = "g7_r6_dronenoise_event_grouped_60_20_20_v1"
EVENT_GROUP_PATTERN = re.compile(
    r"^(?P<operator>[^_]+)_(?P<uav_subtype>[^_]+)_(?P<hagl_m>[^_]+)_"
    r"(?P<operation>[A-Za-z])(?P<speed>\d+)_(?P<payload>[^_]+)_"
    r"(?P<starting>[^_]+)_(?P<direction>[^_]+)_ev(?P<event_index>\d+)$"
)
MANIFEST_COLUMNS = [
    "dataset_role",
    "split",
    "dataset_origin",
    "label",
    "source_path",
    "recording_group",
    "source_group",
    "event_group",
    "uav_subtype",
    "microphone",
    "duration_seconds",
    "sample_rate",
    "frames",
    "audio_sha256",
    "operator",
    "hagl_m",
    "operation_code",
    "speed_code",
    "payload_code",
    "starting_code",
    "direction_code",
    "event_index",
    "license",
]


def parse_event_group(event_group: str) -> dict[str, object]:
    match = EVENT_GROUP_PATTERN.fullmatch(event_group)
    if not match:
        raise ValueError(f"Unsupported DroneNoise event name: {event_group}")
    values: dict[str, object] = match.groupdict()
    values["hagl_m"] = int(str(values["hagl_m"]))
    values["speed"] = int(str(values["speed"]))
    values["event_index"] = int(str(values["event_index"]))
    return values


def assign_event_roles(
    event_groups: list[str],
    seed: int,
    recording_counts: dict[str, int] | None = None,
) -> dict[str, str]:
    """Keep microphones together; reserve complete events for val/test per type."""
    by_subtype: dict[str, list[str]] = defaultdict(list)
    for event in sorted(set(event_groups)):
        subtype = str(parse_event_group(event)["uav_subtype"])
        by_subtype[subtype].append(event)

    roles: dict[str, str] = {}
    for subtype, events in sorted(by_subtype.items()):
        if len(events) < 3:
            raise ValueError(
                f"Subtype {subtype} has only {len(events)} events; "
                "event-grouped train/validation/test requires at least 3"
            )
        counts = recording_counts or {event: 1 for event in events}
        maximum_count = max(counts.get(event, 0) for event in events)
        complete = [event for event in events if counts.get(event, 0) == maximum_count]
        candidates = complete if len(complete) >= 2 else events
        ordered = sorted(
            candidates,
            key=lambda event: hashlib.sha256(f"{seed}:{event}".encode("utf-8")).hexdigest(),
        )
        roles[ordered[0]] = "validation"
        roles[ordered[1]] = "test"
        for event in events:
            if event not in roles:
                roles[event] = "train"
        for event in ordered[2:]:
            roles[event] = "train"
    return roles


def _pairwise_overlap(frames: dict[str, pd.DataFrame], column: str) -> dict[str, int]:
    names = list(frames)
    return {
        f"{left}__{right}": len(
            set(frames[left][column].dropna().astype(str))
            & set(frames[right][column].dropna().astype(str))
        )
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }


def prepare(dataset_root: Path, output_dir: Path, seed: int = 42) -> dict:
    metadata_dir = dataset_root / "metadata"
    download_path = metadata_dir / "download_audit.json"
    overlap_path = metadata_dir / "dads_overlap_audit.json"
    download = json.loads(download_path.read_text(encoding="utf-8"))
    overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
    if not download.get("passed") or not download.get("all_official_md5_verified"):
        raise ValueError("DroneNoise download/MD5 audit must pass before splitting")
    if not overlap.get("passed") or overlap.get(
        "files_with_exact_or_aligned_half_second_dads_overlap"
    ):
        raise ValueError("DroneNoise DADS overlap audit must pass with zero matches")

    inventory = {
        item["name"]: item
        for item in download["inventory"]
        if item.get("is_audio") and not item["name"].startswith("Calib_")
    }
    unique_files: dict[str, dict] = {}
    for item in overlap["files"]:
        if item["is_calibration"]:
            continue
        unique_files.setdefault(item["file"], item)
    event_recording_counts: dict[str, int] = defaultdict(int)
    for item in unique_files.values():
        event_recording_counts[item["event_group"]] += 1
    event_roles = assign_event_roles(
        [item["event_group"] for item in unique_files.values()],
        seed=seed,
        recording_counts=dict(event_recording_counts),
    )

    rows = []
    for name, item in sorted(unique_files.items()):
        source = dataset_root / "raw" / name
        if not source.is_file():
            raise FileNotFoundError(source)
        official = inventory[name]
        event = item["event_group"]
        parsed = parse_event_group(event)
        role = event_roles[event]
        recording_group = f"dronenoise_v3:{event}:M{item['microphone']}"
        rows.append(
            {
                "dataset_role": role,
                "split": role,
                "dataset_origin": "dronenoise_database_v3",
                "label": 1,
                "source_path": str(source),
                "recording_group": recording_group,
                "source_group": f"dronenoise_v3:{event}",
                "event_group": event,
                "uav_subtype": parsed["uav_subtype"],
                "microphone": int(item["microphone"]),
                "duration_seconds": float(official["duration_seconds"]),
                "sample_rate": int(official["sample_rate"]),
                "frames": int(official["frames"]),
                "audio_sha256": official["sha256"],
                "operator": parsed["operator"],
                "hagl_m": parsed["hagl_m"],
                "operation_code": parsed["operation"],
                "speed_code": parsed["speed"],
                "payload_code": parsed["payload"],
                "starting_code": parsed["starting"],
                "direction_code": parsed["direction"],
                "event_index": parsed["event_index"],
                "license": "CC BY 4.0",
            }
        )
    combined = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    roles = ("train", "validation", "test")
    frames = {
        role: combined[combined["dataset_role"].eq(role)].reset_index(drop=True)
        for role in roles
    }
    if any(frame.empty for frame in frames.values()):
        raise ValueError("Every DroneNoise role must be non-empty")

    overlap_checks = {
        column: _pairwise_overlap(frames, column)
        for column in ("audio_sha256", "recording_group", "source_group", "event_group")
    }
    if any(value for checks in overlap_checks.values() for value in checks.values()):
        raise ValueError(f"DroneNoise split leakage detected: {overlap_checks}")

    expected_types = set(combined["uav_subtype"])
    for role in ("validation", "test"):
        if set(frames[role]["uav_subtype"]) != expected_types:
            raise ValueError(f"{role} does not cover every UAV subtype")

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "all_recordings_manifest.csv"
    combined.to_csv(combined_path, index=False)
    paths = {role: output_dir / f"{role}_recordings_manifest.csv" for role in roles}
    for role, path in paths.items():
        frames[role].to_csv(path, index=False)

    event_rows = []
    for event, role in sorted(event_roles.items()):
        parsed = parse_event_group(event)
        selected = combined[combined["event_group"].eq(event)]
        event_rows.append(
            {
                "event_group": event,
                "split": role,
                "uav_subtype": parsed["uav_subtype"],
                "recordings": len(selected),
                "microphones": ",".join(map(str, sorted(selected["microphone"].unique()))),
                "duration_seconds_sum": float(selected["duration_seconds"].sum()),
            }
        )
    event_path = output_dir / "event_split_manifest.csv"
    pd.DataFrame(event_rows).to_csv(event_path, index=False)

    counts = {
        role: {
            "events": int(frames[role]["event_group"].nunique()),
            "recordings": len(frames[role]),
            "duration_seconds_sum": float(frames[role]["duration_seconds"].sum()),
            "events_by_uav_subtype": {
                str(key): int(value)
                for key, value in frames[role]
                .drop_duplicates("event_group")["uav_subtype"]
                .value_counts()
                .sort_index()
                .items()
            },
            "recordings_by_uav_subtype": {
                str(key): int(value)
                for key, value in frames[role]["uav_subtype"].value_counts().sort_index().items()
            },
        }
        for role in roles
    }
    audit = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "split_unit": "flight_event_all_microphones_grouped",
        "allocation_rule": (
            "within_each_uav_subtype prefer events with maximum microphone coverage, "
            "then hash-order with seed; one validation, one test, remaining train"
        ),
        "target_event_ratio": {"train": 0.60, "validation": 0.20, "test": 0.20},
        "positive_only_dataset": True,
        "metric_limitation": (
            "standalone validation/test support positive recall by type/event; "
            "FPR, specificity and binary accuracy require an independently assigned negative corpus"
        ),
        "counts": counts,
        "excluded": {
            "calibration_files": int(overlap["calibration_files_excluded"]),
            "duplicate_official_inventory_entries": len(overlap["official_duplicate_files"]),
            "duplicate_names": overlap["official_duplicate_files"],
        },
        "overlap_checks": overlap_checks,
        "validation_and_test_cover_all_uav_subtypes": True,
        "inputs": {
            "download_audit": {"path": str(download_path), "sha256": file_sha256(download_path)},
            "dads_overlap_audit": {"path": str(overlap_path), "sha256": file_sha256(overlap_path)},
        },
        "outputs": {
            "combined": {"path": str(combined_path), "sha256": file_sha256(combined_path)},
            "events": {"path": str(event_path), "sha256": file_sha256(event_path)},
            **{
                role: {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "rows": len(frames[role]),
                }
                for role, path in paths.items()
            },
        },
        "next_status": "event_split_complete_pending_half_second_cache_and_protocol_merge",
    }
    audit_path = output_dir / "split_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Split DroneNoise by flight event")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/g7_r6_new_sources/drone_noise_v3"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/g7_r6_dronenoise_event_split"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = prepare(args.dataset_root, args.output_dir, seed=args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
