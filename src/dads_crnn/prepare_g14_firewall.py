from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data_firewall import file_sha256


PROTOCOL = "g14_p0_consumed_data_firewall_v1"
EXPECTED_STATUS = "consumed_and_closed"
REQUIRED_CONSUMED_ARTIFACTS = (
    "frozen_protocol",
    "manifest",
    "metrics",
    "predictions",
    "final_report",
)


def _resolve_inside_root(root: Path, value: object) -> Path:
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved_root = root.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Locked artifact escapes the project root: {resolved}")
    return resolved


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _verify_consumed_artifacts(root: Path, consumed: dict[str, Any]) -> list[dict[str, Any]]:
    inventory = []
    for name in REQUIRED_CONSUMED_ARTIFACTS:
        entry = consumed.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"CONSUMED.json lacks {name}")
        path = _resolve_inside_root(root, entry.get("path"))
        expected = str(entry.get("sha256", "")).strip().lower()
        observed = file_sha256(path)
        if observed != expected:
            raise ValueError(f"Locked {name} SHA256 changed: {path}")
        inventory.append(
            {
                "name": name,
                "path": path.relative_to(root.resolve()).as_posix(),
                "sha256": observed,
                "bytes": path.stat().st_size,
            }
        )
    return inventory


def _manifest_hashes(path: Path, expected_rows: int) -> list[str]:
    hashes = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "sha256" not in reader.fieldnames:
            raise ValueError("G13 manifest lacks sha256")
        for row_number, row in enumerate(reader, start=2):
            value = str(row.get("sha256", "")).strip().lower()
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"Malformed G13 audio SHA256 at row {row_number}")
            hashes.append(value)
    if len(hashes) != expected_rows:
        raise ValueError(
            f"G13 manifest row count changed: expected {expected_rows}, got {len(hashes)}"
        )
    if len(set(hashes)) != len(hashes):
        raise ValueError("G13 manifest contains duplicate audio SHA256 values")
    return sorted(hashes)


def prepare(root: Path, output_dir: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    consumed_path = root / "artifacts/g13_external_confirmation/CONSUMED.json"
    consumed = _read_json(consumed_path)
    if consumed.get("status") != EXPECTED_STATUS:
        raise ValueError(f"G13 status must be {EXPECTED_STATUS}")
    policy = consumed.get("future_use_policy")
    if not isinstance(policy, dict) or not all(
        policy.get(name) is True
        for name in (
            "training_forbidden",
            "calibration_forbidden",
            "threshold_selection_forbidden",
            "checkpoint_selection_forbidden",
            "ensemble_weight_selection_forbidden",
            "repeat_final_evaluation_forbidden",
        )
    ):
        raise ValueError("G13 future-use policy is incomplete")

    inventory = _verify_consumed_artifacts(root, consumed)
    entries = {entry["name"]: entry for entry in inventory}
    manifest_path = root / entries["manifest"]["path"]
    expected_rows = int(consumed["manifest"]["rows"])
    hashes = _manifest_hashes(manifest_path, expected_rows)

    protocol = _read_json(root / entries["frozen_protocol"]["path"])
    baseline = protocol.get("inputs", {}).get("baseline_checkpoint", {})
    baseline_path = _resolve_inside_root(root, baseline.get("path"))
    baseline_hash = file_sha256(baseline_path)
    if baseline_hash != str(baseline.get("sha256", "")).strip().lower():
        raise ValueError("G7 baseline checkpoint SHA256 does not match the frozen protocol")
    protocol_values = protocol.get("protocol", {})
    baseline_lock = {
        "protocol": PROTOCOL,
        "status": "locked",
        "model": "PANNs Cnn14_16k",
        "checkpoint": {
            "path": baseline_path.relative_to(root).as_posix(),
            "sha256": baseline_hash,
            "bytes": baseline_path.stat().st_size,
        },
        "temperature": float(protocol_values["baseline_temperature"]),
        "strict_threshold": float(protocol_values["baseline_threshold"]),
        "balanced_mode_policy": str(protocol_values["balanced_mode_policy"]),
        "source": entries["frozen_protocol"],
    }
    consumed_datasets = {
        "protocol": PROTOCOL,
        "status": EXPECTED_STATUS,
        "datasets": [
            {
                "name": "g13_external_confirmation",
                "aliases": ["external_confirmation_v2"],
                "training_forbidden": True,
                "calibration_forbidden": True,
                "model_selection_forbidden": True,
                "repeat_final_evaluation_forbidden": True,
                "audio_hash_registry": (
                    output_dir / "g13_audio_sha256.txt"
                ).relative_to(root).as_posix(),
                "audio_hashes": len(hashes),
            },
            {
                "name": "legacy_unseen",
                "aliases": ["unseen"],
                "training_forbidden": True,
                "calibration_forbidden": True,
                "model_selection_forbidden": True,
            },
            {
                "name": "legacy_real_world",
                "aliases": ["real_world", "real-world", "realworld"],
                "training_forbidden": True,
                "calibration_forbidden": True,
                "model_selection_forbidden": True,
            },
        ],
    }
    hash_inventory = {
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "consumed_marker": {
            "path": consumed_path.relative_to(root).as_posix(),
            "sha256": file_sha256(consumed_path),
        },
        "artifacts": inventory,
        "g13_audio_hashes": len(hashes),
        "g13_audio_hashes_unique": len(set(hashes)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        output_dir / "g13_audio_sha256.txt",
        "".join(f"{value}\n" for value in hashes),
    )
    _atomic_write_text(
        output_dir / "baseline_lock.json",
        json.dumps(baseline_lock, indent=2, ensure_ascii=False) + "\n",
    )
    _atomic_write_text(
        output_dir / "consumed_datasets.json",
        json.dumps(consumed_datasets, indent=2, ensure_ascii=False) + "\n",
    )
    _atomic_write_text(
        output_dir / "g13_hash_inventory.json",
        json.dumps(hash_inventory, indent=2, ensure_ascii=False) + "\n",
    )
    return {
        "passed": True,
        "protocol": PROTOCOL,
        "g13_status": EXPECTED_STATUS,
        "g13_artifacts_verified": len(inventory),
        "g13_audio_hashes_locked": len(hashes),
        "g7_checkpoint_sha256": baseline_hash,
        "locked_dataset_audio_read": False,
        "archival_metadata_read": ["g13_external_confirmation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the G14 P0 consumed-data firewall.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/g14_domain_generalization/p0_firewall"),
    )
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    report = prepare(root, output_dir)
    _atomic_write_text(
        output_dir / "test_report.json",
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
