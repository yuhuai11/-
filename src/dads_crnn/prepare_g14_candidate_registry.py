from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .data_firewall import file_sha256, reject_locked_path


PROTOCOL = "g14_multisource_candidate_registry_v1"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _file_state(
    path: Path,
    *,
    expected_bytes: int | None = None,
    expected_md5: str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "path": path.as_posix(),
        "present": path.is_file(),
        "bytes": None,
        "size_matches": None,
        "md5": None,
        "md5_matches": None,
        "sha256": None,
        "sha256_matches": None,
    }
    if not path.is_file():
        return state
    state["bytes"] = path.stat().st_size
    if expected_bytes is not None:
        state["size_matches"] = path.stat().st_size == expected_bytes
    if expected_md5 is not None:
        state["md5"] = _md5(path)
        state["md5_matches"] = state["md5"] == expected_md5.lower()
    if (expected_bytes is None or state["size_matches"]) and (
        expected_md5 is None or state["md5_matches"]
    ):
        state["sha256"] = file_sha256(path)
        if expected_sha256 is not None:
            state["sha256_matches"] = state["sha256"] == expected_sha256.lower()
    return state


def prepare(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    if config.get("protocol") != "g14_multisource_intake_v1":
        raise ValueError("Unexpected G14 intake protocol")

    output_dir = _path(root, config["output_dir"]).resolve(strict=False)
    sources = config.get("sources", {})
    if not isinstance(sources, dict) or not sources:
        raise ValueError("G14 intake config has no candidate sources")

    candidate_rows = []
    license_rows = []
    file_states: dict[str, Any] = {}
    selected_roles = set()
    selected_classes = set()
    for source_id, settings in sources.items():
        if not isinstance(settings, dict):
            raise ValueError(f"Invalid candidate settings: {source_id}")
        selected = bool(settings.get("selected"))
        if selected:
            selected_roles.add(str(settings.get("role")))
            selected_classes.add(str(settings.get("intake_class", "")))
        for field in ("archive_path", "metadata_path"):
            value = settings.get(field)
            if value:
                path = _path(root, value)
                reject_locked_path(path, context=f"G14 candidate {source_id}")
        candidate_rows.append(
            {
                "source_id": source_id,
                "dataset_name": settings.get("dataset_name", ""),
                "role": settings.get("role", ""),
                "intake_class": settings.get("intake_class", ""),
                "selected": selected,
                "status": settings.get("status", ""),
                "publisher": settings.get("publisher", ""),
                "source_group_unit": settings.get("source_group_unit", ""),
                "reason": settings.get("reason", ""),
            }
        )
        license_settings = settings.get("license", {})
        if isinstance(license_settings, dict) and license_settings:
            license_rows.append(
                {
                    "source_id": source_id,
                    "access_level": license_settings.get("access_level", ""),
                    "expression": license_settings.get("expression", ""),
                    "commercial_use": license_settings.get("commercial_use", ""),
                    "training_allowed": license_settings.get("training_allowed", False),
                    "action": license_settings.get("action", ""),
                }
            )
        if settings.get("archive_path"):
            file_states[source_id] = {
                "archive": _file_state(
                _path(root, settings["archive_path"]),
                expected_bytes=int(settings["expected_bytes"])
                if settings.get("expected_bytes") is not None
                else None,
                expected_sha256=str(settings["expected_sha256"])
                if settings.get("expected_sha256")
                else None,
                )
            }
        elif settings.get("metadata_path"):
            states = {
                "metadata": _file_state(
                    _path(root, settings["metadata_path"]),
                    expected_bytes=int(settings["metadata_expected_bytes"])
                    if settings.get("metadata_expected_bytes") is not None
                    else None,
                    expected_md5=str(settings["metadata_expected_md5"])
                    if settings.get("metadata_expected_md5")
                    else None,
                    expected_sha256=str(settings["metadata_expected_sha256"])
                    if settings.get("metadata_expected_sha256")
                    else None,
                )
            }
            pilot = settings.get("pilot_audio")
            if isinstance(pilot, dict) and pilot.get("path"):
                states["pilot_audio"] = _file_state(
                    _path(root, pilot["path"]),
                    expected_bytes=int(pilot["expected_bytes"])
                    if pilot.get("expected_bytes") is not None
                    else None,
                    expected_md5=str(pilot["expected_md5"])
                    if pilot.get("expected_md5")
                    else None,
                    expected_sha256=str(pilot["expected_sha256"])
                    if pilot.get("expected_sha256")
                    else None,
                )
            supplemental = settings.get("supplemental_audio", [])
            if supplemental is not None and not isinstance(supplemental, list):
                raise ValueError(f"Invalid supplemental_audio settings: {source_id}")
            for item in supplemental or []:
                if not isinstance(item, dict) or not item.get("path") or item.get("part") is None:
                    raise ValueError(f"Invalid supplemental audio item: {source_id}")
                path = _path(root, item["path"])
                reject_locked_path(path, context=f"G14 candidate {source_id}")
                states[f"audio_part_{int(item['part'])}"] = _file_state(
                    path,
                    expected_bytes=int(item["expected_bytes"])
                    if item.get("expected_bytes") is not None
                    else None,
                    expected_md5=str(item["expected_md5"])
                    if item.get("expected_md5")
                    else None,
                    expected_sha256=str(item["expected_sha256"])
                    if item.get("expected_sha256")
                    else None,
                )
            file_states[source_id] = states

    has_selected_uav = (
        "positive_uav" in selected_classes
        or "uav_primary_and_aircraft_hard_negative" in selected_roles
    )
    has_selected_background = (
        "negative_background" in selected_classes
        or "multidevice_background" in selected_roles
    )
    if not has_selected_uav:
        raise ValueError("G14 intake lacks a selected UAV source")
    if not has_selected_background:
        raise ValueError("G14 intake lacks a selected background source")

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate_registry.csv"
    license_path = output_dir / "license_inventory.csv"
    with candidate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    with license_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(license_rows[0]))
        writer.writeheader()
        writer.writerows(license_rows)

    disk = shutil.disk_usage(root)
    unresolved_training_licenses = sorted(
        row["source_id"]
        for row in license_rows
        if sources[row["source_id"]].get("selected")
        and row["training_allowed"] is not True
    )
    selected_missing = sorted(
        source_id
        for source_id, settings in sources.items()
        if settings.get("selected")
        and (
            not file_states.get(source_id)
            or not all(
                state.get("present") and state.get("size_matches") is not False
                and state.get("md5_matches") is not False
                and state.get("sha256_matches") is not False
                for state in file_states[source_id].values()
            )
        )
    )
    coverage_blockers = sorted(
        source_id
        for source_id, settings in sources.items()
        if settings.get("selected")
        and isinstance(settings.get("pilot_audio"), dict)
        and settings["pilot_audio"].get("coverage_status") == "insufficient_alone"
        and settings.get("combined_coverage_status") != "sufficient"
    )
    pilot_only_sources = sorted(
        source_id
        for source_id, settings in sources.items()
        if settings.get("selected") and settings.get("pilot_only") is True
    )
    source_stage_blockers = {
        source_id: list(settings.get("stage_blockers", []))
        for source_id, settings in sources.items()
        if settings.get("selected") and settings.get("stage_blockers")
    }
    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "selection": {
            "selected_sources": sorted(
                source_id
                for source_id, settings in sources.items()
                if settings.get("selected")
            ),
            "reserve_sources": sorted(
                source_id
                for source_id, settings in sources.items()
                if not settings.get("selected")
            ),
            "selected_missing_files": selected_missing,
            "unresolved_training_licenses": unresolved_training_licenses,
            "coverage_blockers": coverage_blockers,
            "pilot_only_sources": pilot_only_sources,
            "source_stage_blockers": source_stage_blockers,
            "ready_for_download": True,
            "ready_for_intake": not selected_missing and not coverage_blockers,
            "ready_for_training": not selected_missing
            and not unresolved_training_licenses
            and not coverage_blockers
            and not pilot_only_sources
            and not source_stage_blockers,
        },
        "file_states": file_states,
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "outputs": {
            "candidate_registry": {
                "path": candidate_path.relative_to(root).as_posix(),
                "sha256": file_sha256(candidate_path),
            },
            "license_inventory": {
                "path": license_path.relative_to(root).as_posix(),
                "sha256": file_sha256(license_path),
            },
        },
        "model_inference_run": False,
        "training_started": False,
        "locked_dataset_audio_read": False,
    }
    _atomic_write_text(
        output_dir / "candidate_preaudit.json",
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit G14 candidate sources before download.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g14_domain_generalization_intake.yaml"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    report = prepare(args.config, args.root)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "protocol": report["protocol"],
                **report["selection"],
                "free_bytes": report["disk"]["free_bytes"],
                "model_inference_run": False,
                "training_started": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
