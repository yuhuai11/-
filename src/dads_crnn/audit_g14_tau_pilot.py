from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from .config import load_config
from .data_firewall import file_sha256, reject_locked_path


PROTOCOL = "g14_tau_urban_2022_pilot_audit_v1"


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
    reject_locked_path(path, context="G14 TAU pilot")
    return path.resolve(strict=True)


def _safe_members(handle: zipfile.ZipFile) -> list[str]:
    names = []
    bad_member = handle.testzip()
    if bad_member is not None:
        raise ValueError(f"TAU ZIP CRC failure: {bad_member}")
    for info in handle.infolist():
        member = PurePosixPath(info.filename)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"Unsafe TAU ZIP member: {info.filename}")
        if not info.is_dir():
            names.append(info.filename)
    return names


def audit(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    settings = config["sources"]["tau_urban_2022"]
    pilot = settings["pilot_audio"]
    metadata_path = _resolve(root, settings["metadata_path"])
    audio_path = _resolve(root, pilot["path"])

    if _md5(metadata_path) != str(settings["metadata_expected_md5"]).lower():
        raise ValueError("TAU metadata MD5 mismatch")
    if file_sha256(metadata_path) != str(settings["metadata_expected_sha256"]).lower():
        raise ValueError("TAU metadata SHA256 mismatch")
    if audio_path.stat().st_size != int(pilot["expected_bytes"]):
        raise ValueError("TAU pilot byte count mismatch")
    if _md5(audio_path) != str(pilot["expected_md5"]).lower():
        raise ValueError("TAU pilot MD5 mismatch")
    if file_sha256(audio_path) != str(pilot["expected_sha256"]).lower():
        raise ValueError("TAU pilot SHA256 mismatch")

    with zipfile.ZipFile(metadata_path) as metadata_zip:
        metadata_names = _safe_members(metadata_zip)
        matches = [name for name in metadata_names if name.endswith("/meta.csv")]
        if len(matches) != 1:
            raise ValueError("TAU metadata archive must contain exactly one meta.csv")
        metadata = pd.read_csv(io.BytesIO(metadata_zip.read(matches[0])), sep="\t")
    required = {"filename", "scene_label", "identifier", "source_label"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"TAU metadata lacks columns: {missing}")

    with zipfile.ZipFile(audio_path) as audio_zip:
        audio_names = _safe_members(audio_zip)
    audio_suffixes = {
        name[name.index("audio/") :]
        for name in audio_names
        if "audio/" in name and name.lower().endswith(".wav")
    }
    selected = metadata.loc[metadata["filename"].isin(audio_suffixes)].copy()
    if len(selected) != len(audio_suffixes):
        raise ValueError(
            f"TAU metadata/audio mismatch: metadata={len(selected)} audio={len(audio_suffixes)}"
        )
    selected["dataset"] = "tau_urban_2022"
    selected["label"] = 0
    selected["city"] = selected["filename"].str.split("-").str[1]
    selected["source_group"] = selected["identifier"].map(
        lambda value: f"tau_urban_2022:{value}"
    )
    selected["device"] = selected["source_label"].astype(str)
    selected = selected[
        [
            "dataset",
            "filename",
            "label",
            "scene_label",
            "city",
            "identifier",
            "device",
            "source_group",
        ]
    ].sort_values(["source_group", "device", "filename"], kind="stable")

    source_groups = int(selected["source_group"].nunique())
    minimum = int(config["minimum"]["source_groups_per_label"])
    coverage_passed = source_groups >= minimum and selected["scene_label"].nunique() >= 2
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / "tau_pilot_registry.csv"
    selected.to_csv(registry_path, index=False)

    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "integrity_passed": True,
        "coverage_passed": coverage_passed,
        "samples": int(len(selected)),
        "source_groups": source_groups,
        "minimum_source_groups": minimum,
        "scenes": dict(sorted(Counter(selected["scene_label"]).items())),
        "cities": dict(sorted(Counter(selected["city"]).items())),
        "devices": dict(sorted(Counter(selected["device"]).items())),
        "unique_devices": int(selected["device"].nunique()),
        "additional_parts_recommended": list(pilot["additional_parts_recommended"]),
        "ready_for_intake": coverage_passed,
        "ready_for_training": False,
        "outputs": {
            "registry": {
                "path": registry_path.relative_to(root).as_posix(),
                "sha256": file_sha256(registry_path),
                "rows": int(len(selected)),
            }
        },
        "model_inference_run": False,
        "training_started": False,
        "locked_dataset_audio_read": False,
    }
    audit_path = output_dir / "tau_pilot_audit.json"
    audit_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": True,
                "integrity_passed": True,
                "coverage_passed": coverage_passed,
                "samples": report["samples"],
                "source_groups": source_groups,
                "scenes": report["scenes"],
                "cities": report["cities"],
                "unique_devices": report["unique_devices"],
                "additional_parts_recommended": report["additional_parts_recommended"],
                "ready_for_intake": coverage_passed,
                "ready_for_training": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the minimal TAU G14 background pilot.")
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
