from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .data_firewall import (
    audit_csv_rows,
    file_sha256,
    load_forbidden_hashes,
    reject_locked_path,
)


PROTOCOL = "g18_p0_open_set_model_id_registry_v1"


def normalize_model(value: object) -> str:
    return "".join(character for character in str(value).upper() if character.isalnum())


def split_recordings(
    hashes: list[str], fractions: dict[str, float], seed: int, model: str
) -> dict[str, set[str]]:
    names = ("train", "tune", "holdout")
    if set(fractions) != set(names) or not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError("Known-model split fractions must be train/tune/holdout and sum to 1")
    ordered = sorted(
        set(hashes),
        key=lambda value: hashlib.sha256(
            f"{seed}:{model}:{value}".encode("utf-8")
        ).hexdigest(),
    )
    if len(ordered) < 3:
        raise ValueError(f"Model {model} has fewer than three raw recordings")
    tune_count = max(1, int(round(len(ordered) * fractions["tune"])))
    holdout_count = max(1, int(round(len(ordered) * fractions["holdout"])))
    if tune_count + holdout_count >= len(ordered):
        tune_count = holdout_count = 1
    train_count = len(ordered) - tune_count - holdout_count
    return {
        "train": set(ordered[:train_count]),
        "tune": set(ordered[train_count : train_count + tune_count]),
        "holdout": set(ordered[train_count + tune_count :]),
    }


def cap_recording_segments(
    rows: list[dict[str, str]], maximum: int
) -> list[dict[str, str]]:
    if maximum <= 0:
        raise ValueError("maximum segments per recording must be positive")
    ordered = sorted(
        rows,
        key=lambda row: (int(row["segment_index"]), row["segment_sha256"]),
    )
    if len(ordered) <= maximum:
        return ordered
    positions = np.linspace(0, len(ordered) - 1, maximum, dtype=np.int64)
    return [ordered[int(position)] for position in positions]


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    reject_locked_path(path, context="G18 model-ID development input")
    return path.resolve(strict=True)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    if not rows:
        raise ValueError(f"Empty G18 partition: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected G18 P0 protocol")

    inputs = config["inputs"]
    segment_manifest = _resolve(root, inputs["segment_manifest"]["path"])
    segment_audit = _resolve(root, inputs["segment_audit"]["path"])
    firewall_report = _resolve(root, inputs["firewall_report"]["path"])
    consumed_registry = _resolve(root, inputs["consumed_hash_registry"]["path"])
    g7_checkpoint = _resolve(root, inputs["g7_checkpoint"]["path"])
    paths = {
        "segment_manifest": segment_manifest,
        "segment_audit": segment_audit,
        "firewall_report": firewall_report,
        "consumed_hash_registry": consumed_registry,
        "g7_checkpoint": g7_checkpoint,
    }
    for name, path in paths.items():
        if file_sha256(path) != str(inputs[name]["sha256"]):
            raise ValueError(f"G18 input SHA256 mismatch: {name}")

    firewall = json.loads(firewall_report.read_text(encoding="utf-8"))
    segment_report = json.loads(segment_audit.read_text(encoding="utf-8"))
    if not (
        firewall.get("passed") is True
        and firewall.get("g13_status") == "consumed_and_closed"
        and firewall.get("locked_dataset_audio_read") is False
    ):
        raise ValueError("G18 consumed-data firewall is not valid")
    if not (
        segment_report.get("passed") is True
        and segment_report.get("protocol") == "g14_controlled_segment_cache_v1"
    ):
        raise ValueError("G18 segment cache audit is not valid")

    forbidden_hashes = load_forbidden_hashes(consumed_registry)
    manifest_rows = audit_csv_rows(
        segment_manifest,
        forbidden_hashes=forbidden_hashes,
        required_columns=(
            "dataset",
            "label",
            "audio_sha256",
            "segment_sha256",
            "segment_index",
            "cache_path",
            "cache_index",
            "subtype",
        ),
    )
    with segment_manifest.open("r", encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    uav_rows = [
        row
        for row in all_rows
        if row["dataset"] == "kielce_17_uav" and int(row["label"]) == 1
    ]
    if not uav_rows:
        raise ValueError("No Kielce UAV segments found")
    for row in uav_rows:
        row["model_id"] = normalize_model(row["subtype"])

    known = [normalize_model(value) for value in config["models"]["known"]]
    unknown_tune = [
        normalize_model(value) for value in config["models"]["unknown_tune"]
    ]
    unknown_holdout = [
        normalize_model(value) for value in config["models"]["unknown_holdout"]
    ]
    configured = known + unknown_tune + unknown_holdout
    if len(configured) != len(set(configured)):
        raise ValueError("G18 model partitions overlap")
    observed_models = {row["model_id"] for row in uav_rows}
    if set(configured) != observed_models:
        raise ValueError(
            f"G18 model registry does not cover data: configured={set(configured)}, "
            f"observed={observed_models}"
        )
    class_to_index = {model: index for index, model in enumerate(known)}

    by_model_recording: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in uav_rows:
        by_model_recording[(row["model_id"], row["audio_sha256"])].append(row)
    capped: dict[tuple[str, str], list[dict[str, str]]] = {
        key: cap_recording_segments(
            values, int(config["sampling"]["maximum_segments_per_recording"])
        )
        for key, values in by_model_recording.items()
    }

    partitions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    recording_partitions: dict[str, str] = {}
    fractions = {
        name: float(config["known_split_fractions"][name])
        for name in ("train", "tune", "holdout")
    }
    seed = int(config["seed"])
    for model in known:
        hashes = [audio_hash for item_model, audio_hash in capped if item_model == model]
        split = split_recordings(hashes, fractions, seed, model)
        for partition, values in split.items():
            for audio_hash in values:
                recording_partitions[audio_hash] = f"known_{partition}"
    for model in unknown_tune:
        for item_model, audio_hash in capped:
            if item_model == model:
                recording_partitions[audio_hash] = "unknown_tune"
    for model in unknown_holdout:
        for item_model, audio_hash in capped:
            if item_model == model:
                recording_partitions[audio_hash] = "unknown_holdout"

    fields = [
        "partition",
        "model_id",
        "target_index",
        "is_known",
        "source_group",
        "audio_sha256",
        "segment_sha256",
        "segment_index",
        "cache_path",
        "cache_index",
        "archive_member",
        "device",
        "distance_m",
        "height_m",
        "rotor_layout",
        "license",
    ]
    for (model, audio_hash), rows in capped.items():
        partition = recording_partitions[audio_hash]
        for row in rows:
            partitions[partition].append(
                {
                    "partition": partition,
                    "model_id": model,
                    "target_index": class_to_index.get(model, -1),
                    "is_known": model in class_to_index,
                    **{field: row[field] for field in fields[4:]},
                }
            )

    expected_partitions = (
        "known_train",
        "known_tune",
        "known_holdout",
        "unknown_tune",
        "unknown_holdout",
    )
    if set(partitions) != set(expected_partitions):
        raise ValueError(f"Incomplete G18 partitions: {sorted(partitions)}")
    hash_sets = {
        name: {row["audio_sha256"] for row in partitions[name]}
        for name in expected_partitions
    }
    overlaps = {}
    for index, left in enumerate(expected_partitions):
        for right in expected_partitions[index + 1 :]:
            overlaps[f"{left}_vs_{right}"] = len(hash_sets[left] & hash_sets[right])
    if any(overlaps.values()):
        raise ValueError(f"G18 raw recording leakage: {overlaps}")

    output_dir = root / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name in expected_partitions:
        rows = sorted(
            partitions[name],
            key=lambda row: (
                row["model_id"],
                row["audio_sha256"],
                int(row["segment_index"]),
            ),
        )
        path = output_dir / f"{name}.csv"
        _write_csv(path, rows, fields)
        outputs[name] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
            "rows": len(rows),
            "raw_recordings": len(hash_sets[name]),
            "model_counts": dict(sorted(Counter(row["model_id"] for row in rows).items())),
        }

    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "formal_training_started": False,
        "checkpoint_written": False,
        "model_inference_run": False,
        "locked_datasets_read": [],
        "manifest_rows_firewall_audited": manifest_rows,
        "uav_segments_available": len(uav_rows),
        "known_models": known,
        "class_to_index": class_to_index,
        "unknown_tune_models": unknown_tune,
        "unknown_holdout_models": unknown_holdout,
        "split_unit": "raw_audio_sha256",
        "maximum_segments_per_recording": int(
            config["sampling"]["maximum_segments_per_recording"]
        ),
        "recording_hash_overlaps": overlaps,
        "limitations": [
            "single_recorder_OLYMPUS_LS11",
            "mostly_single_acquisition_session_per_model",
            "recording_level_split_is_not_cross_device_validation",
            "unknown_tune_and_unknown_holdout_cover_only_two_models_each",
        ],
        "inputs": {
            "config_sha256": file_sha256(config_path),
            **{f"{name}_sha256": file_sha256(path) for name, path in paths.items()},
        },
        "outputs": outputs,
    }
    audit_path = output_dir / "audit.json"
    audit_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the G18 open-set model-ID registry.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g18_model_id_registry.yaml")
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    prepare(args.config, args.root)


if __name__ == "__main__":
    main()
