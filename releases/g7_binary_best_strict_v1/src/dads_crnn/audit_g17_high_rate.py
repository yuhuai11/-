from __future__ import annotations

import argparse
import csv
import json
import math
import wave
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import roc_auc_score

from .config import load_config
from .data_firewall import (
    audit_csv_rows,
    file_sha256,
    load_forbidden_hashes,
    reject_locked_path,
)


PROTOCOL = "g17_p0_high_rate_feasibility_audit_v1"
BANDS = ((0.0, 8000.0), (8000.0, 12000.0), (12000.0, 16000.0), (16000.0, 20000.0))


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    reject_locked_path(path, context="G17 P0 development input")
    return path.resolve(strict=True)


def _safe_member(value: str) -> str:
    member = PurePosixPath(value)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError(f"Unsafe archive member: {value}")
    return member.as_posix()


def select_stratified_rows(
    rows: Iterable[dict[str, str]], max_recordings_per_source_split: int
) -> list[dict[str, str]]:
    if max_recordings_per_source_split <= 0:
        raise ValueError("max_recordings_per_source_split must be positive")
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row["split"], row["label"], row["source_group"])
        groups[key].append(row)
    selected = []
    for key in sorted(groups):
        ordered = sorted(
            groups[key],
            key=lambda row: (row["audio_sha256"], row["archive_member"]),
        )
        if len(ordered) <= max_recordings_per_source_split:
            selected.extend(ordered)
            continue
        positions = np.linspace(
            0, len(ordered) - 1, max_recordings_per_source_split, dtype=np.int64
        )
        selected.extend(ordered[int(position)] for position in positions)
    return selected


def decode_pcm(raw: bytes, sample_width: int, channels: int) -> np.ndarray:
    if channels <= 0:
        raise ValueError("Invalid channel count")
    if sample_width == 1:
        values = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8)
        if packed.size % 3:
            raise ValueError("Malformed 24-bit PCM payload")
        triples = packed.reshape(-1, 3).astype(np.int32)
        integers = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        integers = np.where(integers & 0x800000, integers - 0x1000000, integers)
        values = integers.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        values = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported PCM sample width: {sample_width}")
    usable = values.size - values.size % channels
    return values[:usable].reshape(-1, channels).mean(axis=1)


def spectral_band_features(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    bands: tuple[tuple[float, float], ...] = BANDS,
) -> dict[str, float]:
    if sample_rate <= 0 or waveform.size < 64:
        raise ValueError("Audio window is too short")
    signal = np.asarray(waveform, dtype=np.float64)
    signal = signal - signal.mean()
    window = np.hanning(signal.size)
    power = np.abs(np.fft.rfft(signal * window)) ** 2
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    total = float(power[(frequencies >= 0.0) & (frequencies <= 20000.0)].sum())
    total = max(total, np.finfo(np.float64).tiny)
    features: dict[str, float] = {}
    for low, high in bands:
        mask = (frequencies >= low) & (frequencies < min(high, sample_rate / 2))
        energy = float(power[mask].sum()) if np.any(mask) else 0.0
        features[f"band_{int(low)}_{int(high)}_ratio"] = energy / total
    high_ratio = sum(
        value for key, value in features.items() if not key.startswith("band_0_")
    )
    features["above_8000_ratio"] = high_ratio
    features["above_8000_db"] = 10.0 * math.log10(max(high_ratio, 1.0e-15))
    return features


def _read_windows(
    archive: zipfile.ZipFile,
    member: str,
    *,
    windows_per_recording: int,
    window_seconds: float,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    with archive.open(member) as stream:
        with wave.open(stream, "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
            compression = wav.getcomptype()
            if compression != "NONE":
                raise ValueError(f"Compressed WAV is unsupported: {member}")
            window_frames = min(frames, max(64, int(round(window_seconds * sample_rate))))
            maximum_start = max(0, frames - window_frames)
            starts = np.linspace(
                0, maximum_start, min(windows_per_recording, max(1, frames // window_frames)),
                dtype=np.int64,
            )
            windows = []
            for start in sorted(set(int(value) for value in starts)):
                wav.setpos(start)
                raw = wav.readframes(window_frames)
                windows.append(decode_pcm(raw, sample_width, channels))
    header = {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frames": frames,
        "duration_seconds": frames / sample_rate,
    }
    return header, windows


def _source_explained_fraction(values: np.ndarray, sources: np.ndarray) -> float:
    total = float(np.var(values))
    if total <= 1.0e-15:
        return 1.0
    global_mean = float(np.mean(values))
    between = 0.0
    for source in np.unique(sources):
        group = values[sources == source]
        between += len(group) * (float(np.mean(group)) - global_mean) ** 2
    return min(1.0, between / (len(values) * total))


def summarize_rows(
    rows: list[dict[str, Any]], gates: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = []
    for (split, label), group in sorted(
        _group_rows(rows, ("split", "label")).items()
    ):
        ratios = np.asarray([row["above_8000_ratio"] for row in group], dtype=np.float64)
        summary.append(
            {
                "split": split,
                "label": int(label),
                "recordings": len(group),
                "source_groups": len({row["source_group"] for row in group}),
                **{
                    f"{key}_median": float(
                        np.median([float(row[key]) for row in group])
                    )
                    for key in (
                        "band_0_8000_ratio",
                        "band_8000_12000_ratio",
                        "band_12000_16000_ratio",
                        "band_16000_20000_ratio",
                    )
                },
                "above_8000_ratio_mean": float(np.mean(ratios)),
                "above_8000_ratio_median": float(np.median(ratios)),
                "above_8000_ratio_p10": float(np.quantile(ratios, 0.10)),
                "above_8000_ratio_p90": float(np.quantile(ratios, 0.90)),
                "high_frequency_present_fraction": float(
                    np.mean(ratios >= float(gates["minimum_above_8000_ratio"]))
                ),
            }
        )

    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    ratios = np.asarray([row["above_8000_ratio"] for row in rows], dtype=np.float64)
    auc = float(roc_auc_score(labels, ratios))
    directionless_auc = max(auc, 1.0 - auc)
    positive = labels == 1
    positive_sources = np.asarray(
        [row["source_group"] for row in rows], dtype=object
    )[positive]
    positive_ratios = ratios[positive]
    source_presence = []
    for source in sorted(set(positive_sources)):
        source_values = positive_ratios[positive_sources == source]
        source_presence.append(
            float(np.mean(source_values >= float(gates["minimum_above_8000_ratio"])))
        )
    diagnostics = {
        "uav_background_high_band_auc": auc,
        "uav_background_directionless_auc": directionless_auc,
        "positive_source_groups": len(set(positive_sources)),
        "positive_source_groups_meeting_prevalence": sum(
            value >= float(gates["minimum_source_prevalence"]) for value in source_presence
        ),
        "positive_source_group_prevalence_fraction": float(
            np.mean(
                np.asarray(source_presence) >= float(gates["minimum_source_prevalence"])
            )
        ),
        "positive_source_explained_fraction": _source_explained_fraction(
            np.log10(np.maximum(positive_ratios, 1.0e-15)), positive_sources
        ),
    }
    return summary, diagnostics


def _group_rows(
    rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    result: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[tuple(row[field] for field in fields)].append(row)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected G17 P0 protocol")

    manifest_path = _resolve(root, config["inputs"]["raw_manifest"]["path"])
    raw_audit_path = _resolve(root, config["inputs"]["raw_audit"]["path"])
    firewall_path = _resolve(root, config["inputs"]["firewall_report"]["path"])
    consumed_registry = _resolve(root, config["inputs"]["consumed_hash_registry"]["path"])
    expected = config["inputs"]
    for key, path in (
        ("raw_manifest", manifest_path),
        ("raw_audit", raw_audit_path),
        ("firewall_report", firewall_path),
        ("consumed_hash_registry", consumed_registry),
    ):
        if file_sha256(path) != str(expected[key]["sha256"]):
            raise ValueError(f"G17 input SHA256 mismatch: {key}")

    raw_audit = json.loads(raw_audit_path.read_text(encoding="utf-8"))
    firewall = json.loads(firewall_path.read_text(encoding="utf-8"))
    if not (
        raw_audit.get("passed") is True
        and raw_audit.get("source_group_overlap_across_splits") == 0
        and raw_audit.get("exact_duplicate_hash_groups") == 0
    ):
        raise ValueError("G14 raw manifest audit is not clean")
    if not (
        firewall.get("passed") is True
        and firewall.get("g13_status") == "consumed_and_closed"
        and firewall.get("locked_dataset_audio_read") is False
    ):
        raise ValueError("Consumed-data firewall is not valid")

    forbidden_hashes = load_forbidden_hashes(consumed_registry)
    manifest_rows = audit_csv_rows(
        manifest_path,
        forbidden_hashes=forbidden_hashes,
        required_columns=(
            "archive_path",
            "archive_member",
            "label",
            "split",
            "source_group",
            "audio_sha256",
        ),
    )
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = select_stratified_rows(
        rows, int(config["sampling"]["max_recordings_per_source_split"])
    )

    archive_handles: dict[Path, zipfile.ZipFile] = {}
    spectral_rows = []
    try:
        for index, row in enumerate(selected, start=1):
            archive_path = _resolve(root, row["archive_path"])
            member = _safe_member(row["archive_member"])
            if archive_path not in archive_handles:
                archive_handles[archive_path] = zipfile.ZipFile(archive_path)
            header, windows = _read_windows(
                archive_handles[archive_path],
                member,
                windows_per_recording=int(config["sampling"]["windows_per_recording"]),
                window_seconds=float(config["sampling"]["window_seconds"]),
            )
            features = [
                spectral_band_features(window, int(header["sample_rate"]))
                for window in windows
            ]
            aggregated = {
                key: float(np.median([feature[key] for feature in features]))
                for key in features[0]
            }
            spectral_rows.append(
                {
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "label": int(row["label"]),
                    "source_group": row["source_group"],
                    "audio_sha256": row["audio_sha256"],
                    "archive_path": row["archive_path"],
                    "archive_member": member,
                    **header,
                    "windows_analyzed": len(windows),
                    **aggregated,
                    "conservative_spectral_cutoff_suspected": aggregated[
                        "above_8000_ratio"
                    ]
                    < float(config["gates"]["minimum_above_8000_ratio"]),
                }
            )
            if index % 50 == 0 or index == len(selected):
                print(f"G17 P0 spectral audit: {index}/{len(selected)}", flush=True)
    finally:
        for archive in archive_handles.values():
            archive.close()

    gates = config["gates"]
    summary, diagnostics = summarize_rows(spectral_rows, gates)
    sample_rates = Counter(int(row["sample_rate"]) for row in spectral_rows)
    format_counts = Counter(
        (
            int(row["sample_rate"]),
            int(row["channels"]),
            int(row["sample_width_bytes"]),
        )
        for row in spectral_rows
    )
    license_counts = Counter(str(row["license"]) for row in selected)
    high_rate_fraction = float(
        np.mean(
            [int(row["sample_rate"]) >= int(gates["minimum_original_sample_rate"])
             for row in spectral_rows]
        )
    )
    positive_rows = [row for row in spectral_rows if int(row["label"]) == 1]
    positive_high_fraction = float(
        np.mean(
            [
                row["above_8000_ratio"] >= float(gates["minimum_above_8000_ratio"])
                for row in positive_rows
            ]
        )
    )
    gate_results = {
        "original_high_rate_audio": high_rate_fraction
        >= float(gates["minimum_high_rate_fraction"]),
        "uav_high_frequency_prevalence": positive_high_fraction
        >= float(gates["minimum_uav_high_frequency_fraction"]),
        "uav_background_separability": diagnostics[
            "uav_background_directionless_auc"
        ]
        >= float(gates["minimum_directionless_auc"]),
        "multi_source_support": (
            diagnostics["positive_source_groups"]
            >= int(gates["minimum_positive_source_groups"])
            and diagnostics["positive_source_group_prevalence_fraction"]
            >= float(gates["minimum_positive_source_group_fraction"])
        ),
        "source_dominance_control": diagnostics[
            "positive_source_explained_fraction"
        ]
        <= float(gates["maximum_positive_source_explained_fraction"]),
        "source_split_isolation": raw_audit["source_group_overlap_across_splits"] == 0,
        "audio_hash_isolation": raw_audit["exact_duplicate_hash_groups"] == 0,
        "consumed_data_firewall": len(forbidden_hashes)
        == int(config["firewall"]["expected_consumed_hashes"]),
    }
    passed = all(gate_results.values())
    decision = "proceed_to_g17_p1" if passed else "terminate_high_rate_branch"

    output_dir = root / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "audio_inventory.csv"
    summary_path = output_dir / "spectral_band_summary.csv"
    overlap_path = output_dir / "source_overlap_audit.json"
    decision_path = output_dir / "decision.json"
    _write_csv(inventory_path, spectral_rows)
    _write_csv(summary_path, summary)

    overlap = {
        "protocol": PROTOCOL,
        "source_group_overlap_across_splits": 0,
        "exact_duplicate_hash_groups": 0,
        "consumed_hash_overlap": 0,
        "dataset_label_confounding": True,
        "dataset_label_confounding_note": (
            "Kielce supplies UAV positives and TAU supplies backgrounds; high-band "
            "separability is feasibility evidence, not proof of causal UAV features."
        ),
        "locked_datasets_read": [],
    }
    overlap_path.write_text(
        json.dumps(overlap, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = {
        "passed": passed,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "formal_training_started": False,
        "checkpoint_written": False,
        "model_inference_run": False,
        "locked_datasets_read": [],
        "manifest_rows_audited_for_firewall": manifest_rows,
        "recordings_spectrally_audited": len(spectral_rows),
        "source_groups_spectrally_audited": len(
            {row["source_group"] for row in spectral_rows}
        ),
        "sample_rate_counts": dict(sorted(sample_rates.items())),
        "wav_format_counts": [
            {
                "sample_rate": key[0],
                "channels": key[1],
                "sample_width_bytes": key[2],
                "recordings": count,
            }
            for key, count in sorted(format_counts.items())
        ],
        "license_counts": dict(sorted(license_counts.items())),
        "high_rate_recording_fraction": high_rate_fraction,
        "uav_high_frequency_present_fraction": positive_high_fraction,
        "uav_conservative_spectral_cutoff_suspected_fraction": (
            1.0 - positive_high_fraction
        ),
        "pseudo_upsampling_interpretation": (
            "No original-rate file is labeled pseudo-upsampled. The conservative "
            "spectral-cutoff flag only identifies weak >8 kHz energy and is not, by "
            "itself, proof of resampling history."
        ),
        "diagnostics": diagnostics,
        "gate_results": gate_results,
        "failed_gates": [name for name, value in gate_results.items() if not value],
        "inputs": {
            "config_sha256": file_sha256(config_path),
            "raw_manifest_sha256": file_sha256(manifest_path),
            "raw_audit_sha256": file_sha256(raw_audit_path),
            "firewall_report_sha256": file_sha256(firewall_path),
            "consumed_hash_registry_sha256": file_sha256(consumed_registry),
        },
        "outputs": {},
    }
    for name, path, count in (
        ("audio_inventory", inventory_path, len(spectral_rows)),
        ("spectral_band_summary", summary_path, len(summary)),
        ("source_overlap_audit", overlap_path, 1),
    ):
        report["outputs"][name] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
            "rows": count,
        }
    decision_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the read-only G17-P0 high-rate audit.")
    parser.add_argument("--config", type=Path, default=Path("configs/g17_p0_high_rate.yaml"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    audit(args.config, args.root)


if __name__ == "__main__":
    main()
