from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import wave
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .config import load_config
from .data_firewall import file_sha256, reject_locked_path


PROTOCOL = "g14_controlled_decode_preflight_v1"
SPLITS = ("train", "tune", "dev_holdout")


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    reject_locked_path(path, context="G14 decode preflight")
    return path.resolve(strict=True)


def predicted_resampled_samples(
    frames: int, source_rate: int, target_rate: int
) -> int:
    if frames < 0 or source_rate <= 0 or target_rate <= 0:
        raise ValueError("Invalid WAV frame/rate values")
    return math.ceil(frames * target_rate / source_rate)


def predicted_full_segments(
    frames: int, source_rate: int, target_rate: int, target_samples: int
) -> int:
    return predicted_resampled_samples(frames, source_rate, target_rate) // target_samples


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Combined raw manifest is empty")
    return rows


def audit(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected decode-preflight protocol")
    manifest_path = _resolve(root, config["raw_manifest"])
    raw_audit_path = _resolve(root, config["raw_audit"])
    raw_audit = json.loads(raw_audit_path.read_text(encoding="utf-8"))
    if not (
        raw_audit.get("passed") is True
        and raw_audit.get("protocol") == "g14_combined_raw_manifest_v1"
        and raw_audit.get("outputs", {}).get("manifest", {}).get("sha256")
        == file_sha256(manifest_path)
    ):
        raise ValueError("Combined raw manifest no longer matches its audit")

    rows = _load_rows(manifest_path)
    audio = config["audio"]
    target_rate = int(audio["target_sample_rate"])
    clip_seconds = float(audio["clip_seconds"])
    target_samples = int(round(target_rate * clip_seconds))
    if target_samples <= 0:
        raise ValueError("Invalid target clip length")

    archive_handles: dict[Path, zipfile.ZipFile] = {}
    format_counts: Counter[tuple[Any, ...]] = Counter()
    dataset_formats: Counter[tuple[Any, ...]] = Counter()
    split_label_recordings: Counter[tuple[str, int]] = Counter()
    split_label_segments: Counter[tuple[str, int]] = Counter()
    split_label_duration: Counter[tuple[str, int]] = Counter()
    source_segments: Counter[str] = Counter()
    unsupported = []
    zero_segment_recordings = []
    header_rows = []
    try:
        for row in rows:
            archive_path = _resolve(root, row["archive_path"])
            member_path = PurePosixPath(row["archive_member"])
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe archive member: {row['archive_member']}")
            if archive_path not in archive_handles:
                archive_handles[archive_path] = zipfile.ZipFile(archive_path)
            archive = archive_handles[archive_path]
            try:
                with archive.open(row["archive_member"]) as stream:
                    with wave.open(stream, "rb") as wav:
                        channels = wav.getnchannels()
                        sample_width = wav.getsampwidth()
                        sample_rate = wav.getframerate()
                        frames = wav.getnframes()
                        compression = wav.getcomptype()
            except (KeyError, wave.Error, EOFError) as error:
                raise ValueError(
                    f"Cannot read WAV header: {row['archive_member']}: {error}"
                ) from error
            format_key = (
                sample_rate,
                channels,
                sample_width,
                compression,
            )
            format_counts[format_key] += 1
            dataset_formats[(row["dataset"], *format_key)] += 1
            if compression != "NONE" or sample_width not in (1, 2, 3, 4):
                unsupported.append(
                    {
                        "archive_member": row["archive_member"],
                        "sample_width": sample_width,
                        "compression": compression,
                    }
                )
            segments = predicted_full_segments(
                frames, sample_rate, target_rate, target_samples
            )
            split = row["split"]
            label = int(row["label"])
            split_label_recordings[(split, label)] += 1
            split_label_segments[(split, label)] += segments
            split_label_duration[(split, label)] += frames / sample_rate
            source_segments[row["source_group"]] += segments
            if segments <= 0:
                zero_segment_recordings.append(row["archive_member"])
            header_rows.append(
                {
                    "dataset": row["dataset"],
                    "archive_path": row["archive_path"],
                    "archive_member": row["archive_member"],
                    "split": split,
                    "label": label,
                    "source_group": row["source_group"],
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "sample_width_bytes": sample_width,
                    "compression": compression,
                    "frames": frames,
                    "duration_seconds": frames / sample_rate,
                    "predicted_resampled_samples": predicted_resampled_samples(
                        frames, sample_rate, target_rate
                    ),
                    "predicted_full_segments": segments,
                    "tail_seconds_dropped": (
                        predicted_resampled_samples(frames, sample_rate, target_rate)
                        % target_samples
                    )
                    / target_rate,
                }
            )
    finally:
        for handle in archive_handles.values():
            handle.close()

    if unsupported:
        raise ValueError(f"Unsupported WAV formats found: {len(unsupported)}")
    if zero_segment_recordings:
        raise ValueError(
            f"Recordings shorter than one full clip: {len(zero_segment_recordings)}"
        )

    segments_total = sum(split_label_segments.values())
    data_bytes = segments_total * target_samples * 4
    index_bytes_estimate = segments_total * 512
    estimated_cache_bytes = data_bytes + index_bytes_estimate
    disk = shutil.disk_usage(root)
    safety = int(config["cache"]["safety_free_bytes_after_cache"])
    projected_free = disk.free - estimated_cache_bytes
    disk_passed = projected_free >= safety
    if not disk_passed:
        raise ValueError(
            "Insufficient disk for controlled cache: "
            f"projected_free={projected_free}, required={safety}"
        )

    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    headers_path = output_dir / "wav_header_inventory.csv"
    with headers_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header_rows[0]))
        writer.writeheader()
        writer.writerows(header_rows)

    split_summary = {}
    for split in SPLITS:
        split_summary[split] = {}
        for label, name in ((1, "positive"), (0, "background")):
            recordings = split_label_recordings[(split, label)]
            segments = split_label_segments[(split, label)]
            split_summary[split][name] = {
                "raw_recordings": recordings,
                "duration_hours": split_label_duration[(split, label)] / 3600,
                "predicted_full_segments": segments,
            }
        positive = split_label_segments[(split, 1)]
        background = split_label_segments[(split, 0)]
        split_summary[split]["predicted_background_to_positive_segment_ratio"] = (
            background / positive
        )

    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_recordings": len(rows),
        "wav_format_counts": [
            {
                "sample_rate": key[0],
                "channels": key[1],
                "sample_width_bytes": key[2],
                "compression": key[3],
                "recordings": count,
            }
            for key, count in sorted(format_counts.items())
        ],
        "dataset_wav_format_counts": [
            {
                "dataset": key[0],
                "sample_rate": key[1],
                "channels": key[2],
                "sample_width_bytes": key[3],
                "compression": key[4],
                "recordings": count,
            }
            for key, count in sorted(dataset_formats.items())
        ],
        "unsupported_wav_formats": len(unsupported),
        "zero_segment_recordings": len(zero_segment_recordings),
        "target_sample_rate": target_rate,
        "clip_seconds": clip_seconds,
        "target_samples": target_samples,
        "segmentation": audio["segmentation"],
        "predicted_segments": segments_total,
        "split_summary": split_summary,
        "source_segment_minimum": min(source_segments.values()),
        "source_segment_maximum": max(source_segments.values()),
        "cache_layout": config["cache"]["layout"],
        "estimated_float32_audio_bytes": data_bytes,
        "estimated_index_bytes": index_bytes_estimate,
        "estimated_cache_bytes": estimated_cache_bytes,
        "free_bytes_before_cache": disk.free,
        "projected_free_bytes_after_cache": projected_free,
        "required_safety_free_bytes": safety,
        "disk_gate_passed": disk_passed,
        "ready_for_controlled_decode": True,
        "ready_for_training": False,
        "training_blockers": [
            "controlled_decode_cache_not_built",
            "source_balanced_sampler_not_configured",
        ],
        "audio_payload_decoded": False,
        "model_inference_run": False,
        "training_started": False,
        "inputs": {
            "config_sha256": file_sha256(config_path),
            "raw_manifest_sha256": file_sha256(manifest_path),
            "raw_audit_sha256": file_sha256(raw_audit_path),
        },
        "outputs": {
            "header_inventory": {
                "path": headers_path.relative_to(root).as_posix(),
                "sha256": file_sha256(headers_path),
                "rows": len(header_rows),
            }
        },
    }
    report_path = output_dir / "decode_preflight_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit WAV headers and predict G14 cache size before decoding."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g14_decode_preflight.yaml")
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    audit(args.config, args.root)


if __name__ == "__main__":
    main()
