from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .audio import decode_wav_bytes, ensure_sample_rate
from .config import load_config
from .data_firewall import file_sha256, reject_locked_path


PROTOCOL = "g14_controlled_segment_cache_v1"
SPLITS = ("train", "tune", "dev_holdout")


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    reject_locked_path(path, context="G14 segment cache")
    return path.resolve(strict=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV input: {path}")
    return rows


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _verify_inputs(
    raw_manifest: Path,
    raw_audit_path: Path,
    headers_path: Path,
    preflight_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_audit = json.loads(raw_audit_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not (
        raw_audit.get("passed") is True
        and raw_audit.get("outputs", {}).get("manifest", {}).get("sha256")
        == file_sha256(raw_manifest)
    ):
        raise ValueError("Raw manifest does not match its audit")
    if not (
        preflight.get("passed") is True
        and preflight.get("ready_for_controlled_decode") is True
        and preflight.get("outputs", {}).get("header_inventory", {}).get("sha256")
        == file_sha256(headers_path)
        and preflight.get("inputs", {}).get("raw_manifest_sha256")
        == file_sha256(raw_manifest)
    ):
        raise ValueError("Decode preflight contract mismatch")
    return raw_audit, preflight


def _expected_split_segments(preflight: dict[str, Any]) -> dict[str, int]:
    return {
        split: sum(
            int(preflight["split_summary"][split][label]["predicted_full_segments"])
            for label in ("positive", "background")
        )
        for split in SPLITS
    }


def build(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected segment-cache protocol")
    raw_manifest = _resolve(root, config["raw_manifest"])
    raw_audit_path = _resolve(root, config["raw_audit"])
    headers_path = _resolve(root, config["header_inventory"])
    preflight_path = _resolve(root, config["decode_preflight_audit"])
    _, preflight = _verify_inputs(
        raw_manifest, raw_audit_path, headers_path, preflight_path
    )
    raw_rows = _read_csv(raw_manifest)
    header_rows = _read_csv(headers_path)
    headers = {
        (row["archive_path"], row["archive_member"]): row for row in header_rows
    }
    if len(headers) != len(raw_rows):
        raise ValueError("Header inventory does not cover every raw recording")

    target_rate = int(config["audio"]["target_sample_rate"])
    clip_seconds = float(config["audio"]["clip_seconds"])
    target_samples = int(round(target_rate * clip_seconds))
    expected_by_split = _expected_split_segments(preflight)
    estimated_bytes = sum(expected_by_split.values()) * target_samples * 4
    disk = shutil.disk_usage(root)
    safety = int(config["cache"]["safety_free_bytes_after_cache"])
    if disk.free - estimated_bytes < safety:
        raise ValueError("Disk safety gate failed before cache creation")

    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".build.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("G14 segment cache build is already running") from error
        return _build_locked(
            config,
            config_path,
            root,
            raw_manifest,
            raw_audit_path,
            headers_path,
            preflight_path,
            raw_rows,
            headers,
            output_dir,
            expected_by_split,
            target_rate,
            target_samples,
        )


def _build_locked(
    config: dict[str, Any],
    config_path: Path,
    root: Path,
    raw_manifest: Path,
    raw_audit_path: Path,
    headers_path: Path,
    preflight_path: Path,
    raw_rows: list[dict[str, str]],
    headers: dict[tuple[str, str], dict[str, str]],
    output_dir: Path,
    expected_by_split: dict[str, int],
    target_rate: int,
    target_samples: int,
) -> dict[str, Any]:
    final_paths = {split: output_dir / f"{split}_audio.npy" for split in SPLITS}
    temporary_paths = {
        split: output_dir / f".{split}_audio.npy.building" for split in SPLITS
    }
    report_path = output_dir / "segment_cache_audit.json"
    manifest_path = output_dir / "g14_segment_manifest.csv"
    if report_path.is_file() or manifest_path.is_file() or any(
        path.is_file() for path in final_paths.values()
    ):
        raise FileExistsError(
            "Segment-cache outputs already exist; refusing an implicit overwrite"
        )
    for path in temporary_paths.values():
        if path.exists():
            path.unlink()

    memmaps = {
        split: np.lib.format.open_memmap(
            temporary_paths[split],
            mode="w+",
            dtype=np.float32,
            shape=(expected_by_split[split], target_samples),
        )
        for split in SPLITS
    }
    positions = {split: 0 for split in SPLITS}
    archive_handles: dict[Path, zipfile.ZipFile] = {}
    segment_rows: list[dict[str, Any]] = []
    seen_segment_hashes: dict[str, tuple[str, str, int]] = {}
    zero_rms_segments = 0
    nonfinite_recordings = 0
    raw_hash_mismatches = 0
    source_segments: Counter[tuple[str, str, int]] = Counter()
    split_label_segments: Counter[tuple[str, int]] = Counter()
    try:
        for recording_index, row in enumerate(raw_rows, start=1):
            archive_path = _resolve(root, row["archive_path"])
            if archive_path not in archive_handles:
                archive_handles[archive_path] = zipfile.ZipFile(archive_path)
            archive = archive_handles[archive_path]
            audio_bytes = archive.read(row["archive_member"])
            observed_raw_hash = _sha256_bytes(audio_bytes)
            if observed_raw_hash != row["audio_sha256"]:
                raw_hash_mismatches += 1
                raise ValueError(f"Raw audio SHA256 changed: {row['archive_member']}")
            audio, source_rate = decode_wav_bytes(audio_bytes)
            audio = ensure_sample_rate(audio, source_rate, target_rate)
            if not np.isfinite(audio).all():
                nonfinite_recordings += 1
                raise ValueError(f"Non-finite decoded audio: {row['archive_member']}")
            expected_segments = int(
                headers[(row["archive_path"], row["archive_member"])][
                    "predicted_full_segments"
                ]
            )
            observed_segments = audio.size // target_samples
            if observed_segments != expected_segments:
                raise ValueError(
                    f"Segment count changed for {row['archive_member']}: "
                    f"expected={expected_segments}, observed={observed_segments}"
                )
            split = row["split"]
            label = int(row["label"])
            start_position = positions[split]
            end_position = start_position + observed_segments
            if end_position > expected_by_split[split]:
                raise ValueError(f"Cache position overflow for split={split}")
            clips = audio[: observed_segments * target_samples].reshape(
                observed_segments, target_samples
            )
            memmaps[split][start_position:end_position] = clips
            for segment_index, clip in enumerate(clips):
                segment_hash = hashlib.sha256(
                    clip.astype("<f4", copy=False).tobytes()
                ).hexdigest()
                if segment_hash in seen_segment_hashes:
                    previous = seen_segment_hashes[segment_hash]
                    raise ValueError(
                        "Exact duplicate resampled segment: "
                        f"current={(row['archive_member'], segment_index)}, "
                        f"previous={previous}"
                    )
                seen_segment_hashes[segment_hash] = (
                    row["archive_member"],
                    split,
                    segment_index,
                )
                rms = float(np.sqrt(np.mean(np.square(clip, dtype=np.float64))))
                peak = float(np.max(np.abs(clip)))
                if rms <= 1e-12:
                    zero_rms_segments += 1
                cache_index = start_position + segment_index
                segment_rows.append(
                    {
                        "dataset": row["dataset"],
                        "split": split,
                        "label": label,
                        "source_group": row["source_group"],
                        "cache_path": final_paths[split].relative_to(root).as_posix(),
                        "cache_index": cache_index,
                        "segment_index": segment_index,
                        "start_sample": segment_index * target_samples,
                        "end_sample": (segment_index + 1) * target_samples,
                        "sample_rate": target_rate,
                        "audio_sha256": row["audio_sha256"],
                        "segment_sha256": segment_hash,
                        "rms": rms,
                        "peak": peak,
                        "archive_path": row["archive_path"],
                        "archive_member": row["archive_member"],
                        "domain_type": row["domain_type"],
                        "domain": row["domain"],
                        "subtype": row["subtype"],
                        "device": row["device"],
                        "acquisition_date": row["acquisition_date"],
                        "rotor_layout": row["rotor_layout"],
                        "height_m": row["height_m"],
                        "distance_m": row["distance_m"],
                        "scene_label": row["scene_label"],
                        "license": row["license"],
                    }
                )
            positions[split] = end_position
            split_label_segments[(split, label)] += observed_segments
            source_segments[(split, row["source_group"], label)] += observed_segments
            if recording_index % 2500 == 0 or recording_index == len(raw_rows):
                print(
                    f"G14 cache: {recording_index}/{len(raw_rows)} recordings, "
                    f"segments={len(segment_rows)}",
                    flush=True,
                )
        if positions != expected_by_split:
            raise ValueError(
                f"Final cache positions mismatch: {positions} != {expected_by_split}"
            )
        for memmap in memmaps.values():
            memmap.flush()
        memmaps.clear()
        for split in SPLITS:
            temporary_paths[split].replace(final_paths[split])
    except BaseException:
        memmaps.clear()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise
    finally:
        for handle in archive_handles.values():
            handle.close()

    _atomic_csv(manifest_path, segment_rows)
    cache_outputs = {}
    for split, path in final_paths.items():
        cache_outputs[split] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
            "shape": [expected_by_split[split], target_samples],
            "dtype": "float32",
            "bytes": path.stat().st_size,
        }
    source_counts_by_split_label = Counter(
        (split, label) for split, _, label in source_segments
    )
    segment_counts_by_split_label = {
        split: {
            "positive": split_label_segments[(split, 1)],
            "background": split_label_segments[(split, 0)],
            "positive_source_groups": source_counts_by_split_label[(split, 1)],
            "background_source_groups": source_counts_by_split_label[(split, 0)],
        }
        for split in SPLITS
    }
    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_recordings_decoded": len(raw_rows),
        "segments": len(segment_rows),
        "unique_segment_sha256": len(seen_segment_hashes),
        "exact_duplicate_segments": 0,
        "zero_rms_segments": zero_rms_segments,
        "nonfinite_recordings": nonfinite_recordings,
        "raw_audio_sha256_mismatches": raw_hash_mismatches,
        "target_sample_rate": target_rate,
        "target_samples": target_samples,
        "split_summary": segment_counts_by_split_label,
        "source_segment_minimum": min(source_segments.values()),
        "source_segment_maximum": max(source_segments.values()),
        "free_bytes_after_cache": shutil.disk_usage(root).free,
        "disk_safety_gate_passed": (
            shutil.disk_usage(root).free
            >= int(config["cache"]["safety_free_bytes_after_cache"])
        ),
        "ready_for_sampler_preflight": True,
        "ready_for_training": False,
        "training_blockers": ["source_balanced_sampler_not_configured"],
        "peak_normalization_applied": False,
        "model_inference_run": False,
        "training_started": False,
        "inputs": {
            "config_sha256": file_sha256(config_path),
            "raw_manifest_sha256": file_sha256(raw_manifest),
            "raw_audit_sha256": file_sha256(raw_audit_path),
            "header_inventory_sha256": file_sha256(headers_path),
            "decode_preflight_sha256": file_sha256(preflight_path),
        },
        "outputs": {
            "segment_manifest": {
                "path": manifest_path.relative_to(root).as_posix(),
                "sha256": file_sha256(manifest_path),
                "rows": len(segment_rows),
            },
            "split_memmaps": cache_outputs,
        },
    }
    _atomic_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode G14 ZIP members into deterministic split memmaps."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g14_segment_cache.yaml")
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    build(args.config, args.root)


if __name__ == "__main__":
    main()
