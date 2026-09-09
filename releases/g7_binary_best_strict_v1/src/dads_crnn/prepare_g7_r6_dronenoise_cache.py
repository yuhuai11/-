from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audio import decode_wav_bytes, ensure_sample_rate
from .data_firewall import file_sha256


PROTOCOL = "g7_r6_dronenoise_native_halfsecond_cache_v1"
TARGET_SAMPLE_RATE = 16_000
TARGET_SAMPLES = 8_000
ROLES = ("train", "validation", "test")
MODEL_COLUMNS = [
    "dataset_role",
    "split",
    "dataset_origin",
    "label",
    "cache_path",
    "cache_index",
    "cache_start_sample",
    "cache_end_sample",
    "source_path",
    "recording_group",
    "source_group",
    "domain_bucket",
    "background_mix_eligible",
    "audio_sha256",
    "segment_sha256",
    "half_index",
    "uav_subtype",
    "uav_novelty",
    "device",
    "scene",
    "license",
]


def segment_sha256(waveform: np.ndarray) -> str:
    values = np.asarray(waveform, dtype="<f4")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def complete_window_count(samples: int) -> int:
    if samples < 0:
        raise ValueError("samples must be non-negative")
    return samples // TARGET_SAMPLES


def _load_recordings(split_dir: Path) -> pd.DataFrame:
    frames = []
    for role in ROLES:
        path = split_dir / f"{role}_recordings_manifest.csv"
        frame = pd.read_csv(path, low_memory=False)
        if set(frame["split"].astype(str)) != {role}:
            raise ValueError(f"Unexpected roles in {path}")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined["source_path"].duplicated().any():
        raise ValueError("DroneNoise recording manifests contain duplicate source paths")
    return combined


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return str(path)


def prepare(split_dir: Path, output_dir: Path) -> dict[str, Any]:
    split_audit_path = split_dir / "split_audit.json"
    split_audit = json.loads(split_audit_path.read_text(encoding="utf-8"))
    if not split_audit.get("passed"):
        raise ValueError("DroneNoise event split audit must pass before caching")
    recordings = _load_recordings(split_dir)

    final_paths = {role: output_dir / f"{role}_audio.npy" for role in ROLES}
    temporary_paths = {
        role: output_dir / f".{role}_audio.npy.building" for role in ROLES
    }
    manifest_path = output_dir / "halfsecond_manifest.csv"
    audit_path = output_dir / "cache_audit.json"
    existing = [manifest_path, audit_path, *final_paths.values()]
    if any(path.exists() for path in existing):
        raise FileExistsError("Refusing to overwrite an existing DroneNoise cache")
    output_dir.mkdir(parents=True, exist_ok=True)

    decoded: dict[str, tuple[np.ndarray, pd.Series]] = {}
    expected = {role: 0 for role in ROLES}
    raw_hash_mismatches = 0
    nonfinite_recordings = 0
    for _, row in recordings.iterrows():
        path = Path(str(row["source_path"]))
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != str(row["audio_sha256"]):
            raw_hash_mismatches += 1
            raise ValueError(f"Raw DroneNoise WAV hash changed: {path}")
        audio, source_rate = decode_wav_bytes(payload)
        audio = ensure_sample_rate(audio, source_rate, TARGET_SAMPLE_RATE)
        if not np.isfinite(audio).all():
            nonfinite_recordings += 1
            raise ValueError(f"Non-finite DroneNoise audio: {path}")
        role = str(row["split"])
        expected[role] += complete_window_count(int(audio.size))
        decoded[str(path)] = (audio, row)

    if any(count <= 0 for count in expected.values()):
        raise ValueError(f"Every role must provide complete 0.5-second windows: {expected}")
    memmaps = {
        role: np.lib.format.open_memmap(
            temporary_paths[role],
            mode="w+",
            dtype=np.float32,
            shape=(expected[role], TARGET_SAMPLES),
        )
        for role in ROLES
    }
    positions = {role: 0 for role in ROLES}
    rows: list[dict[str, Any]] = []
    discarded_tail_samples = 0
    zero_rms_windows = 0
    within_role_duplicates = {role: 0 for role in ROLES}
    seen_by_role: dict[str, set[str]] = {role: set() for role in ROLES}
    try:
        for path in sorted(decoded):
            audio, row = decoded[path]
            role = str(row["split"])
            count = complete_window_count(int(audio.size))
            discarded_tail_samples += int(audio.size - count * TARGET_SAMPLES)
            for index in range(count):
                start = index * TARGET_SAMPLES
                waveform = np.asarray(audio[start : start + TARGET_SAMPLES], dtype=np.float32)
                cache_index = positions[role]
                memmaps[role][cache_index] = waveform
                identity = segment_sha256(waveform)
                if identity in seen_by_role[role]:
                    within_role_duplicates[role] += 1
                seen_by_role[role].add(identity)
                rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
                zero_rms_windows += int(rms <= 1e-12)
                rows.append(
                    {
                        "dataset_role": role,
                        "split": role,
                        "dataset_origin": "dronenoise_database_v3",
                        "label": 1,
                        "cache_path": _display_path(final_paths[role]),
                        "cache_index": cache_index,
                        "cache_start_sample": 0,
                        "cache_end_sample": TARGET_SAMPLES,
                        "source_path": path,
                        "recording_group": str(row["recording_group"]),
                        "source_group": str(row["source_group"]),
                        "domain_bucket": f"positive:dronenoise:{row['uav_subtype']}",
                        "background_mix_eligible": False,
                        "audio_sha256": str(row["audio_sha256"]),
                        "segment_sha256": identity,
                        "half_index": index,
                        "uav_subtype": str(row["uav_subtype"]),
                        "uav_novelty": "seen_model_type",
                        "device": f"DroneNoise_M{int(row['microphone'])}",
                        "scene": (
                            f"{row['operation_code']}{int(row['speed_code']):02d}_"
                            f"{row['payload_code']}_{row['starting_code']}_"
                            f"{row['direction_code']}"
                        ),
                        "license": str(row["license"]),
                    }
                )
                positions[role] += 1
        if positions != expected:
            raise ValueError(f"Cache counts changed: {positions} != {expected}")
        for cache in memmaps.values():
            cache.flush()
        memmaps.clear()
        for role in ROLES:
            temporary_paths[role].replace(final_paths[role])
    except BaseException:
        memmaps.clear()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise

    manifest = pd.DataFrame(rows, columns=MODEL_COLUMNS)
    frames = {
        role: manifest[manifest["split"].eq(role)].reset_index(drop=True)
        for role in ROLES
    }
    overlap_checks = {}
    for column in ("audio_sha256", "segment_sha256", "recording_group", "source_group"):
        overlap_checks[column] = {}
        for index, left in enumerate(ROLES):
            for right in ROLES[index + 1 :]:
                overlap_checks[column][f"{left}__{right}"] = len(
                    set(frames[left][column].astype(str))
                    & set(frames[right][column].astype(str))
                )
    if any(value for checks in overlap_checks.values() for value in checks.values()):
        raise ValueError(f"DroneNoise half-second role leakage: {overlap_checks}")
    manifest.to_csv(manifest_path, index=False)

    counts = {
        role: {
            "segments": len(frames[role]),
            "recordings": int(frames[role]["recording_group"].nunique()),
            "events": int(frames[role]["source_group"].nunique()),
            "segments_by_uav_subtype": {
                str(key): int(value)
                for key, value in frames[role]["uav_subtype"].value_counts().sort_index().items()
            },
        }
        for role in ROLES
    }
    audit = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_rate": TARGET_SAMPLE_RATE,
        "clip_seconds": 0.5,
        "target_samples": TARGET_SAMPLES,
        "window_policy": "contiguous_nonoverlapping_from_recording_start_drop_incomplete_tail",
        "amplitude_policy": "store_resampled_float32_without_peak_normalization; dataset_normalizes_on_read",
        "counts": counts,
        "raw_hash_mismatches": raw_hash_mismatches,
        "nonfinite_recordings": nonfinite_recordings,
        "zero_rms_windows": zero_rms_windows,
        "discarded_tail_samples_16k": discarded_tail_samples,
        "within_role_duplicate_segment_hashes": within_role_duplicates,
        "overlap_checks": overlap_checks,
        "inputs": {
            "event_split_audit": {
                "path": str(split_audit_path),
                "sha256": file_sha256(split_audit_path),
            }
        },
        "outputs": {
            "manifest": {
                "path": str(manifest_path),
                "sha256": file_sha256(manifest_path),
                "rows": len(manifest),
            },
            **{
                role: {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "rows": expected[role],
                }
                for role, path in final_paths.items()
            },
        },
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DroneNoise native 0.5-second cache")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("artifacts/g7_r6_dronenoise_event_split"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/g7_r6_dronenoise_halfsec_cache"),
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.split_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
