from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
import wave
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from .audio import decode_wav_bytes, ensure_sample_rate


ALGORITHM = "g13_aerosonic_real_background_v1"
EXPECTED_AUDIO_MD5 = "77605a8ef12a38a289b63ae3457d326e"
EXPECTED_META_MD5 = "aabd99b1b2efe0895e212232bca07e46"
TARGET_RATE = 16000
CLIP_SAMPLES = 16000


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nonoverlap_clip_count(samples: int, clip_samples: int = CLIP_SAMPLES) -> int:
    return max(0, int(samples) // int(clip_samples))


def _safe_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    by_basename: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        pure = PurePosixPath(info.filename)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"Unsafe ZIP member: {info.filename}")
        if info.is_dir() or not info.filename.lower().endswith(".wav"):
            continue
        name = pure.name
        if name in by_basename:
            raise ValueError(f"Duplicate WAV basename in AeroSonicDB ZIP: {name}")
        by_basename[name] = info
    return by_basename


def _write_pcm16(path: Path, audio: np.ndarray) -> None:
    pcm = np.clip(np.rint(audio * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(TARGET_RATE)
        output.writeframes(pcm.tobytes())


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AeroSonicDB as an independent G13 background source")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/external_confirmation_v2/_raw/AeroSonicDB_v1.1.2"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/external_confirmation_v2/_prepared/AeroSonicDB_v1.1.2"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("artifacts/g13_external_confirmation/source_preparations/aerosonic_background_audit.json"),
    )
    parser.add_argument(
        "--registry-fragment",
        type=Path,
        default=Path("artifacts/g13_external_confirmation/source_preparations/aerosonic_source_registry.csv"),
    )
    args = parser.parse_args()

    audio_zip = args.raw_dir / "audio.zip"
    metadata_path = args.raw_dir / "sample_meta.csv"
    final_background = args.output_root / "Background"
    staging = args.output_root / ".Background.tmp"
    if final_background.exists() or args.audit.exists() or args.registry_fragment.exists():
        raise FileExistsError("AeroSonicDB preparation output already exists; refusing overwrite")
    if staging.exists():
        raise FileExistsError(f"AeroSonicDB staging directory already exists: {staging}")
    if md5(audio_zip) != EXPECTED_AUDIO_MD5 or md5(metadata_path) != EXPECTED_META_MD5:
        raise RuntimeError("AeroSonicDB source MD5 verification failed")

    metadata = pd.read_csv(metadata_path, low_memory=False)
    required = {"filename", "class", "session", "location", "mic", "file_length"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"AeroSonicDB metadata missing columns: {missing}")
    if metadata["filename"].duplicated().any():
        raise ValueError("AeroSonicDB metadata filenames must be unique")
    if set(pd.to_numeric(metadata["class"], errors="raise").astype(int)) != {0, 1}:
        raise ValueError("Unexpected AeroSonicDB classes")

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    estimated_bytes = int(pd.to_numeric(metadata["file_length"]).sum()) * (CLIP_SAMPLES * 2 + 44)
    free_bytes = shutil.disk_usage(args.output_root.parent).free
    if free_bytes < estimated_bytes + 2 * 1024**3:
        raise RuntimeError("Insufficient disk for AeroSonicDB derived background")

    staging.mkdir(parents=True)
    output_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    input_sample_rates: Counter[int] = Counter()
    input_channels: Counter[int] = Counter()
    duration_mismatches = 0
    index_rows: list[dict[str, Any]] = []

    with zipfile.ZipFile(audio_zip) as archive:
        members = _safe_members(archive)
        expected_names = set(metadata["filename"].astype(str))
        if set(members) != expected_names:
            missing_audio = sorted(expected_names - set(members))[:20]
            extra_audio = sorted(set(members) - expected_names)[:20]
            raise ValueError(f"AeroSonicDB ZIP/metadata mismatch; missing={missing_audio}, extra={extra_audio}")

        for _, row in metadata.sort_values(["session", "filename"]).iterrows():
            filename = str(row["filename"])
            wav_bytes = archive.read(members[filename])
            audio, sample_rate = decode_wav_bytes(wav_bytes)
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav_header:
                input_channels[wav_header.getnchannels()] += 1
            input_sample_rates[int(sample_rate)] += 1
            audio = ensure_sample_rate(audio, int(sample_rate), TARGET_RATE)
            clips = nonoverlap_clip_count(audio.size)
            if clips != int(float(row["file_length"])):
                duration_mismatches += 1
            session = int(row["session"])
            source_group = f"aerosonic_session_{session:02d}"
            condition = "urban_silence" if int(row["class"]) == 0 else "aircraft"
            output_dir = staging / source_group / condition
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(filename).stem
            for clip_index in range(clips):
                start = clip_index * CLIP_SAMPLES
                output_path = output_dir / f"{stem}_seg{clip_index:03d}.wav"
                _write_pcm16(output_path, audio[start : start + CLIP_SAMPLES])
                output_counts[condition] += 1
                source_counts[source_group] += 1
                index_rows.append(
                    {
                        "path": output_path.relative_to(staging).as_posix(),
                        "source_group": source_group,
                        "condition": condition,
                        "raw_filename": filename,
                        "clip_index": clip_index,
                        "location": int(row["location"]),
                        "microphone": int(row["mic"]),
                    }
                )
            if len(index_rows) and len(index_rows) % 1000 < clips:
                print(f"AeroSonicDB prepared {len(index_rows)} clips", flush=True)

    if min(source_counts.values()) < 50 or len(source_counts) < 10:
        raise RuntimeError(f"AeroSonicDB source requirements failed: {dict(source_counts)}")
    os.replace(staging, final_background)

    args.audit.parent.mkdir(parents=True, exist_ok=True)
    index_path = args.audit.parent / "aerosonic_derived_index.csv"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=index_path.parent, delete=False) as handle:
        temporary_index = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    os.replace(temporary_index, index_path)

    registry_rows = []
    for source_group in sorted(source_counts):
        session = int(source_group.rsplit("_", 1)[1])
        subset = metadata[pd.to_numeric(metadata["session"]).astype(int) == session]
        locations = sorted(int(value) for value in subset["location"].unique())
        microphones = sorted(int(value) for value in subset["mic"].unique())
        registry_rows.append(
            {
                "source_group": source_group,
                "label": 0,
                "acquisition_id": f"aerosonic_ypad_session_{session:02d}",
                "provenance": f"AeroSonicDB v1.1.2 session={session}; locations={locations}; microphones={microphones}",
                "independent_from_existing": "true",
                "license": "CC BY-NC 4.0 non-commercial research",
            }
        )
    args.registry_fragment.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=args.registry_fragment.parent, delete=False) as handle:
        temporary_registry = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(registry_rows[0]))
        writer.writeheader()
        writer.writerows(registry_rows)
    os.replace(temporary_registry, args.registry_fragment)

    report = {
        "algorithm": ALGORITHM,
        "passed": True,
        "source_archive": {"path": audio_zip.as_posix(), "md5": EXPECTED_AUDIO_MD5},
        "source_metadata": {"path": metadata_path.as_posix(), "md5": EXPECTED_META_MD5},
        "recordings": int(len(metadata)),
        "source_groups": len(source_counts),
        "minimum_clips_per_source": min(source_counts.values()),
        "clips": int(sum(output_counts.values())),
        "condition_counts": dict(sorted(output_counts.items())),
        "input_sample_rates": {str(k): v for k, v in sorted(input_sample_rates.items())},
        "input_channels": {str(k): v for k, v in sorted(input_channels.items())},
        "metadata_duration_mismatches": duration_mismatches,
        "output_format": {"sample_rate": TARGET_RATE, "channels": 1, "sample_width_bytes": 2, "seconds": 1.0},
        "output": final_background.as_posix(),
        "derived_index": index_path.as_posix(),
        "registry_fragment": args.registry_fragment.as_posix(),
        "locked_datasets_read": [],
        "model_predictions_read": False,
    }
    _atomic_json(args.audit, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
