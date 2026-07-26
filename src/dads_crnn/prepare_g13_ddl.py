from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .audio import decode_wav_bytes, ensure_sample_rate
from .audit_ddl_real import FILENAME, STANDARD_FORMAT, candidate_clip_count, sha256


ALGORITHM = "g13_ddl_real_uav_v1"
TARGET_RATE = 16000
CLIP_SAMPLES = 16000
SEGMENTS_PER_CLIP = 10


def contiguous_chunks(items: list[tuple[int, Path]], size: int = SEGMENTS_PER_CLIP) -> list[list[tuple[int, Path]]]:
    ordered = sorted(items)
    chunks: list[list[tuple[int, Path]]] = []
    run: list[tuple[int, Path]] = []
    previous: int | None = None
    for item in ordered:
        sequence = item[0]
        if previous is None or sequence == previous + 1:
            run.append(item)
        else:
            chunks.extend(run[index : index + size] for index in range(0, len(run) - size + 1, size))
            run = [item]
        previous = sequence
    chunks.extend(run[index : index + size] for index in range(0, len(run) - size + 1, size))
    return chunks


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
    parser = argparse.ArgumentParser(description="Prepare DDL real UAV recordings for G13")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/external_confirmation_v2/_raw/DDL/extracted/real_data"),
    )
    parser.add_argument(
        "--extraction-audit",
        type=Path,
        default=Path("data/external_confirmation_v2/_raw/DDL/extraction_audit.json"),
    )
    parser.add_argument(
        "--source-audit",
        type=Path,
        default=Path("artifacts/g13_external_confirmation/source_audits/ddl_real_audit.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/external_confirmation_v2/_prepared/DDL_real"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("artifacts/g13_external_confirmation/source_preparations/ddl_uav_audit.json"),
    )
    parser.add_argument(
        "--registry-fragment",
        type=Path,
        default=Path("artifacts/g13_external_confirmation/source_preparations/ddl_source_registry.csv"),
    )
    parser.add_argument("--minimum-clips-per-source", type=int, default=50)
    args = parser.parse_args()

    final_uav = args.output_root / "UAV"
    staging = args.output_root / ".UAV.tmp"
    index_path = args.audit.parent / "ddl_derived_index.csv"
    if any(path.exists() for path in (final_uav, staging, args.audit, args.registry_fragment, index_path)):
        raise FileExistsError("DDL preparation output already exists; refusing overwrite")

    extraction = json.loads(args.extraction_audit.read_text(encoding="utf-8"))
    source_audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if extraction.get("archive_md5") != "4a6d4da4e1c732550c1ccd8d29dd16f8":
        raise RuntimeError("Unexpected DDL extraction audit MD5")
    if source_audit.get("algorithm") != "g13_ddl_real_source_audit_v1":
        raise RuntimeError("Unexpected DDL source audit")

    grouped: dict[tuple[str, str], list[tuple[int, Path]]] = defaultdict(list)
    excluded = Counter()
    for path in sorted(args.raw_root.rglob("*.wav")):
        match = FILENAME.match(path.name)
        if match is None:
            excluded["unparsed_filename"] += 1
            continue
        with wave.open(str(path), "rb") as wav:
            observed = (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getnframes())
        if observed != STANDARD_FORMAT:
            excluded["nonstandard_or_empty"] += 1
            continue
        values = match.groupdict()
        grouped[(values["class_name"], path.parent.name)].append((int(values["sequence"]), path))

    chunks_by_source: dict[tuple[str, str], list[list[tuple[int, Path]]]] = {}
    for key, items in grouped.items():
        chunks = contiguous_chunks(items)
        expected, _ = candidate_clip_count([item[0] for item in items])
        if len(chunks) != expected:
            raise RuntimeError(f"DDL chunk-count mismatch for {key}: {len(chunks)} != {expected}")
        if len(chunks) >= args.minimum_clips_per_source:
            chunks_by_source[key] = chunks
        else:
            excluded["clips_from_ineligible_sources"] += len(chunks)

    if len(chunks_by_source) < 10:
        raise RuntimeError("DDL has fewer than ten eligible source groups")
    args.output_root.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    index_rows: list[dict[str, Any]] = []
    model_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for (class_name, session), chunks in sorted(chunks_by_source.items()):
        source_group = f"ddl_{session.lower()}"
        output_dir = staging / source_group / class_name.lower()
        output_dir.mkdir(parents=True)
        for chunk_index, chunk in enumerate(chunks):
            parts = []
            for _, path in chunk:
                audio, rate = decode_wav_bytes(path.read_bytes())
                if rate != 96000 or audio.size != 9600:
                    raise RuntimeError(f"Unexpected decoded DDL segment: {path}")
                parts.append(audio)
            audio = ensure_sample_rate(np.concatenate(parts), 96000, TARGET_RATE)
            if audio.size != CLIP_SAMPLES:
                raise RuntimeError(f"Unexpected DDL output samples: {audio.size}")
            start_sequence = chunk[0][0]
            end_sequence = chunk[-1][0]
            output_path = output_dir / f"{class_name}_{session}_{start_sequence:06d}_{end_sequence:06d}.wav"
            _write_pcm16(output_path, audio)
            model_counts[class_name] += 1
            source_counts[source_group] += 1
            index_rows.append(
                {
                    "path": output_path.relative_to(staging).as_posix(),
                    "source_group": source_group,
                    "condition": class_name.lower(),
                    "session": session,
                    "model": class_name,
                    "start_sequence": start_sequence,
                    "end_sequence": end_sequence,
                }
            )
        print(f"DDL prepared {source_group}: {len(chunks)} clips", flush=True)

    if min(source_counts.values()) < args.minimum_clips_per_source:
        raise RuntimeError("DDL prepared source count fell below the minimum")
    os.replace(staging, final_uav)

    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=index_path.parent, delete=False) as handle:
        temporary_index = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    os.replace(temporary_index, index_path)

    registry_rows = []
    for source_group in sorted(source_counts):
        session = source_group.removeprefix("ddl_").upper()
        models = sorted(key[0] for key in chunks_by_source if key[1].upper() == session)
        registry_rows.append(
            {
                "source_group": source_group,
                "label": 1,
                "acquisition_id": f"ddl_real_{session.lower()}",
                "provenance": f"DDL real field recording session={session}; models={models}",
                "independent_from_existing": "true",
                "license": "CC BY 4.0",
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
        "source_groups": len(source_counts),
        "minimum_clips_per_source": min(source_counts.values()),
        "clips": len(index_rows),
        "model_counts": dict(sorted(model_counts.items())),
        "excluded": dict(sorted(excluded.items())),
        "output_format": {"sample_rate": TARGET_RATE, "channels": 1, "sample_width_bytes": 2, "seconds": 1.0},
        "output": final_uav.as_posix(),
        "derived_index": index_path.as_posix(),
        "registry_fragment": args.registry_fragment.as_posix(),
        "inputs": {
            "extraction_audit": {"path": args.extraction_audit.as_posix(), "sha256": sha256(args.extraction_audit)},
            "source_audit": {"path": args.source_audit.as_posix(), "sha256": sha256(args.source_audit)},
        },
        "locked_datasets_read": [],
        "model_predictions_read": False,
    }
    _atomic_json(args.audit, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
