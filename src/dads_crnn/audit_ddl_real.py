from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ALGORITHM = "g13_ddl_real_source_audit_v1"
FILENAME = re.compile(
    r"^(?P<timestamp>\d{14})(?P<class_name>MINI|PRO4|XXXX)"
    r"(?P<bearing>\d{3})(?P<range>\d{3})(?P<altitude>\d{3})"
    r"(?P<temperature>\d{4})(?P<kind>[RS])(?P<date>\d{6})-"
    r"(?P<session>T\d{3})-(?P<sequence>\d{6})\.wav$"
)
STANDARD_FORMAT = (8, 96000, 4, 9600)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_clip_count(sequences: list[int], segments_per_clip: int = 10) -> tuple[int, int]:
    if not sequences:
        return 0, 0
    ordered = sorted(set(sequences))
    runs: list[int] = []
    run_length = 1
    for previous, current in zip(ordered, ordered[1:]):
        if current == previous + 1:
            run_length += 1
        else:
            runs.append(run_length)
            run_length = 1
    runs.append(run_length)
    return sum(length // segments_per_clip for length in runs), len(runs)


def audit(root: Path, minimum_clips_per_source: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    class_counts: Counter[str] = Counter()
    standard_counts: Counter[str] = Counter()
    empty_counts: Counter[str] = Counter()
    nonstandard_counts: Counter[str] = Counter()
    formats: Counter[tuple[int, int, int, int]] = Counter()
    session_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    sequences: dict[tuple[str, str], list[int]] = defaultdict(list)
    bad_names: list[str] = []
    bad_wave: list[dict[str, str]] = []

    wav_paths = sorted(root.rglob("*.wav"))
    for path in wav_paths:
        match = FILENAME.match(path.name)
        if match is None:
            bad_names.append(path.relative_to(root).as_posix())
            continue
        values = match.groupdict()
        class_name = values["class_name"]
        source_group = path.parent.name
        class_counts[class_name] += 1
        session_counts[(class_name, source_group)]["parsed"] += 1
        try:
            with wave.open(str(path), "rb") as wav:
                observed = (
                    wav.getnchannels(),
                    wav.getframerate(),
                    wav.getsampwidth(),
                    wav.getnframes(),
                )
        except Exception as error:
            bad_wave.append(
                {"path": path.relative_to(root).as_posix(), "error": repr(error)}
            )
            session_counts[(class_name, source_group)]["bad_wave"] += 1
            continue
        formats[observed] += 1
        if observed[3] == 0:
            empty_counts[class_name] += 1
            session_counts[(class_name, source_group)]["empty"] += 1
        elif observed == STANDARD_FORMAT:
            standard_counts[class_name] += 1
            session_counts[(class_name, source_group)]["standard"] += 1
            sequences[(class_name, source_group)].append(int(values["sequence"]))
        else:
            nonstandard_counts[class_name] += 1
            session_counts[(class_name, source_group)]["nonstandard"] += 1

    source_rows: list[dict[str, Any]] = []
    for class_name, source_group in sorted(session_counts):
        counts = session_counts[(class_name, source_group)]
        clips, runs = candidate_clip_count(sequences[(class_name, source_group)])
        source_rows.append(
            {
                "class_name": class_name,
                "label": 0 if class_name == "XXXX" else 1,
                "source_group": source_group,
                "parsed_wavs": int(counts["parsed"]),
                "standard_wavs": int(counts["standard"]),
                "empty_wavs": int(counts["empty"]),
                "nonstandard_wavs": int(counts["nonstandard"]),
                "bad_wavs": int(counts["bad_wave"]),
                "contiguous_runs": int(runs),
                "candidate_nonoverlap_1s_clips": int(clips),
                "eligible_source": int(clips) >= int(minimum_clips_per_source),
            }
        )

    eligible_positive = sum(
        row["eligible_source"] and row["label"] == 1 for row in source_rows
    )
    eligible_negative = sum(
        row["eligible_source"] and row["label"] == 0 for row in source_rows
    )
    report = {
        "algorithm": ALGORITHM,
        "root": root.as_posix(),
        "wav_files": len(wav_paths),
        "parsed_filenames": int(sum(class_counts.values())),
        "unparsed_filenames": len(bad_names),
        "class_counts": dict(sorted(class_counts.items())),
        "standard_format": {
            "channels": STANDARD_FORMAT[0],
            "sample_rate": STANDARD_FORMAT[1],
            "sample_width_bytes": STANDARD_FORMAT[2],
            "frames": STANDARD_FORMAT[3],
            "duration_seconds": STANDARD_FORMAT[3] / STANDARD_FORMAT[1],
        },
        "standard_counts": dict(sorted(standard_counts.items())),
        "empty_counts": dict(sorted(empty_counts.items())),
        "nonstandard_counts": dict(sorted(nonstandard_counts.items())),
        "bad_wave_count": len(bad_wave),
        "observed_formats": [
            {
                "channels": key[0],
                "sample_rate": key[1],
                "sample_width_bytes": key[2],
                "frames": key[3],
                "count": value,
            }
            for key, value in formats.most_common()
        ],
        "sessions": len(source_rows),
        "eligible_positive_source_groups": int(eligible_positive),
        "eligible_negative_source_groups": int(eligible_negative),
        "candidate_positive_1s_clips": int(
            sum(row["candidate_nonoverlap_1s_clips"] for row in source_rows if row["label"] == 1)
        ),
        "candidate_negative_1s_clips": int(
            sum(row["candidate_nonoverlap_1s_clips"] for row in source_rows if row["label"] == 0)
        ),
        "negative_samples_present": bool(class_counts.get("XXXX", 0)),
        "g13_binary_ready": eligible_positive >= 10 and eligible_negative >= 10,
        "excluded_examples": {
            "unparsed_filenames": bad_names[:50],
            "bad_waves": bad_wave[:50],
        },
        "locked_datasets_read": [],
        "model_predictions_read": False,
    }
    return report, source_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the extracted DDL real-audio dataset")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/external_confirmation_v2/_raw/DDL/extracted/real_data"),
    )
    parser.add_argument(
        "--extraction-audit",
        type=Path,
        default=Path("data/external_confirmation_v2/_raw/DDL/extraction_audit.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/g13_external_confirmation/source_audits/ddl_real_audit.json"),
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=Path("artifacts/g13_external_confirmation/source_audits/ddl_real_sources.csv"),
    )
    parser.add_argument("--minimum-clips-per-source", type=int, default=50)
    args = parser.parse_args()

    if args.output.exists() or args.source_csv.exists():
        raise FileExistsError("DDL source audit outputs already exist; refusing overwrite")
    report, source_rows = audit(args.root, args.minimum_clips_per_source)
    extraction = json.loads(args.extraction_audit.read_text(encoding="utf-8"))
    report["inputs"] = {
        "extraction_audit": {
            "path": args.extraction_audit.as_posix(),
            "sha256": sha256(args.extraction_audit),
        },
        "archive_md5": extraction["archive_md5"],
        "archive_size_bytes": extraction["archive_size_bytes"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=args.output.parent, delete=False
    ) as handle:
        temporary_json = Path(handle.name)
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary_json, args.output)

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=args.source_csv.parent, delete=False
    ) as handle:
        temporary_csv = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    os.replace(temporary_csv, args.source_csv)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
