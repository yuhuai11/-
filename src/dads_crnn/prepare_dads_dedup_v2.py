from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from .data_firewall import audit_csv_rows, file_sha256


PROTOCOL = "dads_raw_audio_sha256_dedup_v2"
LOCATOR_COLUMNS = ["parquet_file", "row_group", "row_in_group"]
SOURCE_COLUMNS = [
    "split",
    "label",
    "parquet_file",
    "row_group",
    "row_in_group",
    "source_path",
    "original_samples",
]
SPLIT_ORDER = ("train", "val", "test")
REQUIRED_MANIFEST_COLUMNS = {
    "split",
    "label",
    "parquet_file",
    "row_group",
    "row_in_group",
    "source_path",
    "segment_index",
    "start_sample",
    "end_sample",
    "original_samples",
    "segment_kind",
    "cache_path",
}


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


def _source_records(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"DADS manifest is missing columns: {missing}")
    if frame.empty:
        raise ValueError("DADS manifest is empty")
    if set(frame["split"].astype(str)) != set(SPLIT_ORDER):
        raise ValueError("DADS manifest must contain exactly train, val and test")
    labels = pd.to_numeric(frame["label"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("DADS labels must be binary")

    uniqueness_columns = ["split", "label", "source_path", "original_samples"]
    variation = (
        frame.groupby(LOCATOR_COLUMNS, sort=False, dropna=False)[uniqueness_columns]
        .nunique(dropna=False)
    )
    if int(variation.to_numpy().max()) != 1:
        raise ValueError("A DADS source locator has inconsistent source metadata")
    duplicate_segments = frame.duplicated(LOCATOR_COLUMNS + ["segment_index"])
    if duplicate_segments.any():
        raise ValueError("DADS manifest contains duplicate source/segment rows")

    sources = frame[SOURCE_COLUMNS].drop_duplicates(LOCATOR_COLUMNS).copy()
    sources["row_group"] = pd.to_numeric(sources["row_group"], errors="raise").astype(int)
    sources["row_in_group"] = pd.to_numeric(
        sources["row_in_group"], errors="raise"
    ).astype(int)
    sources["label"] = pd.to_numeric(sources["label"], errors="raise").astype(int)
    sources["original_split"] = sources["split"].astype(str)
    return sources.sort_values(LOCATOR_COLUMNS, kind="stable").reset_index(drop=True)


def _hash_sources(sources: pd.DataFrame, root: Path) -> pd.DataFrame:
    requested: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in sources.itertuples(index=False):
        requested[(str(row.parquet_file), int(row.row_group))].add(int(row.row_in_group))

    observed: dict[tuple[str, int, int], tuple[str, int]] = {}
    for (parquet_value, row_group), indices in tqdm(
        sorted(requested.items()), desc="Hashing DADS raw audio", unit="row-group"
    ):
        parquet_path = Path(parquet_value)
        if not parquet_path.is_absolute():
            parquet_path = root / parquet_path
        parquet_path = parquet_path.resolve(strict=True)
        parquet = pq.ParquetFile(parquet_path)
        if row_group < 0 or row_group >= parquet.num_row_groups:
            raise ValueError(f"Invalid row group {row_group} in {parquet_path}")
        audios = parquet.read_row_group(row_group, columns=["audio"]).column("audio").to_pylist()
        for row_in_group in sorted(indices):
            if row_in_group < 0 or row_in_group >= len(audios):
                raise ValueError(
                    f"Invalid row {row_in_group} in {parquet_path} row group {row_group}"
                )
            wav_bytes = audios[row_in_group].get("bytes")
            if not isinstance(wav_bytes, bytes) or not wav_bytes:
                raise ValueError(
                    f"Missing audio bytes in {parquet_path}:{row_group}:{row_in_group}"
                )
            observed[(parquet_value, row_group, row_in_group)] = (
                hashlib.sha256(wav_bytes).hexdigest(),
                len(wav_bytes),
            )

    output = sources.copy()
    hashes = []
    byte_counts = []
    for row in output.itertuples(index=False):
        key = (str(row.parquet_file), int(row.row_group), int(row.row_in_group))
        if key not in observed:
            raise RuntimeError(f"Source was not hashed: {key}")
        sha256, byte_count = observed[key]
        hashes.append(sha256)
        byte_counts.append(byte_count)
    output["raw_audio_sha256"] = hashes
    output["raw_audio_bytes"] = byte_counts
    output["recording_group"] = output["raw_audio_sha256"].map(
        lambda value: f"dads_raw_sha256:{value}"
    )
    return output


def _target_counts(sources: pd.DataFrame) -> dict[int, dict[str, int]]:
    targets: dict[int, dict[str, int]] = {}
    for label, rows in sources.groupby("label", sort=True):
        total = int(rows["raw_audio_sha256"].nunique())
        original = rows.drop_duplicates(LOCATOR_COLUMNS)["split"].value_counts()
        ratios = {
            split: float(original.get(split, 0)) / max(1, int(original.sum()))
            for split in SPLIT_ORDER
        }
        counts = {split: int(round(total * ratios[split])) for split in SPLIT_ORDER[:-1]}
        counts[SPLIT_ORDER[-1]] = total - sum(counts.values())
        targets[int(label)] = counts
    return targets


def select_canonical_sources(
    sources: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep one raw-audio record per SHA while minimally disturbing old splits."""
    if "raw_audio_sha256" not in sources.columns:
        raise ValueError("Source registry lacks raw_audio_sha256")
    label_counts = sources.groupby("raw_audio_sha256")["label"].nunique()
    conflicts = label_counts[label_counts != 1]
    if not conflicts.empty:
        raise ValueError(
            f"Identical DADS audio has conflicting labels: {len(conflicts)} hash groups"
        )

    targets = _target_counts(sources)
    selected_counts: dict[int, Counter[str]] = {
        int(label): Counter() for label in sorted(sources["label"].unique())
    }
    selected_indices: set[int] = set()
    duplicate_rows = []

    groups = list(sources.groupby("raw_audio_sha256", sort=True))
    singletons = [(sha256, rows) for sha256, rows in groups if len(rows) == 1]
    duplicates = [(sha256, rows) for sha256, rows in groups if len(rows) > 1]

    for _, rows in singletons:
        index = int(rows.index[0])
        selected_indices.add(index)
        row = rows.loc[index]
        selected_counts[int(row["label"])][str(row["split"])] += 1

    split_rank = {name: index for index, name in enumerate(SPLIT_ORDER)}
    for sha256, rows in duplicates:
        label = int(rows["label"].iloc[0])
        available_splits = sorted(
            set(rows["split"].astype(str)),
            key=lambda split: split_rank.get(split, len(split_rank)),
        )
        chosen_split = max(
            available_splits,
            key=lambda split: (
                targets[label].get(split, 0) - selected_counts[label][split],
                -split_rank.get(split, len(split_rank)),
            ),
        )
        candidates = rows.loc[rows["split"].astype(str) == chosen_split].sort_values(
            LOCATOR_COLUMNS, kind="stable"
        )
        chosen_index = int(candidates.index[0])
        selected_indices.add(chosen_index)
        selected_counts[label][chosen_split] += 1

        cross_split = int(rows["split"].nunique()) > 1
        for index, row in rows.sort_values(LOCATOR_COLUMNS, kind="stable").iterrows():
            duplicate_rows.append(
                {
                    "raw_audio_sha256": sha256,
                    "duplicate_group_size": int(len(rows)),
                    "cross_split_before": bool(cross_split),
                    "kept": int(index) == chosen_index,
                    "original_split": str(row["split"]),
                    "label": int(row["label"]),
                    "parquet_file": str(row["parquet_file"]),
                    "row_group": int(row["row_group"]),
                    "row_in_group": int(row["row_in_group"]),
                    "source_path": str(row["source_path"]),
                }
            )

    selected = sources.loc[sorted(selected_indices)].copy()
    selected["split"] = selected["original_split"]
    selected["kept"] = True
    duplicates_frame = pd.DataFrame(duplicate_rows)
    if not selected["raw_audio_sha256"].is_unique:
        raise AssertionError("Canonical source selection did not remove duplicate hashes")
    return (
        selected.sort_values(["split", "label", *LOCATOR_COLUMNS], kind="stable").reset_index(
            drop=True
        ),
        duplicates_frame,
    )


def _counts(frame: pd.DataFrame, unit: str) -> dict[str, Any]:
    return {
        "unit": unit,
        "total": int(len(frame)),
        "by_label": {
            str(int(key)): int(value)
            for key, value in frame["label"].value_counts().sort_index().items()
        },
        "by_split": {
            str(key): int(value)
            for key, value in frame["split"].value_counts().sort_index().items()
        },
        "by_label_split": {
            str(int(label)): {
                str(split): int(value)
                for split, value in rows["split"].value_counts().sort_index().items()
            }
            for label, rows in frame.groupby("label", sort=True)
        },
    }


def _pairwise_overlap(frame: pd.DataFrame, column: str) -> dict[str, int]:
    values = {
        split: set(rows[column].astype(str))
        for split, rows in frame.groupby("split", sort=True)
    }
    return {
        f"{left}_{right}": len(values.get(left, set()) & values.get(right, set()))
        for index, left in enumerate(SPLIT_ORDER)
        for right in SPLIT_ORDER[index + 1 :]
    }


def prepare(
    root: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    verify_cache: bool,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    output_dir = output_dir.resolve(strict=False)
    audit_csv_rows(manifest_path, required_columns=REQUIRED_MANIFEST_COLUMNS)
    frame = pd.read_csv(manifest_path, low_memory=False)
    sources = _source_records(frame)
    hashed_sources = _hash_sources(sources, root)
    canonical, duplicate_rows = select_canonical_sources(hashed_sources)

    canonical_keys = pd.MultiIndex.from_frame(canonical[LOCATOR_COLUMNS])
    frame_keys = pd.MultiIndex.from_frame(frame[LOCATOR_COLUMNS])
    keep_mask = frame_keys.isin(canonical_keys)
    dedup = frame.loc[keep_mask].copy()
    source_metadata = canonical[
        LOCATOR_COLUMNS + ["raw_audio_sha256", "recording_group", "original_split"]
    ]
    dedup = dedup.merge(
        source_metadata,
        on=LOCATOR_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    if dedup[["raw_audio_sha256", "recording_group"]].isna().any().any():
        raise AssertionError("A deduplicated segment lacks source metadata")
    dedup["split"] = dedup["original_split"]
    dedup = dedup.sort_values(
        ["split", "label", *LOCATOR_COLUMNS, "segment_index"], kind="stable"
    ).reset_index(drop=True)

    cache_missing = None
    if verify_cache:
        cache_missing = int(
            sum(not (root / str(value)).is_file() for value in dedup["cache_path"])
        )
        if cache_missing:
            raise FileNotFoundError(f"{cache_missing} DADS cache files are missing")

    raw_overlap = _pairwise_overlap(canonical, "raw_audio_sha256")
    group_overlap = _pairwise_overlap(canonical, "recording_group")
    source_path_overlap = _pairwise_overlap(canonical, "source_path")
    if any(raw_overlap.values()) or any(group_overlap.values()) or any(
        source_path_overlap.values()
    ):
        raise ValueError(
            "Deduplicated DADS still has cross-split overlap: "
            f"raw={raw_overlap}, group={group_overlap}, source_path={source_path_overlap}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    source_registry_path = output_dir / "source_registry.csv"
    duplicate_path = output_dir / "duplicate_groups.csv"
    combined_path = output_dir / "dads_dedup_v2_all.csv"
    canonical.to_csv(source_registry_path, index=False)
    duplicate_rows.to_csv(duplicate_path, index=False)
    dedup.to_csv(combined_path, index=False)
    split_paths = {}
    for split in SPLIT_ORDER:
        path = output_dir / f"{split}_manifest.csv"
        dedup.loc[dedup["split"] == split].to_csv(path, index=False)
        split_paths[split] = path

    duplicate_groups = int(hashed_sources["raw_audio_sha256"].value_counts().gt(1).sum())
    cross_split_groups = int(
        hashed_sources.groupby("raw_audio_sha256")["split"].nunique().gt(1).sum()
    )
    removed_locators = set(map(tuple, sources[LOCATOR_COLUMNS].to_numpy())) - set(
        map(tuple, canonical[LOCATOR_COLUMNS].to_numpy())
    )
    removed_segments = int(
        frame_keys.isin(pd.MultiIndex.from_tuples(sorted(removed_locators))).sum()
        if removed_locators
        else 0
    )
    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "manifest": manifest_path.relative_to(root).as_posix(),
            "manifest_sha256": file_sha256(manifest_path),
        },
        "original": {
            "sources": _counts(sources, "original_recordings"),
            "segments": _counts(frame, "one_second_segments"),
            "unique_raw_audio_sha256": int(hashed_sources["raw_audio_sha256"].nunique()),
        },
        "dedup_v2": {
            "sources": _counts(canonical, "unique_raw_recordings"),
            "segments": _counts(dedup, "one_second_segments"),
            "sources_removed": int(len(sources) - len(canonical)),
            "segments_removed": removed_segments,
            "duplicate_hash_groups_before": duplicate_groups,
            "cross_split_duplicate_hash_groups_before": cross_split_groups,
            "cross_split_raw_hash_overlap_after": raw_overlap,
            "cross_split_recording_group_overlap_after": group_overlap,
            "cross_split_source_path_overlap_after": source_path_overlap,
            "cache_verified": bool(verify_cache),
            "cache_missing": cache_missing,
        },
        "outputs": {
            "source_registry": {
                "path": source_registry_path.relative_to(root).as_posix(),
                "sha256": file_sha256(source_registry_path),
                "rows": int(len(canonical)),
            },
            "duplicate_groups": {
                "path": duplicate_path.relative_to(root).as_posix(),
                "sha256": file_sha256(duplicate_path),
                "rows": int(len(duplicate_rows)),
            },
            "combined_manifest": {
                "path": combined_path.relative_to(root).as_posix(),
                "sha256": file_sha256(combined_path),
                "rows": int(len(dedup)),
            },
            "split_manifests": {
                split: {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                    "rows": int((dedup["split"] == split).sum()),
                }
                for split, path in split_paths.items()
            },
        },
        "historical_manifest_modified": False,
        "audio_cache_modified": False,
        "training_started": False,
        "locked_dataset_audio_read": False,
    }
    _atomic_write_text(
        output_dir / "audit.json",
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a non-destructive raw-audio-SHA deduplicated DADS v2 manifest."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts_full/manifests/dads_all_seed42.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/dads_dedup_v2")
    )
    parser.add_argument("--skip-cache-verification", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    report = prepare(
        root,
        manifest,
        output,
        verify_cache=not args.skip_cache_verification,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "protocol": report["protocol"],
                "sources_removed": report["dedup_v2"]["sources_removed"],
                "segments_removed": report["dedup_v2"]["segments_removed"],
                "cross_split_duplicate_hash_groups_before": report["dedup_v2"][
                    "cross_split_duplicate_hash_groups_before"
                ],
                "cross_split_raw_hash_overlap_after": report["dedup_v2"][
                    "cross_split_raw_hash_overlap_after"
                ],
                "combined_manifest": report["outputs"]["combined_manifest"],
                "training_started": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
