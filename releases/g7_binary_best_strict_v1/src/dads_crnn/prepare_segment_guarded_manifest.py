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

import numpy as np
import pandas as pd
from tqdm import tqdm

from .audio import peak_normalize, to_fixed_length
from .data_firewall import file_sha256


PROTOCOL = "dads_final_waveform_grouped_dedup_balanced_v3"
SPLIT_ORDER = ("train", "val", "test")
LOCATOR_COLUMNS = ["parquet_file", "row_group", "row_in_group"]
REQUIRED_COLUMNS = {
    "split",
    "label",
    "parquet_file",
    "row_group",
    "row_in_group",
    "source_path",
    "segment_index",
    "cache_path",
}
GUARD_COLUMNS = {
    "split",
    "label",
    "source_id",
    "segment_float32_sha256",
    "segment_pcm16_sha256",
    "manifest_protocol",
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


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _stable_digest(*parts: object) -> str:
    payload = json.dumps(
        [str(part) for part in parts],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_id(parquet_file: object, row_group: object, row_in_group: object) -> str:
    return "dads_source:" + _stable_digest(
        str(parquet_file), int(row_group), int(row_in_group)
    )


def _load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"DADS manifest is missing columns: {missing}")
    if frame.empty:
        raise ValueError("DADS manifest is empty")

    frame = frame.copy()
    frame["split"] = frame["split"].astype(str)
    observed_splits = set(frame["split"])
    if observed_splits != set(SPLIT_ORDER):
        raise ValueError(
            "DADS manifest must contain exactly train, val and test; "
            f"observed={sorted(observed_splits)}"
        )
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not frame["label"].isin([0, 1]).all():
        raise ValueError("DADS labels must be binary")
    for column in ("row_group", "row_in_group", "segment_index"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
        if (frame[column] < 0).any():
            raise ValueError(f"{column} must be non-negative")

    frame["parquet_file"] = frame["parquet_file"].astype(str)
    frame["source_path"] = frame["source_path"].fillna("").astype(str)
    if frame["cache_path"].isna().any() or (
        frame["cache_path"].astype(str).str.strip() == ""
    ).any():
        raise ValueError("Every segment must have a cache_path")
    frame["cache_path"] = frame["cache_path"].astype(str)

    duplicate_segments = frame.duplicated(LOCATOR_COLUMNS + ["segment_index"])
    if duplicate_segments.any():
        raise ValueError("DADS manifest contains duplicate source/segment rows")

    consistency = (
        frame.groupby(LOCATOR_COLUMNS, sort=False, dropna=False)[
            ["label", "source_path"]
        ]
        .nunique(dropna=False)
        .max()
    )
    if int(consistency.max()) != 1:
        raise ValueError("A DADS source locator has inconsistent label or source_path")

    frame["split_before_repair"] = frame["split"]
    frame["source_id"] = [
        _source_id(parquet_file, row_group, row_in_group)
        for parquet_file, row_group, row_in_group in frame[
            LOCATOR_COLUMNS
        ].itertuples(index=False, name=None)
    ]
    return frame


def _cache_index(value: object) -> int | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    index = int(value)
    if index < 0:
        raise ValueError("cache_index must be non-negative")
    return index


def _resolve_cache_path(value: object, root: Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=True)


def final_waveform_hashes(
    waveform: np.ndarray, target_samples: int
) -> tuple[str, str]:
    """Hash the deterministic waveform seen by validation/test inference.

    The float32 hash catches bit-exact duplicates.  The PCM16 hash is the
    grouping key and also catches files that differ only by insignificant
    floating-point encoding or gain before the model's peak normalization.
    """

    audio = np.asarray(waveform)
    if audio.ndim != 1:
        raise ValueError(f"Cached segment must be one-dimensional, got {audio.shape}")
    if int(audio.size) != int(target_samples):
        raise ValueError(
            f"Cached segment has {audio.size} samples; expected {target_samples}"
        )
    if not np.isfinite(audio).all():
        raise ValueError("Cached segment contains NaN or infinite values")

    normalized = peak_normalize(
        to_fixed_length(
            audio.astype(np.float32, copy=False),
            int(target_samples),
            random_crop=False,
        )
    )
    normalized = np.asarray(normalized, dtype="<f4", order="C").copy()
    normalized[normalized == 0.0] = 0.0
    float32_sha256 = hashlib.sha256(normalized.tobytes(order="C")).hexdigest()

    pcm16 = np.rint(np.clip(normalized, -1.0, 1.0) * 32767.0).astype(
        "<i2", copy=False
    )
    pcm16_sha256 = hashlib.sha256(pcm16.tobytes(order="C")).hexdigest()
    return float32_sha256, pcm16_sha256


def _hash_cached_segments(
    frame: pd.DataFrame,
    root: Path,
    target_samples: int,
) -> pd.DataFrame:
    cache_indices: Iterable[object]
    if "cache_index" in frame.columns:
        cache_indices = frame["cache_index"]
    else:
        cache_indices = [None] * len(frame)

    resolved_paths: dict[str, Path] = {}
    keys: list[tuple[str, int | None]] = []
    for cache_path, cache_index in zip(frame["cache_path"], cache_indices):
        raw_path = str(cache_path)
        if raw_path not in resolved_paths:
            resolved_paths[raw_path] = _resolve_cache_path(raw_path, root)
        keys.append((raw_path, _cache_index(cache_index)))

    unique_keys = sorted(set(keys), key=lambda item: (item[0], -1 if item[1] is None else item[1]))
    observed: dict[tuple[str, int | None], tuple[str, str]] = {}
    for raw_path, cache_index in tqdm(
        unique_keys, desc="Hashing final 1-second waveforms", unit="segment"
    ):
        cache = np.load(
            resolved_paths[raw_path],
            mmap_mode="r",
            allow_pickle=False,
        )
        if cache_index is None:
            waveform = np.asarray(cache)
        else:
            if cache.ndim != 2:
                raise ValueError(
                    f"Indexed segment cache must be 2-D: {resolved_paths[raw_path]}"
                )
            if cache_index >= cache.shape[0]:
                raise IndexError(
                    f"cache_index={cache_index} is outside {resolved_paths[raw_path]}"
                )
            waveform = np.asarray(cache[cache_index])
        observed[(raw_path, cache_index)] = final_waveform_hashes(
            waveform, target_samples
        )

    output = frame.copy()
    output["segment_float32_sha256"] = [observed[key][0] for key in keys]
    output["segment_pcm16_sha256"] = [observed[key][1] for key in keys]
    output["content_group"] = output["segment_pcm16_sha256"].map(
        lambda value: f"dads_final_pcm16:{value}"
    )
    return output


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        self.parent[larger] = smaller


def _validate_group_labels(frame: pd.DataFrame, column: str) -> None:
    conflicts = frame.groupby(column, dropna=False)["label"].nunique()
    conflicts = conflicts[conflicts != 1]
    if not conflicts.empty:
        raise ValueError(
            f"Identical/grouped DADS audio has conflicting labels in {column}: "
            f"{len(conflicts)} groups"
        )


def _attach_components(frame: pd.DataFrame) -> pd.DataFrame:
    source_ids = sorted(frame["source_id"].unique())
    union_find = _UnionFind(source_ids)

    grouping_columns = ["segment_pcm16_sha256", "segment_float32_sha256"]
    for optional in ("raw_audio_sha256", "recording_group"):
        if optional in frame.columns and frame[optional].notna().any():
            grouping_columns.append(optional)

    for column in grouping_columns:
        populated = frame.loc[
            frame[column].notna() & (frame[column].astype(str).str.strip() != "")
        ]
        _validate_group_labels(populated, column)
        for _, rows in populated.groupby(column, sort=True):
            members = sorted(rows["source_id"].unique())
            for member in members[1:]:
                union_find.union(members[0], member)

    populated_paths = frame.loc[frame["source_path"].str.strip() != ""]
    _validate_group_labels(populated_paths, "source_path")
    for _, rows in populated_paths.groupby("source_path", sort=True):
        members = sorted(rows["source_id"].unique())
        for member in members[1:]:
            union_find.union(members[0], member)

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for source_id in source_ids:
        members_by_root[union_find.find(source_id)].append(source_id)
    component_for_source = {
        source_id: "dads_component:"
        + _stable_digest(*sorted(members_by_root[union_find.find(source_id)]))
        for source_id in source_ids
    }

    output = frame.copy()
    output["split_component"] = output["source_id"].map(component_for_source)
    component_labels = output.groupby("split_component")["label"].nunique()
    if int(component_labels.max()) != 1:
        raise AssertionError("A split component contains multiple labels")
    return output


def _allocate_counts(
    total: int, split_ratios: dict[str, float]
) -> dict[str, int]:
    if set(split_ratios) != set(SPLIT_ORDER):
        raise ValueError(f"split_ratios must define {SPLIT_ORDER}")
    ratios = np.asarray(
        [float(split_ratios[split]) for split in SPLIT_ORDER], dtype=np.float64
    )
    if not np.isfinite(ratios).all() or (ratios < 0).any() or ratios.sum() <= 0:
        raise ValueError("split ratios must be finite, non-negative and non-zero")
    ratios /= ratios.sum()
    raw = ratios * int(total)
    counts = np.floor(raw).astype(int)
    remainder = int(total) - int(counts.sum())
    order = sorted(
        range(len(SPLIT_ORDER)),
        key=lambda index: (-(raw[index] - counts[index]), index),
    )
    for index in order[:remainder]:
        counts[index] += 1
    return {
        split: int(count)
        for split, count in zip(SPLIT_ORDER, counts)
    }


def _assign_components(
    frame: pd.DataFrame,
    *,
    split_ratios: dict[str, float],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, dict[str, int]]]:
    source_rows = []
    for source_id, rows in frame.groupby("source_id", sort=True):
        original_splits = sorted(set(rows["split_before_repair"].astype(str)))
        first = rows.iloc[0]
        source_rows.append(
            {
                "source_id": source_id,
                "label": int(rows["label"].iloc[0]),
                "parquet_file": str(first["parquet_file"]),
                "row_group": int(first["row_group"]),
                "row_in_group": int(first["row_in_group"]),
                "source_path": str(first["source_path"]),
                "split_component": str(rows["split_component"].iloc[0]),
                "original_splits": "|".join(original_splits),
                "original_split_count": len(original_splits),
                "input_segment_rows": int(len(rows)),
                "unique_content_segments": int(
                    rows["segment_pcm16_sha256"].nunique()
                ),
            }
        )
    sources = pd.DataFrame(source_rows)

    target_counts = {
        int(label): _allocate_counts(int(len(rows)), split_ratios)
        for label, rows in sources.groupby("label", sort=True)
    }
    assignment: dict[str, str] = {}
    assigned_counts: dict[int, Counter[str]] = {
        int(label): Counter()
        for label in sorted(sources["label"].unique())
    }
    components: list[dict[str, Any]] = []

    for component, rows in sources.groupby("split_component", sort=True):
        label = int(rows["label"].iloc[0])
        if int(rows["label"].nunique()) != 1:
            raise AssertionError("A split component contains multiple source labels")
        original_counts: Counter[str] = Counter()
        for value in rows["original_splits"]:
            for split in str(value).split("|"):
                original_counts[split] += 1
        record = {
            "component": str(component),
            "label": label,
            "source_count": int(len(rows)),
            "original_counts": original_counts,
        }
        components.append(record)

    # Re-split every connected component from scratch.  Reusing the legacy
    # singleton assignments would leave almost all of the repeatedly inspected
    # historical test set intact and can also make the grouped source ratios
    # drift.  Original split membership is used only as a deterministic
    # tie-breaker that minimizes needless movement.
    components.sort(
        key=lambda record: (
            -int(record["source_count"]),
            _stable_digest(seed, record["component"]),
        )
    )
    split_rank = {split: index for index, split in enumerate(SPLIT_ORDER)}
    for record in components:
        label = int(record["label"])
        size = int(record["source_count"])
        candidates = []
        for split in SPLIT_ORDER:
            projected = assigned_counts[label].copy()
            projected[split] += size
            squared_error = sum(
                (
                    (projected[name] - target_counts[label][name])
                    / max(1, target_counts[label][name])
                )
                ** 2
                for name in SPLIT_ORDER
            )
            moved_sources = size - int(record["original_counts"].get(split, 0))
            candidates.append(
                (
                    squared_error,
                    moved_sources,
                    _stable_digest(seed, record["component"], split),
                    split_rank[split],
                    split,
                )
            )
        chosen = min(candidates)[-1]
        assignment[str(record["component"])] = chosen
        assigned_counts[label][chosen] += size

    output = frame.copy()
    output["split"] = output["split_component"].map(assignment)
    if output["split"].isna().any():
        raise AssertionError("A split component was not assigned")

    sources["split"] = sources["split_component"].map(assignment)
    sources["moved"] = [
        split not in str(original_splits).split("|")
        or int(original_count) != 1
        for split, original_splits, original_count in zip(
            sources["split"],
            sources["original_splits"],
            sources["original_split_count"],
        )
    ]
    return output, sources, target_counts


def _canonical_segments(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(
        [
            "segment_pcm16_sha256",
            "segment_float32_sha256",
            "source_id",
            *LOCATOR_COLUMNS,
            "segment_index",
            "cache_path",
        ],
        kind="stable",
    )
    canonical = ordered.drop_duplicates("segment_pcm16_sha256", keep="first").copy()
    kept_indices = set(canonical.index)

    duplicate_mask = frame.duplicated(
        "segment_pcm16_sha256", keep=False
    )
    duplicate_rows = frame.loc[duplicate_mask].copy()
    duplicate_rows["kept"] = duplicate_rows.index.map(kept_indices.__contains__)
    group_sizes = duplicate_rows["segment_pcm16_sha256"].value_counts()
    duplicate_rows["duplicate_group_size"] = duplicate_rows[
        "segment_pcm16_sha256"
    ].map(group_sizes)
    cross_split_before = (
        duplicate_rows.groupby("segment_pcm16_sha256")["split_before_repair"]
        .nunique()
        .gt(1)
    )
    duplicate_rows["cross_split_before"] = duplicate_rows[
        "segment_pcm16_sha256"
    ].map(cross_split_before)

    if canonical["segment_pcm16_sha256"].duplicated().any():
        raise AssertionError("Canonicalization did not remove duplicate content")
    split_variation = frame.groupby("segment_pcm16_sha256")["split"].nunique()
    if int(split_variation.max()) != 1:
        raise AssertionError("A content group was assigned to multiple splits")
    return canonical, duplicate_rows


def _round_robin_indices(
    rows: pd.DataFrame,
    count: int,
    *,
    seed: int,
    label: int,
    split: str,
) -> list[int]:
    if len(rows) < count:
        raise ValueError(
            f"Not enough unique label={label} segments in split={split}: "
            f"available={len(rows)}, requested={count}"
        )

    queues: dict[str, list[int]] = {}
    for source_id, source_rows in rows.groupby("source_id", sort=True):
        queues[str(source_id)] = sorted(
            (int(index) for index in source_rows.index),
            key=lambda index: _stable_digest(
                seed,
                label,
                split,
                source_id,
                rows.at[index, "segment_pcm16_sha256"],
            ),
        )
    source_order = sorted(
        queues,
        key=lambda source_id: _stable_digest(
            seed, label, split, source_id
        ),
    )
    cursors = {source_id: 0 for source_id in source_order}
    selected: list[int] = []
    while len(selected) < count:
        progress = False
        for source_id in source_order:
            cursor = cursors[source_id]
            if cursor >= len(queues[source_id]):
                continue
            selected.append(queues[source_id][cursor])
            cursors[source_id] = cursor + 1
            progress = True
            if len(selected) == count:
                break
        if not progress:
            raise AssertionError("Round-robin segment selection exhausted unexpectedly")
    return selected


def _select_balanced_segments(
    canonical: pd.DataFrame,
    *,
    segments_per_class: int | None,
    split_ratios: dict[str, float],
    seed: int,
) -> tuple[pd.DataFrame, dict[int, dict[str, int]] | None]:
    if segments_per_class is None:
        return canonical.copy(), None
    if segments_per_class <= 0:
        raise ValueError("segments_per_class must be positive")

    targets = {
        label: _allocate_counts(segments_per_class, split_ratios)
        for label in (0, 1)
    }
    selected_indices: list[int] = []
    for label in (0, 1):
        for split in SPLIT_ORDER:
            candidates = canonical.loc[
                (canonical["label"] == label) & (canonical["split"] == split)
            ]
            selected_indices.extend(
                _round_robin_indices(
                    candidates,
                    targets[label][split],
                    seed=seed,
                    label=label,
                    split=split,
                )
            )
    selected = canonical.loc[selected_indices].copy()
    return selected, targets


def _pairwise_overlap(
    frame: pd.DataFrame, column: str, *, split_column: str = "split"
) -> dict[str, int]:
    values = {
        split: {
            value
            for value in rows[column].dropna().astype(str)
            if value.strip()
        }
        for split, rows in frame.groupby(split_column, sort=True)
    }
    return {
        f"{left}_{right}": len(
            values.get(left, set()) & values.get(right, set())
        )
        for index, left in enumerate(SPLIT_ORDER)
        for right in SPLIT_ORDER[index + 1 :]
    }


def _counts(frame: pd.DataFrame, *, split_column: str = "split") -> dict[str, Any]:
    return {
        "total": int(len(frame)),
        "by_label": {
            str(int(label)): int(count)
            for label, count in frame["label"].value_counts().sort_index().items()
        },
        "by_split": {
            str(split): int(count)
            for split, count in frame[split_column]
            .value_counts()
            .reindex(SPLIT_ORDER, fill_value=0)
            .items()
        },
        "by_label_split": {
            str(label): {
                split: int(
                    (
                        (frame["label"] == label)
                        & (frame[split_column] == split)
                    ).sum()
                )
                for split in SPLIT_ORDER
            }
            for label in (0, 1)
        },
    }


def validate_guarded_manifest(
    manifest_path: Path,
    *,
    expected_segments_per_class: int | None = None,
    split_ratios: dict[str, float] | None = None,
    verify_cache_content: bool = False,
    root: Path | None = None,
    target_samples: int | None = None,
) -> dict[str, Any]:
    frame = pd.read_csv(manifest_path, low_memory=False)
    missing = sorted(GUARD_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            f"Segment-guarded manifest is missing columns: {missing}"
        )
    if frame.empty:
        raise ValueError("Segment-guarded manifest is empty")
    if set(frame["manifest_protocol"].astype(str)) != {PROTOCOL}:
        raise ValueError("Unexpected or mixed segment-guard protocol")
    if set(frame["split"].astype(str)) != set(SPLIT_ORDER):
        raise ValueError("Guarded manifest must contain train, val and test")
    labels = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not labels.isin([0, 1]).all():
        raise ValueError("Guarded manifest labels must be binary")
    frame["label"] = labels

    for column in ("segment_float32_sha256", "segment_pcm16_sha256"):
        valid = frame[column].astype(str).str.fullmatch(r"[0-9a-f]{64}")
        if not bool(valid.all()):
            raise ValueError(f"Malformed hashes in {column}")
    if frame["segment_pcm16_sha256"].duplicated().any():
        raise ValueError("Guarded manifest contains duplicate final waveforms")
    if int(frame.groupby("source_id")["split"].nunique().max()) != 1:
        raise ValueError("A source_id crosses guarded splits")
    if "source_path" in frame.columns:
        populated = frame.loc[
            frame["source_path"].fillna("").astype(str).str.strip() != ""
        ]
        if not populated.empty and int(
            populated.groupby("source_path")["split"].nunique().max()
        ) != 1:
            raise ValueError("A source_path crosses guarded splits")
    if any(_pairwise_overlap(frame, "segment_float32_sha256").values()):
        raise ValueError("Float32 waveform hashes overlap guarded splits")
    if any(_pairwise_overlap(frame, "segment_pcm16_sha256").values()):
        raise ValueError("PCM16 waveform hashes overlap guarded splits")

    if verify_cache_content:
        if target_samples is None or int(target_samples) <= 0:
            raise ValueError(
                "target_samples must be positive when verifying cache content"
            )
        cache_root = Path.cwd() if root is None else root
        cache_root = cache_root.resolve(strict=True)
        rehashed = _hash_cached_segments(
            frame, cache_root, int(target_samples)
        )
        for column in (
            "segment_float32_sha256",
            "segment_pcm16_sha256",
        ):
            matches = rehashed[column].astype(str) == frame[column].astype(str)
            if not bool(matches.all()):
                raise ValueError(
                    f"{int((~matches).sum())} cached waveforms no longer "
                    f"match {column}"
                )

    if expected_segments_per_class is not None:
        observed = frame["label"].value_counts().to_dict()
        expected = {0: int(expected_segments_per_class), 1: int(expected_segments_per_class)}
        if observed != expected:
            raise ValueError(
                f"Guarded manifest class counts differ: observed={observed}, "
                f"expected={expected}"
            )
        if split_ratios is not None:
            target = _allocate_counts(
                int(expected_segments_per_class), split_ratios
            )
            for label in (0, 1):
                actual = (
                    frame.loc[frame["label"] == label, "split"]
                    .value_counts()
                    .reindex(SPLIT_ORDER, fill_value=0)
                    .to_dict()
                )
                if actual != target:
                    raise ValueError(
                        f"Guarded split counts differ for label={label}: "
                        f"observed={actual}, expected={target}"
                    )
    return {
        "passed": True,
        "protocol": PROTOCOL,
        "cache_content_verified": bool(verify_cache_content),
        "counts": _counts(frame),
        "source_overlap": _pairwise_overlap(frame, "source_id"),
        "float32_overlap": _pairwise_overlap(
            frame, "segment_float32_sha256"
        ),
        "pcm16_overlap": _pairwise_overlap(frame, "segment_pcm16_sha256"),
    }


def prepare(
    root: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    target_samples: int,
    segments_per_class: int | None,
    split_ratios: dict[str, float],
    seed: int,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    output_dir = output_dir.resolve(strict=False)
    output_manifest = output_dir / "manifests" / manifest_path.name
    source_registry_path = output_dir / "source_registry.csv"
    duplicate_path = output_dir / "duplicate_segments.csv"
    audit_path = output_dir / "audit.json"
    outputs = (
        output_manifest,
        source_registry_path,
        duplicate_path,
        audit_path,
    )
    occupied = [path for path in outputs if path.exists()]
    if occupied and not allow_overwrite:
        raise FileExistsError(
            "Refusing to overwrite guarded outputs: "
            + ", ".join(path.as_posix() for path in occupied)
        )

    original = _load_manifest(manifest_path)
    hashed = _hash_cached_segments(original, root, int(target_samples))
    component_rows = _attach_components(hashed)
    assigned, source_registry, source_targets = _assign_components(
        component_rows,
        split_ratios=split_ratios,
        seed=int(seed),
    )
    canonical, duplicate_rows = _canonical_segments(assigned)
    selected, segment_targets = _select_balanced_segments(
        canonical,
        segments_per_class=segments_per_class,
        split_ratios=split_ratios,
        seed=int(seed),
    )

    selected["manifest_protocol"] = PROTOCOL
    split_rank = {split: index for index, split in enumerate(SPLIT_ORDER)}
    selected["_split_rank"] = selected["split"].map(split_rank)
    selected = selected.sort_values(
        [
            "_split_rank",
            "label",
            *LOCATOR_COLUMNS,
            "segment_index",
        ],
        kind="stable",
    ).drop(columns="_split_rank")
    selected = selected.reset_index(drop=True)

    selected_counts = selected["source_id"].value_counts()
    source_registry["selected_segment_rows"] = (
        source_registry["source_id"].map(selected_counts).fillna(0).astype(int)
    )
    source_registry = source_registry.sort_values(
        ["split", "label", "source_id"], kind="stable"
    ).reset_index(drop=True)
    duplicate_rows = duplicate_rows.sort_values(
        ["segment_pcm16_sha256", "source_id", "segment_index"], kind="stable"
    ).reset_index(drop=True)

    validation = validate_guarded_manifest_frame(
        selected,
        expected_segments_per_class=segments_per_class,
        split_ratios=split_ratios if segments_per_class is not None else None,
    )

    _atomic_write_csv(output_manifest, selected)
    _atomic_write_csv(source_registry_path, source_registry)
    _atomic_write_csv(duplicate_path, duplicate_rows)

    cross_split_pcm_groups_before = int(
        hashed.groupby("segment_pcm16_sha256")["split_before_repair"]
        .nunique()
        .gt(1)
        .sum()
    )
    cross_split_float_groups_before = int(
        hashed.groupby("segment_float32_sha256")["split_before_repair"]
        .nunique()
        .gt(1)
        .sum()
    )
    duplicate_pcm_groups_before = int(
        hashed["segment_pcm16_sha256"].value_counts().gt(1).sum()
    )
    cross_split_pcm_hashes = set(
        hashed.groupby("segment_pcm16_sha256")["split_before_repair"]
        .nunique()
        .loc[lambda values: values.gt(1)]
        .index.astype(str)
    )
    cross_split_pcm_rows = hashed.loc[
        hashed["segment_pcm16_sha256"].isin(cross_split_pcm_hashes)
    ]
    moved_sources = int(source_registry["moved"].sum())
    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "target_samples": int(target_samples),
        "segments_per_class": segments_per_class,
        "split_ratios": {
            split: float(split_ratios[split]) for split in SPLIT_ORDER
        },
        "input": {
            "manifest": _display_path(manifest_path, root),
            "manifest_sha256": file_sha256(manifest_path),
            "segments": _counts(
                hashed, split_column="split_before_repair"
            ),
            "sources": _counts(
                hashed.drop_duplicates("source_id"),
                split_column="split_before_repair",
            ),
            "cross_split_source_id_overlap": _pairwise_overlap(
                hashed,
                "source_id",
                split_column="split_before_repair",
            ),
            "cross_split_source_path_overlap": _pairwise_overlap(
                hashed,
                "source_path",
                split_column="split_before_repair",
            ),
            "cross_split_float32_hash_overlap": _pairwise_overlap(
                hashed,
                "segment_float32_sha256",
                split_column="split_before_repair",
            ),
            "cross_split_pcm16_hash_overlap": _pairwise_overlap(
                hashed,
                "segment_pcm16_sha256",
                split_column="split_before_repair",
            ),
            "duplicate_pcm16_groups": duplicate_pcm_groups_before,
            "cross_split_float32_groups": cross_split_float_groups_before,
            "cross_split_pcm16_groups": cross_split_pcm_groups_before,
            "cross_split_pcm16_affected_segments": _counts(
                cross_split_pcm_rows,
                split_column="split_before_repair",
            ),
        },
        "repair": {
            "split_components": int(
                assigned["split_component"].nunique()
            ),
            "source_targets": source_targets,
            "source_assignment": _counts(source_registry),
            "moved_sources": moved_sources,
            "segments_after_content_dedup": int(len(canonical)),
            "duplicate_rows_removed": int(len(hashed) - len(canonical)),
            "segment_targets": segment_targets,
            "balance_rows_discarded": int(len(canonical) - len(selected)),
        },
        "output": {
            "manifest": {
                "path": _display_path(output_manifest, root),
                "sha256": file_sha256(output_manifest),
                "rows": int(len(selected)),
            },
            "source_registry": {
                "path": _display_path(source_registry_path, root),
                "sha256": file_sha256(source_registry_path),
                "rows": int(len(source_registry)),
            },
            "duplicate_segments": {
                "path": _display_path(duplicate_path, root),
                "sha256": file_sha256(duplicate_path),
                "rows": int(len(duplicate_rows)),
            },
            "validation": validation,
        },
        "historical_manifest_modified": False,
        "audio_cache_modified": False,
        "training_started": False,
        "limitations": [
            "Exact and PCM16-equivalent final waveforms are guarded.",
            "Recording-session and near-duplicate similarity require a separate provenance/similarity audit.",
            "The historical test split is not restored to a fresh unseen holdout by this repair.",
        ],
    }
    _atomic_write_text(
        audit_path,
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    )
    return report


def validate_guarded_manifest_frame(
    frame: pd.DataFrame,
    *,
    expected_segments_per_class: int | None = None,
    split_ratios: dict[str, float] | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="dads-guard-validation-") as directory:
        path = Path(directory) / "manifest.csv"
        frame.to_csv(path, index=False)
        return validate_guarded_manifest(
            path,
            expected_segments_per_class=expected_segments_per_class,
            split_ratios=split_ratios,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a non-destructive DADS manifest grouped by final waveform "
            "content, deduplicated globally and balanced by segment."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "artifacts/manifests/dads_balanced_5000_seed42.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/dads_segment_guarded_v3"),
    )
    parser.add_argument("--target-samples", type=int, default=16000)
    parser.add_argument("--segments-per-class", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve(strict=True)
    manifest = (
        args.manifest
        if args.manifest.is_absolute()
        else root / args.manifest
    )
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else root / args.output_dir
    )
    report = prepare(
        root,
        manifest,
        output_dir,
        target_samples=args.target_samples,
        segments_per_class=args.segments_per_class,
        split_ratios={
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        seed=args.seed,
        allow_overwrite=args.allow_overwrite,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "protocol": report["protocol"],
                "cross_split_pcm16_groups_before": report["input"][
                    "cross_split_pcm16_groups"
                ],
                "duplicate_rows_removed": report["repair"][
                    "duplicate_rows_removed"
                ],
                "balance_rows_discarded": report["repair"][
                    "balance_rows_discarded"
                ],
                "manifest": report["output"]["manifest"],
                "training_started": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
