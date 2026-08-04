from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import scipy
from tqdm import tqdm

from .audio import decode_wav_bytes, ensure_sample_rate, peak_normalize
from .data_firewall import file_sha256


PROTOCOL = "dads_native_half_second_content_component_v2"
SPLIT_ORDER = ("train", "val", "test")
TARGET_SAMPLE_RATE = 16000
TARGET_SAMPLES = 8000
MAX_ALLOCATION_RATIO_ERROR = 0.001
LOCATOR_COLUMNS = ("parquet_file", "row_group", "row_in_group")
HASH_COLUMNS = ("segment_float32_sha256", "segment_pcm16_sha256")
REQUIRED_SOURCE_COLUMNS = {
    "split",
    "label",
    "parquet_file",
    "row_group",
    "row_in_group",
    "source_path",
    "original_samples",
    "raw_audio_sha256",
    "recording_group",
    "kept",
}
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
    "cache_index",
    "raw_audio_sha256",
    "recording_group",
    "source_id",
    "split_component",
    "segment_float32_sha256",
    "segment_pcm16_sha256",
    "model_samples",
    "manifest_protocol",
    "evaluation_role",
    "consumption_status",
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


def _strict_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Expected a strict boolean, observed {value!r}")


def _load_sources(
    root: Path,
    source_registry_path: Path,
    dedup_audit_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    audit = json.loads(dedup_audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("passed") is not True
        or audit.get("protocol") != "dads_raw_audio_sha256_dedup_v2"
    ):
        raise ValueError("DADS dedup-v2 audit is not a passed authoritative audit")
    audited = audit.get("outputs", {}).get("source_registry", {})
    if str(audited.get("sha256", "")) != file_sha256(source_registry_path):
        raise ValueError("DADS dedup-v2 source registry no longer matches its audit")

    sources = pd.read_csv(source_registry_path, low_memory=False)
    missing = sorted(REQUIRED_SOURCE_COLUMNS - set(sources.columns))
    if missing:
        raise ValueError(f"DADS source registry is missing columns: {missing}")
    if sources.empty:
        raise ValueError("DADS source registry is empty")
    sources = sources.copy()
    sources["kept"] = [_strict_bool(value) for value in sources["kept"]]
    if not bool(sources["kept"].all()):
        raise ValueError("DADS dedup-v2 source registry contains non-canonical rows")
    sources["label"] = pd.to_numeric(sources["label"], errors="raise").astype(int)
    if not bool(sources["label"].isin([0, 1]).all()):
        raise ValueError("DADS source labels must be binary")
    for column in ("row_group", "row_in_group", "original_samples"):
        sources[column] = pd.to_numeric(sources[column], errors="raise").astype(int)
        if bool((sources[column] < 0).any()):
            raise ValueError(f"{column} must be non-negative")
    for column in ("parquet_file", "source_path", "raw_audio_sha256", "recording_group"):
        sources[column] = sources[column].fillna("").astype(str)
    if not bool(sources["raw_audio_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()):
        raise ValueError("DADS source registry contains malformed raw audio hashes")
    if not bool(sources["raw_audio_sha256"].is_unique):
        raise ValueError("DADS dedup-v2 source registry is not raw-audio unique")
    if bool(sources.duplicated(list(LOCATOR_COLUMNS)).any()):
        raise ValueError("DADS source registry repeats a Parquet source locator")
    if set(sources["split"].astype(str)) != set(SPLIT_ORDER):
        raise ValueError("DADS source registry must contain train, val and test")
    sources["original_split"] = sources["split"].astype(str)
    sources["source_id"] = sources["recording_group"]
    if not bool(sources["source_id"].is_unique):
        raise ValueError("DADS source_id must uniquely identify one raw recording")
    return (
        sources.sort_values(list(LOCATOR_COLUMNS), kind="stable").reset_index(drop=True),
        audit,
    )


def _waveform_hashes(waveform: np.ndarray) -> tuple[str, str]:
    normalized = np.asarray(waveform, dtype="<f4", order="C").copy()
    normalized[normalized == 0.0] = 0.0
    float32_sha256 = hashlib.sha256(normalized.tobytes(order="C")).hexdigest()
    pcm16 = np.rint(np.clip(normalized, -1.0, 1.0) * 32767.0).astype(
        "<i2", copy=False
    )
    pcm16_sha256 = hashlib.sha256(pcm16.tobytes(order="C")).hexdigest()
    return float32_sha256, pcm16_sha256


def _planned_windows(sources: pd.DataFrame) -> int:
    return int((sources["original_samples"] // TARGET_SAMPLES).sum())


def _extract_native_windows(
    root: Path,
    sources: pd.DataFrame,
    cache_path: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    planned = _planned_windows(sources)
    if planned <= 0:
        raise ValueError("No complete native half-second windows are available")

    temporary = cache_path.with_name(f".{cache_path.name}.building.npy")
    if cache_path.exists() or temporary.exists():
        raise FileExistsError(
            f"Refusing to overwrite native half-second cache: {cache_path}"
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(planned, TARGET_SAMPLES),
    )

    requests: dict[tuple[str, int], list[pd.Series]] = defaultdict(list)
    for _, row in sources.iterrows():
        requests[(str(row["parquet_file"]), int(row["row_group"]))].append(row)

    rows: list[dict[str, Any]] = []
    cache_index = 0
    silent_windows = 0
    degenerate_windows = 0
    discarded_tail_samples = 0
    short_sources = 0
    decoded_sources = 0
    try:
        for (parquet_value, row_group), group_sources in tqdm(
            sorted(requests.items()),
            desc="Building native 0.5-second DADS cache",
            unit="row-group",
        ):
            parquet_path = Path(parquet_value)
            if not parquet_path.is_absolute():
                parquet_path = root / parquet_path
            parquet_path = parquet_path.resolve(strict=True)
            parquet = pq.ParquetFile(parquet_path)
            if row_group < 0 or row_group >= parquet.num_row_groups:
                raise ValueError(f"Invalid row group {row_group} in {parquet_path}")
            audios = (
                parquet.read_row_group(row_group, columns=["audio"])
                .column("audio")
                .to_pylist()
            )
            for source in sorted(
                group_sources, key=lambda value: int(value["row_in_group"])
            ):
                row_in_group = int(source["row_in_group"])
                if row_in_group < 0 or row_in_group >= len(audios):
                    raise ValueError(
                        f"Invalid row {row_in_group} in {parquet_path}:{row_group}"
                    )
                wav_bytes = audios[row_in_group].get("bytes")
                if not isinstance(wav_bytes, bytes) or not wav_bytes:
                    raise ValueError(
                        f"Missing WAV bytes in {parquet_path}:{row_group}:{row_in_group}"
                    )
                observed_raw_sha = hashlib.sha256(wav_bytes).hexdigest()
                if observed_raw_sha != str(source["raw_audio_sha256"]):
                    raise ValueError(
                        "DADS raw WAV changed after dedup-v2 audit: "
                        f"{parquet_value}:{row_group}:{row_in_group}"
                    )
                audio, sample_rate = decode_wav_bytes(wav_bytes)
                audio = ensure_sample_rate(audio, sample_rate, TARGET_SAMPLE_RATE)
                if int(audio.size) != int(source["original_samples"]):
                    raise ValueError(
                        "DADS resampled length changed after historical manifest: "
                        f"{parquet_value}:{row_group}:{row_in_group}"
                    )
                decoded_sources += 1
                count = int(audio.size // TARGET_SAMPLES)
                discarded_tail_samples += int(audio.size - count * TARGET_SAMPLES)
                if count == 0:
                    short_sources += 1
                    continue
                for segment_index in range(count):
                    start = segment_index * TARGET_SAMPLES
                    end = start + TARGET_SAMPLES
                    waveform = np.asarray(audio[start:end], dtype=np.float32)
                    peak = float(np.max(np.abs(waveform)))
                    if peak <= 1.0e-8:
                        silent_windows += 1
                        continue
                    waveform = peak_normalize(waveform)
                    if float(np.std(waveform, dtype=np.float64)) < 1.0e-4:
                        degenerate_windows += 1
                        continue
                    float_hash, pcm_hash = _waveform_hashes(waveform)
                    cache[cache_index] = waveform
                    rows.append(
                        {
                            "label": int(source["label"]),
                            "parquet_file": str(source["parquet_file"]),
                            "row_group": int(source["row_group"]),
                            "row_in_group": row_in_group,
                            "source_path": str(source["source_path"]),
                            "segment_index": segment_index,
                            "start_sample": start,
                            "end_sample": end,
                            "original_samples": int(audio.size),
                            "segment_kind": "native_half_second",
                            "cache_path": _display_path(cache_path, root),
                            "cache_index": cache_index,
                            "raw_audio_sha256": str(source["raw_audio_sha256"]),
                            "recording_group": str(source["recording_group"]),
                            "source_id": str(source["source_id"]),
                            "original_split": str(source["original_split"]),
                            "segment_float32_sha256": float_hash,
                            "segment_pcm16_sha256": pcm_hash,
                            "model_samples": TARGET_SAMPLES,
                        }
                    )
                    cache_index += 1
        cache.flush()
        del cache
        temporary.replace(cache_path)
    except BaseException:
        del cache
        temporary.unlink(missing_ok=True)
        raise

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Native half-second preparation produced no usable windows")
    if cache_index > planned:
        raise AssertionError("Native cache wrote beyond its planned capacity")
    return frame, {
        "planned_cache_rows": planned,
        "written_cache_rows": cache_index,
        "unused_cache_rows": planned - cache_index,
        "decoded_sources": decoded_sources,
        "short_sources": short_sources,
        "silent_windows": silent_windows,
        "degenerate_windows": degenerate_windows,
        "discarded_tail_samples": discarded_tail_samples,
    }


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


def _drop_cross_label_content(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    conflicting_values: dict[str, set[str]] = {}
    conflict_mask = np.zeros(len(frame), dtype=bool)
    for column in HASH_COLUMNS:
        label_counts = frame.groupby(column, sort=False)["label"].nunique()
        values = set(label_counts[label_counts > 1].index.astype(str))
        conflicting_values[column] = values
        if values:
            conflict_mask |= frame[column].astype(str).isin(values).to_numpy()
    conflicts = frame.loc[conflict_mask].copy()
    clean = frame.loc[~conflict_mask].copy()
    if clean.empty:
        raise ValueError("Every native DADS window was removed by label conflicts")
    return clean, conflicts, {
        f"{column}_groups": len(values)
        for column, values in conflicting_values.items()
    }


def _attach_components(frame: pd.DataFrame) -> pd.DataFrame:
    source_ids = sorted(frame["source_id"].astype(str).unique())
    union_find = _UnionFind(source_ids)
    for column in (*HASH_COLUMNS, "raw_audio_sha256", "recording_group"):
        for _, rows in frame.groupby(column, sort=True):
            members = sorted(rows["source_id"].astype(str).unique())
            for member in members[1:]:
                union_find.union(members[0], member)
    members_by_root: dict[str, list[str]] = defaultdict(list)
    for source_id in source_ids:
        members_by_root[union_find.find(source_id)].append(source_id)
    component_for_source = {
        source_id: "dads_halfsec_component:"
        + _stable_digest(*sorted(members_by_root[union_find.find(source_id)]))
        for source_id in source_ids
    }
    output = frame.copy()
    output["split_component"] = output["source_id"].map(component_for_source)
    component_labels = output.groupby("split_component")["label"].nunique()
    if int(component_labels.max()) != 1:
        raise ValueError("A native half-second split component contains both labels")
    return output


def _allocate_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    values = np.asarray([float(ratios[name]) for name in SPLIT_ORDER], dtype=np.float64)
    if not np.isfinite(values).all() or bool((values < 0).any()) or values.sum() <= 0:
        raise ValueError("Split ratios must be finite, non-negative and non-zero")
    values /= values.sum()
    raw = values * int(total)
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
        for split, count in zip(SPLIT_ORDER, counts, strict=True)
    }


def _assign_components(
    frame: pd.DataFrame,
    ratios: dict[str, float],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    all_sources = (
        frame[
            [
                "source_id",
                "label",
                "split_component",
                "original_split",
                "raw_audio_sha256",
                "recording_group",
                "parquet_file",
                "row_group",
                "row_in_group",
                "source_path",
            ]
        ]
        .drop_duplicates("source_id")
        .copy()
    )
    effective = (
        frame.sort_values(
            [
                "segment_pcm16_sha256",
                "segment_float32_sha256",
                "source_id",
                "segment_index",
            ],
            kind="stable",
        )
        .drop_duplicates("segment_pcm16_sha256", keep="first")
        .copy()
    )
    sources = (
        effective[
            [
                "source_id",
                "label",
                "split_component",
            ]
        ]
        .drop_duplicates("source_id")
        .copy()
    )
    source_targets = {
        int(label): _allocate_counts(int(len(rows)), ratios)
        for label, rows in sources.groupby("label", sort=True)
    }
    window_targets = {
        int(label): _allocate_counts(
            int(rows["segment_pcm16_sha256"].nunique()), ratios
        )
        for label, rows in effective.groupby("label", sort=True)
    }
    components = (
        effective.groupby(["split_component", "label"], sort=True)
        .agg(
            source_count=("source_id", "nunique"),
            window_count=("segment_pcm16_sha256", "nunique"),
        )
        .reset_index()
    )
    if components["split_component"].duplicated().any():
        raise ValueError("A split component contains multiple labels")
    normalized_ratios = _normalized_ratios(ratios)
    cumulative = []
    running = 0.0
    for split in SPLIT_ORDER:
        running += normalized_ratios[split]
        cumulative.append((split, running))
    assigned_source_counts = {label: Counter() for label in source_targets}
    assigned_window_counts = {label: Counter() for label in window_targets}
    assignment: dict[str, str] = {}
    component_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in components.itertuples(index=False):
        component = str(row.split_component)
        label = int(row.label)
        source_count = int(row.source_count)
        window_count = int(row.window_count)
        unit = int(_stable_digest(seed, component)[:16], 16) / float(1 << 64)
        chosen = next(
            split
            for split, boundary in cumulative
            if unit < boundary or split == SPLIT_ORDER[-1]
        )
        assignment[component] = chosen
        assigned_source_counts[label][chosen] += source_count
        assigned_window_counts[label][chosen] += window_count
        component_rows[label].append(
            {
                "component": component,
                "source_count": source_count,
                "window_count": window_count,
            }
        )

    def move_delta(
        counts: Counter,
        targets: dict[str, int],
        amount: int,
        source_split: str,
        target_split: str,
    ) -> float:
        source_target = max(1, int(targets[source_split]))
        target_target = max(1, int(targets[target_split]))
        before = (
            (counts[source_split] - targets[source_split]) / source_target
        ) ** 2 + (
            (counts[target_split] - targets[target_split]) / target_target
        ) ** 2
        after = (
            (
                counts[source_split]
                - amount
                - targets[source_split]
            )
            / source_target
        ) ** 2 + (
            (
                counts[target_split]
                + amount
                - targets[target_split]
            )
            / target_target
        ) ** 2
        return float(after - before)

    local_moves = {label: 0 for label in source_targets}
    local_passes = {label: 0 for label in source_targets}
    for label, items in component_rows.items():
        ordered = sorted(
            items,
            key=lambda item: _stable_digest(
                seed, "local-balance", item["component"]
            ),
        )
        for pass_index in range(20):
            moves = 0
            for item in ordered:
                component = str(item["component"])
                current = assignment[component]
                choices = []
                for candidate in SPLIT_ORDER:
                    if candidate == current:
                        continue
                    delta = move_delta(
                        assigned_source_counts[label],
                        source_targets[label],
                        int(item["source_count"]),
                        current,
                        candidate,
                    ) + move_delta(
                        assigned_window_counts[label],
                        window_targets[label],
                        int(item["window_count"]),
                        current,
                        candidate,
                    )
                    choices.append(
                        (
                            delta,
                            _stable_digest(
                                seed,
                                "local-move",
                                component,
                                candidate,
                            ),
                            candidate,
                        )
                    )
                delta, _, candidate = min(choices)
                if delta >= -1.0e-18:
                    continue
                source_count = int(item["source_count"])
                window_count = int(item["window_count"])
                assigned_source_counts[label][current] -= source_count
                assigned_source_counts[label][candidate] += source_count
                assigned_window_counts[label][current] -= window_count
                assigned_window_counts[label][candidate] += window_count
                assignment[component] = candidate
                moves += 1
            local_moves[label] += moves
            local_passes[label] = pass_index + 1
            if moves == 0:
                break

    output = frame.copy()
    output["split"] = output["split_component"].map(assignment)
    all_sources["split"] = all_sources["split_component"].map(assignment)
    if output["split"].isna().any() or all_sources["split"].isna().any():
        raise AssertionError("A native half-second component was not assigned")
    all_sources["moved_from_historical_split"] = (
        all_sources["split"].astype(str)
        != all_sources["original_split"].astype(str)
    )
    return output, all_sources, {
        "sources": source_targets,
        "unique_windows": window_targets,
        "method": "seeded_component_hash_then_deterministic_local_balance_v1",
        "local_balance_moves": local_moves,
        "local_balance_passes": local_passes,
    }


def _canonicalize_content(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(
        [
            "segment_pcm16_sha256",
            "segment_float32_sha256",
            "source_id",
            "segment_index",
        ],
        kind="stable",
    )
    canonical = ordered.drop_duplicates("segment_pcm16_sha256", keep="first").copy()
    kept_indices = set(canonical.index)
    duplicates = frame.loc[
        frame.duplicated("segment_pcm16_sha256", keep=False)
    ].copy()
    duplicates["kept"] = duplicates.index.map(kept_indices.__contains__)
    if canonical["segment_pcm16_sha256"].duplicated().any():
        raise AssertionError("Native half-second canonicalization failed")
    if int(frame.groupby("segment_pcm16_sha256")["split"].nunique().max()) != 1:
        raise AssertionError("Equivalent native windows were assigned across splits")
    return canonical, duplicates


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


def _counts(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "total": int(len(frame)),
        "by_label": {
            str(int(label)): int(count)
            for label, count in frame["label"].value_counts().sort_index().items()
        },
        "by_split": {
            split: int(count)
            for split, count in frame["split"]
            .value_counts()
            .reindex(SPLIT_ORDER, fill_value=0)
            .items()
        },
        "by_label_split": {
            str(label): {
                split: int(
                    ((frame["label"] == label) & (frame["split"] == split)).sum()
                )
                for split in SPLIT_ORDER
            }
            for label in (0, 1)
        },
    }


def _allocation_counts(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "sources": {
            str(label): {
                split: int(
                    frame.loc[
                        (frame["label"] == label)
                        & (frame["split"].astype(str) == split),
                        "source_id",
                    ].nunique()
                )
                for split in SPLIT_ORDER
            }
            for label in (0, 1)
        },
        "unique_windows": {
            str(label): {
                split: int(
                    (
                        (frame["label"] == label)
                        & (frame["split"].astype(str) == split)
                    ).sum()
                )
                for split in SPLIT_ORDER
            }
            for label in (0, 1)
        },
    }


def _allocation_max_ratio_error(
    actual: dict[str, Any],
    ratios: dict[str, float],
) -> float:
    normalized = _normalized_ratios(ratios)
    errors = []
    for unit in ("sources", "unique_windows"):
        for label in ("0", "1"):
            total = sum(int(actual[unit][label][split]) for split in SPLIT_ORDER)
            if total <= 0:
                raise ValueError(f"Allocation has no {unit} for label={label}")
            errors.extend(
                abs(
                    int(actual[unit][label][split]) / total
                    - normalized[split]
                )
                for split in SPLIT_ORDER
            )
    return float(max(errors))


def _normalized_ratios(values: dict[str, float]) -> dict[str, float]:
    ratios = {
        split: float(values[split])
        for split in SPLIT_ORDER
    }
    total = sum(ratios.values())
    if (
        not np.isfinite(list(ratios.values())).all()
        or any(value < 0 for value in ratios.values())
        or total <= 0
    ):
        raise ValueError("Split ratios must be finite, non-negative and non-zero")
    return {split: value / total for split, value in ratios.items()}


def _reproducibility_identity() -> dict[str, Any]:
    source_files = [
        Path(__file__).resolve(strict=True),
        Path(__file__).with_name("audio.py").resolve(strict=True),
    ]
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pyarrow.__version__,
        "source_sha256": {
            path.name: file_sha256(path)
            for path in source_files
        },
    }


def validate_manifest(
    manifest_path: Path,
    audit_path: Path,
    *,
    verify_cache_file: bool = False,
    expected_sample_rate: int | None = None,
    expected_clip_seconds: float | None = None,
    expected_split_ratios: dict[str, float] | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve(strict=True)
    audit_path = audit_path.resolve(strict=True)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True or audit.get("protocol") != PROTOCOL:
        raise ValueError("Native half-second DADS audit is not authoritative")
    audited_sample_rate = int(audit.get("sample_rate", -1))
    audited_target_samples = int(audit.get("target_samples", -1))
    audited_clip_seconds = float(audit.get("clip_seconds", -1.0))
    if (
        audited_sample_rate != TARGET_SAMPLE_RATE
        or audited_target_samples != TARGET_SAMPLES
        or not np.isclose(
            audited_clip_seconds,
            TARGET_SAMPLES / TARGET_SAMPLE_RATE,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise ValueError("Native half-second audit protocol parameters changed")
    if (
        expected_sample_rate is not None
        and int(expected_sample_rate) != audited_sample_rate
    ):
        raise ValueError("Training sample rate does not match the DADS audit")
    if (
        expected_clip_seconds is not None
        and not np.isclose(
            float(expected_clip_seconds),
            audited_clip_seconds,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise ValueError("Training clip length does not match the DADS audit")
    audited_ratios = _normalized_ratios(audit.get("split_ratios", {}))
    if expected_split_ratios is not None:
        expected_ratios = _normalized_ratios(expected_split_ratios)
        if any(
            not np.isclose(
                audited_ratios[split],
                expected_ratios[split],
                rtol=0.0,
                atol=1.0e-12,
            )
            for split in SPLIT_ORDER
        ):
            raise ValueError("Training split ratios do not match the DADS audit")
    expected_sha = str(audit.get("output", {}).get("manifest", {}).get("sha256", ""))
    if file_sha256(manifest_path) != expected_sha:
        raise ValueError("Native half-second manifest no longer matches its audit")
    frame = pd.read_csv(manifest_path, low_memory=False)
    missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Native half-second manifest is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Native half-second manifest is empty")
    if set(frame["manifest_protocol"].astype(str)) != {PROTOCOL}:
        raise ValueError("Unexpected native half-second manifest protocol")
    if set(frame["split"].astype(str)) != set(SPLIT_ORDER):
        raise ValueError("Native half-second manifest lacks a required split")
    if set(pd.to_numeric(frame["model_samples"], errors="raise").astype(int)) != {
        TARGET_SAMPLES
    }:
        raise ValueError("Native half-second manifest contains a non-8000 input")
    if set(frame["segment_kind"].astype(str)) != {"native_half_second"}:
        raise ValueError("Padding, loop or tail windows entered the native manifest")
    lengths = (
        pd.to_numeric(frame["end_sample"], errors="raise")
        - pd.to_numeric(frame["start_sample"], errors="raise")
    )
    if not bool((lengths == TARGET_SAMPLES).all()):
        raise ValueError("A native window is not exactly 8000 real samples")
    for column in HASH_COLUMNS:
        if not bool(frame[column].astype(str).str.fullmatch(r"[0-9a-f]{64}").all()):
            raise ValueError(f"Malformed native window hashes in {column}")
    if frame["segment_pcm16_sha256"].duplicated().any():
        raise ValueError("Native half-second manifest contains duplicate model inputs")
    for column in (
        "source_id",
        "recording_group",
        "raw_audio_sha256",
        "split_component",
        *HASH_COLUMNS,
    ):
        if int(frame.groupby(column)["split"].nunique().max()) != 1:
            raise ValueError(f"{column} crosses native DADS splits")
        if any(_pairwise_overlap(frame, column).values()):
            raise ValueError(f"{column} has a native DADS pairwise overlap")
    if int(frame.groupby("split_component")["label"].nunique().max()) != 1:
        raise ValueError("A native DADS split component contains both labels")
    allocation = audit.get("construction", {})
    if allocation.get("allocation_unit") != (
        "per_label_effective_source_and_unique_pcm16_window"
    ):
        raise ValueError("Native DADS allocation unit is missing or ambiguous")
    observed_allocation = _allocation_counts(frame)
    audited_actual = allocation.get("allocation_actual")
    if observed_allocation != audited_actual:
        raise ValueError(
            "Native DADS manifest no longer matches its audited source/window "
            "allocation"
        )
    audited_targets = allocation.get("allocation_targets", {})
    for unit in ("sources", "unique_windows"):
        for label in ("0", "1"):
            total = sum(
                int(observed_allocation[unit][label][split])
                for split in SPLIT_ORDER
            )
            expected_targets = _allocate_counts(total, audited_ratios)
            try:
                recorded_targets = {
                    split: int(audited_targets[unit][label][split])
                    for split in SPLIT_ORDER
                }
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Native DADS audit lacks {unit} allocation targets"
                ) from error
            if recorded_targets != expected_targets:
                raise ValueError(
                    f"Native DADS {unit} allocation targets are inconsistent"
                )
    allowed_error = float(
        allocation.get("max_allowed_abs_ratio_error", -1.0)
    )
    if not np.isclose(
        allowed_error,
        MAX_ALLOCATION_RATIO_ERROR,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("Native DADS allocation tolerance changed")
    observed_error = _allocation_max_ratio_error(
        observed_allocation, audited_ratios
    )
    recorded_error = float(allocation.get("max_abs_ratio_error", -1.0))
    if not np.isclose(
        observed_error, recorded_error, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("Native DADS allocation error no longer matches audit")
    if observed_error > allowed_error:
        raise ValueError(
            "Native DADS source/window allocation exceeds its frozen tolerance"
        )
    roles = {
        "train": "training",
        "val": "internal_model_selection",
        "test": "consumed_internal_development_test",
    }
    for split, role in roles.items():
        observed = set(frame.loc[frame["split"] == split, "evaluation_role"].astype(str))
        if observed != {role}:
            raise ValueError(f"Unexpected evaluation role for split={split}: {observed}")
    if set(frame["consumption_status"].astype(str)) != {
        "underlying_dads_pool_previously_consumed"
    }:
        raise ValueError("DADS consumption status was weakened")

    cache_paths = set(frame["cache_path"].astype(str))
    if len(cache_paths) != 1:
        raise ValueError("Native half-second manifest must bind one immutable cache")
    cache_path = Path(next(iter(cache_paths)))
    if not cache_path.is_absolute():
        cache_path = Path.cwd() / cache_path
    cache_path = cache_path.resolve(strict=True)
    cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    if cache.ndim != 2 or cache.shape[1] != TARGET_SAMPLES:
        raise ValueError(f"Unexpected native half-second cache shape: {cache.shape}")
    audited_cache_shape = audit.get("output", {}).get("cache", {}).get("shape")
    if audited_cache_shape != [int(value) for value in cache.shape]:
        raise ValueError(
            "Native half-second cache shape no longer matches its audit"
        )
    numeric_indices = pd.to_numeric(frame["cache_index"], errors="raise")
    if (
        not np.isfinite(numeric_indices.to_numpy(dtype=np.float64)).all()
        or not bool((numeric_indices == np.floor(numeric_indices)).all())
    ):
        raise ValueError("Native half-second cache indices must be finite integers")
    indices = numeric_indices.astype(int)
    if int(indices.min()) < 0 or int(indices.max()) >= int(cache.shape[0]):
        raise ValueError("Native half-second cache index is outside the cache")
    if indices.duplicated().any():
        raise ValueError(
            "Native half-second manifest maps multiple rows to one cache index"
        )
    if verify_cache_file:
        expected_cache_sha = str(
            audit.get("output", {}).get("cache", {}).get("sha256", "")
        )
        if file_sha256(cache_path) != expected_cache_sha:
            raise ValueError("Native half-second cache no longer matches its audit")
        ordered = frame.assign(cache_index_int=indices).sort_values(
            "cache_index_int", kind="stable"
        )
        for row in tqdm(
            ordered.itertuples(index=False),
            total=len(ordered),
            desc="Verifying native 0.5-second cache rows",
            unit="segment",
        ):
            waveform = np.asarray(
                cache[int(row.cache_index_int)], dtype=np.float32
            )
            float_hash, pcm_hash = _waveform_hashes(waveform)
            if (
                float_hash != str(row.segment_float32_sha256)
                or pcm_hash != str(row.segment_pcm16_sha256)
            ):
                raise ValueError(
                    "Native half-second cache row no longer matches its "
                    f"manifest hashes: cache_index={int(row.cache_index_int)}"
                )
    return {
        "passed": True,
        "protocol": PROTOCOL,
        "counts": _counts(frame),
        "source_overlap": _pairwise_overlap(frame, "source_id"),
        "raw_audio_overlap": _pairwise_overlap(frame, "raw_audio_sha256"),
        "float32_overlap": _pairwise_overlap(frame, "segment_float32_sha256"),
        "pcm16_overlap": _pairwise_overlap(frame, "segment_pcm16_sha256"),
        "split_component_overlap": _pairwise_overlap(
            frame, "split_component"
        ),
        "cache_shape": [int(value) for value in cache.shape],
        "cache_file_verified": bool(verify_cache_file),
        "fresh_final_holdout": False,
        "sample_rate": audited_sample_rate,
        "target_samples": audited_target_samples,
        "clip_seconds": audited_clip_seconds,
        "split_ratios": audited_ratios,
    }


def _seal_audit(
    manifest_path: Path,
    audit_path: Path,
    report: dict[str, Any],
    *,
    verify_cache_file: bool,
) -> dict[str, Any]:
    staging_path = audit_path.with_name(f".{audit_path.name}.validation")
    if staging_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite stale audit staging file: {staging_path}"
        )
    try:
        _atomic_write_text(
            staging_path,
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        )
        validation = validate_manifest(
            manifest_path,
            staging_path,
            verify_cache_file=verify_cache_file,
            expected_sample_rate=int(report["sample_rate"]),
            expected_clip_seconds=float(report["clip_seconds"]),
            expected_split_ratios=report["split_ratios"],
        )
        _atomic_write_text(
            audit_path,
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        )
        return validation
    finally:
        staging_path.unlink(missing_ok=True)


def reseal_existing(
    manifest_path: Path,
    audit_path: Path,
    *,
    verify_cache_file: bool = True,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve(strict=True)
    audit_path = audit_path.resolve(strict=True)
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    if report.get("protocol") != PROTOCOL:
        raise ValueError("Cannot reseal a different DADS protocol")
    frame = pd.read_csv(manifest_path, low_memory=False)
    report["passed"] = True
    report["resealed_at_utc"] = datetime.now(timezone.utc).isoformat()
    construction = report.setdefault("construction", {})
    construction["allocation_unit"] = (
        "per_label_effective_source_and_unique_pcm16_window"
    )
    construction["allocation_actual"] = _allocation_counts(frame)
    construction["max_allowed_abs_ratio_error"] = (
        MAX_ALLOCATION_RATIO_ERROR
    )
    construction["max_abs_ratio_error"] = _allocation_max_ratio_error(
        construction["allocation_actual"], report["split_ratios"]
    )
    report.setdefault("overlap_checks", {})[
        "split_component"
    ] = _pairwise_overlap(frame, "split_component")
    if "training_started" in report:
        report["training_started_at_generation"] = bool(
            report.pop("training_started")
        )
    report["reproducibility"] = _reproducibility_identity()
    temporary_final = audit_path.with_name(f".{audit_path.name}.resealed")
    if temporary_final.exists():
        raise FileExistsError(
            f"Refusing to overwrite stale reseal file: {temporary_final}"
        )
    validation = _seal_audit(
        manifest_path,
        temporary_final,
        report,
        verify_cache_file=verify_cache_file,
    )
    temporary_final.replace(audit_path)
    return validation


def prepare(
    root: Path,
    source_registry_path: Path,
    dedup_audit_path: Path,
    output_dir: Path,
    *,
    split_ratios: dict[str, float],
    seed: int,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    source_registry_path = source_registry_path.resolve(strict=True)
    dedup_audit_path = dedup_audit_path.resolve(strict=True)
    output_dir = output_dir.resolve(strict=False)
    manifest_path = output_dir / "manifest.csv"
    output_sources_path = output_dir / "source_registry.csv"
    duplicates_path = output_dir / "duplicate_windows.csv"
    conflicts_path = output_dir / "cross_label_conflicts.csv"
    cache_path = output_dir / "cache" / "native_half_second_audio.npy"
    audit_path = output_dir / "audit.json"
    occupied = [
        path
        for path in (
            manifest_path,
            output_sources_path,
            duplicates_path,
            conflicts_path,
            cache_path,
            audit_path,
        )
        if path.exists()
    ]
    if occupied:
        raise FileExistsError(
            "Refusing to overwrite leakage-fixed DADS outputs: "
            + ", ".join(path.as_posix() for path in occupied)
        )

    sources, dedup_audit = _load_sources(
        root, source_registry_path, dedup_audit_path
    )
    candidates, extraction = _extract_native_windows(root, sources, cache_path)
    clean, conflicts, conflict_groups = _drop_cross_label_content(candidates)
    connected = _attach_components(clean)
    assigned, output_sources, allocation_plan = _assign_components(
        connected, split_ratios, int(seed)
    )
    canonical, duplicates = _canonicalize_content(assigned)
    role_map = {
        "train": "training",
        "val": "internal_model_selection",
        "test": "consumed_internal_development_test",
    }
    canonical["manifest_protocol"] = PROTOCOL
    canonical["evaluation_role"] = canonical["split"].map(role_map)
    canonical["consumption_status"] = "underlying_dads_pool_previously_consumed"
    split_rank = {split: index for index, split in enumerate(SPLIT_ORDER)}
    canonical["_split_rank"] = canonical["split"].map(split_rank)
    canonical = (
        canonical.sort_values(
            [
                "_split_rank",
                "label",
                "source_id",
                "segment_index",
            ],
            kind="stable",
        )
        .drop(columns="_split_rank")
        .reset_index(drop=True)
    )
    selected_counts = canonical["source_id"].value_counts()
    output_sources["selected_windows"] = (
        output_sources["source_id"].map(selected_counts).fillna(0).astype(int)
    )
    output_sources = output_sources.sort_values(
        ["split", "label", "source_id"], kind="stable"
    ).reset_index(drop=True)
    duplicates = duplicates.sort_values(
        ["segment_pcm16_sha256", "source_id", "segment_index"], kind="stable"
    ).reset_index(drop=True)
    conflicts = conflicts.sort_values(
        ["segment_pcm16_sha256", "label", "source_id", "segment_index"],
        kind="stable",
    ).reset_index(drop=True)
    allocation_actual = _allocation_counts(canonical)
    allocation_error = _allocation_max_ratio_error(
        allocation_actual, split_ratios
    )
    if allocation_error > MAX_ALLOCATION_RATIO_ERROR:
        raise ValueError(
            "Unable to balance effective sources and unique windows within "
            f"the frozen ratio tolerance: {allocation_error:.8f}"
        )

    _atomic_write_csv(manifest_path, canonical)
    _atomic_write_csv(output_sources_path, output_sources)
    _atomic_write_csv(duplicates_path, duplicates)
    _atomic_write_csv(conflicts_path, conflicts)
    cache_sha256 = file_sha256(cache_path)
    output_manifest_sha256 = file_sha256(manifest_path)
    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "sample_rate": TARGET_SAMPLE_RATE,
        "target_samples": TARGET_SAMPLES,
        "clip_seconds": TARGET_SAMPLES / TARGET_SAMPLE_RATE,
        "split_ratios": {
            split: float(split_ratios[split]) for split in SPLIT_ORDER
        },
        "input": {
            "source_registry": {
                "path": _display_path(source_registry_path, root),
                "sha256": file_sha256(source_registry_path),
                "rows": int(len(sources)),
            },
            "dedup_v2_audit": {
                "path": _display_path(dedup_audit_path, root),
                "sha256": file_sha256(dedup_audit_path),
                "protocol": dedup_audit.get("protocol"),
            },
        },
        "construction": {
            **extraction,
            "allocation_unit": (
                "per_label_effective_source_and_unique_pcm16_window"
            ),
            "allocation_method": allocation_plan["method"],
            "allocation_targets": {
                "sources": allocation_plan["sources"],
                "unique_windows": allocation_plan["unique_windows"],
            },
            "allocation_actual": allocation_actual,
            "max_allowed_abs_ratio_error": MAX_ALLOCATION_RATIO_ERROR,
            "max_abs_ratio_error": allocation_error,
            "local_balance_moves": allocation_plan[
                "local_balance_moves"
            ],
            "local_balance_passes": allocation_plan[
                "local_balance_passes"
            ],
            "usable_windows_before_label_conflict_filter": int(len(candidates)),
            "cross_label_conflict_rows_removed": int(len(conflicts)),
            "cross_label_conflict_groups": conflict_groups,
            "components": int(assigned["split_component"].nunique()),
            "sources": _counts(output_sources),
            "duplicate_windows_removed": int(len(assigned) - len(canonical)),
            "final_windows": _counts(canonical),
            "moved_sources_from_historical_split": int(
                output_sources["moved_from_historical_split"].sum()
            ),
        },
        "output": {
            "manifest": {
                "path": _display_path(manifest_path, root),
                "sha256": output_manifest_sha256,
                "rows": int(len(canonical)),
            },
            "source_registry": {
                "path": _display_path(output_sources_path, root),
                "sha256": file_sha256(output_sources_path),
                "rows": int(len(output_sources)),
            },
            "duplicate_windows": {
                "path": _display_path(duplicates_path, root),
                "sha256": file_sha256(duplicates_path),
                "rows": int(len(duplicates)),
            },
            "cross_label_conflicts": {
                "path": _display_path(conflicts_path, root),
                "sha256": file_sha256(conflicts_path),
                "rows": int(len(conflicts)),
            },
            "cache": {
                "path": _display_path(cache_path, root),
                "sha256": cache_sha256,
                "shape": [int(extraction["planned_cache_rows"]), TARGET_SAMPLES],
                "referenced_rows_before_content_filter": int(
                    extraction["written_cache_rows"]
                ),
            },
        },
        "overlap_checks": {
            "source_id": _pairwise_overlap(canonical, "source_id"),
            "recording_group": _pairwise_overlap(canonical, "recording_group"),
            "raw_audio_sha256": _pairwise_overlap(
                canonical, "raw_audio_sha256"
            ),
            "segment_float32_sha256": _pairwise_overlap(
                canonical, "segment_float32_sha256"
            ),
            "segment_pcm16_sha256": _pairwise_overlap(
                canonical, "segment_pcm16_sha256"
            ),
            "split_component": _pairwise_overlap(
                canonical, "split_component"
            ),
        },
        "historical_manifest_modified": False,
        "historical_models_modified": False,
        "training_started_at_generation": False,
        "fresh_final_holdout": False,
        "limitations": [
            "DADS lacks authoritative session, site, device and UAV-model provenance.",
            "The underlying DADS pool has already been consumed by historical development.",
            "The test split is an internal development test, not a fresh final holdout.",
            "Only exact raw-audio and exact/PCM16-equivalent final-window identity is guarded.",
            "Near-duplicate acoustic similarity requires a separate provenance or embedding audit.",
        ],
        "locked_datasets_read": [],
        "reproducibility": _reproducibility_identity(),
    }
    validation = _seal_audit(
        manifest_path,
        audit_path,
        report,
        verify_cache_file=True,
    )
    print(
        json.dumps(
            {
                "passed": validation["passed"],
                "protocol": PROTOCOL,
                "counts": validation["counts"],
                "overlap_checks": report["overlap_checks"],
                "fresh_final_holdout": False,
                "training_started_at_generation": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build or validate a native 0.5-second DADS manifest that groups "
            "raw recordings and final model-input hashes before splitting."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--root", type=Path, default=Path("."))
    prepare_parser.add_argument(
        "--source-registry",
        type=Path,
        default=Path("artifacts/dads_dedup_v2/source_registry.csv"),
    )
    prepare_parser.add_argument(
        "--dedup-audit",
        type=Path,
        default=Path("artifacts/dads_dedup_v2/audit.json"),
    )
    prepare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/g7_leakage_fixed_v2/data"),
    )
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--train-ratio", type=float, default=0.70)
    prepare_parser.add_argument("--val-ratio", type=float, default=0.15)
    prepare_parser.add_argument("--test-ratio", type=float, default=0.15)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/g7_leakage_fixed_v2/data/manifest.csv"),
    )
    validate_parser.add_argument(
        "--audit",
        type=Path,
        default=Path("artifacts/g7_leakage_fixed_v2/data/audit.json"),
    )
    validate_parser.add_argument("--verify-cache-file", action="store_true")
    reseal_parser = subparsers.add_parser("reseal")
    reseal_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/g7_leakage_fixed_v2/data/manifest.csv"),
    )
    reseal_parser.add_argument(
        "--audit",
        type=Path,
        default=Path("artifacts/g7_leakage_fixed_v2/data/audit.json"),
    )
    args = parser.parse_args()

    if args.command == "prepare":
        root = args.root.resolve(strict=True)
        source_registry = (
            args.source_registry
            if args.source_registry.is_absolute()
            else root / args.source_registry
        )
        dedup_audit = (
            args.dedup_audit
            if args.dedup_audit.is_absolute()
            else root / args.dedup_audit
        )
        output_dir = (
            args.output_dir
            if args.output_dir.is_absolute()
            else root / args.output_dir
        )
        prepare(
            root,
            source_registry,
            dedup_audit,
            output_dir,
            split_ratios={
                "train": args.train_ratio,
                "val": args.val_ratio,
                "test": args.test_ratio,
            },
            seed=args.seed,
        )
    elif args.command == "validate":
        result = validate_manifest(
            args.manifest,
            args.audit,
            verify_cache_file=bool(args.verify_cache_file),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        result = reseal_existing(
            args.manifest,
            args.audit,
            verify_cache_file=True,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
