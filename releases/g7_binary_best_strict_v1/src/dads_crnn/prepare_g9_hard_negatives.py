from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .audio import peak_normalize, to_fixed_length
from .config import load_config
from .data_firewall import (
    LOCKED_COMPACT_TOKENS,
    audit_csv_rows,
    reject_locked_path as firewall_reject_locked_path,
    reject_locked_value as firewall_reject_locked_value,
)
from .prepare_external_manifests import dads_audio_hashes


PROTOCOL = "g9_source_isolated_hard_negative_v1"
SOURCE_FAMILY = "github_U16_ESC50"
FIXED_CLASS_MAP = {
    35: "washing_machine",
    36: "vacuum_cleaner",
    40: "helicopter",
    41: "chainsaw",
    44: "engine",
    47: "airplane",
}
FIXED_PROVENANCE = {
    "dataset": "ESC-50-derived U16",
    "upstream": "https://github.com/karolpiczak/ESC-50",
    "license": "CC BY-NC",
    "commercial_use_allowed": False,
    "attribution_file_present": False,
    "publication_ready": False,
}
ESC50_NAME = re.compile(
    r"^(?P<fold>[1-5])-(?P<clip_id>[0-9]+)-(?P<take>[A-Za-z])-"
    r"(?P<class_id>[0-9]{2})(?P<segment_index>[0-4])\.wav$",
    flags=re.IGNORECASE,
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ParsedEsc50Name:
    fold: int
    clip_id: str
    take: str
    class_id: int
    class_name: str
    recording_group: str
    source_group: str
    segment_index: int


@dataclass(frozen=True)
class CandidateClip:
    path: Path
    parsed: ParsedEsc50Name
    sha256: str
    sample_rate: int
    samples: int
    rms_dbfs: float
    audio: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_locked_path(
    path: Path,
    tokens: Iterable[str] = LOCKED_COMPACT_TOKENS,
) -> None:
    firewall_reject_locked_path(path, context="G9 input", tokens=tokens)


def _reject_locked_value(
    value: object,
    *,
    context: str,
    tokens: Iterable[str] = LOCKED_COMPACT_TOKENS,
) -> None:
    firewall_reject_locked_value(value, context=context, tokens=tokens)


def parse_esc50_filename(
    path: Path,
    class_map: Mapping[int, str],
) -> ParsedEsc50Name:
    match = ESC50_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Malformed U16/ESC-50 filename: {path.name}")
    class_id = int(match.group("class_id"))
    if class_id not in class_map:
        raise ValueError(f"ESC-50 class {class_id} is not in the frozen G9 class set")
    segment_index = int(match.group("segment_index"))
    fold = int(match.group("fold"))
    clip_id = match.group("clip_id")
    take = match.group("take").upper()
    recording_group = f"{fold}-{clip_id}-{take}-{match.group('class_id')}"
    source_group = f"{fold}-{clip_id}"
    return ParsedEsc50Name(
        fold=fold,
        clip_id=clip_id,
        take=take,
        class_id=class_id,
        class_name=str(class_map[class_id]),
        recording_group=recording_group,
        source_group=source_group,
        segment_index=segment_index,
    )


def deterministic_group_split(
    groups: Mapping[str, int],
    *,
    class_name: str,
    train_folds: Iterable[int],
    guard_folds: Iterable[int],
) -> dict[str, str]:
    normalized = {str(group): int(fold) for group, fold in groups.items()}
    if not normalized or any(not group for group in normalized):
        raise ValueError("Every hard-negative source group must be non-empty")
    frozen_train_folds = {int(value) for value in train_folds}
    frozen_guard_folds = {int(value) for value in guard_folds}
    if frozen_train_folds != {1, 2, 3, 4} or frozen_guard_folds != {5}:
        raise ValueError("G9 requires official ESC-50 folds 1-4=train and fold 5=guard")
    if frozen_train_folds & frozen_guard_folds:
        raise ValueError("Train and guard ESC-50 folds overlap")
    split = {}
    for group, fold in sorted(normalized.items()):
        if fold in frozen_train_folds:
            split[group] = "train"
        elif fold in frozen_guard_folds:
            split[group] = "guard"
        else:
            raise ValueError(f"Unexpected ESC-50 fold for {class_name}/{group}: {fold}")
    if not set(split.values()) == {"train", "guard"}:
        raise ValueError(f"Both train and guard source groups are required for {class_name}")
    return split


def _read_pcm16_mono(path: Path, expected_sample_rate: int) -> tuple[np.ndarray, int]:
    reject_locked_path(path)
    if path.is_symlink():
        raise ValueError(f"Symlinked hard-negative audio is forbidden: {path}")
    with wave.open(str(path), "rb") as wav:
        if wav.getcomptype() != "NONE":
            raise ValueError(f"Compressed WAV is forbidden: {path}")
        if wav.getnchannels() != 1:
            raise ValueError(f"Expected mono WAV: {path}")
        if wav.getsampwidth() != 2:
            raise ValueError(f"Expected PCM16 WAV: {path}")
        if wav.getframerate() != expected_sample_rate:
            raise ValueError(
                f"Expected {expected_sample_rate} Hz WAV, got {wav.getframerate()}: {path}"
            )
        frames = int(wav.getnframes())
        payload = wav.readframes(frames)
    audio = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    if audio.size != frames or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid PCM payload: {path}")
    return audio, expected_sample_rate


def _rms_dbfs(audio: np.ndarray) -> float:
    power = float(np.mean(np.square(audio), dtype=np.float64))
    return 10.0 * math.log10(power + 1e-12)


def _validate_frozen_config(config: dict[str, Any]) -> dict[int, str]:
    source = config.get("source", {})
    configured = {
        int(class_id): str(name)
        for class_id, name in dict(source.get("classes", {})).items()
    }
    if configured != FIXED_CLASS_MAP:
        raise ValueError(
            f"G9 class set is frozen as {FIXED_CLASS_MAP}; configured {configured}"
        )
    split = config.get("split", {})
    if int(split.get("train_recordings_per_class", -1)) != 32:
        raise ValueError("G9 requires exactly 32 train recording groups per class")
    if int(split.get("guard_recordings_per_class", -1)) != 8:
        raise ValueError("G9 requires exactly 8 guard recording groups per class")
    if {int(value) for value in split.get("train_folds", [])} != {1, 2, 3, 4}:
        raise ValueError("G9 requires ESC-50 folds 1-4 for training")
    if {int(value) for value in split.get("guard_folds", [])} != {5}:
        raise ValueError("G9 requires ESC-50 fold 5 for the guard")
    if float(source.get("minimum_rms_dbfs_exclusive", math.nan)) != -65.0:
        raise ValueError("G9 requires the frozen RMS rule RMS > -65 dBFS")
    if int(source.get("sample_rate", -1)) != 16000:
        raise ValueError("G9 hard negatives must be 16 kHz")
    if int(source.get("channels", -1)) != 1 or int(source.get("sample_width_bytes", -1)) != 2:
        raise ValueError("G9 hard negatives must be mono PCM16")
    if int(source.get("expected_groups_per_class", -1)) != 40:
        raise ValueError("G9 expects 40 original ESC-50 recordings per class")
    if int(source.get("expected_segments_per_group", -1)) != 5:
        raise ValueError("G9 expects five exported segments per ESC-50 recording")
    if str(source.get("source_family", "")) != SOURCE_FAMILY:
        raise ValueError(f"G9 source_family is frozen as {SOURCE_FAMILY}")
    if float(source.get("clip_seconds", math.nan)) != 1.0:
        raise ValueError("G9 hard-negative cache is frozen at one second")
    if config.get("provenance") != FIXED_PROVENANCE:
        raise ValueError(f"G9 provenance metadata is frozen as {FIXED_PROVENANCE}")
    return configured


def _candidate_inventory(
    source_root: Path,
    config: dict[str, Any],
    class_map: Mapping[int, str],
) -> tuple[list[CandidateClip], dict[str, Any]]:
    reject_locked_path(source_root)
    if source_root.is_symlink():
        raise ValueError(f"Symlinked hard-negative root is forbidden: {source_root}")
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    source_cfg = config["source"]
    sample_rate = int(source_cfg["sample_rate"])
    minimum_rms = float(source_cfg["minimum_rms_dbfs_exclusive"])
    expected_groups = int(source_cfg["expected_groups_per_class"])
    expected_segments = int(source_cfg["expected_segments_per_group"])

    targeted: list[tuple[Path, ParsedEsc50Name]] = []
    ignored_wavs = 0
    for path in sorted(source_root.glob("*.wav")):
        reject_locked_path(path)
        match = ESC50_NAME.fullmatch(path.name)
        if match is None or int(match.group("class_id")) not in class_map:
            ignored_wavs += 1
            continue
        parsed = parse_esc50_filename(path, class_map)
        targeted.append((path, parsed))
    if not targeted:
        raise FileNotFoundError(f"No frozen G9 classes found under {source_root}")

    raw_groups: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    group_folds: dict[tuple[str, str], set[int]] = defaultdict(set)
    group_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    clip_id_folds: dict[str, set[int]] = defaultdict(set)
    for _, parsed in targeted:
        raw_groups[parsed.class_name][parsed.recording_group].add(parsed.segment_index)
        group_folds[(parsed.class_name, parsed.recording_group)].add(parsed.fold)
        group_sources[(parsed.class_name, parsed.recording_group)].add(parsed.source_group)
        clip_id_folds[parsed.clip_id].add(parsed.fold)
    expected_indices = set(range(expected_segments))
    for class_name in sorted(class_map.values()):
        groups = raw_groups.get(class_name, {})
        if len(groups) != expected_groups:
            raise ValueError(
                f"Expected {expected_groups} raw groups for {class_name}, got {len(groups)}"
            )
        malformed = {
            group: sorted(indices)
            for group, indices in groups.items()
            if indices != expected_indices
        }
        if malformed:
            raise ValueError(f"Incomplete ESC-50 segment groups for {class_name}: {malformed}")
    if any(len(values) != 1 for values in group_folds.values()):
        raise ValueError("An ESC-50 recording group spans multiple official folds")
    if any(len(values) != 1 for values in group_sources.values()):
        raise ValueError("An ESC-50 recording group has inconsistent source provenance")
    leaked_clip_ids = sorted(
        clip_id for clip_id, folds in clip_id_folds.items() if len(folds) != 1
    )
    if leaked_clip_ids:
        raise ValueError(
            f"ESC-50 clip ID appears in multiple official folds: {leaked_clip_ids[:3]}"
        )
    for class_name in sorted(class_map.values()):
        per_fold = Counter(
            next(iter(group_folds[(class_name, recording)]))
            for recording in raw_groups[class_name]
        )
        if per_fold != Counter({1: 8, 2: 8, 3: 8, 4: 8, 5: 8}):
            raise ValueError(
                f"Expected eight ESC-50 recordings per official fold for {class_name}: "
                f"{dict(per_fold)}"
            )

    kept: list[CandidateClip] = []
    excluded_silent: Counter[str] = Counter()
    raw_counts: Counter[str] = Counter()
    for path, parsed in targeted:
        if path.is_symlink():
            raise ValueError(f"Symlinked hard-negative audio is forbidden: {path}")
        raw_counts[parsed.class_name] += 1
        audio, observed_rate = _read_pcm16_mono(path, sample_rate)
        rms = _rms_dbfs(audio)
        if not rms > minimum_rms:
            excluded_silent[parsed.class_name] += 1
            continue
        kept.append(
            CandidateClip(
                path=path.resolve(strict=True),
                parsed=parsed,
                sha256=sha256(path),
                sample_rate=observed_rate,
                samples=int(audio.size),
                rms_dbfs=float(rms),
                audio=audio,
            )
        )

    selected_hashes = [clip.sha256 for clip in kept]
    duplicates = [value for value, count in Counter(selected_hashes).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate selected hard-negative SHA256 values: {duplicates[:3]}")
    selected_paths = [clip.path.as_posix() for clip in kept]
    if len(selected_paths) != len(set(selected_paths)):
        raise ValueError("Duplicate canonical hard-negative paths")

    kept_groups: dict[str, set[str]] = defaultdict(set)
    kept_sources: dict[str, set[str]] = defaultdict(set)
    kept_counts: Counter[str] = Counter()
    for clip in kept:
        kept_groups[clip.parsed.class_name].add(clip.parsed.recording_group)
        kept_sources[clip.parsed.class_name].add(clip.parsed.source_group)
        kept_counts[clip.parsed.class_name] += 1
    for class_name in sorted(class_map.values()):
        if len(kept_groups[class_name]) != expected_groups:
            raise ValueError(f"RMS filtering removed a complete source group for {class_name}")

    inventory = {
        "source_root": source_root.resolve(strict=True).as_posix(),
        "ignored_wav_files": int(ignored_wavs),
        "raw_targeted_files": int(len(targeted)),
        "selected_files": int(len(kept)),
        "selected_unique_sha256": int(len(set(selected_hashes))),
        "symlinks": 0,
        "minimum_rms_dbfs_exclusive": minimum_rms,
        "raw_class_counts": {name: int(raw_counts[name]) for name in sorted(raw_counts)},
        "selected_class_counts": {
            name: int(kept_counts[name]) for name in sorted(kept_counts)
        },
        "excluded_silent_class_counts": {
            name: int(excluded_silent[name]) for name in sorted(raw_counts)
        },
        "source_groups_per_class": {
            name: int(len(kept_sources[name])) for name in sorted(kept_sources)
        },
        "recording_groups_per_class": {
            name: int(len(kept_groups[name])) for name in sorted(kept_groups)
        },
        "sample_rates": sorted({int(clip.sample_rate) for clip in kept}),
        "sample_count_min": int(min(clip.samples for clip in kept)),
        "sample_count_max": int(max(clip.samples for clip in kept)),
        "rms_dbfs_min": float(min(clip.rms_dbfs for clip in kept)),
        "rms_dbfs_max": float(max(clip.rms_dbfs for clip in kept)),
    }
    return kept, inventory


def _read_csv(
    path: Path,
    *,
    required: set[str],
    locked_columns: Iterable[str],
) -> pd.DataFrame:
    reject_locked_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    audit_csv_rows(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if frame.empty:
        raise ValueError(f"Manifest is empty: {path}")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    for column in locked_columns:
        if column not in frame.columns:
            continue
        for value in frame[column]:
            _reject_locked_value(value, context=f"{path}:{column}")
    return frame


def _resolve_external_source(value: str, repo_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    reject_locked_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve(strict=True)


def _val_ood_inventory(
    tune_manifest: Path,
    holdout_manifest: Path,
    external_repo_root: Path,
) -> dict[str, Any]:
    reject_locked_path(external_repo_root)
    frames = {}
    for split, path in (("tune", tune_manifest), ("holdout", holdout_manifest)):
        frame = _read_csv(
            path,
            required={
                "dataset",
                "path",
                "label",
                "sha256",
                "source_group",
                "uav_source",
                "background_source",
                "ood_split",
            },
            locked_columns=(
                "dataset",
                "path",
                "source_group",
                "uav_source",
                "background_source",
            ),
        )
        if set(frame["dataset"]) != {"val_ood"} or set(frame["ood_split"]) != {split}:
            raise ValueError(f"Expected only val_ood/{split} rows in {path}")
        if not frame["sha256"].str.fullmatch(SHA256_PATTERN).all():
            raise ValueError(f"Malformed val_ood SHA256 in {path}")
        if frame["sha256"].duplicated().any():
            raise ValueError(f"Duplicate val_ood SHA256 in {path}")
        labels = pd.to_numeric(frame["label"], errors="raise")
        if not labels.isin([0, 1]).all():
            raise ValueError(f"Non-binary val_ood label in {path}")
        for row in frame.itertuples(index=False):
            sample_path = Path(row.path)
            reject_locked_path(sample_path)
            if not sample_path.is_file():
                raise FileNotFoundError(sample_path)
            if sha256(sample_path) != row.sha256:
                raise ValueError(f"val_ood manifest SHA mismatch: {sample_path}")
        frames[split] = frame

    tune_hashes = set(frames["tune"]["sha256"])
    holdout_hashes = set(frames["holdout"]["sha256"])
    if tune_hashes & holdout_hashes:
        raise ValueError("val_ood tune/holdout sample SHA overlap")
    tune_groups = set(frames["tune"]["source_group"])
    holdout_groups = set(frames["holdout"]["source_group"])
    if tune_groups & holdout_groups:
        raise ValueError("val_ood tune/holdout source-group overlap")

    canonical_paths: set[str] = set()
    source_values: set[str] = set()
    raw_source_paths: set[Path] = set()
    for frame in frames.values():
        for row in frame.itertuples(index=False):
            sample_path = Path(row.path).resolve(strict=True)
            canonical_paths.add(sample_path.as_posix())
            source_values.add(sample_path.name)
            for value in (row.uav_source, row.background_source):
                if not value:
                    continue
                source_values.add(str(value))
                source_values.add(Path(value).name)
                raw_source_paths.add(_resolve_external_source(str(value), external_repo_root))
    raw_hashes = {sha256(path) for path in raw_source_paths}
    return {
        "frames": frames,
        "sample_hashes": tune_hashes | holdout_hashes,
        "source_groups": tune_groups | holdout_groups,
        "canonical_paths": canonical_paths,
        "source_values": source_values,
        "raw_source_paths": {path.as_posix() for path in raw_source_paths},
        "raw_source_hashes": raw_hashes,
        "audit": {
            "tune": {
                "path": tune_manifest.as_posix(),
                "sha256": sha256(tune_manifest),
                "rows": int(len(frames["tune"])),
                "source_groups": int(len(tune_groups)),
            },
            "holdout": {
                "path": holdout_manifest.as_posix(),
                "sha256": sha256(holdout_manifest),
                "rows": int(len(frames["holdout"])),
                "source_groups": int(len(holdout_groups)),
            },
            "raw_source_files": int(len(raw_source_paths)),
            "raw_source_unique_sha256": int(len(raw_hashes)),
            "tune_holdout_sha256_overlap": 0,
            "tune_holdout_source_group_overlap": 0,
        },
    }


def _dads_inventory(dads_manifest: Path, parquet_dir: Path) -> dict[str, Any]:
    frame = _read_csv(
        dads_manifest,
        required={
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
        },
        locked_columns=("parquet_file", "source_path", "cache_path"),
    )
    if set(frame["split"]) != {"train", "val", "test"}:
        raise ValueError("DADS manifest must contain exactly train/val/test splits")
    if not set(frame["label"]).issubset({"0", "1"}):
        raise ValueError("DADS manifest labels must be binary")
    sources = {
        split: set(rows["source_path"])
        for split, rows in frame.groupby("split", sort=True)
    }
    overlap = {
        "train_val": len(sources["train"] & sources["val"]),
        "train_test": len(sources["train"] & sources["test"]),
        "val_test": len(sources["val"] & sources["test"]),
    }
    if any(overlap.values()):
        raise ValueError(f"DADS source leakage: {overlap}")

    reject_locked_path(parquet_dir)
    if not parquet_dir.is_dir():
        raise FileNotFoundError(parquet_dir)
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No DADS parquet files under {parquet_dir}")
    for path in parquet_files:
        reject_locked_path(path)
        if path.is_symlink():
            raise ValueError(f"Symlinked DADS parquet is forbidden: {path}")
    print("Hashing allowed DADS parquet audio for G9 isolation audit...", flush=True)
    hashes = dads_audio_hashes(parquet_dir)
    if not hashes:
        raise ValueError("DADS parquet hash inventory is empty")
    source_values = set(frame["source_path"])
    source_values.update(Path(value).name for value in frame["source_path"] if value)
    return {
        "frame": frame,
        "hashes": hashes,
        "source_values": source_values,
        "audit": {
            "manifest": {
                "path": dads_manifest.as_posix(),
                "sha256": sha256(dads_manifest),
                "rows": int(len(frame)),
            },
            "parquet_dir": parquet_dir.as_posix(),
            "parquet_files": int(len(parquet_files)),
            "raw_audio_unique_sha256": int(len(hashes)),
            "source_overlap_between_splits": overlap,
        },
    }


def _identity_variants(path: Path) -> set[str]:
    return {path.as_posix(), path.resolve(strict=True).as_posix(), path.name}


def _tree_sha256(entries: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for relative, value in sorted(entries):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.10g")


def _output_entry(final_path: Path, staged_path: Path, rows: int) -> dict[str, Any]:
    return {
        "path": final_path.as_posix(),
        "sha256": sha256(staged_path),
        "rows": int(rows),
    }


def _build_outputs(
    *,
    stage: Path,
    output_dir: Path,
    clips: list[CandidateClip],
    group_splits: dict[tuple[str, str], str],
    dads_frame: pd.DataFrame,
    target_samples: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    cache_entries: list[tuple[str, str]] = []
    for clip in sorted(
        clips,
        key=lambda item: (
            item.parsed.class_name,
            item.parsed.recording_group,
            item.parsed.segment_index,
            item.path.as_posix(),
        ),
    ):
        hn_split = group_splits[(clip.parsed.class_name, clip.parsed.source_group)]
        relative_cache = (
            Path("cache")
            / hn_split
            / clip.parsed.class_name
            / f"{clip.parsed.recording_group}_seg{clip.parsed.segment_index}.npy"
        )
        staged_cache = stage / relative_cache
        staged_cache.parent.mkdir(parents=True, exist_ok=True)
        cached = peak_normalize(to_fixed_length(clip.audio, target_samples, random_crop=False))
        if cached.shape != (target_samples,) or cached.dtype != np.float32:
            cached = np.asarray(cached, dtype=np.float32)
        if cached.shape != (target_samples,) or not np.isfinite(cached).all():
            raise ValueError(f"Invalid cached hard-negative waveform: {clip.path}")
        np.save(staged_cache, cached, allow_pickle=False)
        cache_hash = sha256(staged_cache)
        cache_entries.append((relative_cache.as_posix(), cache_hash))
        final_cache = output_dir / relative_cache
        records.append(
            {
                "split": "train" if hn_split == "train" else "guard",
                "label": 0,
                "parquet_file": "",
                "row_group": -1,
                "row_in_group": -1,
                "source_path": clip.path.as_posix(),
                "segment_index": clip.parsed.segment_index,
                "start_sample": 0,
                "end_sample": clip.samples,
                "original_samples": clip.samples,
                "segment_kind": "hard_negative",
                "cache_path": final_cache.as_posix(),
                "dataset_origin": "g9_hard_negative",
                "hard_negative_class": clip.parsed.class_name,
                "source_family": SOURCE_FAMILY,
                "recording_group": clip.parsed.recording_group,
                "source_group": clip.parsed.source_group,
                "esc_fold": clip.parsed.fold,
                "esc50_clip_id": clip.parsed.clip_id,
                "esc50_take": clip.parsed.take,
                "sha256": clip.sha256,
                "cache_sha256": cache_hash,
                "sample_rate": clip.sample_rate,
                "rms_dbfs": clip.rms_dbfs,
                "hn_split": hn_split,
                "background_mix_eligible": "false",
            }
        )
    details = pd.DataFrame.from_records(records)
    train = details.loc[details["hn_split"] == "train"].reset_index(drop=True)
    guard = details.loc[details["hn_split"] == "guard"].reset_index(drop=True)
    train_recordings = set(
        zip(train["hard_negative_class"], train["recording_group"], strict=False)
    )
    guard_recordings = set(
        zip(guard["hard_negative_class"], guard["recording_group"], strict=False)
    )
    train_groups = set(train["source_group"])
    guard_groups = set(guard["source_group"])
    train_clip_ids = set(train["esc50_clip_id"])
    guard_clip_ids = set(guard["esc50_clip_id"])
    if train_recordings & guard_recordings:
        raise ValueError("Hard-negative recording group leaked between train and guard")
    if train_groups & guard_groups:
        raise ValueError("Hard-negative source group leaked between train and guard")
    if train_clip_ids & guard_clip_ids:
        raise ValueError("ESC-50 clip ID leaked between train and guard")
    if set(train["sha256"]) & set(guard["sha256"]):
        raise ValueError("Hard-negative SHA leaked between train and guard")
    train_recordings_by_class = {
        str(name): int(rows["recording_group"].nunique())
        for name, rows in train.groupby("hard_negative_class", sort=True)
    }
    guard_recordings_by_class = {
        str(name): int(rows["recording_group"].nunique())
        for name, rows in guard.groupby("hard_negative_class", sort=True)
    }
    expected_classes = set(FIXED_CLASS_MAP.values())
    if set(train_recordings_by_class) != expected_classes or any(
        value != 32 for value in train_recordings_by_class.values()
    ):
        raise ValueError(
            f"Expected 32 train ESC recordings per class: {train_recordings_by_class}"
        )
    if set(guard_recordings_by_class) != expected_classes or any(
        value != 8 for value in guard_recordings_by_class.values()
    ):
        raise ValueError(
            f"Expected 8 guard ESC recordings per class: {guard_recordings_by_class}"
        )

    dads = dads_frame.copy()
    dads["dataset_origin"] = "dads"
    dads["hard_negative_class"] = ""
    dads["source_family"] = ""
    dads["recording_group"] = ""
    dads["source_group"] = ""
    dads["esc_fold"] = ""
    dads["esc50_clip_id"] = ""
    dads["esc50_take"] = ""
    dads["sha256"] = ""
    dads["cache_sha256"] = ""
    dads["sample_rate"] = ""
    dads["rms_dbfs"] = ""
    dads["hn_split"] = ""
    dads["background_mix_eligible"] = "true"
    columns = list(dads.columns)
    missing_from_details = [column for column in columns if column not in train.columns]
    if missing_from_details:
        raise AssertionError(f"HN rows miss combined columns: {missing_from_details}")
    train_for_combined = train[columns].copy()
    combined = pd.concat([dads, train_for_combined], ignore_index=True)

    manifest_dir = stage / "manifests"
    train_staged = manifest_dir / "hn_train.csv"
    guard_staged = manifest_dir / "hn_guard.csv"
    combined_staged = manifest_dir / "dads_g9_hn_seed42.csv"
    _write_csv(train, train_staged)
    _write_csv(guard, guard_staged)
    _write_csv(combined, combined_staged)
    outputs = {
        "hn_train": _output_entry(
            output_dir / "manifests/hn_train.csv", train_staged, len(train)
        ),
        "hn_guard": _output_entry(
            output_dir / "manifests/hn_guard.csv", guard_staged, len(guard)
        ),
        "combined_manifest": _output_entry(
            output_dir / "manifests/dads_g9_hn_seed42.csv",
            combined_staged,
            len(combined),
        ),
        "cache": {
            "path": (output_dir / "cache").as_posix(),
            "files": int(len(cache_entries)),
            "tree_sha256": _tree_sha256(cache_entries),
        },
    }
    build_audit = {
        "train_rows": int(len(train)),
        "guard_rows": int(len(guard)),
        "train_recording_groups": int(len(train_recordings)),
        "guard_recording_groups": int(len(guard_recordings)),
        "train_recording_groups_by_class": train_recordings_by_class,
        "guard_recording_groups_by_class": guard_recordings_by_class,
        "train_guard_recording_group_overlap": 0,
        "train_source_groups": int(len(train_groups)),
        "guard_source_groups": int(len(guard_groups)),
        "train_guard_source_group_overlap": 0,
        "train_guard_clip_id_overlap": 0,
        "train_guard_sha256_overlap": 0,
        "combined_rows": int(len(combined)),
        "combined_dads_rows": int(len(dads)),
        "combined_hard_negative_rows": int(len(train)),
        "combined_columns": columns,
        "hard_negative_segment_kind": "hard_negative",
        "hard_negative_background_mix_eligible": False,
    }
    return outputs, build_audit


def prepare(config_path: Path, dads_manifest: Path) -> dict[str, Any]:
    reject_locked_path(config_path)
    reject_locked_path(dads_manifest)
    config = load_config(config_path)
    if not isinstance(config, dict):
        raise ValueError("G9 config must contain a mapping")
    class_map = _validate_frozen_config(config)

    source_root = Path(config["source"]["root"])
    parquet_dir = Path(config["dads"]["parquet_dir"])
    tune_manifest = Path(config["val_ood"]["tune_manifest"])
    holdout_manifest = Path(config["val_ood"]["holdout_manifest"])
    external_repo_root = Path(config["val_ood"]["external_repo_root"])
    output_dir = Path(config["output_dir"])
    all_configured_paths = (
        source_root,
        parquet_dir,
        tune_manifest,
        holdout_manifest,
        external_repo_root,
        output_dir,
    )
    for path in all_configured_paths:
        reject_locked_path(path)
    if output_dir.exists():
        raise FileExistsError(f"G9 output already exists; refusing overwrite: {output_dir}")

    clips, candidate_audit = _candidate_inventory(source_root, config, class_map)
    val = _val_ood_inventory(tune_manifest, holdout_manifest, external_repo_root)
    dads = _dads_inventory(dads_manifest, parquet_dir)

    candidate_hashes = {clip.sha256 for clip in clips}
    candidate_paths: set[str] = set()
    candidate_groups = {clip.parsed.source_group for clip in clips}
    for clip in clips:
        candidate_paths.update(_identity_variants(clip.path))
    candidate_vs_dads_paths = candidate_paths & set(dads["source_values"])
    candidate_vs_dads_hashes = candidate_hashes & set(dads["hashes"])
    candidate_vs_val_paths = candidate_paths & (
        set(val["canonical_paths"]) | set(val["source_values"])
    )
    candidate_vs_val_groups = candidate_groups & set(val["source_groups"])
    candidate_vs_val_sample_hashes = candidate_hashes & set(val["sample_hashes"])
    candidate_vs_val_raw_hashes = candidate_hashes & set(val["raw_source_hashes"])
    isolation = {
        "candidate_vs_dads_source_path_overlap": len(candidate_vs_dads_paths),
        "candidate_vs_dads_raw_sha256_overlap": len(candidate_vs_dads_hashes),
        "candidate_vs_val_ood_source_path_overlap": len(candidate_vs_val_paths),
        "candidate_vs_val_ood_source_group_overlap": len(candidate_vs_val_groups),
        "candidate_vs_val_ood_generated_sha256_overlap": len(
            candidate_vs_val_sample_hashes
        ),
        "candidate_vs_val_ood_raw_source_sha256_overlap": len(
            candidate_vs_val_raw_hashes
        ),
    }
    if any(isolation.values()):
        raise ValueError(f"G9 source/hash isolation failed: {isolation}")

    split_cfg = config["split"]
    group_splits: dict[tuple[str, str], str] = {}
    for class_name in sorted(class_map.values()):
        groups = {
            clip.parsed.source_group: clip.parsed.fold
            for clip in clips
            if clip.parsed.class_name == class_name
        }
        assignment = deterministic_group_split(
            groups,
            class_name=class_name,
            train_folds=split_cfg["train_folds"],
            guard_folds=split_cfg["guard_folds"],
        )
        group_splits.update(
            {(class_name, group): value for group, value in assignment.items()}
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent)
    )
    try:
        outputs, build_audit = _build_outputs(
            stage=stage,
            output_dir=output_dir,
            clips=clips,
            group_splits=group_splits,
            dads_frame=dads["frame"],
            target_samples=int(
                round(
                    int(config["source"]["sample_rate"])
                    * float(config["source"]["clip_seconds"])
                )
            ),
        )
        report = {
            "passed": True,
            "protocol": PROTOCOL,
            "frozen_class_map": {str(key): value for key, value in FIXED_CLASS_MAP.items()},
            "split": {
                "algorithm": "esc50_official_fold",
                "train_folds": sorted(int(value) for value in split_cfg["train_folds"]),
                "guard_folds": sorted(int(value) for value in split_cfg["guard_folds"]),
                "train_recordings_per_class": int(
                    split_cfg["train_recordings_per_class"]
                ),
                "guard_recordings_per_class": int(
                    split_cfg["guard_recordings_per_class"]
                ),
            },
            "provenance": dict(FIXED_PROVENANCE),
            "warnings": [
                "Local non-commercial experiment is allowed to proceed, but attribution/license "
                "files must be restored before publication or redistribution.",
                "No fan or lawn-mower class is present in this frozen first-stage pool.",
            ],
            "inputs": {
                "config": {
                    "path": config_path.as_posix(),
                    "sha256": sha256(config_path),
                },
                "candidates": candidate_audit,
                "dads": dads["audit"],
                "val_ood": val["audit"],
            },
            "isolation": isolation,
            "build": build_audit,
            "outputs": outputs,
            "implementation": {
                "path": Path(__file__).resolve().as_posix(),
                "sha256": sha256(Path(__file__)),
            },
            "locked_datasets_read": [],
        }
        audit_staged = stage / "audit.json"
        with audit_staged.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(stage, output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and audit the source-isolated G9 hard-negative pool"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g9_hard_negative_data.yaml"),
    )
    parser.add_argument("--dads-manifest", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.config, args.dads_manifest)


if __name__ == "__main__":
    main()
