from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from .audio import decode_wav_bytes, ensure_sample_rate, peak_normalize, to_fixed_length
from .config import ensure_dirs, load_config


@dataclass(frozen=True)
class AudioRef:
    parquet_file: str
    row_group: int
    row_in_group: int
    label: int
    source_path: str
    original_samples: int


@dataclass(frozen=True)
class RowRef:
    parquet_file: str
    row_group: int
    row_in_group: int
    label: int
    source_path: str
    segment_index: int
    start_sample: int
    end_sample: int
    original_samples: int
    segment_kind: str


def _segments_for_audio(audio: np.ndarray, target_samples: int) -> list[tuple[int, int, str]]:
    if audio.size < target_samples:
        return [(0, int(audio.size), "loop")]
    if audio.size == target_samples:
        return [(0, target_samples, "exact")]

    segments: list[tuple[int, int, str]] = []
    full_segments = audio.size // target_samples
    for segment_index in range(full_segments):
        start = segment_index * target_samples
        segments.append((start, start + target_samples, "full"))
    tail_start = full_segments * target_samples
    if tail_start < audio.size:
        segments.append((tail_start, int(audio.size), "tail_loop"))
    return segments


def _collect_file_refs(parquet_dir: Path, sample_rate: int) -> dict[int, list[AudioRef]]:
    refs: dict[int, list[AudioRef]] = {0: [], 1: []}
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {parquet_dir}")

    skipped_bad = 0
    skipped_silent = 0
    for parquet_path in tqdm(files, desc="Scanning labels"):
        parquet_file = pq.ParquetFile(parquet_path)
        rel_path = parquet_path.as_posix()
        for row_group in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group, columns=["label", "audio"])
            labels = table.column("label").to_pylist()
            audios = table.column("audio").to_pylist()
            for row_in_group, label in enumerate(labels):
                if label in refs:
                    source_path = audios[row_in_group].get("path") or ""
                    try:
                        audio, original_rate = decode_wav_bytes(audios[row_in_group]["bytes"])
                        audio = ensure_sample_rate(audio, original_rate, sample_rate)
                    except Exception:
                        skipped_bad += 1
                        continue
                    if audio.size == 0 or float(np.max(np.abs(audio))) <= 1e-8:
                        skipped_silent += 1
                        continue

                    refs[int(label)].append(
                        AudioRef(
                            rel_path,
                            row_group,
                            row_in_group,
                            int(label),
                            source_path,
                            int(audio.size),
                        )
    )
    print(
        "Collected usable files:",
        {label: len(label_refs) for label, label_refs in refs.items()},
        f"skipped_bad={skipped_bad}",
        f"skipped_silent={skipped_silent}",
    )
    return refs


def _split_refs(
    refs_by_label: dict[int, list[AudioRef]],
    per_class: int | None,
    seed: int,
    split_ratios: dict[str, float],
) -> list[tuple[str, AudioRef]]:
    rng = np.random.default_rng(seed)
    split_names = ["train", "val", "test"]
    ratios = np.array([split_ratios[name] for name in split_names], dtype=np.float64)
    ratios = ratios / ratios.sum()

    rows: list[tuple[str, AudioRef]] = []
    for label, refs in refs_by_label.items():
        if per_class is not None and len(refs) < per_class:
            raise ValueError(f"Class {label} only has {len(refs)} rows, requested {per_class}")
        selected_count = len(refs) if per_class is None else per_class
        indices = rng.permutation(len(refs))[:selected_count]
        selected = [refs[int(i)] for i in indices]

        train_count = int(round(selected_count * ratios[0]))
        val_count = int(round(selected_count * ratios[1]))
        counts = [train_count, val_count, selected_count - train_count - val_count]
        cursor = 0
        for split_name, count in zip(split_names, counts, strict=True):
            for ref in selected[cursor : cursor + count]:
                rows.append((split_name, ref))
            cursor += count

    rows.sort(
        key=lambda item: (
            item[0],
            item[1].label,
            item[1].parquet_file,
            item[1].row_group,
            item[1].row_in_group,
        )
    )
    return rows


def _sampling_limit(value: object) -> int | None:
    if value is None or (isinstance(value, str) and value.strip().lower() == "all"):
        return None
    limit = int(value)
    if limit <= 0:
        raise ValueError("per_class must be a positive integer or 'all'")
    return limit


def _expand_segments(file_rows: list[tuple[str, AudioRef]], target_samples: int) -> list[tuple[str, RowRef]]:
    rows: list[tuple[str, RowRef]] = []
    for split_name, ref in file_rows:
        placeholder = np.empty(ref.original_samples, dtype=np.float32)
        for segment_index, (start_sample, end_sample, segment_kind) in enumerate(
            _segments_for_audio(placeholder, target_samples)
        ):
            rows.append(
                (
                    split_name,
                    RowRef(
                        ref.parquet_file,
                        ref.row_group,
                        ref.row_in_group,
                        ref.label,
                        ref.source_path,
                        segment_index,
                        start_sample,
                        end_sample,
                        ref.original_samples,
                        segment_kind,
                    ),
                )
            )
    rows.sort(
        key=lambda item: (
            item[0],
            item[1].label,
            item[1].parquet_file,
            item[1].row_group,
            item[1].row_in_group,
            item[1].segment_index,
        )
    )
    return rows


def _extract_cache(
    rows: list[tuple[str, RowRef]],
    cache_dir: Path,
    sample_rate: int,
    target_samples: int,
) -> dict[RowRef, str]:
    ensure_dirs(cache_dir)
    by_group: dict[tuple[str, int], list[tuple[str, RowRef]]] = defaultdict(list)
    for split_name, ref in rows:
        by_group[(ref.parquet_file, ref.row_group)].append((split_name, ref))

    cache_paths: dict[RowRef, str] = {}
    for (parquet_file, row_group), group_rows in tqdm(by_group.items(), desc="Extracting audio cache"):
        table = pq.ParquetFile(parquet_file).read_row_group(row_group, columns=["audio"])
        audios = table.column("audio").to_pylist()
        for split_name, ref in group_rows:
            wav_bytes = audios[ref.row_in_group]["bytes"]
            audio, original_rate = decode_wav_bytes(wav_bytes)
            audio = ensure_sample_rate(audio, original_rate, sample_rate)
            audio = audio[ref.start_sample : ref.end_sample]
            audio = peak_normalize(to_fixed_length(audio, target_samples, random_crop=False))

            stem = (
                f"{Path(ref.parquet_file).stem}_rg{ref.row_group:04d}"
                f"_row{ref.row_in_group:04d}_seg{ref.segment_index:04d}"
            )
            out_path = cache_dir / split_name / str(ref.label) / f"{stem}.npy"
            ensure_dirs(out_path.parent)
            np.save(out_path, audio.astype(np.float32, copy=False))
            cache_paths[ref] = out_path.as_posix()
    return cache_paths


def build_manifest(
    config: dict,
    *,
    per_class: int | None,
    extract_audio: bool,
    all_data: bool = False,
) -> Path:
    data_cfg = config["data"]
    parquet_dir = Path(data_cfg["parquet_dir"])
    manifest_dir = Path(data_cfg["manifest_dir"])
    ensure_dirs(manifest_dir)

    configured_limit = "all" if all_data else (per_class if per_class is not None else data_cfg["per_class"])
    sampling_limit = _sampling_limit(configured_limit)
    seed = int(data_cfg["manifest_seed"])
    sample_rate = int(data_cfg["sample_rate"])
    target_samples = int(sample_rate * float(data_cfg["clip_seconds"]))
    refs_by_label = _collect_file_refs(parquet_dir, sample_rate)
    file_rows = _split_refs(refs_by_label, sampling_limit, seed, data_cfg["splits"])
    rows = _expand_segments(file_rows, target_samples)
    print(
        "Selected files:",
        {label: sum(1 for _, ref in file_rows if ref.label == label) for label in sorted(refs_by_label)},
    )
    print(
        "Expanded selected files into segments:",
        {label: sum(1 for _, ref in rows if ref.label == label) for label in sorted(refs_by_label)},
    )

    cache_paths: dict[RowRef, str] = {}
    if extract_audio:
        cache_paths = _extract_cache(
            rows,
            Path(data_cfg["audio_cache_dir"]),
            sample_rate,
            target_samples,
        )

    sampling_name = "all" if sampling_limit is None else f"balanced_{sampling_limit}"
    manifest_path = manifest_dir / f"dads_{sampling_name}_seed{seed}.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        for split_name, ref in rows:
            writer.writerow(
                {
                    "split": split_name,
                    "label": ref.label,
                    "parquet_file": ref.parquet_file,
                    "row_group": ref.row_group,
                    "row_in_group": ref.row_in_group,
                    "source_path": ref.source_path,
                    "segment_index": ref.segment_index,
                    "start_sample": ref.start_sample,
                    "end_sample": ref.end_sample,
                    "original_samples": ref.original_samples,
                    "segment_kind": ref.segment_kind,
                    "cache_path": cache_paths.get(ref, ""),
                }
            )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a file-level-split DADS manifest.")
    parser.add_argument("--config", default="configs/crnn_dads.yaml")
    parser.add_argument("--per-class", type=int, default=None)
    parser.add_argument("--all-data", action="store_true", help="Use every valid original file from both classes.")
    parser.add_argument("--extract-audio", action="store_true", help="Cache selected 1-second clips as .npy files.")
    args = parser.parse_args()

    if args.all_data and args.per_class is not None:
        parser.error("--all-data and --per-class cannot be used together")

    config = load_config(args.config)
    manifest_path = build_manifest(
        config,
        per_class=args.per_class,
        extract_audio=args.extract_audio,
        all_data=args.all_data,
    )
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
