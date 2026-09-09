from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .audio import decode_wav_bytes, ensure_sample_rate, peak_normalize, to_fixed_length
from .augmentation import WaveformAugmenter


def _strict_boolean(value: object, *, column: str, row_index: int) -> bool:
    """Parse a CSV boolean without accepting truthy strings or numbers."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(
        f"{column} must contain only true/false booleans; "
        f"row {row_index} has {value!r}"
    )


class DADSDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        *,
        sample_rate: int,
        clip_seconds: float,
        training: bool,
        seed: int,
        parquet_cache_groups: int = 4,
        augmentation: dict | None = None,
    ) -> None:
        # G9 appends sparse string metadata columns to the much larger DADS
        # manifest.  Let pandas infer each column from the complete file so it
        # does not emit chunk-level mixed-type warnings; numeric training
        # columns such as label remain numeric.
        self.rows = pd.read_csv(manifest_path, low_memory=False)
        self.rows = self.rows[self.rows["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No rows found for split={split!r} in {manifest_path}")

        # Legacy manifests do not contain this column and must retain their
        # original behavior: every label-0 row is eligible as a background for
        # positive-example mixing.  G9 manifests opt in explicitly and are
        # parsed fail-closed so values such as "yes", 1, blanks or NaN cannot
        # silently become truthy.
        if "background_mix_eligible" in self.rows.columns:
            self.rows["background_mix_eligible"] = [
                _strict_boolean(
                    value,
                    column="background_mix_eligible",
                    row_index=int(index),
                )
                for index, value in self.rows["background_mix_eligible"].items()
            ]
            background_mask = (self.rows["label"] == 0) & self.rows[
                "background_mix_eligible"
            ]
        else:
            background_mask = self.rows["label"] == 0

        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * clip_seconds)
        self.training = training
        self.rng = np.random.default_rng(seed)
        self.parquet_cache_groups = parquet_cache_groups
        self._group_cache: OrderedDict[tuple[str, int], list[dict]] = OrderedDict()
        self._memmap_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.augmenter = (
            WaveformAugmenter(augmentation, sample_rate, seed) if training and augmentation else None
        )
        self.negative_indices = self.rows.index[background_mask].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        audio = self._load_audio(row)
        audio = peak_normalize(
            to_fixed_length(audio, self.target_samples, random_crop=self.training, rng=self.rng)
        )
        if self.augmenter is not None:
            audio, _ = self.augmenter.apply(
                audio,
                int(row["label"]),
                self._sample_background if self.negative_indices.size else None,
            )

        import torch

        waveform = torch.from_numpy(audio.astype(np.float32, copy=False))
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return waveform, label

    def _sample_background(self) -> np.ndarray:
        output = np.zeros(self.target_samples, dtype=np.float32)
        for _ in range(8):
            index = int(self.rng.choice(self.negative_indices))
            row = self.rows.iloc[index]
            audio = self._load_audio(row)
            output = peak_normalize(to_fixed_length(audio, self.target_samples, random_crop=False))
            if float(np.sqrt(np.mean(np.square(output, dtype=np.float64)))) > 1e-8:
                break
        return output

    def _load_audio(self, row: pd.Series) -> np.ndarray:
        cache_path = row.get("cache_path", "")
        if isinstance(cache_path, str) and cache_path:
            cache_index = row.get("cache_index", "")
            if not pd.isna(cache_index) and str(cache_index).strip() != "":
                cache = self._read_memmap(cache_path)
                audio = np.asarray(cache[int(cache_index)], dtype=np.float32)
            else:
                audio = np.load(cache_path).astype(np.float32, copy=False)

            # Some external-domain caches contain audited 1 s windows while
            # G7-R2/R4 consumes native 0.5 s inputs.  Explicit cache offsets
            # let a manifest expose both non-overlapping halves without
            # rewriting several gigabytes of immutable cache data.  Legacy
            # manifests omit the columns and retain their original behavior.
            cache_start = row.get("cache_start_sample", "")
            cache_end = row.get("cache_end_sample", "")
            has_start = not pd.isna(cache_start) and str(cache_start).strip() != ""
            has_end = not pd.isna(cache_end) and str(cache_end).strip() != ""
            if has_start != has_end:
                raise ValueError(
                    "cache_start_sample and cache_end_sample must be provided together"
                )
            if has_start:
                start = int(cache_start)
                end = int(cache_end)
                if start < 0 or end <= start or end > audio.size:
                    raise ValueError(
                        f"Invalid cache sample range [{start}, {end}) for {cache_path} "
                        f"with {audio.size} samples"
                    )
                audio = audio[start:end]
            return audio

        parquet_file = str(row["parquet_file"])
        row_group = int(row["row_group"])
        row_in_group = int(row["row_in_group"])
        group_rows = self._read_group(parquet_file, row_group)
        audio_dict = group_rows[row_in_group]["audio"]
        audio, original_rate = decode_wav_bytes(audio_dict["bytes"])
        audio = ensure_sample_rate(audio, original_rate, self.sample_rate)
        if "start_sample" in row and "end_sample" in row:
            start_sample = int(row["start_sample"])
            end_sample = int(row["end_sample"])
            audio = audio[start_sample:end_sample]
        return audio

    def _read_memmap(self, cache_path: str) -> np.ndarray:
        if cache_path in self._memmap_cache:
            self._memmap_cache.move_to_end(cache_path)
            return self._memmap_cache[cache_path]
        cache = np.load(cache_path, mmap_mode="r")
        if cache.ndim != 2:
            raise ValueError(f"Segment memmap must be 2-D: {cache_path}")
        self._memmap_cache[cache_path] = cache
        if len(self._memmap_cache) > 3:
            self._memmap_cache.popitem(last=False)
        return cache

    def _read_group(self, parquet_file: str, row_group: int) -> list[dict]:
        key = (parquet_file, row_group)
        if key in self._group_cache:
            self._group_cache.move_to_end(key)
            return self._group_cache[key]

        table = pq.ParquetFile(parquet_file).read_row_group(row_group, columns=["audio"])
        rows = table.to_pylist()
        self._group_cache[key] = rows
        if len(self._group_cache) > self.parquet_cache_groups:
            self._group_cache.popitem(last=False)
        return rows
