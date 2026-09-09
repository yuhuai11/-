from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .audio import decode_wav_bytes, ensure_sample_rate, peak_normalize, to_fixed_length


def external_groups(dataset: str, path: Path, label: int) -> tuple[str, str]:
    stem = path.stem
    if dataset == "unseen":
        source = stem.split("_paperdist_", 1)[0]
        condition_match = re.search(r"_(T\d{2})_", stem)
        condition = condition_match.group(1) if condition_match else "unknown"
        return source, condition
    lowered = stem.lower()
    if "helicopter" in lowered:
        condition = "helicopter"
    elif "traffic" in lowered:
        condition = "traffic"
    else:
        condition = "other"
    return ("drone" if label else "background"), condition


def build_external_manifest(
    dataset: str,
    root: Path,
    labels: dict[str, int],
    *,
    hash_files: bool,
) -> pd.DataFrame:
    rows = []
    for folder, label in labels.items():
        class_dir = root / folder
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing external class directory: {class_dir}")
        for path in sorted(class_dir.rglob("*.wav")):
            wav_bytes = path.read_bytes()
            audio, sample_rate = decode_wav_bytes(wav_bytes)
            source_group, condition = external_groups(dataset, path, int(label))
            rows.append(
                {
                    "dataset": dataset,
                    "path": path.resolve().as_posix(),
                    "filename": path.name,
                    "label": int(label),
                    "source_group": source_group,
                    "condition": condition,
                    "sample_rate": int(sample_rate),
                    "samples": int(audio.size),
                    "duration_seconds": float(audio.size / sample_rate),
                    "sha256": hashlib.sha256(wav_bytes).hexdigest() if hash_files else "",
                }
            )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise FileNotFoundError(f"No WAV files found under {root}")
    return manifest.sort_values(["label", "path"]).reset_index(drop=True)


class ExternalAudioDataset:
    def __init__(self, manifest: str | Path, sample_rate: int, clip_seconds: float) -> None:
        self.rows = pd.read_csv(manifest)
        self.sample_rate = int(sample_rate)
        self.target_samples = int(sample_rate * clip_seconds)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        path = Path(str(row["path"]))
        audio, original_rate = decode_wav_bytes(path.read_bytes())
        audio = ensure_sample_rate(audio, original_rate, self.sample_rate)
        audio = peak_normalize(to_fixed_length(audio, self.target_samples, random_crop=False))

        import torch

        return (
            torch.from_numpy(audio.astype(np.float32, copy=False)),
            torch.tensor(float(row["label"]), dtype=torch.float32),
            int(index),
        )
