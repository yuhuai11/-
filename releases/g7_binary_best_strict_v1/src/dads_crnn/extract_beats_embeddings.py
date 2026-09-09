from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .audio import decode_wav_bytes, ensure_sample_rate, peak_normalize, to_fixed_length
from .config import ensure_dirs
from .prepare_beats_probe import EXPECTED_BEATS_SHA256, reject_locked_path, sha256
from .train import resolve_device


def load_beats(checkpoint: Path, vendor_beats: Path, device: torch.device):
    if sha256(checkpoint) != EXPECTED_BEATS_SHA256:
        raise ValueError("BEATs checkpoint hash mismatch")
    sys.path.insert(0, vendor_beats.as_posix())
    try:
        from BEATs import BEATs, BEATsConfig
    finally:
        sys.path.pop(0)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = BEATs(BEATsConfig(payload["cfg"]))
    model.load_state_dict(payload["model"], strict=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _waveform(row: pd.Series, target_samples: int = 16000) -> np.ndarray:
    cache_path = row.get("cache_path", "")
    if isinstance(cache_path, str) and cache_path:
        audio = np.load(cache_path).astype(np.float32, copy=False)
    else:
        path = Path(str(row["path"]))
        audio, sample_rate = decode_wav_bytes(path.read_bytes())
        audio = ensure_sample_rate(audio, sample_rate, 16000)
    return peak_normalize(to_fixed_length(audio, target_samples, random_crop=False))


@torch.inference_mode()
def embedding_batch(model, waveforms: torch.Tensor) -> torch.Tensor:
    features, padding_mask = model.extract_features(waveforms, padding_mask=None)
    if padding_mask is None:
        return features.mean(dim=1)
    valid = (~padding_mask).unsqueeze(-1)
    return (features * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def extract(
    manifest: Path,
    split: str | None,
    checkpoint: Path,
    vendor_beats: Path,
    output_dir: Path,
    batch_size: int,
    device_name: str,
    max_samples: int | None,
) -> dict[str, Any]:
    reject_locked_path(manifest)
    frame = pd.read_csv(manifest)
    if split is not None:
        if "split" not in frame.columns:
            raise ValueError("Manifest has no split column")
        frame = frame.loc[frame["split"] == split].reset_index(drop=True)
    if max_samples is not None:
        frame = frame.head(max_samples).copy()
    if frame.empty or "label" not in frame.columns:
        raise ValueError("No labeled rows selected")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = resolve_device(device_name)
    model = load_beats(checkpoint, vendor_beats, device)
    ensure_dirs(output_dir)
    embedding_path = output_dir / "embeddings.npy"
    label_path = output_dir / "labels.npy"
    metadata_path = output_dir / "metadata.csv"
    embeddings = np.lib.format.open_memmap(
        embedding_path, mode="w+", dtype=np.float32, shape=(len(frame), 768)
    )
    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        audio = np.stack([_waveform(frame.iloc[index]) for index in range(start, stop)])
        batch = torch.from_numpy(audio).to(device)
        values = embedding_batch(model, batch).cpu().numpy().astype(np.float32, copy=False)
        if values.shape != (stop - start, 768) or not np.isfinite(values).all():
            raise ValueError(f"Invalid BEATs embedding batch: {values.shape}")
        embeddings[start:stop] = values
        embeddings.flush()
        print(f"embedded {stop}/{len(frame)}", flush=True)
    labels = frame["label"].to_numpy(dtype=np.int64)
    np.save(label_path, labels)
    frame.to_csv(metadata_path, index=False)
    audit = {
        "manifest": manifest.as_posix(),
        "manifest_sha256": sha256(manifest),
        "split": split,
        "samples": int(len(frame)),
        "embedding_dim": 768,
        "checkpoint_sha256": sha256(checkpoint),
        "device": str(device),
        "embeddings": {"path": embedding_path.as_posix(), "sha256": sha256(embedding_path)},
        "labels": {"path": label_path.as_posix(), "sha256": sha256(label_path)},
        "metadata": {"path": metadata_path.as_posix(), "sha256": sha256(metadata_path)},
        "locked_datasets_read": [],
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen BEATs embeddings")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/p1_beats_probe/BEATs_iter3_plus_AS2M.pt"),
    )
    parser.add_argument(
        "--vendor-beats",
        type=Path,
        default=Path("artifacts/p1_beats_probe/vendor/unilm/beats"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    result = extract(
        args.manifest,
        args.split,
        args.checkpoint,
        args.vendor_beats,
        args.output_dir,
        args.batch_size,
        args.device,
        args.max_samples,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
