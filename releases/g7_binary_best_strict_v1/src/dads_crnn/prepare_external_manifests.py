from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .config import ensure_dirs, load_config
from .external_data import build_external_manifest


def dads_audio_hashes(parquet_dir: Path) -> set[str]:
    import pyarrow.parquet as pq

    hashes: set[str] = set()
    for path in sorted(parquet_dir.glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.num_row_groups):
            audios = parquet.read_row_group(row_group, columns=["audio"]).column("audio").to_pylist()
            for audio in audios:
                wav_bytes = audio.get("bytes")
                if wav_bytes:
                    hashes.add(hashlib.sha256(wav_bytes).hexdigest())
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description="Build immutable manifests for external evaluation sets")
    parser.add_argument("--config", default="configs/external_evaluation.yaml")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--hash-files", action="store_true")
    parser.add_argument("--dads-parquet-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    selected = args.datasets or list(config["datasets"])
    output_dir = Path(config["output_dir"]) / "manifests"
    ensure_dirs(output_dir)
    if args.dads_parquet_dir is not None and not args.hash_files:
        parser.error("--dads-parquet-dir requires --hash-files")
    dads_hashes = dads_audio_hashes(args.dads_parquet_dir) if args.dads_parquet_dir else None
    audit = {
        "hash_files": bool(args.hash_files),
        "dads_overlap_checked": dads_hashes is not None,
        "datasets": {},
    }
    for name in selected:
        dataset_cfg = config["datasets"][name]
        manifest = build_external_manifest(
            name,
            Path(dataset_cfg["root"]),
            {str(key): int(value) for key, value in dataset_cfg["labels"].items()},
            hash_files=args.hash_files,
        )
        path = output_dir / f"{name}_manifest.csv"
        manifest.to_csv(path, index=False)
        audit["datasets"][name] = {
            "manifest": path.as_posix(),
            "samples": int(len(manifest)),
            "label_counts": {str(k): int(v) for k, v in manifest["label"].value_counts().sort_index().items()},
            "sample_rates": sorted(int(value) for value in manifest["sample_rate"].unique()),
            "duration_min": float(manifest["duration_seconds"].min()),
            "duration_max": float(manifest["duration_seconds"].max()),
            "duplicate_hashes": (
                int(manifest.loc[manifest["sha256"] != "", "sha256"].duplicated().sum())
                if args.hash_files
                else None
            ),
            "exact_dads_overlaps": (
                int(manifest["sha256"].isin(dads_hashes).sum()) if dads_hashes is not None else None
            ),
        }
        print(f"{name}: wrote {len(manifest)} rows to {path}")
    (output_dir / "manifest_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
