from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ensure_dirs
from .data_firewall import (
    audit_csv_rows,
    file_sha256 as sha256,
    reject_locked_path,
)


EXPECTED_BEATS_SHA256 = "d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34"


def select_source_segments(frame: pd.DataFrame, maximum: int = 3) -> pd.DataFrame:
    required = {"split", "source_path", "segment_index", "cache_path", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"DADS manifest lacks columns: {sorted(missing)}")
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    selected = []
    for (_, _), group in frame.groupby(["split", "source_path"], sort=True):
        group = group.sort_values(
            ["segment_index", "start_sample", "cache_path"], kind="stable"
        )
        if len(group) <= maximum:
            chosen = group
        else:
            positions = sorted({0, (len(group) - 1) // 2, len(group) - 1})
            chosen = group.iloc[positions[:maximum]]
        selected.append(chosen)
    output = pd.concat(selected, ignore_index=True)
    counts = output.groupby(["split", "source_path"]).size()
    if int(counts.max()) > maximum:
        raise AssertionError("Source segment cap was not enforced")
    return output.sort_values(
        ["split", "label", "source_path", "segment_index"], kind="stable"
    ).reset_index(drop=True)


def _source_overlap(frame: pd.DataFrame) -> dict[str, int]:
    values = {
        split: set(rows["source_path"].astype(str))
        for split, rows in frame.groupby("split")
    }
    required = {"train", "val", "test"}
    if set(values) != required:
        raise ValueError(f"Expected splits {sorted(required)}, got {sorted(values)}")
    return {
        "train_val": len(values["train"] & values["val"]),
        "train_test": len(values["train"] & values["test"]),
        "val_test": len(values["val"] & values["test"]),
    }


def prepare(
    manifest: Path,
    checkpoint: Path,
    vendor_repo: Path,
    python_dir: Path,
    output_dir: Path,
    maximum_segments: int,
) -> dict[str, Any]:
    reject_locked_path(manifest)
    audit_csv_rows(manifest)
    observed_hash = sha256(checkpoint)
    if observed_hash != EXPECTED_BEATS_SHA256:
        raise ValueError(f"Unexpected BEATs checkpoint SHA256: {observed_hash}")
    frame = pd.read_csv(manifest)
    overlap = _source_overlap(frame)
    if any(overlap.values()):
        raise ValueError(f"DADS source leakage: {overlap}")
    selected = select_source_segments(frame, maximum_segments)
    if not selected["cache_path"].map(lambda value: Path(str(value)).is_file()).all():
        raise FileNotFoundError("A selected DADS cache file is missing")
    ensure_dirs(output_dir)
    selected_path = output_dir / "dads_selected_segments.csv"
    selected.to_csv(selected_path, index=False)
    commit = subprocess.run(
        ["git", "-C", str(vendor_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    torchaudio_files = sorted((python_dir / "torchaudio").rglob("*"))
    audit = {
        "manifest": {"path": manifest.as_posix(), "sha256": sha256(manifest)},
        "selected_manifest": {
            "path": selected_path.as_posix(),
            "sha256": sha256(selected_path),
            "rows": int(len(selected)),
            "sources": int(selected["source_path"].nunique()),
            "split_rows": {str(k): int(v) for k, v in selected["split"].value_counts().items()},
            "split_sources": {
                str(k): int(v)
                for k, v in selected.groupby("split")["source_path"].nunique().items()
            },
        },
        "source_overlap": overlap,
        "maximum_segments_per_source": maximum_segments,
        "beats": {
            "checkpoint": checkpoint.as_posix(),
            "checkpoint_sha256": observed_hash,
            "vendor_repo": vendor_repo.as_posix(),
            "vendor_commit": commit,
            "source_sha256": {
                name: sha256(vendor_repo / "beats" / name)
                for name in ("BEATs.py", "backbone.py", "modules.py")
            },
        },
        "isolated_python_dir": python_dir.as_posix(),
        "torchaudio_files": len([path for path in torchaudio_files if path.is_file()]),
        "locked_datasets_read": [],
    }
    (output_dir / "dependency_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the source-isolated BEATs probe")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "archive/historical_models/baselines/"
            "artifacts_15000/manifests/dads_balanced_15000_seed42.csv"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/p1_beats_probe/BEATs_iter3_plus_AS2M.pt"),
    )
    parser.add_argument(
        "--vendor-repo",
        type=Path,
        default=Path("artifacts/p1_beats_probe/vendor/unilm"),
    )
    parser.add_argument(
        "--python-dir", type=Path, default=Path("artifacts/p1_beats_probe/python")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/p1_beats_probe")
    )
    parser.add_argument("--maximum-segments", type=int, default=3)
    args = parser.parse_args()
    result = prepare(
        args.manifest,
        args.checkpoint,
        args.vendor_repo,
        args.python_dir,
        args.output_dir,
        args.maximum_segments,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
