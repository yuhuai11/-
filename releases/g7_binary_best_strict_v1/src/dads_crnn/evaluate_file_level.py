from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .metrics import file_level_metrics


def _evaluate_split(config: dict, manifest_path: Path, run_dir: Path, split: str) -> dict:
    rows = pd.read_csv(manifest_path)
    split_rows = rows[rows["split"] == split].reset_index(drop=True)
    labels_path = run_dir / f"{split}_labels.npy"
    probs_path = run_dir / f"{split}_probabilities.npy"
    if not labels_path.exists() or not probs_path.exists():
        raise FileNotFoundError(f"Missing {split} predictions under {run_dir}")

    y_true = np.load(labels_path)
    y_prob = np.load(probs_path)
    if len(split_rows) != len(y_prob):
        raise ValueError(
            f"Manifest split rows ({len(split_rows)}) do not match predictions ({len(y_prob)})"
        )
    if not np.array_equal(split_rows["label"].to_numpy(dtype=np.int64), y_true.astype(np.int64)):
        raise ValueError(f"{split} labels do not match manifest order")

    return {
        "split": split,
        "segments": int(len(split_rows)),
        "files": int(split_rows[["parquet_file", "row_group", "row_in_group"]].drop_duplicates().shape[0]),
        "mean": file_level_metrics(split_rows, y_prob, config["eval"]["thresholds"], aggregation="mean"),
        "max": file_level_metrics(split_rows, y_prob, config["eval"]["thresholds"], aggregation="max"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute file-level metrics from saved segment predictions.")
    parser.add_argument("--config", default="configs/crnn_dads.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["test"], choices=["val", "test"])
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_path = Path(args.manifest)
    run_dir = Path(args.run_dir)
    results = [_evaluate_split(config, manifest_path, run_dir, split) for split in args.splits]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
