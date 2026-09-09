from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ensure_dirs, load_config
from .sampling import ClassSourceBalancedSampler


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit class-source balanced training sampling")
    parser.add_argument(
        "--config", default="configs/crnn_dads_full_augmented_g3_source_balanced.yaml"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/sampling_audit/g3_source_balanced.json")
    )
    args = parser.parse_args()
    config = load_config(args.config)
    manifest_path = Path(config["data"]["manifest_dir"]) / "dads_all_seed42.csv"
    rows = pd.read_csv(manifest_path)
    rows = rows[rows["split"] == "train"].reset_index(drop=True)
    sampling = config["train"]["sampling"]
    source_column = str(sampling["source_column"])
    sampler = ClassSourceBalancedSampler(
        rows,
        source_column=source_column,
        num_samples=len(rows),
        seed=args.seed,
    )
    sampled = rows.iloc[np.fromiter(iter(sampler), dtype=np.int64, count=len(sampler))]
    original_class = rows["label"].value_counts(normalize=True).sort_index()
    sampled_class = sampled["label"].value_counts(normalize=True).sort_index()
    original_source_sizes = rows.groupby(["label", source_column]).size()
    sampled_source_sizes = sampled.groupby(["label", source_column]).size()
    source_counts = rows.groupby("label")[source_column].nunique().sort_index()
    original_max_share = {
        str(label): float(original_source_sizes.loc[label].max() / len(rows)) for label in (0, 1)
    }
    balanced_expected_max_share = {
        str(label): float(0.5 / source_counts.loc[label]) for label in (0, 1)
    }
    report = {
        "passed": bool(
            len(sampled) == len(rows)
            and max(abs(float(sampled_class.loc[label]) - 0.5) for label in (0, 1)) < 0.01
            and all(
                balanced_expected_max_share[str(label)] < original_max_share[str(label)]
                for label in (0, 1)
            )
        ),
        "config": args.config,
        "manifest": manifest_path.as_posix(),
        "source_column": source_column,
        "samples_per_epoch": len(sampled),
        "source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "original_class_rates": {str(k): float(v) for k, v in original_class.items()},
        "sampled_class_rates": {str(k): float(v) for k, v in sampled_class.items()},
        "original_max_source_share": original_max_share,
        "balanced_expected_source_share": balanced_expected_max_share,
        "sampled_max_source_draws": {
            str(label): int(sampled_source_sizes.loc[label].max()) for label in (0, 1)
        },
        "sampled_unique_sources": {
            str(label): int(sampled_source_sizes.loc[label].size) for label in (0, 1)
        },
        "replacement": True,
        "validation_and_test_sampling_unchanged": True,
    }
    ensure_dirs(args.output.parent)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("Source-balanced sampling audit failed")


if __name__ == "__main__":
    main()
