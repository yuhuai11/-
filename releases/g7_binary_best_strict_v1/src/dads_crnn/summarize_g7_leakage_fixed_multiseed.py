from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from .summarize_panns_runs import collect_runs, write_summary


PROTOCOL = "g7_native_half_second_multiseed_aggregate_v1"
DATA_PROTOCOL = "dads_native_half_second_content_component_v2"
PRIMARY_THRESHOLD = 0.5
METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "false_positive_rate",
    "f1",
    "auc",
    "pr_auc",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(values: list[float]) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError("Multi-seed sample statistics require at least two values")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Cannot aggregate a non-finite metric")
    return {
        "mean": float(statistics.fmean(values)),
        "sample_std": float(statistics.stdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _items_by_threshold(items: list[dict[str, Any]]) -> dict[float, dict[str, Any]]:
    output: dict[float, dict[str, Any]] = {}
    for item in items:
        threshold = float(item["threshold"])
        if threshold in output:
            raise ValueError(f"Duplicate threshold {threshold}")
        output[threshold] = item
    return output


def _validate_run_artifacts(run_root: Path, run: dict[str, Any]) -> None:
    seed = int(run["seed"])
    seed_dir = run_root / f"seed_{seed}"
    checkpoint = seed_dir / "best.pt"
    if _sha256(checkpoint) != run.get("checkpoint_sha256"):
        raise ValueError(f"Checkpoint SHA256 mismatch for seed {seed}")
    prediction_hashes = run.get("prediction_sha256")
    if not isinstance(prediction_hashes, dict):
        raise ValueError(f"Missing prediction SHA256 identity for seed {seed}")
    for name, expected in prediction_hashes.items():
        if _sha256(seed_dir / f"{name}.npy") != expected:
            raise ValueError(f"Prediction SHA256 mismatch for seed {seed}: {name}")


def _validate_comparability(run_root: Path, runs: list[dict[str, Any]]) -> None:
    reference = runs[0]
    identity_fields = (
        "model_type",
        "initialization",
        "feature_type",
        "parameter_count",
        "mixed_precision",
        "amp_init_scale",
        "official_pretraining_sha256",
    )
    reference_identity = {field: reference.get(field) for field in identity_fields}
    reference_segment_thresholds = {
        "val": sorted(_items_by_threshold(reference["val_threshold_metrics"])),
        "test": sorted(_items_by_threshold(reference["threshold_metrics"])),
    }
    for run in runs:
        _validate_run_artifacts(run_root, run)
        identity = {field: run.get(field) for field in identity_fields}
        if identity != reference_identity:
            raise ValueError("Multi-seed model or training settings are not identical")
        thresholds = {
            "val": sorted(_items_by_threshold(run["val_threshold_metrics"])),
            "test": sorted(_items_by_threshold(run["threshold_metrics"])),
        }
        if thresholds != reference_segment_thresholds:
            raise ValueError("Multi-seed threshold grids are not identical")
        audits = run["training_inputs"].get("input_audits", [])
        if not any(audit.get("protocol") == DATA_PROTOCOL for audit in audits):
            raise ValueError(f"Seed {run['seed']} is not bound to the native half-second audit")
    if PRIMARY_THRESHOLD not in reference_segment_thresholds["val"]:
        raise ValueError("The frozen primary threshold 0.50 is missing")


def _aggregate_threshold_items(
    runs: list[dict[str, Any]], key: str
) -> dict[str, dict[str, dict[str, float]]]:
    indexed = [_items_by_threshold(run[key]) for run in runs]
    thresholds = sorted(indexed[0])
    output: dict[str, dict[str, dict[str, float]]] = {}
    for threshold in thresholds:
        output[f"{threshold:.2f}"] = {
            metric: _stats([float(items[threshold][metric]) for items in indexed])
            for metric in METRIC_NAMES
        }
    return output


def _aggregate_recording_items(
    runs: list[dict[str, Any]], key: str
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    output: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for aggregation in ("mean", "max"):
        indexed = [
            _items_by_threshold(run[key][aggregation])
            for run in runs
        ]
        thresholds = sorted(indexed[0])
        output[aggregation] = {}
        for threshold in thresholds:
            output[aggregation][f"{threshold:.2f}"] = {
                metric: _stats([float(items[threshold][metric]) for items in indexed])
                for metric in METRIC_NAMES
            }
    return output


def build_aggregate(run_root: Path, seeds: list[int]) -> dict[str, Any]:
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Expected at least two unique seeds")
    ordered_seeds = sorted(seeds)
    runs = collect_runs(run_root, ordered_seeds)
    _validate_comparability(run_root, runs)
    primary_items = [
        _items_by_threshold(run["threshold_metrics"])[PRIMARY_THRESHOLD]
        for run in runs
    ]
    return {
        "passed": True,
        "protocol": PROTOCOL,
        "data_protocol": DATA_PROTOCOL,
        "seeds": ordered_seeds,
        "seed_count": len(ordered_seeds),
        "test_role": "consumed_internal_development_test",
        "fresh_final_holdout": False,
        "selection_rule": "report_all_seeds_no_seed_selection",
        "primary_threshold": PRIMARY_THRESHOLD,
        "sample_standard_deviation": True,
        "training_inputs": runs[0]["training_inputs"],
        "best_epochs": [int(run["best_epoch"]) for run in runs],
        "elapsed_seconds": _stats([float(run["elapsed_seconds"]) for run in runs]),
        "amp_nonfinite_gradient_skip_steps": [
            int(run["amp_nonfinite_gradient_skip_steps"]) for run in runs
        ],
        "primary_internal_test": {
            metric: _stats([float(item[metric]) for item in primary_items])
            for metric in METRIC_NAMES
        },
        "segment_metrics": {
            "validation": _aggregate_threshold_items(runs, "val_threshold_metrics"),
            "consumed_internal_test": _aggregate_threshold_items(runs, "threshold_metrics"),
        },
        "recording_metrics": {
            "validation": _aggregate_recording_items(runs, "val_file_metrics"),
            "consumed_internal_test": _aggregate_recording_items(runs, "test_file_metrics"),
        },
        "per_seed": [
            {
                "seed": int(run["seed"]),
                "best_epoch": int(run["best_epoch"]),
                "checkpoint_sha256": run["checkpoint_sha256"],
                "primary_internal_test": _items_by_threshold(run["threshold_metrics"])[
                    PRIMARY_THRESHOLD
                ],
            }
            for run in runs
        ],
    }


def write_aggregate(run_root: Path, aggregate: dict[str, Any]) -> Path:
    output = run_root / "multiseed_aggregate.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    temporary.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and aggregate native-half-second G7 multi-seed runs"
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    aggregate = build_aggregate(args.run_root, args.expected_seeds)
    runs = collect_runs(args.run_root, sorted(args.expected_seeds))
    write_summary(args.run_root, runs)
    output = write_aggregate(args.run_root, aggregate)
    primary = aggregate["primary_internal_test"]
    print(
        f"Verified seeds {aggregate['seeds']}; wrote {output}; "
        f"F1={primary['f1']['mean']:.6f}±{primary['f1']['sample_std']:.6f}"
    )


if __name__ == "__main__":
    main()
