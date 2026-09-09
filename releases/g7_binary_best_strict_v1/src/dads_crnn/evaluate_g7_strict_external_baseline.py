from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .data_firewall import file_sha256
from .evaluate_g7_r6_external_suite import (
    _dads_dataset,
    _esc50_dataset,
    _g13_dataset,
    _idmt_dataset,
    _load_model,
    _predict,
    _report,
)
from .evaluate_low_fpr import threshold_at_target_fpr
from .g7_benchmark_validity import build_validity_report
from .train import resolve_device


PROTOCOL = "g7_strict_three_seed_reusable_external_baseline_v1"

INPUTS = {
    "strict_manifest": Path("artifacts/g7_leakage_fixed_v2/data/manifest.csv"),
    "development_fit": Path("artifacts/g7_r6_reusable_multicorpus/fit_manifest.csv"),
    "calibration": Path(
        "artifacts/g7_r6_reusable_multicorpus/threshold_calibration_manifest.csv"
    ),
    "kielce_tau_holdout": Path(
        "artifacts/g7_r5_train_val_test/locked_unseen_external_test/manifest.csv"
    ),
    "g13": Path(
        "artifacts/g13_external_confirmation/intake/external_confirmation_v2_manifest.csv"
    ),
    "idmt": Path("artifacts/g7_improvement/stage_b/development_segments_dedup.csv"),
    "esc50": Path("artifacts/g9_hard_negatives/manifests/hn_guard.csv"),
}

DEFAULT_MODELS = {
    "g7_r2_reference_seed42": Path(
        "artifacts/g7_r2_generalization/ablations/pt_mic_bg_freq/runs/seed_42/best.pt"
    ),
    "g7_strict_seed42": Path("artifacts/g7_strict_retrain_v1/runs/seed_42/best.pt"),
    "g7_strict_seed43": Path("artifacts/g7_strict_retrain_v1/runs/seed_43/best.pt"),
    "g7_strict_seed44": Path("artifacts/g7_strict_retrain_v1/runs/seed_44/best.pt"),
}

IGNORED_SUMMARY_KEYS = {
    "threshold",
    "samples",
    "segments",
    "recordings",
    "source_groups",
    "positive_source_groups",
    "negative_source_groups",
    "rows",
    "tn",
    "fp",
    "fn",
    "tp",
    "false_positives",
}


def _datasets() -> dict[str, Dataset]:
    return {
        "calibration": _dads_dataset(INPUTS["calibration"], "threshold_calibration"),
        "kielce_tau_holdout": _dads_dataset(
            INPUTS["kielce_tau_holdout"], "locked_external_test"
        ),
        "g13_ddl_aerosonic": _g13_dataset(INPUTS["g13"]),
        "idmt_traffic": _idmt_dataset(INPUTS["idmt"]),
        "esc50_fold5_guard": _esc50_dataset(INPUTS["esc50"]),
    }


def _numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    if isinstance(value, dict):
        output: dict[str, float] = {}
        for key, child in value.items():
            if key in IGNORED_SUMMARY_KEYS:
                continue
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            output.update(_numeric_leaves(child, child_prefix))
        return output
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return {prefix: number} if np.isfinite(number) else {}
    return {}


def summarize_strict_seeds(model_reports: dict[str, Any]) -> dict[str, Any]:
    names = [f"g7_strict_seed{seed}" for seed in (42, 43, 44)]
    leaves = [_numeric_leaves(model_reports[name]["datasets"]) for name in names]
    common = set(leaves[0])
    for values in leaves[1:]:
        common &= set(values)
    metrics = {}
    for path in sorted(common):
        values = np.asarray([leaves[index][path] for index in range(3)], dtype=np.float64)
        metrics[path] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "values_by_seed": {
                str(seed): float(value) for seed, value in zip((42, 43, 44), values)
            },
        }
    return {
        "seeds": [42, 43, 44],
        "standard_deviation": "sample_std_ddof_1",
        "metrics": metrics,
    }


def _validate_inputs(models: dict[str, Path]) -> None:
    missing = [str(path) for path in [*INPUTS.values(), *models.values()] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen P0 inputs: " + ", ".join(missing))


def evaluate(
    models: dict[str, Path],
    output_dir: Path,
    device_name: str,
    batch_size: int,
) -> dict[str, Any]:
    _validate_inputs(models)
    datasets = _datasets()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    model_reports: dict[str, Any] = {}

    for model_name, checkpoint in models.items():
        print(f"Loading {model_name}: {checkpoint}", flush=True)
        model = _load_model(checkpoint, device)
        scores_by_dataset: dict[str, np.ndarray] = {}
        for dataset_name, dataset in datasets.items():
            probability_path = output_dir / f"{model_name}_{dataset_name}_probabilities.npy"
            if probability_path.is_file():
                cached = np.load(probability_path)
                if cached.shape == (len(dataset),) and np.isfinite(cached).all():
                    print(
                        f"Reusing {model_name} / {dataset_name}: {len(dataset)} views",
                        flush=True,
                    )
                    scores = cached.astype(np.float64, copy=False)
                else:
                    print(f"Ignoring invalid cache: {probability_path}", flush=True)
                    scores = _predict(model, dataset, device, batch_size)
            else:
                print(
                    f"Predicting {model_name} / {dataset_name}: {len(dataset)} views",
                    flush=True,
                )
                scores = _predict(model, dataset, device, batch_size)
            np.save(probability_path, scores)
            scores_by_dataset[dataset_name] = scores

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        calibration_scores = scores_by_dataset["calibration"]
        thresholds = {"fixed_0_5": 0.5}
        calibration = {}
        for target in (0.01, 0.05):
            item = threshold_at_target_fpr(calibration_scores, target)
            key = f"calibrated_fpr_{target:.2f}"
            thresholds[key] = float(item["threshold"])
            calibration[key] = item

        model_reports[model_name] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "thresholds": thresholds,
            "calibration": calibration,
            "datasets": {
                name: _report(dataset.rows, scores_by_dataset[name], thresholds)
                for name, dataset in datasets.items()
                if name != "calibration"
            },
        }

    validity = build_validity_report(
        pd.read_csv(INPUTS["development_fit"], low_memory=False),
        pd.read_csv(INPUTS["calibration"], low_memory=False),
        pd.read_csv(INPUTS["kielce_tau_holdout"], low_memory=False),
        pd.read_csv(INPUTS["g13"], low_memory=False),
        pd.read_csv(INPUTS["idmt"], low_memory=False),
        pd.read_csv(INPUTS["esc50"], low_memory=False),
    )
    report = {
        "evaluation_completed": True,
        "protocol": PROTOCOL,
        "benchmark_status": "consumed_reusable_development_benchmark",
        "independent_final_claim_allowed": False,
        "model_selection_or_threshold_tuning_allowed": False,
        "input_policy": {
            "native_model_window_seconds": 0.5,
            "recording_aggregation": "mean_probability",
            "threshold_source": "TAU_Prague_calibration_per_model",
            "threshold_targets_fpr": [0.01, 0.05],
        },
        "validity": validity,
        "models": model_reports,
        "strict_three_seed_summary": summarize_strict_seeds(model_reports),
        "inputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in INPUTS.items()
        },
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate strict G7 three-seed checkpoints on reusable external benchmarks"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/g7_strict_retrain_v1/external_baseline"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    _validate_inputs(DEFAULT_MODELS)
    if args.preflight_only:
        datasets = _datasets()
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "passed": True,
                    "models": {
                        name: {"path": str(path), "sha256": file_sha256(path)}
                        for name, path in DEFAULT_MODELS.items()
                    },
                    "dataset_views": {name: len(dataset) for name, dataset in datasets.items()},
                    "writes_predictions": False,
                    "runs_model_inference": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    report = evaluate(DEFAULT_MODELS, args.output_dir, args.device, args.batch_size)
    print(
        json.dumps(
            {
                "evaluation_completed": report["evaluation_completed"],
                "protocol": report["protocol"],
                "output": str(args.output_dir / "metrics.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
