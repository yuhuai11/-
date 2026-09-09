from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data_firewall import file_sha256
from .evaluate_g7_r6_external_suite import _load_model, _predict, _report
from .evaluate_g7_strict_external_baseline import _datasets
from .evaluate_g7_strict_multiscale import aggregate_nonoverlap
from .evaluate_low_fpr import threshold_at_target_fpr
from .train import resolve_device


PROTOCOL = "g7_r7_frequency_mixstyle_seed42_external_screen_v1"
BASELINE_DIR = Path("artifacts/g7_strict_retrain_v1/external_baseline")
BASELINE_CHECKPOINT = Path("artifacts/g7_strict_retrain_v1/runs/seed_42/best.pt")
CANDIDATE_CHECKPOINT = Path("artifacts/g7_r7_freq_mixstyle/runs/seed_42/best.pt")
MODEL_NAMES = ("g7_strict_seed42", "g7_r7_freq_mixstyle_seed42")


def _thresholds(calibration_scores: np.ndarray) -> tuple[dict[str, float], dict[str, Any]]:
    thresholds = {"fixed_0_5": 0.5}
    calibration = {}
    for target in (0.01, 0.05):
        item = threshold_at_target_fpr(calibration_scores, target)
        key = f"calibrated_fpr_{target:.2f}"
        thresholds[key] = float(item["threshold"])
        calibration[key] = item
    return thresholds, calibration


def _load_baseline_scores(datasets: dict[str, Any]) -> dict[str, np.ndarray]:
    scores = {}
    for dataset_name, dataset in datasets.items():
        path = BASELINE_DIR / f"g7_strict_seed42_{dataset_name}_probabilities.npy"
        value = np.load(path)
        if value.shape != (len(dataset),) or not np.isfinite(value).all():
            raise ValueError(f"Invalid frozen baseline probabilities: {path}")
        scores[dataset_name] = value.astype(np.float64, copy=False)
    return scores


def _predict_candidate(
    datasets: dict[str, Any], output_dir: Path, device_name: str, batch_size: int
) -> dict[str, np.ndarray]:
    device = resolve_device(device_name)
    model = _load_model(CANDIDATE_CHECKPOINT, device)
    scores = {}
    for dataset_name, dataset in datasets.items():
        path = output_dir / f"g7_r7_freq_mixstyle_seed42_{dataset_name}_probabilities.npy"
        if path.is_file():
            value = np.load(path)
            if value.shape == (len(dataset),) and np.isfinite(value).all():
                print(f"Reusing R7 / {dataset_name}: {len(dataset)} views", flush=True)
                scores[dataset_name] = value.astype(np.float64, copy=False)
                continue
        print(f"Predicting R7 / {dataset_name}: {len(dataset)} views", flush=True)
        value = _predict(model, dataset, device, batch_size)
        np.save(path, value)
        scores[dataset_name] = value
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores


def _model_report(
    datasets: dict[str, Any], scores: dict[str, np.ndarray], checkpoint: Path
) -> dict[str, Any]:
    native_thresholds, native_calibration = _thresholds(scores["calibration"])
    aggregated = {
        name: aggregate_nonoverlap(dataset.rows, scores[name], windows_per_decision=2)
        for name, dataset in datasets.items()
    }
    one_second_thresholds, one_second_calibration = _thresholds(aggregated["calibration"][1])
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "native_half_second": {
            "thresholds": native_thresholds,
            "calibration": native_calibration,
            "datasets": {
                name: _report(dataset.rows, scores[name], native_thresholds)
                for name, dataset in datasets.items()
                if name != "calibration"
            },
        },
        "nonoverlap_one_second_mean": {
            "thresholds": one_second_thresholds,
            "calibration": one_second_calibration,
            "datasets": {
                name: {
                    **_report(rows, values, one_second_thresholds),
                    "aggregation_audit": audit,
                }
                for name, (rows, values, audit) in aggregated.items()
                if name != "calibration"
            },
        },
    }


def _value(report: dict[str, Any], duration: str, dataset: str, path: tuple[str, ...]) -> float:
    value: Any = report[duration]["datasets"][dataset]
    for key in path:
        value = value[key]
    return float(value)


def _comparison(models: dict[str, Any]) -> dict[str, Any]:
    baseline = models[MODEL_NAMES[0]]
    candidate = models[MODEL_NAMES[1]]
    definitions = {
        "g13_recording_roc_auc": (
            "g13_ddl_aerosonic", ("recording_mean_ranking", "roc_auc"), "higher"
        ),
        "g13_recording_pr_auc": (
            "g13_ddl_aerosonic", ("recording_mean_ranking", "pr_auc"), "higher"
        ),
        "g13_recording_standardized_pauc_fpr_le_0_05": (
            "g13_ddl_aerosonic",
            ("recording_mean_ranking", "standardized_pauc_fpr_le_0_05"),
            "higher",
        ),
        "g13_fixed_recall": (
            "g13_ddl_aerosonic", ("recording_mean_operating_points", "fixed_0_5", "recall"), "higher"
        ),
        "idmt_fixed_fpr": (
            "idmt_traffic",
            ("recording_mean_operating_points", "fixed_0_5", "false_positive_rate"),
            "lower",
        ),
        "esc50_fixed_fpr": (
            "esc50_fold5_guard",
            ("recording_mean_operating_points", "fixed_0_5", "false_positive_rate"),
            "lower",
        ),
    }
    output = {}
    for duration in ("native_half_second", "nonoverlap_one_second_mean"):
        metrics = {}
        for name, (dataset, path, direction) in definitions.items():
            b = _value(baseline, duration, dataset, path)
            c = _value(candidate, duration, dataset, path)
            metrics[name] = {
                "baseline": b,
                "candidate": c,
                "delta_candidate_minus_baseline": c - b,
                "preferred_direction": direction,
            }
        output[duration] = metrics

    primary = output["nonoverlap_one_second_mean"]
    criteria = {
        "g13_roc_auc_not_lower": primary["g13_recording_roc_auc"]["delta_candidate_minus_baseline"] >= 0,
        "g13_pauc_not_lower": primary["g13_recording_standardized_pauc_fpr_le_0_05"]["delta_candidate_minus_baseline"] >= 0,
        "g13_fixed_recall_not_lower": primary["g13_fixed_recall"]["delta_candidate_minus_baseline"] >= 0,
        "idmt_fixed_fpr_not_higher": primary["idmt_fixed_fpr"]["delta_candidate_minus_baseline"] <= 0,
        "esc50_fixed_fpr_not_higher": primary["esc50_fixed_fpr"]["delta_candidate_minus_baseline"] <= 0,
    }
    failed = [name for name, passed in criteria.items() if not passed]
    output["screening_decision"] = {
        "primary_decision_duration": "nonoverlap_one_second_mean",
        "passed": not failed,
        "criteria": criteria,
        "failed_criteria": failed,
        "next_action": (
            "train_seeds_43_and_44" if not failed else "stop_mixstyle_only_branch"
        ),
    }
    return output


def evaluate(output_dir: Path, device_name: str, batch_size: int) -> dict[str, Any]:
    required = [BASELINE_DIR / "metrics.json", BASELINE_CHECKPOINT, CANDIDATE_CHECKPOINT]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen R7 evaluation inputs: " + ", ".join(missing))
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = _datasets()
    baseline_scores = _load_baseline_scores(datasets)
    candidate_scores = _predict_candidate(datasets, output_dir, device_name, batch_size)
    models = {
        MODEL_NAMES[0]: _model_report(datasets, baseline_scores, BASELINE_CHECKPOINT),
        MODEL_NAMES[1]: _model_report(datasets, candidate_scores, CANDIDATE_CHECKPOINT),
    }
    report = {
        "evaluation_completed": True,
        "protocol": PROTOCOL,
        "experiment_type": "single_variable_frequency_mixstyle_seed42_screen",
        "model_selection_or_threshold_tuning_allowed": False,
        "independent_final_claim_allowed": False,
        "models": models,
        "comparison": _comparison(models),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen G7-R7 Frequency MixStyle on frozen external data")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/g7_r7_freq_mixstyle/external_evaluation")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    report = evaluate(args.output_dir, args.device, args.batch_size)
    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
