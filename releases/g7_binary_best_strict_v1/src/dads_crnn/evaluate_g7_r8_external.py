from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .evaluate_g7_r6_external_suite import _load_model, _predict
from .evaluate_g7_r7_external import _model_report
from .evaluate_g7_strict_external_baseline import _datasets
from .train import resolve_device


PROTOCOL = "g7_r8_tau_urban_negative_replay_seed42_external_screen_v1"
BASELINE_NAME = "g7_r7_freq_mixstyle_seed42"
CANDIDATE_NAME = "g7_r8_urban_negatives_seed42"
BASELINE_DIR = Path("artifacts/g7_r7_freq_mixstyle/external_evaluation")
BASELINE_CHECKPOINT = Path("artifacts/g7_r7_freq_mixstyle/runs/seed_42/best.pt")
CANDIDATE_CHECKPOINT = Path("artifacts/g7_r8_urban_negatives/runs/seed_42/best.pt")


def _scores_from_cache(datasets: dict[str, Any], directory: Path, model_name: str) -> dict[str, np.ndarray]:
    output = {}
    for dataset_name, dataset in datasets.items():
        path = directory / f"{model_name}_{dataset_name}_probabilities.npy"
        values = np.load(path)
        if values.shape != (len(dataset),) or not np.isfinite(values).all():
            raise ValueError(f"Invalid frozen probabilities: {path}")
        output[dataset_name] = values.astype(np.float64, copy=False)
    return output


def _candidate_scores(
    datasets: dict[str, Any], output_dir: Path, device_name: str, batch_size: int
) -> dict[str, np.ndarray]:
    device = resolve_device(device_name)
    model = _load_model(CANDIDATE_CHECKPOINT, device)
    output = {}
    for dataset_name, dataset in datasets.items():
        path = output_dir / f"{CANDIDATE_NAME}_{dataset_name}_probabilities.npy"
        if path.is_file():
            values = np.load(path)
            if values.shape == (len(dataset),) and np.isfinite(values).all():
                print(f"Reusing R8 / {dataset_name}: {len(dataset)} views", flush=True)
                output[dataset_name] = values.astype(np.float64, copy=False)
                continue
        print(f"Predicting R8 / {dataset_name}: {len(dataset)} views", flush=True)
        values = _predict(model, dataset, device, batch_size)
        np.save(path, values)
        output[dataset_name] = values
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def _metric(model: dict[str, Any], dataset: str, path: tuple[str, ...]) -> float:
    value: Any = model["nonoverlap_one_second_mean"]["datasets"][dataset]
    for key in path:
        value = value[key]
    return float(value)


def _comparison(models: dict[str, Any]) -> dict[str, Any]:
    baseline = models[BASELINE_NAME]
    candidate = models[CANDIDATE_NAME]
    definitions = {
        "idmt_fixed_fpr": ("idmt_traffic", ("recording_mean_operating_points", "fixed_0_5", "false_positive_rate")),
        "g13_roc_auc": ("g13_ddl_aerosonic", ("recording_mean_ranking", "roc_auc")),
        "g13_pr_auc": ("g13_ddl_aerosonic", ("recording_mean_ranking", "pr_auc")),
        "g13_pauc": ("g13_ddl_aerosonic", ("recording_mean_ranking", "standardized_pauc_fpr_le_0_05")),
        "g13_fixed_recall": ("g13_ddl_aerosonic", ("recording_mean_operating_points", "fixed_0_5", "recall")),
        "esc50_fixed_fpr": ("esc50_fold5_guard", ("recording_mean_operating_points", "fixed_0_5", "false_positive_rate")),
        "kielce_fixed_recall": ("kielce_tau_holdout", ("recording_mean_operating_points", "fixed_0_5", "recall")),
        "kielce_fixed_fpr": ("kielce_tau_holdout", ("recording_mean_operating_points", "fixed_0_5", "false_positive_rate")),
    }
    metrics = {}
    for name, (dataset, path) in definitions.items():
        before = _metric(baseline, dataset, path)
        after = _metric(candidate, dataset, path)
        metrics[name] = {
            "r7": before,
            "r8": after,
            "delta_r8_minus_r7": after - before,
        }
    criteria = {
        "idmt_fpr_le_strict_baseline": metrics["idmt_fixed_fpr"]["r8"] <= 0.15668234171607714,
        "g13_pauc_ge_predeclared_floor": metrics["g13_pauc"]["r8"] >= 0.7792647458238584,
        "g13_recall_ge_predeclared_floor": metrics["g13_fixed_recall"]["r8"] >= 0.8016184112843355,
        "g13_roc_auc_ge_predeclared_floor": metrics["g13_roc_auc"]["r8"] >= 0.9464408349250465,
        "esc50_fpr_le_predeclared_ceiling": metrics["esc50_fixed_fpr"]["r8"] <= 0.021666666666666667,
        "kielce_recall_drop_le_0_010": metrics["kielce_fixed_recall"]["r8"] >= metrics["kielce_fixed_recall"]["r7"] - 0.010,
        "kielce_fpr_le_strict_baseline": metrics["kielce_fixed_fpr"]["r8"] <= 0.006535947712418277,
    }
    failed = [name for name, passed in criteria.items() if not passed]
    return {
        "primary_unit": "nonoverlap_one_second_recording_mean",
        "metrics": metrics,
        "decision": {
            "passed": not failed,
            "criteria": criteria,
            "failed_criteria": failed,
            "next_action": "train_seeds_43_and_44" if not failed else "stop_r8_branch",
        },
    }


def evaluate(output_dir: Path, device_name: str, batch_size: int) -> dict[str, Any]:
    datasets = _datasets()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_scores = _scores_from_cache(datasets, BASELINE_DIR, BASELINE_NAME)
    candidate_scores = _candidate_scores(datasets, output_dir, device_name, batch_size)
    models = {
        BASELINE_NAME: _model_report(datasets, baseline_scores, BASELINE_CHECKPOINT),
        CANDIDATE_NAME: _model_report(datasets, candidate_scores, CANDIDATE_CHECKPOINT),
    }
    report = {
        "evaluation_completed": True,
        "protocol": PROTOCOL,
        "benchmark_status": "consumed_reusable_development_benchmark",
        "independent_final_claim_allowed": False,
        "model_selection_or_threshold_tuning_allowed": False,
        "models": models,
        "comparison": _comparison(models),
        "locked_datasets_read": [],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen G7-R8 against G7-R7")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/g7_r8_urban_negatives/external_evaluation"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    report = evaluate(args.output_dir, args.device, args.batch_size)
    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
