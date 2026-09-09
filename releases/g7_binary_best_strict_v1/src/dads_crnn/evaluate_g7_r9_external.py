from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data_firewall import file_sha256
from .evaluate_g7_r6_external_suite import _load_model, _predict
from .evaluate_g7_r7_external import _model_report
from .evaluate_g7_strict_external_baseline import INPUTS, _datasets
from .train import resolve_device


PROTOCOL = "g7_r9_domain_quota_seed42_external_screen_v1"
BASELINE_NAME = "g7_r7_freq_mixstyle_seed42"
BASELINE_DIR = Path("artifacts/g7_r7_freq_mixstyle/external_evaluation")
BASELINE_CHECKPOINT = Path("artifacts/g7_r7_freq_mixstyle/runs/seed_42/best.pt")
BASELINE_INTERNAL_METRICS = Path("artifacts/g7_r7_freq_mixstyle/runs/seed_42/metrics.json")
ARMS = {
    "control_00": {
        "checkpoint": Path("artifacts/g7_r9_domain_quota/control_00/runs/seed_42/best.pt"),
        "metrics": Path("artifacts/g7_r9_domain_quota/control_00/runs/seed_42/metrics.json"),
        "output": Path("artifacts/g7_r9_domain_quota/control_00/external_evaluation"),
    },
    "tau_05": {
        "checkpoint": Path("artifacts/g7_r9_domain_quota/tau_05/runs/seed_42/best.pt"),
        "metrics": Path("artifacts/g7_r9_domain_quota/tau_05/runs/seed_42/metrics.json"),
        "output": Path("artifacts/g7_r9_domain_quota/tau_05/external_evaluation"),
    },
    "tau_10": {
        "checkpoint": Path("artifacts/g7_r9_domain_quota/tau_10/runs/seed_42/best.pt"),
        "metrics": Path("artifacts/g7_r9_domain_quota/tau_10/runs/seed_42/metrics.json"),
        "output": Path("artifacts/g7_r9_domain_quota/tau_10/external_evaluation"),
    },
}
DATASET_INPUT_KEYS = {
    "calibration": "calibration",
    "kielce_tau_holdout": "kielce_tau_holdout",
    "g13_ddl_aerosonic": "g13",
    "idmt_traffic": "idmt",
    "esc50_fold5_guard": "esc50",
}


def _threshold_item(metrics: dict[str, Any], section: str, threshold: float) -> dict[str, Any]:
    values = metrics[section]
    for item in values:
        if np.isclose(float(item["threshold"]), threshold, rtol=0.0, atol=1e-12):
            return item
    raise ValueError(f"Missing threshold={threshold} in {section}")


def _internal_gate(candidate_path: Path) -> dict[str, Any]:
    baseline = json.loads(BASELINE_INTERNAL_METRICS.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline_segment = _threshold_item(baseline, "threshold_metrics", 0.5)
    candidate_segment = _threshold_item(candidate, "threshold_metrics", 0.5)
    baseline_recording = _threshold_item(baseline["test_file_metrics"], "mean", 0.5)
    candidate_recording = _threshold_item(candidate["test_file_metrics"], "mean", 0.5)
    criteria = {
        "segment_f1_drop_le_0_003": (
            float(candidate_segment["f1"]) >= float(baseline_segment["f1"]) - 0.003
        ),
        "recording_f1_drop_le_0_003": (
            float(candidate_recording["f1"]) >= float(baseline_recording["f1"]) - 0.003
        ),
        "recording_recall_drop_le_0_005": (
            float(candidate_recording["recall"])
            >= float(baseline_recording["recall"]) - 0.005
        ),
    }
    failed = [name for name, passed in criteria.items() if not passed]
    return {
        "passed": not failed,
        "criteria": criteria,
        "failed_criteria": failed,
        "baseline": {
            "segment_f1": float(baseline_segment["f1"]),
            "recording_f1": float(baseline_recording["f1"]),
            "recording_recall": float(baseline_recording["recall"]),
        },
        "candidate": {
            "segment_f1": float(candidate_segment["f1"]),
            "recording_f1": float(candidate_recording["f1"]),
            "recording_recall": float(candidate_recording["recall"]),
        },
    }


def _baseline_scores(datasets: dict[str, Any]) -> dict[str, np.ndarray]:
    baseline_metrics_path = BASELINE_DIR / "metrics.json"
    baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    recorded_sha256 = baseline_metrics["models"][BASELINE_NAME]["checkpoint_sha256"]
    if recorded_sha256 != file_sha256(BASELINE_CHECKPOINT):
        raise ValueError("Frozen R7 report is not bound to the current baseline checkpoint")
    output = {}
    for dataset_name, dataset in datasets.items():
        path = BASELINE_DIR / f"{BASELINE_NAME}_{dataset_name}_probabilities.npy"
        values = np.load(path)
        if values.shape != (len(dataset),) or not np.isfinite(values).all():
            raise ValueError(f"Invalid frozen R7 probabilities: {path}")
        output[dataset_name] = values.astype(np.float64, copy=False)
    return output


def _cache_identity(checkpoint: Path, datasets: dict[str, Any]) -> dict[str, Any]:
    if set(datasets) != set(DATASET_INPUT_KEYS):
        raise ValueError("R9 dataset/input identity mapping is incomplete")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "evaluation_code": {
            "r9": file_sha256(Path(__file__)),
            "r7_reporting": file_sha256(
                Path(__file__).with_name("evaluate_g7_r7_external.py")
            ),
            "dataset_builders": file_sha256(
                Path(__file__).with_name("evaluate_g7_strict_external_baseline.py")
            ),
        },
        "datasets": {
            name: {
                "rows": len(dataset),
                "input_manifest": str(INPUTS[DATASET_INPUT_KEYS[name]]),
                "input_manifest_sha256": file_sha256(
                    INPUTS[DATASET_INPUT_KEYS[name]]
                ),
            }
            for name, dataset in datasets.items()
        },
    }


def _candidate_scores(
    datasets: dict[str, Any],
    checkpoint: Path,
    output_dir: Path,
    model_name: str,
    device_name: str,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    identity = _cache_identity(checkpoint, datasets)
    identity_path = output_dir / "prediction_cache_identity.json"
    cache_allowed = False
    if identity_path.is_file():
        cached_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        cache_allowed = cached_identity == identity

    device = resolve_device(device_name)
    model = None
    output = {}
    for dataset_name, dataset in datasets.items():
        path = output_dir / f"{model_name}_{dataset_name}_probabilities.npy"
        if cache_allowed and path.is_file():
            values = np.load(path)
            if values.shape == (len(dataset),) and np.isfinite(values).all():
                print(f"Reusing bound R9 cache / {dataset_name}: {len(dataset)} views", flush=True)
                output[dataset_name] = values.astype(np.float64, copy=False)
                continue
        if model is None:
            model = _load_model(checkpoint, device)
        print(f"Predicting R9 / {dataset_name}: {len(dataset)} views", flush=True)
        values = _predict(model, dataset, device, batch_size)
        np.save(path, values)
        output[dataset_name] = values

    if model is not None:
        del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    identity_path.write_text(
        json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output, identity


def _metric(model: dict[str, Any], dataset: str, path: tuple[str, ...]) -> float:
    value: Any = model["nonoverlap_one_second_mean"]["datasets"][dataset]
    for key in path:
        value = value[key]
    return float(value)


def _comparison(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
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
            "r9": after,
            "delta_r9_minus_r7": after - before,
        }
    criteria = {
        "idmt_fpr_le_strict_baseline": metrics["idmt_fixed_fpr"]["r9"] <= 0.15668234171607714,
        "g13_pauc_ge_r7_minus_0_010": metrics["g13_pauc"]["r9"] >= 0.7792647458238584,
        "g13_recall_ge_r7_minus_0_010": metrics["g13_fixed_recall"]["r9"] >= 0.8016184112843355,
        "g13_roc_auc_ge_r7_minus_0_005": metrics["g13_roc_auc"]["r9"] >= 0.9464408349250465,
        "esc50_fpr_le_r7_plus_0_005": metrics["esc50_fixed_fpr"]["r9"] <= 0.021666666666666667,
        "kielce_recall_drop_le_0_010": metrics["kielce_fixed_recall"]["r9"] >= metrics["kielce_fixed_recall"]["r7"] - 0.010,
        "kielce_fpr_le_strict_baseline": metrics["kielce_fixed_fpr"]["r9"] <= 0.006535947712418277,
    }
    failed = [name for name, passed in criteria.items() if not passed]
    return {
        "primary_unit": "nonoverlap_one_second_recording_mean",
        "metrics": metrics,
        "decision": {
            "passed": not failed,
            "criteria": criteria,
            "failed_criteria": failed,
            "next_action": "train_seeds_43_and_44" if not failed else "stop_this_r9_arm",
        },
    }


def evaluate(arm: str, device_name: str, batch_size: int) -> dict[str, Any]:
    item = ARMS[arm]
    checkpoint = item["checkpoint"]
    metrics_path = item["metrics"]
    output_dir = item["output"]
    required = [
        BASELINE_CHECKPOINT,
        BASELINE_INTERNAL_METRICS,
        BASELINE_DIR / "metrics.json",
        checkpoint,
        metrics_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen R9 evaluation inputs: " + ", ".join(missing))

    internal_gate = _internal_gate(metrics_path)
    if not internal_gate["passed"]:
        raise RuntimeError(
            "R9 internal safety gate failed; external benchmark must not be consumed: "
            + ", ".join(internal_gate["failed_criteria"])
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = _datasets()
    baseline_scores = _baseline_scores(datasets)
    candidate_name = f"g7_r9_{arm}_seed42"
    candidate_scores, cache_identity = _candidate_scores(
        datasets, checkpoint, output_dir, candidate_name, device_name, batch_size
    )
    baseline_report = _model_report(datasets, baseline_scores, BASELINE_CHECKPOINT)
    candidate_report = _model_report(datasets, candidate_scores, checkpoint)
    report = {
        "evaluation_completed": True,
        "protocol": PROTOCOL,
        "arm": arm,
        "benchmark_status": "consumed_reusable_development_benchmark",
        "independent_final_claim_allowed": False,
        "benchmark_labels_used_for_training_or_early_stopping": False,
        "internal_safety_gate": internal_gate,
        "prediction_cache_identity": cache_identity,
        "models": {
            BASELINE_NAME: baseline_report,
            candidate_name: candidate_report,
        },
        "comparison": _comparison(baseline_report, candidate_report),
        "locked_datasets_read": [],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one frozen G7-R9 dose arm")
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    report = evaluate(args.arm, args.device, args.batch_size)
    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
