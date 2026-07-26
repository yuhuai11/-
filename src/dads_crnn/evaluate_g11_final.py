from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .calibrate_ood import probabilities_from_logits
from .config import ensure_dirs, load_config
from .evaluate_external import _fast_threshold_metrics, stratified_bootstrap_ci
from .external_data import ExternalAudioDataset
from .panns import file_sha256
from .train import resolve_device
from .train_panns import build_model


ALGORITHM = "g11_final_dual_mode_v1"
UNLOCK_PHRASE = "RUN_G11_FINAL_ONCE"


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    ensure_dirs(path.parent)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    ensure_dirs(path.parent)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("algorithm") != ALGORITHM:
        raise ValueError(f"G11 algorithm must be frozen as {ALGORITHM}")
    modes = config.get("evaluation", {}).get("modes", {})
    if set(modes) != {"strict", "balanced"}:
        raise ValueError("G11 requires exactly strict and balanced modes")
    if float(modes["strict"].get("target_fpr", -1)) != 0.01:
        raise ValueError("G11 strict target FPR is frozen as 0.01")
    if modes["strict"].get("candidate_threshold_source") != "source_robust_selected":
        raise ValueError("G11 strict mode requires the source-robust selected threshold")
    if float(modes["balanced"].get("target_fpr", -1)) != 0.05:
        raise ValueError("G11 balanced target FPR is frozen as 0.05")
    if modes["balanced"].get("candidate_threshold_source") != "pooled":
        raise ValueError("G11 balanced mode requires the pooled threshold")
    datasets = config.get("final_data", {}).get("datasets", {})
    if set(datasets) != {"unseen", "real_world"}:
        raise ValueError("G11 final datasets are frozen as unseen and real_world")


def _point_by_target(report: dict[str, Any], target: float) -> dict[str, Any]:
    matches = [item for item in report["operating_points"] if float(item["target_fpr"]) == target]
    if len(matches) != 1:
        raise ValueError(f"Expected one operating point for target FPR {target}")
    return matches[0]


def _threshold_by_target(report: dict[str, Any], target: float) -> dict[str, Any]:
    matches = [item for item in report["thresholds"] if float(item["target_fpr"]) == target]
    if len(matches) != 1:
        raise ValueError(f"Expected one calibration threshold for target FPR {target}")
    return matches[0]


def freeze_protocol(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    _validate_config(config)
    output = Path(config["outputs"]["frozen_protocol"])
    if output.exists():
        raise FileExistsError(f"G11 protocol is already frozen: {output}")
    protocol_document = Path(config["protocol_document"])
    audit_path = Path(config["final_data"]["manifest_audit"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_datasets = config["final_data"]["datasets"]
    for name, settings in expected_datasets.items():
        observed = audit.get("datasets", {}).get(name, {})
        expected_labels = {str(key): int(value) for key, value in settings["expected_labels"].items()}
        if int(observed.get("samples", -1)) != int(settings["expected_rows"]):
            raise ValueError(f"G11 audit row count mismatch for {name}")
        if observed.get("label_counts") != expected_labels:
            raise ValueError(f"G11 audit label count mismatch for {name}")
        if int(observed.get("exact_dads_overlaps", -1)) != 0:
            raise ValueError(f"G11 audit reports DADS overlap for {name}")

    models = config["models"]
    baseline_checkpoint = Path(models["baseline_checkpoint"])
    candidate_checkpoints = [Path(value) for value in models["candidate_checkpoints"]]
    if len(candidate_checkpoints) != 3:
        raise ValueError("G11 requires exactly three candidate checkpoints")
    sources = config["calibration_sources"]
    baseline_calibration_path = Path(sources["baseline_calibration"])
    baseline_low_fpr_path = Path(sources["baseline_low_fpr"])
    candidate_calibration_path = Path(sources["candidate_calibration"])
    baseline_calibration = json.loads(baseline_calibration_path.read_text(encoding="utf-8"))
    baseline_low_fpr = json.loads(baseline_low_fpr_path.read_text(encoding="utf-8"))
    candidate_calibration = json.loads(candidate_calibration_path.read_text(encoding="utf-8"))
    if candidate_calibration.get("protocol", {}).get("holdout_used_for_fitting") is not False:
        raise ValueError("G11 candidate calibration is not tune-only")

    modes = {}
    for name, settings in config["evaluation"]["modes"].items():
        target = float(settings["target_fpr"])
        baseline_point = _point_by_target(baseline_low_fpr, target)
        candidate_point = _threshold_by_target(candidate_calibration, target)
        candidate_threshold = (
            float(candidate_point["threshold"])
            if settings["candidate_threshold_source"] == "source_robust_selected"
            else float(candidate_point["pooled_threshold"])
        )
        modes[name] = {
            "target_fpr": target,
            "baseline_threshold": float(baseline_point["calibration"]["threshold"]),
            "candidate_threshold": candidate_threshold,
            "candidate_threshold_source": settings["candidate_threshold_source"],
        }
    report = {
        "algorithm": ALGORITHM,
        "status": "frozen_before_final_data_read",
        "protocol": {
            "modes": modes,
            "baseline_temperature": float(baseline_calibration["temperature"]),
            "candidate_temperature": float(candidate_calibration["temperature"]),
            "candidate_ensemble": "arithmetic_mean_raw_logit_seeds_42_43_44",
            "final_data_used_for_selection": False,
            "technical_resume_only": True,
            "threshold_changes_after_unlock_forbidden": True,
        },
        "inputs": {
            "config": {"path": config_path.as_posix(), "sha256": _sha256(config_path)},
            "protocol_document": {"path": protocol_document.as_posix(), "sha256": _sha256(protocol_document)},
            "manifest_audit": {"path": audit_path.as_posix(), "sha256": _sha256(audit_path)},
            "baseline_checkpoint": {"path": baseline_checkpoint.as_posix(), "sha256": _sha256(baseline_checkpoint)},
            "candidate_checkpoints": [
                {"path": path.as_posix(), "sha256": _sha256(path)} for path in candidate_checkpoints
            ],
            "baseline_calibration": {"path": baseline_calibration_path.as_posix(), "sha256": _sha256(baseline_calibration_path)},
            "baseline_low_fpr": {"path": baseline_low_fpr_path.as_posix(), "sha256": _sha256(baseline_low_fpr_path)},
            "candidate_calibration": {"path": candidate_calibration_path.as_posix(), "sha256": _sha256(candidate_calibration_path)},
        },
        "final_dataset_contract": {
            name: {
                "manifest_path": str(settings["manifest"]),
                "expected_rows": int(settings["expected_rows"]),
                "expected_labels": {str(key): int(value) for key, value in settings["expected_labels"].items()},
                "manifest_sha256": "sealed_at_explicit_unlock",
            }
            for name, settings in expected_datasets.items()
        },
        "implementation": {"path": Path(__file__).resolve().as_posix(), "sha256": _sha256(Path(__file__))},
        "locked_datasets_read": [],
    }
    _atomic_json(output, report)
    print(json.dumps({"status": report["status"], "modes": modes, "locked_datasets_read": []}, indent=2))
    return report


def _verify_frozen_inputs(config_path: Path, config: dict[str, Any], frozen: dict[str, Any]) -> None:
    if frozen.get("algorithm") != ALGORITHM or frozen.get("status") != "frozen_before_final_data_read":
        raise ValueError("Invalid G11 frozen protocol")
    checks = {
        "config": config_path,
        "protocol_document": Path(config["protocol_document"]),
        "manifest_audit": Path(config["final_data"]["manifest_audit"]),
        "baseline_checkpoint": Path(config["models"]["baseline_checkpoint"]),
        "baseline_calibration": Path(config["calibration_sources"]["baseline_calibration"]),
        "baseline_low_fpr": Path(config["calibration_sources"]["baseline_low_fpr"]),
        "candidate_calibration": Path(config["calibration_sources"]["candidate_calibration"]),
    }
    for key, path in checks.items():
        if frozen["inputs"][key]["sha256"] != _sha256(path):
            raise ValueError(f"G11 frozen input changed: {key}")
    candidate_paths = [Path(value) for value in config["models"]["candidate_checkpoints"]]
    if [item["sha256"] for item in frozen["inputs"]["candidate_checkpoints"]] != [
        _sha256(path) for path in candidate_paths
    ]:
        raise ValueError("G11 candidate checkpoints changed after protocol freeze")
    if frozen["implementation"]["sha256"] != _sha256(Path(__file__)):
        raise ValueError("G11 implementation changed after protocol freeze")


def _validate_final_manifest(path: Path, dataset_name: str, contract: dict[str, Any]) -> pd.DataFrame:
    rows = pd.read_csv(path, low_memory=False)
    required = {"dataset", "path", "label", "sha256", "source_group", "condition"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Final manifest {dataset_name} is missing columns: {missing}")
    if len(rows) != int(contract["expected_rows"]):
        raise ValueError(f"Final manifest row count mismatch for {dataset_name}")
    counts = rows["label"].astype(int).value_counts().sort_index().to_dict()
    expected = {int(key): int(value) for key, value in contract["expected_labels"].items()}
    if counts != expected:
        raise ValueError(f"Final manifest label count mismatch for {dataset_name}")
    if set(rows["dataset"].astype(str).str.lower()) != {dataset_name}:
        raise ValueError(f"Final manifest dataset identity mismatch for {dataset_name}")
    return rows


def _predict_logits(
    checkpoint_path: Path,
    manifest_path: Path,
    *,
    batch_size: int,
    num_workers: int,
    device_name: str,
    mixed_precision: bool,
) -> np.ndarray:
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset = ExternalAudioDataset(
        manifest_path,
        sample_rate=int(config["data"]["sample_rate"]),
        clip_seconds=float(config["data"]["clip_seconds"]),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logits = np.empty(len(dataset), dtype=np.float32)
    with torch.no_grad():
        for waveforms, _, indices in tqdm(loader, desc=checkpoint_path.parent.name):
            waveforms = waveforms.to(device)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda" and mixed_precision):
                output = model(waveforms)
            logits[indices.numpy()] = output.float().cpu().numpy()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return logits


def _score_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float, clusters: np.ndarray, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    result = _fast_threshold_metrics(labels, probabilities, threshold)
    predictions = probabilities >= threshold
    negative = labels == 0
    positive = labels == 1
    result.update(
        {
            "threshold": float(threshold),
            "fpr": float(np.mean(predictions[negative])),
            "tpr": float(np.mean(predictions[positive])),
            "tp": int(np.sum(predictions & positive)),
            "fp": int(np.sum(predictions & negative)),
            "tn": int(np.sum(~predictions & negative)),
            "fn": int(np.sum(~predictions & positive)),
            "bootstrap_95_ci": stratified_bootstrap_ci(labels, probabilities, threshold, samples=bootstrap_samples, seed=seed, clusters=clusters),
        }
    )
    return result


def _ranking(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "partial_auc_fpr_0_05_standardized": float(roc_auc_score(labels, probabilities, max_fpr=0.05)),
    }


def _unique_hash_view(
    rows: pd.DataFrame,
    baseline_probability: np.ndarray,
    candidate_probability: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "sha256": rows["sha256"].astype(str),
            "label": rows["label"].astype(int),
            "baseline_probability": baseline_probability,
            "candidate_probability": candidate_probability,
        }
    )
    if (frame.groupby("sha256")["label"].nunique() > 1).any():
        raise ValueError("A final-test SHA256 appears under conflicting labels")
    return frame.groupby("sha256", sort=True, as_index=False).agg(
        label=("label", "first"),
        baseline_probability=("baseline_probability", "mean"),
        candidate_probability=("candidate_probability", "mean"),
    )


def evaluate_final(config_path: Path, unlock_phrase: str) -> dict[str, Any]:
    if unlock_phrase != UNLOCK_PHRASE:
        raise PermissionError("Explicit G11 final-test unlock phrase is required")
    config = load_config(config_path)
    _validate_config(config)
    frozen_path = Path(config["outputs"]["frozen_protocol"])
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    _verify_frozen_inputs(config_path, config, frozen)
    output_dir = Path(config["outputs"]["evaluation_dir"])
    final_metrics = output_dir / "metrics.json"
    if final_metrics.exists():
        raise FileExistsError("G11 final evaluation is already complete and cannot be rerun")
    ensure_dirs(output_dir)
    marker_path = output_dir / "FINAL_TEST_UNLOCKED.json"
    protocol_sha = _sha256(frozen_path)
    technical_resume_used = marker_path.exists()
    if technical_resume_used:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("protocol_sha256") != protocol_sha:
            raise ValueError("Final-test resume protocol mismatch")
    else:
        marker = {
            "status": "in_progress",
            "unlocked_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": protocol_sha,
            "technical_resume_only": True,
        }
        _atomic_json(marker_path, marker)

    evaluation = config["evaluation"]
    checkpoint_paths = {
        "g7": Path(config["models"]["baseline_checkpoint"]),
        **{
            f"g9_seed_{seed}": Path(path)
            for seed, path in zip((42, 43, 44), config["models"]["candidate_checkpoints"], strict=True)
        },
    }
    results = {}
    read_datasets = []
    for dataset_index, (dataset_name, settings) in enumerate(config["final_data"]["datasets"].items()):
        manifest_path = Path(settings["manifest"])
        rows = _validate_final_manifest(manifest_path, dataset_name, frozen["final_dataset_contract"][dataset_name])
        read_datasets.append(dataset_name)
        manifest_sha = _sha256(manifest_path)
        logits = {}
        cache_dir = output_dir / "cache" / dataset_name
        for model_name, checkpoint_path in checkpoint_paths.items():
            cache_path = cache_dir / f"{model_name}.npz"
            if cache_path.exists():
                cached = np.load(cache_path)
                if str(cached["manifest_sha256"].item()) != manifest_sha or str(cached["checkpoint_sha256"].item()) != _sha256(checkpoint_path):
                    raise ValueError(f"G11 prediction cache identity mismatch: {cache_path}")
                model_logits = cached["logits"].astype(np.float32)
            else:
                model_logits = _predict_logits(
                    checkpoint_path,
                    manifest_path,
                    batch_size=int(evaluation["batch_size"]),
                    num_workers=int(evaluation["num_workers"]),
                    device_name=str(evaluation["device"]),
                    mixed_precision=bool(evaluation["mixed_precision"]),
                )
                _atomic_npz(
                    cache_path,
                    logits=model_logits,
                    manifest_sha256=np.asarray(manifest_sha),
                    checkpoint_sha256=np.asarray(_sha256(checkpoint_path)),
                )
            if model_logits.shape != (len(rows),) or not np.isfinite(model_logits).all():
                raise ValueError(f"Invalid final logits for {dataset_name}/{model_name}")
            logits[model_name] = model_logits
        baseline_probability = probabilities_from_logits(logits["g7"], frozen["protocol"]["baseline_temperature"])
        candidate_logit = np.mean([logits[f"g9_seed_{seed}"] for seed in (42, 43, 44)], axis=0)
        candidate_probability = probabilities_from_logits(candidate_logit, frozen["protocol"]["candidate_temperature"])
        labels = rows["label"].to_numpy(dtype=np.int64)
        clusters = rows["sha256"].astype(str).to_numpy()
        ranking = {"baseline": _ranking(labels, baseline_probability), "candidate": _ranking(labels, candidate_probability)}
        ranking_checks = [
            {"name": name, "baseline": ranking["baseline"][name], "candidate": ranking["candidate"][name], "passed": ranking["candidate"][name] >= ranking["baseline"][name]}
            for name in ("roc_auc", "pr_auc", "partial_auc_fpr_0_05_standardized")
        ]
        modes = {}
        for mode_index, (mode_name, mode) in enumerate(frozen["protocol"]["modes"].items()):
            baseline_metrics = _score_metrics(labels, baseline_probability, float(mode["baseline_threshold"]), clusters, int(evaluation["bootstrap_samples"]), int(evaluation["bootstrap_seed"]) + dataset_index * 100 + mode_index * 10)
            candidate_metrics = _score_metrics(labels, candidate_probability, float(mode["candidate_threshold"]), clusters, int(evaluation["bootstrap_samples"]), int(evaluation["bootstrap_seed"]) + dataset_index * 100 + mode_index * 10 + 1)
            checks = [
                {"name": "fpr", "baseline": baseline_metrics["fpr"], "candidate": candidate_metrics["fpr"], "passed": candidate_metrics["fpr"] <= baseline_metrics["fpr"]},
                {"name": "tpr", "baseline": baseline_metrics["tpr"], "candidate": candidate_metrics["tpr"], "passed": candidate_metrics["tpr"] >= baseline_metrics["tpr"]},
                {"name": "f1", "baseline": baseline_metrics["f1"], "candidate": candidate_metrics["f1"], "passed": candidate_metrics["f1"] >= baseline_metrics["f1"]},
            ]
            modes[mode_name] = {"baseline": baseline_metrics, "candidate": candidate_metrics, "checks": checks}
        prediction_rows = rows.copy()
        prediction_rows["g7_probability"] = baseline_probability
        prediction_rows["g11_probability"] = candidate_probability
        prediction_rows.to_csv(output_dir / f"{dataset_name}_predictions.csv", index=False)
        unique = _unique_hash_view(rows, baseline_probability, candidate_probability)
        unique_labels = unique["label"].to_numpy(dtype=np.int64)
        unique_summary = {
            "samples": int(len(unique)),
            "duplicates_removed": int(len(rows) - len(unique)),
            "ranking": {
                "baseline": _ranking(
                    unique_labels,
                    unique["baseline_probability"].to_numpy(dtype=np.float64),
                ),
                "candidate": _ranking(
                    unique_labels,
                    unique["candidate_probability"].to_numpy(dtype=np.float64),
                ),
            },
            "modes": {},
        }
        for mode_name, mode in frozen["protocol"]["modes"].items():
            unique_summary["modes"][mode_name] = {
                "baseline": _score_metrics(
                    unique_labels,
                    unique["baseline_probability"].to_numpy(dtype=np.float64),
                    float(mode["baseline_threshold"]),
                    unique["sha256"].to_numpy(),
                    0,
                    0,
                ),
                "candidate": _score_metrics(
                    unique_labels,
                    unique["candidate_probability"].to_numpy(dtype=np.float64),
                    float(mode["candidate_threshold"]),
                    unique["sha256"].to_numpy(),
                    0,
                    0,
                ),
            }
        results[dataset_name] = {
            "manifest": {"path": manifest_path.as_posix(), "sha256": manifest_sha, "rows": int(len(rows))},
            "ranking": ranking,
            "ranking_checks": ranking_checks,
            "modes": modes,
            "unique_sha256_analysis": unique_summary,
        }
    passed = all(
        item["passed"]
        for dataset in results.values()
        for item in (
            *dataset["ranking_checks"],
            *[check for mode in dataset["modes"].values() for check in mode["checks"]],
        )
    )
    report = {
        "algorithm": ALGORITHM,
        "decision": "promote" if passed else "do_not_promote",
        "protocol": {"path": frozen_path.as_posix(), "sha256": protocol_sha},
        "datasets": results,
        "final_datasets_read": read_datasets,
        "technical_resume_used": technical_resume_used,
    }
    _atomic_json(final_metrics, report)
    marker["status"] = "complete"
    marker["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    marker["metrics_sha256"] = _sha256(final_metrics)
    _atomic_json(marker_path, marker)
    print(json.dumps({"decision": report["decision"], "final_datasets_read": read_datasets}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze or run the one-time G11 final evaluation")
    parser.add_argument("mode", choices=("freeze", "evaluate"))
    parser.add_argument("--config", type=Path, default=Path("configs/g11_final_dual_mode.yaml"))
    parser.add_argument("--unlock-final-evaluation", default="")
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze_protocol(args.config)
    else:
        evaluate_final(args.config, args.unlock_final_evaluation)


if __name__ == "__main__":
    main()
