from __future__ import annotations

import argparse
import json
import math
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
from .evaluate_external import _fast_threshold_metrics, stratified_bootstrap_ci, subgroup_metrics
from .evaluate_low_fpr import clopper_pearson, source_macro
from .external_data import ExternalAudioDataset
from .panns import file_sha256
from .train import resolve_device
from .train_panns import build_model


ALGORITHM = "g13_external_confirmation_once_v1"
UNLOCK_PHRASE = "RUN_G13_EXTERNAL_CONFIRMATION_ONCE"
SEEDS = (42, 43, 44)


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
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: pd.DataFrame) -> None:
    ensure_dirs(path.parent)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        rows.to_csv(handle, index=False)
    os.replace(temporary, path)


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("algorithm") != ALGORITHM:
        raise ValueError(f"G13 algorithm must be frozen as {ALGORITHM}")
    contract = config.get("contract", {})
    if tuple(int(value) for value in contract.get("candidate_seeds", [])) != SEEDS:
        raise ValueError("G13 candidate seeds must be [42, 43, 44]")
    if not math.isclose(float(contract.get("target_fpr", -1)), 0.01, abs_tol=1e-12):
        raise ValueError("G13 strict target FPR must be 0.01")
    if contract.get("candidate_ensemble") != "arithmetic_mean_raw_logit":
        raise ValueError("G13 must average raw logits")
    if contract.get("balanced_mode_policy") != "retain_g7":
        raise ValueError("G13 balanced mode must retain G7")
    if len(config.get("models", {}).get("candidate_checkpoints", [])) != 3:
        raise ValueError("G13 requires exactly three candidate checkpoints")


def _item_at_target(rows: list[dict[str, Any]], target: float, name: str) -> dict[str, Any]:
    matches = [row for row in rows if math.isclose(float(row["target_fpr"]), target, abs_tol=1e-12)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name} entry for target FPR {target}")
    return matches[0]


def _derive_protocol(config: dict[str, Any]) -> dict[str, Any]:
    sources = config["frozen_sources"]
    intake = json.loads(Path(sources["intake_audit"]).read_text(encoding="utf-8"))
    if intake.get("passed") is not True or intake.get("model_predictions_read") is not False:
        raise ValueError("G13 intake audit is not a clean, prediction-free pass")
    contract = config["contract"]
    if int(intake.get("samples", -1)) != int(contract["expected_rows"]):
        raise ValueError("G13 intake row count differs from the frozen contract")
    if int(intake.get("unique_sha256", -1)) != int(contract["expected_unique_sha256"]):
        raise ValueError("G13 intake unique-hash count differs from the frozen contract")
    expected_labels = {str(key): int(value) for key, value in contract["expected_labels"].items()}
    if intake.get("label_counts") != expected_labels:
        raise ValueError("G13 intake label counts differ from the frozen contract")
    expected_sources = {
        str(key): int(value) for key, value in contract["expected_source_groups_by_label"].items()
    }
    if intake.get("source_groups_by_label") != expected_sources:
        raise ValueError("G13 intake source counts differ from the frozen contract")

    baseline_calibration = json.loads(
        Path(sources["baseline_calibration"]).read_text(encoding="utf-8")
    )
    baseline_low_fpr = json.loads(Path(sources["baseline_low_fpr"]).read_text(encoding="utf-8"))
    candidate = json.loads(Path(sources["candidate_calibration"]).read_text(encoding="utf-8"))
    candidate_gate = json.loads(Path(sources["candidate_gate"]).read_text(encoding="utf-8"))
    target = float(contract["target_fpr"])
    baseline_point = _item_at_target(
        baseline_low_fpr["operating_points"], target, "baseline operating point"
    )
    if candidate.get("algorithm") != "g12_strict_source_conformal_v1":
        raise ValueError("G13 candidate calibration must be the frozen G12 calibration")
    if candidate.get("protocol", {}).get("locked_final_tests_used") is not False:
        raise ValueError("G12 candidate calibration does not prove final-test isolation")
    if candidate_gate.get("decision") != "strict_candidate_ready_for_new_external_confirmation":
        raise ValueError("G12 candidate did not pass its continuation gate")
    if candidate_gate.get("inputs", {}).get("calibration", {}).get("sha256") != _sha256(
        Path(sources["candidate_calibration"])
    ):
        raise ValueError("G12 gate is not bound to the selected candidate calibration")
    strict = candidate["strict_calibration"]
    return {
        "baseline_temperature": float(baseline_calibration["temperature"]),
        "baseline_threshold": float(baseline_point["calibration"]["threshold"]),
        "candidate_temperature": float(candidate["temperature"]),
        "candidate_threshold": float(strict["threshold"]),
        "candidate_ensemble": "arithmetic_mean_raw_logit_seeds_42_43_44",
        "comparison": "probability >= threshold",
        "balanced_mode_policy": "retain_g7",
        "threshold_or_temperature_refit_on_g13_forbidden": True,
    }


def freeze_protocol(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    _validate_config(config)
    output = Path(config["outputs"]["frozen_protocol"])
    if output.exists():
        raise FileExistsError(f"G13 protocol is already frozen: {output}")
    protocol = _derive_protocol(config)
    sources = config["frozen_sources"]
    checkpoints = [
        Path(config["models"]["baseline_checkpoint"]),
        *[Path(path) for path in config["models"]["candidate_checkpoints"]],
    ]
    source_paths = {
        name: Path(path)
        for name, path in sources.items()
    }
    report = {
        "algorithm": ALGORITHM,
        "status": "frozen_before_g13_model_inference",
        "protocol": protocol,
        "contract": config["contract"],
        "inputs": {
            "config": {"path": config_path.as_posix(), "sha256": _sha256(config_path)},
            "protocol_document": {
                "path": str(config["protocol_document"]),
                "sha256": _sha256(Path(config["protocol_document"])),
            },
            **{
                name: {"path": path.as_posix(), "sha256": _sha256(path)}
                for name, path in source_paths.items()
            },
            "baseline_checkpoint": {
                "path": checkpoints[0].as_posix(),
                "sha256": _sha256(checkpoints[0]),
            },
            "candidate_checkpoints": [
                {"seed": seed, "path": path.as_posix(), "sha256": _sha256(path)}
                for seed, path in zip(SEEDS, checkpoints[1:], strict=True)
            ],
        },
        "implementation": {
            "path": Path(__file__).resolve().as_posix(),
            "sha256": _sha256(Path(__file__)),
        },
        "g13_audio_read": False,
        "g13_model_predictions_read": False,
    }
    _atomic_json(output, report)
    print(json.dumps({"status": report["status"], "protocol": protocol}, indent=2))
    return report


def _verify_frozen(config_path: Path, config: dict[str, Any], frozen: dict[str, Any]) -> None:
    if frozen.get("algorithm") != ALGORITHM:
        raise ValueError("Unexpected G13 frozen protocol algorithm")
    if frozen.get("status") != "frozen_before_g13_model_inference":
        raise ValueError("G13 protocol is not in the frozen pre-inference state")
    checks = {
        "config": config_path,
        "protocol_document": Path(config["protocol_document"]),
        **{name: Path(path) for name, path in config["frozen_sources"].items()},
        "baseline_checkpoint": Path(config["models"]["baseline_checkpoint"]),
    }
    for name, path in checks.items():
        if frozen["inputs"][name]["sha256"] != _sha256(path):
            raise ValueError(f"G13 frozen input changed: {name}")
    candidate_paths = [Path(path) for path in config["models"]["candidate_checkpoints"]]
    if [item["sha256"] for item in frozen["inputs"]["candidate_checkpoints"]] != [
        _sha256(path) for path in candidate_paths
    ]:
        raise ValueError("G13 candidate checkpoint changed after freeze")
    if frozen["implementation"]["sha256"] != _sha256(Path(__file__)):
        raise ValueError("G13 evaluator implementation changed after freeze")
    if frozen["protocol"] != _derive_protocol(config):
        raise ValueError("G13 derived protocol changed after freeze")


def _load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, config, checkpoint


def preflight(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    _validate_config(config)
    frozen = json.loads(Path(config["outputs"]["frozen_protocol"]).read_text(encoding="utf-8"))
    _verify_frozen(config_path, config, frozen)
    device = resolve_device(str(config["evaluation"]["device"]))
    model_paths = [
        ("g7", Path(config["models"]["baseline_checkpoint"])),
        *[
            (f"g9_seed_{seed}", Path(path))
            for seed, path in zip(SEEDS, config["models"]["candidate_checkpoints"], strict=True)
        ],
    ]
    results = []
    raw_logits = []
    for name, path in model_paths:
        model, model_config, checkpoint = _load_model(path, device)
        sample_rate = int(model_config["data"]["sample_rate"])
        samples = int(round(sample_rate * float(model_config["data"]["clip_seconds"])))
        waveform = torch.zeros((1, samples), dtype=torch.float32, device=device)
        with torch.no_grad(), torch.amp.autocast(
            device.type,
            enabled=device.type == "cuda" and bool(config["evaluation"]["mixed_precision"]),
        ):
            output = model(waveform).float()
        finite = bool(torch.isfinite(output).all().item())
        if output.numel() != 1 or not finite:
            raise RuntimeError(f"G13 synthetic preflight failed for {name}")
        logit = float(output.reshape(-1)[0].cpu())
        results.append(
            {
                "model": name,
                "checkpoint_sha256": _sha256(path),
                "checkpoint_seed": int(checkpoint.get("seed", -1)),
                "synthetic_logit": logit,
                "finite": finite,
            }
        )
        if name.startswith("g9_"):
            raw_logits.append(logit)
        del model, checkpoint, waveform, output
        if device.type == "cuda":
            torch.cuda.empty_cache()
    candidate_probability = float(
        probabilities_from_logits(
            np.asarray([np.mean(raw_logits)], dtype=np.float64),
            float(frozen["protocol"]["candidate_temperature"]),
        )[0]
    )
    report = {
        "algorithm": ALGORITHM,
        "passed": True,
        "device": str(device),
        "models": results,
        "candidate_synthetic_probability": candidate_probability,
        "candidate_probability_finite": math.isfinite(candidate_probability),
        "g13_audio_read": False,
        "g13_model_predictions_read": False,
        "frozen_protocol_sha256": _sha256(Path(config["outputs"]["frozen_protocol"])),
    }
    _atomic_json(Path(config["outputs"]["preflight"]), report)
    print(json.dumps(report, indent=2))
    return report


def _validate_manifest(path: Path, frozen: dict[str, Any]) -> pd.DataFrame:
    rows = pd.read_csv(path, low_memory=False)
    required = {"dataset", "path", "label", "sha256", "source_group", "condition"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"G13 manifest is missing columns: {missing}")
    contract = frozen["contract"]
    if len(rows) != int(contract["expected_rows"]):
        raise ValueError("G13 manifest row count changed")
    if rows["sha256"].astype(str).nunique() != int(contract["expected_unique_sha256"]):
        raise ValueError("G13 manifest unique-hash count changed")
    observed_labels = rows["label"].astype(int).value_counts().sort_index().to_dict()
    expected_labels = {int(key): int(value) for key, value in contract["expected_labels"].items()}
    if observed_labels != expected_labels:
        raise ValueError("G13 manifest label counts changed")
    observed_sources = (
        rows.assign(label=rows["label"].astype(int))
        .groupby("label")["source_group"]
        .nunique()
        .to_dict()
    )
    expected_sources = {
        int(key): int(value) for key, value in contract["expected_source_groups_by_label"].items()
    }
    if observed_sources != expected_sources:
        raise ValueError("G13 manifest source-group counts changed")
    if set(rows["dataset"].astype(str)) != {"external_confirmation_v2"}:
        raise ValueError("G13 manifest dataset identity changed")
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
    model, model_config, _ = _load_model(checkpoint_path, device)
    dataset = ExternalAudioDataset(
        manifest_path,
        sample_rate=int(model_config["data"]["sample_rate"]),
        clip_seconds=float(model_config["data"]["clip_seconds"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    logits = np.empty(len(dataset), dtype=np.float32)
    with torch.no_grad():
        for waveforms, _, indices in tqdm(loader, desc=checkpoint_path.parent.name):
            waveforms = waveforms.to(device, non_blocking=True)
            with torch.amp.autocast(
                device.type, enabled=device.type == "cuda" and mixed_precision
            ):
                output = model(waveforms)
            logits[indices.numpy()] = output.float().cpu().numpy()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return logits


def _ranking(labels: np.ndarray, probabilities: np.ndarray, max_fpr: float) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "partial_auc_fpr_0_05_standardized": float(
            roc_auc_score(labels, probabilities, max_fpr=max_fpr)
        ),
    }


def _operating_metrics(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    labels = rows["label"].to_numpy(dtype=np.int64)
    predictions = (probabilities >= threshold).astype(np.int64)
    positive = labels == 1
    negative = labels == 0
    tp = int(np.sum((predictions == 1) & positive))
    fp = int(np.sum((predictions == 1) & negative))
    tn = int(np.sum((predictions == 0) & negative))
    fn = int(np.sum((predictions == 0) & positive))
    source_rows = rows.copy()
    source_rows["uav_source"] = np.where(positive, rows["source_group"].astype(str), "")
    source_rows["background_source"] = np.where(negative, rows["source_group"].astype(str), "")
    result = _fast_threshold_metrics(labels, probabilities, threshold)
    result.update(
        {
            "threshold": float(threshold),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "tpr": float(tp / positive.sum()),
            "fpr": float(fp / negative.sum()),
            "tpr_clopper_pearson_95_ci": clopper_pearson(tp, int(positive.sum())),
            "fpr_clopper_pearson_95_ci": clopper_pearson(fp, int(negative.sum())),
            "source_cluster_bootstrap_95_ci": stratified_bootstrap_ci(
                labels,
                probabilities,
                threshold,
                samples=bootstrap_samples,
                seed=seed,
                clusters=rows["source_group"].astype(str).to_numpy(),
            ),
            "uav_source_macro": source_macro(
                source_rows,
                predictions,
                label=1,
                group_field="uav_source",
                bootstrap_samples=bootstrap_samples,
                seed=seed + 1,
            ),
            "background_source_macro": source_macro(
                source_rows,
                predictions,
                label=0,
                group_field="background_source",
                bootstrap_samples=bootstrap_samples,
                seed=seed + 2,
            ),
            "subgroups": subgroup_metrics(rows, probabilities, threshold),
        }
    )
    return result


def evaluate_once(config_path: Path, unlock_phrase: str) -> dict[str, Any]:
    if unlock_phrase != UNLOCK_PHRASE:
        raise PermissionError("Explicit G13 one-time evaluation unlock phrase is required")
    config = load_config(config_path)
    _validate_config(config)
    frozen_path = Path(config["outputs"]["frozen_protocol"])
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    _verify_frozen(config_path, config, frozen)
    preflight_path = Path(config["outputs"]["preflight"])
    if not preflight_path.exists():
        raise FileNotFoundError("G13 synthetic preflight has not been completed")
    preflight_report = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight_report.get("passed") is not True:
        raise RuntimeError("G13 synthetic preflight did not pass")
    if preflight_report.get("frozen_protocol_sha256") != _sha256(frozen_path):
        raise ValueError("G13 preflight is not bound to the current frozen protocol")

    output_dir = Path(config["outputs"]["evaluation_dir"])
    final_metrics = output_dir / "metrics.json"
    if final_metrics.exists():
        raise FileExistsError("G13 final evaluation is already complete and cannot be rerun")
    ensure_dirs(output_dir)
    marker_path = output_dir / "FINAL_TEST_UNLOCKED.json"
    frozen_sha = _sha256(frozen_path)
    technical_resume_used = marker_path.exists()
    if technical_resume_used:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("frozen_protocol_sha256") != frozen_sha:
            raise ValueError("G13 technical-resume protocol mismatch")
    else:
        marker = {
            "status": "in_progress",
            "unlocked_at_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_protocol_sha256": frozen_sha,
            "technical_resume_only": True,
        }
        _atomic_json(marker_path, marker)

    manifest_path = Path(config["frozen_sources"]["manifest"])
    rows = _validate_manifest(manifest_path, frozen)
    manifest_sha = _sha256(manifest_path)
    paths = {
        "g7": Path(config["models"]["baseline_checkpoint"]),
        **{
            f"g9_seed_{seed}": Path(path)
            for seed, path in zip(SEEDS, config["models"]["candidate_checkpoints"], strict=True)
        },
    }
    logits: dict[str, np.ndarray] = {}
    for name, checkpoint_path in paths.items():
        cache_path = output_dir / "cache" / f"{name}.npz"
        checkpoint_sha = _sha256(checkpoint_path)
        if cache_path.exists():
            with np.load(cache_path) as cached:
                if (
                    str(cached["manifest_sha256"].item()) != manifest_sha
                    or str(cached["checkpoint_sha256"].item()) != checkpoint_sha
                ):
                    raise ValueError(f"G13 cache identity mismatch: {cache_path}")
                model_logits = cached["logits"].astype(np.float32)
        else:
            model_logits = _predict_logits(
                checkpoint_path,
                manifest_path,
                batch_size=int(config["evaluation"]["batch_size"]),
                num_workers=int(config["evaluation"]["num_workers"]),
                device_name=str(config["evaluation"]["device"]),
                mixed_precision=bool(config["evaluation"]["mixed_precision"]),
            )
            _atomic_npz(
                cache_path,
                logits=model_logits,
                manifest_sha256=np.asarray(manifest_sha),
                checkpoint_sha256=np.asarray(checkpoint_sha),
            )
        if model_logits.shape != (len(rows),) or not np.isfinite(model_logits).all():
            raise ValueError(f"Invalid G13 logits for {name}")
        logits[name] = model_logits

    protocol = frozen["protocol"]
    baseline_probability = probabilities_from_logits(
        logits["g7"], float(protocol["baseline_temperature"])
    )
    candidate_logit = np.mean(
        [logits[f"g9_seed_{seed}"].astype(np.float64) for seed in SEEDS], axis=0
    )
    candidate_probability = probabilities_from_logits(
        candidate_logit, float(protocol["candidate_temperature"])
    )
    labels = rows["label"].to_numpy(dtype=np.int64)
    settings = config["evaluation"]
    baseline = _operating_metrics(
        rows,
        baseline_probability,
        float(protocol["baseline_threshold"]),
        int(settings["bootstrap_samples"]),
        int(settings["bootstrap_seed"]),
    )
    candidate = _operating_metrics(
        rows,
        candidate_probability,
        float(protocol["candidate_threshold"]),
        int(settings["bootstrap_samples"]),
        int(settings["bootstrap_seed"]) + 100,
    )
    ranking = {
        "baseline": _ranking(labels, baseline_probability, float(settings["low_fpr_max"])),
        "candidate": _ranking(labels, candidate_probability, float(settings["low_fpr_max"])),
    }
    checks = [
        {"name": "fpr", "baseline": baseline["fpr"], "candidate": candidate["fpr"], "passed": candidate["fpr"] <= baseline["fpr"]},
        {"name": "tpr", "baseline": baseline["tpr"], "candidate": candidate["tpr"], "passed": candidate["tpr"] >= baseline["tpr"]},
        {"name": "f1", "baseline": baseline["f1"], "candidate": candidate["f1"], "passed": candidate["f1"] >= baseline["f1"]},
        {
            "name": "background_source_macro_fpr",
            "baseline": baseline["background_source_macro"]["macro_error"],
            "candidate": candidate["background_source_macro"]["macro_error"],
            "passed": candidate["background_source_macro"]["macro_error"] <= baseline["background_source_macro"]["macro_error"],
        },
        {
            "name": "uav_source_macro_tpr",
            "baseline": baseline["uav_source_macro"]["macro_correct"],
            "candidate": candidate["uav_source_macro"]["macro_correct"],
            "passed": candidate["uav_source_macro"]["macro_correct"] >= baseline["uav_source_macro"]["macro_correct"],
        },
        {
            "name": "partial_auc_fpr_0_05_standardized",
            "baseline": ranking["baseline"]["partial_auc_fpr_0_05_standardized"],
            "candidate": ranking["candidate"]["partial_auc_fpr_0_05_standardized"],
            "passed": ranking["candidate"]["partial_auc_fpr_0_05_standardized"] >= ranking["baseline"]["partial_auc_fpr_0_05_standardized"],
        },
    ]
    prediction_rows = rows.copy()
    prediction_rows["g7_logit"] = logits["g7"]
    for seed in SEEDS:
        prediction_rows[f"g9_seed_{seed}_logit"] = logits[f"g9_seed_{seed}"]
    prediction_rows["g9_ensemble_logit"] = candidate_logit
    prediction_rows["g7_probability"] = baseline_probability
    prediction_rows["g12_probability"] = candidate_probability
    prediction_rows["g7_prediction"] = (
        baseline_probability >= float(protocol["baseline_threshold"])
    ).astype(np.int64)
    prediction_rows["g12_prediction"] = (
        candidate_probability >= float(protocol["candidate_threshold"])
    ).astype(np.int64)
    predictions_path = output_dir / "predictions.csv"
    _atomic_csv(predictions_path, prediction_rows)

    passed = all(check["passed"] for check in checks)
    report = {
        "algorithm": ALGORITHM,
        "decision": "promote_strict_candidate" if passed else "do_not_promote",
        "balanced_mode_policy": "retain_g7",
        "protocol": {"path": frozen_path.as_posix(), "sha256": frozen_sha},
        "manifest": {"path": manifest_path.as_posix(), "sha256": manifest_sha, "rows": len(rows)},
        "predictions": {"path": predictions_path.as_posix(), "sha256": _sha256(predictions_path)},
        "baseline": baseline,
        "candidate": candidate,
        "ranking": ranking,
        "promotion_checks": checks,
        "technical_resume_used": technical_resume_used,
        "g13_audio_read": True,
        "g13_model_predictions_generated": True,
    }
    _atomic_json(final_metrics, report)
    marker.update(
        {
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "metrics_sha256": _sha256(final_metrics),
        }
    )
    _atomic_json(marker_path, marker)
    print(json.dumps({"decision": report["decision"], "promotion_checks": checks}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze, preflight, or run the one-time G13 external confirmation"
    )
    parser.add_argument("mode", choices=("freeze", "preflight", "evaluate"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g13_external_confirmation_eval.yaml")
    )
    parser.add_argument("--unlock-final-evaluation", default="")
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze_protocol(args.config)
    elif args.mode == "preflight":
        preflight(args.config)
    else:
        evaluate_once(args.config, args.unlock_final_evaluation)


if __name__ == "__main__":
    main()
