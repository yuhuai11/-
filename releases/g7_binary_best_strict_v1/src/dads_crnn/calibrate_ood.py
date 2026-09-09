from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ensure_dirs
from .data_firewall import audit_csv_rows, load_forbidden_hashes, reject_consumed_hash
from .evaluate_external import _fast_threshold_metrics, subgroup_metrics
from .external_data import ExternalAudioDataset
from .metrics import binary_metrics
from .prepare_beats_probe import reject_locked_path
from .train import _build_feature_extractor, _build_model, resolve_device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest_audio_hashes(manifest_path: Path) -> dict[str, Any]:
    reject_locked_path(manifest_path)
    audit_csv_rows(manifest_path, required_columns=("path", "sha256"))
    frame = pd.read_csv(manifest_path, usecols=["path", "sha256"])
    if frame.empty or frame[["path", "sha256"]].isna().any().any():
        raise ValueError(f"Manifest lacks complete audio hashes: {manifest_path}")
    unique = frame[["path", "sha256"]].drop_duplicates().reset_index(drop=True)
    if unique["path"].duplicated().any():
        raise ValueError(f"A manifest path has multiple expected hashes: {manifest_path}")
    forbidden_hashes = load_forbidden_hashes()
    for row in unique.itertuples(index=False):
        path = Path(str(row.path))
        reject_locked_path(path)
        reject_consumed_hash(
            row.sha256,
            context=f"{manifest_path}:sha256",
            forbidden_hashes=forbidden_hashes,
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != str(row.sha256):
            raise ValueError(f"Audio hash mismatch: {path}")
    identity = hashlib.sha256(
        unique.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()
    return {
        "manifest": manifest_path.as_posix(),
        "rows": int(len(frame)),
        "unique_files": int(len(unique)),
        "identity_sha256": identity,
        "verified": True,
    }


def probabilities_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    return expit(scaled)


def _prediction_frame(
    rows: pd.DataFrame,
    logits: np.ndarray,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Build a lossless-enough CSV table for the strict calibration-chain audit."""
    predictions = rows.copy()
    # Model inference returns float32 logits.  Casting before ``to_csv`` makes
    # pandas serialize enough significant digits for the gate to reconstruct
    # both probability columns at its strict 1e-10/1e-12 tolerances.
    predictions["logit"] = np.asarray(logits, dtype=np.float64)
    predictions["raw_probability"] = np.asarray(raw_probabilities, dtype=np.float64)
    predictions["calibrated_probability"] = np.asarray(
        calibrated_probabilities, dtype=np.float64
    )
    predictions["selected_prediction"] = (
        np.asarray(calibrated_probabilities, dtype=np.float64) >= float(threshold)
    ).astype(np.int64)
    return predictions


def negative_log_likelihood(labels: np.ndarray, logits: np.ndarray, temperature: float) -> float:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    labels = np.asarray(labels, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, scaled) - labels * scaled))


def fit_temperature(labels: np.ndarray, logits: np.ndarray) -> float:
    result = minimize_scalar(
        lambda log_temperature: negative_log_likelihood(labels, logits, np.exp(log_temperature)),
        bounds=(-5.0, 5.0),
        method="bounded",
        options={"xatol": 1e-7},
    )
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(np.exp(result.x))


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int
) -> tuple[float, list[dict[str, float]]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    predicted = probabilities >= 0.5
    rows = []
    ece = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if not mask.any():
            continue
        confidence = np.where(predicted[mask], probabilities[mask], 1.0 - probabilities[mask])
        accuracy = np.mean(predicted[mask] == labels[mask])
        mean_confidence = float(np.mean(confidence))
        weight = float(mask.mean())
        ece += weight * abs(float(accuracy) - mean_confidence)
        rows.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "samples": int(mask.sum()),
                "accuracy": float(accuracy),
                "mean_confidence": mean_confidence,
                "weight": weight,
            }
        )
    return float(ece), rows


def probability_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    *,
    ece_bins: int,
) -> dict[str, Any]:
    metrics = binary_metrics(labels, probabilities, threshold)
    clipped = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    metrics["nll"] = float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))
    metrics["brier"] = float(np.mean((probabilities - labels) ** 2))
    metrics["ece"], _ = expected_calibration_error(labels, probabilities, ece_bins)
    return metrics


def search_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    target_recall: float,
    target_specificity: float,
) -> tuple[float, bool, list[dict[str, Any]]]:
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    rows = [
        {"threshold": float(threshold), **_fast_threshold_metrics(labels, probabilities, float(threshold))}
        for threshold in candidates
    ]
    feasible = [
        row
        for row in rows
        if row["recall"] >= target_recall and row["specificity"] >= target_specificity
    ]
    if feasible:
        selected = max(feasible, key=lambda row: (row["f1"], row["balanced_accuracy"]))
        return float(selected["threshold"]), True, rows

    def diagnostic_score(row: dict[str, Any]) -> tuple[float, float, float]:
        recall_ratio = min(row["recall"] / target_recall, 1.0) if target_recall else 1.0
        specificity_ratio = (
            min(row["specificity"] / target_specificity, 1.0) if target_specificity else 1.0
        )
        return min(recall_ratio, specificity_ratio), row["f1"], row["balanced_accuracy"]

    selected = max(rows, key=diagnostic_score)
    return float(selected["threshold"]), False, rows


def _predict_logits(
    checkpoint_path: Path,
    manifest_path: Path,
    *,
    batch_size: int,
    num_workers: int,
    device_name: str,
) -> tuple[pd.DataFrame, np.ndarray, dict, int]:
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    seed = int(checkpoint.get("seed", checkpoint_path.parent.name.removeprefix("seed_")))
    model_type = str(config["model"].get("type", "crnn"))
    if model_type == "panns_cnn14_16k":
        from .train_panns import build_model as build_panns_model

        model = build_panns_model(config).to(device)
        feature_extractor = None
    else:
        model = _build_model(config).to(device)
        feature_extractor = _build_feature_extractor(config).to(device)
        feature_extractor.eval()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = ExternalAudioDataset(
        manifest_path,
        sample_rate=int(config["data"]["sample_rate"]),
        clip_seconds=float(config["data"]["clip_seconds"]),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logits = np.empty(len(dataset), dtype=np.float32)
    use_amp = model_type == "panns_cnn14_16k" and device.type == "cuda" and bool(
        config["train"].get("mixed_precision", True)
    )
    with torch.no_grad():
        for waveform, _, indices in tqdm(loader, desc=f"seed {seed} {manifest_path.stem}"):
            waveform = waveform.to(device)
            with torch.amp.autocast(device.type, enabled=use_amp):
                output = model(waveform) if feature_extractor is None else model(
                    feature_extractor(waveform)
                )
            logits[indices.numpy()] = output.cpu().numpy()
    return dataset.rows.copy(), logits, config, seed


def calibrate_checkpoint(
    checkpoint_path: Path,
    tune_manifest: Path,
    holdout_manifest: Path,
    output_dir: Path,
    experiment: str,
    *,
    batch_size: int,
    num_workers: int,
    device_name: str,
    target_recall: float,
    target_specificity: float,
    ece_bins: int,
) -> dict[str, Any]:
    # Reject every external input before reading a manifest, checkpoint, or audio file.
    for path in (checkpoint_path, tune_manifest, holdout_manifest):
        reject_locked_path(path)
    audio_hash_audit = {
        "tune": verify_manifest_audio_hashes(tune_manifest),
        "holdout": verify_manifest_audio_hashes(holdout_manifest),
    }
    tune_rows, tune_logits, config, seed = _predict_logits(
        checkpoint_path,
        tune_manifest,
        batch_size=batch_size,
        num_workers=num_workers,
        device_name=device_name,
    )
    holdout_rows, holdout_logits, _, holdout_seed = _predict_logits(
        checkpoint_path,
        holdout_manifest,
        batch_size=batch_size,
        num_workers=num_workers,
        device_name=device_name,
    )
    if seed != holdout_seed:
        raise ValueError("Tune and holdout checkpoint seeds disagree")
    tune_labels = tune_rows["label"].to_numpy(dtype=np.int64)
    holdout_labels = holdout_rows["label"].to_numpy(dtype=np.int64)
    temperature = fit_temperature(tune_labels, tune_logits)
    tune_raw = probabilities_from_logits(tune_logits)
    tune_calibrated = probabilities_from_logits(tune_logits, temperature)
    holdout_raw = probabilities_from_logits(holdout_logits)
    holdout_calibrated = probabilities_from_logits(holdout_logits, temperature)
    threshold, feasible, search_rows = search_threshold(
        tune_labels,
        tune_calibrated,
        target_recall=target_recall,
        target_specificity=target_specificity,
    )

    scenarios = {
        "raw_050": (holdout_raw, 0.5),
        "temperature_050": (holdout_calibrated, 0.5),
        "temperature_selected": (holdout_calibrated, threshold),
    }
    holdout_metrics = {
        name: probability_metrics(holdout_labels, probabilities, scenario_threshold, ece_bins=ece_bins)
        for name, (probabilities, scenario_threshold) in scenarios.items()
    }
    tune_metrics = {
        "raw_050": probability_metrics(tune_labels, tune_raw, 0.5, ece_bins=ece_bins),
        "temperature_050": probability_metrics(tune_labels, tune_calibrated, 0.5, ece_bins=ece_bins),
        "temperature_selected": probability_metrics(
            tune_labels, tune_calibrated, threshold, ece_bins=ece_bins
        ),
    }
    result = {
        "experiment": experiment,
        "seed": seed,
        "model_type": str(config["model"].get("type", "crnn")),
        "feature_type": str(config["features"].get("type", "log_mel")),
        "temporal_pooling": (
            str(config["model"].get("temporal_pooling", "mean"))
            if str(config["model"].get("type", "crnn")) == "crnn"
            else None
        ),
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "tune_manifest": tune_manifest.as_posix(),
        "tune_manifest_sha256": _sha256(tune_manifest),
        "holdout_manifest": holdout_manifest.as_posix(),
        "holdout_manifest_sha256": _sha256(holdout_manifest),
        "audio_hash_audit": audio_hash_audit,
        "temperature": temperature,
        "selected_threshold": threshold,
        "constraints_feasible_on_tune": feasible,
        "target_recall": target_recall,
        "target_specificity": target_specificity,
        "tune_metrics": tune_metrics,
        "holdout_metrics": holdout_metrics,
        "holdout_subgroups": subgroup_metrics(holdout_rows, holdout_calibrated, threshold),
    }
    run_dir = output_dir / "calibration" / experiment / f"seed_{seed}"
    ensure_dirs(run_dir)
    pd.DataFrame(search_rows).to_csv(run_dir / "threshold_search.csv", index=False)
    _, raw_reliability = expected_calibration_error(holdout_labels, holdout_raw, ece_bins)
    _, calibrated_reliability = expected_calibration_error(
        holdout_labels, holdout_calibrated, ece_bins
    )
    pd.concat(
        [
            pd.DataFrame(raw_reliability).assign(scenario="raw"),
            pd.DataFrame(calibrated_reliability).assign(scenario="temperature"),
        ],
        ignore_index=True,
    ).to_csv(run_dir / "reliability.csv", index=False)
    prediction_hashes = {}
    for split, rows, logits, raw, calibrated in (
        ("tune", tune_rows, tune_logits, tune_raw, tune_calibrated),
        ("holdout", holdout_rows, holdout_logits, holdout_raw, holdout_calibrated),
    ):
        prediction_dir = output_dir / "predictions" / split / experiment / f"seed_{seed}"
        ensure_dirs(prediction_dir)
        predictions = _prediction_frame(rows, logits, raw, calibrated, threshold)
        prediction_path = prediction_dir / "predictions.csv"
        predictions.to_csv(prediction_path, index=False)
        prediction_hashes[split] = _sha256(prediction_path)
    result["prediction_sha256"] = prediction_hashes
    (run_dir / "calibration.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate one frozen checkpoint on val_ood")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tune-manifest", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/val_ood"))
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-recall", type=float, default=0.80)
    parser.add_argument("--target-specificity", type=float, default=0.90)
    parser.add_argument("--ece-bins", type=int, default=15)
    args = parser.parse_args()
    calibrate_checkpoint(
        args.checkpoint,
        args.tune_manifest,
        args.holdout_manifest,
        args.output_dir,
        args.experiment,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
        target_recall=args.target_recall,
        target_specificity=args.target_specificity,
        ece_bins=args.ece_bins,
    )


if __name__ == "__main__":
    main()
