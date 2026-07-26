from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ensure_dirs
from .external_data import ExternalAudioDataset
from .metrics import binary_metrics
from .train import _build_feature_extractor, _build_model, resolve_device


CI_METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1")


def _fast_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = y_prob >= threshold
    positive = y_true == 1
    negative = ~positive
    tp = int(np.sum(pred & positive))
    tn = int(np.sum(~pred & negative))
    fp = int(np.sum(pred & negative))
    fn = int(np.sum(~pred & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "accuracy": (tp + tn) / len(y_true),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def stratified_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    *,
    samples: int,
    seed: int,
    clusters: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    if samples <= 0:
        return {}
    rng = np.random.default_rng(seed)
    if clusters is None:
        by_class = [[np.array([index], dtype=np.int64) for index in np.flatnonzero(y_true == label)] for label in (0, 1)]
    else:
        clusters = np.asarray(clusters).astype(str)
        by_class = []
        for label in (0, 1):
            label_indices = np.flatnonzero(y_true == label)
            grouped = [label_indices[clusters[label_indices] == cluster] for cluster in np.unique(clusters[label_indices])]
            by_class.append(grouped)
    values = {metric: [] for metric in CI_METRICS}
    for _ in range(samples):
        sampled_groups = []
        for groups in by_class:
            chosen = rng.integers(0, len(groups), size=len(groups))
            sampled_groups.extend(groups[index] for index in chosen)
        selected = np.concatenate(sampled_groups)
        metrics = _fast_threshold_metrics(y_true[selected], y_prob[selected], threshold)
        for metric in CI_METRICS:
            values[metric].append(metrics[metric])
    return {
        metric: {
            "low": float(np.quantile(metric_values, 0.025)),
            "high": float(np.quantile(metric_values, 0.975)),
        }
        for metric, metric_values in values.items()
    }


def subgroup_metrics(rows: pd.DataFrame, y_prob: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    frame = rows[["source_group", "condition", "label"]].copy()
    frame["probability"] = y_prob
    frame["prediction"] = (y_prob >= threshold).astype(np.int64)
    output = []
    for field in ("source_group", "condition"):
        for value, group in frame.groupby(field, dropna=False, sort=True):
            labels = group["label"].to_numpy(dtype=np.int64)
            predictions = group["prediction"].to_numpy(dtype=np.int64)
            positives = labels == 1
            negatives = labels == 0
            output.append(
                {
                    "group_field": field,
                    "group": str(value),
                    "samples": int(len(group)),
                    "positives": int(positives.sum()),
                    "negatives": int(negatives.sum()),
                    "accuracy": float(np.mean(predictions == labels)),
                    "mean_probability": float(group["probability"].mean()),
                    "recall": float(np.mean(predictions[positives] == 1)) if positives.any() else None,
                    "specificity": float(np.mean(predictions[negatives] == 0)) if negatives.any() else None,
                    "false_positives": int(np.sum((predictions == 1) & negatives)),
                    "false_negatives": int(np.sum((predictions == 0) & positives)),
                }
            )
    return output


def _internal_test_metrics(checkpoint: Path, threshold: float) -> dict[str, Any] | None:
    metrics_path = checkpoint.parent / "metrics.json"
    if not metrics_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    candidates = metrics.get("threshold_metrics", [])
    return next((item for item in candidates if abs(float(item["threshold"]) - threshold) < 1e-9), None)


def evaluate_checkpoint(
    checkpoint_path: Path,
    manifest_path: Path,
    output_dir: Path,
    dataset_name: str,
    experiment_name: str,
    training_scale: str,
    *,
    thresholds: list[float],
    primary_threshold: float,
    batch_size: int,
    num_workers: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    device_name: str,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    seed = int(checkpoint.get("seed", checkpoint_path.parent.name.removeprefix("seed_")))
    model = _build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    feature_extractor = _build_feature_extractor(config).to(device)
    feature_extractor.eval()

    dataset = ExternalAudioDataset(
        manifest_path,
        sample_rate=int(config["data"]["sample_rate"]),
        clip_seconds=float(config["data"]["clip_seconds"]),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probabilities = np.empty(len(dataset), dtype=np.float32)
    labels = dataset.rows["label"].to_numpy(dtype=np.int64)
    with torch.no_grad():
        for waveform, _, indices in tqdm(loader, desc=f"{experiment_name} seed {seed} {dataset_name}"):
            waveform = waveform.to(device)
            logits = model(feature_extractor(waveform))
            probabilities[indices.numpy()] = torch.sigmoid(logits).cpu().numpy()

    threshold_metrics = [binary_metrics(labels, probabilities, threshold) for threshold in thresholds]
    primary = next(item for item in threshold_metrics if abs(item["threshold"] - primary_threshold) < 1e-9)
    primary["bootstrap_95_ci"] = stratified_bootstrap_ci(
        labels,
        probabilities,
        primary_threshold,
        samples=bootstrap_samples,
        seed=bootstrap_seed + seed,
        clusters=dataset.rows["sha256"].where(dataset.rows["sha256"].astype(str) != "", dataset.rows["path"]).to_numpy(),
    )
    internal = _internal_test_metrics(checkpoint_path, primary_threshold)
    result = {
        "dataset": dataset_name,
        "experiment": experiment_name,
        "training_scale": training_scale,
        "seed": seed,
        "model_type": str(config["model"].get("type", "crnn")),
        "feature_type": str(config["features"].get("type", "log_mel")),
        "temporal_pooling": (
            str(config["model"].get("temporal_pooling", "mean"))
            if str(config["model"].get("type", "crnn")) == "crnn"
            else None
        ),
        "checkpoint": checkpoint_path.as_posix(),
        "manifest": manifest_path.as_posix(),
        "samples": int(len(dataset)),
        "device": str(device),
        "primary_threshold": primary_threshold,
        "threshold_metrics": threshold_metrics,
        "internal_dads_test_at_primary_threshold": internal,
        "subgroup_metrics": subgroup_metrics(dataset.rows, probabilities, primary_threshold),
    }

    run_dir = output_dir / dataset_name / experiment_name / f"seed_{seed}"
    ensure_dirs(run_dir)
    prediction_rows = dataset.rows.copy()
    prediction_rows["probability"] = probabilities
    prediction_rows["prediction"] = (probabilities >= primary_threshold).astype(np.int64)
    prediction_rows["experiment"] = experiment_name
    prediction_rows["training_scale"] = training_scale
    prediction_rows["seed"] = seed
    prediction_rows.to_csv(run_dir / "predictions.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one frozen checkpoint on one external manifest")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--training-scale", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/external_evaluation/predictions"))
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.40, 0.50, 0.65])
    parser.add_argument("--primary-threshold", type=float, default=0.50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.primary_threshold not in args.thresholds:
        parser.error("--primary-threshold must also appear in --thresholds")
    result = evaluate_checkpoint(
        args.checkpoint,
        args.manifest,
        args.output_dir,
        args.dataset_name,
        args.experiment_name,
        args.training_scale,
        thresholds=args.thresholds,
        primary_threshold=args.primary_threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        device_name=args.device,
    )
    primary = next(item for item in result["threshold_metrics"] if item["threshold"] == result["primary_threshold"])
    print(json.dumps(primary, indent=2))


if __name__ == "__main__":
    main()
