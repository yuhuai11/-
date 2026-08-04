from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data_firewall import file_sha256
from .dataset import DADSDataset
from .evaluate_low_fpr import threshold_at_target_fpr
from .train import resolve_device
from .train_panns import build_model


PROTOCOL = "g7_r6_paired_dronenoise_positive_generalization_v1"


def _load_model(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    model = build_model(config)
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval(), config


def _predict(
    checkpoint: Path,
    manifest: Path,
    split: str,
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    model, config = _load_model(checkpoint, device)
    dataset = DADSDataset(
        manifest,
        split,
        sample_rate=16_000,
        clip_seconds=0.5,
        training=False,
        seed=42,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities = []
    amp = bool(config["train"].get("mixed_precision", True)) and device.type == "cuda"
    with torch.no_grad():
        for waveform, _ in loader:
            waveform = waveform.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=amp):
                probabilities.append(torch.sigmoid(model(waveform)).float().cpu().numpy())
    return dataset.rows.copy(), np.concatenate(probabilities).astype(np.float64)


def _positive_recall_report(
    rows: pd.DataFrame, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    if set(rows["label"].astype(int)) != {1}:
        raise ValueError("DroneNoise positive test must contain only label=1")
    frame = rows[["recording_group", "source_group", "uav_subtype"]].copy()
    frame["probability"] = probabilities
    recording = frame.groupby("recording_group", as_index=False).agg(
        probability=("probability", "mean"),
        source_group=("source_group", "first"),
        uav_subtype=("uav_subtype", "first"),
    )
    event = recording.groupby("source_group", as_index=False).agg(
        probability=("probability", "mean"),
        uav_subtype=("uav_subtype", "first"),
    )

    def summarize(values: pd.DataFrame) -> dict[str, Any]:
        return {
            "units": int(len(values)),
            "recall": float((values["probability"] >= threshold).mean()),
            "mean_probability": float(values["probability"].mean()),
            "minimum_probability": float(values["probability"].min()),
            "by_uav_subtype": {
                str(subtype): {
                    "units": int(len(group)),
                    "recall": float((group["probability"] >= threshold).mean()),
                    "mean_probability": float(group["probability"].mean()),
                }
                for subtype, group in values.groupby("uav_subtype", sort=True)
            },
        }

    return {
        "threshold": float(threshold),
        "segment": summarize(frame),
        "recording_mean": summarize(recording),
        "event_mean": summarize(event),
    }


def evaluate(
    baseline_checkpoint: Path,
    candidate_checkpoint: Path,
    positive_manifest: Path,
    calibration_manifest: Path,
    output_dir: Path,
    device_name: str,
    batch_size: int,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    checkpoints = {"g7_r2_control": baseline_checkpoint, "g7_r6_candidate": candidate_checkpoint}
    predictions: dict[str, np.ndarray] = {}
    calibration_scores: dict[str, np.ndarray] = {}
    positive_rows = None
    for name, checkpoint in checkpoints.items():
        rows, scores = _predict(
            checkpoint, positive_manifest, "reusable_positive_test", device, batch_size
        )
        calibration_rows, negative_scores = _predict(
            checkpoint, calibration_manifest, "threshold_calibration", device, batch_size
        )
        if set(calibration_rows["label"].astype(int)) != {0}:
            raise ValueError("Threshold calibration manifest must be negative-only")
        if positive_rows is not None and not rows["segment_sha256"].equals(
            positive_rows["segment_sha256"]
        ):
            raise ValueError("Baseline and candidate DroneNoise prediction order changed")
        positive_rows = rows
        predictions[name] = scores
        calibration_scores[name] = negative_scores

    assert positive_rows is not None
    models = {}
    for name in checkpoints:
        fixed = {
            str(threshold): _positive_recall_report(positive_rows, predictions[name], threshold)
            for threshold in (0.4, 0.5, 0.65)
        }
        low_fpr = {}
        for target in (0.01, 0.05):
            calibration = threshold_at_target_fpr(calibration_scores[name], target)
            low_fpr[str(target)] = {
                "calibration": calibration,
                "dronenoise": _positive_recall_report(
                    positive_rows, predictions[name], float(calibration["threshold"])
                ),
            }
        models[name] = {
            "checkpoint": str(checkpoints[name]),
            "checkpoint_sha256": file_sha256(checkpoints[name]),
            "fixed_thresholds": fixed,
            "calibrated_low_fpr": low_fpr,
        }

    paired = {
        "mean_segment_probability_delta_candidate_minus_control": float(
            np.mean(predictions["g7_r6_candidate"] - predictions["g7_r2_control"])
        ),
        "segments_improved": int(
            np.sum(predictions["g7_r6_candidate"] > predictions["g7_r2_control"])
        ),
        "segments_degraded": int(
            np.sum(predictions["g7_r6_candidate"] < predictions["g7_r2_control"])
        ),
        "segments_equal": int(
            np.sum(predictions["g7_r6_candidate"] == predictions["g7_r2_control"])
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame = positive_rows.copy()
    for name, scores in predictions.items():
        prediction_frame[f"probability_{name}"] = scores
    prediction_path = output_dir / "paired_predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False)
    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "device": str(device),
        "positive_test_is_reusable": True,
        "positive_only_limitation": "recall is reportable; FPR and binary accuracy require negative data",
        "models": models,
        "paired": paired,
        "inputs": {
            "positive_manifest": {"path": str(positive_manifest), "sha256": file_sha256(positive_manifest)},
            "calibration_manifest": {"path": str(calibration_manifest), "sha256": file_sha256(calibration_manifest)},
        },
        "outputs": {
            "predictions": {"path": str(prediction_path), "sha256": file_sha256(prediction_path)}
        },
    }
    report_path = output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired G7-R2/G7-R6 DroneNoise evaluation")
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=Path("artifacts/g7_r2_generalization/ablations/pt_mic_bg_freq/runs/seed_42/best.pt"),
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=Path("artifacts/g7_r6_dronenoise_control/runs/seed_42/best.pt"),
    )
    parser.add_argument(
        "--positive-manifest",
        type=Path,
        default=Path("artifacts/g7_r6_reusable_multicorpus/reusable_positive_test_manifest.csv"),
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=Path("artifacts/g7_r6_reusable_multicorpus/threshold_calibration_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/g7_r6_dronenoise_control/dronenoise_evaluation"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    report = evaluate(
        args.baseline_checkpoint,
        args.candidate_checkpoint,
        args.positive_manifest,
        args.calibration_manifest,
        args.output_dir,
        args.device,
        args.batch_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
