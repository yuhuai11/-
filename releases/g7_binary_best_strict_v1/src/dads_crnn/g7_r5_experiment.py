from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from .config import load_config
from .dataset import DADSDataset
from .evaluate_low_fpr import threshold_at_target_fpr
from .metrics import binary_metrics
from .panns import file_sha256
from .train import resolve_device
from .train_panns import build_model


def load_checkpoint_model(
    path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict, int | None]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config = saved["config"]
    model = build_model(config)
    model.load_state_dict(saved["model"], strict=True)
    model.to(device).eval()
    checkpoint_seed = saved.get("seed")
    return model, config, int(checkpoint_seed) if checkpoint_seed is not None else None


def predict_system(
    checkpoint_paths: list[Path],
    manifest: Path,
    split: str,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, np.ndarray, list[dict[str, Any]]]:
    dataset = DADSDataset(
        manifest,
        split,
        sample_rate=16000,
        clip_seconds=0.5,
        training=False,
        seed=42,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    member_probabilities = []
    identities = []
    for checkpoint_path in checkpoint_paths:
        model, config, checkpoint_seed = load_checkpoint_model(checkpoint_path, device)
        values = []
        amp = bool(config["train"].get("mixed_precision", True)) and device.type == "cuda"
        with torch.no_grad():
            for waveform, _ in loader:
                waveform = waveform.to(device)
                with torch.amp.autocast(device.type, enabled=amp):
                    values.append(torch.sigmoid(model(waveform)).float().cpu().numpy())
        member_probabilities.append(np.concatenate(values))
        identities.append(
            {
                "path": str(checkpoint_path),
                "sha256": file_sha256(checkpoint_path),
                "seed": checkpoint_seed,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    probabilities = np.mean(np.stack(member_probabilities), axis=0).astype(np.float32)
    return dataset.rows.copy(), probabilities, identities


def ranking(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "standardized_pauc_fpr_le_0_05": float(
            roc_auc_score(labels, probabilities, max_fpr=0.05)
        ),
    }


def recording_frame(rows: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    frame = rows[
        ["recording_group", "label", "dataset_origin", "source_group", "uav_subtype", "uav_novelty"]
    ].copy()
    frame["probability"] = probabilities
    if bool((frame.groupby("recording_group")["label"].nunique() > 1).any()):
        raise ValueError("Recording group has inconsistent labels")
    return frame.groupby(["recording_group", "label"], as_index=False).agg(
        probability=("probability", "mean"),
        dataset_origin=("dataset_origin", "first"),
        source_group=("source_group", "first"),
        uav_subtype=("uav_subtype", "first"),
        uav_novelty=("uav_novelty", "first"),
    )


def threshold_report(rows: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> dict:
    labels = rows["label"].to_numpy(dtype=np.int64)
    return binary_metrics(labels, probabilities, threshold)


def subgroup_recall(
    rows: pd.DataFrame, probabilities: np.ndarray, threshold: float, column: str
) -> dict[str, Any]:
    positive = rows[rows["label"].astype(int).eq(1)]
    output = {}
    values = []
    for value, group in positive.groupby(column, sort=True):
        scores = probabilities[group.index.to_numpy(dtype=np.int64)]
        recall = float((scores >= threshold).mean())
        output[str(value)] = {"rows": int(len(group)), "recall": recall}
        values.append(recall)
    return {
        "by_group": output,
        "macro_recall": float(np.mean(values)),
        "worst_group_recall": float(np.min(values)),
    }


def external_report(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    labels = rows["label"].to_numpy(dtype=np.int64)
    report = {
        "rows": int(len(rows)),
        "recordings": int(rows["recording_group"].nunique()),
        "ranking": ranking(labels, probabilities),
        "operating_points": {
            name: threshold_report(rows, probabilities, threshold)
            for name, threshold in thresholds.items()
        },
        "subgroups": {
            name: {
                "uav_subtype": subgroup_recall(rows, probabilities, threshold, "uav_subtype"),
                "uav_novelty": subgroup_recall(rows, probabilities, threshold, "uav_novelty"),
                "source_group": subgroup_recall(rows, probabilities, threshold, "source_group"),
            }
            for name, threshold in thresholds.items()
        },
    }
    recordings = recording_frame(rows, probabilities)
    recording_probabilities = recordings["probability"].to_numpy(dtype=np.float64)
    recording_labels = recordings["label"].to_numpy(dtype=np.int64)
    report["recording_level"] = {
        "rows": int(len(recordings)),
        "ranking": ranking(recording_labels, recording_probabilities),
        "operating_points": {
            name: binary_metrics(recording_labels, recording_probabilities, threshold)
            for name, threshold in thresholds.items()
        },
        "subgroups": {
            name: {
                "uav_subtype": subgroup_recall(
                    recordings, recording_probabilities, threshold, "uav_subtype"
                ),
                "uav_novelty": subgroup_recall(
                    recordings, recording_probabilities, threshold, "uav_novelty"
                ),
                "source_group": subgroup_recall(
                    recordings, recording_probabilities, threshold, "source_group"
                ),
            }
            for name, threshold in thresholds.items()
        },
    }
    return report


def validation_freeze_report(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    target_fprs: list[float],
) -> tuple[dict[str, Any], dict[str, float]]:
    external = rows[rows["dataset_origin"].isin(["kielce_17_uav", "tau_urban_2022"])].copy()
    external_probabilities = probabilities[external.index.to_numpy(dtype=np.int64)]
    tau = rows[rows["dataset_origin"].eq("tau_urban_2022")]
    tau_probabilities = probabilities[tau.index.to_numpy(dtype=np.int64)]
    thresholds = {"fixed_0_5": 0.5}
    calibration = {}
    for target in target_fprs:
        item = threshold_at_target_fpr(tau_probabilities, target)
        key = f"calibrated_fpr_{target:.2f}"
        thresholds[key] = float(item["threshold"])
        calibration[key] = item
    dads = rows[rows["dataset_origin"].eq("dads_halfsec")]
    dads_probabilities = probabilities[dads.index.to_numpy(dtype=np.int64)]
    return (
        {
            "dads_fixed_0_5": threshold_report(dads, dads_probabilities, 0.5),
            "external": external_report(external.reset_index(drop=True), external_probabilities, thresholds),
            "calibration": calibration,
        },
        thresholds,
    )


def load_experiment(path: Path) -> dict:
    config = load_config(path)
    if config.get("protocol") != "g7_r5_locked_external_comparative_test_v1":
        raise ValueError("Unexpected G7-R5 experiment protocol")
    return config
