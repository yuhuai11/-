from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from .dataset import DADSDataset
from .evaluate_g14_counterfactual import CounterfactualPairDataset
from .evaluate_low_fpr import threshold_at_target_fpr
from .metrics import binary_metrics
from .panns import file_sha256
from .train import resolve_device
from .train_panns import build_model


PROTOCOL = "g7_r4_development_domain_and_lowsnr_evaluation_v1"


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = saved["config"]
    model = build_model(config)
    model.load_state_dict(saved["model"], strict=True)
    model.to(device).eval()
    return model, config


def _predict_manifest(
    model: torch.nn.Module,
    config: dict,
    manifest: Path,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    dataset = DADSDataset(
        manifest,
        "val",
        sample_rate=16000,
        clip_seconds=0.5,
        training=False,
        seed=42,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities = []
    amp = bool(config["train"].get("mixed_precision", True)) and device.type == "cuda"
    with torch.no_grad():
        for waveform, _ in loader:
            waveform = waveform.to(device)
            with torch.amp.autocast(device.type, enabled=amp):
                probabilities.append(torch.sigmoid(model(waveform)).float().cpu().numpy())
    return dataset.rows.copy(), np.concatenate(probabilities)


def _recording_metrics(rows: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> dict:
    frame = rows[["recording_group", "label", "domain_bucket"]].copy()
    frame["probability"] = probabilities
    if (frame.groupby("recording_group")["label"].nunique() > 1).any():
        raise ValueError("R4 recording_group contains inconsistent labels")
    grouped = frame.groupby(["recording_group", "label"], as_index=False).agg(
        probability=("probability", "mean"),
        domain_bucket=("domain_bucket", "first"),
    )
    return binary_metrics(
        grouped["label"].to_numpy(dtype=np.int64),
        grouped["probability"].to_numpy(dtype=np.float64),
        threshold,
    )


def _domain_report(rows: pd.DataFrame, probabilities: np.ndarray) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset, group in rows.groupby("dataset_origin", sort=True):
        indices = group.index.to_numpy(dtype=np.int64)
        labels = group["label"].to_numpy(dtype=np.int64)
        scores = probabilities[indices]
        item: dict[str, Any] = {
            "rows": int(len(group)),
            "label": int(labels[0]) if np.unique(labels).size == 1 else "binary",
            "mean_probability": float(scores.mean()),
        }
        if np.unique(labels).size == 2:
            item["threshold_0_5"] = binary_metrics(labels, scores, 0.5)
        elif labels[0] == 1:
            item["recall_at_0_5"] = float((scores >= 0.5).mean())
            source_values = []
            by_source = {}
            for source, source_group in group.groupby("domain_bucket", sort=True):
                source_scores = probabilities[source_group.index.to_numpy(dtype=np.int64)]
                recall = float((source_scores >= 0.5).mean())
                by_source[str(source)] = {"rows": len(source_group), "recall": recall}
                source_values.append(recall)
            item["source_macro_recall_at_0_5"] = float(np.mean(source_values))
            item["worst_source_recall_at_0_5"] = float(np.min(source_values))
            item["by_source"] = by_source
        else:
            item["fpr_at_0_5"] = float((scores >= 0.5).mean())
        item["recording_level_mean_at_0_5"] = _recording_metrics(group, scores, 0.5)
        output[str(dataset)] = item
    return output


def _low_fpr(rows: pd.DataFrame, probabilities: np.ndarray) -> dict[str, Any]:
    tau = rows[rows["dataset_origin"].eq("tau_urban_2022")]
    kielce = rows[rows["dataset_origin"].eq("kielce_17_uav")]
    tau_scores = probabilities[tau.index.to_numpy(dtype=np.int64)]
    kielce_scores = probabilities[kielce.index.to_numpy(dtype=np.int64)]
    combined_labels = np.concatenate(
        [np.zeros(len(tau_scores), dtype=np.int64), np.ones(len(kielce_scores), dtype=np.int64)]
    )
    combined_scores = np.concatenate([tau_scores, kielce_scores])
    points = {}
    for target in (0.01, 0.05):
        calibration = threshold_at_target_fpr(tau_scores, target)
        threshold = float(calibration["threshold"])
        points[str(target)] = {
            "calibration": calibration,
            "kielce_tpr": float((kielce_scores >= threshold).mean()),
            "tau_fpr": float((tau_scores >= threshold).mean()),
        }
    return {
        "roc_auc": float(roc_auc_score(combined_labels, combined_scores)),
        "pr_auc": float(average_precision_score(combined_labels, combined_scores)),
        "standardized_pauc_fpr_le_0_05": float(
            roc_auc_score(combined_labels, combined_scores, max_fpr=0.05)
        ),
        "operating_points": points,
    }


def _counterfactual(
    model: torch.nn.Module,
    config: dict,
    manifest: Path,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    dataset = CounterfactualPairDataset(
        manifest,
        epsilon=1.0e-8,
        peak_limit=0.99,
        share_gain_across_snr=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    negative_probability = np.empty(len(dataset), dtype=np.float32)
    positive_probability = np.empty(len(dataset), dtype=np.float32)
    amp = bool(config["train"].get("mixed_precision", True)) and device.type == "cuda"
    with torch.no_grad():
        for negative, positive, indices in loader:
            # Preserve native 0.5 s semantics and aggregate the two halves back
            # to the audited 1 s counterfactual pair.
            waveforms = torch.cat(
                [negative[:, :8000], negative[:, 8000:], positive[:, :8000], positive[:, 8000:]],
                dim=0,
            ).to(device)
            with torch.amp.autocast(device.type, enabled=amp):
                score = torch.sigmoid(model(waveforms)).float().cpu().numpy()
            count = len(indices)
            negative_probability[indices.numpy()] = (score[:count] + score[count:2 * count]) / 2.0
            positive_probability[indices.numpy()] = (
                score[2 * count:3 * count] + score[3 * count:]
            ) / 2.0
    output = {}
    for snr, group in dataset.rows.groupby("target_snr_db", sort=True):
        idx = group.index.to_numpy(dtype=np.int64)
        negative = negative_probability[idx]
        positive = positive_probability[idx]
        lift = positive - negative
        labels = np.concatenate(
            [np.zeros(len(idx), dtype=np.int64), np.ones(len(idx), dtype=np.int64)]
        )
        scores = np.concatenate([negative, positive])
        operating_points = {}
        for target_fpr in (0.01, 0.05):
            calibration = threshold_at_target_fpr(negative, target_fpr)
            threshold = float(calibration["threshold"])
            operating_points[str(target_fpr)] = {
                "calibration": calibration,
                "positive_tpr": float((positive >= threshold).mean()),
            }
        output[str(float(snr))] = {
            "pairs": int(len(idx)),
            "mean_lift": float(lift.mean()),
            "ordering_accuracy": float((lift > 0).mean()),
            "negative_fpr_at_0_5": float((negative >= 0.5).mean()),
            "positive_tpr_at_0_5": float((positive >= 0.5).mean()),
            "roc_auc": float(roc_auc_score(labels, scores)),
            "pr_auc": float(average_precision_score(labels, scores)),
            "standardized_pauc_fpr_le_0_05": float(
                roc_auc_score(labels, scores, max_fpr=0.05)
            ),
            "operating_points": operating_points,
        }
    return {"by_snr": output, "aggregation": "mean_probability_of_two_half_second_views"}


def _selection(baseline: dict, candidate: dict) -> dict[str, Any]:
    dads_base = baseline["domains"]["dads_halfsec"]["threshold_0_5"]["f1"]
    dads_candidate = candidate["domains"]["dads_halfsec"]["threshold_0_5"]["f1"]
    kielce_base = baseline["domains"]["kielce_17_uav"]["recall_at_0_5"]
    kielce_candidate = candidate["domains"]["kielce_17_uav"]["recall_at_0_5"]
    tau_base = baseline["domains"]["tau_urban_2022"]["fpr_at_0_5"]
    tau_candidate = candidate["domains"]["tau_urban_2022"]["fpr_at_0_5"]
    low_deltas = {
        snr: candidate["low_snr"]["by_snr"][snr]["positive_tpr_at_0_5"]
        - baseline["low_snr"]["by_snr"][snr]["positive_tpr_at_0_5"]
        for snr in ("-15.0", "-10.0")
    }
    checks = {
        "dads_f1_safe": dads_candidate >= dads_base - 0.003,
        "tau_fpr_safe": tau_candidate <= tau_base + 0.005,
        "kielce_recall_gain": kielce_candidate >= kielce_base + 0.005,
        "low_snr_recall_noninferior": min(low_deltas.values()) >= 0.0,
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "deltas": {
            "dads_f1": float(dads_candidate - dads_base),
            "tau_fpr": float(tau_candidate - tau_base),
            "kielce_recall": float(kielce_candidate - kielce_base),
            "low_snr_positive_tpr": {k: float(v) for k, v in low_deltas.items()},
        },
    }


def run(
    baseline_checkpoint: Path,
    candidate_checkpoint: Path,
    manifest: Path,
    pair_manifest: Path,
    output_dir: Path,
    *,
    device_name: str,
    batch_size: int,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    reports = {}
    prediction_tables = {}
    for name, checkpoint in (("baseline", baseline_checkpoint), ("candidate", candidate_checkpoint)):
        model, config = _load_model(checkpoint, device)
        rows, probabilities = _predict_manifest(
            model, config, manifest, device=device, batch_size=batch_size
        )
        reports[name] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "domains": _domain_report(rows, probabilities),
            "low_fpr": _low_fpr(rows, probabilities),
            "low_snr": _counterfactual(
                model, config, pair_manifest, device=device, batch_size=max(1, batch_size // 4)
            ),
        }
        table = rows[["dataset_origin", "recording_group", "domain_bucket", "label"]].copy()
        table["probability"] = probabilities
        prediction_tables[name] = table
    selection = _selection(reports["baseline"], reports["candidate"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in prediction_tables.items():
        table.to_csv(output_dir / f"{name}_predictions.csv", index=False)
    result = {
        "passed": True,
        "protocol": PROTOCOL,
        "decision": (
            "promote_candidate_to_multiseed"
            if selection["eligible"]
            else "do_not_promote_candidate"
        ),
        "selection": selection,
        **reports,
        "inputs": {
            "manifest": {"path": str(manifest), "sha256": file_sha256(manifest)},
            "pair_manifest": {"path": str(pair_manifest), "sha256": file_sha256(pair_manifest)},
        },
        "locked_datasets_read": [],
        "external_dev_holdout_read": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate G7-R4 development domains")
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    result = run(
        args.baseline_checkpoint,
        args.candidate_checkpoint,
        args.manifest,
        args.pair_manifest,
        args.output_dir,
        device_name=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
