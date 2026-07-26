from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ensure_dirs
from .dataset import DADSDataset
from .prepare_g9_hard_negatives import FIXED_CLASS_MAP, PROTOCOL
from .train import resolve_device
from .train_panns import build_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_guard_manifest(
    manifest_path: Path,
    audit_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = audit.get("outputs", {}).get("hn_guard", {})
    if not audit.get("passed") or audit.get("protocol") != PROTOCOL:
        raise ValueError("G9 audit did not pass the frozen hard-negative protocol")
    if audit.get("locked_datasets_read") != []:
        raise ValueError("G9 audit reports access to a locked final-test dataset")
    if sha256(manifest_path) != expected.get("sha256"):
        raise ValueError("Guard manifest SHA256 does not match the G9 audit")

    rows = pd.read_csv(manifest_path, low_memory=False)
    if len(rows) != int(expected.get("rows", -1)) or len(rows) != 240:
        raise ValueError("G9 guard must contain exactly 240 audited rows")
    required = {
        "label",
        "cache_path",
        "hard_negative_class",
        "recording_group",
        "source_group",
        "esc_fold",
        "sha256",
        "hn_split",
        "background_mix_eligible",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"G9 guard is missing columns: {missing}")
    if set(rows["label"].astype(int)) != {0}:
        raise ValueError("G9 guard must contain negative labels only")
    if set(rows["esc_fold"].astype(int)) != {5} or set(rows["hn_split"].astype(str)) != {
        "guard"
    }:
        raise ValueError("G9 guard must contain only ESC-50 fold 5 guard rows")
    if set(rows["hard_negative_class"].astype(str)) != set(FIXED_CLASS_MAP.values()):
        raise ValueError("G9 guard class set does not match the frozen protocol")
    class_counts = rows.groupby("hard_negative_class").size().to_dict()
    if any(int(value) != 40 for value in class_counts.values()):
        raise ValueError(f"G9 guard requires 40 segments per class: {class_counts}")
    if rows["sha256"].astype(str).duplicated().any():
        raise ValueError("G9 guard contains duplicate audio hashes")
    if rows["cache_path"].astype(str).eq("").any():
        raise ValueError("G9 guard contains an empty cache path")
    normalized_mix = rows["background_mix_eligible"].astype(str).str.strip().str.lower()
    if set(normalized_mix) != {"false"}:
        raise ValueError("G9 guard rows must not be eligible for background mixing")
    return rows, audit


def negative_metrics(rows: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> dict:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(rows),) or not np.isfinite(probabilities).all():
        raise ValueError("Guard probabilities must be a finite vector aligned with the manifest")
    predictions = probabilities >= threshold

    by_recording = (
        pd.DataFrame(
            {
                "recording_group": rows["recording_group"].astype(str),
                "probability": probabilities,
            }
        )
        .groupby("recording_group", sort=True)["probability"]
        .max()
    )
    by_source = (
        pd.DataFrame(
            {
                "source_group": rows["source_group"].astype(str),
                "probability": probabilities,
            }
        )
        .groupby("source_group", sort=True)["probability"]
        .max()
    )
    per_class = []
    for name, indices in rows.groupby("hard_negative_class", sort=True).groups.items():
        selected = np.asarray(list(indices), dtype=np.int64)
        class_predictions = predictions[selected]
        per_class.append(
            {
                "hard_negative_class": str(name),
                "segments": int(selected.size),
                "false_positives": int(class_predictions.sum()),
                "false_positive_rate": float(class_predictions.mean()),
                "mean_probability": float(probabilities[selected].mean()),
                "max_probability": float(probabilities[selected].max()),
            }
        )
    return {
        "threshold": float(threshold),
        "segments": int(len(rows)),
        "false_positives": int(predictions.sum()),
        "segment_false_positive_rate": float(predictions.mean()),
        "recording_groups": int(len(by_recording)),
        "recording_any_false_positives": int((by_recording >= threshold).sum()),
        "recording_any_false_positive_rate": float((by_recording >= threshold).mean()),
        "source_groups": int(len(by_source)),
        "source_any_false_positives": int((by_source >= threshold).sum()),
        "source_any_false_positive_rate": float((by_source >= threshold).mean()),
        "mean_probability": float(probabilities.mean()),
        "max_probability": float(probabilities.max()),
        "per_class": per_class,
    }


def predict_guard(
    checkpoint_path: Path,
    manifest_path: Path,
    *,
    batch_size: int,
    num_workers: int,
    device_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    if str(config.get("model", {}).get("type")) != "panns_cnn14_16k":
        raise ValueError("G9 guard evaluation requires a PANNs Cnn14 16 kHz checkpoint")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset = DADSDataset(
        manifest_path,
        "guard",
        sample_rate=int(config["data"]["sample_rate"]),
        clip_seconds=float(config["data"]["clip_seconds"]),
        training=False,
        seed=int(checkpoint.get("seed", 42)),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    batches = []
    use_amp = device.type == "cuda" and bool(config["train"].get("mixed_precision", True))
    with torch.no_grad():
        for waveforms, _ in tqdm(loader, desc=f"{checkpoint_path.parent.parent.name} guard"):
            waveforms = waveforms.to(device)
            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(waveforms)
            batches.append(torch.sigmoid(logits).cpu().numpy())
    probabilities = np.concatenate(batches).astype(np.float32, copy=False)
    return probabilities, checkpoint


def evaluate_pair(
    baseline_checkpoint: Path,
    candidate_checkpoint: Path,
    manifest_path: Path,
    audit_path: Path,
    output_dir: Path,
    *,
    thresholds: list[float],
    batch_size: int,
    num_workers: int,
    device_name: str,
) -> dict[str, Any]:
    rows, audit = validate_guard_manifest(manifest_path, audit_path)
    baseline_probabilities, baseline = predict_guard(
        baseline_checkpoint,
        manifest_path,
        batch_size=batch_size,
        num_workers=num_workers,
        device_name=device_name,
    )
    candidate_probabilities, candidate = predict_guard(
        candidate_checkpoint,
        manifest_path,
        batch_size=batch_size,
        num_workers=num_workers,
        device_name=device_name,
    )
    candidate_inputs = candidate.get("training_inputs", {})
    if candidate_inputs.get("g9_audit_sha256") != sha256(audit_path):
        raise ValueError("Candidate checkpoint is not bound to this G9 audit")

    baseline_metrics = [
        negative_metrics(rows, baseline_probabilities, threshold) for threshold in thresholds
    ]
    candidate_metrics = [
        negative_metrics(rows, candidate_probabilities, threshold) for threshold in thresholds
    ]
    comparisons = []
    for baseline_item, candidate_item in zip(
        baseline_metrics, candidate_metrics, strict=True
    ):
        comparisons.append(
            {
                "threshold": candidate_item["threshold"],
                "segment_fpr_change": candidate_item["segment_false_positive_rate"]
                - baseline_item["segment_false_positive_rate"],
                "recording_any_fpr_change": candidate_item[
                    "recording_any_false_positive_rate"
                ]
                - baseline_item["recording_any_false_positive_rate"],
                "source_any_fpr_change": candidate_item["source_any_false_positive_rate"]
                - baseline_item["source_any_false_positive_rate"],
                "mean_probability_change": candidate_item["mean_probability"]
                - baseline_item["mean_probability"],
            }
        )
    report = {
        "protocol": "g9_guard_pairwise_negative_evaluation_v1",
        "baseline_seed": int(baseline["seed"]),
        "candidate_seed": int(candidate["seed"]),
        "manifest": {
            "path": manifest_path.as_posix(),
            "sha256": sha256(manifest_path),
            "rows": int(len(rows)),
        },
        "audit": {
            "path": audit_path.as_posix(),
            "sha256": sha256(audit_path),
            "protocol": audit["protocol"],
        },
        "baseline": {
            "checkpoint": baseline_checkpoint.as_posix(),
            "checkpoint_sha256": sha256(baseline_checkpoint),
            "metrics": baseline_metrics,
        },
        "candidate": {
            "checkpoint": candidate_checkpoint.as_posix(),
            "checkpoint_sha256": sha256(candidate_checkpoint),
            "metrics": candidate_metrics,
        },
        "comparison_candidate_minus_baseline": comparisons,
        "locked_datasets_read": [],
    }
    ensure_dirs(output_dir)
    prediction_rows = rows.copy()
    prediction_rows["g7_probability"] = baseline_probabilities
    prediction_rows["g9_probability"] = candidate_probabilities
    prediction_rows.to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate G7 and G9 on the frozen G9 guard")
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.40, 0.50, 0.65])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not args.thresholds or any(not 0.0 < value < 1.0 for value in args.thresholds):
        parser.error("--thresholds must contain values strictly between zero and one")
    result = evaluate_pair(
        args.baseline_checkpoint,
        args.candidate_checkpoint,
        args.manifest,
        args.audit,
        args.output_dir,
        thresholds=sorted(set(args.thresholds)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
    )
    print(json.dumps(result["comparison_candidate_minus_baseline"], indent=2))


if __name__ == "__main__":
    main()
