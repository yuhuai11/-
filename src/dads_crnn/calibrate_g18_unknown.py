from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from .config import load_config
from .data_firewall import audit_csv_rows, file_sha256, reject_locked_path
from .model_identification import G7ModelIdentifier
from .panns import PannsCnn14Binary
from .train import resolve_device, set_seed
from .train_g18_model_id import RegistryDataset, _batches, _load_head


PROTOCOL = "g18_p3_unknown_msp_calibration_v1"


def aggregate_recording_logits(
    frame: pd.DataFrame, logits: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    """Average segment logits without requiring a known-class target."""
    if len(frame) != len(logits) or logits.ndim != 2:
        raise ValueError("G18 P3 recording aggregation inputs are misaligned")
    rows: list[dict[str, Any]] = []
    values: list[np.ndarray] = []
    for audio_hash, indices in frame.groupby("audio_sha256", sort=True).indices.items():
        positions = np.asarray(indices, dtype=np.int64)
        metadata = frame.iloc[positions]
        model_ids = metadata["model_id"].astype(str).unique()
        known_flags = metadata["is_known"].astype(bool).unique()
        if len(model_ids) != 1 or len(known_flags) != 1:
            raise ValueError("One G18 recording has conflicting open-set metadata")
        target_values = metadata["target_index"].astype(int).unique()
        if len(target_values) != 1:
            raise ValueError("One G18 recording has conflicting target indices")
        rows.append(
            {
                "audio_sha256": str(audio_hash),
                "model_id": str(model_ids[0]),
                "is_known": bool(known_flags[0]),
                "target_index": int(target_values[0]),
                "segments": int(len(positions)),
            }
        )
        values.append(np.asarray(logits[positions], dtype=np.float64).mean(axis=0))
    if not rows:
        raise ValueError("G18 P3 cannot aggregate an empty manifest")
    return pd.DataFrame(rows), np.stack(values)


def maximum_softmax_scores(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits_tensor = torch.as_tensor(logits, dtype=torch.float64)
    if logits_tensor.ndim != 2 or not torch.isfinite(logits_tensor).all():
        raise ValueError("Invalid G18 P3 logits")
    probabilities = F.softmax(logits_tensor, dim=1)
    confidence, prediction = probabilities.max(dim=1)
    return confidence.numpy(), prediction.numpy().astype(np.int64)


def threshold_metrics(
    known_scores: np.ndarray, unknown_scores: np.ndarray, threshold: float
) -> dict[str, float]:
    known_scores = np.asarray(known_scores, dtype=np.float64)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64)
    if (
        known_scores.size == 0
        or unknown_scores.size == 0
        or not np.isfinite(known_scores).all()
        or not np.isfinite(unknown_scores).all()
    ):
        raise ValueError("Open-set threshold metrics require finite non-empty scores")
    known_acceptance = float(np.mean(known_scores >= threshold))
    unknown_recall = float(np.mean(unknown_scores < threshold))
    return {
        "known_acceptance_rate": known_acceptance,
        "known_rejection_rate": 1.0 - known_acceptance,
        "unknown_recall": unknown_recall,
        "unknown_false_acceptance_rate": 1.0 - unknown_recall,
        "balanced_accuracy": 0.5 * (known_acceptance + unknown_recall),
    }


def select_threshold(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    *,
    minimum_known_acceptance: float,
) -> tuple[float, dict[str, float]]:
    if not (0.0 < minimum_known_acceptance <= 1.0):
        raise ValueError("minimum_known_acceptance must be inside (0, 1]")
    scores = np.concatenate(
        [
            np.asarray(known_scores, dtype=np.float64),
            np.asarray(unknown_scores, dtype=np.float64),
        ]
    )
    candidates = np.unique(
        np.concatenate(
            (
                [np.nextafter(0.0, 1.0)],
                scores[(scores > 0.0) & (scores < 1.0)],
                [np.nextafter(1.0, 0.0)],
            )
        )
    )
    eligible: list[tuple[float, dict[str, float]]] = []
    for threshold in candidates:
        metrics = threshold_metrics(known_scores, unknown_scores, float(threshold))
        if metrics["known_acceptance_rate"] + 1.0e-12 >= minimum_known_acceptance:
            eligible.append((float(threshold), metrics))
    if not eligible:
        raise RuntimeError("No G18 P3 threshold satisfies the known-acceptance constraint")
    return max(
        eligible,
        key=lambda item: (
            item[1]["balanced_accuracy"],
            item[1]["unknown_recall"],
            item[1]["known_acceptance_rate"],
            -item[0],
        ),
    )


def calibration_gate(
    metrics: dict[str, float], gates: dict[str, Any]
) -> tuple[bool, dict[str, dict[str, float | bool]]]:
    requirements = {
        "balanced_accuracy": float(gates["minimum_tune_balanced_accuracy"]),
        "unknown_recall": float(gates["minimum_tune_unknown_recall"]),
        "known_end_to_end_accuracy": float(
            gates["minimum_tune_known_end_to_end_accuracy"]
        ),
        "known_unknown_auroc": float(
            gates["minimum_tune_known_unknown_auroc"]
        ),
    }
    checks = {}
    for name, minimum in requirements.items():
        if not (0.0 <= minimum <= 1.0):
            raise ValueError(f"Invalid G18 P3 gate for {name}")
        value = float(metrics[name])
        checks[name] = {
            "value": value,
            "minimum": minimum,
            "passed": value + 1.0e-12 >= minimum,
        }
    return all(bool(item["passed"]) for item in checks.values()), checks


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    reject_locked_path(path, context="G18 P3 development input")
    return path.resolve(strict=True)


def _verify_inputs(
    config: dict[str, Any], root: Path
) -> tuple[dict[str, Path], dict[str, str], dict[str, Any], dict[str, Any]]:
    input_names = (
        "known_tune",
        "unknown_tune",
        "registry_audit",
        "p1_preflight",
        "p2_checkpoint",
        "p2_summary",
    )
    if any(
        name in config.get("inputs", {})
        for name in ("known_holdout", "unknown_holdout")
    ):
        raise ValueError("G18 P3 must not bind holdout inputs")
    paths = {
        name: _resolve(root, config["inputs"][name]["path"]) for name in input_names
    }
    paths["official_checkpoint"] = _resolve(root, config["model"]["checkpoint_path"])
    paths["g7_checkpoint"] = _resolve(
        root, config["model"]["binary_checkpoint_path"]
    )
    expected = {
        name: str(config["inputs"][name]["sha256"]) for name in input_names
    }
    expected["official_checkpoint"] = str(config["model"]["checkpoint_sha256"])
    expected["g7_checkpoint"] = str(
        config["model"]["binary_checkpoint_sha256"]
    )
    observed = {name: file_sha256(path) for name, path in paths.items()}
    mismatches = {
        name: (expected[name], observed[name])
        for name in expected
        if expected[name] != observed[name]
    }
    if mismatches:
        raise ValueError(f"G18 P3 input SHA256 mismatch: {mismatches}")
    registry = json.loads(paths["registry_audit"].read_text(encoding="utf-8"))
    p2_summary = json.loads(paths["p2_summary"].read_text(encoding="utf-8"))
    if not (
        registry.get("passed") is True
        and registry.get("locked_datasets_read") == []
        and p2_summary.get("passed") is True
        and p2_summary.get("feasibility_gate_passed") is True
        and p2_summary.get("decision") == "proceed_to_unknown_calibration"
        and p2_summary.get("unknown_tune_read") is False
        and p2_summary.get("known_holdout_read") is False
        and p2_summary.get("unknown_holdout_read") is False
    ):
        raise ValueError("G18 P0/P2 prerequisites do not authorize P3")
    return paths, observed, registry, p2_summary


def preflight(config_path: Path, root: Path) -> dict[str, Any]:
    """Validate P3 identities and partitions without reading waveform payloads."""
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected G18 P3 protocol")
    paths, observed, registry, p2_summary = _verify_inputs(config, root)
    required = (
        "model_id",
        "target_index",
        "is_known",
        "audio_sha256",
        "cache_path",
        "cache_index",
    )
    row_counts = {
        name: audit_csv_rows(paths[name], required_columns=required)
        for name in ("known_tune", "unknown_tune")
    }
    known_frame = pd.read_csv(paths["known_tune"])
    unknown_frame = pd.read_csv(paths["unknown_tune"])
    if not known_frame["is_known"].astype(bool).all():
        raise ValueError("known_tune contains unknown rows")
    if unknown_frame["is_known"].astype(bool).any():
        raise ValueError("unknown_tune contains known rows")
    if (unknown_frame["target_index"].astype(int) != -1).any():
        raise ValueError("unknown_tune target indices must be -1")
    known_hashes = set(known_frame["audio_sha256"].astype(str))
    unknown_hashes = set(unknown_frame["audio_sha256"].astype(str))
    if known_hashes & unknown_hashes:
        raise ValueError("G18 P3 tune recording partitions overlap")
    checkpoint = torch.load(
        paths["p2_checkpoint"], map_location="cpu", weights_only=True
    )
    if (
        checkpoint.get("protocol")
        != "g18_p2_seed42_frozen_g7_model_id_feasibility_v1"
        or checkpoint.get("known_models") != list(registry["known_models"])
        or not isinstance(checkpoint.get("head_state"), dict)
    ):
        raise ValueError("G18 P2 checkpoint identity is invalid")
    report = {
        "passed": True,
        "protocol": f"{PROTOCOL}_preflight",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ready_for_calibration": True,
        "audio_payload_read": False,
        "optimizer_step_exercised": False,
        "checkpoint_written": False,
        "formal_calibration_started": False,
        "known_holdout_read": False,
        "unknown_holdout_read": False,
        "locked_datasets_read": [],
        "rows": row_counts,
        "recordings": {
            "known": len(known_hashes),
            "unknown": len(unknown_hashes),
        },
        "unknown_models": sorted(
            unknown_frame["model_id"].astype(str).unique().tolist()
        ),
        "known_unknown_recording_overlap": 0,
        "p2_best_epoch": int(p2_summary["best_epoch"]),
        "inputs": {
            "config_sha256": file_sha256(config_path),
            "implementation_sha256": file_sha256(Path(__file__)),
            **{f"{name}_sha256": value for name, value in observed.items()},
        },
    }
    output_dir = root / str(config["preflight_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


@torch.no_grad()
def infer_logits(
    model: G7ModelIdentifier,
    dataset: RegistryDataset,
    *,
    batch_size: int,
    device: torch.device,
    label: str,
) -> np.ndarray:
    model.eval()
    values = []
    indices = np.arange(len(dataset.frame), dtype=np.int64)
    processed = 0
    for batch_indices in _batches(indices, batch_size):
        waveforms, _ = dataset.batch(batch_indices)
        _, logits = model(waveforms.to(device))
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"G18 P3 produced non-finite {label} logits")
        values.append(logits.float().cpu().numpy())
        processed += len(batch_indices)
        if processed % 1024 == 0 or processed == len(indices):
            print(f"G18 P3 {label}: {processed}/{len(indices)}", flush=True)
    return np.concatenate(values)


def calibrate(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected G18 P3 protocol")
    paths, observed, registry, p2_summary = _verify_inputs(config, root)
    output_dir = root / str(config["output_dir"])
    report_path = output_dir / "calibration.json"
    if report_path.exists():
        raise FileExistsError(
            "Refusing to overwrite completed G18 P3 calibration report"
        )
    preflight_path = root / str(config["preflight_output_dir"]) / "report.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("G18 P3 preflight report is missing")
    preflight_report = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not (
        preflight_report.get("passed") is True
        and preflight_report.get("ready_for_calibration") is True
        and preflight_report.get("audio_payload_read") is False
        and preflight_report.get("formal_calibration_started") is False
        and preflight_report.get("known_holdout_read") is False
        and preflight_report.get("unknown_holdout_read") is False
        and preflight_report.get("inputs", {}).get("config_sha256")
        == file_sha256(config_path)
        and preflight_report.get("inputs", {}).get("implementation_sha256")
        == file_sha256(Path(__file__))
    ):
        raise ValueError("G18 P3 preflight report is invalid or stale")
    required = (
        "model_id",
        "target_index",
        "is_known",
        "audio_sha256",
        "cache_path",
        "cache_index",
    )
    for name in ("known_tune", "unknown_tune"):
        audit_csv_rows(paths[name], required_columns=required)
    known_frame = pd.read_csv(paths["known_tune"])
    unknown_frame = pd.read_csv(paths["unknown_tune"])
    if not known_frame["is_known"].astype(bool).all():
        raise ValueError("known_tune contains unknown rows")
    if unknown_frame["is_known"].astype(bool).any():
        raise ValueError("unknown_tune contains known rows")
    if (unknown_frame["target_index"].astype(int) != -1).any():
        raise ValueError("unknown_tune target indices must be -1")

    seed = int(config["calibration"]["seed"])
    set_seed(seed)
    device = resolve_device(str(config["calibration"]["device"]))
    if device.type != "cuda":
        raise RuntimeError("G18 P3 calibration requires the server CUDA GPU")
    known_models = list(registry["known_models"])
    detector = PannsCnn14Binary(
        initialization=str(config["model"]["initialization"]),
        vendor_dir=str(config["model"]["vendor_dir"]),
        checkpoint_path=paths["official_checkpoint"].as_posix(),
        checkpoint_sha256=observed["official_checkpoint"],
        spec_augment=False,
        frontend_precision="float32",
        binary_checkpoint_path=paths["g7_checkpoint"].as_posix(),
        binary_checkpoint_sha256=observed["g7_checkpoint"],
        trainable_scope="binary_head_only",
    )
    model = G7ModelIdentifier(
        detector,
        embedding_dim=int(config["model_id_head"]["embedding_dim"]),
        classes=len(known_models),
    ).to(device)
    checkpoint = torch.load(paths["p2_checkpoint"], map_location="cpu", weights_only=True)
    if (
        checkpoint.get("protocol")
        != "g18_p2_seed42_frozen_g7_model_id_feasibility_v1"
        or checkpoint.get("known_models") != known_models
    ):
        raise ValueError("G18 P2 checkpoint identity is invalid")
    _load_head(model, checkpoint["head_state"])
    if any(parameter.requires_grad for parameter in model.detector.parameters()):
        raise RuntimeError("G18 P3 detector is not frozen")

    target_samples = int(
        config["data"]["sample_rate"] * config["data"]["clip_seconds"]
    )
    known_data = RegistryDataset(paths["known_tune"], target_samples)
    unknown_data = RegistryDataset(paths["unknown_tune"], target_samples)
    batch_size = int(config["calibration"]["batch_size"])
    known_logits = infer_logits(
        model, known_data, batch_size=batch_size, device=device, label="known_tune"
    )
    unknown_logits = infer_logits(
        model, unknown_data, batch_size=batch_size, device=device, label="unknown_tune"
    )
    known_recordings, known_recording_logits = aggregate_recording_logits(
        known_data.frame, known_logits
    )
    unknown_recordings, unknown_recording_logits = aggregate_recording_logits(
        unknown_data.frame, unknown_logits
    )
    known_scores, known_predictions = maximum_softmax_scores(known_recording_logits)
    unknown_scores, unknown_predictions = maximum_softmax_scores(
        unknown_recording_logits
    )
    threshold, metrics = select_threshold(
        known_scores,
        unknown_scores,
        minimum_known_acceptance=float(
            config["calibration"]["minimum_known_acceptance"]
        ),
    )
    binary_labels = np.concatenate(
        [np.ones(len(known_scores)), np.zeros(len(unknown_scores))]
    )
    binary_scores = np.concatenate([known_scores, unknown_scores])
    metrics["known_unknown_auroc"] = float(
        roc_auc_score(binary_labels, binary_scores)
    )
    known_correct = (
        known_predictions
        == known_recordings["target_index"].to_numpy(dtype=np.int64)
    )
    metrics["known_closed_set_accuracy"] = float(np.mean(known_correct))
    metrics["known_end_to_end_accuracy"] = float(
        np.mean(known_correct & (known_scores >= threshold))
    )
    gate_passed, gate_checks = calibration_gate(metrics, config["gates"])

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "tune_recording_predictions.csv"
    prediction_rows = []
    for frame, scores, predictions in (
        (known_recordings, known_scores, known_predictions),
        (unknown_recordings, unknown_scores, unknown_predictions),
    ):
        for row, score, prediction in zip(
            frame.to_dict("records"), scores, predictions, strict=True
        ):
            prediction_rows.append(
                {
                    **row,
                    "predicted_index": int(prediction),
                    "predicted_model": known_models[int(prediction)],
                    "msp_confidence": float(score),
                    "decision": (
                        "known_uav" if float(score) >= threshold else "unknown_uav"
                    ),
                }
            )
    predictions_temporary = predictions_path.with_suffix(".csv.tmp")
    with predictions_temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    os.replace(predictions_temporary, predictions_path)

    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": (
            "proceed_to_multiseed_replication"
            if gate_passed
            else "stop_model_identification_branch"
        ),
        "calibration_gate_passed": gate_passed,
        "calibration_gate_checks": gate_checks,
        "score": "recording_mean_logits_max_softmax_probability",
        "known_confidence_threshold": threshold,
        "selection_rule": (
            "maximize_tune_balanced_accuracy_subject_to_known_acceptance_constraint"
        ),
        "minimum_known_acceptance": float(
            config["calibration"]["minimum_known_acceptance"]
        ),
        "tune_metrics": metrics,
        "recordings": {
            "known": int(len(known_recordings)),
            "unknown": int(len(unknown_recordings)),
        },
        "unknown_models_read": sorted(
            unknown_recordings["model_id"].astype(str).unique().tolist()
        ),
        "unknown_model_recording_counts": {
            str(key): int(value)
            for key, value in Counter(
                unknown_recordings["model_id"].astype(str)
            ).items()
        },
        "known_holdout_read": False,
        "unknown_holdout_read": False,
        "locked_datasets_read": [],
        "development_datasets_read": ["g18_known_tune", "g18_unknown_tune"],
        "outputs": {
            "tune_recording_predictions": {
                "path": predictions_path.relative_to(root).as_posix(),
                "sha256": file_sha256(predictions_path),
            }
        },
        "inputs": {
            "config_sha256": file_sha256(config_path),
            "implementation_sha256": file_sha256(Path(__file__)),
            "preflight_sha256": file_sha256(preflight_path),
            **{f"{name}_sha256": value for name, value in observed.items()},
            "p2_best_epoch": int(p2_summary["best_epoch"]),
        },
    }
    report_temporary = report_path.with_suffix(".json.tmp")
    report_temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(report_temporary, report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the G18 recording-level MSP Unknown threshold."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g18_unknown_calibration.yaml")
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        preflight(args.config, args.root)
    else:
        calibrate(args.config, args.root)


if __name__ == "__main__":
    main()
