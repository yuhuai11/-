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
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from .calibrate_g18_unknown import (
    calibration_gate,
    select_threshold,
)
from .config import load_config
from .data_firewall import audit_csv_rows, file_sha256, reject_locked_path
from .model_identification import G7ModelIdentifier
from .panns import PannsCnn14Binary
from .train import resolve_device, set_seed
from .train_g18_model_id import RegistryDataset, _batches, _load_head


PROTOCOL = "g18_p3b_pca32_cosine_prototype_probe_v1"
P2_PROTOCOL = "g18_p2_seed42_frozen_g7_model_id_feasibility_v1"
P3_PROTOCOL = "g18_p3_unknown_msp_calibration_v1"
REQUIRED_COLUMNS = (
    "model_id",
    "target_index",
    "is_known",
    "audio_sha256",
    "cache_path",
    "cache_index",
)


def aggregate_recording_embeddings(
    frame: pd.DataFrame, embeddings: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(frame) != len(embeddings) or embeddings.ndim != 2:
        raise ValueError("G18 P3b recording embeddings are misaligned")
    rows: list[dict[str, Any]] = []
    values: list[np.ndarray] = []
    for audio_hash, indices in frame.groupby("audio_sha256", sort=True).indices.items():
        positions = np.asarray(indices, dtype=np.int64)
        metadata = frame.iloc[positions]
        model_ids = metadata["model_id"].astype(str).unique()
        target_values = metadata["target_index"].astype(int).unique()
        known_values = metadata["is_known"].astype(bool).unique()
        if len(model_ids) != 1 or len(target_values) != 1 or len(known_values) != 1:
            raise ValueError("One G18 P3b recording has conflicting metadata")
        rows.append(
            {
                "audio_sha256": str(audio_hash),
                "model_id": str(model_ids[0]),
                "target_index": int(target_values[0]),
                "is_known": bool(known_values[0]),
                "segments": int(len(positions)),
            }
        )
        values.append(
            np.asarray(embeddings[positions], dtype=np.float64).mean(axis=0)
        )
    if not rows:
        raise ValueError("G18 P3b cannot aggregate an empty manifest")
    result = np.stack(values)
    if not np.isfinite(result).all():
        raise ValueError("G18 P3b aggregated non-finite embeddings")
    return pd.DataFrame(rows), result


def l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("G18 P3b normalization requires a finite matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise ValueError("G18 P3b encountered a zero-length embedding")
    return values / norms


def fit_prototype_space(
    train_embeddings: np.ndarray,
    train_targets: np.ndarray,
    *,
    components: int,
    classes: int,
    seed: int,
) -> tuple[PCA, np.ndarray]:
    train_embeddings = np.asarray(train_embeddings, dtype=np.float64)
    train_targets = np.asarray(train_targets, dtype=np.int64)
    if (
        train_embeddings.ndim != 2
        or len(train_embeddings) != len(train_targets)
        or not np.isfinite(train_embeddings).all()
    ):
        raise ValueError("Invalid G18 P3b prototype training inputs")
    if not (1 <= components < min(train_embeddings.shape)):
        raise ValueError("Invalid G18 P3b PCA dimension")
    if set(train_targets.tolist()) != set(range(classes)):
        raise ValueError("G18 P3b prototype classes are incomplete")
    pca = PCA(
        n_components=components,
        whiten=False,
        svd_solver="randomized",
        random_state=seed,
    )
    projected = l2_normalize(pca.fit_transform(train_embeddings))
    prototypes = []
    for target in range(classes):
        class_values = projected[train_targets == target]
        if len(class_values) < 2:
            raise ValueError("G18 P3b needs at least two recordings per class")
        prototypes.append(class_values.mean(axis=0))
    return pca, l2_normalize(np.stack(prototypes))


def prototype_scores(
    embeddings: np.ndarray, pca: PCA, prototypes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    projected = l2_normalize(pca.transform(np.asarray(embeddings, dtype=np.float64)))
    prototypes = l2_normalize(prototypes)
    similarities = projected @ prototypes.T
    predictions = similarities.argmax(axis=1).astype(np.int64)
    maximum = similarities[np.arange(len(similarities)), predictions]
    scores = np.clip((maximum + 1.0) / 2.0, 0.0, 1.0)
    return scores, predictions


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    reject_locked_path(path, context="G18 P3b development input")
    return path.resolve(strict=True)


def _verify_inputs(
    config: dict[str, Any], root: Path
) -> tuple[dict[str, Path], dict[str, str], dict[str, Any], dict[str, Any]]:
    input_names = (
        "known_train",
        "known_tune",
        "unknown_tune",
        "registry_audit",
        "p1_preflight",
        "p2_checkpoint",
        "p2_summary",
        "p3_calibration",
    )
    if any(
        name in config.get("inputs", {})
        for name in ("known_holdout", "unknown_holdout")
    ):
        raise ValueError("G18 P3b must not bind holdout inputs")
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
        raise ValueError(f"G18 P3b input SHA256 mismatch: {mismatches}")
    registry = json.loads(paths["registry_audit"].read_text(encoding="utf-8"))
    p2_summary = json.loads(paths["p2_summary"].read_text(encoding="utf-8"))
    p3_report = json.loads(paths["p3_calibration"].read_text(encoding="utf-8"))
    if not (
        registry.get("passed") is True
        and registry.get("locked_datasets_read") == []
        and p2_summary.get("passed") is True
        and p2_summary.get("feasibility_gate_passed") is True
        and p2_summary.get("known_holdout_read") is False
        and p2_summary.get("unknown_holdout_read") is False
        and p3_report.get("passed") is True
        and p3_report.get("protocol") == P3_PROTOCOL
        and p3_report.get("calibration_gate_passed") is False
        and p3_report.get("decision") == "stop_model_identification_branch"
        and p3_report.get("known_holdout_read") is False
        and p3_report.get("unknown_holdout_read") is False
        and p3_report.get("locked_datasets_read") == []
    ):
        raise ValueError("G18 prerequisites do not authorize P3b")
    return paths, observed, registry, p2_summary


def _load_frames(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    frames = {}
    for name in ("known_train", "known_tune", "unknown_tune"):
        audit_csv_rows(paths[name], required_columns=REQUIRED_COLUMNS)
        frames[name] = pd.read_csv(paths[name])
    for name in ("known_train", "known_tune"):
        if not frames[name]["is_known"].astype(bool).all():
            raise ValueError(f"{name} contains unknown rows")
    if frames["unknown_tune"]["is_known"].astype(bool).any():
        raise ValueError("unknown_tune contains known rows")
    if (frames["unknown_tune"]["target_index"].astype(int) != -1).any():
        raise ValueError("unknown_tune target indices must be -1")
    hash_sets = {
        name: set(frame["audio_sha256"].astype(str))
        for name, frame in frames.items()
    }
    names = tuple(hash_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if hash_sets[left] & hash_sets[right]:
                raise ValueError(f"G18 P3b recording overlap: {left} vs {right}")
    return frames


def preflight(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected G18 P3b protocol")
    paths, observed, registry, p2_summary = _verify_inputs(config, root)
    frames = _load_frames(paths)
    checkpoint = torch.load(
        paths["p2_checkpoint"], map_location="cpu", weights_only=True
    )
    if (
        checkpoint.get("protocol") != P2_PROTOCOL
        or checkpoint.get("known_models") != list(registry["known_models"])
        or not isinstance(checkpoint.get("head_state"), dict)
    ):
        raise ValueError("G18 P3b P2 checkpoint identity is invalid")
    recordings = {
        name: int(frame["audio_sha256"].nunique())
        for name, frame in frames.items()
    }
    components = int(config["prototype"]["pca_components"])
    if not (1 <= components < recordings["known_train"]):
        raise ValueError("G18 P3b PCA components exceed train recordings")
    report = {
        "passed": True,
        "protocol": f"{PROTOCOL}_preflight",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ready_for_probe": True,
        "audio_payload_read": False,
        "optimizer_step_exercised": False,
        "formal_probe_started": False,
        "known_holdout_read": False,
        "unknown_holdout_read": False,
        "locked_datasets_read": [],
        "rows": {name: int(len(frame)) for name, frame in frames.items()},
        "recordings": recordings,
        "unknown_models": sorted(
            frames["unknown_tune"]["model_id"].astype(str).unique().tolist()
        ),
        "all_recording_overlaps": 0,
        "pca_components": components,
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
def infer_embeddings(
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
        embedding = model.detector.extract_embedding(waveforms.to(device))
        embedding = model.embedding_norm(embedding)
        if not torch.isfinite(embedding).all():
            raise RuntimeError(f"G18 P3b produced non-finite {label} embeddings")
        values.append(embedding.float().cpu().numpy())
        processed += len(batch_indices)
        if processed % 1024 == 0 or processed == len(indices):
            print(f"G18 P3b {label}: {processed}/{len(indices)}", flush=True)
    return np.concatenate(values)


def probe(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected G18 P3b protocol")
    paths, observed, registry, p2_summary = _verify_inputs(config, root)
    output_dir = root / str(config["output_dir"])
    report_path = output_dir / "summary.json"
    if report_path.exists():
        raise FileExistsError("Refusing to overwrite completed G18 P3b report")
    preflight_path = root / str(config["preflight_output_dir"]) / "report.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("G18 P3b preflight report is missing")
    preflight_report = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not (
        preflight_report.get("passed") is True
        and preflight_report.get("ready_for_probe") is True
        and preflight_report.get("audio_payload_read") is False
        and preflight_report.get("known_holdout_read") is False
        and preflight_report.get("unknown_holdout_read") is False
        and preflight_report.get("inputs", {}).get("config_sha256")
        == file_sha256(config_path)
        and preflight_report.get("inputs", {}).get("implementation_sha256")
        == file_sha256(Path(__file__))
    ):
        raise ValueError("G18 P3b preflight report is invalid or stale")
    frames = _load_frames(paths)
    seed = int(config["prototype"]["seed"])
    set_seed(seed)
    device = resolve_device(str(config["prototype"]["device"]))
    if device.type != "cuda":
        raise RuntimeError("G18 P3b probe requires the server CUDA GPU")
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
    checkpoint = torch.load(
        paths["p2_checkpoint"], map_location="cpu", weights_only=True
    )
    if checkpoint.get("protocol") != P2_PROTOCOL:
        raise ValueError("G18 P3b P2 checkpoint protocol is invalid")
    _load_head(model, checkpoint["head_state"])
    if any(parameter.requires_grad for parameter in model.detector.parameters()):
        raise RuntimeError("G18 P3b detector is not frozen")

    target_samples = int(
        config["data"]["sample_rate"] * config["data"]["clip_seconds"]
    )
    datasets = {
        name: RegistryDataset(paths[name], target_samples)
        for name in ("known_train", "known_tune", "unknown_tune")
    }
    batch_size = int(config["prototype"]["batch_size"])
    recording_frames = {}
    recording_embeddings = {}
    for name, dataset in datasets.items():
        segment_embeddings = infer_embeddings(
            model,
            dataset,
            batch_size=batch_size,
            device=device,
            label=name,
        )
        recording_frames[name], recording_embeddings[name] = (
            aggregate_recording_embeddings(dataset.frame, segment_embeddings)
        )
    train_targets = recording_frames["known_train"]["target_index"].to_numpy(
        dtype=np.int64
    )
    pca, prototypes = fit_prototype_space(
        recording_embeddings["known_train"],
        train_targets,
        components=int(config["prototype"]["pca_components"]),
        classes=len(known_models),
        seed=seed,
    )
    known_scores, known_predictions = prototype_scores(
        recording_embeddings["known_tune"], pca, prototypes
    )
    unknown_scores, unknown_predictions = prototype_scores(
        recording_embeddings["unknown_tune"], pca, prototypes
    )
    threshold, metrics = select_threshold(
        known_scores,
        unknown_scores,
        minimum_known_acceptance=float(
            config["prototype"]["minimum_known_acceptance"]
        ),
    )
    binary_labels = np.concatenate(
        [np.ones(len(known_scores)), np.zeros(len(unknown_scores))]
    )
    metrics["known_unknown_auroc"] = float(
        roc_auc_score(
            binary_labels, np.concatenate([known_scores, unknown_scores])
        )
    )
    known_targets = recording_frames["known_tune"]["target_index"].to_numpy(
        dtype=np.int64
    )
    known_correct = known_predictions == known_targets
    metrics["known_closed_set_accuracy"] = float(np.mean(known_correct))
    metrics["known_end_to_end_accuracy"] = float(
        np.mean(known_correct & (known_scores >= threshold))
    )
    gate_passed, gate_checks = calibration_gate(metrics, config["gates"])

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "tune_recording_predictions.csv"
    prediction_rows = []
    for name, scores, predictions in (
        ("known_tune", known_scores, known_predictions),
        ("unknown_tune", unknown_scores, unknown_predictions),
    ):
        frame = recording_frames[name]
        for row, score, prediction in zip(
            frame.to_dict("records"), scores, predictions, strict=True
        ):
            prediction_rows.append(
                {
                    **row,
                    "predicted_index": int(prediction),
                    "predicted_model": known_models[int(prediction)],
                    "prototype_score": float(score),
                    "decision": (
                        "known_uav" if float(score) >= threshold else "unknown_uav"
                    ),
                }
            )
    predictions_temporary = predictions_path.with_suffix(".csv.tmp")
    with predictions_temporary.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    os.replace(predictions_temporary, predictions_path)

    probe_path = output_dir / "prototype_space.npz"
    probe_temporary = probe_path.with_suffix(".npz.tmp")
    with probe_temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            pca_mean=pca.mean_,
            pca_components=pca.components_,
            prototypes=prototypes,
            known_models=np.asarray(known_models),
            threshold=np.asarray([threshold], dtype=np.float64),
        )
    os.replace(probe_temporary, probe_path)
    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": (
            "proceed_to_multiseed_replication"
            if gate_passed
            else "stop_prototype_open_set_branch"
        ),
        "probe_gate_passed": gate_passed,
        "probe_gate_checks": gate_checks,
        "score": "rescaled_max_cosine_to_pca32_known_train_prototype",
        "known_confidence_threshold": threshold,
        "selection_rule": (
            "maximize_tune_balanced_accuracy_subject_to_known_acceptance_constraint"
        ),
        "minimum_known_acceptance": float(
            config["prototype"]["minimum_known_acceptance"]
        ),
        "tune_metrics": metrics,
        "pca": {
            "components": int(pca.n_components_),
            "explained_variance_ratio_sum": float(
                pca.explained_variance_ratio_.sum()
            ),
            "fit_partition": "known_train_recordings_only",
        },
        "recordings": {
            name: int(len(frame)) for name, frame in recording_frames.items()
        },
        "unknown_models_read": sorted(
            recording_frames["unknown_tune"]["model_id"]
            .astype(str)
            .unique()
            .tolist()
        ),
        "unknown_model_recording_counts": {
            str(key): int(value)
            for key, value in Counter(
                recording_frames["unknown_tune"]["model_id"].astype(str)
            ).items()
        },
        "optimizer_step_exercised": False,
        "known_holdout_read": False,
        "unknown_holdout_read": False,
        "locked_datasets_read": [],
        "development_datasets_read": [
            "g18_known_train",
            "g18_known_tune",
            "g18_unknown_tune",
        ],
        "outputs": {
            "prototype_space": {
                "path": probe_path.relative_to(root).as_posix(),
                "sha256": file_sha256(probe_path),
            },
            "tune_recording_predictions": {
                "path": predictions_path.relative_to(root).as_posix(),
                "sha256": file_sha256(predictions_path),
            },
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
        description="Run the G18 PCA-cosine prototype open-set probe."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g18_prototype_probe.yaml")
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        preflight(args.config, args.root)
    else:
        probe(args.config, args.root)


if __name__ == "__main__":
    main()
