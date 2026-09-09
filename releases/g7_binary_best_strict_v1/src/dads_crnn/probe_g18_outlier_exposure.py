from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .calibrate_g18_unknown import (
    aggregate_recording_logits,
    calibration_gate,
    select_threshold,
)
from .config import load_config
from .data_firewall import audit_csv_rows, file_sha256, reject_locked_path
from .model_identification import G7ModelIdentifier
from .panns import PannsCnn14Binary
from .probe_g18_knn import infer_outputs
from .probe_g18_prototypes import aggregate_recording_embeddings
from .train import resolve_device, set_seed
from .train_g18_model_id import RegistryDataset, _load_head


PROTOCOL = "g18_p3d_existing_unknown_linear_oe_v1"
P2_PROTOCOL = "g18_p2_seed42_frozen_g7_model_id_feasibility_v1"
P3C_PROTOCOL = "g18_p3c_pca32_class_conditional_knn5_probe_v1"
REQUIRED_COLUMNS = (
    "model_id",
    "target_index",
    "is_known",
    "audio_sha256",
    "cache_path",
    "cache_index",
)


def split_unknown_recordings(
    frame: pd.DataFrame, *, train_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("Unknown OE train fraction must be inside (0, 1)")
    train_hashes: set[str] = set()
    calibration_hashes: set[str] = set()
    for model, group in frame.groupby("model_id", sort=True):
        hashes = sorted(
            group["audio_sha256"].astype(str).unique().tolist(),
            key=lambda value: hashlib.sha256(
                f"{seed}:{model}:{value}".encode("utf-8")
            ).hexdigest(),
        )
        if len(hashes) < 4:
            raise ValueError("Unknown model has too few recordings for OE splitting")
        train_count = int(round(len(hashes) * train_fraction))
        train_count = min(max(train_count, 1), len(hashes) - 1)
        train_hashes.update(hashes[:train_count])
        calibration_hashes.update(hashes[train_count:])
    if train_hashes & calibration_hashes:
        raise RuntimeError("Unknown OE recording split overlaps")
    train = frame[frame["audio_sha256"].astype(str).isin(train_hashes)].copy()
    calibration = frame[
        frame["audio_sha256"].astype(str).isin(calibration_hashes)
    ].copy()
    if len(train) + len(calibration) != len(frame):
        raise RuntimeError("Unknown OE split lost rows")
    return train.reset_index(drop=True), calibration.reset_index(drop=True)


def fit_linear_ood_head(
    known_train: np.ndarray,
    unknown_train: np.ndarray,
    *,
    components: int,
    regularization_c: float,
    seed: int,
) -> tuple[PCA, StandardScaler, LogisticRegression]:
    known_train = np.asarray(known_train, dtype=np.float64)
    unknown_train = np.asarray(unknown_train, dtype=np.float64)
    if (
        known_train.ndim != 2
        or unknown_train.ndim != 2
        or known_train.shape[1] != unknown_train.shape[1]
        or not np.isfinite(known_train).all()
        or not np.isfinite(unknown_train).all()
    ):
        raise ValueError("Invalid G18 P3d OE training features")
    if not (1 <= components < min(known_train.shape)):
        raise ValueError("Invalid G18 P3d PCA dimension")
    if regularization_c <= 0.0:
        raise ValueError("G18 P3d regularization C must be positive")
    pca = PCA(
        n_components=components,
        whiten=False,
        svd_solver="randomized",
        random_state=seed,
    )
    known_projected = pca.fit_transform(known_train)
    unknown_projected = pca.transform(unknown_train)
    features = np.vstack([known_projected, unknown_projected])
    labels = np.concatenate(
        [np.ones(len(known_projected)), np.zeros(len(unknown_projected))]
    )
    scaler = StandardScaler()
    standardized = scaler.fit_transform(features)
    classifier = LogisticRegression(
        C=regularization_c,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=seed,
    )
    classifier.fit(standardized, labels)
    return pca, scaler, classifier


def known_scores(
    embeddings: np.ndarray,
    pca: PCA,
    scaler: StandardScaler,
    classifier: LogisticRegression,
) -> np.ndarray:
    features = scaler.transform(pca.transform(np.asarray(embeddings, dtype=np.float64)))
    index = int(np.flatnonzero(classifier.classes_ == 1)[0])
    scores = classifier.predict_proba(features)[:, index]
    if not np.isfinite(scores).all():
        raise ValueError("G18 P3d produced non-finite Known probabilities")
    return scores


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    reject_locked_path(path, context="G18 P3d development input")
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
        "p3c_summary",
    )
    if any(
        name in config.get("inputs", {})
        for name in ("known_holdout", "unknown_holdout")
    ):
        raise ValueError("G18 P3d must not bind holdout inputs")
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
        raise ValueError(f"G18 P3d input SHA256 mismatch: {mismatches}")
    registry = json.loads(paths["registry_audit"].read_text(encoding="utf-8"))
    p2_summary = json.loads(paths["p2_summary"].read_text(encoding="utf-8"))
    p3c_summary = json.loads(paths["p3c_summary"].read_text(encoding="utf-8"))
    if not (
        registry.get("passed") is True
        and registry.get("locked_datasets_read") == []
        and p2_summary.get("passed") is True
        and p2_summary.get("feasibility_gate_passed") is True
        and p2_summary.get("known_holdout_read") is False
        and p2_summary.get("unknown_holdout_read") is False
        and p3c_summary.get("passed") is True
        and p3c_summary.get("protocol") == P3C_PROTOCOL
        and p3c_summary.get("probe_gate_passed") is False
        and p3c_summary.get("decision") == "stop_knn_open_set_branch"
        and p3c_summary.get("known_holdout_read") is False
        and p3c_summary.get("unknown_holdout_read") is False
        and p3c_summary.get("locked_datasets_read") == []
    ):
        raise ValueError("G18 prerequisites do not authorize P3d")
    return paths, observed, registry, p2_summary


def _read_and_audit(path: Path) -> pd.DataFrame:
    audit_csv_rows(path, required_columns=REQUIRED_COLUMNS)
    return pd.read_csv(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(temporary, path)


def preflight(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected G18 P3d protocol")
    paths, observed, registry, p2_summary = _verify_inputs(config, root)
    known_train = _read_and_audit(paths["known_train"])
    known_tune = _read_and_audit(paths["known_tune"])
    unknown = _read_and_audit(paths["unknown_tune"])
    if not known_train["is_known"].astype(bool).all():
        raise ValueError("known_train contains unknown rows")
    if not known_tune["is_known"].astype(bool).all():
        raise ValueError("known_tune contains unknown rows")
    if unknown["is_known"].astype(bool).any():
        raise ValueError("unknown_tune contains known rows")
    unknown_oe, unknown_calibration = split_unknown_recordings(
        unknown,
        train_fraction=float(config["outlier_exposure"]["unknown_train_fraction"]),
        seed=int(config["outlier_exposure"]["split_seed"]),
    )
    hash_sets = {
        "known_train": set(known_train["audio_sha256"].astype(str)),
        "known_tune": set(known_tune["audio_sha256"].astype(str)),
        "unknown_oe_train": set(unknown_oe["audio_sha256"].astype(str)),
        "unknown_calibration": set(
            unknown_calibration["audio_sha256"].astype(str)
        ),
    }
    names = tuple(hash_sets)
    overlaps = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlaps[f"{left}_vs_{right}"] = len(hash_sets[left] & hash_sets[right])
    if any(overlaps.values()):
        raise ValueError(f"G18 P3d recording overlap: {overlaps}")
    checkpoint = torch.load(
        paths["p2_checkpoint"], map_location="cpu", weights_only=True
    )
    if (
        checkpoint.get("protocol") != P2_PROTOCOL
        or checkpoint.get("known_models") != list(registry["known_models"])
        or not isinstance(checkpoint.get("head_state"), dict)
    ):
        raise ValueError("G18 P3d P2 checkpoint identity is invalid")
    output_dir = root / str(config["preflight_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    oe_path = output_dir / "unknown_oe_train.csv"
    calibration_path = output_dir / "unknown_calibration.csv"
    _atomic_csv(oe_path, unknown_oe)
    _atomic_csv(calibration_path, unknown_calibration)
    report = {
        "passed": True,
        "protocol": f"{PROTOCOL}_preflight",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ready_for_probe": True,
        "audio_payload_read": False,
        "formal_probe_started": False,
        "known_holdout_read": False,
        "unknown_holdout_read": False,
        "locked_datasets_read": [],
        "split": {
            "seed": int(config["outlier_exposure"]["split_seed"]),
            "unknown_train_fraction": float(
                config["outlier_exposure"]["unknown_train_fraction"]
            ),
            "recording_overlaps": overlaps,
            "unknown_oe_train_recordings_by_model": {
                str(key): int(value)
                for key, value in unknown_oe.drop_duplicates("audio_sha256")[
                    "model_id"
                ].value_counts().sort_index().items()
            },
            "unknown_calibration_recordings_by_model": {
                str(key): int(value)
                for key, value in unknown_calibration.drop_duplicates(
                    "audio_sha256"
                )["model_id"].value_counts().sort_index().items()
            },
        },
        "outputs": {
            "unknown_oe_train": {
                "path": oe_path.relative_to(root).as_posix(),
                "sha256": file_sha256(oe_path),
            },
            "unknown_calibration": {
                "path": calibration_path.relative_to(root).as_posix(),
                "sha256": file_sha256(calibration_path),
            },
        },
        "p2_best_epoch": int(p2_summary["best_epoch"]),
        "inputs": {
            "config_sha256": file_sha256(config_path),
            "implementation_sha256": file_sha256(Path(__file__)),
            **{f"{name}_sha256": value for name, value in observed.items()},
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def probe(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected G18 P3d protocol")
    paths, observed, registry, p2_summary = _verify_inputs(config, root)
    output_dir = root / str(config["output_dir"])
    report_path = output_dir / "summary.json"
    if report_path.exists():
        raise FileExistsError("Refusing to overwrite completed G18 P3d report")
    preflight_path = root / str(config["preflight_output_dir"]) / "report.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("G18 P3d preflight report is missing")
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
        raise ValueError("G18 P3d preflight report is invalid or stale")
    split_paths = {}
    for name in ("unknown_oe_train", "unknown_calibration"):
        split_path = root / preflight_report["outputs"][name]["path"]
        if file_sha256(split_path) != preflight_report["outputs"][name]["sha256"]:
            raise ValueError(f"G18 P3d split manifest hash changed: {name}")
        split_paths[name] = split_path
    paths.update(split_paths)
    seed = int(config["outlier_exposure"]["seed"])
    set_seed(seed)
    device = resolve_device(str(config["outlier_exposure"]["device"]))
    if device.type != "cuda":
        raise RuntimeError("G18 P3d probe requires the server CUDA GPU")
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
    _load_head(model, checkpoint["head_state"])
    target_samples = int(
        config["data"]["sample_rate"] * config["data"]["clip_seconds"]
    )
    datasets = {
        name: RegistryDataset(paths[name], target_samples)
        for name in (
            "known_train",
            "known_tune",
            "unknown_oe_train",
            "unknown_calibration",
        )
    }
    batch_size = int(config["outlier_exposure"]["batch_size"])
    recording_frames = {}
    recording_embeddings = {}
    recording_logits = {}
    for name, dataset in datasets.items():
        segment_embeddings, segment_logits = infer_outputs(
            model,
            dataset,
            batch_size=batch_size,
            device=device,
            label=name,
        )
        recording_frames[name], recording_embeddings[name] = (
            aggregate_recording_embeddings(dataset.frame, segment_embeddings)
        )
        logit_frame, recording_logits[name] = aggregate_recording_logits(
            dataset.frame, segment_logits
        )
        if (
            recording_frames[name]["audio_sha256"].tolist()
            != logit_frame["audio_sha256"].tolist()
        ):
            raise RuntimeError("G18 P3d recording aggregation order mismatch")
    pca, scaler, classifier = fit_linear_ood_head(
        recording_embeddings["known_train"],
        recording_embeddings["unknown_oe_train"],
        components=int(config["outlier_exposure"]["pca_components"]),
        regularization_c=float(
            config["outlier_exposure"]["logistic_regularization_c"]
        ),
        seed=seed,
    )
    tune_known_scores = known_scores(
        recording_embeddings["known_tune"], pca, scaler, classifier
    )
    tune_unknown_scores = known_scores(
        recording_embeddings["unknown_calibration"], pca, scaler, classifier
    )
    threshold, metrics = select_threshold(
        tune_known_scores,
        tune_unknown_scores,
        minimum_known_acceptance=float(
            config["outlier_exposure"]["minimum_known_acceptance"]
        ),
    )
    labels = np.concatenate(
        [np.ones(len(tune_known_scores)), np.zeros(len(tune_unknown_scores))]
    )
    metrics["known_unknown_auroc"] = float(
        roc_auc_score(
            labels, np.concatenate([tune_known_scores, tune_unknown_scores])
        )
    )
    known_predictions = recording_logits["known_tune"].argmax(axis=1)
    known_targets = recording_frames["known_tune"]["target_index"].to_numpy(
        dtype=np.int64
    )
    correct = known_predictions == known_targets
    metrics["known_closed_set_accuracy"] = float(np.mean(correct))
    metrics["known_end_to_end_accuracy"] = float(
        np.mean(correct & (tune_known_scores >= threshold))
    )
    gate_passed, gate_checks = calibration_gate(metrics, config["gates"])

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "calibration_recording_predictions.csv"
    rows = []
    for name, scores, predictions in (
        ("known_tune", tune_known_scores, known_predictions),
        (
            "unknown_calibration",
            tune_unknown_scores,
            recording_logits["unknown_calibration"].argmax(axis=1),
        ),
    ):
        for row, score, prediction in zip(
            recording_frames[name].to_dict("records"),
            scores,
            predictions,
            strict=True,
        ):
            rows.append(
                {
                    **row,
                    "predicted_index": int(prediction),
                    "predicted_model": known_models[int(prediction)],
                    "known_probability": float(score),
                    "decision": (
                        "known_uav" if float(score) >= threshold else "unknown_uav"
                    ),
                }
            )
    temporary = predictions_path.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, predictions_path)
    model_path = output_dir / "linear_oe_space.npz"
    model_temporary = model_path.with_suffix(".npz.tmp")
    with model_temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            pca_mean=pca.mean_,
            pca_components=pca.components_,
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            logistic_coef=classifier.coef_,
            logistic_intercept=classifier.intercept_,
            known_models=np.asarray(known_models),
            threshold=np.asarray([threshold]),
        )
    os.replace(model_temporary, model_path)
    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": (
            "proceed_to_multiseed_replication"
            if gate_passed
            else "stop_existing_unknown_oe_branch"
        ),
        "probe_gate_passed": gate_passed,
        "probe_gate_checks": gate_checks,
        "classifier": "frozen_p2_model_id_plus_linear_known_unknown_head",
        "known_confidence_threshold": threshold,
        "tune_metrics": metrics,
        "outlier_exposure": {
            "pca_components": int(pca.n_components_),
            "logistic_regularization_c": float(
                config["outlier_exposure"]["logistic_regularization_c"]
            ),
            "known_train_recordings": int(
                len(recording_frames["known_train"])
            ),
            "unknown_oe_train_recordings": int(
                len(recording_frames["unknown_oe_train"])
            ),
            "unknown_calibration_recordings": int(
                len(recording_frames["unknown_calibration"])
            ),
            "unknown_models": sorted(
                recording_frames["unknown_oe_train"]["model_id"]
                .astype(str)
                .unique()
                .tolist()
            ),
        },
        "network_optimizer_step_exercised": False,
        "linear_detector_fitted": True,
        "known_holdout_read": False,
        "unknown_holdout_read": False,
        "locked_datasets_read": [],
        "outputs": {
            "linear_oe_space": {
                "path": model_path.relative_to(root).as_posix(),
                "sha256": file_sha256(model_path),
            },
            "calibration_recording_predictions": {
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
        description="Run G18 existing-Unknown linear outlier exposure."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g18_outlier_exposure.yaml"),
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
