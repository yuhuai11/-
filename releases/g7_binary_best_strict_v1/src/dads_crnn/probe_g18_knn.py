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
    aggregate_recording_logits,
    calibration_gate,
    select_threshold,
)
from .config import load_config
from .data_firewall import audit_csv_rows, file_sha256, reject_locked_path
from .model_identification import G7ModelIdentifier
from .panns import PannsCnn14Binary
from .probe_g18_prototypes import aggregate_recording_embeddings, l2_normalize
from .train import resolve_device, set_seed
from .train_g18_model_id import RegistryDataset, _batches, _load_head


PROTOCOL = "g18_p3c_pca32_class_conditional_knn5_probe_v1"
P2_PROTOCOL = "g18_p2_seed42_frozen_g7_model_id_feasibility_v1"
P3B_PROTOCOL = "g18_p3b_pca32_cosine_prototype_probe_v1"
REQUIRED_COLUMNS = (
    "model_id",
    "target_index",
    "is_known",
    "audio_sha256",
    "cache_path",
    "cache_index",
)


def fit_knn_space(
    train_embeddings: np.ndarray,
    train_targets: np.ndarray,
    *,
    components: int,
    classes: int,
    neighbors: int,
    seed: int,
) -> tuple[PCA, np.ndarray, np.ndarray]:
    train_embeddings = np.asarray(train_embeddings, dtype=np.float64)
    train_targets = np.asarray(train_targets, dtype=np.int64)
    if (
        train_embeddings.ndim != 2
        or len(train_embeddings) != len(train_targets)
        or not np.isfinite(train_embeddings).all()
    ):
        raise ValueError("Invalid G18 P3c kNN training inputs")
    if not (1 <= components < min(train_embeddings.shape)):
        raise ValueError("Invalid G18 P3c PCA dimension")
    if set(train_targets.tolist()) != set(range(classes)):
        raise ValueError("G18 P3c reference classes are incomplete")
    counts = Counter(train_targets.tolist())
    if neighbors <= 0 or min(counts.values()) < neighbors:
        raise ValueError("G18 P3c k exceeds a class reference count")
    pca = PCA(
        n_components=components,
        whiten=False,
        svd_solver="randomized",
        random_state=seed,
    )
    references = l2_normalize(pca.fit_transform(train_embeddings))
    return pca, references, train_targets.copy()


def class_conditional_knn_scores(
    query_embeddings: np.ndarray,
    predictions: np.ndarray,
    *,
    pca: PCA,
    references: np.ndarray,
    reference_targets: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=np.int64)
    references = l2_normalize(references)
    reference_targets = np.asarray(reference_targets, dtype=np.int64)
    queries = l2_normalize(
        pca.transform(np.asarray(query_embeddings, dtype=np.float64))
    )
    if len(queries) != len(predictions) or len(references) != len(reference_targets):
        raise ValueError("G18 P3c kNN inputs are misaligned")
    scores = []
    for query, prediction in zip(queries, predictions, strict=True):
        candidates = references[reference_targets == int(prediction)]
        if len(candidates) < neighbors:
            raise ValueError("G18 P3c predicted class lacks k references")
        similarities = candidates @ query
        top = np.partition(similarities, len(similarities) - neighbors)[-neighbors:]
        scores.append(float(np.mean(top)))
    return np.clip((np.asarray(scores) + 1.0) / 2.0, 0.0, 1.0)


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    reject_locked_path(path, context="G18 P3c development input")
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
        "p3b_summary",
    )
    if any(
        name in config.get("inputs", {})
        for name in ("known_holdout", "unknown_holdout")
    ):
        raise ValueError("G18 P3c must not bind holdout inputs")
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
        raise ValueError(f"G18 P3c input SHA256 mismatch: {mismatches}")
    registry = json.loads(paths["registry_audit"].read_text(encoding="utf-8"))
    p2_summary = json.loads(paths["p2_summary"].read_text(encoding="utf-8"))
    p3b_summary = json.loads(paths["p3b_summary"].read_text(encoding="utf-8"))
    if not (
        registry.get("passed") is True
        and registry.get("locked_datasets_read") == []
        and p2_summary.get("passed") is True
        and p2_summary.get("feasibility_gate_passed") is True
        and p2_summary.get("known_holdout_read") is False
        and p2_summary.get("unknown_holdout_read") is False
        and p3b_summary.get("passed") is True
        and p3b_summary.get("protocol") == P3B_PROTOCOL
        and p3b_summary.get("probe_gate_passed") is False
        and p3b_summary.get("decision") == "stop_prototype_open_set_branch"
        and p3b_summary.get("known_holdout_read") is False
        and p3b_summary.get("unknown_holdout_read") is False
        and p3b_summary.get("locked_datasets_read") == []
    ):
        raise ValueError("G18 prerequisites do not authorize P3c")
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
                raise ValueError(f"G18 P3c recording overlap: {left} vs {right}")
    return frames


def preflight(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected G18 P3c protocol")
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
        raise ValueError("G18 P3c P2 checkpoint identity is invalid")
    recording_counts = {
        name: int(frame["audio_sha256"].nunique())
        for name, frame in frames.items()
    }
    train_recordings = (
        frames["known_train"]
        .drop_duplicates("audio_sha256")
        .groupby("target_index")
        .size()
    )
    components = int(config["knn"]["pca_components"])
    neighbors = int(config["knn"]["neighbors"])
    if not (1 <= components < recording_counts["known_train"]):
        raise ValueError("G18 P3c PCA components exceed train recordings")
    if neighbors <= 0 or int(train_recordings.min()) < neighbors:
        raise ValueError("G18 P3c k exceeds per-class train recordings")
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
        "recordings": recording_counts,
        "known_train_recordings_per_class": {
            str(int(key)): int(value) for key, value in train_recordings.items()
        },
        "unknown_models": sorted(
            frames["unknown_tune"]["model_id"].astype(str).unique().tolist()
        ),
        "all_recording_overlaps": 0,
        "pca_components": components,
        "neighbors": neighbors,
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
def infer_outputs(
    model: G7ModelIdentifier,
    dataset: RegistryDataset,
    *,
    batch_size: int,
    device: torch.device,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    embeddings = []
    logits_values = []
    indices = np.arange(len(dataset.frame), dtype=np.int64)
    processed = 0
    for batch_indices in _batches(indices, batch_size):
        waveforms, _ = dataset.batch(batch_indices)
        base_embedding = model.detector.extract_embedding(waveforms.to(device))
        embedding = model.embedding_norm(base_embedding)
        logits = model.classifier(embedding)
        if not torch.isfinite(embedding).all() or not torch.isfinite(logits).all():
            raise RuntimeError(f"G18 P3c produced non-finite {label} outputs")
        embeddings.append(embedding.float().cpu().numpy())
        logits_values.append(logits.float().cpu().numpy())
        processed += len(batch_indices)
        if processed % 1024 == 0 or processed == len(indices):
            print(f"G18 P3c {label}: {processed}/{len(indices)}", flush=True)
    return np.concatenate(embeddings), np.concatenate(logits_values)


def probe(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected G18 P3c protocol")
    paths, observed, registry, p2_summary = _verify_inputs(config, root)
    output_dir = root / str(config["output_dir"])
    report_path = output_dir / "summary.json"
    if report_path.exists():
        raise FileExistsError("Refusing to overwrite completed G18 P3c report")
    preflight_path = root / str(config["preflight_output_dir"]) / "report.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("G18 P3c preflight report is missing")
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
        raise ValueError("G18 P3c preflight report is invalid or stale")
    frames = _load_frames(paths)
    seed = int(config["knn"]["seed"])
    set_seed(seed)
    device = resolve_device(str(config["knn"]["device"]))
    if device.type != "cuda":
        raise RuntimeError("G18 P3c probe requires the server CUDA GPU")
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
        raise ValueError("G18 P3c P2 checkpoint protocol is invalid")
    _load_head(model, checkpoint["head_state"])
    if any(parameter.requires_grad for parameter in model.detector.parameters()):
        raise RuntimeError("G18 P3c detector is not frozen")

    target_samples = int(
        config["data"]["sample_rate"] * config["data"]["clip_seconds"]
    )
    datasets = {
        name: RegistryDataset(paths[name], target_samples)
        for name in ("known_train", "known_tune", "unknown_tune")
    }
    batch_size = int(config["knn"]["batch_size"])
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
            raise RuntimeError("G18 P3c recording aggregation order mismatch")

    train_targets = recording_frames["known_train"]["target_index"].to_numpy(
        dtype=np.int64
    )
    pca, references, reference_targets = fit_knn_space(
        recording_embeddings["known_train"],
        train_targets,
        components=int(config["knn"]["pca_components"]),
        classes=len(known_models),
        neighbors=int(config["knn"]["neighbors"]),
        seed=seed,
    )
    known_predictions = recording_logits["known_tune"].argmax(axis=1)
    unknown_predictions = recording_logits["unknown_tune"].argmax(axis=1)
    known_scores = class_conditional_knn_scores(
        recording_embeddings["known_tune"],
        known_predictions,
        pca=pca,
        references=references,
        reference_targets=reference_targets,
        neighbors=int(config["knn"]["neighbors"]),
    )
    unknown_scores = class_conditional_knn_scores(
        recording_embeddings["unknown_tune"],
        unknown_predictions,
        pca=pca,
        references=references,
        reference_targets=reference_targets,
        neighbors=int(config["knn"]["neighbors"]),
    )
    threshold, metrics = select_threshold(
        known_scores,
        unknown_scores,
        minimum_known_acceptance=float(config["knn"]["minimum_known_acceptance"]),
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
                    "knn_score": float(score),
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

    space_path = output_dir / "knn_space.npz"
    space_temporary = space_path.with_suffix(".npz.tmp")
    with space_temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            pca_mean=pca.mean_,
            pca_components=pca.components_,
            references=references,
            reference_targets=reference_targets,
            known_models=np.asarray(known_models),
            neighbors=np.asarray([int(config["knn"]["neighbors"])], dtype=np.int64),
            threshold=np.asarray([threshold], dtype=np.float64),
        )
    os.replace(space_temporary, space_path)
    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": (
            "proceed_to_multiseed_replication"
            if gate_passed
            else "stop_knn_open_set_branch"
        ),
        "probe_gate_passed": gate_passed,
        "probe_gate_checks": gate_checks,
        "classifier": "frozen_p2_recording_mean_logits",
        "score": "rescaled_mean_top5_cosine_within_p2_predicted_class",
        "known_confidence_threshold": threshold,
        "selection_rule": (
            "maximize_tune_balanced_accuracy_subject_to_known_acceptance_constraint"
        ),
        "minimum_known_acceptance": float(
            config["knn"]["minimum_known_acceptance"]
        ),
        "tune_metrics": metrics,
        "knn": {
            "neighbors": int(config["knn"]["neighbors"]),
            "pca_components": int(pca.n_components_),
            "pca_explained_variance_ratio_sum": float(
                pca.explained_variance_ratio_.sum()
            ),
            "reference_partition": "known_train_recordings_only",
            "reference_recordings": int(len(references)),
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
            "knn_space": {
                "path": space_path.relative_to(root).as_posix(),
                "sha256": file_sha256(space_path),
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
        description="Run the G18 class-conditional kNN open-set probe."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g18_knn_probe.yaml")
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
