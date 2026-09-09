from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    top_k_accuracy_score,
)

from .config import load_config
from .data_firewall import file_sha256
from .g19_open_set import (
    ClassConditionalBoundary,
    calibrate_class_thresholds,
    fit_class_conditional_boundary,
    save_class_conditional_boundary,
)
from .g19_recording_data import RecordingExample, RecordingFeatureDataset
from .train import resolve_device, set_seed
from .train_g18_model_id import classification_metrics
from .train_g19_representation import (
    PROTOCOL as REPRESENTATION_PROTOCOL,
    _atomic_json,
    _protocol_source_sha256,
    _read_validated_frames,
    _verify_inputs,
    build_head,
    ensure_feature_cache,
    infer_head,
    resolve_g19_output,
)


PROTOCOL = "g19_p2_class_conditional_open_set_calibration_v1"


def split_unknown_development_recordings(
    frame: pd.DataFrame,
    *,
    calibration_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratify Unknown Tune recordings into threshold-fit and audit subsets."""
    if not (0.0 < calibration_fraction < 1.0):
        raise ValueError("Unknown calibration fraction must be inside (0, 1)")
    calibration_hashes: set[str] = set()
    audit_hashes: set[str] = set()
    for model_id, group in frame.groupby("model_id", sort=True):
        hashes = group["audio_sha256"].astype(str).str.lower().unique().tolist()
        hashes = sorted(
            hashes,
            key=lambda value: hashlib.sha256(
                f"g19:{seed}:{model_id}:{value}".encode("utf-8")
            ).hexdigest(),
        )
        if len(hashes) < 4:
            raise ValueError("Each Unknown Tune model needs at least four recordings")
        calibration_count = int(round(len(hashes) * calibration_fraction))
        calibration_count = min(max(calibration_count, 1), len(hashes) - 1)
        calibration_hashes.update(hashes[:calibration_count])
        audit_hashes.update(hashes[calibration_count:])
    if calibration_hashes & audit_hashes:
        raise RuntimeError("G19 Unknown calibration/audit split overlaps")
    normalized = frame["audio_sha256"].astype(str).str.lower()
    calibration = frame[normalized.isin(calibration_hashes)].copy()
    audit = frame[normalized.isin(audit_hashes)].copy()
    if len(calibration) + len(audit) != len(frame):
        raise RuntimeError("G19 Unknown split lost manifest rows")
    return calibration.reset_index(drop=True), audit.reset_index(drop=True)


def _metadata_mask(
    metadata: tuple[RecordingExample, ...] | list[RecordingExample],
    hashes: set[str],
) -> np.ndarray:
    mask = np.asarray(
        [example.audio_sha256.lower() in hashes for example in metadata],
        dtype=np.bool_,
    )
    if not mask.any():
        raise ValueError("G19 metadata selection produced an empty subset")
    return mask


def _open_set_metrics(
    boundary: ClassConditionalBoundary,
    known_embeddings: np.ndarray,
    known_logits: np.ndarray,
    known_targets: np.ndarray,
    unknown_embeddings: np.ndarray,
    unknown_logits: np.ndarray,
) -> dict[str, Any]:
    known_decision = boundary.score(known_embeddings, known_logits)
    unknown_decision = boundary.score(unknown_embeddings, unknown_logits)
    labels = np.concatenate(
        [
            np.ones(len(known_embeddings), dtype=np.int64),
            np.zeros(len(unknown_embeddings), dtype=np.int64),
        ]
    )
    margins = np.concatenate(
        [
            known_decision.thresholds - known_decision.distances,
            unknown_decision.thresholds - unknown_decision.distances,
        ]
    )
    known_acceptance = float(known_decision.accepted.mean())
    unknown_recall = float((~unknown_decision.accepted).mean())
    correct_model = known_decision.predicted_classes == known_targets
    accepted_correct = known_decision.accepted & correct_model
    accepted_count = int(known_decision.accepted.sum())
    per_predicted_class_known_acceptance: list[dict[str, Any]] = []
    supported_acceptance: list[float] = []
    for class_index, model_id in enumerate(boundary.known_models):
        class_mask = known_decision.predicted_classes == class_index
        support = int(class_mask.sum())
        acceptance = (
            float(known_decision.accepted[class_mask].mean())
            if support
            else None
        )
        if acceptance is not None:
            supported_acceptance.append(acceptance)
        per_predicted_class_known_acceptance.append(
            {
                "class_index": class_index,
                "model_id": model_id,
                "support": support,
                "acceptance_rate": acceptance,
            }
        )
    metrics: dict[str, Any] = {
        "known_acceptance_rate": known_acceptance,
        "known_rejection_rate": 1.0 - known_acceptance,
        "unknown_recall": unknown_recall,
        "unknown_false_acceptance_rate": 1.0 - unknown_recall,
        "balanced_open_set_accuracy": 0.5 * (known_acceptance + unknown_recall),
        "known_correct_and_accepted_rate": float(accepted_correct.mean()),
        "known_classification_accuracy_when_accepted": (
            float(accepted_correct.sum() / accepted_count)
            if accepted_count
            else 0.0
        ),
        "known_conservative_fallback_rate": float(
            known_decision.used_conservative_fallback.mean()
        ),
        "unknown_conservative_fallback_rate": float(
            unknown_decision.used_conservative_fallback.mean()
        ),
        "known_unknown_roc_auc": float(roc_auc_score(labels, margins)),
        "known_unknown_pr_auc": float(average_precision_score(labels, margins)),
        "standardized_pauc_fpr_le_0_05": float(
            roc_auc_score(labels, margins, max_fpr=0.05)
        ),
        "known_recordings": int(len(known_embeddings)),
        "unknown_recordings": int(len(unknown_embeddings)),
        "minimum_supported_predicted_class_known_acceptance": float(
            min(supported_acceptance)
        ),
        "per_predicted_class_known_acceptance": (
            per_predicted_class_known_acceptance
        ),
        "ranking_positive_class": "known",
        "fpr_definition": "unknown_false_acceptance_rate",
    }
    return metrics


def development_gate_checks(
    development_metrics: dict[str, Any],
    per_unknown_model: dict[str, Any],
    gates: dict[str, Any],
) -> tuple[bool, dict[str, dict[str, float | bool]]]:
    if not per_unknown_model:
        raise ValueError("G19 development gate requires per-model Unknown metrics")
    minimum_unknown_model_recall = min(
        float(values["unknown_recall"])
        for values in per_unknown_model.values()
    )
    gate_values = {
        "known_acceptance_rate": (
            development_metrics["known_acceptance_rate"],
            gates["minimum_tune_known_acceptance"],
        ),
        "known_correct_and_accepted_rate": (
            development_metrics["known_correct_and_accepted_rate"],
            gates["minimum_known_correct_and_accepted_rate"],
        ),
        "minimum_supported_predicted_class_known_acceptance": (
            development_metrics[
                "minimum_supported_predicted_class_known_acceptance"
            ],
            gates["minimum_supported_predicted_class_known_acceptance"],
        ),
        "same_unknown_recording_recall": (
            development_metrics["unknown_recall"],
            gates["minimum_same_unknown_recording_recall"],
        ),
        "minimum_same_unknown_model_recall": (
            minimum_unknown_model_recall,
            gates["minimum_same_unknown_model_recall"],
        ),
        "balanced_open_set_accuracy": (
            development_metrics["balanced_open_set_accuracy"],
            gates["minimum_development_balanced_open_set_accuracy"],
        ),
        "known_unknown_roc_auc": (
            development_metrics["known_unknown_roc_auc"],
            gates["minimum_tune_known_unknown_auroc"],
        ),
    }
    checks = {
        name: {
            "observed": float(observed_value),
            "minimum": float(minimum_value),
            "passed": bool(float(observed_value) >= float(minimum_value)),
        }
        for name, (observed_value, minimum_value) in gate_values.items()
    }
    return all(values["passed"] for values in checks.values()), checks


def _prediction_rows(
    *,
    split_name: str,
    metadata: list[RecordingExample] | tuple[RecordingExample, ...],
    logits: np.ndarray,
    embeddings: np.ndarray,
    attention_entropies: np.ndarray,
    boundary: ClassConditionalBoundary,
) -> list[dict[str, Any]]:
    decision = boundary.score(embeddings, logits)
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(metadata):
        predicted_index = int(decision.predicted_classes[index])
        predicted_model = boundary.known_models[predicted_index]
        accepted = bool(decision.accepted[index])
        rows.append(
            {
                "split": split_name,
                "audio_sha256": example.audio_sha256,
                "true_model_id": example.model_id,
                "true_target_index": example.target,
                "predicted_model_id": predicted_model,
                "predicted_target_index": predicted_index,
                "accepted_as_known": accepted,
                "open_set_output": predicted_model if accepted else "UNKNOWN",
                "distance": float(decision.distances[index]),
                "threshold": float(decision.thresholds[index]),
                "known_margin": float(
                    decision.thresholds[index] - decision.distances[index]
                ),
                "used_conservative_fallback": bool(
                    decision.used_conservative_fallback[index]
                ),
                "normalized_attention_entropy": float(
                    attention_entropies[index]
                ),
            }
        )
    return rows


def _atomic_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("Cannot write an empty G19 prediction table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def calibrate(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    paths, observed, registry = _verify_inputs(config, root)
    known_models = [str(value) for value in registry["known_models"]]
    frames = _read_validated_frames(paths, known_models)
    seed = int(config["train"]["seed"])
    set_seed(seed)
    device = resolve_device(str(config["train"]["device"]))
    output_dir = resolve_g19_output(
        root, config["output_dir"], context="G19 representation output"
    )
    calibration_dir = resolve_g19_output(
        root,
        config.get(
            "open_set_output_dir",
            Path(str(config["output_dir"])) / "open_set",
        ),
        context="G19 open-set output",
    )
    boundary_path = (
        calibration_dir / "candidate_class_conditional_boundary_v1.npz"
    )
    predictions_path = calibration_dir / "recording_predictions.csv"
    calibration_summary_path = calibration_dir / "summary.json"
    if any(
        path.exists()
        for path in (boundary_path, predictions_path, calibration_summary_path)
    ):
        raise ValueError(
            "G19 open-set artifacts already exist; refusing to overwrite them"
        )
    configured_checkpoint = config["open_set"].get(
        "representation_checkpoint_path",
        Path(str(config["output_dir"])) / "best.pt",
    )
    checkpoint_path = Path(str(configured_checkpoint))
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    checkpoint_path = checkpoint_path.resolve(strict=False)
    expected_checkpoint = (output_dir / "best.pt").resolve(strict=False)
    if checkpoint_path != expected_checkpoint:
        raise ValueError(
            "G19 open-set calibration must use this run's representation checkpoint"
        )
    summary_path = checkpoint_path.parent / "summary.json"
    if not checkpoint_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("G19 representation training outputs are missing")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    checkpoint_sha256 = file_sha256(checkpoint_path)
    representation_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not (
        checkpoint.get("protocol") == REPRESENTATION_PROTOCOL
        and checkpoint.get("known_models") == known_models
        and representation_summary.get("protocol") == REPRESENTATION_PROTOCOL
        and representation_summary.get("passed") is True
        and representation_summary.get("representation_gate_passed") is True
        and representation_summary.get("identity") == checkpoint.get("identity")
        and representation_summary.get("best_checkpoint_sha256")
        == checkpoint_sha256
        and representation_summary.get("best_epoch") == checkpoint.get("epoch")
        and representation_summary.get("best_tune_recording_metrics")
        == checkpoint.get("tune_recording_metrics")
        and checkpoint.get("architecture") == dict(config["representation"])
        and checkpoint.get("identity", {}).get("source_sha256")
        == _protocol_source_sha256()
        and checkpoint.get("known_holdout_read") is False
        and checkpoint.get("unknown_holdout_read") is False
        and checkpoint.get("locked_datasets_read") == []
        and representation_summary.get("known_holdout_read") is False
        and representation_summary.get("unknown_holdout_read") is False
    ):
        raise ValueError("G19 representation checkpoint is not calibration-ready")
    if checkpoint["identity"].get("config_sha256") != file_sha256(config_path):
        raise ValueError("G19 representation config identity changed")

    detector = None
    feature_dir = resolve_g19_output(
        root,
        config.get("feature_cache_dir", Path(str(config["output_dir"])) / "features"),
        context="G19 feature-cache output",
    )
    feature_arrays: dict[str, np.ndarray] = {}
    feature_metadata: dict[str, dict[str, Any]] = {}
    for name in ("known_train", "known_tune", "unknown_tune"):
        values, metadata, detector = ensure_feature_cache(
            split_name=name,
            manifest_path=paths[name],
            manifest_sha256=observed[name],
            output_dir=output_dir,
            feature_dir=feature_dir,
            workspace_root=root,
            config=config,
            paths=paths,
            observed=observed,
            device=device,
            detector=detector,
        )
        feature_arrays[name] = values
        feature_metadata[name] = metadata
    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    head = build_head(config, len(known_models)).to(device)
    head.load_state_dict(checkpoint["head_state"], strict=True)
    head.eval()
    inference: dict[
        str,
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            list[RecordingExample] | tuple[RecordingExample, ...],
        ],
    ] = {}
    batch_size = max(
        len(known_models) * int(config["train"]["recordings_per_class"]),
        32,
    )
    for name in ("known_train", "known_tune", "unknown_tune"):
        dataset = RecordingFeatureDataset(frames[name], feature_arrays[name])
        inference[name] = infer_head(
            head,
            dataset,
            batch_size=batch_size,
            device=device,
        )

    (
        train_logits,
        train_embeddings,
        train_targets,
        train_entropies,
        train_metadata,
    ) = inference["known_train"]
    (
        tune_logits,
        tune_embeddings,
        tune_targets,
        tune_entropies,
        tune_metadata,
    ) = inference["known_tune"]
    (
        unknown_logits,
        unknown_embeddings,
        _,
        unknown_entropies,
        unknown_metadata,
    ) = inference["unknown_tune"]
    open_config = config["open_set"]
    boundary = fit_class_conditional_boundary(
        train_embeddings,
        train_targets,
        known_models,
        pca_components=int(open_config["pca_components"]),
        initial_known_acceptance=float(
            open_config.get("initial_known_acceptance", 0.95)
        ),
    )

    unknown_calibration, unknown_audit = split_unknown_development_recordings(
        frames["unknown_tune"],
        calibration_fraction=float(
            open_config.get("unknown_calibration_fraction", 0.40)
        ),
        seed=int(open_config.get("split_seed", seed)),
    )
    calibration_hashes = set(
        unknown_calibration["audio_sha256"].astype(str).str.lower()
    )
    audit_hashes = set(unknown_audit["audio_sha256"].astype(str).str.lower())
    calibration_mask = _metadata_mask(unknown_metadata, calibration_hashes)
    audit_mask = _metadata_mask(unknown_metadata, audit_hashes)
    if np.any(calibration_mask & audit_mask) or not np.all(
        calibration_mask | audit_mask
    ):
        raise RuntimeError("G19 Unknown inference split is incomplete or overlapping")

    calibrated, diagnostics = calibrate_class_thresholds(
        boundary,
        tune_embeddings,
        tune_logits,
        unknown_embeddings[calibration_mask],
        unknown_logits[calibration_mask],
        minimum_known_acceptance=float(
            open_config["minimum_known_acceptance"]
        ),
        minimum_known_support=int(open_config["minimum_known_support"]),
        minimum_unknown_support=int(open_config["minimum_unknown_support"]),
    )
    calibration_identity_payload = {
        "protocol": PROTOCOL,
        "config_sha256": file_sha256(config_path),
        "representation_checkpoint_sha256": checkpoint_sha256,
        "known_train_sha256": observed["known_train"],
        "known_tune_sha256": observed["known_tune"],
        "unknown_tune_sha256": observed["unknown_tune"],
        "segment_cache_audit_sha256": observed["segment_cache_audit"],
        "feature_sha256": {
            name: metadata["feature_sha256"]
            for name, metadata in feature_metadata.items()
        },
        "unknown_split_seed": int(open_config.get("split_seed", seed)),
        "unknown_calibration_audio_sha256": sorted(calibration_hashes),
        "unknown_audit_audio_sha256": sorted(audit_hashes),
        "open_set_config": dict(open_config),
        "source_sha256": _protocol_source_sha256(),
    }
    calibration_identity_sha256 = hashlib.sha256(
        json.dumps(
            calibration_identity_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    calibrated = replace(
        calibrated,
        representation_checkpoint_sha256=checkpoint_sha256,
        calibration_identity_sha256=calibration_identity_sha256,
    )
    calibration_metrics = _open_set_metrics(
        calibrated,
        tune_embeddings,
        tune_logits,
        tune_targets,
        unknown_embeddings[calibration_mask],
        unknown_logits[calibration_mask],
    )
    development_audit_metrics = _open_set_metrics(
        calibrated,
        tune_embeddings,
        tune_logits,
        tune_targets,
        unknown_embeddings[audit_mask],
        unknown_logits[audit_mask],
    )
    tune_classifier_metrics = classification_metrics(
        tune_targets, tune_logits, len(known_models)
    )
    tune_classifier_metrics["top3_accuracy"] = float(
        top_k_accuracy_score(
            tune_targets,
            tune_logits,
            k=min(3, len(known_models)),
            labels=list(range(len(known_models))),
        )
    )
    tune_classifier_metrics["confusion_matrix"] = confusion_matrix(
        tune_targets,
        tune_logits.argmax(axis=1),
        labels=list(range(len(known_models))),
    ).astype(int).tolist()
    tune_classifier_metrics["mean_normalized_attention_entropy"] = float(
        tune_entropies.mean()
    )

    unknown_audit_rows = [
        example
        for example, selected in zip(
            unknown_metadata, audit_mask, strict=True
        )
        if selected
    ]
    audit_decision = calibrated.score(
        unknown_embeddings[audit_mask], unknown_logits[audit_mask]
    )
    per_unknown_model: dict[str, Any] = {}
    for model_id in sorted({example.model_id for example in unknown_audit_rows}):
        model_mask = np.asarray(
            [example.model_id == model_id for example in unknown_audit_rows],
            dtype=np.bool_,
        )
        per_unknown_model[model_id] = {
            "recordings": int(model_mask.sum()),
            "unknown_recall": float(
                (~audit_decision.accepted[model_mask]).mean()
            ),
            "false_acceptance_rate": float(
                audit_decision.accepted[model_mask].mean()
            ),
        }
    by_predicted_known_class: dict[str, Any] = {}
    for class_index, model_id in enumerate(known_models):
        class_mask = audit_decision.predicted_classes == class_index
        by_predicted_known_class[model_id] = {
            "predicted_unknown_recordings": int(class_mask.sum()),
            "false_acceptances": int(
                (audit_decision.accepted & class_mask).sum()
            ),
            "used_conservative_fallback": bool(
                calibrated.fallback_mask[class_index]
            ),
        }

    save_class_conditional_boundary(
        boundary_path,
        calibrated,
        require_identity=True,
    )
    prediction_rows = _prediction_rows(
        split_name="known_train_geometry",
        metadata=train_metadata,
        logits=train_logits,
        embeddings=train_embeddings,
        attention_entropies=train_entropies,
        boundary=calibrated,
    )
    prediction_rows.extend(
        _prediction_rows(
            split_name="known_tune_threshold_fit",
            metadata=tune_metadata,
            logits=tune_logits,
            embeddings=tune_embeddings,
            attention_entropies=tune_entropies,
            boundary=calibrated,
        )
    )
    calibration_metadata = [
        example
        for example, selected in zip(
            unknown_metadata, calibration_mask, strict=True
        )
        if selected
    ]
    prediction_rows.extend(
        _prediction_rows(
            split_name="unknown_tune_threshold_fit",
            metadata=calibration_metadata,
            logits=unknown_logits[calibration_mask],
            embeddings=unknown_embeddings[calibration_mask],
            attention_entropies=unknown_entropies[calibration_mask],
            boundary=calibrated,
        )
    )
    prediction_rows.extend(
        _prediction_rows(
            split_name="unknown_tune_recording_audit",
            metadata=unknown_audit_rows,
            logits=unknown_logits[audit_mask],
            embeddings=unknown_embeddings[audit_mask],
            attention_entropies=unknown_entropies[audit_mask],
            boundary=calibrated,
        )
    )
    _atomic_csv(prediction_rows, predictions_path)

    audit_gate, gate_checks = development_gate_checks(
        development_audit_metrics,
        per_unknown_model,
        config["gates"],
    )
    report = {
        "passed": audit_gate,
        "execution_completed": True,
        "protocol": PROTOCOL,
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": (
            "g19_development_gate_passed_new_final_models_required"
            if audit_gate
            else "revise_g19_before_acquiring_new_final_models"
        ),
        "development_gate_passed": audit_gate,
        "status": "same_known_and_same_unknown_model_development_candidate_only",
        "gate_checks": gate_checks,
        "known_models": known_models,
        "representation_checkpoint_sha256": checkpoint_sha256,
        "calibration_identity_sha256": calibration_identity_sha256,
        "calibration_identity": calibration_identity_payload,
        "boundary_sha256": file_sha256(boundary_path),
        "prediction_table_sha256": file_sha256(predictions_path),
        "known_tune_classifier_metrics": tune_classifier_metrics,
        "threshold_calibration_metrics": calibration_metrics,
        "same_unknown_model_recording_audit_metrics": development_audit_metrics,
        "same_unknown_model_per_model_audit": per_unknown_model,
        "same_unknown_model_by_predicted_known_class": by_predicted_known_class,
        "threshold_diagnostics": diagnostics,
        "fallback_model_ids": [
            known_models[index] for index in diagnostics["fallback_classes"]
        ],
        "unknown_split": {
            "unit": "audio_sha256",
            "calibration_fraction": float(
                open_config.get("unknown_calibration_fraction", 0.40)
            ),
            "calibration_recordings": int(calibration_mask.sum()),
            "recording_audit_recordings": int(audit_mask.sum()),
            "models_shared_between_subsets": sorted(
                frames["unknown_tune"]["model_id"].astype(str).unique().tolist()
            ),
            "warning": (
                "The audit uses different Unknown recordings but the same two "
                "Unknown models as threshold calibration, and its Known side "
                "reuses known_tune; it is not an independent final open-set test."
            ),
        },
        "feature_caches": feature_metadata,
        "artifacts": {
            "boundary": boundary_path.relative_to(root).as_posix(),
            "recording_predictions": predictions_path.relative_to(root).as_posix(),
        },
        "known_holdout_read": False,
        "unknown_holdout_read": False,
        "consumed_p6_reused": False,
        "deployable_final_boundary": False,
        "locked_datasets_read": [],
    }
    _atomic_json(report, calibration_summary_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the G19 class-conditional open-set boundary"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    arguments = parser.parse_args()
    result = calibrate(arguments.config, arguments.root)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
