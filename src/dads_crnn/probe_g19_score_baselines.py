from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp, softmax
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import load_config
from .data_firewall import file_sha256
from .g19_recording_data import RecordingFeatureDataset
from .probe_g19_class_conditional import split_unknown_development_recordings
from .train import resolve_device, set_seed
from .train_g19_representation import (
    _atomic_json,
    _read_validated_frames,
    _verify_inputs,
    build_head,
    infer_head,
    resolve_g19_output,
)


PROTOCOL = "g19_p3_known_only_score_baselines_v1"
METHODS = ("msp", "maximum_logit", "energy")


def knownness_scores(logits: np.ndarray, method: str, *, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("Score probe requires a non-empty [recordings, classes] logit matrix")
    if not np.isfinite(values).all():
        raise ValueError("Score probe logits must be finite")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Energy temperature must be positive")
    if method == "msp":
        return softmax(values, axis=1).max(axis=1)
    if method == "maximum_logit":
        return values.max(axis=1)
    if method == "energy":
        # Liu et al. (NeurIPS 2020): E(x)=-T*logsumexp(f/T).
        # Higher -E is used here as the Known score for a common orientation.
        return temperature * logsumexp(values / temperature, axis=1)
    raise ValueError(f"Unsupported Known/Unknown score: {method}")


def threshold_from_known_only(scores: np.ndarray, minimum_acceptance: float) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Known-only calibration requires finite 1D scores")
    if not 0.0 < minimum_acceptance <= 1.0:
        raise ValueError("minimum_acceptance must be inside (0, 1]")
    required = int(math.ceil(minimum_acceptance * values.size))
    descending = np.sort(values)[::-1]
    threshold = float(descending[required - 1])
    accepted = values >= threshold
    if int(accepted.sum()) < required:
        raise RuntimeError("Known-only threshold violated its acceptance constraint")
    return {
        "threshold": threshold,
        "known_recordings": int(values.size),
        "minimum_acceptance": float(minimum_acceptance),
        "required_acceptances": required,
        "actual_acceptances": int(accepted.sum()),
        "actual_acceptance_rate": float(accepted.mean()),
    }


def score_metrics(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    threshold: float,
    *,
    known_targets: np.ndarray,
    known_logits: np.ndarray,
) -> dict[str, Any]:
    known = np.asarray(known_scores, dtype=np.float64)
    unknown = np.asarray(unknown_scores, dtype=np.float64)
    known_accepted = known >= threshold
    unknown_accepted = unknown >= threshold
    labels = np.concatenate(
        [np.ones(len(known), dtype=np.int64), np.zeros(len(unknown), dtype=np.int64)]
    )
    scores = np.concatenate([known, unknown])
    correct = np.asarray(known_logits).argmax(axis=1) == np.asarray(known_targets)
    known_acceptance = float(known_accepted.mean())
    unknown_recall = float((~unknown_accepted).mean())
    return {
        "known_acceptance_rate": known_acceptance,
        "known_rejection_rate": 1.0 - known_acceptance,
        "unknown_recall": unknown_recall,
        "unknown_false_acceptance_rate": 1.0 - unknown_recall,
        "balanced_open_set_accuracy": 0.5 * (known_acceptance + unknown_recall),
        "known_correct_and_accepted_rate": float((correct & known_accepted).mean()),
        "known_unknown_roc_auc": float(roc_auc_score(labels, scores)),
        "known_unknown_pr_auc": float(average_precision_score(labels, scores)),
        "standardized_pauc_fpr_le_0_05": float(
            roc_auc_score(labels, scores, max_fpr=0.05)
        ),
        "known_recordings": int(len(known)),
        "unknown_recordings": int(len(unknown)),
    }


def run_probe(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    paths, observed, registry = _verify_inputs(config, root)
    known_models = [str(value) for value in registry["known_models"]]
    frames = _read_validated_frames(paths, known_models)
    seed = int(config["train"]["seed"])
    set_seed(seed)
    device = resolve_device(str(config["train"]["device"]))

    checkpoint_path = resolve_g19_output(
        root,
        config["open_set"]["representation_checkpoint_path"],
        context="G19 score-probe representation checkpoint",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("known_models") != known_models
        or checkpoint.get("known_holdout_read") is not False
        or checkpoint.get("unknown_holdout_read") is not False
        or checkpoint.get("locked_datasets_read") != []
    ):
        raise ValueError("G19 representation checkpoint is not development-safe")

    feature_dir = resolve_g19_output(
        root,
        config["feature_cache_dir"],
        context="G19 score-probe feature cache",
    )
    features: dict[str, np.ndarray] = {}
    feature_identity: dict[str, Any] = {}
    for name in ("known_tune", "unknown_tune"):
        metadata_path = feature_dir / f"{name}.json"
        feature_path = feature_dir / f"{name}.npy"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("manifest_sha256") != observed[name]
            or metadata.get("feature_sha256") != file_sha256(feature_path)
        ):
            raise ValueError(f"G19 {name} feature cache identity changed")
        features[name] = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        feature_identity[name] = {
            "path": feature_path.as_posix(),
            "sha256": metadata["feature_sha256"],
        }

    head = build_head(config, len(known_models)).to(device)
    head.load_state_dict(checkpoint["head_state"], strict=True)
    head.eval()
    inference = {}
    for name in ("known_tune", "unknown_tune"):
        inference[name] = infer_head(
            head,
            RecordingFeatureDataset(frames[name], features[name]),
            batch_size=32,
            device=device,
        )
    known_logits, _, known_targets, _, known_metadata = inference["known_tune"]
    unknown_logits, _, _, _, unknown_metadata = inference["unknown_tune"]

    _, unknown_audit = split_unknown_development_recordings(
        frames["unknown_tune"],
        calibration_fraction=float(config["open_set"]["unknown_calibration_fraction"]),
        seed=int(config["open_set"]["split_seed"]),
    )
    audit_hashes = set(unknown_audit["audio_sha256"].astype(str).str.lower())
    audit_mask = np.asarray(
        [item.audio_sha256.lower() in audit_hashes for item in unknown_metadata],
        dtype=np.bool_,
    )
    if int(audit_mask.sum()) != int(unknown_audit["audio_sha256"].nunique()):
        raise RuntimeError("G19 score-probe Unknown audit selection is incomplete")
    audit_logits = unknown_logits[audit_mask]
    audit_metadata = [
        item for item, selected in zip(unknown_metadata, audit_mask, strict=True) if selected
    ]

    minimum_acceptance = float(config["open_set"]["minimum_known_acceptance"])
    results = []
    prediction_rows = []
    for method in METHODS:
        known_scores = knownness_scores(known_logits, method)
        unknown_scores = knownness_scores(audit_logits, method)
        calibration = threshold_from_known_only(known_scores, minimum_acceptance)
        threshold = float(calibration["threshold"])
        metrics = score_metrics(
            known_scores,
            unknown_scores,
            threshold,
            known_targets=known_targets,
            known_logits=known_logits,
        )
        per_unknown_model = {}
        for model_id in sorted({item.model_id for item in audit_metadata}):
            mask = np.asarray([item.model_id == model_id for item in audit_metadata])
            accepted = unknown_scores[mask] >= threshold
            per_unknown_model[model_id] = {
                "recordings": int(mask.sum()),
                "unknown_recall": float((~accepted).mean()),
                "false_acceptance_rate": float(accepted.mean()),
            }
        results.append(
            {
                "method": method,
                "threshold_calibration": calibration,
                "audit_metrics": metrics,
                "per_unknown_model": per_unknown_model,
            }
        )
        for split, logits, scores, metadata in (
            ("known_tune_threshold_fit", known_logits, known_scores, known_metadata),
            ("unknown_tune_recording_audit", audit_logits, unknown_scores, audit_metadata),
        ):
            predicted = logits.argmax(axis=1)
            for index, item in enumerate(metadata):
                accepted = bool(scores[index] >= threshold)
                prediction_rows.append(
                    {
                        "method": method,
                        "split": split,
                        "audio_sha256": item.audio_sha256,
                        "true_model_id": item.model_id,
                        "predicted_model_id": known_models[int(predicted[index])],
                        "knownness_score": float(scores[index]),
                        "threshold": threshold,
                        "accepted_as_known": accepted,
                        "open_set_output": (
                            known_models[int(predicted[index])] if accepted else "UNKNOWN"
                        ),
                    }
                )

    selected = max(
        results,
        key=lambda item: (
            item["audit_metrics"]["unknown_recall"],
            item["audit_metrics"]["known_unknown_roc_auc"],
            item["audit_metrics"]["standardized_pauc_fpr_le_0_05"],
        ),
    )
    output_dir = resolve_g19_output(
        root,
        "artifacts/g19_supcon_attention/p3_known_only_score_baselines",
        context="G19 score-probe output",
    )
    summary_path = output_dir / "summary.json"
    predictions_path = output_dir / "recording_predictions.csv"
    if summary_path.exists() or predictions_path.exists():
        raise ValueError("G19 score-probe outputs already exist; refusing to overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    report = {
        "passed": True,
        "execution_completed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "development_score_candidate_selected_without_holdout",
        "selected_method": selected["method"],
        "selection_is_development_only": True,
        "results": results,
        "unknown_audit_models": sorted({item.model_id for item in audit_metadata}),
        "threshold_uses_unknown": False,
        "known_holdout_read": False,
        "unknown_holdout_read": False,
        "consumed_p6_reused": False,
        "deployable_final_boundary": False,
        "inputs": {
            "config": config_path.as_posix(),
            "config_sha256": file_sha256(config_path),
            "representation_checkpoint": checkpoint_path.as_posix(),
            "representation_checkpoint_sha256": file_sha256(checkpoint_path),
            "feature_cache": feature_identity,
        },
        "artifacts": {"recording_predictions": predictions_path.as_posix()},
        "locked_datasets_read": [],
    }
    _atomic_json(report, summary_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe G19 MSP, MLS, and energy scores")
    parser.add_argument("--config", type=Path, default=Path("configs/g19_supcon_attention.yaml"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run_probe(args.config, args.root)


if __name__ == "__main__":
    main()
