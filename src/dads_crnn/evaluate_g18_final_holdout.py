from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from torch.nn import functional as F

from .calibrate_g18_unknown import aggregate_recording_logits
from .config import load_config
from .data_firewall import audit_csv_rows, file_sha256
from .panns import PannsCnn14Binary
from .probe_g18_prototypes import aggregate_recording_embeddings
from .train import resolve_device, set_seed
from .train_g18_model_id import (
    MULTISEED_PROTOCOL,
    PROTOCOL as SEED42_PROTOCOL,
    RegistryDataset,
    _batches,
)


PROTOCOL = "g18_p6_single_final_holdout_v1"
P5_AGGREGATE_PROTOCOL = "g18_p5_multiseed_fixed_linear_oe_v1_aggregate_v1"
P3D_PROTOCOL = "g18_p3d_existing_unknown_linear_oe_v1"
P5_PROTOCOL = "g18_p5_multiseed_fixed_linear_oe_v1"
SEEDS = (42, 43, 44)
DATASETS = ("known_holdout", "unknown_holdout")
REQUIRED_COLUMNS = (
    "model_id",
    "target_index",
    "is_known",
    "audio_sha256",
    "cache_path",
    "cache_index",
)
AGGREGATE_METRICS = (
    "known_acceptance_rate",
    "unknown_recall",
    "open_set_balanced_accuracy",
    "known_unknown_auroc",
    "known_closed_set_accuracy",
    "known_macro_f1",
    "known_minimum_recall",
    "known_model_id_end_to_end_accuracy",
    "strict_known_full_chain_accuracy",
    "strict_unknown_full_chain_recall",
    "balanced_known_full_chain_accuracy",
    "balanced_unknown_full_chain_recall",
)


def _source_manifest(root: Path) -> dict[str, str]:
    paths = sorted((root / "src" / "dads_crnn").glob("*.py"))
    if not paths:
        raise FileNotFoundError("G18 P6 source package is missing")
    return {
        path.relative_to(root).as_posix(): file_sha256(path) for path in paths
    }


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=True)


def _verify(path: Path, expected: object, name: str) -> str:
    observed = file_sha256(path)
    if observed != str(expected):
        raise ValueError(
            f"G18 P6 SHA256 mismatch for {name}: "
            f"expected={expected}, observed={observed}"
        )
    return observed


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def score_oe_space(embeddings: np.ndarray, space: dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    projected = (values - space["pca_mean"]) @ space["pca_components"].T
    standardized = (projected - space["scaler_mean"]) / space["scaler_scale"]
    logits = standardized @ space["logistic_coef"].reshape(-1)
    logits = logits + float(space["logistic_intercept"].reshape(-1)[0])
    scores = _sigmoid(logits)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("G18 P6 produced invalid Known probabilities")
    return scores


def summarize_seed(
    *,
    known_frame: pd.DataFrame,
    unknown_frame: pd.DataFrame,
    known_logits: np.ndarray,
    unknown_logits: np.ndarray,
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    threshold: float,
    known_g7_probability: np.ndarray,
    unknown_g7_probability: np.ndarray,
    strict_threshold: float,
    balanced_threshold: float,
    known_models: list[str],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    known_targets = known_frame["target_index"].to_numpy(dtype=np.int64)
    known_predictions = np.asarray(known_logits).argmax(axis=1).astype(np.int64)
    unknown_predictions = np.asarray(unknown_logits).argmax(axis=1).astype(np.int64)
    accepted = np.asarray(known_scores) >= threshold
    rejected = np.asarray(unknown_scores) < threshold
    correct = known_predictions == known_targets
    labels = list(range(len(known_models)))
    recalls = recall_score(
        known_targets,
        known_predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )
    known_acceptance = float(accepted.mean())
    unknown_recall = float(rejected.mean())
    combined_labels = np.concatenate(
        [np.ones(len(known_scores)), np.zeros(len(unknown_scores))]
    )
    metrics: dict[str, Any] = {
        "known_acceptance_rate": known_acceptance,
        "unknown_recall": unknown_recall,
        "unknown_false_acceptance_rate": 1.0 - unknown_recall,
        "open_set_balanced_accuracy": 0.5
        * (known_acceptance + unknown_recall),
        "known_unknown_auroc": float(
            roc_auc_score(
                combined_labels,
                np.concatenate([known_scores, unknown_scores]),
            )
        ),
        "known_closed_set_accuracy": float(
            accuracy_score(known_targets, known_predictions)
        ),
        "known_macro_f1": float(
            f1_score(
                known_targets,
                known_predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "known_minimum_recall": float(recalls.min()),
        "known_per_class_recall": {
            model: float(value) for model, value in zip(known_models, recalls, strict=True)
        },
        "known_confusion_matrix": confusion_matrix(
            known_targets, known_predictions, labels=labels
        ).astype(int).tolist(),
        "known_model_id_end_to_end_accuracy": float(
            np.mean(correct & accepted)
        ),
        "unknown_recall_by_model": {
            str(model): float(
                rejected[unknown_frame["model_id"].astype(str).to_numpy() == str(model)]
                .mean()
            )
            for model in sorted(unknown_frame["model_id"].astype(str).unique())
        },
    }
    for mode, detection_threshold in (
        ("strict", strict_threshold),
        ("balanced", balanced_threshold),
    ):
        known_detected = known_g7_probability >= detection_threshold
        unknown_detected = unknown_g7_probability >= detection_threshold
        metrics[f"{mode}_known_detection_recall"] = float(known_detected.mean())
        metrics[f"{mode}_unknown_detection_recall"] = float(
            unknown_detected.mean()
        )
        metrics[f"{mode}_known_full_chain_accuracy"] = float(
            np.mean(known_detected & accepted & correct)
        )
        metrics[f"{mode}_unknown_full_chain_recall"] = float(
            np.mean(unknown_detected & rejected)
        )
        metrics[f"{mode}_hierarchical_balanced_accuracy"] = 0.5 * (
            metrics[f"{mode}_known_full_chain_accuracy"]
            + metrics[f"{mode}_unknown_full_chain_recall"]
        )
    return metrics, known_predictions, unknown_predictions


def _paths_and_hashes(
    config: dict[str, Any], root: Path
) -> tuple[
    dict[str, Path],
    dict[str, str],
    dict[int, dict[str, Path]],
    dict[int, dict[str, str]],
]:
    paths = {
        name: _resolve(root, config["inputs"][name]["path"])
        for name in (
            "registry_audit",
            "known_holdout",
            "unknown_holdout",
            "p5_summary",
            "g7_frozen_protocol",
        )
    }
    hashes = {
        name: _verify(path, config["inputs"][name]["sha256"], name)
        for name, path in paths.items()
    }
    paths["official_checkpoint"] = _resolve(
        root, config["model"]["checkpoint_path"]
    )
    paths["g7_checkpoint"] = _resolve(
        root, config["model"]["binary_checkpoint_path"]
    )
    hashes["official_checkpoint"] = _verify(
        paths["official_checkpoint"],
        config["model"]["checkpoint_sha256"],
        "official_checkpoint",
    )
    hashes["g7_checkpoint"] = _verify(
        paths["g7_checkpoint"],
        config["model"]["binary_checkpoint_sha256"],
        "g7_checkpoint",
    )
    system_paths: dict[int, dict[str, Path]] = {}
    system_hashes: dict[int, dict[str, str]] = {}
    for seed in SEEDS:
        entry = config["systems"][str(seed)]
        system_paths[seed] = {
            name: _resolve(root, entry[name]["path"])
            for name in (
                "model_id_checkpoint",
                "model_id_summary",
                "oe_space",
                "oe_summary",
            )
        }
        system_hashes[seed] = {
            name: _verify(
                system_paths[seed][name],
                entry[name]["sha256"],
                f"seed_{seed}_{name}",
            )
            for name in system_paths[seed]
        }
    return paths, hashes, system_paths, system_hashes


def validate_inputs(
    config_path: Path, root: Path, *, audit_holdout_manifests: bool
) -> tuple[
    dict[str, Any],
    dict[str, Path],
    dict[str, str],
    dict[int, dict[str, Path]],
    dict[int, dict[str, str]],
    dict[str, Any],
]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected G18 P6 protocol")
    if tuple(config["reporting"]["seeds"]) != SEEDS:
        raise ValueError("G18 P6 must report seeds 42, 43 and 44 in order")
    if config["reporting"]["seed_selection"] != "forbidden":
        raise ValueError("G18 P6 seed selection must remain forbidden")
    paths, hashes, system_paths, system_hashes = _paths_and_hashes(config, root)
    registry = json.loads(paths["registry_audit"].read_text(encoding="utf-8"))
    p5 = json.loads(paths["p5_summary"].read_text(encoding="utf-8"))
    g7_protocol = json.loads(
        paths["g7_frozen_protocol"].read_text(encoding="utf-8")
    )
    if not (
        registry.get("passed") is True
        and registry.get("locked_datasets_read") == []
        and p5.get("passed") is True
        and p5.get("protocol") == P5_AGGREGATE_PROTOCOL
        and p5.get("decision") == "ready_to_freeze_single_holdout_protocol"
        and p5.get("all_seed_gates_passed") is True
        and p5.get("known_holdout_read") is False
        and p5.get("unknown_holdout_read") is False
        and p5.get("selection_rule")
        == "report_mean_sample_std_and_worst_seed_no_seed_selection"
        and g7_protocol.get("status") == "frozen_before_final_data_read"
        and g7_protocol.get("protocol", {}).get("final_data_used_for_selection")
        is False
    ):
        raise ValueError("G18 P0/P5/G7 prerequisites do not authorize P6")
    if (
        float(config["g7_diagnostics"]["temperature"])
        != float(g7_protocol["protocol"]["baseline_temperature"])
        or float(config["g7_diagnostics"]["strict_threshold"])
        != float(g7_protocol["protocol"]["modes"]["strict"]["baseline_threshold"])
        or float(config["g7_diagnostics"]["balanced_threshold"])
        != float(g7_protocol["protocol"]["modes"]["balanced"]["baseline_threshold"])
    ):
        raise ValueError("G18 P6 changed frozen G7 diagnostic calibration")
    expected_models = list(config["reporting"]["known_models"])
    if expected_models != list(registry["known_models"]):
        raise ValueError("G18 P6 Known model order differs from P0")
    if list(config["reporting"]["unknown_models"]) != list(
        registry["unknown_holdout_models"]
    ):
        raise ValueError("G18 P6 Unknown Holdout models differ from P0")
    for name in DATASETS:
        contract = config["inputs"][name]
        registered = registry["outputs"][name]
        if not (
            Path(contract["path"]).as_posix() == registered["path"]
            and contract["sha256"] == registered["sha256"]
            and int(contract["rows"]) == int(registered["rows"])
            and int(contract["recordings"]) == int(registered["raw_recordings"])
        ):
            raise ValueError(f"G18 P6 {name} contract differs from P0")
    for seed in SEEDS:
        checkpoint = torch.load(
            system_paths[seed]["model_id_checkpoint"],
            map_location="cpu",
            weights_only=True,
        )
        model_summary = json.loads(
            system_paths[seed]["model_id_summary"].read_text(encoding="utf-8")
        )
        oe_summary = json.loads(
            system_paths[seed]["oe_summary"].read_text(encoding="utf-8")
        )
        expected_model_protocol = (
            SEED42_PROTOCOL if seed == 42 else MULTISEED_PROTOCOL
        )
        expected_oe_protocol = P3D_PROTOCOL if seed == 42 else P5_PROTOCOL
        with np.load(system_paths[seed]["oe_space"], allow_pickle=False) as space:
            oe_threshold = float(space["threshold"].reshape(-1)[0])
            oe_models = space["known_models"].astype(str).tolist()
        if not (
            checkpoint.get("protocol") == expected_model_protocol
            and checkpoint.get("seed") == seed
            and checkpoint.get("known_models") == expected_models
            and isinstance(checkpoint.get("head_state"), dict)
            and model_summary.get("passed") is True
            and model_summary.get("seed") == seed
            and model_summary.get("feasibility_gate_passed") is True
            and oe_summary.get("passed") is True
            and oe_summary.get("protocol") == expected_oe_protocol
            and oe_summary.get("probe_gate_passed") is True
            and oe_summary.get("known_holdout_read") is False
            and oe_summary.get("unknown_holdout_read") is False
            and oe_models == expected_models
            and np.isclose(
                oe_threshold,
                float(oe_summary["known_confidence_threshold"]),
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raise ValueError(f"G18 P6 seed {seed} system identity is invalid")
    if audit_holdout_manifests:
        for name, expected_known in (
            ("known_holdout", True),
            ("unknown_holdout", False),
        ):
            rows = audit_csv_rows(
                paths[name],
                forbidden_hashes=frozenset(),
                required_columns=REQUIRED_COLUMNS,
            )
            frame = pd.read_csv(paths[name])
            flags = frame["is_known"].astype(bool)
            if rows != int(config["inputs"][name]["rows"]):
                raise ValueError(f"G18 P6 {name} row count changed")
            if frame["audio_sha256"].nunique() != int(
                config["inputs"][name]["recordings"]
            ):
                raise ValueError(f"G18 P6 {name} recording count changed")
            if expected_known and not bool(flags.all()):
                raise ValueError("G18 P6 Known Holdout contains Unknown rows")
            if not expected_known and bool(flags.any()):
                raise ValueError("G18 P6 Unknown Holdout contains Known rows")
            expected = (
                set(config["reporting"]["known_models"])
                if expected_known
                else set(config["reporting"]["unknown_models"])
            )
            if set(frame["model_id"].astype(str)) != expected:
                raise ValueError(f"G18 P6 {name} model set changed")
        known_hashes = set(
            pd.read_csv(paths["known_holdout"], usecols=["audio_sha256"])[
                "audio_sha256"
            ].astype(str)
        )
        unknown_hashes = set(
            pd.read_csv(paths["unknown_holdout"], usecols=["audio_sha256"])[
                "audio_sha256"
            ].astype(str)
        )
        if known_hashes & unknown_hashes:
            raise ValueError("G18 P6 Holdout recording hashes overlap")
    return config, paths, hashes, system_paths, system_hashes, registry


def preflight(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config, paths, hashes, system_paths, system_hashes, registry = validate_inputs(
        config_path, root, audit_holdout_manifests=True
    )
    output_dir = root / str(config["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("G18 P6 final output directory is not empty")
    report = {
        "passed": True,
        "protocol": f"{PROTOCOL}_preflight",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "authorized_for_one_formal_run",
        "formal_holdout_audio_read": False,
        "holdout_manifest_metadata_read": True,
        "optimizer_step_exercised": False,
        "threshold_search_exercised": False,
        "seed_selection_exercised": False,
        "technical_resume_only": True,
        "holdout_contract": {
            name: {
                "rows": int(config["inputs"][name]["rows"]),
                "recordings": int(config["inputs"][name]["recordings"]),
                "sha256": hashes[name],
            }
            for name in DATASETS
        },
        "known_models": list(registry["known_models"]),
        "unknown_models": list(registry["unknown_holdout_models"]),
        "inputs": {
            "config_sha256": file_sha256(config_path),
            "implementation_sha256": file_sha256(Path(__file__)),
            "source_manifest": _source_manifest(root),
            **{f"{name}_sha256": value for name, value in hashes.items()},
            "systems": {
                str(seed): {
                    f"{name}_sha256": value
                    for name, value in system_hashes[seed].items()
                }
                for seed in SEEDS
            },
        },
    }
    output_dir = root / str(config["preflight_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "report.json"
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def _validate_preflight(
    config_path: Path, root: Path, config: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    path = root / str(config["preflight_output_dir"]) / "report.json"
    if not path.is_file():
        raise FileNotFoundError("G18 P6 preflight report is missing")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not (
        report.get("passed") is True
        and report.get("decision") == "authorized_for_one_formal_run"
        and report.get("formal_holdout_audio_read") is False
        and report.get("threshold_search_exercised") is False
        and report.get("seed_selection_exercised") is False
        and report.get("inputs", {}).get("config_sha256")
        == file_sha256(config_path)
        and report.get("inputs", {}).get("implementation_sha256")
        == file_sha256(Path(__file__))
        and report.get("inputs", {}).get("source_manifest")
        == _source_manifest(root)
    ):
        raise ValueError("G18 P6 preflight report is invalid or stale")
    return path, report


@torch.no_grad()
def _infer_base(
    detector: PannsCnn14Binary,
    dataset: RegistryDataset,
    *,
    batch_size: int,
    device: torch.device,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    detector.eval()
    embeddings = []
    logits = []
    indices = np.arange(len(dataset.frame), dtype=np.int64)
    processed = 0
    for batch_indices in _batches(indices, batch_size):
        waveforms, _ = dataset.batch(batch_indices)
        embedding = detector.extract_embedding(waveforms.to(device))
        drone_logit = detector.backbone.fc_audioset(embedding).squeeze(1)
        if not torch.isfinite(embedding).all() or not torch.isfinite(
            drone_logit
        ).all():
            raise RuntimeError(f"G18 P6 produced non-finite {label} outputs")
        embeddings.append(embedding.float().cpu().numpy())
        logits.append(drone_logit.float().cpu().numpy())
        processed += len(batch_indices)
        if processed % 1024 == 0 or processed == len(indices):
            print(f"G18 P6 {label}: {processed}/{len(indices)}", flush=True)
    return np.concatenate(embeddings), np.concatenate(logits)


def _head_outputs(
    base_embeddings: np.ndarray,
    head_state: dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    values = torch.as_tensor(base_embeddings, dtype=torch.float32)
    normalized = F.layer_norm(
        values,
        (values.shape[1],),
        weight=head_state["embedding_norm.weight"].float(),
        bias=head_state["embedding_norm.bias"].float(),
    )
    logits = F.linear(
        normalized,
        head_state["classifier.weight"].float(),
        head_state["classifier.bias"].float(),
    )
    if not torch.isfinite(normalized).all() or not torch.isfinite(logits).all():
        raise RuntimeError("G18 P6 model-ID head produced non-finite outputs")
    return normalized.numpy(), logits.numpy()


def _load_oe(path: Path) -> dict[str, np.ndarray]:
    required = {
        "pca_mean",
        "pca_components",
        "scaler_mean",
        "scaler_scale",
        "logistic_coef",
        "logistic_intercept",
        "known_models",
        "threshold",
    }
    with np.load(path, allow_pickle=False) as values:
        if set(values.files) != required:
            raise ValueError("G18 P6 OE space fields changed")
        return {name: np.asarray(values[name]) for name in values.files}


def _aggregate_scalar(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    _, aggregated = aggregate_recording_logits(
        frame, np.asarray(values, dtype=np.float64).reshape(-1, 1)
    )
    return aggregated.reshape(-1)


def _seed_predictions_frame(
    *,
    seed: int,
    known_frame: pd.DataFrame,
    unknown_frame: pd.DataFrame,
    known_predictions: np.ndarray,
    unknown_predictions: np.ndarray,
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    threshold: float,
    known_g7: np.ndarray,
    unknown_g7: np.ndarray,
    known_models: list[str],
) -> pd.DataFrame:
    rows = []
    for partition, frame, predictions, scores, g7_values in (
        (
            "known_holdout",
            known_frame,
            known_predictions,
            known_scores,
            known_g7,
        ),
        (
            "unknown_holdout",
            unknown_frame,
            unknown_predictions,
            unknown_scores,
            unknown_g7,
        ),
    ):
        for row, prediction, score, g7_value in zip(
            frame.to_dict("records"),
            predictions,
            scores,
            g7_values,
            strict=True,
        ):
            rows.append(
                {
                    **row,
                    "partition": partition,
                    "seed": seed,
                    "predicted_index": int(prediction),
                    "predicted_model": known_models[int(prediction)],
                    "known_probability": float(score),
                    "known_threshold": float(threshold),
                    "open_set_decision": (
                        "known_uav" if float(score) >= threshold else "unknown_uav"
                    ),
                    "g7_recording_probability": float(g7_value),
                }
            )
    return pd.DataFrame(rows)


def evaluate(
    config_path: Path, root: Path, *, resume: bool
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    (
        config,
        paths,
        hashes,
        system_paths,
        system_hashes,
        registry,
    ) = validate_inputs(config_path, root, audit_holdout_manifests=True)
    preflight_path, _ = _validate_preflight(config_path, root, config)
    output_dir = root / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "summary.json"
    marker_path = output_dir / "FORMAL_RUN_STARTED.json"
    if final_path.exists():
        raise FileExistsError("G18 P6 final Holdout was already consumed")
    if resume:
        if not marker_path.is_file():
            raise FileNotFoundError("G18 P6 technical-resume marker is missing")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if not (
            marker.get("protocol") == PROTOCOL
            and marker.get("config_sha256") == file_sha256(config_path)
            and marker.get("preflight_sha256") == file_sha256(preflight_path)
        ):
            raise ValueError("G18 P6 resume marker is invalid")
    else:
        marker = {
            "protocol": PROTOCOL,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": file_sha256(config_path),
            "preflight_sha256": file_sha256(preflight_path),
            "technical_resume_only": True,
        }
        with marker_path.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(marker, indent=2, ensure_ascii=False) + "\n"
            )
    set_seed(42)
    device = resolve_device("auto")
    if device.type != "cuda":
        raise RuntimeError("G18 P6 final evaluation requires the server CUDA GPU")
    detector = PannsCnn14Binary(
        initialization=str(config["model"]["initialization"]),
        vendor_dir=str(config["model"]["vendor_dir"]),
        checkpoint_path=paths["official_checkpoint"].as_posix(),
        checkpoint_sha256=hashes["official_checkpoint"],
        spec_augment=False,
        frontend_precision="float32",
        binary_checkpoint_path=paths["g7_checkpoint"].as_posix(),
        binary_checkpoint_sha256=hashes["g7_checkpoint"],
        trainable_scope="binary_head_only",
    ).to(device)
    target_samples = int(
        config["data"]["sample_rate"] * config["data"]["clip_seconds"]
    )
    datasets = {
        name: RegistryDataset(paths[name], target_samples) for name in DATASETS
    }
    base_embeddings = {}
    drone_logits = {}
    for name in DATASETS:
        base_embeddings[name], drone_logits[name] = _infer_base(
            detector,
            datasets[name],
            batch_size=int(config["data"]["batch_size"]),
            device=device,
            label=name,
        )
    del detector
    temperature = float(config["g7_diagnostics"]["temperature"])
    recording_g7 = {
        name: _sigmoid(
            _aggregate_scalar(datasets[name].frame, drone_logits[name])
            / temperature
        )
        for name in DATASETS
    }
    known_models = list(registry["known_models"])
    metrics_by_seed = {}
    output_hashes = {}
    for seed in SEEDS:
        checkpoint = torch.load(
            system_paths[seed]["model_id_checkpoint"],
            map_location="cpu",
            weights_only=True,
        )
        recording_frames = {}
        recording_embeddings = {}
        recording_logits = {}
        for name in DATASETS:
            embeddings, logits = _head_outputs(
                base_embeddings[name], checkpoint["head_state"]
            )
            recording_frames[name], recording_embeddings[name] = (
                aggregate_recording_embeddings(datasets[name].frame, embeddings)
            )
            logit_frame, recording_logits[name] = aggregate_recording_logits(
                datasets[name].frame, logits
            )
            if (
                recording_frames[name]["audio_sha256"].tolist()
                != logit_frame["audio_sha256"].tolist()
            ):
                raise RuntimeError("G18 P6 recording aggregation order mismatch")
        oe_space = _load_oe(system_paths[seed]["oe_space"])
        threshold = float(oe_space["threshold"].reshape(-1)[0])
        known_scores = score_oe_space(
            recording_embeddings["known_holdout"], oe_space
        )
        unknown_scores = score_oe_space(
            recording_embeddings["unknown_holdout"], oe_space
        )
        metrics, known_predictions, unknown_predictions = summarize_seed(
            known_frame=recording_frames["known_holdout"],
            unknown_frame=recording_frames["unknown_holdout"],
            known_logits=recording_logits["known_holdout"],
            unknown_logits=recording_logits["unknown_holdout"],
            known_scores=known_scores,
            unknown_scores=unknown_scores,
            threshold=threshold,
            known_g7_probability=recording_g7["known_holdout"],
            unknown_g7_probability=recording_g7["unknown_holdout"],
            strict_threshold=float(
                config["g7_diagnostics"]["strict_threshold"]
            ),
            balanced_threshold=float(
                config["g7_diagnostics"]["balanced_threshold"]
            ),
            known_models=known_models,
        )
        metrics_by_seed[seed] = metrics
        prediction_frame = _seed_predictions_frame(
            seed=seed,
            known_frame=recording_frames["known_holdout"],
            unknown_frame=recording_frames["unknown_holdout"],
            known_predictions=known_predictions,
            unknown_predictions=unknown_predictions,
            known_scores=known_scores,
            unknown_scores=unknown_scores,
            threshold=threshold,
            known_g7=recording_g7["known_holdout"],
            unknown_g7=recording_g7["unknown_holdout"],
            known_models=known_models,
        )
        prediction_path = output_dir / f"seed_{seed}_recording_predictions.csv"
        temporary = prediction_path.with_suffix(".csv.tmp")
        prediction_frame.to_csv(temporary, index=False)
        os.replace(temporary, prediction_path)
        output_hashes[str(seed)] = {
            "path": prediction_path.relative_to(root).as_posix(),
            "sha256": file_sha256(prediction_path),
        }
    aggregate_metrics = {}
    for name in AGGREGATE_METRICS:
        values = np.asarray(
            [metrics_by_seed[seed][name] for seed in SEEDS],
            dtype=np.float64,
        )
        aggregate_metrics[name] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "final_holdout_consumed_and_closed",
        "formal_holdout_audio_read": True,
        "technical_resume_used": bool(resume),
        "threshold_search_exercised": False,
        "seed_selection_exercised": False,
        "seeds": list(SEEDS),
        "metrics_by_seed": {
            str(seed): metrics_by_seed[seed] for seed in SEEDS
        },
        "aggregate_metrics": aggregate_metrics,
        "primary_scope": "model_identification_conditional_on_uav_input",
        "g7_diagnostics": {
            **config["g7_diagnostics"],
            "holdout_contains_background": False,
            "background_fpr_estimable": False,
        },
        "limitations": [
            "single_recorder_OLYMPUS_LS11",
            "mostly_single_acquisition_session_per_model",
            "unknown_holdout_contains_only_X6D_and_Y6",
            "holdout_has_no_background_for_estimating_g7_false_positive_rate",
        ],
        "outputs": {"recording_predictions": output_hashes},
        "inputs": {
            "config_sha256": file_sha256(config_path),
            "implementation_sha256": file_sha256(Path(__file__)),
            "preflight_sha256": file_sha256(preflight_path),
            "formal_start_marker_sha256": file_sha256(marker_path),
            "source_manifest": _source_manifest(root),
            **{f"{name}_sha256": value for name, value in hashes.items()},
            "systems": {
                str(seed): {
                    f"{name}_sha256": value
                    for name, value in system_hashes[seed].items()
                }
                for seed in SEEDS
            },
        },
    }
    temporary = final_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, final_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the one-time G18 final Known/Unknown Holdout evaluation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g18_final_holdout.yaml"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.preflight_only and args.resume:
        parser.error("--preflight-only and --resume are mutually exclusive")
    if args.preflight_only:
        preflight(args.config, args.root)
    else:
        evaluate(args.config, args.root, resume=args.resume)


if __name__ == "__main__":
    main()
