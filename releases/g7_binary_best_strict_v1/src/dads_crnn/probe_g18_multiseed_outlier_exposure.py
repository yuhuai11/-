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
from .probe_g18_knn import infer_outputs
from .probe_g18_outlier_exposure import fit_linear_ood_head, known_scores
from .probe_g18_prototypes import aggregate_recording_embeddings
from .train import resolve_device, set_seed
from .train_g18_model_id import MULTISEED_PROTOCOL, RegistryDataset, _load_head


PROTOCOL = "g18_p5_multiseed_fixed_linear_oe_v1"
P3D_PROTOCOL = "g18_p3d_existing_unknown_linear_oe_v1"
P4_PREFLIGHT_PROTOCOL = "g18_p4_multiseed_preflight_v1"
SEEDS = (43, 44)
DATASET_NAMES = (
    "known_train",
    "known_tune",
    "unknown_oe_train",
    "unknown_calibration",
)
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
    "balanced_accuracy",
    "known_unknown_auroc",
    "known_closed_set_accuracy",
    "known_end_to_end_accuracy",
)


def _source_manifest(root: Path) -> dict[str, str]:
    source_root = root / "src" / "dads_crnn"
    paths = sorted(source_root.glob("*.py"))
    if not paths:
        raise FileNotFoundError("G18 P5 could not enumerate dads_crnn sources")
    return {
        path.relative_to(root).as_posix(): file_sha256(path) for path in paths
    }


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    reject_locked_path(path, context="G18 P5 development input")
    return path.resolve(strict=True)


def _verify_hash(path: Path, expected: object, name: str) -> str:
    observed = file_sha256(path)
    if observed != str(expected):
        raise ValueError(
            f"G18 P5 input SHA256 mismatch for {name}: "
            f"expected={expected}, observed={observed}"
        )
    return observed


def _common_paths(
    config: dict[str, Any], root: Path
) -> tuple[dict[str, Path], dict[str, str]]:
    if any(
        name in config.get("inputs", {})
        for name in ("unknown_tune", "known_holdout", "unknown_holdout")
    ):
        raise ValueError("G18 P5 must bind frozen splits and must not bind Holdout")
    paths = {
        name: _resolve(root, config["inputs"][name]["path"])
        for name in (
            *DATASET_NAMES,
            "registry_audit",
            "p3d_config",
            "p3d_preflight",
            "p3d_summary",
            "p4_preflight",
        )
    }
    paths["official_checkpoint"] = _resolve(
        root, config["model"]["checkpoint_path"]
    )
    paths["g7_checkpoint"] = _resolve(
        root, config["model"]["binary_checkpoint_path"]
    )
    observed = {
        name: _verify_hash(path, config["inputs"][name]["sha256"], name)
        for name, path in paths.items()
        if name in config["inputs"]
    }
    observed["official_checkpoint"] = _verify_hash(
        paths["official_checkpoint"],
        config["model"]["checkpoint_sha256"],
        "official_checkpoint",
    )
    observed["g7_checkpoint"] = _verify_hash(
        paths["g7_checkpoint"],
        config["model"]["binary_checkpoint_sha256"],
        "g7_checkpoint",
    )
    return paths, observed


def _replication_paths(
    config: dict[str, Any], root: Path, seed: int
) -> tuple[dict[str, Path], dict[str, str]]:
    entry = config["replications"][str(seed)]
    paths = {
        name: _resolve(root, entry[name]["path"])
        for name in ("checkpoint", "summary")
    }
    observed = {
        name: _verify_hash(paths[name], entry[name]["sha256"], f"seed_{seed}_{name}")
        for name in paths
    }
    return paths, observed


def _validate_frozen_protocol(
    config: dict[str, Any],
    paths: dict[str, Path],
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = json.loads(paths["registry_audit"].read_text(encoding="utf-8"))
    p3d_config = load_config(paths["p3d_config"])
    p3d_preflight = json.loads(
        paths["p3d_preflight"].read_text(encoding="utf-8")
    )
    p3d_summary = json.loads(paths["p3d_summary"].read_text(encoding="utf-8"))
    p4_preflight = json.loads(
        paths["p4_preflight"].read_text(encoding="utf-8")
    )
    if not (
        registry.get("passed") is True
        and registry.get("locked_datasets_read") == []
        and p3d_preflight.get("passed") is True
        and p3d_preflight.get("protocol") == f"{P3D_PROTOCOL}_preflight"
        and p3d_preflight.get("known_holdout_read") is False
        and p3d_preflight.get("unknown_holdout_read") is False
        and p3d_summary.get("passed") is True
        and p3d_summary.get("protocol") == P3D_PROTOCOL
        and p3d_summary.get("probe_gate_passed") is True
        and p3d_summary.get("decision") == "proceed_to_multiseed_replication"
        and p3d_summary.get("known_holdout_read") is False
        and p3d_summary.get("unknown_holdout_read") is False
        and p4_preflight.get("passed") is True
        and p4_preflight.get("protocol") == P4_PREFLIGHT_PROTOCOL
        and p4_preflight.get("unknown_tune_read") is False
        and p4_preflight.get("known_holdout_read") is False
        and p4_preflight.get("unknown_holdout_read") is False
    ):
        raise ValueError("G18 P3d/P4 prerequisites do not authorize P5")
    for name in ("unknown_oe_train", "unknown_calibration"):
        frozen = p3d_preflight["outputs"][name]
        if (
            Path(frozen["path"]).as_posix()
            != paths[name].relative_to(root).as_posix()
            or frozen["sha256"] != file_sha256(paths[name])
        ):
            raise ValueError(f"G18 P5 does not use the frozen P3d split: {name}")
    frozen_oe = p3d_config["outlier_exposure"]
    current_oe = config["outlier_exposure"]
    fixed_pairs = {
        "split_seed": (current_oe["split_seed"], frozen_oe["split_seed"]),
        "unknown_train_fraction": (
            current_oe["unknown_train_fraction"],
            frozen_oe["unknown_train_fraction"],
        ),
        "pca_components": (
            current_oe["pca_components"],
            frozen_oe["pca_components"],
        ),
        "logistic_regularization_c": (
            current_oe["logistic_regularization_c"],
            frozen_oe["logistic_regularization_c"],
        ),
        "minimum_known_acceptance": (
            current_oe["minimum_known_acceptance"],
            frozen_oe["minimum_known_acceptance"],
        ),
    }
    if any(left != right for left, right in fixed_pairs.values()):
        raise ValueError(f"G18 P5 changed frozen P3d parameters: {fixed_pairs}")
    if config["gates"] != p3d_config["gates"]:
        raise ValueError("G18 P5 changed frozen P3d gates")
    if int(current_oe["fit_seed"]) != int(frozen_oe["seed"]):
        raise ValueError("G18 P5 OE fit seed must remain frozen at P3d seed 42")
    return registry, p3d_summary


def _read_registry(path: Path, *, expected_known: bool) -> pd.DataFrame:
    audit_csv_rows(path, required_columns=REQUIRED_COLUMNS)
    frame = pd.read_csv(path)
    values = frame["is_known"].astype(bool)
    if bool(values.all()) != expected_known or (
        not expected_known and bool(values.any())
    ):
        raise ValueError(f"G18 P5 is_known mismatch in {path}")
    return frame


def _validate_recording_disjoint(frames: dict[str, pd.DataFrame]) -> dict[str, int]:
    hashes = {
        name: set(frame["audio_sha256"].astype(str))
        for name, frame in frames.items()
    }
    names = tuple(hashes)
    overlaps = {
        f"{left}_vs_{right}": len(hashes[left] & hashes[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }
    if any(overlaps.values()):
        raise ValueError(f"G18 P5 recording overlap: {overlaps}")
    return overlaps


def validate_inputs(
    config_path: Path, root: Path
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
        raise ValueError("Unexpected G18 P5 protocol")
    if sorted(int(seed) for seed in config["replications"]) != list(SEEDS):
        raise ValueError("G18 P5 replication seeds must be exactly 43 and 44")
    paths, hashes = _common_paths(config, root)
    registry, _ = _validate_frozen_protocol(config, paths, root)
    replication_paths = {}
    replication_hashes = {}
    for seed in SEEDS:
        seed_paths, seed_hashes = _replication_paths(config, root, seed)
        summary = json.loads(seed_paths["summary"].read_text(encoding="utf-8"))
        checkpoint = torch.load(
            seed_paths["checkpoint"], map_location="cpu", weights_only=True
        )
        if not (
            summary.get("passed") is True
            and summary.get("protocol") == MULTISEED_PROTOCOL
            and summary.get("seed") == seed
            and summary.get("decision")
            == "proceed_to_fixed_outlier_exposure_replication"
            and summary.get("feasibility_gate_passed") is True
            and summary.get("known_holdout_read") is False
            and summary.get("unknown_holdout_read") is False
            and summary.get("outputs", {})
            .get("best_checkpoint", {})
            .get("sha256")
            == seed_hashes["checkpoint"]
            and checkpoint.get("protocol") == MULTISEED_PROTOCOL
            and checkpoint.get("seed") == seed
            and checkpoint.get("known_models") == list(registry["known_models"])
            and isinstance(checkpoint.get("head_state"), dict)
        ):
            raise ValueError(f"G18 P5 seed {seed} model identity is invalid")
        replication_paths[seed] = seed_paths
        replication_hashes[seed] = seed_hashes
    return (
        config,
        paths,
        hashes,
        replication_paths,
        replication_hashes,
        registry,
    )


def preflight(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    (
        config,
        paths,
        hashes,
        replication_paths,
        replication_hashes,
        _,
    ) = validate_inputs(config_path, root)
    frames = {
        "known_train": _read_registry(paths["known_train"], expected_known=True),
        "known_tune": _read_registry(paths["known_tune"], expected_known=True),
        "unknown_oe_train": _read_registry(
            paths["unknown_oe_train"], expected_known=False
        ),
        "unknown_calibration": _read_registry(
            paths["unknown_calibration"], expected_known=False
        ),
    }
    overlaps = _validate_recording_disjoint(frames)
    output_root = root / str(config["output_dir"])
    existing = {
        str(seed): sorted((output_root / f"seed_{seed}").iterdir())
        if (output_root / f"seed_{seed}").is_dir()
        else []
        for seed in SEEDS
    }
    if any(existing.values()):
        raise FileExistsError("G18 P5 seed output directories are not empty")
    report = {
        "passed": True,
        "protocol": f"{PROTOCOL}_preflight",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ready_for_sequential_probe": True,
        "seeds": list(SEEDS),
        "frozen_oe_fit_seed": int(config["outlier_exposure"]["fit_seed"]),
        "recording_overlaps": overlaps,
        "recordings": {
            name: int(frame["audio_sha256"].nunique())
            for name, frame in frames.items()
        },
        "audio_payload_read": False,
        "formal_probe_started": False,
        "known_holdout_read": False,
        "unknown_holdout_read": False,
        "locked_datasets_read": [],
        "inputs": {
            "config_sha256": file_sha256(config_path),
            "implementation_sha256": file_sha256(Path(__file__)),
            "source_manifest": _source_manifest(root),
            **{f"{name}_sha256": value for name, value in hashes.items()},
            "replications": {
                str(seed): {
                    f"{name}_sha256": value
                    for name, value in replication_hashes[seed].items()
                }
                for seed in SEEDS
            },
        },
    }
    output_dir = root / str(config["preflight_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def _validate_preflight(config_path: Path, root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    report_path = root / str(config["preflight_output_dir"]) / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError("G18 P5 preflight report is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not (
        report.get("passed") is True
        and report.get("ready_for_sequential_probe") is True
        and report.get("audio_payload_read") is False
        and report.get("formal_probe_started") is False
        and report.get("known_holdout_read") is False
        and report.get("unknown_holdout_read") is False
        and report.get("inputs", {}).get("config_sha256")
        == file_sha256(config_path)
        and report.get("inputs", {}).get("implementation_sha256")
        == file_sha256(Path(__file__))
        and report.get("inputs", {}).get("source_manifest")
        == _source_manifest(root)
    ):
        raise ValueError("G18 P5 preflight report is invalid or stale")
    return report


def run_seed(config_path: Path, root: Path, seed: int) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    if seed not in SEEDS:
        raise ValueError(f"G18 P5 seed must be one of {SEEDS}")
    (
        config,
        paths,
        hashes,
        replication_paths,
        replication_hashes,
        registry,
    ) = validate_inputs(config_path, root)
    preflight_report = _validate_preflight(config_path, root)
    preflight_path = root / str(config["preflight_output_dir"]) / "report.json"
    output_dir = root / str(config["output_dir"]) / f"seed_{seed}"
    report_path = output_dir / "summary.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed G18 P5 seed {seed}")
    set_seed(int(config["outlier_exposure"]["fit_seed"]))
    device = resolve_device(str(config["outlier_exposure"]["device"]))
    if device.type != "cuda":
        raise RuntimeError("G18 P5 probe requires the server CUDA GPU")
    known_models = list(registry["known_models"])
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
    )
    model = G7ModelIdentifier(
        detector,
        embedding_dim=int(config["model_id_head"]["embedding_dim"]),
        classes=len(known_models),
    ).to(device)
    checkpoint = torch.load(
        replication_paths[seed]["checkpoint"],
        map_location="cpu",
        weights_only=True,
    )
    _load_head(model, checkpoint["head_state"])
    target_samples = int(
        config["data"]["sample_rate"] * config["data"]["clip_seconds"]
    )
    datasets = {
        name: RegistryDataset(paths[name], target_samples)
        for name in DATASET_NAMES
    }
    batch_size = int(config["outlier_exposure"]["batch_size"])
    recording_frames: dict[str, pd.DataFrame] = {}
    recording_embeddings: dict[str, np.ndarray] = {}
    recording_logits: dict[str, np.ndarray] = {}
    for name, dataset in datasets.items():
        segment_embeddings, segment_logits = infer_outputs(
            model,
            dataset,
            batch_size=batch_size,
            device=device,
            label=f"g18_p5_seed_{seed}_{name}",
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
            raise RuntimeError("G18 P5 recording aggregation order mismatch")
    pca, scaler, classifier = fit_linear_ood_head(
        recording_embeddings["known_train"],
        recording_embeddings["unknown_oe_train"],
        components=int(config["outlier_exposure"]["pca_components"]),
        regularization_c=float(
            config["outlier_exposure"]["logistic_regularization_c"]
        ),
        seed=int(config["outlier_exposure"]["fit_seed"]),
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
    predictions = recording_logits["known_tune"].argmax(axis=1)
    targets = recording_frames["known_tune"]["target_index"].to_numpy(
        dtype=np.int64
    )
    correct = predictions == targets
    metrics["known_closed_set_accuracy"] = float(np.mean(correct))
    metrics["known_end_to_end_accuracy"] = float(
        np.mean(correct & (tune_known_scores >= threshold))
    )
    gate_passed, gate_checks = calibration_gate(metrics, config["gates"])
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "calibration_recording_predictions.csv"
    rows = []
    for name, scores, model_predictions in (
        ("known_tune", tune_known_scores, predictions),
        (
            "unknown_calibration",
            tune_unknown_scores,
            recording_logits["unknown_calibration"].argmax(axis=1),
        ),
    ):
        for row, score, prediction in zip(
            recording_frames[name].to_dict("records"),
            scores,
            model_predictions,
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
    temporary = model_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
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
    os.replace(temporary, model_path)
    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "decision": (
            "eligible_for_multiseed_aggregation"
            if gate_passed
            else "stop_before_holdout"
        ),
        "probe_gate_passed": gate_passed,
        "probe_gate_checks": gate_checks,
        "known_confidence_threshold": threshold,
        "tune_metrics": metrics,
        "frozen_outlier_exposure": {
            "fit_seed": int(config["outlier_exposure"]["fit_seed"]),
            "split_seed": int(config["outlier_exposure"]["split_seed"]),
            "pca_components": int(pca.n_components_),
            "logistic_regularization_c": float(
                config["outlier_exposure"]["logistic_regularization_c"]
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
            "source_manifest": preflight_report["inputs"]["source_manifest"],
            "preflight_sha256": file_sha256(preflight_path),
            **{f"{name}_sha256": value for name, value in hashes.items()},
            "model_id_checkpoint_sha256": replication_hashes[seed]["checkpoint"],
            "model_id_summary_sha256": replication_hashes[seed]["summary"],
        },
    }
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def aggregate(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config, *_ = validate_inputs(config_path, root)
    preflight_path = root / str(config["preflight_output_dir"]) / "report.json"
    _validate_preflight(config_path, root)
    seed_reports = {}
    for seed in SEEDS:
        path = root / str(config["output_dir"]) / f"seed_{seed}" / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"G18 P5 seed {seed} summary is missing")
        report = json.loads(path.read_text(encoding="utf-8"))
        if not (
            report.get("passed") is True
            and report.get("protocol") == PROTOCOL
            and report.get("seed") == seed
            and report.get("probe_gate_passed") is True
            and report.get("known_holdout_read") is False
            and report.get("unknown_holdout_read") is False
            and report.get("inputs", {}).get("config_sha256")
            == file_sha256(config_path)
            and report.get("inputs", {}).get("implementation_sha256")
            == file_sha256(Path(__file__))
            and report.get("inputs", {}).get("preflight_sha256")
            == file_sha256(preflight_path)
        ):
            raise ValueError(f"G18 P5 seed {seed} result is invalid or failed")
        seed_reports[seed] = (path, report)
    p3d_path = root / config["inputs"]["p3d_summary"]["path"]
    p3d_report = json.loads(p3d_path.read_text(encoding="utf-8"))
    metrics_by_seed = {
        42: p3d_report["tune_metrics"],
        **{seed: report["tune_metrics"] for seed, (_, report) in seed_reports.items()},
    }
    aggregate_metrics = {}
    for name in AGGREGATE_METRICS:
        values = np.asarray(
            [metrics_by_seed[seed][name] for seed in (42, 43, 44)],
            dtype=np.float64,
        )
        aggregate_metrics[name] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    output_dir = root / str(config["output_dir"])
    report_path = output_dir / "summary.json"
    if report_path.exists():
        raise FileExistsError("Refusing to overwrite G18 P5 aggregate summary")
    report = {
        "passed": True,
        "protocol": f"{PROTOCOL}_aggregate_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "ready_to_freeze_single_holdout_protocol",
        "seeds": [42, 43, 44],
        "all_seed_gates_passed": True,
        "metrics_by_seed": {
            str(seed): metrics_by_seed[seed] for seed in (42, 43, 44)
        },
        "aggregate_metrics": aggregate_metrics,
        "selection_rule": "report_mean_sample_std_and_worst_seed_no_seed_selection",
        "known_holdout_read": False,
        "unknown_holdout_read": False,
        "locked_datasets_read": [],
        "inputs": {
            "config_sha256": file_sha256(config_path),
            "implementation_sha256": file_sha256(Path(__file__)),
            "source_manifest": _source_manifest(root),
            "preflight_sha256": file_sha256(preflight_path),
            "seed_42_summary_sha256": file_sha256(p3d_path),
            **{
                f"seed_{seed}_summary_sha256": file_sha256(path)
                for seed, (path, _) in seed_reports.items()
            },
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run G18 P5 fixed multiseed outlier-exposure replication."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g18_multiseed_outlier_exposure.yaml"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    selected = sum(
        [bool(args.preflight_only), args.seed is not None, bool(args.aggregate)]
    )
    if selected != 1:
        parser.error("select exactly one of --preflight-only, --seed, or --aggregate")
    if args.preflight_only:
        preflight(args.config, args.root)
    elif args.aggregate:
        aggregate(args.config, args.root)
    else:
        run_seed(args.config, args.root, int(args.seed))


if __name__ == "__main__":
    main()
