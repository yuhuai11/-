from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .config import load_config
from .data_firewall import (
    audit_csv_rows,
    compact_token,
    file_sha256,
    reject_locked_path,
)
from .g19_model import G19RecordingHead, supervised_contrastive_loss
from .g19_recording_data import (
    RecordingExample,
    RecordingFeatureDataset,
    assert_disjoint_recordings,
    balanced_recording_batches,
    build_recording_examples,
    collate_recordings,
)
from .panns import PannsCnn14Binary
from .train import resolve_device, set_seed
from .train_g18_model_id import RegistryDataset, classification_metrics


PROTOCOL = "g19_p1_supcon_attention_recording_v1"
SCHEMA_VERSION = 1
FORBIDDEN_SPLIT_NAMES = {
    "known_holdout",
    "unknown_holdout",
    "holdout",
    "final",
    "test",
}
FORBIDDEN_G18_VALUE_TOKENS = {
    "knownholdout",
    "unknownholdout",
    "finalholdout",
    "g18p6",
}
REQUIRED_COLUMNS = (
    "model_id",
    "target_index",
    "is_known",
    "audio_sha256",
    "segment_index",
    "cache_path",
    "cache_index",
)
HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "train_cross_entropy",
    "train_supcon",
    "tune_loss",
    "tune_accuracy",
    "tune_macro_f1",
    "tune_minimum_recall",
    "tune_attention_entropy",
    "improved",
    "bad_epochs",
)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_numpy_save(values: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def _write_history(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "state": torch.from_numpy(
                np.asarray(numpy_state[1], dtype=np.int64).copy()
            ),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise ValueError("G19 checkpoint RNG state is incomplete")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_state["state"].cpu().numpy().astype(np.uint32, copy=False),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = list(state["torch_cuda"])
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("G19 CUDA RNG state cannot be restored without CUDA")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("G19 CUDA device count changed across resume")
        torch.cuda.set_rng_state_all(cuda_states)


def _resolve(root: Path, value: object, *, context: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    reject_locked_path(path, context=context)
    return path.resolve(strict=True)


def resolve_g19_output(root: Path, value: object, *, context: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{context} must remain inside the workspace root") from error
    reject_locked_path(path, context=context)
    return path


def _contains_forbidden_split(value: object) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_SPLIT_NAMES or "holdout" in normalized:
                return True
            if _contains_forbidden_split(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_forbidden_split(child) for child in value)
    elif isinstance(value, (str, Path)):
        normalized = compact_token(value)
        if any(token in normalized for token in FORBIDDEN_G18_VALUE_TOKENS):
            return True
        components = {
            token
            for token in re.split(r"[^a-z0-9]+", str(value).lower())
            if token
        }
        return bool(components & {"x6d", "y6"})
    return False


def _validate_protocol_config(config: dict[str, Any]) -> None:
    expected = {
        ("data", "split_unit"): "audio_sha256",
        ("data", "sample_rate"): 16000,
        ("data", "clip_seconds"): 1.0,
        ("data", "maximum_segments_per_recording"): 20,
        ("model", "freeze_g7"): True,
        ("representation", "normalize_segment_embedding"): True,
        ("representation", "normalize_recording_embedding"): True,
        ("representation", "aggregation"): "gated_attention",
        ("open_set", "covariance_estimator"): "oas",
        ("open_set", "distance"): "normalized_squared_mahalanobis",
        ("open_set", "class_assignment"): "classifier_argmax",
        (
            "open_set",
            "insufficient_support_policy",
        ): "global_with_predicted_class_known_protection",
        ("reporting", "primary_unit"): "recording",
        ("reporting", "report_attention_entropy"): True,
        ("reporting", "report_conservative_fallback_classes"): True,
        ("reporting", "report_per_class_recall"): True,
    }
    mismatches = {}
    for (section, key), required in expected.items():
        observed = config.get(section, {}).get(key)
        if observed != required:
            mismatches[f"{section}.{key}"] = {
                "required": required,
                "observed": observed,
            }
    if mismatches:
        raise ValueError(f"G19 config contradicts its fixed protocol: {mismatches}")
    if int(config["open_set"]["seed"]) != int(config["train"]["seed"]):
        raise ValueError("G19 representation and open-set seeds must match")
    if not math.isclose(
        float(config["gates"]["minimum_tune_known_acceptance"]),
        float(config["open_set"]["minimum_known_acceptance"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("G19 Known-acceptance gate and calibration constraint differ")
    for name, raw_value in config["gates"].items():
        value = float(raw_value)
        if not math.isfinite(value) or not (0.0 <= value <= 1.0):
            raise ValueError(f"G19 gate {name} must be inside [0, 1]")


def _verify_inputs(
    config: dict[str, Any], root: Path
) -> tuple[dict[str, Path], dict[str, str], dict[str, Any]]:
    if str(config.get("protocol")) != PROTOCOL:
        raise ValueError(f"Unexpected G19 protocol; expected {PROTOCOL}")
    _validate_protocol_config(config)
    if _contains_forbidden_split(config.get("inputs", {})):
        raise ValueError("G19 development config must not bind Holdout/final/test inputs")

    input_names = (
        "known_train",
        "known_tune",
        "unknown_tune",
        "registry_audit",
        "segment_cache_audit",
    )
    paths = {
        name: _resolve(
            root,
            config["inputs"][name]["path"],
            context=f"G19 {name} development input",
        )
        for name in input_names
    }
    paths["official_checkpoint"] = _resolve(
        root,
        config["model"]["checkpoint_path"],
        context="G19 official PANNs checkpoint",
    )
    paths["g7_checkpoint"] = _resolve(
        root,
        config["model"]["binary_checkpoint_path"],
        context="G19 frozen G7 checkpoint",
    )
    paths["vendor_dir"] = _resolve(
        root,
        config["model"]["vendor_dir"],
        context="G19 PANNs vendor source",
    )
    expected = {
        name: str(config["inputs"][name]["sha256"]) for name in input_names
    }
    expected.update(
        {
            "official_checkpoint": str(config["model"]["checkpoint_sha256"]),
            "g7_checkpoint": str(config["model"]["binary_checkpoint_sha256"]),
        }
    )
    observed = {name: file_sha256(paths[name]) for name in expected}
    mismatches = {
        name: {"expected": expected[name], "observed": observed[name]}
        for name in expected
        if observed[name] != expected[name]
    }
    if mismatches:
        raise ValueError(f"G19 input SHA256 mismatch: {mismatches}")

    for name in ("known_train", "known_tune", "unknown_tune"):
        audit_csv_rows(paths[name], required_columns=REQUIRED_COLUMNS)
    registry = json.loads(paths["registry_audit"].read_text(encoding="utf-8"))
    segment_cache_audit = json.loads(
        paths["segment_cache_audit"].read_text(encoding="utf-8")
    )
    if not (
        registry.get("passed") is True
        and registry.get("protocol") == "g18_p0_open_set_model_id_registry_v1"
        and registry.get("locked_datasets_read") == []
    ):
        raise ValueError("G19 requires a valid, firewall-audited G18 registry")
    if not (
        segment_cache_audit.get("passed") is True
        and segment_cache_audit.get("protocol")
        == "g14_controlled_segment_cache_v1"
        and int(segment_cache_audit.get("target_sample_rate", -1)) == 16000
        and int(segment_cache_audit.get("target_samples", -1)) == 16000
        and registry.get("inputs", {}).get("segment_audit_sha256")
        == observed["segment_cache_audit"]
    ):
        raise ValueError("G19 requires the registry-bound segment-cache audit")
    consumed_manifest_sha256 = {
        str(registry["outputs"][name]["sha256"])
        for name in ("known_holdout", "unknown_holdout")
    }
    for name in ("known_train", "known_tune", "unknown_tune"):
        registry_sha256 = str(registry["outputs"][name]["sha256"])
        if expected[name] != registry_sha256 or observed[name] != registry_sha256:
            raise ValueError(
                f"G19 {name} is not the registry-bound development manifest"
            )
    if any(
        observed[name] in consumed_manifest_sha256
        for name in ("known_train", "known_tune", "unknown_tune")
    ):
        raise ValueError("G19 input is a consumed G18 P6 manifest")
    if [
        str(value) for value in config["reporting"]["known_models"]
    ] != [str(value) for value in registry["known_models"]]:
        raise ValueError("G19 reporting Known-model order differs from the registry")
    if {
        str(value) for value in config["reporting"]["development_unknown_models"]
    } != {str(value) for value in registry["unknown_tune_models"]}:
        raise ValueError("G19 reporting Unknown models differ from the registry")
    return paths, observed, registry


def _protocol_source_sha256() -> dict[str, str]:
    source_directory = Path(__file__).parent
    names = (
        "g19_model.py",
        "g19_open_set.py",
        "g19_recording_data.py",
        "probe_g19_class_conditional.py",
        "train_g19_representation.py",
    )
    return {
        name: file_sha256(source_directory / name)
        for name in names
    }


def _preflight_directory(config: dict[str, Any], root: Path) -> Path:
    return resolve_g19_output(
        root,
        config.get(
            "preflight_output_dir",
            Path(str(config["output_dir"])) / "preflight",
        ),
        context="G19 preflight output",
    )


def _require_preflight(
    config: dict[str, Any],
    config_path: Path,
    root: Path,
    observed_inputs: dict[str, str],
    paths: dict[str, Path],
) -> dict[str, Any]:
    report_path = _preflight_directory(config, root) / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError("G19 preflight report is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required = (
        report.get("passed") is True
        and report.get("ready_for_training") is True
        and report.get("protocol") == PROTOCOL
        and report.get("config_sha256") == file_sha256(config_path)
        and report.get("input_sha256") == observed_inputs
        and report.get("source_sha256") == _protocol_source_sha256()
        and report.get("extractor_source_sha256")
        == _extractor_source_sha256(paths)
        and report.get("known_holdout_read") is False
        and report.get("unknown_holdout_read") is False
        and report.get("locked_datasets_read") == []
    )
    if not required:
        raise ValueError(
            "G19 preflight is stale or does not authorize this exact training run"
        )
    return report


def _read_validated_frames(
    paths: dict[str, Path], known_models: list[str]
) -> dict[str, pd.DataFrame]:
    frames = {
        name: pd.read_csv(paths[name])
        for name in ("known_train", "known_tune", "unknown_tune")
    }
    examples = {
        name: build_recording_examples(frame) for name, frame in frames.items()
    }
    for name in ("known_train", "known_tune"):
        frame = frames[name]
        if not all(example.is_known for example in examples[name]):
            raise ValueError(f"G19 {name} contains Unknown rows")
        expected_targets = set(range(len(known_models)))
        if {example.target for example in examples[name]} != expected_targets:
            raise ValueError(f"G19 {name} does not contain every Known target")
        mapping = sorted(
            {(example.target, example.model_id) for example in examples[name]}
        )
        if (
            [model_id for _, model_id in mapping] != known_models
            or [target for target, _ in mapping] != list(range(len(known_models)))
        ):
            raise ValueError(f"G19 {name} model-to-target mapping changed")
    if any(example.is_known for example in examples["unknown_tune"]):
        raise ValueError("G19 unknown_tune contains Known rows")
    if {example.target for example in examples["unknown_tune"]} != {-1}:
        raise ValueError("G19 unknown_tune must use target_index=-1")
    assert_disjoint_recordings(frames)
    return frames


def preflight(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    paths, observed, registry = _verify_inputs(config, root)
    known_models = [str(value) for value in registry["known_models"]]
    frames = _read_validated_frames(paths, known_models)
    if {
        str(value) for value in config["reporting"]["development_unknown_models"]
    } != set(frames["unknown_tune"]["model_id"].astype(str)):
        raise ValueError("G19 reporting Unknown Tune models differ from the manifest")
    examples = {
        name: build_recording_examples(frame) for name, frame in frames.items()
    }

    recording_counts = {name: len(values) for name, values in examples.items()}
    segment_counts = {name: int(len(frame)) for name, frame in frames.items()}
    for name in frames:
        configured = config["inputs"][name]
        if "rows" in configured and int(configured["rows"]) != segment_counts[name]:
            raise ValueError(f"G19 configured row count changed for {name}")
        if (
            "recordings" in configured
            and int(configured["recordings"]) != recording_counts[name]
        ):
            raise ValueError(f"G19 configured recording count changed for {name}")
    maximum_segments = int(config["data"]["maximum_segments_per_recording"])
    if maximum_segments <= 0 or any(
        len(example.indices) > maximum_segments
        for values in examples.values()
        for example in values
    ):
        raise ValueError("G19 recording exceeds maximum_segments_per_recording")
    minimum_per_class = int(config["train"]["recordings_per_class"])
    if minimum_per_class < 2:
        raise ValueError("G19 SupCon requires recordings_per_class >= 2")
    class_recordings = (
        frames["known_train"]
        .groupby("target_index")["audio_sha256"]
        .nunique()
        .sort_index()
        .astype(int)
        .tolist()
    )
    if min(class_recordings) < minimum_per_class:
        raise ValueError("A G19 Known class cannot populate one balanced batch")
    preview_batches = balanced_recording_batches(
        examples["known_train"],
        recordings_per_class=minimum_per_class,
        seed=int(config["train"]["seed"]),
        epoch=0,
    )
    head = build_head(config, len(known_models))
    train_values = (
        float(config["train"]["learning_rate"]),
        float(config["train"]["weight_decay"]),
        float(config["train"]["temperature"]),
        float(config["train"]["supcon_weight"]),
        float(config["train"]["gradient_clip_norm"]),
    )
    if (
        not all(math.isfinite(value) for value in train_values)
        or train_values[0] <= 0.0
        or train_values[1] < 0.0
        or train_values[2] <= 0.0
        or train_values[3] < 0.0
        or train_values[4] <= 0.0
    ):
        raise ValueError("G19 training hyperparameters are invalid")

    report = {
        "passed": True,
        "execution_completed": True,
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": file_sha256(config_path),
        "input_sha256": observed,
        "source_sha256": _protocol_source_sha256(),
        "extractor_source_sha256": _extractor_source_sha256(paths),
        "known_models": known_models,
        "segments": segment_counts,
        "raw_recordings": recording_counts,
        "known_train_recordings_per_class": class_recordings,
        "balanced_batches_per_epoch": len(preview_batches),
        "balanced_batch_recordings": int(len(preview_batches[0])),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in head.parameters())
        ),
        "optimization_inputs": ["known_train"],
        "model_selection_inputs": ["known_tune"],
        "open_set_calibration_inputs": ["known_tune", "unknown_tune"],
        "known_holdout_read": False,
        "unknown_holdout_read": False,
        "locked_datasets_read": [],
        "ready_for_training": True,
    }
    output = _preflight_directory(config, root)
    _atomic_json(report, output / "report.json")
    return report


def _build_detector(
    config: dict[str, Any],
    paths: dict[str, Path],
    observed: dict[str, str],
    device: torch.device,
) -> PannsCnn14Binary:
    detector = PannsCnn14Binary(
        initialization=str(config["model"]["initialization"]),
        vendor_dir=paths["vendor_dir"].as_posix(),
        checkpoint_path=paths["official_checkpoint"].as_posix(),
        checkpoint_sha256=observed["official_checkpoint"],
        spec_augment=False,
        frontend_precision="float32",
        binary_checkpoint_path=paths["g7_checkpoint"].as_posix(),
        binary_checkpoint_sha256=observed["g7_checkpoint"],
        trainable_scope="binary_head_only",
    )
    for parameter in detector.parameters():
        parameter.requires_grad = False
    detector.eval()
    return detector.to(device)


@torch.no_grad()
def _extract_segment_features(
    detector: PannsCnn14Binary,
    dataset: RegistryDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("G19 extraction batch size must be positive")
    detector.eval()
    chunks: list[np.ndarray] = []
    for start in range(0, len(dataset.frame), batch_size):
        positions = np.arange(
            start, min(start + batch_size, len(dataset.frame)), dtype=np.int64
        )
        waveforms, _ = dataset.batch(positions)
        features = detector.extract_embedding(waveforms.to(device))
        chunks.append(features.float().cpu().numpy())
    result = np.concatenate(chunks).astype(np.float32, copy=False)
    if len(result) != len(dataset.frame) or not np.isfinite(result).all():
        raise RuntimeError("G19 extracted invalid frozen-G7 features")
    return result


def _feature_identity(
    *,
    split_name: str,
    manifest_sha256: str,
    g7_checkpoint_sha256: str,
    official_checkpoint_sha256: str,
    segment_cache_audit_sha256: str,
    rows: int,
    dimension: int,
    target_samples: int,
    extractor_source_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "g19_frozen_g7_segment_features",
        "split_name": split_name,
        "manifest_sha256": manifest_sha256,
        "g7_checkpoint_sha256": g7_checkpoint_sha256,
        "official_checkpoint_sha256": official_checkpoint_sha256,
        "segment_cache_audit_sha256": segment_cache_audit_sha256,
        "rows": int(rows),
        "dimension": int(dimension),
        "target_samples": int(target_samples),
        "dtype": "float32",
        "extractor_source_sha256": extractor_source_sha256,
    }


def _python_tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in directory.rglob("*.py") if path.is_file())
    if not files:
        raise ValueError(f"G19 source tree contains no Python files: {directory}")
    for path in files:
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def _extractor_source_sha256(paths: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    panns_source = Path(__file__).with_name("panns.py")
    registry_dataset_source = Path(__file__).with_name("train_g18_model_id.py")
    for label, value in (
        ("dads_crnn_panns", file_sha256(panns_source)),
        ("registry_dataset", file_sha256(registry_dataset_source)),
        ("vendor_python_tree", _python_tree_sha256(paths["vendor_dir"])),
    ):
        digest.update(label.encode("utf-8"))
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _root_registry_cache_paths(
    dataset: RegistryDataset, workspace_root: Path
) -> None:
    resolved: dict[str, str] = {}
    for value in dataset.frame["cache_path"].astype(str).unique():
        path = Path(value)
        if not path.is_absolute():
            path = workspace_root / path
        reject_locked_path(path, context="G19 frozen-G7 waveform cache")
        resolved[value] = path.resolve(strict=True).as_posix()
    dataset.frame["cache_path"] = dataset.frame["cache_path"].astype(str).map(resolved)


def ensure_feature_cache(
    *,
    split_name: str,
    manifest_path: Path,
    manifest_sha256: str,
    output_dir: Path,
    feature_dir: Path | None,
    workspace_root: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    observed: dict[str, str],
    device: torch.device,
    detector: PannsCnn14Binary | None = None,
) -> tuple[np.ndarray, dict[str, Any], PannsCnn14Binary | None]:
    """Load or deterministically create a segment-feature cache.

    The returned optional detector lets callers reuse the expensive PANNs
    instance when creating several split caches in one process.
    """
    cache_dir = output_dir / "features" if feature_dir is None else feature_dir
    feature_path = cache_dir / f"{split_name}.npy"
    metadata_path = cache_dir / f"{split_name}.json"
    expected_dimension = int(config["representation"]["g7_embedding_dim"])
    target_samples = int(config["data"]["sample_rate"]) * float(
        config["data"]["clip_seconds"]
    )
    if not float(target_samples).is_integer():
        raise ValueError("G19 target sample count must be an integer")
    extractor_hash = _extractor_source_sha256(paths)
    frame = pd.read_csv(manifest_path)
    expected_identity = _feature_identity(
        split_name=split_name,
        manifest_sha256=manifest_sha256,
        g7_checkpoint_sha256=observed["g7_checkpoint"],
        official_checkpoint_sha256=observed["official_checkpoint"],
        segment_cache_audit_sha256=observed["segment_cache_audit"],
        rows=len(frame),
        dimension=expected_dimension,
        target_samples=int(target_samples),
        extractor_source_sha256=extractor_hash,
    )
    if feature_path.is_file() or metadata_path.is_file():
        if not (feature_path.is_file() and metadata_path.is_file()):
            raise ValueError(f"Incomplete G19 feature cache for {split_name}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        identity = {key: metadata.get(key) for key in expected_identity}
        if identity != expected_identity:
            raise ValueError(f"Stale G19 feature cache identity for {split_name}")
        if metadata.get("feature_sha256") != file_sha256(feature_path):
            raise ValueError(f"Corrupt G19 feature cache for {split_name}")
        values = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if (
            values.shape != (len(frame), expected_dimension)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"Invalid G19 feature cache payload for {split_name}")
        return values, metadata, detector

    if detector is None:
        detector = _build_detector(config, paths, observed, device)
    dataset = RegistryDataset(manifest_path, int(target_samples))
    _root_registry_cache_paths(dataset, workspace_root)
    values = _extract_segment_features(
        detector,
        dataset,
        batch_size=int(config["train"]["extraction_batch_size"]),
        device=device,
    )
    if values.shape[1] != expected_dimension:
        raise ValueError("Frozen G7 emitted an unexpected embedding dimension")
    _atomic_numpy_save(values, feature_path)
    metadata = {
        **expected_identity,
        "feature_path": feature_path.as_posix(),
        "feature_sha256": file_sha256(feature_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(metadata, metadata_path)
    return np.load(feature_path, mmap_mode="r", allow_pickle=False), metadata, detector


def build_head(config: dict[str, Any], classes: int) -> G19RecordingHead:
    representation = config["representation"]
    return G19RecordingHead(
        g7_embedding_dim=int(representation["g7_embedding_dim"]),
        projection_hidden_dim=int(representation["projection_hidden_dim"]),
        embedding_dim=int(representation["embedding_dim"]),
        attention_hidden_dim=int(representation["attention_hidden_dim"]),
        classes=classes,
        dropout=float(representation["dropout"]),
    )


def _head_state(head: G19RecordingHead) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu() for name, value in head.state_dict().items()
    }


def _attention_entropy(attention: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_counts = mask.sum(dim=1)
    entropy = -torch.sum(
        torch.where(
            mask,
            attention.clamp_min(torch.finfo(attention.dtype).tiny).log()
            * attention,
            torch.zeros_like(attention),
        ),
        dim=1,
    )
    denominator = torch.log(valid_counts.float())
    return torch.where(
        valid_counts > 1,
        entropy / denominator.clamp_min(torch.finfo(entropy.dtype).tiny),
        torch.zeros_like(entropy),
    )


def _dataset_batch(
    dataset: RecordingFeatureDataset,
    indices: Iterable[int],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[RecordingExample, ...],
]:
    return collate_recordings([dataset[int(index)] for index in indices])


@torch.no_grad()
def evaluate_head(
    head: G19RecordingHead,
    dataset: RecordingFeatureDataset,
    *,
    batch_size: int,
    device: torch.device,
    classes: int,
) -> tuple[
    float,
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    list[RecordingExample],
]:
    head.eval()
    all_logits: list[np.ndarray] = []
    all_embeddings: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    metadata: list[RecordingExample] = []
    total_loss = 0.0
    entropy_sum = 0.0
    for start in range(0, len(dataset), batch_size):
        indices = range(start, min(start + batch_size, len(dataset)))
        features, mask, targets, batch_metadata = _dataset_batch(dataset, indices)
        output = head(features.to(device), mask.to(device))
        total_loss += float(
            F.cross_entropy(
                output.logits.float(),
                targets.to(device),
                reduction="sum",
            ).cpu()
        )
        entropy_sum += float(
            _attention_entropy(output.attention, mask.to(device)).sum().cpu()
        )
        all_logits.append(output.logits.float().cpu().numpy())
        all_embeddings.append(output.embedding.float().cpu().numpy())
        all_targets.append(targets.numpy())
        metadata.extend(batch_metadata)
    logits = np.concatenate(all_logits)
    embeddings = np.concatenate(all_embeddings)
    targets = np.concatenate(all_targets)
    metrics = classification_metrics(targets, logits, classes)
    metrics["mean_normalized_attention_entropy"] = entropy_sum / len(dataset)
    return total_loss / len(dataset), metrics, logits, embeddings, metadata


@torch.no_grad()
def infer_head(
    head: G19RecordingHead,
    dataset: RecordingFeatureDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[RecordingExample],
]:
    """Infer arbitrary Known or Unknown recordings without using their labels."""
    head.eval()
    all_logits: list[np.ndarray] = []
    all_embeddings: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_entropies: list[np.ndarray] = []
    metadata: list[RecordingExample] = []
    for start in range(0, len(dataset), batch_size):
        indices = range(start, min(start + batch_size, len(dataset)))
        features, mask, targets, batch_metadata = _dataset_batch(dataset, indices)
        output = head(features.to(device), mask.to(device))
        all_logits.append(output.logits.float().cpu().numpy())
        all_embeddings.append(output.embedding.float().cpu().numpy())
        all_targets.append(targets.numpy())
        all_entropies.append(
            _attention_entropy(output.attention, mask.to(device)).cpu().numpy()
        )
        metadata.extend(batch_metadata)
    return (
        np.concatenate(all_logits),
        np.concatenate(all_embeddings),
        np.concatenate(all_targets),
        np.concatenate(all_entropies),
        metadata,
    )


def train(config_path: Path, root: Path, *, resume: bool = False) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    paths, observed, registry = _verify_inputs(config, root)
    known_models = [str(value) for value in registry["known_models"]]
    frames = _read_validated_frames(paths, known_models)
    preflight_report = _require_preflight(
        config, config_path, root, observed, paths
    )
    seed = int(config["train"]["seed"])
    set_seed(seed)
    device = resolve_device(str(config["train"]["device"]))
    output_dir = resolve_g19_output(
        root, config["output_dir"], context="G19 representation output"
    )
    feature_dir = resolve_g19_output(
        root,
        config.get("feature_cache_dir", Path(str(config["output_dir"])) / "features"),
        context="G19 feature-cache output",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    history_path = output_dir / "history.csv"
    summary_path = output_dir / "summary.json"
    if resume and summary_path.exists():
        raise ValueError("G19 representation run is already complete")
    if not resume and any(
        path.exists()
        for path in (latest_path, best_path, history_path, summary_path)
    ):
        raise ValueError(
            "G19 representation artifacts already exist; use --resume if incomplete"
        )

    detector: PannsCnn14Binary | None = None
    train_features, train_cache, detector = ensure_feature_cache(
        split_name="known_train",
        manifest_path=paths["known_train"],
        manifest_sha256=observed["known_train"],
        output_dir=output_dir,
        feature_dir=feature_dir,
        workspace_root=root,
        config=config,
        paths=paths,
        observed=observed,
        device=device,
        detector=detector,
    )
    tune_features, tune_cache, detector = ensure_feature_cache(
        split_name="known_tune",
        manifest_path=paths["known_tune"],
        manifest_sha256=observed["known_tune"],
        output_dir=output_dir,
        feature_dir=feature_dir,
        workspace_root=root,
        config=config,
        paths=paths,
        observed=observed,
        device=device,
        detector=detector,
    )
    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_data = RecordingFeatureDataset(frames["known_train"], train_features)
    tune_data = RecordingFeatureDataset(frames["known_tune"], tune_features)
    classes = len(known_models)
    head = build_head(config, classes).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "config_sha256": file_sha256(config_path),
        "known_train_sha256": observed["known_train"],
        "known_tune_sha256": observed["known_tune"],
        "registry_audit_sha256": observed["registry_audit"],
        "segment_cache_audit_sha256": observed["segment_cache_audit"],
        "official_checkpoint_sha256": observed["official_checkpoint"],
        "g7_checkpoint_sha256": observed["g7_checkpoint"],
        "known_train_feature_sha256": train_cache["feature_sha256"],
        "known_tune_feature_sha256": tune_cache["feature_sha256"],
        "source_sha256": preflight_report["source_sha256"],
        "extractor_source_sha256": preflight_report[
            "extractor_source_sha256"
        ],
    }
    start_epoch = 1
    best_macro_f1 = -1.0
    best_minimum_recall = -1.0
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    if resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"G19 resume checkpoint not found: {latest_path}")
        saved = torch.load(latest_path, map_location="cpu", weights_only=True)
        if saved.get("identity") != identity:
            raise ValueError("G19 resume identity mismatch")
        head.load_state_dict(saved["head_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved["epoch"]) + 1
        best_macro_f1 = float(saved["best_macro_f1"])
        best_minimum_recall = float(saved["best_minimum_recall"])
        best_epoch = int(saved["best_epoch"])
        bad_epochs = int(saved["bad_epochs"])
        history = list(saved["history"])
        _restore_rng_state(saved["rng_state"])

    epochs = int(config["train"]["epochs"])
    patience = int(config["train"]["patience"])
    recordings_per_class = int(config["train"]["recordings_per_class"])
    temperature = float(config["train"]["temperature"])
    supcon_weight = float(config["train"]["supcon_weight"])
    label_smoothing = float(config["train"].get("label_smoothing", 0.0))
    log_every = int(config["train"].get("log_every_steps", 10))
    tune_batch_size = max(classes * recordings_per_class, 32)
    if resume and bad_epochs >= patience:
        print(
            "G19 resume checkpoint had already reached early stopping; "
            "finalizing its existing best checkpoint",
            flush=True,
        )
        start_epoch = epochs + 1

    for epoch in range(start_epoch, epochs + 1):
        head.train()
        running_total = 0.0
        running_ce = 0.0
        running_supcon = 0.0
        processed = 0
        batches = balanced_recording_batches(
            train_data.examples,
            recordings_per_class=recordings_per_class,
            seed=seed,
            epoch=epoch,
        )
        for step, indices in enumerate(batches, start=1):
            features, mask, targets, _ = _dataset_batch(train_data, indices)
            features = features.to(device)
            mask = mask.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = head(features, mask)
            cross_entropy = F.cross_entropy(
                output.logits.float(),
                targets,
                label_smoothing=label_smoothing,
            )
            supcon = supervised_contrastive_loss(
                output.embedding,
                targets,
                temperature=temperature,
            )
            loss = cross_entropy + supcon_weight * supcon
            if not torch.isfinite(loss):
                raise RuntimeError("G19 produced a non-finite training loss")
            loss.backward()
            gradients = [
                parameter.grad for parameter in head.parameters() if parameter.requires_grad
            ]
            if not all(
                gradient is not None and torch.isfinite(gradient).all()
                for gradient in gradients
            ):
                raise RuntimeError("G19 produced missing or non-finite gradients")
            torch.nn.utils.clip_grad_norm_(
                head.parameters(),
                float(config["train"]["gradient_clip_norm"]),
            )
            optimizer.step()
            count = len(targets)
            running_total += float(loss.detach().cpu()) * count
            running_ce += float(cross_entropy.detach().cpu()) * count
            running_supcon += float(supcon.detach().cpu()) * count
            processed += count
            if log_every > 0 and step % log_every == 0:
                print(
                    f"G19 epoch {epoch} step {step}: "
                    f"loss={float(loss.detach().cpu()):.6f} "
                    f"ce={float(cross_entropy.detach().cpu()):.6f} "
                    f"supcon={float(supcon.detach().cpu()):.6f}",
                    flush=True,
                )

        tune_loss, tune_metrics, _, _, _ = evaluate_head(
            head,
            tune_data,
            batch_size=tune_batch_size,
            device=device,
            classes=classes,
        )
        improved = (
            tune_metrics["macro_f1"] > best_macro_f1 + 1.0e-8
            or (
                abs(tune_metrics["macro_f1"] - best_macro_f1) <= 1.0e-8
                and tune_metrics["minimum_recall"]
                > best_minimum_recall + 1.0e-8
            )
        )
        if improved:
            best_macro_f1 = float(tune_metrics["macro_f1"])
            best_minimum_recall = float(tune_metrics["minimum_recall"])
            best_epoch = epoch
            bad_epochs = 0
            _atomic_torch_save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "protocol": PROTOCOL,
                    "epoch": epoch,
                    "seed": seed,
                    "head_state": _head_state(head),
                    "architecture": dict(config["representation"]),
                    "known_models": known_models,
                    "identity": identity,
                    "tune_recording_metrics": tune_metrics,
                    "known_holdout_read": False,
                    "unknown_holdout_read": False,
                    "locked_datasets_read": [],
                },
                best_path,
            )
        else:
            bad_epochs += 1
        row = {
            "epoch": epoch,
            "train_loss": running_total / processed,
            "train_cross_entropy": running_ce / processed,
            "train_supcon": running_supcon / processed,
            "tune_loss": tune_loss,
            "tune_accuracy": tune_metrics["accuracy"],
            "tune_macro_f1": tune_metrics["macro_f1"],
            "tune_minimum_recall": tune_metrics["minimum_recall"],
            "tune_attention_entropy": tune_metrics[
                "mean_normalized_attention_entropy"
            ],
            "improved": improved,
            "bad_epochs": bad_epochs,
        }
        history.append(row)
        _write_history(history, history_path)
        _atomic_torch_save(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "epoch": epoch,
                "head_state": _head_state(head),
                "optimizer": optimizer.state_dict(),
                "identity": identity,
                "best_macro_f1": best_macro_f1,
                "best_minimum_recall": best_minimum_recall,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "history": history,
                "rng_state": _capture_rng_state(),
            },
            latest_path,
        )
        print(
            f"G19 epoch {epoch}: tune_macro_f1={tune_metrics['macro_f1']:.6f} "
            f"tune_min_recall={tune_metrics['minimum_recall']:.6f} "
            f"attention_entropy="
            f"{tune_metrics['mean_normalized_attention_entropy']:.6f} "
            f"improved={improved}",
            flush=True,
        )
        if bad_epochs >= patience:
            print(f"G19 early stopping at epoch {epoch}", flush=True)
            break

    best = torch.load(best_path, map_location="cpu", weights_only=True)
    gates = config.get("gates", {})
    representation_gate = (
        best["tune_recording_metrics"]["macro_f1"]
        >= float(gates.get("minimum_tune_recording_macro_f1", 0.0))
        and best["tune_recording_metrics"]["minimum_recall"]
        >= float(gates.get("minimum_tune_recording_recall", 0.0))
    )
    report = {
        "passed": representation_gate,
        "execution_completed": True,
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": (
            "proceed_to_class_conditional_open_set_calibration"
            if representation_gate
            else "stop_g19_representation_branch"
        ),
        "representation_gate_passed": representation_gate,
        "best_epoch": int(best["epoch"]),
        "best_checkpoint_sha256": file_sha256(best_path),
        "best_tune_recording_metrics": best["tune_recording_metrics"],
        "known_models": known_models,
        "trainable_modules": ["projector", "aggregator", "classifier"],
        "frozen_modules": ["G7_PANNs_Cnn14_16k"],
        "loss": {
            "formula": "recording_cross_entropy + supcon_weight * supervised_contrastive",
            "supcon_weight": supcon_weight,
            "temperature": temperature,
        },
        "feature_caches": {
            "known_train": train_cache,
            "known_tune": tune_cache,
        },
        "identity": identity,
        "known_holdout_read": False,
        "unknown_holdout_read": False,
        "locked_datasets_read": [],
    }
    _atomic_json(report, summary_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train G19 supervised-contrastive recording representation"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.preflight_only:
        result = preflight(arguments.config, arguments.root)
    else:
        result = train(arguments.config, arguments.root, resume=arguments.resume)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
