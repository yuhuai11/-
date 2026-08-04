from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import load_config
from .data_firewall import audit_csv_rows, file_sha256, reject_locked_path
from .g22_harmonic_features import (
    aggregate_recording_features,
    extract_segment_harmonic_features,
    normalize_class_scores,
)
from .train_g18_model_id import aggregate_recording_logits, classification_metrics


PROTOCOL = "g22_p0_known_harmonic_fusion_probe_v1"
SCHEMA_VERSION = 1


def _resolve(root: Path, raw: object, *, context: str) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = root / path
    reject_locked_path(path, context=context)
    return path.resolve(strict=True)


def _verify_config(config: dict[str, Any]) -> None:
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"G22 requires protocol {PROTOCOL}")
    if config.get("data", {}).get("split_unit") != "audio_sha256":
        raise ValueError("G22 requires recording-identity isolation")
    if config.get("features", {}).get("recording_aggregation") != "mean_std":
        raise ValueError("G22 v1 requires fixed mean/std aggregation")
    if int(config["features"]["segment_dimension"]) != 61:
        raise ValueError("G22 v1 segment feature definition changed")
    if float(config["fusion"]["primary_harmonic_weight"]) != 0.25:
        raise ValueError("G22 primary fusion weight must remain pre-registered")
    serialized = json.dumps(config, ensure_ascii=False).lower()
    forbidden = ("unknown_tune", "known_holdout", "unknown_holdout", "x6d", "y6")
    if any(value in serialized for value in forbidden):
        raise ValueError("G22 must not bind Unknown or Holdout inputs")


def _verify_inputs(
    config: dict[str, Any], root: Path
) -> tuple[dict[str, Path], dict[str, str], dict[str, Any]]:
    _verify_config(config)
    paths: dict[str, Path] = {}
    expected: dict[str, str] = {}
    for split in ("known_train", "known_tune"):
        spec = config["inputs"][split]
        paths[split] = _resolve(root, spec["path"], context=f"G22 {split}")
        paths[f"{split}_g7_feature"] = _resolve(
            root, spec["g7_feature_path"], context=f"G22 {split} G7 features"
        )
        expected[split] = str(spec["sha256"])
        expected[f"{split}_g7_feature"] = str(spec["g7_feature_sha256"])
    for name in ("registry_audit", "g18_checkpoint"):
        paths[name] = _resolve(
            root, config["inputs"][name]["path"], context=f"G22 {name}"
        )
        expected[name] = str(config["inputs"][name]["sha256"])
    observed = {name: file_sha256(path) for name, path in paths.items()}
    mismatches = {
        name: {"expected": expected[name], "observed": observed[name]}
        for name in expected
        if expected[name] != observed[name]
    }
    if mismatches:
        raise ValueError(f"G22 input identity mismatch: {mismatches}")
    registry = json.loads(paths["registry_audit"].read_text(encoding="utf-8"))
    if registry.get("passed") is not True or registry.get("locked_datasets_read") != []:
        raise ValueError("G22 registry audit is invalid")
    for split in ("known_train", "known_tune"):
        rows = audit_csv_rows(
            paths[split],
            required_columns=(
                "audio_sha256",
                "target_index",
                "model_id",
                "cache_path",
                "cache_index",
            ),
        )
        if rows != int(config["inputs"][split]["rows"]):
            raise ValueError(f"G22 {split} row count changed")
        if registry["outputs"][split]["sha256"] != observed[split]:
            raise ValueError(f"G22 {split} is not registry-bound")
    return paths, observed, registry


def _extract_split(
    frame: pd.DataFrame,
    *,
    sample_rate: int,
    n_fft: int,
    bands: int,
    harmonics: int,
) -> np.ndarray:
    result = np.empty((len(frame), 61), dtype=np.float32)
    maps: OrderedDict[str, np.ndarray] = OrderedDict()
    for position, row in frame.iterrows():
        cache_path = str(row["cache_path"])
        reject_locked_path(Path(cache_path), context="G22 segment cache")
        if cache_path not in maps:
            maps[cache_path] = np.load(cache_path, mmap_mode="r")
        waveform = np.asarray(
            maps[cache_path][int(row["cache_index"])], dtype=np.float32
        ).reshape(-1)
        result[int(position)] = extract_segment_harmonic_features(
            waveform,
            sample_rate=sample_rate,
            n_fft=n_fft,
            bands=bands,
            harmonics=harmonics,
        )
        if (int(position) + 1) % 500 == 0:
            print(f"G22 extracted {int(position) + 1}/{len(frame)} segments", flush=True)
    return result


def _g18_segment_logits(features: np.ndarray, checkpoint: Path) -> np.ndarray:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = saved["head_state"]
    # Memmapped G7 caches are read-only; copy before exposing them to PyTorch.
    values = torch.from_numpy(np.array(features, dtype=np.float32, copy=True))
    weight = state["embedding_norm.weight"].float()
    bias = state["embedding_norm.bias"].float()
    normalized = torch.nn.functional.layer_norm(
        values, (values.shape[1],), weight=weight, bias=bias
    )
    logits = torch.nn.functional.linear(
        normalized,
        state["classifier.weight"].float(),
        state["classifier.bias"].float(),
    )
    return logits.numpy()


def _metrics_with_confusion(
    targets: np.ndarray, scores: np.ndarray, classes: int
) -> dict[str, Any]:
    metrics = classification_metrics(targets, scores, classes)
    metrics["confusion_matrix"] = confusion_matrix(
        targets, scores.argmax(axis=1), labels=list(range(classes))
    ).astype(int).tolist()
    return metrics


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def run(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    paths, observed, registry = _verify_inputs(config, root)
    output_dir = root / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        split: pd.read_csv(paths[split]).reset_index(drop=True)
        for split in ("known_train", "known_tune")
    }
    recording_sets = {
        split: set(frame["audio_sha256"].astype(str))
        for split, frame in frames.items()
    }
    if recording_sets["known_train"] & recording_sets["known_tune"]:
        raise ValueError("G22 detected train/tune recording leakage")

    harmonic_segments = {}
    for split in ("known_train", "known_tune"):
        cache_path = output_dir / f"{split}_segment_harmonic.npy"
        metadata_path = output_dir / f"{split}_segment_harmonic.json"
        expected_metadata = {
            "protocol": PROTOCOL,
            "manifest_sha256": observed[split],
            "rows": len(frames[split]),
            "dimension": int(config["features"]["segment_dimension"]),
            "extractor_sha256": file_sha256(Path(__file__).parent / "g22_harmonic_features.py"),
        }
        if cache_path.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if all(metadata.get(key) == value for key, value in expected_metadata.items()):
                harmonic_segments[split] = np.load(cache_path, allow_pickle=False)
                continue
        harmonic_segments[split] = _extract_split(
            frames[split],
            sample_rate=int(config["data"]["sample_rate"]),
            n_fft=int(config["features"]["n_fft"]),
            bands=int(config["features"]["logarithmic_bands"]),
            harmonics=int(config["features"]["harmonics"]),
        )
        np.save(cache_path, harmonic_segments[split], allow_pickle=False)
        _atomic_json(
            {
                **expected_metadata,
                "feature_sha256": file_sha256(cache_path),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            metadata_path,
        )

    recording_features = {}
    targets = {}
    identities = {}
    g18_recording_logits = {}
    classes = len(registry["known_models"])
    for split in ("known_train", "known_tune"):
        (
            recording_features[split],
            targets[split],
            identities[split],
        ) = aggregate_recording_features(frames[split], harmonic_segments[split])
        g7_features = np.load(
            paths[f"{split}_g7_feature"], mmap_mode="r", allow_pickle=False
        )
        segment_logits = _g18_segment_logits(g7_features, paths["g18_checkpoint"])
        baseline_targets, g18_recording_logits[split] = aggregate_recording_logits(
            frames[split], segment_logits
        )
        if not np.array_equal(targets[split], baseline_targets):
            raise ValueError("G22 harmonic and G18 recording orders differ")

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(config["classifier"]["c"]),
            class_weight=str(config["classifier"]["class_weight"]),
            max_iter=int(config["classifier"]["maximum_iterations"]),
            random_state=int(config["classifier"]["seed"]),
        ),
    )
    classifier.fit(recording_features["known_train"], targets["known_train"])
    harmonic_scores = {
        split: classifier.decision_function(recording_features[split])
        for split in ("known_train", "known_tune")
    }
    baseline_metrics = _metrics_with_confusion(
        targets["known_tune"], g18_recording_logits["known_tune"], classes
    )
    harmonic_metrics = _metrics_with_confusion(
        targets["known_tune"], harmonic_scores["known_tune"], classes
    )
    diagnostic = {}
    for weight in config["fusion"]["diagnostic_harmonic_weights"]:
        value = float(weight)
        scores = normalize_class_scores(g18_recording_logits["known_tune"]) + value * (
            normalize_class_scores(harmonic_scores["known_tune"])
        )
        diagnostic[str(value)] = _metrics_with_confusion(
            targets["known_tune"], scores, classes
        )
    primary_weight = float(config["fusion"]["primary_harmonic_weight"])
    primary = diagnostic[str(primary_weight)]
    baseline = config["baseline"]
    noninferior_accuracy = primary["accuracy"] >= float(
        baseline["tune_recording_accuracy"]
    )
    improved = (
        primary["macro_f1"] > float(baseline["tune_recording_macro_f1"]) + 1.0e-8
        or primary["minimum_recall"]
        > float(baseline["tune_recording_minimum_recall"]) + 1.0e-8
    )
    passed = bool(noninferior_accuracy and improved)

    predictions = pd.DataFrame(
        {
            "audio_sha256": identities["known_tune"],
            "target_index": targets["known_tune"],
            "g18_prediction": g18_recording_logits["known_tune"].argmax(axis=1),
            "harmonic_prediction": harmonic_scores["known_tune"].argmax(axis=1),
            "primary_fusion_prediction": (
                normalize_class_scores(g18_recording_logits["known_tune"])
                + primary_weight * normalize_class_scores(harmonic_scores["known_tune"])
            ).argmax(axis=1),
        }
    )
    predictions.to_csv(output_dir / "known_tune_predictions.csv", index=False)
    report = {
        "passed": passed,
        "execution_completed": True,
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": (
            "proceed_to_g22_formal_ablation"
            if passed
            else "stop_harmonic_fusion_and_retain_g18"
        ),
        "baseline_metrics_recomputed": baseline_metrics,
        "harmonic_only_metrics": harmonic_metrics,
        "primary_fusion_weight": primary_weight,
        "primary_fusion_metrics": primary,
        "diagnostic_weight_curve": diagnostic,
        "gate": {
            "accuracy_noninferior": noninferior_accuracy,
            "macro_f1_or_minimum_recall_strictly_improved": improved,
        },
        "known_models": registry["known_models"],
        "feature_dimensions": {
            "segment": harmonic_segments["known_train"].shape[1],
            "recording": recording_features["known_train"].shape[1],
        },
        "input_sha256": observed,
        "source_sha256": {
            "extractor": file_sha256(Path(__file__).parent / "g22_harmonic_features.py"),
            "probe": file_sha256(Path(__file__)),
        },
        "optimization_inputs": ["known_train"],
        "model_selection_inputs": ["known_tune"],
        "known_holdout_read": False,
        "unknown_inputs_read": False,
        "locked_datasets_read": [],
    }
    _atomic_json(report, output_dir / "summary.json")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the G22 harmonic fusion probe.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g22_harmonic_fusion_probe.yaml"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    run(args.config, args.root)


if __name__ == "__main__":
    main()
