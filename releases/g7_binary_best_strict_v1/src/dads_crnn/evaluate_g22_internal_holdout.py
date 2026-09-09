from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import load_config
from .data_firewall import audit_csv_rows, file_sha256
from .evaluate_g18_final_holdout import _infer_base
from .g22_harmonic_features import (
    aggregate_recording_features,
    normalize_class_scores,
)
from .panns import PannsCnn14Binary
from .probe_g22_harmonic_fusion import (
    _atomic_json,
    _extract_split,
    _g18_segment_logits,
)
from .train import resolve_device, set_seed
from .train_g18_model_id import RegistryDataset, aggregate_recording_logits


PROTOCOL = "g22_p2_internal_known_holdout_report_v1"
SEEDS = (42, 43, 44)


def _resolve(root: Path, raw: object) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=True)


def _verify(path: Path, expected: object, name: str) -> str:
    observed = file_sha256(path)
    if observed != str(expected):
        raise ValueError(
            f"G22 internal Holdout SHA256 mismatch for {name}: "
            f"expected={expected}, observed={observed}"
        )
    return observed


def classification_report(
    targets: np.ndarray,
    scores: np.ndarray,
    known_models: list[str],
) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    classes = len(known_models)
    if scores.shape != (len(targets), classes) or not np.isfinite(scores).all():
        raise ValueError("Invalid G22 Holdout class scores")
    predictions = scores.argmax(axis=1)
    labels = list(range(classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": {
            model: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, model in enumerate(known_models)
        },
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=labels
        ).astype(int).tolist(),
    }


def _aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
    }


def aggregate_reports(
    reports: dict[int, dict[str, Any]], known_models: list[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        metric: _aggregate([reports[seed][metric] for seed in SEEDS])
        for metric in ("accuracy", "macro_f1")
    }
    result["per_class"] = {
        model: {
            metric: _aggregate(
                [reports[seed]["per_class"][model][metric] for seed in SEEDS]
            )
            for metric in ("precision", "recall", "f1")
        }
        for model in known_models
    }
    return result


def _load_or_extract_harmonic(
    frame: pd.DataFrame,
    manifest_hash: str,
    config: dict[str, Any],
    output_dir: Path,
) -> np.ndarray:
    path = output_dir / "known_holdout_segment_harmonic.npy"
    metadata_path = output_dir / "known_holdout_segment_harmonic.json"
    expected = {
        "protocol": PROTOCOL,
        "manifest_sha256": manifest_hash,
        "rows": len(frame),
        "dimension": 61,
        "extractor_sha256": file_sha256(
            Path(__file__).parent / "g22_harmonic_features.py"
        ),
    }
    if path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected.items()):
            return np.load(path, allow_pickle=False)
    values = _extract_split(
        frame,
        sample_rate=int(config["data"]["sample_rate"]),
        n_fft=int(config["features"]["n_fft"]),
        bands=int(config["features"]["logarithmic_bands"]),
        harmonics=int(config["features"]["harmonics"]),
    )
    np.save(path, values, allow_pickle=False)
    _atomic_json(
        {
            **expected,
            "feature_sha256": file_sha256(path),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        metadata_path,
    )
    return values


def _load_or_extract_g7(
    frame: pd.DataFrame,
    manifest_hash: str,
    config: dict[str, Any],
    paths: dict[str, Path],
    output_dir: Path,
) -> np.ndarray:
    path = output_dir / "known_holdout_g7_features.npy"
    metadata_path = output_dir / "known_holdout_g7_features.json"
    expected = {
        "protocol": PROTOCOL,
        "manifest_sha256": manifest_hash,
        "rows": len(frame),
        "dimension": int(config["model"]["embedding_dim"]),
        "binary_checkpoint_sha256": str(
            config["model"]["binary_checkpoint_sha256"]
        ),
    }
    if path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected.items()):
            return np.load(path, mmap_mode="r", allow_pickle=False)

    set_seed(42)
    device = resolve_device(str(config["runtime"]["device"]))
    detector = PannsCnn14Binary(
        initialization=str(config["model"]["initialization"]),
        vendor_dir=str(config["model"]["vendor_dir"]),
        checkpoint_path=paths["official_checkpoint"].as_posix(),
        checkpoint_sha256=str(config["model"]["checkpoint_sha256"]),
        spec_augment=False,
        frontend_precision="float32",
        binary_checkpoint_path=paths["binary_checkpoint"].as_posix(),
        binary_checkpoint_sha256=str(config["model"]["binary_checkpoint_sha256"]),
        trainable_scope="binary_head_only",
    ).to(device)
    dataset = RegistryDataset(paths["known_holdout"], int(config["data"]["sample_rate"]))
    features, _ = _infer_base(
        detector,
        dataset,
        batch_size=int(config["runtime"]["batch_size"]),
        device=device,
        label="G22 known_holdout",
    )
    del detector
    np.save(path, features.astype(np.float32), allow_pickle=False)
    _atomic_json(
        {
            **expected,
            "device": str(device),
            "feature_sha256": file_sha256(path),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        metadata_path,
    )
    return np.load(path, mmap_mode="r", allow_pickle=False)


def run(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"Expected {PROTOCOL}")
    if float(config["fusion"]["harmonic_weight"]) != 0.25:
        raise ValueError("G22 Holdout evaluation must keep harmonic weight 0.25")
    if list(map(int, config["reporting"]["seeds"])) != list(SEEDS):
        raise ValueError("G22 Holdout evaluation requires seeds 42, 43 and 44")

    paths = {
        name: _resolve(root, spec["path"])
        for name, spec in config["inputs"].items()
        if name != "checkpoints"
    }
    hashes = {
        name: _verify(paths[name], config["inputs"][name]["sha256"], name)
        for name in paths
    }
    for seed in SEEDS:
        name = f"checkpoint_{seed}"
        spec = config["inputs"]["checkpoints"][str(seed)]
        paths[name] = _resolve(root, spec["path"])
        hashes[name] = _verify(paths[name], spec["sha256"], name)

    development_summary = json.loads(
        paths["development_summary"].read_text(encoding="utf-8")
    )
    if (
        development_summary.get("passed") is not True
        or development_summary.get("decision")
        != "retain_g22_as_development_champion_candidate"
        or float(development_summary.get("harmonic_weight", -1)) != 0.25
        or development_summary.get("known_holdout_read") is not False
    ):
        raise ValueError("G22 development result does not authorize fixed Holdout testing")

    registry = json.loads(paths["registry_audit"].read_text(encoding="utf-8"))
    known_models = list(registry["known_models"])
    for split in ("known_train", "known_holdout"):
        expected_rows = int(config["inputs"][split]["rows"])
        if audit_csv_rows(
            paths[split],
            required_columns=(
                "audio_sha256",
                "target_index",
                "model_id",
                "cache_path",
                "cache_index",
            ),
        ) != expected_rows:
            raise ValueError(f"G22 {split} row count changed")
    frames = {
        split: pd.read_csv(paths[split]).reset_index(drop=True)
        for split in ("known_train", "known_holdout")
    }
    if set(frames["known_train"]["audio_sha256"]) & set(
        frames["known_holdout"]["audio_sha256"]
    ):
        raise ValueError("G22 detected train/Holdout recording leakage")

    output_dir = root / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_harmonic = np.load(paths["known_train_harmonic"], allow_pickle=False)
    holdout_harmonic = _load_or_extract_harmonic(
        frames["known_holdout"], hashes["known_holdout"], config, output_dir
    )
    train_features, train_targets, _ = aggregate_recording_features(
        frames["known_train"], train_harmonic
    )
    holdout_features, targets, identities = aggregate_recording_features(
        frames["known_holdout"], holdout_harmonic
    )
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(config["classifier"]["c"]),
            class_weight="balanced",
            max_iter=int(config["classifier"]["maximum_iterations"]),
            random_state=int(config["classifier"]["seed"]),
        ),
    )
    classifier.fit(train_features, train_targets)
    harmonic_scores = classifier.decision_function(holdout_features)

    g7_features = _load_or_extract_g7(
        frames["known_holdout"],
        hashes["known_holdout"],
        config,
        paths,
        output_dir,
    )
    weight = float(config["fusion"]["harmonic_weight"])
    reports: dict[int, dict[str, Any]] = {}
    fused_by_seed: dict[int, np.ndarray] = {}
    rows = []
    for seed in SEEDS:
        segment_logits = _g18_segment_logits(g7_features, paths[f"checkpoint_{seed}"])
        baseline_targets, baseline_scores = aggregate_recording_logits(
            frames["known_holdout"], segment_logits
        )
        if not np.array_equal(targets, baseline_targets):
            raise ValueError(f"G22 seed {seed} Holdout recording order changed")
        fused = normalize_class_scores(baseline_scores) + weight * (
            normalize_class_scores(harmonic_scores)
        )
        fused_by_seed[seed] = fused
        reports[seed] = classification_report(targets, fused, known_models)
        predictions = fused.argmax(axis=1)
        for index, identity in enumerate(identities):
            rows.append(
                {
                    "seed": seed,
                    "audio_sha256": identity,
                    "target_index": int(targets[index]),
                    "target_model": known_models[int(targets[index])],
                    "predicted_index": int(predictions[index]),
                    "predicted_model": known_models[int(predictions[index])],
                }
            )

    ensemble_scores = np.mean([fused_by_seed[seed] for seed in SEEDS], axis=0)
    ensemble_report = classification_report(targets, ensemble_scores, known_models)
    pd.DataFrame(rows).to_csv(
        output_dir / "known_holdout_predictions_by_seed.csv", index=False
    )
    report = {
        "passed": True,
        "execution_completed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "internal_known_holdout_recording_level",
        "holdout_recordings": len(identities),
        "harmonic_weight": weight,
        "metrics_by_seed": {str(seed): reports[seed] for seed in SEEDS},
        "aggregate_metrics": aggregate_reports(reports, known_models),
        "mean_score_ensemble_supplement": ensemble_report,
        "known_models": known_models,
        "input_sha256": hashes,
        "optimization_inputs": ["known_train"],
        "model_selection_inputs": ["known_tune"],
        "test_inputs": ["known_holdout"],
        "limitations": [
            "known_holdout_was_previously_consumed_by_g18_p6",
            "not_a_new_independent_confirmation_set",
            "single_recorder_and_mostly_single_session_per_model",
        ],
    }
    _atomic_json(report, output_dir / "summary.json")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed G22 on the previously consumed internal Known Holdout."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g22_internal_holdout.yaml"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    run(args.config, args.root)


if __name__ == "__main__":
    main()
