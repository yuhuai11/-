from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import load_config
from .data_firewall import file_sha256, reject_locked_path
from .g22_harmonic_features import (
    aggregate_recording_features,
    normalize_class_scores,
)
from .probe_g22_harmonic_fusion import (
    _g18_segment_logits,
    _metrics_with_confusion,
    _verify_inputs,
)
from .train_g18_model_id import aggregate_recording_logits


PROTOCOL = "g22_p1_known_harmonic_fusion_multiseed_v1"
SCHEMA_VERSION = 1


def _resolve(root: Path, raw: object, *, context: str) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = root / path
    reject_locked_path(path, context=context)
    return path.resolve(strict=True)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def run(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"Expected {PROTOCOL}")
    if float(config["fusion"]["harmonic_weight"]) != 0.25:
        raise ValueError("G22 multiseed must preserve the probe's primary weight")
    serialized = json.dumps(config, ensure_ascii=False).lower()
    if any(
        value in serialized
        for value in ("unknown_tune", "known_holdout", "unknown_holdout", "x6d", "y6")
    ):
        raise ValueError("G22 multiseed must not bind Unknown or Holdout inputs")

    paths = {}
    observed = {}
    for name in (
        "probe_config",
        "probe_summary",
        "known_train_harmonic",
        "known_tune_harmonic",
    ):
        paths[name] = _resolve(root, config["inputs"][name]["path"], context=f"G22 {name}")
        observed[name] = file_sha256(paths[name])
        if observed[name] != str(config["inputs"][name]["sha256"]):
            raise ValueError(f"G22 {name} SHA256 mismatch")
    for seed, spec in config["inputs"]["checkpoints"].items():
        name = f"checkpoint_{seed}"
        paths[name] = _resolve(root, spec["path"], context=f"G22 {name}")
        observed[name] = file_sha256(paths[name])
        if observed[name] != str(spec["sha256"]):
            raise ValueError(f"G22 checkpoint {seed} SHA256 mismatch")

    probe = json.loads(paths["probe_summary"].read_text(encoding="utf-8"))
    if (
        probe.get("passed") is not True
        or probe.get("decision") != "proceed_to_g22_formal_ablation"
        or float(probe.get("primary_fusion_weight")) != 0.25
        or probe.get("known_holdout_read") is not False
        or probe.get("unknown_inputs_read") is not False
    ):
        raise ValueError("G22 probe does not authorize multiseed ablation")
    probe_config = load_config(paths["probe_config"])
    base_paths, _, registry = _verify_inputs(probe_config, root)
    frames = {
        split: pd.read_csv(base_paths[split]).reset_index(drop=True)
        for split in ("known_train", "known_tune")
    }
    harmonic_segments = {
        "known_train": np.load(paths["known_train_harmonic"], allow_pickle=False),
        "known_tune": np.load(paths["known_tune_harmonic"], allow_pickle=False),
    }
    recording_features = {}
    targets = {}
    for split in ("known_train", "known_tune"):
        recording_features[split], targets[split], _ = aggregate_recording_features(
            frames[split], harmonic_segments[split]
        )
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(probe_config["classifier"]["c"]),
            class_weight=str(probe_config["classifier"]["class_weight"]),
            max_iter=int(probe_config["classifier"]["maximum_iterations"]),
            random_state=int(probe_config["classifier"]["seed"]),
        ),
    )
    classifier.fit(recording_features["known_train"], targets["known_train"])
    harmonic_scores = classifier.decision_function(recording_features["known_tune"])
    classes = len(registry["known_models"])
    weight = float(config["fusion"]["harmonic_weight"])
    by_seed = {}
    prediction_rows = []
    for seed in sorted(config["inputs"]["checkpoints"], key=int):
        segment_logits = _g18_segment_logits(
            np.load(
                base_paths["known_tune_g7_feature"],
                mmap_mode="r",
                allow_pickle=False,
            ),
            paths[f"checkpoint_{seed}"],
        )
        baseline_targets, baseline_scores = aggregate_recording_logits(
            frames["known_tune"], segment_logits
        )
        if not np.array_equal(baseline_targets, targets["known_tune"]):
            raise ValueError(f"G22 seed {seed} recording order changed")
        fused_scores = normalize_class_scores(baseline_scores) + weight * (
            normalize_class_scores(harmonic_scores)
        )
        baseline_metrics = _metrics_with_confusion(
            baseline_targets, baseline_scores, classes
        )
        fusion_metrics = _metrics_with_confusion(
            baseline_targets, fused_scores, classes
        )
        by_seed[seed] = {
            "baseline": baseline_metrics,
            "fusion": fusion_metrics,
            "delta": {
                key: fusion_metrics[key] - baseline_metrics[key]
                for key in ("accuracy", "macro_f1", "minimum_recall")
            },
        }
        for index, target in enumerate(baseline_targets):
            prediction_rows.append(
                {
                    "seed": int(seed),
                    "audio_sha256": sorted(
                        frames["known_tune"]["audio_sha256"].astype(str).unique()
                    )[index],
                    "target_index": int(target),
                    "baseline_prediction": int(baseline_scores[index].argmax()),
                    "fusion_prediction": int(fused_scores[index].argmax()),
                }
            )

    aggregate = {}
    for method in ("baseline", "fusion"):
        aggregate[method] = {
            metric: _aggregate(
                [by_seed[seed][method][metric] for seed in by_seed]
            )
            for metric in ("accuracy", "macro_f1", "minimum_recall")
        }
    checks = {
        "mean_accuracy_noninferior": (
            aggregate["fusion"]["accuracy"]["mean"]
            >= aggregate["baseline"]["accuracy"]["mean"]
        ),
        "mean_macro_f1_strictly_improved": (
            aggregate["fusion"]["macro_f1"]["mean"]
            > aggregate["baseline"]["macro_f1"]["mean"] + 1.0e-8
        ),
        "every_seed_minimum_recall_noninferior": all(
            by_seed[seed]["fusion"]["minimum_recall"]
            >= by_seed[seed]["baseline"]["minimum_recall"]
            for seed in by_seed
        ),
    }
    passed = all(checks.values())
    output_dir = root / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "known_tune_predictions.csv", index=False
    )
    report = {
        "passed": passed,
        "execution_completed": True,
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": (
            "retain_g22_as_development_champion_candidate"
            if passed
            else "stop_g22_and_retain_g18"
        ),
        "harmonic_weight": weight,
        "metrics_by_seed": by_seed,
        "aggregate_metrics": aggregate,
        "gate_checks": checks,
        "known_models": registry["known_models"],
        "input_sha256": observed,
        "source_sha256": {
            "evaluator": file_sha256(Path(__file__)),
            "extractor": file_sha256(Path(__file__).parent / "g22_harmonic_features.py"),
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
    parser = argparse.ArgumentParser(description="Evaluate fixed G22 fusion over G18 seeds.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g22_harmonic_fusion_multiseed.yaml"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    run(args.config, args.root)


if __name__ == "__main__":
    main()
