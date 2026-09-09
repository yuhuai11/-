from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .calibrate_ood import _predict_logits, probabilities_from_logits
from .config import ensure_dirs, load_config
from .data_firewall import audit_csv_rows
from .dataset import DADSDataset
from .evaluate_g9_guard import negative_metrics, predict_guard, validate_guard_manifest
from .metrics import binary_metrics
from .panns import file_sha256
from .prepare_beats_probe import reject_locked_path
from .train import resolve_device
from .train_panns import build_model


G14_MANIFEST = Path(
    "artifacts/g14_domain_generalization/segment_cache/g14_segment_manifest.csv"
)
G14_SUMMARY = Path("artifacts/g14_domain_generalization/runs/head_only/summary.json")


def _verify_file(path: Path, expected_sha256: str | None = None) -> str:
    reject_locked_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = file_sha256(path)
    if expected_sha256 and observed != str(expected_sha256).lower():
        raise ValueError(f"SHA256 mismatch for {path}: {observed}")
    return observed


def _load_prediction_csv(path: Path, manifest: pd.DataFrame, probability: str) -> np.ndarray:
    _verify_file(path)
    rows = pd.read_csv(path, low_memory=False)
    if len(rows) != len(manifest):
        raise ValueError(f"Prediction row count mismatch: {path}")
    for column in ("label", "sha256", probability):
        if column not in rows:
            raise ValueError(f"Missing {column} in {path}")
    if not np.array_equal(
        rows["label"].to_numpy(dtype=np.int64),
        manifest["label"].to_numpy(dtype=np.int64),
    ):
        raise ValueError(f"Prediction labels do not align with {path}")
    if not np.array_equal(
        rows["sha256"].astype(str).to_numpy(),
        manifest["sha256"].astype(str).to_numpy(),
    ):
        raise ValueError(f"Prediction hashes do not align with {path}")
    values = rows[probability].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite probabilities in {path}")
    return values


def _validate_candidate_checkpoint(path: Path, seed: int, expected_sha256: str) -> dict:
    _verify_file(path, expected_sha256)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if int(checkpoint.get("seed", -1)) != int(seed):
        raise ValueError(f"Candidate seed mismatch: {path}")
    config = checkpoint.get("config", {})
    model = config.get("model", {})
    if model.get("trainable_scope") != "binary_head_only":
        raise ValueError(f"Candidate is not a G14 head-only checkpoint: {path}")
    if model.get("binary_checkpoint_sha256") != (
        "d357f194105ec27a61838e0f4c2e7575932e2dd861e529ef970b1fb467954cdc"
    ):
        raise ValueError(f"Candidate G7 parent identity changed: {path}")
    inputs = checkpoint.get("training_inputs", {})
    if Path(str(inputs.get("manifest_path", ""))).resolve() != G14_MANIFEST.resolve():
        raise ValueError(f"Candidate is not bound to the G14 manifest: {path}")
    if not inputs.get("input_audits"):
        raise ValueError(f"Candidate lacks bound G14 input audits: {path}")
    return checkpoint


def _preflight(config: dict) -> dict[str, Any]:
    protocol = str(config["protocol"])
    if protocol != "g14_b_cross_domain_regression_v1":
        raise ValueError(f"Unsupported G14-B protocol: {protocol}")

    baseline = config["baseline"]
    checked = {
        "baseline_checkpoint": _verify_file(
            Path(baseline["checkpoint"]), baseline["checkpoint_sha256"]
        )
    }
    for key in (
        "dads_test_probabilities",
        "dads_test_labels",
        "val_ood_tune_predictions",
        "val_ood_holdout_predictions",
        "g9_guard_predictions",
    ):
        checked[key] = _verify_file(Path(baseline[key]))

    for name, spec in (
        ("dads_test_manifest", config["datasets"]["dads_test"]),
        ("val_ood_tune_manifest", {
            "manifest": config["datasets"]["val_ood"]["tune_manifest"],
            "manifest_sha256": config["datasets"]["val_ood"]["tune_manifest_sha256"],
        }),
        ("val_ood_holdout_manifest", {
            "manifest": config["datasets"]["val_ood"]["holdout_manifest"],
            "manifest_sha256": config["datasets"]["val_ood"]["holdout_manifest_sha256"],
        }),
        ("g9_guard_manifest", config["datasets"]["g9_guard"]),
    ):
        path = Path(spec["manifest"])
        audit_csv_rows(path)
        checked[name] = _verify_file(path, spec["manifest_sha256"])
    checked["g9_guard_audit"] = _verify_file(
        Path(config["datasets"]["g9_guard"]["audit"]),
        config["datasets"]["g9_guard"]["audit_sha256"],
    )

    candidates = []
    summary = json.loads(G14_SUMMARY.read_text(encoding="utf-8"))
    summary_by_seed = {int(item["seed"]): item for item in summary}
    for spec in config["candidates"]:
        seed = int(spec["seed"])
        checkpoint = Path(spec["checkpoint"])
        _validate_candidate_checkpoint(checkpoint, seed, str(spec["checkpoint_sha256"]))
        metrics = summary_by_seed.get(seed)
        if metrics is None or metrics.get("checkpoint_sha256") != spec["checkpoint_sha256"]:
            raise ValueError(f"G14 summary/checkpoint mismatch for seed {seed}")
        candidates.append(
            {
                "seed": seed,
                "checkpoint": checkpoint.as_posix(),
                "checkpoint_sha256": spec["checkpoint_sha256"],
            }
        )

    g14_rows = pd.read_csv(G14_MANIFEST, low_memory=False)
    label_by_dataset = {
        str(dataset): sorted(int(value) for value in values)
        for dataset, values in g14_rows.groupby("dataset")["label"].unique().items()
    }
    dataset_label_confounding = all(len(values) == 1 for values in label_by_dataset.values())
    report = {
        "passed": True,
        "protocol": protocol,
        "mode": "preflight",
        "ready_for_gpu_regression": True,
        "training_started": False,
        "model_parameters_updated": False,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "checked_inputs": checked,
        "shortcut_risk": {
            "dataset_label_confounding": dataset_label_confounding,
            "labels_by_dataset": label_by_dataset,
            "promotion_blocked_until_counterfactual_control": dataset_label_confounding,
        },
        "locked_datasets_read": [],
    }
    output = Path(config["output_dir"])
    ensure_dirs(output)
    (output / "preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _predict_dads(
    checkpoint_path: Path,
    manifest_path: Path,
    split: str,
    *,
    batch_size: int,
    num_workers: int,
    device_name: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset = DADSDataset(
        manifest_path,
        split,
        sample_rate=int(config["data"]["sample_rate"]),
        clip_seconds=float(config["data"]["clip_seconds"]),
        training=False,
        seed=int(checkpoint["seed"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    batches = []
    use_amp = device.type == "cuda" and bool(config["train"].get("mixed_precision", True))
    with torch.no_grad():
        for waveform, _ in tqdm(loader, desc=f"G14 seed {checkpoint['seed']} DADS/{split}"):
            waveform = waveform.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                batches.append(torch.sigmoid(model(waveform)).cpu().numpy())
    return dataset.rows.copy(), np.concatenate(batches).astype(np.float32, copy=False)


def _metric_delta(candidate: dict, baseline: dict) -> dict[str, float]:
    return {
        "f1": float(candidate["f1"] - baseline["f1"]),
        "auc": float(candidate["auc"] - baseline["auc"]),
        "recall": float(candidate["recall"] - baseline["recall"]),
        "specificity": float(candidate["specificity"] - baseline["specificity"]),
    }


def _run(config: dict) -> dict[str, Any]:
    preflight = _preflight(config)
    settings = config["evaluation"]
    threshold = float(settings["threshold"])
    batch_size = int(settings["batch_size"])
    num_workers = int(settings["num_workers"])
    device_name = str(settings["device"])
    output = Path(config["output_dir"])

    dads_spec = config["datasets"]["dads_test"]
    dads_manifest = Path(dads_spec["manifest"])
    dads_rows = pd.read_csv(dads_manifest, low_memory=False)
    dads_rows = dads_rows.loc[dads_rows["split"].astype(str).eq(str(dads_spec["split"]))].reset_index(drop=True)
    baseline_dads_labels = np.load(config["baseline"]["dads_test_labels"])
    baseline_dads_probabilities = np.load(config["baseline"]["dads_test_probabilities"])
    if not np.array_equal(
        baseline_dads_labels.astype(np.int64),
        dads_rows["label"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("Stored G7 DADS labels do not align with the frozen test split")
    baseline_dads = binary_metrics(
        baseline_dads_labels.astype(np.int64), baseline_dads_probabilities, threshold
    )

    tune_manifest_path = Path(config["datasets"]["val_ood"]["tune_manifest"])
    holdout_manifest_path = Path(config["datasets"]["val_ood"]["holdout_manifest"])
    tune_manifest = pd.read_csv(tune_manifest_path, low_memory=False)
    holdout_manifest = pd.read_csv(holdout_manifest_path, low_memory=False)
    baseline_tune_probability = _load_prediction_csv(
        Path(config["baseline"]["val_ood_tune_predictions"]),
        tune_manifest,
        "raw_probability",
    )
    baseline_holdout_probability = _load_prediction_csv(
        Path(config["baseline"]["val_ood_holdout_predictions"]),
        holdout_manifest,
        "raw_probability",
    )
    baseline_tune = binary_metrics(
        tune_manifest["label"].to_numpy(dtype=np.int64),
        baseline_tune_probability,
        threshold,
    )
    baseline_holdout = binary_metrics(
        holdout_manifest["label"].to_numpy(dtype=np.int64),
        baseline_holdout_probability,
        threshold,
    )

    guard_spec = config["datasets"]["g9_guard"]
    guard_rows, _ = validate_guard_manifest(
        Path(guard_spec["manifest"]), Path(guard_spec["audit"])
    )
    guard_predictions = pd.read_csv(
        config["baseline"]["g9_guard_predictions"], low_memory=False
    )
    if len(guard_predictions) != len(guard_rows):
        raise ValueError("Stored G7 guard predictions do not align with the guard manifest")
    baseline_guard = negative_metrics(
        guard_rows,
        guard_predictions["g7_probability"].to_numpy(dtype=np.float64),
        threshold,
    )

    gates = config["gates"]
    seed_results = []
    for spec in config["candidates"]:
        seed = int(spec["seed"])
        checkpoint_path = Path(spec["checkpoint"])
        candidate_dir = output / f"seed_{seed}"
        ensure_dirs(candidate_dir)

        candidate_dads_rows, candidate_dads_probability = _predict_dads(
            checkpoint_path,
            dads_manifest,
            str(dads_spec["split"]),
            batch_size=batch_size,
            num_workers=num_workers,
            device_name=device_name,
        )
        if not np.array_equal(
            candidate_dads_rows["label"].to_numpy(dtype=np.int64),
            dads_rows["label"].to_numpy(dtype=np.int64),
        ):
            raise ValueError(f"G14 seed {seed} DADS prediction order changed")
        candidate_dads = binary_metrics(
            dads_rows["label"].to_numpy(dtype=np.int64),
            candidate_dads_probability,
            threshold,
        )

        _, tune_logits, _, tune_seed = _predict_logits(
            checkpoint_path,
            tune_manifest_path,
            batch_size=batch_size,
            num_workers=num_workers,
            device_name=device_name,
        )
        _, holdout_logits, _, holdout_seed = _predict_logits(
            checkpoint_path,
            holdout_manifest_path,
            batch_size=batch_size,
            num_workers=num_workers,
            device_name=device_name,
        )
        if tune_seed != seed or holdout_seed != seed:
            raise ValueError(f"G14 seed identity changed during val_ood inference: {seed}")
        candidate_tune_probability = probabilities_from_logits(tune_logits)
        candidate_holdout_probability = probabilities_from_logits(holdout_logits)
        candidate_tune = binary_metrics(
            tune_manifest["label"].to_numpy(dtype=np.int64),
            candidate_tune_probability,
            threshold,
        )
        candidate_holdout = binary_metrics(
            holdout_manifest["label"].to_numpy(dtype=np.int64),
            candidate_holdout_probability,
            threshold,
        )

        candidate_guard_probability, guard_checkpoint = predict_guard(
            checkpoint_path,
            Path(guard_spec["manifest"]),
            batch_size=batch_size,
            num_workers=num_workers,
            device_name=device_name,
        )
        if int(guard_checkpoint["seed"]) != seed:
            raise ValueError(f"G14 seed identity changed during guard inference: {seed}")
        candidate_guard = negative_metrics(
            guard_rows, candidate_guard_probability, threshold
        )

        np.save(candidate_dir / "dads_test_probabilities.npy", candidate_dads_probability)
        np.save(candidate_dir / "val_ood_tune_probabilities.npy", candidate_tune_probability)
        np.save(
            candidate_dir / "val_ood_holdout_probabilities.npy",
            candidate_holdout_probability,
        )
        np.save(candidate_dir / "g9_guard_probabilities.npy", candidate_guard_probability)

        dads_delta = _metric_delta(candidate_dads, baseline_dads)
        holdout_delta = _metric_delta(candidate_holdout, baseline_holdout)
        guard_fpr_delta = float(
            candidate_guard["segment_false_positive_rate"]
            - baseline_guard["segment_false_positive_rate"]
        )
        checks = [
            {
                "name": "dads_test_f1",
                "value": candidate_dads["f1"],
                "minimum": baseline_dads["f1"]
                - float(gates["dads_test"]["maximum_f1_drop"]),
                "passed": dads_delta["f1"]
                >= -float(gates["dads_test"]["maximum_f1_drop"]),
            },
            {
                "name": "dads_test_auc",
                "value": candidate_dads["auc"],
                "minimum": baseline_dads["auc"]
                - float(gates["dads_test"]["maximum_auc_drop"]),
                "passed": dads_delta["auc"]
                >= -float(gates["dads_test"]["maximum_auc_drop"]),
            },
            {
                "name": "val_ood_holdout_f1",
                "value": candidate_holdout["f1"],
                "minimum": baseline_holdout["f1"]
                - float(gates["val_ood_holdout"]["maximum_f1_drop"]),
                "passed": holdout_delta["f1"]
                >= -float(gates["val_ood_holdout"]["maximum_f1_drop"]),
            },
            {
                "name": "val_ood_holdout_auc",
                "value": candidate_holdout["auc"],
                "minimum": baseline_holdout["auc"]
                - float(gates["val_ood_holdout"]["maximum_auc_drop"]),
                "passed": holdout_delta["auc"]
                >= -float(gates["val_ood_holdout"]["maximum_auc_drop"]),
            },
            {
                "name": "g9_guard_segment_fpr",
                "value": candidate_guard["segment_false_positive_rate"],
                "maximum": baseline_guard["segment_false_positive_rate"]
                + float(gates["g9_guard"]["maximum_segment_fpr_increase"]),
                "passed": guard_fpr_delta
                <= float(gates["g9_guard"]["maximum_segment_fpr_increase"]),
            },
        ]
        seed_result = {
            "seed": seed,
            "checkpoint": checkpoint_path.as_posix(),
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "dads_test": candidate_dads,
            "dads_test_delta_from_g7": dads_delta,
            "val_ood_tune": candidate_tune,
            "val_ood_holdout": candidate_holdout,
            "val_ood_holdout_delta_from_g7": holdout_delta,
            "g9_guard": candidate_guard,
            "g9_guard_segment_fpr_delta_from_g7": guard_fpr_delta,
            "checks": checks,
            "regression_gate_passed": all(item["passed"] for item in checks),
        }
        (candidate_dir / "metrics.json").write_text(
            json.dumps(seed_result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        seed_results.append(seed_result)

    every_seed_passed = all(item["regression_gate_passed"] for item in seed_results)
    report = {
        "passed": True,
        "protocol": config["protocol"],
        "regression_gate_passed": every_seed_passed,
        "decision": (
            "continue_controlled_development_pending_counterfactual"
            if every_seed_passed
            else "reject_g14_head_only_due_to_regression"
        ),
        "baseline": {
            "checkpoint": config["baseline"]["checkpoint"],
            "checkpoint_sha256": config["baseline"]["checkpoint_sha256"],
            "dads_test": baseline_dads,
            "val_ood_tune": baseline_tune,
            "val_ood_holdout": baseline_holdout,
            "g9_guard": baseline_guard,
        },
        "seeds": seed_results,
        "shortcut_risk": preflight["shortcut_risk"],
        "training_started": False,
        "model_parameters_updated": False,
        "locked_datasets_read": [],
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the G14-B frozen regression audit")
    parser.add_argument("--config", type=Path, default=Path("configs/g14_b_regression.yaml"))
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    args = parser.parse_args()
    config = load_config(args.config)
    result = _preflight(config) if args.mode == "preflight" else _run(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
