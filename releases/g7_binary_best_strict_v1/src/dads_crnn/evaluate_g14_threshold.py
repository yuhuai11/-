from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_dirs, load_config
from .evaluate_g14_regression import _predict_dads, _verify_file
from .evaluate_g9_guard import negative_metrics, validate_guard_manifest
from .metrics import binary_metrics
from .panns import file_sha256


def _pattern(config: dict, key: str, seed: int) -> Path:
    return Path(str(config["candidates"][key]).format(seed=seed))


def _preflight(config: dict) -> dict[str, Any]:
    if config["protocol"] != "g14_c_tune_only_shared_threshold_v1":
        raise ValueError("Unsupported G14-C protocol")
    seeds = [int(value) for value in config["seeds"]]
    checked = {}
    tune_only_paths = [
        Path(config["baseline"]["dads_val_probabilities"]),
        Path(config["baseline"]["dads_val_labels"]),
        Path(config["baseline"]["val_ood_tune_predictions"]),
        Path(config["manifests"]["dads"]),
        Path(config["manifests"]["val_ood_tune"]),
    ]
    for seed in seeds:
        tune_only_paths.extend(
            [
                _pattern(config, "checkpoint_pattern", seed),
                _pattern(config, "g14_tune_probability_pattern", seed),
                _pattern(config, "g14_tune_label_pattern", seed),
                _pattern(config, "val_ood_tune_probability_pattern", seed),
            ]
        )
    for path in tune_only_paths:
        checked[path.as_posix()] = _verify_file(path)
    selection = config["selection"]
    if selection["rule"] != "lowest_shared_threshold_passing_every_seed":
        raise ValueError("G14-C threshold selection rule is not locked")
    if float(selection["threshold_step"]) <= 0:
        raise ValueError("Threshold step must be positive")
    report = {
        "passed": True,
        "protocol": config["protocol"],
        "mode": "preflight",
        "ready_for_fit": True,
        "selection_inputs_are_tune_only": True,
        "selection_forbidden_inputs": [
            "DADS test",
            "val_ood holdout",
            "G9 guard",
            "G13 external confirmation",
        ],
        "holdouts_previously_observed_in_g14_b": True,
        "therefore_not_final_evidence": True,
        "checked_tune_inputs": checked,
        "training_started": False,
        "model_parameters_updated": False,
        "locked_datasets_read": [],
    }
    output = Path(config["output_dir"])
    ensure_dirs(output)
    (output / "preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    return binary_metrics(
        np.asarray(labels, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
        float(threshold),
    )


def _fit(config: dict) -> dict[str, Any]:
    _preflight(config)
    seeds = [int(value) for value in config["seeds"]]
    runtime = config["runtime"]
    output = Path(config["output_dir"])
    dads_manifest = Path(config["manifests"]["dads"])
    dads_frame = pd.read_csv(dads_manifest, low_memory=False)
    dads_val = dads_frame.loc[dads_frame["split"].astype(str).eq("val")].reset_index(drop=True)
    baseline_dads_labels = np.load(config["baseline"]["dads_val_labels"])
    baseline_dads_probability = np.load(config["baseline"]["dads_val_probabilities"])
    if not np.array_equal(
        baseline_dads_labels.astype(np.int64),
        dads_val["label"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("G7 DADS val labels do not align with the frozen manifest")
    baseline_dads = _metrics(baseline_dads_labels, baseline_dads_probability, 0.5)

    val_ood_frame = pd.read_csv(config["baseline"]["val_ood_tune_predictions"], low_memory=False)
    val_ood_labels = val_ood_frame["label"].to_numpy(dtype=np.int64)
    baseline_val_ood = _metrics(
        val_ood_labels,
        val_ood_frame["raw_probability"].to_numpy(dtype=np.float64),
        0.5,
    )

    tune_data = {}
    for seed in seeds:
        checkpoint = _pattern(config, "checkpoint_pattern", seed)
        rows, dads_probability = _predict_dads(
            checkpoint,
            dads_manifest,
            "val",
            batch_size=int(runtime["batch_size"]),
            num_workers=int(runtime["num_workers"]),
            device_name=str(runtime["device"]),
        )
        if not np.array_equal(
            rows["label"].to_numpy(dtype=np.int64),
            dads_val["label"].to_numpy(dtype=np.int64),
        ):
            raise ValueError(f"G14 seed {seed} DADS val order changed")
        np.save(output / f"seed_{seed}_dads_val_probabilities.npy", dads_probability)
        g14_labels = np.load(_pattern(config, "g14_tune_label_pattern", seed))
        g14_probability = np.load(_pattern(config, "g14_tune_probability_pattern", seed))
        val_ood_probability = np.load(
            _pattern(config, "val_ood_tune_probability_pattern", seed)
        )
        if len(val_ood_probability) != len(val_ood_labels):
            raise ValueError(f"G14 seed {seed} val_ood tune length changed")
        tune_data[seed] = {
            "dads_labels": rows["label"].to_numpy(dtype=np.int64),
            "dads_probability": dads_probability,
            "val_ood_labels": val_ood_labels,
            "val_ood_probability": val_ood_probability,
            "g14_labels": g14_labels,
            "g14_probability": g14_probability,
        }

    selection = config["selection"]
    thresholds = np.arange(
        float(selection["threshold_minimum"]),
        float(selection["threshold_maximum"]) + float(selection["threshold_step"]) / 2,
        float(selection["threshold_step"]),
    )
    search = []
    selected = None
    for threshold in thresholds:
        seed_rows = []
        for seed in seeds:
            data = tune_data[seed]
            dads = _metrics(data["dads_labels"], data["dads_probability"], threshold)
            val_ood = _metrics(
                data["val_ood_labels"], data["val_ood_probability"], threshold
            )
            g14 = _metrics(data["g14_labels"], data["g14_probability"], threshold)
            checks = {
                "dads_f1": dads["f1"]
                >= baseline_dads["f1"]
                - float(selection["dads_val_maximum_f1_drop"]),
                "dads_specificity": dads["specificity"]
                >= baseline_dads["specificity"]
                - float(selection["dads_val_maximum_specificity_drop"]),
                "val_ood_f1": val_ood["f1"]
                >= baseline_val_ood["f1"]
                + float(selection["val_ood_tune_minimum_f1_gain"]),
                "g14_f1": g14["f1"] >= float(selection["g14_tune_minimum_f1"]),
            }
            seed_rows.append(
                {
                    "seed": seed,
                    "dads_val": dads,
                    "val_ood_tune": val_ood,
                    "g14_tune": g14,
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
        item = {
            "threshold": float(round(threshold, 10)),
            "seeds": seed_rows,
            "passed_every_seed": all(row["passed"] for row in seed_rows),
        }
        search.append(item)
        if selected is None and item["passed_every_seed"]:
            selected = item

    calibration = {
        "passed": selected is not None,
        "protocol": config["protocol"],
        "mode": "fit",
        "selection_rule": selection["rule"],
        "selected_threshold": selected["threshold"] if selected else None,
        "selected_tune_metrics": selected["seeds"] if selected else [],
        "baseline_tune_metrics": {
            "dads_val_at_0_5": baseline_dads,
            "val_ood_tune_at_0_5": baseline_val_ood,
        },
        "thresholds_evaluated": int(len(search)),
        "holdout_inputs_read": [],
        "training_started": False,
        "model_parameters_updated": False,
        "locked_datasets_read": [],
    }
    (output / "threshold_search.json").write_text(
        json.dumps(search, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    calibration_path = output / "calibration.json"
    calibration_path.write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    calibration["calibration_sha256"] = file_sha256(calibration_path)
    return calibration


def _evaluate(config: dict) -> dict[str, Any]:
    output = Path(config["output_dir"])
    calibration_path = output / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("passed") is not True or calibration.get("selected_threshold") is None:
        raise RuntimeError("G14-C tune-only fitting did not produce a feasible threshold")
    if calibration.get("holdout_inputs_read") != []:
        raise ValueError("G14-C calibration is not tune-only")
    threshold = float(calibration["selected_threshold"])
    prior = json.loads(Path(config["evaluation"]["g14_b_summary"]).read_text(encoding="utf-8"))
    baseline = prior["baseline"]
    dads_frame = pd.read_csv(config["evaluation"]["dads_test_manifest"], low_memory=False)
    dads_test = dads_frame.loc[dads_frame["split"].astype(str).eq("test")].reset_index(drop=True)
    holdout = pd.read_csv(config["evaluation"]["val_ood_holdout_manifest"], low_memory=False)
    guard_rows, _ = validate_guard_manifest(
        Path(config["evaluation"]["g9_guard_manifest"]),
        Path(config["evaluation"]["g9_guard_audit"]),
    )
    seed_results = []
    for seed in [int(value) for value in config["seeds"]]:
        dads_probability = np.load(
            str(config["evaluation"]["dads_test_probability_pattern"]).format(seed=seed)
        )
        holdout_probability = np.load(
            str(config["evaluation"]["val_ood_holdout_probability_pattern"]).format(seed=seed)
        )
        guard_probability = np.load(
            str(config["evaluation"]["g9_guard_probability_pattern"]).format(seed=seed)
        )
        dads = _metrics(dads_test["label"].to_numpy(), dads_probability, threshold)
        val_ood = _metrics(holdout["label"].to_numpy(), holdout_probability, threshold)
        guard = negative_metrics(guard_rows, guard_probability, threshold)
        checks = [
            dads["f1"] >= baseline["dads_test"]["f1"] - 0.005,
            dads["auc"] >= baseline["dads_test"]["auc"] - 0.002,
            val_ood["f1"] >= baseline["val_ood_holdout"]["f1"] - 0.02,
            val_ood["auc"] >= baseline["val_ood_holdout"]["auc"] - 0.01,
            guard["segment_false_positive_rate"]
            <= baseline["g9_guard"]["segment_false_positive_rate"] + 0.01,
        ]
        seed_results.append(
            {
                "seed": seed,
                "threshold": threshold,
                "dads_test": dads,
                "val_ood_holdout": val_ood,
                "g9_guard": guard,
                "checks": checks,
                "passed": all(checks),
            }
        )
    gate = all(row["passed"] for row in seed_results)
    report = {
        "passed": True,
        "protocol": config["protocol"],
        "selected_threshold": threshold,
        "calibration_path": calibration_path.as_posix(),
        "calibration_sha256": file_sha256(calibration_path),
        "regression_gate_passed": gate,
        "decision": (
            "continue_counterfactual_control_not_promote"
            if gate
            else "terminate_g14_head_only"
        ),
        "seeds": seed_results,
        "holdouts_previously_observed_in_g14_b": True,
        "therefore_not_final_evidence": True,
        "training_started": False,
        "model_parameters_updated": False,
        "locked_datasets_read": [],
    }
    (output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="G14-C tune-only threshold calibration")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g14_c_tune_only_threshold.yaml")
    )
    parser.add_argument("--mode", choices=("preflight", "fit", "evaluate"), required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    functions = {"preflight": _preflight, "fit": _fit, "evaluate": _evaluate}
    print(json.dumps(functions[args.mode](config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
