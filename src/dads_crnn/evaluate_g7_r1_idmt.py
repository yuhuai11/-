from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .audit_g7_r1 import ALGORITHM
from .calibrate_ood import fit_temperature, probabilities_from_logits
from .config import load_config
from .data_firewall import file_sha256
from .evaluate_g7_idmt_r0 import (
    IdmtSegmentDataset,
    _atomic_csv,
    _atomic_json,
    _atomic_npz,
    _group_metrics,
    _load_model,
    _recording_metrics,
    _segment_metrics,
    _session_macro,
    _predict,
    validate_development_manifest,
)
from .evaluate_g7_r1_guards import (
    _checkpoint_and_metrics,
    _verify_frozen as _verify_training_frozen,
)
from .evaluate_low_fpr import threshold_at_target_fpr
from .train import resolve_device


IDMT_ALGORITHM = "g7_r1_idmt_development_per_model_tune_calibration_v1"


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _output_paths(root: Path, protocol: dict) -> dict[str, Path]:
    return {
        name: _resolve(root, value)
        for name, value in protocol["outputs"].items()
    }


def _frozen_evaluation_inputs(
    root: Path,
    protocol: dict,
    high_level_frozen: dict,
    checkpoint_path: Path,
) -> dict[str, Path]:
    outputs = _output_paths(root, protocol)
    return {
        "high_level_frozen_protocol": outputs["frozen_protocol"],
        "guard_evaluation": outputs["guard_evaluation"],
        "candidate_checkpoint": checkpoint_path,
        "r0_idmt_config": _resolve(root, protocol["inputs"]["r0_idmt_config"]),
        "stage_b_audit": _resolve(root, protocol["inputs"]["stage_b_audit"]),
        "idmt_development_manifest": _resolve(
            root, protocol["inputs"]["idmt_development_manifest"]
        ),
        "r0_frozen_protocol": _resolve(
            root, protocol["inputs"]["r0_frozen_protocol"]
        ),
        "r0_idmt_metrics": _resolve(root, protocol["inputs"]["r0_idmt_metrics"]),
        "r0_idmt_predictions": _resolve(
            root, protocol["inputs"]["r0_idmt_predictions"]
        ),
    }


def _validated_guard_score_mappings(
    root: Path,
    protocol: dict,
    outputs: dict[str, Path],
    guard: dict[str, Any],
) -> dict[str, Any]:
    score_protocol = protocol["contract"]["score_protocol"]
    if guard.get("score_protocol") != score_protocol:
        raise ValueError("R1 guard score protocol differs from the frozen contract")
    expected_prediction_paths = {
        "val_ood_tune": (
            outputs["guard_evaluation"].parent
            / "predictions"
            / "val_ood_tune.csv"
        ),
        "val_ood_holdout": (
            outputs["guard_evaluation"].parent
            / "predictions"
            / "val_ood_holdout.csv"
        ),
        "g9_guard": (
            outputs["guard_evaluation"].parent / "predictions" / "g9_guard.csv"
        ),
    }
    for name, expected_path in expected_prediction_paths.items():
        artifact = guard.get("predictions", {}).get(name, {})
        declared_path = _resolve(root, artifact.get("path", ""))
        if declared_path.resolve(strict=False) != expected_path.resolve(strict=False):
            raise ValueError(f"R1 guard prediction path changed: {name}")
        if (
            not declared_path.is_file()
            or artifact.get("sha256") != file_sha256(declared_path)
        ):
            raise ValueError(f"R1 guard prediction hash changed: {name}")

    tune_manifest_path = _resolve(
        root, protocol["inputs"]["val_ood_tune_manifest"]
    )
    tune_manifest = pd.read_csv(tune_manifest_path, low_memory=False)
    tune_predictions = pd.read_csv(
        expected_prediction_paths["val_ood_tune"], low_memory=False
    )
    for column in ("sha256", "label", "r1_logit"):
        if column not in tune_predictions:
            raise ValueError(f"R1 Tune predictions lack {column}")
    if len(tune_predictions) != len(tune_manifest):
        raise ValueError("R1 Tune prediction row count changed")
    for column in ("sha256", "label"):
        if not np.array_equal(
            tune_predictions[column].astype(str).to_numpy(),
            tune_manifest[column].astype(str).to_numpy(),
        ):
            raise ValueError(f"R1 Tune predictions do not align by {column}")
    labels = tune_manifest["label"].to_numpy(dtype=np.int64)
    logits = tune_predictions["r1_logit"].to_numpy(dtype=np.float64)
    if not np.isfinite(logits).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("R1 Tune calibration inputs are invalid")
    temperature = fit_temperature(labels, logits)
    probabilities = probabilities_from_logits(logits, temperature)
    calibrations = {
        float(target): threshold_at_target_fpr(
            probabilities[labels == 0], float(target)
        )
        for target in score_protocol["target_fprs"]
    }
    reported = guard.get("candidate_calibration", {})
    if (
        not np.isclose(
            float(reported.get("temperature", float("nan"))),
            temperature,
            rtol=0.0,
            atol=1e-12,
        )
        or reported.get("strict_target_fpr_1") != calibrations[0.01]
        or reported.get("sensitivity_target_fpr_5") != calibrations[0.05]
        or reported.get("holdout_used_for_temperature_or_threshold") is not False
        or reported.get("idmt_used_for_temperature_or_threshold") is not False
    ):
        raise ValueError("R1 guard calibration cannot be reproduced from Tune logits")
    return {
        "primary_comparison": score_protocol["primary_comparison"],
        "baseline_g7": score_protocol["baseline_g7_mapping"],
        "candidate_r1": {
            "temperature": temperature,
            "strict_threshold": float(calibrations[0.01]["threshold"]),
            "sensitivity_threshold": float(calibrations[0.05]["threshold"]),
            "source": "same_val_ood_tune_candidate_specific_calibration",
        },
        "fixed_g7_mapping_role": score_protocol["fixed_g7_mapping_role"],
    }


def freeze(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, high_level_frozen, outputs = _verify_training_frozen(
        root, protocol_path
    )
    frozen_path = outputs["idmt_frozen_protocol"]
    if frozen_path.exists():
        raise FileExistsError(f"R1 IDMT protocol already frozen: {frozen_path}")
    guard_path = outputs["guard_evaluation"]
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    if (
        guard.get("status") != "guard_evaluation_complete"
        or guard.get("decision") != "advance_to_idmt_development"
        or guard.get("gate_passed") is not True
        or not guard.get("checks")
        or not all(item.get("passed") is True for item in guard["checks"])
        or guard.get("locked_datasets_read") != []
        or guard.get("idmt_calibration_audio_read") is not False
        or guard.get("idmt_development_audio_read") is not False
        or guard.get("idmt_final_holdout_audio_read") is not False
    ):
        raise ValueError("R1 did not pass the pre-IDMT regression guards")
    checkpoint_path, checkpoint, _ = _checkpoint_and_metrics(
        root, protocol, high_level_frozen
    )
    if guard.get("checkpoint", {}).get("sha256") != file_sha256(checkpoint_path):
        raise ValueError("R1 guard result is not bound to the candidate checkpoint")
    score_mappings = _validated_guard_score_mappings(
        root, protocol, outputs, guard
    )
    inputs = _frozen_evaluation_inputs(
        root, protocol, high_level_frozen, checkpoint_path
    )
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing R1 IDMT input {name}: {path}")

    r0_config = load_config(inputs["r0_idmt_config"])
    development = validate_development_manifest(
        inputs["idmt_development_manifest"], r0_config
    )
    stage_b = json.loads(inputs["stage_b_audit"].read_text(encoding="utf-8"))
    if (
        stage_b.get("passed") is not True
        or int(stage_b.get("locked_audio_members_read", -1)) != 0
        or stage_b.get("model_inference_started") is not False
        or int(stage_b.get("label_conflicts", -1)) != 0
    ):
        raise ValueError("Stage-B identity audit is not a clean pass")

    report = {
        "algorithm": IDMT_ALGORITHM,
        "status": "frozen_before_r1_idmt_inference",
        "candidate": {
            "seed": int(checkpoint["seed"]),
            "checkpoint": checkpoint_path.relative_to(root).as_posix(),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "selection": "best_dads_validation_f1_under_frozen_training_protocol",
        },
        "score_protocol": protocol["contract"]["score_protocol"],
        "score_mappings": score_mappings,
        "promotion_gates": protocol["promotion_gates"][
            "idmt_development_at_candidate_tune_calibrated_operating_points"
        ],
        "inputs": {
            name: {
                "path": path.relative_to(root).as_posix(),
                "sha256": file_sha256(path),
            }
            for name, path in inputs.items()
        },
        "implementation": {
            "path": Path(__file__).resolve().relative_to(root).as_posix(),
            "sha256": file_sha256(Path(__file__)),
        },
        "development_manifest": {
            "segments": int(len(development)),
            "recordings": int(development["recording_id"].nunique()),
            "sessions": int(development["session_id"].nunique()),
            "events": int(development["event_group"].nunique()),
        },
        "training_complete": True,
        "guard_gate_passed": True,
        "idmt_calibration_audio_read": False,
        "idmt_development_audio_read": False,
        "idmt_final_holdout_audio_read": False,
        "model_predictions_generated": False,
        "locked_datasets_read": [],
    }
    _atomic_json(frozen_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _verify_evaluation_frozen(
    root: Path, protocol_path: Path
) -> tuple[dict, dict, dict[str, Path], Path]:
    protocol, high_level_frozen, outputs = _verify_training_frozen(
        root, protocol_path
    )
    frozen_path = outputs["idmt_frozen_protocol"]
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if (
        frozen.get("algorithm") != IDMT_ALGORITHM
        or frozen.get("status") != "frozen_before_r1_idmt_inference"
    ):
        raise ValueError("R1 IDMT evaluation protocol is not frozen")
    checkpoint_path, _, _ = _checkpoint_and_metrics(
        root, protocol, high_level_frozen
    )
    inputs = _frozen_evaluation_inputs(
        root, protocol, high_level_frozen, checkpoint_path
    )
    for name, path in inputs.items():
        if frozen["inputs"][name]["sha256"] != file_sha256(path):
            raise ValueError(f"Frozen R1 IDMT input changed: {name}")
    if frozen["implementation"]["sha256"] != file_sha256(Path(__file__)):
        raise ValueError("R1 IDMT evaluator changed after evaluation freeze")
    if frozen["score_protocol"] != protocol["contract"]["score_protocol"]:
        raise ValueError("R1 score calibration protocol changed after evaluation freeze")
    guard = json.loads(inputs["guard_evaluation"].read_text(encoding="utf-8"))
    expected_mappings = _validated_guard_score_mappings(
        root, protocol, outputs, guard
    )
    if frozen.get("score_mappings") != expected_mappings:
        raise ValueError("R1 candidate score mapping changed after evaluation freeze")
    expected_gates = protocol["promotion_gates"][
        "idmt_development_at_candidate_tune_calibrated_operating_points"
    ]
    if frozen["promotion_gates"] != expected_gates:
        raise ValueError("R1 IDMT gates changed after evaluation freeze")
    return protocol, frozen, outputs, checkpoint_path


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, _, outputs, checkpoint_path = _verify_evaluation_frozen(
        root, protocol_path
    )
    device = resolve_device("cuda")
    model, model_config, checkpoint = _load_model(checkpoint_path, device)
    samples = int(
        round(
            int(model_config["data"]["sample_rate"])
            * float(model_config["data"]["clip_seconds"])
        )
    )
    waveform = torch.zeros((8, samples), dtype=torch.float32, device=device)
    with torch.no_grad(), torch.amp.autocast(
        device.type,
        enabled=bool(model_config["train"].get("mixed_precision", True)),
    ):
        logits = model(waveform).float()
    if logits.shape != (8,) or not torch.isfinite(logits).all():
        raise RuntimeError("R1 IDMT synthetic CUDA preflight failed")
    report = {
        "algorithm": IDMT_ALGORITHM,
        "passed": True,
        "device": str(device),
        "checkpoint_seed": int(checkpoint["seed"]),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "batch_size": 8,
        "idmt_calibration_audio_read": False,
        "idmt_development_audio_read": False,
        "idmt_final_holdout_audio_read": False,
        "frozen_protocol_sha256": file_sha256(outputs["idmt_frozen_protocol"]),
        "locked_datasets_read": [],
    }
    _atomic_json(outputs["idmt_preflight"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _aligned_baseline_predictions(
    rows: pd.DataFrame,
    path: Path,
    *,
    temperature: float,
    strict_threshold: float,
    sensitivity_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline = pd.read_csv(path, low_memory=False)
    identity = ("model_pcm_sha256", "recording_id", "segment_index")
    if len(baseline) != len(rows):
        raise ValueError("R0/R1 IDMT prediction row counts differ")
    for column in (*identity, "g7_logit"):
        if column not in baseline:
            raise ValueError(f"R0 predictions lack {column}")
    for column in identity:
        if not np.array_equal(
            baseline[column].astype(str).to_numpy(),
            rows[column].astype(str).to_numpy(),
        ):
            raise ValueError(f"R0/R1 IDMT rows are not aligned by {column}")
    logits = baseline["g7_logit"].to_numpy(dtype=np.float64)
    probabilities = probabilities_from_logits(logits, temperature)
    strict = probabilities >= strict_threshold
    sensitivity = probabilities >= sensitivity_threshold
    return probabilities, strict, sensitivity


def _paired_session_bootstrap(
    rows: pd.DataFrame,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            "session_id": rows["session_id"].astype(str),
            "baseline": np.asarray(baseline_predictions, dtype=np.float64),
            "candidate": np.asarray(candidate_predictions, dtype=np.float64),
        }
    )
    sessions = [
        group[["baseline", "candidate"]].to_numpy(dtype=np.float64)
        for _, group in frame.groupby("session_id", sort=True)
    ]
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        chosen = rng.integers(0, len(sessions), size=len(sessions))
        sample = np.concatenate([sessions[item] for item in chosen], axis=0)
        deltas[index] = float(sample[:, 1].mean() - sample[:, 0].mean())
    return {
        "candidate_minus_baseline_fpr_delta": float(
            np.asarray(candidate_predictions, dtype=np.float64).mean()
            - np.asarray(baseline_predictions, dtype=np.float64).mean()
        ),
        "one_sided_95pct_upper": float(np.quantile(deltas, 0.95)),
        "two_sided_95pct": {
            "low": float(np.quantile(deltas, 0.025)),
            "high": float(np.quantile(deltas, 0.975)),
        },
        "bootstrap_unit": "recording_session",
        "sessions": len(sessions),
        "samples": int(samples),
        "seed": int(seed),
    }


def _subset_fpr(
    rows: pd.DataFrame,
    predictions: np.ndarray,
    **conditions: str,
) -> dict[str, Any]:
    mask = np.ones(len(rows), dtype=bool)
    for field, value in conditions.items():
        mask &= rows[field].astype(str).to_numpy() == str(value)
    selected = np.asarray(predictions, dtype=bool)[mask]
    if selected.size == 0:
        raise ValueError(f"Empty IDMT subgroup: {conditions}")
    return {
        "conditions": conditions,
        "segments": int(selected.size),
        "false_positives": int(selected.sum()),
        "fpr": float(selected.mean()),
    }


def _gate_check(
    name: str,
    value: float,
    *,
    maximum: float | None = None,
    require_below_zero: bool = False,
) -> dict[str, Any]:
    if maximum is not None:
        return {
            "name": name,
            "value": float(value),
            "maximum": float(maximum),
            "passed": float(value) <= float(maximum),
        }
    if require_below_zero:
        return {
            "name": name,
            "value": float(value),
            "required": "strictly_below_zero",
            "passed": float(value) < 0.0,
        }
    raise ValueError("A gate check requires a criterion")


def evaluate(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, frozen, outputs, checkpoint_path = _verify_evaluation_frozen(
        root, protocol_path
    )
    preflight_report = json.loads(
        outputs["idmt_preflight"].read_text(encoding="utf-8")
    )
    if (
        preflight_report.get("passed") is not True
        or preflight_report.get("frozen_protocol_sha256")
        != file_sha256(outputs["idmt_frozen_protocol"])
    ):
        raise ValueError("R1 IDMT CUDA preflight is missing or stale")
    evaluation_dir = outputs["idmt_evaluation_dir"]
    metrics_path = evaluation_dir / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError(f"R1 IDMT evaluation already complete: {metrics_path}")

    r0_config = load_config(_resolve(root, protocol["inputs"]["r0_idmt_config"]))
    manifest_path = _resolve(
        root, protocol["inputs"]["idmt_development_manifest"]
    )
    rows = validate_development_manifest(manifest_path, r0_config)
    device = resolve_device("cuda")
    model, model_config, checkpoint = _load_model(checkpoint_path, device)
    if (
        int(model_config["data"]["sample_rate"])
        != int(r0_config["contract"]["sample_rate"])
        or not math.isclose(
            float(model_config["data"]["clip_seconds"]),
            float(r0_config["contract"]["clip_seconds"]),
        )
    ):
        raise ValueError("R1 checkpoint audio protocol differs from G7-R0")
    dataset = IdmtSegmentDataset(
        rows,
        sample_rate=int(r0_config["contract"]["sample_rate"]),
        clip_seconds=float(r0_config["contract"]["clip_seconds"]),
    )
    cache_path = evaluation_dir / "r1_logits.npz"
    logits = _predict(
        model,
        dataset,
        device,
        batch_size=128,
        num_workers=0,
        mixed_precision=True,
    )
    if logits.shape != (len(rows),) or not np.isfinite(logits).all():
        raise ValueError("Invalid R1 IDMT logits")
    _atomic_npz(
        cache_path,
        logits=logits,
        manifest_sha256=np.asarray(file_sha256(manifest_path)),
        checkpoint_sha256=np.asarray(file_sha256(checkpoint_path)),
    )

    baseline_mapping = frozen["score_mappings"]["baseline_g7"]
    candidate_mapping = frozen["score_mappings"]["candidate_r1"]
    baseline_temperature = float(baseline_mapping["temperature"])
    baseline_strict_threshold = float(baseline_mapping["strict_threshold"])
    baseline_sensitivity_threshold = float(
        baseline_mapping["sensitivity_threshold"]
    )
    candidate_temperature = float(candidate_mapping["temperature"])
    candidate_strict_threshold = float(candidate_mapping["strict_threshold"])
    candidate_sensitivity_threshold = float(
        candidate_mapping["sensitivity_threshold"]
    )
    probabilities = probabilities_from_logits(logits, candidate_temperature)
    candidate_strict = probabilities >= candidate_strict_threshold
    candidate_sensitivity = probabilities >= candidate_sensitivity_threshold
    fixed_g7_mapping_probabilities = probabilities_from_logits(
        logits, baseline_temperature
    )
    candidate_fixed_g7_strict = (
        fixed_g7_mapping_probabilities >= baseline_strict_threshold
    )
    baseline_probabilities, baseline_strict, baseline_sensitivity = (
        _aligned_baseline_predictions(
            rows,
            _resolve(root, protocol["inputs"]["r0_idmt_predictions"]),
            temperature=baseline_temperature,
            strict_threshold=baseline_strict_threshold,
            sensitivity_threshold=baseline_sensitivity_threshold,
        )
    )

    baseline = {
        "strict": _segment_metrics(
            baseline_probabilities, baseline_strict_threshold
        ),
        "sensitivity": _segment_metrics(
            baseline_probabilities, baseline_sensitivity_threshold
        ),
        "me_vehicle_strict": _subset_fpr(
            rows,
            baseline_strict,
            microphone="ME",
            traffic_content="vehicle_passing",
        ),
        "se_strict": _subset_fpr(rows, baseline_strict, microphone="SE"),
        "background_only_strict": _subset_fpr(
            rows, baseline_strict, traffic_content="background_only"
        ),
    }
    candidate = {
        "strict": _segment_metrics(probabilities, candidate_strict_threshold),
        "sensitivity": _segment_metrics(
            probabilities, candidate_sensitivity_threshold
        ),
        "me_vehicle_strict": _subset_fpr(
            rows,
            candidate_strict,
            microphone="ME",
            traffic_content="vehicle_passing",
        ),
        "se_strict": _subset_fpr(rows, candidate_strict, microphone="SE"),
        "background_only_strict": _subset_fpr(
            rows, candidate_strict, traffic_content="background_only"
        ),
        "fixed_g7_mapping_diagnostic": {
            "strict": _segment_metrics(
                fixed_g7_mapping_probabilities, baseline_strict_threshold
            ),
            "me_vehicle_strict": _subset_fpr(
                rows,
                candidate_fixed_g7_strict,
                microphone="ME",
                traffic_content="vehicle_passing",
            ),
        },
    }
    gate = protocol["promotion_gates"][
        "idmt_development_at_candidate_tune_calibrated_operating_points"
    ]
    paired = _paired_session_bootstrap(
        rows,
        baseline_strict,
        candidate_strict,
        samples=int(gate["paired_session_bootstrap_samples"]),
        seed=int(gate["paired_session_bootstrap_seed"]),
    )
    checks = [
        _gate_check(
            "idmt_overall_fpr_strict",
            candidate["strict"]["fpr"],
            maximum=float(gate["maximum_overall_fpr_at_strict_threshold"]),
        ),
        _gate_check(
            "idmt_me_vehicle_fpr_strict",
            candidate["me_vehicle_strict"]["fpr"],
            maximum=float(gate["maximum_me_vehicle_fpr_at_strict_threshold"]),
        ),
        _gate_check(
            "idmt_se_fpr_strict",
            candidate["se_strict"]["fpr"],
            maximum=float(gate["maximum_se_fpr_at_strict_threshold"]),
        ),
        _gate_check(
            "idmt_background_only_fpr_strict",
            candidate["background_only_strict"]["fpr"],
            maximum=float(
                gate["maximum_background_only_fpr_at_strict_threshold"]
            ),
        ),
        _gate_check(
            "idmt_overall_fpr_sensitivity",
            candidate["sensitivity"]["fpr"],
            maximum=float(
                gate["maximum_overall_fpr_at_sensitivity_threshold"]
            ),
        ),
        _gate_check(
            "paired_session_bootstrap_one_sided_upper",
            paired["one_sided_95pct_upper"],
            require_below_zero=True,
        ),
    ]
    gate_passed = all(item["passed"] for item in checks)

    prediction_path = evaluation_dir / "predictions.csv"
    prediction_rows = rows.copy()
    prediction_rows["g7_r0_probability"] = baseline_probabilities
    prediction_rows["g7_r1_logit"] = logits
    prediction_rows["g7_r1_candidate_calibrated_probability"] = probabilities
    prediction_rows["g7_r1_fixed_g7_mapping_probability"] = (
        fixed_g7_mapping_probabilities
    )
    prediction_rows["g7_r0_strict_prediction"] = baseline_strict.astype(np.int64)
    prediction_rows["g7_r1_strict_prediction"] = candidate_strict.astype(np.int64)
    prediction_rows["g7_r0_sensitivity_prediction"] = baseline_sensitivity.astype(
        np.int64
    )
    prediction_rows["g7_r1_sensitivity_prediction"] = (
        candidate_sensitivity.astype(np.int64)
    )
    prediction_rows["g7_r1_fixed_g7_strict_prediction"] = (
        candidate_fixed_g7_strict.astype(np.int64)
    )
    _atomic_csv(prediction_path, prediction_rows)

    report = {
        "algorithm": IDMT_ALGORITHM,
        "status": "complete",
        "decision": (
            "advance_r1_seed42_to_seeds_43_44"
            if gate_passed
            else "reject_r1_seed42_after_idmt_development"
        ),
        "candidate": {
            "checkpoint": checkpoint_path.relative_to(root).as_posix(),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "seed": int(checkpoint["seed"]),
        },
        "protocol": {
            "path": outputs["idmt_frozen_protocol"].relative_to(root).as_posix(),
            "sha256": file_sha256(outputs["idmt_frozen_protocol"]),
            "score_protocol": frozen["score_protocol"],
            "score_mappings": frozen["score_mappings"],
        },
        "dataset": {
            "name": "IDMT-TRAFFIC",
            "role": "development_test",
            "segments": int(len(rows)),
            "recordings": int(rows["recording_id"].nunique()),
            "events": int(rows["event_group"].nunique()),
            "sessions": int(rows["session_id"].nunique()),
            "all_labels_negative": True,
        },
        "baseline_g7_r0": baseline,
        "candidate_g7_r1": candidate,
        "candidate_details": {
            "recording_and_event_strict": _recording_metrics(
                rows, probabilities, candidate_strict_threshold
            ),
            "session_macro_strict": _session_macro(
                rows, probabilities, candidate_strict_threshold
            ),
            "subgroups_strict": [
                item
                for field in (
                    "location",
                    "microphone",
                    "traffic_content",
                    "weather",
                    "vehicle",
                )
                for item in _group_metrics(
                    rows, probabilities, candidate_strict_threshold, field
                )
            ],
        },
        "paired_session_bootstrap": paired,
        "checks": checks,
        "gate_passed": gate_passed,
        "artifacts": {
            "predictions": {
                "path": prediction_path.relative_to(root).as_posix(),
                "sha256": file_sha256(prediction_path),
            },
            "logit_cache": {
                "path": cache_path.relative_to(root).as_posix(),
                "sha256": file_sha256(cache_path),
            },
        },
        "idmt_calibration_audio_read": False,
        "idmt_development_audio_read": True,
        "idmt_final_holdout_audio_read": False,
        "locked_datasets_read": [],
        "unsupported_metrics": [
            "TPR",
            "ROC-AUC",
            "PR-AUC",
            "pAUC",
            "precision",
            "F1",
        ],
    }
    _atomic_json(metrics_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate G7-R1 on IDMT development")
    parser.add_argument("mode", choices=("freeze", "preflight", "evaluate"))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/g7_r1_protocol.yaml"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = _resolve(root, args.protocol).resolve(strict=True)
    if args.mode == "freeze":
        freeze(root, protocol_path)
    elif args.mode == "preflight":
        preflight(root, protocol_path)
    else:
        evaluate(root, protocol_path)


if __name__ == "__main__":
    main()
