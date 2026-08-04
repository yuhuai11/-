from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .audio import decode_wav_bytes, ensure_sample_rate, peak_normalize, to_fixed_length
from .calibrate_ood import probabilities_from_logits
from .config import load_config
from .data_firewall import file_sha256
from .evaluate_low_fpr import clopper_pearson
from .train import resolve_device
from .train_panns import build_model


ALGORITHM = "g7_idmt_r0_frozen_background_v1"


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _input_paths(
    root: Path, config: dict[str, Any]
) -> tuple[Path, dict[str, Path], dict[str, Path]]:
    checkpoint = _resolve(root, config["model"]["checkpoint"]["path"])
    input_paths = {
        name: _resolve(root, settings["path"])
        for name, settings in config["inputs"].items()
    }
    outputs = {
        name: _resolve(root, value) for name, value in config["outputs"].items()
    }
    return checkpoint, input_paths, outputs


def _verify_declared_inputs(
    root: Path, config_path: Path, config: dict[str, Any]
) -> tuple[Path, dict[str, Path], dict[str, Path]]:
    if config.get("algorithm") != ALGORITHM:
        raise ValueError(f"R0 algorithm must be {ALGORITHM}")
    checkpoint, inputs, outputs = _input_paths(root, config)
    declared = {
        "checkpoint": (
            checkpoint,
            str(config["model"]["checkpoint"]["sha256"]),
        ),
        **{
            name: (path, str(config["inputs"][name]["sha256"]))
            for name, path in inputs.items()
        },
    }
    for name, (path, expected) in declared.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen R0 input {name}: {path}")
        observed = file_sha256(path)
        if observed != expected:
            raise ValueError(
                f"R0 input changed: {name} expected={expected} observed={observed}"
            )
    if int(config["model"]["seed"]) != 42:
        raise ValueError("Formal G7 R0 must use seed 42")
    contract = config["contract"]
    if contract.get("calibration_audio_used_in_r0") is not False:
        raise ValueError("R0 must not refit on IDMT calibration audio")
    if contract.get("final_holdout_audio_used") is not False:
        raise ValueError("R0 must not use final holdout audio")
    if str(contract["primary_operating_point"]) != "strict_target_fpr_1":
        raise ValueError("R0 primary operating point changed")
    return checkpoint, inputs, outputs


def _forbidden_tokens(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(value).strip().lower()
        for value in config["contract"]["forbidden_tokens"]
        if str(value).strip()
    )


def validate_development_manifest(
    path: Path, config: dict[str, Any]
) -> pd.DataFrame:
    rows = pd.read_csv(path)
    required = {
        "path",
        "label",
        "role",
        "location",
        "session_id",
        "event_group",
        "microphone",
        "traffic_content",
        "recording_id",
        "recording_sha256",
        "segment_index",
        "start_sample_16k",
        "end_sample_16k",
        "model_pcm_sha256",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"R0 development manifest is missing columns: {missing}")
    contract = config["contract"]
    if len(rows) != int(contract["expected_segments"]):
        raise ValueError(f"Unexpected R0 segment count: {len(rows)}")
    if set(rows["role"].astype(str)) != {str(contract["evaluation_role"])}:
        raise ValueError("R0 manifest contains a non-development role")
    if set(rows["location"].astype(str)) != set(
        str(value) for value in contract["allowed_locations"]
    ):
        raise ValueError("R0 manifest location set changed")
    if set(pd.to_numeric(rows["label"], errors="raise").astype(int)) != {0}:
        raise ValueError("IDMT R0 must be a pure-negative evaluation")
    if rows["recording_id"].nunique() != int(contract["expected_recordings"]):
        raise ValueError("Unexpected R0 recording count")
    if rows["session_id"].nunique() != int(contract["expected_sessions"]):
        raise ValueError("Unexpected R0 session count")
    if rows["event_group"].nunique() != int(contract["expected_events"]):
        raise ValueError("Unexpected R0 event count")
    per_recording = rows.groupby("recording_id")["segment_index"].agg(
        ["count", "nunique"]
    )
    expected_windows = int(contract["windows_per_recording"])
    if not (
        (per_recording["count"] == expected_windows)
        & (per_recording["nunique"] == expected_windows)
    ).all():
        raise ValueError("Every R0 recording must contain two unique windows")
    forbidden = _forbidden_tokens(config)
    for row_number, row in enumerate(rows.to_dict("records"), start=2):
        text = "|".join(str(value).lower() for value in row.values())
        matches = [token for token in forbidden if token in text]
        if matches:
            raise ValueError(
                f"Locked IDMT value in R0 manifest row {row_number}: {matches}"
            )
    rows["segment_index"] = pd.to_numeric(
        rows["segment_index"], errors="raise"
    ).astype(int)
    return rows.reset_index(drop=True)


class IdmtSegmentDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, sample_rate: int, clip_seconds: float):
        self.rows = rows
        self.sample_rate = int(sample_rate)
        self.target_samples = int(round(sample_rate * clip_seconds))
        self._cached_path = ""
        self._cached_recording_sha = ""
        self._cached_audio = np.empty(0, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def _recording(self, row: pd.Series) -> np.ndarray:
        path = Path(str(row["path"]))
        path_value = path.as_posix()
        expected_sha = str(row["recording_sha256"])
        if path_value != self._cached_path or expected_sha != self._cached_recording_sha:
            wav_bytes = path.read_bytes()
            observed = hashlib.sha256(wav_bytes).hexdigest()
            if observed != expected_sha:
                raise ValueError(f"R0 WAV SHA256 mismatch: {path}")
            audio, original_rate = decode_wav_bytes(wav_bytes)
            audio = ensure_sample_rate(audio, original_rate, self.sample_rate)
            audio = to_fixed_length(
                audio, self.target_samples * 2, random_crop=False
            )
            self._cached_path = path_value
            self._cached_recording_sha = expected_sha
            self._cached_audio = audio
        return self._cached_audio

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        recording = self._recording(row)
        start = int(row["start_sample_16k"])
        end = int(row["end_sample_16k"])
        waveform = peak_normalize(recording[start:end])
        observed_pcm = hashlib.sha256(
            waveform.astype("<f4", copy=False).tobytes()
        ).hexdigest()
        if observed_pcm != str(row["model_pcm_sha256"]):
            raise ValueError(f"R0 model PCM hash mismatch at row {index}")
        return (
            torch.from_numpy(waveform.astype(np.float32, copy=False)),
            index,
        )


def _load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, config, checkpoint


def freeze(config_path: Path, root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    checkpoint, inputs, outputs = _verify_declared_inputs(root, config_path, config)
    output = outputs["frozen_protocol"]
    if output.exists():
        raise FileExistsError(f"R0 protocol is already frozen: {output}")
    development = validate_development_manifest(inputs["development_manifest"], config)
    stage_b = json.loads(inputs["stage_b_audit"].read_text(encoding="utf-8"))
    if (
        stage_b.get("passed") is not True
        or int(stage_b.get("locked_audio_members_read", -1)) != 0
        or stage_b.get("model_inference_started") is not False
        or int(stage_b.get("label_conflicts", -1)) != 0
    ):
        raise ValueError("Stage B audit is not a clean prediction-free pass")
    report = {
        "algorithm": ALGORITHM,
        "status": "frozen_before_idmt_model_inference",
        "contract": config["contract"],
        "inputs": {
            "config": {
                "path": config_path.relative_to(root).as_posix(),
                "sha256": file_sha256(config_path),
            },
            "checkpoint": {
                "path": checkpoint.relative_to(root).as_posix(),
                "sha256": file_sha256(checkpoint),
            },
            **{
                name: {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                }
                for name, path in inputs.items()
            },
        },
        "implementation": {
            "path": Path(__file__).resolve().relative_to(root).as_posix(),
            "sha256": file_sha256(Path(__file__)),
        },
        "development_manifest": {
            "segments": len(development),
            "recordings": int(development["recording_id"].nunique()),
            "sessions": int(development["session_id"].nunique()),
            "events": int(development["event_group"].nunique()),
        },
        "idmt_calibration_audio_read": False,
        "idmt_development_audio_read": False,
        "idmt_final_holdout_audio_read": False,
        "model_predictions_generated": False,
        "training_started": False,
    }
    _atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _verify_frozen(
    config_path: Path,
    root: Path,
    config: dict[str, Any],
    frozen: dict[str, Any],
) -> tuple[Path, dict[str, Path], dict[str, Path]]:
    checkpoint, inputs, outputs = _verify_declared_inputs(root, config_path, config)
    if frozen.get("algorithm") != ALGORITHM:
        raise ValueError("Unexpected R0 frozen algorithm")
    if frozen.get("status") != "frozen_before_idmt_model_inference":
        raise ValueError("R0 protocol is not in its frozen pre-inference state")
    checks = {
        "config": config_path,
        "checkpoint": checkpoint,
        **inputs,
    }
    for name, path in checks.items():
        if frozen["inputs"][name]["sha256"] != file_sha256(path):
            raise ValueError(f"Frozen R0 input changed: {name}")
    if frozen["implementation"]["sha256"] != file_sha256(Path(__file__)):
        raise ValueError("R0 evaluator implementation changed after freeze")
    if frozen["contract"] != config["contract"]:
        raise ValueError("R0 contract changed after freeze")
    return checkpoint, inputs, outputs


def preflight(config_path: Path, root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    _, _, outputs = _verify_declared_inputs(root, config_path, config)
    frozen = json.loads(outputs["frozen_protocol"].read_text(encoding="utf-8"))
    checkpoint, _, _ = _verify_frozen(config_path, root, config, frozen)
    device = resolve_device(str(config["evaluation"]["device"]))
    model, model_config, checkpoint_payload = _load_model(checkpoint, device)
    samples = int(
        round(
            int(model_config["data"]["sample_rate"])
            * float(model_config["data"]["clip_seconds"])
        )
    )
    batch_size = 2 if device.type == "cpu" else 8
    waveform = torch.zeros((batch_size, samples), dtype=torch.float32, device=device)
    started = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast(
        device.type,
        enabled=device.type == "cuda"
        and bool(config["evaluation"]["mixed_precision"]),
    ):
        output = model(waveform).float()
    elapsed = time.perf_counter() - started
    if output.shape != (batch_size,) or not torch.isfinite(output).all():
        raise RuntimeError("R0 synthetic model preflight failed")
    report = {
        "algorithm": ALGORITHM,
        "passed": True,
        "device": str(device),
        "checkpoint_seed": int(checkpoint_payload.get("seed", -1)),
        "batch_size": batch_size,
        "synthetic_seconds": elapsed,
        "synthetic_items_per_second": batch_size / elapsed,
        "development_audio_read": False,
        "calibration_audio_read": False,
        "final_holdout_audio_read": False,
        "frozen_protocol_sha256": file_sha256(outputs["frozen_protocol"]),
    }
    _atomic_json(outputs["preflight"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _negative_rate_metrics(
    probabilities: np.ndarray,
    threshold: float,
    *,
    unit_name: str,
    exposure_hours: float | None = None,
) -> dict[str, Any]:
    predictions = probabilities >= threshold
    false_positives = int(predictions.sum())
    total = int(predictions.size)
    fpr = float(false_positives / total)
    result = {
        "threshold": float(threshold),
        unit_name: total,
        "false_positives": false_positives,
        "true_negatives": total - false_positives,
        "fpr": fpr,
        "specificity": float(1.0 - fpr),
        "fpr_clopper_pearson_95_ci": clopper_pearson(false_positives, total),
    }
    if exposure_hours is not None:
        result["exposure_hours"] = float(exposure_hours)
        result["false_positive_snippets_per_channel_hour"] = float(
            false_positives / exposure_hours
        )
    return result


def _segment_metrics(
    probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    return _negative_rate_metrics(
        probabilities,
        threshold,
        unit_name="segments",
        exposure_hours=float(probabilities.size / 3600.0),
    )


def _group_metrics(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    field: str,
) -> list[dict[str, Any]]:
    frame = rows[[field]].copy()
    frame["probability"] = probabilities
    frame["prediction"] = probabilities >= threshold
    output = []
    for value, group in frame.groupby(field, dropna=False, sort=True):
        count = int(len(group))
        false_positives = int(group["prediction"].sum())
        output.append(
            {
                "group_field": field,
                "group": "" if pd.isna(value) else str(value),
                "segments": count,
                "false_positives": false_positives,
                "fpr": float(false_positives / count),
                "mean_probability": float(group["probability"].mean()),
                "p95_probability": float(group["probability"].quantile(0.95)),
            }
        )
    return output


def _cluster_bootstrap(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    frame = pd.DataFrame(
        {
            "session_id": rows["session_id"].astype(str),
            "prediction": probabilities >= threshold,
        }
    )
    sessions = [
        group["prediction"].to_numpy(dtype=np.float64)
        for _, group in frame.groupby("session_id", sort=True)
    ]
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        chosen = rng.integers(0, len(sessions), size=len(sessions))
        sampled = np.concatenate([sessions[item] for item in chosen])
        values[index] = float(sampled.mean())
    return {
        "low": float(np.quantile(values, 0.025)),
        "high": float(np.quantile(values, 0.975)),
        "bootstrap_unit": "recording_session",
        "sessions": len(sessions),
    }


def _recording_metrics(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    frame = rows[["recording_id", "event_group", "session_id"]].copy()
    frame["probability"] = probabilities
    recordings = frame.groupby("recording_id", sort=False)["probability"].agg(
        ["max", "mean", "min"]
    )
    events = frame.groupby("event_group", sort=False)["probability"].max()
    return {
        "recordings": int(len(recordings)),
        "recording_any_window_positive": _negative_rate_metrics(
            recordings["max"].to_numpy(dtype=np.float64),
            threshold,
            unit_name="recordings",
        ),
        "recording_mean_probability_positive": _negative_rate_metrics(
            recordings["mean"].to_numpy(dtype=np.float64),
            threshold,
            unit_name="recordings",
        ),
        "recording_both_windows_positive": _negative_rate_metrics(
            recordings["min"].to_numpy(dtype=np.float64),
            threshold,
            unit_name="recordings",
        ),
        "events": int(len(events)),
        "event_any_sensor_any_window_positive": _negative_rate_metrics(
            events.to_numpy(dtype=np.float64),
            threshold,
            unit_name="events",
        ),
    }


def _session_macro(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            "session_id": rows["session_id"].astype(str),
            "prediction": probabilities >= threshold,
        }
    )
    values = frame.groupby("session_id", sort=True)["prediction"].mean()
    return {
        "sessions": int(len(values)),
        "macro_fpr": float(values.mean()),
        "worst_session_fpr": float(values.max()),
        "best_session_fpr": float(values.min()),
        "by_session": [
            {"session_id": str(name), "fpr": float(value)}
            for name, value in values.items()
        ],
    }


def _predict(
    model,
    dataset: IdmtSegmentDataset,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    mixed_precision: bool,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    logits = np.empty(len(dataset), dtype=np.float32)
    with torch.no_grad():
        for waveforms, indices in tqdm(loader, desc="G7-R0 IDMT development"):
            waveforms = waveforms.to(device, non_blocking=True)
            with torch.amp.autocast(
                device.type,
                enabled=device.type == "cuda" and mixed_precision,
            ):
                output = model(waveforms)
            logits[indices.numpy()] = output.float().cpu().numpy()
    return logits


def evaluate(config_path: Path, root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    _, _, outputs = _verify_declared_inputs(root, config_path, config)
    frozen_path = outputs["frozen_protocol"]
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    checkpoint, inputs, outputs = _verify_frozen(
        config_path, root, config, frozen
    )
    preflight_report = json.loads(outputs["preflight"].read_text(encoding="utf-8"))
    if (
        preflight_report.get("passed") is not True
        or preflight_report.get("frozen_protocol_sha256")
        != file_sha256(frozen_path)
    ):
        raise ValueError("R0 preflight is missing or not bound to this protocol")
    evaluation_dir = outputs["evaluation_dir"]
    metrics_path = evaluation_dir / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError(f"R0 evaluation is already complete: {metrics_path}")
    rows = validate_development_manifest(inputs["development_manifest"], config)
    device = resolve_device(str(config["evaluation"]["device"]))
    model, model_config, _ = _load_model(checkpoint, device)
    if (
        int(model_config["data"]["sample_rate"])
        != int(config["contract"]["sample_rate"])
        or not math.isclose(
            float(model_config["data"]["clip_seconds"]),
            float(config["contract"]["clip_seconds"]),
        )
    ):
        raise ValueError("G7 checkpoint audio protocol differs from frozen R0")
    dataset = IdmtSegmentDataset(
        rows,
        sample_rate=int(config["contract"]["sample_rate"]),
        clip_seconds=float(config["contract"]["clip_seconds"]),
    )
    cache_path = evaluation_dir / "g7_logits.npz"
    manifest_sha = file_sha256(inputs["development_manifest"])
    checkpoint_sha = file_sha256(checkpoint)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            if (
                str(cache["manifest_sha256"].item()) != manifest_sha
                or str(cache["checkpoint_sha256"].item()) != checkpoint_sha
            ):
                raise ValueError("R0 logit cache identity mismatch")
            logits = cache["logits"].astype(np.float32)
    else:
        batch_settings = config["evaluation"]["batch_size"]
        batch_size = int(batch_settings[device.type])
        logits = _predict(
            model,
            dataset,
            device,
            batch_size=batch_size,
            num_workers=int(config["evaluation"]["num_workers"]),
            mixed_precision=bool(config["evaluation"]["mixed_precision"]),
        )
        _atomic_npz(
            cache_path,
            logits=logits,
            manifest_sha256=np.asarray(manifest_sha),
            checkpoint_sha256=np.asarray(checkpoint_sha),
        )
    if logits.shape != (len(rows),) or not np.isfinite(logits).all():
        raise ValueError("Invalid G7 R0 logits")
    probabilities = probabilities_from_logits(
        logits, float(config["contract"]["temperature"])
    )
    predictions = rows.copy()
    predictions["g7_logit"] = logits
    predictions["calibrated_probability"] = probabilities
    operating_results = {}
    group_fields = (
        "location",
        "microphone",
        "traffic_content",
        "weather",
        "vehicle",
    )
    for index, (name, settings) in enumerate(
        config["contract"]["operating_points"].items()
    ):
        threshold = float(settings["threshold"])
        segment = _segment_metrics(probabilities, threshold)
        segment["session_cluster_bootstrap_95_ci"] = _cluster_bootstrap(
            rows,
            probabilities,
            threshold,
            samples=int(config["evaluation"]["bootstrap_samples"]),
            seed=int(config["evaluation"]["bootstrap_seed"]) + index,
        )
        operating_results[name] = {
            "source_target_fpr": settings["source_target_fpr"],
            "segment": segment,
            "recording_and_event": _recording_metrics(
                rows, probabilities, threshold
            ),
            "session_macro": _session_macro(rows, probabilities, threshold),
            "subgroups": [
                item
                for field in group_fields
                for item in _group_metrics(rows, probabilities, threshold, field)
            ],
        }
        predictions[f"prediction_{name}"] = (
            probabilities >= threshold
        ).astype(np.int64)
    prediction_path = evaluation_dir / "predictions.csv"
    _atomic_csv(prediction_path, predictions)
    report = {
        "algorithm": ALGORITHM,
        "status": "complete",
        "model": {
            "checkpoint": checkpoint.relative_to(root).as_posix(),
            "checkpoint_sha256": checkpoint_sha,
            "seed": 42,
        },
        "protocol": {
            "path": frozen_path.relative_to(root).as_posix(),
            "sha256": file_sha256(frozen_path),
            "temperature": float(config["contract"]["temperature"]),
            "primary_operating_point": config["contract"][
                "primary_operating_point"
            ],
        },
        "dataset": {
            "name": "IDMT-TRAFFIC",
            "role": "development_test",
            "segments": len(rows),
            "recordings": int(rows["recording_id"].nunique()),
            "events": int(rows["event_group"].nunique()),
            "sessions": int(rows["session_id"].nunique()),
            "locations": sorted(rows["location"].astype(str).unique()),
            "all_labels_negative": True,
        },
        "score_quantiles": {
            str(value): float(np.quantile(probabilities, value))
            for value in (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)
        },
        "operating_points": operating_results,
        "predictions": {
            "path": prediction_path.relative_to(root).as_posix(),
            "sha256": file_sha256(prediction_path),
        },
        "cache": {
            "path": cache_path.relative_to(root).as_posix(),
            "sha256": file_sha256(cache_path),
        },
        "idmt_calibration_audio_read": False,
        "idmt_development_audio_read": True,
        "idmt_final_holdout_audio_read": False,
        "training_started": False,
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
    primary = operating_results[
        str(config["contract"]["primary_operating_point"])
    ]["segment"]
    print(json.dumps(primary, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "preflight", "evaluate"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g7_idmt_r0.yaml")
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = _resolve(root, args.config).resolve(strict=True)
    if args.mode == "freeze":
        freeze(config_path, root)
    elif args.mode == "preflight":
        preflight(config_path, root)
    else:
        evaluate(config_path, root)


if __name__ == "__main__":
    main()
