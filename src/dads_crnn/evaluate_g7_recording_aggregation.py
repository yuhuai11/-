from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .audio import decode_wav_bytes, ensure_sample_rate, peak_normalize
from .calibrate_ood import fit_temperature, verify_manifest_audio_hashes
from .config import ensure_dirs, load_config
from .evaluate_low_fpr import ranking_metrics, threshold_at_target_fpr
from .metrics import binary_metrics
from .prepare_beats_probe import reject_locked_path
from .train import _build_feature_extractor, _build_model, resolve_device


REQUIRED_MANIFEST_COLUMNS = {"path", "sha256", "label", "ood_split"}
SUPPORTED_METHODS = {
    "constrained_autopool_probability",
    "mean_probability",
    "max_probability",
    "mean_logit_then_sigmoid",
    "topk_mean_probability",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_window_count(samples: int, sample_rate: int, target_rate: int, window_samples: int) -> int:
    """Return the number of complete native windows after deterministic resampling."""
    if samples < 0 or sample_rate <= 0 or target_rate <= 0 or window_samples <= 0:
        raise ValueError("Invalid audio/window dimensions")
    resampled_samples = int(math.ceil(samples * target_rate / sample_rate))
    return resampled_samples // window_samples


class NativeWindowDataset(Dataset):
    """Expose every complete 0.5 s window without looping, padding, or stretching."""

    def __init__(self, manifest_path: Path, *, sample_rate: int, clip_seconds: float) -> None:
        self.rows = pd.read_csv(manifest_path)
        missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(self.rows.columns))
        if missing:
            raise ValueError(f"Missing manifest columns: {missing}")
        if self.rows.empty:
            raise ValueError(f"Empty manifest: {manifest_path}")
        self.sample_rate = int(sample_rate)
        self.window_samples = int(round(self.sample_rate * float(clip_seconds)))
        if not math.isclose(float(clip_seconds), 0.5, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Recording aggregation protocol requires native 0.5 s windows")

        index_rows: list[dict[str, Any]] = []
        for recording_index, row in self.rows.reset_index(drop=True).iterrows():
            path = Path(str(row["path"]))
            audio, original_rate = decode_wav_bytes(path.read_bytes())
            audio = ensure_sample_rate(audio, original_rate, self.sample_rate)
            complete_windows = int(audio.size // self.window_samples)
            if complete_windows == 0:
                raise ValueError(
                    f"Recording shorter than one native 0.5 s window; padding is forbidden: {path}"
                )
            for segment_index in range(complete_windows):
                start = segment_index * self.window_samples
                index_rows.append(
                    {
                        "recording_index": int(recording_index),
                        "recording_id": str(row["sha256"]),
                        "segment_index": int(segment_index),
                        "start_sample": int(start),
                        "end_sample": int(start + self.window_samples),
                        "resampled_samples": int(audio.size),
                        "discarded_tail_samples": int(audio.size % self.window_samples),
                        "label": int(row["label"]),
                    }
                )
        self.window_rows = pd.DataFrame(index_rows)

    def __len__(self) -> int:
        return len(self.window_rows)

    def __getitem__(self, index: int):
        window = self.window_rows.iloc[index]
        recording = self.rows.iloc[int(window["recording_index"])]
        audio, original_rate = decode_wav_bytes(Path(str(recording["path"])).read_bytes())
        audio = ensure_sample_rate(audio, original_rate, self.sample_rate)
        start = int(window["start_sample"])
        end = int(window["end_sample"])
        segment = audio[start:end]
        if segment.size != self.window_samples:
            raise RuntimeError("Native window index no longer matches decoded audio")
        segment = peak_normalize(segment)
        return (
            torch.from_numpy(segment.astype(np.float32, copy=False)),
            torch.tensor(float(window["label"]), dtype=torch.float32),
            int(index),
        )


def aggregate_scores(
    probabilities: np.ndarray,
    *,
    method: str,
    topk_fraction: float | None = None,
    max_instance_weight: float | None = None,
) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Aggregation requires a non-empty finite 1D score array")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Probabilities must be between zero and one")
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported recording aggregation method: {method}")
    if method == "mean_probability":
        return float(values.mean())
    if method == "max_probability":
        return float(values.max())
    if method == "mean_logit_then_sigmoid":
        clipped = np.clip(values, 1e-8, 1.0 - 1e-8)
        return float(expit(logit(clipped).mean()))
    if method == "constrained_autopool_probability":
        if values.size == 1:
            return float(values[0])
        minimum_weight = 1.0 / values.size
        if (
            max_instance_weight is None
            or not minimum_weight <= max_instance_weight < 1.0
        ):
            raise ValueError(
                "constrained_autopool_probability requires "
                "1 / windows <= max_instance_weight < 1"
            )
        # McFee et al. (2018), eqs. (8) and (11).  We use the largest
        # admissible alpha, so one extreme instance can never receive more
        # than max_instance_weight of the recording-level responsibility.
        alpha = math.log(max_instance_weight / (1.0 - max_instance_weight))
        alpha += math.log(values.size - 1)
        scaled = alpha * values
        weights = np.exp(scaled - scaled.max())
        weights /= weights.sum()
        return float(np.sum(values * weights))
    if topk_fraction is None or not 0.0 < topk_fraction <= 1.0:
        raise ValueError("topk_mean_probability requires 0 < topk_fraction <= 1")
    count = max(1, int(math.ceil(values.size * topk_fraction)))
    return float(np.partition(values, values.size - count)[-count:].mean())


def aggregate_recordings(
    segment_rows: pd.DataFrame,
    *,
    method: str,
    topk_fraction: float | None = None,
    max_instance_weight: float | None = None,
) -> pd.DataFrame:
    required = {"recording_index", "recording_id", "segment_index", "label", "probability"}
    missing = sorted(required - set(segment_rows.columns))
    if missing:
        raise ValueError(f"Missing segment prediction columns: {missing}")
    output = []
    for recording_index, group in segment_rows.groupby("recording_index", sort=True):
        group = group.sort_values("segment_index", kind="stable")
        expected = np.arange(len(group), dtype=np.int64)
        if not np.array_equal(group["segment_index"].to_numpy(dtype=np.int64), expected):
            raise ValueError("Recording windows must be complete, consecutive, and zero-indexed")
        if group["label"].nunique() != 1 or group["recording_id"].nunique() != 1:
            raise ValueError("A recording has inconsistent identity or labels")
        output.append(
            {
                "recording_index": int(recording_index),
                "recording_id": str(group["recording_id"].iloc[0]),
                "label": int(group["label"].iloc[0]),
                "windows": int(len(group)),
                "probability": aggregate_scores(
                    group["probability"].to_numpy(dtype=np.float64),
                    method=method,
                    topk_fraction=topk_fraction,
                    max_instance_weight=max_instance_weight,
                ),
            }
        )
    return pd.DataFrame(output)


def _method_name(specification: dict[str, Any]) -> str:
    method = str(specification["method"])
    if method == "topk_mean_probability":
        return f"topk_mean_probability_{float(specification['topk_fraction']):g}"
    if method == "constrained_autopool_probability":
        return (
            "constrained_autopool_probability_"
            f"{float(specification['max_instance_weight']):g}"
        )
    return method


def _predict_segments(
    checkpoint_path: Path,
    manifest_path: Path,
    *,
    batch_size: int,
    num_workers: int,
    device_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, Any], int]:
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    seed = int(checkpoint.get("seed", checkpoint_path.parent.name.removeprefix("seed_")))
    model_type = str(config["model"].get("type", "crnn"))
    if model_type == "panns_cnn14_16k":
        from .train_panns import build_model as build_panns_model

        model = build_panns_model(config).to(device)
        feature_extractor = None
    else:
        model = _build_model(config).to(device)
        feature_extractor = _build_feature_extractor(config).to(device)
        feature_extractor.eval()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = NativeWindowDataset(
        manifest_path,
        sample_rate=int(config["data"]["sample_rate"]),
        clip_seconds=float(config["data"]["clip_seconds"]),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logits = np.empty(len(dataset), dtype=np.float32)
    use_amp = model_type == "panns_cnn14_16k" and device.type == "cuda" and bool(
        config["train"].get("mixed_precision", True)
    )
    with torch.no_grad():
        for waveform, _, indices in tqdm(loader, desc=f"seed {seed} {manifest_path.stem} windows"):
            waveform = waveform.to(device)
            with torch.amp.autocast(device.type, enabled=use_amp):
                output = model(waveform) if feature_extractor is None else model(
                    feature_extractor(waveform)
                )
            logits[indices.numpy()] = output.detach().cpu().numpy()
    return dataset.rows.copy(), dataset.window_rows.copy(), logits, config, seed


def _attach_recording_metadata(recordings: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    metadata = manifest.reset_index(drop=True).reset_index(names="recording_index")
    merged = recordings.merge(metadata, on=["recording_index", "label"], how="left", validate="one_to_one")
    if merged["path"].isna().any():
        raise RuntimeError("Recording aggregation metadata merge failed")
    return merged


def _evaluate_method_on_tune(
    segment_rows: pd.DataFrame,
    specification: dict[str, Any],
    target_fprs: list[float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    method = str(specification["method"])
    topk_fraction = specification.get("topk_fraction")
    max_instance_weight = specification.get("max_instance_weight")
    recordings = aggregate_recordings(
        segment_rows,
        method=method,
        topk_fraction=topk_fraction,
        max_instance_weight=max_instance_weight,
    )
    labels = recordings["label"].to_numpy(dtype=np.int64)
    scores = recordings["probability"].to_numpy(dtype=np.float64)
    calibrations = {}
    for target in target_fprs:
        calibration = threshold_at_target_fpr(scores[labels == 0], target)
        threshold = float(calibration["threshold"])
        calibrations[f"{target:g}"] = {
            **calibration,
            "metrics": binary_metrics(labels, scores, threshold),
        }
    result = {
        "name": _method_name(specification),
        "method": method,
        "topk_fraction": None if topk_fraction is None else float(topk_fraction),
        "max_instance_weight": (
            None if max_instance_weight is None else float(max_instance_weight)
        ),
        "eligible_for_selection": bool(
            specification.get("eligible_for_selection", True)
        ),
        "ranking": ranking_metrics(labels, scores),
        "operating_points": calibrations,
    }
    return recordings, result


def _selection_key(result: dict[str, Any], target_fprs: list[float]) -> tuple[float, ...]:
    primary = result["operating_points"][f"{target_fprs[0]:g}"]["metrics"]["recall"]
    secondary = result["operating_points"][f"{target_fprs[-1]:g}"]["metrics"]["recall"]
    return (
        float(primary),
        float(secondary),
        float(result["ranking"]["partial_auc_fpr_0_05_standardized"]),
        float(result["ranking"]["pr_auc"]),
    )


def evaluate_recording_aggregation(
    checkpoint_path: Path,
    tune_manifest_path: Path,
    holdout_manifest_path: Path,
    protocol_path: Path,
    output_dir: Path,
    *,
    experiment: str,
    batch_size: int,
    num_workers: int,
    device_name: str,
) -> dict[str, Any]:
    for path in (checkpoint_path, tune_manifest_path, holdout_manifest_path, protocol_path):
        reject_locked_path(path)
    protocol = load_config(protocol_path)
    target_fprs = sorted({float(value) for value in protocol["target_fprs"]})
    methods = list(protocol["methods"])
    if not methods or not target_fprs:
        raise ValueError("At least one aggregation method and target FPR are required")

    audio_audit = {
        "tune": verify_manifest_audio_hashes(tune_manifest_path),
        "holdout": verify_manifest_audio_hashes(holdout_manifest_path),
    }
    tune_manifest, tune_windows, tune_logits, config, seed = _predict_segments(
        checkpoint_path,
        tune_manifest_path,
        batch_size=batch_size,
        num_workers=num_workers,
        device_name=device_name,
    )
    holdout_manifest, holdout_windows, holdout_logits, _, holdout_seed = _predict_segments(
        checkpoint_path,
        holdout_manifest_path,
        batch_size=batch_size,
        num_workers=num_workers,
        device_name=device_name,
    )
    if seed != holdout_seed:
        raise ValueError("Tune and Holdout checkpoint seeds disagree")
    if set(tune_manifest["sha256"].astype(str)) & set(holdout_manifest["sha256"].astype(str)):
        raise ValueError("Tune and Holdout recording hashes overlap")

    tune_window_labels = tune_windows["label"].to_numpy(dtype=np.int64)
    temperature = fit_temperature(tune_window_labels, tune_logits)
    tune_windows["logit"] = tune_logits.astype(np.float64)
    tune_windows["probability"] = expit(tune_logits.astype(np.float64) / temperature)
    holdout_windows["logit"] = holdout_logits.astype(np.float64)
    holdout_windows["probability"] = expit(holdout_logits.astype(np.float64) / temperature)

    candidates: list[dict[str, Any]] = []
    tune_recordings_by_name: dict[str, pd.DataFrame] = {}
    for specification in methods:
        recordings, result = _evaluate_method_on_tune(
            tune_windows, specification, target_fprs
        )
        candidates.append(result)
        tune_recordings_by_name[result["name"]] = recordings
    eligible = [item for item in candidates if item["eligible_for_selection"]]
    if not eligible:
        raise ValueError("At least one aggregation method must be eligible for selection")
    selected = max(eligible, key=lambda item: _selection_key(item, target_fprs))
    selected_specification = next(
        item for item in methods if _method_name(item) == selected["name"]
    )
    tune_recordings = tune_recordings_by_name[selected["name"]]
    holdout_recordings = aggregate_recordings(
        holdout_windows,
        method=str(selected_specification["method"]),
        topk_fraction=selected_specification.get("topk_fraction"),
        max_instance_weight=selected_specification.get("max_instance_weight"),
    )
    tune_recordings = _attach_recording_metadata(tune_recordings, tune_manifest)
    holdout_recordings = _attach_recording_metadata(holdout_recordings, holdout_manifest)

    holdout_labels = holdout_recordings["label"].to_numpy(dtype=np.int64)
    holdout_scores = holdout_recordings["probability"].to_numpy(dtype=np.float64)
    holdout_operating_points = {}
    operating_rows = []
    for target in target_fprs:
        calibration = selected["operating_points"][f"{target:g}"]
        threshold = float(calibration["threshold"])
        metrics = binary_metrics(holdout_labels, holdout_scores, threshold)
        holdout_operating_points[f"{target:g}"] = metrics
        operating_rows.append(
            {
                "target_fpr": target,
                "threshold_frozen_on_tune": threshold,
                "tune_fpr": calibration["metrics"]["false_positive_rate"],
                "tune_tpr": calibration["metrics"]["recall"],
                "holdout_fpr": metrics["false_positive_rate"],
                "holdout_tpr": metrics["recall"],
                "holdout_precision": metrics["precision"],
                "holdout_f1": metrics["f1"],
            }
        )

    ensure_dirs(output_dir)
    tune_windows.to_csv(output_dir / "tune_segment_predictions.csv", index=False)
    holdout_windows.to_csv(output_dir / "holdout_segment_predictions.csv", index=False)
    tune_recordings.to_csv(output_dir / "tune_recording_predictions.csv", index=False)
    holdout_recordings.to_csv(output_dir / "holdout_recording_predictions.csv", index=False)
    pd.DataFrame(operating_rows).to_csv(output_dir / "operating_points.csv", index=False)
    candidate_rows = []
    for candidate in candidates:
        row = {
            "name": candidate["name"],
            "eligible_for_selection": candidate["eligible_for_selection"],
            **candidate["ranking"],
        }
        for target in target_fprs:
            point = candidate["operating_points"][f"{target:g}"]["metrics"]
            row[f"tpr_at_fpr_{target:g}"] = point["recall"]
            row[f"empirical_fpr_{target:g}"] = point["false_positive_rate"]
        candidate_rows.append(row)
    pd.DataFrame(candidate_rows).to_csv(output_dir / "tune_candidate_metrics.csv", index=False)

    report = {
        "experiment": experiment,
        "seed": seed,
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "protocol": {
            "input_window_seconds": float(config["data"]["clip_seconds"]),
            "window_rule": "all_complete_native_windows_no_loop_no_padding_no_stretch",
            "temperature_fit_split": "tune_segments_only",
            "aggregation_selection_split": "tune_recordings_only",
            "threshold_selection_split": "tune_negative_recordings_only",
            "holdout_used_for_selection": False,
            "target_fprs": target_fprs,
        },
        "temperature": temperature,
        "selected_method": selected,
        "tune_candidates": candidates,
        "holdout": {
            "ranking": ranking_metrics(holdout_labels, holdout_scores),
            "operating_points": holdout_operating_points,
        },
        "coverage": {
            "tune_recordings": int(len(tune_manifest)),
            "tune_windows": int(len(tune_windows)),
            "holdout_recordings": int(len(holdout_manifest)),
            "holdout_windows": int(len(holdout_windows)),
            "tune_discarded_tail_samples": int(
                tune_windows.groupby("recording_index")["discarded_tail_samples"].first().sum()
            ),
            "holdout_discarded_tail_samples": int(
                holdout_windows.groupby("recording_index")["discarded_tail_samples"].first().sum()
            ),
        },
        "inputs": {
            "tune_manifest": tune_manifest_path.as_posix(),
            "tune_manifest_sha256": _sha256(tune_manifest_path),
            "holdout_manifest": holdout_manifest_path.as_posix(),
            "holdout_manifest_sha256": _sha256(holdout_manifest_path),
            "audio_hash_audit": audio_audit,
        },
        "locked_datasets_read": [],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate complete native 0.5 s windows with Tune-selected recording aggregation"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tune-manifest", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/g7_r3_recording_aggregation.yaml")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    evaluate_recording_aggregation(
        args.checkpoint,
        args.tune_manifest,
        args.holdout_manifest,
        args.protocol,
        args.output_dir,
        experiment=args.experiment,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
