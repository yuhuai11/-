from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from .audio import decode_wav_bytes, ensure_sample_rate, peak_normalize
from .data_firewall import file_sha256
from .dataset import DADSDataset
from .evaluate_low_fpr import threshold_at_target_fpr
from .g7_benchmark_validity import build_validity_report
from .metrics import binary_metrics
from .train import resolve_device
from .train_panns import build_model


PROTOCOL = "g7_r6_reusable_benchmark_native_halfsecond_v2"
SAMPLE_RATE = 16_000
TARGET_SAMPLES = 8_000


class WindowedWavDataset(Dataset):
    def __init__(self, base_rows: pd.DataFrame, *, expected_windows: int):
        self.base_rows = base_rows.reset_index(drop=True)
        views = []
        for base_index, row in self.base_rows.iterrows():
            for half_index in range(expected_windows):
                views.append(
                    {
                        "base_index": base_index,
                        "half_index": half_index,
                        "label": int(row["label"]),
                        "recording_group": str(row["recording_group"]),
                        "source_group": str(row["source_group"]),
                        "dataset_origin": str(row["dataset_origin"]),
                        "subtype": str(row.get("subtype", "")),
                    }
                )
        self.rows = pd.DataFrame(views)
        self._cached_index = -1
        self._cached_audio = np.empty(0, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def _audio(self, base_index: int) -> np.ndarray:
        if base_index != self._cached_index:
            row = self.base_rows.iloc[base_index]
            path = Path(str(row["path"]))
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != str(row["audio_sha256"]):
                raise ValueError(f"External WAV SHA256 mismatch: {path}")
            audio, rate = decode_wav_bytes(payload)
            audio = ensure_sample_rate(audio, rate, SAMPLE_RATE)
            self._cached_index = base_index
            self._cached_audio = audio
        return self._cached_audio

    def __getitem__(self, index: int):
        view = self.rows.iloc[index]
        base_index = int(view["base_index"])
        audio = self._audio(base_index)
        start = int(view["half_index"]) * TARGET_SAMPLES
        end = start + TARGET_SAMPLES
        if end > audio.size:
            raise ValueError(f"External recording is shorter than declared views: {base_index}")
        waveform = peak_normalize(audio[start:end])
        return torch.from_numpy(waveform.astype(np.float32, copy=False))


class WindowedNpyDataset(Dataset):
    def __init__(self, base_rows: pd.DataFrame):
        self.base_rows = base_rows.reset_index(drop=True)
        views = []
        self._counts = []
        for base_index, row in self.base_rows.iterrows():
            values = np.load(str(row["cache_path"]), mmap_mode="r")
            count = int(values.size // TARGET_SAMPLES)
            if count < 1:
                raise ValueError(f"ESC-50 cache has no complete half-second view: {row['cache_path']}")
            self._counts.append(count)
            for half_index in range(count):
                views.append(
                    {
                        "base_index": base_index,
                        "half_index": half_index,
                        "label": int(row["label"]),
                        "recording_group": str(row["recording_group"]),
                        "source_group": str(row["source_group"]),
                        "dataset_origin": "esc50_fold5_guard",
                        "subtype": str(row["hard_negative_class"]),
                    }
                )
        self.rows = pd.DataFrame(views)
        self._cached_index = -1
        self._cached_audio = np.empty(0, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        view = self.rows.iloc[index]
        base_index = int(view["base_index"])
        if base_index != self._cached_index:
            row = self.base_rows.iloc[base_index]
            self._cached_audio = np.load(str(row["cache_path"])).astype(np.float32, copy=False)
            self._cached_index = base_index
        start = int(view["half_index"]) * TARGET_SAMPLES
        waveform = peak_normalize(self._cached_audio[start : start + TARGET_SAMPLES])
        return torch.from_numpy(waveform.astype(np.float32, copy=False))


class WaveformOnlyDataset(Dataset):
    def __init__(self, dataset: DADSDataset):
        self.dataset = dataset
        self.rows = dataset.rows.copy()
        if "subtype" not in self.rows:
            positive = self.rows.get("uav_subtype", pd.Series("", index=self.rows.index)).fillna("").astype(str)
            fallback = self.rows["dataset_origin"].fillna("unknown").astype(str)
            self.rows["subtype"] = positive.where(positive.str.len() > 0, fallback)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index][0]


def _dads_dataset(path: Path, split: str) -> WaveformOnlyDataset:
    return WaveformOnlyDataset(
        DADSDataset(
            path,
            split,
            sample_rate=SAMPLE_RATE,
            clip_seconds=0.5,
            training=False,
            seed=42,
        )
    )


def _g13_dataset(path: Path) -> WindowedWavDataset:
    source = pd.read_csv(path, low_memory=False)
    base = pd.DataFrame(
        {
            "path": source["path"].astype(str),
            "audio_sha256": source["sha256"].astype(str),
            "label": source["label"].astype(int),
            "recording_group": "g13:" + source["sha256"].astype(str),
            "source_group": source["source_group"].astype(str),
            "dataset_origin": "g13_ddl_aerosonic",
            "subtype": source["condition"].astype(str),
        }
    )
    return WindowedWavDataset(base, expected_windows=2)


def _idmt_dataset(path: Path) -> WindowedWavDataset:
    source = pd.read_csv(path, low_memory=False).drop_duplicates("recording_id")
    base = pd.DataFrame(
        {
            "path": source["path"].astype(str),
            "audio_sha256": source["recording_sha256"].astype(str),
            "label": 0,
            "recording_group": "idmt:" + source["recording_id"].astype(str),
            "source_group": source["session_id"].astype(str),
            "dataset_origin": "idmt_traffic_development",
            "subtype": source["traffic_content"].astype(str),
        }
    )
    return WindowedWavDataset(base, expected_windows=4)


def _esc50_dataset(path: Path) -> WindowedNpyDataset:
    source = pd.read_csv(path, low_memory=False)
    return WindowedNpyDataset(source)


def _predict(model: torch.nn.Module, dataset: Dataset, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    values = []
    with torch.no_grad():
        for batch_index, waveform in enumerate(loader, start=1):
            waveform = waveform.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                values.append(torch.sigmoid(model(waveform)).float().cpu().numpy())
            if batch_index % 100 == 0 or batch_index == len(loader):
                print(f"  batches {batch_index}/{len(loader)}", flush=True)
    return np.concatenate(values).astype(np.float64)


def _load_model(path: Path, device: torch.device) -> torch.nn.Module:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(saved["config"])
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval()


def _ranking(labels: np.ndarray, scores: np.ndarray) -> dict[str, float] | None:
    if np.unique(labels).size != 2:
        return None
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "standardized_pauc_fpr_le_0_05": float(roc_auc_score(labels, scores, max_fpr=0.05)),
    }


def _negative_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    if set(np.unique(labels)) != {0}:
        raise ValueError("Pure-negative metric received positive rows")
    return {
        "threshold": float(threshold),
        "samples": int(labels.size),
        "false_positives": int(np.sum(scores >= threshold)),
        "false_positive_rate": float(np.mean(scores >= threshold)),
        "mean_probability": float(scores.mean()),
        "maximum_probability": float(scores.max()),
    }


def _report(rows: pd.DataFrame, scores: np.ndarray, thresholds: dict[str, float]) -> dict[str, Any]:
    labels = rows["label"].to_numpy(dtype=np.int64)
    if scores.shape != (len(rows),) or not np.isfinite(scores).all():
        raise ValueError("External predictions are not aligned finite values")
    grouped = rows.copy()
    grouped["probability"] = scores
    if bool((grouped.groupby("recording_group")["label"].nunique() > 1).any()):
        raise ValueError("External recording group contains conflicting labels")
    recording = grouped.groupby("recording_group", as_index=False).agg(
        label=("label", "first"),
        probability=("probability", "mean"),
        source_group=("source_group", "first"),
        dataset_origin=("dataset_origin", "first"),
        subtype=("subtype", "first"),
    )
    recording_labels = recording["label"].to_numpy(dtype=np.int64)
    recording_scores = recording["probability"].to_numpy(dtype=np.float64)

    source_conflicts = grouped.groupby("source_group")["label"].nunique().gt(1)
    if bool(source_conflicts.any()):
        source_group = None
        source_labels = np.empty(0, dtype=np.int64)
        source_scores = np.empty(0, dtype=np.float64)
    else:
        source_group = grouped.groupby("source_group", as_index=False).agg(
            label=("label", "first"),
            probability=("probability", "mean"),
            recordings=("recording_group", "nunique"),
            segments=("label", "size"),
            dataset_origin=("dataset_origin", "first"),
        )
        source_labels = source_group["label"].to_numpy(dtype=np.int64)
        source_scores = source_group["probability"].to_numpy(dtype=np.float64)

    def operating(frame_labels: np.ndarray, frame_scores: np.ndarray) -> dict[str, Any]:
        return {
            name: (
                binary_metrics(frame_labels, frame_scores, threshold)
                if np.unique(frame_labels).size == 2
                else _negative_metrics(frame_labels, frame_scores, threshold)
            )
            for name, threshold in thresholds.items()
        }

    def source_group_macro() -> dict[str, Any]:
        output = {}
        for name, threshold in thresholds.items():
            classified = recording.copy()
            classified["positive_prediction"] = (
                classified["probability"] >= threshold
            ).astype(float)
            rates = classified.groupby("source_group", as_index=False).agg(
                label=("label", "first"),
                positive_prediction_rate=("positive_prediction", "mean"),
                recordings=("recording_group", "size"),
            )
            positive = rates[rates["label"].eq(1)]["positive_prediction_rate"]
            negative = rates[rates["label"].eq(0)]["positive_prediction_rate"]
            macro_recall = float(positive.mean()) if len(positive) else None
            macro_fpr = float(negative.mean()) if len(negative) else None
            output[name] = {
                "threshold": float(threshold),
                "source_groups": int(len(rates)),
                "positive_source_groups": int(len(positive)),
                "negative_source_groups": int(len(negative)),
                "macro_recall": macro_recall,
                "macro_false_positive_rate": macro_fpr,
                "macro_balanced_accuracy": (
                    float((macro_recall + 1.0 - macro_fpr) / 2.0)
                    if macro_recall is not None and macro_fpr is not None
                    else None
                ),
            }
        return output

    subgroup = {}
    for subtype, indices in grouped.groupby("subtype", sort=True).groups.items():
        selected = np.asarray(list(indices), dtype=np.int64)
        subtype_labels = labels[selected]
        subgroup[str(subtype)] = {
            name: (
                {"recall": float(np.mean(scores[selected] >= threshold)), "rows": int(selected.size)}
                if set(np.unique(subtype_labels)) == {1}
                else {"fpr": float(np.mean(scores[selected] >= threshold)), "rows": int(selected.size)}
            )
            for name, threshold in thresholds.items()
        }
    return {
        "segments": int(len(rows)),
        "recordings": int(len(recording)),
        "source_groups": int(grouped["source_group"].nunique()),
        "source_groups_with_conflicting_labels": int(source_conflicts.sum()),
        "segment_ranking": _ranking(labels, scores),
        "recording_mean_ranking": _ranking(recording_labels, recording_scores),
        "source_group_mean_ranking": (
            _ranking(source_labels, source_scores) if source_group is not None else None
        ),
        "segment_operating_points": operating(labels, scores),
        "recording_mean_operating_points": operating(recording_labels, recording_scores),
        "source_group_macro_recording_operating_points": source_group_macro(),
        "source_group_mean_operating_points": (
            operating(source_labels, source_scores) if source_group is not None else None
        ),
        "segment_subtypes": subgroup,
    }


def _promotion_decision(model_reports: dict[str, Any]) -> dict[str, Any]:
    baseline = model_reports["g7_r2_control"]["datasets"]
    candidate = model_reports["g7_r6_candidate"]["datasets"]
    point = "fixed_0_5"

    def clustered(dataset: dict[str, Any], metric: str) -> float:
        return float(
            dataset["source_group_macro_recording_operating_points"][point][metric]
        )

    criteria = {
        "kielce_tau_group_balanced_accuracy_not_lower": (
            clustered(candidate["kielce_tau_holdout"], "macro_balanced_accuracy")
            >= clustered(baseline["kielce_tau_holdout"], "macro_balanced_accuracy")
        ),
        "kielce_tau_group_fpr_not_higher": (
            clustered(candidate["kielce_tau_holdout"], "macro_false_positive_rate")
            <= clustered(baseline["kielce_tau_holdout"], "macro_false_positive_rate")
        ),
        "g13_group_balanced_accuracy_not_lower": (
            clustered(candidate["g13_ddl_aerosonic"], "macro_balanced_accuracy")
            >= clustered(baseline["g13_ddl_aerosonic"], "macro_balanced_accuracy")
        ),
        "g13_recording_roc_auc_not_lower": (
            candidate["g13_ddl_aerosonic"]["recording_mean_ranking"]["roc_auc"]
            >= baseline["g13_ddl_aerosonic"]["recording_mean_ranking"]["roc_auc"]
        ),
        "idmt_group_fpr_not_higher": (
            clustered(candidate["idmt_traffic"], "macro_false_positive_rate")
            <= clustered(baseline["idmt_traffic"], "macro_false_positive_rate")
        ),
        "esc50_group_fpr_not_higher": (
            clustered(candidate["esc50_fold5_guard"], "macro_false_positive_rate")
            <= clustered(baseline["esc50_fold5_guard"], "macro_false_positive_rate")
        ),
    }
    failed = [name for name, passed in criteria.items() if not passed]
    return {
        "promoted_on_reusable_benchmark": not failed,
        "independent_final_claim_allowed": False,
        "operating_point": point,
        "criteria": criteria,
        "failed_criteria": failed,
        "reason": (
            "All reusable benchmark non-regression gates passed."
            if not failed
            else "Candidate failed reusable benchmark non-regression gates: "
            + ", ".join(failed)
        ),
    }


def evaluate(
    baseline: Path,
    candidate: Path,
    output_dir: Path,
    device_name: str,
    batch_size: int,
) -> dict[str, Any]:
    inputs = {
        "development_fit": Path("artifacts/g7_r6_reusable_multicorpus/fit_manifest.csv"),
        "calibration": Path("artifacts/g7_r6_reusable_multicorpus/threshold_calibration_manifest.csv"),
        "kielce_tau_holdout": Path("artifacts/g7_r5_train_val_test/locked_unseen_external_test/manifest.csv"),
        "g13": Path("artifacts/g13_external_confirmation/intake/external_confirmation_v2_manifest.csv"),
        "idmt": Path("artifacts/g7_improvement/stage_b/development_segments_dedup.csv"),
        "esc50": Path("artifacts/g9_hard_negatives/manifests/hn_guard.csv"),
    }
    datasets: dict[str, Dataset] = {
        "calibration": _dads_dataset(inputs["calibration"], "threshold_calibration"),
        "kielce_tau_holdout": _dads_dataset(inputs["kielce_tau_holdout"], "locked_external_test"),
        "g13_ddl_aerosonic": _g13_dataset(inputs["g13"]),
        "idmt_traffic": _idmt_dataset(inputs["idmt"]),
        "esc50_fold5_guard": _esc50_dataset(inputs["esc50"]),
    }
    device = resolve_device(device_name)
    checkpoints = {"g7_r2_control": baseline, "g7_r6_candidate": candidate}
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_by_model: dict[str, dict[str, np.ndarray]] = {}
    thresholds_by_model = {}
    model_reports = {}
    for model_name, checkpoint in checkpoints.items():
        print(f"Loading {model_name}: {checkpoint}", flush=True)
        model = _load_model(checkpoint, device)
        scores_by_model[model_name] = {}
        for dataset_name, dataset in datasets.items():
            probability_path = output_dir / f"{model_name}_{dataset_name}_probabilities.npy"
            if probability_path.is_file():
                cached = np.load(probability_path)
                if cached.shape == (len(dataset),) and np.isfinite(cached).all():
                    print(f"Reusing {model_name} / {dataset_name}: {len(dataset)} saved views", flush=True)
                    scores = cached.astype(np.float64, copy=False)
                else:
                    print(f"Ignoring invalid saved probabilities: {probability_path}", flush=True)
                    scores = _predict(model, dataset, device, batch_size)
            else:
                print(f"Predicting {model_name} / {dataset_name}: {len(dataset)} views", flush=True)
                scores = _predict(model, dataset, device, batch_size)
            scores_by_model[model_name][dataset_name] = scores
            np.save(probability_path, scores)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        calibration_scores = scores_by_model[model_name]["calibration"]
        thresholds = {"fixed_0_5": 0.5}
        calibration = {}
        for target in (0.01, 0.05):
            item = threshold_at_target_fpr(calibration_scores, target)
            key = f"calibrated_fpr_{target:.2f}"
            thresholds[key] = float(item["threshold"])
            calibration[key] = item
        thresholds_by_model[model_name] = thresholds
        model_reports[model_name] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "calibration": calibration,
            "datasets": {
                dataset_name: _report(dataset.rows, scores_by_model[model_name][dataset_name], thresholds)
                for dataset_name, dataset in datasets.items()
                if dataset_name != "calibration"
            },
        }

    comparisons = {}
    for dataset_name in ("kielce_tau_holdout", "g13_ddl_aerosonic", "idmt_traffic", "esc50_fold5_guard"):
        comparisons[dataset_name] = {}
        for point in thresholds_by_model["g7_r2_control"]:
            b = model_reports["g7_r2_control"]["datasets"][dataset_name]["segment_operating_points"][point]
            c = model_reports["g7_r6_candidate"]["datasets"][dataset_name]["segment_operating_points"][point]
            if "f1" in b:
                comparisons[dataset_name][point] = {
                    "accuracy_delta": float(c["accuracy"] - b["accuracy"]),
                    "f1_delta": float(c["f1"] - b["f1"]),
                    "recall_delta": float(c["recall"] - b["recall"]),
                    "fpr_delta": float(c["false_positive_rate"] - b["false_positive_rate"]),
                }
            else:
                comparisons[dataset_name][point] = {
                    "fpr_delta": float(c["false_positive_rate"] - b["false_positive_rate"])
                }
    validity = build_validity_report(
        pd.read_csv(inputs["development_fit"], low_memory=False),
        pd.read_csv(inputs["calibration"], low_memory=False),
        pd.read_csv(inputs["kielce_tau_holdout"], low_memory=False),
        pd.read_csv(inputs["g13"], low_memory=False),
        pd.read_csv(inputs["idmt"], low_memory=False),
        pd.read_csv(inputs["esc50"], low_memory=False),
    )
    report = {
        "evaluation_completed": True,
        "promotion_decision": _promotion_decision(model_reports),
        "protocol": PROTOCOL,
        "validity": validity,
        "input_policy": {
            "native_halfsecond_views": True,
            "g13_views_per_recording": 2,
            "idmt_views_per_recording": 4,
            "recording_aggregation": "mean_probability",
            "threshold_source": "TAU_Prague_calibration_per_model",
        },
        "models": model_reports,
        "comparisons_candidate_minus_control": comparisons,
        "inputs": {
            name: {"path": str(path), "sha256": file_sha256(path)} for name, path in inputs.items()
        },
    }
    path = output_dir / "metrics.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run corrected reusable G7-R6 benchmark suite")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("artifacts/g7_r2_generalization/ablations/pt_mic_bg_freq/runs/seed_42/best.pt"),
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path("artifacts/g7_r6_dronenoise_control/runs/seed_42/best.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/g7_r6_dronenoise_control/external_suite"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    report = evaluate(args.baseline, args.candidate, args.output_dir, args.device, args.batch_size)
    print(
        json.dumps(
            {
                "evaluation_completed": report["evaluation_completed"],
                "promotion_decision": report["promotion_decision"],
                "comparisons": report["comparisons_candidate_minus_control"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
