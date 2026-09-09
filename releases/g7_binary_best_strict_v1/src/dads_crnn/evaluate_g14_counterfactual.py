from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import ensure_dirs, load_config
from .panns import file_sha256
from .train import resolve_device
from .train_panns import build_model


PROTOCOL_V1 = "g14_d_paired_counterfactual_evaluation_v1"
PROTOCOL_V2 = "g14_d_paired_counterfactual_evaluation_v2"
SUPPORTED_PROTOCOLS = {PROTOCOL_V1, PROTOCOL_V2}


def synthesize_pair(
    background: np.ndarray,
    uav: np.ndarray,
    target_snr_db: float,
    *,
    epsilon: float,
    peak_limit: float,
    common_gain_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    background = np.asarray(background, dtype=np.float32)
    uav = np.asarray(uav, dtype=np.float32)
    if background.shape != uav.shape or background.ndim != 1:
        raise ValueError("Counterfactual source waveforms must be aligned 1-D arrays")
    background_rms = float(np.sqrt(np.mean(np.square(background, dtype=np.float64))))
    uav_rms = float(np.sqrt(np.mean(np.square(uav, dtype=np.float64))))
    if background_rms <= epsilon or uav_rms <= epsilon:
        raise ValueError("Counterfactual source waveform has near-zero RMS")

    target_ratio = 10.0 ** (float(target_snr_db) / 20.0)
    uav_component = uav * np.float32(background_rms * target_ratio / uav_rms)
    mixed = background + uav_component
    mixed_rms = float(np.sqrt(np.mean(np.square(mixed, dtype=np.float64))))
    if mixed_rms <= epsilon:
        raise ValueError("Counterfactual mixture has near-zero RMS")
    rms_match_gain = background_rms / mixed_rms
    positive_background = background * np.float32(rms_match_gain)
    positive_uav = uav_component * np.float32(rms_match_gain)
    positive = positive_background + positive_uav
    negative = background.copy()

    joint_peak = float(
        max(np.max(np.abs(negative)), np.max(np.abs(positive)))
    )
    required_common_gain = min(1.0, float(peak_limit) / max(joint_peak, epsilon))
    common_gain = (
        required_common_gain
        if common_gain_override is None
        else float(common_gain_override)
    )
    if not 0.0 < common_gain <= required_common_gain + 1e-7:
        raise ValueError(
            "Shared common gain must be positive and no larger than the "
            "condition-specific anti-clipping gain"
        )
    negative *= np.float32(common_gain)
    positive *= np.float32(common_gain)
    positive_background *= np.float32(common_gain)
    positive_uav *= np.float32(common_gain)

    negative_rms = float(np.sqrt(np.mean(np.square(negative, dtype=np.float64))))
    positive_rms = float(np.sqrt(np.mean(np.square(positive, dtype=np.float64))))
    component_snr = 20.0 * np.log10(
        max(
            float(np.sqrt(np.mean(np.square(positive_uav, dtype=np.float64)))),
            epsilon,
        )
        / max(
            float(np.sqrt(np.mean(np.square(positive_background, dtype=np.float64)))),
            epsilon,
        )
    )
    return (
        negative.astype(np.float32, copy=False),
        positive.astype(np.float32, copy=False),
        {
            "negative_rms": negative_rms,
            "positive_rms": positive_rms,
            "achieved_snr_db": float(component_snr),
            "common_gain": float(common_gain),
            "required_common_gain": float(required_common_gain),
            "joint_peak": float(
                max(np.max(np.abs(negative)), np.max(np.abs(positive)))
            ),
        },
    )


class CounterfactualPairDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        *,
        epsilon: float,
        peak_limit: float,
        share_gain_across_snr: bool = False,
    ) -> None:
        self.rows = pd.read_csv(manifest, low_memory=False)
        self.epsilon = float(epsilon)
        self.peak_limit = float(peak_limit)
        self.share_gain_across_snr = bool(share_gain_across_snr)
        self._memmaps: OrderedDict[str, np.ndarray] = OrderedDict()
        self._shared_gains: dict[str, float] = {}
        if self.share_gain_across_snr:
            self._shared_gains = self._compute_shared_gains()

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, path: str, index: int) -> np.ndarray:
        if path not in self._memmaps:
            array = np.load(path, mmap_mode="r")
            if array.ndim != 2:
                raise ValueError(f"Expected a 2-D segment memmap: {path}")
            self._memmaps[path] = array
        return np.asarray(self._memmaps[path][int(index)], dtype=np.float32)

    def _compute_shared_gains(self) -> dict[str, float]:
        required_columns = {
            "base_pair_id",
            "target_snr_db",
            "background_cache_path",
            "background_cache_index",
            "uav_cache_path",
            "uav_cache_index",
        }
        missing = required_columns.difference(self.rows.columns)
        if missing:
            raise ValueError(f"Counterfactual manifest lacks columns: {sorted(missing)}")
        gains: dict[str, float] = {}
        for base_pair_id, group in self.rows.groupby("base_pair_id", sort=False):
            identity_columns = [
                "background_cache_path",
                "background_cache_index",
                "uav_cache_path",
                "uav_cache_index",
            ]
            if any(group[column].nunique(dropna=False) != 1 for column in identity_columns):
                raise ValueError(f"Inconsistent source identity for {base_pair_id}")
            if group["target_snr_db"].duplicated().any() or len(group) < 2:
                raise ValueError(f"Invalid SNR conditions for {base_pair_id}")
            first = group.iloc[0]
            background = self._load(
                str(first["background_cache_path"]),
                int(first["background_cache_index"]),
            )
            uav = self._load(
                str(first["uav_cache_path"]), int(first["uav_cache_index"])
            )
            condition_gains = []
            for target_snr_db in group["target_snr_db"]:
                _, _, diagnostic = synthesize_pair(
                    background,
                    uav,
                    float(target_snr_db),
                    epsilon=self.epsilon,
                    peak_limit=self.peak_limit,
                )
                condition_gains.append(diagnostic["required_common_gain"])
            gains[str(base_pair_id)] = float(min(condition_gains))
        return gains

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        background = self._load(
            str(row["background_cache_path"]), int(row["background_cache_index"])
        )
        uav = self._load(str(row["uav_cache_path"]), int(row["uav_cache_index"]))
        negative, positive, _ = synthesize_pair(
            background,
            uav,
            float(row["target_snr_db"]),
            epsilon=self.epsilon,
            peak_limit=self.peak_limit,
            common_gain_override=self._shared_gains.get(str(row["base_pair_id"])),
        )
        return (
            torch.from_numpy(negative.copy()),
            torch.from_numpy(positive.copy()),
            int(index),
        )


def _verify(path: Path, expected: str | None = None) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = file_sha256(path)
    if expected and observed != str(expected):
        raise ValueError(f"SHA256 mismatch for {path}")
    return observed


def preflight(config: dict) -> dict[str, Any]:
    protocol = str(config.get("protocol"))
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError("Unsupported G14-D evaluation protocol")
    share_gain_across_snr = protocol == PROTOCOL_V2
    audit_path = Path(config["pair_audit"])
    _verify(audit_path, config["pair_audit_sha256"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True or audit.get("locked_datasets_read") != []:
        raise ValueError("G14-D pair audit is not valid")
    checked = {"pair_audit": file_sha256(audit_path)}
    datasets = {}
    mixing = config["mixing"]
    for split, value in config["pair_manifests"].items():
        path = Path(value)
        expected = audit["splits"][split]["manifest_sha256"]
        checked[f"{split}_manifest"] = _verify(path, expected)
        dataset = CounterfactualPairDataset(
            path,
            epsilon=float(mixing["epsilon"]),
            peak_limit=float(mixing["common_pair_peak_limit"]),
            share_gain_across_snr=share_gain_across_snr,
        )
        if share_gain_across_snr:
            expected_snrs = sorted(
                float(value) for value in mixing["expected_target_snr_db"]
            )
            for base_pair_id, group in dataset.rows.groupby(
                "base_pair_id", sort=False
            ):
                observed_snrs = sorted(group["target_snr_db"].astype(float).tolist())
                if observed_snrs != expected_snrs:
                    raise ValueError(
                        f"Unexpected SNR grid for {split}/{base_pair_id}: "
                        f"{observed_snrs}"
                    )
        maximum_rms_error = 0.0
        maximum_snr_error = 0.0
        maximum_peak = 0.0
        probe_indices = np.unique(
            np.linspace(0, len(dataset) - 1, num=min(32, len(dataset)), dtype=np.int64)
        )
        for index in probe_indices:
            row = dataset.rows.iloc[int(index)]
            background = dataset._load(
                str(row["background_cache_path"]), int(row["background_cache_index"])
            )
            uav = dataset._load(
                str(row["uav_cache_path"]), int(row["uav_cache_index"])
            )
            negative, positive, diagnostic = synthesize_pair(
                background,
                uav,
                float(row["target_snr_db"]),
                epsilon=float(mixing["epsilon"]),
                peak_limit=float(mixing["common_pair_peak_limit"]),
                common_gain_override=dataset._shared_gains.get(
                    str(row["base_pair_id"])
                ),
            )
            if not np.isfinite(negative).all() or not np.isfinite(positive).all():
                raise ValueError("Non-finite counterfactual waveform")
            maximum_rms_error = max(
                maximum_rms_error,
                abs(diagnostic["positive_rms"] - diagnostic["negative_rms"])
                / diagnostic["negative_rms"],
            )
            maximum_snr_error = max(
                maximum_snr_error,
                abs(diagnostic["achieved_snr_db"] - float(row["target_snr_db"])),
            )
            maximum_peak = max(maximum_peak, diagnostic["joint_peak"])
        if maximum_rms_error > float(mixing["rms_relative_tolerance"]):
            raise ValueError(f"RMS control failed for {split}")
        if maximum_snr_error > float(mixing["snr_absolute_tolerance_db"]):
            raise ValueError(f"SNR control failed for {split}")
        if maximum_peak > float(mixing["common_pair_peak_limit"]) + 1e-6:
            raise ValueError(f"Peak control failed for {split}")
        exact_negative_identity_pairs = 0
        if share_gain_across_snr:
            base_pair_ids = dataset.rows["base_pair_id"].drop_duplicates()
            identity_probe_ids = base_pair_ids.iloc[
                np.unique(
                    np.linspace(
                        0,
                        len(base_pair_ids) - 1,
                        num=min(32, len(base_pair_ids)),
                        dtype=np.int64,
                    )
                )
            ]
            for base_pair_id in identity_probe_ids:
                indices = dataset.rows.index[
                    dataset.rows["base_pair_id"] == base_pair_id
                ].tolist()
                negatives = [dataset[int(index)][0].numpy() for index in indices]
                if not all(
                    np.array_equal(negatives[0], negative)
                    for negative in negatives[1:]
                ):
                    raise ValueError(
                        f"Cross-SNR negative identity failed for {base_pair_id}"
                    )
                exact_negative_identity_pairs += 1
        datasets[split] = {
            "rows": len(dataset),
            "probed_pairs": len(probe_indices),
            "maximum_relative_rms_error": maximum_rms_error,
            "maximum_absolute_snr_error_db": maximum_snr_error,
            "maximum_peak": maximum_peak,
            "cross_snr_shared_gain": share_gain_across_snr,
            "base_pairs_with_precomputed_shared_gain": len(dataset._shared_gains),
            "minimum_shared_gain": (
                min(dataset._shared_gains.values())
                if dataset._shared_gains
                else None
            ),
            "exact_negative_identity_pairs_probed": exact_negative_identity_pairs,
        }

    models = []
    for spec in config["models"]:
        models.append(
            {
                "name": spec["name"],
                "role": spec["role"],
                "checkpoint_sha256": _verify(
                    Path(spec["checkpoint"]), spec["checkpoint_sha256"]
                ),
            }
        )
    report = {
        "passed": True,
        "protocol": protocol,
        "ready_for_gpu_evaluation": True,
        "checked_inputs": checked,
        "datasets": datasets,
        "models": models,
        "synthetic_audio_written": False,
        "model_inference_started": False,
        "training_started": False,
        "locked_datasets_read": [],
    }
    output = Path(config["output_dir"])
    ensure_dirs(output)
    (output / "preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _predict_model(
    spec: dict,
    dataset: CounterfactualPairDataset,
    runtime: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = torch.load(spec["checkpoint"], map_location=device, weights_only=True)
    model = build_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=int(runtime["batch_size_pairs"]),
        shuffle=False,
        num_workers=int(runtime["num_workers"]),
    )
    negative_probability = np.empty(len(dataset), dtype=np.float32)
    positive_probability = np.empty(len(dataset), dtype=np.float32)
    amp = device.type == "cuda" and bool(runtime["mixed_precision"])
    with torch.no_grad():
        for negative, positive, indices in tqdm(loader, desc=f"{spec['name']} pairs"):
            waveforms = torch.cat((negative, positive), dim=0).to(device)
            with torch.amp.autocast(device.type, enabled=amp):
                probabilities = torch.sigmoid(model(waveforms)).cpu().numpy()
            batch = len(indices)
            index_array = indices.numpy()
            negative_probability[index_array] = probabilities[:batch]
            positive_probability[index_array] = probabilities[batch:]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return negative_probability, positive_probability


def _bootstrap(values: np.ndarray, statistic: str, samples: int, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = values[rng.integers(0, len(values), size=len(values))]
        estimates[index] = draw.mean()
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def _group_metrics(
    rows: pd.DataFrame,
    *,
    threshold: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    lift = rows["positive_probability"].to_numpy() - rows["negative_probability"].to_numpy()
    ordered = lift > 0
    return {
        "pairs": int(len(rows)),
        "mean_negative_probability": float(rows["negative_probability"].mean()),
        "mean_positive_probability": float(rows["positive_probability"].mean()),
        "mean_lift": float(lift.mean()),
        "median_lift": float(np.median(lift)),
        "mean_lift_bootstrap_95": _bootstrap(
            lift, "mean", bootstrap_samples, bootstrap_seed
        ),
        "paired_ordering_accuracy": float(ordered.mean()),
        "paired_ordering_bootstrap_95": _bootstrap(
            ordered.astype(np.float64), "mean", bootstrap_samples, bootstrap_seed + 1
        ),
        "negative_fpr": float((rows["negative_probability"] >= threshold).mean()),
        "positive_tpr": float((rows["positive_probability"] >= threshold).mean()),
    }


def _summarize_predictions(
    rows: pd.DataFrame, statistics: dict, seed_offset: int
) -> dict[str, Any]:
    threshold = float(statistics["threshold"])
    samples = int(statistics["bootstrap_samples"])
    seed = int(statistics["bootstrap_seed"]) + seed_offset
    by_snr = {}
    for offset, (snr, group) in enumerate(
        rows.groupby("target_snr_db", sort=True)
    ):
        by_snr[str(float(snr))] = _group_metrics(
            group,
            threshold=threshold,
            bootstrap_samples=samples,
            bootstrap_seed=seed + offset * 10,
        )
    positive_pivot = rows.pivot(
        index="base_pair_id",
        columns="target_snr_db",
        values="positive_probability",
    ).sort_index(axis=1)
    negative_pivot = rows.pivot(
        index="base_pair_id",
        columns="target_snr_db",
        values="negative_probability",
    ).sort_index(axis=1)
    monotonic = np.all(np.diff(positive_pivot.to_numpy(), axis=1) >= -1e-6, axis=1)
    negative_spread = (
        negative_pivot.max(axis=1).to_numpy() - negative_pivot.min(axis=1).to_numpy()
    )
    return {
        "by_snr": by_snr,
        "snr_monotonic_non_decreasing_fraction": float(monotonic.mean()),
        "same_background_probability_max_spread": float(negative_spread.max()),
    }


def evaluate(config: dict) -> dict[str, Any]:
    preflight_report = preflight(config)
    output = Path(config["output_dir"])
    protocol = str(config["protocol"])
    share_gain_across_snr = protocol == PROTOCOL_V2
    device = resolve_device(str(config["runtime"]["device"]))
    if device.type != "cuda":
        raise RuntimeError("G14-D formal evaluation requires the server CUDA GPU")
    results: dict[str, dict[str, Any]] = {}
    for split, manifest_value in config["pair_manifests"].items():
        dataset = CounterfactualPairDataset(
            Path(manifest_value),
            epsilon=float(config["mixing"]["epsilon"]),
            peak_limit=float(config["mixing"]["common_pair_peak_limit"]),
            share_gain_across_snr=share_gain_across_snr,
        )
        prediction_frame = dataset.rows.copy()
        split_results = {}
        for model_index, spec in enumerate(config["models"]):
            negative, positive = _predict_model(
                spec, dataset, config["runtime"], device
            )
            prediction_frame[f"{spec['name']}_negative_probability"] = negative
            prediction_frame[f"{spec['name']}_positive_probability"] = positive
            model_rows = dataset.rows.copy()
            model_rows["negative_probability"] = negative
            model_rows["positive_probability"] = positive
            split_results[spec["name"]] = _summarize_predictions(
                model_rows, config["statistics"], model_index * 1000
            )
            if share_gain_across_snr:
                maximum_spread = split_results[spec["name"]][
                    "same_background_probability_max_spread"
                ]
                tolerance = float(
                    config["statistics"]["negative_probability_invariance_tolerance"]
                )
                if maximum_spread > tolerance:
                    raise RuntimeError(
                        f"Shared-background probability invariance failed for "
                        f"{split}/{spec['name']}: {maximum_spread} > {tolerance}"
                    )

            detail_rows = []
            for fields in (
                ["target_snr_db", "background_device"],
                ["target_snr_db", "background_scene"],
                ["target_snr_db", "uav_subtype"],
            ):
                for keys, group in model_rows.groupby(fields, sort=True):
                    values = keys if isinstance(keys, tuple) else (keys,)
                    item = dict(zip(fields, values, strict=True))
                    item.update(
                        _group_metrics(
                            group,
                            threshold=float(config["statistics"]["threshold"]),
                            bootstrap_samples=int(config["statistics"]["bootstrap_samples"]),
                            bootstrap_seed=int(config["statistics"]["bootstrap_seed"])
                            + model_index * 1000,
                        )
                    )
                    item["grouping"] = "+".join(fields)
                    detail_rows.append(item)
            pd.DataFrame(detail_rows).to_csv(
                output / f"{split}_{spec['name']}_subgroups.csv", index=False
            )
        prediction_frame.to_csv(output / f"{split}_predictions.csv", index=False)
        results[split] = split_results

    baseline_name = next(
        spec["name"] for spec in config["models"] if spec["role"] == "baseline"
    )
    comparisons = {}
    for split, split_results in results.items():
        baseline = split_results[baseline_name]["by_snr"]
        comparisons[split] = {}
        for spec in config["models"]:
            if spec["role"] == "baseline":
                continue
            candidate = split_results[spec["name"]]["by_snr"]
            comparisons[split][spec["name"]] = {
                snr: {
                    "mean_lift_delta_from_g7": candidate[snr]["mean_lift"]
                    - baseline[snr]["mean_lift"],
                    "ordering_accuracy_delta_from_g7": candidate[snr][
                        "paired_ordering_accuracy"
                    ]
                    - baseline[snr]["paired_ordering_accuracy"],
                    "negative_probability_delta_from_g7": candidate[snr][
                        "mean_negative_probability"
                    ]
                    - baseline[snr]["mean_negative_probability"],
                }
                for snr in baseline
            }
    report = {
        "passed": True,
        "protocol": protocol,
        "scope": "mechanistic_development_audit_only",
        "models_remain_ineligible_for_promotion": True,
        "cross_snr_monotonicity_valid": share_gain_across_snr,
        "results": results,
        "g14_minus_g7": comparisons,
        "preflight_sha256": file_sha256(output / "preflight.json"),
        "synthetic_audio_written": False,
        "training_started": False,
        "locked_datasets_read": [],
    }
    (output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate G14-D paired counterfactuals")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g14_d_counterfactual_evaluation.yaml"),
    )
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    args = parser.parse_args()
    config = load_config(args.config)
    result = preflight(config) if args.mode == "preflight" else evaluate(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
