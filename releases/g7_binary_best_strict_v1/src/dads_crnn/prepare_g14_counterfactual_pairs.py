from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ensure_dirs, load_config
from .panns import file_sha256
from .prepare_beats_probe import reject_locked_path


PROTOCOL = "g14_d_paired_counterfactual_manifest_v1"


def _stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _balanced_select(
    rows: pd.DataFrame,
    count: int,
    *,
    strata: list[str],
    seed: int,
) -> pd.DataFrame:
    if count > len(rows):
        raise ValueError(f"Requested {count} rows from only {len(rows)} available")
    work = rows.copy()
    for column in strata:
        if column not in work:
            raise ValueError(f"Missing selection stratum: {column}")
        work[column] = work[column].fillna("<NA>").astype(str)
    work["_stratum"] = work[strata].agg("|".join, axis=1)
    work["_rank"] = [
        _stable_rank(seed, str(value)) for value in work["segment_sha256"].astype(str)
    ]
    queues = {
        str(name): deque(group.sort_values("_rank").index.tolist())
        for name, group in work.groupby("_stratum", sort=True)
    }
    order = sorted(queues, key=lambda value: _stable_rank(seed, value))
    selected: list[int] = []
    while len(selected) < count:
        progressed = False
        for name in order:
            if queues[name]:
                selected.append(queues[name].popleft())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise RuntimeError("Balanced selection exhausted before reaching its target")
    return rows.loc[selected].reset_index(drop=True)


def _load_source(config: dict) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = Path(config["source_manifest"])
    audit_path = Path(config["source_audit"])
    reject_locked_path(manifest_path)
    reject_locked_path(audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True:
        raise ValueError("G14 segment-cache audit did not pass")
    manifest = pd.read_csv(manifest_path, low_memory=False)
    required = {
        "split",
        "label",
        "dataset",
        "source_group",
        "cache_path",
        "cache_index",
        "segment_sha256",
        "audio_sha256",
        "device",
        "scene_label",
        "subtype",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"G14 segment manifest lacks columns: {missing}")
    if manifest["segment_sha256"].duplicated().any():
        raise ValueError("G14 source manifest contains duplicate segment hashes")
    return manifest, audit


def build_pairs(config: dict) -> dict[str, Any]:
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported G14-D counterfactual-pair protocol")
    source, source_audit = _load_source(config)
    mixing = config["mixing"]
    snrs = [float(value) for value in mixing["target_snr_db"]]
    if len(snrs) != len(set(snrs)) or not snrs:
        raise ValueError("G14-D requires distinct SNR conditions")
    if mixing.get("require_same_background_within_pair") is not True:
        raise ValueError("Counterfactual pairs must share the exact background")
    if mixing.get("positive_rms_matched_to_negative") is not True:
        raise ValueError("Counterfactual pairs must control output RMS")

    output = Path(config["output_dir"])
    ensure_dirs(output)
    reports = {}
    split_selected_audio: dict[str, set[str]] = {}
    all_rows = []
    for split, split_config in config["splits"].items():
        split_rows = source.loc[source["split"].astype(str).eq(str(split))].copy()
        background = split_rows.loc[split_rows["label"].astype(int).eq(0)].copy()
        uav = split_rows.loc[split_rows["label"].astype(int).eq(1)].copy()
        count = int(split_config["background_samples"])
        seed = int(split_config["seed"])
        selected_background = _balanced_select(
            background,
            count,
            strata=[str(value) for value in config["selection"]["background_strata"]],
            seed=seed,
        )
        selected_uav = _balanced_select(
            uav,
            count,
            strata=[str(value) for value in config["selection"]["uav_strata"]],
            seed=seed + 1,
        )
        if selected_background["segment_sha256"].duplicated().any():
            raise RuntimeError("A background segment was selected more than once")
        if selected_uav["segment_sha256"].duplicated().any():
            raise RuntimeError("A UAV segment was selected more than once")

        pair_rows = []
        for pair_index, (background_row, uav_row) in enumerate(
            zip(
                selected_background.itertuples(index=False),
                selected_uav.itertuples(index=False),
                strict=True,
            )
        ):
            base_identity = (
                f"{split}:{background_row.segment_sha256}:{uav_row.segment_sha256}"
            )
            base_pair_id = hashlib.sha256(base_identity.encode("utf-8")).hexdigest()
            for snr in snrs:
                pair_rows.append(
                    {
                        "split": str(split),
                        "pair_index": pair_index,
                        "base_pair_id": base_pair_id,
                        "target_snr_db": snr,
                        "negative_label": 0,
                        "positive_label": 1,
                        "background_dataset": str(background_row.dataset),
                        "background_cache_path": str(background_row.cache_path),
                        "background_cache_index": int(background_row.cache_index),
                        "background_segment_sha256": str(background_row.segment_sha256),
                        "background_audio_sha256": str(background_row.audio_sha256),
                        "background_source_group": str(background_row.source_group),
                        "background_device": str(background_row.device),
                        "background_scene": str(background_row.scene_label),
                        "uav_dataset": str(uav_row.dataset),
                        "uav_cache_path": str(uav_row.cache_path),
                        "uav_cache_index": int(uav_row.cache_index),
                        "uav_segment_sha256": str(uav_row.segment_sha256),
                        "uav_audio_sha256": str(uav_row.audio_sha256),
                        "uav_source_group": str(uav_row.source_group),
                        "uav_subtype": str(uav_row.subtype),
                        "rms_control": "positive_matched_to_negative",
                        "shared_peak_limit": float(mixing["common_pair_peak_limit"]),
                    }
                )
        pairs = pd.DataFrame(pair_rows)
        expected_rows = count * len(snrs)
        if len(pairs) != expected_rows:
            raise RuntimeError("Counterfactual pair row count changed")
        if not pairs.groupby("base_pair_id").size().eq(len(snrs)).all():
            raise RuntimeError("A base pair does not contain every SNR condition")
        manifest_path = output / f"{split}_pairs.csv"
        pairs.to_csv(manifest_path, index=False)
        selected_audio = set(selected_background["audio_sha256"].astype(str)) | set(
            selected_uav["audio_sha256"].astype(str)
        )
        split_selected_audio[str(split)] = selected_audio
        reports[str(split)] = {
            "manifest": manifest_path.as_posix(),
            "manifest_sha256": file_sha256(manifest_path),
            "base_pairs": count,
            "rows": expected_rows,
            "snr_db": snrs,
            "background_segments_unique": int(
                pairs["background_segment_sha256"].nunique()
            ),
            "uav_segments_unique": int(pairs["uav_segment_sha256"].nunique()),
            "background_source_groups": int(pairs["background_source_group"].nunique()),
            "background_devices": sorted(pairs["background_device"].unique().tolist()),
            "background_scenes": sorted(pairs["background_scene"].unique().tolist()),
            "uav_source_groups": int(pairs["uav_source_group"].nunique()),
            "uav_subtypes": sorted(pairs["uav_subtype"].unique().tolist()),
        }
        all_rows.append(pairs)

    split_names = list(split_selected_audio)
    overlaps = {}
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlaps[f"{left}__{right}"] = len(
                split_selected_audio[left] & split_selected_audio[right]
            )
    if any(overlaps.values()):
        raise ValueError(f"G14-D selected raw audio crosses splits: {overlaps}")

    combined = pd.concat(all_rows, ignore_index=True)
    combined_path = output / "all_pairs.csv"
    combined.to_csv(combined_path, index=False)
    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "source_manifest": {
            "path": config["source_manifest"],
            "sha256": file_sha256(Path(config["source_manifest"])),
            "source_audit_protocol": source_audit.get("protocol"),
        },
        "splits": reports,
        "combined_manifest": {
            "path": combined_path.as_posix(),
            "sha256": file_sha256(combined_path),
            "rows": int(len(combined)),
        },
        "cross_split_selected_audio_overlap": overlaps,
        "pair_controls": {
            "same_background_for_negative_and_positive": True,
            "positive_rms_matched_to_negative": True,
            "same_common_peak_gain_within_pair": True,
            "background_dataset_and_device_held_constant_within_pair": True,
        },
        "audio_payload_read": False,
        "synthetic_audio_written": False,
        "model_inference_started": False,
        "training_started": False,
        "locked_datasets_read": [],
    }
    (output / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build G14-D paired counterfactual manifests")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g14_d_counterfactual_pairs.yaml")
    )
    args = parser.parse_args()
    print(json.dumps(build_pairs(load_config(args.config)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
