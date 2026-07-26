from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config
from .data_firewall import file_sha256, load_forbidden_hashes


PROTOCOL = "g15_p0_four_source_training_registry_v1"
OUTPUT_COLUMNS = [
    "training_source",
    "split",
    "label",
    "source_group",
    "recording_group",
    "cache_format",
    "cache_path",
    "cache_index",
    "audio_sha256",
    "segment_sha256",
    "hard_negative_class",
    "background_mix_eligible",
    "teacher_distillation_eligible",
    "paired_ranking_eligible",
    "original_dataset",
    "original_row",
]


def _verify(path: Path, expected_sha256: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = file_sha256(path)
    if observed != str(expected_sha256):
        raise ValueError(f"SHA256 mismatch for {path}: {observed}")
    return observed


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} lacks required columns: {missing}")


def _reject_consumed_hashes(
    frame: pd.DataFrame,
    *,
    column: str,
    forbidden_hashes: frozenset[str],
    name: str,
) -> None:
    values = set(frame[column].dropna().astype(str).str.lower())
    overlap = values.intersection(forbidden_hashes)
    if overlap:
        raise ValueError(f"{name} overlaps consumed G13 audio hashes: {len(overlap)}")


def _normalize_dads(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        frame,
        {"split", "label", "cache_path", "raw_audio_sha256", "recording_group"},
        "DADS train",
    )
    if set(frame["split"].astype(str)) != {"train"}:
        raise ValueError("DADS registry input must contain only the train split")
    result = pd.DataFrame(
        {
            "training_source": "dads_replay",
            "split": "train",
            "label": frame["label"].astype(int),
            "source_group": "dads:" + frame["recording_group"].astype(str),
            "recording_group": "dads:" + frame["recording_group"].astype(str),
            "cache_format": "individual_npy",
            "cache_path": frame["cache_path"].astype(str),
            "cache_index": -1,
            "audio_sha256": frame["raw_audio_sha256"].astype(str).str.lower(),
            "segment_sha256": "",
            "hard_negative_class": "",
            "background_mix_eligible": frame["label"].astype(int).eq(0),
            "teacher_distillation_eligible": True,
            "paired_ranking_eligible": False,
            "original_dataset": "dads_dedup_v2",
            "original_row": frame.index.astype(int),
        }
    )
    return result[OUTPUT_COLUMNS]


def _normalize_g9(frame: pd.DataFrame, classes: set[str]) -> pd.DataFrame:
    _require_columns(
        frame,
        {
            "split",
            "label",
            "cache_path",
            "sha256",
            "cache_sha256",
            "recording_group",
            "source_group",
            "hard_negative_class",
            "background_mix_eligible",
        },
        "G9 hard-negative train",
    )
    if set(frame["split"].astype(str)) != {"train"}:
        raise ValueError("G9 registry input must contain only the train split")
    selected = frame[frame["hard_negative_class"].astype(str).isin(classes)].copy()
    observed = set(selected["hard_negative_class"].astype(str))
    if observed != classes:
        raise ValueError(f"G9 requested classes are incomplete: {observed} != {classes}")
    if not selected["label"].astype(int).eq(0).all():
        raise ValueError("G9 hard negatives must all have label 0")
    eligible = selected["background_mix_eligible"].astype(str).str.lower()
    if eligible.isin({"true", "1"}).any():
        raise ValueError("G9 hard negatives must never be background-mix eligible")
    result = pd.DataFrame(
        {
            "training_source": "g9_mechanical_hard_negative",
            "split": "train",
            "label": 0,
            "source_group": "g9:" + selected["source_group"].astype(str),
            "recording_group": "g9:" + selected["recording_group"].astype(str),
            "cache_format": "individual_npy",
            "cache_path": selected["cache_path"].astype(str),
            "cache_index": -1,
            "audio_sha256": selected["sha256"].astype(str).str.lower(),
            "segment_sha256": selected["cache_sha256"].astype(str).str.lower(),
            "hard_negative_class": selected["hard_negative_class"].astype(str),
            "background_mix_eligible": False,
            "teacher_distillation_eligible": True,
            "paired_ranking_eligible": False,
            "original_dataset": "g9_hard_negative",
            "original_row": selected.index.astype(int),
        }
    )
    return result[OUTPUT_COLUMNS].reset_index(drop=True)


def _normalize_g14(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(
        frame,
        {
            "dataset",
            "split",
            "label",
            "source_group",
            "cache_path",
            "cache_index",
            "audio_sha256",
            "segment_sha256",
        },
        "G14 segment cache",
    )
    train = frame[frame["split"].astype(str).eq("train")].copy()
    if len(train) != int(frame["split"].astype(str).eq("train").sum()):
        raise AssertionError("G14 train filtering failed")
    expected_datasets = {"kielce_17_uav", "tau_urban_2022"}
    if set(train["dataset"].astype(str)) != expected_datasets:
        raise ValueError("G14 train does not contain the expected two datasets")

    def normalized(dataset: str, label: int, source: str) -> pd.DataFrame:
        selected = train[train["dataset"].astype(str).eq(dataset)].copy()
        if not selected["label"].astype(int).eq(label).all():
            raise ValueError(f"Unexpected labels in {dataset}")
        result = pd.DataFrame(
            {
                "training_source": source,
                "split": "train",
                "label": label,
                "source_group": selected["source_group"].astype(str),
                "recording_group": selected["audio_sha256"].map(
                    lambda value: f"{dataset}:{str(value).lower()}"
                ),
                "cache_format": "memmap_npy",
                "cache_path": selected["cache_path"].astype(str),
                "cache_index": selected["cache_index"].astype(int),
                "audio_sha256": selected["audio_sha256"].astype(str).str.lower(),
                "segment_sha256": selected["segment_sha256"].astype(str).str.lower(),
                "hard_negative_class": "",
                "background_mix_eligible": label == 0,
                "teacher_distillation_eligible": False,
                "paired_ranking_eligible": True,
                "original_dataset": dataset,
                "original_row": selected.index.astype(int),
            }
        )
        return result[OUTPUT_COLUMNS].reset_index(drop=True)

    return (
        normalized("kielce_17_uav", 1, "kielce_uav"),
        normalized("tau_urban_2022", 0, "tau_background"),
    )


def build_registry(
    dads: pd.DataFrame,
    g9: pd.DataFrame,
    g14: pd.DataFrame,
    *,
    g9_classes: set[str],
    forbidden_hashes: frozenset[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, int]]:
    sources = {}
    sources["dads_replay"] = _normalize_dads(dads)
    sources["g9_mechanical_hard_negative"] = _normalize_g9(g9, g9_classes)
    kielce, tau = _normalize_g14(g14)
    sources["kielce_uav"] = kielce
    sources["tau_background"] = tau

    for name, frame in sources.items():
        _reject_consumed_hashes(
            frame,
            column="audio_sha256",
            forbidden_hashes=forbidden_hashes,
            name=name,
        )
    overlaps: dict[str, int] = {}
    names = list(sources)
    for index, left in enumerate(names):
        left_hashes = set(sources[left]["audio_sha256"])
        for right in names[index + 1 :]:
            overlap = left_hashes.intersection(sources[right]["audio_sha256"])
            key = f"{left}__{right}"
            overlaps[key] = len(overlap)
            if overlap:
                raise ValueError(f"Cross-source audio overlap for {key}: {len(overlap)}")

    combined = pd.concat([sources[name] for name in names], ignore_index=True)
    if combined[["training_source", "original_row"]].duplicated().any():
        raise ValueError("Duplicate source-row identity in G15 registry")
    return combined, sources, overlaps


def prepare(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported G15 training-registry protocol")
    inputs = config["inputs"]
    verified = {
        name: _verify(Path(spec["path"]), str(spec["sha256"]))
        for name, spec in inputs.items()
        if name not in {"g7_checkpoint"}
    }
    g7_spec = inputs["g7_checkpoint"]
    verified["g7_checkpoint"] = _verify(
        Path(g7_spec["path"]), str(g7_spec["sha256"])
    )

    for audit_name in ("dads_audit", "g9_audit", "g14_segment_audit"):
        audit = json.loads(Path(inputs[audit_name]["path"]).read_text(encoding="utf-8"))
        if audit.get("passed") is not True:
            raise ValueError(f"Input audit did not pass: {audit_name}")

    forbidden_hashes = load_forbidden_hashes(Path(inputs["g13_hash_registry"]["path"]))
    if len(forbidden_hashes) != int(config["firewall"]["expected_g13_hashes"]):
        raise ValueError("Consumed G13 hash-registry size changed")

    dads = pd.read_csv(inputs["dads_train_manifest"]["path"], low_memory=False)
    g9 = pd.read_csv(inputs["g9_train_manifest"]["path"], low_memory=False)
    g14 = pd.read_csv(inputs["g14_segment_manifest"]["path"], low_memory=False)
    g14_metadata_split_counts = {
        str(key): int(value)
        for key, value in g14["split"].value_counts(dropna=False).items()
    }
    combined, sources, overlaps = build_registry(
        dads,
        g9,
        g14,
        g9_classes=set(config["g9_mechanical_classes"]),
        forbidden_hashes=forbidden_hashes,
    )

    output_dir = Path(config["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite G15 P0 output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent))
    try:
        manifests = {}
        for name, frame in {**sources, "combined_train": combined}.items():
            path = stage / f"{name}.csv"
            frame.to_csv(path, index=False, lineterminator="\n")
            manifests[name] = {
                "path": (output_dir / path.name).as_posix(),
                "sha256": file_sha256(path),
                "rows": int(len(frame)),
            }
        report = {
            "passed": True,
            "protocol": PROTOCOL,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": verified,
            "g7_checkpoint_locked": True,
            "g7_checkpoint_modified": False,
            "source_rows": {name: int(len(frame)) for name, frame in sources.items()},
            "source_label_counts": {
                name: {
                    str(key): int(value)
                    for key, value in frame["label"].value_counts().sort_index().items()
                }
                for name, frame in sources.items()
            },
            "g9_mechanical_classes": sorted(config["g9_mechanical_classes"]),
            "cross_source_audio_hash_overlap": overlaps,
            "consumed_g13_hashes_checked": len(forbidden_hashes),
            "consumed_g13_audio_overlap": 0,
            "training_split_only": True,
            "g14_source_manifest_metadata_splits": g14_metadata_split_counts,
            "g14_nontraining_metadata_rows_filtered": int(
                len(g14) - g14["split"].astype(str).eq("train").sum()
            ),
            "nontraining_audio_payload_read": False,
            "forbidden_manifests_read": [],
            "audio_payload_read": False,
            "model_inference_started": False,
            "training_started": False,
            "outputs": manifests,
        }
        audit_path = stage / "audit.json"
        audit_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        stage.replace(output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the G15 four-source registry")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g15_training_registry.yaml"),
    )
    args = parser.parse_args()
    report = prepare(args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
