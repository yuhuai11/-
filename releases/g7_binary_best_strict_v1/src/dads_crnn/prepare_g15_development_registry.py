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
from .prepare_g15_training_registry import OUTPUT_COLUMNS


PROTOCOL = "g15_p2_development_registry_v1"


def _verify(path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = file_sha256(path)
    if observed != str(expected):
        raise ValueError(f"SHA256 mismatch for {path}")
    return observed


def normalize_dads_validation(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"split", "label", "cache_path", "raw_audio_sha256", "recording_group"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"DADS validation lacks columns: {sorted(missing)}")
    if set(frame["split"].astype(str)) != {"val"}:
        raise ValueError("DADS development input must contain only split=val")
    result = pd.DataFrame(
        {
            "training_source": "dads_validation",
            "split": "validation",
            "label": frame["label"].astype(int),
            "source_group": "dads:" + frame["recording_group"].astype(str),
            "recording_group": "dads:" + frame["recording_group"].astype(str),
            "cache_format": "individual_npy",
            "cache_path": frame["cache_path"].astype(str),
            "cache_index": -1,
            "audio_sha256": frame["raw_audio_sha256"].astype(str).str.lower(),
            "segment_sha256": "",
            "hard_negative_class": "",
            "background_mix_eligible": False,
            "teacher_distillation_eligible": False,
            "paired_ranking_eligible": False,
            "original_dataset": "dads_dedup_v2",
            "original_row": frame.index.astype(int),
        }
    )
    return result[OUTPUT_COLUMNS]


def normalize_g14_tune(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    required = {
        "dataset",
        "split",
        "label",
        "source_group",
        "cache_path",
        "cache_index",
        "audio_sha256",
        "segment_sha256",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"G14 segment manifest lacks columns: {sorted(missing)}")
    tune = frame[frame["split"].astype(str).eq("tune")].copy()

    def convert(dataset: str, label: int, name: str) -> pd.DataFrame:
        selected = tune[tune["dataset"].astype(str).eq(dataset)].copy()
        if selected.empty or not selected["label"].astype(int).eq(label).all():
            raise ValueError(f"Invalid G14 tune source: {dataset}")
        result = pd.DataFrame(
            {
                "training_source": name,
                "split": "validation",
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
                "background_mix_eligible": False,
                "teacher_distillation_eligible": False,
                "paired_ranking_eligible": False,
                "original_dataset": dataset,
                "original_row": selected.index.astype(int),
            }
        )
        return result[OUTPUT_COLUMNS].reset_index(drop=True)

    return (
        convert("kielce_17_uav", 1, "kielce_tune"),
        convert("tau_urban_2022", 0, "tau_tune"),
        int(len(frame) - len(tune)),
    )


def validate_isolation(
    development: dict[str, pd.DataFrame],
    *,
    training_hashes: set[str],
    consumed_hashes: frozenset[str],
) -> dict[str, int]:
    result = {}
    names = list(development)
    for name, frame in development.items():
        hashes = set(frame["audio_sha256"].astype(str).str.lower())
        training_overlap = hashes.intersection(training_hashes)
        consumed_overlap = hashes.intersection(consumed_hashes)
        result[f"{name}_vs_training"] = len(training_overlap)
        result[f"{name}_vs_consumed_g13"] = len(consumed_overlap)
        if training_overlap or consumed_overlap:
            raise ValueError(f"Development isolation failed for {name}")
    for index, left in enumerate(names):
        left_hashes = set(development[left]["audio_sha256"])
        for right in names[index + 1 :]:
            overlap = left_hashes.intersection(development[right]["audio_sha256"])
            result[f"{left}__{right}"] = len(overlap)
            if overlap:
                raise ValueError(f"Development-source overlap: {left}/{right}")
    return result


def prepare(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported G15 development-registry protocol")
    inputs = config["inputs"]
    verified = {
        name: _verify(Path(spec["path"]), str(spec["sha256"]))
        for name, spec in inputs.items()
    }
    p0 = json.loads(Path(inputs["p0_audit"]["path"]).read_text(encoding="utf-8"))
    p1 = json.loads(Path(inputs["p1_preflight"]["path"]).read_text(encoding="utf-8"))
    if p0.get("passed") is not True or p1.get("passed") is not True:
        raise ValueError("G15 P0/P1 prerequisite did not pass")

    training = pd.read_csv(inputs["combined_training_registry"]["path"], low_memory=False)
    dads = pd.read_csv(inputs["dads_val_manifest"]["path"], low_memory=False)
    g14 = pd.read_csv(inputs["g14_segment_manifest"]["path"], low_memory=False)
    development = {"dads_validation": normalize_dads_validation(dads)}
    kielce, tau, filtered = normalize_g14_tune(g14)
    development["kielce_tune"] = kielce
    development["tau_tune"] = tau
    consumed = load_forbidden_hashes(Path(inputs["g13_hash_registry"]["path"]))
    isolation = validate_isolation(
        development,
        training_hashes=set(training["audio_sha256"].astype(str).str.lower()),
        consumed_hashes=consumed,
    )

    output_dir = Path(config["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite G15 P2 output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent))
    try:
        outputs = {}
        for name, frame in development.items():
            path = stage / f"{name}.csv"
            frame.to_csv(path, index=False, lineterminator="\n")
            outputs[name] = {
                "path": (output_dir / path.name).as_posix(),
                "sha256": file_sha256(path),
                "rows": int(len(frame)),
            }
        report = {
            "passed": True,
            "protocol": PROTOCOL,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": verified,
            "source_rows": {
                name: int(len(frame)) for name, frame in development.items()
            },
            "source_label_counts": {
                name: {
                    str(key): int(value)
                    for key, value in frame["label"].value_counts().sort_index().items()
                }
                for name, frame in development.items()
            },
            "isolation": isolation,
            "g14_non_tune_metadata_rows_filtered": filtered,
            "selection_allowed_splits": ["dads_val", "g14_tune"],
            "dads_test_read": False,
            "g9_guard_read": False,
            "g14_dev_holdout_audio_read": False,
            "g13_audio_read": False,
            "audio_payload_read": False,
            "model_inference_started": False,
            "training_started": False,
            "outputs": outputs,
        }
        (stage / "audit.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        stage.replace(output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build G15 development registry")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g15_development_registry.yaml"),
    )
    args = parser.parse_args()
    report = prepare(args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
