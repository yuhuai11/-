from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import load_config
from .data_firewall import file_sha256
from .sampling import ClassDomainQuotaBatchSampler


PROTOCOL = "g7_r9_domain_quota_preflight_v1"
BASELINE_CONFIG = Path("configs/g7_r7_freq_mixstyle.yaml")
MANIFEST = Path("artifacts/g7_r8_urban_negatives/data/manifest.csv")
MANIFEST_AUDIT = Path("artifacts/g7_r8_urban_negatives/data/audit.json")
CONFIGS = {
    "control_00": Path("configs/g7_r9_domain_quota_control.yaml"),
    "tau_05": Path("configs/g7_r9_domain_quota_tau05.yaml"),
    "tau_10": Path("configs/g7_r9_domain_quota_tau10.yaml"),
}
EXPECTED_FRACTIONS = {
    "control_00": {"negative:dads": 1.00, "negative:tau_urban_train": 0.00},
    "tau_05": {"negative:dads": 0.95, "negative:tau_urban_train": 0.05},
    "tau_10": {"negative:dads": 0.90, "negative:tau_urban_train": 0.10},
}
BASE_POS_WEIGHT = 155685 / 136368


def _assert_training_controls(baseline: dict, candidate: dict) -> None:
    if candidate["model"] != baseline["model"]:
        raise ValueError("R9 model must remain identical to R7 Frequency MixStyle")
    if candidate["features"] != baseline["features"]:
        raise ValueError("R9 features must remain identical to R7")
    for key, value in baseline["train"].items():
        if key == "pos_weight":
            continue
        if key not in candidate["train"] or candidate["train"][key] != value:
            raise ValueError(f"R9 changed frozen R7 training field: {key}")
    if abs(float(candidate["train"]["pos_weight"]) - BASE_POS_WEIGHT) > 1e-12:
        raise ValueError("R9 must preserve the effective R7/DADS positive-class weight")


def audit(output: Path = Path("artifacts/g7_r9_domain_quota/preflight/audit.json")) -> dict:
    baseline = load_config(BASELINE_CONFIG)
    manifest_audit = json.loads(MANIFEST_AUDIT.read_text(encoding="utf-8"))
    if manifest_audit.get("passed") is not True:
        raise ValueError("The reused R8 manifest audit has not passed")
    if file_sha256(MANIFEST) != manifest_audit["output"]["manifest"]["sha256"]:
        raise ValueError("The R8 manifest no longer matches its bound audit")
    if manifest_audit.get("locked_datasets_read") != []:
        raise ValueError("The reused manifest audit read a locked dataset")

    rows = pd.read_csv(MANIFEST, low_memory=False)
    train = rows[rows["split"].eq("train")].reset_index(drop=True)
    observed = {
        (int(label), str(origin)): int(len(group))
        for (label, origin), group in train.groupby(["label", "dataset_origin"])
    }
    expected_counts = {
        (0, "dads_halfsec"): 155685,
        (1, "dads_halfsec"): 136368,
        (0, "tau_urban_2022"): 43320,
    }
    if observed != expected_counts:
        raise ValueError(f"Unexpected R9 training population: {observed}")
    tau = train[train["dataset_origin"].eq("tau_urban_2022")]
    if not bool(tau["label"].eq(0).all()) or bool(tau["background_mix_eligible"].any()):
        raise ValueError("TAU rows must remain direct negative examples only")
    if set(tau["domain_bucket"].astype(str)) != {"negative:tau_urban_train"}:
        raise ValueError("Unexpected TAU sampling domain")

    arms = {}
    configs = {name: load_config(path) for name, path in CONFIGS.items()}
    frozen_reference = None
    for name, config in configs.items():
        _assert_training_controls(baseline, config)
        sampling = config["train"]["sampling"]
        if sampling["type"] != "class_domain_quota_batch":
            raise ValueError(f"{name} does not use the R9 quota sampler")
        fractions = {str(key): float(value) for key, value in sampling["negative_domain_fractions"].items()}
        if fractions != EXPECTED_FRACTIONS[name]:
            raise ValueError(f"Unexpected negative-domain dose for {name}: {fractions}")
        if int(sampling["samples_per_epoch"]) != 292096:
            raise ValueError("R9 must preserve 2,282 batches per epoch")
        if int(sampling["positive_per_batch"]) != 60:
            raise ValueError("R9 must use 60 positive and 68 negative rows per batch")

        frozen = json.loads(json.dumps(config))
        frozen.pop("protocol")
        frozen.pop("output_dir")
        frozen["train"]["sampling"]["negative_domain_fractions"] = "DOSE"
        if frozen_reference is None:
            frozen_reference = frozen
        elif frozen != frozen_reference:
            raise ValueError("R9 arms differ outside protocol, output and TAU dose")

        sampler = ClassDomainQuotaBatchSampler(
            train,
            domain_column=str(sampling["domain_column"]),
            batch_size=int(config["train"]["batch_size"]),
            positive_per_batch=int(sampling["positive_per_batch"]),
            num_batches=int(sampling["samples_per_epoch"]) // int(config["train"]["batch_size"]),
            positive_domain=str(sampling["positive_domain"]),
            negative_domain_fractions=fractions,
            seed=42,
        )
        arms[name] = {
            "config": str(CONFIGS[name]),
            "config_sha256": file_sha256(CONFIGS[name]),
            "batches_per_epoch": len(sampler),
            "samples_per_epoch": len(sampler) * sampler.batch_size,
            "positive_draws_per_epoch": len(sampler) * sampler.positive_per_batch,
            "negative_draws_per_epoch": len(sampler) * sampler.negative_per_batch,
            "negative_domain_draws": sampler.negative_domain_draws,
            "expected_label_probabilities": sampler.expected_label_probabilities,
        }

    result = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experimental_question": (
            "Can controlled 5% or 10% TAU exposure recover IDMT false positives "
            "without sacrificing the G13 gains of Frequency MixStyle?"
        ),
        "single_variable": "negative-domain dose under a fixed class/domain quota sampler",
        "inputs": {
            "r7_baseline_config": {
                "path": str(BASELINE_CONFIG),
                "sha256": file_sha256(BASELINE_CONFIG),
            },
            "manifest": {"path": str(MANIFEST), "sha256": file_sha256(MANIFEST)},
            "manifest_audit": {
                "path": str(MANIFEST_AUDIT),
                "sha256": file_sha256(MANIFEST_AUDIT),
            },
        },
        "training_population": {
            f"label_{label}:{origin}": count
            for (label, origin), count in observed.items()
        },
        "arms": arms,
        "controls": {
            "r7_model_and_augmentations_preserved": True,
            "fixed_optimizer_steps_per_epoch": 2282,
            "fixed_batch_composition": {"positive": 60, "negative": 68},
            "fixed_r7_pos_weight": BASE_POS_WEIGHT,
            "validation_and_internal_test_are_dads_only": True,
            "tau_rows_are_never_background_mix_sources": True,
            "external_benchmarks_are_not_training_or_early_stopping_inputs": True,
        },
        "benchmark_policy": {
            "status": "consumed_reusable_development_benchmark",
            "allowed": "predeclared seed42 screening and paired reporting",
            "forbidden": "claiming a new independent final test",
        },
        "locked_datasets_read": [],
        "formal_training_started": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    print(json.dumps(audit(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
