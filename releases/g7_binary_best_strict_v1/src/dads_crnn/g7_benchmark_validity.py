from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PROTOCOL = "g7_reusable_benchmark_validity_v1"


def _normalized_set(values: pd.Series) -> set[str]:
    return set(values.dropna().astype(str).str.strip().str.lower()) - {""}


def exact_overlap(left: pd.Series, right: pd.Series) -> int:
    return len(_normalized_set(left) & _normalized_set(right))


def label_purity(frame: pd.DataFrame, group_column: str) -> dict[str, Any]:
    grouped = frame.groupby(group_column, dropna=False)["label"]
    label_counts = grouped.nunique()
    sizes = grouped.size()
    positive_rates = grouped.mean()
    pure = label_counts.eq(1)
    return {
        "group_column": group_column,
        "groups": int(label_counts.size),
        "pure_label_groups": int(pure.sum()),
        "mixed_label_groups": int((~pure).sum()),
        "pure_label_group_fraction": float(pure.mean()) if len(pure) else 0.0,
        "rows_in_pure_label_groups": int(sizes[pure].sum()),
        "group_positive_rates": {
            str(group): float(rate) for group, rate in positive_rates.items()
        },
    }


def majority_lookup(
    development: pd.DataFrame,
    benchmark: pd.DataFrame,
    group_column: str,
) -> dict[str, Any]:
    mapping = development.groupby(group_column)["label"].mean().ge(0.5).astype(int)
    predicted = benchmark[group_column].map(mapping)
    covered = predicted.notna()
    return {
        "group_column": group_column,
        "development_groups": int(mapping.size),
        "benchmark_groups": int(benchmark[group_column].nunique()),
        "covered_rows": int(covered.sum()),
        "coverage": float(covered.mean()),
        "accuracy_on_covered_rows": (
            float(
                np.mean(
                    predicted[covered].to_numpy(dtype=np.int64)
                    == benchmark.loc[covered, "label"].to_numpy(dtype=np.int64)
                )
            )
            if bool(covered.any())
            else None
        ),
    }


def build_validity_report(
    development: pd.DataFrame,
    calibration: pd.DataFrame,
    kielce_tau: pd.DataFrame,
    g13: pd.DataFrame,
    idmt: pd.DataFrame,
    esc50: pd.DataFrame,
) -> dict[str, Any]:
    development = development[
        development["dataset_role"].isin(["train", "model_validation"])
    ].copy()
    g13_normalized = pd.DataFrame(
        {
            "label": g13["label"].astype(int),
            "source_group": g13["source_group"].astype(str),
            "dataset_origin": "g13_ddl_aerosonic",
        }
    )
    idmt_normalized = pd.DataFrame(
        {
            "label": 0,
            "source_group": idmt["session_id"].astype(str),
            "dataset_origin": "idmt_traffic",
        }
    )
    esc_normalized = pd.DataFrame(
        {
            "label": 0,
            "source_group": esc50["source_group"].astype(str),
            "dataset_origin": "esc50_fold5_guard",
        }
    )
    exact = {
        "kielce_tau_vs_development": {
            column: exact_overlap(development[column], kielce_tau[column])
            for column in (
                "audio_sha256",
                "segment_sha256",
                "recording_group",
                "source_group",
            )
        },
        "kielce_tau_vs_calibration": {
            column: exact_overlap(calibration[column], kielce_tau[column])
            for column in (
                "audio_sha256",
                "segment_sha256",
                "recording_group",
                "source_group",
            )
        },
        "g13_raw_audio_vs_development": exact_overlap(
            development["audio_sha256"], g13["sha256"]
        ),
        "idmt_raw_audio_vs_development": exact_overlap(
            development["audio_sha256"], idmt["recording_sha256"]
        ),
        "esc50_audio_vs_development": exact_overlap(
            development["audio_sha256"], esc50["sha256"]
        ),
        "esc50_cache_vs_development": exact_overlap(
            development["segment_sha256"], esc50["cache_sha256"]
        ),
    }
    exact_values = []
    for value in exact.values():
        exact_values.extend(value.values() if isinstance(value, dict) else [value])

    origin_lookup = majority_lookup(development, kielce_tau, "dataset_origin")
    corpus_confound = bool(
        origin_lookup["coverage"] == 1.0
        and origin_lookup["accuracy_on_covered_rows"] == 1.0
    )
    return {
        "protocol": PROTOCOL,
        "benchmark_status": "consumed_reusable_development_benchmark",
        "independent_final_claim_allowed": False,
        "hard_exact_metadata_overlap_detected": bool(any(exact_values)),
        "exact_overlap_checks": exact,
        "corpus_label_confounding": {
            "kielce_tau_dataset_origin_lookup": origin_lookup,
            "kielce_tau_corpus_label_shortcut_detected": corpus_confound,
            "development_origin_label_purity": label_purity(
                development, "dataset_origin"
            ),
            "kielce_tau_origin_label_purity": label_purity(
                kielce_tau, "dataset_origin"
            ),
            "g13_source_group_label_purity": label_purity(
                g13_normalized, "source_group"
            ),
        },
        "stress_set_scope": {
            "idmt": {
                "labels": sorted(idmt_normalized["label"].unique().tolist()),
                "supports_only_negative_fpr": True,
                "source_groups": int(idmt_normalized["source_group"].nunique()),
            },
            "esc50": {
                "labels": sorted(esc_normalized["label"].unique().tolist()),
                "supports_only_negative_fpr": True,
                "source_groups": int(esc_normalized["source_group"].nunique()),
                "same_corpus_fold_as_training_hard_negatives": True,
            },
        },
        "interpretation": {
            "primary_evidence_unit": "source_group_or_recording_not_segment",
            "kielce_tau": "same-corpus/session-isolated reusable benchmark with corpus-label confounding",
            "g13": "consumed cross-corpus stress benchmark with source-group labels that are pure",
            "idmt": "consumed pure-negative traffic stress benchmark",
            "esc50": "consumed pure-negative same-corpus fold guard",
        },
        "overall_risk": "high_corpus_confounding_without_exact_identity_leakage",
    }
