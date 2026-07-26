from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "false_positive_rate": float(1.0 - specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_prob)
        metrics["pr_auc"] = float(auc(pr_recall, pr_precision))
    except ValueError:
        metrics["auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def file_level_metrics(
    rows: pd.DataFrame,
    y_prob: np.ndarray,
    thresholds: list[float],
    *,
    aggregation: str,
) -> list[dict[str, float]]:
    if aggregation not in {"mean", "max"}:
        raise ValueError(f"Unsupported file-level aggregation: {aggregation}")

    identity_candidates = (
        ("parquet_file", "row_group", "row_in_group"),
        ("archive_path", "archive_member"),
        ("audio_sha256",),
        ("source_path",),
    )
    identity_columns = next(
        (
            list(candidate)
            for candidate in identity_candidates
            if all(column in rows.columns for column in candidate)
        ),
        None,
    )
    if identity_columns is None:
        raise ValueError(
            "Cannot compute file-level metrics: no supported original-file identity "
            "columns are present"
        )
    if len(rows) != len(y_prob):
        raise ValueError(
            f"File-level probability count mismatch: rows={len(rows)}, probabilities={len(y_prob)}"
        )

    eval_rows = rows[[*identity_columns, "label"]].copy()
    eval_rows["probability"] = y_prob
    label_counts = eval_rows.groupby(identity_columns, sort=False, dropna=False)["label"].nunique()
    if bool((label_counts > 1).any()):
        raise ValueError("An original audio file has inconsistent labels")
    grouped = eval_rows.groupby(
        [*identity_columns, "label"], sort=False, dropna=False
    )
    if aggregation == "mean":
        file_probs = grouped["probability"].mean()
    else:
        file_probs = grouped["probability"].max()

    file_df = file_probs.reset_index()
    y_true = file_df["label"].to_numpy(dtype=np.int64)
    probs = file_df["probability"].to_numpy(dtype=np.float64)
    metrics = []
    for threshold in thresholds:
        item = binary_metrics(y_true, probs, threshold)
        item["aggregation"] = aggregation
        item["files"] = int(len(file_df))
        metrics.append(item)
    return metrics
