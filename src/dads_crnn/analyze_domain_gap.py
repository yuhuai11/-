from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_dirs
from .metrics import binary_metrics


IDENTITY_COLUMNS = (
    "path",
    "sha256",
    "label",
    "source_group",
    "uav_source",
    "background_source",
    "condition",
    "ood_split",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _validate_frame(frame: pd.DataFrame, *, context: str) -> None:
    required = set(IDENTITY_COLUMNS) | {
        "calibrated_probability",
        "selected_prediction",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for {context}: {sorted(missing)}")
    labels = frame["label"].to_numpy(dtype=np.int64)
    probabilities = frame["calibrated_probability"].to_numpy(dtype=np.float64)
    predictions = frame["selected_prediction"].to_numpy(dtype=np.int64)
    if (
        frame.empty
        or not np.isin(labels, [0, 1]).all()
        or not np.isfinite(probabilities).all()
        or not ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
        or not np.isin(predictions, [0, 1]).all()
    ):
        raise ValueError(f"Invalid prediction content for {context}")


def validate_paired_identity(left: pd.DataFrame, right: pd.DataFrame, *, split: str) -> str:
    _validate_frame(left, context=f"baseline {split}")
    _validate_frame(right, context=f"candidate {split}")
    left_identity = left[list(IDENTITY_COLUMNS)].fillna("").astype(str)
    right_identity = right[list(IDENTITY_COLUMNS)].fillna("").astype(str)
    if not left_identity.equals(right_identity):
        raise ValueError(f"Prediction identity/order mismatch for {split}")
    return hashlib.sha256(
        left_identity.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _overall_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    labels = frame["label"].to_numpy(dtype=np.int64)
    probabilities = frame["calibrated_probability"].to_numpy(dtype=np.float64)
    metrics = binary_metrics(labels, probabilities, threshold)
    return {
        key: metrics[key]
        for key in (
            "f1",
            "balanced_accuracy",
            "recall",
            "specificity",
            "auc",
            "tn",
            "fp",
            "fn",
            "tp",
        )
    }


def _group_metrics(
    model: str, split: str, frame: pd.DataFrame, threshold: float
) -> list[dict[str, Any]]:
    working = frame.copy()
    working["prediction"] = (
        working["calibrated_probability"].to_numpy(dtype=np.float64) >= threshold
    ).astype(np.int64)
    specifications = (
        ("condition", None),
        ("source_group", None),
        ("uav_source", 1),
        ("background_source", 0),
    )
    rows: list[dict[str, Any]] = []
    for field, label_filter in specifications:
        selected = working if label_filter is None else working.loc[working["label"] == label_filter]
        values = selected[field].fillna("").astype(str)
        selected = selected.loc[values != ""].copy()
        if selected.empty:
            continue
        for value, group in selected.groupby(field, sort=True, dropna=False):
            labels = group["label"].to_numpy(dtype=np.int64)
            predictions = group["prediction"].to_numpy(dtype=np.int64)
            positive = labels == 1
            negative = labels == 0
            tp = int(np.sum((predictions == 1) & positive))
            tn = int(np.sum((predictions == 0) & negative))
            fp = int(np.sum((predictions == 1) & negative))
            fn = int(np.sum((predictions == 0) & positive))
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "group_field": field,
                    "group": str(value),
                    "samples": int(len(group)),
                    "positives": int(positive.sum()),
                    "negatives": int(negative.sum()),
                    "recall": tp / int(positive.sum()) if positive.any() else None,
                    "specificity": tn / int(negative.sum()) if negative.any() else None,
                    "errors": fp + fn,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "mean_probability": float(group["calibrated_probability"].mean()),
                }
            )
    return rows


def _error_concentration(
    frame: pd.DataFrame, threshold: float, *, label: int, group_field: str
) -> dict[str, Any]:
    probabilities = frame["calibrated_probability"].to_numpy(dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    mask = (frame["label"].to_numpy(dtype=np.int64) == label) & (predictions != label)
    errors = frame.loc[mask].copy()
    if errors.empty:
        return {"errors": 0, "groups": 0, "top_groups": [], "top_3_share": 0.0}
    counts = (
        errors[group_field]
        .fillna("(missing)")
        .astype(str)
        .value_counts(dropna=False)
    )
    total = int(counts.sum())
    top = [
        {"group": group, "errors": int(count), "share": float(count / total)}
        for group, count in counts.head(10).items()
    ]
    return {
        "errors": total,
        "groups": int(len(counts)),
        "top_groups": top,
        "top_3_share": float(counts.head(3).sum() / total),
    }


def _margin_summary(frame: pd.DataFrame, threshold: float) -> list[dict[str, Any]]:
    probabilities = frame["calibrated_probability"].to_numpy(dtype=np.float64)
    labels = frame["label"].to_numpy(dtype=np.int64)
    predictions = (probabilities >= threshold).astype(np.int64)
    margins = np.abs(probabilities - threshold)
    edges = (0.0, 0.02, 0.05, 0.10, 0.20, np.inf)
    output = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (margins >= lower) & (margins < upper)
        if not selected.any():
            continue
        errors = int(np.sum(predictions[selected] != labels[selected]))
        output.append(
            {
                "margin_min": lower,
                "margin_max": None if np.isinf(upper) else upper,
                "samples": int(selected.sum()),
                "errors": errors,
                "error_rate": float(errors / selected.sum()),
            }
        )
    return output


def _paired_error_cases(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    baseline_threshold: float,
    candidate_threshold: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    output = baseline.copy()
    labels = output["label"].to_numpy(dtype=np.int64)
    baseline_probability = baseline["calibrated_probability"].to_numpy(dtype=np.float64)
    candidate_probability = candidate["calibrated_probability"].to_numpy(dtype=np.float64)
    baseline_prediction = (baseline_probability >= baseline_threshold).astype(np.int64)
    candidate_prediction = (candidate_probability >= candidate_threshold).astype(np.int64)
    baseline_correct = baseline_prediction == labels
    candidate_correct = candidate_prediction == labels
    patterns = np.select(
        [
            baseline_correct & candidate_correct,
            baseline_correct & ~candidate_correct,
            ~baseline_correct & candidate_correct,
        ],
        ["both_correct", "g2_only_correct", "g6_only_correct"],
        default="both_wrong",
    )
    summary = {
        pattern: int(np.sum(patterns == pattern))
        for pattern in ("both_correct", "g2_only_correct", "g6_only_correct", "both_wrong")
    }
    output["g2_probability"] = baseline_probability
    output["g6_probability"] = candidate_probability
    output["g2_prediction"] = baseline_prediction
    output["g6_prediction"] = candidate_prediction
    output["g2_margin"] = np.abs(baseline_probability - baseline_threshold)
    output["g6_margin"] = np.abs(candidate_probability - candidate_threshold)
    output["error_pattern"] = patterns
    keep = [
        column
        for column in (
            "path",
            "filename",
            "sha256",
            "label",
            "condition",
            "source_group",
            "uav_source",
            "background_source",
            "target_snr_db",
            "achieved_snr_db",
            "g2_probability",
            "g6_probability",
            "g2_prediction",
            "g6_prediction",
            "g2_margin",
            "g6_margin",
            "error_pattern",
        )
        if column in output.columns
    ]
    errors = output.loc[patterns != "both_correct", keep].copy()
    errors = errors.sort_values(
        ["error_pattern", "label", "source_group", "path"], kind="stable"
    ).reset_index(drop=True)
    return errors, summary


def _source_isolation(
    tune: pd.DataFrame, holdout: pd.DataFrame, field: str, label: int | None
) -> dict[str, int]:
    if label is not None:
        tune = tune.loc[tune["label"] == label]
        holdout = holdout.loc[holdout["label"] == label]
    tune_sources = set(tune[field].dropna().astype(str)) - {""}
    holdout_sources = set(holdout[field].dropna().astype(str)) - {""}
    return {
        "tune_sources": len(tune_sources),
        "holdout_sources": len(holdout_sources),
        "overlap": len(tune_sources & holdout_sources),
    }


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# G2/G6 val_ood域差距与错误来源审计",
        "",
        "> 本报告只使用已解锁的val_ood开发集；未读取Unseen或Real-world。",
        "",
        "## 总体结果",
        "",
        "| Split | 模型 | F1 | Recall | Specificity | AUC | FN | FP |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("tune", "holdout"):
        for model in ("g2", "g6"):
            metrics = report["models"][model][split]["metrics"]
            lines.append(
                f"| {split} | {model.upper()} | {metrics['f1']:.5f} | "
                f"{metrics['recall']:.5f} | {metrics['specificity']:.5f} | "
                f"{metrics['auc']:.5f} | {metrics['fn']} | {metrics['fp']} |"
            )
    lines += [
        "",
        "## Holdout错误集中度",
        "",
        "| 模型 | 错误类型 | 错误数 | 来源数 | Top-3来源占比 |",
        "|---|---|---:|---:|---:|",
    ]
    for model in ("g2", "g6"):
        for key, label in (("false_negatives", "漏检/UAV来源"), ("false_positives", "误报/背景来源")):
            value = report["models"][model]["holdout"][key]
            lines.append(
                f"| {model.upper()} | {label} | {value['errors']} | {value['groups']} | "
                f"{value['top_3_share']:.1%} |"
            )
    disagreement = report["holdout_disagreement"]
    lines += [
        "",
        "## 模型分歧",
        "",
        f"- 两者都正确：{disagreement['both_correct']}；",
        f"- 只有G2正确：{disagreement['g2_only_correct']}；",
        f"- 只有G6正确：{disagreement['g6_only_correct']}；",
        f"- 两者都错误：{disagreement['both_wrong']}。",
        "",
        "## 条件级定位",
        "",
        "| 条件 | G2 Recall/Specificity | G6 Recall/Specificity | 样本数 |",
        "|---|---:|---:|---:|",
    ]
    conditions = {
        model: {row["group"]: row for row in report["diagnostics"][model]["conditions"]}
        for model in ("g2", "g6")
    }
    for condition in sorted(conditions["g2"]):
        left = conditions["g2"][condition]
        right = conditions["g6"][condition]
        metric = "specificity" if condition == "background_only" else "recall"
        lines.append(
            f"| {condition} | {left[metric]:.5f} | {right[metric]:.5f} | "
            f"{left['samples']} |"
        )
    lines += [
        "",
        "## G2最困难来源",
        "",
        "| 类型 | 来源 | 样本 | 指标 | 错误数 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in report["diagnostics"]["g2"]["worst_uav_sources"][:5]:
        lines.append(
            f"| UAV漏检 | {Path(row['group']).name} | {row['samples']} | "
            f"Recall {row['recall']:.5f} | {row['false_negatives']} |"
        )
    for row in report["diagnostics"]["g2"]["worst_background_sources"][:5]:
        lines.append(
            f"| 背景误报 | {Path(row['group']).name} | {row['samples']} | "
            f"Specificity {row['specificity']:.5f} | {row['false_positives']} |"
        )
    source_isolation = report["source_isolation"]
    lines += [
        "",
        "## 来源隔离与高置信错误",
        "",
        f"- UAV来源：tune {source_isolation['uav_source']['tune_sources']}个，holdout "
        f"{source_isolation['uav_source']['holdout_sources']}个，重叠"
        f"{source_isolation['uav_source']['overlap']}个；",
        f"- 背景来源：tune {source_isolation['background_source']['tune_sources']}个，holdout "
        f"{source_isolation['background_source']['holdout_sources']}个，重叠"
        f"{source_isolation['background_source']['overlap']}个；",
        f"- 距冻结阈值至少0.10仍判错：G2 {report['high_margin_errors']['g2']}条，"
        f"G6 {report['high_margin_errors']['g6']}条。",
        "",
        "## 结论",
        "",
        "1. DADS内部接近饱和，但val_ood仍存在显著差距，主要瓶颈不是小规模结构容量；",
        "2. G6提高AUC和Specificity但损失Recall，说明注意力改变了跨来源工作点而未解决稳定泛化；",
        "3. 下一步优先处理错误集中的真实UAV/背景来源，并用来源隔离的预训练音频表征探针验证迁移收益；",
        "4. 本报告不得用于对当前holdout事后调阈值并宣称模型晋级。",
    ]
    return "\n".join(lines) + "\n"


def analyze_domain_gap(
    baseline_tune: Path,
    baseline_holdout: Path,
    baseline_calibration: Path,
    candidate_tune: Path,
    candidate_holdout: Path,
    candidate_calibration: Path,
    output_dir: Path,
) -> dict[str, Any]:
    paths = {
        "g2": {
            "tune": baseline_tune,
            "holdout": baseline_holdout,
            "calibration": baseline_calibration,
        },
        "g6": {
            "tune": candidate_tune,
            "holdout": candidate_holdout,
            "calibration": candidate_calibration,
        },
    }
    frames: dict[str, dict[str, pd.DataFrame]] = {"g2": {}, "g6": {}}
    calibrations = {}
    for model in ("g2", "g6"):
        calibrations[model] = _load_json(paths[model]["calibration"])
        for split in ("tune", "holdout"):
            frames[model][split] = pd.read_csv(paths[model][split])
    identity = {
        split: validate_paired_identity(
            frames["g2"][split], frames["g6"][split], split=split
        )
        for split in ("tune", "holdout")
    }
    report: dict[str, Any] = {
        "scope": "val_ood_development_only",
        "locked_datasets_read": [],
        "identity_sha256": identity,
        "inputs": {
            model: {
                key: {"path": path.as_posix(), "sha256": _sha256(path)}
                for key, path in model_paths.items()
            }
            for model, model_paths in paths.items()
        },
        "models": {},
    }
    source_rows: list[dict[str, Any]] = []
    for model in ("g2", "g6"):
        threshold = float(calibrations[model]["selected_threshold"])
        report["models"][model] = {"threshold": threshold}
        for split in ("tune", "holdout"):
            frame = frames[model][split]
            split_result = {
                "samples": int(len(frame)),
                "metrics": _overall_metrics(frame, threshold),
                "margin_summary": _margin_summary(frame, threshold),
            }
            if split == "holdout":
                split_result["false_negatives"] = _error_concentration(
                    frame, threshold, label=1, group_field="uav_source"
                )
                split_result["false_positives"] = _error_concentration(
                    frame, threshold, label=0, group_field="background_source"
                )
            report["models"][model][split] = split_result
            source_rows.extend(_group_metrics(model, split, frame, threshold))
        tune_metrics = report["models"][model]["tune"]["metrics"]
        holdout_metrics = report["models"][model]["holdout"]["metrics"]
        report["models"][model]["tune_to_holdout_delta"] = {
            metric: float(holdout_metrics[metric] - tune_metrics[metric])
            for metric in ("f1", "balanced_accuracy", "recall", "specificity", "auc")
        }
    error_cases, disagreement = _paired_error_cases(
        frames["g2"]["holdout"],
        frames["g6"]["holdout"],
        report["models"]["g2"]["threshold"],
        report["models"]["g6"]["threshold"],
    )
    report["holdout_disagreement"] = disagreement
    report["source_isolation"] = {
        "uav_source": _source_isolation(
            frames["g2"]["tune"], frames["g2"]["holdout"], "uav_source", 1
        ),
        "background_source": _source_isolation(
            frames["g2"]["tune"],
            frames["g2"]["holdout"],
            "background_source",
            0,
        ),
    }
    source_frame = pd.DataFrame(source_rows)
    report["diagnostics"] = {}
    for model in ("g2", "g6"):
        selected = source_frame.loc[
            (source_frame["model"] == model) & (source_frame["split"] == "holdout")
        ]
        conditions = selected.loc[selected["group_field"] == "condition"]
        uav_sources = selected.loc[selected["group_field"] == "uav_source"].sort_values(
            ["recall", "samples"], ascending=[True, False]
        )
        background_sources = selected.loc[
            selected["group_field"] == "background_source"
        ].sort_values(["specificity", "samples"], ascending=[True, False])
        report["diagnostics"][model] = {
            "conditions": _json_records(conditions),
            "worst_uav_sources": _json_records(uav_sources.head(10)),
            "worst_background_sources": _json_records(background_sources.head(10)),
        }
    report["high_margin_errors"] = {
        model: int(
            (
                (error_cases[f"{model}_prediction"] != error_cases["label"])
                & (error_cases[f"{model}_margin"] >= 0.10)
            ).sum()
        )
        for model in ("g2", "g6")
    }
    ensure_dirs(output_dir)
    source_frame.to_csv(output_dir / "source_metrics.csv", index=False)
    error_cases.to_csv(output_dir / "error_cases.csv", index=False)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit val_ood domain gap and error sources")
    parser.add_argument(
        "--baseline-tune",
        type=Path,
        default=Path("artifacts/val_ood/predictions/tune/crnn_full_augmented_g2/seed_42/predictions.csv"),
    )
    parser.add_argument(
        "--baseline-holdout",
        type=Path,
        default=Path("artifacts/val_ood/predictions/holdout/crnn_full_augmented_g2/seed_42/predictions.csv"),
    )
    parser.add_argument(
        "--baseline-calibration",
        type=Path,
        default=Path("artifacts/val_ood/calibration/crnn_full_augmented_g2/seed_42/calibration.json"),
    )
    parser.add_argument(
        "--candidate-tune",
        type=Path,
        default=Path("artifacts/val_ood/predictions/tune/crnn_full_augmented_g6_temporal_attention/seed_42/predictions.csv"),
    )
    parser.add_argument(
        "--candidate-holdout",
        type=Path,
        default=Path("artifacts/val_ood/predictions/holdout/crnn_full_augmented_g6_temporal_attention/seed_42/predictions.csv"),
    )
    parser.add_argument(
        "--candidate-calibration",
        type=Path,
        default=Path("artifacts/val_ood/calibration/crnn_full_augmented_g6_temporal_attention/seed_42/calibration.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/domain_gap_audit")
    )
    args = parser.parse_args()
    result = analyze_domain_gap(
        args.baseline_tune,
        args.baseline_holdout,
        args.baseline_calibration,
        args.candidate_tune,
        args.candidate_holdout,
        args.candidate_calibration,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
