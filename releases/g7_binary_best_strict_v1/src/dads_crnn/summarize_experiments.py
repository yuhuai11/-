from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .metrics import binary_metrics


EXPERIMENTS = {
    "artifacts": {"dataset_size": "5000_per_class", "model_type": "crnn", "feature_type": "log_mel", "parameter_count": 1012193},
    "archive/historical_models/baselines/artifacts_15000": {"dataset_size": "15000_per_class", "model_type": "crnn", "feature_type": "log_mel", "parameter_count": 1012193},
    "archive/historical_models/baselines/artifacts_crnn_full": {"dataset_size": "full", "model_type": "crnn", "feature_type": "log_mel", "parameter_count": 1012193},
    "archive/historical_models/baselines/artifacts_resnet10_cbam_15000": {
        "dataset_size": "15000_per_class",
        "model_type": "resnet10_cbam",
        "feature_type": "mfcc",
        "parameter_count": 4898083,
    },
    "archive/historical_models/baselines/artifacts_resnet10_cbam_full": {
        "dataset_size": "full",
        "model_type": "resnet10_cbam",
        "feature_type": "mfcc",
        "parameter_count": 4898083,
    },
}

METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "false_positive_rate",
    "f1",
    "auc",
    "pr_auc",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finite(values: Iterable[Any]) -> list[float]:
    result = []
    for value in values:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            result.append(float(value))
    return result


def _aggregate(rows: list[dict[str, Any]], group_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)

    output = []
    for key, members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        result = dict(zip(group_fields, key))
        result["seed_count"] = len({row["seed"] for row in members})
        result["seeds"] = ",".join(str(seed) for seed in sorted({row["seed"] for row in members}))
        for metric in METRIC_NAMES:
            values = _finite(row.get(metric) for row in members)
            if values:
                result[f"{metric}_mean"] = statistics.fmean(values)
                result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        for metric in ("parameter_count", "elapsed_minutes", "best_epoch"):
            values = _finite(row.get(metric) for row in members)
            if values:
                result[f"{metric}_mean"] = statistics.fmean(values)
                result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(result)
    return output


def _fmt(value: Any, digits: int = 4) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _build_report(
    segment_agg: list[dict[str, Any]],
    file_agg: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
) -> str:
    lines = [
        "# 实验结果统一汇总",
        "",
        "本报告由 `python -m dads_crnn.summarize_experiments` 从各运行目录的 `metrics.json` 自动生成。",
        "标准差为跨随机种子的样本标准差；只有一个种子时显示为 0，但该组不能用于稳定性结论。",
        "",
        "## 实验覆盖情况",
        "",
        "| 数据规模 | 模型 | 特征 | 已有种子 | 完整性 |",
        "|---|---|---|---:|---|",
    ]
    for row in coverage:
        complete = "完整" if row["seed_count"] >= 3 else "不完整"
        lines.append(
            f"| {row['dataset_size']} | {row['model_type']} | {row['feature_type']} | "
            f"{row['seeds']} | {complete} |"
        )

    lines += [
        "",
        "## Test segment-level（阈值 0.50）",
        "",
        "| 数据规模 | 模型 | 种子数 | F1 mean±std | AUC mean±std | Recall mean±std | FPR mean±std | 参数量 | 训练分钟 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    chosen = [row for row in segment_agg if row["split"] == "test" and float(row["threshold"]) == 0.5]
    for row in chosen:
        lines.append(
            f"| {row['dataset_size']} | {row['model_type']} | {row['seed_count']} | "
            f"{_fmt(row.get('f1_mean'))}±{_fmt(row.get('f1_std'))} | "
            f"{_fmt(row.get('auc_mean'))}±{_fmt(row.get('auc_std'))} | "
            f"{_fmt(row.get('recall_mean'))}±{_fmt(row.get('recall_std'))} | "
            f"{_fmt(row.get('false_positive_rate_mean'))}±{_fmt(row.get('false_positive_rate_std'))} | "
            f"{_fmt(row.get('parameter_count_mean'), 0)} | {_fmt(row.get('elapsed_minutes_mean'), 1)} |"
        )

    lines += [
        "",
        "## Test file-level mean aggregation（阈值 0.50）",
        "",
        "| 数据规模 | 模型 | 种子数 | F1 mean±std | AUC mean±std | Recall mean±std | FPR mean±std |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    chosen_file = [
        row
        for row in file_agg
        if row["split"] == "test" and row["aggregation"] == "mean" and float(row["threshold"]) == 0.5
    ]
    for row in chosen_file:
        lines.append(
            f"| {row['dataset_size']} | {row['model_type']} | {row['seed_count']} | "
            f"{_fmt(row.get('f1_mean'))}±{_fmt(row.get('f1_std'))} | "
            f"{_fmt(row.get('auc_mean'))}±{_fmt(row.get('auc_std'))} | "
            f"{_fmt(row.get('recall_mean'))}±{_fmt(row.get('recall_std'))} | "
            f"{_fmt(row.get('false_positive_rate_mean'))}±{_fmt(row.get('false_positive_rate_std'))} |"
        )

    lines += [
        "",
        "## 文件说明",
        "",
        "- `segment_metrics_by_seed.csv`：每个种子、split、阈值的 segment-level 明细。",
        "- `segment_metrics_aggregate.csv`：segment-level 跨种子均值与标准差。",
        "- `file_metrics_by_seed.csv`：每个种子、split、聚合方式、阈值的 file-level 明细。",
        "- `file_metrics_aggregate.csv`：file-level 跨种子均值与标准差。",
        "- `experiment_coverage.csv`：实验组和随机种子覆盖情况。",
        "",
    ]
    return "\n".join(lines)


def summarize(project_root: Path, output_dir: Path) -> None:
    segment_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    coverage_groups: dict[tuple[str, str, str], set[int]] = defaultdict(set)

    for directory, defaults in EXPERIMENTS.items():
        for path in sorted((project_root / directory / "runs").glob("seed_*/metrics.json")):
            metrics = json.loads(path.read_text(encoding="utf-8"))
            seed = int(metrics["seed"])
            base = {
                "dataset_size": defaults["dataset_size"],
                "model_type": metrics.get("model_type", defaults["model_type"]),
                "feature_type": metrics.get("feature_type", defaults["feature_type"]),
                "seed": seed,
                "parameter_count": metrics.get("parameter_count", defaults["parameter_count"]),
                "best_epoch": metrics.get("best_epoch"),
                "elapsed_minutes": metrics.get("elapsed_minutes"),
                "source": str(path.relative_to(project_root)),
            }
            coverage_groups[(base["dataset_size"], base["model_type"], base["feature_type"])].add(seed)
            for split, key in (("val", "val_threshold_metrics"), ("test", "threshold_metrics")):
                labels_path = path.parent / f"{split}_labels.npy"
                probabilities_path = path.parent / f"{split}_probabilities.npy"
                labels = np.load(labels_path) if labels_path.exists() else None
                probabilities = np.load(probabilities_path) if probabilities_path.exists() else None
                for item in metrics.get(key, []):
                    normalized = dict(item)
                    if labels is not None and probabilities is not None:
                        # Older runs predate the extended confusion-matrix metrics.
                        # Recalculate from their saved predictions so all runs share one schema.
                        normalized = binary_metrics(
                            labels.astype(np.int64), probabilities, threshold=float(item["threshold"])
                        )
                    segment_rows.append({**base, "split": split, **normalized})
            for split, key in (("val", "val_file_metrics"), ("test", "test_file_metrics")):
                for aggregation, items in metrics.get(key, {}).items():
                    for item in items:
                        file_rows.append({**base, "split": split, "aggregation": aggregation, **item})

    if not segment_rows:
        raise FileNotFoundError("No seed_*/metrics.json files were found in known experiment directories")

    output_dir.mkdir(parents=True, exist_ok=True)
    segment_agg = _aggregate(segment_rows, ("dataset_size", "model_type", "feature_type", "split", "threshold"))
    file_agg = _aggregate(
        file_rows,
        ("dataset_size", "model_type", "feature_type", "split", "aggregation", "threshold"),
    )
    coverage = [
        {
            "dataset_size": key[0],
            "model_type": key[1],
            "feature_type": key[2],
            "seed_count": len(seeds),
            "seeds": ",".join(map(str, sorted(seeds))),
            "expected_seed_count": 3,
            "complete": len(seeds) >= 3,
        }
        for key, seeds in sorted(coverage_groups.items())
    ]
    _write_csv(output_dir / "segment_metrics_by_seed.csv", segment_rows)
    _write_csv(output_dir / "segment_metrics_aggregate.csv", segment_agg)
    _write_csv(output_dir / "file_metrics_by_seed.csv", file_rows)
    _write_csv(output_dir / "file_metrics_aggregate.csv", file_agg)
    _write_csv(output_dir / "experiment_coverage.csv", coverage)
    report_path = output_dir / "实验结果汇总.md"
    report_path.write_text(
        _build_report(segment_agg, file_agg, coverage), encoding="utf-8"
    )
    # Remove the report name used before the Chinese naming convention.
    (output_dir / "RESULTS_SUMMARY.md").unlink(missing_ok=True)
    print(f"Summarized {len(coverage_groups)} experiment groups and {len({row['source'] for row in segment_rows})} runs")
    print(f"Results written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate all completed experiment metrics")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/comparison"))
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    summarize(project_root, output_dir)


if __name__ == "__main__":
    main()
