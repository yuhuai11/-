from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "specificity", "false_positive_rate", "f1", "auc", "pr_auc")


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate external metrics across seeds")
    parser.add_argument("--root", type=Path, default=Path("artifacts/external_evaluation"))
    args = parser.parse_args()

    rows = []
    subgroup_rows = []
    for path in sorted((args.root / "predictions").glob("*/*/seed_*/metrics.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        for metrics in result["threshold_metrics"]:
            row = {
                "dataset": result["dataset"],
                "experiment": result["experiment"],
                "training_scale": result["training_scale"],
                "model_type": result["model_type"],
                "feature_type": result["feature_type"],
                "seed": result["seed"],
                **{key: value for key, value in metrics.items() if key != "bootstrap_95_ci"},
            }
            internal = result.get("internal_dads_test_at_primary_threshold")
            if internal and metrics["threshold"] == result["primary_threshold"]:
                for metric in METRICS:
                    if metric in internal and metric in metrics:
                        row[f"internal_{metric}"] = internal[metric]
                        row[f"generalization_gap_{metric}"] = metrics[metric] - internal[metric]
                        row[f"retention_{metric}"] = metrics[metric] / internal[metric] if internal[metric] else None
            rows.append(row)
        for subgroup in result["subgroup_metrics"]:
            subgroup_rows.append(
                {
                    "dataset": result["dataset"],
                    "experiment": result["experiment"],
                    "training_scale": result["training_scale"],
                    "model_type": result["model_type"],
                    "feature_type": result["feature_type"],
                    "seed": result["seed"],
                    **subgroup,
                }
            )
    if not rows:
        raise FileNotFoundError(f"No external metrics found below {args.root / 'predictions'}")

    metrics_dir = args.root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(metrics_dir / "metrics_by_seed.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(metrics_dir / "subgroup_metrics_by_seed.csv", index=False)

    group_fields = ["dataset", "experiment", "training_scale", "model_type", "feature_type", "threshold"]
    numeric_fields = [column for column in by_seed.columns if column not in group_fields + ["seed"]]
    aggregate_rows = []
    for key, group in by_seed.groupby(group_fields, sort=True, dropna=False):
        row = dict(zip(group_fields, key))
        row["seed_count"] = int(group["seed"].nunique())
        row["seeds"] = ",".join(str(value) for value in sorted(group["seed"].unique()))
        for field in numeric_fields:
            values = [float(value) for value in group[field].dropna()]
            if values:
                row[f"{field}_mean"] = statistics.fmean(values)
                row[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(metrics_dir / "metrics_aggregate.csv", index=False)

    import matplotlib.pyplot as plt

    figures_dir = args.root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    primary_plot = aggregate[aggregate["threshold"] == 0.5].copy()
    primary_plot["series"] = primary_plot["dataset"] + ":" + primary_plot["experiment"]
    primary_plot = primary_plot.sort_values(["dataset", "experiment"])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(primary_plot["series"], primary_plot["f1_mean"], yerr=primary_plot["f1_std"], capsize=3)
    ax.set_ylabel("External F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("External generalization at threshold 0.50")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(figures_dir / "external_f1_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for (dataset, experiment), group in aggregate.groupby(["dataset", "experiment"], sort=True):
        group = group.sort_values("threshold")
        ax.plot(group["threshold"], group["f1_mean"], marker="o", label=f"{dataset}:{experiment}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 mean across seeds")
    ax.set_ylim(0, 1.05)
    ax.set_title("External threshold sensitivity")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "threshold_sensitivity.png", dpi=180)
    plt.close(fig)

    primary = aggregate[aggregate["threshold"] == 0.5]
    lines = [
        "# 外部泛化测试报告",
        "",
        "本报告由冻结的现有检查点生成。外部数据不参与训练、阈值选择或概率校准。",
        "",
        "## 主结果（阈值 0.50）",
        "",
        "| 外部数据 | 实验 | 模型 | 种子数 | F1 mean±std | Recall mean±std | Specificity mean±std | AUC mean±std | F1保持率 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in primary.sort_values(["dataset", "experiment"]).iterrows():
        lines.append(
            f"| {row['dataset']} | {row['experiment']} | {row['model_type']} | {int(row['seed_count'])} | "
            f"{_fmt(row.get('f1_mean'))}±{_fmt(row.get('f1_std'))} | "
            f"{_fmt(row.get('recall_mean'))}±{_fmt(row.get('recall_std'))} | "
            f"{_fmt(row.get('specificity_mean'))}±{_fmt(row.get('specificity_std'))} | "
            f"{_fmt(row.get('auc_mean'))}±{_fmt(row.get('auc_std'))} | "
            f"{_fmt(row.get('retention_f1_mean'))} |"
        )
    lines += [
        "",
        "## 解释约束",
        "",
        "- 以三随机种子均值和标准差作为模型结论，单种子结果仅用于排错。",
        "- Real-world 重点同时观察 Recall 与 Specificity，避免以全预测为 UAV 换取高 Recall。",
        "- 外部结果只能用于最终比较；后续模型选择必须使用独立的 OOD validation 数据。",
        "",
    ]
    (args.root / "外部泛化测试报告.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Summarized {len(by_seed)} threshold rows from {by_seed[['dataset', 'experiment', 'seed']].drop_duplicates().shape[0]} evaluations")


if __name__ == "__main__":
    main()
