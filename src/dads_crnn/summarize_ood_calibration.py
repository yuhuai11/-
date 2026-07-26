from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "false_positive_rate",
    "f1",
    "auc",
    "pr_auc",
    "nll",
    "brier",
    "ece",
)


def _fmt(value: float) -> str:
    return f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate OOD calibration results")
    parser.add_argument("--root", type=Path, default=Path("artifacts/val_ood"))
    args = parser.parse_args()
    rows = []
    subgroup_rows = []
    for path in sorted((args.root / "calibration").glob("*/seed_*/calibration.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        for split_key in ("tune_metrics", "holdout_metrics"):
            split = split_key.removesuffix("_metrics")
            for scenario, metrics in result[split_key].items():
                rows.append(
                    {
                        "experiment": result["experiment"],
                        "model_type": result["model_type"],
                        "feature_type": result["feature_type"],
                        "seed": result["seed"],
                        "split": split,
                        "scenario": scenario,
                        "temperature": result["temperature"],
                        "selected_threshold": result["selected_threshold"],
                        "scenario_threshold": metrics["threshold"],
                        "constraints_feasible_on_tune": result["constraints_feasible_on_tune"],
                        **metrics,
                    }
                )
        for subgroup in result["holdout_subgroups"]:
            subgroup_rows.append(
                {
                    "experiment": result["experiment"],
                    "seed": result["seed"],
                    "selected_threshold": result["selected_threshold"],
                    **subgroup,
                }
            )
    if not rows:
        raise FileNotFoundError(f"No calibration results found below {args.root}")
    metrics_dir = args.root / "metrics"
    figures_dir = args.root / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(metrics_dir / "calibration_metrics_by_seed.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(metrics_dir / "subgroup_metrics.csv", index=False)

    groups = ["experiment", "model_type", "feature_type", "split", "scenario"]
    aggregate_rows = []
    for key, group in by_seed.groupby(groups, sort=True):
        row = dict(zip(groups, key))
        row["seed_count"] = int(group["seed"].nunique())
        row["feasible_seed_count"] = int(group["constraints_feasible_on_tune"].sum())
        for metric in ("temperature", "selected_threshold", "scenario_threshold", *METRICS):
            values = [float(value) for value in group[metric].dropna()]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(metrics_dir / "calibration_metrics_aggregate.csv", index=False)

    holdout = aggregate[aggregate["split"] == "holdout"].copy()
    lines = [
        "# OOD校准评估报告",
        "",
        "温度和阈值只由val_ood tune产生；holdout不参与拟合或阈值选择。",
        "",
        "| 实验 | 场景 | 可行种子 | 阈值 | F1 mean±std | Recall mean±std | Specificity mean±std | ECE | NLL |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in holdout.sort_values(["experiment", "scenario"]).iterrows():
        lines.append(
            f"| {row['experiment']} | {row['scenario']} | {int(row['feasible_seed_count'])}/{int(row['seed_count'])} | "
            f"{_fmt(row['scenario_threshold_mean'])} | "
            f"{_fmt(row['f1_mean'])}±{_fmt(row['f1_std'])} | "
            f"{_fmt(row['recall_mean'])}±{_fmt(row['recall_std'])} | "
            f"{_fmt(row['specificity_mean'])}±{_fmt(row['specificity_std'])} | "
            f"{_fmt(row['ece_mean'])} | {_fmt(row['nll_mean'])} |"
        )
    lines += [
        "",
        "判定门槛：holdout Recall ≥ 0.80、Specificity ≥ 0.90，并且校准后的ECE/NLL下降。",
        "",
    ]
    (args.root / "OOD校准评估报告.md").write_text("\n".join(lines), encoding="utf-8")

    selected = holdout[holdout["scenario"] == "temperature_selected"].sort_values("experiment")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(selected))
    ax.bar([value - 0.18 for value in x], selected["recall_mean"], width=0.36, label="Recall")
    ax.bar([value + 0.18 for value in x], selected["specificity_mean"], width=0.36, label="Specificity")
    ax.axhline(0.8, color="C0", linestyle="--", alpha=0.6)
    ax.axhline(0.9, color="C1", linestyle="--", alpha=0.6)
    ax.set_xticks(list(x), selected["experiment"])
    ax.set_ylim(0, 1.05)
    ax.set_title("OOD holdout after tune-only calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "recall_specificity_calibrated.png", dpi=180)
    plt.close(fig)
    print(f"Summarized {len(by_seed)} scenario rows from {by_seed[['experiment', 'seed']].drop_duplicates().shape[0]} checkpoints")


if __name__ == "__main__":
    main()
