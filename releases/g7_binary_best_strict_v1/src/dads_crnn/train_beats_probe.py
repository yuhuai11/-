from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ensure_dirs
from .metrics import binary_metrics
from .prepare_beats_probe import sha256


def _load_source_embeddings(directory: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    embeddings = np.load(directory / "embeddings.npy", mmap_mode="r")
    labels = np.load(directory / "labels.npy").astype(np.int64)
    metadata = pd.read_csv(directory / "metadata.csv")
    if (
        embeddings.ndim != 2
        or embeddings.shape[1] != 768
        or len(metadata) != len(labels)
        or embeddings.shape[0] != len(labels)
        or "source_path" not in metadata.columns
        or not np.array_equal(labels, metadata["label"].to_numpy(dtype=np.int64))
    ):
        raise ValueError(f"Invalid embedding bundle: {directory}")
    groups = metadata.groupby("source_path", sort=True).indices
    output = np.empty((len(groups), embeddings.shape[1]), dtype=np.float32)
    source_labels = np.empty(len(groups), dtype=np.int64)
    source_rows = []
    for position, (source, indices) in enumerate(groups.items()):
        indices = np.asarray(indices, dtype=np.int64)
        unique_labels = np.unique(labels[indices])
        if unique_labels.size != 1:
            raise ValueError(f"A source has inconsistent labels: {source}")
        output[position] = np.asarray(embeddings[indices], dtype=np.float32).mean(axis=0)
        source_labels[position] = int(unique_labels[0])
        source_rows.append(
            {"source_path": source, "label": int(unique_labels[0]), "segments": int(len(indices))}
        )
    return output, source_labels, pd.DataFrame(source_rows)


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    return binary_metrics(labels, probabilities, 0.5)


def train_probe(
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    train_x, train_y, train_sources = _load_source_embeddings(train_dir)
    val_x, val_y, val_sources = _load_source_embeddings(val_dir)
    test_x, test_y, test_sources = _load_source_embeddings(test_dir)
    source_sets = {
        "train": set(train_sources["source_path"]),
        "val": set(val_sources["source_path"]),
        "test": set(test_sources["source_path"]),
    }
    overlaps = {
        "train_val": len(source_sets["train"] & source_sets["val"]),
        "train_test": len(source_sets["train"] & source_sets["test"]),
        "val_test": len(source_sets["val"] & source_sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Source leakage in embedding bundles: {overlaps}")
    candidates = []
    fitted = {}
    for c_value in (0.01, 0.1, 1.0):
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(C=c_value, max_iter=2000, random_state=42),
                ),
            ]
        )
        model.fit(train_x, train_y)
        val_probability = model.predict_proba(val_x)[:, 1]
        metrics = _metrics(val_y, val_probability)
        candidates.append({"c": c_value, "val_metrics": metrics})
        fitted[c_value] = model
    selected = max(
        candidates,
        key=lambda row: (row["val_metrics"]["auc"], row["val_metrics"]["f1"], -row["c"]),
    )
    model = fitted[float(selected["c"])]
    results = {}
    for split, features, labels, sources in (
        ("train", train_x, train_y, train_sources),
        ("val", val_x, val_y, val_sources),
        ("test", test_x, test_y, test_sources),
    ):
        probabilities = model.predict_proba(features)[:, 1]
        results[split] = {
            "sources": int(len(labels)),
            "metrics": _metrics(labels, probabilities),
        }
        ensure_dirs(output_dir / split)
        np.save(output_dir / split / "probabilities.npy", probabilities)
        np.save(output_dir / split / "labels.npy", labels)
        sources.assign(probability=probabilities).to_csv(
            output_dir / split / "predictions.csv", index=False
        )
    ensure_dirs(output_dir)
    model_path = output_dir / "linear_probe.joblib"
    joblib.dump(model, model_path)
    report = {
        "selection_metric": ["val_auc", "val_f1", "smaller_c"],
        "candidates": candidates,
        "selected_c": selected["c"],
        "source_overlap": overlaps,
        "splits": results,
        "model": {"path": model_path.as_posix(), "sha256": sha256(model_path)},
        "input_audits": {
            split: {
                "path": (directory / "audit.json").as_posix(),
                "sha256": sha256(directory / "audit.json"),
            }
            for split, directory in (("train", train_dir), ("val", val_dir), ("test", test_dir))
        },
        "locked_datasets_read": [],
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a frozen-BEATs linear source probe")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/p1_beats_probe/linear")
    )
    args = parser.parse_args()
    result = train_probe(args.train_dir, args.val_dir, args.test_dir, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
