from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audio_identity import spectral_fingerprint
from .data_firewall import file_sha256


PROTOCOL = "dads_cross_split_spectral_and_waveform_near_duplicate_audit_v2"
SPLITS = ("train", "val", "test")
REQUIRED_COLUMNS = {
    "split",
    "label",
    "source_id",
    "source_path",
    "cache_path",
    "cache_index",
}
NUMBERED_WAV = re.compile(r"^(?P<prefix>.*?)(?P<number>\d+)\.wav$", re.IGNORECASE)


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {str(q): None for q in (0.5, 0.9, 0.95, 0.99, 1.0)}
    return {
        str(q): float(np.quantile(values, q))
        for q in (0.5, 0.9, 0.95, 0.99, 1.0)
    }


def _source_table(manifest: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    if not set(manifest["split"].astype(str)).issubset(SPLITS):
        raise ValueError("Manifest contains unsupported splits")
    conflicts = manifest.groupby("source_id").agg(
        split_count=("split", "nunique"),
        label_count=("label", "nunique"),
        path_count=("source_path", "nunique"),
        cache_count=("cache_path", "nunique"),
    )
    if bool((conflicts != 1).any().any()):
        raise ValueError("A source has conflicting split, label, path or cache metadata")
    return (
        manifest.sort_values(["source_id", "cache_index"], kind="stable")
        .groupby("source_id", as_index=False, sort=True)
        .agg(
            split=("split", "first"),
            label=("label", "first"),
            source_path=("source_path", "first"),
            cache_path=("cache_path", "first"),
            representative_cache_index=("cache_index", "first"),
            windows=("cache_index", "size"),
        )
    )


def _extract_features(
    sources: pd.DataFrame,
    root: Path,
    output: Path,
) -> np.ndarray:
    if output.exists():
        with np.load(output, allow_pickle=False) as archive:
            ids = archive["source_id"].astype(str)
            features = archive["features"].astype(np.float32)
        if np.array_equal(ids, sources["source_id"].astype(str).to_numpy()):
            return features
        raise ValueError("Existing fingerprint cache does not match current source order")

    cache_paths = sources["cache_path"].astype(str).unique()
    if len(cache_paths) != 1:
        raise ValueError("Near-duplicate audit currently requires one shared audio cache")
    cache_path = Path(cache_paths[0])
    if not cache_path.is_absolute():
        cache_path = root / cache_path
    cache = np.load(cache_path.resolve(strict=True), mmap_mode="r")
    indices = sources["representative_cache_index"].to_numpy(dtype=np.int64)
    features = np.empty((len(sources), 48), dtype=np.float32)
    for index, cache_index in enumerate(indices):
        features[index] = spectral_fingerprint(cache[int(cache_index)], 16_000)
        if (index + 1) % 10_000 == 0:
            print(f"fingerprints {index + 1}/{len(sources)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        source_id=sources["source_id"].astype(str).to_numpy(dtype=str),
        features=features,
    )
    return features


def _signatures(features: np.ndarray, planes: np.ndarray) -> np.ndarray:
    bits = (features @ planes.T) >= 0.0
    weights = np.left_shift(np.uint32(1), np.arange(bits.shape[1], dtype=np.uint32))
    return (bits.astype(np.uint32) * weights).sum(axis=1, dtype=np.uint32)


def lsh_cross_split_nearest(
    features: np.ndarray,
    reference_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    tables: int = 8,
    bits: int = 18,
    seed: int = 20260804,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate cross-split cosine nearest neighbours using random-hyperplane LSH."""
    if bits <= 0 or bits > 31 or tables <= 0:
        raise ValueError("LSH requires tables > 0 and 1 <= bits <= 31")
    reference_indices = np.asarray(reference_indices, dtype=np.int64)
    query_indices = np.asarray(query_indices, dtype=np.int64)
    best_similarity = np.full(len(query_indices), -np.inf, dtype=np.float32)
    best_reference = np.full(len(query_indices), -1, dtype=np.int64)
    candidate_comparisons = np.zeros(len(query_indices), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for _ in range(tables):
        planes = rng.standard_normal((bits, features.shape[1])).astype(np.float32)
        reference_signatures = _signatures(features[reference_indices], planes)
        query_signatures = _signatures(features[query_indices], planes)
        order = np.argsort(reference_signatures, kind="stable")
        sorted_signatures = reference_signatures[order]
        for query_position, signature in enumerate(query_signatures):
            left = int(np.searchsorted(sorted_signatures, signature, side="left"))
            right = int(np.searchsorted(sorted_signatures, signature, side="right"))
            if left == right:
                continue
            candidates = reference_indices[order[left:right]]
            scores = features[candidates] @ features[query_indices[query_position]]
            candidate_comparisons[query_position] += len(candidates)
            local = int(np.argmax(scores))
            score = float(scores[local])
            if score > float(best_similarity[query_position]):
                best_similarity[query_position] = score
                best_reference[query_position] = int(candidates[local])
    best_similarity[~np.isfinite(best_similarity)] = np.nan
    return best_similarity, best_reference, candidate_comparisons


def _numbered_adjacency(sources: pd.DataFrame, features: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    parsed = []
    for index, row in sources.iterrows():
        match = NUMBERED_WAV.fullmatch(Path(str(row["source_path"])).name)
        if match is None:
            continue
        parsed.append(
            {
                "index": int(index),
                "prefix": match.group("prefix"),
                "number": int(match.group("number")),
                "split": str(row["split"]),
                "label": int(row["label"]),
                "source_id": str(row["source_id"]),
                "source_path": str(row["source_path"]),
            }
        )
    frame = pd.DataFrame(parsed)
    if frame.empty:
        return pd.DataFrame()
    for (_, _,), group in frame.groupby(["prefix", "label"], sort=True):
        ordered = group.sort_values("number", kind="stable").reset_index(drop=True)
        for left, right in zip(
            ordered.iloc[:-1].to_dict("records"),
            ordered.iloc[1:].to_dict("records"),
            strict=True,
        ):
            if int(right["number"]) - int(left["number"]) != 1:
                continue
            similarity = float(features[int(left["index"])] @ features[int(right["index"])])
            rows.append(
                {
                    "prefix": left["prefix"],
                    "label": int(left["label"]),
                    "left_number": int(left["number"]),
                    "left_split": left["split"],
                    "right_split": right["split"],
                    "cross_split": left["split"] != right["split"],
                    "pair_role": "__".join(sorted((left["split"], right["split"]))),
                    "spectral_cosine_similarity": similarity,
                    "left_source_id": left["source_id"],
                    "right_source_id": right["source_id"],
                    "left_source_path": left["source_path"],
                    "right_source_path": right["source_path"],
                }
            )
    return pd.DataFrame(rows)


def _aligned_waveform_cosines(
    root: Path,
    sources: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    batch_size: int = 512,
) -> np.ndarray:
    """Confirm retrieved spectral candidates with aligned representative waveforms."""
    cache_paths = sources["cache_path"].astype(str).unique()
    if len(cache_paths) != 1:
        raise ValueError("Waveform confirmation requires one shared audio cache")
    cache_path = Path(cache_paths[0])
    if not cache_path.is_absolute():
        cache_path = root / cache_path
    cache = np.load(cache_path.resolve(strict=True), mmap_mode="r")
    lookup = dict(
        zip(
            sources["source_id"].astype(str),
            sources["representative_cache_index"].astype(int),
            strict=True,
        )
    )
    query = np.asarray(
        [lookup.get(str(value), -1) for value in candidates["query_source_id"]],
        dtype=np.int64,
    )
    reference = np.asarray(
        [lookup.get(str(value), -1) for value in candidates["reference_source_id"]],
        dtype=np.int64,
    )
    output = np.full(len(candidates), np.nan, dtype=np.float32)
    for start in range(0, len(candidates), batch_size):
        end = min(len(candidates), start + batch_size)
        valid = (query[start:end] >= 0) & (reference[start:end] >= 0)
        if not bool(valid.any()):
            continue
        left = np.asarray(cache[query[start:end][valid]], dtype=np.float32)
        right = np.asarray(cache[reference[start:end][valid]], dtype=np.float32)
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        values = np.einsum("ij,ij->i", left, right) / np.maximum(
            denominator, 1.0e-12
        )
        output[np.flatnonzero(valid) + start] = values.astype(np.float32)
    return output


def audit(
    root: Path,
    manifest_path: Path,
    split_audit_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = (root / manifest_path).resolve(strict=True)
    split_audit_path = (root / split_audit_path).resolve(strict=True)
    output_dir = (root / output_dir).resolve()
    split_audit = json.loads(split_audit_path.read_text(encoding="utf-8"))
    audited_manifest = split_audit.get("output", {}).get("manifest", {})
    if split_audit.get("passed") is not True:
        raise ValueError("DADS split audit is not passed")
    if str(audited_manifest.get("sha256")) != file_sha256(manifest_path):
        raise ValueError("Manifest no longer matches its split audit")

    manifest = pd.read_csv(manifest_path, low_memory=False)
    sources = _source_table(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "source_representative_spectral_fingerprints.npz"
    features = _extract_features(sources, root, features_path)

    candidate_frames = []
    retrieval_reports = {}
    train_indices = np.flatnonzero(sources["split"].astype(str).eq("train").to_numpy())
    for query_split in ("val", "test"):
        query_indices = np.flatnonzero(
            sources["split"].astype(str).eq(query_split).to_numpy()
        )
        similarities, references, comparisons = lsh_cross_split_nearest(
            features, train_indices, query_indices
        )
        valid = references >= 0
        query = sources.iloc[query_indices].reset_index(drop=True)
        reference = sources.iloc[np.maximum(references, 0)].reset_index(drop=True)
        candidates = pd.DataFrame(
            {
                "query_split": query_split,
                "query_source_id": query["source_id"],
                "query_source_path": query["source_path"],
                "query_label": query["label"],
                "reference_split": "train",
                "reference_source_id": reference["source_id"],
                "reference_source_path": reference["source_path"],
                "reference_label": reference["label"],
                "spectral_cosine_similarity": similarities,
                "candidate_comparisons": comparisons,
                "review_status": "candidate_only_unreviewed",
                "automatic_exclusion": False,
            }
        )
        candidates.loc[~valid, ["reference_source_id", "reference_source_path"]] = ""
        candidate_frames.append(candidates)
        finite = similarities[np.isfinite(similarities)]
        retrieval_reports[query_split] = {
            "queries": int(len(query_indices)),
            "queries_with_candidates": int(np.isfinite(similarities).sum()),
            "top1_similarity_quantiles": _quantiles(finite),
            "candidate_counts_at_or_above": {
                str(threshold): int(np.sum(finite >= threshold))
                for threshold in (0.95, 0.98, 0.99, 0.995, 0.999)
            },
            "same_label_candidates_at_or_above_0.99": int(
                (
                    valid
                    & (similarities >= 0.99)
                    & (
                        query["label"].to_numpy()
                        == reference["label"].to_numpy()
                    )
                ).sum()
            ),
            "lsh_candidate_comparisons": int(comparisons.sum()),
        }
    candidates = pd.concat(candidate_frames, ignore_index=True)
    candidates["aligned_waveform_cosine"] = _aligned_waveform_cosines(
        root, sources, candidates
    )
    candidates.to_csv(output_dir / "train_cross_split_candidates.csv", index=False)

    waveform_reports = {}
    for query_split in ("val", "test"):
        values = candidates.loc[
            candidates["query_split"].eq(query_split), "aligned_waveform_cosine"
        ].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        waveform_reports[query_split] = {
            "confirmed_candidate_pairs": int(len(finite)),
            "aligned_waveform_cosine_quantiles": _quantiles(finite),
            "counts_at_or_above": {
                str(threshold): int(np.sum(finite >= threshold))
                for threshold in (0.9, 0.95, 0.98, 0.99, 0.995, 0.999)
            },
        }

    adjacency = _numbered_adjacency(sources, features)
    adjacency.to_csv(output_dir / "numbered_adjacent_pairs.csv", index=False)
    adjacency_reports = {}
    if not adjacency.empty:
        for name, rows in {
            "all": adjacency,
            "within_split": adjacency.loc[~adjacency["cross_split"]],
            "cross_split": adjacency.loc[adjacency["cross_split"]],
            "train_test": adjacency.loc[adjacency["pair_role"].eq("test__train")],
        }.items():
            values = rows["spectral_cosine_similarity"].to_numpy(dtype=np.float64)
            adjacency_reports[name] = {
                "pairs": int(len(rows)),
                "similarity_quantiles": _quantiles(values),
                "counts_at_or_above": {
                    str(threshold): int(np.sum(values >= threshold))
                    for threshold in (0.95, 0.98, 0.99, 0.995, 0.999)
                },
            }

    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "candidate_screen_not_confirmed_duplicate_removal",
        "sources": {
            "total": int(len(sources)),
            "by_split": {
                split: int(sources["split"].astype(str).eq(split).sum())
                for split in SPLITS
            },
            "representative_window_policy": "lowest_cache_index_per_source",
        },
        "global_lsh_retrieval": retrieval_reports,
        "aligned_waveform_confirmation": waveform_reports,
        "numbered_filename_adjacency": adjacency_reports,
        "interpretation": {
            "automatic_duplicate_threshold": None,
            "automatic_exclusions": 0,
            "reason": (
                "Spectral similarity and numbered adjacency are candidate evidence; "
                "DADS has no authoritative session provenance or calibrated duplicate threshold."
            ),
            "exact_identity_audit_still_authoritative": True,
            "high_similarity_requires_waveform_or_provenance_review": True,
        },
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
            "split_audit": {
                "path": str(split_audit_path),
                "sha256": file_sha256(split_audit_path),
            },
            "fingerprints": {
                "path": str(features_path),
                "sha256": file_sha256(features_path),
            },
        },
    }
    (output_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DADS cross-split acoustic near duplicates")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/g7_leakage_fixed_v2/data/manifest.csv"),
    )
    parser.add_argument(
        "--split-audit",
        type=Path,
        default=Path("artifacts/g7_leakage_fixed_v2/data/audit.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/g7_strict_retrain_v1/near_duplicate_audit"),
    )
    args = parser.parse_args()
    audit(args.root, args.manifest, args.split_audit, args.output_dir)


if __name__ == "__main__":
    main()
