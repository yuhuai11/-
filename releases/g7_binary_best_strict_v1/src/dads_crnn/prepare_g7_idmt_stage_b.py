from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from .audio import ensure_sample_rate, peak_normalize, to_fixed_length
from .audio_identity import (
    decode_wav_channel_variants,
    quantized_pcm_sha256,
    spectral_fingerprint,
)
from .config import load_config
from .data_firewall import file_sha256


PROTOCOL = "g7_idmt_stage_b_identity_v1"
FORBIDDEN_TEXT = ("hohenwarte", "final_holdout")


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _validate_input(path: Path, expected_sha256: str, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(
            f"{name} SHA256 mismatch: expected={expected_sha256} observed={observed}"
        )


def _load(root: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"Stage B protocol must be {PROTOCOL}")
    paths = {
        name: _resolve(root, settings["path"])
        for name, settings in config["inputs"].items()
    }
    for name, settings in config["inputs"].items():
        _validate_input(paths[name], str(settings["sha256"]), name)
    for name, value in config["outputs"].items():
        paths[name] = _resolve(root, value)
    if config.get("state") != {
        "model_inference_started": False,
        "training_started": False,
    }:
        raise ValueError("Stage B must remain prediction-free and training-free")
    return config, paths


def _reject_locked(row: pd.Series | dict[str, Any], *, context: str) -> None:
    text = "|".join(str(value).lower() for value in dict(row).values())
    matched = [token for token in FORBIDDEN_TEXT if token in text]
    if matched:
        raise ValueError(f"Locked IDMT audio is forbidden in {context}: {matched}")


def _native_pcm_sha(audio: np.ndarray, sample_rate: int) -> str:
    values = np.asarray(audio, dtype="<f4")
    header = f"native_float32_mono|rate={sample_rate}|samples={values.size}|".encode(
        "ascii"
    )
    return hashlib.sha256(header + values.tobytes()).hexdigest()


def _event_group(row: pd.Series) -> str:
    return (
        f"{row['date_time']}|{row['location_id']}|{row['sample_position']}|"
        f"{row['traffic_content']}"
    )


def _write_raw(path: Path, wav_bytes: bytes, expected_sha256: str) -> None:
    if path.exists():
        if file_sha256(path) != expected_sha256:
            raise ValueError(f"Existing extracted WAV hash mismatch: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(wav_bytes)
    if file_sha256(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Extracted WAV hash mismatch: {path}")
    os.replace(temporary, path)


def prepare_idmt(config_path: Path, root: Path) -> dict[str, Any]:
    config, paths = _load(root, config_path)
    contract = config["contract"]
    intake = pd.read_csv(paths["intake_manifest"])
    expected_total = sum(int(value) for value in contract["expected_files"].values())
    if len(intake) != expected_total:
        raise ValueError(f"Unexpected IDMT intake rows: {len(intake)}")
    counts = intake["intended_role"].value_counts().to_dict()
    if counts != {
        role: int(value) for role, value in contract["expected_files"].items()
    }:
        raise ValueError(f"Unexpected IDMT role counts: {counts}")

    allowed_roles = set(str(value) for value in contract["allowed_roles"])
    selected = intake.loc[intake["intended_role"].isin(allowed_roles)].copy()
    if len(selected) != int(contract["expected_files"]["calibration"]) + int(
        contract["expected_files"]["development_test"]
    ):
        raise ValueError("IDMT active role selection changed")
    for row in selected.to_dict("records"):
        _reject_locked(row, context="IDMT active intake")

    identity_rows: list[dict[str, Any]] = []
    segment_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fingerprint_rows: list[np.ndarray] = []
    fingerprint_ids: list[str] = []
    fingerprint_roles: list[str] = []
    opened_members = 0
    locked_members_read = 0
    audio_root = paths["audio_root"]
    target_rate = int(contract["target_sample_rate"])
    target_samples = int(
        round(target_rate * float(contract["target_clip_seconds"]))
    )

    with zipfile.ZipFile(paths["archive"]) as archive:
        names = set(archive.namelist())
        for row in tqdm(
            selected.itertuples(index=False),
            total=len(selected),
            desc="Preparing active IDMT audio",
            unit="file",
        ):
            values = row._asdict()
            _reject_locked(values, context="IDMT ZIP member open")
            member = str(values["archive_member"])
            if member not in names:
                raise FileNotFoundError(f"IDMT archive member is missing: {member}")
            wav_bytes = archive.read(member)
            opened_members += 1
            if any(token in member.lower() for token in FORBIDDEN_TEXT):
                locked_members_read += 1
                raise AssertionError(f"Locked member was opened: {member}")
            raw_sha = hashlib.sha256(wav_bytes).hexdigest()
            variants, sample_rate = decode_wav_channel_variants(wav_bytes)
            primary_name = "mono" if "mono" in variants else "mono_mean"
            primary = variants[primary_name]
            role = str(values["intended_role"])
            output_path = (
                audio_root
                / role
                / str(values["location_id"])
                / Path(member).name
            )
            _write_raw(output_path, wav_bytes, raw_sha)
            event_group = (
                f"{values['date_time']}|{values['location_id']}|"
                f"{values['sample_position']}|{values['traffic_content']}"
            )

            for variant_name, audio in variants.items():
                identity_rows.append(
                    {
                        "dataset": "IDMT-TRAFFIC",
                        "recording_id": values["recording_id"],
                        "role": role,
                        "location": values["location_id"],
                        "session_id": values["session_id"],
                        "event_group": event_group,
                        "microphone": values["microphone_id"],
                        "variant": variant_name,
                        "sample_rate": sample_rate,
                        "samples": int(audio.size),
                        "raw_wav_sha256": raw_sha,
                        "native_pcm_sha256": _native_pcm_sha(audio, sample_rate),
                        "canonical_pcm_sha256": quantized_pcm_sha256(
                            audio, sample_rate, normalize_gain=False
                        ),
                        "gain_invariant_pcm_sha256": quantized_pcm_sha256(
                            audio, sample_rate, normalize_gain=True
                        ),
                        "path": output_path.resolve().as_posix(),
                    }
                )

            canonical = ensure_sample_rate(primary, sample_rate, target_rate)
            canonical_samples_before_fix = int(canonical.size)
            canonical = to_fixed_length(
                canonical,
                target_samples * int(contract["expected_segments_per_file"]),
                random_crop=False,
            )
            fingerprint_rows.append(
                spectral_fingerprint(primary, sample_rate)
            )
            fingerprint_ids.append(str(values["recording_id"]))
            fingerprint_roles.append(role)
            for segment_index in range(int(contract["expected_segments_per_file"])):
                start = segment_index * target_samples
                waveform = peak_normalize(canonical[start : start + target_samples])
                model_pcm_sha = hashlib.sha256(
                    waveform.astype("<f4", copy=False).tobytes()
                ).hexdigest()
                segment_rows[role].append(
                    {
                        "dataset": "IDMT-TRAFFIC",
                        "path": output_path.resolve().as_posix(),
                        "filename": output_path.name,
                        "label": 0,
                        "source_group": values["session_id"],
                        "condition": values["traffic_content"],
                        "role": role,
                        "location": values["location_id"],
                        "session_id": values["session_id"],
                        "event_group": event_group,
                        "microphone": values["microphone_id"],
                        "channels": values["channels"],
                        "traffic_content": values["traffic_content"],
                        "weather": values["weather"],
                        "vehicle": values["vehicle"],
                        "recording_id": values["recording_id"],
                        "recording_sha256": raw_sha,
                        "canonical_samples_before_fix": canonical_samples_before_fix,
                        "segment_index": segment_index,
                        "start_sample_16k": start,
                        "end_sample_16k": start + target_samples,
                        "model_pcm_sha256": model_pcm_sha,
                    }
                )

    if locked_members_read != 0:
        raise AssertionError("Locked IDMT audio member count must remain zero")
    identities = pd.DataFrame(identity_rows).sort_values(
        ["role", "recording_id", "variant"], kind="stable"
    )
    _atomic_csv(paths["idmt_identity"], identities)
    _atomic_npz(
        paths["idmt_fingerprints"],
        features=np.stack(fingerprint_rows).astype(np.float32),
        recording_id=np.asarray(fingerprint_ids),
        role=np.asarray(fingerprint_roles),
    )
    role_outputs = {
        "calibration": paths["calibration_manifest"],
        "development_test": paths["development_manifest"],
    }
    manifest_reports = {}
    for role, output in role_outputs.items():
        frame = pd.DataFrame(segment_rows[role]).sort_values(
            ["location", "session_id", "recording_id", "segment_index"],
            kind="stable",
        )
        expected = (
            int(contract["expected_files"][role])
            * int(contract["expected_segments_per_file"])
        )
        if len(frame) != expected or set(frame["role"]) != {role}:
            raise ValueError(f"Unexpected {role} segment manifest")
        for row in frame.to_dict("records"):
            _reject_locked(row, context=f"{role} segment manifest")
        _atomic_csv(output, frame)
        manifest_reports[role] = {
            "path": output.relative_to(root).as_posix(),
            "sha256": file_sha256(output),
            "rows": len(frame),
            "recordings": int(frame["recording_id"].nunique()),
            "sessions": int(frame["session_id"].nunique()),
            "events": int(frame["event_group"].nunique()),
        }
    report = {
        "protocol": PROTOCOL,
        "phase": "idmt_identity",
        "passed": True,
        "opened_audio_members": opened_members,
        "locked_audio_members_read": locked_members_read,
        "model_inference_started": False,
        "training_started": False,
        "identity": {
            "path": paths["idmt_identity"].relative_to(root).as_posix(),
            "sha256": file_sha256(paths["idmt_identity"]),
            "rows": len(identities),
            "recordings": int(identities["recording_id"].nunique()),
        },
        "fingerprints": {
            "path": paths["idmt_fingerprints"].relative_to(root).as_posix(),
            "sha256": file_sha256(paths["idmt_fingerprints"]),
            "rows": len(fingerprint_rows),
            "policy": "candidate retrieval only; never an automatic exclusion",
        },
        "manifests": manifest_reports,
    }
    _atomic_json(paths["stage_dir"] / "idmt_identity_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def prepare_dads(config_path: Path, root: Path) -> dict[str, Any]:
    _, paths = _load(root, config_path)
    registry = pd.read_csv(paths["dads_source_registry"])
    required = {
        "label",
        "parquet_file",
        "row_group",
        "row_in_group",
        "source_path",
        "raw_audio_sha256",
    }
    missing = sorted(required - set(registry.columns))
    if missing:
        raise ValueError(f"DADS source registry is missing: {missing}")
    if len(registry) != 180_305 or not registry["raw_audio_sha256"].is_unique:
        raise ValueError("Unexpected DADS dedup-v2 source registry")

    requested: dict[tuple[str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in registry.itertuples(index=False):
        requested[(str(row.parquet_file), int(row.row_group))][
            int(row.row_in_group)
        ] = row._asdict()

    identity_rows: list[dict[str, Any]] = []
    fingerprint_rows: list[np.ndarray] = []
    fingerprint_ids: list[str] = []
    fingerprint_paths: list[str] = []
    fingerprint_splits: list[str] = []
    for (parquet_value, row_group), selected in tqdm(
        sorted(requested.items()),
        desc="Preparing DADS PCM identity",
        unit="row-group",
    ):
        parquet_path = _resolve(root, parquet_value).resolve(strict=True)
        parquet = pq.ParquetFile(parquet_path)
        audios = parquet.read_row_group(row_group, columns=["audio"]).column(
            "audio"
        ).to_pylist()
        for row_in_group, metadata in selected.items():
            wav_bytes = audios[row_in_group].get("bytes")
            if not isinstance(wav_bytes, bytes) or not wav_bytes:
                raise ValueError(
                    f"Missing DADS audio: {parquet_path}:{row_group}:{row_in_group}"
                )
            raw_sha = hashlib.sha256(wav_bytes).hexdigest()
            if raw_sha != str(metadata["raw_audio_sha256"]):
                raise ValueError("DADS raw SHA changed after dedup-v2")
            variants, sample_rate = decode_wav_channel_variants(wav_bytes)
            source_id = (
                f"{parquet_value}|{row_group}|{row_in_group}"
            )
            for variant_name, audio in variants.items():
                identity_rows.append(
                    {
                        "dataset": "DADS",
                        "source_id": source_id,
                        "split": metadata["split"],
                        "label": int(metadata["label"]),
                        "source_path": metadata["source_path"],
                        "variant": variant_name,
                        "sample_rate": sample_rate,
                        "samples": int(audio.size),
                        "raw_wav_sha256": raw_sha,
                        "canonical_pcm_sha256": quantized_pcm_sha256(
                            audio, sample_rate, normalize_gain=False
                        ),
                        "gain_invariant_pcm_sha256": quantized_pcm_sha256(
                            audio, sample_rate, normalize_gain=True
                        ),
                    }
                )
            if int(metadata["label"]) == 0:
                primary_name = "mono" if "mono" in variants else "mono_mean"
                fingerprint_rows.append(
                    spectral_fingerprint(variants[primary_name], sample_rate)
                )
                fingerprint_ids.append(source_id)
                fingerprint_paths.append(str(metadata["source_path"]))
                fingerprint_splits.append(str(metadata["split"]))

    identities = pd.DataFrame(identity_rows).sort_values(
        ["label", "source_id", "variant"], kind="stable"
    )
    _atomic_csv(paths["dads_identity"], identities)
    _atomic_npz(
        paths["dads_fingerprints"],
        features=np.stack(fingerprint_rows).astype(np.float32),
        source_id=np.asarray(fingerprint_ids),
        source_path=np.asarray(fingerprint_paths),
        split=np.asarray(fingerprint_splits),
    )
    report = {
        "protocol": PROTOCOL,
        "phase": "dads_identity",
        "passed": True,
        "sources": int(registry.shape[0]),
        "background_fingerprints": len(fingerprint_rows),
        "identity_rows": len(identities),
        "identity_sha256": file_sha256(paths["dads_identity"]),
        "fingerprints_sha256": file_sha256(paths["dads_fingerprints"]),
        "model_inference_started": False,
        "training_started": False,
    }
    _atomic_json(paths["stage_dir"] / "dads_identity_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def compare(config_path: Path, root: Path) -> dict[str, Any]:
    config, paths = _load(root, config_path)
    idmt = pd.read_csv(paths["idmt_identity"])
    dads = pd.read_csv(paths["dads_identity"])
    for frame, name in ((idmt, "IDMT"), (dads, "DADS")):
        required = {
            "raw_wav_sha256",
            "canonical_pcm_sha256",
            "gain_invariant_pcm_sha256",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} identity is missing columns: {missing}")

    matches = []
    layers = (
        ("raw_wav_sha256", "exact_wav_bytes"),
        ("canonical_pcm_sha256", "canonical_pcm"),
        ("gain_invariant_pcm_sha256", "gain_invariant_pcm"),
    )
    for column, layer in layers:
        left = idmt.merge(
            dads,
            on=column,
            how="inner",
            suffixes=("_idmt", "_dads"),
        )
        for row in left.to_dict("records"):
            matches.append(
                {
                    "match_layer": layer,
                    "identity_sha256": row[column],
                    "idmt_recording_id": row["recording_id"],
                    "idmt_role": row["role"],
                    "idmt_variant": row["variant_idmt"],
                    "dads_source_id": row["source_id"],
                    "dads_source_path": row["source_path"],
                    "dads_label": int(row["label"]),
                    "dads_variant": row["variant_dads"],
                }
            )
    exact = pd.DataFrame(
        matches,
        columns=[
            "match_layer",
            "identity_sha256",
            "idmt_recording_id",
            "idmt_role",
            "idmt_variant",
            "dads_source_id",
            "dads_source_path",
            "dads_label",
            "dads_variant",
        ],
    )
    _atomic_csv(paths["exact_matches"], exact)
    if not exact.empty and (exact["dads_label"] == 1).any():
        raise RuntimeError("IDMT identity matches a DADS UAV-positive source")

    with np.load(paths["idmt_fingerprints"], allow_pickle=False) as archive:
        idmt_features = archive["features"].astype(np.float32)
        idmt_ids = archive["recording_id"].astype(str)
        idmt_roles = archive["role"].astype(str)
    with np.load(paths["dads_fingerprints"], allow_pickle=False) as archive:
        dads_features = archive["features"].astype(np.float32)
        dads_ids = archive["source_id"].astype(str)
        dads_paths = archive["source_path"].astype(str)
        dads_splits = archive["split"].astype(str)
    nearest = NearestNeighbors(
        n_neighbors=3, metric="cosine", algorithm="brute", n_jobs=-1
    ).fit(dads_features)
    distances, indices = nearest.kneighbors(idmt_features)
    candidate_rows = []
    for idmt_index, recording_id in enumerate(idmt_ids):
        for rank, (distance, dads_index) in enumerate(
            zip(distances[idmt_index], indices[idmt_index], strict=True), start=1
        ):
            candidate_rows.append(
                {
                    "idmt_recording_id": recording_id,
                    "idmt_role": idmt_roles[idmt_index],
                    "rank": rank,
                    "spectral_cosine_similarity": float(1.0 - distance),
                    "dads_source_id": dads_ids[dads_index],
                    "dads_source_path": dads_paths[dads_index],
                    "dads_split": dads_splits[dads_index],
                    "review_status": "candidate_only_unreviewed",
                    "automatic_exclusion": False,
                }
            )
    candidates = pd.DataFrame(candidate_rows)
    _atomic_csv(paths["fingerprint_candidates"], candidates)

    exact_ids = set(exact["idmt_recording_id"].astype(str)) if not exact.empty else set()
    manifest_reports = {}
    for source_name, output_name, role in (
        ("calibration_manifest", "clean_calibration_manifest", "calibration"),
        ("development_manifest", "clean_development_manifest", "development_test"),
    ):
        frame = pd.read_csv(paths[source_name])
        for row in frame.to_dict("records"):
            _reject_locked(row, context=f"clean {role} manifest")
        clean = frame.loc[~frame["recording_id"].astype(str).isin(exact_ids)].copy()
        _atomic_csv(paths[output_name], clean)
        manifest_reports[role] = {
            "path": paths[output_name].relative_to(root).as_posix(),
            "sha256": file_sha256(paths[output_name]),
            "rows": len(clean),
            "recordings": int(clean["recording_id"].nunique()),
            "removed_exact_recordings": int(
                frame["recording_id"].nunique() - clean["recording_id"].nunique()
            ),
        }

    similarities = candidates.loc[
        candidates["rank"] == 1, "spectral_cosine_similarity"
    ].to_numpy(dtype=np.float64)
    audit = {
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "idmt_active_recordings": int(idmt["recording_id"].nunique()),
        "dads_unique_sources": int(dads["source_id"].nunique()),
        "dads_background_fingerprint_sources": int(dads_features.shape[0]),
        "exact_match_rows": int(len(exact)),
        "exact_match_recordings": int(len(exact_ids)),
        "exact_matches_by_layer": {
            key: int(value)
            for key, value in exact["match_layer"].value_counts().sort_index().items()
        }
        if not exact.empty
        else {},
        "label_conflicts": int((exact["dads_label"] == 1).sum())
        if not exact.empty
        else 0,
        "near_duplicate_policy": config["contract"]["near_duplicate_policy"],
        "fingerprint_candidates": {
            "rows": len(candidates),
            "top1_similarity_quantiles": {
                str(quantile): float(np.quantile(similarities, quantile))
                for quantile in (0.5, 0.9, 0.95, 0.99, 1.0)
            },
            "automatic_exclusions": 0,
            "reason": (
                "Spectral retrieval is a candidate screen. No scientifically "
                "calibrated confirmation threshold exists yet."
            ),
        },
        "clean_manifests": manifest_reports,
        "locked_audio_members_read": 0,
        "model_inference_started": False,
        "training_started": False,
        "historical_dads_manifest_modified": False,
    }
    _atomic_json(paths["audit"], audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("idmt", "dads", "compare", "all"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g7_idmt_stage_b.yaml")
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = _resolve(root, args.config).resolve(strict=True)
    if args.mode in {"idmt", "all"}:
        prepare_idmt(config_path, root)
    if args.mode in {"dads", "all"}:
        prepare_dads(config_path, root)
    if args.mode in {"compare", "all"}:
        compare(config_path, root)


if __name__ == "__main__":
    main()
