from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Iterable, Mapping


LOCKED_COMPACT_TOKENS = (
    "unseen",
    "realworld",
    "testaugmented",
    "rawrecordedaudios",
    "externalconfirmationv2",
    "g13externalconfirmation",
)
PATH_LIKE_COLUMNS = {
    "background_source",
    "cache_path",
    "dataset",
    "dataset_origin",
    "filename",
    "parquet_file",
    "path",
    "provenance_group",
    "recording_group",
    "source_family",
    "source_group",
    "source_path",
    "uav_source",
    "uri",
    "url",
}
HASH_COLUMNS = {"sha256", "audio_sha256", "cache_sha256", "file_sha256"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_HASH_REGISTRY = Path(
    "artifacts/g14_domain_generalization/p0_firewall/g13_audio_sha256.txt"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _normalized_tokens(tokens: Iterable[str]) -> tuple[str, ...]:
    return tuple(token for value in tokens if (token := compact_token(value)))


def reject_locked_value(
    value: object,
    *,
    context: str,
    tokens: Iterable[str] = LOCKED_COMPACT_TOKENS,
) -> None:
    text = str(value).strip()
    if not text:
        return
    compact = compact_token(text)
    if any(token in compact for token in _normalized_tokens(tokens)):
        raise ValueError(f"Locked final-test value is forbidden in {context}: {text}")


def reject_locked_path(
    path: Path,
    *,
    context: str = "development input",
    tokens: Iterable[str] = LOCKED_COMPACT_TOKENS,
) -> None:
    """Reject lexical paths and symlink-resolved targets before opening them."""
    candidates = [path]
    try:
        candidates.append(path.resolve(strict=True))
    except (FileNotFoundError, OSError):
        candidates.append(path.resolve(strict=False))
    for candidate in candidates:
        try:
            reject_locked_value(candidate.as_posix(), context=context, tokens=tokens)
        except ValueError as error:
            raise ValueError(f"Locked final-test path is forbidden: {candidate}") from error


def load_forbidden_hashes(path: Path | None = None) -> frozenset[str]:
    registry = DEFAULT_HASH_REGISTRY if path is None else path
    if not registry.is_file():
        return frozenset()
    hashes: set[str] = set()
    with registry.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip().lower()
            if not value:
                continue
            if not SHA256_PATTERN.fullmatch(value):
                raise ValueError(
                    f"Malformed SHA256 in consumed-data registry {registry}:{line_number}"
                )
            hashes.add(value)
    return frozenset(hashes)


def reject_consumed_hash(
    value: object,
    *,
    context: str,
    forbidden_hashes: frozenset[str],
) -> None:
    candidate = str(value).strip().lower()
    if candidate and candidate in forbidden_hashes:
        raise ValueError(f"Consumed final-test audio hash is forbidden in {context}")


def audit_csv_rows(
    path: Path,
    *,
    forbidden_hashes: frozenset[str] | None = None,
    required_columns: Iterable[str] = (),
) -> int:
    """Stream a CSV and reject locked provenance or consumed sample hashes."""
    reject_locked_path(path, context="manifest path")
    hashes = load_forbidden_hashes() if forbidden_hashes is None else forbidden_hashes
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Manifest lacks a CSV header: {path}")
        fields = set(reader.fieldnames)
        missing = sorted(set(required_columns) - fields)
        if missing:
            raise ValueError(f"Manifest {path} is missing columns: {missing}")
        value_columns = [
            column
            for column in reader.fieldnames
            if column.lower() in PATH_LIKE_COLUMNS or "path" in column.lower()
        ]
        hash_columns = [
            column for column in reader.fieldnames if column.lower() in HASH_COLUMNS
        ]
        rows = 0
        for row_number, row in enumerate(reader, start=2):
            rows += 1
            for column in value_columns:
                reject_locked_value(
                    row.get(column, ""),
                    context=f"{path}: row {row_number}, column {column}",
                )
            for column in hash_columns:
                reject_consumed_hash(
                    row.get(column, ""),
                    context=f"{path}: row {row_number}, column {column}",
                    forbidden_hashes=hashes,
                )
    if rows <= 0:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def reject_mapping(
    values: Mapping[str, object],
    *,
    context: str,
    forbidden_hashes: frozenset[str] | None = None,
) -> None:
    """Check config/manifest-like mappings without requiring a CSV file."""
    hashes = load_forbidden_hashes() if forbidden_hashes is None else forbidden_hashes
    for key, value in values.items():
        lowered = str(key).lower()
        if lowered in PATH_LIKE_COLUMNS or "path" in lowered:
            reject_locked_value(value, context=f"{context}:{key}")
        if lowered in HASH_COLUMNS:
            reject_consumed_hash(
                value,
                context=f"{context}:{key}",
                forbidden_hashes=hashes,
            )
