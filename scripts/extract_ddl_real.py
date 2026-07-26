#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path("/home/user1/JJZ/ABDDV-CRNN")
SOURCE_DIR = ROOT / "data/external_confirmation_v2/_raw/DDL"
ARCHIVE = SOURCE_DIR / "MLSP_2022_Real_Data.zip"
STAGING = SOURCE_DIR / ".extracted.tmp"
FINAL = SOURCE_DIR / "extracted"
AUDIT = SOURCE_DIR / "extraction_audit.json"
EXPECTED_MD5 = "4a6d4da4e1c732550c1ccd8d29dd16f8"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"Unsafe archive path: {name!r}")
    return Path(*pure.parts)


def main() -> int:
    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)
    if FINAL.exists() or AUDIT.exists():
        raise FileExistsError("DDL extraction output already exists; refusing overwrite")
    if STAGING.exists() and any(STAGING.iterdir()):
        raise FileExistsError(f"Non-empty staging directory already exists: {STAGING}")

    actual_md5 = md5(ARCHIVE)
    if actual_md5 != EXPECTED_MD5:
        raise RuntimeError(f"DDL archive MD5 mismatch: {actual_md5}")

    with zipfile.ZipFile(ARCHIVE) as archive:
        entries = archive.infolist()
        total_bytes = sum(info.file_size for info in entries if not info.is_dir())
        file_entries = sum(not info.is_dir() for info in entries)
        free_bytes = shutil.disk_usage(SOURCE_DIR).free
        required_bytes = total_bytes + 2 * 1024**3
        if free_bytes < required_bytes:
            raise RuntimeError(
                f"Insufficient disk: free={free_bytes}, required={required_bytes}"
            )

        for info in entries:
            safe_relative(info.filename)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"Symlink entry is not allowed: {info.filename}")

        STAGING.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        extracted_files = 0
        extracted_bytes = 0
        for info in entries:
            relative = safe_relative(info.filename)
            destination = STAGING / relative
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            extracted_files += 1
            extracted_bytes += info.file_size
            if extracted_files % 500 == 0 or extracted_files == file_entries:
                elapsed = max(time.monotonic() - started, 0.001)
                print(
                    f"DDL extract {extracted_files}/{file_entries} files "
                    f"({100.0 * extracted_bytes / max(total_bytes, 1):.1f}%), "
                    f"{extracted_bytes / 1024**3:.2f} GiB, "
                    f"{extracted_bytes / elapsed / 1024**2:.1f} MiB/s",
                    flush=True,
                )

    os.replace(STAGING, FINAL)
    report = {
        "archive": str(ARCHIVE),
        "archive_size_bytes": ARCHIVE.stat().st_size,
        "archive_md5": actual_md5,
        "entries": len(entries),
        "files": file_entries,
        "uncompressed_bytes": total_bytes,
        "output": str(FINAL),
        "zip_crc_verified_during_extraction": True,
    }
    temporary_audit = AUDIT.with_suffix(".json.tmp")
    temporary_audit.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_audit, AUDIT)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"DDL extraction failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise
