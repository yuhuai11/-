#!/usr/bin/env python3
"""Build an inode-aware inventory before removing generated experiment files."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    candidates: list[Path] = []
    for name in ("last.pt", "*.log~", "*.tmp", "*.lock", "*.incomplete"):
        candidates.extend(root.rglob(name))

    rows = []
    seen: dict[tuple[int, int], str] = {}
    for path in sorted(set(candidates)):
        if ".git" in path.parts or not path.is_file():
            continue
        stat = path.stat()
        inode_key = (stat.st_dev, stat.st_ino)
        canonical = seen.setdefault(inode_key, str(path.relative_to(root)))
        sibling_best = path.with_name("best.pt")
        evidence = [
            name
            for name in ("metrics.json", "history.csv", "summary.json", "training_summary.json")
            if path.with_name(name).exists()
        ]
        if path.name == "last.pt" and sibling_best.exists():
            proposed_action = "DELETE_AFTER_CHECK"
            reason = "completed run has sibling best.pt; last.pt is resume-only"
        elif path.name == "last.pt":
            proposed_action = "REVIEW"
            reason = "no sibling best.pt; interrupted or non-promoted run"
        else:
            proposed_action = "DELETE_IF_STALE"
            reason = "generated temporary, backup, lock, or incomplete file"
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": stat.st_size,
                "sha256": sha256(path),
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "link_count": stat.st_nlink,
                "canonical_inode_path": canonical,
                "sibling_best_exists": sibling_best.exists(),
                "evidence": ";".join(evidence),
                "proposed_action": proposed_action,
                "reason": reason,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["path"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
