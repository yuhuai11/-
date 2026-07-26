from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ALGORITHM = "g13_external_confirmation_assembly_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble prepared DDL and AeroSonicDB sources for G13")
    parser.add_argument("--root", type=Path, default=Path("data/external_confirmation_v2"))
    parser.add_argument(
        "--background",
        type=Path,
        default=Path("data/external_confirmation_v2/_prepared/AeroSonicDB_v1.1.2/Background"),
    )
    parser.add_argument(
        "--uav", type=Path, default=Path("data/external_confirmation_v2/_prepared/DDL_real/UAV")
    )
    parser.add_argument(
        "--background-registry",
        type=Path,
        default=Path("artifacts/g13_external_confirmation/source_preparations/aerosonic_source_registry.csv"),
    )
    parser.add_argument(
        "--uav-registry",
        type=Path,
        default=Path("artifacts/g13_external_confirmation/source_preparations/ddl_source_registry.csv"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("artifacts/g13_external_confirmation/intake/assembly_audit.json"),
    )
    args = parser.parse_args()

    final_background = args.root / "Background"
    final_uav = args.root / "UAV"
    final_registry = args.root / "source_registry.csv"
    staging = args.root / ".assembly.tmp"
    if any(path.exists() for path in (final_background, final_uav, final_registry, staging, args.audit)):
        raise FileExistsError("G13 assembly output already exists; refusing overwrite")

    staging.mkdir(parents=True)
    seen: dict[str, Path] = {}
    kept_by_label: Counter[str] = Counter()
    kept_by_source: Counter[str] = Counter()
    dropped_by_label: Counter[str] = Counter()
    dropped_by_source: Counter[str] = Counter()
    hard_links = 0
    try:
        for label_name, source_root in (("Background", args.background), ("UAV", args.uav)):
            for source_path in sorted(source_root.rglob("*.wav")):
                relative = source_path.relative_to(source_root)
                source_group = relative.parts[0]
                digest = sha256(source_path)
                if digest in seen:
                    dropped_by_label[label_name] += 1
                    dropped_by_source[source_group] += 1
                    continue
                seen[digest] = source_path
                destination = staging / label_name / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source_path, destination)
                    hard_links += 1
                except OSError:
                    shutil.copy2(source_path, destination)
                kept_by_label[label_name] += 1
                kept_by_source[source_group] += 1

        observed_sources = set(kept_by_source)
        registry = pd.concat(
            [pd.read_csv(args.background_registry, dtype=str), pd.read_csv(args.uav_registry, dtype=str)],
            ignore_index=True,
        )
        registry["source_group"] = registry["source_group"].str.strip()
        if registry["source_group"].duplicated().any():
            raise ValueError("G13 assembled registry contains duplicate source groups")
        if set(registry["source_group"]) != observed_sources:
            raise ValueError("G13 assembled registry does not match observed source groups")
        registry.sort_values(["label", "source_group"]).to_csv(staging / "source_registry.csv", index=False)

        label_sources = registry.groupby("label")["source_group"].nunique()
        if min(kept_by_label.values()) < 1000:
            raise RuntimeError(f"G13 label sample requirement failed: {dict(kept_by_label)}")
        if int(label_sources.min()) < 10:
            raise RuntimeError(f"G13 label source requirement failed: {label_sources.to_dict()}")
        if min(kept_by_source.values()) < 50:
            raise RuntimeError("G13 per-source sample requirement failed after deduplication")

        os.replace(staging / "Background", final_background)
        os.replace(staging / "UAV", final_uav)
        os.replace(staging / "source_registry.csv", final_registry)
        staging.rmdir()
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    report = {
        "algorithm": ALGORITHM,
        "passed": True,
        "unique_audio_sha256": len(seen),
        "kept_by_label": dict(sorted(kept_by_label.items())),
        "dropped_duplicates_by_label": dict(sorted(dropped_by_label.items())),
        "dropped_duplicates_by_source": dict(sorted(dropped_by_source.items())),
        "source_groups": len(kept_by_source),
        "source_groups_by_label": {str(k): int(v) for k, v in label_sources.items()},
        "minimum_samples_per_source": min(kept_by_source.values()),
        "hard_links_created": hard_links,
        "outputs": {
            "background": final_background.as_posix(),
            "uav": final_uav.as_posix(),
            "source_registry": final_registry.as_posix(),
        },
        "inputs": {
            "background_registry": {"path": args.background_registry.as_posix(), "sha256": sha256(args.background_registry)},
            "uav_registry": {"path": args.uav_registry.as_posix(), "sha256": sha256(args.uav_registry)},
        },
        "locked_datasets_read": [],
        "model_predictions_read": False,
    }
    _atomic_json(args.audit, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
