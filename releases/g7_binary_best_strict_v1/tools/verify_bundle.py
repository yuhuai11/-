from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ASSETS = {
    "configs/g7_strict_retrain_v1.yaml": (
        2024,
        "670de60957cfd306903ae0a6ea1e4e89701d446285a0519d5b62c57e8bc910cd",
        False,
    ),
    "src/dads_crnn/panns.py": (
        14090,
        "41f28fea3f623452d77303182fd0450bdc7cf3a2e67b16894ea30e6c4494bdd6",
        False,
    ),
    "src/dads_crnn/train_panns.py": (
        33390,
        "ecd6d8cb270d9e930c2b6565d7a94b37d5a9779cd1f5d02fb91ff7f31d0d2042",
        False,
    ),
    "artifacts/g7_panns/checkpoints/Cnn14_16k_mAP=0.438.pth": (
        358668570,
        "e2ee543a27919542c2ea03eabaa70b24dcd4e6c8e05621de6b67a94e4c5058e6",
        True,
    ),
    "artifacts/g7_leakage_fixed_v2/data/manifest.csv": (
        311312051,
        "46973865f4a99aeddd59f22be31d4daa0f5826c6dcd964fa1ff0f8c14c38a3f6",
        True,
    ),
    "artifacts/g7_leakage_fixed_v2/data/cache/native_half_second_audio.npy": (
        13940256128,
        "709d2fd5e8c565f0465fad9c83945fe6299991e3e9583df9785c4948d850c970",
        True,
    ),
    "artifacts/g7_strict_retrain_v1/runs/seed_42/best.pt": (
        319919130,
        "30cdf88691a3f357ca00568f5f34fd91d491d896b45c087448c2506ac79862ac",
        True,
    ),
    "artifacts/g7_strict_retrain_v1/runs/seed_43/best.pt": (
        319919130,
        "717c8ed8a958aad76df47267a79d1583bf9be05a4dd5c38d59a5c8ae9e756a81",
        True,
    ),
    "artifacts/g7_strict_retrain_v1/runs/seed_44/best.pt": (
        319919130,
        "65879b3cee424edda47c47604a99f0dcf172c0231917536725c894bb42019529",
        True,
    ),
    "artifacts/g7_strict_retrain_v1/external_baseline/metrics.json": (
        251141,
        "bb7b533479a179a9bc2acacbbeaf285a4239fecd5572050e006b6bccac37f4a9",
        False,
    ),
    "artifacts/g7_strict_retrain_v1/multiscale_aggregation/metrics.json": (
        142064,
        "4ce372ff586f9bf828d1bee6f796c7846db2919c27d9683fb0613c1cb3fc9be7",
        False,
    ),
}

DATA_MANIFESTS = {
    "artifacts/g13_external_confirmation/intake/external_confirmation_v2_manifest.csv": (
        "path", 55000, 55000
    ),
    "artifacts/g7_improvement/stage_b/development_segments_dedup.csv": (
        "path", 29346, 14673
    ),
    "artifacts/g7_r6_reusable_multicorpus/threshold_calibration_manifest.csv": (
        "cache_path", 4480, 1
    ),
    "artifacts/g7_r5_train_val_test/locked_unseen_external_test/manifest.csv": (
        "cache_path", 38126, 1
    ),
    "artifacts/g9_hard_negatives/manifests/hn_guard.csv": (
        "cache_path", 240, 240
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_data_paths() -> list[str]:
    failures = []
    for relative, (column, expected_rows, expected_unique) in DATA_MANIFESTS.items():
        manifest = ROOT / relative
        if not manifest.is_file():
            failures.append(f"missing data manifest: {relative}")
            continue
        rows = 0
        paths = set()
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows += 1
                value = str(row.get(column, "")).strip()
                if value:
                    paths.add(value)
        if rows != expected_rows:
            failures.append(f"row mismatch: {relative}: {rows} != {expected_rows}")
        if len(paths) != expected_unique:
            failures.append(
                f"unique path mismatch: {relative}: {len(paths)} != {expected_unique}"
            )
        missing = [value for value in paths if not (ROOT / value).is_file()]
        if missing:
            failures.append(
                f"missing referenced data: {relative}: {missing[0]} "
                f"(+{len(missing) - 1} more)"
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the frozen G7 binary bundle")
    parser.add_argument("--fast", action="store_true", help="Skip hashes for large assets")
    args = parser.parse_args()
    failures = []
    checked_hashes = 0
    for relative, (expected_size, expected_sha256, large) in ASSETS.items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        observed_size = path.stat().st_size
        if observed_size != expected_size:
            failures.append(
                f"size mismatch: {relative}: {observed_size} != {expected_size}"
            )
            continue
        if args.fast and large:
            continue
        observed_sha256 = sha256(path)
        checked_hashes += 1
        if observed_sha256 != expected_sha256:
            failures.append(
                f"sha256 mismatch: {relative}: {observed_sha256} != {expected_sha256}"
            )
    failures.extend(validate_data_paths())
    if failures:
        raise SystemExit("Bundle verification failed:\n- " + "\n- ".join(failures))
    mode = "fast" if args.fast else "full"
    print(
        f"G7 bundle verification passed: mode={mode}, assets={len(ASSETS)}, "
        f"hashes_checked={checked_hashes}, data_manifests={len(DATA_MANIFESTS)}"
    )


if __name__ == "__main__":
    main()
