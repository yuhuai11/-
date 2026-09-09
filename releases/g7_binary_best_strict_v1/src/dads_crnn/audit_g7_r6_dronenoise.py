from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .audio import ensure_sample_rate
from .audio_identity import decode_wav_channel_variants, quantized_pcm_sha256


EVENT_PATTERN = re.compile(r"^(?P<event>.+)_M(?P<microphone>\d+)\.wav$")


def audit(
    dataset_root: Path,
    dads_identity_path: Path,
    output_path: Path,
) -> dict:
    download_audit = json.loads(
        (dataset_root / "metadata/download_audit.json").read_text(encoding="utf-8")
    )
    if not download_audit.get("passed"):
        raise ValueError("DroneNoise download audit did not pass")
    dads = pd.read_csv(dads_identity_path, low_memory=False)
    dads_positive = dads[dads["label"].astype(int).eq(1)]
    dads_half = dads_positive[dads_positive["samples"].astype(int).eq(8000)]
    dads_canonical = set(dads_half["canonical_pcm_sha256"].dropna().astype(str))
    dads_gain = set(dads_half["gain_invariant_pcm_sha256"].dropna().astype(str))
    dads_raw = set(dads_positive["raw_wav_sha256"].dropna().astype(str))

    files = []
    window_matches = []
    event_groups: dict[str, list[str]] = {}
    observed_sha256: set[str] = set()
    official_duplicates = []
    for item in download_audit["inventory"]:
        if not item["is_audio"]:
            continue
        path = dataset_root / "raw" / item["name"]
        match = EVENT_PATTERN.match(item["name"])
        is_calibration = item["name"].startswith("Calib_")
        event = match.group("event") if match else item["name"].removesuffix(".wav")
        microphone = int(match.group("microphone")) if match else None
        event_groups.setdefault(event, []).append(item["name"])
        if item["sha256"] in observed_sha256:
            official_duplicates.append(item["name"])
        observed_sha256.add(item["sha256"])

        payload = path.read_bytes()
        variants, sample_rate = decode_wav_channel_variants(payload)
        audio = variants["mono"]
        values = ensure_sample_rate(audio, sample_rate, 16000)
        exact_raw_match = item["sha256"] in dads_raw
        canonical_matches = 0
        gain_matches = 0
        if not is_calibration:
            for window_index, start in enumerate(range(0, len(values) - 7999, 8000)):
                window = np.asarray(values[start : start + 8000], dtype=np.float32)
                canonical = quantized_pcm_sha256(window, 16000, normalize_gain=False)
                gain = quantized_pcm_sha256(window, 16000, normalize_gain=True)
                canonical_match = canonical in dads_canonical
                gain_match = gain in dads_gain
                canonical_matches += int(canonical_match)
                gain_matches += int(gain_match)
                if canonical_match or gain_match:
                    window_matches.append(
                        {
                            "file": item["name"],
                            "event_group": event,
                            "microphone": microphone,
                            "window_index": window_index,
                            "start_sample_16k": start,
                            "canonical_match": canonical_match,
                            "gain_invariant_match": gain_match,
                        }
                    )
        files.append(
            {
                "file": item["name"],
                "event_group": event,
                "microphone": microphone,
                "is_calibration": is_calibration,
                "sha256": item["sha256"],
                "exact_raw_match_to_dads_positive": exact_raw_match,
                "aligned_half_second_canonical_matches": canonical_matches,
                "aligned_half_second_gain_invariant_matches": gain_matches,
            }
        )

    data_files = [item for item in files if not item["is_calibration"]]
    matched_files = [
        item
        for item in data_files
        if item["exact_raw_match_to_dads_positive"]
        or item["aligned_half_second_canonical_matches"]
        or item["aligned_half_second_gain_invariant_matches"]
    ]
    report = {
        "passed": True,
        "protocol": "g7_r6_dronenoise_dads_overlap_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "DroneNoise Database v3",
        "license": "CC BY 4.0",
        "audio_files": len(files),
        "data_audio_files": len(data_files),
        "calibration_files_excluded": len(files) - len(data_files),
        "event_groups": len([key for key in event_groups if not key.startswith("Calib_")]),
        "official_duplicate_files": official_duplicates,
        "dads_positive_half_second_identities": len(dads_half),
        "files_with_exact_or_aligned_half_second_dads_overlap": len(matched_files),
        "matched_half_second_windows": len(window_matches),
        "interpretation": (
            "no_exact_or_aligned_half_second_overlap_detected"
            if not matched_files
            else "content_overlap_detected_do_not_merge_matched_files"
        ),
        "limitations": [
            "zero exact matches do not exclude offset, codec, filtering, or other transformed copies",
            "all microphones from one event must remain in the same dataset role",
        ],
        "training_status": (
            "eligible_for_session_grouped_intake_after_manual_metadata_mapping"
            if not matched_files
            else "quarantined_overlap_detected"
        ),
        "files": files,
        "window_matches": window_matches,
        "inputs": {
            "download_audit": str(dataset_root / "metadata/download_audit.json"),
            "download_audit_sha256": hashlib.sha256(
                (dataset_root / "metadata/download_audit.json").read_bytes()
            ).hexdigest(),
            "dads_identity": str(dads_identity_path),
            "dads_identity_sha256": hashlib.sha256(dads_identity_path.read_bytes()).hexdigest(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DroneNoise v3 against DADS identities")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/g7_r6_new_sources/drone_noise_v3"),
    )
    parser.add_argument(
        "--dads-identity",
        type=Path,
        default=Path("artifacts/g7_improvement/stage_b/dads_pcm_identity.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/g7_r6_new_sources/drone_noise_v3/metadata/dads_overlap_audit.json"),
    )
    args = parser.parse_args()
    report = audit(args.dataset_root, args.dads_identity, args.output)
    keys = (
        "passed",
        "audio_files",
        "data_audio_files",
        "event_groups",
        "files_with_exact_or_aligned_half_second_dads_overlap",
        "matched_half_second_windows",
        "interpretation",
        "training_status",
    )
    print(json.dumps({key: report[key] for key in keys}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
