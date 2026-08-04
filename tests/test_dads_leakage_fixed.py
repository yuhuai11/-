from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dads_crnn.data_firewall import file_sha256
from dads_crnn.prepare_dads_leakage_fixed import (
    PROTOCOL,
    TARGET_SAMPLES,
    _assign_components,
    _attach_components,
    _canonicalize_content,
    _drop_cross_label_content,
    _waveform_hashes,
    validate_manifest,
)


class DadsLeakageFixedTests(unittest.TestCase):
    def _row(
        self,
        source: str,
        label: int,
        waveform: np.ndarray,
        *,
        original_split: str,
        cache_index: int,
    ) -> dict:
        float_hash, pcm_hash = _waveform_hashes(waveform)
        return {
            "label": label,
            "parquet_file": "data.parquet",
            "row_group": 0,
            "row_in_group": cache_index,
            "source_path": f"{source}.wav",
            "segment_index": 0,
            "start_sample": 0,
            "end_sample": TARGET_SAMPLES,
            "original_samples": TARGET_SAMPLES,
            "segment_kind": "native_half_second",
            "cache_path": "cache.npy",
            "cache_index": cache_index,
            "raw_audio_sha256": f"{cache_index + 1:064x}",
            "recording_group": source,
            "source_id": source,
            "original_split": original_split,
            "segment_float32_sha256": float_hash,
            "segment_pcm16_sha256": pcm_hash,
            "model_samples": TARGET_SAMPLES,
        }

    def test_cross_label_equal_content_is_removed(self) -> None:
        waveform = np.linspace(-1.0, 1.0, TARGET_SAMPLES, dtype=np.float32)
        frame = pd.DataFrame(
            [
                self._row("negative", 0, waveform, original_split="train", cache_index=0),
                self._row("positive", 1, waveform, original_split="test", cache_index=1),
                self._row(
                    "positive-unique",
                    1,
                    waveform[::-1].copy(),
                    original_split="val",
                    cache_index=2,
                ),
            ]
        )
        clean, conflicts, groups = _drop_cross_label_content(frame)
        self.assertEqual(len(clean), 1)
        self.assertEqual(len(conflicts), 2)
        self.assertEqual(groups["segment_pcm16_sha256_groups"], 1)

    def test_equal_content_sources_form_one_split_component(self) -> None:
        shared = np.sin(
            np.linspace(0.0, 20.0, TARGET_SAMPLES, dtype=np.float32)
        ).astype(np.float32)
        other = np.cos(
            np.linspace(0.0, 20.0, TARGET_SAMPLES, dtype=np.float32)
        ).astype(np.float32)
        frame = pd.DataFrame(
            [
                self._row("a", 0, shared, original_split="train", cache_index=0),
                self._row("b", 0, shared, original_split="test", cache_index=1),
                self._row("c", 0, other, original_split="val", cache_index=2),
            ]
        )
        connected = _attach_components(frame)
        self.assertEqual(
            connected.loc[connected["source_id"].isin(["a", "b"]), "split_component"].nunique(),
            1,
        )
        assigned, _, _ = _assign_components(
            connected,
            {"train": 0.70, "val": 0.15, "test": 0.15},
            42,
        )
        self.assertEqual(
            assigned.loc[assigned["source_id"].isin(["a", "b"]), "split"].nunique(),
            1,
        )
        canonical, duplicates = _canonicalize_content(assigned)
        self.assertEqual(len(canonical), 2)
        self.assertEqual(len(duplicates), 2)

    def test_validator_accepts_a_small_native_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = root / "cache.npy"
            cache = np.lib.format.open_memmap(
                cache_path,
                mode="w+",
                dtype=np.float32,
                shape=(6, TARGET_SAMPLES),
            )
            rows = []
            for index, split in enumerate(("train", "train", "val", "val", "test", "test")):
                label = index % 2
                waveform = np.sin(
                    np.linspace(
                        0.0,
                        float(index + 1) * 5.0,
                        TARGET_SAMPLES,
                        dtype=np.float32,
                    )
                ).astype(np.float32)
                cache[index] = waveform
                row = self._row(
                    f"source-{index}",
                    label,
                    waveform,
                    original_split=split,
                    cache_index=index,
                )
                row.update(
                    split=split,
                    split_component=f"component-{index}",
                    cache_path=cache_path.as_posix(),
                    manifest_protocol=PROTOCOL,
                    evaluation_role={
                        "train": "training",
                        "val": "internal_model_selection",
                        "test": "consumed_internal_development_test",
                    }[split],
                    consumption_status="underlying_dads_pool_previously_consumed",
                )
                rows.append(row)
            cache.flush()
            del cache
            manifest_path = root / "manifest.csv"
            manifest = pd.DataFrame(rows)
            manifest.to_csv(manifest_path, index=False)
            audit_path = root / "audit.json"
            audit = {
                "passed": True,
                "protocol": PROTOCOL,
                "sample_rate": 16000,
                "target_samples": TARGET_SAMPLES,
                "clip_seconds": 0.5,
                "split_ratios": {
                    "train": 1.0,
                    "val": 1.0,
                    "test": 1.0,
                },
                "construction": {
                    "allocation_unit": (
                        "per_label_effective_source_and_unique_pcm16_window"
                    ),
                    "allocation_targets": {
                        "sources": {
                            "0": {"train": 1, "val": 1, "test": 1},
                            "1": {"train": 1, "val": 1, "test": 1},
                        },
                        "unique_windows": {
                            "0": {"train": 1, "val": 1, "test": 1},
                            "1": {"train": 1, "val": 1, "test": 1},
                        }
                    },
                    "allocation_actual": {
                        "sources": {
                            "0": {"train": 1, "val": 1, "test": 1},
                            "1": {"train": 1, "val": 1, "test": 1},
                        },
                        "unique_windows": {
                            "0": {"train": 1, "val": 1, "test": 1},
                            "1": {"train": 1, "val": 1, "test": 1},
                        },
                    },
                    "max_allowed_abs_ratio_error": 0.001,
                    "max_abs_ratio_error": 0.0,
                },
                "output": {
                    "manifest": {"sha256": file_sha256(manifest_path)},
                    "cache": {
                        "sha256": file_sha256(cache_path),
                        "shape": [6, TARGET_SAMPLES],
                    }
                },
            }
            audit_path.write_text(
                json.dumps(audit),
                encoding="utf-8",
            )
            observed = validate_manifest(
                manifest_path, audit_path, verify_cache_file=True
            )
            self.assertTrue(observed["passed"])
            self.assertEqual(observed["cache_shape"], [6, TARGET_SAMPLES])
            self.assertFalse(observed["fresh_final_holdout"])

            crossed_component = manifest.copy()
            crossed_component.loc[2, "split_component"] = crossed_component.loc[
                0, "split_component"
            ]
            crossed_component.to_csv(manifest_path, index=False)
            audit["output"]["manifest"]["sha256"] = file_sha256(manifest_path)
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "split_component crosses"
            ):
                validate_manifest(
                    manifest_path, audit_path, verify_cache_file=True
                )

            swapped = manifest.copy()
            swapped.loc[[0, 1], "cache_index"] = [1, 0]
            swapped.to_csv(manifest_path, index=False)
            audit["output"]["manifest"]["sha256"] = file_sha256(manifest_path)
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "cache row no longer matches"
            ):
                validate_manifest(
                    manifest_path, audit_path, verify_cache_file=True
                )

            duplicated = manifest.copy()
            duplicated.loc[1, "cache_index"] = 0
            duplicated.to_csv(manifest_path, index=False)
            audit["output"]["manifest"]["sha256"] = file_sha256(manifest_path)
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "multiple rows to one cache index"
            ):
                validate_manifest(
                    manifest_path, audit_path, verify_cache_file=True
                )


if __name__ == "__main__":
    unittest.main()
