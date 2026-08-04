from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dads_crnn.prepare_segment_guarded_manifest import (
    PROTOCOL,
    prepare,
    validate_guarded_manifest,
)


SPLIT_RATIOS = {"train": 1.0, "val": 1.0, "test": 1.0}


def _waveform(position: int, samples: int = 8) -> np.ndarray:
    waveform = np.zeros(samples, dtype=np.float32)
    waveform[position % samples] = 1.0
    waveform[(position + 3) % samples] = -0.5
    return waveform


def _row(
    *,
    split: str,
    label: int,
    source_number: int,
    cache_path: Path,
) -> dict[str, object]:
    return {
        "split": split,
        "label": label,
        "parquet_file": "synthetic.parquet",
        "row_group": 0,
        "row_in_group": source_number,
        "source_path": f"source-{source_number}.wav",
        "segment_index": 0,
        "start_sample": 0,
        "end_sample": 8,
        "original_samples": 8,
        "segment_kind": "exact",
        "cache_path": cache_path.as_posix(),
    }


class SegmentGuardedManifestTests(unittest.TestCase):
    def _fixture(
        self, root: Path, *, conflicting_label: bool = False
    ) -> Path:
        cache_dir = root / "cache"
        cache_dir.mkdir()
        rows = []
        source_number = 0

        for label in (0, 1):
            for split_index, split in enumerate(("train", "val", "test")):
                cache_path = cache_dir / f"{source_number}.npy"
                np.save(cache_path, _waveform(label * 3 + split_index))
                rows.append(
                    _row(
                        split=split,
                        label=label,
                        source_number=source_number,
                        cache_path=cache_path,
                    )
                )
                source_number += 1

        duplicate_waveform = _waveform(7)
        first_duplicate = cache_dir / f"{source_number}.npy"
        np.save(first_duplicate, duplicate_waveform)
        rows.append(
            _row(
                split="train",
                label=0,
                source_number=source_number,
                cache_path=first_duplicate,
            )
        )
        source_number += 1

        second_duplicate = cache_dir / f"{source_number}.npy"
        np.save(second_duplicate, duplicate_waveform)
        rows.append(
            _row(
                split="test",
                label=1 if conflicting_label else 0,
                source_number=source_number,
                cache_path=second_duplicate,
            )
        )

        manifest_path = root / "dads_balanced_3_seed42.csv"
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        return manifest_path

    def test_prepare_groups_deduplicates_and_balances_segments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self._fixture(root)
            report = prepare(
                root,
                manifest_path,
                root / "guarded",
                target_samples=8,
                segments_per_class=3,
                split_ratios=SPLIT_RATIOS,
                seed=42,
            )

            output_path = root / "guarded" / "manifests" / manifest_path.name
            output = pd.read_csv(output_path)
            self.assertTrue(report["passed"])
            self.assertEqual(report["protocol"], PROTOCOL)
            self.assertEqual(
                report["input"]["cross_split_pcm16_groups"], 1
            )
            self.assertEqual(report["repair"]["duplicate_rows_removed"], 1)
            self.assertEqual(len(output), 6)
            self.assertEqual(output.groupby(["split", "label"]).size().to_dict(), {
                ("test", 0): 1,
                ("test", 1): 1,
                ("train", 0): 1,
                ("train", 1): 1,
                ("val", 0): 1,
                ("val", 1): 1,
            })
            self.assertTrue(output["segment_pcm16_sha256"].is_unique)
            self.assertEqual(
                int(output.groupby("source_id")["split"].nunique().max()), 1
            )
            validation = validate_guarded_manifest(
                output_path,
                expected_segments_per_class=3,
                split_ratios=SPLIT_RATIOS,
            )
            self.assertTrue(validation["passed"])
            self.assertFalse(any(validation["pcm16_overlap"].values()))

    def test_output_is_independent_of_input_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self._fixture(root)
            shuffled_path = root / "shuffled.csv"
            pd.read_csv(manifest_path).sample(
                frac=1.0, random_state=7
            ).to_csv(shuffled_path, index=False)

            prepare(
                root,
                manifest_path,
                root / "first",
                target_samples=8,
                segments_per_class=3,
                split_ratios=SPLIT_RATIOS,
                seed=42,
            )
            prepare(
                root,
                shuffled_path,
                root / "second",
                target_samples=8,
                segments_per_class=3,
                split_ratios=SPLIT_RATIOS,
                seed=42,
            )
            first = pd.read_csv(
                root / "first" / "manifests" / manifest_path.name
            )
            second = pd.read_csv(
                root / "second" / "manifests" / shuffled_path.name
            )
            pd.testing.assert_frame_equal(first, second)

    def test_conflicting_labels_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self._fixture(root, conflicting_label=True)
            with self.assertRaisesRegex(ValueError, "conflicting labels"):
                prepare(
                    root,
                    manifest_path,
                    root / "guarded",
                    target_samples=8,
                    segments_per_class=3,
                    split_ratios=SPLIT_RATIOS,
                    seed=42,
                )

    def test_cache_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self._fixture(root)
            prepare(
                root,
                manifest_path,
                root / "guarded",
                target_samples=8,
                segments_per_class=3,
                split_ratios=SPLIT_RATIOS,
                seed=42,
            )
            output_path = (
                root / "guarded" / "manifests" / manifest_path.name
            )
            output = pd.read_csv(output_path)
            mutated_cache = Path(str(output.iloc[0]["cache_path"]))
            np.save(
                mutated_cache,
                np.linspace(-1.0, 1.0, 8, dtype=np.float32),
            )

            with self.assertRaisesRegex(ValueError, "no longer match"):
                validate_guarded_manifest(
                    output_path,
                    expected_segments_per_class=3,
                    split_ratios=SPLIT_RATIOS,
                    verify_cache_content=True,
                    root=root,
                    target_samples=8,
                )


if __name__ == "__main__":
    unittest.main()
