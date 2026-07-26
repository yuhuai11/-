from __future__ import annotations

import unittest

import pandas as pd

from dads_crnn.prepare_dads_dedup_v2 import (
    LOCATOR_COLUMNS,
    _pairwise_overlap,
    select_canonical_sources,
)


def _source(
    parquet_file: str,
    row: int,
    split: str,
    label: int,
    sha256: str,
) -> dict[str, object]:
    return {
        "split": split,
        "original_split": split,
        "label": label,
        "parquet_file": parquet_file,
        "row_group": 0,
        "row_in_group": row,
        "source_path": f"source-{row}.wav",
        "original_samples": 16000,
        "raw_audio_sha256": sha256,
        "raw_audio_bytes": 32044,
        "recording_group": f"dads_raw_sha256:{sha256}",
    }


class DadsDedupV2Tests(unittest.TestCase):
    def test_duplicate_hash_keeps_one_source_and_zero_split_overlap(self) -> None:
        duplicate = "a" * 64
        frame = pd.DataFrame(
            [
                _source("a.parquet", 0, "train", 0, duplicate),
                _source("a.parquet", 1, "test", 0, duplicate),
                _source("a.parquet", 2, "val", 0, "b" * 64),
                _source("a.parquet", 3, "train", 1, "c" * 64),
                _source("a.parquet", 4, "val", 1, "d" * 64),
                _source("a.parquet", 5, "test", 1, "e" * 64),
            ]
        )
        selected, duplicates = select_canonical_sources(frame)
        self.assertEqual(len(selected), 5)
        self.assertEqual(int((selected["raw_audio_sha256"] == duplicate).sum()), 1)
        self.assertEqual(int(duplicates["kept"].sum()), 1)
        self.assertTrue(duplicates["cross_split_before"].all())
        self.assertFalse(any(_pairwise_overlap(selected, "raw_audio_sha256").values()))

    def test_selection_is_independent_of_input_order(self) -> None:
        duplicate = "f" * 64
        frame = pd.DataFrame(
            [
                _source("b.parquet", 2, "test", 0, duplicate),
                _source("a.parquet", 1, "train", 0, duplicate),
                _source("a.parquet", 3, "val", 0, "1" * 64),
            ]
        )
        first, _ = select_canonical_sources(frame)
        second, _ = select_canonical_sources(
            frame.sample(frac=1.0, random_state=42).reset_index(drop=True)
        )
        columns = ["raw_audio_sha256", *LOCATOR_COLUMNS, "split"]
        self.assertEqual(
            first[columns].sort_values(columns).to_dict("records"),
            second[columns].sort_values(columns).to_dict("records"),
        )

    def test_conflicting_labels_fail_closed(self) -> None:
        duplicate = "9" * 64
        frame = pd.DataFrame(
            [
                _source("a.parquet", 0, "train", 0, duplicate),
                _source("a.parquet", 1, "test", 1, duplicate),
            ]
        )
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            select_canonical_sources(frame)


if __name__ == "__main__":
    unittest.main()
