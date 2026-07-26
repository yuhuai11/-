from __future__ import annotations

import unittest

from dads_crnn.prepare_g14_segment_cache import _expected_split_segments


class G14SegmentCacheTests(unittest.TestCase):
    def test_expected_split_segments_sum_both_labels(self) -> None:
        preflight = {
            "split_summary": {
                split: {
                    "positive": {"predicted_full_segments": index + 1},
                    "background": {"predicted_full_segments": index + 2},
                }
                for index, split in enumerate(("train", "tune", "dev_holdout"))
            }
        }
        self.assertEqual(
            _expected_split_segments(preflight),
            {"train": 3, "tune": 5, "dev_holdout": 7},
        )


if __name__ == "__main__":
    unittest.main()
