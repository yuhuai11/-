from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dads_crnn.prepare_g7_r6_protocol import PROTOCOL, _matches_prefixes, _overlap


class G7R6ProtocolTest(unittest.TestCase):
    def test_active_protocol_is_dronenoise_integrated_version(self) -> None:
        self.assertEqual(PROTOCOL, "g7_r6_reuter_reusable_multicorpus_v3")

    def test_prefix_partition_is_source_group_based(self) -> None:
        values = pd.Series(
            ["tau_urban_2022:lisbon-1000", "tau_urban_2022:prague-1027"]
        )
        self.assertEqual(
            _matches_prefixes(values, ["tau_urban_2022:lisbon-"]).tolist(),
            [True, False],
        )

    def test_overlap_counts_unique_identities(self) -> None:
        left = pd.DataFrame({"audio_sha256": ["a", "a", "b"]})
        right = pd.DataFrame({"audio_sha256": ["b", "c"]})
        self.assertEqual(_overlap(left, right, "audio_sha256"), 1)


if __name__ == "__main__":
    unittest.main()
