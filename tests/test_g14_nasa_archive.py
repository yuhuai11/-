from __future__ import annotations

import unittest

from dads_crnn.audit_g14_nasa_archive import parse_member


class G14NasaArchiveTests(unittest.TestCase):
    positive = {"hex", "phantom", "y6"}
    negative = {"edge", "cub"}

    def test_parses_uav_and_fixed_wing_flights(self) -> None:
        uav = parse_member(
            "data/phantom_flyover_120.mat",
            positive_tokens=self.positive,
            negative_tokens=self.negative,
        )
        negative = parse_member(
            "data/cub_flyover_107.mat",
            positive_tokens=self.positive,
            negative_tokens=self.negative,
        )
        self.assertEqual(uav["label"], 1)
        self.assertEqual(uav["source_group"], "nasa_suas:phantom:flyover:120")
        self.assertEqual(uav["acquisition_group"], "virginia_beach_2014")
        self.assertEqual(negative["label"], 0)
        self.assertEqual(negative["acquisition_group"], "ap_hill_2015")

    def test_metadata_and_unknown_audio_names_are_not_silently_labeled(self) -> None:
        self.assertIsNone(
            parse_member(
                "Data Description 20160203.pdf",
                positive_tokens=self.positive,
                negative_tokens=self.negative,
            )
        )
        self.assertIsNone(
            parse_member(
                "data/unknown_flyover_001.mat",
                positive_tokens=self.positive,
                negative_tokens=self.negative,
            )
        )


if __name__ == "__main__":
    unittest.main()
