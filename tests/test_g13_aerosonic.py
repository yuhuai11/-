import unittest

from dads_crnn.prepare_g13_aerosonic import nonoverlap_clip_count


class AeroSonicPreparationTests(unittest.TestCase):
    def test_nonoverlap_clip_count_drops_short_tail(self) -> None:
        self.assertEqual(nonoverlap_clip_count(16000 * 10 + 15999), 10)
        self.assertEqual(nonoverlap_clip_count(15999), 0)


if __name__ == "__main__":
    unittest.main()
