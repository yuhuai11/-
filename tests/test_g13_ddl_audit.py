import unittest

from dads_crnn.audit_ddl_real import FILENAME, candidate_clip_count


class DDLRealAuditTests(unittest.TestCase):
    def test_actual_filename_layout(self) -> None:
        name = "20210329141240MINI0030240312886R290321-T004-005236.wav"
        parsed = FILENAME.match(name)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.group("class_name"), "MINI")
        self.assertEqual(parsed.group("session"), "T004")
        self.assertEqual(parsed.group("sequence"), "005236")

    def test_candidate_clips_do_not_cross_gaps(self) -> None:
        clips, runs = candidate_clip_count(list(range(20)) + list(range(30, 45)))
        self.assertEqual(clips, 3)
        self.assertEqual(runs, 2)


if __name__ == "__main__":
    unittest.main()
