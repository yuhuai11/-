from __future__ import annotations

import unittest

from dads_crnn.audit_g7_r6_dronenoise import EVENT_PATTERN


class G7R6DroneNoiseTest(unittest.TestCase):
    def test_event_and_microphone_are_separated(self) -> None:
        match = EVENT_PATTERN.match("Ed_M3_10_F15_N_E_dw_ev4_M7.wav")
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.group("event"), "Ed_M3_10_F15_N_E_dw_ev4")
        self.assertEqual(match.group("microphone"), "7")

    def test_calibration_name_is_not_an_event_microphone(self) -> None:
        self.assertIsNone(EVENT_PATTERN.match("Calib_Scotland_ch1.wav"))


if __name__ == "__main__":
    unittest.main()
