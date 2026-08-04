import unittest

from dads_crnn.prepare_g7_r6_dronenoise_splits import (
    assign_event_roles,
    parse_event_group,
)


class DroneNoiseSplitTests(unittest.TestCase):
    def test_parse_event_group(self) -> None:
        values = parse_event_group("Ed_M3_10_F15_N_E_dw_ev4")
        self.assertEqual(values["uav_subtype"], "M3")
        self.assertEqual(values["hagl_m"], 10)
        self.assertEqual(values["operation"], "F")
        self.assertEqual(values["speed"], 15)
        self.assertEqual(values["event_index"], 4)

    def test_each_type_has_disjoint_validation_and_test_event(self) -> None:
        events = [
            f"Ed_{kind}_10_F15_N_W_dw_ev{event}"
            for kind in ("A", "B", "C", "D")
            for event in range(1, 4)
        ]
        roles = assign_event_roles(events, seed=42)
        for kind in ("A", "B", "C", "D"):
            selected = {
                event: role for event, role in roles.items() if f"Ed_{kind}_" in event
            }
            self.assertEqual(list(selected.values()).count("train"), 1)
            self.assertEqual(list(selected.values()).count("validation"), 1)
            self.assertEqual(list(selected.values()).count("test"), 1)
        self.assertEqual(roles, assign_event_roles(events, seed=42))

    def test_validation_and_test_prefer_complete_microphone_events(self) -> None:
        events = [f"Ed_A_10_F15_N_W_dw_ev{event}" for event in range(1, 5)]
        counts = {events[0]: 1, events[1]: 9, events[2]: 9, events[3]: 9}
        roles = assign_event_roles(events, seed=42, recording_counts=counts)
        self.assertEqual(roles[events[0]], "train")
        held_out = [event for event in events if roles[event] in {"validation", "test"}]
        self.assertTrue(all(counts[event] == 9 for event in held_out))


if __name__ == "__main__":
    unittest.main()
