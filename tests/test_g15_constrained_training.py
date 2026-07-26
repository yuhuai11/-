from __future__ import annotations

import unittest

import pandas as pd
import torch

from dads_crnn.train_g15_constrained import (
    PROTOCOL_V2,
    build_epoch_plan,
    final_decision,
    gradients_are_finite,
    update_safe_early_stopping,
)


class G15ConstrainedTrainingTests(unittest.TestCase):
    def test_safe_g14_improvement_resets_patience_before_promotion_gate(self) -> None:
        best, bad_epochs, improved = update_safe_early_stopping(
            dads_safe=True,
            g14_score=0.9912,
            best_safe_score=0.9910,
            bad_epochs=2,
        )
        self.assertTrue(improved)
        self.assertEqual(best, 0.9912)
        self.assertEqual(bad_epochs, 0)

    def test_unsafe_or_flat_epoch_increments_patience(self) -> None:
        for dads_safe, score in ((False, 0.9920), (True, 0.9910)):
            best, bad_epochs, improved = update_safe_early_stopping(
                dads_safe=dads_safe,
                g14_score=score,
                best_safe_score=0.9910,
                bad_epochs=1,
            )
            self.assertFalse(improved)
            self.assertEqual(best, 0.9910)
            self.assertEqual(bad_epochs, 2)

    def test_v2_noneligible_safe_history_has_accurate_decision(self) -> None:
        decision = final_decision(
            protocol=PROTOCOL_V2,
            best_checkpoint_exists=False,
            history=[
                {
                    "dads_floor_passed": True,
                    "g14_gain_passed": False,
                    "checkpoint_eligible": False,
                }
            ],
        )
        self.assertEqual(decision, "terminate_no_dual_gate_checkpoint")

    def test_gradient_finiteness_rejects_inf_and_missing_gradients(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        parameter.grad = torch.tensor([0.5])
        self.assertTrue(gradients_are_finite({"weight": parameter}))
        parameter.grad = torch.tensor([float("inf")])
        self.assertFalse(gradients_are_finite({"weight": parameter}))
        parameter.grad = None
        self.assertFalse(gradients_are_finite({"weight": parameter}))

    def test_epoch_plan_has_conservative_fixed_composition(self) -> None:
        frames = {
            "dads_replay": pd.DataFrame(
                {
                    "label": [0] * 100 + [1] * 100,
                    "source_group": [f"d-{index // 5}" for index in range(200)],
                }
            ),
            "g9_mechanical_hard_negative": pd.DataFrame(
                {
                    "label": [0] * 40,
                    "source_group": [f"g-{index // 2}" for index in range(40)],
                    "hard_negative_class": [
                        name
                        for name in ("chainsaw", "engine", "vacuum", "washing")
                        for _ in range(10)
                    ],
                }
            ),
            "kielce_uav": pd.DataFrame(
                {"label": [1] * 100, "source_group": ["k"] * 100}
            ),
            "tau_background": pd.DataFrame(
                {"label": [0] * 100, "source_group": ["t"] * 100}
            ),
        }
        composition = {
            "dads_replay": 60,
            "g9_mechanical_hard_negative": 4,
            "kielce_uav": 32,
            "tau_background": 32,
        }
        first = build_epoch_plan(
            frames, batches=3, composition=composition, seed=42, epoch=1
        )
        second = build_epoch_plan(
            frames, batches=3, composition=composition, seed=42, epoch=1
        )
        self.assertEqual(
            {name: value.shape for name, value in first.items()},
            {
                "dads_replay": (3, 60),
                "g9_mechanical_hard_negative": (3, 4),
                "kielce_uav": (3, 32),
                "tau_background": (3, 32),
            },
        )
        for name in first:
            self.assertTrue((first[name] == second[name]).all())
        for row in first["dads_replay"]:
            labels = frames["dads_replay"].iloc[row]["label"].value_counts().to_dict()
            self.assertEqual(labels, {0: 30, 1: 30})
        for row in first["g9_mechanical_hard_negative"]:
            classes = set(
                frames["g9_mechanical_hard_negative"].iloc[row][
                    "hard_negative_class"
                ]
            )
            self.assertEqual(len(classes), 4)


if __name__ == "__main__":
    unittest.main()
