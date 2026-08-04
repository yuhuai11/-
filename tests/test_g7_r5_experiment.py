import unittest

from dads_crnn.g7_r5_experiment import load_experiment


class G7R5ExperimentTests(unittest.TestCase):
    def test_experiment_is_comparative_and_forbids_retest(self):
        config = load_experiment(
            __import__("pathlib").Path("configs/g7_r5_locked_test_experiment.yaml")
        )
        self.assertEqual(
            config["purpose"], "independent_comparison_not_test_set_model_selection"
        )
        self.assertFalse(config["post_test_policy"]["allow_repeat_on_same_locked_test"])
        self.assertEqual(
            config["systems"]["g7_r4c_ensemble"]["aggregation"],
            "uniform_probability_mean",
        )


if __name__ == "__main__":
    unittest.main()
