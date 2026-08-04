import unittest

from dads_crnn.data_firewall import reject_locked_path


class G7R5SplitTests(unittest.TestCase):
    def test_locked_test_path_is_rejected_by_development_firewall(self):
        with self.assertRaisesRegex(ValueError, "Locked final-test path"):
            reject_locked_path(
                __import__("pathlib").Path(
                    "artifacts/g7_r5_train_val_test/locked_unseen_external_test/manifest.csv"
                )
            )


if __name__ == "__main__":
    unittest.main()
