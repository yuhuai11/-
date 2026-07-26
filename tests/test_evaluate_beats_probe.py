import unittest

import numpy as np
import pandas as pd

from dads_crnn.evaluate_beats_probe import source_macro_metric


class EvaluateBeatsProbeTests(unittest.TestCase):
    def test_source_macro_weights_sources_equally(self) -> None:
        rows = pd.DataFrame(
            {
                "label": [1, 1, 1, 1],
                "uav_source": ["large", "large", "large", "small"],
            }
        )
        predictions = np.array([1, 1, 1, 0])
        result = source_macro_metric(
            rows, predictions, label=1, group_field="uav_source"
        )
        self.assertEqual(result["groups"], 2)
        self.assertEqual(result["macro"], 0.5)


if __name__ == "__main__":
    unittest.main()
