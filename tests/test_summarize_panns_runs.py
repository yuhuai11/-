from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dads_crnn.summarize_panns_runs import collect_runs, write_summary


class SummarizePannsRunsTests(unittest.TestCase):
    def _write_metrics(self, root: Path, seed: int, identity: dict) -> None:
        run_dir = root / f"seed_{seed}"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "training_inputs": identity,
                    "locked_datasets_read": [],
                }
            ),
            encoding="utf-8",
        )

    def test_collects_ordered_runs_and_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (42, 43, 44):
                self._write_metrics(root, seed, {"manifest_sha256": "same"})
            results = collect_runs(root, [42, 43, 44])
            self.assertEqual([row["seed"] for row in results], [42, 43, 44])
            path = write_summary(root, results)
            self.assertEqual(
                [row["seed"] for row in json.loads(path.read_text(encoding="utf-8"))],
                [42, 43, 44],
            )

    def test_rejects_mismatched_training_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_metrics(root, 42, {"manifest_sha256": "a"})
            self._write_metrics(root, 43, {"manifest_sha256": "b"})
            with self.assertRaises(ValueError):
                collect_runs(root, [42, 43])

    def test_rejects_locked_dataset_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_metrics(root, 42, {"manifest_sha256": "same"})
            path = root / "seed_42" / "metrics.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["locked_datasets_read"] = ["unseen"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                collect_runs(root, [42])


if __name__ == "__main__":
    unittest.main()
