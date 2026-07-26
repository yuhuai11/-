from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import pandas as pd

from dads_crnn.prepare_g13_confirmation import build_manifest, validate_registry


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.zeros(16000, dtype=np.int16)
    with wave.open(path.as_posix(), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(audio.tobytes())


class G13ConfirmationTests(unittest.TestCase):
    def test_manifest_requires_registered_source_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = root / "source_registry.csv"
            pd.DataFrame(
                [
                    {
                        "source_group": "bg-new",
                        "label": "0",
                        "acquisition_id": "new-1",
                        "provenance": "field",
                        "independent_from_existing": "true",
                        "license": "research",
                    },
                    {
                        "source_group": "uav-new",
                        "label": "1",
                        "acquisition_id": "new-1",
                        "provenance": "field",
                        "independent_from_existing": "true",
                        "license": "research",
                    },
                ]
            ).to_csv(registry_path, index=False)
            _write_wav(root / "Background" / "bg-new" / "a.wav")
            _write_wav(root / "UAV" / "uav-new" / "a.wav")
            registry = validate_registry(registry_path, {0, 1}, True)
            manifest = build_manifest(
                root, "external_confirmation_v2", {"Background": 0, "UAV": 1}, registry
            )
            self.assertEqual(len(manifest), 2)
            self.assertEqual(set(manifest["source_group"]), {"bg-new", "uav-new"})

    def test_registry_rejects_nonindependent_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.csv"
            pd.DataFrame(
                [
                    {
                        "source_group": "old",
                        "label": "0",
                        "acquisition_id": "old-1",
                        "provenance": "existing",
                        "independent_from_existing": "false",
                        "license": "research",
                    }
                ]
            ).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                validate_registry(path, {0, 1}, True)


if __name__ == "__main__":
    unittest.main()
