from __future__ import annotations

import unittest

import torch

from dads_crnn.panns import FrequencyMixStyle


class FrequencyMixStyleTests(unittest.TestCase):
    def test_eval_is_exact_identity(self) -> None:
        module = FrequencyMixStyle(probability=1.0, beta_alpha=0.6).eval()
        features = torch.randn(4, 3, 5, 8)
        self.assertTrue(torch.equal(module(features), features))
        self.assertFalse(module.last_applied)

    def test_forced_training_mix_preserves_shape_and_gradients(self) -> None:
        torch.manual_seed(42)
        module = FrequencyMixStyle(probability=0.0, beta_alpha=0.6).train()
        features = torch.randn(4, 3, 5, 8, requires_grad=True)
        mixed = module(features, force=True)
        self.assertEqual(mixed.shape, features.shape)
        self.assertTrue(torch.isfinite(mixed).all())
        self.assertFalse(torch.equal(mixed, features))
        mixed.square().mean().backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_single_item_batch_is_identity(self) -> None:
        module = FrequencyMixStyle(probability=1.0, beta_alpha=0.6).train()
        features = torch.randn(1, 2, 3, 8)
        self.assertTrue(torch.equal(module(features), features))

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FrequencyMixStyle(probability=1.1, beta_alpha=0.6)
        with self.assertRaises(ValueError):
            FrequencyMixStyle(probability=0.5, beta_alpha=0.0)


if __name__ == "__main__":
    unittest.main()
