from __future__ import annotations

import math
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - lightweight CPU Python
    torch = None

from RecursiveMAS.inference_utils.linkradius import (
    PerturbationSubspace,
    sample_stable_lifted_direction,
    sample_stable_unit_direction,
)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ProbeDirectionTests(unittest.TestCase):
    def test_subspaces_are_isometric_with_declared_q(self) -> None:
        for name, expected_q in (("full_tensor", 15), ("channel_broadcast", 5)):
            with self.subTest(name=name):
                subspace = PerturbationSubspace(name, 3, 5)
                coefficients = torch.randn(expected_q, dtype=torch.float32)
                lifted = subspace.lift(coefficients)
                self.assertEqual(subspace.q, expected_q)
                self.assertEqual(tuple(lifted.shape), (3, 5))
                self.assertAlmostEqual(
                    float(torch.linalg.vector_norm(coefficients)),
                    float(torch.linalg.vector_norm(lifted)),
                    places=5,
                )

    def test_channel_broadcast_adjoint_matches_inner_product(self) -> None:
        subspace = PerturbationSubspace("channel_broadcast", 4, 3)
        coefficients = torch.tensor([1.0, -2.0, 0.5])
        relay_gradient = torch.tensor(
            [
                [1.0, 0.0, 2.0],
                [0.0, 3.0, -1.0],
                [2.0, -1.0, 0.0],
                [-2.0, 1.0, 4.0],
            ]
        )
        lifted_inner = torch.sum(subspace.lift(coefficients) * relay_gradient)
        coordinate_inner = torch.sum(
            coefficients * subspace.adjoint(relay_gradient)
        )
        self.assertAlmostEqual(float(lifted_inner), float(coordinate_inner), places=6)

    def test_direction_is_sample_id_stable(self) -> None:
        subspace = PerturbationSubspace("full_tensor", 3, 4)
        kwargs = dict(
            global_seed=42,
            raw_sample_id="stable-raw-id",
            edge="p2c@0",
            subspace=subspace,
            probe_seed=17,
            direction_id=6,
        )
        first = sample_stable_unit_direction(**kwargs)
        # Batch order/shard/batch size are deliberately absent from the seed API.
        _ = sample_stable_unit_direction(42, "other", "p2c@0", subspace, 17, 6)
        second = sample_stable_unit_direction(**kwargs)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.device.type, "cpu")
        self.assertEqual(first.dtype, torch.float32)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(first)), 1.0, places=6)

    def test_lifted_plus_and_minus_reuse_one_direction(self) -> None:
        subspace = PerturbationSubspace("channel_broadcast", 4, 7)
        lifted = sample_stable_lifted_direction(1, "x", "c2s@1", subspace, 2, 3)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(lifted)), 1.0, places=6)
        self.assertTrue(torch.equal(-lifted, lifted * -1))


if __name__ == "__main__":
    unittest.main()
