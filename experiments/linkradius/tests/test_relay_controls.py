from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from RecursiveMAS.inference_utils.linkradius import (
    ProbeAcceptanceThresholds,
    additive_intervention,
    deterministic_donor_mapping,
    identity_intervention,
    mismatch_intervention,
    moment_noise_intervention,
    probe_pair_diagnostics,
    realized_delta_diagnostics,
    requested_additive_delta,
    zero_intervention,
)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class RelayControlTests(unittest.TestCase):
    def test_identity_zero_and_additive(self) -> None:
        z = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        delta = torch.ones_like(z)
        self.assertTrue(torch.equal(identity_intervention(z), z))
        self.assertTrue(torch.equal(zero_intervention(z), torch.zeros_like(z)))
        self.assertTrue(torch.equal(additive_intervention(z, delta), z + 1))

    def test_donor_mapping_is_balanced_nonself_and_strictly_stratified(self) -> None:
        records = []
        for label, ids in (("A", ("a1", "a2", "a3")), ("B", ("b1", "b2"))):
            for raw_id in ids:
                records.append({
                    "raw_sample_id": raw_id,
                    "partition": "validation",
                    "gold_label": label,
                    "edge_id": "p2c@0",
                    "R": 2,
                    "tensor_shape": [32, 8],
                    "length_bucket": 32,
                })
        mapping = deterministic_donor_mapping(records, donor_seed=99)
        reverse_mapping = deterministic_donor_mapping(list(reversed(records)), donor_seed=99)
        self.assertEqual(mapping, reverse_mapping)
        by_label = {item["raw_sample_id"]: item["gold_label"] for item in records}
        for recipient, donor in mapping.items():
            self.assertIsNotNone(donor)
            self.assertNotEqual(recipient, donor)
            self.assertEqual(by_label[recipient], by_label[donor])
        self.assertEqual(set(mapping.values()), set(mapping))  # cyclic and balanced

    def test_singleton_donor_is_unavailable(self) -> None:
        row = {
            "raw_sample_id": "only",
            "partition": "test",
            "gold": "C",
            "site": "c2s",
            "round_idx": 1,
            "horizon": 2,
            "shape": (2, 3),
        }
        self.assertEqual(deterministic_donor_mapping([row]), {"only": None})

    def test_mismatch_rescales_and_zero_norm_is_unavailable(self) -> None:
        recipient = torch.tensor([[3.0, 4.0]])
        donor = torch.tensor([[0.0, 2.0]])
        requested, meta = mismatch_intervention(recipient, donor)
        self.assertTrue(meta.available)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(requested)), 5.0)
        unavailable, zero_meta = mismatch_intervention(recipient, torch.zeros_like(donor))
        self.assertIsNone(unavailable)
        self.assertEqual(zero_meta.reason, "donor_zero_norm")

    def test_moment_noise_matches_each_recipients_global_moments(self) -> None:
        z = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        first, meta = moment_noise_intervention(
            z, global_seed=42, raw_sample_id="raw", edge="p2c@0", return_diagnostics=True
        )
        second = moment_noise_intervention(z, global_seed=42, raw_sample_id="raw", edge="p2c@0")
        self.assertTrue(torch.equal(first, second))
        self.assertAlmostEqual(float(first.mean()), float(z.mean()), places=6)
        self.assertAlmostEqual(float(first.std(correction=0)), float(z.std(correction=0)), places=5)
        self.assertAlmostEqual(meta.requested_norm, float(torch.linalg.vector_norm(first)), places=5)

        constant = torch.full((3, 4), 7.0)
        noise, constant_meta = moment_noise_intervention(
            constant, global_seed=0, raw_sample_id="constant", edge="c2s@0", return_diagnostics=True
        )
        self.assertTrue(constant_meta.zero_variance)
        self.assertTrue(torch.equal(noise, constant))

    def test_realized_coordinates_use_cast_separation_not_requested_2h(self) -> None:
        z = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        direction = torch.ones_like(z)
        direction /= torch.linalg.vector_norm(direction)
        h = 0.01
        plus_delta = requested_additive_delta(z, direction, h, +1)
        minus_delta = requested_additive_delta(z, direction, h, -1)
        plus = realized_delta_diagnostics(
            z, plus_delta, consumer_dtype=torch.bfloat16, lifted_unit_direction=direction
        )
        minus = realized_delta_diagnostics(
            z, minus_delta, consumer_dtype=torch.bfloat16, lifted_unit_direction=direction
        )
        pair = probe_pair_diagnostics(plus, minus, ProbeAcceptanceThresholds())
        self.assertTrue(pair.accepted)
        self.assertAlmostEqual(
            pair.realized_separation,
            plus.realized_signed_coordinate - minus.realized_signed_coordinate,
        )
        self.assertNotAlmostEqual(pair.realized_separation, 2 * h, places=6)

    def test_bfloat16_cast_collapse_is_rejected(self) -> None:
        z = torch.ones((2, 2), dtype=torch.float32)
        direction = torch.ones_like(z) / 2
        plus = realized_delta_diagnostics(
            z,
            requested_additive_delta(z, direction, 1e-8, +1),
            consumer_dtype=torch.bfloat16,
            lifted_unit_direction=direction,
        )
        minus = realized_delta_diagnostics(
            z,
            requested_additive_delta(z, direction, 1e-8, -1),
            consumer_dtype=torch.bfloat16,
            lifted_unit_direction=direction,
        )
        self.assertTrue(plus.collapsed)
        pair = probe_pair_diagnostics(plus, minus)
        self.assertFalse(pair.accepted)
        self.assertIn("plus_collapsed", pair.rejection_reasons)


if __name__ == "__main__":
    unittest.main()

