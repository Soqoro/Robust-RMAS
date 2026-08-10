from __future__ import annotations

import math
import unittest

from RecursiveMAS.inference_utils.linkradius import (
    ProbePairObservation,
    central_difference,
    estimate_competitor_radii,
    estimate_probe_prefix,
)


class ProbeEstimatorTests(unittest.TestCase):
    def test_realized_coordinate_denominator(self) -> None:
        self.assertAlmostEqual(central_difference(3.0, 1.0, 0.3, -0.1), 5.0)
        with self.assertRaises(ValueError):
            central_difference(3.0, 1.0, 0.1, 0.1)

    def test_analytic_susceptibility_radius_and_binding_competitor(self) -> None:
        estimate = estimate_competitor_radii(
            {"B": 8.0, "C": 3.0, "D": 2.0},
            {
                "B": [2.0, -2.0],  # chi=sqrt(4/2 * 8)=4, rho=2
                "C": [0.0, 0.0],   # zero susceptibility => infinity
                "D": [0.5, -0.5],  # chi=1, rho=2
            },
            q=4,
        )
        self.assertAlmostEqual(estimate.competitors["B"].susceptibility, 4.0)
        self.assertAlmostEqual(estimate.competitors["B"].radius, 2.0)
        self.assertTrue(math.isinf(estimate.competitors["C"].radius))
        self.assertTrue(estimate.competitors["C"].zero_susceptibility)
        self.assertEqual(estimate.radius, 2.0)
        self.assertEqual(estimate.binding_competitors, ("B", "D"))
        self.assertEqual(estimate.binding_competitor, "B")

    def test_nonpositive_clean_margin_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            estimate_competitor_radii({"B": 0.0}, {"B": [1.0]}, q=2)

    def test_primary_requires_every_requested_prefix_pair(self) -> None:
        pairs = [
            ProbePairObservation(
                direction_id=0,
                t_plus=0.2,
                t_minus=-0.2,
                margins_plus={"B": 1.4},
                margins_minus={"B": 0.6},
                accepted=True,
            ),
            ProbePairObservation(
                direction_id=1,
                t_plus=0.2,
                t_minus=-0.2,
                margins_plus={"B": 1.0},
                margins_minus={"B": 1.0},
                accepted=False,
                rejection_reason="cast_collapse",
            ),
        ]
        result = estimate_probe_prefix({"B": 1.0}, pairs, q=2, requested_k=2)
        self.assertFalse(result.primary_available)
        self.assertIsNone(result.primary)
        self.assertIsNotNone(result.incomplete_sensitivity)
        self.assertEqual(result.k_eff, 1)
        self.assertEqual(result.rejected_direction_ids, (1,))

        complete = estimate_probe_prefix({"B": 1.0}, [pairs[0]], q=2, requested_k=1)
        self.assertTrue(complete.primary_available)
        self.assertIsNotNone(complete.primary)


if __name__ == "__main__":
    unittest.main()
