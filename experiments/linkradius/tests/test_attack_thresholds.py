from __future__ import annotations

import unittest

from experiments.linkradius.aggregate_attack_thresholds import first_crossing
from experiments.linkradius.schemas import ContractError


class AttackThresholdTests(unittest.TestCase):
    def test_first_exact_crossing_and_recovery(self) -> None:
        result = first_crossing(
            [
                {"requested_epsilon": 0.1, "minimum_margin": 1.0},
                {"requested_epsilon": 0.2, "minimum_margin": 0.0, "binding_competitor": "B", "realized_epsilon": 0.19},
                {"requested_epsilon": 0.3, "minimum_margin": 0.2},
            ],
            tie_tolerance=0.5,
        )
        self.assertEqual(result["first_scored_crossing"], 0.2)
        self.assertEqual(result["binding_competitor"], "B")
        self.assertTrue(result["non_monotonic_recovery"])
        self.assertEqual(result["realized_epsilon_at_crossing"], 0.19)

    def test_positive_margin_inside_tolerance_is_not_crossing(self) -> None:
        result = first_crossing(
            [{"epsilon": 0.1, "minimum_margin": 0.01}], tie_tolerance=0.1
        )
        self.assertEqual(result["threshold_status"], "right_censored")
        self.assertIsNone(result["first_scored_crossing"])

    def test_right_censoring_and_invalid_generation_missing(self) -> None:
        result = first_crossing(
            [
                {"epsilon": 0.1, "minimum_margin": 1.0, "answer_invalid": True},
                {"epsilon": 0.4, "minimum_margin": 0.1, "answer_invalid": True},
            ]
        )
        self.assertEqual(result["right_censoring_limit"], 0.4)
        self.assertIsNone(result["first_generated_flip"])

    def test_duplicate_requested_budget_rejected(self) -> None:
        with self.assertRaises(ContractError):
            first_crossing(
                [
                    {"epsilon": 0.1, "minimum_margin": 1.0},
                    {"epsilon": 0.1, "minimum_margin": -1.0},
                ]
            )


if __name__ == "__main__":
    unittest.main()

