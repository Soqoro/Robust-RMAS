from __future__ import annotations

import unittest

from experiments.linkradius.build_system_curves import build_predicted_system_curves, epsilon50


class SystemCurveTests(unittest.TestCase):
    def test_worst_random_fixed_and_useful_are_distinct(self) -> None:
        rows = [
            {"raw_sample_id": "a", "edge_id": "p2c@0", "edge_radius": 1.0},
            {"raw_sample_id": "a", "edge_id": "c2s@0", "edge_radius": 3.0},
            {"raw_sample_id": "b", "edge_id": "p2c@0", "edge_radius": 2.0},
            {"raw_sample_id": "b", "edge_id": "c2s@0", "edge_radius": 4.0},
        ]
        curves, summaries = build_predicted_system_curves(
            rows,
            [0.0, 2.0, 4.0],
            fixed_edge="p2c@0",
            useful_edges=["c2s@0"],
        )
        at_two = {row["curve_type"]: row["vulnerability"] for row in curves if row["epsilon"] == 2.0}
        self.assertEqual(at_two["worst_accessible_site"], 1.0)
        self.assertEqual(at_two["uniform_random_site"], 0.5)
        self.assertEqual(at_two["fixed_validation_site"], 1.0)
        self.assertEqual(at_two["causally_useful_edge"], 0.0)
        self.assertEqual(
            {row["curve_type"] for row in summaries},
            {"worst_accessible_site", "uniform_random_site", "fixed_validation_site", "causally_useful_edge"},
        )

    def test_epsilon50_reports_censoring(self) -> None:
        value = epsilon50(
            [
                {"epsilon": 0.1, "vulnerability": 0.1},
                {"epsilon": 0.2, "vulnerability": 0.4},
            ]
        )
        self.assertEqual(value["epsilon50_status"], "not_reached_right_censored")
        self.assertEqual(value["epsilon50_censoring_limit"], 0.2)


if __name__ == "__main__":
    unittest.main()

