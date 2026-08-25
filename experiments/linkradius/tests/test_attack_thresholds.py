from __future__ import annotations

import unittest

from experiments.linkradius.aggregate_attack_thresholds import (
    aggregate_thresholds,
    first_crossing,
)
from experiments.linkradius.schemas import ContractError


def _row(
    epsilon: float,
    margin: float,
    *,
    raw_id: str = "raw-1",
    edge: str = "p2c@0",
    family: str = "pgd_autograd",
    **extra: object,
) -> dict[str, object]:
    return {
        "record_type": "sample",
        "raw_sample_id": raw_id,
        "edge_id": edge,
        "attack_family": family,
        "requested_epsilon": epsilon,
        "minimum_margin": margin,
        **extra,
    }


class AttackThresholdTests(unittest.TestCase):
    def test_first_budget_crossing_uses_clean_zero_lower_bound(self) -> None:
        result = first_crossing(
            [_row(0.1, -0.1), _row(0.2, -0.2)],
            requested_budget_grid=(0.1, 0.2),
        )
        self.assertEqual(result["crossing_status"], "interval_crossed")
        self.assertEqual(result["requested_interval_lower"], 0.0)
        self.assertEqual(result["requested_interval_upper"], 0.1)
        self.assertTrue(result["implicit_clean_epsilon_zero"])

    def test_later_crossing_reports_requested_and_realized_interval(self) -> None:
        result = first_crossing(
            [
                _row(0.1, 1.0, realized_epsilon=0.09),
                _row(
                    0.2,
                    -0.1,
                    realized_epsilon=0.19,
                    binding_competitor="B",
                ),
                _row(0.3, -0.2, realized_epsilon=0.29),
            ],
            requested_budget_grid=(0.1, 0.2, 0.3),
        )
        self.assertEqual(result["first_scored_crossing"], 0.2)
        self.assertEqual(result["binding_competitor"], "B")
        self.assertEqual(result["requested_interval_lower"], 0.1)
        self.assertEqual(result["requested_interval_upper"], 0.2)
        self.assertEqual(result["realized_interval_lower"], 0.09)
        self.assertEqual(result["realized_interval_upper"], 0.19)

    def test_right_censoring_is_beyond_maximum_budget(self) -> None:
        result = first_crossing(
            [_row(0.1, 1.0), _row(0.4, 0.1)],
            requested_budget_grid=(0.1, 0.4),
        )
        self.assertEqual(result["crossing_status"], "right_censored")
        self.assertEqual(result["requested_interval_lower"], 0.4)
        self.assertIsNone(result["requested_interval_upper"])
        self.assertEqual(result["right_censoring_limit"], 0.4)
        self.assertIsNone(result["first_scored_crossing"])

    def test_explicit_failed_zero_budget_is_left_censored(self) -> None:
        result = first_crossing(
            [_row(0.0, -0.1), _row(0.1, -0.2)],
            requested_budget_grid=(0.0, 0.1),
        )
        self.assertEqual(result["crossing_status"], "left_censored")
        self.assertIsNone(result["requested_interval_lower"])
        self.assertEqual(result["requested_interval_upper"], 0.0)

    def test_score_tie_is_crossing_even_with_nonzero_tolerance(self) -> None:
        result = first_crossing(
            [_row(0.1, 0.01), _row(0.2, 0.0)],
            requested_budget_grid=(0.1, 0.2),
            tie_tolerance=0.5,
        )
        self.assertEqual(result["first_scored_crossing"], 0.2)

    def test_reentry_count_and_nonmonotonic_flag(self) -> None:
        result = first_crossing(
            [
                _row(0.1, -1.0),
                _row(0.2, 1.0),
                _row(0.3, -1.0),
                _row(0.4, 1.0),
            ],
            requested_budget_grid=(0.1, 0.2, 0.3, 0.4),
        )
        self.assertEqual(result["reentry_count"], 2)
        self.assertTrue(result["nonmonotonic"])
        self.assertTrue(result["non_monotonic_recovery"])

    def test_duplicate_requested_budget_rejected(self) -> None:
        with self.assertRaisesRegex(ContractError, "duplicate"):
            first_crossing([_row(0.1, 1.0), _row(0.1, -1.0)])

    def test_partial_realized_budget_curve_is_rejected(self) -> None:
        with self.assertRaisesRegex(ContractError, "complete curve"):
            first_crossing(
                [
                    _row(0.1, 1.0, realized_epsilon=0.09),
                    _row(0.2, -1.0),
                ],
                requested_budget_grid=(0.1, 0.2),
            )

    def test_nonincreasing_realized_curve_keeps_requested_interval(self) -> None:
        result = first_crossing(
            [
                _row(0.1, 1.0, realized_epsilon=0.09),
                _row(0.2, -1.0, realized_epsilon=0.09),
            ],
            requested_budget_grid=(0.1, 0.2),
        )
        self.assertEqual(result["requested_interval_lower"], 0.1)
        self.assertEqual(result["requested_interval_upper"], 0.2)
        self.assertTrue(result["realized_interval_available"])
        self.assertEqual(result["realized_interval_lower"], 0.0)
        self.assertEqual(result["realized_interval_upper"], 0.09)
        self.assertEqual(
            result["realized_grid_status"], "nonincreasing_or_collapsed"
        )

    def test_missing_frozen_budget_rejected(self) -> None:
        with self.assertRaisesRegex(ContractError, "missing=\\[0.2\\]"):
            first_crossing(
                [_row(0.1, 1.0), _row(0.3, -1.0)],
                requested_budget_grid=(0.1, 0.2, 0.3),
            )

    def test_seed_identity_keeps_curves_separate(self) -> None:
        rows = [
            _row(epsilon, margin, attack_seed=seed)
            for seed, margins in ((11, (1.0, -1.0)), (12, (1.0, 1.0)))
            for epsilon, margin in zip((0.1, 0.2), margins)
        ]
        output = aggregate_thresholds(
            rows, requested_budget_grid=(0.1, 0.2)
        )
        self.assertEqual(len(output), 2)
        self.assertEqual({row["attack_seed"] for row in output}, {11, 12})
        by_seed = {row["attack_seed"]: row for row in output}
        self.assertEqual(by_seed[11]["crossing_status"], "interval_crossed")
        self.assertEqual(by_seed[12]["crossing_status"], "right_censored")

    def test_seeded_random_and_unseeded_pgd_can_coexist(self) -> None:
        rows = [
            _row(epsilon, margin, family="pgd_autograd")
            for epsilon, margin in ((0.1, 1.0), (0.2, -1.0))
        ] + [
            _row(
                epsilon,
                margin,
                family="random_independent",
                attack_seed=91,
            )
            for epsilon, margin in ((0.1, 1.0), (0.2, 1.0))
        ]
        output = aggregate_thresholds(
            rows, requested_budget_grid=(0.1, 0.2)
        )
        self.assertEqual(len(output), 2)
        by_family = {row["attack_family"]: row for row in output}
        self.assertNotIn("attack_seed", by_family["pgd_autograd"])
        self.assertEqual(by_family["random_independent"]["attack_seed"], 91)

    def test_competitor_targets_do_not_collide_with_edge_summary(self) -> None:
        rows: list[dict[str, object]] = [
            _row(0.1, 1.0),
            _row(0.2, -1.0),
        ]
        for target, margins in (("B", (0.5, -0.5)), ("C", (0.6, 0.1))):
            for epsilon, margin in zip((0.1, 0.2), margins):
                target_row = _row(epsilon, margin)
                target_row.update(
                    {
                        "record_type": "attack_target",
                        "target_label": target,
                        "target_margin": margin,
                    }
                )
                target_row.pop("minimum_margin")
                rows.append(target_row)

        output = aggregate_thresholds(
            rows, requested_budget_grid=(0.1, 0.2)
        )
        self.assertEqual(len(output), 3)
        summaries = [row for row in output if row["curve_kind"] == "edge_summary"]
        targets = [row for row in output if row["curve_kind"] == "attack_target"]
        self.assertEqual(len(summaries), 1)
        self.assertEqual({row["target_label"] for row in targets}, {"B", "C"})

    def test_missing_budget_in_one_seed_curve_rejected(self) -> None:
        rows = [
            _row(0.1, 1.0, attack_seed=11),
            _row(0.2, -1.0, attack_seed=11),
            _row(0.1, 1.0, attack_seed=12),
        ]
        with self.assertRaisesRegex(ContractError, "missing=\\[0.2\\]"):
            aggregate_thresholds(rows, requested_budget_grid=(0.1, 0.2))

    def test_failed_or_unsupported_rows_are_not_silently_dropped(self) -> None:
        for bad in (
            _row(0.1, 1.0, failure="oom"),
            {
                **_row(0.1, 1.0),
                "record_type": "unsupported",
                "unsupported_reason": "no kernel",
            },
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ContractError):
                    aggregate_thresholds([bad], requested_budget_grid=(0.1,))


if __name__ == "__main__":
    unittest.main()
