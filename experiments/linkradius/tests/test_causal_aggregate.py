from __future__ import annotations

import unittest

from experiments.linkradius.aggregate_causal_use import (
    aggregate_causal_rows,
    eligible_complete_causal_rows,
)
from experiments.linkradius.schemas import ContractError


def _row(raw_id: str, mode: str, margin: float, *, eligible: bool) -> dict:
    return {
        "record_type": "sample",
        "raw_sample_id": raw_id,
        "edge_id": "p2c@0",
        "intervention_mode": mode,
        "minimum_margin": margin,
        "scorer_correct": margin > 0,
        "analysis_eligible": eligible,
    }


class CausalAggregateTests(unittest.TestCase):
    def test_ineligible_opposite_effect_cannot_change_summary(self) -> None:
        rows = [
            _row("eligible", "identity", 1.0, eligible=True),
            _row("eligible", "mismatch", 0.5, eligible=True),
            _row("filler", "identity", -10.0, eligible=False),
            _row("filler", "mismatch", 10.0, eligible=False),
        ]
        selected = eligible_complete_causal_rows(
            rows,
            expected_edges=("p2c@0",),
            expected_modes=("identity", "mismatch"),
        )
        self.assertEqual({row["raw_sample_id"] for row in selected}, {"eligible"})
        _, summary = aggregate_causal_rows(selected, bootstrap_draws=10)
        self.assertEqual(summary[0]["paired_gold_margin_effect"], 0.5)

    def test_missing_eligible_control_fails_closed(self) -> None:
        with self.assertRaises(ContractError):
            eligible_complete_causal_rows(
                [_row("eligible", "identity", 1.0, eligible=True)],
                expected_edges=("p2c@0",),
                expected_modes=("identity", "mismatch"),
            )

    def test_entire_missing_eligible_sample_fails_frozen_coverage(self) -> None:
        rows = [
            _row("observed", "identity", 1.0, eligible=True),
            _row("observed", "mismatch", 0.5, eligible=True),
        ]
        with self.assertRaises(ContractError):
            eligible_complete_causal_rows(
                rows,
                expected_edges=("p2c@0",),
                expected_modes=("identity", "mismatch"),
                expected_raw_ids=("observed", "entirely-missing"),
            )


if __name__ == "__main__":
    unittest.main()
