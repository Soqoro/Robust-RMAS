from __future__ import annotations

import copy
import math
import unittest

from experiments.linkradius.assemble_failure_boundary import (
    assemble_failure_boundary,
)
from experiments.linkradius.schemas import ContractError


PROVENANCE = {
    "partition": "test",
    "source_hash": "1" * 64,
    "split_manifest_hash": "2" * 64,
    "execution_manifest_hash": "3" * 64,
    "model_hash": "4" * 64,
    "scorer_hash": "5" * 64,
    "subspace_hash": "6" * 64,
}


def _clean(
    raw_id: str,
    *,
    generated_correct: bool = True,
    scorer_correct: bool = True,
) -> dict:
    gold = "A"
    return {
        "record_type": "sample",
        "raw_sample_id": raw_id,
        "analysis_eligible": True,
        "gold": gold,
        "strict_generated_choice": gold if generated_correct else "B",
        "strict_generated_valid": True,
        "answer_invalid": False,
        "answer_conflict": False,
        "scorer_prediction": gold if scorer_correct else "B",
        "scorer_numerically_valid": True,
        "score_tie": False,
        "option_scores": {
            "A": 6.0 if scorer_correct else 1.0,
            "B": 1.0 if scorer_correct else 6.0,
            "C": 0.0,
            "D": -1.0,
        },
        "margins": {"B": 2.0, "C": 4.0, "D": 6.0},
        **PROVENANCE,
    }


def _pairs(
    raw_id: str,
    *,
    directions: tuple[int, ...] = (0, 1),
    seeds: tuple[int, ...] = (101, 202, 303),
) -> list[dict]:
    return [
        {
            "record_type": "probe_pair",
            "raw_sample_id": raw_id,
            "edge_id": "p2c@0",
            "direction_id": direction,
            "probe_seed": seed,
            "h": 0.001,
            "q": 2,
            "accepted": True,
            "clean_margins": {"B": 2.0, "C": 4.0, "D": 6.0},
            "central_differences": {"B": 1.0, "C": 0.5, "D": 0.25},
            **PROVENANCE,
        }
        for seed in seeds
        for direction in directions
    ]


def _attacks(
    raw_id: str,
    *,
    families: tuple[str, ...] = ("pgd_autograd", "random_independent"),
    budgets: tuple[float, ...] = (0.1, 0.2),
) -> list[dict]:
    return [
        {
            "record_type": "sample",
            "run_id": f"{raw_id}-{family}-{budget}",
            "raw_sample_id": raw_id,
            "edge_id": "p2c@0",
            "attack_family": family,
            "requested_epsilon": budget,
            "realized_epsilon": budget * 0.99,
            "minimum_margin": 0.0 if budget == 0.2 else 0.5,
            **PROVENANCE,
        }
        for family in families
        for budget in budgets
    ]


def _assemble(clean, pairs, attacks):
    if isinstance(clean, dict):
        clean = [clean]
    return assemble_failure_boundary(
        clean,
        pairs,
        attacks,
        frozen_edges=("p2c@0",),
        frozen_budgets=(0.1, 0.2),
        frozen_families=("pgd_autograd", "random_independent"),
        requested_K=2,
        selected_h=0.001,
        probe_seeds=(101, 202, 303),
    )


class AssembleFailureBoundaryTests(unittest.TestCase):
    def test_happy_synthetic_cube(self) -> None:
        result = _assemble(_clean("raw-1"), _pairs("raw-1"), _attacks("raw-1"))

        self.assertEqual(result["counts"]["edge_predictors"], 3)
        self.assertEqual(result["counts"]["prediction_units"], 12)
        predictor = result["edge_predictors"][0]
        self.assertAlmostEqual(predictor["edge_radius"], math.sqrt(2.0))
        self.assertEqual(predictor["minimum_clean_margin"], 2.0)
        self.assertAlmostEqual(predictor["maximum_susceptibility"], math.sqrt(2.0))
        self.assertEqual(
            {row["attack_family"] for row in result["prediction_units"]},
            {"pgd_autograd", "random_independent"},
        )
        self.assertEqual(
            {row["requested_epsilon"] for row in result["prediction_units"]},
            {0.1, 0.2},
        )
        self.assertEqual(
            {row["probe_seed"] for row in result["prediction_units"]},
            {101, 202, 303},
        )
        for row in result["prediction_units"]:
            self.assertEqual(row["clean_margin"], row["minimum_clean_margin"])
            self.assertEqual(row["susceptibility"], row["maximum_susceptibility"])
            self.assertAlmostEqual(
                row["linkradius_score"],
                row["requested_epsilon"] / row["edge_radius"],
            )
            self.assertEqual(
                row["linkradius_score"], row["linkradius_requested_score"]
            )
            self.assertAlmostEqual(
                row["linkradius_realized_score"],
                row["realized_epsilon"] / row["edge_radius"],
            )
            self.assertNotEqual(
                row["linkradius_score"], row["linkradius_realized_score"]
            )

    def test_attack_cube_rejects_missing_and_duplicate_coordinates(self) -> None:
        attacks = _attacks("raw-1")
        with self.assertRaisesRegex(ContractError, "cube is incomplete"):
            _assemble(_clean("raw-1"), _pairs("raw-1"), attacks[:-1])

        with self.assertRaisesRegex(ContractError, "duplicate attack cube"):
            _assemble(
                _clean("raw-1"),
                _pairs("raw-1"),
                attacks + [copy.deepcopy(attacks[0])],
            )

    def test_rejects_split_and_source_provenance_mismatches(self) -> None:
        for field in ("partition", "source_hash", "split_manifest_hash"):
            with self.subTest(field=field):
                pairs = _pairs("raw-1")
                pairs[0][field] = "validation" if field == "partition" else "f" * 64
                with self.assertRaisesRegex(ContractError, "provenance mismatch"):
                    _assemble(_clean("raw-1"), pairs, _attacks("raw-1"))

    def test_generation_is_diagnostic_not_an_inclusion_requirement(self) -> None:
        generation_wrong = _clean("generation-wrong", generated_correct=False)
        generation_wrong.update(
            {
                "strict_generated_valid": False,
                "answer_invalid": True,
                "answer_conflict": True,
            }
        )
        result = _assemble(
            generation_wrong,
            _pairs("generation-wrong"),
            _attacks("generation-wrong"),
        )

        self.assertEqual(result["eligible_raw_sample_ids"], ["generation-wrong"])
        self.assertEqual(result["excluded_raw_sample_ids"], [])
        self.assertEqual(
            {row["raw_sample_id"] for row in result["prediction_units"]},
            {"generation-wrong"},
        )

    def test_dual_correct_policy_excludes_generation_wrong_rows(self) -> None:
        generation_wrong = _clean("generation-wrong", generated_correct=False)
        good = _clean("good")
        result = assemble_failure_boundary(
            [generation_wrong, good],
            [*_pairs("generation-wrong"), *_pairs("good")],
            [*_attacks("generation-wrong"), *_attacks("good")],
            frozen_edges=("p2c@0",),
            frozen_budgets=(0.1, 0.2),
            frozen_families=("pgd_autograd", "random_independent"),
            requested_K=2,
            selected_h=0.001,
            probe_seeds=(101, 202, 303),
            clean_correct_policy="dual_correct",
        )

        self.assertEqual(result["eligible_raw_sample_ids"], ["good"])
        self.assertEqual(
            result["frozen_configuration"]["clean_correct_policy"],
            "dual_correct",
        )

    def test_filters_rows_without_valid_unique_correct_forced_margins(self) -> None:
        bad_rows = {
            "analysis-ineligible": {"analysis_eligible": False},
            "scorer-wrong": {"scorer_prediction": "B"},
            "scorer-nonfinite": {"scorer_numerically_valid": False},
            "score-tie": {"score_tie": True},
            "margins-missing": {"margins": None},
            "margin-zero": {"margins": {"B": 0.0, "C": 4.0, "D": 6.0}},
            "margin-negative": {"margins": {"B": -0.1, "C": 4.0, "D": 6.0}},
            "margin-nonfinite": {
                "margins": {"B": float("inf"), "C": 4.0, "D": 6.0}
            },
        }
        clean = [_clean("good")]
        pairs = _pairs("good")
        attacks = _attacks("good")
        for raw_id, changes in bad_rows.items():
            row = _clean(raw_id)
            row.update(changes)
            clean.append(row)
            pairs.extend(_pairs(raw_id))
            attacks.extend(_attacks(raw_id))

        result = _assemble(clean, pairs, attacks)

        self.assertEqual(result["eligible_raw_sample_ids"], ["good"])
        self.assertEqual(
            result["excluded_raw_sample_ids"], sorted(bad_rows)
        )
        self.assertEqual(
            {row["raw_sample_id"] for row in result["prediction_units"]}, {"good"}
        )

    def test_recomputes_forced_margin_status_when_cached_flags_are_missing(self) -> None:
        row = _clean("derived")
        row["scorer_numerically_valid"] = None
        row["score_tie"] = None
        result = _assemble(row, _pairs("derived"), _attacks("derived"))

        self.assertEqual(result["eligible_raw_sample_ids"], ["derived"])

    def test_requires_at_least_one_scorer_margin_clean_row(self) -> None:
        with self.assertRaisesRegex(
            ContractError, "no eligible forced_margin clean-correct examples"
        ):
            _assemble(
                _clean("bad", scorer_correct=False),
                _pairs("bad"),
                _attacks("bad"),
            )

    def test_zero_margin_is_a_failure_boundary_crossing(self) -> None:
        result = _assemble(_clean("raw-1"), _pairs("raw-1"), _attacks("raw-1"))
        by_budget = {
            (row["attack_family"], row["requested_epsilon"]): row["flipped"]
            for row in result["prediction_units"]
        }
        self.assertFalse(by_budget[("pgd_autograd", 0.1)])
        self.assertTrue(by_budget[("pgd_autograd", 0.2)])

    def test_rejects_incomplete_primary_probe_prefix(self) -> None:
        with self.assertRaisesRegex(ContractError, "incomplete frozen probe prefix"):
            _assemble(_clean("raw-1"), _pairs("raw-1", directions=(0,)), _attacks("raw-1"))

    def test_excludes_rejected_probe_prefix_but_still_validates_attack_cube(self) -> None:
        pairs = _pairs("raw-1")
        pairs[1]["accepted"] = False
        pairs[1]["central_differences"] = {"B": None, "C": None, "D": None}
        result = _assemble(_clean("raw-1"), pairs, _attacks("raw-1"))

        self.assertEqual(len(result["prediction_units"]), 8)
        self.assertEqual(len(result["edge_predictors"]), 2)
        self.assertEqual(result["counts"]["probe_exclusions"], 1)
        self.assertEqual(
            result["probe_exclusions"][0]["reason"],
            "insufficient_accepted_probe_directions",
        )

    def test_empirical_minimum_k_eff_retains_a_reported_partial_estimate(self) -> None:
        pairs = _pairs("raw-1")
        pairs[1]["accepted"] = False
        pairs[1]["central_differences"] = {"B": None, "C": None, "D": None}
        result = assemble_failure_boundary(
            [_clean("raw-1")],
            pairs,
            _attacks("raw-1"),
            frozen_edges=("p2c@0",),
            frozen_budgets=(0.1, 0.2),
            frozen_families=("pgd_autograd", "random_independent"),
            requested_K=2,
            minimum_K_eff=1,
            selected_h=0.001,
            probe_seeds=(101, 202, 303),
        )

        self.assertEqual(result["counts"]["edge_predictors"], 3)
        partial = [
            row for row in result["edge_predictors"] if row["K_eff"] == 1
        ]
        self.assertEqual(len(partial), 1)
        self.assertEqual(partial[0]["estimate_kind"], "accepted_direction_subset")
        self.assertFalse(partial[0]["primary_available"])

    def test_requires_three_unique_frozen_probe_seeds(self) -> None:
        with self.assertRaisesRegex(ContractError, "at least three unique"):
            assemble_failure_boundary(
                [_clean("raw-1")],
                _pairs("raw-1", seeds=(101, 202)),
                _attacks("raw-1"),
                frozen_edges=("p2c@0",),
                frozen_budgets=(0.1, 0.2),
                frozen_families=("pgd_autograd", "random_independent"),
                requested_K=2,
                selected_h=0.001,
                probe_seeds=(101, 202),
            )


if __name__ == "__main__":
    unittest.main()
