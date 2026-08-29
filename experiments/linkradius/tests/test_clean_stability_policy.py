from __future__ import annotations

from copy import deepcopy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.linkradius import run_linkradius as runner
from experiments.linkradius.io_utils import load_jsonl, verify_completion
from experiments.linkradius.run_linkradius import _clean_execution_coverage
from experiments.linkradius.schemas import ContractError
from experiments.linkradius.select_clean_correct import (
    annotate_screening_rows,
    audit_clean_stability,
    classify_forced_margin_row,
)


def _fresh_row(
    raw_id: str,
    *,
    generated: str,
    scorer: str,
    gold: str = "A",
) -> dict:
    scores = {label: 0.0 for label in ("A", "B", "C", "D")}
    scores[scorer] = 1.0
    return {
        "record_type": "sample",
        "raw_sample_id": raw_id,
        "gold": gold,
        "strict_generated_choice": generated,
        "strict_generated_valid": True,
        "answer_invalid": False,
        "answer_conflict": False,
        "scorer_prediction": scorer,
        "scorer_numerically_valid": True,
        "scorer_nonfinite_fields": [],
        "score_tie": False,
        "forward_finiteness": {"all_relay_interfaces_finite": True},
        "option_scores": scores,
        "margins": {
            label: 1.0 for label in {"A", "B", "C", "D"} - {gold}
        },
    }


class CleanStabilityPolicyTests(unittest.TestCase):
    def test_forced_margin_primary_does_not_gate_on_generation(self) -> None:
        rows, summary = annotate_screening_rows(
            [_fresh_row("row-0", generated="B", scorer="A")]
        )

        self.assertTrue(rows[0]["forced_margin_correct"])
        self.assertFalse(rows[0]["dual_correct"])
        self.assertTrue(rows[0]["clean_correct"])
        self.assertTrue(rows[0]["analysis_eligible"])
        self.assertEqual(rows[0]["clean_correct_policy"], "forced_margin")
        self.assertEqual(summary["forced_margin_correct_count"], 1)
        self.assertEqual(summary["dual_correct_count"], 0)
        self.assertEqual(summary["analysis_eligible_count"], 1)

    def test_legacy_dual_policy_remains_available(self) -> None:
        rows, summary = annotate_screening_rows(
            [_fresh_row("row-0", generated="B", scorer="A")],
            clean_correct_policy="dual_correct",
        )

        self.assertTrue(rows[0]["forced_margin_correct"])
        self.assertFalse(rows[0]["dual_correct"])
        self.assertFalse(rows[0]["clean_correct"])
        self.assertFalse(rows[0]["analysis_eligible"])
        self.assertEqual(rows[0]["exclusion_reason"], "generated_incorrect")
        self.assertEqual(summary["clean_correct_policy"], "dual_correct")

    def test_forced_margin_endpoint_fails_closed_on_required_inputs(self) -> None:
        valid = _fresh_row("row-0", generated="A", scorer="A")
        cases = {}

        invalid_gold = deepcopy(valid)
        invalid_gold["gold"] = "Z"
        cases["invalid_gold"] = invalid_gold

        relay_nonfinite = deepcopy(valid)
        relay_nonfinite["forward_finiteness"][
            "all_relay_interfaces_finite"
        ] = False
        cases["relay_nonfinite"] = relay_nonfinite

        score_nonfinite = deepcopy(valid)
        score_nonfinite["option_scores"]["A"] = math.inf
        cases["scorer_nonfinite"] = score_nonfinite

        score_tie = deepcopy(valid)
        score_tie["option_scores"]["B"] = score_tie["option_scores"]["A"]
        cases["score_tie"] = score_tie

        scorer_incorrect = _fresh_row("row-0", generated="A", scorer="B")
        cases["scorer_incorrect"] = scorer_incorrect

        nonpositive_margin = deepcopy(valid)
        nonpositive_margin["margins"]["B"] = 0.0
        cases["nonpositive_clean_margin"] = nonpositive_margin

        for expected_reason, row in cases.items():
            with self.subTest(expected_reason=expected_reason):
                passed, reason = classify_forced_margin_row(row)
                self.assertFalse(passed)
                self.assertEqual(reason, expected_reason)

    def test_strict_policy_preserves_status_change_failure(self) -> None:
        with self.assertRaisesRegex(
            ContractError,
            "fresh clean dual-correct status differs from screening",
        ):
            audit_clean_stability(
                [_fresh_row("row-0", generated="B", scorer="A")],
                raw_sample_ids=["row-0"],
                screening_dual_correct=[True],
                screening_analysis_eligible=[True],
                screening_exclusion_reasons=[""],
                screening_generated_choices=["A"],
                screening_scorer_predictions=["A"],
                policy="strict",
            )

    def test_empirical_policy_demotes_but_never_promotes(self) -> None:
        rows, summary = audit_clean_stability(
            [
                _fresh_row("eligible", generated="B", scorer="A"),
                _fresh_row("frozen-out", generated="A", scorer="A"),
            ],
            raw_sample_ids=["eligible", "frozen-out"],
            screening_dual_correct=[True, False],
            screening_analysis_eligible=[True, False],
            screening_exclusion_reasons=["", "generated_incorrect"],
            screening_generated_choices=["A", "B"],
            screening_scorer_predictions=["A", "A"],
            policy="empirical",
        )

        self.assertEqual(
            [row["effective_analysis_eligible"] for row in rows],
            [False, False],
        )
        self.assertTrue(rows[0]["analysis_demoted"])
        self.assertEqual(
            rows[0]["effective_exclusion_reason"],
            "clean_replay_not_dual_correct",
        )
        self.assertFalse(rows[1]["analysis_demoted"])
        self.assertTrue(rows[1]["fresh_dual_correct"])
        self.assertEqual(summary["dual_correct_status_changed_count"], 2)
        self.assertEqual(summary["demoted_count"], 1)
        self.assertEqual(summary["promoted_count"], 0)
        self.assertEqual(summary["generated_choice_flip_rate"], 1.0)
        self.assertEqual(summary["scorer_prediction_flip_rate"], 0.0)

    def test_unknown_heldout_status_is_not_demoted(self) -> None:
        rows, summary = audit_clean_stability(
            [_fresh_row("heldout", generated="B", scorer="A")],
            raw_sample_ids=["heldout"],
            screening_dual_correct=[None],
            screening_analysis_eligible=[True],
            screening_exclusion_reasons=[""],
            policy="empirical",
        )
        self.assertTrue(rows[0]["effective_analysis_eligible"])
        self.assertFalse(rows[0]["analysis_demoted"])
        self.assertEqual(summary["known_screening_status_count"], 0)
        self.assertIsNone(summary["dual_correct_status_flip_rate"])

    def test_forced_margin_strict_audit_ignores_generation_flip(self) -> None:
        rows, summary = audit_clean_stability(
            [_fresh_row("row-0", generated="B", scorer="A")],
            raw_sample_ids=["row-0"],
            screening_clean_correct=[True],
            screening_dual_correct=[True],
            screening_analysis_eligible=[True],
            screening_exclusion_reasons=[""],
            screening_generated_choices=["A"],
            screening_scorer_predictions=["A"],
            clean_correct_policy="forced_margin",
            policy="strict",
        )

        self.assertTrue(rows[0]["fresh_clean_correct"])
        self.assertFalse(rows[0]["fresh_dual_correct"])
        self.assertFalse(rows[0]["clean_status_changed"])
        self.assertTrue(rows[0]["dual_correct_status_changed"])
        self.assertEqual(summary["clean_correct_status_changed_count"], 0)
        self.assertEqual(summary["dual_correct_status_changed_count"], 1)
        self.assertEqual(summary["generated_choice_flip_count"], 1)

    def test_forced_margin_empirical_audit_demotes_scorer_failure(self) -> None:
        rows, summary = audit_clean_stability(
            [_fresh_row("row-0", generated="A", scorer="B")],
            raw_sample_ids=["row-0"],
            screening_clean_correct=[True],
            screening_dual_correct=[True],
            screening_analysis_eligible=[True],
            screening_exclusion_reasons=[""],
            screening_generated_choices=["A"],
            screening_scorer_predictions=["A"],
            clean_correct_policy="forced_margin",
            policy="empirical",
        )

        self.assertFalse(rows[0]["fresh_clean_correct"])
        self.assertTrue(rows[0]["analysis_demoted"])
        self.assertEqual(
            rows[0]["effective_exclusion_reason"],
            "clean_replay_not_forced_margin_correct",
        )
        self.assertEqual(summary["demoted_count"], 1)


class CleanExecutionCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = {
            "ordered_raw_sample_ids": ["row-0", "row-1"],
            "analysis_eligible": [True, False],
            "ordered_cohort_hash": "ordered",
            "batch_boundary_hash": "boundaries",
            "batch_boundaries": [
                {"execution_batch_id": 0, "start": 0, "stop": 1},
                {"execution_batch_id": 1, "start": 1, "stop": 2},
            ],
        }

    @staticmethod
    def _row(raw_id: str, batch_id: int, eligible: bool) -> dict:
        return {
            "raw_sample_id": raw_id,
            "analysis_eligible": eligible,
            "ordered_cohort_hash": "ordered",
            "batch_boundary_hash": "boundaries",
            "task": {"execution_batch_id": batch_id},
        }

    def test_empirical_coverage_accepts_authenticated_demotion(self) -> None:
        demoted = {
            **self._row("row-0", 0, False),
            "clean_stability_policy": "empirical",
            "screening_dual_correct": True,
            "screening_analysis_eligible": True,
            "fresh_dual_correct": False,
            "clean_status_changed": True,
            "analysis_demoted": True,
            "exclusion_reason": "clean_replay_not_dual_correct",
            "generated_choice_comparable": True,
            "generated_choice_changed": True,
        }
        frozen_out = self._row("row-1", 1, False)
        report = _clean_execution_coverage(
            [demoted, frozen_out],
            self.manifest,
            clean_stability_policy="empirical",
        )
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["frozen_eligible_rows"], 1)
        self.assertEqual(report["eligible_rows"], 0)
        self.assertEqual(report["demoted_rows"], 1)
        self.assertEqual(report["generated_choice_flip_rate"], 1.0)

    def test_strict_coverage_rejects_the_same_demotion(self) -> None:
        rows = [
            self._row("row-0", 0, False),
            self._row("row-1", 1, False),
        ]
        report = _clean_execution_coverage(
            rows,
            self.manifest,
            clean_stability_policy="strict",
        )
        self.assertFalse(report["passed"])
        self.assertIn("row-0:eligibility", report["violation_examples"])

    def test_empirical_coverage_rejects_post_outcome_promotion(self) -> None:
        rows = [
            self._row("row-0", 0, True),
            self._row("row-1", 1, True),
        ]
        report = _clean_execution_coverage(
            rows,
            self.manifest,
            clean_stability_policy="empirical",
        )
        self.assertFalse(report["passed"])
        self.assertIn(
            "row-1:post_outcome_promotion", report["violation_examples"]
        )


class EmpiricalDownstreamSkipTests(unittest.TestCase):
    def test_empty_effective_batch_publishes_authenticated_skip(self) -> None:
        args = runner.build_parser().parse_args(
            [
                "--workflow",
                "smoke",
                "--stage",
                "causal",
                "--clean-stability-policy",
                "empirical",
                "--overwrite",
                "1",
            ]
        )
        args.overwrite = True
        task = {
            "stage": "causal",
            "partition": "validation",
            "edge_id": "p2c@0",
            "config_key": "a" * 64,
            "array_index": 3,
            "execution_batch_id": 7,
        }
        trajectory = SimpleNamespace(
            execution_manifest_hash="execution-hash",
            analysis_eligibility_mask=[False],
        )

        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            (task_dir / "manifest.json").write_text(
                json.dumps({"task": task}), encoding="utf-8"
            )
            (task_dir / "command.txt").write_text("test\n", encoding="utf-8")
            with mock.patch.object(
                runner, "_resolve_trajectory_path", return_value=Path("clean.pt")
            ), mock.patch.object(
                runner, "_load_trajectory", return_value=trajectory
            ), mock.patch.object(
                runner, "_execution_manifest_path", return_value="execution.json"
            ), mock.patch.object(
                runner,
                "_authenticated_execution_manifest",
                return_value=(Path("execution.json"), {}, "execution-hash"),
            ), mock.patch.object(runner, "_runtime") as runtime:
                runner._replay_stage(args, task_dir, task, Path.cwd())

            runtime.assert_not_called()
            rows = load_jsonl(task_dir / "causal_runs.jsonl")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["row_count"], 0)
            self.assertEqual(
                rows[0]["skip_reason"],
                "no_effective_analysis_eligible_row",
            )
            completion = verify_completion(task_dir)
            self.assertEqual(completion["analysis_row_count"], 0)


if __name__ == "__main__":
    unittest.main()
