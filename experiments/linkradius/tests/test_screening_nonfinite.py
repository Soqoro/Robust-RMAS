from types import SimpleNamespace
import unittest

from experiments.linkradius.io_utils import canonical_json_bytes
from experiments.linkradius.run_linkradius import _trajectory_rows
from experiments.linkradius.schemas import ContractError
from experiments.linkradius.select_clean_correct import annotate_screening_rows


class ScreeningNonfiniteTests(unittest.TestCase):
    @staticmethod
    def trajectory():
        labels = ("A", "B", "C", "D")
        return SimpleNamespace(
            sample_ids=["sample-0"],
            raw_sample_ids=["raw-0"],
            raw_indices=[0],
            gold_labels=["A"],
            clean_generation_audit=[
                {
                    "strict_choice": "A",
                    "answer_invalid": False,
                    "answer_conflict": False,
                }
            ],
            clean_scoring=SimpleNamespace(
                labels=labels,
                scores=[[float("nan"), -1.0, -2.0, -3.0]],
                summed_logprobs=[[float("-inf"), -2.0, -4.0, -6.0]],
                mean_logprobs=[[float("nan"), -1.0, -2.0, -3.0]],
                token_counts=[[2, 2, 2, 2]],
                predictions=(None,),
                score_ties=(False,),
                metadata={"scorer_version": "test"},
            ),
            clean_margins=[{"B": float("nan"), "C": float("nan"), "D": float("nan")}],
            analysis_eligibility_mask=[True],
            provenance={},
            execution_manifest_hash="execution-hash",
            ordered_cohort_hash="cohort-hash",
            batch_boundary_hash="boundary-hash",
        )

    @staticmethod
    def task(stage):
        return {
            "stage": stage,
            "workflow": "engineering",
            "partition": "validation",
            "dataset": "gpqa",
            "style": "sequential_scaled",
            "method": "ours_recursive",
            "R": 2,
            "config_key": "config-hash",
        }

    def test_screening_serializes_nulls_and_records_explicit_exclusion(self):
        rows = _trajectory_rows(
            self.trajectory(),
            task=self.task("discover"),
        )
        row = rows[0]
        self.assertFalse(row["scorer_numerically_valid"])
        self.assertIsNone(row["option_scores"]["A"])
        self.assertIsNone(row["summed_option_logprobs"]["A"])
        self.assertIsNone(row["minimum_margin"])
        self.assertIsNone(row["binding_competitor"])
        self.assertIn("option_scores.A", row["scorer_nonfinite_fields"])
        canonical_json_bytes(rows)

        annotated, summary = annotate_screening_rows(rows)
        self.assertFalse(annotated[0]["dual_correct"])
        self.assertFalse(annotated[0]["analysis_eligible"])
        self.assertEqual(annotated[0]["exclusion_reason"], "scorer_nonfinite")
        self.assertEqual(summary["exclusion_counts"], {"scorer_nonfinite": 1})

    def test_authenticated_clean_capture_still_rejects_nonfinite_scores(self):
        with self.assertRaisesRegex(
            ContractError,
            "clean trajectory row raw-0 contains non-finite scorer values",
        ):
            _trajectory_rows(
                self.trajectory(),
                task=self.task("clean"),
            )


if __name__ == "__main__":
    unittest.main()
