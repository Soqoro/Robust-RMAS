from types import SimpleNamespace
import unittest

from experiments.linkradius.io_utils import canonical_json_bytes
from experiments.linkradius.run_linkradius import _trajectory_rows
from experiments.linkradius.schemas import ContractError
from experiments.linkradius.select_clean_correct import annotate_screening_rows

try:
    import torch
except ModuleNotFoundError:
    torch = None


class ScreeningNonfiniteTests(unittest.TestCase):
    @staticmethod
    def relay_tensor(offset=0.0):
        return [[
            [1.0 + offset, 2.0 + offset],
            [3.0 + offset, 4.0 + offset],
        ]]

    @staticmethod
    def trajectory():
        labels = ("A", "B", "C", "D")
        edges = ("p2c@0", "c2s@0", "s2p@0", "p2c@1", "c2s@1")
        return SimpleNamespace(
            rounds=2,
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
            transport_messages={
                edge: ScreeningNonfiniteTests.relay_tensor(index * 10.0)
                for index, edge in enumerate(edges)
            },
            receiver_reference_messages={
                edge: ScreeningNonfiniteTests.relay_tensor(index * 10.0)
                for index, edge in enumerate(edges)
            },
            edge_dtypes={
                edge: SimpleNamespace(
                    transport_dtype="float32",
                    consumer_dtype="bfloat16",
                    requested_transfer_mode="cpu_staged",
                    realized_transfer_mode="cpu_float32_staged_cross_device",
                )
                for edge in edges
            },
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
        diagnostics = row["forward_finiteness"]
        self.assertTrue(diagnostics["all_relay_interfaces_finite"])
        self.assertFalse(diagnostics["all_observed_numeric_outputs_finite"])
        self.assertEqual(
            diagnostics["first_nonfinite"]["stage"],
            "terminal_solver_scoring",
        )
        self.assertEqual(
            list(diagnostics["edges"]),
            ["p2c@0", "c2s@0", "s2p@0", "p2c@1", "c2s@1"],
        )
        self.assertEqual(
            diagnostics["edges"]["p2c@0"]["requested_transfer_mode"],
            "cpu_staged",
        )
        self.assertEqual(
            diagnostics["edges"]["p2c@0"]["realized_transfer_mode"],
            "cpu_float32_staged_cross_device",
        )
        canonical_json_bytes(rows)

        annotated, summary = annotate_screening_rows(rows)
        self.assertFalse(annotated[0]["dual_correct"])
        self.assertFalse(annotated[0]["analysis_eligible"])
        self.assertEqual(annotated[0]["exclusion_reason"], "scorer_nonfinite")
        self.assertEqual(summary["exclusion_counts"], {"scorer_nonfinite": 1})

    def test_authenticated_clean_capture_still_rejects_nonfinite_scores(self):
        with self.assertRaisesRegex(
            ContractError,
            "clean trajectory row raw-0 contains non-finite forward values",
        ):
            _trajectory_rows(
                self.trajectory(),
                task=self.task("clean"),
            )

    def test_first_nonfinite_relay_stage_and_latent_step_are_explicit(self):
        trajectory = self.trajectory()
        trajectory.transport_messages["s2p@0"][0][1][1] = float("inf")
        trajectory.receiver_reference_messages["s2p@0"][0][1][1] = float(
            "inf"
        )

        row = _trajectory_rows(
            trajectory,
            task=self.task("discover"),
        )[0]
        diagnostics = row["forward_finiteness"]
        first = diagnostics["first_nonfinite"]
        self.assertFalse(diagnostics["all_relay_interfaces_finite"])
        self.assertEqual(first["stage"], "solver_feedback")
        self.assertEqual(first["edge_id"], "s2p@0")
        self.assertEqual(first["interface"], "transport")
        self.assertEqual(first["first_nonfinite_index"], [1, 1])
        self.assertEqual(first["first_nonfinite_latent_step"], 1)

        transport = diagnostics["edges"]["s2p@0"]["transport"]
        self.assertEqual(transport["nonfinite_count"], 1)
        self.assertEqual(transport["posinf_count"], 1)
        self.assertEqual(transport["nan_count"], 0)
        self.assertTrue(transport["latent_step_stats"][0]["all_finite"])
        self.assertFalse(transport["latent_step_stats"][1]["all_finite"])
        canonical_json_bytes(row)

        annotated, summary = annotate_screening_rows([row])
        self.assertEqual(annotated[0]["exclusion_reason"], "relay_nonfinite")
        self.assertEqual(summary["exclusion_counts"], {"relay_nonfinite": 1})

    def test_receiver_cast_boundary_is_distinguished_from_transport(self):
        trajectory = self.trajectory()
        trajectory.receiver_reference_messages["c2s@0"][0][0][0] = float("nan")

        row = _trajectory_rows(
            trajectory,
            task=self.task("discover"),
        )[0]
        first = row["forward_finiteness"]["first_nonfinite"]
        self.assertEqual(first["stage"], "critic")
        self.assertEqual(first["edge_id"], "c2s@0")
        self.assertEqual(first["interface"], "receiver")
        receiver = row["forward_finiteness"]["edges"]["c2s@0"]["receiver"]
        self.assertEqual(receiver["nan_count"], 1)
        self.assertEqual(receiver["first_nonfinite_index"], [0, 0])

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_real_tensor_diagnostics_are_json_safe(self):
        trajectory = self.trajectory()
        for edge, values in tuple(trajectory.transport_messages.items()):
            trajectory.transport_messages[edge] = torch.tensor(values)
        for edge, values in tuple(
            trajectory.receiver_reference_messages.items()
        ):
            trajectory.receiver_reference_messages[edge] = torch.tensor(values)
        trajectory.transport_messages["p2c@1"][0, 0, 1] = float("-inf")

        row = _trajectory_rows(
            trajectory,
            task=self.task("discover"),
        )[0]
        stats = row["forward_finiteness"]["edges"]["p2c@1"]["transport"]
        self.assertEqual(stats["stored_dtype"], "float32")
        self.assertEqual(stats["neginf_count"], 1)
        self.assertEqual(stats["first_nonfinite_index"], [0, 1])
        canonical_json_bytes(row)


if __name__ == "__main__":
    unittest.main()
