from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.linkradius import run_linkradius as runner
from experiments.linkradius.grid import GridConfig, WORKFLOW_STAGES, build_grid
from experiments.linkradius.make_execution_manifest import (
    build_execution_manifest,
    verify_execution_manifest,
)
from experiments.linkradius.make_split_manifest import build_split_manifest
from experiments.linkradius.run_linkradius import build_parser
from experiments.linkradius.schemas import ContractError, EARLY_R2_EDGES


REPO_ROOT = Path(__file__).resolve().parents[3]


class FailureBoundaryGridTests(unittest.TestCase):
    def test_attacks_rejects_ambiguous_global_scope_and_manifest_override(self) -> None:
        with self.assertRaisesRegex(ContractError, "exactly DATASETS=gpqa"):
            runner.main(
                [
                    "--workflow",
                    "attacks",
                    "--stage",
                    "split",
                    "--datasets",
                    "gpqa medqa",
                ]
            )
        with self.assertRaisesRegex(ContractError, "unqualified EXECUTION_MANIFEST"):
            runner.main(
                [
                    "--workflow",
                    "attacks",
                    "--stage",
                    "split",
                    "--execution-manifest-path",
                    "/tmp/ambiguous.json",
                ]
            )

    def test_attacks_lifecycle_exposes_the_complete_frozen_workflow(self) -> None:
        self.assertEqual(
            WORKFLOW_STAGES["attacks"],
            (
                "split",
                "freeze_execution",
                "val_grid",
                "val",
                "freeze_attack",
                "clean_grid",
                "clean",
                "test_probe_grid",
                "test_probe",
                "test_grid",
                "test",
                "thresholds",
                "analyze",
                "validate",
                "grid",
            ),
        )

    def test_validation_and_test_attack_grids_sweep_budgets_in_each_task(self) -> None:
        budgets = (3e-4, 1e-3, 3e-3)
        families = ("pgd_autograd", "random_independent")
        for stage, partition, batches in (
            ("val", "validation", 2),
            ("test", "test", 3),
        ):
            with self.subTest(stage=stage):
                tasks = build_grid(
                    GridConfig(
                        workflow="attacks",
                        stage=stage,
                        partitions=(partition,),
                        num_batches=batches,
                        attack_families=families,
                        attack_epsilons=budgets,
                    )
                )
                self.assertEqual(
                    len(tasks), batches * len(EARLY_R2_EDGES) * len(families)
                )
                self.assertEqual({task.edge_id for task in tasks}, set(EARLY_R2_EDGES))
                self.assertEqual({task.attack_family for task in tasks}, set(families))
                self.assertEqual(
                    {task.execution_batch_id for task in tasks}, set(range(batches))
                )
                self.assertTrue(all(task.epsilon is None for task in tasks))
                self.assertTrue(
                    all(
                        task.metadata == {"attack_epsilons": list(budgets)}
                        for task in tasks
                    )
                )

    def test_test_probe_grid_is_restricted_to_early_r2_edges(self) -> None:
        tasks = build_grid(
            GridConfig(
                workflow="attacks",
                stage="test_probe",
                partitions=("test",),
                num_batches=2,
                probe_radii=(1e-3,),
                probe_seeds=(101, 202, 303),
                K=8,
            )
        )
        self.assertEqual(len(tasks), 2 * len(EARLY_R2_EDGES) * 3)
        self.assertEqual({task.edge_id for task in tasks}, set(EARLY_R2_EDGES))
        self.assertEqual({task.probe_seed for task in tasks}, {101, 202, 303})

    def test_attack_task_key_binds_the_complete_budget_sweep(self) -> None:
        common = dict(
            workflow="attacks",
            stage="test",
            partitions=("test",),
            num_batches=1,
            attack_families=("pgd_autograd",),
        )
        original = build_grid(
            GridConfig(**common, attack_epsilons=(1e-3, 3e-3))
        )[0]
        changed = build_grid(
            GridConfig(**common, attack_epsilons=(1e-3, 1e-2))
        )[0]
        self.assertNotEqual(original.config_key, changed.config_key)
        self.assertNotEqual(original.metadata, changed.metadata)


class FailureBoundaryExecutionFreezeTests(unittest.TestCase):
    def test_fresh_dual_correct_filter_is_recomputed_from_clean_primitives(self) -> None:
        trajectory = SimpleNamespace(
            gold_labels=["A", "A", "A"],
            analysis_eligibility_mask=[True, True, False],
            clean_generation_audit=[
                {
                    "strict_choice": "A",
                    "answer_invalid": False,
                    "answer_conflict": False,
                },
                {
                    "strict_choice": "B",
                    "answer_invalid": False,
                    "answer_conflict": False,
                },
                {
                    "strict_choice": "A",
                    "answer_invalid": False,
                    "answer_conflict": False,
                },
            ],
            clean_scoring=SimpleNamespace(
                predictions=["A", "A", "A"], score_ties=[False, False, False]
            ),
            clean_margins=[
                {"B": 1.0, "C": 2.0, "D": 3.0},
                {"B": 1.0, "C": 2.0, "D": 3.0},
                {"B": 1.0, "C": 2.0, "D": 3.0},
            ],
        )
        self.assertEqual(
            runner._fresh_dual_correct_trajectory_indices(trajectory), [0]
        )

    def test_heldout_manifest_can_freeze_unknown_outcomes_as_null(self) -> None:
        split = build_split_manifest(
            [
                {"raw_sample_id": f"raw-{index}", "raw_index": index}
                for index in range(15)
            ]
        )
        test_rows = list(split["partitions"]["test"])
        screening = [
            {
                "raw_sample_id": row["raw_sample_id"],
                "sample_id": row["raw_sample_id"],
                "raw_index": row["raw_index"],
                "analysis_eligible": True,
                "dual_correct": None,
                "exclusion_reason": "",
            }
            for row in test_rows
        ]
        manifest = build_execution_manifest(
            split_manifest=split,
            partition="test",
            screening_rows=screening,
            batch_size=2,
            screening_config_hash="a" * 64,
            retain_all_partition_rows=True,
        )
        verify_execution_manifest(manifest, split_manifest=split)
        self.assertEqual(manifest["ordered_raw_sample_ids"], [
            row["raw_sample_id"] for row in test_rows
        ])
        self.assertEqual(manifest["analysis_eligible"], [True] * len(test_rows))
        self.assertEqual(manifest["screening_dual_correct"], [None] * len(test_rows))
        self.assertEqual(manifest["retained_filler_rows"], 0)

    def test_heldout_freeze_rejects_any_completed_test_outcome_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contaminated = (
                root
                / "attacks"
                / "gpqa"
                / "R2"
                / "test"
                / "clean"
                / "global"
                / ("f" * 64)
            )
            contaminated.mkdir(parents=True)
            (contaminated / ".complete.json").write_text("{}\n", encoding="utf-8")
            args = build_parser().parse_args(
                [
                    "--workflow",
                    "attacks",
                    "--stage",
                    "freeze_execution",
                    "--out-root",
                    directory,
                ]
            )
            task = build_grid(runner._build_grid_config(args))[0].as_dict()
            with mock.patch.object(runner, "_authenticated_split_manifest") as split:
                with self.assertRaisesRegex(
                    ContractError, "before any test outcome task"
                ):
                    runner._freeze_heldout_execution(
                        args, root / "freeze-task", task, REPO_ROOT
                    )
            split.assert_not_called()

    def test_heldout_freeze_rejects_interrupted_test_outcome_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interrupted = (
                root
                / "attacks"
                / "gpqa"
                / "R2"
                / "test"
                / "test_probe"
                / "p2c_r0"
                / ("f" * 64)
            )
            interrupted.mkdir(parents=True)
            (interrupted / "manifest.json").write_text("{}\n", encoding="utf-8")
            args = build_parser().parse_args(
                [
                    "--workflow",
                    "attacks",
                    "--stage",
                    "freeze_execution",
                    "--out-root",
                    directory,
                ]
            )
            task = build_grid(runner._build_grid_config(args))[0].as_dict()
            with mock.patch.object(runner, "_authenticated_split_manifest") as split:
                with self.assertRaisesRegex(
                    ContractError, "before any test outcome task"
                ):
                    runner._freeze_heldout_execution(
                        args, root / "freeze-task", task, REPO_ROOT
                    )
            split.assert_not_called()


class FrozenFailureBoundaryProtocolTests(unittest.TestCase):
    @staticmethod
    def _arguments():
        return build_parser().parse_args(
            [
                "--workflow",
                "attacks",
                "--stage",
                "test",
                "--datasets",
                "gpqa",
                "--rounds",
                "2",
                "--style",
                "sequential_scaled",
                "--method",
                "ours_recursive",
                "--batch-size",
                "1",
                "--latent-length",
                "48",
                "--subspace",
                "full_tensor",
                "--attack-families",
                "pgd_autograd random_independent",
                "--attack-epsilons",
                "0.001 0.003 0.01",
                "--pgd-steps",
                "20",
                "--random-attack-seed-offset",
                "1000000",
                "--probe-radii",
                "0.003",
                "--probe-seeds",
                "101 202 303",
                "--K",
                "32",
                "--device",
                "cuda:0",
                "--planner-device",
                "cuda:0",
                "--critic-device",
                "cuda:1",
                "--solver-device",
                "cuda:2",
                "--terminal-solver-device",
                "cuda:3",
                "--relay-transfer-mode",
                "cpu_staged",
                "--autograd-memory-mode",
                "checkpoint",
            ]
        )

    @staticmethod
    def _frozen() -> dict:
        return {
            "dataset": "gpqa",
            "R": 2,
            "seed": 42,
            "style": "sequential_scaled",
            "method": "ours_recursive",
            "batch_size": 1,
            "latent_length": 48,
            "subspace": "full_tensor",
            "attack_families": ["pgd_autograd", "random_independent"],
            "attack_epsilons": [0.001, 0.003, 0.01],
            "pgd": {"steps": 20},
            "random_independent": {"seed_offset": 1000000},
            "probe": {
                "h": 0.003,
                "seeds": [101, 202, 303],
                "primary_seed": 101,
                "K": 32,
            },
            "runtime": {
                "role_devices": {
                    "planner": "cuda:0",
                    "critic": "cuda:1",
                    "solver": "cuda:2",
                    "terminal_solver": "cuda:3",
                },
                "relay_transfer_mode": "cpu_staged",
                "autograd_memory_mode": "checkpoint",
                "trust_remote_code": 1,
                "round_label_mode": "legacy",
                "environment": runner._runtime_environment_identity(),
            },
            "test_execution_manifest_hash": "e" * 64,
        }

    @staticmethod
    def _execution_authentication():
        return mock.patch.multiple(
            runner,
            _execution_manifest_path=mock.DEFAULT,
            load_json=mock.DEFAULT,
            verify_execution_manifest=mock.DEFAULT,
        )

    def test_exact_frozen_arguments_are_accepted(self) -> None:
        args = self._arguments()
        frozen = self._frozen()
        with self._execution_authentication() as patched:
            patched["_execution_manifest_path"].return_value = "execution.json"
            patched["load_json"].return_value = {}
            patched["verify_execution_manifest"].return_value = "e" * 64
            for stage in (
                "test_probe",
                "test",
                "thresholds",
                "analyze",
                "validate",
            ):
                runner._assert_frozen_attack_arguments(args, frozen, stage=stage)

    def test_attack_and_probe_drift_are_rejected(self) -> None:
        frozen = self._frozen()
        mutations = (
            ("attack_families", "random_independent pgd_autograd", "test"),
            ("attack_epsilons", "0.001 0.01", "test"),
            ("pgd_steps", 21, "test"),
            ("random_attack_seed_offset", 7, "test"),
            ("probe_radii", "0.001", "test_probe"),
            ("probe_seeds", "202", "test_probe"),
            ("K", 16, "test_probe"),
        )
        with self._execution_authentication() as patched:
            patched["_execution_manifest_path"].return_value = "execution.json"
            patched["load_json"].return_value = {}
            patched["verify_execution_manifest"].return_value = "e" * 64
            for field, value, stage in mutations:
                with self.subTest(field=field, stage=stage):
                    args = self._arguments()
                    setattr(args, field, value)
                    with self.assertRaises(ContractError):
                        runner._assert_frozen_attack_arguments(
                            args, frozen, stage=stage
                        )

    def test_shared_and_runtime_drift_are_rejected(self) -> None:
        frozen = self._frozen()
        mutations = (
            ("seeds", "43"),
            ("style", "sequential_light"),
            ("batch_size", 2),
            ("latent_length", 32),
            ("planner_device", "cuda:3"),
            ("critic_device", "cuda:2"),
            ("solver_device", "cuda:1"),
            ("terminal_solver_device", "cuda:0"),
            ("relay_transfer_mode", "direct"),
            ("autograd_memory_mode", "none"),
            ("trust_remote_code", 0),
            ("round_label_mode", "actual"),
        )
        with self._execution_authentication() as patched:
            patched["_execution_manifest_path"].return_value = "execution.json"
            patched["load_json"].return_value = {}
            patched["verify_execution_manifest"].return_value = "e" * 64
            for field, value in mutations:
                with self.subTest(field=field):
                    args = self._arguments()
                    setattr(args, field, value)
                    with self.assertRaises(ContractError):
                        runner._assert_frozen_attack_arguments(
                            args, frozen, stage="test"
                        )


class CurrentGridRowCollectionTests(unittest.TestCase):
    def test_historical_retune_rows_are_not_reintroduced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "val"
            current_key = "a" * 64
            historical_key = "b" * 64
            for array_index, config_key, marker in (
                (0, current_key, "current"),
                (1, historical_key, "historical"),
            ):
                task_dir = root / "p2c_r0" / config_key
                task_dir.mkdir(parents=True)
                (task_dir / "manifest.json").write_text(
                    json.dumps(
                        {
                            "task": {
                                "stage": "val",
                                "array_index": array_index,
                                "config_key": config_key,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                rows = [
                    {"record_type": "sample", "marker": marker},
                    {
                        "record_type": "shard_metadata",
                        "array_index": array_index,
                        "config_key": config_key,
                        "row_count": 1,
                    },
                ]
                (task_dir / "attack_results.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                (task_dir / ".complete.json").write_text("{}\n", encoding="utf-8")

            completion = {
                "source_hash": "s" * 64,
                "config_hash": current_key,
                "array_index": 0,
                "artifacts": [
                    {"path": "manifest.json"},
                    {"path": "attack_results.jsonl", "row_count": 2},
                ],
            }
            with mock.patch.object(
                runner, "verify_completion", return_value=completion
            ) as verify:
                rows = runner._completed_rows(
                    root,
                    "attack_results.jsonl",
                    expected_source_hash="s" * 64,
                    expected_config_keys={current_key},
                )

            self.assertEqual(
                [row.get("marker") for row in rows if row.get("marker")],
                ["current"],
            )
            verify.assert_called_once()


class PGDTargetEvidenceTests(unittest.TestCase):
    @staticmethod
    def _rows() -> list[dict]:
        common = {
            "raw_sample_id": "raw-1",
            "edge_id": "p2c@0",
            "requested_epsilon": 0.01,
            "attack_seed": 42,
            "attack_restart": 0,
            "attack_family": "pgd_autograd",
            "gold": "A",
            "pgd_target_count": 3,
            "clean_margins": {"B": 1.0, "C": 1.0, "D": 1.0},
        }
        target_margins = {
            "B": {"B": 0.5, "C": 0.4, "D": 0.3},
            "C": {"B": 0.1, "C": 0.2, "D": 0.4},
            "D": {"B": 0.6, "C": 0.7, "D": 0.5},
        }
        targets = []
        for label, margins in target_margins.items():
            scores = {"A": 1.0, **{key: 1.0 - value for key, value in margins.items()}}
            realized = {
                "target_label": label,
                "initial_margin": 1.0,
                "final_margin": margins[label],
                "requested_delta_norm": 1.0,
                "realized_delta_norm": 0.9,
                "requested_relative_norm": 0.01,
                "realized_relative_norm": 0.009,
                "absolute_budget": 1.0,
                "budget_respected": True,
            }
            targets.append(
                {
                    **common,
                    "record_type": "attack_target",
                    "target_label": label,
                    "competitor": label,
                    "binding_competitor": label,
                    "margins": margins,
                    "minimum_margin": margins[label],
                    "option_scores": scores,
                    "realized_epsilon": 0.009,
                    "realized_intervention": realized,
                }
            )
        strongest = targets[1]
        summary = {
            **common,
            "record_type": "sample",
            "margins": strongest["margins"],
            "minimum_margin": min(strongest["margins"].values()),
            "option_scores": strongest["option_scores"],
            "realized_epsilon": strongest["realized_epsilon"],
            "realized_intervention": strongest["realized_intervention"],
        }
        return [summary, *targets]

    def test_exact_three_target_cube_reconstructs_summary(self) -> None:
        report = runner._validate_pgd_target_evidence(
            self._rows(), where="unit test"
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["summary_rows"], 1)
        self.assertEqual(report["target_rows"], 3)

    def test_missing_target_or_wrong_summary_is_rejected(self) -> None:
        with self.assertRaises(ContractError):
            runner._validate_pgd_target_evidence(
                self._rows()[:-1], where="unit test"
            )
        rows = self._rows()
        rows[0]["realized_intervention"] = {"target_label": "B"}
        with self.assertRaises(ContractError):
            runner._validate_pgd_target_evidence(rows, where="unit test")

    def test_each_target_norm_and_score_margin_identity_are_authenticated(self) -> None:
        rows = self._rows()
        rows[-1]["realized_intervention"] = {
            **rows[-1]["realized_intervention"],
            "budget_respected": False,
        }
        with self.assertRaisesRegex(ContractError, "malformed PGD target evidence"):
            runner._validate_pgd_target_evidence(rows, where="unit test")

        rows = self._rows()
        rows[-1]["margins"] = {**rows[-1]["margins"], "B": 123.0}
        with self.assertRaisesRegex(ContractError, "malformed PGD target evidence"):
            runner._validate_pgd_target_evidence(rows, where="unit test")

        rows = self._rows()
        rows[-1]["realized_intervention"] = {
            **rows[-1]["realized_intervention"],
            "realized_delta_norm": 99.0,
            "absolute_budget": 0.1,
        }
        with self.assertRaisesRegex(ContractError, "malformed PGD target evidence"):
            runner._validate_pgd_target_evidence(rows, where="unit test")

    def test_validation_straddle_must_be_on_the_same_curve(self) -> None:
        unpaired = [
            {
                "raw_sample_id": "safe",
                "edge_id": "p2c@0",
                "requested_epsilon": budget,
                "minimum_margin": 1.0,
            }
            for budget in (0.001, 0.01)
        ] + [
            {
                "raw_sample_id": "failed",
                "edge_id": "p2c@0",
                "requested_epsilon": budget,
                "minimum_margin": -1.0,
            }
            for budget in (0.001, 0.01)
        ]
        report = runner._validation_pgd_straddle(unpaired, (0.001, 0.01))
        self.assertEqual(report["safe_at_smallest"], 1)
        self.assertEqual(report["crossed_at_largest"], 1)
        self.assertEqual(report["paired_straddled_curves"], 0)

        paired = [dict(row) for row in unpaired]
        paired[1]["minimum_margin"] = -0.5
        report = runner._validation_pgd_straddle(paired, (0.001, 0.01))
        self.assertEqual(report["paired_straddled_curves"], 1)

class ReplayArgumentIsolationTests(unittest.TestCase):
    def test_trajectory_resolution_authenticates_only_requested_batch(self) -> None:
        args = SimpleNamespace(trajectory="")
        task = {"execution_batch_id": 7}
        expected = Path("/authenticated/batch-7/clean_trajectory.pt")
        with mock.patch.object(
            runner,
            "_completed_clean_trajectories",
            return_value=[expected],
        ) as completed:
            result = runner._resolve_trajectory_path(
                args, task, producer_workflow="attacks"
            )
        self.assertEqual(result, expected)
        completed.assert_called_once_with(
            args,
            task,
            producer_workflow="attacks",
            execution_batch_ids=(7,),
        )

    def test_replay_stage_does_not_mutate_explicit_trajectory_argument(self) -> None:
        args = build_parser().parse_args(
            [
                "--workflow",
                "engineering",
                "--stage",
                "replay",
                "--trajectory",
                "/caller/selected-clean-trajectory.pt",
            ]
        )
        task = build_grid(runner._build_grid_config(args))[0].as_dict()
        trajectory = SimpleNamespace(execution_manifest_hash="execution-hash")
        runtime = mock.Mock()
        runtime.replay.side_effect = RuntimeError("stop after replay dispatch")

        with mock.patch.object(
            runner, "_resolve_trajectory_path", return_value=Path("/resolved/clean.pt")
        ), mock.patch.object(
            runner, "_load_trajectory", return_value=trajectory
        ), mock.patch.object(
            runner, "_execution_manifest_path", return_value="execution.json"
        ), mock.patch.object(
            runner,
            "_authenticated_execution_manifest",
            return_value=(Path("execution.json"), {}, "execution-hash"),
        ), mock.patch.object(runner, "_runtime", return_value=runtime):
            with self.assertRaisesRegex(RuntimeError, "stop after replay dispatch"):
                runner._replay_stage(args, Path("/unused"), task, REPO_ROOT)

        self.assertEqual(args.trajectory, "/caller/selected-clean-trajectory.pt")
        runtime.unload.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
