from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

from experiments.linkradius.io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    publish_completion,
    source_hash,
)
from experiments.linkradius.schemas import ContractError
from experiments.linkradius.validate_engineering import (
    EXPECTED_R2_EDGES,
    _legacy_latent_contagion_regression_check,
    assemble_engineering_evidence,
)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class EngineeringArtifactAuthenticationTests(unittest.TestCase):
    def setUp(self) -> None:
        from RecursiveMAS.inference_utils.linkradius import valid_edges
        from RecursiveMAS.inference_utils.linkradius_runtime import (
            CleanTrajectory,
            EdgeDtypeMetadata,
            ForcedChoiceBatch,
            RUNTIME_VERSION,
            TRAJECTORY_VERSION,
        )

        self._valid_edges = valid_edges
        self._CleanTrajectory = CleanTrajectory
        self._EdgeDtypeMetadata = EdgeDtypeMetadata
        self._ForcedChoiceBatch = ForcedChoiceBatch
        self._runtime_version = RUNTIME_VERSION
        self._trajectory_version = TRAJECTORY_VERSION
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo_root = Path(__file__).resolve().parents[3]
        self.current_source_hash = source_hash(self.repo_root)
        self.shared = {
            "split_manifest_hash": "1" * 64,
            "execution_manifest_hash": "2" * 64,
            "ordered_cohort_hash": "3" * 64,
            "batch_boundary_hash": "4" * 64,
            "model_hash": "5" * 64,
            "scorer_hash": "6" * 64,
        }
        self.scores = {"A": 2.0, "B": 1.0, "C": 0.5, "D": -1.0}
        self.margins = {"B": 1.0, "C": 1.5, "D": 3.0}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _config_hash(name: str) -> str:
        return hashlib.sha256(name.encode("utf-8")).hexdigest()

    def _trajectory(self, *, tensor_steps: int = 32, recorded_steps: int = 32):
        edges = self._valid_edges(2)
        messages = {
            edge: torch.ones((1, tensor_steps, 2), dtype=torch.float32)
            for edge in edges
        }
        score_tensor = torch.tensor([[2.0, 1.0, 0.5, -1.0]], dtype=torch.float32)
        scoring = self._ForcedChoiceBatch(
            labels=("A", "B", "C", "D"),
            scores=score_tensor,
            summed_logprobs=score_tensor,
            mean_logprobs=score_tensor,
            token_counts=torch.ones_like(score_tensor, dtype=torch.long),
            predictions=("A",),
            score_ties=(False,),
            encodings={},
            metadata={},
        )
        return self._CleanTrajectory(
            schema_version=self._trajectory_version,
            runtime_version=self._runtime_version,
            rounds=2,
            sample_ids=["sample-1"],
            raw_sample_ids=["raw-1"],
            raw_indices=[7],
            questions=["question"],
            gold_labels=["A"],
            option_permutations=[None],
            choice_metadata=[None],
            execution_manifest_hash=self.shared["execution_manifest_hash"],
            ordered_batch_ids=[0],
            batch_boundaries=[(0, 1)],
            analysis_eligibility_mask=[True],
            transport_messages=dict(messages),
            receiver_reference_messages=dict(messages),
            edge_dtypes={
                edge: self._EdgeDtypeMetadata("float32", "float32") for edge in edges
            },
            clean_scoring=scoring,
            clean_margins=[dict(self.margins)],
            clean_generation_audit=[{"strict_choice": "A"}],
            provenance={
                "runtime_config": {"latent_steps": recorded_steps},
                "split_manifest_hash": self.shared["split_manifest_hash"],
                "global_ordered_cohort_hash": self.shared["ordered_cohort_hash"],
                "global_batch_boundary_hash": self.shared["batch_boundary_hash"],
                "model_hash": self.shared["model_hash"],
                "scorer_hash": self.shared["scorer_hash"],
            },
        )

    def _sample_row(self, *, config_hash: str) -> dict[str, object]:
        return {
            "schema_version": "linkradius.v1",
            "record_type": "sample",
            "sample_id": "sample-1",
            "raw_sample_id": "raw-1",
            "raw_index": 7,
            "analysis_eligible": True,
            "gold": "A",
            "option_scores": dict(self.scores),
            "scorer_prediction": "A",
            "scorer_correct": True,
            "score_tie": False,
            "margins": dict(self.margins),
            "minimum_margin": 1.0,
            "binding_competitor": "B",
            "R": 2,
            **self.shared,
            "source_hash": self.current_source_hash,
            "config_hash": config_hash,
        }

    def _write_manifest(
        self,
        directory: Path,
        *,
        stage: str,
        config_hash: str,
        array_index: int,
        edge_id: str | None = None,
        extra_task: dict[str, object] | None = None,
    ) -> None:
        task: dict[str, object] = {
            "array_index": array_index,
            "workflow": "engineering",
            "stage": stage,
            "dataset": "gpqa",
            "partition": "validation",
            "style": "sequential_light",
            "method": "ours_recursive",
            "R": 2,
            "batch_size": 1,
            "latent_length": 32,
            "execution_manifest_hash": self.shared["execution_manifest_hash"],
            "execution_batch_id": 0,
            "config_key": config_hash,
        }
        if edge_id is not None:
            task["edge_id"] = edge_id
        task.update(extra_task or {})
        atomic_write_json(
            directory / "manifest.json",
            {
                "schema_version": "linkradius.task_manifest.v1",
                "task": task,
                "source_hash": self.current_source_hash,
            },
        )

    def _publish_clean(
        self,
        name: str = "clean",
        *,
        completion_source_hash: str | None = None,
        tensor_steps: int = 32,
        recorded_steps: int = 32,
    ) -> Path:
        directory = self.root / name
        directory.mkdir(parents=True)
        config_hash = self._config_hash(name)
        trajectory_path = directory / "clean_trajectory.pt"
        torch.save(
            self._trajectory(tensor_steps=tensor_steps, recorded_steps=recorded_steps),
            trajectory_path,
        )
        clean_path = directory / "clean_baseline.jsonl"
        atomic_write_jsonl(
            clean_path,
            [
                self._sample_row(config_hash=config_hash),
                {
                    "record_type": "shard_metadata",
                    "array_index": 0,
                    "config_key": config_hash,
                    "row_count": 1,
                },
            ],
        )
        self._write_manifest(
            directory,
            stage="clean",
            config_hash=config_hash,
            array_index=0,
        )
        publish_completion(
            directory,
            config_hash=config_hash,
            source_hash_value=completion_source_hash or self.current_source_hash,
            artifact_paths=["manifest.json", "clean_trajectory.pt", "clean_baseline.jsonl"],
            row_counts={"clean_baseline.jsonl": 2},
            extra={"array_index": 0, "execution_batch_id": 0},
        )
        return directory

    def _publish_replay(
        self,
        *,
        edge_id: str,
        mode: str,
        suffix: str = "",
        provenance_overrides: dict[str, object] | None = None,
    ) -> None:
        token = f"{mode}-{edge_id}-{suffix}"
        directory = self.root / "replay" / token.replace("@", "_")
        directory.mkdir(parents=True)
        config_hash = self._config_hash(token)
        row = self._sample_row(config_hash=config_hash)
        row.update(
            {
                "edge_id": edge_id,
                "intervention_mode": mode,
            }
        )
        row.update(provenance_overrides or {})
        shard = len(list((self.root / "replay").glob("*"))) - 1
        atomic_write_jsonl(
            directory / "replay_runs.jsonl",
            [
                row,
                {
                    "record_type": "shard_metadata",
                    "array_index": shard,
                    "config_key": config_hash,
                    "row_count": 1,
                },
            ],
        )
        self._write_manifest(
            directory,
            stage="replay",
            config_hash=config_hash,
            array_index=shard,
            edge_id=edge_id,
            extra_task={"intervention_mode": mode},
        )
        publish_completion(
            directory,
            config_hash=config_hash,
            source_hash_value=self.current_source_hash,
            artifact_paths=["manifest.json", "replay_runs.jsonl"],
            row_counts={"replay_runs.jsonl": 2},
            extra={"array_index": shard, "execution_batch_id": 0},
        )

    def _publish_probe(
        self,
        *,
        h: float,
        probe_seed: int,
        tamper_central_difference: bool = False,
        direction_count: int = 8,
    ) -> None:
        token = f"probe-{h}-{probe_seed}"
        directory = self.root / "probe" / token
        directory.mkdir(parents=True)
        config_hash = self._config_hash(token)
        common = {
            "sample_id": "sample-1",
            "raw_sample_id": "raw-1",
            "raw_index": 7,
            "analysis_eligible": True,
            "edge_id": "p2c@0",
            "probe_seed": probe_seed,
            "h": h,
            "q": 64,
            "subspace_id": "full-tensor",
            "partition": "validation",
            "split_manifest_hash": self.shared["split_manifest_hash"],
            "execution_manifest_hash": self.shared["execution_manifest_hash"],
            "ordered_cohort_hash": self.shared["ordered_cohort_hash"],
            "batch_boundary_hash": self.shared["batch_boundary_hash"],
            "model_hash": self.shared["model_hash"],
            "scorer_hash": self.shared["scorer_hash"],
            "adapter_hash": "7" * 64,
            "prompt_hash": "8" * 64,
            "subspace_hash": "9" * 64,
            "source_hash": self.current_source_hash,
            "config_hash": config_hash,
            "R": 2,
        }
        rows: list[dict[str, object]] = []
        for direction_id in range(direction_count):
            derivatives = {
                "B": 1.0 + 0.01 * direction_id,
                "C": 0.2 + 0.001 * direction_id,
                "D": 0.1 + 0.001 * direction_id,
            }
            plus_margins = {
                label: self.margins[label] + derivative * h
                for label, derivative in derivatives.items()
            }
            minus_margins = {
                label: self.margins[label] - derivative * h
                for label, derivative in derivatives.items()
            }
            plus_id = f"plus-{h}-{probe_seed}-{direction_id}"
            minus_id = f"minus-{h}-{probe_seed}-{direction_id}"
            plus_diag = {
                "realized_signed_coordinate": h,
                "requested_realized_cosine": 1.0,
                "off_direction_relative": 0.0,
                "collapsed": False,
            }
            minus_diag = {
                "realized_signed_coordinate": -h,
                "requested_realized_cosine": 1.0,
                "off_direction_relative": 0.0,
                "collapsed": False,
            }
            central = {
                label: (plus_margins[label] - minus_margins[label]) / (2.0 * h)
                for label in derivatives
            }
            if tamper_central_difference and direction_id == 0:
                central["B"] += 1.0
            rows.extend(
                (
                    {
                        **common,
                        "record_type": "sample",
                        "intervention_mode": "additive_antithetic",
                        "run_id": plus_id,
                        "direction_id": direction_id,
                        "sign": 1,
                        "margins": plus_margins,
                        "diagnostics": plus_diag,
                    },
                    {
                        **common,
                        "record_type": "sample",
                        "intervention_mode": "additive_antithetic",
                        "run_id": minus_id,
                        "direction_id": direction_id,
                        "sign": -1,
                        "margins": minus_margins,
                        "diagnostics": minus_diag,
                    },
                    {
                        **common,
                        "record_type": "probe_pair",
                        "intervention_mode": "additive_antithetic_pair",
                        "run_id": f"pair-{h}-{probe_seed}-{direction_id}",
                        "direction_id": direction_id,
                        "plus_run_id": plus_id,
                        "minus_run_id": minus_id,
                        "clean_margins": dict(self.margins),
                        "t_plus": h,
                        "t_minus": -h,
                        "realized_separation": 2.0 * h,
                        "antipodality": 1.0,
                        "accepted": False,
                        "central_differences": central,
                        "margins_plus": plus_margins,
                        "margins_minus": minus_margins,
                    },
                )
            )
        array_index = len(list((self.root / "probe").glob("*"))) - 1
        rows.append(
            {
                "record_type": "shard_metadata",
                "array_index": array_index,
                "config_key": config_hash,
                "row_count": 3 * direction_count,
            }
        )
        atomic_write_jsonl(directory / "probe_runs.jsonl", rows)
        self._write_manifest(
            directory,
            stage="probe",
            config_hash=config_hash,
            array_index=array_index,
            edge_id="p2c@0",
            extra_task={
                "h": h,
                "probe_seed": probe_seed,
                "K": 8,
                "subspace": "full_tensor",
                "metadata": {"direction_ids": list(range(8))},
            },
        )
        publish_completion(
            directory,
            config_hash=config_hash,
            source_hash_value=self.current_source_hash,
            artifact_paths=["manifest.json", "probe_runs.jsonl"],
            row_counts={"probe_runs.jsonl": len(rows)},
            extra={"array_index": array_index, "execution_batch_id": 0},
        )

    def _publish_gradient(
        self,
        *,
        task_execution_hash: str,
        tamper_separation: bool = False,
        include_row_provenance: bool = True,
        requested_coordinate_scale: float = 1.0,
    ) -> None:
        directory = self.root / "gradient"
        directory.mkdir(parents=True)
        config_hash = self._config_hash("gradient")
        atomic_write_jsonl(
            directory / "gradient_runs.jsonl",
            [
                {
                    "record_type": "gradient",
                    "sample_id": "sample-1",
                    "raw_sample_id": "raw-1",
                    "edge_id": "c2s@1",
                    "gradient_norm": 1.0,
                    "target_label": "B",
                    "autograd_semantics": "continuous_consumer_input",
                    **(
                        {
                            "split_manifest_hash": self.shared["split_manifest_hash"],
                            "execution_manifest_hash": self.shared["execution_manifest_hash"],
                            "source_hash": self.current_source_hash,
                        }
                        if include_row_provenance
                        else {}
                    ),
                    "finite_difference": {
                        "h": 0.01,
                        "target": "B",
                        "realized_separation": 0.019 if tamper_separation else 0.018,
                        "finite_difference_derivative": 1.0,
                        "autograd_dimensionless_derivative": 0.9,
                        "relative_error": 0.1,
                        "agrees": True,
                        "plus_diagnostics": {
                            "requested_signed_coordinate": (
                                0.01 * requested_coordinate_scale
                            ),
                            "realized_signed_coordinate": 0.009,
                            "realized_delta_norm": 1.0,
                            "requested_realized_cosine": 1.0,
                            "collapsed": False,
                            "consumer_dtype": "float32",
                        },
                        "minus_diagnostics": {
                            "requested_signed_coordinate": (
                                -0.01 * requested_coordinate_scale
                            ),
                            "realized_signed_coordinate": -0.009,
                            "realized_delta_norm": 1.0,
                            "requested_realized_cosine": 1.0,
                            "collapsed": False,
                            "consumer_dtype": "float32",
                        },
                    },
                    "pgd": {"supported": False, "targets": []},
                },
                {
                    "record_type": "shard_metadata",
                    "array_index": 0,
                    "config_key": config_hash,
                    "row_count": 1,
                },
            ],
        )
        self._write_manifest(
            directory,
            stage="gradient",
            config_hash=config_hash,
            array_index=0,
            edge_id="c2s@1",
            extra_task={"execution_manifest_hash": task_execution_hash},
        )
        publish_completion(
            directory,
            config_hash=config_hash,
            source_hash_value=self.current_source_hash,
            artifact_paths=["manifest.json", "gradient_runs.jsonl"],
            row_counts={"gradient_runs.jsonl": 2},
            extra={"array_index": 0, "execution_batch_id": 0},
        )

    def test_orphan_clean_trajectory_is_rejected(self) -> None:
        directory = self.root / "orphan"
        directory.mkdir()
        torch.save(self._trajectory(), directory / "clean_trajectory.pt")
        with self.assertRaisesRegex(ContractError, "orphan"):
            assemble_engineering_evidence(self.root)

    def test_stale_source_completion_is_rejected(self) -> None:
        self._publish_clean(completion_source_hash="f" * 64)
        with self.assertRaisesRegex(ContractError, "stale-source"):
            assemble_engineering_evidence(self.root)

    def test_duplicate_clean_trajectories_are_rejected(self) -> None:
        self._publish_clean("clean-a")
        self._publish_clean("clean-b")
        with self.assertRaisesRegex(ContractError, "exactly one"):
            assemble_engineering_evidence(self.root)

    def test_wrong_recorded_latent_length_is_rejected(self) -> None:
        self._publish_clean(tensor_steps=31, recorded_steps=32)
        with self.assertRaisesRegex(ContractError, r"shape \[1,32,D\]"):
            assemble_engineering_evidence(self.root)

    def test_mixed_replay_provenance_is_rejected(self) -> None:
        self._publish_clean()
        self._publish_replay(
            edge_id="p2c@0",
            mode="identity",
            provenance_overrides={"execution_manifest_hash": "e" * 64},
        )
        with self.assertRaisesRegex(ContractError, "mixes execution_manifest_hash"):
            assemble_engineering_evidence(self.root)

    def test_exact_five_rows_per_replay_mode_and_repeated_scores(self) -> None:
        self._publish_clean()
        for mode in ("identity", "additive_zero"):
            for edge_id in sorted(EXPECTED_R2_EDGES):
                self._publish_replay(edge_id=edge_id, mode=mode)
        evidence = assemble_engineering_evidence(self.root)
        self.assertTrue(evidence["checks"]["identity_replay_scores"]["passed"])
        self.assertTrue(evidence["checks"]["zero_additive_scores"]["passed"])
        self.assertTrue(evidence["checks"]["repeated_scoring_deterministic"]["passed"])

    def test_antithetic_pairs_are_joined_recomputed_and_cardinality_checked(self) -> None:
        self._publish_clean()
        self._publish_probe(h=1e-3, probe_seed=101)
        self._publish_probe(h=3e-3, probe_seed=101)
        evidence = assemble_engineering_evidence(self.root)
        check = evidence["checks"]["antithetic_cast_survival"]
        self.assertTrue(check["passed"], check)
        self.assertEqual(check["detail"]["pair_rows"], 16)

    def test_tampered_probe_central_difference_is_rejected(self) -> None:
        self._publish_clean()
        self._publish_probe(
            h=1e-3,
            probe_seed=101,
            tamper_central_difference=True,
        )
        self._publish_probe(h=3e-3, probe_seed=101)
        with self.assertRaisesRegex(ContractError, "central differences"):
            assemble_engineering_evidence(self.root)

    def test_incomplete_k8_probe_task_is_rejected(self) -> None:
        self._publish_clean()
        self._publish_probe(h=1e-3, probe_seed=101, direction_count=7)
        self._publish_probe(h=3e-3, probe_seed=101)
        with self.assertRaisesRegex(ContractError, "directions 0..7"):
            assemble_engineering_evidence(self.root)

    def test_legacy_defaults_behavior_and_schema_are_exercised(self) -> None:
        check = _legacy_latent_contagion_regression_check()
        self.assertTrue(check["passed"], check)

    def test_gradient_without_row_provenance_is_bound_through_task_manifest(self) -> None:
        self._publish_clean()
        self._publish_gradient(
            task_execution_hash="e" * 64,
            include_row_provenance=False,
        )
        with self.assertRaisesRegex(ContractError, "execution_manifest_hash"):
            assemble_engineering_evidence(self.root)

    def test_gradient_finite_difference_cast_evidence_is_recomputed(self) -> None:
        self._publish_clean()
        self._publish_gradient(
            task_execution_hash=self.shared["execution_manifest_hash"]
        )
        evidence = assemble_engineering_evidence(self.root)
        check = evidence["checks"]["terminal_autograd_finite_difference_agreement"]
        self.assertTrue(check["passed"], check)
        self.assertTrue(check["detail"]["cast_survived"])

    def test_gradient_float32_projection_roundoff_is_accepted(self) -> None:
        self._publish_clean()
        self._publish_gradient(
            task_execution_hash=self.shared["execution_manifest_hash"],
            requested_coordinate_scale=1.0000059349474493,
        )
        evidence = assemble_engineering_evidence(self.root)
        check = evidence["checks"]["terminal_autograd_finite_difference_agreement"]
        self.assertTrue(check["passed"], check)

    def test_materially_wrong_requested_coordinate_is_rejected(self) -> None:
        self._publish_clean()
        self._publish_gradient(
            task_execution_hash=self.shared["execution_manifest_hash"],
            requested_coordinate_scale=1.0001,
        )
        with self.assertRaisesRegex(ContractError, "requested signed coordinates"):
            assemble_engineering_evidence(self.root)

    def test_tampered_gradient_realized_separation_is_rejected(self) -> None:
        self._publish_clean()
        self._publish_gradient(
            task_execution_hash=self.shared["execution_manifest_hash"],
            tamper_separation=True,
        )
        with self.assertRaisesRegex(ContractError, "realized separation"):
            assemble_engineering_evidence(self.root)


if __name__ == "__main__":
    unittest.main()
