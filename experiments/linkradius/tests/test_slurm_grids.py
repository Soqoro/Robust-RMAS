from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.linkradius.grid import GridConfig, build_grid, canonical_edge_pairs, select_task
from experiments.linkradius.io_utils import atomic_write_json, atomic_write_text, publish_completion, source_hash, verify_completion
from experiments.linkradius import run_linkradius as runner
from experiments.linkradius.run_linkradius import _build_grid_config, build_parser
from experiments.linkradius.io_utils import atomic_write_json
from experiments.linkradius.make_execution_manifest import build_execution_manifest
from experiments.linkradius.make_split_manifest import build_split_manifest
from experiments.linkradius.schemas import ContractError


REPO_ROOT = Path(__file__).resolve().parents[3]


class SlurmGridTests(unittest.TestCase):
    @staticmethod
    def _runner_task_key(*arguments: str) -> str:
        args = build_parser().parse_args(list(arguments))
        return build_grid(_build_grid_config(args))[0].config_key

    def test_valid_edges_and_deterministic_selection(self) -> None:
        for rounds in range(1, 6):
            pairs = canonical_edge_pairs(rounds)
            self.assertEqual(len(pairs), 3 * rounds - 1)
            self.assertNotIn(("s2p", rounds - 1), pairs)
        config = GridConfig(workflow="smoke", stage="probe", num_batches=2)
        first, second = build_grid(config), build_grid(config)
        self.assertEqual(first, second)
        self.assertEqual(select_task(first, 3), first[3])
        with self.assertRaises(ContractError):
            select_task(first, len(first))

    def test_smoke_grid_only_has_three_early_edges(self) -> None:
        tasks = build_grid(GridConfig(workflow="smoke", stage="probe", num_batches=1))
        self.assertEqual({task.edge_id for task in tasks}, {"p2c@0", "c2s@0", "s2p@0"})
        self.assertNotIn("s2p@1", {task.edge_id for task in tasks})

    def test_clean_capture_rejects_missing_execution_manifest_before_loading_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(
                [
                    "--workflow",
                    "smoke",
                    "--stage",
                    "clean",
                    "--out-root",
                    directory,
                ]
            )
            task = build_grid(_build_grid_config(args))[0].as_dict()
            with mock.patch.object(runner, "_records_for_task") as records:
                with self.assertRaisesRegex(
                    ContractError, "clean capture requires a frozen execution manifest"
                ):
                    runner._capture_stage(
                        args,
                        Path(directory) / "clean-task",
                        task,
                        REPO_ROOT,
                    )
            records.assert_not_called()

    def test_pilot_validate_is_the_same_canonical_task_as_validate_probe(self) -> None:
        validate = build_parser().parse_args(
            ["--workflow", "pilot", "--stage", "validate"]
        )
        validate_probe = build_parser().parse_args(
            ["--workflow", "pilot", "--stage", "validate_probe"]
        )
        self.assertEqual(
            build_grid(_build_grid_config(validate)),
            build_grid(_build_grid_config(validate_probe)),
        )

    def test_generic_grid_can_target_a_concrete_stage(self) -> None:
        args = build_parser().parse_args(
            [
                "--workflow",
                "engineering",
                "--stage",
                "grid",
                "--grid-target-stage",
                "replay",
            ]
        )
        tasks = build_grid(_build_grid_config(args))
        self.assertEqual(len(tasks), 10)
        self.assertEqual({task.stage for task in tasks}, {"replay"})

    def test_deferred_completion_authenticates_closed_shell_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            for name, value in (
                ("manifest.json", "{}\n"),
                ("command.txt", "python task\n"),
                ("result.json", "{}\n"),
                ("warnings.txt", ""),
                (".run.log.pending", "finished\n"),
                (".launcher_command.pending.txt", "python task\n"),
            ):
                atomic_write_text(task_dir / name, value)
            previous = runner._DEFER_COMPLETION
            runner._DEFER_COMPLETION = True
            try:
                runner.publish_completion(
                    task_dir,
                    config_hash="a" * 64,
                    source_hash_value=source_hash(REPO_ROOT),
                    artifact_paths=("manifest.json", "command.txt", "result.json"),
                )
            finally:
                runner._DEFER_COMPLETION = previous
            runner._finalize_deferred_completion(task_dir)
            completion = verify_completion(task_dir, expected_config_hash="a" * 64)
            declared = {item["path"] for item in completion["artifacts"]}
            self.assertTrue(
                {"manifest.json", "command.txt", "result.json", "run.log", "warnings.txt", "launcher_command.txt"}.issubset(declared)
            )
            self.assertFalse((task_dir / ".completion.pending.json").exists())
            atomic_write_text(task_dir / ".run.log.pending", "reused\n")
            atomic_write_text(
                task_dir / ".launcher_command.pending.txt", "python task\n"
            )
            reused = runner._finalize_deferred_completion(task_dir)
            self.assertEqual(reused["status"], "reused_complete")
            self.assertFalse((task_dir / ".run.log.pending").exists())

    def test_config_key_changes_with_execution_context(self) -> None:
        base = build_grid(GridConfig(workflow="engineering", stage="probe", num_batches=1))[0]
        changed = build_grid(GridConfig(workflow="engineering", stage="probe", num_batches=1, batch_size=2))[0]
        self.assertNotEqual(base.config_key, changed.config_key)

    def test_config_key_binds_stage_relevant_runner_options(self) -> None:
        common = ("--workflow", "engineering", "--stage", "replay")
        donor_42 = self._runner_task_key(*common, "--donor-seed", "42")
        donor_999 = self._runner_task_key(*common, "--donor-seed", "999")
        self.assertNotEqual(donor_42, donor_999)

        legacy = self._runner_task_key(
            "--workflow", "engineering", "--stage", "clean", "--round-label-mode", "legacy"
        )
        actual = self._runner_task_key(
            "--workflow", "engineering", "--stage", "clean", "--round-label-mode", "actual"
        )
        self.assertNotEqual(legacy, actual)

        draws_100 = self._runner_task_key(
            "--workflow", "smoke", "--stage", "aggregate", "--bootstrap-draws", "100"
        )
        draws_9999 = self._runner_task_key(
            "--workflow", "smoke", "--stage", "aggregate", "--bootstrap-draws", "9999"
        )
        self.assertNotEqual(draws_100, draws_9999)

        pgd_2 = self._runner_task_key(
            "--workflow", "engineering", "--stage", "gradient", "--pgd-steps", "2"
        )
        pgd_7 = self._runner_task_key(
            "--workflow", "engineering", "--stage", "gradient", "--pgd-steps", "7"
        )
        self.assertNotEqual(pgd_2, pgd_7)

    def test_prefreeze_grid_identity_ignores_later_execution_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            arguments = [
                "--workflow",
                "smoke",
                "--stage",
                "screen",
                "--out-root",
                directory,
                "--num-batches",
                "2",
            ]
            args = build_parser().parse_args(arguments)
            before = build_grid(_build_grid_config(args))
            manifest = Path(directory) / "smoke" / "gpqa" / "R2" / "validation" / "execution_manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}\n", encoding="utf-8")
            after = build_grid(_build_grid_config(args))
        self.assertEqual(before, after)

    def test_aggregate_lifecycle_reconstructs_prefreeze_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_args = build_parser().parse_args(
                [
                    "--workflow",
                    "smoke",
                    "--stage",
                    "screen",
                    "--out-root",
                    directory,
                    "--num-batches",
                    "2",
                    "--max-eligible",
                    "16",
                ]
            )
            original = build_grid(_build_grid_config(source_args))
            manifest = Path(directory) / "smoke" / "gpqa" / "R2" / "validation" / "execution_manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}\n", encoding="utf-8")

            aggregate_args = build_parser().parse_args(
                [
                    "--workflow",
                    "aggregate",
                    "--stage",
                    "verify",
                    "--aggregate-phase",
                    "smoke",
                    "--out-root",
                    directory,
                    "--num-batches",
                    "2",
                    "--max-eligible",
                    "16",
                ]
            )
            # The verifier reconstructs source-workflow grids with its shared
            # scientific options.  A later manifest must not alter screen IDs.
            aggregate_args.workflow = "smoke"
            aggregate_args.stage = "screen"
            reconstructed = build_grid(_build_grid_config(aggregate_args))
        self.assertEqual(original, reconstructed)

    def test_aggregate_verify_default_covers_required_smoke_stages(self) -> None:
        args = build_parser().parse_args(["--workflow", "aggregate", "--stage", "verify"])
        self.assertEqual(
            set(args.verify_stages.split()),
            {
                "split",
                "screen",
                "freeze_execution",
                "clean",
                "causal",
                "probe",
                "gradient",
                "attack",
                "estimate",
                "aggregate",
                "validate",
            },
        )

    def test_aggregate_gate_requires_completed_canonical_verify_pointer(self) -> None:
        current_source_hash = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(
                [
                    "--workflow",
                    "aggregate",
                    "--stage",
                    "causal",
                    "--aggregate-phase",
                    "smoke",
                    "--out-root",
                    directory,
                ]
            )
            args._current_source_hash = current_source_hash
            consumer_task = {"dataset": "gpqa", "R": 2, "seed": 42}
            verify_task = runner._canonical_global_task(
                args, consumer_task, stage="verify"
            )
            verified_stages = list(runner.REQUIRED_AGGREGATE_STAGES["smoke"])
            inventory_hash = runner.content_hash(
                [
                    {"stage": stage, "items": []}
                    for stage in verified_stages
                ],
                domain="linkradius:aggregate_completion_inventory:v1",
            )
            scope_hash = runner.content_hash(
                {
                    "aggregate_phase": "smoke",
                    "verified_stages": verified_stages,
                    "source_hash": current_source_hash,
                },
                domain="linkradius:aggregate_verification_scope:v1",
            )
            gate = runner.make_gate(
                gate_type="aggregate_verification_gate",
                checks=[
                    {
                        "name": f"expected_stage:{stage}",
                        "passed": True,
                        "completion_inventory": [],
                    }
                    for stage in verified_stages
                ],
                config_hash=str(verify_task["config_key"]),
                source_hash=current_source_hash,
                prerequisite_hashes={
                    "aggregate_phase": "smoke",
                    "verified_stages": verified_stages,
                    "verification_scope_hash": scope_hash,
                    "completion_inventory_hash": inventory_hash,
                },
            )
            gate_path = runner._aggregate_gate_path(args)
            atomic_write_json(gate_path, gate)

            with self.assertRaises(ContractError):
                runner.enforce_prerequisites(
                    args, "causal", consumer_task
                )

            verify_dir = runner.task_output_dir(args, verify_task)
            atomic_write_json(
                verify_dir / "manifest.json",
                {
                    "task": verify_task,
                    "source_hash": current_source_hash,
                },
            )
            atomic_write_text(verify_dir / "command.txt", "verify\n")
            atomic_write_json(
                verify_dir / "verification.json",
                {
                    "schema_version": "linkradius.aggregate_verification.v1",
                    "aggregate_phase": "smoke",
                    "source_hash": current_source_hash,
                    "verified_stages": verified_stages,
                    "completion_inventory_hash": inventory_hash,
                    "passed": True,
                    "stages": [],
                },
            )
            atomic_write_json(
                verify_dir / "aggregate_verification_gate.json", gate
            )
            atomic_write_json(
                verify_dir / "aggregate_verification_result.json",
                {
                    "schema_version": "linkradius.aggregate_verification_result.v1",
                    "aggregate_phase": "smoke",
                    "aggregate_verification_gate": str(gate_path.resolve()),
                    "gate_content_hash": gate["gate_content_hash"],
                    "local_gate_sha256": runner.file_sha256(
                        verify_dir / "aggregate_verification_gate.json"
                    ),
                    "source_hash": current_source_hash,
                },
            )
            publish_completion(
                verify_dir,
                config_hash=str(verify_task["config_key"]),
                source_hash_value=current_source_hash,
                artifact_paths=[
                    "manifest.json",
                    "command.txt",
                    "verification.json",
                    "aggregate_verification_gate.json",
                    "aggregate_verification_result.json",
                ],
                extra={
                    "array_index": int(verify_task["array_index"]),
                    "gate_content_hash": gate["gate_content_hash"],
                },
            )
            runner.enforce_prerequisites(args, "causal", consumer_task)

    def test_verified_row_input_is_mandatory_for_every_expected_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "causal-task"
            atomic_write_json(task_dir / "manifest.json", {})
            atomic_write_text(task_dir / "command.txt", "causal\n")
            publish_completion(
                task_dir,
                config_hash="b" * 64,
                source_hash_value="c" * 64,
                artifact_paths=["manifest.json", "command.txt"],
            )
            with self.assertRaisesRegex(ContractError, "causal_runs.jsonl"):
                runner._rows_from_verified_directories(
                    {"causal": [task_dir]},
                    stage="causal",
                    filename="causal_runs.jsonl",
                )

    def test_single_public_source_rejects_undeclared_and_duplicate_sidecars(self) -> None:
        current_source_hash = "d" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            undeclared = root / "undeclared"
            atomic_write_json(undeclared / "manifest.json", {})
            atomic_write_text(
                undeclared / "linkradius_edges.csv", "raw_sample_id,edge_id\n"
            )
            publish_completion(
                undeclared,
                config_hash="e" * 64,
                source_hash_value=current_source_hash,
                artifact_paths=["manifest.json"],
            )
            with self.assertRaisesRegex(ContractError, "linkradius_edges.csv"):
                runner._single_required_verified_artifact(
                    {"estimate": [undeclared]},
                    stage="estimate",
                    filename="linkradius_edges.csv",
                    expected_source_hash=current_source_hash,
                )

            declared = []
            for index in range(2):
                task_dir = root / f"declared-{index}"
                atomic_write_json(task_dir / "manifest.json", {})
                atomic_write_text(
                    task_dir / "linkradius_edges.csv",
                    "raw_sample_id,edge_id\n",
                )
                publish_completion(
                    task_dir,
                    config_hash=f"{index + 1:064x}",
                    source_hash_value=current_source_hash,
                    artifact_paths=["manifest.json", "linkradius_edges.csv"],
                )
                declared.append(task_dir)
            with self.assertRaisesRegex(ContractError, "found 2"):
                runner._single_required_verified_artifact(
                    {"estimate": declared},
                    stage="estimate",
                    filename="linkradius_edges.csv",
                    expected_source_hash=current_source_hash,
                )

    def test_smoke_validation_requires_exact_canonical_estimate_artifacts(self) -> None:
        current_source_hash = "9" * 64
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(
                [
                    "--workflow",
                    "smoke",
                    "--stage",
                    "validate",
                    "--out-root",
                    directory,
                ]
            )
            args._current_source_hash = current_source_hash
            consumer_task = {"dataset": "gpqa", "R": 2, "seed": 42}
            expected_task = runner._canonical_global_task(
                args, consumer_task, stage="estimate"
            )
            expected_dir = runner.task_output_dir(args, expected_task)

            rogue_dir = expected_dir.parent / ("0" * 64)
            atomic_write_json(
                rogue_dir / "manifest.json",
                {
                    "task": {
                        **expected_task,
                        "config_key": "0" * 64,
                    },
                    "source_hash": current_source_hash,
                },
            )
            atomic_write_text(rogue_dir / "linkradius_edges.csv", "edge_id\n")
            atomic_write_text(
                rogue_dir / "linkradius_competitors.csv", "competitor\n"
            )
            publish_completion(
                rogue_dir,
                config_hash="0" * 64,
                source_hash_value=current_source_hash,
                artifact_paths=[
                    "manifest.json",
                    "linkradius_edges.csv",
                    "linkradius_competitors.csv",
                ],
                extra={"array_index": int(expected_task["array_index"])},
            )
            with self.assertRaises(ContractError):
                runner._authenticated_canonical_task_artifacts(
                    args,
                    consumer_task,
                    REPO_ROOT,
                    stage="estimate",
                    filenames=(
                        "linkradius_edges.csv",
                        "linkradius_competitors.csv",
                    ),
                )

            atomic_write_json(
                expected_dir / "manifest.json",
                {
                    "task": expected_task,
                    "source_hash": current_source_hash,
                },
            )
            atomic_write_text(expected_dir / "linkradius_edges.csv", "edge_id\n")
            atomic_write_text(
                expected_dir / "linkradius_competitors.csv", "competitor\n"
            )
            publish_completion(
                expected_dir,
                config_hash=str(expected_task["config_key"]),
                source_hash_value=current_source_hash,
                artifact_paths=["manifest.json", "linkradius_edges.csv"],
                extra={"array_index": int(expected_task["array_index"])},
            )
            with self.assertRaisesRegex(ContractError, "artifact declarations"):
                runner._authenticated_canonical_task_artifacts(
                    args,
                    consumer_task,
                    REPO_ROOT,
                    stage="estimate",
                    filenames=(
                        "linkradius_edges.csv",
                        "linkradius_competitors.csv",
                    ),
                )

            publish_completion(
                expected_dir,
                config_hash=str(expected_task["config_key"]),
                source_hash_value=current_source_hash,
                artifact_paths=[
                    "manifest.json",
                    "linkradius_edges.csv",
                    "linkradius_competitors.csv",
                ],
                extra={"array_index": int(expected_task["array_index"])},
                overwrite=True,
            )
            _, artifacts = runner._authenticated_canonical_task_artifacts(
                args,
                consumer_task,
                REPO_ROOT,
                stage="estimate",
                filenames=(
                    "linkradius_edges.csv",
                    "linkradius_competitors.csv",
                ),
            )
            self.assertEqual(
                set(artifacts),
                {"linkradius_edges.csv", "linkradius_competitors.csv"},
            )

    def test_upstream_artifact_change_invalidates_freeze_task_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task_dir = (
                Path(directory)
                / "smoke"
                / "gpqa"
                / "R2"
                / "validation"
                / "screen"
                / "global"
                / ("1" * 64)
            )
            task_dir.mkdir(parents=True)
            atomic_write_json(
                task_dir / "manifest.json",
                {"task": {"stage": "screen"}},
            )
            atomic_write_text(task_dir / "screening_rows.jsonl", "{}\n")
            publish_completion(
                task_dir,
                config_hash="1" * 64,
                source_hash_value="2" * 64,
                artifact_paths=["manifest.json", "screening_rows.jsonl"],
            )
            args = build_parser().parse_args(
                [
                    "--workflow",
                    "smoke",
                    "--stage",
                    "freeze_execution",
                    "--out-root",
                    directory,
                ]
            )
            first = build_grid(_build_grid_config(args))[0].config_key
            atomic_write_text(task_dir / "screening_rows.jsonl", '{"changed":true}\n')
            publish_completion(
                task_dir,
                config_hash="1" * 64,
                source_hash_value="2" * 64,
                artifact_paths=["manifest.json", "screening_rows.jsonl"],
                overwrite=True,
            )
            second = build_grid(_build_grid_config(args))[0].config_key
        self.assertNotEqual(first, second)

    def test_shell_imports_from_arbitrary_cwd_without_pythonpath(self) -> None:
        script = REPO_ROOT / "experiments" / "linkradius" / "run_linkradius_smoke.sh"
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["LR_STAGE"] = "probe_grid"
        env["NUM_BATCHES"] = "1"
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                ["bash", str(script)],
                cwd=directory,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("total_tasks\t12", completed.stdout)
        self.assertNotIn("s2p@1", completed.stdout)

    def test_slurm_spool_copies_resolve_common_from_submit_directory(self) -> None:
        cases = (
            ("run_linkradius_engineering.sh", "grid", 0, "total_tasks\t2"),
            ("run_linkradius_smoke.sh", "probe_grid", 0, "total_tasks\t12"),
            ("run_linkradius_pilot.sh", "probe_calibration_grid", 0, "total_tasks\t60"),
            ("run_linkradius_attacks.sh", "train_grid", 0, "total_tasks\t20"),
            ("run_linkradius_expansion.sh", "grid", 0, "total_tasks\t5"),
            ("run_linkradius_aggregate.sh", "invalid_stage", 2, "unsupported LR_STAGE"),
        )
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            for index, (name, stage, returncode, marker) in enumerate(cases):
                spool = temporary_root / f"job-{index}"
                spool.mkdir()
                source = REPO_ROOT / "experiments" / "linkradius" / name
                copied_script = spool / "slurm_script"
                copied_script.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
                env = dict(os.environ)
                env.pop("PYTHONPATH", None)
                env.update(
                    {
                        "SLURM_JOB_ID": str(9000 + index),
                        "SLURM_SUBMIT_DIR": str(REPO_ROOT),
                        "PYTHON_BIN": sys.executable,
                        "LR_STAGE": stage,
                        "NUM_BATCHES": "1",
                        "BATCH_COUNTS": "attack_train=1 validation=1 test=1",
                        "OUT_ROOT": str(temporary_root / f"outputs-{index}"),
                    }
                )
                completed = subprocess.run(
                    ["bash", str(copied_script)],
                    cwd=spool,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                combined = completed.stdout + completed.stderr
                self.assertEqual(completed.returncode, returncode, (name, combined))
                self.assertIn(marker, combined, (name, combined))
                self.assertNotIn("linkradius_common.sh: No such file", combined)

    def test_frozen_test_grid_is_blocked_before_attack_freeze_gate(self) -> None:
        completed = subprocess.run(
            [
                "python",
                "-m",
                "experiments.linkradius.run_linkradius",
                "--workflow",
                "attacks",
                "--stage",
                "test_grid",
                "--datasets",
                "gpqa",
                "--rounds",
                "2",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("gate", completed.stderr.lower())

    def test_gated_output_dir_preflight_cannot_be_bypassed(self) -> None:
        cases = (("attacks", "test_probe"), ("attacks", "test"), ("expansion", "r2"))
        with tempfile.TemporaryDirectory() as directory:
            for workflow, stage in cases:
                completed = subprocess.run(
                    [
                        "python",
                        "-m",
                        "experiments.linkradius.run_linkradius",
                        "--workflow",
                        workflow,
                        "--stage",
                        stage,
                        "--out-root",
                        directory,
                        "--engineering-gate",
                        str(Path(directory) / "missing_engineering_gate.json"),
                        "--pilot-gate",
                        str(Path(directory) / "missing_pilot_gate.json"),
                        "--print-output-dir",
                    ],
                    cwd=REPO_ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0, (workflow, stage))
                self.assertEqual(completed.stdout.strip(), "", (workflow, stage))

    def test_autograd_grid_skips_frozen_batches_without_eligible_rows(self) -> None:
        split = build_split_manifest(
            [{"raw_sample_id": f"id-{index}", "raw_index": index} for index in range(10)]
        )
        partition = "attack_train"
        ids = [row["raw_sample_id"] for row in split["partitions"][partition]]
        screening = [
            {
                "raw_sample_id": raw_id,
                "sample_id": raw_id,
                "analysis_eligible": index == 2,
                "dual_correct": index == 2,
                "exclusion_reason": "" if index == 2 else "not_selected",
            }
            for index, raw_id in enumerate(ids)
        ]
        manifest = build_execution_manifest(
            split_manifest=split,
            partition=partition,
            screening_rows=screening,
            batch_size=2,
            screening_config_hash="a" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "execution_manifest.json"
            atomic_write_json(path, manifest)
            tasks = build_grid(
                GridConfig(
                    workflow="smoke",
                    stage="gradient",
                    partitions=(partition,),
                    execution_manifests={partition: str(path)},
                )
            )
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].execution_batch_id, 1)
        self.assertEqual(tasks[0].edge_id, "c2s@1")


if __name__ == "__main__":
    unittest.main()
