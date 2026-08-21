from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.linkradius.compare_legacy_equivalence import (
    RELEASE_PROVENANCE_SCHEMA_VERSION,
    build_equivalence_report,
    verify_equivalence_report,
)
from experiments.linkradius.io_utils import atomic_write_jsonl, load_jsonl
from experiments.linkradius.schemas import ContractError

try:
    import torch
    from RecursiveMAS.inference_utils import inference_mas
    from RecursiveMAS.inference_utils.linkradius_runtime import (
        ForcedChoiceBatch,
        LinkRadiusRuntime,
        RelayEmission,
        RuntimeConfig,
    )
except (ImportError, ModuleNotFoundError):  # pragma: no cover - lightweight CPU env.
    torch = None
    inference_mas = None
    ForcedChoiceBatch = None
    LinkRadiusRuntime = object
    RelayEmission = None
    RuntimeConfig = None


REPO_ROOT = Path(__file__).resolve().parents[3]


class LegacyToleranceContractTests(unittest.TestCase):
    def test_noncanonical_equivalence_tolerances_are_rejected_before_loading(self) -> None:
        with self.assertRaisesRegex(ContractError, "fixed tolerances"):
            build_equivalence_report(
                trajectory_path="missing-trajectory.pt",
                legacy_trace_path="missing-trace.pt",
                legacy_results_path="missing-results.jsonl",
                repo_root=REPO_ROOT,
                atol=1.0,
                rtol=1.0,
            )


@unittest.skipIf(torch is None, "PyTorch/legacy inference dependencies are unavailable")
class LegacyEquivalenceTests(unittest.TestCase):
    class ToyRuntime(LinkRadiusRuntime):
        def __init__(self) -> None:
            super().__init__(
                RuntimeConfig(
                    rounds=2,
                    latent_steps=1,
                    batch_size=1,
                    device="cpu",
                    style="sequential_light",
                    dataset="gpqa",
                )
            )

        @property
        def device(self):
            return torch.device("cpu")

        @staticmethod
        def emit(value):
            return RelayEmission(value, value, "float32", "float32")

        def run_initial_planner(self, questions, **kwargs):
            return self.emit(torch.ones((1, 1, 2), dtype=torch.float32))

        def run_critic(self, questions, planner_message, round_idx, **kwargs):
            return self.emit(planner_message + 1.0)

        def run_solver_feedback(self, questions, critic_message, round_idx, **kwargs):
            return self.emit(critic_message + 1.0)

        def run_planner_feedback(self, questions, solver_message, round_idx, **kwargs):
            return self.emit(solver_message + 1.0)

        def score_terminal(self, questions, critic_message, **kwargs):
            base = critic_message.float().mean(dim=(1, 2))
            scores = torch.stack((base, base - 1, base - 2, base - 3), dim=-1)
            return ForcedChoiceBatch(
                labels=("A", "B", "C", "D"),
                scores=scores,
                summed_logprobs=scores,
                mean_logprobs=scores,
                token_counts=torch.ones_like(scores, dtype=torch.long),
                predictions=("A",),
                score_ties=(False,),
                encodings={},
                metadata={
                    "scorer_version": self.config.scorer_version,
                    "normalization": self.config.scorer_normalization,
                    "prefix": self.config.scorer_prefix,
                    "tie_atol": self.config.score_tie_atol,
                    "tie_rtol": self.config.score_tie_rtol,
                },
            )

    def _fixtures(self, directory: Path):
        runtime = self.ToyRuntime()
        trajectory = runtime.capture_clean(
            sample_ids=["gpqa_diamond:train:7"],
            raw_sample_ids=["raw-7"],
            raw_indices=[7],
            questions=["question"],
            gold_labels=["A"],
            batch_boundaries=[(0, 1)],
            include_generation=False,
        )
        trajectory.clean_generation_audit = [{"final_text": "Final Choice: A"}]
        trajectory_path = directory / "trajectory.pt"
        torch.save(trajectory, trajectory_path)
        role_devices = dict(trajectory.provenance["role_devices"])
        relay_transfer_mode = str(
            trajectory.provenance["runtime_config"]["relay_transfer_mode"]
        )
        release_source_sha256 = inference_mas.source_tree_sha256()

        latents = {"p2c": {}, "c2s": {}, "s2p": {}}
        for edge in ("p2c@0", "c2s@0", "s2p@0", "p2c@1", "c2s@1"):
            site, round_text = edge.split("@")
            latents[site][int(round_text)] = trajectory.message(edge, receiver=False)
        trace = {
            "metadata": {
                "provenance_schema_version": RELEASE_PROVENANCE_SCHEMA_VERSION,
                "source_tree_sha256": release_source_sha256,
                "dataset": "gpqa",
                "style": "sequential_light",
                "method": "ours_recursive",
                "R": 2,
                "seed": 42,
                "num_samples": 1,
                "trace_sites": ["p2c", "c2s", "s2p"],
                "trace_rounds": [0, 1],
                "trace_dtype": "float32",
                "role_devices": role_devices,
                "relay_transfer_mode": relay_transfer_mode,
            },
            "sample_ids": ["gpqa_diamond:train:7"],
            "sample_indices": [7],
            "latents": latents,
        }
        trace_path = directory / "legacy_trace.pt"
        torch.save(trace, trace_path)

        generation_config = {
            "provenance_schema_version": RELEASE_PROVENANCE_SCHEMA_VERSION,
            "source_tree_sha256": release_source_sha256,
            "experiment": {
                "style": "sequential_light",
                "method": "ours_recursive",
                "num_recursive_rounds": 2,
                "latent_steps": 1,
                "choice_old_prompt": 2,
                "solver_pre_question": 0,
                "planner_feedback_round_label_mode": "legacy",
            },
            "dataset": {"split": "train", "gpqa_option_shuffle": True},
            "generation": {
                "seed": 42,
                "num_rollouts": 1,
                "batch_size": 1,
                "max_new_tokens": 4000,
                "do_sample": False,
                "ans": True,
                "enable_thinking": False,
            },
            "runtime": {
                "role_devices": role_devices,
                "relay_transfer_mode": relay_transfer_mode,
            },
        }
        summary = {
            "type": "summary",
            "generation_config": generation_config,
            "generation_config_sha256": inference_mas.stable_json_sha256(generation_config),
        }
        results_path = directory / "legacy.jsonl"
        atomic_write_jsonl(
            results_path,
            [
                {
                    "sample_id": "gpqa_diamond:train:7",
                    "question": "question",
                    "ground_truth": "A",
                    "raw_final_output": "Final Choice: A",
                },
                summary,
            ],
        )
        return trajectory_path, trace_path, results_path

    @staticmethod
    def _rewrite_results(results_path: Path, mutate) -> None:
        rows = load_jsonl(results_path)
        summaries = [row for row in rows if row.get("type") == "summary"]
        if len(summaries) != 1:
            raise AssertionError(f"expected one summary row, found {len(summaries)}")
        generation_config = summaries[0]["generation_config"]
        mutate(generation_config)
        summaries[0]["generation_config_sha256"] = (
            inference_mas.stable_json_sha256(generation_config)
        )
        atomic_write_jsonl(results_path, rows)

    @staticmethod
    def _rewrite_trace(trace_path: Path, mutate) -> None:
        trace = torch.load(trace_path, map_location="cpu", weights_only=False)
        mutate(trace["metadata"])
        torch.save(trace, trace_path)

    def _assert_rejected(
        self,
        trajectory: Path,
        trace: Path,
        results: Path,
    ) -> None:
        report = build_equivalence_report(
            trajectory_path=trajectory,
            legacy_trace_path=trace,
            legacy_results_path=results,
            repo_root=REPO_ROOT,
        )
        self.assertFalse(report["passed"])
        self.assertTrue(
            any(not check["passed"] for check in report["checks"]),
            report,
        )
        with self.assertRaises(ContractError):
            verify_equivalence_report(report, repo_root=REPO_ROOT)

    def test_real_artifacts_are_recomputed_and_authenticated(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            trajectory, trace, results = self._fixtures(directory)
            report = build_equivalence_report(
                trajectory_path=trajectory,
                legacy_trace_path=trace,
                legacy_results_path=results,
                repo_root=REPO_ROOT,
            )
            self.assertTrue(report["passed"])
            self.assertEqual(
                verify_equivalence_report(report, repo_root=REPO_ROOT), report
            )

    def test_boolean_or_artifact_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            trajectory, trace, results = self._fixtures(directory)
            report = build_equivalence_report(
                trajectory_path=trajectory,
                legacy_trace_path=trace,
                legacy_results_path=results,
                repo_root=REPO_ROOT,
            )
            report["checks"][0]["passed"] = False
            with self.assertRaises(ContractError):
                verify_equivalence_report(report, repo_root=REPO_ROOT)

    def test_rehashed_result_topology_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            trajectory, trace, results = self._fixtures(Path(raw_directory))

            def mutate(config):
                config["runtime"]["role_devices"]["critic"] = "cuda:1"

            self._rewrite_results(results, mutate)
            self._assert_rejected(trajectory, trace, results)

    def test_trace_topology_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            trajectory, trace, results = self._fixtures(Path(raw_directory))

            def mutate(metadata):
                metadata["role_devices"]["solver"] = "cuda:2"

            self._rewrite_trace(trace, mutate)
            self._assert_rejected(trajectory, trace, results)

    def test_result_execution_fields_are_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            trajectory, trace, results = self._fixtures(Path(raw_directory))

            def mutate(config):
                config["runtime"].pop("role_devices")
                config["runtime"].pop("relay_transfer_mode")

            self._rewrite_results(results, mutate)
            self._assert_rejected(trajectory, trace, results)

    def test_trace_execution_fields_are_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            trajectory, trace, results = self._fixtures(Path(raw_directory))

            def mutate(metadata):
                metadata.pop("role_devices")
                metadata.pop("relay_transfer_mode")

            self._rewrite_trace(trace, mutate)
            self._assert_rejected(trajectory, trace, results)

    def test_wrong_result_relay_policy_is_rejected_after_rehash(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            trajectory, trace, results = self._fixtures(Path(raw_directory))

            def mutate(config):
                config["runtime"]["relay_transfer_mode"] = "direct"

            self._rewrite_results(results, mutate)
            self._assert_rejected(trajectory, trace, results)

    def test_wrong_trace_relay_policy_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            trajectory, trace, results = self._fixtures(Path(raw_directory))

            def mutate(metadata):
                metadata["relay_transfer_mode"] = "direct"

            self._rewrite_trace(trace, mutate)
            self._assert_rejected(trajectory, trace, results)

    def test_mutually_stale_release_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            trajectory, trace, results = self._fixtures(Path(raw_directory))
            stale_source = "0" * 64

            def mutate_results(config):
                config["source_tree_sha256"] = stale_source

            def mutate_trace(metadata):
                metadata["source_tree_sha256"] = stale_source

            self._rewrite_results(results, mutate_results)
            self._rewrite_trace(trace, mutate_trace)
            self._assert_rejected(trajectory, trace, results)

    def test_mutually_consistent_but_wrong_topology_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            trajectory, trace, results = self._fixtures(Path(raw_directory))
            wrong_topology = {
                "planner": "cuda:0",
                "critic": "cuda:1",
                "solver": "cuda:2",
                "terminal_solver": "cuda:3",
            }

            def mutate_results(config):
                config["runtime"]["role_devices"] = dict(wrong_topology)

            def mutate_trace(metadata):
                metadata["role_devices"] = dict(wrong_topology)

            self._rewrite_results(results, mutate_results)
            self._rewrite_trace(trace, mutate_trace)
            self._assert_rejected(trajectory, trace, results)


if __name__ == "__main__":
    unittest.main()
