from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from RecursiveMAS.inference_utils.linkradius import Edge, valid_edges
from RecursiveMAS.inference_utils.linkradius_runtime import (
    ForcedChoiceBatch,
    LinkRadiusRuntime,
    RelayEmission,
    ReplayIntervention,
    RuntimeConfig,
    _portable_artifact_identity,
    clean_audit_rows,
)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class RuntimeReplayTests(unittest.TestCase):
    class ToyRuntime(LinkRadiusRuntime):
        def __init__(self):
            super().__init__(RuntimeConfig(rounds=2, latent_steps=1, batch_size=1, device="cpu"))
            self.calls = []

        @staticmethod
        def emit(value):
            return RelayEmission(value, value, "float32", "float32")

        def run_initial_planner(self, questions, **kwargs):
            self.calls.append("planner_initial@0")
            return self.emit(torch.ones((len(questions), 2, 3)))

        def run_critic(self, questions, planner_message, round_idx, **kwargs):
            self.calls.append(f"critic@{round_idx}")
            return self.emit(planner_message + 1.0)

        def run_solver_feedback(self, questions, critic_message, round_idx, **kwargs):
            self.calls.append(f"solver_feedback@{round_idx}")
            return self.emit(critic_message + 1.0)

        def run_planner_feedback(self, questions, solver_message, round_idx, **kwargs):
            self.calls.append(f"planner_feedback@{round_idx}")
            return self.emit(solver_message + 1.0)

        def score_terminal(self, questions, critic_message, **kwargs):
            self.calls.append("score_final@1")
            base = critic_message.float().mean(dim=(1, 2))
            scores = torch.stack((base, base - 1, base - 2, base - 3), dim=-1)
            count = torch.ones_like(scores, dtype=torch.long)
            return ForcedChoiceBatch(
                labels=("A", "B", "C", "D"),
                scores=scores,
                summed_logprobs=scores,
                mean_logprobs=scores,
                token_counts=count,
                predictions=tuple("A" for _ in questions),
                score_ties=tuple(False for _ in questions),
                encodings={},
                metadata={
                    "toy": True,
                    "scorer_version": self.config.scorer_version,
                    "normalization": self.config.scorer_normalization,
                    "prefix": self.config.scorer_prefix,
                    "tie_atol": self.config.score_tie_atol,
                    "tie_rtol": self.config.score_tie_rtol,
                },
            )

    def setUp(self):
        self.runtime = self.ToyRuntime()
        self.trajectory = self.runtime.capture_clean(
            sample_ids=["sample"],
            raw_sample_ids=["raw"],
            raw_indices=[7],
            questions=["q"],
            gold_labels=["A"],
            batch_boundaries=[(0, 1)],
            include_generation=False,
        )

    def test_capture_has_every_transport_and_receiver_reference(self):
        self.assertEqual(set(self.trajectory.transport_messages), set(valid_edges(2)))
        self.assertEqual(set(self.trajectory.receiver_reference_messages), set(valid_edges(2)))
        for edge in valid_edges(2):
            self.assertEqual(self.trajectory.message(edge).dtype, torch.float32)
            self.assertEqual(self.trajectory.message(edge, receiver=False).dtype, torch.float32)

    def test_early_replay_recomputes_only_exact_descendants(self):
        self.runtime.calls.clear()
        result = self.runtime.replay(self.trajectory, Edge("p2c", 0), "identity")
        self.assertEqual(
            self.runtime.calls,
            ["critic@0", "solver_feedback@0", "planner_feedback@1", "critic@1", "score_final@1"],
        )
        self.assertEqual(
            [step.token for step in result.schedule],
            ["critic@0", "solver_feedback@0", "planner_feedback@1", "critic@1", "score_final@1"],
        )

    def test_terminal_replay_scores_without_recomputing_an_agent(self):
        self.runtime.calls.clear()
        self.runtime.replay(self.trajectory, Edge("c2s", 1), "identity")
        self.assertEqual(self.runtime.calls, ["score_final@1"])

    def test_zero_additive_delta_is_identity(self):
        clean = self.runtime.replay(self.trajectory, Edge("s2p", 0), "identity")
        zero = self.runtime.replay(
            self.trajectory,
            Edge("s2p", 0),
            ReplayIntervention(
                mode="additive",
                delta=torch.zeros_like(self.trajectory.message(Edge("s2p", 0))),
            ),
        )
        self.assertTrue(torch.equal(clean.scoring.scores, zero.scoring.scores))

    def test_invalid_terminal_feedback_fails_before_stage_execution(self):
        self.runtime.calls.clear()
        with self.assertRaises(ValueError):
            self.runtime.replay(self.trajectory, "s2p@1", "identity")
        self.assertEqual(self.runtime.calls, [])

    def test_replay_rejects_runtime_configuration_drift(self):
        self.runtime.config.latent_steps = 2
        self.runtime.calls.clear()
        with self.assertRaisesRegex(ValueError, "runtime configuration differs"):
            self.runtime.replay(self.trajectory, "c2s@1", "identity")
        self.assertEqual(self.runtime.calls, [])

    def test_direct_mismatch_rejects_unchecked_replacement(self):
        replacement = torch.zeros_like(self.trajectory.message("c2s@1"))
        with self.assertRaisesRegex(ValueError, "unchecked replacement"):
            self.runtime.replay(
                self.trajectory,
                "c2s@1",
                ReplayIntervention(mode="mismatch", replacement=replacement),
            )

    def test_replacement_mode_remains_available_for_validated_external_controls(self):
        replacement = torch.zeros_like(self.trajectory.message("c2s@1"))
        result = self.runtime.replay(
            self.trajectory,
            "c2s@1",
            ReplayIntervention(mode="replacement", replacement=replacement),
        )
        self.assertEqual(result.intervention_metadata[0]["mode"], "replacement")

    def test_direct_mismatch_requires_deterministic_label_matched_mapping(self):
        trajectory = self.runtime.capture_clean(
            sample_ids=["sample-a", "sample-b"],
            raw_sample_ids=["raw-a", "raw-b"],
            raw_indices=[0, 1],
            questions=["qa", "qb"],
            gold_labels=["A", "A"],
            batch_boundaries=[(0, 2)],
            include_generation=False,
        )
        result = self.runtime.replay(
            trajectory,
            "c2s@1",
            ReplayIntervention(mode="mismatch", donor_indices=[1, 0], seed=42),
        )
        self.assertEqual(
            [row["donor_raw_sample_id"] for row in result.intervention_metadata],
            ["raw-b", "raw-a"],
        )
        with self.assertRaisesRegex(ValueError, "deterministic cyclic mapping"):
            self.runtime.replay(
                trajectory,
                "c2s@1",
                ReplayIntervention(mode="mismatch", donor_indices=[0, 1], seed=42),
            )

    def test_manifest_eligibility_does_not_change_underlying_dual_correctness(self):
        self.trajectory.analysis_eligibility_mask = [False]
        self.trajectory.clean_generation_audit = [
            {
                "strict_choice": "A",
                "answer_invalid": False,
                "answer_conflict": False,
            }
        ]
        row = clean_audit_rows(self.trajectory)[0]
        self.assertTrue(row["dual_correct"])
        self.assertFalse(row["analysis_eligible"])
        self.assertFalse(row["analysis_dual_correct"])
        self.assertEqual(
            row["analysis_exclusion_reasons"],
            ["execution_manifest_ineligible"],
        )
        self.assertEqual(row["dual_correct_exclusion_reasons"], [])


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PortableSystemIdentityTests(unittest.TestCase):
    @staticmethod
    def runtime(cache_root: Path) -> LinkRadiusRuntime:
        identity = torch.nn.Identity()
        agent = SimpleNamespace(model=identity, tokenizer=object(), inner_adapter=identity)
        repo_ids = {
            "planner": "Org/Planner",
            "critic": "Org/Critic",
            "solver": "Org/Solver",
            "outer": "Org/Outer",
        }
        repo_paths = {
            role: cache_root / f"models--Org--{role.title()}" / "snapshots" / "revision-1"
            for role in ("planner", "critic", "solver", "outer")
        }
        paths = SimpleNamespace(
            style="sequential_light",
            family="sequential",
            dataset="gpqa",
            repo_ids=repo_ids,
            repo_paths=repo_paths,
            inner_adapter_paths={
                role: cache_root / f"models--Org--{role.title()}" / "blobs" / f"{role}-blob"
                for role in ("planner", "critic", "solver")
            },
            outer_adapter_paths={
                key: cache_root / "models--Org--Outer" / "blobs" / f"{key}-blob"
                for key in ("outer_12", "outer_23", "outer_31")
            },
        )
        system = SimpleNamespace(
            style="sequential_light",
            family="sequential",
            dataset="gpqa",
            agents={role: agent for role in ("planner", "critic", "solver")},
            outer_adapters={key: identity for key in ("outer_12", "outer_23", "outer_31")},
            paths=paths,
        )
        return LinkRadiusRuntime(
            RuntimeConfig(rounds=1, latent_steps=1, device="cpu"),
            system=system,
        )

    def test_cache_root_changes_only_diagnostic_paths(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_provenance = self.runtime(Path(first))._system_provenance()
            second_provenance = self.runtime(Path(second))._system_provenance()
        self.assertEqual(first_provenance["model_hash"], second_provenance["model_hash"])
        self.assertEqual(first_provenance["adapter_hash"], second_provenance["adapter_hash"])
        self.assertEqual(
            first_provenance["system_resolution"],
            second_provenance["system_resolution"],
        )
        self.assertNotEqual(
            first_provenance["system_diagnostic_paths"],
            second_provenance["system_diagnostic_paths"],
        )

    def test_local_identity_is_path_independent_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_path = Path(first) / "adapter.pt"
            second_path = Path(second) / "adapter.pt"
            first_path.write_bytes(b"same adapter bytes")
            second_path.write_bytes(b"same adapter bytes")
            first_identity = _portable_artifact_identity(first_path, repo_id="Org/Adapter")
            second_identity = _portable_artifact_identity(second_path, repo_id="Org/Adapter")
            self.assertEqual(first_identity, second_identity)
            second_path.write_bytes(b"changed adapter bytes")
            changed_identity = _portable_artifact_identity(second_path, repo_id="Org/Adapter")
        self.assertNotEqual(first_identity, changed_identity)


if __name__ == "__main__":
    unittest.main()
