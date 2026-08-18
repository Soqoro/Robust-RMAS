import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from RecursiveMAS.inference_utils.linkradius_runtime import (
    ForcedChoiceBatch,
    LinkRadiusRuntime,
    RelayEmission,
    RuntimeConfig,
)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class RuntimeGradientTests(unittest.TestCase):
    class DifferentiableToyRuntime(LinkRadiusRuntime):
        def __init__(self):
            super().__init__(RuntimeConfig(rounds=1, latent_steps=1, batch_size=1, device="cpu"))
            self.scored_batch_sizes = []
            self.scored_candidate_labels = []

        @property
        def device(self):
            return torch.device("cpu")

        @staticmethod
        def emit(value):
            return RelayEmission(value, value, "float32", "float32")

        def run_initial_planner(self, questions, **kwargs):
            return self.emit(torch.ones((len(questions), 1, 2)))

        def run_critic(self, questions, planner_message, round_idx, **kwargs):
            return self.emit(planner_message.clone())

        def score_terminal(self, questions, critic_message, **kwargs):
            self.scored_batch_sizes.append(len(questions))
            selected = tuple(kwargs.get("candidate_labels") or ("A", "B", "C", "D"))
            self.scored_candidate_labels.append(selected)
            x = critic_message.float().mean(dim=(1, 2))
            all_scores = torch.stack((x, -x, -x - 1.0, -x - 2.0), dim=-1)
            columns = [("A", "B", "C", "D").index(label) for label in selected]
            scores = all_scores[:, columns]
            row = scores.detach().cpu().tolist()
            predictions = tuple(
                selected[max(range(len(values)), key=values.__getitem__)]
                for values in row
            )
            return ForcedChoiceBatch(
                labels=selected,
                scores=scores,
                summed_logprobs=scores,
                mean_logprobs=scores,
                token_counts=torch.ones_like(scores, dtype=torch.long),
                predictions=predictions,
                score_ties=tuple(False for _ in row),
                encodings={},
                metadata={
                    "scorer_version": self.config.scorer_version,
                    "normalization": self.config.scorer_normalization,
                    "prefix": self.config.scorer_prefix,
                    "tie_atol": self.config.score_tie_atol,
                    "tie_rtol": self.config.score_tie_rtol,
                },
            )

    def setUp(self):
        self.runtime = self.DifferentiableToyRuntime()
        self.trajectory = self.runtime.capture_clean(
            sample_ids=["sample"],
            raw_sample_ids=["raw"],
            raw_indices=[0],
            questions=["q"],
            gold_labels=["A"],
            batch_boundaries=[(0, 1)],
            include_generation=False,
        )

    def test_terminal_consumer_leaf_gradient(self):
        clean = self.trajectory.message("c2s@0").clone().requires_grad_(True)
        full = self.runtime.score_terminal(
            self.trajectory.questions,
            clean,
            differentiable=True,
        )
        full_objective = self.runtime._score_margin_tensor(full, "A", "B")
        full_gradient = torch.autograd.grad(full_objective, clean)[0]
        self.runtime.scored_candidate_labels.clear()

        result = self.runtime.terminal_gradient(self.trajectory, target_label="B")
        self.assertEqual(result.autograd_semantics, "continuous_consumer_input")
        self.assertGreater(result.gradient_norm, 0.0)
        self.assertTrue(torch.isfinite(result.gradient).all())
        self.assertAlmostEqual(result.objective_value, float(full_objective.detach()))
        self.assertTrue(torch.allclose(result.gradient, full_gradient))
        self.assertEqual(self.runtime.scored_candidate_labels, [("A",), ("B",)])

    def test_early_edge_gradient_scores_one_candidate_graph_at_a_time(self):
        self.runtime.scored_candidate_labels.clear()
        result = self.runtime.autograd_gradient(
            self.trajectory,
            "p2c@0",
            target_label="B",
        )
        self.assertGreater(result.gradient_norm, 0.0)
        self.assertTrue(torch.isfinite(result.gradient).all())
        self.assertEqual(self.runtime.scored_candidate_labels, [("A",), ("B",)])

    def test_terminal_pgd_improves_target_margin_and_respects_realized_budget(self):
        self.runtime.scored_candidate_labels.clear()
        result = self.runtime.autograd_pgd(
            self.trajectory,
            "c2s@0",
            epsilon=1.0,
            steps=4,
            targets=["B"],
        )
        target = result.targets[0]
        self.assertEqual(result.autograd_semantics, "relaxed_autograd")
        self.assertTrue(target.improved)
        self.assertLess(target.final_margin, target.initial_margin)
        self.assertTrue(target.budget_respected)
        self.assertLessEqual(target.realized_delta_norm, target.budget + 1e-6)
        self.assertEqual(self.runtime.scored_candidate_labels[-1], ("A", "B", "C", "D"))
        self.assertTrue(
            all(
                len(labels) == 1
                for labels in self.runtime.scored_candidate_labels[:-1]
            )
        )

    def test_gradient_selects_one_row_without_rebatching_frozen_context(self):
        trajectory = self.runtime.capture_clean(
            sample_ids=["filler-0", "eligible", "filler-2"],
            raw_sample_ids=["raw-0", "raw-1", "raw-2"],
            raw_indices=[0, 1, 2],
            questions=["q0", "q1", "q2"],
            gold_labels=["A", "A", "A"],
            batch_boundaries=[(0, 3)],
            analysis_eligibility_mask=[False, True, False],
            include_generation=False,
        )
        self.runtime.scored_batch_sizes.clear()
        self.runtime.scored_candidate_labels.clear()
        result = self.runtime.terminal_gradient(
            trajectory,
            target_label="B",
            sample_index=1,
        )
        self.assertEqual(result.sample_index, 1)
        self.assertEqual(result.sample_id, "eligible")
        self.assertEqual(tuple(result.gradient.shape), (1, 1, 2))
        self.assertGreater(result.gradient_norm, 0.0)
        self.assertEqual(self.runtime.scored_batch_sizes, [3, 3])
        self.assertEqual(self.runtime.scored_candidate_labels, [("A",), ("B",)])
        with self.assertRaisesRegex(ValueError, "analysis-eligible"):
            self.runtime.terminal_gradient(trajectory, sample_index=0)

    def test_pgd_selects_one_row_without_rebatching_frozen_context(self):
        trajectory = self.runtime.capture_clean(
            sample_ids=["filler-0", "eligible", "filler-2"],
            raw_sample_ids=["raw-0", "raw-1", "raw-2"],
            raw_indices=[0, 1, 2],
            questions=["q0", "q1", "q2"],
            gold_labels=["A", "A", "A"],
            batch_boundaries=[(0, 3)],
            analysis_eligibility_mask=[False, True, False],
            include_generation=False,
        )
        self.runtime.scored_batch_sizes.clear()
        result = self.runtime.autograd_pgd(
            trajectory,
            "c2s@0",
            epsilon=1.0,
            steps=2,
            targets=["B"],
            sample_index=1,
        )
        self.assertEqual(result.sample_index, 1)
        self.assertEqual(result.sample_id, "eligible")
        self.assertTrue(result.targets[0].improved)
        self.assertTrue(result.targets[0].budget_respected)
        self.assertTrue(self.runtime.scored_batch_sizes)
        self.assertEqual(set(self.runtime.scored_batch_sizes), {3})


if __name__ == "__main__":
    unittest.main()
