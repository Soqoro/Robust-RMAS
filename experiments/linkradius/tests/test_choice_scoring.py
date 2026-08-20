from types import SimpleNamespace
import unittest

try:
    import torch
except ModuleNotFoundError:  # Lightweight CPU-only manifest environment.
    torch = None

from RecursiveMAS.inference_utils.linkradius_runtime import (
    LinkRadiusRuntime,
    RuntimeConfig,
    causal_token_log_probs,
    prediction_from_scores,
    tokenize_joint_candidate,
    tokenize_joint_candidates,
)


class _BoundaryTokenizer:
    """Tiny tokenizer with a prefix/candidate boundary merge."""

    def __init__(self, offsets=True):
        self.offsets = offsets

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        table = {
            "P{": ([10, 11], [(0, 1), (1, 2)]),
            "P{A}": ([10, 101, 200], [(0, 1), (1, 3), (3, 4)]),
            "P{B}": ([10, 102, 200], [(0, 1), (1, 3), (3, 4)]),
            "P{C}": ([10, 103, 200], [(0, 1), (1, 3), (3, 4)]),
            "P{D}": ([10, 104, 200], [(0, 1), (1, 3), (3, 4)]),
        }
        ids, offsets = table[text]
        result = {"input_ids": ids}
        if return_offsets_mapping and self.offsets:
            result["offset_mapping"] = offsets
        return result


class ChoiceScoringTests(unittest.TestCase):
    def test_joint_tokenization_includes_boundary_merge_with_offsets(self):
        encodings = tokenize_joint_candidates(
            _BoundaryTokenizer(offsets=True),
            "P{",
            {label: f"{label}}}" for label in "ABCD"},
        )
        self.assertEqual(encodings["A"].candidate_start, 1)
        self.assertEqual(encodings["A"].candidate_token_ids, (101, 200))
        self.assertEqual(encodings["A"].span_method, "offset_mapping")

    def test_lcp_fallback_includes_boundary_merge(self):
        encodings = tokenize_joint_candidates(
            _BoundaryTokenizer(offsets=False),
            "P{",
            {label: f"{label}}}" for label in "ABCD"},
        )
        self.assertEqual(encodings["B"].candidate_start, 1)
        self.assertEqual(encodings["B"].candidate_token_ids, (102, 200))
        self.assertEqual(encodings["B"].span_method, "longest_common_token_prefix")

    def test_single_candidate_lcp_uses_prefix_only_comparison(self):
        encoding = tokenize_joint_candidate(
            _BoundaryTokenizer(offsets=False),
            "P{",
            "A}",
            label="A",
        )
        self.assertEqual(encoding.candidate_start, 1)
        self.assertEqual(encoding.candidate_token_ids, (101, 200))

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_causal_alignment_and_float32_log_softmax(self):
        logits = torch.zeros((1, 4, 5), dtype=torch.float16)
        logits[0, 1, 3] = 2.0
        logits[0, 2, 4] = 1.0
        gathered = causal_token_log_probs(logits, [[3, 4]], [[2, 3]])
        expected_first = torch.log_softmax(logits[0, 1].float(), dim=-1)[3]
        expected_second = torch.log_softmax(logits[0, 2].float(), dim=-1)[4]
        self.assertEqual(gathered.dtype, torch.float32)
        self.assertTrue(torch.allclose(gathered[0], torch.stack((expected_first, expected_second))))

    def test_ties_are_not_broken_alphabetically(self):
        self.assertEqual(prediction_from_scores([1.0, 1.0, 0.0, -1.0]), (None, True))
        self.assertEqual(prediction_from_scores([1.0, 0.9, 0.8, 0.7]), ("A", False))


class _CharTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    chat_template = "toy-chat-v1"

    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
        enable_thinking=False,
    ):
        text = "<s>" + "|".join(message["content"] for message in messages) + "|assistant:"
        return self(text, add_special_tokens=False)["input_ids"] if tokenize else text

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
        if isinstance(text, list):
            raise AssertionError("The score-only toy never batch-tokenizes text")
        ids = [2 + (ord(character) % 110) for character in text]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result

    def decode(self, ids, skip_special_tokens=False):
        return "".join("?" for _ in ids)


class _ToyLM(torch.nn.Module if torch is not None else object):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 8)
        bias = torch.zeros(128)
        # Char IDs are 2 + ord(char) % 110. D receives the largest score.
        for rank, label in enumerate("ABCD", start=1):
            bias[2 + ord(label) % 110] = float(rank)
        self.register_buffer("bias", bias)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask=None, logits_to_keep=None, **kwargs):
        length = inputs_embeds.size(1)
        keep = min(length, int(logits_to_keep)) if logits_to_keep is not None else length
        # Keep a zero-valued dependency on inputs so differentiable callers retain
        # the correct graph even though this toy's probabilities are positional constants.
        dependency = inputs_embeds[:, -keep:, :].sum(dim=-1, keepdim=True) * 0.0
        logits = self.bias.view(1, 1, -1) + dependency
        return SimpleNamespace(logits=logits, hidden_states=(inputs_embeds,))


class _CumulativeToyLM(_ToyLM):
    """Toy hidden states where every relay token influences the final state."""

    def forward(self, inputs_embeds, attention_mask=None, logits_to_keep=None, **kwargs):
        hidden = inputs_embeds.cumsum(dim=1)
        length = hidden.size(1)
        keep = min(length, int(logits_to_keep)) if logits_to_keep is not None else length
        dependency = hidden[:, -keep:, :].sum(dim=-1, keepdim=True) * 0.0
        logits = self.bias.view(1, 1, -1) + dependency
        return SimpleNamespace(logits=logits, hidden_states=(hidden,))


class _GradientToyLM(_ToyLM):
    """Toy LM whose candidate scores have a nonzero input-embedding gradient."""

    def forward(self, inputs_embeds, attention_mask=None, logits_to_keep=None, **kwargs):
        hidden = inputs_embeds.cumsum(dim=1)
        length = hidden.size(1)
        keep = min(length, int(logits_to_keep)) if logits_to_keep is not None else length
        signal = hidden[:, -keep:, :].sum(dim=-1, keepdim=True)
        scale = torch.linspace(
            -0.5,
            0.5,
            self.bias.numel(),
            dtype=signal.dtype,
            device=signal.device,
        ).view(1, 1, -1)
        logits = self.bias.to(dtype=signal.dtype).view(1, 1, -1) + signal * scale
        return SimpleNamespace(logits=logits, hidden_states=(hidden,))


class _Identity(torch.nn.Module if torch is not None else object):
    def forward(self, value):
        return value


@unittest.skipIf(torch is None, "PyTorch is not installed")
class EndToEndToyScorerTests(unittest.TestCase):
    @staticmethod
    def runtime(rounds=1, *, autograd_memory_mode="none"):
        tokenizer = _CharTokenizer()
        model = _ToyLM()
        agent = SimpleNamespace(model=model, tokenizer=tokenizer, inner_adapter=_Identity())
        system = SimpleNamespace(
            family="sequential",
            agents={"planner": agent, "critic": agent, "solver": agent},
            outer_adapters={
                "outer_12": _Identity(),
                "outer_23": _Identity(),
                "outer_31": _Identity(),
            },
        )
        return LinkRadiusRuntime(
            RuntimeConfig(
                rounds=rounds,
                latent_steps=1,
                batch_size=1,
                device="cpu",
                autograd_memory_mode=autograd_memory_mode,
            ),
            system=system,
        )

    def test_terminal_scoring_uses_the_dedicated_replica(self):
        tokenizer = _CharTokenizer()
        primary = _ToyLM()
        terminal = _ToyLM()
        with torch.no_grad():
            terminal.bias.zero_()
            terminal.bias[2 + ord("A") % 110] = 20.0
        primary_agent = SimpleNamespace(
            model=primary,
            tokenizer=tokenizer,
            inner_adapter=_Identity(),
        )
        terminal_agent = SimpleNamespace(
            model=terminal,
            tokenizer=tokenizer,
            inner_adapter=None,
        )
        system = SimpleNamespace(
            family="sequential",
            agents={
                "planner": primary_agent,
                "critic": primary_agent,
                "solver": primary_agent,
            },
            terminal_solver=terminal_agent,
            outer_adapters={
                "outer_12": _Identity(),
                "outer_23": _Identity(),
                "outer_31": _Identity(),
            },
        )
        runtime = LinkRadiusRuntime(
            RuntimeConfig(rounds=1, latent_steps=1, batch_size=1, device="cpu"),
            system=system,
        )
        result = runtime.score_terminal(
            ["Question\nA. x\nB. y\nC. z\nD. w"],
            torch.zeros((1, 2, 8)),
        )
        self.assertEqual(result.predictions, ("A",))

    def test_checkpointed_terminal_scorer_matches_scores_and_gradients(self):
        tokenizer = _CharTokenizer()

        def evaluate(mode):
            torch.manual_seed(1234)
            model = _GradientToyLM()
            agent = SimpleNamespace(
                model=model,
                tokenizer=tokenizer,
                inner_adapter=_Identity(),
            )
            terminal_model = _GradientToyLM()
            terminal_model.load_state_dict(model.state_dict())
            terminal_agent = SimpleNamespace(
                model=terminal_model,
                tokenizer=tokenizer,
                inner_adapter=None,
            )
            system = SimpleNamespace(
                family="sequential",
                agents={"planner": agent, "critic": agent, "solver": agent},
                terminal_solver=terminal_agent,
                outer_adapters={
                    "outer_12": _Identity(),
                    "outer_23": _Identity(),
                    "outer_31": _Identity(),
                },
            )
            runtime = LinkRadiusRuntime(
                RuntimeConfig(
                    rounds=1,
                    latent_steps=1,
                    batch_size=1,
                    device="cpu",
                    autograd_memory_mode=mode,
                ),
                system=system,
            )
            relay = torch.linspace(-1.0, 1.0, 16).reshape(1, 2, 8)
            relay.requires_grad_(True)
            scoring = runtime.score_terminal(
                ["Question\nA. x\nB. y\nC. z\nD. w"],
                relay,
                differentiable=True,
                candidate_labels=("A", "D"),
            )
            margin = scoring.scores[0, 0] - scoring.scores[0, 1]
            gradient = torch.autograd.grad(margin, relay)[0]
            return scoring.scores.detach(), gradient.detach()

        ordinary_scores, ordinary_gradient = evaluate("none")
        checkpoint_scores, checkpoint_gradient = evaluate("checkpoint")
        self.assertTrue(torch.allclose(checkpoint_scores, ordinary_scores))
        self.assertTrue(torch.allclose(checkpoint_gradient, ordinary_gradient))
        self.assertGreater(float(checkpoint_gradient.abs().sum()), 0.0)

    def test_checkpointed_latent_rollout_matches_output_and_gradient(self):
        tokenizer = _CharTokenizer()

        def evaluate(mode):
            torch.manual_seed(4321)
            model = _CumulativeToyLM()
            agent = SimpleNamespace(
                model=model,
                tokenizer=tokenizer,
                inner_adapter=_Identity(),
            )
            system = SimpleNamespace(
                family="sequential",
                agents={"planner": agent, "critic": agent, "solver": agent},
                outer_adapters={
                    "outer_12": _Identity(),
                    "outer_23": _Identity(),
                    "outer_31": _Identity(),
                },
            )
            runtime = LinkRadiusRuntime(
                RuntimeConfig(
                    rounds=2,
                    latent_steps=2,
                    batch_size=1,
                    device="cpu",
                    autograd_memory_mode=mode,
                ),
                system=system,
            )
            relay = torch.linspace(-1.0, 1.0, 16).reshape(1, 2, 8)
            relay.requires_grad_(True)
            emission = runtime.run_critic(
                ["Question\nA. x\nB. y\nC. z\nD. w"],
                relay,
                0,
                differentiable=True,
            )
            gradient = torch.autograd.grad(emission.receiver.square().sum(), relay)[0]
            return emission.receiver.detach(), gradient.detach()

        ordinary_output, ordinary_gradient = evaluate("none")
        checkpoint_output, checkpoint_gradient = evaluate("checkpoint")
        self.assertTrue(torch.allclose(checkpoint_output, ordinary_output))
        self.assertTrue(torch.allclose(checkpoint_gradient, ordinary_gradient))
        self.assertGreater(float(checkpoint_gradient.abs().sum()), 0.0)

    def test_checkpointed_early_edge_margin_matches_full_graph(self):
        tokenizer = _CharTokenizer()

        def evaluate(mode):
            torch.manual_seed(9876)
            model = _GradientToyLM()
            agent = SimpleNamespace(
                model=model,
                tokenizer=tokenizer,
                inner_adapter=_Identity(),
            )
            terminal_model = _GradientToyLM()
            terminal_model.load_state_dict(model.state_dict())
            terminal_agent = SimpleNamespace(
                model=terminal_model,
                tokenizer=tokenizer,
                inner_adapter=None,
            )
            system = SimpleNamespace(
                family="sequential",
                agents={"planner": agent, "critic": agent, "solver": agent},
                terminal_solver=terminal_agent,
                outer_adapters={
                    "outer_12": _Identity(),
                    "outer_23": _Identity(),
                    "outer_31": _Identity(),
                },
            )
            runtime = LinkRadiusRuntime(
                RuntimeConfig(
                    rounds=2,
                    latent_steps=2,
                    batch_size=1,
                    device="cpu",
                    autograd_memory_mode=mode,
                ),
                system=system,
            )
            trajectory = runtime.capture_clean(
                sample_ids=["sample"],
                raw_sample_ids=["raw"],
                raw_indices=[0],
                questions=["Question\nA. x\nB. y\nC. z\nD. w"],
                gold_labels=["D"],
                batch_boundaries=[(0, 1)],
                analysis_eligibility_mask=[True],
                include_generation=False,
            )
            result = runtime.autograd_gradient(
                trajectory,
                "p2c@0",
                target_label="A",
                sample_index=0,
            )
            return result.objective_value, result.gradient

        ordinary_value, ordinary_gradient = evaluate("none")
        checkpoint_value, checkpoint_gradient = evaluate("checkpoint")
        self.assertAlmostEqual(checkpoint_value, ordinary_value, places=6)
        self.assertTrue(
            torch.allclose(
                checkpoint_gradient,
                ordinary_gradient,
                atol=1e-6,
                rtol=1e-5,
            )
        )
        self.assertGreater(float(checkpoint_gradient.abs().sum()), 0.0)

    def test_terminal_scorer_returns_four_float32_mean_scores(self):
        runtime = self.runtime()
        result = runtime.score_terminal(
            ["Question\nA. x\nB. y\nC. z\nD. w"],
            torch.zeros((1, 2, 8)),
        )
        self.assertEqual(tuple(result.scores.shape), (1, 4))
        self.assertEqual(result.scores.dtype, torch.float32)
        self.assertEqual(result.predictions, ("D",))
        self.assertEqual(result.score_ties, (False,))
        self.assertEqual(result.token_counts.tolist(), [[2, 2, 2, 2]])
        self.assertTrue(torch.isfinite(result.scores).all())
        self.assertEqual(result.metadata["log_softmax_dtype"], "float32")
        self.assertEqual(set(result.metadata["joint_token_ids"]), set("ABCD"))
        self.assertEqual(
            result.metadata["candidate_token_counts"],
            {label: 2 for label in "ABCD"},
        )
        self.assertTrue(
            all(result.metadata["candidate_token_ids"][label] for label in "ABCD")
        )

    def test_candidate_subset_matches_full_scorer_columns(self):
        runtime = self.runtime()
        questions = ["Question\nA. x\nB. y\nC. z\nD. w"]
        relay = torch.zeros((1, 2, 8), requires_grad=True)
        full = runtime.score_terminal(questions, relay, differentiable=True)
        subset = runtime.score_terminal(
            questions,
            relay,
            differentiable=True,
            candidate_labels=("D", "B"),
        )
        self.assertEqual(subset.labels, ("D", "B"))
        self.assertTrue(
            torch.equal(
                subset.scores,
                full.scores[:, [full.labels.index("D"), full.labels.index("B")]],
            )
        )
        # Candidate span discovery remains bound to the complete frozen
        # comparison set even though only two model forwards are requested.
        self.assertEqual(set(subset.metadata["joint_token_ids"]), set("ABCD"))

    def test_candidate_subset_validation(self):
        runtime = self.runtime()
        questions = ["Question\nA. x\nB. y\nC. z\nD. w"]
        relay = torch.zeros((1, 2, 8))
        for invalid in ((), ("A", "A"), ("E",)):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "candidate_labels"):
                    runtime.score_terminal(
                        questions,
                        relay,
                        candidate_labels=invalid,
                    )

    def test_concrete_persistent_stages_capture_all_r2_edges(self):
        runtime = self.runtime(rounds=2)
        trajectory = runtime.capture_clean(
            sample_ids=["sample"],
            raw_sample_ids=["raw"],
            raw_indices=[0],
            questions=["Question\nA. x\nB. y\nC. z\nD. w"],
            gold_labels=["D"],
            batch_boundaries=[(0, 1)],
            include_generation=False,
        )
        self.assertEqual(
            {edge.edge_id for edge in trajectory.receiver_reference_messages},
            {"p2c@0", "c2s@0", "s2p@0", "p2c@1", "c2s@1"},
        )
        self.assertEqual(trajectory.clean_scoring.predictions, ("D",))

    def test_role_routing_and_differentiable_consumer_cast(self):
        tokenizer = _CharTokenizer()

        def agent(dtype):
            model = _CumulativeToyLM().to(dtype=dtype)
            return SimpleNamespace(
                model=model,
                tokenizer=tokenizer,
                inner_adapter=_Identity(),
            )

        system = SimpleNamespace(
            family="sequential",
            agents={
                "planner": agent(torch.float32),
                "critic": agent(torch.float64),
                "solver": agent(torch.float32),
            },
            outer_adapters={
                "outer_12": _Identity(),
                "outer_23": _Identity(),
                "outer_31": _Identity(),
            },
        )

        class RecordingRuntime(LinkRadiusRuntime):
            def __init__(self):
                self.role_requests = []
                super().__init__(
                    RuntimeConfig(rounds=2, latent_steps=1, batch_size=1, device="cpu"),
                    system=system,
                )

            @property
            def device(self):  # Any remaining global-device use must fail.
                raise AssertionError("concrete stages must use a role-specific device")

            def role_device(self, role):
                self.role_requests.append(role)
                return torch.device("cpu")

        runtime = RecordingRuntime()
        question = ["Question\nA. x\nB. y\nC. z\nD. w"]
        incoming = torch.ones((1, 2, 8), dtype=torch.float32, requires_grad=True)

        runtime.role_requests.clear()
        emission = runtime.run_critic(question, incoming, 0, differentiable=True)
        self.assertEqual(runtime.role_requests, ["critic", "solver"])
        self.assertEqual(emission.transport.dtype, torch.float64)
        self.assertEqual(emission.receiver.dtype, torch.float32)
        emission.receiver.sum().backward()
        self.assertIsNotNone(incoming.grad)
        self.assertGreater(float(incoming.grad.abs().sum()), 0.0)

        expected = (
            (lambda: runtime.run_initial_planner(question), {"planner", "critic"}),
            (
                lambda: runtime.run_critic(question, incoming.detach(), 1),
                {"critic", "terminal_solver"},
            ),
            (lambda: runtime.run_solver_feedback(question, incoming.detach(), 0), {"solver", "planner"}),
            (lambda: runtime.run_planner_feedback(question, incoming.detach(), 1), {"planner", "critic"}),
            (
                lambda: runtime.score_terminal(question, incoming.detach()),
                {"terminal_solver"},
            ),
        )
        for operation, roles in expected:
            runtime.role_requests.clear()
            operation()
            self.assertEqual(set(runtime.role_requests), roles)


if __name__ == "__main__":
    unittest.main()
