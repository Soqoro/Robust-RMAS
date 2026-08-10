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


class _Identity(torch.nn.Module if torch is not None else object):
    def forward(self, value):
        return value


@unittest.skipIf(torch is None, "PyTorch is not installed")
class EndToEndToyScorerTests(unittest.TestCase):
    @staticmethod
    def runtime(rounds=1):
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
            RuntimeConfig(rounds=rounds, latent_steps=1, batch_size=1, device="cpu"),
            system=system,
        )

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


if __name__ == "__main__":
    unittest.main()
