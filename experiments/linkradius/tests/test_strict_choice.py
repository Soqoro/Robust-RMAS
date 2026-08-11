from __future__ import annotations

import unittest

from RecursiveMAS.inference_utils.linkradius import (
    STRICT_CHOICE_VERSION,
    check_strict_choice,
    forced_choice_prediction,
    parse_strict_choice,
)


class StrictChoiceTests(unittest.TestCase):
    def test_no_default_a(self) -> None:
        for text in ("", "No final answer was produced.", "I compared option A against B."):
            result = parse_strict_choice(text)
            self.assertIsNone(result.choice)
            self.assertTrue(result.answer_invalid)
            self.assertFalse(result.answer_conflict)
            self.assertEqual(result.checker_version, STRICT_CHOICE_VERSION)

    def test_declared_terminal_forms(self) -> None:
        cases = {
            r"Reasoning. Therefore \boxed{B}": "B",
            r"Reasoning. Therefore the correct option is \(\boxed{A}\).": "A",
            r"Reasoning. Therefore the correct option is \[\boxed{D}\]": "D",
            "Reasoning mentions A.\nFinal Choice: C": "C",
            "Reasoning mentions A and D.\n(D)": "D",
            "Answer: a.": "A",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                result = parse_strict_choice(text)
                self.assertEqual(result.choice, expected)
                self.assertFalse(result.answer_invalid)

    def test_boxed_labels_mentioned_in_reasoning_are_not_answers(self) -> None:
        cases = (
            r"I considered \boxed{A}, but it is wrong; no final answer.",
            r"I considered \(\boxed{D}\), but did not choose it.",
            "Answer: B\nThat was only a guess, so no final answer.",
            "Quoted reasoning: `\\boxed{C}` is one candidate.",
        )
        for text in cases:
            with self.subTest(text=text):
                result = parse_strict_choice(text)
                self.assertIsNone(result.choice)
                self.assertTrue(result.answer_invalid)
                self.assertFalse(result.answer_conflict)

    def test_only_contiguous_terminal_answer_block_is_considered(self) -> None:
        result = parse_strict_choice(
            "Earlier speculation: \\boxed{A}\nNot final.\nFinal Choice: D"
        )
        self.assertEqual(result.choice, "D")
        self.assertEqual([span.label for span in result.matched_spans], ["D"])

    def test_conflicting_final_spans_fail_closed(self) -> None:
        result = parse_strict_choice(r"First \boxed{A}. Revised: \boxed{D}.")
        self.assertIsNone(result.choice)
        self.assertTrue(result.answer_invalid)
        self.assertTrue(result.answer_conflict)
        self.assertEqual({span.label for span in result.matched_spans}, {"A", "D"})

    def test_repeated_same_label_is_unambiguous(self) -> None:
        result = parse_strict_choice("Answer: B\nFinal Choice: B")
        self.assertEqual(result.choice, "B")
        self.assertGreaterEqual(len(result.matched_spans), 2)

    def test_ambiguous_explicit_line_is_invalid(self) -> None:
        self.assertTrue(parse_strict_choice("Final Choice: A or B").answer_invalid)

    def test_check_and_score_ties(self) -> None:
        self.assertTrue(check_strict_choice(r"\boxed{C}", "C").is_correct)
        unique = forced_choice_prediction({"A": 0.1, "B": 0.4, "C": -1, "D": 0})
        self.assertEqual(unique.prediction, "B")
        tied = forced_choice_prediction({"A": 1.0, "B": 1.0, "C": 0.0, "D": 0.0})
        self.assertIsNone(tied.prediction)
        self.assertTrue(tied.score_tie)
        self.assertEqual(tied.tied_labels, ("A", "B"))


if __name__ == "__main__":
    unittest.main()
