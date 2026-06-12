#!/usr/bin/env python3
"""Self-tests for the strict Math500 answer checker."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from RecursiveMAS.inference_utils.answer_utils import compare_answers_detailed  # noqa: E402


def assert_correct(gold: str, pred: str) -> None:
    detail = compare_answers_detailed(gold, pred, dataset_name="math500")
    assert detail["correct"], f"expected correct: gold={gold!r} pred={pred!r} detail={detail}"


def assert_wrong(gold: str, pred: str) -> None:
    detail = compare_answers_detailed(gold, pred, dataset_name="math500")
    assert not detail["correct"], f"expected wrong: gold={gold!r} pred={pred!r} detail={detail}"


def assert_invalid(gold: str, pred: str) -> None:
    detail = compare_answers_detailed(gold, pred, dataset_name="math500")
    assert not detail["correct"], f"expected invalid/wrong: gold={gold!r} pred={pred!r} detail={detail}"
    assert detail["answer_invalid"], f"expected invalid flag: gold={gold!r} pred={pred!r} detail={detail}"


def main() -> None:
    correct_cases = [
        (r"\frac{1}{2}", r"\frac{2}{4}"),
        (r"\frac{13}{4}", "3.25"),
        (r"90^\circ", "90"),
        (r"\text{even}", "even"),
        (r"\boxed{5}", "5"),
        (r"2\sqrt{113}", r"2\sqrt{113}"),
    ]
    for gold, pred in correct_cases:
        assert_correct(gold, pred)

    incorrect_cases = [
        (r"\frac{3}{56}", r"\frac{8}{63}"),
        (r"\frac{1}{16}", r"-\frac{1}{16}"),
        (r"2\sqrt{113}", r"2\sqrt{34}"),
        ("2k", "y = ax^2 - 2ahx + ah^2 + k"),
        (r"\frac{243}{625}", r"\frac{81}{200}"),
        ("-1", "1"),
        (r"\begin{pmatrix} 1/5 \\ -18/5 \end{pmatrix}", r"\begin{pmatrix} \frac{1}{5"),
    ]
    for gold, pred in incorrect_cases:
        assert_wrong(gold, pred)

    malformed_cases = [
        ("1", r"\frac{1 + \sqrt{5}"),
        ("1", r"\begin{pmatrix}"),
        ("1", r"\boxed{y = ax^2 - 2ahx + ah^2 + k"),
    ]
    for gold, pred in malformed_cases:
        assert_invalid(gold, pred)

    print("strict Math500 answer checker self-test passed")


if __name__ == "__main__":
    main()
