from __future__ import annotations

import hashlib
import json
import unittest
import unicodedata
from unittest import mock

from RecursiveMAS.inference_utils.linkradius import (
    GPQA_OPTION_PERMUTATION_VERSION,
    build_gpqa_raw_record,
    gpqa_option_permutation,
    stable_gpqa_raw_id,
)

try:
    from RecursiveMAS.inference_utils.inference_mas import (
        _build_gpqa_question_and_choice as legacy_build_gpqa,
        build_sample_id as legacy_build_sample_id,
        load_eval_questions_and_answers,
    )
except (ImportError, ModuleNotFoundError):  # lightweight CPU Python has no ML stack
    legacy_build_gpqa = None
    legacy_build_sample_id = None
    load_eval_questions_and_answers = None


def fixture() -> dict[str, str]:
    return {
        "Question": "  Which particle?\r\nExplain.  ",
        "Correct Answer": "Muon",
        "Incorrect Answer 1": "Electron",
        "Incorrect Answer 2": "Proton",
        "Incorrect Answer 3": "Photon",
    }


class GPQAIdentityTests(unittest.TestCase):
    def test_fallback_id_exact_domain_separated_canonical_json(self) -> None:
        row = fixture()
        canonical = {
            "question": "Which particle?\nExplain.",
            "correct_answer": "Muon",
            "incorrect_answer_1": "Electron",
            "incorrect_answer_2": "Proton",
            "incorrect_answer_3": "Photon",
        }
        encoded = json.dumps(
            canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        expected = hashlib.sha256(b"linkradius:gpqa_raw_id:v1\0" + encoded).hexdigest()
        self.assertEqual(stable_gpqa_raw_id(row), expected)

    def test_id_is_independent_of_presentation_seed_and_batching(self) -> None:
        first = build_gpqa_raw_record(fixture(), raw_index=9, seed=42)
        second = build_gpqa_raw_record(fixture(), raw_index=0, seed=987)
        self.assertEqual(first.raw_sample_id, second.raw_sample_id)
        self.assertEqual(first.raw_id_algorithm, "linkradius_gpqa_raw_id_v1")
        self.assertEqual(first.option_permutation_algorithm, GPQA_OPTION_PERMUTATION_VERSION)
        self.assertNotEqual(first.raw_index, second.raw_index)

    def test_native_id_and_nfkc_normalization(self) -> None:
        row = fixture()
        row["id"] = "  id-\uff21  "
        self.assertEqual(stable_gpqa_raw_id(row), "id-A")

    def test_release_permutation_and_rendering_are_exact(self) -> None:
        row = fixture()
        record = build_gpqa_raw_record(row, raw_index=0, seed=42)
        expected_perm = gpqa_option_permutation(row["Question"].strip(), seed=42)
        self.assertEqual(record.option_permutation, expected_perm)
        options = ["Muon", "Electron", "Proton", "Photon"]
        lines = [f"{label}. {options[index]}" for label, index in zip("ABCD", expected_perm)]
        expected_question = row["Question"].strip() + "\n" + "\n".join(lines)
        expected_question += "\n\nChoose the correct option (A/B/C/D)."
        self.assertEqual(record.rendered_question, expected_question)
        self.assertEqual(record.gold_label, "ABCD"[expected_perm.index(0)])
        self.assertEqual(dict(record)["raw_sample_id"], record.raw_sample_id)

    @unittest.skipIf(legacy_build_gpqa is None, "legacy inference dependencies are unavailable")
    def test_legacy_rendering_and_positional_id_remain_byte_identical(self) -> None:
        row = fixture()
        expected_question, expected_gold = legacy_build_gpqa(row, seed=42, shuffle_options=True)
        linkradius_record = build_gpqa_raw_record(row, raw_index=17, seed=42)
        self.assertEqual(linkradius_record.rendered_question.encode("utf-8"), expected_question.encode("utf-8"))
        self.assertEqual(linkradius_record.gold_label, expected_gold)
        # The opt-in LinkRadius raw ID must not alter the legacy positional path.
        self.assertEqual(legacy_build_sample_id("gpqa_diamond", "train", 17, None), "gpqa_diamond:train:17")

    def test_wrong_answer_numbering_is_identity_significant(self) -> None:
        row = fixture()
        swapped = fixture()
        swapped["Incorrect Answer 1"], swapped["Incorrect Answer 2"] = (
            swapped["Incorrect Answer 2"],
            swapped["Incorrect Answer 1"],
        )
        self.assertNotEqual(stable_gpqa_raw_id(row), stable_gpqa_raw_id(swapped))

    @unittest.skipIf(
        load_eval_questions_and_answers is None,
        "legacy inference dependencies are unavailable",
    )
    def test_optional_raw_index_selector_preserves_positional_legacy_id(self) -> None:
        class FakeDataset(list):
            def select(self, indices):
                return FakeDataset([self[index] for index in indices])

            def shuffle(self, seed):  # pragma: no cover - selector rejects shuffle.
                raise AssertionError(f"unexpected shuffle with seed {seed}")

        rows = [fixture(), {**fixture(), "Question": "second"}]
        with mock.patch(
            "RecursiveMAS.inference_utils.inference_mas.load_dataset",
            return_value=FakeDataset(rows),
        ):
            dataset, questions, gold, metadata = load_eval_questions_and_answers(
                "gpqa",
                "train",
                num_samples=1,
                shuffle=False,
                seed=42,
                return_metadata=True,
                sample_indices=[1],
            )
        self.assertEqual(dataset, "gpqa_diamond")
        self.assertEqual(len(questions), 1)
        self.assertEqual(len(gold), 1)
        self.assertEqual(metadata, [{"id": 1}])
        self.assertEqual(
            legacy_build_sample_id(dataset, "train", 0, metadata),
            "gpqa_diamond:train:1",
        )

    def test_empty_required_field_rejected(self) -> None:
        row = fixture()
        row["Correct Answer"] = "  "
        with self.assertRaises(ValueError):
            stable_gpqa_raw_id(row)


if __name__ == "__main__":
    unittest.main()
