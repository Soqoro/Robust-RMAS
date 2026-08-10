from __future__ import annotations

import unittest

from RecursiveMAS.inference_utils.linkradius import largest_remainder_counts, split_raw_ids


class RawSplitTests(unittest.TestCase):
    def test_198_uses_exact_largest_remainder_counts(self) -> None:
        self.assertEqual(largest_remainder_counts(198), (79, 40, 79))
        split = split_raw_ids([f"raw-{index:03d}" for index in range(198)], seed=42)
        self.assertEqual({name: len(rows) for name, rows in split.items()}, {
            "attack_train": 79,
            "validation": 40,
            "test": 79,
        })

    def test_split_is_disjoint_exhaustive_and_input_order_invariant(self) -> None:
        ids = [f"id-{index}" for index in range(37)]
        forward = split_raw_ids(ids, seed=42)
        reverse = split_raw_ids(list(reversed(ids)), seed=42)
        self.assertEqual(forward, reverse)
        sets = [set(values) for values in forward.values()]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])
        self.assertEqual(set().union(*sets), set(ids))

    def test_duplicate_ids_fail_before_splitting(self) -> None:
        with self.assertRaises(ValueError):
            split_raw_ids(["same", "same"])


if __name__ == "__main__":
    unittest.main()

