from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from RecursiveMAS.inference_utils.linkradius import largest_remainder_counts, split_raw_ids
from experiments.linkradius.io_utils import atomic_write_json, load_json
from experiments.linkradius.make_split_manifest import build_split_manifest, verify_split_manifest


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

    def test_serialized_manifest_accepts_canonical_json_key_order(self) -> None:
        manifest = build_split_manifest(
            [
                {"raw_sample_id": f"raw-{index:03d}", "raw_index": index}
                for index in range(10)
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split_manifest.json"
            atomic_write_json(path, manifest)
            reloaded = load_json(path)
        self.assertEqual(
            tuple(reloaded["partitions"]),
            ("attack_train", "test", "validation"),
        )
        self.assertEqual(verify_split_manifest(reloaded), manifest["content_hash"])


if __name__ == "__main__":
    unittest.main()
