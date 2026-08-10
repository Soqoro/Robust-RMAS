from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.linkradius.io_utils import atomic_write_jsonl, atomic_write_text, publish_completion
from experiments.linkradius.merge_shards import validate_and_merge_shards
from experiments.linkradius.schemas import ContractError


class MergeShardTests(unittest.TestCase):
    def test_deterministic_merge_has_one_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            atomic_write_jsonl(first, [
                {"raw_sample_id": "b", "edge_id": "p2c@0"},
                {"record_type": "shard_metadata", "array_index": 0, "config_key": "x", "row_count": 1},
            ])
            atomic_write_jsonl(second, [
                {"raw_sample_id": "a", "edge_id": "c2s@0"},
                {"record_type": "shard_metadata", "array_index": 1, "config_key": "y", "row_count": 1},
            ])
            rows, summary = validate_and_merge_shards(
                [second, first],
                expected_tasks=[{"array_index": 0, "config_key": "x"}, {"array_index": 1, "config_key": "y"}],
                require_completion=False,
            )
            self.assertEqual([row["raw_sample_id"] for row in rows], ["b", "a"])
            self.assertEqual(summary["type"], "summary")
            self.assertEqual(summary["num_shards"], 2)

    def test_duplicate_sample_config_key_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index in range(2):
                path = root / f"{index}.jsonl"
                atomic_write_jsonl(path, [
                    {"raw_sample_id": "same", "edge_id": "p2c@0"},
                    {"record_type": "shard_metadata", "array_index": index, "config_key": "same-config", "row_count": 1},
                ])
                paths.append(path)
            with self.assertRaises(ContractError):
                validate_and_merge_shards(
                    paths,
                    expected_tasks=[{"array_index": 0, "config_key": "same-config"}, {"array_index": 1, "config_key": "same-config"}],
                    require_completion=False,
                )

    def test_shard_must_be_declared_by_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "probe_runs.jsonl"
            atomic_write_jsonl(
                shard,
                [
                    {"raw_sample_id": "a"},
                    {"record_type": "shard_metadata", "array_index": 0, "config_key": "x", "row_count": 1},
                ],
            )
            atomic_write_text(root / "other.txt", "bound\n")
            publish_completion(
                root,
                config_hash="1" * 64,
                source_hash_value="2" * 64,
                artifact_paths=["other.txt"],
            )
            with self.assertRaises(ContractError):
                validate_and_merge_shards(
                    [shard],
                    expected_tasks=[{"array_index": 0, "config_key": "x"}],
                )


if __name__ == "__main__":
    unittest.main()
