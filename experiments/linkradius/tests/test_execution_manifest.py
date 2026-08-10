from __future__ import annotations

import unittest

from experiments.linkradius.make_execution_manifest import (
    batch_rows,
    build_execution_manifest,
    shard_batch_ids,
    verify_execution_manifest,
)
from experiments.linkradius.make_split_manifest import build_split_manifest
from experiments.linkradius.schemas import ContractError


class ExecutionManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.split = build_split_manifest(
            [{"raw_sample_id": f"id-{index}", "raw_index": index} for index in range(10)]
        )
        self.partition = "attack_train"
        ids = [row["raw_sample_id"] for row in self.split["partitions"][self.partition]]
        self.screening = [
            {
                "raw_sample_id": raw_id,
                "sample_id": f"sample-{raw_id}",
                "analysis_eligible": index % 2 == 0,
                "dual_correct": index % 2 == 0,
                "exclusion_reason": "" if index % 2 == 0 else "generated_incorrect",
            }
            for index, raw_id in enumerate(ids)
        ]

    def test_boundaries_and_shards_preserve_whole_batches(self) -> None:
        manifest = build_execution_manifest(
            split_manifest=self.split,
            partition=self.partition,
            screening_rows=self.screening,
            batch_size=3,
            batches_per_shard=1,
            screening_config_hash="a" * 64,
        )
        verify_execution_manifest(manifest, split_manifest=self.split)
        self.assertEqual(list(batch_rows(manifest, 0)), [0, 1, 2])
        assigned = [batch for index in range(len(manifest["array_shards"])) for batch in shard_batch_ids(manifest, index)]
        self.assertEqual(assigned, list(range(len(manifest["batch_boundaries"]))))

    def test_partial_or_duplicate_batch_assignment_rejected(self) -> None:
        manifest = build_execution_manifest(
            split_manifest=self.split,
            partition=self.partition,
            screening_rows=self.screening,
            batch_size=2,
            screening_config_hash="a" * 64,
        )
        manifest["array_shards"][0]["execution_batch_ids"].append(1)
        with self.assertRaises(ContractError):
            verify_execution_manifest(manifest)


if __name__ == "__main__":
    unittest.main()

