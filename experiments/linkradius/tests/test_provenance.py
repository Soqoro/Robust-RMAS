from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.linkradius.build_attack_bank import (
    TRANSFER_MAP_VERSION,
    build_bank,
    transfer_map_hash,
    validate_bank_for_evaluation,
)
from experiments.linkradius.io_utils import source_hash
from experiments.linkradius.schemas import ContractError
from experiments.linkradius.schemas import validate_gate
from experiments.linkradius.validate_stage import make_gate


H = "a" * 64


def bank():
    return build_bank(
        family="universal_margin",
        direction=[3.0, 4.0],
        training_raw_ids=["train-a", "train-b"],
        split_manifest_hash=H,
        execution_manifest_hash="b" * 64,
        edge_id="p2c@0",
        trained_R=2,
        scorer_hash="c" * 64,
        subspace={"name": "full_tensor", "q": 2},
        source_hash="d" * 64,
        hyperparameters={"seed": 42},
    )


class ProvenanceTests(unittest.TestCase):
    def test_gate_content_hash_detects_tampering(self) -> None:
        gate = make_gate(
            gate_type="probe_gate",
            checks=[{"name": "ok", "passed": True}],
            config_hash="1" * 64,
            source_hash="2" * 64,
        )
        validate_gate(gate, gate_type="probe_gate")
        gate["checks"][0]["name"] = "tampered"
        with self.assertRaises(ContractError):
            validate_gate(gate, gate_type="probe_gate")

    def test_attack_training_overlap_rejected(self) -> None:
        with self.assertRaises(ContractError):
            validate_bank_for_evaluation(
                bank(),
                eval_raw_ids=["train-a", "test-x"],
                eval_partition="test",
                split_manifest_hash=H,
                execution_manifest_hash="e" * 64,
                edge_id="p2c@0",
                eval_R=2,
                scorer_hash="c" * 64,
                subspace={"name": "full_tensor", "q": 2},
            )

    def test_cross_horizon_requires_exact_frozen_map(self) -> None:
        kwargs = dict(
            eval_raw_ids=["test-x"],
            eval_partition="test",
            split_manifest_hash=H,
            execution_manifest_hash="e" * 64,
            edge_id="p2c@0",
            eval_R=4,
            scorer_hash="c" * 64,
            subspace={"name": "full_tensor", "q": 2},
        )
        with self.assertRaises(ContractError):
            validate_bank_for_evaluation(bank(), **kwargs)
        transfer = {
            "schema_version": TRANSFER_MAP_VERSION,
            "mappings": [
                {"trained_R": 2, "eval_R": 4, "trained_edge": "p2c@0", "eval_edge": "p2c@0"}
            ],
        }
        transfer["content_hash"] = transfer_map_hash(transfer)
        validate_bank_for_evaluation(bank(), transfer_map=transfer, **kwargs)
        transfer["mappings"][0]["eval_edge"] = "c2s@0"
        with self.assertRaises(ContractError):
            validate_bank_for_evaluation(bank(), transfer_map=transfer, **kwargs)

    def test_source_hash_covers_runtime_and_shell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "RecursiveMAS" / "runtime.py"
            shell = root / "experiments" / "linkradius" / "run.sh"
            runtime.parent.mkdir(parents=True)
            shell.parent.mkdir(parents=True)
            runtime.write_text("one", encoding="utf-8")
            shell.write_text("two", encoding="utf-8")
            first = source_hash(root)
            shell.write_text("changed", encoding="utf-8")
            second = source_hash(root)
            self.assertNotEqual(first, second)
            runtime.write_text("changed too", encoding="utf-8")
            self.assertNotEqual(second, source_hash(root))

    def test_source_hash_ignores_hidden_editor_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "RecursiveMAS" / "runtime.py"
            shell = root / "experiments" / "linkradius" / "run.sh"
            runtime.parent.mkdir(parents=True)
            shell.parent.mkdir(parents=True)
            runtime.write_text("runtime", encoding="utf-8")
            shell.write_text("launcher", encoding="utf-8")
            expected = source_hash(root)

            checkpoint = (
                shell.parent
                / ".ipynb_checkpoints"
                / "run-checkpoint.sh"
            )
            checkpoint.parent.mkdir()
            checkpoint.write_text("first editor copy", encoding="utf-8")
            self.assertEqual(source_hash(root), expected)
            checkpoint.write_text("changed editor copy", encoding="utf-8")
            self.assertEqual(source_hash(root), expected)


if __name__ == "__main__":
    unittest.main()
