from __future__ import annotations

import copy
import hashlib
import json
import unittest

from experiments.linkradius.schemas import ContractError, validate_intervention_row


def _stable_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def valid_row() -> dict[str, object]:
    common = {
        "identity_kind": "linkradius_portable_system_identity_v1",
        "style": "sequential_light",
        "family": "sequential",
        "dataset": "gpqa",
    }
    model_identity = {
        **common,
        "artifacts": {
            "planner": {
                "identity_kind": "huggingface_snapshot",
                "repo_id": "example/planner",
                "revision": "frozen-revision",
            }
        },
    }
    adapter_identity = {
        **common,
        "inner_artifacts": {},
        "outer_artifacts": {},
    }
    scores = {"A": 2.0, "B": 1.0, "C": 0.5, "D": -1.0}
    margins = {label: scores["A"] - score for label, score in scores.items() if label != "A"}
    return {
        "schema_version": "linkradius.v1",
        "record_type": "sample",
        "run_id": "run-1",
        "phase": "pilot",
        "partition": "validation",
        "raw_sample_id": "raw-1",
        "sample_id": "sample-1",
        "raw_index": 0,
        "analysis_eligible": True,
        "dataset": "gpqa",
        "source_split": "train",
        "style": "sequential_light",
        "method": "identity",
        "R": 2,
        "site": "p2c",
        "code_round": 0,
        "paper_round": 1,
        "edge_id": "p2c@0",
        "split_manifest_hash": "1" * 64,
        "execution_manifest_hash": "2" * 64,
        "ordered_cohort_hash": "3" * 64,
        "batch_boundary_hash": "4" * 64,
        "config_hash": "5" * 64,
        "source_hash": "6" * 64,
        "scorer_hash": "7" * 64,
        "subspace_hash": "8" * 64,
        "model_hash": _stable_hash(model_identity),
        "adapter_hash": _stable_hash(adapter_identity),
        "prompt_hash": "9" * 64,
        "system_resolution": {
            **common,
            "model_identity": model_identity,
            "adapter_identity": adapter_identity,
        },
        "strict_generated_choice": None,
        "strict_generated_valid": None,
        "strict_generated_correct": None,
        "gold": "A",
        "scorer_prediction": "A",
        "scorer_correct": True,
        "score_tie": False,
        "option_scores": scores,
        "margins": margins,
        "minimum_margin": min(margins.values()),
        "binding_competitor": "B",
        "intervention_mode": "identity",
        "intervention_family": None,
        "requested_intervention": {},
        "realized_intervention": {"mode": "identity"},
        "runtime": {
            "rounds": 2,
            "style": "sequential_light",
            "dataset": "gpqa",
            "score_tie_atol": 0.0,
            "score_tie_rtol": 0.0,
        },
        "failure": None,
        "warnings": [],
    }


class InterventionSchemaTests(unittest.TestCase):
    def assert_rejected(self, row: dict[str, object]) -> None:
        with self.assertRaises(ContractError):
            validate_intervention_row(row)

    def test_valid_success_row(self) -> None:
        validate_intervention_row(valid_row())

    def test_requires_analysis_system_tie_and_binding_metadata(self) -> None:
        required = (
            "analysis_eligible",
            "system_resolution",
            "score_tie",
            "minimum_margin",
            "binding_competitor",
            "intervention_family",
        )
        for field in required:
            with self.subTest(field=field):
                row = valid_row()
                del row[field]
                self.assert_rejected(row)

    def test_model_adapter_and_prompt_hashes_are_validated(self) -> None:
        mutations = {
            "model_hash": "a" * 64,
            "adapter_hash": "b" * 64,
            "prompt_hash": "C" * 64,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                row = valid_row()
                row[field] = value
                self.assert_rejected(row)

    def test_model_and_adapter_hashes_are_bound_to_system_resolution(self) -> None:
        row = valid_row()
        resolution = row["system_resolution"]
        assert isinstance(resolution, dict)
        model_identity = resolution["model_identity"]
        assert isinstance(model_identity, dict)
        model_identity["dataset"] = "tampered"
        self.assert_rejected(row)

        row = valid_row()
        resolution = row["system_resolution"]
        assert isinstance(resolution, dict)
        resolution["identity_kind"] = "unresolved_injected_system"
        self.assert_rejected(row)

    def test_scorer_claims_follow_scores_and_tie_tolerances(self) -> None:
        for field, value in (
            ("scorer_prediction", "B"),
            ("scorer_correct", False),
            ("score_tie", True),
        ):
            with self.subTest(field=field):
                row = valid_row()
                row[field] = value
                self.assert_rejected(row)

        tied = valid_row()
        scores = tied["option_scores"]
        runtime = tied["runtime"]
        assert isinstance(scores, dict)
        assert isinstance(runtime, dict)
        scores["B"] = 2.0 - 5e-7
        runtime["score_tie_atol"] = 1e-6
        margins = {
            label: float(scores["A"]) - float(score)
            for label, score in scores.items()
            if label != "A"
        }
        tied["margins"] = margins
        tied["minimum_margin"] = min(margins.values())
        tied["binding_competitor"] = "B"
        tied["score_tie"] = True
        tied["scorer_prediction"] = None
        tied["scorer_correct"] = False
        validate_intervention_row(tied)

        forged = copy.deepcopy(tied)
        forged["score_tie"] = False
        forged["scorer_prediction"] = "A"
        forged["scorer_correct"] = True
        self.assert_rejected(forged)

    def test_gold_margins_minimum_and_binding_are_recomputed(self) -> None:
        mutations = (
            ("margins", {"B": 0.9, "C": 1.5, "D": 3.0}),
            ("minimum_margin", 1.5),
            ("binding_competitor", "C"),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                row = valid_row()
                row[field] = value
                self.assert_rejected(row)

    def test_success_metadata_types_and_invariants_fail_closed(self) -> None:
        mutations = (
            ("analysis_eligible", 1),
            ("requested_intervention", []),
            ("realized_intervention", []),
            ("runtime", []),
            ("failure", {"message": "pretend success"}),
            ("warnings", "warning"),
            ("warnings", [""]),
            ("intervention_family", ""),
        )
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                row = valid_row()
                row[field] = value
                self.assert_rejected(row)

    def test_runtime_must_describe_the_claimed_experiment(self) -> None:
        for field, value in (("rounds", 4), ("style", "other"), ("dataset", "other")):
            with self.subTest(field=field):
                row = valid_row()
                runtime = row["runtime"]
                assert isinstance(runtime, dict)
                runtime[field] = value
                self.assert_rejected(row)

        row = valid_row()
        runtime = row["runtime"]
        assert isinstance(runtime, dict)
        runtime["score_tie_atol"] = -1.0
        self.assert_rejected(row)

    def test_strict_generation_fields_are_jointly_consistent(self) -> None:
        generated = valid_row()
        generated["strict_generated_choice"] = "B"
        generated["strict_generated_valid"] = True
        generated["strict_generated_correct"] = False
        validate_intervention_row(generated)

        invalid = valid_row()
        invalid["strict_generated_choice"] = None
        invalid["strict_generated_valid"] = False
        invalid["strict_generated_correct"] = False
        validate_intervention_row(invalid)

        mutations = (
            ("strict_generated_choice", "E"),
            ("strict_generated_valid", True),
            ("strict_generated_correct", True),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                row = valid_row()
                row[field] = value
                self.assert_rejected(row)

        row = copy.deepcopy(generated)
        row["strict_generated_correct"] = True
        self.assert_rejected(row)

    def test_numeric_strings_booleans_and_nonfinite_scores_are_rejected(self) -> None:
        for value in ("2.0", True, float("nan")):
            with self.subTest(value=value):
                row = valid_row()
                scores = row["option_scores"]
                assert isinstance(scores, dict)
                scores["A"] = value
                self.assert_rejected(row)


if __name__ == "__main__":
    unittest.main()
