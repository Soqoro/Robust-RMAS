from __future__ import annotations

import copy
import unittest

from experiments.linkradius.probe_validation import (
    calibrate_probe_configuration,
    probe_autograd_agreement,
    reclassify_probe_pairs,
    select_causally_useful_edges,
    stability_checks,
)
from experiments.linkradius.schemas import ContractError


HEX = "a" * 64


def _probe_rows() -> list[dict]:
    rows: list[dict] = []
    for unit in range(3):
        raw_id = f"raw-{unit}"
        clean = {"B": 1.0 + unit, "C": 4.0 + unit, "D": 6.0 + unit}
        for h in (0.1, 0.2):
            for seed in (101, 202):
                for direction in range(8):
                    t_plus, t_minus = h, -h
                    derivatives = {
                        "B": 1.0 + 0.2 * unit + 0.01 * direction,
                        "C": 0.1 + 0.001 * direction,
                        "D": 0.05 + 0.001 * direction,
                    }
                    margins_plus = {
                        label: clean[label] + value * h
                        for label, value in derivatives.items()
                    }
                    margins_minus = {
                        label: clean[label] - value * h
                        for label, value in derivatives.items()
                    }
                    common = {
                        "record_type": "sample",
                        "intervention_mode": "additive_antithetic",
                        "raw_sample_id": raw_id,
                        "sample_id": f"sample-{unit}",
                        "edge_id": "p2c@0",
                        "direction_id": direction,
                        "probe_seed": seed,
                        "h": h,
                        "q": 32,
                        "subspace_id": "full-tensor",
                        "partition": "validation",
                        "split_manifest_hash": HEX,
                        "execution_manifest_hash": HEX,
                        "source_hash": HEX,
                        "config_hash": HEX,
                        "model_hash": HEX,
                        "adapter_hash": HEX,
                        "prompt_hash": HEX,
                        "scorer_hash": HEX,
                        "subspace_hash": HEX,
                        "analysis_eligible": True,
                    }
                    diagnostics_plus = {
                        "realized_signed_coordinate": t_plus,
                        "requested_realized_cosine": 0.99,
                        "off_direction_relative": 0.01,
                        "collapsed": False,
                    }
                    diagnostics_minus = {
                        "realized_signed_coordinate": t_minus,
                        "requested_realized_cosine": 0.99,
                        "off_direction_relative": 0.01,
                        "collapsed": False,
                    }
                    plus_id = f"plus-{unit}-{h}-{seed}-{direction}"
                    minus_id = f"minus-{unit}-{h}-{seed}-{direction}"
                    rows.extend(
                        (
                            {
                                **common,
                                "run_id": plus_id,
                                "sign": 1,
                                "margins": margins_plus,
                                "diagnostics": diagnostics_plus,
                            },
                            {
                                **common,
                                "run_id": minus_id,
                                "sign": -1,
                                "margins": margins_minus,
                                "diagnostics": diagnostics_minus,
                            },
                            {
                                **common,
                                "record_type": "probe_pair",
                                "intervention_mode": "additive_antithetic_pair",
                                "run_id": f"pair-{unit}-{h}-{seed}-{direction}",
                                "plus_run_id": plus_id,
                                "minus_run_id": minus_id,
                                "clean_margins": clean,
                                "t_plus": t_plus,
                                "t_minus": t_minus,
                                "realized_separation": t_plus - t_minus,
                                "antipodality": 0.99,
                                # Calibration must not trust these flags.
                                "accepted": False,
                                "rejection_reasons": ["stored_flag_is_not_authoritative"],
                                "central_differences": {
                                    label: (margins_plus[label] - margins_minus[label])
                                    / (t_plus - t_minus)
                                    for label in derivatives
                                },
                                "margins_plus": margins_plus,
                                "margins_minus": margins_minus,
                            },
                        )
                    )
    return rows


class ProbeValidationTests(unittest.TestCase):
    def test_validation_thresholds_and_stability_are_recomputed(self) -> None:
        calibration, pairs = calibrate_probe_configuration(
            _probe_rows(), candidate_K=(4, 8), minimum_acceptance=0.5
        )
        self.assertEqual(calibration["selected_h"], 0.1)
        self.assertEqual(calibration["selected_K"], 8)
        self.assertEqual(calibration["acceptance_thresholds"]["minimum_requested_realized_cosine"], 0.99)
        self.assertTrue(all(pair["accepted"] for pair in pairs))
        checks = stability_checks(
            calibration["stability"],
            minimum_rank_correlation=0.9,
            minimum_binding_agreement=0.9,
        )
        self.assertTrue(all(check["passed"] for check in checks), checks)

    def test_pair_numeric_identity_tamper_fails_closed(self) -> None:
        rows = _probe_rows()
        thresholds = calibrate_probe_configuration(rows, candidate_K=(4, 8))[0][
            "acceptance_thresholds"
        ]
        tampered = copy.deepcopy(rows)
        pair = next(row for row in tampered if row["record_type"] == "probe_pair")
        pair["central_differences"]["B"] += 0.25
        with self.assertRaises(ContractError):
            reclassify_probe_pairs(tampered, thresholds)

    def test_causally_useful_rule_is_paired_and_content_hashed(self) -> None:
        rows = []
        for raw_id in ("one", "two"):
            rows.extend(
                (
                    {
                        "record_type": "sample",
                        "analysis_eligible": True,
                        "raw_sample_id": raw_id,
                        "edge_id": "p2c@0",
                        "intervention_mode": "identity",
                        "minimum_margin": 1.0,
                        "scorer_correct": True,
                    },
                    {
                        "record_type": "sample",
                        "analysis_eligible": True,
                        "raw_sample_id": raw_id,
                        "edge_id": "p2c@0",
                        "intervention_mode": "mismatch",
                        "minimum_margin": 0.25,
                        "scorer_correct": False,
                    },
                )
            )
        rule = select_causally_useful_edges(rows, minimum_pairs=2)
        self.assertEqual(rule["useful_edges"], ["p2c@0"])
        self.assertEqual(len(rule["content_hash"]), 64)
        with self.assertRaises(ContractError):
            select_causally_useful_edges(rows[:-1], minimum_pairs=2)

        unavailable = copy.deepcopy(rows)
        for row in unavailable:
            if row["intervention_mode"] == "mismatch":
                row["intervention_unavailable"] = True
        excluded = select_causally_useful_edges(
            unavailable, expected_edges=("p2c@0",), minimum_pairs=2
        )
        self.assertEqual(excluded["edge_summaries"][0]["pair_count"], 0)
        self.assertEqual(excluded["edge_summaries"][0]["unavailable_pair_count"], 2)
        self.assertFalse(excluded["edge_summaries"][0]["useful"])

    def test_ineligible_filler_cannot_rescue_probe_acceptance(self) -> None:
        rows = _probe_rows()
        filler = copy.deepcopy(rows)
        for row in filler:
            row["analysis_eligible"] = False
            row["run_id"] = "filler-" + row["run_id"]
            if "plus_run_id" in row:
                row["plus_run_id"] = "filler-" + row["plus_run_id"]
                row["minus_run_id"] = "filler-" + row["minus_run_id"]
        for row in rows:
            if row["record_type"] == "sample":
                row["diagnostics"]["collapsed"] = True
        classified = reclassify_probe_pairs(
            rows + filler,
            {
                "minimum_requested_realized_cosine": 0.0,
                "maximum_off_direction_relative": 1.0,
                "minimum_signed_separation": 0.0,
                "minimum_antipodality": -1.0,
            },
        )
        self.assertTrue(classified)
        self.assertTrue(all(not row["accepted"] for row in classified))
        self.assertTrue(all(row["analysis_eligible"] for row in classified))

    def test_incomplete_eligible_probe_cube_fails_calibration(self) -> None:
        rows = _probe_rows()
        target = next(
            row
            for row in rows
            if row["record_type"] == "probe_pair" and row["raw_sample_id"] == "raw-0"
        )
        run_ids = {target["run_id"], target["plus_run_id"], target["minus_run_id"]}
        incomplete = [row for row in rows if row["run_id"] not in run_ids]
        with self.assertRaises(ContractError):
            calibrate_probe_configuration(incomplete, candidate_K=(4, 8))

    def test_canonical_probe_cube_detects_configuration_missing_for_every_sample(self) -> None:
        rows = _probe_rows()
        expected = [
            ("p2c@0", h, seed, direction)
            for h in (0.1, 0.2)
            for seed in (101, 202)
            for direction in range(8)
        ]
        expected.append(("c2s@0", 0.1, 101, 0))
        with self.assertRaises(ContractError):
            calibrate_probe_configuration(
                rows,
                candidate_K=(4, 8),
                expected_raw_ids=("raw-0", "raw-1", "raw-2"),
                expected_configurations=expected,
            )

    def test_probe_susceptibility_is_joined_to_matching_autograd_subspace(self) -> None:
        calibration, pairs = calibrate_probe_configuration(
            _probe_rows(), candidate_K=(4, 8)
        )
        selected = [
            row
            for row in pairs
            if row["raw_sample_id"] == "raw-0"
            and row["edge_id"] == "p2c@0"
            and row["h"] == calibration["selected_h"]
            and row["probe_seed"] == 101
            and row["direction_id"] < calibration["selected_K"]
        ]
        q = selected[0]["q"]
        derivatives = [row["central_differences"]["B"] for row in selected]
        susceptibility = (q * sum(value * value for value in derivatives) / len(derivatives)) ** 0.5
        gradient = {
            "record_type": "gradient",
            "raw_sample_id": "raw-0",
            "edge_id": "p2c@0",
            "target_label": "B",
            "q": q,
            "subspace_id": selected[0]["subspace_id"],
            "finite_difference": {
                "autograd_dimensionless_derivative": susceptibility,
                "agrees": True,
                "realized_separation": 0.2,
                "plus_diagnostics": {"collapsed": False},
                "minus_diagnostics": {"collapsed": False},
            },
        }
        agreement = probe_autograd_agreement(
            pairs,
            [gradient],
            selected_h=calibration["selected_h"],
            selected_K=calibration["selected_K"],
        )
        self.assertEqual(agreement["comparison_count"], 2)
        self.assertAlmostEqual(agreement["median_relative_error"], 0.0)
        mismatch = copy.deepcopy(gradient)
        mismatch["q"] += 1
        with self.assertRaises(ContractError):
            probe_autograd_agreement(
                pairs,
                [mismatch],
                selected_h=calibration["selected_h"],
                selected_K=calibration["selected_K"],
            )


if __name__ == "__main__":
    unittest.main()
