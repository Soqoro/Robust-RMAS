from __future__ import annotations

import math
import unittest

from experiments.linkradius.evaluate_failure_boundary import (
    binary_auprc,
    binary_auroc,
    calibration_bins,
    cluster_bootstrap,
    family_budget_metrics,
    interval_censored_concordance,
    mean_over_probe_seeds,
    site_ranking_metrics,
    threshold_spearman,
)
from experiments.linkradius.schemas import ContractError


class FailureBoundaryMetricTests(unittest.TestCase):
    def test_binary_metrics_are_tie_and_order_invariant(self) -> None:
        labels = [True, False, True, False]
        scores = [1.0, 1.0, 0.0, 0.0]
        permutation = [3, 1, 2, 0]

        auroc = binary_auroc(labels, scores)
        auprc = binary_auprc(labels, scores)
        self.assertEqual(
            auroc,
            binary_auroc(
                [labels[index] for index in permutation],
                [scores[index] for index in permutation],
            ),
        )
        self.assertEqual(
            auprc,
            binary_auprc(
                [labels[index] for index in permutation],
                [scores[index] for index in permutation],
            ),
        )
        self.assertAlmostEqual(auroc, 0.5)
        self.assertAlmostEqual(auprc, 0.5)

    def test_interval_concordance_excludes_overlap_and_credits_prediction_tie(self) -> None:
        rows = [
            {"predicted_radius": 1.0, "threshold_lower": 0.0, "threshold_upper": 1.0},
            {"predicted_radius": 1.0, "threshold_lower": 2.0, "threshold_upper": 3.0},
            {"predicted_radius": 9.0, "threshold_lower": 0.5, "threshold_upper": 2.5},
            {"predicted_radius": 4.0, "threshold_lower": 4.0, "threshold_upper": None},
        ]
        result = interval_censored_concordance(rows)
        # Interval 2 overlaps 0 and 1.  It is disjoint from the final
        # right-censored interval, giving four comparable pairs in total.
        self.assertEqual(result["comparable_pairs"], 4)
        self.assertAlmostEqual(result["concordance"], 2.5 / 4.0)

    def test_adjacent_open_closed_intervals_are_ordered(self) -> None:
        rows = [
            {
                "predicted_radius": 0.1,
                "threshold_lower": 0.0,
                "threshold_upper": 0.1,
            },
            {
                "predicted_radius": 0.2,
                "threshold_lower": 0.1,
                "threshold_upper": 0.2,
            },
            {
                "predicted_radius": 0.3,
                "threshold_lower": 0.2,
                "threshold_upper": None,
            },
        ]
        result = interval_censored_concordance(rows)
        self.assertEqual(result["comparable_pairs"], 3)
        self.assertEqual(result["concordance"], 1.0)

        site_rows = [
            {
                **row,
                "raw_id": "x",
                "attack_family": "pgd",
                "edge_id": edge,
            }
            for row, edge in zip(rows, ("a", "b", "c"))
        ]
        site = site_ranking_metrics(site_rows)
        self.assertEqual(site["comparable_site_pairs"], 3)
        self.assertEqual(site["site_kendall"], 1.0)

    def test_threshold_spearman_excludes_censored_rows(self) -> None:
        result = threshold_spearman(
            [
                {"predicted_radius": 1.0, "threshold_lower": 0.0, "threshold_upper": 1.0},
                {"predicted_radius": 2.0, "threshold_lower": 1.0, "threshold_upper": 3.0},
                {"predicted_radius": 0.0, "threshold_lower": 4.0, "threshold_upper": None},
            ]
        )
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["spearman"], 1.0)

    def test_site_top1_fractional_credit_for_predicted_tie(self) -> None:
        rows = [
            {"raw_id": "x", "attack_family": "pgd", "edge_id": "a", "predicted_radius": 1.0, "threshold_lower": 0.0, "threshold_upper": 1.0},
            {"raw_id": "x", "attack_family": "pgd", "edge_id": "b", "predicted_radius": 1.0, "threshold_lower": 2.0, "threshold_upper": 3.0},
            {"raw_id": "x", "attack_family": "pgd", "edge_id": "c", "predicted_radius": 4.0, "threshold_lower": 4.0, "threshold_upper": None},
        ]
        result = site_ranking_metrics(rows)
        self.assertEqual(result["top1_groups"], 1)
        self.assertEqual(result["top1_accuracy"], 0.5)
        self.assertEqual(result["comparable_site_pairs"], 3)
        self.assertAlmostEqual(result["site_kendall"], 2.5 / 3.0)

    def test_component_score_orientations(self) -> None:
        rows = [
            {"attack_family": "pgd", "requested_epsilon": 0.1, "flipped": True, "edge_radius": 1.0, "clean_margin": 1.0, "susceptibility": 3.0},
            {"attack_family": "pgd", "requested_epsilon": 0.1, "flipped": False, "edge_radius": 2.0, "clean_margin": 2.0, "susceptibility": 1.0},
        ]
        metrics = {
            row["predictor"]: row for row in family_budget_metrics(rows)
        }
        for predictor in ("linkradius", "margin_only", "susceptibility_only"):
            self.assertEqual(metrics[predictor]["auroc"], 1.0)
            self.assertEqual(metrics[predictor]["auprc"], 1.0)

    def test_synthetic_linkradius_beats_both_components(self) -> None:
        labels = [False, False, False, True, True, True]
        radii = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        margins = [3.0, 1.0, 5.0, 4.0, 2.0, 6.0]
        susceptibilities = [3.0, 5.0, 1.0, 2.0, 4.0, 0.0]
        rows = [
            {
                "attack_family": "pgd",
                "requested_epsilon": 0.1,
                "flipped": label,
                "edge_radius": radius,
                "clean_margin": margin,
                "susceptibility": susceptibility,
            }
            for label, radius, margin, susceptibility in zip(
                labels, radii, margins, susceptibilities
            )
        ]
        metrics = {
            row["predictor"]: row["auroc"]
            for row in family_budget_metrics(rows)
        }
        self.assertEqual(metrics["linkradius"], 1.0)
        self.assertGreater(metrics["linkradius"], metrics["margin_only"])
        self.assertGreater(metrics["linkradius"], metrics["susceptibility_only"])

    def test_cluster_bootstrap_is_deterministic_and_clustered(self) -> None:
        rows = [
            {"raw_id": "a", "value": 0.0},
            {"raw_id": "a", "value": 2.0},
            {"raw_id": "b", "value": 10.0},
            {"raw_id": "b", "value": 12.0},
            {"raw_id": "c", "value": 20.0},
            {"raw_id": "c", "value": 22.0},
        ]

        def mean(sample):
            return sum(float(row["value"]) for row in sample) / len(sample)

        first = cluster_bootstrap(rows, mean, repetitions=100, seed=71)
        second = cluster_bootstrap(rows, mean, repetitions=100, seed=71)
        self.assertEqual(first, second)
        self.assertEqual(first["estimate"], 11.0)
        self.assertEqual(first["cluster_count"], 3)
        self.assertLess(first["ci_lower"], first["estimate"])
        self.assertGreater(first["ci_upper"], first["estimate"])

    def test_probe_seed_mean_does_not_pool_repeated_labels(self) -> None:
        rows = [
            {"raw_id": raw_id, "probe_seed": seed, "value": value}
            for seed, values in (
                (101, (0.0, 2.0)),
                (202, (10.0, 14.0)),
                (303, (20.0, 26.0)),
            )
            for raw_id, value in zip(("a", "b"), values)
        ]

        def seed_range(seed_rows):
            values = [float(row["value"]) for row in seed_rows]
            return max(values) - min(values)

        self.assertEqual(
            mean_over_probe_seeds(
                rows,
                seed_range,
                expected_seeds=(101, 202, 303),
            ),
            4.0,
        )
        interval = cluster_bootstrap(
            rows,
            lambda sampled: mean_over_probe_seeds(
                sampled,
                seed_range,
                expected_seeds=(101, 202, 303),
            ),
            repetitions=50,
            seed=9,
        )
        self.assertEqual(interval["cluster_count"], 2)
        self.assertEqual(interval["estimate"], 4.0)

    def test_probe_seed_mean_requires_every_seed_statistic_to_be_finite(self) -> None:
        rows = [
            {"raw_id": "a", "probe_seed": seed, "value": value}
            for seed, value in ((101, 1.0), (202, 2.0), (303, 3.0))
        ]
        result = mean_over_probe_seeds(
            rows,
            lambda seed_rows: (
                math.nan
                if int(seed_rows[0]["probe_seed"]) == 202
                else float(seed_rows[0]["value"])
            ),
            expected_seeds=(101, 202, 303),
        )
        self.assertTrue(math.isnan(result))

    def test_calibration_does_not_split_tied_scores(self) -> None:
        rows = [
            {"failure_score": 0.0, "flipped": False},
            {"failure_score": 1.0, "flipped": False},
            {"failure_score": 1.0, "flipped": True},
            {"failure_score": 2.0, "flipped": True},
        ]
        bins = calibration_bins(rows, num_bins=3)
        tie_bins = [row for row in bins if row["score_min"] <= 1.0 <= row["score_max"]]
        self.assertEqual(len(tie_bins), 1)
        self.assertGreaterEqual(tie_bins[0]["n"], 2)

    def test_nonfinite_inputs_are_rejected(self) -> None:
        with self.assertRaises(ContractError):
            binary_auroc([False, True], [0.0, math.inf])


if __name__ == "__main__":
    unittest.main()
