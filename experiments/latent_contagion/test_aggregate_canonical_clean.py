#!/usr/bin/env python3
"""Regression tests for canonical-clean latent-contagion aggregation."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.latent_contagion import aggregate_latent_contagion as aggregate  # noqa: E402


class CanonicalCleanAggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        aggregate.load_required_packages()

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _record(
        sample_id: str,
        correct: bool,
        *,
        eps: float,
        site: str,
        seed: int = 7,
        lc_seed: int | None = None,
        R: int = 2,
        lc_round: int = 0,
        lc_mode: str = "one_shot",
        direction: str = "random",
        steering_method: str = "",
        steering_id: str = "",
        cohort_hash: str = "cohort-v1",
        generation_hash: str = "generation-v1",
        evaluation_hash: str = "evaluation-v1",
        evaluation_protocol: str = "native",
        attack_hash: str = "attack-v1",
    ) -> dict:
        return {
            "sample_id": sample_id,
            "sample_idx": int(sample_id.removeprefix("sample-")),
            "dataset": "math500",
            "style": "sequential",
            "method": "ours_recursive",
            "role_response_regime": "neutral",
            "mas_shape": "chain",
            "recursion_rounds": R,
            "seed": seed,
            "lc_seed": seed if lc_seed is None else lc_seed,
            "lc_mode": lc_mode,
            "lc_site": site,
            "lc_epsilon": eps,
            "lc_round": lc_round,
            "lc_direction": direction,
            "lc_steering_method": steering_method,
            "lc_steering_id": steering_id,
            "raw_final_output": str(correct),
            "final_answer": str(correct),
            "pred_answer_parsed": str(correct),
            "is_correct": correct,
            "correct": correct,
            "answer_invalid": False,
            "provenance_schema_version": 2,
            "sample_cohort_sha256": cohort_hash,
            "sample_ids_sha256": f"ids:{cohort_hash}",
            "questions_sha256": f"questions:{cohort_hash}",
            "ground_truths_sha256": f"golds:{cohort_hash}",
            "generation_config_sha256": generation_hash,
            "evaluation_config_sha256": evaluation_hash,
            "evaluation_protocol": evaluation_protocol,
            "attack_config_sha256": attack_hash,
            "sample_input_sha256": f"input:{cohort_hash}:{sample_id}",
        }

    def _write_run(
        self,
        relative_path: str,
        correctness: list[bool],
        *,
        eps: float,
        site: str,
        seed: int = 7,
        lc_seed: int | None = None,
        R: int = 2,
        lc_round: int = 0,
        lc_mode: str = "one_shot",
        direction: str = "random",
        steering_method: str = "",
        steering_id: str = "",
        cohort_hash: str = "cohort-v1",
        generation_hash: str = "generation-v1",
        evaluation_hash: str = "evaluation-v1",
        evaluation_protocol: str = "native",
        attack_hash: str | None = None,
        include_summary: bool = True,
        declared_total: int | None = None,
    ) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        records = [
            self._record(
                f"sample-{idx}",
                correct,
                eps=eps,
                site=site,
                seed=seed,
                lc_seed=lc_seed,
                R=R,
                lc_round=lc_round,
                lc_mode=lc_mode,
                direction=direction,
                steering_method=steering_method,
                steering_id=steering_id,
                cohort_hash=cohort_hash,
                generation_hash=generation_hash,
                evaluation_hash=evaluation_hash,
                evaluation_protocol=evaluation_protocol,
                attack_hash=(
                    attack_hash
                    or f"attack:{lc_mode}:{site}:{eps}:{lc_round}:{direction}:{steering_id}"
                ),
            )
            for idx, correct in enumerate(correctness)
        ]
        if include_summary:
            records.append(
                {
                    "type": "summary",
                    "provenance_schema_version": 2,
                    "num_samples": len(correctness) if declared_total is None else declared_total,
                    "accuracy": 100.0 * sum(correctness) / len(correctness),
                    "sample_cohort_sha256": cohort_hash,
                    "sample_ids_sha256": f"ids:{cohort_hash}",
                    "questions_sha256": f"questions:{cohort_hash}",
                    "ground_truths_sha256": f"golds:{cohort_hash}",
                    "generation_config_sha256": generation_hash,
                    "evaluation_config_sha256": evaluation_hash,
                    "evaluation_protocol": evaluation_protocol,
                    "attack_config_sha256": (
                        attack_hash
                        or f"attack:{lc_mode}:{site}:{eps}:{lc_round}:{direction}:{steering_id}"
                    ),
                    "attack_config": {
                        "question_suffix_path": "",
                        "prompt_footer_path": "",
                        "latent_contagion": {
                            "mode": lc_mode,
                            "epsilon": eps,
                        },
                        "role_profile_probe": {
                            "mode": "none",
                            "epsilon": 0.0,
                        },
                    },
                    "lc_enabled": bool(lc_mode != "none" and eps > 0.0),
                    "lc_mode": lc_mode,
                    "lc_epsilon": eps,
                    "question_suffix_path": "",
                    "prompt_footer_path": "",
                }
            )
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return path

    def _canonical_inputs(self, reference_path: Path, control_path: Path | None = None):
        warnings: list[str] = []
        reference = aggregate.build_canonical_clean_index(
            [reference_path], "math500", "reference", warnings
        )
        control = (
            aggregate.build_canonical_clean_index(
                [control_path], "math500", "control", warnings
            )
            if control_path is not None
            else {}
        )
        metrics, metrics_by_core = aggregate.compute_fixed_canonical_clean_metrics(
            reference,
            control,
            control_supplied=control_path is not None,
            warnings=warnings,
        )
        return warnings, reference, metrics, metrics_by_core

    @staticmethod
    def _attack_row(per_condition, *, site: str = "p2c", eps: float = 0.1):
        selected = per_condition[
            (per_condition["site"] == site)
            & aggregate.np.isclose(per_condition["eps"].astype(float), eps)
        ]
        if len(selected) != 1:
            raise AssertionError(f"expected one attack row, found {len(selected)}")
        return selected.iloc[0]

    def test_fixed_control_is_invariant_to_in_grid_clean_sites(self) -> None:
        reference_path = self._write_run(
            "reference.jsonl", [True, True], eps=0.0, site="", lc_mode="none"
        )
        control_path = self._write_run(
            "control.jsonl", [True, False], eps=0.0, site="", lc_mode="none"
        )
        attack_path = self._write_run(
            "attack.jsonl",
            [False, True],
            eps=0.1,
            site="p2c",
            lc_round=1,
            direction="bank",
            steering_method="diffmean",
            steering_id="bank-a",
        )
        first_grid_clean = self._write_run(
            "grid-c2s.jsonl", [True, False], eps=0.0, site="c2s"
        )
        extra_grid_clean = self._write_run(
            "grid-s2p.jsonl", [False, True], eps=0.0, site="s2p"
        )
        warnings, reference, metrics, metrics_by_core = self._canonical_inputs(
            reference_path, control_path
        )
        self.assertEqual(len(metrics), 1)
        self.assertAlmostEqual(float(metrics.iloc[0]["clean_asr"]), 0.5)

        observed = []
        for attack_files in (
            [attack_path, first_grid_clean],
            [attack_path, first_grid_clean, extra_grid_clean],
        ):
            frame = aggregate.build_condition_dataframe(attack_files, "math500", warnings)
            frame = aggregate.filter_canonical_attack_dataframe(frame, "math500", warnings)
            self.assertTrue(frame["eps"].apply(aggregate._finite_positive).all())
            per_condition = aggregate.compute_per_condition_metrics_canonical(
                frame, reference, metrics_by_core, warnings
            )
            row = self._attack_row(per_condition)
            observed.append(
                (
                    float(row["clean_accuracy"]),
                    float(row["asrcc"]),
                    float(row["clean_asr"]),
                    float(row["excess_asrcc"]),
                )
            )

        self.assertEqual(observed[0], observed[1])
        self.assertEqual(observed[0], (1.0, 0.5, 0.5, 0.0))
        self.assertTrue(any("only strict positive-epsilon attacks" in warning for warning in warnings))

    def test_reference_self_control_is_zero_and_ignores_perturb_metadata(self) -> None:
        reference_path = self._write_run(
            "reference.jsonl", [True, False], eps=0.0, site="", lc_mode="none"
        )
        attack_path = self._write_run(
            "attack.jsonl",
            [False, False],
            eps=0.2,
            site="s2p",
            lc_round=9,
            lc_mode="persistent",
            direction="bank",
            steering_method="other-method",
            steering_id="other-bank",
        )
        warnings, reference, metrics, metrics_by_core = self._canonical_inputs(reference_path)
        frame = aggregate.build_condition_dataframe([attack_path], "math500", warnings)
        per_condition = aggregate.compute_per_condition_metrics_canonical(
            frame, reference, metrics_by_core, warnings
        )
        row = self._attack_row(per_condition, site="s2p", eps=0.2)

        self.assertEqual(float(metrics.iloc[0]["clean_asr"]), 0.0)
        self.assertEqual(float(row["clean_accuracy"]), 0.5)
        self.assertEqual(float(row["asrcc"]), 1.0)
        self.assertEqual(float(row["clean_asr"]), 0.0)
        self.assertEqual(float(row["excess_asrcc"]), 1.0)

    def test_attack_requires_exact_sample_id_set(self) -> None:
        reference_path = self._write_run(
            "reference.jsonl", [True, True], eps=0.0, site="", lc_mode="none"
        )
        attack_path = self._write_run(
            "attack.jsonl", [False], eps=0.1, site="p2c"
        )
        warnings, reference, _, metrics_by_core = self._canonical_inputs(reference_path)
        frame = aggregate.build_condition_dataframe([attack_path], "math500", warnings)
        per_condition = aggregate.compute_per_condition_metrics_canonical(
            frame, reference, metrics_by_core, warnings
        )
        row = self._attack_row(per_condition)

        self.assertTrue(math.isnan(float(row["asrcc"])))
        self.assertTrue(math.isnan(float(row["excess_asrcc"])))
        self.assertTrue(any("do not exactly match" in warning for warning in warnings))

    def test_attack_provenance_mismatch_leaves_asr_nan(self) -> None:
        reference_path = self._write_run(
            "reference.jsonl", [True, True], eps=0.0, site="", lc_mode="none"
        )
        attack_path = self._write_run(
            "attack.jsonl",
            [False, True],
            eps=0.1,
            site="p2c",
            generation_hash="generation-other",
        )
        warnings, reference, _, metrics_by_core = self._canonical_inputs(reference_path)
        frame = aggregate.build_condition_dataframe([attack_path], "math500", warnings)
        per_condition = aggregate.compute_per_condition_metrics_canonical(
            frame, reference, metrics_by_core, warnings
        )
        row = self._attack_row(per_condition)

        self.assertTrue(math.isnan(float(row["asrcc"])))
        self.assertTrue(any("generation_config_sha256 differs" in warning for warning in warnings))

    def test_partial_attack_provenance_is_rejected(self) -> None:
        reference_path = self._write_run(
            "reference.jsonl", [True, True], eps=0.0, site="", lc_mode="none"
        )
        attack_path = self._write_run(
            "attack.jsonl", [False, True], eps=0.1, site="p2c"
        )
        rows = [json.loads(line) for line in attack_path.read_text(encoding="utf-8").splitlines()]
        rows[0]["generation_config_sha256"] = ""
        attack_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

        warnings, reference, _, metrics_by_core = self._canonical_inputs(reference_path)
        frame = aggregate.build_condition_dataframe([attack_path], "math500", warnings)
        per_condition = aggregate.compute_per_condition_metrics_canonical(
            frame, reference, metrics_by_core, warnings
        )
        row = self._attack_row(per_condition)

        self.assertTrue(math.isnan(float(row["asrcc"])))
        self.assertTrue(any("inconsistent generation_config_sha256" in warning for warning in warnings))

    def test_evaluation_protocol_mismatch_is_rejected(self) -> None:
        reference_path = self._write_run(
            "reference.jsonl", [True, True], eps=0.0, site="", lc_mode="none"
        )
        attack_path = self._write_run(
            "attack.jsonl",
            [False, True],
            eps=0.1,
            site="p2c",
            evaluation_hash="evaluation-strict-v2",
            evaluation_protocol="math500_strict_rejudge",
        )
        warnings, reference, _, metrics_by_core = self._canonical_inputs(reference_path)
        frame = aggregate.build_condition_dataframe([attack_path], "math500", warnings)
        per_condition = aggregate.compute_per_condition_metrics_canonical(
            frame, reference, metrics_by_core, warnings
        )
        row = self._attack_row(per_condition)

        self.assertTrue(math.isnan(float(row["asrcc"])))
        self.assertTrue(any("evaluation_config_sha256 differs" in warning for warning in warnings))

    def test_generation_seed_and_lc_seed_are_distinct_conditions(self) -> None:
        filename_metadata = aggregate.parse_metadata_from_filename(
            Path("site=p2c_eps=0.1_R=2_lc_round=0_seed=7_lc_seed=11.jsonl")
        )
        self.assertEqual(filename_metadata["seed"], 7)
        self.assertEqual(filename_metadata["lc_seed"], 11)

        first = self._write_run(
            "attack-seed-11.jsonl",
            [False, True],
            eps=0.1,
            site="p2c",
            seed=7,
            lc_seed=11,
        )
        second = self._write_run(
            "attack-seed-12.jsonl",
            [True, False],
            eps=0.1,
            site="p2c",
            seed=7,
            lc_seed=12,
        )
        warnings: list[str] = []
        frame = aggregate.build_condition_dataframe([first, second], "math500", warnings)

        self.assertEqual(set(frame["seed"].tolist()), {7})
        self.assertEqual(set(frame["lc_seed"].tolist()), {11, 12})
        self.assertFalse(any("duplicate condition" in warning for warning in warnings))

    def test_epsilon50_uses_excess_asr(self) -> None:
        reference_path = self._write_run(
            "reference.jsonl", [True] * 20, eps=0.0, site="", lc_mode="none"
        )
        control_path = self._write_run(
            "control.jsonl",
            [False] * 4 + [True] * 16,
            eps=0.0,
            site="",
            lc_mode="none",
        )
        low_attack = self._write_run(
            "attack-low.jsonl",
            [False] * 9 + [True] * 11,
            eps=0.01,
            site="p2c",
        )
        high_attack = self._write_run(
            "attack-high.jsonl",
            [False] * 11 + [True] * 9,
            eps=0.1,
            site="p2c",
        )
        warnings, reference, _, metrics_by_core = self._canonical_inputs(
            reference_path, control_path
        )
        frame = aggregate.build_condition_dataframe(
            [low_attack, high_attack], "math500", warnings
        )
        per_condition = aggregate.compute_per_condition_metrics_canonical(
            frame, reference, metrics_by_core, warnings
        )
        epsilon50 = aggregate.compute_epsilon50(per_condition)

        self.assertEqual(len(epsilon50), 1)
        row = epsilon50.iloc[0]
        self.assertEqual(row["epsilon50_metric"], "excess_asrcc")
        self.assertEqual(row["epsilon50_status"], "not_reached")
        self.assertTrue(math.isnan(float(row["epsilon50"])))
        self.assertAlmostEqual(float(row["max_asrcc"]), 0.55)
        self.assertAlmostEqual(float(row["max_excess_asrcc"]), 0.35)

    def test_zero_clean_correct_denominator_is_nan_without_control(self) -> None:
        reference_path = self._write_run(
            "reference.jsonl", [False, False], eps=0.0, site="", lc_mode="none"
        )
        warnings, _, metrics, _ = self._canonical_inputs(reference_path)

        self.assertEqual(len(metrics), 1)
        self.assertTrue(math.isnan(float(metrics.iloc[0]["clean_asr"])))
        self.assertTrue(any("denominator" in warning for warning in warnings))

    def test_main_auto_detects_conventional_clean_roots(self) -> None:
        self._write_run(
            "clean/reference/math500/R2/seed7/result.jsonl",
            [True, True],
            eps=0.0,
            site="",
            lc_mode="none",
        )
        self._write_run(
            "clean/control/math500/R2/seed7/result.jsonl",
            [True, False],
            eps=0.0,
            site="",
            lc_mode="none",
        )
        self._write_run(
            "math500/oneshot/attack.jsonl",
            [False, True],
            eps=0.1,
            site="p2c",
        )
        original_argv = sys.argv
        try:
            sys.argv = [
                "aggregate_latent_contagion.py",
                "--root",
                str(self.root),
                "--dataset",
                "math500",
                "--subdir",
                "oneshot",
                "--make_plots",
                "false",
            ]
            aggregate.main()
        finally:
            sys.argv = original_argv

        metrics_path = (
            self.root
            / "aggregate"
            / f"math500_{self.root.name}_canonical_clean_metrics.csv"
        )
        per_condition_path = (
            self.root
            / "aggregate"
            / f"math500_{self.root.name}_per_condition.csv"
        )
        metrics = aggregate.pd.read_csv(metrics_path)
        per_condition = aggregate.pd.read_csv(per_condition_path)
        self.assertAlmostEqual(float(metrics.iloc[0]["clean_asr"]), 0.5)
        self.assertAlmostEqual(float(per_condition.iloc[0]["clean_asr"]), 0.5)
        self.assertAlmostEqual(float(per_condition.iloc[0]["excess_asrcc"]), 0.0)

    def test_missing_requested_attack_subdir_does_not_scan_stale_root(self) -> None:
        stale = self._write_run(
            "math500/stale-run/attack.jsonl",
            [False, True],
            eps=0.1,
            site="p2c",
        )

        strict = aggregate.find_jsonl_files(
            self.root, "math500", "requested-run", allow_root_fallback=False
        )
        fallback = aggregate.find_jsonl_files(
            self.root, "math500", "requested-run", allow_root_fallback=True
        )

        self.assertEqual(strict, [])
        self.assertIn(stale, fallback)

    def test_control_arrival_never_publishes_temporary_zero_clean_asr(self) -> None:
        self._write_run(
            "clean/reference/math500/R2/seed7/result.jsonl",
            [True, True],
            eps=0.0,
            site="",
            lc_mode="none",
        )
        self._write_run(
            "math500/oneshot/attack.jsonl",
            [False, True],
            eps=0.1,
            site="p2c",
        )

        def run_aggregate():
            original_argv = sys.argv
            try:
                sys.argv = [
                    "aggregate_latent_contagion.py",
                    "--root",
                    str(self.root),
                    "--dataset",
                    "math500",
                    "--subdir",
                    "oneshot",
                    "--make_plots",
                    "false",
                ]
                aggregate.main()
            finally:
                sys.argv = original_argv
            return aggregate.pd.read_csv(
                self.root
                / "aggregate"
                / f"math500_{self.root.name}_per_condition.csv"
            )

        before_control = run_aggregate()
        self.assertTrue(math.isnan(float(before_control.iloc[0]["clean_asr"])))
        self.assertTrue(math.isnan(float(before_control.iloc[0]["excess_asrcc"])))

        self._write_run(
            "clean/control/math500/R2/seed7/result.jsonl",
            [True, False],
            eps=0.0,
            site="",
            lc_mode="none",
        )
        after_control = run_aggregate()
        self.assertAlmostEqual(float(after_control.iloc[0]["clean_asr"]), 0.5)
        self.assertAlmostEqual(float(after_control.iloc[0]["excess_asrcc"]), 0.0)

    def test_incomplete_canonical_file_is_not_indexed(self) -> None:
        missing_summary = self._write_run(
            "missing-summary.jsonl",
            [True, True],
            eps=0.0,
            site="",
            lc_mode="none",
            include_summary=False,
        )
        wrong_count = self._write_run(
            "wrong-count.jsonl",
            [True, True],
            eps=0.0,
            site="",
            lc_mode="none",
            declared_total=3,
        )
        warnings: list[str] = []
        reference = aggregate.build_canonical_clean_index(
            [missing_summary, wrong_count], "math500", "reference", warnings
        )

        self.assertEqual(reference, {})
        self.assertTrue(any("exactly one summary" in warning for warning in warnings))
        self.assertTrue(any("is incomplete" in warning for warning in warnings))

    def test_nominal_clean_with_prompt_attack_is_not_indexed(self) -> None:
        attacked_clean = self._write_run(
            "attacked-clean.jsonl",
            [True, True],
            eps=0.0,
            site="",
            lc_mode="none",
        )
        rows = [
            json.loads(line)
            for line in attacked_clean.read_text(encoding="utf-8").splitlines()
        ]
        rows[-1]["question_suffix_path"] = "suffix.txt"
        rows[-1]["attack_config"]["question_suffix_path"] = "suffix.txt"
        attacked_clean.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        warnings: list[str] = []
        reference = aggregate.build_canonical_clean_index(
            [attacked_clean], "math500", "reference", warnings
        )

        self.assertEqual(reference, {})
        self.assertTrue(any("not attack-free" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
