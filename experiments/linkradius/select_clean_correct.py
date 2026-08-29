#!/usr/bin/env python3
"""Classify screening rows without hiding scorer/generated disagreement."""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io_utils import atomic_write_json, atomic_write_jsonl, content_hash, load_jsonl
from .schemas import (
    CLEAN_CORRECT_POLICIES,
    DEFAULT_CLEAN_CORRECT_POLICY,
    ContractError,
)


CLEAN_STABILITY_POLICIES = ("strict", "empirical")


def _choice(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    return normalized if normalized in {"A", "B", "C", "D"} else None


def _prediction(row: Mapping[str, Any]) -> str | None:
    value = row.get("scorer_prediction")
    if value is None:
        scores = row.get("option_scores")
        if not isinstance(scores, Mapping) or set(scores) != {"A", "B", "C", "D"}:
            return None
        try:
            numeric = {label: float(score) for label, score in scores.items()}
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(score) for score in numeric.values()):
            return None
        maximum = max(numeric.values())
        winners = [label for label, score in numeric.items() if score == maximum]
        return winners[0] if len(winners) == 1 else None
    value = str(value).strip().upper()
    return value if value in {"A", "B", "C", "D"} else None


def _clean_correct_policy(value: str) -> str:
    policy = str(value).strip()
    if policy not in CLEAN_CORRECT_POLICIES:
        raise ContractError(
            "clean-correct policy must be one of "
            f"{CLEAN_CORRECT_POLICIES!r}, got: {value!r}"
        )
    return policy


def classify_forced_margin_row(row: Mapping[str, Any]) -> tuple[bool, str]:
    """Classify the proposal's primary, generation-independent clean endpoint."""

    gold = str(row.get("gold", row.get("gold_choice", ""))).strip().upper()
    if gold not in {"A", "B", "C", "D"}:
        return False, "invalid_gold"
    forward_finiteness = row.get("forward_finiteness")
    if (
        isinstance(forward_finiteness, Mapping)
        and forward_finiteness.get("all_relay_interfaces_finite") is False
    ):
        return False, "relay_nonfinite"
    if row.get("scorer_numerically_valid") is False or row.get(
        "scorer_nonfinite_fields"
    ):
        return False, "scorer_nonfinite"

    scores = row.get("option_scores")
    if not isinstance(scores, Mapping) or set(scores) != {"A", "B", "C", "D"}:
        return False, "option_scores_invalid"
    try:
        numeric_scores = {label: float(value) for label, value in scores.items()}
    except (TypeError, ValueError):
        return False, "option_scores_invalid"
    if not all(math.isfinite(value) for value in numeric_scores.values()):
        return False, "scorer_nonfinite"

    maximum = max(numeric_scores.values())
    winners = [
        label for label, value in numeric_scores.items() if value == maximum
    ]
    if bool(row.get("score_tie", False)) or len(winners) != 1:
        return False, "score_tie"
    scored = _prediction(row)
    if scored is None:
        return False, "scorer_prediction_invalid"
    if scored != winners[0]:
        return False, "scorer_prediction_inconsistent"
    if scored != gold:
        return False, "scorer_incorrect"

    margins = row.get("margins")
    if not isinstance(margins, Mapping) or set(margins) != ({"A", "B", "C", "D"} - {gold}):
        return False, "margins_invalid"
    values = []
    for value in margins.values():
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False, "margins_invalid"
        if not math.isfinite(number):
            return False, "margins_nonfinite"
        values.append(number)
    if min(values) <= 0:
        return False, "nonpositive_clean_margin"
    return True, ""


def classify_dual_correct_row(row: Mapping[str, Any]) -> tuple[bool, str]:
    """Classify the legacy endpoint requiring both generation and scorer correctness."""

    forced_correct, forced_reason = classify_forced_margin_row(row)
    if not forced_correct:
        return False, forced_reason
    gold = str(row.get("gold", row.get("gold_choice", ""))).strip().upper()
    generated = _choice(
        row.get("strict_generated_choice", row.get("generated_choice_strict"))
    )
    if bool(row.get("answer_conflict", False)):
        return False, "generated_answer_conflict"
    if (
        row.get("strict_generated_valid") is False
        or generated is None
        or bool(row.get("answer_invalid", False))
    ):
        return False, "generated_answer_invalid"
    if generated != gold:
        return False, "generated_incorrect"
    return True, ""


def classify_screening_row(
    row: Mapping[str, Any],
    *,
    clean_correct_policy: str = DEFAULT_CLEAN_CORRECT_POLICY,
) -> tuple[bool, str]:
    """Classify a row using the selected clean-correct cohort endpoint."""

    selected = _clean_correct_policy(clean_correct_policy)
    if selected == "forced_margin":
        return classify_forced_margin_row(row)
    return classify_dual_correct_row(row)


def annotate_screening_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_eligible: int | None = None,
    clean_correct_policy: str = DEFAULT_CLEAN_CORRECT_POLICY,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_policy = _clean_correct_policy(clean_correct_policy)
    annotated: list[dict[str, Any]] = []
    eligible_seen = 0
    clean_correct_seen = 0
    forced_margin_correct_seen = 0
    dual_correct_seen = 0
    reasons: collections.Counter[str] = collections.Counter()
    generated_scored_agree = 0
    generated_scored_comparable = 0
    ids: set[str] = set()
    for source in rows:
        row = dict(source)
        raw_id = str(row.get("raw_sample_id", "")).strip()
        if not raw_id or raw_id in ids:
            raise ContractError("screening rows require unique nonempty raw_sample_id values")
        ids.add(raw_id)
        forced_margin_correct, forced_reason = classify_forced_margin_row(row)
        dual_correct, dual_reason = classify_dual_correct_row(row)
        clean_correct, clean_reason = classify_screening_row(
            row, clean_correct_policy=selected_policy
        )
        clean_correct_seen += int(clean_correct)
        forced_margin_correct_seen += int(forced_margin_correct)
        dual_correct_seen += int(dual_correct)
        eligible = clean_correct
        reason = clean_reason
        if eligible and max_eligible is not None and eligible_seen >= max_eligible:
            eligible, reason = False, "cohort_limit_filler"
        if eligible:
            eligible_seen += 1
        else:
            reasons[reason] += 1
        row["clean_correct_policy"] = selected_policy
        row["forced_margin_correct"] = forced_margin_correct
        row["forced_margin_exclusion_reason"] = forced_reason
        row["clean_correct"] = clean_correct
        row["clean_correct_exclusion_reason"] = clean_reason
        row["analysis_eligible"] = eligible
        row["dual_correct"] = dual_correct
        row["dual_correct_exclusion_reason"] = dual_reason
        row["exclusion_reason"] = reason
        scored = _prediction(row)
        generated = row.get("strict_generated_choice", row.get("generated_choice_strict"))
        if scored is not None and generated in {"A", "B", "C", "D"}:
            generated_scored_comparable += 1
            generated_scored_agree += int(scored == generated)
        annotated.append(row)
    summary = {
        "schema_version": "linkradius.screening_summary.v2",
        "num_rows": len(annotated),
        "clean_correct_policy": selected_policy,
        "clean_correct_count": clean_correct_seen,
        "analysis_eligible_count": eligible_seen,
        "forced_margin_correct_count": forced_margin_correct_seen,
        "dual_correct_count": dual_correct_seen,
        "cohort_limit_excluded_count": clean_correct_seen - eligible_seen,
        "exclusion_counts": dict(sorted(reasons.items())),
        "generated_scored_comparable": generated_scored_comparable,
        "generated_scored_agree": generated_scored_agree,
        "generated_scored_agreement": (
            generated_scored_agree / generated_scored_comparable
            if generated_scored_comparable
            else None
        ),
        "ordered_raw_id_hash": content_hash(
            [row["raw_sample_id"] for row in annotated],
            domain="linkradius:screening_order:v1",
        ),
    }
    return annotated, summary


def verify_fresh_clean_eligibility(
    screening_rows: Sequence[Mapping[str, Any]],
    fresh_rows: Sequence[Mapping[str, Any]],
    *,
    clean_correct_policy: str = DEFAULT_CLEAN_CORRECT_POLICY,
) -> None:
    selected_policy = _clean_correct_policy(clean_correct_policy)
    screening = {
        str(row["raw_sample_id"]): bool(
            row.get(
                "clean_correct",
                classify_screening_row(
                    row, clean_correct_policy=selected_policy
                )[0],
            )
        )
        for row in screening_rows
    }
    fresh = {
        str(row["raw_sample_id"]): classify_screening_row(
            row, clean_correct_policy=selected_policy
        )[0]
        for row in fresh_rows
    }
    if screening.keys() != fresh.keys():
        raise ContractError("fresh clean capture row IDs differ from frozen screening rows")
    changed = sorted(raw_id for raw_id in screening if screening[raw_id] != fresh[raw_id])
    if changed:
        raise ContractError(
            f"fresh clean {selected_policy} status differs from screening under "
            "the frozen execution manifest for "
            f"{len(changed)} row(s): {', '.join(changed[:10])}"
        )


def audit_clean_stability(
    fresh_rows: Sequence[Mapping[str, Any]],
    *,
    raw_sample_ids: Sequence[str],
    screening_dual_correct: Sequence[bool | None] | None,
    screening_analysis_eligible: Sequence[bool],
    screening_exclusion_reasons: Sequence[str],
    screening_generated_choices: Sequence[str | None] | None = None,
    screening_scorer_predictions: Sequence[str | None] | None = None,
    screening_clean_correct: Sequence[bool | None] | None = None,
    clean_correct_policy: str | None = None,
    policy: str = "strict",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit a fresh clean capture against its frozen screening outcomes.

    ``strict`` preserves the certification-oriented contract and rejects any
    known selected clean-correct status change.  ``empirical`` treats that
    change as an observed clean-repeat instability: a previously eligible row
    may be demoted, but a frozen-ineligible row is never promoted after
    outcomes are observed.  Calls from pre-policy manifests remain legacy
    dual-correct audits when ``screening_clean_correct`` and
    ``clean_correct_policy`` are omitted.
    """

    if policy not in CLEAN_STABILITY_POLICIES:
        raise ContractError(
            "clean stability policy must be one of "
            f"{CLEAN_STABILITY_POLICIES!r}, got: {policy!r}"
        )
    if clean_correct_policy is None:
        selected_policy = (
            "dual_correct"
            if screening_clean_correct is None
            else DEFAULT_CLEAN_CORRECT_POLICY
        )
    else:
        selected_policy = _clean_correct_policy(clean_correct_policy)
    expected_ids = [str(value) for value in raw_sample_ids]
    dual_before = (
        list(screening_dual_correct)
        if screening_dual_correct is not None
        else [None] * len(expected_ids)
    )
    if screening_clean_correct is None:
        if selected_policy != "dual_correct":
            raise ContractError(
                "screening_clean_correct is required for a forced_margin "
                "clean-stability audit"
            )
        clean_before = list(dual_before)
    else:
        clean_before = list(screening_clean_correct)
    arrays: dict[str, Sequence[Any]] = {
        "screening_dual_correct": dual_before,
        "screening_clean_correct": clean_before,
        "screening_analysis_eligible": screening_analysis_eligible,
        "screening_exclusion_reasons": screening_exclusion_reasons,
    }
    if screening_generated_choices is not None:
        arrays["screening_generated_choices"] = screening_generated_choices
    if screening_scorer_predictions is not None:
        arrays["screening_scorer_predictions"] = screening_scorer_predictions
    for name, values in arrays.items():
        if len(values) != len(expected_ids):
            raise ContractError(
                f"{name} length differs from the clean execution batch"
            )

    annotated, _ = annotate_screening_rows(
        fresh_rows, clean_correct_policy=selected_policy
    )
    observed_ids = [str(row.get("raw_sample_id") or "") for row in annotated]
    if observed_ids != expected_ids:
        raise ContractError(
            "fresh clean capture row IDs or order differ from the frozen execution batch"
        )

    generated_before = (
        list(screening_generated_choices)
        if screening_generated_choices is not None
        else [None] * len(expected_ids)
    )
    scorer_before = (
        list(screening_scorer_predictions)
        if screening_scorer_predictions is not None
        else [None] * len(expected_ids)
    )
    audit_rows: list[dict[str, Any]] = []
    changed_ids: list[str] = []
    dual_changed_ids: list[str] = []
    generated_comparable = 0
    generated_flips = 0
    scorer_comparable = 0
    scorer_flips = 0
    for index, (raw_id, fresh) in enumerate(zip(expected_ids, annotated)):
        expected_clean = clean_before[index]
        if expected_clean is not None and not isinstance(expected_clean, bool):
            raise ContractError(
                "screening_clean_correct entries must be booleans or null"
            )
        expected_dual = dual_before[index]
        if expected_dual is not None and not isinstance(expected_dual, bool):
            raise ContractError(
                "screening_dual_correct entries must be booleans or null"
            )
        frozen_eligible = screening_analysis_eligible[index]
        if not isinstance(frozen_eligible, bool):
            raise ContractError(
                "screening_analysis_eligible entries must be booleans"
            )
        if expected_clean is False and frozen_eligible:
            raise ContractError(
                f"frozen eligible row is not screening clean-correct: {raw_id}"
            )

        fresh_clean = bool(fresh["clean_correct"])
        status_changed = (
            expected_clean is not None and bool(expected_clean) != fresh_clean
        )
        if status_changed:
            changed_ids.append(raw_id)

        fresh_dual = bool(fresh["dual_correct"])
        dual_status_changed = (
            expected_dual is not None and bool(expected_dual) != fresh_dual
        )
        if dual_status_changed:
            dual_changed_ids.append(raw_id)

        screening_generated = _choice(generated_before[index])
        fresh_generated = _choice(fresh.get("strict_generated_choice"))
        generated_comparable_here = (
            screening_generated is not None and fresh_generated is not None
        )
        generated_changed = (
            generated_comparable_here
            and screening_generated != fresh_generated
        )
        generated_comparable += int(generated_comparable_here)
        generated_flips += int(generated_changed)

        screening_scorer = _choice(scorer_before[index])
        fresh_scorer = _prediction(fresh)
        scorer_comparable_here = (
            screening_scorer is not None and fresh_scorer is not None
        )
        scorer_changed = (
            scorer_comparable_here and screening_scorer != fresh_scorer
        )
        scorer_comparable += int(scorer_comparable_here)
        scorer_flips += int(scorer_changed)

        demoted = bool(
            policy == "empirical"
            and expected_clean is not None
            and frozen_eligible
            and not fresh_clean
        )
        effective_eligible = bool(frozen_eligible and not demoted)
        frozen_reason = str(screening_exclusion_reasons[index] or "").strip()
        if frozen_eligible and frozen_reason:
            raise ContractError(
                f"frozen eligible row has an exclusion reason: {raw_id}"
            )
        if not frozen_eligible and not frozen_reason:
            frozen_reason = "not_clean_correct"
        replay_exclusion = (
            "clean_replay_not_dual_correct"
            if selected_policy == "dual_correct"
            else "clean_replay_not_forced_margin_correct"
        )
        effective_reason = replay_exclusion if demoted else frozen_reason
        if effective_eligible:
            effective_reason = ""

        audit_rows.append(
            {
                "schema_version": "linkradius.clean_stability_row.v1",
                "raw_sample_id": raw_id,
                "clean_stability_policy": policy,
                "clean_correct_policy": selected_policy,
                "screening_clean_correct": expected_clean,
                "fresh_clean_correct": fresh_clean,
                "fresh_clean_correct_exclusion_reason": str(
                    fresh.get("clean_correct_exclusion_reason") or ""
                ),
                "screening_dual_correct": expected_dual,
                "screening_analysis_eligible": frozen_eligible,
                "screening_exclusion_reason": frozen_reason,
                "fresh_dual_correct": fresh_dual,
                "fresh_dual_correct_exclusion_reason": str(
                    fresh.get("dual_correct_exclusion_reason") or ""
                ),
                "clean_status_changed": status_changed,
                "clean_correct_status_changed": status_changed,
                "dual_correct_status_changed": dual_status_changed,
                "screening_generated_choice": screening_generated,
                "fresh_generated_choice": fresh_generated,
                "generated_choice_comparable": generated_comparable_here,
                "generated_choice_changed": generated_changed,
                "screening_scorer_prediction": screening_scorer,
                "fresh_scorer_prediction": fresh_scorer,
                "scorer_prediction_comparable": scorer_comparable_here,
                "scorer_prediction_changed": scorer_changed,
                "analysis_demoted": demoted,
                "effective_analysis_eligible": effective_eligible,
                "effective_exclusion_reason": effective_reason,
            }
        )

    if policy == "strict" and changed_ids:
        if selected_policy == "dual_correct":
            endpoint = "dual-correct"
        else:
            endpoint = "forced-margin clean-correct"
        raise ContractError(
            f"fresh clean {endpoint} status differs from screening under the "
            f"frozen execution settings: {', '.join(changed_ids)}"
        )

    known_status_count = sum(value is not None for value in clean_before)
    known_dual_status_count = sum(value is not None for value in dual_before)
    demoted_count = sum(bool(row["analysis_demoted"]) for row in audit_rows)
    summary = {
        "schema_version": "linkradius.clean_stability_summary.v1",
        "policy": policy,
        "clean_correct_policy": selected_policy,
        "row_count": len(audit_rows),
        "known_screening_status_count": known_status_count,
        "clean_correct_status_changed_count": len(changed_ids),
        "clean_correct_status_flip_rate": (
            len(changed_ids) / known_status_count if known_status_count else None
        ),
        "known_screening_dual_correct_count": known_dual_status_count,
        "dual_correct_status_changed_count": len(dual_changed_ids),
        "dual_correct_status_flip_rate": (
            len(dual_changed_ids) / known_dual_status_count
            if known_dual_status_count
            else None
        ),
        "frozen_eligible_count": sum(
            bool(value) for value in screening_analysis_eligible
        ),
        "effective_eligible_count": sum(
            bool(row["effective_analysis_eligible"]) for row in audit_rows
        ),
        "demoted_count": demoted_count,
        "promoted_count": 0,
        "generated_choice_comparable_count": generated_comparable,
        "generated_choice_flip_count": generated_flips,
        "generated_choice_flip_rate": (
            generated_flips / generated_comparable
            if generated_comparable
            else None
        ),
        "scorer_prediction_comparable_count": scorer_comparable,
        "scorer_prediction_flip_count": scorer_flips,
        "scorer_prediction_flip_rate": (
            scorer_flips / scorer_comparable if scorer_comparable else None
        ),
        "changed_raw_sample_ids": changed_ids,
        "dual_correct_changed_raw_sample_ids": dual_changed_ids,
        "demoted_raw_sample_ids": [
            str(row["raw_sample_id"])
            for row in audit_rows
            if bool(row["analysis_demoted"])
        ],
    }
    return audit_rows, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--max-eligible", type=int)
    parser.add_argument("--min-eligible", type=int, default=1)
    parser.add_argument(
        "--clean-correct-policy",
        choices=CLEAN_CORRECT_POLICIES,
        default=DEFAULT_CLEAN_CORRECT_POLICY,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    rows, summary = annotate_screening_rows(
        load_jsonl(args.input),
        max_eligible=args.max_eligible,
        clean_correct_policy=args.clean_correct_policy,
    )
    if summary["analysis_eligible_count"] < args.min_eligible:
        raise ContractError(
            f"found {summary['analysis_eligible_count']} eligible "
            f"{args.clean_correct_policy} rows; need {args.min_eligible}"
        )
    atomic_write_jsonl(args.output, rows, overwrite=args.overwrite)
    atomic_write_json(args.summary, summary, overwrite=args.overwrite)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
