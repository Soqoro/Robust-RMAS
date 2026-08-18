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
from .schemas import ContractError


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


def classify_screening_row(row: Mapping[str, Any]) -> tuple[bool, str]:
    gold = str(row.get("gold", row.get("gold_choice", ""))).strip().upper()
    generated = row.get("strict_generated_choice", row.get("generated_choice_strict"))
    generated = None if generated is None else str(generated).strip().upper()
    if gold not in {"A", "B", "C", "D"}:
        return False, "invalid_gold"
    forward_finiteness = row.get("forward_finiteness")
    if (
        isinstance(forward_finiteness, Mapping)
        and forward_finiteness.get("all_relay_interfaces_finite") is False
    ):
        return False, "relay_nonfinite"
    if bool(row.get("answer_conflict", False)):
        return False, "generated_answer_conflict"
    if generated not in {"A", "B", "C", "D"} or bool(row.get("answer_invalid", False)):
        return False, "generated_answer_invalid"
    if generated != gold:
        return False, "generated_incorrect"
    if bool(row.get("score_tie", False)):
        return False, "score_tie"
    if row.get("scorer_numerically_valid") is False or row.get(
        "scorer_nonfinite_fields"
    ):
        return False, "scorer_nonfinite"
    scored = _prediction(row)
    if scored is None:
        return False, "scorer_prediction_invalid"
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


def annotate_screening_rows(
    rows: Sequence[Mapping[str, Any]], *, max_eligible: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    eligible_seen = 0
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
        dual_correct, reason = classify_screening_row(row)
        eligible = dual_correct
        if eligible and max_eligible is not None and eligible_seen >= max_eligible:
            eligible, reason = False, "cohort_limit_filler"
        if eligible:
            eligible_seen += 1
        else:
            reasons[reason] += 1
        row["analysis_eligible"] = eligible
        row["dual_correct"] = dual_correct
        row["exclusion_reason"] = reason
        scored = _prediction(row)
        generated = row.get("strict_generated_choice", row.get("generated_choice_strict"))
        if scored is not None and generated in {"A", "B", "C", "D"}:
            generated_scored_comparable += 1
            generated_scored_agree += int(scored == generated)
        annotated.append(row)
    summary = {
        "schema_version": "linkradius.screening_summary.v1",
        "num_rows": len(annotated),
        "dual_correct_count": eligible_seen,
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
    screening_rows: Sequence[Mapping[str, Any]], fresh_rows: Sequence[Mapping[str, Any]]
) -> None:
    screening = {
        str(row["raw_sample_id"]): bool(
            row.get("dual_correct", classify_screening_row(row)[0])
        )
        for row in screening_rows
    }
    fresh = {str(row["raw_sample_id"]): classify_screening_row(row)[0] for row in fresh_rows}
    if screening.keys() != fresh.keys():
        raise ContractError("fresh clean capture row IDs differ from frozen screening rows")
    changed = sorted(raw_id for raw_id in screening if screening[raw_id] != fresh[raw_id])
    if changed:
        raise ContractError(
            "fresh clean dual-correct status differs from screening under the frozen execution "
            f"manifest for {len(changed)} row(s): {', '.join(changed[:10])}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--max-eligible", type=int)
    parser.add_argument("--min-eligible", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    rows, summary = annotate_screening_rows(load_jsonl(args.input), max_eligible=args.max_eligible)
    if summary["dual_correct_count"] < args.min_eligible:
        raise ContractError(
            f"found {summary['dual_correct_count']} dual-correct rows; need {args.min_eligible}"
        )
    atomic_write_jsonl(args.output, rows, overwrite=args.overwrite)
    atomic_write_json(args.summary, summary, overwrite=args.overwrite)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
