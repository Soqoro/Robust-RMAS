#!/usr/bin/env python3
"""Derive observed attack crossings on the frozen requested budget grid."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io_utils import atomic_write_csv, load_jsonl
from .schemas import ContractError


def _minimum_margin(row: Mapping[str, Any]) -> float:
    if "minimum_margin" in row:
        return float(row["minimum_margin"])
    margins = row.get("margins")
    if not isinstance(margins, Mapping) or not margins:
        raise ContractError("attack row has no minimum_margin or margins mapping")
    return min(float(value) for value in margins.values())


def first_crossing(
    rows: Sequence[Mapping[str, Any]], *, tie_tolerance: float = 0.0
) -> dict[str, Any]:
    if tie_tolerance < 0 or not math.isfinite(tie_tolerance):
        raise ContractError("tie_tolerance must be finite and non-negative")
    by_budget: dict[float, Mapping[str, Any]] = {}
    for row in rows:
        epsilon = float(row.get("requested_epsilon", row.get("epsilon")))
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ContractError("requested attack budgets must be finite and non-negative")
        if epsilon in by_budget:
            raise ContractError(f"duplicate attack result at requested epsilon {epsilon}")
        by_budget[epsilon] = row
    if not by_budget:
        raise ContractError("cannot derive a threshold from no attack rows")
    ordered = [(epsilon, by_budget[epsilon], _minimum_margin(by_budget[epsilon])) for epsilon in sorted(by_budget)]
    if not all(math.isfinite(margin) for _, _, margin in ordered):
        raise ContractError("attack margins must be finite")
    # The theorem-aligned primary crossing is exact: a non-positive gold
    # margin, including a score tie.  ``tie_tolerance`` is retained only as
    # recorded scorer provenance and must not move this boundary.
    crossing_index = next(
        (idx for idx, (_, _, margin) in enumerate(ordered) if margin <= 0.0), None
    )
    if crossing_index is None:
        threshold = None
        status = "right_censored"
        binding = None
        recovery = False
    else:
        epsilon, crossing_row, _ = ordered[crossing_index]
        threshold = epsilon
        status = "crossed"
        binding = crossing_row.get("binding_competitor")
        recovery = any(margin > 0.0 for _, _, margin in ordered[crossing_index + 1 :])

    generated_flip_epsilon = None
    for epsilon, row, _ in ordered:
        valid = row.get("strict_generated_valid")
        if valid is None:
            valid = not bool(row.get("answer_invalid", False))
        choice = row.get("strict_generated_choice")
        clean_choice = row.get("clean_strict_generated_choice", row.get("gold"))
        if bool(valid) and choice in {"A", "B", "C", "D"} and choice != clean_choice:
            generated_flip_epsilon = epsilon
            break
    realized = None
    if crossing_index is not None:
        crossing_row = ordered[crossing_index][1]
        value = crossing_row.get("realized_epsilon")
        realized = None if value is None else float(value)
    return {
        "first_scored_crossing": threshold,
        "threshold_status": status,
        "right_censoring_limit": ordered[-1][0] if status == "right_censored" else None,
        "first_generated_flip": generated_flip_epsilon,
        "binding_competitor": binding,
        "non_monotonic_recovery": recovery,
        "realized_epsilon_at_crossing": realized,
        "num_requested_budgets": len(ordered),
        "scorer_tie_tolerance": tie_tolerance,
    }


def aggregate_thresholds(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_fields: Sequence[str] = ("raw_sample_id", "edge_id", "attack_family"),
    tie_tolerance: float = 0.0,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field) for field in group_fields)
        if any(value in (None, "") for value in key):
            raise ContractError(f"attack row is missing a grouping field: {group_fields}")
        groups[key].append(row)
    return [
        {
            **dict(zip(group_fields, key)),
            **first_crossing(group, tie_tolerance=tie_tolerance),
        }
        for key, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0]))
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tie-tolerance", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output = aggregate_thresholds(load_jsonl(args.input), tie_tolerance=args.tie_tolerance)
    atomic_write_csv(args.output, output, overwrite=args.overwrite)
    print(json.dumps({"rows": len(output), "path": str(Path(args.output).resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
