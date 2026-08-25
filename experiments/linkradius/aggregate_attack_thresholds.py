#!/usr/bin/env python3
"""Derive interval-censored attack thresholds on a frozen budget grid.

The requested grid defines the pre-registered dose-response ordering.  The
main scientific threshold is also derived on the actual post-consumer-cast
norm coordinate; achieved observations are sorted by that norm because a
constrained optimizer may finish inside its requested ball.  A clean execution
is the safe epsilon=0 lower endpoint unless an explicit epsilon=0 attack result
says that the clean margin is non-positive.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io_utils import atomic_write_csv, load_jsonl
from .schemas import ContractError


_DEFAULT_GROUP_FIELDS = ("raw_sample_id", "edge_id", "attack_family")
_OPTIONAL_IDENTITY_FIELDS = (
    "attack_seed",
    "attack_restart",
    "restart",
    "restart_index",
    "target",
    "target_label",
    "attack_target",
    "target_competitor",
    "competitor",
    "competitor_label",
)


def _minimum_margin(row: Mapping[str, Any]) -> float:
    if "minimum_margin" in row:
        return float(row["minimum_margin"])
    if row.get("record_type") == "attack_target":
        for field in ("target_margin", "margin"):
            if field in row:
                return float(row[field])
    margins = row.get("margins")
    if not isinstance(margins, Mapping) or not margins:
        raise ContractError("attack row has no minimum_margin or margins mapping")
    return min(float(value) for value in margins.values())


def _requested_epsilon(row: Mapping[str, Any]) -> float:
    value = row.get("requested_epsilon", row.get("epsilon"))
    if value is None:
        raise ContractError("attack row is missing requested_epsilon")
    epsilon = float(value)
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ContractError("requested attack budgets must be finite and non-negative")
    return epsilon


def _realized_epsilon(row: Mapping[str, Any]) -> float | None:
    value = row.get("realized_epsilon")
    if value is None:
        realized = row.get("realized_intervention")
        if isinstance(realized, Mapping):
            value = realized.get("realized_relative_norm")
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ContractError("realized attack budgets must be finite and non-negative")
    return result


def _validate_result_row(row: Mapping[str, Any]) -> None:
    record_type = row.get("record_type")
    if record_type not in (None, "sample", "attack_target"):
        raise ContractError(f"cannot aggregate attack record_type={record_type!r}")
    for field in ("failure", "unsupported_reason", "error"):
        if row.get(field) not in (None, "", False):
            raise ContractError(f"cannot aggregate attack row with {field}: {row[field]}")
    if row.get("unsupported") is True:
        raise ContractError("cannot aggregate an unsupported attack row")


def _validate_budget_grid(values: Sequence[float]) -> tuple[float, ...]:
    grid = tuple(float(value) for value in values)
    if not grid:
        raise ContractError("the frozen requested budget grid cannot be empty")
    if any(not math.isfinite(value) or value < 0 for value in grid):
        raise ContractError("frozen requested budgets must be finite and non-negative")
    if any(right <= left for left, right in zip(grid, grid[1:])):
        raise ContractError("the frozen requested budget grid must be strictly increasing")
    return grid


def first_crossing(
    rows: Sequence[Mapping[str, Any]],
    *,
    requested_budget_grid: Sequence[float] | None = None,
    tie_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Return the first non-positive-margin crossing and its censoring interval.

    ``requested_budget_grid`` should be the grid recorded by the frozen attack
    gate. It is optional only for compatibility with older callers; when
    omitted, the exact sorted budgets in ``rows`` become the expected grid.
    """

    if tie_tolerance < 0 or not math.isfinite(tie_tolerance):
        raise ContractError("tie_tolerance must be finite and non-negative")
    by_budget: dict[float, Mapping[str, Any]] = {}
    for row in rows:
        _validate_result_row(row)
        epsilon = _requested_epsilon(row)
        if epsilon in by_budget:
            raise ContractError(f"duplicate attack result at requested epsilon {epsilon}")
        by_budget[epsilon] = row
    if not by_budget:
        raise ContractError("cannot derive a threshold from no attack rows")

    observed_grid = tuple(sorted(by_budget))
    frozen_grid = (
        observed_grid
        if requested_budget_grid is None
        else _validate_budget_grid(requested_budget_grid)
    )
    if observed_grid != frozen_grid:
        missing = [value for value in frozen_grid if value not in by_budget]
        unexpected = [value for value in observed_grid if value not in frozen_grid]
        raise ContractError(
            "attack results do not exactly cover the frozen requested budget grid; "
            f"missing={missing}, unexpected={unexpected}"
        )

    ordered = [
        (epsilon, by_budget[epsilon], _minimum_margin(by_budget[epsilon]))
        for epsilon in frozen_grid
    ]
    if not all(math.isfinite(margin) for _, _, margin in ordered):
        raise ContractError("attack margins must be finite")
    realized_by_budget = {
        epsilon: _realized_epsilon(row) for epsilon, row, _ in ordered
    }
    realized_values = [realized_by_budget[epsilon] for epsilon in frozen_grid]
    realized_grid_status = "unavailable"
    realized_grid_strictly_increasing: bool | None = None
    if any(value is not None for value in realized_values):
        if any(value is None for value in realized_values):
            raise ContractError(
                "realized attack budgets must be present for the complete curve"
            )
        complete_realized = [
            float(value) for value in realized_values if value is not None
        ]
        realized_grid_strictly_increasing = not (
            (frozen_grid[0] > 0.0 and complete_realized[0] <= 0.0)
            or any(
                right <= left
                for left, right in zip(
                    complete_realized, complete_realized[1:]
                )
            )
        )
        realized_grid_status = (
            "strictly_increasing"
            if realized_grid_strictly_increasing
            else "nonincreasing_or_collapsed"
        )

    # Ties cross by definition. The scorer tolerance is provenance only and
    # never moves the empirical boundary.
    crossing_index = next(
        (index for index, (_, _, margin) in enumerate(ordered) if margin <= 0.0),
        None,
    )
    first_epsilon = ordered[0][0]
    first_is_zero = first_epsilon == 0.0
    if crossing_index is None:
        crossing_status = "right_censored"
        requested_lower = ordered[-1][0]
        requested_upper = None
        threshold = None
        binding = None
    else:
        epsilon, crossing_row, _ = ordered[crossing_index]
        threshold = epsilon
        binding = crossing_row.get("binding_competitor")
        if crossing_index == 0 and first_is_zero:
            # There is no observed safe point below a failed clean endpoint.
            crossing_status = "left_censored"
            requested_lower = None
        else:
            crossing_status = "interval_crossed"
            requested_lower = (
                0.0 if crossing_index == 0 else ordered[crossing_index - 1][0]
            )
        requested_upper = epsilon

    # The manuscript requires actual post-consumer-cast norms in the main
    # analysis.  A PGD optimizer may finish inside a larger requested ball, so
    # requested-grid order need not equal achieved-norm order.  Derive the
    # realized threshold from the complete set of achieved observations rather
    # than discarding such curves or pretending requested epsilon was inserted.
    realized_interval_available = all(
        value is not None for value in realized_values
    )
    realized_crossing_status = "unavailable"
    realized_lower = None
    realized_upper = None
    realized_threshold = None
    realized_binding = None
    if realized_interval_available:
        realized_observations = sorted(
            (
                float(realized_by_budget[epsilon]),
                epsilon,
                row,
                margin,
            )
            for epsilon, row, margin in ordered
        )
        realized_failures = [
            value for value in realized_observations if value[3] <= 0.0
        ]
        if not realized_failures:
            realized_crossing_status = "right_censored"
            realized_lower = max(value[0] for value in realized_observations)
        else:
            realized_threshold, _, realized_row, _ = realized_failures[0]
            realized_upper = realized_threshold
            realized_binding = realized_row.get("binding_competitor")
            if realized_threshold == 0.0:
                realized_crossing_status = "left_censored"
            else:
                realized_crossing_status = "interval_crossed"
                safe_below = [
                    value[0]
                    for value in realized_observations
                    if value[0] < realized_threshold and value[3] > 0.0
                ]
                # Every analyzed example is freshly clean-correct, so the
                # unperturbed execution supplies the safe epsilon=0 endpoint.
                realized_lower = max([0.0, *safe_below])

    # Count every return from the failure region (margin <= 0) to the safe
    # region (margin > 0), not merely whether one occurred.
    margins = [margin for _, _, margin in ordered]
    reentry_count = sum(
        int(left <= 0.0 and right > 0.0)
        for left, right in zip(margins, margins[1:])
    )

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

    # Keep old scalar/status columns for downstream readers while making the
    # interval-censoring semantics explicit in new columns.
    legacy_status = (
        "right_censored" if crossing_status == "right_censored" else "crossed"
    )
    return {
        "crossing_status": crossing_status,
        "requested_interval_lower": requested_lower,
        "requested_interval_upper": requested_upper,
        "realized_interval_lower": realized_lower,
        "realized_interval_upper": realized_upper,
        "realized_interval_available": realized_interval_available,
        "realized_crossing_status": realized_crossing_status,
        "realized_first_scored_crossing": realized_threshold,
        "realized_binding_competitor": realized_binding,
        "realized_grid_status": realized_grid_status,
        "realized_grid_strictly_increasing": realized_grid_strictly_increasing,
        "first_scored_crossing": threshold,
        "threshold_status": legacy_status,
        "right_censoring_limit": (
            requested_lower if crossing_status == "right_censored" else None
        ),
        "first_generated_flip": generated_flip_epsilon,
        "binding_competitor": binding,
        "reentry_count": reentry_count,
        "nonmonotonic": reentry_count > 0,
        "non_monotonic_recovery": reentry_count > 0,
        "realized_epsilon_at_requested_crossing": (
            realized_by_budget[threshold] if threshold is not None else None
        ),
        "realized_epsilon_at_crossing": (
            realized_threshold
        ),
        "num_requested_budgets": len(ordered),
        "scorer_tie_tolerance": tie_tolerance,
        "implicit_clean_epsilon_zero": 0.0 not in frozen_grid,
    }


def _identity_value(row: Mapping[str, Any], field: str) -> Any:
    value = row.get(field)
    if field == "attack_seed" and value in (None, ""):
        realized = row.get("realized_intervention")
        if isinstance(realized, Mapping):
            value = realized.get("attack_seed")
    return value


def aggregate_thresholds(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_fields: Sequence[str] = _DEFAULT_GROUP_FIELDS,
    requested_budget_grid: Sequence[float] | None = None,
    tie_tolerance: float = 0.0,
) -> list[dict[str, Any]]:
    """Aggregate independent attack curves without merging seed/target runs."""

    if not rows:
        raise ContractError("cannot aggregate thresholds from no attack rows")
    for row in rows:
        _validate_result_row(row)

    # Edge summaries and per-competitor PGD targets are distinct curves at the
    # same budgets. Partitioning before identity discovery prevents the target
    # records from looking like duplicate summary budgets.
    partitions: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        curve_kind = "attack_target" if row.get("record_type") == "attack_target" else "edge_summary"
        partitions[curve_kind].append(row)

    outputs: list[dict[str, Any]] = []
    inferred_expected_grid: tuple[float, ...] | None = (
        None
        if requested_budget_grid is None
        else _validate_budget_grid(requested_budget_grid)
    )
    for curve_kind in sorted(partitions):
        partition_rows = partitions[curve_kind]
        if curve_kind == "attack_target" and not any(
            any(_identity_value(row, field) not in (None, "") for row in partition_rows)
            for field in ("target", "target_label", "attack_target", "target_competitor", "competitor", "competitor_label")
        ):
            raise ContractError("attack_target row is missing target/competitor identity")

        # Discover optional identities within each required base curve. This
        # permits, for example, seeded random attacks and unseeded PGD attacks
        # to coexist without making attack_seed spuriously mandatory for PGD.
        base_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for row in partition_rows:
            base_key = tuple(_identity_value(row, field) for field in group_fields)
            if any(value in (None, "") for value in base_key):
                raise ContractError(
                    "attack row is missing a required curve identity field: "
                    f"{tuple(group_fields)}"
                )
            base_groups[base_key].append(row)

        for base_key, base_rows in sorted(
            base_groups.items(),
            key=lambda item: tuple(str(value) for value in item[0]),
        ):
            optional_fields = [
                field
                for field in _OPTIONAL_IDENTITY_FIELDS
                if field not in group_fields
                and any(
                    _identity_value(row, field) not in (None, "")
                    for row in base_rows
                )
            ]
            identity_fields = [*group_fields, *optional_fields]
            groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
            for row in base_rows:
                key = tuple(_identity_value(row, field) for field in identity_fields)
                if any(value in (None, "") for value in key):
                    raise ContractError(
                        "attack row is missing a curve identity field: "
                        f"{tuple(identity_fields)}"
                    )
                groups[key].append(row)

            sorted_groups = sorted(
                groups.items(),
                key=lambda item: tuple(str(value) for value in item[0]),
            )
            if inferred_expected_grid is None:
                inferred_expected_grid = _validate_budget_grid(
                    tuple(
                        sorted(
                            _requested_epsilon(row)
                            for row in sorted_groups[0][1]
                        )
                    )
                )

            outputs.extend(
                {
                    "curve_kind": curve_kind,
                    **dict(zip(identity_fields, key)),
                    **first_crossing(
                        group,
                        requested_budget_grid=inferred_expected_grid,
                        tie_tolerance=tie_tolerance,
                    ),
                }
                for key, group in sorted_groups
            )

    return sorted(
        outputs,
        key=lambda row: tuple(
            str(row.get(field, ""))
            for field in ("raw_sample_id", "edge_id", "attack_family", "curve_kind", "target_label", "competitor", "attack_seed", "restart_index")
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--requested-budget-grid",
        nargs="+",
        type=float,
        help="exact increasing grid frozen before test execution",
    )
    parser.add_argument("--tie-tolerance", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    input_rows = load_jsonl(args.input)
    # Shard metadata is not an attack result. Unsupported and failed result
    # records are deliberately retained so aggregate_thresholds rejects them.
    rows = [row for row in input_rows if row.get("record_type") != "shard_metadata"]
    output = aggregate_thresholds(
        rows,
        requested_budget_grid=args.requested_budget_grid,
        tie_tolerance=args.tie_tolerance,
    )
    atomic_write_csv(
        args.output,
        output,
        fieldnames=sorted({str(key) for row in output for key in row}),
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {"rows": len(output), "path": str(Path(args.output).resolve())},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
