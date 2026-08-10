#!/usr/bin/env python3
"""Build distinct worst, random, fixed, useful, and observed system curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io_utils import atomic_write_csv
from .schemas import ContractError


def _trapz(points: Sequence[tuple[float, float]], low: float, high: float) -> float:
    if high <= low:
        raise ContractError("AUC epsilon range must have positive width")
    if not points:
        return math.nan
    by_x = {float(x): float(y) for x, y in points if low <= float(x) <= high}
    xs = sorted(set([low, high, *by_x]))
    source = sorted((float(x), float(y)) for x, y in points)

    def step_value(x: float) -> float:
        eligible = [y for px, y in source if px <= x]
        return eligible[-1] if eligible else 0.0

    area = 0.0
    for left, right in zip(xs, xs[1:]):
        area += (right - left) * (step_value(left) + step_value(right)) / 2.0
    return area


def epsilon50(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(points, key=lambda row: float(row["epsilon"]))
    reached = next((row for row in ordered if float(row["vulnerability"]) >= 0.5), None)
    if reached is None:
        return {
            "epsilon50": None,
            "epsilon50_status": "not_reached_right_censored",
            "epsilon50_censoring_limit": float(ordered[-1]["epsilon"]) if ordered else None,
        }
    return {
        "epsilon50": float(reached["epsilon"]),
        "epsilon50_status": "reached",
        "epsilon50_censoring_limit": None,
    }


def build_predicted_system_curves(
    rows: Sequence[Mapping[str, Any]],
    epsilons: Sequence[float],
    *,
    fixed_edge: str | None = None,
    useful_edges: Iterable[str] = (),
    radius_field: str = "edge_radius",
    sample_field: str = "raw_sample_id",
    edge_field: str = "edge_id",
    auc_range: tuple[float, float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_sample: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        sample_id, edge_id = str(row[sample_field]), str(row[edge_field])
        if edge_id in by_sample[sample_id]:
            raise ContractError(f"duplicate predicted radius for {sample_id}/{edge_id}")
        radius = float(row[radius_field])
        if math.isnan(radius) or radius < 0:
            raise ContractError("predicted radii must be non-negative or infinite")
        by_sample[sample_id][edge_id] = radius
    if not by_sample:
        raise ContractError("system curves require at least one predicted radius row")
    epsilon_grid = sorted(set(float(value) for value in epsilons))
    if not epsilon_grid or any(not math.isfinite(value) or value < 0 for value in epsilon_grid):
        raise ContractError("epsilon grid must contain finite non-negative values")
    useful = set(useful_edges)
    if fixed_edge is not None and not any(fixed_edge in values for values in by_sample.values()):
        raise ContractError(f"fixed edge {fixed_edge!r} is absent")

    output: list[dict[str, Any]] = []
    for epsilon in epsilon_grid:
        worst_values: list[float] = []
        random_values: list[float] = []
        fixed_values: list[float] = []
        useful_values: list[float] = []
        for edges in by_sample.values():
            worst_values.append(float(min(edges.values()) <= epsilon))
            random_values.append(sum(radius <= epsilon for radius in edges.values()) / len(edges))
            if fixed_edge is not None and fixed_edge in edges:
                fixed_values.append(float(edges[fixed_edge] <= epsilon))
            useful_radii = [radius for edge, radius in edges.items() if edge in useful]
            if useful_radii:
                useful_values.append(float(min(useful_radii) <= epsilon))
        curve_values = {
            "worst_accessible_site": worst_values,
            "uniform_random_site": random_values,
        }
        if fixed_edge is not None:
            curve_values["fixed_validation_site"] = fixed_values
        if useful:
            curve_values["causally_useful_edge"] = useful_values
        for curve_type, values in curve_values.items():
            if not values:
                raise ContractError(f"curve {curve_type} has no eligible samples")
            output.append(
                {
                    "curve_type": curve_type,
                    "epsilon": epsilon,
                    "vulnerability": sum(values) / len(values),
                    "n": len(values),
                    "fixed_edge": fixed_edge if curve_type == "fixed_validation_site" else None,
                    "num_exposed_edges": (
                        len({edge for edges in by_sample.values() for edge in edges})
                    ),
                }
            )

    summaries: list[dict[str, Any]] = []
    types = sorted({row["curve_type"] for row in output})
    low, high = auc_range or (epsilon_grid[0], epsilon_grid[-1])
    for curve_type in types:
        points = [row for row in output if row["curve_type"] == curve_type]
        summaries.append(
            {
                "curve_type": curve_type,
                **epsilon50(points),
                "auc": _trapz(
                    [(float(row["epsilon"]), float(row["vulnerability"])) for row in points],
                    low,
                    high,
                ),
                "auc_epsilon_min": low,
                "auc_epsilon_max": high,
            }
        )
    return output, summaries


def build_observed_attack_curves(
    rows: Sequence[Mapping[str, Any]],
    epsilons: Sequence[float],
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[str(row["attack_family"])][str(row["raw_sample_id"])].append(row)
    output = []
    for family, samples in sorted(groups.items()):
        for epsilon in sorted(set(float(value) for value in epsilons)):
            outcomes = []
            for sample_rows in samples.values():
                eligible = [
                    row
                    for row in sample_rows
                    if float(row.get("requested_epsilon", row.get("epsilon"))) == epsilon
                ]
                if not eligible:
                    continue
                outcomes.append(any(float(row.get("minimum_margin", math.inf)) <= 0 for row in eligible))
            if outcomes:
                output.append(
                    {
                        "curve_type": "observed_attack_family",
                        "attack_family": family,
                        "epsilon": epsilon,
                        "vulnerability": sum(outcomes) / len(outcomes),
                        "n": len(outcomes),
                    }
                )
    return output


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linkradius-edges", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--epsilons", required=True)
    parser.add_argument("--fixed-edge", default=None)
    parser.add_argument("--useful-edges", default="")
    parser.add_argument("--auc-min", type=float)
    parser.add_argument("--auc-max", type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    epsilons = [float(value) for value in args.epsilons.split()]
    auc_range = None
    if args.auc_min is not None or args.auc_max is not None:
        if args.auc_min is None or args.auc_max is None:
            raise ContractError("--auc-min and --auc-max must be provided together")
        auc_range = (args.auc_min, args.auc_max)
    curves, summaries = build_predicted_system_curves(
        _read_csv(args.linkradius_edges),
        epsilons,
        fixed_edge=args.fixed_edge,
        useful_edges=args.useful_edges.split(),
        auc_range=auc_range,
    )
    atomic_write_csv(args.output, curves, overwrite=args.overwrite)
    atomic_write_csv(args.summary, summaries, overwrite=args.overwrite)
    print(json.dumps({"curve_rows": len(curves), "summary_rows": len(summaries)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

