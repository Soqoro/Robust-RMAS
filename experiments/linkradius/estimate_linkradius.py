#!/usr/bin/env python3
"""Estimate competitor-specific susceptibilities and LinkRadius values."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io_utils import atomic_write_csv
from .schemas import ContractError


def estimate_competitor_radii(
    clean_margins: Mapping[str, float],
    derivatives: Mapping[str, Sequence[float]],
    *,
    q: int,
    requested_K: int | None = None,
    accepted_direction_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Apply ``chi^2=q/K sum d^2`` without coercing invalid clean margins."""

    if isinstance(q, bool) or int(q) <= 0:
        raise ContractError("q must be a positive integer")
    competitors = tuple(sorted(clean_margins))
    if not competitors or set(derivatives) != set(competitors):
        raise ContractError("margins and derivatives must contain the same competitors")
    lengths = {len(derivatives[label]) for label in competitors}
    if len(lengths) != 1:
        raise ContractError("all competitors must use the same accepted direction pairs")
    K_eff = lengths.pop()
    if K_eff <= 0:
        raise ContractError("at least one accepted antithetic pair is required")
    if requested_K is None:
        requested_K = K_eff
    if requested_K <= 0 or K_eff > requested_K:
        raise ContractError("requested_K must be positive and at least K_eff")
    ids = tuple(range(K_eff)) if accepted_direction_ids is None else tuple(accepted_direction_ids)
    if len(ids) != K_eff or len(set(ids)) != K_eff:
        raise ContractError("accepted direction IDs must be unique and match K_eff")
    primary_available = K_eff == requested_K and ids == tuple(range(requested_K))

    rows: list[dict[str, Any]] = []
    for competitor in competitors:
        margin = float(clean_margins[competitor])
        if not math.isfinite(margin):
            raise ContractError(f"clean margin for {competitor} is non-finite")
        if margin <= 0:
            raise ContractError(f"clean margin for {competitor} must be positive")
        values = [float(value) for value in derivatives[competitor]]
        if not all(math.isfinite(value) for value in values):
            raise ContractError(f"derivatives for {competitor} contain non-finite values")
        susceptibility = math.sqrt(float(q) * sum(value * value for value in values) / K_eff)
        radius = math.inf if susceptibility == 0.0 else margin / susceptibility
        rows.append(
            {
                "competitor": competitor,
                "clean_margin": margin,
                "susceptibility": susceptibility,
                "radius": radius,
                "q": int(q),
                "requested_K": int(requested_K),
                "K_eff": K_eff,
                "primary_available": primary_available,
                "estimate_kind": "primary" if primary_available else "incomplete_sensitivity",
            }
        )
    binding = min(rows, key=lambda row: (row["radius"], row["competitor"]))
    return {
        "competitors": rows,
        "edge_radius": binding["radius"],
        "binding_competitor": binding["competitor"],
        "primary_available": primary_available,
        "K_eff": K_eff,
        "requested_K": int(requested_K),
        "accepted_direction_ids": list(ids),
    }


def central_difference(
    margin_plus: float, margin_minus: float, t_plus: float, t_minus: float
) -> float:
    numerator = float(margin_plus) - float(margin_minus)
    denominator = float(t_plus) - float(t_minus)
    if not all(math.isfinite(value) for value in (numerator, denominator)):
        raise ContractError("finite-difference values must be finite")
    if denominator <= 0.0:
        raise ContractError("realized signed separation must be positive")
    return numerator / denominator


def estimate_from_pair_rows(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    clean_margins: Mapping[str, float],
    q: int,
    requested_K: int,
) -> dict[str, Any]:
    by_direction: dict[int, Mapping[str, Any]] = {}
    for row in pair_rows:
        direction_id = int(row["direction_id"])
        if direction_id in by_direction:
            raise ContractError(f"duplicate antithetic pair for direction {direction_id}")
        by_direction[direction_id] = row
    accepted_ids = sorted(
        direction_id
        for direction_id, row in by_direction.items()
        if direction_id < requested_K and bool(row.get("accepted", False))
    )
    derivatives: dict[str, list[float]] = {label: [] for label in clean_margins}
    for direction_id in accepted_ids:
        row = by_direction[direction_id]
        if isinstance(row.get("derivatives"), Mapping):
            for label in clean_margins:
                derivatives[label].append(float(row["derivatives"][label]))
        else:
            for label in clean_margins:
                derivatives[label].append(
                    central_difference(
                        row["margins_plus"][label],
                        row["margins_minus"][label],
                        row["t_plus"],
                        row["t_minus"],
                    )
                )
    if not accepted_ids:
        raise ContractError("no accepted probe pairs in requested prefix")
    return estimate_competitor_radii(
        clean_margins,
        derivatives,
        q=q,
        requested_K=requested_K,
        accepted_direction_ids=accepted_ids,
    )


def direction_bootstrap_interval(
    clean_margin: float,
    derivatives: Sequence[float],
    *,
    q: int,
    draws: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not derivatives or draws <= 0 or not 0 < alpha < 1:
        raise ContractError("invalid direction-bootstrap arguments")
    rng = random.Random(seed)
    values: list[float] = []
    source = [float(value) for value in derivatives]
    for _ in range(draws):
        sample = [source[rng.randrange(len(source))] for _ in source]
        chi = math.sqrt(float(q) * sum(value * value for value in sample) / len(sample))
        values.append(math.inf if chi == 0 else float(clean_margin) / chi)
    values.sort()
    low_idx = max(0, min(draws - 1, int(math.floor((alpha / 2) * draws))))
    high_idx = max(0, min(draws - 1, int(math.ceil((1 - alpha / 2) * draws)) - 1))
    return values[low_idx], values[high_idx]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="CSV with accepted, derivative_<label> fields")
    parser.add_argument("--output-competitors", required=True)
    parser.add_argument("--output-edges", required=True)
    parser.add_argument("--K", type=int, action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    requested_values = sorted(set(args.K or [4, 8, 16, 32]))
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    group_fields = ("raw_sample_id", "sample_id", "edge_id", "probe_seed", "h", "subspace_id")
    for row in _read_csv(args.pairs):
        groups[tuple(row.get(field, "") for field in group_fields)].append(row)
    competitor_output: list[dict[str, Any]] = []
    edge_output: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        gold = rows[0]["gold"]
        competitors = tuple(label for label in "ABCD" if label != gold)
        margins = {label: float(rows[0][f"clean_margin_{label}"]) for label in competitors}
        normalized = []
        for row in rows:
            normalized.append(
                {
                    "direction_id": int(row["direction_id"]),
                    "accepted": row.get("accepted", "").lower() in {"1", "true", "yes"},
                    "derivatives": {label: float(row[f"derivative_{label}"]) for label in competitors},
                }
            )
        q = int(rows[0]["q"])
        for requested_K in requested_values:
            try:
                estimate = estimate_from_pair_rows(
                    normalized, clean_margins=margins, q=q, requested_K=requested_K
                )
            except ContractError:
                continue
            common = dict(zip(group_fields, key))
            for result in estimate["competitors"]:
                competitor_output.append({**common, **result})
            edge_output.append(
                {
                    **common,
                    "requested_K": requested_K,
                    "K_eff": estimate["K_eff"],
                    "primary_available": estimate["primary_available"],
                    "edge_radius": estimate["edge_radius"],
                    "binding_competitor": estimate["binding_competitor"],
                }
            )
    atomic_write_csv(args.output_competitors, competitor_output, overwrite=args.overwrite)
    atomic_write_csv(args.output_edges, edge_output, overwrite=args.overwrite)
    print(json.dumps({"competitor_rows": len(competitor_output), "edge_rows": len(edge_output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

