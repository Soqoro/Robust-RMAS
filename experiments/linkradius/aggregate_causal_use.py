#!/usr/bin/env python3
"""Aggregate paired causal-use controls with example-level bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .io_utils import atomic_write_csv, load_jsonl
from .schemas import ContractError


PROVENANCE_FIELDS = (
    "split_manifest_hash",
    "execution_manifest_hash",
    "ordered_cohort_hash",
    "batch_boundary_hash",
)


def common_provenance(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Return one complete provenance identity, rejecting mixed/partial inputs."""

    relevant = [row for row in rows if row.get("record_type", "sample") != "shard_metadata"]
    if not relevant:
        return {}
    if not any(any(row.get(field) is not None for field in PROVENANCE_FIELDS) for row in relevant):
        # Preserve the small pure-function fixtures while real experiment rows
        # are required by their schema/runner to carry all four fields.
        return {}
    result: dict[str, str] = {}
    for field in PROVENANCE_FIELDS:
        values = {str(row.get(field) or "") for row in relevant}
        if len(values) != 1 or "" in values:
            raise ContractError(f"causal aggregation has mixed or missing {field}")
        result[field] = values.pop()
    return result


def _minimum_margin(row: Mapping[str, Any]) -> float:
    if row.get("minimum_margin") is not None:
        return float(row["minimum_margin"])
    margins = row.get("margins")
    if not isinstance(margins, Mapping) or not margins:
        raise ContractError("causal row has no gold-margin information")
    return min(float(value) for value in margins.values())


def _correct(row: Mapping[str, Any]) -> float:
    if "scorer_correct" in row:
        return float(bool(row["scorer_correct"]))
    return float(row.get("scorer_prediction") == row.get("gold"))


def eligible_complete_causal_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_edges: Sequence[str],
    expected_modes: Sequence[str],
    expected_raw_ids: Sequence[str] | None = None,
) -> list[Mapping[str, Any]]:
    """Select the analysis cohort and require its exact paired control cube."""

    selected = [
        row
        for row in rows
        if row.get("record_type", "sample") == "sample"
        and bool(row.get("analysis_eligible", False))
    ]
    if not selected:
        raise ContractError("causal aggregation has no analysis-eligible rows")
    common_provenance(selected)
    edges, modes = set(expected_edges), set(expected_modes)
    if {str(row.get("edge_id")) for row in selected} != edges:
        raise ContractError("eligible causal rows do not cover the exact expected edges")
    if {str(row.get("intervention_mode")) for row in selected} != modes:
        raise ContractError("eligible causal rows do not cover the exact expected controls")
    by_sample: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in selected:
        raw_id = str(row.get("raw_sample_id") or "")
        key = (str(row.get("edge_id")), str(row.get("intervention_mode")))
        if not raw_id or key in by_sample[raw_id]:
            raise ContractError("eligible causal cube contains a duplicate or invalid row")
        by_sample[raw_id].add(key)
    if expected_raw_ids is not None:
        required_ids = {str(value) for value in expected_raw_ids}
        observed_ids = set(by_sample)
        if not required_ids or observed_ids != required_ids:
            missing = sorted(required_ids - observed_ids)
            extra = sorted(observed_ids - required_ids)
            raise ContractError(
                "eligible causal sample coverage differs from the frozen execution; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
    expected = {(edge, mode) for edge in edges for mode in modes}
    incomplete = [raw_id for raw_id, values in by_sample.items() if values != expected]
    if incomplete:
        raise ContractError(f"eligible causal rows have incomplete paired controls: {incomplete[:5]}")
    return selected


def paired_bootstrap_interval(
    differences: Sequence[float],
    *,
    draws: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    values = [float(value) for value in differences]
    if not values or draws <= 0:
        raise ContractError("paired bootstrap requires observations and positive draws")
    rng = random.Random(seed)
    boot = []
    for _ in range(draws):
        boot.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    boot.sort()
    low = boot[max(0, int((alpha / 2) * draws))]
    high = boot[min(draws - 1, int((1 - alpha / 2) * draws))]
    return low, high


def aggregate_causal_rows(
    rows: Sequence[Mapping[str, Any]], *, bootstrap_draws: int = 2000, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    provenance = common_provenance(rows)
    by_key: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        sample_id = str(row.get("raw_sample_id", ""))
        edge_id = str(row.get("edge_id", ""))
        mode = str(row.get("intervention_mode", ""))
        if not sample_id or not edge_id or not mode:
            raise ContractError("causal rows require raw_sample_id, edge_id, and intervention_mode")
        key = (sample_id, edge_id)
        if mode in by_key[key]:
            raise ContractError(f"duplicate causal row for {key}/{mode}")
        by_key[key][mode] = row

    paired_rows: list[dict[str, Any]] = []
    summary_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (sample_id, edge_id), modes in sorted(by_key.items()):
        if "identity" not in modes:
            raise ContractError(f"causal controls are missing identity for {sample_id}/{edge_id}")
        identity = modes["identity"]
        for mode, row in sorted(modes.items()):
            if mode == "identity":
                continue
            paired = {
                **provenance,
                "raw_sample_id": sample_id,
                "edge_id": edge_id,
                "intervention_mode": mode,
                "identity_correct": _correct(identity),
                "intervention_correct": _correct(row),
                "accuracy_difference": _correct(identity) - _correct(row),
                "identity_minimum_margin": _minimum_margin(identity),
                "intervention_minimum_margin": _minimum_margin(row),
                "gold_margin_difference": _minimum_margin(identity) - _minimum_margin(row),
                "available": not bool(row.get("intervention_unavailable", False)),
                "unavailable_reason": row.get("unavailable_reason", ""),
            }
            paired_rows.append(paired)
            if paired["available"]:
                summary_groups[(edge_id, mode)].append(paired)

    summaries: list[dict[str, Any]] = []
    for (edge_id, mode), group in sorted(summary_groups.items()):
        accuracy = [float(row["accuracy_difference"]) for row in group]
        margins = [float(row["gold_margin_difference"]) for row in group]
        acc_low, acc_high = paired_bootstrap_interval(
            accuracy, draws=bootstrap_draws, seed=seed
        )
        margin_low, margin_high = paired_bootstrap_interval(
            margins, draws=bootstrap_draws, seed=seed + 1
        )
        summaries.append(
            {
                **provenance,
                "edge_id": edge_id,
                "intervention_mode": mode,
                "n": len(group),
                "paired_accuracy_effect": sum(accuracy) / len(accuracy),
                "paired_accuracy_ci_low": acc_low,
                "paired_accuracy_ci_high": acc_high,
                "paired_gold_margin_effect": sum(margins) / len(margins),
                "paired_gold_margin_ci_low": margin_low,
                "paired_gold_margin_ci_high": margin_high,
            }
        )
    return paired_rows, summaries


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--rows-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    paired, summary = aggregate_causal_rows(
        load_jsonl(args.input), bootstrap_draws=args.bootstrap_draws, seed=args.seed
    )
    atomic_write_csv(args.rows_output, paired, overwrite=args.overwrite)
    atomic_write_csv(args.summary_output, summary, overwrite=args.overwrite)
    print(json.dumps({"paired_rows": len(paired), "summary_rows": len(summary)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
