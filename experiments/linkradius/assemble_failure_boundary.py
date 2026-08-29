"""Assemble the authenticated held-out LinkRadius failure-boundary cube.

This module is deliberately CPU-only and does not read artifacts from disk.  A
caller must authenticate completions before passing their rows here.  The
assembler then enforces the scientific join contract: clean examples with an
eligible, uniquely correct forced scorer and positive finite competitor
margins; one LinkRadius realization per frozen edge and probe seed; and exactly
one attack outcome per frozen family and requested budget.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .estimate_linkradius import estimate_from_pair_rows
from .probe_validation import reclassify_probe_pairs
from .schemas import ContractError, DEFAULT_CLEAN_CORRECT_POLICY
from .select_clean_correct import classify_screening_row


ASSEMBLY_VERSION = "linkradius.failure_boundary_assembly.v4"

# Config hashes are task-specific and therefore intentionally absent.  These
# fields describe identities that must not change between clean, probe, and
# attack execution.
_REQUIRED_PROVENANCE_FIELDS = (
    "partition",
    "source_hash",
    "split_manifest_hash",
    "execution_manifest_hash",
    "model_hash",
    "scorer_hash",
)
_OPTIONAL_PROVENANCE_FIELDS = (
    "dataset",
    "source_split",
    "style",
    "method",
    "R",
    "adapter_hash",
    "prompt_hash",
    "ordered_cohort_hash",
    "batch_boundary_hash",
    "subspace_hash",
)


def _nonempty(value: Any, *, field: str) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ContractError(f"failure-boundary row is missing {field}")
    return value


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{field} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{field} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ContractError(f"{field} must be a finite number")
    return parsed


def _provenance(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        field: _nonempty(row.get(field), field=field)
        for field in _REQUIRED_PROVENANCE_FIELDS
    }
    for field in _OPTIONAL_PROVENANCE_FIELDS:
        if field in row and row[field] is not None:
            result[field] = row[field]
    return result


def _require_matching_provenance(
    expected: Mapping[str, Any], row: Mapping[str, Any], *, where: str
) -> None:
    observed = _provenance(row)
    for field, value in expected.items():
        if field not in observed:
            raise ContractError(f"{where} provenance is missing {field}")
        if observed[field] != value:
            raise ContractError(f"{where} provenance mismatch for {field}")
    # An optional identity may first appear on a later artifact.  Reject that
    # ambiguity rather than silently accepting two differently shaped claims.
    for field in _REQUIRED_PROVENANCE_FIELDS:
        if observed[field] != expected[field]:
            raise ContractError(f"{where} provenance mismatch for {field}")


def _clean_margins(row: Mapping[str, Any]) -> dict[str, float]:
    raw = row.get("margins")
    if not isinstance(raw, Mapping) or not raw:
        raise ContractError("scorer-correct clean rows require competitor margins")
    margins = {
        str(label): _finite(value, field=f"clean margins.{label}")
        for label, value in raw.items()
    }
    if any(value <= 0.0 for value in margins.values()):
        raise ContractError("scorer-correct clean margins must be strictly positive")
    return margins


def _canonical_budget(value: Any, budgets: Sequence[float]) -> float:
    parsed = _finite(value, field="requested_epsilon")
    matches = [
        budget
        for budget in budgets
        if math.isclose(parsed, budget, rel_tol=1e-12, abs_tol=1e-12)
    ]
    if len(matches) != 1:
        raise ContractError(f"attack row uses an unfrozen requested budget: {parsed}")
    return matches[0]


def _failure_score(epsilon: float, radius: float) -> float:
    """Return epsilon/radius with deterministic zero/infinity conventions."""

    if epsilon == 0.0:
        return 0.0
    if radius == 0.0:
        return math.inf
    if math.isinf(radius):
        return 0.0
    return epsilon / radius


def assemble_failure_boundary(
    clean_rows: Sequence[Mapping[str, Any]],
    probe_rows: Sequence[Mapping[str, Any]],
    attack_rows: Sequence[Mapping[str, Any]],
    *,
    frozen_edges: Sequence[str],
    frozen_budgets: Sequence[float],
    frozen_families: Sequence[str],
    requested_K: int,
    selected_h: float,
    probe_seeds: Sequence[int],
    probe_acceptance_thresholds: Mapping[str, Any] | None = None,
    clean_correct_policy: str = DEFAULT_CLEAN_CORRECT_POLICY,
    minimum_K_eff: int | None = None,
) -> dict[str, Any]:
    """Build seed-specific prediction and predictor rows for held-out RQ2.

    Rows with a non-``sample``/``probe_pair`` record type are treated as shard
    metadata.  Attack/probe rows for known examples without an eligible,
    uniquely correct forced scorer and positive finite competitor margins are
    excluded; an unknown example or an extra frozen-grid coordinate fails
    closed.  Free-form generation fields are retained upstream as diagnostics
    but do not determine this forced-margin analysis cohort.
    """

    edges = tuple(str(value).strip() for value in frozen_edges)
    families = tuple(str(value).strip() for value in frozen_families)
    budgets = tuple(_finite(value, field="frozen budget") for value in frozen_budgets)
    h = _finite(selected_h, field="selected_h")
    if (
        not edges
        or any(not value for value in edges)
        or len(set(edges)) != len(edges)
        or not families
        or any(not value for value in families)
        or len(set(families)) != len(families)
        or not budgets
        or any(value < 0.0 for value in budgets)
        or len(set(budgets)) != len(budgets)
        or isinstance(requested_K, bool)
        or int(requested_K) <= 0
    ):
        raise ContractError("frozen failure-boundary configuration is invalid")
    K = int(requested_K)
    min_K_eff = K if minimum_K_eff is None else int(minimum_K_eff)
    if (
        isinstance(minimum_K_eff, bool)
        or min_K_eff < 1
        or min_K_eff > K
    ):
        raise ContractError("minimum_K_eff must lie in [1, requested_K]")
    if any(isinstance(value, bool) for value in probe_seeds):
        raise ContractError("frozen failure-boundary probe seeds are invalid")
    seeds = tuple(int(value) for value in probe_seeds)
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ContractError(
            "failure-boundary evaluation requires at least three unique probe seeds"
        )

    clean_by_id: dict[str, Mapping[str, Any]] = {}
    for row in clean_rows:
        if row.get("record_type", "sample") != "sample":
            continue
        raw_id = str(row.get("raw_sample_id") or "")
        if not raw_id or raw_id in clean_by_id:
            raise ContractError("clean rows require unique nonempty raw_sample_id values")
        clean_by_id[raw_id] = row
    if not clean_by_id:
        raise ContractError("failure-boundary assembly requires clean sample rows")

    eligible = {}
    for raw_id, row in clean_by_id.items():
        clean_correct, _ = classify_screening_row(
            row, clean_correct_policy=clean_correct_policy
        )
        if bool(row.get("analysis_eligible", False)) and clean_correct:
            eligible[raw_id] = row
    if not eligible:
        raise ContractError(
            f"no eligible {clean_correct_policy} clean-correct examples are available"
        )

    # Every clean row must describe one execution identity, including excluded
    # rows.  This prevents a mixed split from being hidden by cohort filtering.
    first_clean = next(iter(clean_by_id.values()))
    common_provenance = _provenance(first_clean)
    for raw_id, row in clean_by_id.items():
        _require_matching_provenance(
            common_provenance, row, where=f"clean row {raw_id}"
        )

    classified_pairs: Sequence[Mapping[str, Any]]
    if probe_acceptance_thresholds is not None:
        classified_pairs = reclassify_probe_pairs(
            probe_rows, probe_acceptance_thresholds
        )
    else:
        classified_pairs = [
            row for row in probe_rows if row.get("record_type") == "probe_pair"
        ]
        if any("accepted" not in row for row in classified_pairs):
            raise ContractError(
                "probe pairs require stored classification or acceptance thresholds"
            )

    pair_groups: dict[
        tuple[str, str, int], dict[int, Mapping[str, Any]]
    ] = defaultdict(dict)
    probe_subspace_by_edge: dict[str, str] = {}
    for row in classified_pairs:
        raw_id = str(row.get("raw_sample_id") or "")
        if raw_id not in clean_by_id:
            raise ContractError(f"probe row references unknown clean example {raw_id!r}")
        if raw_id not in eligible:
            continue
        edge = str(row.get("edge_id") or "")
        if edge not in edges:
            raise ContractError(f"probe row uses an unfrozen edge: {edge!r}")
        if not math.isclose(
            _finite(row.get("h"), field="probe h"), h, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ContractError("probe row uses an unfrozen h")
        seed = int(row.get("probe_seed"))
        if seed not in seeds:
            raise ContractError("probe row uses an unfrozen probe seed")
        _require_matching_provenance(
            common_provenance, row, where=f"probe row {raw_id}/{edge}"
        )
        subspace_hash = str(row.get("subspace_hash") or "")
        if not subspace_hash:
            raise ContractError("probe row is missing edge-specific subspace_hash")
        previous_subspace = probe_subspace_by_edge.setdefault(edge, subspace_hash)
        if previous_subspace != subspace_hash:
            raise ContractError(f"probe subspace identity differs within edge {edge}")
        direction = int(row.get("direction_id"))
        if direction not in range(K):
            raise ContractError("probe row lies outside the frozen K prefix")
        key = (raw_id, edge, seed)
        if direction in pair_groups[key]:
            raise ContractError(
                f"duplicate probe direction for {raw_id}/{edge}/seed={seed}"
            )
        pair_groups[key][direction] = row

    edge_predictors: list[dict[str, Any]] = []
    probe_exclusions: list[dict[str, Any]] = []
    predictor_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    expected_directions = set(range(K))
    for raw_id in sorted(eligible):
        margins = _clean_margins(eligible[raw_id])
        for edge in edges:
            for seed in seeds:
                by_direction = pair_groups.get((raw_id, edge, seed), {})
                if set(by_direction) != expected_directions:
                    raise ContractError(
                        f"incomplete frozen probe prefix for {raw_id}/{edge}/seed={seed}: "
                        f"observed={sorted(by_direction)}, expected={sorted(expected_directions)}"
                    )
                normalized: list[dict[str, Any]] = []
                q_values: set[int] = set()
                accepted_directions: set[int] = set()
                for direction in range(K):
                    pair = by_direction[direction]
                    pair_clean = pair.get("clean_margins")
                    if not isinstance(pair_clean, Mapping) or set(pair_clean) != set(margins):
                        raise ContractError("probe and clean competitor sets differ")
                    for label, value in margins.items():
                        if not math.isclose(
                            _finite(pair_clean[label], field=f"probe clean margin {label}"),
                            value,
                            rel_tol=1e-9,
                            abs_tol=1e-9,
                        ):
                            raise ContractError("probe clean margins differ from clean baseline")
                    q = int(pair.get("q"))
                    if q <= 0:
                        raise ContractError("probe q must be positive")
                    q_values.add(q)
                    derivatives = pair.get("central_differences", pair.get("derivatives"))
                    if not isinstance(derivatives, Mapping) or set(derivatives) != set(margins):
                        raise ContractError("classified probe pair has incomplete derivatives")
                    accepted = bool(pair.get("accepted", False))
                    if accepted:
                        accepted_directions.add(direction)
                    normalized.append(
                        {
                            "direction_id": direction,
                            "accepted": accepted,
                            "derivatives": {
                                # Rejected cast-domain probes may deliberately have
                                # no derivative.  Their values are never consumed by
                                # the primary estimate, but accepted probes must be
                                # finite.
                                label: (
                                    _finite(
                                        derivatives[label],
                                        field=f"probe derivative {label}",
                                    )
                                    if accepted
                                    else 0.0
                                )
                                for label in margins
                            },
                        }
                    )
                if len(q_values) != 1:
                    raise ContractError("probe q differs within one primary estimate")
                if len(accepted_directions) < min_K_eff:
                    probe_exclusions.append(
                        {
                            "raw_sample_id": raw_id,
                            "raw_id": raw_id,
                            "edge_id": edge,
                            "probe_seed": seed,
                            "reason": "insufficient_accepted_probe_directions",
                            "requested_K": K,
                            "minimum_K_eff": min_K_eff,
                            "K_eff": len(accepted_directions),
                            "rejected_direction_ids": sorted(
                                expected_directions - accepted_directions
                            ),
                        }
                    )
                    continue
                estimate = estimate_from_pair_rows(
                    normalized,
                    clean_margins=margins,
                    q=next(iter(q_values)),
                    requested_K=K,
                )
                minimum_margin = min(float(row["clean_margin"]) for row in estimate["competitors"])
                maximum_susceptibility = max(
                    float(row["susceptibility"]) for row in estimate["competitors"]
                )
                predictor = {
                    "schema_version": ASSEMBLY_VERSION,
                    "raw_sample_id": raw_id,
                    "raw_id": raw_id,
                    "edge_id": edge,
                    "edge_radius": float(estimate["edge_radius"]),
                    "predicted_radius": float(estimate["edge_radius"]),
                    "binding_competitor": str(estimate["binding_competitor"]),
                    "minimum_clean_margin": minimum_margin,
                    "clean_margin": minimum_margin,
                    "maximum_susceptibility": maximum_susceptibility,
                    "susceptibility": maximum_susceptibility,
                    "q": next(iter(q_values)),
                    "requested_K": K,
                    "K_eff": int(estimate["K_eff"]),
                    "minimum_K_eff": min_K_eff,
                    "primary_available": bool(estimate["primary_available"]),
                    "estimate_kind": (
                        "complete_K"
                        if estimate["primary_available"]
                        else "accepted_direction_subset"
                    ),
                    "h": h,
                    "probe_seed": seed,
                    "clean_correct_policy": clean_correct_policy,
                    "subspace_hash": probe_subspace_by_edge[edge],
                    **common_provenance,
                }
                edge_predictors.append(predictor)
                predictor_by_key[(raw_id, edge, seed)] = predictor

    attacks: dict[tuple[str, str, str, float], Mapping[str, Any]] = {}
    for row in attack_rows:
        if row.get("record_type", "sample") != "sample":
            continue
        raw_id = str(row.get("raw_sample_id") or "")
        if raw_id not in clean_by_id:
            raise ContractError(f"attack row references unknown clean example {raw_id!r}")
        if raw_id not in eligible:
            continue
        edge = str(row.get("edge_id") or "")
        family = str(row.get("attack_family") or "")
        if edge not in edges:
            raise ContractError(f"attack row uses an unfrozen edge: {edge!r}")
        if family not in families:
            raise ContractError(f"attack row uses an unfrozen family: {family!r}")
        budget = _canonical_budget(row.get("requested_epsilon"), budgets)
        _require_matching_provenance(
            common_provenance,
            row,
            where=f"attack row {raw_id}/{edge}/{family}/{budget}",
        )
        if row.get("subspace_hash") != probe_subspace_by_edge.get(edge):
            raise ContractError(
                f"attack/probe subspace provenance mismatch for edge {edge}"
            )
        key = (raw_id, edge, family, budget)
        if key in attacks:
            raise ContractError(f"duplicate attack cube coordinate: {key}")
        attacks[key] = row

    expected_attack_coordinates = {
        (raw_id, edge, family, budget)
        for raw_id in eligible
        for edge in edges
        for family in families
        for budget in budgets
    }
    missing = sorted(expected_attack_coordinates - set(attacks))
    extra = sorted(set(attacks) - expected_attack_coordinates)
    if missing or extra:
        raise ContractError(
            "attack result cube is incomplete or contains unexpected coordinates; "
            f"missing={missing[:1]}, extra={extra[:1]}"
        )

    prediction_units: list[dict[str, Any]] = []
    for raw_id, edge, seed in sorted(predictor_by_key):
        predictor = predictor_by_key[(raw_id, edge, seed)]
        for family in families:
            for budget in budgets:
                key = (raw_id, edge, family, budget)
                attack = attacks[key]
                minimum_margin = _finite(
                    attack.get("minimum_margin"), field="attack minimum_margin"
                )
                realized_epsilon = _finite(
                    attack.get("realized_epsilon"),
                    field="attack realized_epsilon",
                )
                if realized_epsilon <= 0.0 or realized_epsilon > budget + 1e-6 * max(
                    1.0, budget
                ):
                    raise ContractError(
                        "attack realized epsilon violates its requested budget"
                    )
                if family == "random_independent" and probe_acceptance_thresholds:
                    diagnostics = attack.get("realized_intervention")
                    if not isinstance(diagnostics, Mapping):
                        raise ContractError(
                            "random attack is missing post-cast diagnostics"
                        )
                    cosine = _finite(
                        diagnostics.get("requested_realized_cosine"),
                        field="random requested-realized cosine",
                    )
                    off_direction = _finite(
                        diagnostics.get("off_direction_relative"),
                        field="random off-direction relative norm",
                    )
                    if (
                        bool(diagnostics.get("collapsed", True))
                        or cosine
                        < float(
                            probe_acceptance_thresholds[
                                "minimum_requested_realized_cosine"
                            ]
                        )
                        or off_direction
                        > float(
                            probe_acceptance_thresholds[
                                "maximum_off_direction_relative"
                            ]
                        )
                    ):
                        raise ContractError(
                            "random attack violates frozen post-cast direction quality"
                        )
                prediction_units.append(
                    {
                        **predictor,
                        "attack_family": family,
                        "requested_epsilon": budget,
                        "realized_epsilon": realized_epsilon,
                        "attack_minimum_margin": minimum_margin,
                        "flipped": minimum_margin <= 0.0,
                        # Binary ranking is evaluated within one frozen
                        # requested-budget cell.  Using each attack's achieved
                        # norm here would leak attack optimization behavior into
                        # LinkRadius alone and make the component comparison
                        # unfair.  Achieved norm remains the primary threshold
                        # coordinate and a separately named diagnostic score.
                        "linkradius_score": _failure_score(
                            budget, float(predictor["edge_radius"])
                        ),
                        "linkradius_requested_score": _failure_score(
                            budget, float(predictor["edge_radius"])
                        ),
                        "linkradius_realized_score": _failure_score(
                            realized_epsilon, float(predictor["edge_radius"])
                        ),
                        "margin_score": -float(predictor["minimum_clean_margin"]),
                        "susceptibility_score": float(
                            predictor["maximum_susceptibility"]
                        ),
                        "attack_run_id": attack.get("run_id"),
                    }
                )
    return {
        "schema_version": ASSEMBLY_VERSION,
        "prediction_units": prediction_units,
        "edge_predictors": edge_predictors,
        "probe_exclusions": probe_exclusions,
        "eligible_raw_sample_ids": sorted(eligible),
        "evaluated_raw_sample_ids": sorted(
            {raw_id for raw_id, _, _ in predictor_by_key}
        ),
        "excluded_raw_sample_ids": sorted(set(clean_by_id) - set(eligible)),
        "counts": {
            "clean_rows": len(clean_by_id),
            "eligible_rows": len(eligible),
            "edge_predictors": len(edge_predictors),
            "probe_exclusions": len(probe_exclusions),
            "prediction_units": len(prediction_units),
        },
        "frozen_configuration": {
            "edges": list(edges),
            "budgets": list(budgets),
            "families": list(families),
            "requested_K": K,
            "minimum_K_eff": min_K_eff,
            "selected_h": h,
            "probe_seeds": list(seeds),
            "clean_correct_policy": clean_correct_policy,
        },
        "provenance": common_provenance,
    }


__all__ = ["ASSEMBLY_VERSION", "assemble_failure_boundary"]
