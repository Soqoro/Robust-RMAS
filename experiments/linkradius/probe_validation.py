"""Validation-only probe calibration, reclassification, and stability checks.

The functions in this module are deliberately CPU-only.  They derive every
quality threshold from validation diagnostics, reconstruct pair acceptance
from the signed records, and compute primary (complete-prefix) estimates for
the nested direction counts.  No stored ``accepted`` flag or stored central
difference is trusted during calibration.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .io_utils import content_hash
from .schemas import ContractError


THRESHOLD_DERIVATION_VERSION = "linkradius.probe_threshold_derivation.v1"
PAIR_RECLASSIFICATION_VERSION = "linkradius.probe_pair_reclassification.v1"
CALIBRATION_VERSION = "linkradius.probe_calibration.v1"
SELECTION_ALGORITHM = "validation_primary_coverage_acceptance_then_small_h_v1"


def empirical_quantile(values: Sequence[float], probability: float) -> float:
    """Return the deterministic type-7 linear empirical quantile."""

    if not 0.0 <= float(probability) <= 1.0:
        raise ContractError("quantile probability must lie in [0,1]")
    parsed = sorted(float(value) for value in values)
    if not parsed or not all(math.isfinite(value) for value in parsed):
        raise ContractError("quantiles require at least one finite observation")
    if len(parsed) == 1:
        return parsed[0]
    position = (len(parsed) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return parsed[lower] * (1.0 - weight) + parsed[upper] * weight


def _signed_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row.get("record_type") == "sample"
        and row.get("intervention_mode") == "additive_antithetic"
        and row.get("sign") in {-1, 1}
        and bool(row.get("analysis_eligible", False))
    ]


def _pair_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row.get("record_type") == "probe_pair"
        and bool(row.get("analysis_eligible", False))
    ]


def _diagnostics(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("diagnostics", row.get("realized_intervention"))
    if not isinstance(value, Mapping):
        raise ContractError("signed probe row is missing realized diagnostics")
    return value


def _finite_optional(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def derive_acceptance_thresholds(
    rows: Sequence[Mapping[str, Any]],
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive cast-quality cutoffs exclusively from the supplied rows."""

    if not 0.0 <= lower_quantile < 0.5 or not 0.5 < upper_quantile <= 1.0:
        raise ContractError("probe threshold quantiles must bracket the median")
    signed = _signed_rows(rows)
    pairs = _pair_rows(rows)
    if not signed or not pairs:
        raise ContractError("threshold derivation requires signed rows and pair rows")

    cosines: list[float] = []
    residuals: list[float] = []
    for row in signed:
        diagnostics = _diagnostics(row)
        cosine = _finite_optional(diagnostics.get("requested_realized_cosine"))
        residual = _finite_optional(diagnostics.get("off_direction_relative"))
        if cosine is not None:
            cosines.append(cosine)
        if residual is not None and residual >= 0.0:
            residuals.append(residual)
    separations = [
        value
        for row in pairs
        if (value := _finite_optional(row.get("realized_separation"))) is not None
        and value > 0.0
    ]
    antipodalities = [
        value
        for row in pairs
        if (value := _finite_optional(row.get("antipodality"))) is not None
    ]
    if not cosines or not residuals or not separations or not antipodalities:
        raise ContractError("validation diagnostics cannot identify all four probe thresholds")

    thresholds = {
        "minimum_requested_realized_cosine": max(
            -1.0, min(1.0, empirical_quantile(cosines, lower_quantile))
        ),
        "maximum_off_direction_relative": max(
            0.0, empirical_quantile(residuals, upper_quantile)
        ),
        "minimum_signed_separation": max(
            0.0, empirical_quantile(separations, lower_quantile)
        ),
        "minimum_antipodality": max(
            -1.0, min(1.0, empirical_quantile(antipodalities, lower_quantile))
        ),
        "version": "linkradius_probe_thresholds_v1",
    }
    derivation = {
        "schema_version": THRESHOLD_DERIVATION_VERSION,
        "partition": "validation",
        "quantile_algorithm": "type7_linear_v1",
        "lower_quantile": float(lower_quantile),
        "upper_quantile": float(upper_quantile),
        "observation_counts": {
            "requested_realized_cosine": len(cosines),
            "off_direction_relative": len(residuals),
            "positive_signed_separation": len(separations),
            "antipodality": len(antipodalities),
        },
    }
    derivation["content_hash"] = content_hash(
        {"thresholds": thresholds, **derivation},
        domain="linkradius:probe_threshold_derivation:v1",
    )
    return thresholds, derivation


def _same_number(left: Any, right: Any, *, tolerance: float) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        left_value, right_value = float(left), float(right)
    except (TypeError, ValueError):
        return False
    return math.isfinite(left_value) and math.isfinite(right_value) and math.isclose(
        left_value,
        right_value,
        rel_tol=tolerance,
        abs_tol=tolerance,
    )


def _require_pair_identity(
    pair: Mapping[str, Any],
    signed: Mapping[str, Any],
    *,
    sign: int,
) -> None:
    if int(signed.get("sign", 0)) != sign:
        raise ContractError("probe pair signed-run ID references the wrong sign")
    for field in (
        "raw_sample_id",
        "sample_id",
        "edge_id",
        "direction_id",
        "probe_seed",
        "h",
        "q",
        "subspace_id",
        "partition",
        "split_manifest_hash",
        "execution_manifest_hash",
        "source_hash",
        "config_hash",
        "model_hash",
        "adapter_hash",
        "prompt_hash",
        "scorer_hash",
        "subspace_hash",
    ):
        if field not in pair or field not in signed:
            raise ContractError(f"probe pair provenance is missing {field}")
        if pair[field] != signed[field]:
            raise ContractError(f"probe pair and signed record disagree on {field}")


def reclassify_probe_pairs(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    *,
    numeric_tolerance: float = 1e-9,
) -> list[dict[str, Any]]:
    """Reconstruct pair acceptance and derivatives from signed evidence."""

    required_thresholds = (
        "minimum_requested_realized_cosine",
        "maximum_off_direction_relative",
        "minimum_signed_separation",
        "minimum_antipodality",
    )
    if any(field not in thresholds for field in required_thresholds):
        raise ContractError("frozen probe thresholds are incomplete")
    if numeric_tolerance < 0.0:
        raise ContractError("numeric tolerance must be non-negative")

    signed_by_id: dict[str, Mapping[str, Any]] = {}
    for row in _signed_rows(rows):
        run_id = str(row.get("run_id") or "")
        if not run_id or run_id in signed_by_id:
            raise ContractError("signed probe run IDs must be nonempty and unique")
        signed_by_id[run_id] = row

    output: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    used_signed_ids: set[str] = set()
    for pair in _pair_rows(rows):
        pair_id = str(pair.get("run_id") or "")
        if not pair_id or pair_id in pair_ids:
            raise ContractError("probe pair run IDs must be nonempty and unique")
        pair_ids.add(pair_id)
        try:
            plus_id = str(pair["plus_run_id"])
            minus_id = str(pair["minus_run_id"])
            plus = signed_by_id[plus_id]
            minus = signed_by_id[minus_id]
        except KeyError as exc:
            raise ContractError("probe pair references a missing signed run") from exc
        if plus_id in used_signed_ids or minus_id in used_signed_ids:
            raise ContractError("a signed probe run is reused by multiple pairs")
        used_signed_ids.update((plus_id, minus_id))
        _require_pair_identity(pair, plus, sign=1)
        _require_pair_identity(pair, minus, sign=-1)
        plus_diag, minus_diag = _diagnostics(plus), _diagnostics(minus)

        t_plus = _finite_optional(plus_diag.get("realized_signed_coordinate"))
        t_minus = _finite_optional(minus_diag.get("realized_signed_coordinate"))
        if not _same_number(pair.get("t_plus"), t_plus, tolerance=numeric_tolerance):
            raise ContractError("probe pair t_plus differs from its signed record")
        if not _same_number(pair.get("t_minus"), t_minus, tolerance=numeric_tolerance):
            raise ContractError("probe pair t_minus differs from its signed record")
        separation = None if t_plus is None or t_minus is None else t_plus - t_minus
        if not _same_number(
            pair.get("realized_separation"), separation, tolerance=numeric_tolerance
        ):
            raise ContractError("probe pair separation is not numerically reproducible")

        plus_margins = plus.get("margins")
        minus_margins = minus.get("margins")
        if not isinstance(plus_margins, Mapping) or not isinstance(minus_margins, Mapping):
            raise ContractError("signed probe records require competitor margins")
        if set(plus_margins) != set(minus_margins):
            raise ContractError("signed probe margin competitors differ")
        for stored_name, signed_margins in (
            ("margins_plus", plus_margins),
            ("margins_minus", minus_margins),
        ):
            stored = pair.get(stored_name)
            if not isinstance(stored, Mapping) or set(stored) != set(signed_margins):
                raise ContractError(f"probe pair {stored_name} has incompatible competitors")
            if any(
                not _same_number(stored[label], signed_margins[label], tolerance=numeric_tolerance)
                for label in stored
            ):
                raise ContractError(f"probe pair {stored_name} differs from signed evidence")

        derivatives: dict[str, float | None] = {}
        if separation is not None and math.isfinite(separation) and separation > 0.0:
            derivatives = {
                label: (float(plus_margins[label]) - float(minus_margins[label]))
                / separation
                for label in sorted(plus_margins)
            }
        else:
            derivatives = {label: None for label in sorted(plus_margins)}
        stored_derivatives = pair.get("central_differences")
        if not isinstance(stored_derivatives, Mapping) or set(stored_derivatives) != set(derivatives):
            raise ContractError("probe pair central differences have incompatible competitors")
        if any(
            not _same_number(
                stored_derivatives[label], derivatives[label], tolerance=numeric_tolerance
            )
            for label in derivatives
        ):
            raise ContractError("probe pair central differences are not numerically reproducible")

        reasons: list[str] = []
        for name, diagnostics in (("plus", plus_diag), ("minus", minus_diag)):
            if bool(diagnostics.get("collapsed")):
                reasons.append(f"{name}_collapsed")
            cosine = _finite_optional(diagnostics.get("requested_realized_cosine"))
            if cosine is None:
                reasons.append(f"{name}_cosine_unavailable")
            elif cosine < float(thresholds["minimum_requested_realized_cosine"]):
                reasons.append(f"{name}_cosine_below_threshold")
            residual = _finite_optional(diagnostics.get("off_direction_relative"))
            if residual is None:
                reasons.append(f"{name}_residual_unavailable")
            elif residual > float(thresholds["maximum_off_direction_relative"]):
                reasons.append(f"{name}_residual_above_threshold")
        if separation is None:
            reasons.append("separation_unavailable")
        elif separation <= 0.0:
            reasons.append("nonpositive_signed_separation")
        elif separation < float(thresholds["minimum_signed_separation"]):
            reasons.append("separation_below_threshold")
        antipodality = _finite_optional(pair.get("antipodality"))
        if antipodality is None:
            reasons.append("antipodality_unavailable")
        elif antipodality < float(thresholds["minimum_antipodality"]):
            reasons.append("antipodality_below_threshold")

        output.append(
            {
                **dict(pair),
                "t_plus": t_plus,
                "t_minus": t_minus,
                "realized_separation": separation,
                "accepted": not reasons,
                "rejection_reasons": list(dict.fromkeys(reasons)),
                "central_differences": derivatives,
                "plus_requested_realized_cosine": _finite_optional(
                    plus_diag.get("requested_realized_cosine")
                ),
                "minus_requested_realized_cosine": _finite_optional(
                    minus_diag.get("requested_realized_cosine")
                ),
                "plus_off_direction_relative": _finite_optional(
                    plus_diag.get("off_direction_relative")
                ),
                "minus_off_direction_relative": _finite_optional(
                    minus_diag.get("off_direction_relative")
                ),
                "plus_collapsed": bool(plus_diag.get("collapsed")),
                "minus_collapsed": bool(minus_diag.get("collapsed")),
                "classification_version": PAIR_RECLASSIFICATION_VERSION,
            }
        )
    if not output:
        raise ContractError("probe calibration requires at least one pair")
    if used_signed_ids != set(signed_by_id):
        raise ContractError("eligible signed probe records are not paired exactly once")
    return sorted(
        output,
        key=lambda row: (
            str(row["raw_sample_id"]),
            str(row["edge_id"]),
            float(row["h"]),
            int(row["probe_seed"]),
            int(row["direction_id"]),
        ),
    )


def validate_complete_eligible_cube(
    pairs: Sequence[Mapping[str, Any]],
    *,
    expected_raw_ids: Sequence[str] | None = None,
    expected_configurations: Sequence[tuple[str, float, int, int]] | None = None,
) -> dict[str, Any]:
    """Require an exact eligible sample × canonical probe-configuration cube."""

    by_sample: dict[str, set[tuple[str, float, int, int]]] = defaultdict(set)
    for row in pairs:
        raw_id = str(row.get("raw_sample_id") or "")
        key = (
            str(row.get("edge_id") or ""),
            float(row["h"]),
            int(row["probe_seed"]),
            int(row["direction_id"]),
        )
        if not raw_id or key in by_sample[raw_id]:
            raise ContractError("eligible probe cube contains a duplicate or invalid pair")
        by_sample[raw_id].add(key)
    if not by_sample:
        raise ContractError("eligible probe cube is empty")
    observed_raw_ids = set(by_sample)
    required_raw_ids = (
        {str(value) for value in expected_raw_ids}
        if expected_raw_ids is not None
        else observed_raw_ids
    )
    if not required_raw_ids or observed_raw_ids != required_raw_ids:
        missing = sorted(required_raw_ids - observed_raw_ids)
        extra = sorted(observed_raw_ids - required_raw_ids)
        raise ContractError(
            f"eligible probe sample coverage differs from the frozen execution; missing={missing[:5]}, extra={extra[:5]}"
        )
    expected = (
        {
            (str(edge), float(h), int(seed), int(direction))
            for edge, h, seed, direction in expected_configurations
        }
        if expected_configurations is not None
        else set().union(*by_sample.values())
    )
    if not expected:
        raise ContractError("canonical probe configuration cube is empty")
    incomplete = [raw_id for raw_id, values in by_sample.items() if values != expected]
    if incomplete:
        raise ContractError(f"eligible probe rows have incomplete configuration coverage: {incomplete[:5]}")
    by_configuration: dict[tuple[str, float, int], set[int]] = defaultdict(set)
    for edge, h, seed, direction in expected:
        by_configuration[(edge, h, seed)].add(direction)
    for configuration, directions in by_configuration.items():
        if directions != set(range(max(directions) + 1)):
            raise ContractError(f"probe directions are not a complete nested prefix: {configuration}")
    result = {
        "raw_sample_count": len(required_raw_ids),
        "configuration_count": len(expected),
        "pair_count": len(required_raw_ids) * len(expected),
        "raw_sample_ids_hash": content_hash(
            sorted(required_raw_ids), domain="linkradius:probe_cube_raw_ids:v1"
        ),
        "configuration_hash": content_hash(
            sorted(expected), domain="linkradius:probe_cube_configurations:v1"
        ),
    }
    result["content_hash"] = content_hash(
        result, domain="linkradius:probe_cube_coverage:v1"
    )
    return result


def _primary_estimates(
    pairs: Sequence[Mapping[str, Any]], candidate_K: Sequence[int]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairs:
        if not bool(row.get("analysis_eligible", False)):
            continue
        groups[
            (
                str(row["raw_sample_id"]),
                str(row["edge_id"]),
                float(row["h"]),
                int(row["probe_seed"]),
            )
        ].append(row)
    estimates: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        by_direction: dict[int, Mapping[str, Any]] = {}
        for row in group:
            direction_id = int(row["direction_id"])
            if direction_id in by_direction:
                raise ContractError("duplicate direction in a probe estimate group")
            by_direction[direction_id] = row
        first = group[0]
        clean_margins = first.get("clean_margins")
        if not isinstance(clean_margins, Mapping) or not clean_margins:
            raise ContractError("probe pairs require clean competitor margins")
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in clean_margins.values()):
            # LinkRadius is undefined for rows that are not scorer-correct.
            continue
        q = int(first["q"])
        if q <= 0:
            raise ContractError("probe subspace dimension q must be positive")
        for requested_K in candidate_K:
            direction_ids = tuple(range(int(requested_K)))
            if any(direction_id not in by_direction for direction_id in direction_ids):
                continue
            selected = [by_direction[direction_id] for direction_id in direction_ids]
            if not all(bool(row.get("accepted")) for row in selected):
                continue
            competitors = tuple(sorted(str(label) for label in clean_margins))
            derivatives: dict[str, list[float]] = {label: [] for label in competitors}
            valid = True
            for row in selected:
                values = row.get("central_differences")
                if not isinstance(values, Mapping) or set(values) != set(competitors):
                    valid = False
                    break
                for label in competitors:
                    value = _finite_optional(values[label])
                    if value is None:
                        valid = False
                        break
                    derivatives[label].append(value)
                if not valid:
                    break
            if not valid:
                continue
            competitor_rows = []
            for label in competitors:
                susceptibility = math.sqrt(
                    float(q)
                    * sum(value * value for value in derivatives[label])
                    / int(requested_K)
                )
                radius = (
                    math.inf
                    if susceptibility == 0.0
                    else float(clean_margins[label]) / susceptibility
                )
                competitor_rows.append((radius, label, susceptibility))
            radius, binding, _ = min(competitor_rows, key=lambda item: (item[0], item[1]))
            estimates.append(
                {
                    "raw_sample_id": key[0],
                    "edge_id": key[1],
                    "h": key[2],
                    "probe_seed": key[3],
                    "K": int(requested_K),
                    "edge_radius": radius,
                    "binding_competitor": binding,
                }
            )
    return estimates


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: float(values[index]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + 1 + stop) / 2.0
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks, right_ranks = _average_ranks(left), _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_ranks, right_ranks)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left_ranks)
        * sum((value - right_mean) ** 2 for value in right_ranks)
    )
    if denominator == 0.0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def stability_summary(estimates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare primary estimates across radii, seeds, and nested K values."""

    by_config: dict[tuple[float, int, int], dict[tuple[str, str], Mapping[str, Any]]] = defaultdict(dict)
    for row in estimates:
        config = (float(row["h"]), int(row["probe_seed"]), int(row["K"]))
        unit = (str(row["raw_sample_id"]), str(row["edge_id"]))
        if unit in by_config[config]:
            raise ContractError("duplicate primary estimate for one unit/configuration")
        by_config[config][unit] = row

    comparisons: dict[str, list[dict[str, Any]]] = {
        "radius": [],
        "seed": [],
        "K": [],
    }
    configs = sorted(by_config)
    for left_index, left in enumerate(configs):
        for right in configs[left_index + 1 :]:
            differing = [index for index in range(3) if left[index] != right[index]]
            if len(differing) != 1:
                continue
            dimension = ("radius", "seed", "K")[differing[0]]
            common = sorted(set(by_config[left]) & set(by_config[right]))
            finite = [
                unit
                for unit in common
                if math.isfinite(float(by_config[left][unit]["edge_radius"]))
                and math.isfinite(float(by_config[right][unit]["edge_radius"]))
            ]
            correlation = _spearman(
                [float(by_config[left][unit]["edge_radius"]) for unit in finite],
                [float(by_config[right][unit]["edge_radius"]) for unit in finite],
            )
            binding = (
                sum(
                    by_config[left][unit]["binding_competitor"]
                    == by_config[right][unit]["binding_competitor"]
                    for unit in common
                )
                / len(common)
                if common
                else None
            )
            if common:
                comparisons[dimension].append(
                    {
                        "left": {"h": left[0], "probe_seed": left[1], "K": left[2]},
                        "right": {"h": right[0], "probe_seed": right[1], "K": right[2]},
                        "common_units": len(common),
                        "finite_radius_units": len(finite),
                        "rank_correlation": correlation,
                        "binding_agreement": binding,
                    }
                )

    dimensions: dict[str, dict[str, Any]] = {}
    for name, records in comparisons.items():
        correlations = [
            float(record["rank_correlation"])
            for record in records
            if record["rank_correlation"] is not None
        ]
        binding_values = [
            float(record["binding_agreement"])
            for record in records
            if record["binding_agreement"] is not None
        ]
        dimensions[name] = {
            "comparison_count": len(records),
            "finite_rank_comparison_count": len(correlations),
            "median_rank_correlation": _median(correlations),
            "minimum_rank_correlation": min(correlations) if correlations else None,
            "median_binding_agreement": _median(binding_values),
            "minimum_binding_agreement": min(binding_values) if binding_values else None,
            "comparisons": records,
        }
    result = {
        "schema_version": "linkradius.probe_stability.v1",
        "dimensions": dimensions,
    }
    result["content_hash"] = content_hash(result, domain="linkradius:probe_stability:v1")
    return result


def calibrate_probe_configuration(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_K: Sequence[int] = (4, 8, 16, 32),
    minimum_acceptance: float = 0.5,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    expected_raw_ids: Sequence[str] | None = None,
    expected_configurations: Sequence[tuple[str, float, int, int]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Derive a deterministic frozen-configuration fragment from validation."""

    candidates = tuple(sorted(set(int(value) for value in candidate_K)))
    if len(candidates) < 2 or candidates[0] < 1:
        raise ContractError("probe calibration requires at least two nested K candidates")
    if not 0.0 <= minimum_acceptance <= 1.0:
        raise ContractError("minimum probe acceptance must lie in [0,1]")
    thresholds, derivation = derive_acceptance_thresholds(
        rows,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )
    pairs = reclassify_probe_pairs(rows, thresholds)
    cube_coverage = validate_complete_eligible_cube(
        pairs,
        expected_raw_ids=expected_raw_ids,
        expected_configurations=expected_configurations,
    )
    estimates = _primary_estimates(pairs, candidates)
    if not estimates:
        raise ContractError("validation probes contain no complete primary estimate")

    acceptance_by_h: dict[float, list[bool]] = defaultdict(list)
    for row in pairs:
        acceptance_by_h[float(row["h"])].append(bool(row["accepted"]))
    acceptance = {
        h: sum(values) / len(values) for h, values in sorted(acceptance_by_h.items())
    }
    initially_usable = {
        h for h, value in acceptance.items() if value >= float(minimum_acceptance)
    }
    viable_K: list[int] = []
    for requested_K in candidates:
        relevant = [
            row
            for row in estimates
            if int(row["K"]) == requested_K and float(row["h"]) in initially_usable
        ]
        if (
            len({float(row["h"]) for row in relevant}) >= 2
            and len({int(row["probe_seed"]) for row in relevant}) >= 2
        ):
            viable_K.append(requested_K)
    if not viable_K:
        raise ContractError(
            "validation requires complete primary estimates for at least two radii and seeds"
        )
    selected_K = max(viable_K)
    usable_h = sorted(
        h
        for h in initially_usable
        if any(int(row["K"]) == selected_K and float(row["h"]) == h for row in estimates)
    )
    if len(usable_h) < 2:
        raise ContractError("at least two validation probe radii must remain usable")
    selected_h = min(
        usable_h,
        key=lambda h: (
            -sum(
                int(row["K"]) == selected_K and float(row["h"]) == h
                for row in estimates
            ),
            -acceptance[h],
            h,
        ),
    )
    selected_estimates = [
        row
        for row in estimates
        if float(row["h"]) in usable_h and int(row["K"]) <= selected_K
    ]
    stability = stability_summary(selected_estimates)

    inventory_payload = [
        {
            **{key: value for key, value in row.items() if key != "edge_radius"},
            "edge_radius": (
                float(row["edge_radius"])
                if math.isfinite(float(row["edge_radius"]))
                else "infinity"
            ),
        }
        for row in sorted(
            estimates,
            key=lambda item: (
                str(item["raw_sample_id"]),
                str(item["edge_id"]),
                float(item["h"]),
                int(item["probe_seed"]),
                int(item["K"]),
            ),
        )
    ]
    result = {
        "schema_version": CALIBRATION_VERSION,
        "selection_algorithm": SELECTION_ALGORITHM,
        "minimum_acceptance": float(minimum_acceptance),
        "candidate_K": list(candidates),
        "selected_h": selected_h,
        "selected_K": selected_K,
        "nested_K": [value for value in candidates if value <= selected_K],
        "usable_h": usable_h,
        "probe_seeds": sorted({int(row["probe_seed"]) for row in pairs}),
        "acceptance_thresholds": thresholds,
        "threshold_derivation": derivation,
        "eligible_cube_coverage": cube_coverage,
        "candidate_h_acceptance": {str(h): acceptance[h] for h in sorted(acceptance)},
        "reclassified_pair_hash": content_hash(
            pairs, domain="linkradius:reclassified_probe_pairs:v1"
        ),
        "primary_estimate_count": len(estimates),
        "primary_estimate_inventory_hash": content_hash(
            inventory_payload, domain="linkradius:primary_probe_estimates:v1"
        ),
        "stability": stability,
    }
    result["content_hash"] = content_hash(result, domain="linkradius:probe_calibration:v1")
    return result, pairs


def stability_checks(
    stability: Mapping[str, Any],
    *,
    minimum_rank_correlation: float,
    minimum_binding_agreement: float,
    minimum_comparisons: int = 1,
) -> list[dict[str, Any]]:
    """Return fail-closed gate checks for all predeclared stability axes."""

    checks: list[dict[str, Any]] = []
    dimensions = stability.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise ContractError("probe stability summary is malformed")
    for name in ("radius", "seed", "K"):
        summary = dimensions.get(name)
        if not isinstance(summary, Mapping):
            raise ContractError(f"probe stability is missing the {name} dimension")
        rank = summary.get("median_rank_correlation")
        binding = summary.get("median_binding_agreement")
        finite_count = int(summary.get("finite_rank_comparison_count", 0))
        comparison_count = int(summary.get("comparison_count", 0))
        checks.extend(
            (
                {
                    "name": f"{name}_rank_stability",
                    "passed": finite_count >= int(minimum_comparisons)
                    and rank is not None
                    and float(rank) >= float(minimum_rank_correlation),
                    "median_rank_correlation": rank,
                    "finite_comparisons": finite_count,
                    "minimum": float(minimum_rank_correlation),
                },
                {
                    "name": f"{name}_binding_stability",
                    "passed": comparison_count >= int(minimum_comparisons)
                    and binding is not None
                    and float(binding) >= float(minimum_binding_agreement),
                    "median_binding_agreement": binding,
                    "comparisons": comparison_count,
                    "minimum": float(minimum_binding_agreement),
                },
            )
        )
    return checks


def probe_autograd_agreement(
    pairs: Sequence[Mapping[str, Any]],
    gradient_rows: Sequence[Mapping[str, Any]],
    *,
    selected_h: float,
    selected_K: int,
) -> dict[str, Any]:
    """Join random-probe susceptibility to the exact autograd reference."""

    gradients: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in gradient_rows:
        if row.get("record_type") != "gradient":
            continue
        key = (str(row.get("raw_sample_id") or ""), str(row.get("edge_id") or ""))
        if not all(key) or key in gradients:
            raise ContractError("autograd reference rows have duplicate or invalid sample/edge keys")
        gradients[key] = row
    groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairs:
        if (
            bool(row.get("analysis_eligible", False))
            and float(row.get("h", -1.0)) == float(selected_h)
            and int(row.get("direction_id", -1)) < int(selected_K)
        ):
            groups[
                (
                    str(row["raw_sample_id"]),
                    str(row["edge_id"]),
                    int(row["probe_seed"]),
                )
            ].append(row)
    comparisons: list[dict[str, Any]] = []
    for (raw_id, edge, seed), rows in sorted(groups.items()):
        gradient = gradients.get((raw_id, edge))
        if gradient is None:
            continue
        by_direction = {int(row["direction_id"]): row for row in rows}
        if set(by_direction) != set(range(int(selected_K))):
            continue
        selected = [by_direction[index] for index in range(int(selected_K))]
        if not all(bool(row.get("accepted")) for row in selected):
            continue
        target = str(gradient.get("target_label") or "")
        if not target or any(
            target not in row.get("central_differences", {}) for row in selected
        ):
            raise ContractError("probe/autograd join has an incompatible target competitor")
        q_values = {int(row["q"]) for row in selected}
        subspace_ids = {str(row["subspace_id"]) for row in selected}
        if len(q_values) != 1 or len(subspace_ids) != 1:
            raise ContractError("probe/autograd join mixes subspace definitions")
        if (
            int(gradient.get("q", -1)) not in q_values
            or str(gradient.get("subspace_id") or "") not in subspace_ids
        ):
            raise ContractError("probe and autograd rows use different subspaces")
        finite_difference = gradient.get("finite_difference")
        if not isinstance(finite_difference, Mapping):
            continue
        plus_diagnostics = finite_difference.get("plus_diagnostics")
        minus_diagnostics = finite_difference.get("minus_diagnostics")
        if (
            finite_difference.get("agrees") is not True
            or not isinstance(plus_diagnostics, Mapping)
            or not isinstance(minus_diagnostics, Mapping)
            or plus_diagnostics.get("collapsed") is not False
            or minus_diagnostics.get("collapsed") is not False
            or (_finite_optional(finite_difference.get("realized_separation")) or 0.0)
            <= 0.0
        ):
            continue
        reference = _finite_optional(
            finite_difference.get("autograd_dimensionless_derivative")
        )
        if reference is None:
            continue
        derivatives = [
            _finite_optional(row["central_differences"][target]) for row in selected
        ]
        if any(value is None for value in derivatives):
            continue
        susceptibility = math.sqrt(
            float(next(iter(q_values)))
            * sum(float(value) ** 2 for value in derivatives)
            / int(selected_K)
        )
        reference = abs(reference)
        relative_error = abs(susceptibility - reference) / max(
            susceptibility, reference, 1e-12
        )
        comparisons.append(
            {
                "raw_sample_id": raw_id,
                "edge_id": edge,
                "probe_seed": seed,
                "target_label": target,
                "subspace_id": next(iter(subspace_ids)),
                "q": next(iter(q_values)),
                "probe_susceptibility": susceptibility,
                "autograd_dimensionless_derivative": reference,
                "relative_error": relative_error,
            }
        )
    errors = [float(row["relative_error"]) for row in comparisons]
    usable_gradient_keys: set[tuple[str, str]] = set()
    for key, gradient in gradients.items():
        finite_difference = gradient.get("finite_difference")
        if not isinstance(finite_difference, Mapping):
            continue
        plus = finite_difference.get("plus_diagnostics")
        minus = finite_difference.get("minus_diagnostics")
        if (
            isinstance(plus, Mapping)
            and isinstance(minus, Mapping)
            and plus.get("collapsed") is False
            and minus.get("collapsed") is False
            and (_finite_optional(finite_difference.get("realized_separation")) or 0.0)
            > 0.0
            and _finite_optional(
                finite_difference.get("autograd_dimensionless_derivative")
            )
            is not None
        ):
            usable_gradient_keys.add(key)
    matched_gradient_keys = {
        (str(row["raw_sample_id"]), str(row["edge_id"])) for row in comparisons
    }
    result = {
        "schema_version": "linkradius.probe_autograd_agreement.v1",
        "selected_h": float(selected_h),
        "selected_K": int(selected_K),
        "comparison_count": len(comparisons),
        "total_gradient_rows": len(gradients),
        "usable_finite_difference_gradient_rows": len(usable_gradient_keys),
        "usable_finite_difference_coverage": (
            len(usable_gradient_keys) / len(gradients) if gradients else 0.0
        ),
        "matched_gradient_rows": len(matched_gradient_keys),
        "matched_gradient_coverage": (
            len(matched_gradient_keys) / len(gradients) if gradients else 0.0
        ),
        "median_relative_error": _median(errors),
        "maximum_relative_error": max(errors) if errors else None,
        "comparisons": comparisons,
    }
    result["content_hash"] = content_hash(
        result, domain="linkradius:probe_autograd_agreement:v1"
    )
    return result


def select_causally_useful_edges(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_edges: Sequence[str] | None = None,
    minimum_pairs: int = 1,
    minimum_accuracy_effect: float = 0.0,
    minimum_margin_effect: float = 0.0,
) -> dict[str, Any]:
    """Freeze a validation-only identity-minus-mismatch useful-edge rule."""

    if minimum_pairs < 1:
        raise ContractError("minimum causal pair count must be positive")
    if not all(
        math.isfinite(float(value))
        for value in (minimum_accuracy_effect, minimum_margin_effect)
    ):
        raise ContractError("causal-use thresholds must be finite")
    by_unit: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("record_type") != "sample" or not bool(
            row.get("analysis_eligible", False)
        ):
            continue
        mode = str(row.get("intervention_mode") or "")
        if mode not in {"identity", "mismatch"}:
            continue
        key = (str(row.get("raw_sample_id") or ""), str(row.get("edge_id") or ""))
        if not all(key) or mode in by_unit[key]:
            raise ContractError("causal validation rows contain a duplicate/invalid unit and mode")
        by_unit[key][mode] = row

    by_edge: dict[str, list[dict[str, float]]] = defaultdict(list)
    attempted_by_edge: dict[str, int] = defaultdict(int)
    unavailable_by_edge: dict[str, int] = defaultdict(int)
    for (_, edge), modes in sorted(by_unit.items()):
        if set(modes) != {"identity", "mismatch"}:
            raise ContractError("causal-use calibration requires paired identity and mismatch rows")
        identity, mismatch = modes["identity"], modes["mismatch"]
        attempted_by_edge[edge] += 1
        if bool(mismatch.get("intervention_unavailable", False)):
            unavailable_by_edge[edge] += 1
            continue
        try:
            identity_margin = float(identity["minimum_margin"])
            mismatch_margin = float(mismatch["minimum_margin"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("causal-use rows require finite minimum margins") from exc
        if not math.isfinite(identity_margin) or not math.isfinite(mismatch_margin):
            raise ContractError("causal-use rows require finite minimum margins")
        by_edge[edge].append(
            {
                "accuracy_effect": float(bool(identity.get("scorer_correct")))
                - float(bool(mismatch.get("scorer_correct"))),
                "margin_effect": identity_margin - mismatch_margin,
            }
        )
    observed_edges = set(attempted_by_edge)
    required_edges = set(expected_edges or observed_edges)
    if not required_edges or observed_edges != required_edges:
        raise ContractError("causal-use calibration does not cover the exact expected edges")

    summaries: list[dict[str, Any]] = []
    for edge in sorted(required_edges):
        effects = by_edge[edge]
        accuracy_effect = (
            sum(item["accuracy_effect"] for item in effects) / len(effects)
            if effects
            else None
        )
        margin_effect = (
            sum(item["margin_effect"] for item in effects) / len(effects)
            if effects
            else None
        )
        # Strict improvement on either predeclared signal prevents a no-effect
        # edge from passing merely because the default cutoff is zero.
        useful = len(effects) >= int(minimum_pairs) and (
            float(accuracy_effect) > float(minimum_accuracy_effect)
            or float(margin_effect) > float(minimum_margin_effect)
        )
        summaries.append(
            {
                "edge_id": edge,
                "pair_count": len(effects),
                "attempted_pair_count": attempted_by_edge[edge],
                "unavailable_pair_count": unavailable_by_edge[edge],
                "identity_minus_mismatch_accuracy": accuracy_effect,
                "identity_minus_mismatch_minimum_margin": margin_effect,
                "useful": useful,
            }
        )
    result = {
        "schema_version": "linkradius.causally_useful_edge_rule.v1",
        "partition": "validation",
        "rule": "identity_minus_mismatch_strict_improvement_on_accuracy_or_minimum_margin_v1",
        "minimum_pairs": int(minimum_pairs),
        "minimum_accuracy_effect": float(minimum_accuracy_effect),
        "minimum_margin_effect": float(minimum_margin_effect),
        "expected_edges": sorted(required_edges),
        "edge_summaries": summaries,
        "useful_edges": [row["edge_id"] for row in summaries if row["useful"]],
    }
    result["content_hash"] = content_hash(
        result, domain="linkradius:causally_useful_edge_rule:v1"
    )
    return result


__all__ = [
    "calibrate_probe_configuration",
    "derive_acceptance_thresholds",
    "empirical_quantile",
    "probe_autograd_agreement",
    "reclassify_probe_pairs",
    "select_causally_useful_edges",
    "stability_checks",
    "stability_summary",
]
