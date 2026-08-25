#!/usr/bin/env python3
"""Pure metrics for the held-out LinkRadius failure-boundary experiment.

The functions in this module deliberately do no file I/O.  They accept
CSV-like rows (``Sequence[Mapping[str, Any]]``), validate the fields used by
each metric, and return ordinary dictionaries suitable for authenticated
artifact writers in the experiment runner.

Conventions
-----------
* A larger binary-prediction score means *more likely to fail*.
* A smaller predicted radius means *a smaller failure threshold*.
* Observed threshold intervals are represented by ``threshold_lower`` and
  ``threshold_upper``.  ``None`` lower/upper bounds mean negative/positive
  infinity respectively.  Only disjoint intervals are order-comparable.
* Threshold Spearman uses the first crossed grid budget (the finite upper
  endpoint) and excludes left- and right-censored observations.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .schemas import ContractError


def _finite_float(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ContractError(f"{field} must be finite")
    return parsed


def _optional_finite_float(value: Any, *, field: str) -> float | None:
    if value is None or value == "":
        return None
    return _finite_float(value, field=field)


def _required(row: Mapping[str, Any], field: str) -> Any:
    if field not in row:
        raise ContractError(f"missing required field: {field}")
    return row[field]


def _bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes"}:
            return True
        if normalized in {"0", "false", "no"}:
            return False
    raise ContractError(f"{field} must be boolean")


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        # Ranks are one-indexed; all members of a tie receive their mean rank.
        rank = (start + 1 + stop) / 2.0
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Return tie-aware Spearman correlation, or NaN when undefined."""

    if len(x) != len(y):
        raise ContractError("Spearman arrays differ in length")
    if len(x) < 2:
        return math.nan
    left = [_finite_float(value, field="spearman.x") for value in x]
    right = [_finite_float(value, field="spearman.y") for value in y]
    rx, ry = _average_ranks(left), _average_ranks(right)
    mean_x, mean_y = sum(rx) / len(rx), sum(ry) / len(ry)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in rx)
        * sum((b - mean_y) ** 2 for b in ry)
    )
    return numerator / denominator if denominator else math.nan


def binary_auroc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Return rank AUROC with half credit for score ties."""

    if len(labels) != len(scores):
        raise ContractError("AUROC labels/scores length mismatch")
    parsed_labels = [_bool(value, field="label") for value in labels]
    parsed_scores = [_finite_float(value, field="score") for value in scores]
    positives = sum(parsed_labels)
    negatives = len(parsed_labels) - positives
    if not positives or not negatives:
        return math.nan
    ranks = _average_ranks(parsed_scores)
    positive_rank_sum = sum(
        rank for rank, label in zip(ranks, parsed_labels) if label
    )
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def binary_auprc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Return average precision evaluated at complete tied-score groups.

    Processing a tied group atomically makes the value independent of input
    order.  This is the right-continuous precision-recall step integral (often
    called average precision), not trapezoidal PR AUC.
    """

    if len(labels) != len(scores):
        raise ContractError("AUPRC labels/scores length mismatch")
    parsed_labels = [_bool(value, field="label") for value in labels]
    parsed_scores = [_finite_float(value, field="score") for value in scores]
    positives = sum(parsed_labels)
    if not positives:
        return math.nan

    groups: dict[float, list[bool]] = defaultdict(list)
    for score, label in zip(parsed_scores, parsed_labels):
        groups[score].append(label)

    true_positives = 0
    predicted_positives = 0
    previous_recall = 0.0
    area = 0.0
    for score in sorted(groups, reverse=True):
        group = groups[score]
        true_positives += sum(group)
        predicted_positives += len(group)
        recall = true_positives / positives
        precision = true_positives / predicted_positives
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def _threshold_interval(
    row: Mapping[str, Any],
    *,
    lower_field: str,
    upper_field: str,
) -> tuple[float | None, float | None]:
    lower = _optional_finite_float(
        _required(row, lower_field), field=lower_field
    )
    upper = _optional_finite_float(
        _required(row, upper_field), field=upper_field
    )
    if lower is not None and upper is not None and lower > upper:
        raise ContractError(
            f"invalid threshold interval: {lower_field} exceeds {upper_field}"
        )
    if lower is None and upper is None:
        raise ContractError("threshold interval cannot be unbounded on both sides")
    return lower, upper


def interval_censored_concordance(
    rows: Sequence[Mapping[str, Any]],
    *,
    prediction_field: str = "predicted_radius",
    lower_field: str = "threshold_lower",
    upper_field: str = "threshold_upper",
) -> dict[str, Any]:
    """Compare predicted thresholds using only disjoint observed intervals.

    An interval pair is comparable only when one finite upper endpoint is at
    or below the other's finite lower endpoint.  Equality is ordered because
    empirical threshold intervals are open on the left and closed on the
    right: ``(a,b]`` precedes ``(b,c]``.  Predicted ties receive half credit.
    Lower predicted radii are expected for lower intervals.
    """

    parsed: list[tuple[float, float | None, float | None]] = []
    for row in rows:
        prediction = _finite_float(
            _required(row, prediction_field), field=prediction_field
        )
        lower, upper = _threshold_interval(
            row, lower_field=lower_field, upper_field=upper_field
        )
        parsed.append((prediction, lower, upper))

    comparable = 0
    concordance_credit = 0.0
    for left_index in range(len(parsed)):
        left_prediction, left_lower, left_upper = parsed[left_index]
        for right_prediction, right_lower, right_upper in parsed[left_index + 1 :]:
            observed_relation = 0
            if (
                left_upper is not None
                and right_lower is not None
                and left_upper <= right_lower
            ):
                observed_relation = -1
            elif (
                right_upper is not None
                and left_lower is not None
                and right_upper <= left_lower
            ):
                observed_relation = 1
            if not observed_relation:
                continue

            comparable += 1
            predicted_relation = (
                -1
                if left_prediction < right_prediction
                else (1 if left_prediction > right_prediction else 0)
            )
            if predicted_relation == observed_relation:
                concordance_credit += 1.0
            elif predicted_relation == 0:
                concordance_credit += 0.5

    return {
        "n": len(parsed),
        "comparable_pairs": comparable,
        "concordance": (
            concordance_credit / comparable if comparable else math.nan
        ),
    }


def threshold_spearman(
    rows: Sequence[Mapping[str, Any]],
    *,
    prediction_field: str = "predicted_radius",
    lower_field: str = "threshold_lower",
    upper_field: str = "threshold_upper",
) -> dict[str, Any]:
    """Spearman against the first crossed grid budget for crossed rows only.

    Exact and interval-crossed rows have two finite endpoints.  Left-censored
    and right-censored rows are excluded; no within-interval interpolation is
    performed.
    """

    predictions: list[float] = []
    observed: list[float] = []
    for row in rows:
        lower, upper = _threshold_interval(
            row, lower_field=lower_field, upper_field=upper_field
        )
        if lower is None or upper is None:
            continue
        predictions.append(
            _finite_float(
                _required(row, prediction_field), field=prediction_field
            )
        )
        observed.append(upper)
    return {
        "n": len(predictions),
        "spearman": spearman(predictions, observed),
    }


def site_ranking_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_fields: Sequence[str] = ("raw_id", "attack_family"),
    site_field: str = "edge_id",
    prediction_field: str = "predicted_radius",
    lower_field: str = "threshold_lower",
    upper_field: str = "threshold_upper",
) -> dict[str, Any]:
    """Return tie-aware vulnerable-site top-1 accuracy and pair concordance.

    Within each example/family group, the observed vulnerable set contains all
    sites tied at the smallest finite first-crossing budget.  The predicted set
    contains all sites tied at the smallest radius.  Top-1 credit is
    ``|predicted ∩ observed| / |predicted|``.  Groups without a crossed site
    are excluded.  ``site_kendall`` is a concordance-style ordering score over
    disjoint observed site intervals, with half credit for predicted ties.
    """

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(_required(row, field)) for field in group_fields)
        lower, upper = _threshold_interval(
            row, lower_field=lower_field, upper_field=upper_field
        )
        grouped[key].append(
            {
                "site": str(_required(row, site_field)),
                "prediction": _finite_float(
                    _required(row, prediction_field), field=prediction_field
                ),
                "lower": lower,
                "upper": upper,
            }
        )

    top1_credits: list[float] = []
    pair_credit = 0.0
    comparable_pairs = 0
    for values in grouped.values():
        crossed = [item for item in values if item["lower"] is not None and item["upper"] is not None]
        if crossed:
            minimum_observed = min(item["upper"] for item in crossed)
            observed_sites = {
                item["site"] for item in crossed if item["upper"] == minimum_observed
            }
            minimum_prediction = min(item["prediction"] for item in values)
            predicted_sites = {
                item["site"]
                for item in values
                if item["prediction"] == minimum_prediction
            }
            top1_credits.append(
                len(predicted_sites & observed_sites) / len(predicted_sites)
            )

        for left_index in range(len(values)):
            left = values[left_index]
            for right in values[left_index + 1 :]:
                observed_relation = 0
                if (
                    left["upper"] is not None
                    and right["lower"] is not None
                    and left["upper"] <= right["lower"]
                ):
                    observed_relation = -1
                elif (
                    right["upper"] is not None
                    and left["lower"] is not None
                    and right["upper"] <= left["lower"]
                ):
                    observed_relation = 1
                if not observed_relation:
                    continue
                comparable_pairs += 1
                predicted_relation = (
                    -1
                    if left["prediction"] < right["prediction"]
                    else (
                        1
                        if left["prediction"] > right["prediction"]
                        else 0
                    )
                )
                if predicted_relation == observed_relation:
                    pair_credit += 1.0
                elif predicted_relation == 0:
                    pair_credit += 0.5

    return {
        "groups": len(grouped),
        "top1_groups": len(top1_credits),
        "top1_accuracy": (
            sum(top1_credits) / len(top1_credits) if top1_credits else math.nan
        ),
        "comparable_site_pairs": comparable_pairs,
        "site_kendall": (
            pair_credit / comparable_pairs if comparable_pairs else math.nan
        ),
    }


def family_budget_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    family_field: str = "attack_family",
    budget_field: str = "requested_epsilon",
    label_field: str = "flipped",
    radius_field: str = "edge_radius",
    margin_field: str = "clean_margin",
    susceptibility_field: str = "susceptibility",
) -> list[dict[str, Any]]:
    """Evaluate LinkRadius and component baselines in each family/budget stratum.

    Predictor orientations are fixed here: smaller radius and clean margin mean
    greater risk, while larger susceptibility means greater risk.
    """

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family = str(_required(row, family_field))
        if not family:
            raise ContractError(f"{family_field} must be non-empty")
        budget = _finite_float(_required(row, budget_field), field=budget_field)
        try:
            radius = float(_required(row, radius_field))
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{radius_field} must be numeric") from exc
        if math.isnan(radius):
            raise ContractError(f"{radius_field} must not be NaN")
        margin = _finite_float(_required(row, margin_field), field=margin_field)
        susceptibility = _finite_float(
            _required(row, susceptibility_field), field=susceptibility_field
        )
        if radius < 0 or susceptibility < 0:
            raise ContractError("radius and susceptibility must be non-negative")
        linkradius_risk = (
            _finite_float(row["linkradius_score"], field="linkradius_score")
            if "linkradius_score" in row
            else (-radius if math.isfinite(radius) else -1e300)
        )
        grouped[(family, budget)].append(
            {
                "label": _bool(_required(row, label_field), field=label_field),
                "linkradius": linkradius_risk,
                "margin_only": -margin,
                "susceptibility_only": susceptibility,
            }
        )

    output: list[dict[str, Any]] = []
    for (family, budget), values in sorted(grouped.items()):
        labels = [item["label"] for item in values]
        positives = sum(labels)
        for predictor in ("linkradius", "margin_only", "susceptibility_only"):
            scores = [item[predictor] for item in values]
            output.append(
                {
                    "attack_family": family,
                    "requested_epsilon": budget,
                    "predictor": predictor,
                    "n": len(values),
                    "positives": positives,
                    "auroc": binary_auroc(labels, scores),
                    "auprc": binary_auprc(labels, scores),
                }
            )
    return output


def calibration_bins(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_field: str = "failure_score",
    label_field: str = "flipped",
    num_bins: int = 10,
) -> list[dict[str, Any]]:
    """Build deterministic equal-frequency-style bins without splitting ties."""

    if num_bins <= 0:
        raise ContractError("num_bins must be positive")
    groups: dict[float, list[bool]] = defaultdict(list)
    for row in rows:
        score = _finite_float(_required(row, score_field), field=score_field)
        label = _bool(_required(row, label_field), field=label_field)
        groups[score].append(label)
    if not groups:
        return []

    total = sum(len(labels) for labels in groups.values())
    bins: dict[int, list[tuple[float, list[bool]]]] = defaultdict(list)
    offset = 0
    for score in sorted(groups):
        labels = groups[score]
        midpoint = offset + (len(labels) - 1) / 2.0
        bin_index = min(num_bins - 1, int(midpoint * num_bins / total))
        bins[bin_index].append((score, labels))
        offset += len(labels)

    output: list[dict[str, Any]] = []
    for bin_index in sorted(bins):
        values = bins[bin_index]
        count = sum(len(labels) for _, labels in values)
        score_sum = sum(score * len(labels) for score, labels in values)
        positives = sum(sum(labels) for _, labels in values)
        output.append(
            {
                "bin": bin_index,
                "n": count,
                "score_min": values[0][0],
                "score_max": values[-1][0],
                "score_mean": score_sum / count,
                "flip_rate": positives / count,
            }
        )
    return output


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    statistic: Callable[[Sequence[Mapping[str, Any]]], float],
    *,
    cluster_field: str = "raw_id",
    repetitions: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Return a deterministic percentile CI from a raw-example bootstrap.

    Entire clusters are sampled with replacement.  A cluster selected multiple
    times contributes duplicate rows under distinct bootstrap-unit IDs,
    preserving within-example dependence across sites, budgets, and attack
    families without merging repeated draws in grouped statistics.
    """

    if repetitions <= 0:
        raise ContractError("bootstrap repetitions must be positive")
    if not 0.0 < confidence < 1.0:
        raise ContractError("bootstrap confidence must lie in (0, 1)")
    clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        cluster = str(_required(row, cluster_field))
        if not cluster:
            raise ContractError(f"{cluster_field} must be non-empty")
        clusters[cluster].append(row)
    if not clusters:
        raise ContractError("cluster bootstrap requires rows")

    estimate = _finite_float(statistic(rows), field="bootstrap estimate")
    cluster_ids = sorted(clusters)
    generator = random.Random(seed)
    replicates: list[float] = []
    for _ in range(repetitions):
        sampled: list[Mapping[str, Any]] = []
        for draw_index in range(len(cluster_ids)):
            cluster = cluster_ids[generator.randrange(len(cluster_ids))]
            # Give repeated draws of one raw example distinct bootstrap-unit
            # identities.  Statistics that regroup sites by raw ID must treat
            # two sampled copies as two clusters rather than merging them and
            # inventing within-example cross-pairs.
            for row in clusters[cluster]:
                copied = dict(row)
                copied[cluster_field] = f"{cluster}#bootstrap-{draw_index}"
                sampled.append(copied)
        value = float(statistic(sampled))
        if math.isfinite(value):
            replicates.append(value)
    if not replicates:
        raise ContractError("bootstrap statistic was non-finite in every replicate")

    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": estimate,
        "ci_lower": _percentile(replicates, alpha),
        "ci_upper": _percentile(replicates, 1.0 - alpha),
        "confidence": confidence,
        "cluster_count": len(cluster_ids),
        "repetitions": repetitions,
        "valid_repetitions": len(replicates),
        "seed": seed,
    }


def mean_over_probe_seeds(
    rows: Sequence[Mapping[str, Any]],
    statistic: Callable[[Sequence[Mapping[str, Any]]], float],
    *,
    expected_seeds: Sequence[int],
    seed_field: str = "probe_seed",
) -> float:
    """Average a statistic over frozen probe-seed realizations.

    The probe directions are Monte Carlo realizations of one predictor, not
    independent examples.  Computing the statistic separately within each
    seed and then taking an equal-weight mean prevents duplicated attack labels
    from inflating the effective sample size.  In a raw-ID cluster bootstrap,
    every sampled example still carries its complete set of seed realizations.
    """

    seeds = tuple(int(value) for value in expected_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ContractError("expected probe seeds must be unique and non-empty")
    groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = _required(row, seed_field)
        if isinstance(value, bool):
            raise ContractError(f"{seed_field} must be an integer")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{seed_field} must be an integer") from exc
        if parsed not in seeds:
            raise ContractError(f"unexpected frozen probe seed: {parsed}")
        groups[parsed].append(row)
    missing = sorted(set(seeds) - set(groups))
    if missing:
        # A raw-ID bootstrap replicate can lose all usable rows for one seed
        # when cast-quality exclusions are seed-specific.  Mark that replicate
        # non-estimable; ``cluster_bootstrap`` will exclude it transparently.
        return math.nan

    values = [float(statistic(groups[seed])) for seed in seeds]
    if any(not math.isfinite(value) for value in values):
        return math.nan
    return sum(values) / len(values)
