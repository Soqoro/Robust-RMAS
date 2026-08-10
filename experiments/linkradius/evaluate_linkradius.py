#!/usr/bin/env python3
"""Held-out LinkRadius diagnostic metrics and component ablations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io_utils import atomic_write_csv
from .schemas import ContractError


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        stop = pos + 1
        while stop < len(order) and values[order[stop]] == values[order[pos]]:
            stop += 1
        rank = (pos + 1 + stop) / 2.0
        for idx in order[pos:stop]:
            ranks[idx] = rank
        pos = stop
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return math.nan
    rx, ry = _average_ranks([float(value) for value in x]), _average_ranks([float(value) for value in y])
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return numerator / denominator if denominator else math.nan


def binary_auroc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    if len(labels) != len(scores):
        raise ContractError("AUROC labels/scores length mismatch")
    positives = sum(bool(value) for value in labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return math.nan
    ranks = _average_ranks([float(value) for value in scores])
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def binary_auprc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    if len(labels) != len(scores):
        raise ContractError("AUPRC labels/scores length mismatch")
    positives = sum(bool(value) for value in labels)
    if not positives:
        return math.nan
    ordered = sorted(zip(scores, labels), key=lambda item: -float(item[0]))
    true_positives = 0
    previous_recall = 0.0
    area = 0.0
    for rank, (_, label) in enumerate(ordered, start=1):
        if label:
            true_positives += 1
            recall = true_positives / positives
            precision = true_positives / rank
            area += (recall - previous_recall) * precision
            previous_recall = recall
    return area


def censored_concordance(
    predictions: Sequence[float], thresholds: Sequence[float | None], censor_limits: Sequence[float | None]
) -> float:
    if not (len(predictions) == len(thresholds) == len(censor_limits)):
        raise ContractError("censored concordance arrays differ in length")
    concordant = 0.0
    comparable = 0
    for left in range(len(predictions)):
        for right in range(left + 1, len(predictions)):
            lt, rt = thresholds[left], thresholds[right]
            relation = None
            if lt is not None and rt is not None and lt != rt:
                relation = -1 if lt < rt else 1
            elif lt is not None and rt is None and censor_limits[right] is not None and lt <= censor_limits[right]:
                relation = -1
            elif rt is not None and lt is None and censor_limits[left] is not None and rt <= censor_limits[left]:
                relation = 1
            if relation is None:
                continue
            comparable += 1
            predicted_relation = -1 if predictions[left] < predictions[right] else (1 if predictions[left] > predictions[right] else 0)
            concordant += 1.0 if predicted_relation == relation else (0.5 if predicted_relation == 0 else 0.0)
    return concordant / comparable if comparable else math.nan


def competitor_baselines(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ContractError("competitor baselines require rows")
    parsed = []
    for row in rows:
        margin = float(row["clean_margin"])
        susceptibility = float(row["susceptibility"])
        radius = math.inf if susceptibility == 0 else margin / susceptibility
        parsed.append((str(row["competitor"]), margin, susceptibility, radius))
    margin_choice = min(parsed, key=lambda item: (item[1], item[0]))
    susceptibility_choice = max(parsed, key=lambda item: (item[2], tuple(-ord(c) for c in item[0])))
    radius_choice = min(parsed, key=lambda item: (item[3], item[0]))
    return {
        "margin_only": margin_choice[1],
        "margin_only_competitor": margin_choice[0],
        "susceptibility_only": susceptibility_choice[2],
        "susceptibility_only_competitor": susceptibility_choice[0],
        "linkradius": radius_choice[3],
        "linkradius_binding_competitor": radius_choice[0],
    }


def calibration_bins(
    scores: Sequence[float], labels: Sequence[bool], *, num_bins: int = 10
) -> list[dict[str, Any]]:
    if len(scores) != len(labels) or num_bins <= 0:
        raise ContractError("invalid calibration-bin arguments")
    ordered = sorted(zip(scores, labels), key=lambda item: float(item[0]))
    output = []
    for bin_idx in range(num_bins):
        start = len(ordered) * bin_idx // num_bins
        stop = len(ordered) * (bin_idx + 1) // num_bins
        values = ordered[start:stop]
        if not values:
            continue
        output.append(
            {
                "bin": bin_idx,
                "n": len(values),
                "score_min": float(values[0][0]),
                "score_max": float(values[-1][0]),
                "score_mean": sum(float(value) for value, _ in values) / len(values),
                "flip_rate": sum(bool(label) for _, label in values) / len(values),
            }
        )
    return output


def evaluate_flip_prediction(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    labels = [bool(row["flipped"]) for row in rows]
    metrics = []
    for name in ("linkradius_score", "margin_score", "susceptibility_score"):
        values = [float(row[name]) for row in rows]
        metrics.append(
            {
                "predictor": name,
                "n": len(rows),
                "auroc": binary_auroc(labels, values),
                "auprc": binary_auprc(labels, values),
            }
        )
    return metrics


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, help="joined held-out evaluation CSV")
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--calibration-output", required=True)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    rows = _read_csv(args.rows)
    parsed = []
    for row in rows:
        epsilon, radius = float(row["requested_epsilon"]), float(row["edge_radius"])
        parsed.append(
            {
                **row,
                "flipped": row["flipped"].lower() in {"1", "true", "yes"},
                "linkradius_score": math.inf if radius == 0 else epsilon / radius,
                "margin_score": -float(row["minimum_clean_margin"]),
                "susceptibility_score": float(row["maximum_susceptibility"]),
            }
        )
    metrics = evaluate_flip_prediction(parsed)
    bins = calibration_bins(
        [float(row["linkradius_score"]) for row in parsed],
        [bool(row["flipped"]) for row in parsed],
        num_bins=args.calibration_bins,
    )
    atomic_write_csv(args.metrics_output, metrics, overwrite=args.overwrite)
    atomic_write_csv(args.calibration_output, bins, overwrite=args.overwrite)
    print(json.dumps({"metrics": len(metrics), "calibration_bins": len(bins)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
