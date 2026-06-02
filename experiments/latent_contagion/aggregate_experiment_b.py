#!/usr/bin/env python3
"""Aggregate latent-contagion Experiment B JSONL logs.

This script intentionally computes all metrics from per-sample JSONL rows.
Summary rows are used only for an accuracy sanity check.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "aggregate_experiment_b.py requires numpy and pandas. "
        "Install those packages, and install matplotlib as well if --make_plots true."
    ) from exc


plt = None


CONDITION_COLUMNS = [
    "dataset",
    "style",
    "method",
    "mas_shape",
    "lc_mode",
    "site",
    "R",
    "lc_round",
    "seed",
    "eps",
]

BASELINE_COLUMNS = [
    "dataset",
    "style",
    "method",
    "mas_shape",
    "lc_mode",
    "site",
    "R",
    "lc_round",
    "seed",
]

SITELESS_BASELINE_COLUMNS = [
    "dataset",
    "style",
    "method",
    "mas_shape",
    "lc_mode",
    "R",
    "lc_round",
    "seed",
]

PER_CONDITION_COLUMNS = CONDITION_COLUMNS + [
    "n_total",
    "clean_n_total",
    "clean_correct_n",
    "clean_accuracy",
    "perturbed_accuracy",
    "delta_accuracy",
    "asrcc",
    "invalid_rate",
    "clean_to_wrong_count",
    "clean_to_invalid_count",
    "clean_to_invalid_rate",
]

EPSILON50_COLUMNS = BASELINE_COLUMNS + [
    "epsilon50",
    "epsilon50_status",
    "max_eps",
    "max_asrcc",
    "clean_accuracy",
    "clean_correct_n",
]

DISAGREEMENT_COLUMNS = SITELESS_BASELINE_COLUMNS + [
    "sample_id",
    "site_a",
    "site_b",
    "correct_a",
    "correct_b",
    "final_answer_a",
    "final_answer_b",
]

FILENAME_TOKEN_RE = re.compile(r"(?P<key>site|eps|epsilon|R|rounds|seed|lc_round)=(?P<value>[^_]+)")


def parse_bool(value: Any) -> bool:
    """Coerce common serialized boolean values to bool."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value) and float(value) != 0.0)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "f", "no", "n", "off", "", "none", "null", "nan"}:
            return False
    return bool(value)


def is_summary_record(record: Mapping[str, Any]) -> bool:
    return str(record.get("type", "")).lower() == "summary"


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def is_invalid_record(record: Mapping[str, Any]) -> bool:
    """Conservative invalid-output rule for Math500/freeform tasks."""
    return (
        _is_empty_value(record.get("raw_final_output"))
        or _is_empty_value(record.get("final_answer"))
        or _is_empty_value(record.get("pred_answer_parsed"))
    )


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _to_int(value: Any) -> Optional[int]:
    number = _to_float(value)
    if number is None:
        return None
    return int(number)


def _clean_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _metadata_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if not _is_empty_value(value):
            return value
    return None


def _key_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def _row_key(row: Mapping[str, Any], columns: Sequence[str]) -> Tuple[Any, ...]:
    return tuple(_key_value(row[col]) for col in columns)


def _frame_key(frame: pd.DataFrame, columns: Sequence[str]) -> Tuple[Any, ...]:
    row = frame.iloc[0]
    return tuple(_key_value(row[col]) for col in columns)


def _safe_mean_bool(values: pd.Series) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values.astype(bool).to_numpy()))


def _is_zero_eps(value: Any) -> bool:
    number = _to_float(value)
    return bool(number is not None and np.isclose(number, 0.0, rtol=0.0, atol=1e-12))


def _finite_positive(value: Any) -> bool:
    number = _to_float(value)
    return bool(number is not None and number > 0.0)


def parse_metadata_from_filename(path: Path) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for match in FILENAME_TOKEN_RE.finditer(path.stem):
        key = match.group("key")
        value = match.group("value")
        if key == "site":
            metadata["site"] = value
        elif key in {"eps", "epsilon"}:
            metadata["eps"] = _to_float(value)
        elif key in {"R", "rounds"}:
            metadata["R"] = _to_int(value)
        elif key == "seed":
            metadata["seed"] = _to_int(value)
        elif key == "lc_round":
            metadata["lc_round"] = _to_int(value)
    return metadata


def load_jsonl_file(
    path: Path,
    warnings: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    samples: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    if warnings is not None:
                        warnings.append(f"{path}:{line_number}: invalid JSON skipped ({exc})")
                    continue
                if not isinstance(record, dict):
                    if warnings is not None:
                        warnings.append(f"{path}:{line_number}: non-object JSONL row skipped")
                    continue
                record["__source_file"] = str(path)
                record["__line_number"] = line_number
                if is_summary_record(record):
                    summaries.append(record)
                else:
                    samples.append(record)
    except OSError as exc:
        if warnings is not None:
            warnings.append(f"{path}: could not read file ({exc})")
    return samples, summaries


def _normalize_sample_record(
    record: Mapping[str, Any],
    filename_metadata: Mapping[str, Any],
    dataset_default: str,
) -> Dict[str, Any]:
    sample_id = record.get("sample_id")
    if _is_empty_value(sample_id):
        sample_idx = record.get("sample_idx")
        sample_id = f"sample_idx={sample_idx}" if not _is_empty_value(sample_idx) else (
            f"{Path(str(record.get('__source_file', 'unknown'))).name}:"
            f"{record.get('__line_number', 0)}"
        )

    correctness_value = _first_nonempty(record.get("is_correct"), record.get("correct"))
    site = _first_nonempty(record.get("lc_site"), filename_metadata.get("site"))
    eps = _first_nonempty(record.get("lc_epsilon"), filename_metadata.get("eps"))
    recursion_rounds = _first_nonempty(record.get("recursion_rounds"), filename_metadata.get("R"))
    lc_round = _first_nonempty(record.get("lc_round"), filename_metadata.get("lc_round"), 0)
    seed = _first_nonempty(record.get("lc_seed"), filename_metadata.get("seed"))
    dataset = _first_nonempty(record.get("dataset"), dataset_default)

    return {
        "dataset": _metadata_text(dataset),
        "style": _metadata_text(record.get("style")),
        "method": _metadata_text(record.get("method")),
        "mas_shape": _metadata_text(record.get("mas_shape")),
        "lc_mode": _metadata_text(record.get("lc_mode")),
        "site": _metadata_text(site),
        "R": _to_int(recursion_rounds),
        "lc_round": _to_int(lc_round),
        "seed": _to_int(seed),
        "eps": _to_float(eps),
        "sample_id": str(sample_id),
        "correct_bool": parse_bool(correctness_value),
        "invalid_bool": is_invalid_record(record),
        "final_answer": _clean_text_value(record.get("final_answer")),
        "source_file": str(record.get("__source_file", "")),
        "line_number": int(record.get("__line_number", 0) or 0),
    }


def _check_summary_accuracy(
    path: Path,
    samples: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    warnings: List[str],
) -> None:
    if not samples:
        return
    computed_accuracy = float(
        np.mean([parse_bool(sample.get("is_correct", sample.get("correct"))) for sample in samples])
    )
    for summary in summaries:
        if "accuracy" not in summary:
            continue
        summary_accuracy = _to_float(summary.get("accuracy"))
        if summary_accuracy is None:
            continue
        summary_fraction = summary_accuracy / 100.0
        if abs(summary_fraction - computed_accuracy) > 1e-6:
            warnings.append(
                f"{path}: summary accuracy {summary_accuracy}%, as fraction "
                f"{summary_fraction:.12g}, differs from per-sample accuracy "
                f"{computed_accuracy:.12g}"
            )


def build_condition_dataframe(
    jsonl_files: Sequence[Path],
    dataset_default: str,
    warnings: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(jsonl_files):
        filename_metadata = parse_metadata_from_filename(path)
        samples, summaries = load_jsonl_file(path, warnings=warnings)
        _check_summary_accuracy(path, samples, summaries, warnings)
        for sample in samples:
            rows.append(_normalize_sample_record(sample, filename_metadata, dataset_default))

    columns = CONDITION_COLUMNS + [
        "sample_id",
        "correct_bool",
        "invalid_bool",
        "final_answer",
        "source_file",
        "line_number",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    for column in columns:
        if column not in df.columns:
            df[column] = None

    df = df.sort_values(["source_file", "line_number"], kind="mergesort").reset_index(drop=True)
    duplicate_mask = df.duplicated(CONDITION_COLUMNS + ["sample_id"], keep=False)
    if duplicate_mask.any():
        duplicate_pairs = df.loc[duplicate_mask, CONDITION_COLUMNS + ["sample_id"]].drop_duplicates()
        warnings.append(
            f"found {len(duplicate_pairs)} duplicate condition/sample_id pairs; "
            "keeping the first row by file path and line number"
        )
        df = df.drop_duplicates(CONDITION_COLUMNS + ["sample_id"], keep="first").reset_index(drop=True)

    return df[columns]


def _sample_ids(frame: pd.DataFrame) -> set:
    return set(frame["sample_id"].astype(str).tolist())


def _select_clean_baseline(
    condition: Mapping[str, Any],
    perturbed: pd.DataFrame,
    clean_by_key: Mapping[Tuple[Any, ...], pd.DataFrame],
    compatible_clean: Mapping[Tuple[Any, ...], List[Tuple[str, pd.DataFrame]]],
    warnings: List[str],
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    baseline_key = tuple(_key_value(condition[col]) for col in BASELINE_COLUMNS)
    exact = clean_by_key.get(baseline_key)
    if exact is not None:
        return exact, None

    siteless_key = tuple(_key_value(condition[col]) for col in SITELESS_BASELINE_COLUMNS)
    candidates = compatible_clean.get(siteless_key, [])
    perturbed_ids = _sample_ids(perturbed)
    matching: List[Tuple[str, pd.DataFrame]] = []
    for site, candidate in candidates:
        if site == condition.get("site"):
            continue
        if _sample_ids(candidate) == perturbed_ids:
            matching.append((site, candidate))

    if matching:
        fallback_site, fallback = sorted(matching, key=lambda item: item[0])[0]
        warnings.append(
            "using eps=0 baseline from site="
            f"{fallback_site} for site={condition.get('site')} "
            f"R={condition.get('R')} lc_round={condition.get('lc_round')} "
            f"seed={condition.get('seed')} eps={condition.get('eps')}"
        )
        return fallback, fallback_site

    warnings.append(
        "clean eps=0 baseline missing for "
        f"dataset={condition.get('dataset')} style={condition.get('style')} "
        f"method={condition.get('method')} mas_shape={condition.get('mas_shape')} "
        f"lc_mode={condition.get('lc_mode')} site={condition.get('site')} "
        f"R={condition.get('R')} lc_round={condition.get('lc_round')} "
        f"seed={condition.get('seed')} eps={condition.get('eps')}"
    )
    return None, None


def compute_per_condition_metrics(df: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=PER_CONDITION_COLUMNS)

    clean_df = df[df["eps"].apply(_is_zero_eps)].copy()
    clean_by_key: Dict[Tuple[Any, ...], pd.DataFrame] = {}
    compatible_clean: Dict[Tuple[Any, ...], List[Tuple[str, pd.DataFrame]]] = defaultdict(list)

    for _, clean_group in clean_df.groupby(BASELINE_COLUMNS, dropna=False, sort=True):
        clean_group = clean_group.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
        key = _frame_key(clean_group, BASELINE_COLUMNS)
        clean_by_key[key] = clean_group
        siteless_key = _frame_key(clean_group, SITELESS_BASELINE_COLUMNS)
        compatible_clean[siteless_key].append((str(clean_group.iloc[0]["site"]), clean_group))

    rows: List[Dict[str, Any]] = []
    for _, group in df.groupby(CONDITION_COLUMNS, dropna=False, sort=True):
        group = group.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
        condition = {column: _key_value(group.iloc[0][column]) for column in CONDITION_COLUMNS}

        perturbed_accuracy = _safe_mean_bool(group["correct_bool"])
        invalid_rate = _safe_mean_bool(group["invalid_bool"])
        n_total = int(len(group))

        clean = None
        if not clean_df.empty:
            clean, _ = _select_clean_baseline(condition, group, clean_by_key, compatible_clean, warnings)
        else:
            warnings.append(
                "clean eps=0 baseline missing for "
                f"dataset={condition.get('dataset')} style={condition.get('style')} "
                f"method={condition.get('method')} mas_shape={condition.get('mas_shape')} "
                f"lc_mode={condition.get('lc_mode')} site={condition.get('site')} "
                f"R={condition.get('R')} lc_round={condition.get('lc_round')} "
                f"seed={condition.get('seed')} eps={condition.get('eps')}"
            )

        clean_n_total = 0
        clean_correct_n = 0
        clean_accuracy = float("nan")
        clean_to_wrong_count = 0
        clean_to_invalid_count = 0
        asrcc = float("nan")
        clean_to_invalid_rate = float("nan")

        if clean is not None:
            clean_n_total = int(len(clean))
            clean_accuracy = _safe_mean_bool(clean["correct_bool"])

            clean_ids = _sample_ids(clean)
            perturbed_ids = _sample_ids(group)
            if clean_ids != perturbed_ids:
                warnings.append(
                    "perturbed sample_ids do not match clean sample_ids for "
                    f"dataset={condition.get('dataset')} site={condition.get('site')} "
                    f"R={condition.get('R')} lc_round={condition.get('lc_round')} "
                    f"seed={condition.get('seed')} eps={condition.get('eps')} "
                    f"(clean={len(clean_ids)}, perturbed={len(perturbed_ids)}, "
                    f"common={len(clean_ids & perturbed_ids)})"
                )

            common_ids = sorted(clean_ids & perturbed_ids)
            clean_indexed = clean.set_index("sample_id", drop=False)
            perturbed_indexed = group.set_index("sample_id", drop=False)
            if common_ids:
                aligned_clean = clean_indexed.loc[common_ids]
                aligned_perturbed = perturbed_indexed.loc[common_ids]
                dcc_mask = aligned_clean["correct_bool"].astype(bool).to_numpy()
                clean_correct_n = int(np.sum(dcc_mask))
                if clean_correct_n > 0:
                    perturbed_correct = aligned_perturbed["correct_bool"].astype(bool).to_numpy()
                    perturbed_invalid = aligned_perturbed["invalid_bool"].astype(bool).to_numpy()
                    clean_to_wrong_count = int(np.sum(dcc_mask & ~perturbed_correct))
                    clean_to_invalid_count = int(np.sum(dcc_mask & perturbed_invalid))
                    eps_value = _to_float(condition.get("eps"))
                    if eps_value is not None and np.isclose(eps_value, 0.0, rtol=0.0, atol=1e-12):
                        asrcc = 0.0
                        clean_to_wrong_count = 0
                    else:
                        asrcc = clean_to_wrong_count / clean_correct_n
                    clean_to_invalid_rate = clean_to_invalid_count / clean_correct_n
                else:
                    warnings.append(
                        "clean-correct set is empty for "
                        f"dataset={condition.get('dataset')} site={condition.get('site')} "
                        f"R={condition.get('R')} lc_round={condition.get('lc_round')} "
                        f"seed={condition.get('seed')} eps={condition.get('eps')}"
                    )

        delta_accuracy = (
            perturbed_accuracy - clean_accuracy if math.isfinite(clean_accuracy) else float("nan")
        )
        row = {
            **condition,
            "n_total": n_total,
            "clean_n_total": clean_n_total,
            "clean_correct_n": clean_correct_n,
            "clean_accuracy": clean_accuracy,
            "perturbed_accuracy": perturbed_accuracy,
            "delta_accuracy": delta_accuracy,
            "asrcc": asrcc,
            "invalid_rate": invalid_rate,
            "clean_to_wrong_count": clean_to_wrong_count,
            "clean_to_invalid_count": clean_to_invalid_count,
            "clean_to_invalid_rate": clean_to_invalid_rate,
        }
        rows.append(row)

    return pd.DataFrame(rows, columns=PER_CONDITION_COLUMNS).sort_values(
        CONDITION_COLUMNS, kind="mergesort"
    )


def compute_clean_disagreements(df: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=DISAGREEMENT_COLUMNS)

    clean_df = df[df["eps"].apply(_is_zero_eps)].copy()
    if clean_df.empty:
        return pd.DataFrame(columns=DISAGREEMENT_COLUMNS)

    rows: List[Dict[str, Any]] = []
    grouping_columns = SITELESS_BASELINE_COLUMNS + ["sample_id"]
    for _, group in clean_df.groupby(grouping_columns, dropna=False, sort=True):
        by_site = []
        for site, site_group in group.groupby("site", dropna=False, sort=True):
            row = site_group.sort_values(["source_file", "line_number"], kind="mergesort").iloc[0]
            by_site.append((str(site), row))
        if len(by_site) < 2:
            continue
        for (site_a, row_a), (site_b, row_b) in combinations(by_site, 2):
            correct_a = bool(row_a["correct_bool"])
            correct_b = bool(row_b["correct_bool"])
            final_answer_a = _clean_text_value(row_a["final_answer"])
            final_answer_b = _clean_text_value(row_b["final_answer"])
            if correct_a == correct_b and final_answer_a == final_answer_b:
                continue
            base = {column: _key_value(row_a[column]) for column in SITELESS_BASELINE_COLUMNS}
            rows.append(
                {
                    **base,
                    "sample_id": str(row_a["sample_id"]),
                    "site_a": site_a,
                    "site_b": site_b,
                    "correct_a": correct_a,
                    "correct_b": correct_b,
                    "final_answer_a": final_answer_a,
                    "final_answer_b": final_answer_b,
                }
            )

    if rows:
        warnings.append(
            f"found {len(rows)} eps=0 baseline disagreements across sites; "
            "see clean_disagreements CSV"
        )

    return pd.DataFrame(rows, columns=DISAGREEMENT_COLUMNS)


def compute_epsilon50(per_condition: pd.DataFrame) -> pd.DataFrame:
    if per_condition.empty:
        return pd.DataFrame(columns=EPSILON50_COLUMNS)

    rows: List[Dict[str, Any]] = []
    for _, group in per_condition.groupby(BASELINE_COLUMNS, dropna=False, sort=True):
        group = group.sort_values("eps", kind="mergesort").reset_index(drop=True)
        base = {column: _key_value(group.iloc[0][column]) for column in BASELINE_COLUMNS}
        clean_accuracy = float(group["clean_accuracy"].dropna().iloc[0]) if group["clean_accuracy"].notna().any() else float("nan")
        clean_correct_n = int(group["clean_correct_n"].dropna().iloc[0]) if group["clean_correct_n"].notna().any() else 0

        positive = group[group["eps"].apply(_finite_positive)].copy()
        positive = positive[np.isfinite(positive["asrcc"].astype(float))]
        if positive.empty:
            rows.append(
                {
                    **base,
                    "epsilon50": float("nan"),
                    "epsilon50_status": "insufficient_points",
                    "max_eps": float("nan"),
                    "max_asrcc": float("nan"),
                    "clean_accuracy": clean_accuracy,
                    "clean_correct_n": clean_correct_n,
                }
            )
            continue

        positive = positive.sort_values("eps", kind="mergesort")
        eps_values = positive["eps"].astype(float).to_numpy()
        y_values = positive["asrcc"].astype(float).to_numpy()
        y_monotone = np.maximum.accumulate(y_values)
        max_eps = float(eps_values[-1])
        max_asrcc = float(y_monotone[-1])

        if y_monotone[0] >= 0.5:
            epsilon50 = float(eps_values[0])
            status = "below_min_positive_eps"
        elif max_asrcc < 0.5:
            epsilon50 = float("nan")
            status = "not_reached"
        else:
            crossing_idx = int(np.argmax(y_monotone >= 0.5))
            e1 = float(eps_values[crossing_idx - 1])
            e2 = float(eps_values[crossing_idx])
            y1 = float(y_monotone[crossing_idx - 1])
            y2 = float(y_monotone[crossing_idx])
            log_e1 = math.log10(e1)
            log_e2 = math.log10(e2)
            log_eps50 = log_e1 + (0.5 - y1) * (log_e2 - log_e1) / (y2 - y1)
            epsilon50 = float(10 ** log_eps50)
            status = "interpolated"

        rows.append(
            {
                **base,
                "epsilon50": epsilon50,
                "epsilon50_status": status,
                "max_eps": max_eps,
                "max_asrcc": max_asrcc,
                "clean_accuracy": clean_accuracy,
                "clean_correct_n": clean_correct_n,
            }
        )

    return pd.DataFrame(rows, columns=EPSILON50_COLUMNS).sort_values(
        BASELINE_COLUMNS, kind="mergesort"
    )


def _format_tick(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return str(value)
    if number == 0:
        return "0"
    if abs(number) < 1e-3 or abs(number) >= 1e3:
        return f"{number:.0e}"
    return f"{number:g}"


def _site_for_path(site: Any) -> str:
    text = str(site) if site is not None else "unknown"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text.strip())
    return text or "unknown"


def _load_pyplot(warnings: List[str]) -> bool:
    global plt
    if plt is not None:
        return True
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ImportError as exc:
        warnings.append(f"matplotlib is unavailable; plots were not created ({exc})")
        return False
    plt = pyplot
    return True


def _plot_heatmap(
    per_condition: pd.DataFrame,
    metric: str,
    dataset: str,
    out_dir: Path,
    filename_middle: str,
    colorbar_label: str,
) -> None:
    for site in sorted(per_condition["site"].dropna().unique()):
        site_df = per_condition[per_condition["site"] == site].copy()
        site_df = site_df[site_df["R"].notna() & site_df["eps"].notna()]
        if site_df.empty:
            continue

        averaged = site_df.groupby(["eps", "R"], dropna=False, as_index=False)[metric].mean()
        eps_values = sorted(averaged["eps"].astype(float).unique())
        r_values = sorted(averaged["R"].astype(int).unique())
        matrix = np.full((len(eps_values), len(r_values)), np.nan)
        eps_to_i = {eps: i for i, eps in enumerate(eps_values)}
        r_to_j = {r: j for j, r in enumerate(r_values)}
        for _, row in averaged.iterrows():
            matrix[eps_to_i[float(row["eps"])], r_to_j[int(row["R"])]] = float(row[metric])

        fig_width = max(5.5, 1.1 * len(r_values) + 2.0)
        fig_height = max(4.0, 0.45 * len(eps_values) + 1.8)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        image = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(f"{dataset} {colorbar_label} ({site})")
        ax.set_xlabel("R")
        ax.set_ylabel("epsilon")
        ax.set_xticks(np.arange(len(r_values)))
        ax.set_xticklabels([str(r) for r in r_values])
        ax.set_yticks(np.arange(len(eps_values)))
        ax.set_yticklabels([_format_tick(eps) for eps in eps_values])
        fig.colorbar(image, ax=ax, label=colorbar_label)
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset}_{filename_middle}_{_site_for_path(site)}.png", dpi=200)
        plt.close(fig)


def _plot_epsilon50_vs_r(epsilon50: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    finite = epsilon50[
        epsilon50["R"].notna()
        & epsilon50["epsilon50"].notna()
        & np.isfinite(epsilon50["epsilon50"].astype(float))
        & (epsilon50["epsilon50"].astype(float) > 0)
    ].copy()

    if finite.empty:
        ax.text(0.5, 0.5, "No finite epsilon50 values", ha="center", va="center", transform=ax.transAxes)
    else:
        plotted_values: List[float] = []
        for site in sorted(finite["site"].dropna().unique()):
            site_df = finite[finite["site"] == site]
            averaged = site_df.groupby("R", as_index=False)["epsilon50"].mean().sort_values("R")
            if averaged.empty:
                continue
            x_values = averaged["R"].astype(int).to_numpy()
            y_values = averaged["epsilon50"].astype(float).to_numpy()
            plotted_values.extend(y_values.tolist())
            ax.plot(x_values, y_values, marker="o", linewidth=1.8, label=str(site))
        if plotted_values and all(value > 0 for value in plotted_values):
            ax.set_yscale("log")
        ax.legend(title="site", frameon=False)

    omitted = int((epsilon50["epsilon50_status"] == "not_reached").sum()) if not epsilon50.empty else 0
    if omitted:
        ax.text(
            0.01,
            0.02,
            f"{omitted} not_reached rows omitted",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
        )
    ax.set_title(f"{dataset} epsilon50 vs R")
    ax.set_xlabel("R")
    ax.set_ylabel("epsilon50")
    fig.tight_layout()
    fig.savefig(out_dir / f"{dataset}_epsilon50_vs_R.png", dpi=200)
    plt.close(fig)


def _plot_clean_accuracy_vs_r(per_condition: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    clean_base = per_condition.drop_duplicates(BASELINE_COLUMNS)
    clean_base = clean_base[
        clean_base["R"].notna()
        & clean_base["clean_accuracy"].notna()
        & np.isfinite(clean_base["clean_accuracy"].astype(float))
    ].copy()

    if clean_base.empty:
        ax.text(0.5, 0.5, "No clean accuracy values", ha="center", va="center", transform=ax.transAxes)
    else:
        for site in sorted(clean_base["site"].dropna().unique()):
            site_df = clean_base[clean_base["site"] == site]
            averaged = site_df.groupby("R", as_index=False)["clean_accuracy"].mean().sort_values("R")
            if averaged.empty:
                continue
            ax.plot(
                averaged["R"].astype(int).to_numpy(),
                averaged["clean_accuracy"].astype(float).to_numpy(),
                marker="o",
                linewidth=1.8,
                label=str(site),
            )
        ax.legend(title="site", frameon=False)

    ax.set_title(f"{dataset} clean accuracy vs R")
    ax.set_xlabel("R")
    ax.set_ylabel("clean_accuracy")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_dir / f"{dataset}_clean_accuracy_vs_R.png", dpi=200)
    plt.close(fig)


def make_plots(
    per_condition: pd.DataFrame,
    epsilon50: pd.DataFrame,
    dataset: str,
    out_dir: Path,
    warnings: List[str],
) -> None:
    if per_condition.empty:
        return
    if not _load_pyplot(warnings):
        return
    _plot_heatmap(per_condition, "asrcc", dataset, out_dir, "asrcc_heatmap", "ASRcc")
    _plot_heatmap(per_condition, "invalid_rate", dataset, out_dir, "invalid_rate_heatmap", "invalid_rate")
    _plot_epsilon50_vs_r(epsilon50, dataset, out_dir)
    _plot_clean_accuracy_vs_r(per_condition, dataset, out_dir)


def find_jsonl_files(root: Path, dataset: str, subdir: str) -> List[Path]:
    preferred = root / dataset / subdir
    search_root = preferred if preferred.exists() else root
    if not search_root.exists():
        return []
    return sorted(path for path in search_root.rglob("*.jsonl") if path.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate latent-contagion Experiment B one-shot JSONL runs."
    )
    parser.add_argument("--root", default="outputs/latent_contagion/experiment_b")
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--subdir", default="oneshot")
    parser.add_argument("--out_dir", default="outputs/latent_contagion/experiment_b/aggregate")
    parser.add_argument("--exclude_s2p_r1", nargs="?", const=True, default=True, type=parse_bool)
    parser.add_argument("--make_plots", nargs="?", const=True, default=True, type=parse_bool)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    jsonl_files = find_jsonl_files(root, args.dataset, args.subdir)
    if not jsonl_files:
        warnings.append(
            f"no JSONL files found under {root / args.dataset / args.subdir} "
            f"or fallback root {root}"
        )

    sample_df = build_condition_dataframe(jsonl_files, args.dataset, warnings)
    sample_records_parsed = int(len(sample_df))

    if args.exclude_s2p_r1 and not sample_df.empty:
        sample_df = sample_df[~((sample_df["site"] == "s2p") & (sample_df["R"] == 1))].reset_index(drop=True)

    per_condition = compute_per_condition_metrics(sample_df, warnings)
    epsilon50 = compute_epsilon50(per_condition)
    disagreements = compute_clean_disagreements(sample_df, warnings)

    per_condition_path = out_dir / f"{args.dataset}_experiment_b_per_condition.csv"
    epsilon50_path = out_dir / f"{args.dataset}_experiment_b_epsilon50.csv"
    disagreements_path = out_dir / f"{args.dataset}_experiment_b_clean_disagreements.csv"
    warnings_path = out_dir / f"{args.dataset}_experiment_b_warnings.txt"

    per_condition.to_csv(per_condition_path, index=False)
    epsilon50.to_csv(epsilon50_path, index=False)
    disagreements.to_csv(disagreements_path, index=False)
    if args.make_plots:
        make_plots(per_condition, epsilon50, args.dataset, out_dir, warnings)

    warnings_path.write_text("\n".join(warnings) + ("\n" if warnings else "No warnings.\n"), encoding="utf-8")

    print(f"files parsed: {len(jsonl_files)}")
    print(f"sample records parsed: {sample_records_parsed}")
    print(f"conditions aggregated: {len(per_condition)}")
    print(f"epsilon50 rows: {len(epsilon50)}")
    print(f"warnings count: {len(warnings)}")
    print(f"out_dir: {out_dir}")


if __name__ == "__main__":
    main()
