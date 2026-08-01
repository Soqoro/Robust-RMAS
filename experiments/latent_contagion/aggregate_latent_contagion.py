#!/usr/bin/env python3
"""Aggregate latent-contagion JSONL logs.

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


np = None
pd = None
plt = None


def load_required_packages() -> None:
    global np, pd
    try:
        import numpy as np_module
        import pandas as pd_module
    except ImportError as exc:
        raise SystemExit(
            "aggregate_latent_contagion.py requires numpy and pandas. "
            "Install those packages, and install matplotlib as well if --make_plots true."
        ) from exc
    np = np_module
    pd = pd_module


CONDITION_COLUMNS = [
    "dataset",
    "style",
    "method",
    "role_response_regime",
    "mas_shape",
    "lc_mode",
    "lc_direction",
    "lc_steering_method",
    "lc_steering_id",
    "site",
    "R",
    "lc_round",
    "seed",
    "lc_seed",
    "eps",
]

BASELINE_COLUMNS = [
    "dataset",
    "style",
    "method",
    "role_response_regime",
    "mas_shape",
    "lc_mode",
    "lc_direction",
    "lc_steering_method",
    "lc_steering_id",
    "site",
    "R",
    "lc_round",
    "seed",
    "lc_seed",
]

SITELESS_BASELINE_COLUMNS = [
    "dataset",
    "style",
    "method",
    "role_response_regime",
    "mas_shape",
    "lc_mode",
    "lc_direction",
    "lc_steering_method",
    "lc_steering_id",
    "R",
    "lc_round",
    "seed",
    "lc_seed",
]

# A canonical clean run is independent of every latent-contagion control.  In
# particular, site/round/direction/steering metadata must not select a different
# clean answer set when epsilon is zero.
CORE_CLEAN_COLUMNS = [
    "dataset",
    "style",
    "method",
    "role_response_regime",
    "mas_shape",
    "R",
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
    "clean_flip_floor",
    "clean_asr",
    "excess_asrcc",
    "clean_to_invalid_floor",
    "excess_clean_to_invalid_rate",
]

EPSILON50_COLUMNS = BASELINE_COLUMNS + [
    "epsilon50",
    "epsilon50_status",
    "max_eps",
    "max_asrcc",
    "max_excess_asrcc",
    "epsilon50_metric",
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

CLEAN_FLIP_FLOOR_COLUMNS = SITELESS_BASELINE_COLUMNS + [
    "site_a",
    "site_b",
    "n_common",
    "clean_correct_a_n",
    "clean_flip_count",
    "clean_flip_rate",
    "clean_to_invalid_count",
    "clean_to_invalid_rate",
]

CLEAN_FLIP_FLOOR_POOLED_COLUMNS = SITELESS_BASELINE_COLUMNS + [
    "n_ordered_pairs",
    "pooled_clean_correct_a_n",
    "pooled_clean_flip_count",
    "pooled_clean_flip_rate",
    "pooled_clean_to_invalid_count",
    "pooled_clean_to_invalid_rate",
]

EXCESS_ASR_COLUMNS = [
    "clean_flip_floor",
    "clean_asr",
    "excess_asrcc",
    "clean_to_invalid_floor",
    "excess_clean_to_invalid_rate",
]

CANONICAL_CLEAN_METRIC_COLUMNS = CORE_CLEAN_COLUMNS + [
    "reference_source_file",
    "control_source_file",
    "reference_n_total",
    "reference_correct_n",
    "clean_accuracy",
    "control_n_total",
    "clean_flip_count",
    "clean_asr",
    "clean_to_invalid_count",
    "clean_to_invalid_rate",
]

COHORT_PROVENANCE_COLUMNS = [
    "provenance_schema_version",
    "sample_cohort_sha256",
    "sample_ids_sha256",
    "questions_sha256",
    "ground_truths_sha256",
]

GENERATION_PROVENANCE_COLUMNS = [
    "generation_config_sha256",
    "evaluation_config_sha256",
    "evaluation_protocol",
]
PROVENANCE_MATCH_COLUMNS = COHORT_PROVENANCE_COLUMNS + GENERATION_PROVENANCE_COLUMNS
CANONICAL_FILE_PROVENANCE_COLUMNS = PROVENANCE_MATCH_COLUMNS + ["attack_config_sha256"]

FILENAME_TOKEN_RE = re.compile(
    r"(?:^|_)(?P<key>lc_round|lc_seed|site|eps|epsilon|R|rounds|seed)=(?P<value>[^_]+)"
)


def parse_bool(value: Any) -> bool:
    """Coerce common serialized boolean values to bool."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    integer_types = (int,) if np is None else (int, np.integer)
    float_types = (float,) if np is None else (float, np.floating)
    if isinstance(value, integer_types):
        return int(value) != 0
    if isinstance(value, float_types):
        return bool(math.isfinite(float(value)) and float(value) != 0.0)
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
    invalid_value = _first_nonempty(record.get("answer_invalid_strict"), record.get("answer_invalid"))
    if invalid_value is not None:
        return parse_bool(invalid_value)
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


def _preferred_correctness_value(record: Mapping[str, Any]) -> Any:
    return _first_nonempty(
        record.get("is_correct_strict"),
        record.get("correct_strict"),
        record.get("is_correct"),
        record.get("correct"),
    )


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
        elif key == "lc_seed":
            metadata["lc_seed"] = _to_int(value)
        elif key == "lc_round":
            metadata["lc_round"] = _to_int(value)
    return metadata


def infer_role_response_regime(record: Mapping[str, Any]) -> str:
    value = _first_nonempty(record.get("role_response_regime"))
    if value is not None:
        return str(value).strip().lower() or "neutral"
    source = str(record.get("__source_file", "")).replace("\\", "/").lower()
    path_text = f"/{source.strip('/')}/"
    for regime in ("amplifying", "corrective", "neutral"):
        if f"/{regime}/" in path_text:
            return regime
    return "neutral"


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

    correctness_value = _preferred_correctness_value(record)
    site = _first_nonempty(record.get("lc_site"), filename_metadata.get("site"))
    eps = _first_nonempty(record.get("lc_epsilon"), filename_metadata.get("eps"))
    recursion_rounds = _first_nonempty(record.get("recursion_rounds"), filename_metadata.get("R"))
    lc_round = _first_nonempty(record.get("lc_round"), filename_metadata.get("lc_round"), 0)
    seed = _first_nonempty(
        record.get("seed"),
        filename_metadata.get("seed"),
        record.get("lc_seed"),
    )
    lc_seed = _first_nonempty(
        record.get("lc_seed"),
        filename_metadata.get("lc_seed"),
        filename_metadata.get("seed"),
        seed,
    )
    dataset = _first_nonempty(record.get("dataset"), dataset_default)
    lc_direction = _first_nonempty(record.get("lc_direction"), "random")
    lc_steering_method = _first_nonempty(record.get("lc_steering_method"), "")
    lc_steering_id = _first_nonempty(record.get("lc_steering_id"), "")
    role_response_regime = infer_role_response_regime(record)

    return {
        "dataset": _metadata_text(dataset),
        "style": _metadata_text(record.get("style")),
        "method": _metadata_text(record.get("method")),
        "role_response_regime": _metadata_text(role_response_regime),
        "mas_shape": _metadata_text(record.get("mas_shape")),
        "lc_mode": _metadata_text(record.get("lc_mode")),
        "lc_direction": _metadata_text(lc_direction),
        "lc_steering_method": _metadata_text(lc_steering_method),
        "lc_steering_id": _metadata_text(lc_steering_id),
        "site": _metadata_text(site),
        "R": _to_int(recursion_rounds),
        "lc_round": _to_int(lc_round),
        "seed": _to_int(seed),
        "lc_seed": _to_int(lc_seed),
        "eps": _to_float(eps),
        "sample_id": str(sample_id),
        "correct_bool": parse_bool(correctness_value),
        "invalid_bool": is_invalid_record(record),
        "final_answer": _clean_text_value(record.get("final_answer")),
        "judge_method": _metadata_text(record.get("judge_method")),
        "judge_method_strict": _metadata_text(record.get("judge_method_strict")),
        "answer_invalid": parse_bool(record.get("answer_invalid")) if record.get("answer_invalid") is not None else None,
        "answer_invalid_strict": (
            parse_bool(record.get("answer_invalid_strict"))
            if record.get("answer_invalid_strict") is not None
            else None
        ),
        "invalid_reason": _metadata_text(record.get("invalid_reason")),
        "invalid_reason_strict": _metadata_text(record.get("invalid_reason_strict")),
        "checker_version": _metadata_text(record.get("checker_version")),
        "provenance_schema_version": _to_int(record.get("provenance_schema_version")),
        "sample_cohort_sha256": _metadata_text(record.get("sample_cohort_sha256")),
        "sample_ids_sha256": _metadata_text(record.get("sample_ids_sha256")),
        "questions_sha256": _metadata_text(record.get("questions_sha256")),
        "ground_truths_sha256": _metadata_text(record.get("ground_truths_sha256")),
        "generation_config_sha256": _metadata_text(record.get("generation_config_sha256")),
        "evaluation_config_sha256": _metadata_text(record.get("evaluation_config_sha256")),
        "evaluation_protocol": _metadata_text(
            _first_nonempty(
                record.get("evaluation_protocol"),
                "strict" if (
                    "is_correct_strict" in record or "correct_strict" in record
                ) else "native",
            )
        ),
        "sample_input_sha256": _metadata_text(record.get("sample_input_sha256")),
        "effective_sample_input_sha256": _metadata_text(
            record.get("effective_sample_input_sha256")
        ),
        "attack_config_sha256": _metadata_text(record.get("attack_config_sha256")),
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
        np.mean([parse_bool(_preferred_correctness_value(sample)) for sample in samples])
    )
    has_strict_samples = any(
        "is_correct_strict" in sample or "correct_strict" in sample for sample in samples
    )
    for summary in summaries:
        summary_tracks_strict = (
            "strict" in str(summary.get("evaluation_protocol", "")).lower()
            or any("strict" in str(key) for key in summary)
        )
        if has_strict_samples and not summary_tracks_strict:
            warnings.append(f"{path}: summary row was not rejudged; using strict per-sample fields when present")
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
        "judge_method",
        "judge_method_strict",
        "answer_invalid",
        "answer_invalid_strict",
        "invalid_reason",
        "invalid_reason_strict",
        "checker_version",
        "provenance_schema_version",
        "sample_cohort_sha256",
        "sample_ids_sha256",
        "questions_sha256",
        "ground_truths_sha256",
        "generation_config_sha256",
        "evaluation_config_sha256",
        "evaluation_protocol",
        "sample_input_sha256",
        "effective_sample_input_sha256",
        "attack_config_sha256",
        "condition_ambiguous",
        "source_file",
        "line_number",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    for column in columns:
        if column not in df.columns:
            df[column] = None
    df["condition_ambiguous"] = False

    df = df.sort_values(["source_file", "line_number"], kind="mergesort").reset_index(drop=True)
    duplicate_mask = df.duplicated(CONDITION_COLUMNS + ["sample_id"], keep=False)
    if duplicate_mask.any():
        duplicate_pairs = df.loc[duplicate_mask, CONDITION_COLUMNS + ["sample_id"]].drop_duplicates()
        ambiguous_condition_keys = {
            tuple(_key_value(row[column]) for column in CONDITION_COLUMNS)
            for _, row in duplicate_pairs.iterrows()
        }
        warnings.append(
            f"found {len(duplicate_pairs)} duplicate condition/sample_id pairs; "
            "keeping the first row for diagnostics and marking those conditions ambiguous"
        )
        df = df.drop_duplicates(CONDITION_COLUMNS + ["sample_id"], keep="first").reset_index(drop=True)
        df["condition_ambiguous"] = df.apply(
            lambda row: tuple(_key_value(row[column]) for column in CONDITION_COLUMNS)
            in ambiguous_condition_keys,
            axis=1,
        )

    return df[columns]


def _canonical_path_metadata(path: Path) -> Dict[str, Any]:
    """Read core R/seed metadata from standard canonical-clean path tokens."""
    metadata = parse_metadata_from_filename(path)
    for part in reversed(path.parts):
        r_match = re.fullmatch(r"R(?:=)?(-?\d+)", part)
        if r_match and "R" not in metadata:
            metadata["R"] = int(r_match.group(1))
        seed_match = re.fullmatch(r"seed(?:=)?(-?\d+)", part, flags=re.IGNORECASE)
        if seed_match and "seed" not in metadata:
            metadata["seed"] = int(seed_match.group(1))
    return metadata


def _has_complete_summary(
    path: Path,
    samples: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    label: str,
    warnings: List[str],
) -> bool:
    if not samples:
        warnings.append(f"{label} canonical file has no sample rows and was skipped: {path}")
        return False
    if len(summaries) != 1:
        warnings.append(
            f"{label} canonical file requires exactly one summary row; "
            f"found {len(summaries)} and skipped: {path}"
        )
        return False
    declared_total = _to_int(
        _first_nonempty(summaries[0].get("num_samples"), summaries[0].get("n_total"))
    )
    if declared_total is None:
        warnings.append(
            f"{label} canonical file summary has no valid num_samples/n_total and was skipped: {path}"
        )
        return False
    if declared_total != len(samples):
        warnings.append(
            f"{label} canonical file is incomplete: summary declares {declared_total} samples "
            f"but {len(samples)} rows were read; skipped: {path}"
        )
        return False
    return True


def _nonempty_serialized_values(values: Sequence[Any]) -> set:
    serialized = set()
    for value in values:
        normalized = _key_value(value)
        if normalized is None:
            continue
        text = str(normalized).strip()
        if text:
            serialized.add(text)
    return serialized


def _has_consistent_file_provenance(
    path: Path,
    samples: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    label: str,
    warnings: List[str],
) -> bool:
    """Require complete, internally consistent provenance for canonical files."""
    for column in CANONICAL_FILE_PROVENANCE_COLUMNS:
        sample_values = [sample.get(column) for sample in samples]
        present_sample_values = [
            value for value in sample_values if value is not None and str(value).strip()
        ]
        summary_value = summary.get(column)
        summary_present = summary_value is not None and str(summary_value).strip() != ""
        if len(present_sample_values) != len(samples) or not summary_present:
            warnings.append(
                f"{label} canonical file has incomplete {column} provenance and was skipped: {path}"
            )
            return False
        combined = _nonempty_serialized_values(present_sample_values + [summary_value])
        if len(combined) != 1:
            warnings.append(
                f"{label} canonical file has inconsistent {column} provenance and was skipped: {path}"
            )
            return False
    return True


def _has_attack_free_canonical_summary(
    path: Path,
    summary: Mapping[str, Any],
    label: str,
    warnings: List[str],
) -> bool:
    """Reject a nominal clean file if any supported attack mechanism is active."""
    problems: List[str] = []
    if "lc_enabled" not in summary or parse_bool(summary.get("lc_enabled")):
        problems.append("lc_enabled is not explicitly false")
    if not _is_zero_eps(summary.get("lc_epsilon")):
        problems.append("lc_epsilon is not zero")
    if str(summary.get("lc_mode", "")).strip().lower() != "none":
        problems.append("lc_mode is not none")
    if str(summary.get("question_suffix_path", "")).strip():
        problems.append("question suffix is active")
    if str(summary.get("prompt_footer_path", "")).strip():
        problems.append("prompt footer is active")

    attack_config = summary.get("attack_config")
    if not isinstance(attack_config, Mapping):
        problems.append("attack_config metadata is missing")
    else:
        latent = attack_config.get("latent_contagion")
        probe = attack_config.get("role_profile_probe")
        if not isinstance(latent, Mapping):
            problems.append("latent attack metadata is missing")
        else:
            if str(latent.get("mode", "")).strip().lower() != "none":
                problems.append("latent attack mode is not none")
            if not _is_zero_eps(latent.get("epsilon")):
                problems.append("latent attack epsilon is not zero")
        if not isinstance(probe, Mapping):
            problems.append("role-profile attack metadata is missing")
        else:
            if str(probe.get("mode", "")).strip().lower() != "none":
                problems.append("role-profile probe mode is not none")
            if not _is_zero_eps(probe.get("epsilon")):
                problems.append("role-profile probe epsilon is not zero")
        if str(attack_config.get("question_suffix_path", "")).strip():
            problems.append("attack_config has a question suffix")
        if str(attack_config.get("prompt_footer_path", "")).strip():
            problems.append("attack_config has a prompt footer")

    if problems:
        warnings.append(
            f"{label} canonical file is not attack-free ({'; '.join(problems)}) and was skipped: {path}"
        )
        return False
    return True


def _single_frame_value(frame: pd.DataFrame, column: str) -> Tuple[Optional[str], bool]:
    if column not in frame.columns:
        return None, True
    raw_values = frame[column].tolist()
    present_count = sum(
        1 for value in raw_values if _nonempty_serialized_values([value])
    )
    if 0 < present_count < len(raw_values):
        return None, False
    values = _nonempty_serialized_values(raw_values)
    if len(values) > 1:
        return None, False
    return (next(iter(values)) if values else None), True


def _provenance_compatibility_error(
    reference: pd.DataFrame,
    other: pd.DataFrame,
) -> Optional[str]:
    """Return why two frames cannot be paired, or None when compatible."""
    for column in PROVENANCE_MATCH_COLUMNS:
        reference_value, reference_consistent = _single_frame_value(reference, column)
        other_value, other_consistent = _single_frame_value(other, column)
        if not reference_consistent:
            return f"canonical reference has inconsistent {column}"
        if not other_consistent:
            return f"paired run has inconsistent {column}"
        if reference_value is None and other_value is None:
            continue
        if reference_value is None or other_value is None:
            return f"{column} is missing on one side"
        if reference_value != other_value:
            return f"{column} differs"

    reference_ids = _sample_ids(reference)
    other_ids = _sample_ids(other)
    if reference_ids != other_ids:
        return None  # The caller emits a more informative set-size warning.

    reference_inputs = reference.set_index("sample_id")["sample_input_sha256"]
    other_inputs = other.set_index("sample_id")["sample_input_sha256"]
    for sample_id in sorted(reference_ids):
        reference_values = _nonempty_serialized_values([reference_inputs.loc[sample_id]])
        other_values = _nonempty_serialized_values([other_inputs.loc[sample_id]])
        reference_value = next(iter(reference_values)) if reference_values else ""
        other_value = next(iter(other_values)) if other_values else ""
        if not reference_value and not other_value:
            continue
        if not reference_value or not other_value:
            return f"sample_input_sha256 is missing on one side for sample_id={sample_id}"
        if reference_value != other_value:
            return f"sample_input_sha256 differs for sample_id={sample_id}"
    return None


def find_canonical_jsonl_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() == ".jsonl" else []
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.jsonl") if path.is_file())


def build_canonical_clean_index(
    jsonl_files: Sequence[Path],
    dataset_default: str,
    label: str,
    warnings: List[str],
) -> Dict[Tuple[Any, ...], pd.DataFrame]:
    """Load one complete, unambiguous canonical clean file per core key."""
    candidates: Dict[Tuple[Any, ...], List[pd.DataFrame]] = defaultdict(list)

    for path in sorted(jsonl_files):
        path_metadata = _canonical_path_metadata(path)
        samples, summaries = load_jsonl_file(path, warnings=warnings)
        _check_summary_accuracy(path, samples, summaries, warnings)
        if not _has_complete_summary(path, samples, summaries, label, warnings):
            continue
        if not _has_attack_free_canonical_summary(path, summaries[0], label, warnings):
            continue
        if not _has_consistent_file_provenance(path, samples, summaries[0], label, warnings):
            continue

        normalized_rows: List[Dict[str, Any]] = []
        for sample in samples:
            row = _normalize_sample_record(sample, path_metadata, dataset_default)
            # Older dedicated clean runs logged only lc_seed, which could be an
            # irrelevant default under lc_mode=none.  Prefer the new top-level
            # experiment seed; otherwise a standard seed path is authoritative.
            if _is_empty_value(sample.get("seed")) and path_metadata.get("seed") is not None:
                row["seed"] = _to_int(path_metadata.get("seed"))
            if _is_empty_value(sample.get("recursion_rounds")) and path_metadata.get("R") is not None:
                row["R"] = _to_int(path_metadata.get("R"))
            normalized_rows.append(row)

        file_frame = pd.DataFrame(normalized_rows)
        for _, group in file_frame.groupby(CORE_CLEAN_COLUMNS, dropna=False, sort=True):
            group = group.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
            key = _frame_key(group, CORE_CLEAN_COLUMNS)
            if str(key[CORE_CLEAN_COLUMNS.index("dataset")]) != str(dataset_default):
                continue
            context = " ".join(
                f"{column}={value}" for column, value in zip(CORE_CLEAN_COLUMNS, key)
            )
            if any(value is None or (isinstance(value, str) and not value.strip()) for value in key):
                warnings.append(
                    f"{label} canonical clean core key is incomplete ({context}); skipped: {path}"
                )
                continue
            if not group["eps"].apply(_is_zero_eps).all():
                warnings.append(
                    f"{label} canonical clean contains nonzero or missing epsilon ({context}); "
                    f"skipped: {path}"
                )
                continue
            if group["sample_id"].astype(str).duplicated().any():
                warnings.append(
                    f"{label} canonical clean has duplicate sample_ids ({context}); skipped: {path}"
                )
                continue
            candidates[key].append(group)

    clean_by_core: Dict[Tuple[Any, ...], pd.DataFrame] = {}
    for key, groups in candidates.items():
        context = " ".join(
            f"{column}={value}" for column, value in zip(CORE_CLEAN_COLUMNS, key)
        )
        if len(groups) != 1:
            source_files = sorted(
                {str(path) for group in groups for path in group["source_file"].unique()}
            )
            warnings.append(
                f"{label} canonical clean is ambiguous for {context}: found {len(groups)} "
                f"candidate files ({', '.join(source_files)}); core key was skipped"
            )
            continue
        clean_by_core[key] = groups[0]
    return clean_by_core


def _core_context(key: Tuple[Any, ...]) -> str:
    return " ".join(f"{column}={value}" for column, value in zip(CORE_CLEAN_COLUMNS, key))


def compute_fixed_canonical_clean_metrics(
    reference_by_core: Mapping[Tuple[Any, ...], pd.DataFrame],
    control_by_core: Mapping[Tuple[Any, ...], pd.DataFrame],
    control_supplied: bool,
    warnings: List[str],
) -> Tuple[pd.DataFrame, Dict[Tuple[Any, ...], Dict[str, Any]]]:
    """Compute one immutable reference-to-control clean ASR per core key."""
    rows: List[Dict[str, Any]] = []
    metrics_by_core: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    for key, reference in sorted(reference_by_core.items(), key=lambda item: tuple(map(str, item[0]))):
        base = dict(zip(CORE_CLEAN_COLUMNS, key))
        context = _core_context(key)
        reference = reference.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
        reference_ids = _sample_ids(reference)
        reference_indexed = reference.set_index("sample_id", drop=False)
        reference_correct = reference_indexed["correct_bool"].astype(bool).to_numpy()
        reference_correct_n = int(np.sum(reference_correct))
        reference_source = ",".join(sorted(reference["source_file"].astype(str).unique()))

        control_source = "<reference-self>"
        control_n_total = int(len(reference))
        clean_flip_count = 0
        clean_to_invalid_count = 0
        clean_asr = 0.0
        clean_to_invalid_rate = 0.0

        if reference_correct_n == 0:
            warnings.append(
                f"fixed clean ASR denominator reference_correct_n is zero for {context}"
            )
            clean_flip_count = float("nan")
            clean_to_invalid_count = float("nan")
            clean_asr = float("nan")
            clean_to_invalid_rate = float("nan")
        elif control_supplied:
            control = control_by_core.get(key)
            if control is None:
                warnings.append(f"fixed clean control missing for {context}")
                control_source = ""
                control_n_total = 0
                clean_flip_count = float("nan")
                clean_to_invalid_count = float("nan")
                clean_asr = float("nan")
                clean_to_invalid_rate = float("nan")
            else:
                control = control.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
                control_ids = _sample_ids(control)
                control_source = ",".join(sorted(control["source_file"].astype(str).unique()))
                control_n_total = int(len(control))
                compatibility_error = _provenance_compatibility_error(reference, control)
                if compatibility_error is not None:
                    warnings.append(
                        "fixed clean control provenance does not match reference for "
                        f"{context}: {compatibility_error}"
                    )
                    clean_flip_count = float("nan")
                    clean_to_invalid_count = float("nan")
                    clean_asr = float("nan")
                    clean_to_invalid_rate = float("nan")
                elif control_ids != reference_ids:
                    warnings.append(
                        "fixed clean control sample_ids do not exactly match reference for "
                        f"{context} (reference={len(reference_ids)}, control={len(control_ids)}, "
                        f"common={len(reference_ids & control_ids)})"
                    )
                    clean_flip_count = float("nan")
                    clean_to_invalid_count = float("nan")
                    clean_asr = float("nan")
                    clean_to_invalid_rate = float("nan")
                else:
                    ordered_ids = sorted(reference_ids)
                    aligned_reference = reference_indexed.loc[ordered_ids]
                    aligned_control = control.set_index("sample_id", drop=False).loc[ordered_ids]
                    reference_correct = aligned_reference["correct_bool"].astype(bool).to_numpy()
                    control_correct = aligned_control["correct_bool"].astype(bool).to_numpy()
                    control_invalid = aligned_control["invalid_bool"].astype(bool).to_numpy()
                    clean_flip_count = int(np.sum(reference_correct & ~control_correct))
                    clean_to_invalid_count = int(np.sum(reference_correct & control_invalid))
                    clean_asr = clean_flip_count / reference_correct_n
                    clean_to_invalid_rate = clean_to_invalid_count / reference_correct_n

        row = {
            **base,
            "reference_source_file": reference_source,
            "control_source_file": control_source,
            "reference_n_total": int(len(reference)),
            "reference_correct_n": reference_correct_n,
            "clean_accuracy": _safe_mean_bool(reference["correct_bool"]),
            "control_n_total": control_n_total,
            "clean_flip_count": clean_flip_count,
            "clean_asr": clean_asr,
            "clean_to_invalid_count": clean_to_invalid_count,
            "clean_to_invalid_rate": clean_to_invalid_rate,
        }
        rows.append(row)
        metrics_by_core[key] = row

    frame = pd.DataFrame(rows, columns=CANONICAL_CLEAN_METRIC_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(CORE_CLEAN_COLUMNS, kind="mergesort")
    return frame, metrics_by_core


def compute_per_condition_metrics_canonical(
    df: pd.DataFrame,
    reference_by_core: Mapping[Tuple[Any, ...], pd.DataFrame],
    clean_metrics_by_core: Mapping[Tuple[Any, ...], Mapping[str, Any]],
    warnings: List[str],
) -> pd.DataFrame:
    """Aggregate attacks against a fixed canonical clean reference."""
    if df.empty:
        return pd.DataFrame(columns=PER_CONDITION_COLUMNS)

    rows: List[Dict[str, Any]] = []
    missing_reference_keys: set = set()
    for _, group in df.groupby(CONDITION_COLUMNS, dropna=False, sort=True):
        group = group.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
        condition = {column: _key_value(group.iloc[0][column]) for column in CONDITION_COLUMNS}
        core_key = tuple(_key_value(condition[column]) for column in CORE_CLEAN_COLUMNS)
        context = _core_context(core_key)

        perturbed_accuracy = _safe_mean_bool(group["correct_bool"])
        invalid_rate = _safe_mean_bool(group["invalid_bool"])
        n_total = int(len(group))
        reference = reference_by_core.get(core_key)
        clean_metric = clean_metrics_by_core.get(core_key, {})

        clean_n_total = 0
        clean_correct_n = 0
        clean_accuracy = float("nan")
        clean_to_wrong_count = 0
        clean_to_invalid_count = 0
        asrcc = float("nan")
        clean_to_invalid_rate = float("nan")
        clean_asr = _to_float(clean_metric.get("clean_asr"))
        clean_to_invalid_floor = _to_float(clean_metric.get("clean_to_invalid_rate"))

        condition_ambiguous = (
            "condition_ambiguous" in group.columns
            and group["condition_ambiguous"].fillna(False).astype(bool).any()
        )
        attack_config_value, attack_config_consistent = _single_frame_value(
            group, "attack_config_sha256"
        )
        if condition_ambiguous:
            warnings.append(
                "attack condition has duplicate source rows for "
                f"{context} site={condition.get('site')} eps={condition.get('eps')} "
                f"lc_round={condition.get('lc_round')} lc_seed={condition.get('lc_seed')}; "
                "ASR left NaN"
            )
        elif not attack_config_consistent or attack_config_value is None:
            warnings.append(
                "attack condition has missing or inconsistent attack_config_sha256 for "
                f"{context} site={condition.get('site')} eps={condition.get('eps')} "
                f"lc_round={condition.get('lc_round')} lc_seed={condition.get('lc_seed')}; "
                "ASR left NaN"
            )
        elif reference is None:
            if core_key not in missing_reference_keys:
                warnings.append(f"canonical clean reference missing for {context}")
                missing_reference_keys.add(core_key)
        else:
            reference = reference.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
            clean_n_total = int(len(reference))
            clean_correct_n = int(np.sum(reference["correct_bool"].astype(bool).to_numpy()))
            clean_accuracy = _safe_mean_bool(reference["correct_bool"])
            reference_ids = _sample_ids(reference)
            perturbed_ids = _sample_ids(group)
            compatibility_error = _provenance_compatibility_error(reference, group)

            if compatibility_error is not None:
                warnings.append(
                    "attack provenance does not match canonical clean reference for "
                    f"{context} site={condition.get('site')} eps={condition.get('eps')} "
                    f"lc_round={condition.get('lc_round')}: {compatibility_error}; ASR left NaN"
                )
            elif reference_ids != perturbed_ids:
                warnings.append(
                    "attack sample_ids do not exactly match canonical clean reference for "
                    f"{context} site={condition.get('site')} eps={condition.get('eps')} "
                    f"lc_round={condition.get('lc_round')} "
                    f"(reference={len(reference_ids)}, attack={len(perturbed_ids)}, "
                    f"common={len(reference_ids & perturbed_ids)}); ASR left NaN"
                )
            elif clean_correct_n == 0:
                warnings.append(
                    "canonical clean-correct set is empty for "
                    f"{context} site={condition.get('site')} eps={condition.get('eps')}"
                )
            else:
                ordered_ids = sorted(reference_ids)
                aligned_reference = reference.set_index("sample_id", drop=False).loc[ordered_ids]
                aligned_perturbed = group.set_index("sample_id", drop=False).loc[ordered_ids]
                reference_correct = aligned_reference["correct_bool"].astype(bool).to_numpy()
                perturbed_correct = aligned_perturbed["correct_bool"].astype(bool).to_numpy()
                perturbed_invalid = aligned_perturbed["invalid_bool"].astype(bool).to_numpy()
                clean_to_wrong_count = int(np.sum(reference_correct & ~perturbed_correct))
                clean_to_invalid_count = int(np.sum(reference_correct & perturbed_invalid))
                eps_value = _to_float(condition.get("eps"))
                if eps_value is not None and np.isclose(eps_value, 0.0, rtol=0.0, atol=1e-12):
                    asrcc = 0.0
                    clean_to_wrong_count = 0
                else:
                    asrcc = clean_to_wrong_count / clean_correct_n
                clean_to_invalid_rate = clean_to_invalid_count / clean_correct_n

        clean_asr_value = clean_asr if clean_asr is not None else float("nan")
        clean_to_invalid_floor_value = (
            clean_to_invalid_floor if clean_to_invalid_floor is not None else float("nan")
        )
        excess_asrcc = (
            asrcc - clean_asr_value
            if math.isfinite(asrcc) and math.isfinite(clean_asr_value)
            else float("nan")
        )
        excess_clean_to_invalid_rate = (
            clean_to_invalid_rate - clean_to_invalid_floor_value
            if math.isfinite(clean_to_invalid_rate)
            and math.isfinite(clean_to_invalid_floor_value)
            else float("nan")
        )
        delta_accuracy = (
            perturbed_accuracy - clean_accuracy if math.isfinite(clean_accuracy) else float("nan")
        )
        rows.append(
            {
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
                "clean_flip_floor": clean_asr_value,
                "clean_asr": clean_asr_value,
                "excess_asrcc": excess_asrcc,
                "clean_to_invalid_floor": clean_to_invalid_floor_value,
                "excess_clean_to_invalid_rate": excess_clean_to_invalid_rate,
            }
        )

    return pd.DataFrame(rows, columns=PER_CONDITION_COLUMNS).sort_values(
        CONDITION_COLUMNS, kind="mergesort"
    )


def filter_canonical_attack_dataframe(
    df: pd.DataFrame,
    dataset: str,
    warnings: List[str],
) -> pd.DataFrame:
    """Keep only positive-epsilon attacks for canonical aggregation."""
    if df.empty:
        return df.copy()
    dataset_mask = df["dataset"].astype(str) == str(dataset)
    if not dataset_mask.all():
        warnings.append(
            f"canonical mode ignored {int((~dataset_mask).sum())} sample rows from other datasets"
        )
    dataset_frame = df[dataset_mask].copy()
    positive_mask = dataset_frame["eps"].apply(_finite_positive)
    ignored_count = int((~positive_mask).sum())
    if ignored_count:
        ignored_conditions = int(
            dataset_frame.loc[~positive_mask, CONDITION_COLUMNS].drop_duplicates().shape[0]
        )
        warnings.append(
            "canonical mode ignored "
            f"{ignored_count} sample rows across {ignored_conditions} non-positive or missing-epsilon "
            "conditions; only strict positive-epsilon attacks are aggregated"
        )
    return dataset_frame[positive_mask].reset_index(drop=True)


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


def _siteless_warning_context(base: Mapping[str, Any]) -> str:
    return " ".join(f"{column}={base.get(column)}" for column in SITELESS_BASELINE_COLUMNS)


def compute_clean_flip_floor(
    df: pd.DataFrame,
    warnings: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pairwise_empty = pd.DataFrame(columns=CLEAN_FLIP_FLOOR_COLUMNS)
    pooled_empty = pd.DataFrame(columns=CLEAN_FLIP_FLOOR_POOLED_COLUMNS)
    if df.empty:
        return pairwise_empty, pooled_empty

    pairwise_rows: List[Dict[str, Any]] = []
    pooled_rows: List[Dict[str, Any]] = []
    for _, full_group in df.groupby(SITELESS_BASELINE_COLUMNS, dropna=False, sort=True):
        clean_group = full_group[full_group["eps"].apply(_is_zero_eps)].copy()
        group = clean_group if not clean_group.empty else full_group
        group = group.sort_values(["site", "sample_id"], kind="mergesort").reset_index(drop=True)
        base = dict(zip(SITELESS_BASELINE_COLUMNS, _frame_key(group, SITELESS_BASELINE_COLUMNS)))
        context = _siteless_warning_context(base)

        site_frames: Dict[str, pd.DataFrame] = {}
        for site, site_group in clean_group.groupby("site", dropna=False, sort=True):
            site_key = str(_key_value(site))
            site_frames[site_key] = (
                site_group.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
            )

        sites = sorted(site_frames)
        if len(sites) < 2:
            warnings.append(
                f"fewer than two eps=0 clean sites for {context} (found {len(sites)})"
            )
            pooled_rows.append(
                {
                    **base,
                    "n_ordered_pairs": 0,
                    "pooled_clean_correct_a_n": 0,
                    "pooled_clean_flip_count": 0,
                    "pooled_clean_flip_rate": float("nan"),
                    "pooled_clean_to_invalid_count": 0,
                    "pooled_clean_to_invalid_rate": float("nan"),
                }
            )
            continue

        site_ids = {site: _sample_ids(site_frames[site]) for site in sites}
        site_indexed = {
            site: site_frames[site].set_index("sample_id", drop=False) for site in sites
        }

        pooled_ordered_pairs = 0
        pooled_clean_correct_a_n = 0
        pooled_clean_flip_count = 0
        pooled_clean_to_invalid_count = 0

        for i, site_a in enumerate(sites):
            for j, site_b in enumerate(sites):
                if i == j:
                    continue

                common_ids = sorted(site_ids[site_a] & site_ids[site_b])
                n_common = int(len(common_ids))
                clean_correct_a_n = 0
                clean_flip_count = 0
                clean_to_invalid_count = 0

                if n_common == 0:
                    if i < j:
                        warnings.append(
                            "eps=0 clean sample_id sets do not overlap for "
                            f"{context} site_a={site_a} site_b={site_b}"
                        )
                else:
                    aligned_a = site_indexed[site_a].loc[common_ids]
                    aligned_b = site_indexed[site_b].loc[common_ids]
                    correct_a = aligned_a["correct_bool"].astype(bool).to_numpy()
                    correct_b = aligned_b["correct_bool"].astype(bool).to_numpy()
                    invalid_b = aligned_b["invalid_bool"].astype(bool).to_numpy()
                    clean_correct_a_n = int(np.sum(correct_a))
                    if clean_correct_a_n > 0:
                        clean_flip_count = int(np.sum(correct_a & ~correct_b))
                        clean_to_invalid_count = int(np.sum(correct_a & invalid_b))

                if clean_correct_a_n == 0:
                    warnings.append(
                        "clean flip floor denominator clean_correct_a_n is zero for "
                        f"{context} site_a={site_a} site_b={site_b} n_common={n_common}"
                    )

                clean_flip_rate = (
                    clean_flip_count / clean_correct_a_n
                    if clean_correct_a_n > 0
                    else float("nan")
                )
                clean_to_invalid_rate = (
                    clean_to_invalid_count / clean_correct_a_n
                    if clean_correct_a_n > 0
                    else float("nan")
                )

                pairwise_rows.append(
                    {
                        **base,
                        "site_a": site_a,
                        "site_b": site_b,
                        "n_common": n_common,
                        "clean_correct_a_n": clean_correct_a_n,
                        "clean_flip_count": clean_flip_count,
                        "clean_flip_rate": clean_flip_rate,
                        "clean_to_invalid_count": clean_to_invalid_count,
                        "clean_to_invalid_rate": clean_to_invalid_rate,
                    }
                )

                pooled_ordered_pairs += 1
                pooled_clean_correct_a_n += clean_correct_a_n
                pooled_clean_flip_count += clean_flip_count
                pooled_clean_to_invalid_count += clean_to_invalid_count

        pooled_clean_flip_rate = (
            pooled_clean_flip_count / pooled_clean_correct_a_n
            if pooled_clean_correct_a_n > 0
            else float("nan")
        )
        pooled_clean_to_invalid_rate = (
            pooled_clean_to_invalid_count / pooled_clean_correct_a_n
            if pooled_clean_correct_a_n > 0
            else float("nan")
        )
        pooled_rows.append(
            {
                **base,
                "n_ordered_pairs": pooled_ordered_pairs,
                "pooled_clean_correct_a_n": pooled_clean_correct_a_n,
                "pooled_clean_flip_count": pooled_clean_flip_count,
                "pooled_clean_flip_rate": pooled_clean_flip_rate,
                "pooled_clean_to_invalid_count": pooled_clean_to_invalid_count,
                "pooled_clean_to_invalid_rate": pooled_clean_to_invalid_rate,
            }
        )

    pairwise = pd.DataFrame(pairwise_rows, columns=CLEAN_FLIP_FLOOR_COLUMNS)
    pooled = pd.DataFrame(pooled_rows, columns=CLEAN_FLIP_FLOOR_POOLED_COLUMNS)
    if not pairwise.empty:
        pairwise = pairwise.sort_values(
            SITELESS_BASELINE_COLUMNS + ["site_a", "site_b"], kind="mergesort"
        )
    if not pooled.empty:
        pooled = pooled.sort_values(SITELESS_BASELINE_COLUMNS, kind="mergesort")
    return pairwise, pooled


def add_clean_floor_excess_columns(
    per_condition: pd.DataFrame,
    clean_flip_floor_pooled: pd.DataFrame,
) -> pd.DataFrame:
    if per_condition.empty:
        return pd.DataFrame(columns=PER_CONDITION_COLUMNS)

    out = per_condition.copy()
    existing_excess_columns = [column for column in EXCESS_ASR_COLUMNS if column in out.columns]
    if existing_excess_columns:
        out = out.drop(columns=existing_excess_columns)

    if clean_flip_floor_pooled.empty:
        out["clean_flip_floor"] = float("nan")
        out["clean_to_invalid_floor"] = float("nan")
    else:
        floor = clean_flip_floor_pooled[
            SITELESS_BASELINE_COLUMNS
            + ["pooled_clean_flip_rate", "pooled_clean_to_invalid_rate"]
        ].copy()
        floor = floor.rename(
            columns={
                "pooled_clean_flip_rate": "clean_flip_floor",
                "pooled_clean_to_invalid_rate": "clean_to_invalid_floor",
            }
        )
        floor = floor.drop_duplicates(SITELESS_BASELINE_COLUMNS, keep="first")
        out = out.merge(floor, on=SITELESS_BASELINE_COLUMNS, how="left")

    # Preserve the historical clean_flip_floor name while exposing the metric
    # directly as clean_asr.  In legacy mode this remains the pooled cross-site
    # estimate; canonical mode supplies a fixed reference-to-control value.
    out["clean_asr"] = pd.to_numeric(out["clean_flip_floor"], errors="coerce")
    out["excess_asrcc"] = pd.to_numeric(out["asrcc"], errors="coerce") - out["clean_asr"]
    out["excess_clean_to_invalid_rate"] = pd.to_numeric(
        out["clean_to_invalid_rate"], errors="coerce"
    ) - pd.to_numeric(out["clean_to_invalid_floor"], errors="coerce")

    for column in PER_CONDITION_COLUMNS:
        if column not in out.columns:
            out[column] = float("nan")

    return out[PER_CONDITION_COLUMNS].sort_values(CONDITION_COLUMNS, kind="mergesort")


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
    """Compute epsilon where the primary excess-ASR metric reaches 0.5."""
    if per_condition.empty:
        return pd.DataFrame(columns=EPSILON50_COLUMNS)

    rows: List[Dict[str, Any]] = []
    for _, group in per_condition.groupby(BASELINE_COLUMNS, dropna=False, sort=True):
        group = group.sort_values("eps", kind="mergesort").reset_index(drop=True)
        base = {column: _key_value(group.iloc[0][column]) for column in BASELINE_COLUMNS}
        clean_accuracy = float(group["clean_accuracy"].dropna().iloc[0]) if group["clean_accuracy"].notna().any() else float("nan")
        clean_correct_n = int(group["clean_correct_n"].dropna().iloc[0]) if group["clean_correct_n"].notna().any() else 0

        positive = group[group["eps"].apply(_finite_positive)].copy()
        positive = positive[np.isfinite(positive["excess_asrcc"].astype(float))]
        if positive.empty:
            rows.append(
                {
                    **base,
                    "epsilon50": float("nan"),
                    "epsilon50_status": "insufficient_points",
                    "max_eps": float("nan"),
                    "max_asrcc": float("nan"),
                    "max_excess_asrcc": float("nan"),
                    "epsilon50_metric": "excess_asrcc",
                    "clean_accuracy": clean_accuracy,
                    "clean_correct_n": clean_correct_n,
                }
            )
            continue

        positive = positive.sort_values("eps", kind="mergesort")
        eps_values = positive["eps"].astype(float).to_numpy()
        y_values = positive["excess_asrcc"].astype(float).to_numpy()
        y_monotone = np.maximum.accumulate(y_values)
        max_eps = float(eps_values[-1])
        raw_asr_values = positive["asrcc"].astype(float).to_numpy()
        finite_raw_asr = raw_asr_values[np.isfinite(raw_asr_values)]
        max_asrcc = float(np.max(finite_raw_asr)) if len(finite_raw_asr) else float("nan")
        max_excess_asrcc = float(y_monotone[-1])

        if y_monotone[0] >= 0.5:
            epsilon50 = float(eps_values[0])
            status = "below_min_positive_eps"
        elif max_excess_asrcc < 0.5:
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
                "max_excess_asrcc": max_excess_asrcc,
                "epsilon50_metric": "excess_asrcc",
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
    *,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
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
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
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
    _plot_heatmap(
        per_condition,
        "excess_asrcc",
        dataset,
        out_dir,
        "excess_asrcc_heatmap",
        "excess ASRcc",
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
    )
    _plot_heatmap(per_condition, "asrcc", dataset, out_dir, "asrcc_heatmap", "raw ASRcc")
    _plot_heatmap(per_condition, "invalid_rate", dataset, out_dir, "invalid_rate_heatmap", "invalid_rate")
    _plot_epsilon50_vs_r(epsilon50, dataset, out_dir)
    _plot_clean_accuracy_vs_r(per_condition, dataset, out_dir)


def find_jsonl_files(
    root: Path,
    dataset: str,
    subdir: str,
    allow_root_fallback: bool = False,
) -> List[Path]:
    preferred = root / dataset / subdir
    search_root = preferred if preferred.exists() else (root if allow_root_fallback else preferred)
    if not search_root.exists():
        return []
    return sorted(path for path in search_root.rglob("*.jsonl") if path.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate latent-contagion JSONL runs."
    )
    parser.add_argument("--root", default="outputs/latent_contagion/experiment_b")
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--subdir", default="oneshot")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Directory for aggregate outputs. Defaults to <root>/aggregate.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Label used in output filenames. Defaults to the final path component of --root.",
    )
    parser.add_argument(
        "--clean_reference_root",
        "--clean-reference-root",
        default="",
        help=(
            "File or directory containing complete canonical clean-reference JSONLs. "
            "When supplied, in-grid eps=0 runs are never used as baselines."
        ),
    )
    parser.add_argument(
        "--clean_control_root",
        "--clean-control-root",
        default="",
        help=(
            "Optional file or directory containing a fixed clean-control replicate. "
            "With an explicitly supplied reference root, omitting this option uses "
            "the reference as its own control (clean ASR zero). Under automatic "
            "canonical discovery, a missing control leaves clean/excess ASR as NaN."
        ),
    )
    parser.add_argument("--exclude_s2p_r1", nargs="?", const=True, default=True, type=parse_bool)
    parser.add_argument("--make_plots", nargs="?", const=True, default=True, type=parse_bool)
    parser.add_argument(
        "--allow_root_fallback",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Explicitly scan all of --root when <root>/<dataset>/<subdir> is absent.",
    )
    parser.add_argument(
        "--legacy_baselines",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Disable automatic <root>/clean/{reference,control} discovery.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_required_packages()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_label = args.label or root.name or "latent_contagion"

    warnings: List[str] = []
    if args.legacy_baselines and (args.clean_reference_root or args.clean_control_root):
        raise SystemExit("--legacy_baselines cannot be combined with canonical clean roots")

    if not args.legacy_baselines and not str(args.clean_reference_root).strip():
        conventional_reference_root = root / "clean" / "reference"
        conventional_control_root = root / "clean" / "control"
        args.clean_reference_root = str(conventional_reference_root)
        if not str(args.clean_control_root).strip():
            args.clean_control_root = str(conventional_control_root)
        warnings.append(
            "using canonical clean roots under "
            f"{root / 'clean'}; missing roles remain NaN until their fixed jobs finish. "
            "Use --legacy_baselines true only for deliberate legacy analysis"
        )

    canonical_mode = bool(str(args.clean_reference_root).strip())
    if args.clean_control_root and not canonical_mode:
        raise SystemExit("--clean_control_root requires --clean_reference_root")

    reference_files: List[Path] = []
    control_files: List[Path] = []
    if canonical_mode:
        reference_root = Path(args.clean_reference_root)
        reference_files = find_canonical_jsonl_files(reference_root)
        if not reference_files:
            warnings.append(f"no canonical clean reference JSONL files found under {reference_root}")
        if args.clean_control_root:
            control_root = Path(args.clean_control_root)
            control_files = find_canonical_jsonl_files(control_root)
            if not control_files:
                warnings.append(f"no fixed clean control JSONL files found under {control_root}")

    jsonl_files = find_jsonl_files(
        root,
        args.dataset,
        args.subdir,
        allow_root_fallback=bool(args.allow_root_fallback),
    )
    if canonical_mode:
        canonical_paths = {
            path.resolve(strict=False) for path in reference_files + control_files
        }
        jsonl_files = [
            path for path in jsonl_files if path.resolve(strict=False) not in canonical_paths
        ]
    if not jsonl_files:
        warnings.append(
            f"no JSONL files found under {root / args.dataset / args.subdir}"
            + (f" or explicit fallback root {root}" if args.allow_root_fallback else "")
        )

    sample_df = build_condition_dataframe(jsonl_files, args.dataset, warnings)
    sample_records_parsed = int(len(sample_df))

    if args.exclude_s2p_r1 and not sample_df.empty:
        sample_df = sample_df[~((sample_df["site"] == "s2p") & (sample_df["R"] == 1))].reset_index(drop=True)
    if canonical_mode:
        sample_df = filter_canonical_attack_dataframe(sample_df, args.dataset, warnings)

    canonical_clean_metrics = pd.DataFrame(columns=CANONICAL_CLEAN_METRIC_COLUMNS)
    if canonical_mode:
        reference_by_core = build_canonical_clean_index(
            reference_files,
            args.dataset,
            "reference",
            warnings,
        )
        control_by_core = build_canonical_clean_index(
            control_files,
            args.dataset,
            "control",
            warnings,
        ) if args.clean_control_root else {}
        canonical_clean_metrics, clean_metrics_by_core = compute_fixed_canonical_clean_metrics(
            reference_by_core,
            control_by_core,
            control_supplied=bool(args.clean_control_root),
            warnings=warnings,
        )
        per_condition = compute_per_condition_metrics_canonical(
            sample_df,
            reference_by_core,
            clean_metrics_by_core,
            warnings,
        )
        clean_flip_floor = pd.DataFrame(columns=CLEAN_FLIP_FLOOR_COLUMNS)
        clean_flip_floor_pooled = pd.DataFrame(columns=CLEAN_FLIP_FLOOR_POOLED_COLUMNS)
        disagreements = pd.DataFrame(columns=DISAGREEMENT_COLUMNS)
    else:
        legacy_sample_df = sample_df
        if "condition_ambiguous" in sample_df.columns and not sample_df.empty:
            ambiguous_mask = sample_df["condition_ambiguous"].fillna(False).astype(bool)
            if ambiguous_mask.any():
                ambiguous_conditions = int(
                    sample_df.loc[ambiguous_mask, CONDITION_COLUMNS].drop_duplicates().shape[0]
                )
                warnings.append(
                    f"legacy mode excluded {ambiguous_conditions} duplicate/stale conditions "
                    "instead of selecting a result by file-path order"
                )
                legacy_sample_df = sample_df.loc[~ambiguous_mask].reset_index(drop=True)
        per_condition = compute_per_condition_metrics(legacy_sample_df, warnings)
        clean_flip_floor, clean_flip_floor_pooled = compute_clean_flip_floor(
            legacy_sample_df, warnings
        )
        per_condition = add_clean_floor_excess_columns(per_condition, clean_flip_floor_pooled)
        disagreements = compute_clean_disagreements(legacy_sample_df, warnings)

    epsilon50 = compute_epsilon50(per_condition)

    per_condition_path = out_dir / f"{args.dataset}_{output_label}_per_condition.csv"
    epsilon50_path = out_dir / f"{args.dataset}_{output_label}_epsilon50.csv"
    disagreements_path = out_dir / f"{args.dataset}_{output_label}_clean_disagreements.csv"
    clean_flip_floor_path = out_dir / f"{args.dataset}_{output_label}_clean_flip_floor.csv"
    clean_flip_floor_pooled_path = (
        out_dir / f"{args.dataset}_{output_label}_clean_flip_floor_pooled.csv"
    )
    canonical_clean_metrics_path = (
        out_dir / f"{args.dataset}_{output_label}_canonical_clean_metrics.csv"
    )
    warnings_path = out_dir / f"{args.dataset}_{output_label}_warnings.txt"

    per_condition.to_csv(per_condition_path, index=False)
    epsilon50.to_csv(epsilon50_path, index=False)
    disagreements.to_csv(disagreements_path, index=False)
    clean_flip_floor.to_csv(clean_flip_floor_path, index=False)
    clean_flip_floor_pooled.to_csv(clean_flip_floor_pooled_path, index=False)
    canonical_clean_metrics.to_csv(canonical_clean_metrics_path, index=False)
    if args.make_plots:
        make_plots(per_condition, epsilon50, args.dataset, out_dir, warnings)

    warnings_path.write_text("\n".join(warnings) + ("\n" if warnings else "No warnings.\n"), encoding="utf-8")

    print(f"files parsed: {len(jsonl_files)}")
    print(f"sample records parsed: {sample_records_parsed}")
    print(f"conditions aggregated: {len(per_condition)}")
    print(f"epsilon50 rows: {len(epsilon50)}")
    print(f"clean flip floor pairwise CSV: {clean_flip_floor_path}")
    print(f"clean flip floor pooled CSV: {clean_flip_floor_pooled_path}")
    print(f"canonical clean mode: {canonical_mode}")
    if canonical_mode:
        print(f"canonical reference files: {len(reference_files)}")
        print(f"canonical control files: {len(control_files)}")
        print(f"canonical clean metrics CSV: {canonical_clean_metrics_path}")
    print(f"warnings count: {len(warnings)}")
    print(f"out_dir: {out_dir}")


if __name__ == "__main__":
    main()
