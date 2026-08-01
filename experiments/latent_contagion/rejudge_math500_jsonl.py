#!/usr/bin/env python3
"""Rejudge Math500 JSONL outputs with the strict answer checker."""

from __future__ import annotations

import argparse
from collections import Counter
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from RecursiveMAS.inference_utils.answer_utils import (  # noqa: E402
    MATH500_CHECKER_VERSION,
    compare_answers_detailed,
)


EVALUATION_PROTOCOL = "math500_strict_rejudge"
ANSWER_UTILS_PATH = REPO_ROOT / "RecursiveMAS" / "inference_utils" / "answer_utils.py"


def normalize_math500_dataset(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"math500", "math-500", "huggingfaceh4/math-500"}:
        return "math500"
    raise ValueError(
        "rejudge_math500_jsonl.py only supports Math500 "
        f"(math500, math-500, or HuggingFaceH4/MATH-500); got {value!r}"
    )


def validate_record_dataset(record: Mapping[str, Any], source: Path, row_number: int) -> None:
    """Reject labelled non-Math500 rows before applying the Math500 checker.

    Older result files may omit these fields, so missing/blank labels remain
    supported.  When labels are present, both must identify Math500.
    """
    for field in ("dataset", "dataset_name"):
        value = record.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        try:
            normalize_math500_dataset(str(value))
        except ValueError as exc:
            raise ValueError(
                f"{source}: record {row_number} has non-Math500 {field}={value!r}; "
                "refusing to apply the Math500 checker"
            ) from exc


def build_evaluation_provenance(dataset: str) -> Tuple[Dict[str, Any], str]:
    normalized_dataset = normalize_math500_dataset(dataset)
    config = {
        "protocol": EVALUATION_PROTOCOL,
        "dataset": normalized_dataset,
        "checker_version": MATH500_CHECKER_VERSION,
        "answer_utils_sha256": hashlib.sha256(ANSWER_UTILS_PATH.read_bytes()).hexdigest(),
        "rejudge_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    payload = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return config, hashlib.sha256(payload).hexdigest()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "f", "no", "n", "off", "", "none", "null"}:
            return False
    return bool(value)


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def is_summary(record: Mapping[str, Any]) -> bool:
    return str(record.get("type", "")).lower() == "summary"


def input_output_pairs(input_path: Path, output_path: Path, recursive: bool, suffix: str) -> List[Tuple[Path, Path]]:
    if input_path.is_file():
        if output_path.suffix == ".jsonl":
            return [(input_path, output_path)]
        return [(input_path, output_path / f"{input_path.stem}{suffix}{input_path.suffix}")]

    pattern = "**/*.jsonl" if recursive else "*.jsonl"
    pairs: List[Tuple[Path, Path]] = []
    for src in sorted(input_path.glob(pattern)):
        if not src.is_file():
            continue
        rel = src.relative_to(input_path)
        dst = output_path / rel.parent / f"{src.stem}{suffix}{src.suffix}"
        pairs.append((src, dst))
    return pairs


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc})") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def build_summary(
    original: Mapping[str, Any],
    samples: List[Mapping[str, Any]],
    evaluation_config: Mapping[str, Any],
    evaluation_config_sha256: str,
) -> Dict[str, Any]:
    strict_correct = [parse_bool(row.get("is_correct_strict", row.get("correct_strict"))) for row in samples]
    num_correct = sum(1 for value in strict_correct if value)
    num_total = len(strict_correct)
    out = dict(original)
    for key in ("accuracy", "num_correct", "num_wrong"):
        if key in original:
            out[f"original_{key}"] = original.get(key)
    out["type"] = "summary"
    out["checker_version"] = MATH500_CHECKER_VERSION
    out["evaluation_protocol"] = EVALUATION_PROTOCOL
    out["evaluation_config_sha256"] = evaluation_config_sha256
    out["evaluation_config"] = dict(evaluation_config)
    out["num_samples"] = num_total
    out["num_correct"] = num_correct
    out["num_wrong"] = num_total - num_correct
    out["accuracy"] = 100.0 * num_correct / num_total if num_total else 0.0
    return out


def rejudge_record(
    record: Mapping[str, Any],
    dataset: str,
    preserve_original_canonical: bool,
    evaluation_config_sha256: str,
) -> Tuple[Dict[str, Any], bool, bool, bool]:
    out = dict(record)
    original_correct_value = first_nonempty(record.get("is_correct"), record.get("correct"))
    original_correct = parse_bool(original_correct_value)

    gold_text = first_nonempty(record.get("gold_answer_raw"), record.get("ground_truth"), "")
    pred_text = first_nonempty(record.get("raw_final_output"), record.get("raw_output"), record.get("final_answer"), "")
    detail = compare_answers_detailed(str(gold_text), str(pred_text), dataset_name=dataset)
    strict_correct = bool(detail.get("correct", False))

    out["original_is_correct"] = record.get("is_correct")
    out["original_correct"] = record.get("correct")
    out["original_gold_norm"] = record.get("gold_norm")
    out["original_pred_norm"] = record.get("pred_norm")
    out["is_correct_strict"] = strict_correct
    out["correct_strict"] = strict_correct
    out["gold_norm_strict"] = detail.get("gold_norm", "")
    out["pred_norm_strict"] = detail.get("pred_norm", "")
    out["judge_method_strict"] = detail.get("judge_method", "")
    out["answer_invalid_strict"] = bool(detail.get("answer_invalid", False))
    out["invalid_reason_strict"] = detail.get("invalid_reason", "")
    out["checker_version"] = detail.get("checker_version", MATH500_CHECKER_VERSION)
    out["evaluation_protocol"] = EVALUATION_PROTOCOL
    out["evaluation_config_sha256"] = evaluation_config_sha256
    out["gold_answer_parsed_strict"] = detail.get("gold_answer_parsed")
    out["pred_answer_parsed_strict"] = detail.get("pred_answer_parsed")

    if not preserve_original_canonical:
        out["is_correct"] = strict_correct
        out["correct"] = strict_correct
        out["gold_norm"] = detail.get("gold_norm", "")
        out["pred_norm"] = detail.get("pred_norm", "")
        out["gold_answer_parsed"] = detail.get("gold_answer_parsed")
        out["pred_answer_parsed"] = detail.get("pred_answer_parsed")
        out["judge_method"] = detail.get("judge_method", "")
        out["answer_invalid"] = bool(detail.get("answer_invalid", False))
        out["invalid_reason"] = detail.get("invalid_reason", "")

    false_positive = original_correct and not strict_correct
    false_negative = (not original_correct) and strict_correct
    invalid = bool(detail.get("answer_invalid", False))
    return out, false_positive, false_negative, invalid


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        try:
            target_mode = stat.S_IMODE(path.stat().st_mode)
        except FileNotFoundError:
            target_mode = None
        if target_mode is not None:
            os.fchmod(fd, target_mode)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        try:
            directory_fd = os.open(
                str(path.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as exc:
            unsupported = {errno.EACCES, errno.EINVAL, errno.EROFS}
            if hasattr(errno, "ENOTSUP"):
                unsupported.add(errno.ENOTSUP)
            if exc.errno not in unsupported:
                raise
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def rejudge_file(
    src: Path,
    dst: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if dst.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite true to replace: {dst}")

    rows = read_jsonl(src)
    evaluation_config, evaluation_config_sha256 = build_evaluation_provenance(args.dataset)
    normalized_dataset = str(evaluation_config["dataset"])
    sample_rows: List[Dict[str, Any]] = []
    output_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    false_positive_to_wrong = 0
    false_negative_to_correct = 0
    invalid_count = 0
    judge_methods: Counter[str] = Counter()
    invalid_reasons: Counter[str] = Counter()
    original_correct: List[bool] = []
    strict_correct: List[bool] = []

    for row_number, row in enumerate(rows, start=1):
        validate_record_dataset(row, src, row_number)
        if is_summary(row):
            summaries.append(row)
            continue
        rejudged, false_positive, false_negative, invalid = rejudge_record(
            row,
            dataset=normalized_dataset,
            preserve_original_canonical=args.preserve_original_canonical,
            evaluation_config_sha256=evaluation_config_sha256,
        )
        sample_rows.append(rejudged)
        output_rows.append(rejudged)
        false_positive_to_wrong += int(false_positive)
        false_negative_to_correct += int(false_negative)
        invalid_count += int(invalid)
        original_correct.append(parse_bool(first_nonempty(row.get("is_correct"), row.get("correct"))))
        strict_correct.append(bool(rejudged.get("correct_strict", False)))
        judge_methods[str(rejudged.get("judge_method_strict", ""))] += 1
        invalid_reasons[str(rejudged.get("invalid_reason_strict", ""))] += 1

    if args.summary:
        if summaries:
            output_rows.extend(
                build_summary(
                    summary,
                    sample_rows,
                    evaluation_config,
                    evaluation_config_sha256,
                )
                for summary in summaries
            )
        else:
            output_rows.append(
                build_summary(
                    {"type": "summary", "dataset": normalized_dataset},
                    sample_rows,
                    evaluation_config,
                    evaluation_config_sha256,
                )
            )

    write_jsonl(dst, output_rows)

    total = len(sample_rows)
    original_accuracy = sum(original_correct) / total if total else 0.0
    strict_accuracy = sum(strict_correct) / total if total else 0.0
    report = {
        "input": str(src),
        "output": str(dst),
        "rows_processed": total,
        "original_accuracy": original_accuracy,
        "strict_accuracy": strict_accuracy,
        "false_positive_to_wrong": false_positive_to_wrong,
        "false_negative_to_correct": false_negative_to_correct,
        "invalid_count": invalid_count,
        "judge_method_counts": dict(judge_methods.most_common()),
        "invalid_reason_counts": dict(invalid_reasons.most_common()),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rejudge Math500 JSONL files with strict answer checking.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--recursive", type=parse_bool, default=True)
    parser.add_argument("--suffix", default="_strict")
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--summary", type=parse_bool, default=True)
    parser.add_argument("--preserve_original_canonical", type=parse_bool, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    pairs = input_output_pairs(input_path, output_path, recursive=bool(args.recursive), suffix=args.suffix)
    if not pairs:
        raise SystemExit(f"No JSONL files found under {input_path}")

    total_rows = 0
    total_false_positive = 0
    total_false_negative = 0
    total_invalid = 0
    method_counts: Counter[str] = Counter()
    invalid_counts: Counter[str] = Counter()

    for src, dst in pairs:
        report = rejudge_file(src, dst, args)
        total_rows += int(report["rows_processed"])
        total_false_positive += int(report["false_positive_to_wrong"])
        total_false_negative += int(report["false_negative_to_correct"])
        total_invalid += int(report["invalid_count"])
        method_counts.update(report["judge_method_counts"])
        invalid_counts.update(report["invalid_reason_counts"])
        print(
            f"{src} -> {dst} rows={report['rows_processed']} "
            f"original_acc={100.0 * report['original_accuracy']:.2f}% "
            f"strict_acc={100.0 * report['strict_accuracy']:.2f}% "
            f"false_positive_to_wrong={report['false_positive_to_wrong']} "
            f"false_negative_to_correct={report['false_negative_to_correct']} "
            f"invalid={report['invalid_count']}"
        )

    print("===== total =====")
    print(f"rows processed: {total_rows}")
    print(f"false_positive_to_wrong count: {total_false_positive}")
    print(f"false_negative_to_correct count: {total_false_negative}")
    print(f"invalid count: {total_invalid}")
    print(f"most common judge_method_strict counts: {dict(method_counts.most_common())}")
    print(f"most common invalid_reason_strict counts: {dict(invalid_counts.most_common())}")


if __name__ == "__main__":
    main()
