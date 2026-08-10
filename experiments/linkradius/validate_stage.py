#!/usr/bin/env python3
"""Verify exact expected completions and atomically publish a stage gate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .grid import GridConfig, GridTask, build_grid
from .io_utils import atomic_write_json, content_hash, load_json, require_passed_gate, verify_completion
from .schemas import GATE_VERSION, ContractError, validate_gate


def discover_completion_records(root: str | Path) -> list[tuple[Path, Mapping[str, Any]]]:
    output = []
    for path in sorted(Path(root).rglob(".complete.json")):
        record = verify_completion(path.parent)
        output.append((path, record))
    return output


def verify_expected_completions(
    expected_tasks: Sequence[GridTask | Mapping[str, Any]],
    completion_records: Sequence[tuple[Path, Mapping[str, Any]]],
    *,
    expected_source_hash: str | None = None,
) -> dict[str, Any]:
    expected: dict[int, str] = {}
    for task in expected_tasks:
        data = task.as_dict() if isinstance(task, GridTask) else task
        index, key = int(data["array_index"]), str(data["config_key"])
        if index in expected:
            raise ContractError(f"duplicate expected array index: {index}")
        expected[index] = key
    found: dict[int, tuple[Path, Mapping[str, Any]]] = {}
    stale: list[int] = []
    unexpected: list[int] = []
    for path, record in completion_records:
        if "array_index" not in record:
            continue
        index = int(record["array_index"])
        if index not in expected:
            unexpected.append(index)
            continue
        if index in found:
            raise ContractError(f"duplicate completion for array index {index}")
        found[index] = (path, record)
        if record.get("config_hash") != expected[index] or (
            expected_source_hash is not None
            and record.get("source_hash") != expected_source_hash
        ):
            stale.append(index)
    missing = sorted(set(expected) - set(found))
    return {
        "passed": not missing and not stale and not unexpected,
        "expected_count": len(expected),
        "compatible_count": len(expected) - len(missing) - len(stale),
        "missing_array_indices": missing,
        "stale_array_indices": sorted(stale),
        "unexpected_array_indices": sorted(unexpected),
    }


def make_gate(
    *,
    gate_type: str,
    checks: Sequence[Mapping[str, Any]],
    config_hash: str,
    source_hash: str,
    prerequisite_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    normalized_checks = [dict(check) for check in checks]
    passed = bool(normalized_checks) and all(check.get("passed") is True for check in normalized_checks)
    gate: dict[str, Any] = {
        "schema_version": GATE_VERSION,
        "gate_type": gate_type,
        "passed": passed,
        "config_hash": config_hash,
        "source_hash": source_hash,
        "checks": normalized_checks,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    gate.update(dict(prerequisite_hashes or {}))
    gate["gate_content_hash"] = content_hash(gate, domain="linkradius:gate_content:v1")
    if passed:
        validate_gate(gate, gate_type=gate_type)
    return gate


def verify_prerequisites(
    prerequisites: Sequence[tuple[str | Path, str]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    checks = []
    hashes = {}
    for path, gate_type in prerequisites:
        gate = require_passed_gate(path, gate_type=gate_type)
        key = f"{gate_type}_hash"
        hashes[key] = str(gate.get("gate_content_hash", content_hash(gate)))
        checks.append({"name": f"prerequisite:{gate_type}", "passed": True, "path": str(path)})
    return checks, hashes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-grid", required=True)
    parser.add_argument("--completion-root", required=True)
    parser.add_argument("--gate-output", required=True)
    parser.add_argument("--gate-type", required=True)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--prerequisite", action="append", default=[], help="TYPE=PATH")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    expected_value = load_json(args.expected_grid)
    expected = expected_value.get("tasks") if isinstance(expected_value, Mapping) else expected_value
    if not isinstance(expected, list):
        raise ContractError("expected grid JSON must contain a task list")
    completion_check = verify_expected_completions(
        expected,
        discover_completion_records(args.completion_root),
        expected_source_hash=args.source_hash,
    )
    checks = [{"name": "expected_completions", **completion_check}]
    prereqs = []
    for value in args.prerequisite:
        gate_type, path = value.split("=", 1)
        prereqs.append((path, gate_type))
    prereq_checks, prereq_hashes = verify_prerequisites(prereqs)
    checks.extend(prereq_checks)
    gate = make_gate(
        gate_type=args.gate_type,
        checks=checks,
        config_hash=args.config_hash,
        source_hash=args.source_hash,
        prerequisite_hashes=prereq_hashes,
    )
    atomic_write_json(args.gate_output, gate, overwrite=args.overwrite)
    if not gate["passed"]:
        print(json.dumps(gate, sort_keys=True))
        return 1
    print(json.dumps({"path": str(Path(args.gate_output).resolve()), "passed": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
