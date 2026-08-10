#!/usr/bin/env python3
"""Deterministically merge validated array shards into one public JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io_utils import atomic_write_jsonl, content_hash, load_json, load_jsonl, verify_completion
from .schemas import ContractError


def _record_type(row: Mapping[str, Any]) -> str:
    return str(row.get("record_type", row.get("type", "sample"))).lower()


def validate_and_merge_shards(
    shard_paths: Sequence[str | Path],
    *,
    expected_tasks: Sequence[Mapping[str, Any]],
    key_fields: Sequence[str] = (
        "raw_sample_id",
        "edge_id",
        "intervention_mode",
        "probe_seed",
        "direction_id",
        "sign",
        "requested_epsilon",
        "attack_family",
    ),
    require_completion: bool = True,
    expected_source_hash: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_by_index = {int(task["array_index"]): dict(task) for task in expected_tasks}
    expected = {
        array_index: str(task["config_key"])
        for array_index, task in expected_by_index.items()
    }
    if len(expected) != len(expected_tasks):
        raise ContractError("expected grid contains duplicate array indices")
    seen_tasks: dict[int, Path] = {}
    sample_rows: list[tuple[int, tuple[str, ...], dict[str, Any]]] = []
    seen_sample_keys: set[tuple[str, ...]] = set()
    for raw_path in shard_paths:
        path = Path(raw_path)
        completion: Mapping[str, Any] | None = None
        if require_completion:
            completion = verify_completion(path.parent)
            declared = [
                artifact
                for artifact in completion["artifacts"]
                if str(artifact["path"]) == path.name
            ]
            if len(declared) != 1 or "row_count" not in declared[0]:
                raise ContractError(f"shard is not declared by its completion: {path}")
            if (
                expected_source_hash is not None
                and completion.get("source_hash") != expected_source_hash
            ):
                raise ContractError(f"shard completion has a stale source hash: {path}")
        rows = load_jsonl(path)
        summaries = [row for row in rows if _record_type(row) == "summary"]
        if summaries:
            raise ContractError(f"array shard contains a competing public summary: {path}")
        metadata_rows = [row for row in rows if _record_type(row) == "shard_metadata"]
        if len(metadata_rows) != 1:
            raise ContractError(f"array shard must contain exactly one shard_metadata row: {path}")
        metadata = metadata_rows[0]
        array_index = int(metadata.get("array_index", -1))
        config_key = str(metadata.get("config_key", ""))
        if array_index not in expected:
            raise ContractError(f"unexpected shard array index {array_index}: {path}")
        if expected[array_index] != config_key:
            raise ContractError(f"stale config hash for array index {array_index}: {path}")
        if completion is not None:
            manifest = load_json(path.parent / "manifest.json")
            if (
                manifest.get("task") != expected_by_index[array_index]
                or completion.get("config_hash") != config_key
                or int(completion.get("array_index", -1)) != array_index
                or int(declared[0]["row_count"]) != len(rows)
            ):
                raise ContractError(
                    f"shard completion/task identity or declared row count is stale: {path}"
                )
        if array_index in seen_tasks:
            raise ContractError(
                f"duplicate shard for array index {array_index}: {seen_tasks[array_index]}, {path}"
            )
        seen_tasks[array_index] = path
        samples = [dict(row) for row in rows if _record_type(row) not in {"shard_metadata", "summary"}]
        if int(metadata.get("row_count", -1)) != len(samples):
            raise ContractError(f"shard row count does not match metadata: {path}")
        for row in samples:
            row.setdefault("config_key", config_key)
            key = tuple(str(row.get(field, "")) for field in key_fields) + (config_key,)
            if key in seen_sample_keys:
                raise ContractError(f"duplicate sample/config key across shards: {key}")
            seen_sample_keys.add(key)
            sample_rows.append((array_index, key, row))
    missing = sorted(set(expected) - set(seen_tasks))
    if missing:
        raise ContractError(f"missing array indices: {','.join(str(value) for value in missing)}")
    ordered_rows = [row for _, _, row in sorted(sample_rows, key=lambda item: (item[0], item[1]))]
    summary = {
        "type": "summary",
        "schema_version": "linkradius.public_summary.v1",
        "num_samples": len(ordered_rows),
        "num_shards": len(seen_tasks),
        "expected_array_indices": sorted(expected),
        "ordered_row_hash": content_hash(ordered_rows, domain="linkradius:merged_rows:v1"),
    }
    return ordered_rows, summary


def merge_to_path(
    shard_paths: Sequence[str | Path],
    *,
    expected_tasks: Sequence[Mapping[str, Any]],
    output: str | Path,
    require_completion: bool = True,
    overwrite: bool = False,
    expected_source_hash: str | None = None,
) -> Path:
    rows, summary = validate_and_merge_shards(
        shard_paths,
        expected_tasks=expected_tasks,
        require_completion=require_completion,
        expected_source_hash=expected_source_hash,
    )
    return atomic_write_jsonl(output, [*rows, summary], overwrite=overwrite)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-json", required=True)
    parser.add_argument("--shard", action="append", default=[])
    parser.add_argument("--shard-list", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-completion-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected-source-hash", default="")
    args = parser.parse_args(argv)
    grid_value = load_json(args.grid_json)
    expected = grid_value.get("tasks") if isinstance(grid_value, Mapping) else grid_value
    if not isinstance(expected, list):
        raise ContractError("grid JSON must be a task list or contain a tasks list")
    paths = list(args.shard)
    if args.shard_list:
        paths.extend(
            line.strip()
            for line in Path(args.shard_list).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    merge_to_path(
        paths,
        expected_tasks=expected,
        output=args.output,
        require_completion=not args.no_completion_check,
        overwrite=args.overwrite,
        expected_source_hash=args.expected_source_hash or None,
    )
    print(json.dumps({"path": str(Path(args.output).resolve()), "shards": len(paths)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
