#!/usr/bin/env python3
"""Freeze execution rows, batch boundaries, eligibility, and whole-batch shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io_utils import atomic_write_json, content_hash, load_json, load_jsonl, ordered_hash
from .make_split_manifest import verify_split_manifest
from .schemas import EXECUTION_MANIFEST_VERSION, ContractError, validate_execution_manifest


def execution_manifest_hash(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("content_hash", None)
    return content_hash(payload, domain="linkradius:execution_manifest:v1")


def _partition_rows(split_manifest: Mapping[str, Any], partition: str) -> list[Mapping[str, Any]]:
    if partition not in split_manifest["partitions"]:
        raise ContractError(f"unknown split partition: {partition}")
    result = []
    for value in split_manifest["partitions"][partition]:
        result.append(value if isinstance(value, Mapping) else {"raw_sample_id": value})
    return result


def _batch_boundaries(n: int, batch_size: int) -> list[dict[str, int]]:
    return [
        {"execution_batch_id": idx, "start": start, "stop": min(start + batch_size, n)}
        for idx, start in enumerate(range(0, n, batch_size))
    ]


def _array_shards(num_batches: int, batches_per_shard: int) -> list[dict[str, Any]]:
    return [
        {
            "array_index": idx,
            "execution_batch_ids": list(
                range(start, min(start + batches_per_shard, num_batches))
            ),
        }
        for idx, start in enumerate(range(0, num_batches, batches_per_shard))
    ]


def build_execution_manifest(
    *,
    split_manifest: Mapping[str, Any],
    partition: str,
    screening_rows: Sequence[Mapping[str, Any]],
    batch_size: int,
    batches_per_shard: int = 1,
    padding_policy: str = "release_tokenizer_longest_in_frozen_batch_v1",
    screening_config_hash: str,
    screening_run_hash: str = "",
    retain_all_partition_rows: bool = True,
) -> dict[str, Any]:
    split_hash = verify_split_manifest(split_manifest)
    if batch_size <= 0 or batches_per_shard <= 0:
        raise ContractError("batch_size and batches_per_shard must be positive")
    partition_rows = _partition_rows(split_manifest, partition)
    partition_ids = [str(row["raw_sample_id"]) for row in partition_rows]
    screening_by_id: dict[str, Mapping[str, Any]] = {}
    screening_order: list[str] = []
    for row in screening_rows:
        raw_id = str(row.get("raw_sample_id", "")).strip()
        if not raw_id:
            raise ContractError("screening row is missing raw_sample_id")
        if raw_id in screening_by_id:
            raise ContractError(f"duplicate screening raw_sample_id: {raw_id}")
        if raw_id not in set(partition_ids):
            raise ContractError(f"screening row is outside partition {partition}: {raw_id}")
        screening_by_id[raw_id] = row
        screening_order.append(raw_id)
    if retain_all_partition_rows:
        ordered_raw_ids = partition_ids
    else:
        ordered_raw_ids = screening_order
    if not ordered_raw_ids:
        raise ContractError("execution manifest cannot be empty")

    ordered_sample_ids: list[str] = []
    eligibility: list[bool] = []
    dual_correct_values: list[bool | None] = []
    reasons: list[str] = []
    raw_indices: list[int | None] = []
    partition_by_id = {str(row["raw_sample_id"]): row for row in partition_rows}
    for raw_id in ordered_raw_ids:
        row = screening_by_id.get(raw_id)
        partition_row = partition_by_id[raw_id]
        sample_id = str((row or {}).get("sample_id", raw_id)).strip()
        if not sample_id:
            raise ContractError(f"empty sample_id for raw ID {raw_id}")
        is_eligible = bool((row or {}).get("analysis_eligible", False))
        dual_correct = None if row is None else bool(row.get("dual_correct", False))
        reason = str((row or {}).get("exclusion_reason", "")).strip()
        if row is None:
            reason = "not_screened"
        elif not is_eligible and not reason:
            reason = "not_dual_correct"
        if is_eligible:
            reason = ""
        ordered_sample_ids.append(sample_id)
        eligibility.append(is_eligible)
        dual_correct_values.append(dual_correct)
        reasons.append(reason)
        raw_index = (row or {}).get("raw_index", partition_row.get("raw_index"))
        raw_indices.append(None if raw_index is None else int(raw_index))

    boundaries = _batch_boundaries(len(ordered_raw_ids), int(batch_size))
    manifest: dict[str, Any] = {
        "schema_version": EXECUTION_MANIFEST_VERSION,
        "split_manifest_hash": split_hash,
        "partition": partition,
        "ordered_raw_sample_ids": ordered_raw_ids,
        "ordered_sample_ids": ordered_sample_ids,
        "raw_indices": raw_indices,
        "batch_size": int(batch_size),
        "batch_boundaries": boundaries,
        "padding_policy": padding_policy,
        "array_shards": _array_shards(len(boundaries), int(batches_per_shard)),
        "analysis_eligible": eligibility,
        "screening_dual_correct": dual_correct_values,
        "exclusion_reasons": reasons,
        "screening_config_hash": screening_config_hash,
        "screening_run_hash": screening_run_hash,
        "retained_filler_rows": int(len(ordered_raw_ids) - sum(eligibility)),
        "ordered_cohort_hash": ordered_hash(
            ordered_raw_ids, domain="linkradius:ordered_execution_cohort:v1"
        ),
        "batch_boundary_hash": content_hash(
            boundaries, domain="linkradius:batch_boundaries:v1"
        ),
    }
    manifest["content_hash"] = execution_manifest_hash(manifest)
    validate_execution_manifest(manifest)
    return manifest


def verify_execution_manifest(
    manifest: Mapping[str, Any], *, split_manifest: Mapping[str, Any] | None = None
) -> str:
    validate_execution_manifest(manifest)
    expected = execution_manifest_hash(manifest)
    if manifest.get("content_hash") != expected:
        raise ContractError("execution manifest content_hash is missing or stale")
    if split_manifest is not None and manifest["split_manifest_hash"] != verify_split_manifest(split_manifest):
        raise ContractError("execution manifest references a different split manifest")
    return expected


def batch_rows(manifest: Mapping[str, Any], execution_batch_id: int) -> range:
    validate_execution_manifest(manifest)
    boundaries = manifest["batch_boundaries"]
    if not 0 <= int(execution_batch_id) < len(boundaries):
        raise ContractError(f"execution batch index out of range: {execution_batch_id}")
    boundary = boundaries[int(execution_batch_id)]
    return range(int(boundary["start"]), int(boundary["stop"]))


def shard_batch_ids(manifest: Mapping[str, Any], array_index: int) -> tuple[int, ...]:
    validate_execution_manifest(manifest)
    shards = manifest["array_shards"]
    if not 0 <= int(array_index) < len(shards):
        raise ContractError(f"array shard index out of range: {array_index}")
    return tuple(int(value) for value in shards[int(array_index)]["execution_batch_ids"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--screening-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--batches-per-shard", type=int, default=1)
    parser.add_argument("--screening-config-hash", required=True)
    parser.add_argument("--screening-run-hash", default="")
    parser.add_argument("--selected-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    split = load_json(args.split_manifest)
    manifest = build_execution_manifest(
        split_manifest=split,
        partition=args.partition,
        screening_rows=load_jsonl(args.screening_jsonl),
        batch_size=args.batch_size,
        batches_per_shard=args.batches_per_shard,
        screening_config_hash=args.screening_config_hash,
        screening_run_hash=args.screening_run_hash,
        retain_all_partition_rows=not args.selected_only,
    )
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        existing = load_json(output)
        if verify_execution_manifest(existing, split_manifest=split) != manifest["content_hash"]:
            raise ContractError("refusing to replace incompatible execution manifest")
    else:
        atomic_write_json(output, manifest, overwrite=True)
    print(json.dumps({"path": str(output.resolve()), "content_hash": manifest["content_hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
