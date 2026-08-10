#!/usr/bin/env python3
"""Create the immutable raw GPQA Diamond 40/20/40 split manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io_utils import atomic_write_json, content_hash, load_json
from .schemas import PARTITIONS, SPLIT_MANIFEST_VERSION, ContractError, validate_split_manifest


DEFAULT_DATASET = "Idavidrein/gpqa"
DEFAULT_CONFIG = "gpqa_diamond"
DEFAULT_SOURCE_SPLIT = "train"
DEFAULT_SEED = 42
DEFAULT_RATIOS = (0.4, 0.2, 0.4)


def largest_remainder_counts(total: int, ratios: Sequence[float]) -> list[int]:
    from RecursiveMAS.inference_utils.linkradius import largest_remainder_counts as canonical

    return list(canonical(total, ratios))


def _raw_id(record: Any) -> str:
    if isinstance(record, str):
        value = record
    elif isinstance(record, Mapping):
        value = record.get("raw_sample_id", record.get("id", ""))
    else:
        value = getattr(record, "raw_sample_id", "")
    value = str(value or "").strip()
    if not value:
        raise ContractError("every raw record must have a stable raw_sample_id")
    return value


def split_raw_ids(
    raw_ids: Iterable[str], *, seed: int = DEFAULT_SEED, ratios: Sequence[float] = DEFAULT_RATIOS
) -> dict[str, list[str]]:
    from RecursiveMAS.inference_utils.linkradius import split_raw_ids as canonical

    result = canonical(list(raw_ids), seed=seed, ratios=ratios, partition_names=PARTITIONS)
    return {name: list(result[name]) for name in PARTITIONS}


def split_manifest_hash(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("content_hash", None)
    return content_hash(payload, domain="linkradius:split_manifest:v1")


def build_split_manifest(
    raw_records: Sequence[Any],
    *,
    seed: int = DEFAULT_SEED,
    dataset: str = DEFAULT_DATASET,
    dataset_config: str = DEFAULT_CONFIG,
    source_split: str = DEFAULT_SOURCE_SPLIT,
    ratios: Sequence[float] = DEFAULT_RATIOS,
) -> dict[str, Any]:
    ids = [_raw_id(record) for record in raw_records]
    split = split_raw_ids(ids, seed=seed, ratios=ratios)
    by_id = {_raw_id(record): record for record in raw_records}
    partitions: dict[str, list[dict[str, Any]]] = {}
    for name in PARTITIONS:
        rows: list[dict[str, Any]] = []
        for raw_id in split[name]:
            source = by_id[raw_id]
            if isinstance(source, Mapping):
                raw_index = source.get("raw_index")
            else:
                raw_index = getattr(source, "raw_index", None)
            row: dict[str, Any] = {"raw_sample_id": raw_id}
            if raw_index is not None:
                row["raw_index"] = int(raw_index)
            rows.append(row)
        partitions[name] = rows
    counts = {name: len(partitions[name]) for name in PARTITIONS}
    manifest: dict[str, Any] = {
        "schema_version": SPLIT_MANIFEST_VERSION,
        "dataset": dataset,
        "dataset_config": dataset_config,
        "source_split": source_split,
        "seed": int(seed),
        "ratios": {name: float(ratios[idx]) for idx, name in enumerate(PARTITIONS)},
        "allocation_algorithm": "largest_remainder_stable_partition_order_v1",
        "shuffle_algorithm": "python_random_seeded_sorted_raw_ids_v1",
        "num_records": len(ids),
        "counts": counts,
        "partitions": partitions,
    }
    manifest["content_hash"] = split_manifest_hash(manifest)
    validate_split_manifest(manifest)
    return manifest


def verify_split_manifest(manifest: Mapping[str, Any]) -> str:
    validate_split_manifest(manifest)
    expected = split_manifest_hash(manifest)
    if manifest.get("content_hash") != expected:
        raise ContractError("split manifest content_hash is missing or stale")
    return expected


def _mapping_from_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"cannot convert GPQA record to mapping: {type(value).__name__}")


def load_gpqa_raw_records(
    *, dataset: str = DEFAULT_DATASET, config: str = DEFAULT_CONFIG, split: str = DEFAULT_SOURCE_SPLIT
) -> list[dict[str, Any]]:
    """Load raw records lazily; no dataset access occurs in grid modes."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("the split stage requires the 'datasets' package") from exc
    try:
        from RecursiveMAS.inference_utils.linkradius import build_gpqa_raw_record
    except ImportError as exc:
        raise RuntimeError("LinkRadius core GPQA identity utilities are unavailable") from exc
    source = load_dataset(dataset, config, split=split)
    records: list[dict[str, Any]] = []
    for raw_index, row in enumerate(source):
        built = build_gpqa_raw_record(row, raw_index=raw_index, seed=DEFAULT_SEED)
        records.append(_mapping_from_object(built))
    return records


def create_or_verify(
    output: str | Path,
    manifest: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> str:
    path = Path(output)
    expected_hash = verify_split_manifest(manifest)
    if path.exists() and not overwrite:
        existing = load_json(path)
        existing_hash = verify_split_manifest(existing)
        if existing_hash != expected_hash:
            raise ContractError(
                f"existing split manifest is incompatible: {existing_hash} != {expected_hash}"
            )
        return existing_hash
    atomic_write_json(path, manifest, overwrite=True)
    return expected_hash


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--records-json", default="", help="CPU fixture: JSON list of stable raw records")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_CONFIG)
    parser.add_argument("--source-split", default=DEFAULT_SOURCE_SPLIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.records_json:
        value = load_json(args.records_json)
        if not isinstance(value, list):
            raise ContractError("--records-json must contain a JSON list")
        records = value
    else:
        records = load_gpqa_raw_records(
            dataset=args.dataset, config=args.dataset_config, split=args.source_split
        )
    manifest = build_split_manifest(
        records,
        seed=args.seed,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        source_split=args.source_split,
    )
    digest = create_or_verify(args.output, manifest, overwrite=args.overwrite)
    print(json.dumps({"path": str(Path(args.output).resolve()), "content_hash": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
