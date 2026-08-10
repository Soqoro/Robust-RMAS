#!/usr/bin/env python3
"""Construct and validate attack-training-only LinkRadius direction banks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io_utils import atomic_write_json, content_hash, load_json
from .schemas import ContractError


ATTACK_BANK_VERSION = "linkradius.attack_bank.v1"
TRANSFER_MAP_VERSION = "linkradius.transfer_map.v1"


def _normalize(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    if not values or not all(math.isfinite(value) for value in values):
        raise ContractError("attack vectors must be nonempty and finite")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        raise ContractError("cannot normalize a zero attack vector")
    return [value / norm for value in values]


def universal_margin_direction(vectors: Sequence[Sequence[float]]) -> list[float]:
    normalized = [_normalize(vector) for vector in vectors]
    if not normalized:
        raise ContractError("universal_margin needs at least one training vector")
    width = len(normalized[0])
    if any(len(vector) != width for vector in normalized):
        raise ContractError("training vectors have inconsistent dimensions")
    return _normalize([sum(vector[idx] for vector in normalized) for idx in range(width)])


def diffmean_direction(
    clean_vectors: Sequence[Sequence[float]], attack_vectors: Sequence[Sequence[float]]
) -> list[float]:
    if not clean_vectors or len(clean_vectors) != len(attack_vectors):
        raise ContractError("DiffMean requires paired nonempty clean and target-attack vectors")
    width = len(clean_vectors[0])
    if any(len(vector) != width for vector in [*clean_vectors, *attack_vectors]):
        raise ContractError("DiffMean vectors have inconsistent dimensions")
    deltas = [
        [float(attack[idx]) - float(clean[idx]) for idx in range(width)]
        for clean, attack in zip(clean_vectors, attack_vectors)
    ]
    return _normalize([sum(delta[idx] for delta in deltas) / len(deltas) for idx in range(width)])


def pca_direction(deltas: Sequence[Sequence[float]], *, iterations: int = 100) -> list[float]:
    if not deltas:
        raise ContractError("PCA requires at least one training delta")
    matrix = [[float(value) for value in row] for row in deltas]
    width = len(matrix[0])
    if width == 0 or any(len(row) != width for row in matrix):
        raise ContractError("PCA deltas have inconsistent dimensions")
    means = [sum(row[idx] for row in matrix) / len(matrix) for idx in range(width)]
    centered = [[row[idx] - means[idx] for idx in range(width)] for row in matrix]
    vector = _normalize([1.0 + idx / max(1, width) for idx in range(width)])
    for _ in range(iterations):
        projection = [sum(row[idx] * vector[idx] for idx in range(width)) for row in centered]
        updated = [
            sum(projection[row_idx] * centered[row_idx][idx] for row_idx in range(len(centered)))
            for idx in range(width)
        ]
        try:
            next_vector = _normalize(updated)
        except ContractError:
            raise ContractError("PCA covariance is zero") from None
        if sum((a - b) ** 2 for a, b in zip(vector, next_vector)) < 1e-24:
            vector = next_vector
            break
        vector = next_vector
    # Deterministic sign; scientific sign selection may override using attack-train margins.
    first_nonzero = next((value for value in vector if value != 0), 1.0)
    return vector if first_nonzero > 0 else [-value for value in vector]


def attack_bank_hash(bank: Mapping[str, Any]) -> str:
    payload = dict(bank)
    payload.pop("content_hash", None)
    return content_hash(payload, domain="linkradius:attack_bank:v1")


def build_bank(
    *,
    family: str,
    direction: Sequence[float],
    training_raw_ids: Sequence[str],
    split_manifest_hash: str,
    execution_manifest_hash: str,
    edge_id: str,
    trained_R: int,
    scorer_hash: str,
    subspace: Mapping[str, Any],
    source_hash: str,
    hyperparameters: Mapping[str, Any],
) -> dict[str, Any]:
    ids = [str(value) for value in training_raw_ids]
    if not ids or len(ids) != len(set(ids)):
        raise ContractError("attack-bank training raw IDs must be nonempty and unique")
    unit = _normalize(direction)
    bank: dict[str, Any] = {
        "schema_version": ATTACK_BANK_VERSION,
        "family": family,
        "training_partition": "attack_train",
        "training_raw_ids": ids,
        "training_raw_id_hash": content_hash(ids, domain="linkradius:attack_bank_training_ids:v1"),
        "split_manifest_hash": split_manifest_hash,
        "execution_manifest_hash": execution_manifest_hash,
        "edge_id": edge_id,
        "trained_R": int(trained_R),
        "scorer_hash": scorer_hash,
        "subspace": dict(subspace),
        "direction": unit,
        "direction_norm": math.sqrt(sum(value * value for value in unit)),
        "source_hash": source_hash,
        "hyperparameters": dict(hyperparameters),
    }
    bank["content_hash"] = attack_bank_hash(bank)
    return bank


def transfer_map_hash(transfer_map: Mapping[str, Any]) -> str:
    payload = dict(transfer_map)
    payload.pop("content_hash", None)
    return content_hash(payload, domain="linkradius:transfer_map:v1")


def validate_bank_for_evaluation(
    bank: Mapping[str, Any],
    *,
    eval_raw_ids: Iterable[str],
    eval_partition: str,
    split_manifest_hash: str,
    execution_manifest_hash: str,
    edge_id: str,
    eval_R: int,
    scorer_hash: str,
    subspace: Mapping[str, Any],
    transfer_map: Mapping[str, Any] | None = None,
) -> None:
    if bank.get("schema_version") != ATTACK_BANK_VERSION:
        raise ContractError("unsupported attack-bank schema")
    if bank.get("training_partition") != "attack_train":
        raise ContractError("learned attack banks must be trained only on attack_train")
    if bank.get("content_hash") != attack_bank_hash(bank):
        raise ContractError("attack-bank content hash is stale")
    overlap = set(str(value) for value in bank["training_raw_ids"]) & set(
        str(value) for value in eval_raw_ids
    )
    if eval_partition != "attack_train" and overlap:
        raise ContractError(f"attack-bank split leakage: {len(overlap)} training/evaluation IDs overlap")
    for name, expected in (
        ("split_manifest_hash", split_manifest_hash),
        ("scorer_hash", scorer_hash),
    ):
        if bank.get(name) != expected:
            raise ContractError(f"attack-bank {name} mismatch")
    if bank.get("subspace") != dict(subspace):
        raise ContractError("attack-bank subspace mismatch")
    trained_R = int(bank["trained_R"])
    if trained_R == int(eval_R):
        if bank.get("edge_id") != edge_id:
            raise ContractError("attack-bank edge mismatch")
        if eval_partition == "attack_train" and bank.get("execution_manifest_hash") != execution_manifest_hash:
            raise ContractError("attack-bank training execution manifest mismatch")
        return
    if transfer_map is None:
        raise ContractError("cross-horizon attack-bank loading requires a frozen transfer map")
    if transfer_map.get("schema_version") != TRANSFER_MAP_VERSION:
        raise ContractError("unsupported transfer-map schema")
    if transfer_map.get("content_hash") != transfer_map_hash(transfer_map):
        raise ContractError("transfer-map hash is missing or stale")
    mappings = transfer_map.get("mappings", [])
    allowed = any(
        int(item.get("trained_R", -1)) == trained_R
        and int(item.get("eval_R", -1)) == int(eval_R)
        and item.get("trained_edge") == bank.get("edge_id")
        and item.get("eval_edge") == edge_id
        for item in mappings
    )
    if not allowed:
        raise ContractError("cross-horizon edge transfer is not present in the frozen transfer map")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="JSON bank metadata and training vectors")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    spec = load_json(args.spec)
    family = spec["family"]
    if family == "universal_margin":
        direction = universal_margin_direction(spec["vectors"])
    elif family == "diffmean":
        direction = diffmean_direction(spec["clean_vectors"], spec["attack_vectors"])
    elif family == "pca":
        direction = pca_direction(spec["deltas"])
        if float(spec.get("sign_score", 1.0)) < 0:
            direction = [-value for value in direction]
    else:
        raise ContractError(f"unsupported learned bank family: {family}")
    metadata = {key: value for key, value in spec.items() if key not in {"vectors", "clean_vectors", "attack_vectors", "deltas", "sign_score"}}
    bank = build_bank(direction=direction, **metadata)
    atomic_write_json(args.output, bank, overwrite=args.overwrite)
    print(json.dumps({"path": str(Path(args.output).resolve()), "content_hash": bank["content_hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
