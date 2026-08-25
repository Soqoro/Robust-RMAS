#!/usr/bin/env python3
"""Single canonical grid implementation for all LinkRadius entrypoints."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io_utils import content_hash, load_json
from .make_execution_manifest import verify_execution_manifest
from .schemas import EARLY_R2_EDGES, GRID_VERSION, ContractError


GPU_STAGES = frozenset(
    {
        "discover",
        "screen",
        "screen_clean",
        "clean",
        "replay",
        "causal",
        "probe",
        "probe_calibration",
        "gradient",
        "attack",
        "train",
        "val",
        "test_probe",
        "test",
    }
)

WORKFLOW_STAGES: dict[str, tuple[str, ...]] = {
    "engineering": (
        "split",
        "discover",
        "freeze_execution",
        "clean",
        "replay",
        "probe",
        "gradient",
        "validate",
        "all",
        "grid",
    ),
    "smoke": (
        "split",
        "screen",
        "freeze_execution",
        "clean",
        "causal_grid",
        "causal",
        "probe_grid",
        "probe",
        "gradient_grid",
        "gradient",
        "attack_grid",
        "attack",
        "estimate",
        "aggregate",
        "validate",
        "all",
        "grid",
    ),
    "pilot": (
        "split",
        "screen_clean_grid",
        "screen_clean",
        "freeze_execution",
        "clean_grid",
        "clean",
        "causal_grid",
        "causal",
        "probe_calibration_grid",
        "probe_calibration",
        "gradient_grid",
        "gradient",
        "freeze_probe",
        "validate_probe",
        "aggregate",
        "validate",
        "grid",
    ),
    "attacks": (
        "split",
        "freeze_execution",
        "val_grid",
        "val",
        "freeze_attack",
        "clean_grid",
        "clean",
        "test_probe_grid",
        "test_probe",
        "test_grid",
        "test",
        "thresholds",
        "analyze",
        "validate",
        "grid",
    ),
    "aggregate": ("verify", "causal", "linkradius", "attacks", "metrics", "system_curves", "all"),
    "expansion": ("r2", "r4", "rounds", "medqa", "systems", "prompts", "protection", "grid"),
}

GRID_ALIAS: dict[str, str] = {
    "causal_grid": "causal",
    "probe_grid": "probe",
    "gradient_grid": "gradient",
    "attack_grid": "attack",
    "screen_clean_grid": "screen_clean",
    "clean_grid": "clean",
    "probe_calibration_grid": "probe_calibration",
    "train_grid": "train",
    "val_grid": "val",
    "test_probe_grid": "test_probe",
    "test_grid": "test",
}


def _tokens(value: str | Sequence[Any], cast=str) -> tuple[Any, ...]:
    raw = value.split() if isinstance(value, str) else list(value)
    result = tuple(cast(item) for item in raw)
    if not result:
        raise ContractError("grid list cannot be empty")
    return result


def canonical_edge_pairs(R: int) -> tuple[tuple[str, int], ...]:
    if isinstance(R, bool) or int(R) < 1:
        raise ContractError("R must be a positive integer")
    pairs: list[tuple[str, int]] = []
    for round_idx in range(int(R)):
        pairs.extend((site, round_idx) for site in ("p2c", "c2s"))
        if round_idx < int(R) - 1:
            pairs.append(("s2p", round_idx))
    return tuple(pairs)


def parse_edge_token(token: str, R: int) -> tuple[str, int]:
    try:
        site, round_text = token.split("@", 1)
        pair = (site, int(round_text))
    except (ValueError, TypeError) as exc:
        raise ContractError(f"invalid edge token: {token!r}") from exc
    if pair not in canonical_edge_pairs(R):
        raise ContractError(f"invalid edge {token!r} for R={R}")
    return pair


@dataclass(frozen=True)
class GridTask:
    array_index: int
    workflow: str
    stage: str
    dataset: str
    R: int
    partition: str
    style: str = "sequential_light"
    method: str = "ours_recursive"
    batch_size: int = 16
    latent_length: int = 32
    runner_config_hash: str = ""
    execution_manifest_hash: str | None = None
    execution_batch_id: int | None = None
    site: str | None = None
    code_round: int | None = None
    paper_round: int | None = None
    edge_id: str | None = None
    seed: int = 42
    intervention_mode: str | None = None
    attack_family: str | None = None
    epsilon: float | None = None
    h: float | None = None
    probe_seed: int | None = None
    K: int | None = None
    subspace: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_key: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GridConfig:
    workflow: str
    stage: str
    datasets: tuple[str, ...] = ("gpqa",)
    rounds: tuple[int, ...] = (2,)
    seeds: tuple[int, ...] = (42,)
    partitions: tuple[str, ...] = ()
    num_batches: int = 1
    batch_counts: Mapping[str, int] = field(default_factory=dict)
    probe_radii: tuple[float, ...] = (1e-3, 3e-3)
    probe_seeds: tuple[int, ...] = (101, 202)
    K: int = 8
    subspace: str = "full_tensor"
    interventions: tuple[str, ...] = ("identity", "mismatch", "zero", "moment_noise")
    attack_families: tuple[str, ...] = ("random_independent", "pgd_autograd")
    attack_epsilons: tuple[float, ...] = (1e-3, 3e-3, 1e-2)
    execution_manifests: Mapping[str, str] = field(default_factory=dict)
    style: str = "sequential_light"
    method: str = "ours_recursive"
    batch_size: int = 16
    latent_length: int = 32
    discovery_batches: int = 20
    runner_config_hash: str = ""


def _runner_config_hash(config: GridConfig) -> str:
    if config.runner_config_hash:
        value = str(config.runner_config_hash)
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ContractError("runner_config_hash must be a lowercase SHA-256 digest")
        return value
    payload = asdict(config)
    payload.pop("runner_config_hash", None)
    # Paths are not scientific identity; their verified manifest content hashes
    # are already included separately in each task.
    payload.pop("execution_manifests", None)
    return content_hash(payload, domain="linkradius:grid_runner_config:v1")


def _default_partitions(workflow: str, stage: str) -> tuple[str, ...]:
    if workflow == "engineering":
        return ("validation",)
    if workflow == "smoke":
        return ("validation",)
    if workflow == "pilot":
        # Phase 3 is forbidden from loading test outcomes.
        return ("attack_train", "validation")
    if workflow == "attacks":
        if stage == "val":
            return ("validation",)
        if stage in {"freeze_execution", "clean", "test_probe", "test"}:
            return ("test",)
    return ("validation",)


def _batch_ids(config: GridConfig, partition: str) -> tuple[int, ...]:
    manifest_path = config.execution_manifests.get(partition)
    if manifest_path:
        manifest = load_json(manifest_path)
        verify_execution_manifest(manifest)
        if manifest["partition"] != partition:
            raise ContractError("execution manifest partition differs from grid partition")
        return tuple(range(len(manifest["batch_boundaries"])))
    count = int(config.batch_counts.get(partition, config.num_batches))
    if count <= 0:
        raise ContractError(f"batch count for {partition} must be positive")
    return tuple(range(count))


def _eligible_batch_ids(
    config: GridConfig,
    partition: str,
    batches: tuple[int, ...],
) -> tuple[int, ...]:
    """Keep only frozen batches containing at least one analysis row."""

    manifest_path = config.execution_manifests.get(partition)
    if not manifest_path:
        return batches
    manifest = load_json(manifest_path)
    verify_execution_manifest(manifest)
    eligibility = list(manifest["analysis_eligible"])
    selected: list[int] = []
    for batch_id in batches:
        boundary = manifest["batch_boundaries"][int(batch_id)]
        start = int(boundary["start"])
        stop = int(boundary["stop"])
        if any(bool(value) for value in eligibility[start:stop]):
            selected.append(int(batch_id))
    return tuple(selected)


def _execution_hash(config: GridConfig, partition: str) -> str | None:
    path = config.execution_manifests.get(partition)
    if not path:
        return None
    manifest = load_json(path)
    return verify_execution_manifest(manifest)


def _base_payload(
    config: GridConfig,
    *,
    workflow: str,
    stage: str,
    dataset: str,
    R: int,
    partition: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "workflow": workflow,
        "stage": stage,
        "dataset": dataset,
        "R": int(R),
        "partition": partition,
        "seed": int(seed),
        "style": config.style,
        "method": config.method,
        "batch_size": int(config.batch_size),
        "latent_length": int(config.latent_length),
        "runner_config_hash": _runner_config_hash(config),
        "execution_manifest_hash": _execution_hash(config, partition),
    }


def _edge_tokens_for(config: GridConfig, R: int, stage: str) -> tuple[str, ...]:
    all_edges = tuple(f"{site}@{round_idx}" for site, round_idx in canonical_edge_pairs(R))
    if config.workflow == "smoke":
        if R != 2:
            raise ContractError("the smoke early-edge design is frozen at R=2")
        if stage in {"causal", "probe", "attack"}:
            return EARLY_R2_EDGES
        if stage == "gradient":
            return ("c2s@1",)
    if config.workflow == "engineering":
        if R != 2:
            raise ContractError("the engineering design is frozen at R=2")
        if stage == "probe":
            return ("p2c@0",)
        if stage == "gradient":
            return ("c2s@1", "p2c@0")
    if config.workflow == "attacks":
        if R != 2:
            raise ContractError(
                "the first held-out failure-boundary experiment is frozen at R=2"
            )
        if stage in {"val", "test_probe", "test"}:
            return EARLY_R2_EDGES
    return all_edges


def _with_edge(base: dict[str, Any], edge_token: str, R: int) -> dict[str, Any]:
    site, code_round = parse_edge_token(edge_token, R)
    return {
        **base,
        "site": site,
        "code_round": code_round,
        "paper_round": code_round + 1,
        "edge_id": edge_token,
    }


def _task_payloads(config: GridConfig) -> list[dict[str, Any]]:
    if config.workflow not in WORKFLOW_STAGES:
        raise ContractError(f"unknown workflow: {config.workflow}")
    requested_stage = config.stage
    if requested_stage not in WORKFLOW_STAGES[config.workflow]:
        raise ContractError(
            f"stage {requested_stage!r} is not valid for workflow {config.workflow!r}"
        )
    stage = GRID_ALIAS.get(requested_stage, requested_stage)
    partitions = config.partitions or _default_partitions(config.workflow, stage)
    payloads: list[dict[str, Any]] = []
    if config.workflow == "aggregate":
        return [
            _base_payload(
                config,
                workflow=config.workflow,
                stage=stage,
                dataset=dataset,
                R=int(R),
                partition="global",
                seed=int(seed),
            )
            for dataset, R, seed in itertools.product(
                config.datasets, config.rounds, config.seeds
            )
        ]
    if config.workflow == "expansion":
        if stage in {"r2", "r4", "rounds"}:
            horizons = (2,) if stage == "r2" else ((4,) if stage == "r4" else config.rounds)
            for dataset, horizon, partition, seed in itertools.product(
                config.datasets, horizons, partitions, config.seeds
            ):
                base = _base_payload(config, workflow=config.workflow, stage=stage, dataset=dataset, R=int(horizon), partition=partition, seed=int(seed))
                for batch_id, (site, round_idx) in itertools.product(
                    _batch_ids(config, partition), canonical_edge_pairs(int(horizon))
                ):
                    payloads.append(
                        {
                            **_with_edge(base, f"{site}@{round_idx}", int(horizon)),
                            "execution_batch_id": batch_id,
                            "intervention_mode": "frozen_expansion_pending",
                        }
                    )
            return payloads
        if stage in {"medqa", "systems", "prompts", "protection"}:
            return [
                _base_payload(config, workflow=config.workflow, stage=stage, dataset=dataset, R=int(R), partition=partition, seed=int(seed))
                for dataset, R, partition, seed in itertools.product(
                    config.datasets, config.rounds, partitions, config.seeds
                )
            ]
    singleton_stages = {
        "split",
        "freeze_execution",
        "freeze_probe",
        "validate_probe",
        "freeze_attack",
        "thresholds",
        "analyze",
        "estimate",
        "aggregate",
        "validate",
        "verify",
        "linkradius",
        "attacks",
        "metrics",
        "system_curves",
        "medqa",
        "systems",
        "prompts",
        "protection",
    }
    if stage in singleton_stages:
        global_stages = {
            "split",
            "freeze_probe",
            "validate_probe",
            "freeze_attack",
            "thresholds",
            "analyze",
            "estimate",
            "aggregate",
            "validate",
            "verify",
            "linkradius",
            "attacks",
            "metrics",
            "system_curves",
        }
        selected_partitions = ("global",) if stage in global_stages else partitions
        return [
            _base_payload(config, workflow=config.workflow, stage=stage, dataset=dataset, R=R, partition=partition, seed=seed)
            for dataset, R, partition, seed in itertools.product(
                config.datasets, config.rounds, selected_partitions, config.seeds
            )
        ]

    for dataset, R, partition, seed in itertools.product(
        config.datasets, config.rounds, partitions, config.seeds
    ):
        base = _base_payload(config, workflow=config.workflow, stage=stage, dataset=dataset, R=int(R), partition=partition, seed=int(seed))
        batches = (
            tuple(range(int(config.discovery_batches)))
            if config.workflow == "engineering" and stage == "discover"
            else _batch_ids(config, partition)
        )
        if stage in {"discover", "screen", "screen_clean", "clean"}:
            for batch_id in batches:
                payloads.append({**base, "execution_batch_id": batch_id})
        elif stage == "replay":
            for batch_id, edge, mode in itertools.product(
                batches, _edge_tokens_for(config, R, stage), ("identity", "additive_zero")
            ):
                payloads.append(
                    {**_with_edge(base, edge, R), "execution_batch_id": batch_id, "intervention_mode": mode}
                )
        elif stage == "causal":
            for batch_id, edge, mode in itertools.product(
                batches, _edge_tokens_for(config, R, stage), config.interventions
            ):
                payloads.append(
                    {**_with_edge(base, edge, R), "execution_batch_id": batch_id, "intervention_mode": mode}
                )
        elif stage in {"probe", "probe_calibration", "test_probe"}:
            for batch_id, edge, h, probe_seed in itertools.product(
                batches,
                _edge_tokens_for(config, R, stage),
                config.probe_radii,
                config.probe_seeds,
            ):
                payloads.append(
                    {
                        **_with_edge(base, edge, R),
                        "execution_batch_id": batch_id,
                        "intervention_mode": "additive_antithetic",
                        "h": float(h),
                        "probe_seed": int(probe_seed),
                        "K": int(config.K),
                        "subspace": config.subspace,
                        "metadata": {"direction_ids": list(range(int(config.K)))},
                    }
                )
        elif stage == "gradient":
            eligible_batches = _eligible_batch_ids(config, partition, batches)
            for batch_id, edge in itertools.product(eligible_batches, _edge_tokens_for(config, R, stage)):
                payloads.append(
                    {
                        **_with_edge(base, edge, R),
                        "execution_batch_id": batch_id,
                        "intervention_mode": "continuous_consumer_autograd",
                        "subspace": config.subspace,
                    }
                )
        elif stage in {"attack", "val", "test"}:
            for family in config.attack_families:
                if config.workflow == "smoke" and family == "pgd_autograd":
                    family_edges = ("c2s@1",)
                else:
                    family_edges = _edge_tokens_for(config, R, stage)
                family_batches = _eligible_batch_ids(config, partition, batches)
                if config.workflow == "attacks":
                    # Sweep every dose under one loaded runtime.  Metadata is
                    # included in the canonical task key, so changing any
                    # budget invalidates all dependent completions.
                    for batch_id, edge in itertools.product(
                        family_batches, family_edges
                    ):
                        payloads.append(
                            {
                                **_with_edge(base, edge, R),
                                "execution_batch_id": batch_id,
                                "intervention_mode": "additive",
                                "attack_family": family,
                                "subspace": config.subspace,
                                "metadata": {
                                    "attack_epsilons": [
                                        float(value)
                                        for value in config.attack_epsilons
                                    ]
                                },
                            }
                        )
                else:
                    for batch_id, edge, epsilon in itertools.product(
                        family_batches, family_edges, config.attack_epsilons
                    ):
                        payloads.append(
                            {
                                **_with_edge(base, edge, R),
                                "execution_batch_id": batch_id,
                                "intervention_mode": "additive",
                                "attack_family": family,
                                "epsilon": float(epsilon),
                                "subspace": config.subspace,
                            }
                        )
        else:
            raise ContractError(f"no canonical grid is defined for {config.workflow}/{stage}")
    return payloads


def build_grid(config: GridConfig) -> tuple[GridTask, ...]:
    payloads = _task_payloads(config)
    tasks: list[GridTask] = []
    seen: set[str] = set()
    for array_index, payload in enumerate(payloads):
        key_payload = {"schema_version": GRID_VERSION, **payload}
        key = content_hash(key_payload, domain="linkradius:grid_task:v1")
        if key in seen:
            raise ContractError("canonical grid produced a duplicate configuration key")
        seen.add(key)
        tasks.append(GridTask(array_index=array_index, config_key=key, **payload))
    return tuple(tasks)


def select_task(tasks: Sequence[GridTask], array_index: int) -> GridTask:
    if isinstance(array_index, bool) or not 0 <= int(array_index) < len(tasks):
        maximum = len(tasks) - 1
        raise ContractError(f"array index {array_index} is out of range 0..{maximum}")
    task = tasks[int(array_index)]
    if task.array_index != int(array_index):
        raise ContractError("grid indices are not canonical and contiguous")
    return task


TSV_COLUMNS = (
    "array_index",
    "workflow",
    "stage",
    "dataset",
    "R",
    "partition",
    "style",
    "method",
    "batch_size",
    "latent_length",
    "runner_config_hash",
    "execution_manifest_hash",
    "execution_batch_id",
    "site",
    "code_round",
    "paper_round",
    "edge_id",
    "seed",
    "intervention_mode",
    "attack_family",
    "epsilon",
    "h",
    "probe_seed",
    "K",
    "subspace",
    "metadata",
    "config_key",
)


def grid_tsv(tasks: Sequence[GridTask]) -> str:
    lines = ["\t".join(TSV_COLUMNS)]
    for task in tasks:
        data = task.as_dict()
        lines.append(
            "\t".join(
                json.dumps(data.get(column), sort_keys=True)
                if column == "metadata"
                else ""
                if data.get(column) is None
                else str(data.get(column))
                for column in TSV_COLUMNS
            )
        )
    return "\n".join(lines)


def config_from_namespace(args: argparse.Namespace) -> GridConfig:
    manifests: dict[str, str] = {}
    for item in args.execution_manifest:
        try:
            partition, path = item.split("=", 1)
        except ValueError as exc:
            raise ContractError("--execution-manifest must use PARTITION=PATH") from exc
        manifests[partition] = path
    batch_counts: dict[str, int] = {}
    for item in args.batch_count:
        try:
            partition, count = item.split("=", 1)
            batch_counts[partition] = int(count)
        except ValueError as exc:
            raise ContractError("--batch-count must use PARTITION=N") from exc
    return GridConfig(
        workflow=args.workflow,
        stage=args.stage,
        datasets=_tokens(args.datasets),
        rounds=_tokens(args.rounds, int),
        seeds=_tokens(args.seeds, int),
        partitions=() if not args.partitions.strip() else _tokens(args.partitions),
        num_batches=args.num_batches,
        batch_counts=batch_counts,
        probe_radii=_tokens(args.probe_radii, float),
        probe_seeds=_tokens(args.probe_seeds, int),
        K=args.K,
        subspace=args.subspace,
        interventions=_tokens(args.interventions),
        attack_families=_tokens(args.attack_families),
        attack_epsilons=_tokens(args.attack_epsilons, float),
        execution_manifests=manifests,
        style=getattr(args, "style", "sequential_light"),
        method=getattr(args, "method", "ours_recursive"),
        batch_size=getattr(args, "batch_size", 16),
        latent_length=getattr(args, "latent_length", 32),
        discovery_batches=getattr(args, "discovery_batches", 20),
    )


def add_grid_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workflow", required=True, choices=tuple(WORKFLOW_STAGES))
    parser.add_argument("--stage", required=True)
    parser.add_argument("--datasets", default="gpqa")
    parser.add_argument("--rounds", default="2")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--partitions", default="")
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--batch-count", action="append", default=[])
    parser.add_argument("--execution-manifest", action="append", default=[])
    parser.add_argument("--probe-radii", default="1e-3 3e-3")
    parser.add_argument("--probe-seeds", default="101 202")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--subspace", default="full_tensor", choices=("full_tensor", "channel_broadcast"))
    parser.add_argument("--interventions", default="identity mismatch zero moment_noise")
    parser.add_argument("--attack-families", default="random_independent pgd_autograd")
    parser.add_argument("--attack-epsilons", default="1e-3 3e-3 1e-2")
    parser.add_argument("--discovery-batches", type=int, default=20)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_grid_arguments(parser)
    parser.add_argument("--format", choices=("tsv", "json", "count"), default="tsv")
    parser.add_argument("--index", type=int)
    args = parser.parse_args(argv)
    tasks = build_grid(config_from_namespace(args))
    if args.index is not None:
        print(json.dumps(select_task(tasks, args.index).as_dict(), sort_keys=True))
    elif args.format == "json":
        print(json.dumps([task.as_dict() for task in tasks], sort_keys=True))
    elif args.format == "count":
        print(len(tasks))
    else:
        print(f"total_tasks\t{len(tasks)}")
        print(f"max_array_index\t{len(tasks) - 1}")
        print(grid_tsv(tasks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
