#!/usr/bin/env python3
"""Dedicated LinkRadius runner with pre-model stage and provenance gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Support the exact absolute-script invocation from an arbitrary submission cwd
# with PYTHONPATH unset.  Package imports happen only after this single bootstrap.
if __package__ in {None, ""}:  # pragma: no cover - covered by subprocess tests.
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    __package__ = "experiments.linkradius"

from .grid import GPU_STAGES, GRID_ALIAS, WORKFLOW_STAGES, GridConfig, add_grid_arguments, build_grid, canonical_edge_pairs, config_from_namespace, grid_tsv, select_task
from .io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    atomic_write_csv,
    compatible_complete,
    content_hash,
    file_sha256,
    load_json,
    load_jsonl,
    publish_completion as _publish_completion_immediate,
    require_passed_gate,
    source_hash,
    verify_completion,
)
from .make_execution_manifest import build_execution_manifest, verify_execution_manifest
from .make_split_manifest import build_split_manifest, create_or_verify, load_gpqa_raw_records, verify_split_manifest
from .schemas import (
    EARLY_R2_EDGES,
    ContractError,
    validate_completion_record,
    validate_intervention_row,
)
from .select_clean_correct import annotate_screening_rows
from .validate_stage import make_gate


PHASE_DIR = {
    "engineering": "engineering",
    "smoke": "smoke",
    "pilot": "pilot",
    "attacks": "attacks",
    "aggregate": "aggregate",
    "expansion": "expansion",
}


_DEFER_COMPLETION = False


def publish_completion(
    output_dir: str | os.PathLike[str],
    *,
    config_hash: str,
    source_hash_value: str,
    artifact_paths: Sequence[str | os.PathLike[str]],
    row_counts: Mapping[str, int] | None = None,
    extra: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Publish immediately for direct Python use, or stage a shell finalization."""

    if not _DEFER_COMPLETION:
        return _publish_completion_immediate(
            output_dir,
            config_hash=config_hash,
            source_hash_value=source_hash_value,
            artifact_paths=artifact_paths,
            row_counts=row_counts,
            extra=extra,
            overwrite=overwrite,
        )
    directory = Path(output_dir)
    normalized_paths: list[str] = []
    for raw_path in artifact_paths:
        path = Path(raw_path)
        if path.is_absolute():
            if path.parent.resolve() != directory.resolve():
                raise ContractError("deferred completion artifacts must be direct task children")
            normalized_paths.append(path.name)
        else:
            normalized_paths.append(path.as_posix())
    pending = {
        "schema_version": "linkradius.deferred_completion.v1",
        "config_hash": str(config_hash),
        "source_hash": str(source_hash_value),
        "artifact_paths": normalized_paths,
        "row_counts": {str(key): int(value) for key, value in (row_counts or {}).items()},
        "extra": dict(extra or {}),
        "overwrite": bool(overwrite),
    }
    atomic_write_json(
        directory / ".completion.pending.json", pending, overwrite=overwrite
    )
    return directory / ".completion.pending.json"


def _finalize_deferred_completion(output_dir: str | os.PathLike[str]) -> Mapping[str, Any]:
    """Finalize only after the shell's tee process has closed its pending log."""

    directory = Path(output_dir).resolve()
    pending_path = directory / ".completion.pending.json"
    log_pending = directory / ".run.log.pending"
    launcher_pending = directory / ".launcher_command.pending.txt"
    if not pending_path.is_file():
        if (directory / ".complete.json").is_file():
            record = verify_completion(directory)
            for temporary in (log_pending, launcher_pending):
                if temporary.is_file():
                    temporary.unlink()
            return {"status": "reused_complete", "config_hash": record["config_hash"]}
        raise ContractError("deferred completion has no pending publication record")
    pending = load_json(pending_path)
    if pending.get("schema_version") != "linkradius.deferred_completion.v1":
        raise ContractError("unsupported deferred completion record")
    if source_hash(Path(__file__).resolve().parents[2]) != pending.get("source_hash"):
        raise ContractError("deferred completion source changed before finalization")
    if not log_pending.is_file() or not launcher_pending.is_file():
        raise ContractError("deferred completion requires closed run and launcher logs")
    os.replace(log_pending, directory / "run.log")
    os.replace(launcher_pending, directory / "launcher_command.txt")
    required = [
        *[str(value) for value in pending["artifact_paths"]],
        "warnings.txt",
        "run.log",
        "launcher_command.txt",
    ]
    artifact_paths = list(dict.fromkeys(required))
    completion_path = _publish_completion_immediate(
        directory,
        config_hash=str(pending["config_hash"]),
        source_hash_value=str(pending["source_hash"]),
        artifact_paths=artifact_paths,
        row_counts={str(key): int(value) for key, value in pending.get("row_counts", {}).items()},
        extra=dict(pending.get("extra", {})),
        overwrite=bool(pending.get("overwrite", False)),
    )
    pending_path.unlink()
    return {
        "status": "finalized_complete",
        "completion": str(completion_path),
        "config_hash": pending["config_hash"],
    }

GRID_DEFAULT_STAGE = {
    "engineering": "probe",
    "smoke": "probe",
    "pilot": "probe_calibration",
    "attacks": "val",
    "expansion": "r2",
}


# A verification gate covers a complete source workflow, not an operator-picked
# subset.  This prevents ``VERIFY_STAGES=clean`` from authorizing unrelated CPU
# summaries.
REQUIRED_AGGREGATE_STAGES = {
    "engineering": (
        "split",
        "discover",
        "freeze_execution",
        "clean",
        "replay",
        "probe",
        "gradient",
        "validate",
    ),
    "smoke": (
        "split",
        "screen",
        "freeze_execution",
        "clean",
        "causal",
        "probe",
        "gradient",
        "attack",
        "estimate",
        "aggregate",
        "validate",
    ),
    "pilot": (
        "split",
        "screen_clean",
        "freeze_execution",
        "clean",
        "causal",
        "probe_calibration",
        "gradient",
        "freeze_probe",
        "validate_probe",
        "aggregate",
    ),
    "attacks": (
        "split",
        "freeze_execution",
        "val",
        "freeze_attack",
        "clean",
        "test_probe",
        "test",
        "thresholds",
        "analyze",
        "validate",
    ),
}


def _bool_env(value: str | int | bool) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_plain(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _bind_probe_run_ids(
    diagnostics: Any,
    *,
    run_id: str | None = None,
    plus_run_id: str | None = None,
    minus_run_id: str | None = None,
) -> dict[str, Any]:
    """Replace runtime-local probe IDs with canonical task-bound row IDs."""

    plain = _to_plain(diagnostics)
    if not isinstance(plain, Mapping):
        raise ContractError("probe diagnostics must be a mapping")
    result = dict(plain)
    for field, value in (
        ("run_id", run_id),
        ("plus_run_id", plus_run_id),
        ("minus_run_id", minus_run_id),
    ):
        if value is not None:
            if not str(value):
                raise ContractError(f"canonical probe {field} must be nonempty")
            result[field] = str(value)
    return result


def _cached_source_hash(args: argparse.Namespace, repo_root: Path) -> str:
    """Compute the immutable per-process experiment source identity once."""

    value = getattr(args, "_current_source_hash", None)
    if value is None:
        value = source_hash(repo_root)
        args._current_source_hash = value
    return str(value)


def _phase_root(args: argparse.Namespace) -> Path:
    return Path(args.out_root).resolve() / PHASE_DIR[args.workflow]


def _producer_namespace(
    args: argparse.Namespace, workflow: str | None
) -> argparse.Namespace:
    """Return an isolated namespace for a cross-phase authenticated producer.

    Reconstructing a producer grid must never mutate the consumer's arguments:
    those arguments are themselves part of canonical task identity.
    """

    if workflow is None or workflow == args.workflow:
        return args
    if workflow not in PHASE_DIR:
        raise ContractError(f"unknown producer workflow: {workflow}")
    producer = argparse.Namespace(**vars(args))
    producer.workflow = workflow
    producer.stage = "grid"
    producer._current_source_hash = getattr(args, "_current_source_hash", None)
    return producer


def _execution_manifest_path(
    args: argparse.Namespace, partition: str, task: Mapping[str, Any] | None = None
) -> str:
    if args.execution_manifest_path:
        return str(args.execution_manifest_path)
    for item in args.execution_manifest:
        if "=" not in item:
            raise ContractError("--execution-manifest must use PARTITION=PATH")
        candidate_partition, path = item.split("=", 1)
        if candidate_partition == partition:
            return path
    dataset = str(task["dataset"]) if task is not None else str(args.datasets).split()[0]
    round_value = int(task["R"]) if task is not None else int(str(args.rounds).split()[0])
    candidate = (
        _phase_root(args)
        / dataset
        / f"R{round_value}"
        / partition
        / "execution_manifest.json"
    )
    return str(candidate) if candidate.is_file() else ""


def task_output_dir(args: argparse.Namespace, task: Mapping[str, Any]) -> Path:
    edge_token = str(task.get("edge_id") or "global").replace("@", "_r")
    return (
        _phase_root(args)
        / str(task["dataset"])
        / f"R{int(task['R'])}"
        / str(task["partition"])
        / str(task["stage"])
        / edge_token
        / str(task["config_key"])
    )


def _single_stage_task(
    args: argparse.Namespace,
    task: Mapping[str, Any],
    stage: str,
    partition: str,
) -> Mapping[str, Any]:
    matches = [
        candidate.as_dict()
        for candidate in build_grid(_build_grid_config(args, stage))
        if candidate.dataset == str(task["dataset"])
        and int(candidate.R) == int(task["R"])
        and candidate.partition == partition
    ]
    if len(matches) != 1:
        raise ContractError(f"expected one canonical {stage}/{partition} task")
    return matches[0]


def _authenticated_split_manifest(
    args: argparse.Namespace,
    task: Mapping[str, Any],
    repo_root: Path,
) -> tuple[Mapping[str, Any], str]:
    if not args.split_manifest:
        raise ContractError("a canonical split manifest path is required")
    canonical_path = Path(args.split_manifest)
    canonical = load_json(canonical_path)
    canonical_hash = verify_split_manifest(canonical)
    split_task = _single_stage_task(args, task, "split", "global")
    split_dir = task_output_dir(args, split_task)
    completion = verify_completion(
        split_dir, expected_config_hash=str(split_task["config_key"])
    )
    if completion.get("source_hash") != _cached_source_hash(args, repo_root):
        raise ContractError("split completion has a stale source hash")
    frozen_copy = load_json(split_dir / "split_manifest.json")
    pointer = load_json(split_dir / "split_result.json")
    if (
        verify_split_manifest(frozen_copy) != canonical_hash
        or pointer.get("split_manifest_hash") != canonical_hash
        or Path(str(pointer.get("split_manifest_path"))).resolve()
        != canonical_path.resolve()
        or completion.get("split_manifest_hash") != canonical_hash
    ):
        raise ContractError("canonical split manifest differs from its authenticated freeze")
    return canonical, canonical_hash


def _authenticated_execution_manifest(
    args: argparse.Namespace,
    partition: str,
    task: Mapping[str, Any],
    repo_root: Path,
) -> tuple[Path, Mapping[str, Any], str]:
    path_text = _execution_manifest_path(args, partition, task)
    if not path_text:
        raise ContractError(f"missing frozen execution manifest for {partition}")
    canonical_path = Path(path_text)
    split, _ = _authenticated_split_manifest(args, task, repo_root)
    canonical = load_json(canonical_path)
    canonical_hash = verify_execution_manifest(canonical, split_manifest=split)
    freeze_task = _single_stage_task(args, task, "freeze_execution", partition)
    freeze_dir = task_output_dir(args, freeze_task)
    completion = verify_completion(
        freeze_dir, expected_config_hash=str(freeze_task["config_key"])
    )
    if completion.get("source_hash") != _cached_source_hash(args, repo_root):
        raise ContractError("execution-freeze completion has a stale source hash")
    frozen_copy = load_json(freeze_dir / "execution_manifest.json")
    pointer = load_json(freeze_dir / "freeze_execution_result.json")
    if (
        verify_execution_manifest(frozen_copy, split_manifest=split) != canonical_hash
        or pointer.get("content_hash") != canonical_hash
        or Path(str(pointer.get("path"))).resolve() != canonical_path.resolve()
        or completion.get("execution_manifest_hash") != canonical_hash
    ):
        raise ContractError("canonical execution manifest differs from its authenticated freeze")
    return canonical_path, canonical, canonical_hash


def _task_manifest(args: argparse.Namespace, task: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    manifest = {
        "schema_version": "linkradius.task_manifest.v1",
        "task": dict(task),
        "style": args.style,
        "method": args.method,
        "batch_size": args.batch_size,
        "latent_length": args.latent_length,
        "role_devices": _resolved_role_devices(args),
        "relay_transfer_mode": str(
            getattr(args, "relay_transfer_mode", "cpu_staged")
        ),
        "autograd_memory_mode": str(
            getattr(args, "autograd_memory_mode", "none")
        ),
        # Scheduler placement is diagnostic rather than scientific identity:
        # logical role devices are hashed above, while physical allocation may
        # change safely between otherwise identical jobs.
        "scheduler_environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        },
        "source_hash": _cached_source_hash(args, repo_root),
        "argv": list(sys.argv),
        "cwd": str(Path.cwd()),
        "python": sys.executable,
    }
    if str(task.get("stage")) in GPU_STAGES:
        manifest["runtime_environment"] = _runtime_environment_identity()
    return manifest


def _resolved_role_devices(args: argparse.Namespace) -> dict[str, str]:
    fallback = str(args.device).strip()
    if not fallback:
        raise ContractError("--device must not be empty")
    result: dict[str, str] = {}
    for role in ("planner", "critic", "solver"):
        explicit = str(getattr(args, f"{role}_device", "") or "").strip()
        result[role] = explicit or fallback
    terminal_explicit = str(
        getattr(args, "terminal_solver_device", "") or ""
    ).strip()
    result["terminal_solver"] = terminal_explicit or result["solver"]
    return result


def _gate_default(args: argparse.Namespace, name: str) -> Path:
    explicit = getattr(args, name.replace("-", "_"), "")
    if explicit:
        return Path(explicit)
    filename = name.replace("_path", "")
    return Path(args.out_root).resolve() / f"{filename}.json"


def _configured_artifact_identity(path_value: str) -> Mapping[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value)
    return {
        "configured_path": str(path),
        "sha256": file_sha256(path) if path.is_file() else None,
    }


_RUNTIME_DISTRIBUTIONS = (
    "torch",
    "transformers",
    "triton",
    "fla-core",
    "flash-linear-attention",
    "causal-conv1d",
    "einops",
    "ninja",
)


def _runtime_environment_identity() -> dict[str, Any]:
    """Return the software identity that can change model numerics.

    Physical CUDA allocation is intentionally excluded: a clean GPU on a
    different node is a safe placement change.  Python and inference-backend
    packages are included so artifacts from the fallback and FLA paths cannot
    be mixed under one scientific task identity.
    """

    distributions: dict[str, str | None] = {}
    for name in _RUNTIME_DISTRIBUTIONS:
        try:
            distributions[name] = package_version(name)
        except PackageNotFoundError:
            distributions[name] = None
    return {
        "schema_version": "linkradius.runtime_environment.v1",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "distributions": distributions,
    }


UPSTREAM_COMPLETION_STAGES: Mapping[tuple[str, str], tuple[str, ...]] = {
    ("engineering", "freeze_execution"): ("discover",),
    ("engineering", "replay"): ("clean",),
    ("engineering", "probe"): ("clean",),
    ("engineering", "gradient"): ("clean",),
    ("engineering", "validate"): ("clean", "replay", "probe", "gradient"),
    ("smoke", "freeze_execution"): ("screen",),
    ("smoke", "causal"): ("clean",),
    ("smoke", "probe"): ("clean",),
    ("smoke", "gradient"): ("clean",),
    ("smoke", "attack"): ("clean",),
    ("smoke", "estimate"): ("probe",),
    ("smoke", "aggregate"): ("causal",),
    ("smoke", "validate"): (
        "clean",
        "causal",
        "probe",
        "gradient",
        "attack",
        "estimate",
        "aggregate",
    ),
    ("pilot", "freeze_execution"): ("screen_clean",),
    ("pilot", "causal"): ("clean",),
    ("pilot", "probe_calibration"): ("clean",),
    ("pilot", "gradient"): ("clean",),
    ("pilot", "freeze_probe"): ("causal", "probe_calibration"),
    ("pilot", "validate_probe"): ("clean", "causal", "probe_calibration", "gradient", "freeze_probe"),
    ("pilot", "validate"): ("clean", "causal", "probe_calibration", "gradient", "freeze_probe"),
    ("pilot", "aggregate"): ("causal", "probe_calibration", "freeze_probe", "validate_probe"),
    ("attacks", "freeze_attack"): ("freeze_execution", "val"),
    ("attacks", "clean"): ("freeze_attack",),
    ("attacks", "test_probe"): ("clean",),
    ("attacks", "test"): ("clean",),
    ("attacks", "thresholds"): ("clean", "test"),
    ("attacks", "analyze"): ("clean", "test_probe", "test", "thresholds"),
    ("attacks", "validate"): (
        "freeze_execution",
        "val",
        "freeze_attack",
        "clean",
        "test_probe",
        "test",
        "thresholds",
        "analyze",
    ),
}


def _upstream_completion_fingerprint(
    args: argparse.Namespace,
    *,
    workflow: str,
    stage: str,
) -> Mapping[str, Any] | None:
    stages = UPSTREAM_COMPLETION_STAGES.get((workflow, stage), ())
    if not stages:
        return None
    root = Path(args.out_root).resolve() / PHASE_DIR[workflow]
    items: list[dict[str, Any]] = []
    if root.is_dir():
        for completion_path in sorted(root.rglob(".complete.json")):
            manifest_path = completion_path.parent / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = load_json(manifest_path)
            upstream_stage = manifest.get("task", {}).get("stage")
            if upstream_stage not in stages:
                continue
            # The completion record already commits to every artifact hash.
            # Fingerprinting that small record is sufficient for stage-global
            # task identity; the concrete artifact consumed by a task is
            # authenticated separately.  Re-hashing every clean trajectory in
            # every array process would create an avoidable shared-filesystem
            # storm.
            record = load_json(completion_path)
            validate_completion_record(record)
            items.append(
                {
                    "stage": upstream_stage,
                    "directory": completion_path.parent.relative_to(root).as_posix(),
                    "completion_sha256": file_sha256(completion_path),
                    "config_hash": record["config_hash"],
                    "source_hash": record["source_hash"],
                    "artifacts": [
                        {"path": item["path"], "sha256": item["sha256"]}
                        for item in record["artifacts"]
                    ],
                }
            )
    return {
        "required_stages": list(stages),
        "items": items,
        "content_hash": content_hash(
            {"required_stages": list(stages), "items": items},
            domain="linkradius:upstream_completion_inventory:v1",
        ),
    }


def _aggregate_source_completion_fingerprint(
    args: argparse.Namespace,
) -> Mapping[str, Any]:
    phase = str(args.aggregate_phase)
    if phase not in REQUIRED_AGGREGATE_STAGES:
        raise ContractError("aggregate phase has no canonical source workflow")
    root = Path(args.out_root).resolve() / PHASE_DIR[phase]
    stages = set(REQUIRED_AGGREGATE_STAGES[phase])
    items: list[dict[str, Any]] = []
    if root.is_dir():
        for completion_path in sorted(root.rglob(".complete.json")):
            manifest_path = completion_path.parent / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = load_json(manifest_path)
            stage = manifest.get("task", {}).get("stage")
            if stage not in stages:
                continue
            record = verify_completion(completion_path.parent)
            items.append(
                {
                    "stage": stage,
                    "directory": completion_path.parent.relative_to(root).as_posix(),
                    "completion_sha256": file_sha256(completion_path),
                    "config_hash": record["config_hash"],
                    "source_hash": record["source_hash"],
                }
            )
    result = {"phase": phase, "required_stages": sorted(stages), "items": items}
    return {
        **result,
        "content_hash": content_hash(
            result, domain="linkradius:aggregate_source_completion_tree:v1"
        ),
    }


def _runner_config_digest(
    args: argparse.Namespace,
    *,
    workflow: str,
    stage: str,
) -> str:
    """Hash every stage-relevant option not already explicit in ``GridTask``."""

    payload: dict[str, Any] = {
        "schema_version": "linkradius.runner_config.v1",
        "workflow": workflow,
        "stage": stage,
        "upstream_completions": _upstream_completion_fingerprint(
            args, workflow=workflow, stage=stage
        ),
    }
    if stage != "split" and workflow in {"engineering", "smoke", "pilot", "attacks"}:
        payload["split_manifest"] = _configured_artifact_identity(args.split_manifest)
    if stage in GPU_STAGES:
        payload["runtime"] = {
            "trust_remote_code": int(args.trust_remote_code),
            "round_label_mode": args.round_label_mode,
            "device": args.device,
            "role_devices": _resolved_role_devices(args),
            "relay_transfer_mode": str(
                getattr(args, "relay_transfer_mode", "cpu_staged")
            ),
            "autograd_memory_mode": str(
                getattr(args, "autograd_memory_mode", "none")
            ),
            "environment": _runtime_environment_identity(),
        }
    if stage in {
        "replay",
        "causal",
        "probe",
        "probe_calibration",
        "gradient",
        "attack",
        "val",
        "test_probe",
        "test",
    }:
        payload["trajectory_input"] = {
            "explicit_trajectory": _configured_artifact_identity(args.trajectory)
        }
    if stage in {"replay", "causal"}:
        payload["donor"] = {
            "donor_seed": int(args.donor_seed),
            "donor_trajectories": [
                _configured_artifact_identity(path) for path in args.donor_trajectory
            ],
        }
    if stage == "gradient":
        payload["gradient_validation"] = {
            "pgd_steps": int(args.pgd_steps),
            "finite_difference_radii": [float(value) for value in args.finite_difference_radii],
            "autograd_fd_relative_tolerance": float(args.autograd_fd_relative_tolerance),
            "engineering_pgd_epsilon": float(args.engineering_pgd_epsilon),
        }
    if stage in {"attack", "val", "test"}:
        payload["attack_runtime"] = {
            "pgd_steps": int(args.pgd_steps),
            "random_attack_seed_offset": int(args.random_attack_seed_offset),
        }
    if workflow == "attacks" and stage in {
        "freeze_execution",
        "val",
        "freeze_attack",
    }:
        pilot_args = _producer_namespace(args, "pilot")
        payload["failure_boundary_prerequisites"] = {
            "engineering_gate": _configured_artifact_identity(args.engineering_gate),
            "smoke_gate": _configured_artifact_identity(args.smoke_gate),
            "probe_gate": _configured_artifact_identity(args.probe_gate),
            "frozen_probe_config": _configured_artifact_identity(args.frozen_config),
            "validation_execution_manifest": _configured_artifact_identity(
                _execution_manifest_path(pilot_args, "validation")
            ),
            # Attack validation consumes Phase-3 clean trajectories.  Bind the
            # complete producer inventory so an overwritten trajectory cannot
            # leave a previously compatible attack-validation task reusable.
            "validation_clean_completions": _upstream_completion_fingerprint(
                pilot_args, workflow="pilot", stage="causal"
            ),
        }
    if workflow == "attacks" and stage in {"val", "freeze_attack"}:
        payload["failure_boundary_attack_design"] = {
            "attack_families": str(args.attack_families).split(),
            "attack_epsilons": [
                float(value) for value in str(args.attack_epsilons).split()
            ],
            "pgd_steps": int(args.pgd_steps),
            "random_attack_seed_offset": int(args.random_attack_seed_offset),
            "subspace": str(args.subspace),
        }
    if workflow == "attacks" and stage in {
        "clean",
        "test_probe",
        "test",
        "thresholds",
        "analyze",
        "validate",
    }:
        payload["frozen_attack_protocol"] = {
            "gate": _configured_artifact_identity(args.attack_freeze_gate),
            "config": _configured_artifact_identity(args.frozen_attack_config),
        }
    if workflow == "attacks" and stage == "analyze":
        payload["failure_boundary_analysis"] = {
            "bootstrap_draws": int(args.bootstrap_draws),
            "calibration_bins": 10,
        }
    if stage == "freeze_execution":
        payload["execution_freeze"] = {
            "screening_jsonl": [
                _configured_artifact_identity(path) for path in args.screening_jsonl
            ],
            "max_eligible": int(args.max_eligible),
            "batches_per_shard": int(args.batches_per_shard),
            "retain_all_partition_rows": bool(args.retain_all_partition_rows),
        }
    if workflow == "engineering" and stage == "validate":
        payload["engineering_validation"] = {
            "legacy_equivalence": _configured_artifact_identity(args.legacy_equivalence),
            "autograd_fd_relative_tolerance": float(args.autograd_fd_relative_tolerance),
        }
    if workflow == "smoke" and stage == "validate":
        payload["smoke_validation"] = {
            "minimum_scorer_agreement": float(args.minimum_scorer_agreement),
            "minimum_probe_acceptance": float(args.minimum_probe_acceptance),
            "autograd_fd_relative_tolerance": float(args.autograd_fd_relative_tolerance),
            "K": int(args.K),
            "probe_radii": args.probe_radii.split(),
            "probe_seeds": args.probe_seeds.split(),
            "interventions": args.interventions.split(),
        }
    if workflow == "smoke" and stage in {"estimate"}:
        payload["estimate"] = {"K": int(args.K), "subspace": args.subspace}
    if workflow == "smoke" and stage == "aggregate":
        payload["causal_aggregate"] = {"bootstrap_draws": int(args.bootstrap_draws)}
    if workflow == "pilot" and stage in {"freeze_probe", "validate_probe", "validate"}:
        payload["probe_validation"] = {
            "K": int(args.K),
            "subspace": args.subspace,
            "minimum_scorer_agreement": float(args.minimum_scorer_agreement),
            "minimum_probe_acceptance": float(args.minimum_probe_acceptance),
            "threshold_lower_quantile": float(args.probe_threshold_lower_quantile),
            "threshold_upper_quantile": float(args.probe_threshold_upper_quantile),
            "minimum_rank_stability": float(args.minimum_rank_stability),
            "minimum_binding_stability": float(args.minimum_binding_stability),
            "minimum_stability_comparisons": int(args.minimum_stability_comparisons),
            "minimum_autograd_agreement": float(args.minimum_autograd_agreement),
            "maximum_probe_autograd_relative_error": float(args.maximum_probe_autograd_relative_error),
            "identity_replay_tolerance": float(args.identity_replay_tolerance),
            "minimum_causal_pairs": int(args.minimum_causal_pairs),
            "minimum_causal_accuracy_effect": float(args.minimum_causal_accuracy_effect),
            "minimum_causal_margin_effect": float(args.minimum_causal_margin_effect),
            "frozen_config": (
                _configured_artifact_identity(args.frozen_config)
                if stage in {"validate_probe", "validate"}
                else None
            ),
        }
    if workflow == "pilot" and stage == "aggregate":
        payload["pilot_aggregate"] = {
            "bootstrap_draws": int(args.bootstrap_draws),
            "interventions": args.interventions.split(),
            "frozen_config": _configured_artifact_identity(args.frozen_config),
        }
    if workflow == "aggregate":
        payload["aggregate"] = {
            "aggregate_phase": args.aggregate_phase,
            "verify_stages": args.verify_stages.split(),
        }
        if stage == "verify":
            payload["aggregate"]["source_completions"] = (
                _aggregate_source_completion_fingerprint(args)
            )
        if stage == "causal":
            payload["aggregate"]["bootstrap_draws"] = int(args.bootstrap_draws)
        if stage == "system_curves":
            payload["aggregate"].update(
                {
                    "fixed_edge": args.fixed_edge,
                    "useful_edges": args.useful_edges.split(),
                    "attack_epsilons": [
                        float(value) for value in args.attack_epsilons.split()
                    ],
                }
            )
        if stage != "verify":
            payload["aggregate"]["verification_gate"] = _configured_artifact_identity(
                str(_aggregate_gate_path(args))
            )
    return content_hash(payload, domain="linkradius:runner_config:v1")


def _authenticate_completed_global_pointer(
    args: argparse.Namespace,
    task: Mapping[str, Any] | None,
    *,
    producer_workflow: str,
    producer_stage: str,
    pointer_name: str,
    pointer_path_field: str,
    canonical_path: Path,
    pointer_hash_field: str,
    expected_hash: str,
    current_source_hash: str,
    expected_task: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Bind a global file to the exact completed task that published its pointer."""

    dataset = str(task["dataset"]) if task is not None else str(args.datasets).split()[0]
    horizon = int(task["R"]) if task is not None else int(str(args.rounds).split()[0])
    pointer_root = (
        Path(args.out_root).resolve()
        / PHASE_DIR[producer_workflow]
        / dataset
        / f"R{horizon}"
        / "global"
        / producer_stage
        / "global"
    )
    if expected_task is None:
        pointer_paths = sorted(pointer_root.glob(f"*/{pointer_name}"))
        expected_array_index = 0
        expected_config_hash = None
    else:
        expected_array_index = int(expected_task["array_index"])
        expected_config_hash = str(expected_task["config_key"])
        pointer_paths = [pointer_root / expected_config_hash / pointer_name]
    matches: list[Mapping[str, Any]] = []
    for pointer_path in pointer_paths:
        if not pointer_path.is_file():
            continue
        try:
            completion = verify_completion(pointer_path.parent)
            manifest = load_json(pointer_path.parent / "manifest.json")
            pointer = load_json(pointer_path)
        except (OSError, ValueError, TypeError, ContractError):
            # Failed/incomplete historical attempts are never authoritative and
            # must not shadow the one exact completed publisher.
            continue
        declared_pointer = [
            value
            for value in completion["artifacts"]
            if value.get("path") == pointer_name
        ]
        declared_manifest = [
            value
            for value in completion["artifacts"]
            if value.get("path") == "manifest.json"
        ]
        manifest_task = manifest.get("task")
        if not isinstance(manifest_task, Mapping):
            continue
        try:
            identity_matches = (
                completion.get("source_hash") == current_source_hash
                and len(declared_pointer) == 1
                and len(declared_manifest) == 1
                and manifest.get("source_hash") == current_source_hash
                and manifest_task.get("workflow") == producer_workflow
                and manifest_task.get("stage") == producer_stage
                and manifest_task.get("dataset") == dataset
                and int(manifest_task.get("R", -1)) == horizon
                and manifest_task.get("partition") == "global"
                and int(manifest_task.get("array_index", -1))
                == expected_array_index
                and completion.get("config_hash") == manifest_task.get("config_key")
                and int(completion.get("array_index", -1))
                == expected_array_index
                and pointer_path.parent.name == manifest_task.get("config_key")
                and (
                    expected_config_hash is None
                    or manifest_task.get("config_key") == expected_config_hash
                )
                and Path(str(pointer.get(pointer_path_field, ""))).resolve()
                == canonical_path.resolve()
                and pointer.get(pointer_hash_field) == expected_hash
                and completion.get(pointer_hash_field) == expected_hash
            )
        except (TypeError, ValueError, OSError):
            continue
        if identity_matches:
            matches.append(pointer)
    if len(matches) != 1:
        raise ContractError(
            f"{canonical_path.name} is not bound to exactly one current completed "
            f"{producer_workflow}/{producer_stage} pointer"
        )
    return matches[0]


def _canonical_global_task(
    args: argparse.Namespace,
    task: Mapping[str, Any],
    *,
    stage: str,
) -> Mapping[str, Any]:
    """Resolve one exact singleton task identity for the consumer's scope."""

    matches = [
        candidate.as_dict()
        for candidate in build_grid(_build_grid_config(args, stage))
        if candidate.dataset == str(task["dataset"])
        and int(candidate.R) == int(task["R"])
        and candidate.partition == "global"
        and int(candidate.seed) == int(task["seed"])
    ]
    if len(matches) != 1:
        raise ContractError(
            f"expected exactly one canonical {args.workflow}/{stage} global task"
        )
    return matches[0]


def _authenticated_canonical_task_artifacts(
    args: argparse.Namespace,
    task: Mapping[str, Any],
    repo_root: Path,
    *,
    stage: str,
    filenames: Sequence[str],
) -> tuple[Mapping[str, Any], dict[str, Path]]:
    """Authenticate artifacts from one exact canonical global producer task."""

    expected_task = _canonical_global_task(args, task, stage=stage)
    expected_dir = task_output_dir(args, expected_task)
    try:
        completion = verify_completion(
            expected_dir, expected_config_hash=str(expected_task["config_key"])
        )
        manifest = load_json(expected_dir / "manifest.json")
    except (OSError, ValueError, TypeError, ContractError) as exc:
        raise ContractError(
            f"canonical {args.workflow}/{stage} task is not compatibly complete"
        ) from exc
    current_source_hash = _cached_source_hash(args, repo_root)
    required_names = ("manifest.json", *filenames)
    declarations = {
        name: [
            artifact
            for artifact in completion["artifacts"]
            if artifact.get("path") == name
        ]
        for name in required_names
    }
    if (
        completion.get("source_hash") != current_source_hash
        or any(len(values) != 1 for values in declarations.values())
        or manifest.get("source_hash") != current_source_hash
        or manifest.get("task") != expected_task
        or int(completion.get("array_index", -1))
        != int(expected_task["array_index"])
    ):
        raise ContractError(
            f"canonical {args.workflow}/{stage} completion or artifact declarations "
            "are incompatible"
        )
    paths = {name: expected_dir / name for name in filenames}
    if not all(path.is_file() for path in paths.values()):
        raise ContractError(
            f"canonical {args.workflow}/{stage} completion is missing a required artifact"
        )
    return expected_task, paths


def _authenticate_gate_completion(
    args: argparse.Namespace,
    task: Mapping[str, Any] | None,
    *,
    gate_path: Path,
    gate_type: str,
    gate: Mapping[str, Any],
    current_source_hash: str,
) -> None:
    producers = {
        "engineering_gate": (
            "engineering",
            "validate",
            "engineering_validation_result.json",
            "engineering_gate",
        ),
        "smoke_gate": (
            "smoke",
            "validate",
            "smoke_validation_result.json",
            "smoke_gate",
        ),
        "probe_gate": (
            "pilot",
            "validate_probe",
            "probe_validation_result.json",
            "probe_gate",
        ),
        "attack_freeze_gate": (
            "attacks",
            "freeze_attack",
            "attack_freeze_result.json",
            "attack_freeze_gate",
        ),
        "attack_validation_gate": (
            "attacks",
            "validate",
            "attack_validation_result.json",
            "attack_validation_gate",
        ),
        "aggregate_verification_gate": (
            "aggregate",
            "verify",
            "aggregate_verification_result.json",
            "aggregate_verification_gate",
        ),
    }
    producer = producers.get(gate_type)
    if producer is None:
        return
    workflow, producer_stage, pointer_name, path_field = producer
    expected_task = None
    if gate_type == "aggregate_verification_gate":
        if task is None:
            raise ContractError(
                "aggregate verification gate authentication requires a consumer task"
            )
        expected_task = _canonical_global_task(args, task, stage="verify")
    pointer = _authenticate_completed_global_pointer(
        args,
        task,
        producer_workflow=workflow,
        producer_stage=producer_stage,
        pointer_name=pointer_name,
        pointer_path_field=path_field,
        canonical_path=gate_path,
        pointer_hash_field="gate_content_hash",
        expected_hash=str(gate["gate_content_hash"]),
        current_source_hash=current_source_hash,
        expected_task=expected_task,
    )
    if gate_type == "aggregate_verification_gate":
        assert expected_task is not None
        producer_dir = task_output_dir(args, expected_task)
        completion = verify_completion(
            producer_dir, expected_config_hash=str(expected_task["config_key"])
        )
        local_gate_name = "aggregate_verification_gate.json"
        required_declarations = {
            name: [
                artifact
                for artifact in completion["artifacts"]
                if artifact.get("path") == name
            ]
            for name in (
                "manifest.json",
                "command.txt",
                "verification.json",
                local_gate_name,
                "aggregate_verification_result.json",
            )
        }
        local_gate_path = producer_dir / local_gate_name
        if any(len(values) != 1 for values in required_declarations.values()):
            raise ContractError(
                "aggregate verify completion does not declare its exact required artifacts"
            )
        try:
            verification = load_json(producer_dir / "verification.json")
            local_gate = load_json(local_gate_path)
            local_gate_sha256 = file_sha256(local_gate_path)
        except (OSError, ValueError, TypeError) as exc:
            raise ContractError(
                "aggregate verify completion artifacts are unreadable"
            ) from exc
        if (
            gate.get("config_hash") != expected_task.get("config_key")
            or local_gate != dict(gate)
            or pointer.get("local_gate_sha256") != local_gate_sha256
            or local_gate_sha256 != file_sha256(gate_path)
            or pointer.get("schema_version")
            != "linkradius.aggregate_verification_result.v1"
            or pointer.get("aggregate_phase") != args.aggregate_phase
            or pointer.get("source_hash") != current_source_hash
            or verification.get("schema_version")
            != "linkradius.aggregate_verification.v1"
            or verification.get("passed") is not True
            or verification.get("aggregate_phase") != args.aggregate_phase
            or verification.get("source_hash") != current_source_hash
            or verification.get("verified_stages") != gate.get("verified_stages")
            or verification.get("completion_inventory_hash")
            != gate.get("completion_inventory_hash")
        ):
            raise ContractError(
                "aggregate verification gate differs from its completed canonical copy"
            )


def _authenticated_frozen_probe_config(
    args: argparse.Namespace,
    task: Mapping[str, Any] | None,
    *,
    current_source_hash: str,
) -> Mapping[str, Any]:
    if not args.frozen_config or not Path(args.frozen_config).is_file():
        raise ContractError("a completed frozen_config.json is required")
    path = Path(args.frozen_config)
    frozen = load_json(path)
    expected_hash = content_hash(
        {key: value for key, value in frozen.items() if key != "content_hash"},
        domain="linkradius:frozen_probe_config:v1",
    )
    if (
        frozen.get("content_hash") != expected_hash
        or frozen.get("source_hash") != current_source_hash
        or frozen.get("test_accessed") is not False
        or not isinstance(frozen.get("runtime"), Mapping)
        or frozen["runtime"].get("environment")
        != _runtime_environment_identity()
    ):
        raise ContractError("frozen probe configuration is stale or test-contaminated")
    _authenticate_completed_global_pointer(
        args,
        task,
        producer_workflow="pilot",
        producer_stage="freeze_probe",
        pointer_name="freeze_probe_result.json",
        pointer_path_field="frozen_config",
        canonical_path=path,
        pointer_hash_field="content_hash",
        expected_hash=expected_hash,
        current_source_hash=current_source_hash,
    )
    return frozen


def _authenticated_frozen_attack_config(
    args: argparse.Namespace,
    task: Mapping[str, Any] | None,
    *,
    current_source_hash: str,
) -> Mapping[str, Any]:
    if not args.frozen_attack_config or not Path(args.frozen_attack_config).is_file():
        raise ContractError("a completed frozen_attack_config.json is required")
    path = Path(args.frozen_attack_config)
    frozen = load_json(path)
    expected_hash = content_hash(
        {key: value for key, value in frozen.items() if key != "content_hash"},
        domain="linkradius:frozen_attack_config:v1",
    )
    if (
        frozen.get("schema_version")
        != "linkradius.frozen_attack_config.v1"
        or frozen.get("content_hash") != expected_hash
        or frozen.get("source_hash") != current_source_hash
        or frozen.get("test_outcomes_accessed_before_freeze") is not False
        or not isinstance(frozen.get("runtime"), Mapping)
        or frozen["runtime"].get("environment")
        != _runtime_environment_identity()
    ):
        raise ContractError("frozen attack configuration is stale or test-contaminated")
    frozen_probe_seeds = frozen.get("probe", {}).get("seeds", [])
    if (
        not isinstance(frozen_probe_seeds, list)
        or len(frozen_probe_seeds) < 3
        or len({int(value) for value in frozen_probe_seeds})
        != len(frozen_probe_seeds)
    ):
        raise ContractError(
            "frozen attack configuration requires at least three unique probe seeds"
        )
    _authenticate_completed_global_pointer(
        args,
        task,
        producer_workflow="attacks",
        producer_stage="freeze_attack",
        pointer_name="attack_freeze_result.json",
        pointer_path_field="frozen_attack_config",
        canonical_path=path,
        pointer_hash_field="content_hash",
        expected_hash=expected_hash,
        current_source_hash=current_source_hash,
    )
    return frozen


def _assert_frozen_attack_arguments(
    args: argparse.Namespace,
    frozen: Mapping[str, Any],
    *,
    stage: str,
) -> None:
    """Reject test-time protocol drift, even when no override flag was used."""

    shared = {
        "dataset": str(args.datasets).split(),
        "R": [int(value) for value in str(args.rounds).split()],
        "seeds": [int(value) for value in str(args.seeds).split()],
        "style": str(args.style),
        "method": str(args.method),
        "batch_size": int(args.batch_size),
        "latent_length": int(args.latent_length),
        "subspace": str(args.subspace),
        "runtime": {
            "role_devices": _resolved_role_devices(args),
            "relay_transfer_mode": str(args.relay_transfer_mode),
            "autograd_memory_mode": str(args.autograd_memory_mode),
            "trust_remote_code": int(args.trust_remote_code),
            "round_label_mode": str(args.round_label_mode),
            "environment": _runtime_environment_identity(),
        },
    }
    expected_shared = {
        "dataset": [str(frozen["dataset"])],
        "R": [int(frozen["R"])],
        "seeds": [int(frozen["seed"])],
        "style": str(frozen["style"]),
        "method": str(frozen["method"]),
        "batch_size": int(frozen["batch_size"]),
        "latent_length": int(frozen["latent_length"]),
        "subspace": str(frozen["subspace"]),
        "runtime": dict(frozen["runtime"]),
    }
    if shared != expected_shared:
        raise ContractError(
            f"{stage} runtime arguments differ from the frozen attack protocol"
        )
    if args.split_manifest and Path(args.split_manifest).is_file():
        if verify_split_manifest(load_json(args.split_manifest)) != frozen.get(
            "split_manifest_hash"
        ):
            raise ContractError(f"{stage} split manifest differs from the frozen protocol")
    if stage in {"clean", "test_probe", "test", "thresholds", "analyze", "validate"}:
        execution_path = _execution_manifest_path(args, "test")
        if not execution_path or verify_execution_manifest(
            load_json(execution_path)
        ) != frozen.get("test_execution_manifest_hash"):
            raise ContractError(
                f"{stage} test execution manifest differs from the frozen protocol"
            )
    if stage in {"test", "thresholds", "analyze", "validate"}:
        actual_attack = {
            "families": str(args.attack_families).split(),
            "epsilons": [float(value) for value in str(args.attack_epsilons).split()],
            "pgd_steps": int(args.pgd_steps),
            "random_attack_seed_offset": int(args.random_attack_seed_offset),
        }
        expected_attack = {
            "families": list(frozen["attack_families"]),
            "epsilons": [float(value) for value in frozen["attack_epsilons"]],
            "pgd_steps": int(frozen["pgd"]["steps"]),
            "random_attack_seed_offset": int(
                frozen["random_independent"]["seed_offset"]
            ),
        }
        if actual_attack != expected_attack:
            raise ContractError(
                f"{stage} attack arguments differ from frozen_attack_config.json"
            )
    if stage in {"test_probe", "analyze", "validate"}:
        actual_probe = {
            "radii": [float(value) for value in str(args.probe_radii).split()],
            "seeds": [int(value) for value in str(args.probe_seeds).split()],
            "K": int(args.K),
        }
        expected_probe = {
            "radii": [float(frozen["probe"]["h"])],
            "seeds": [int(value) for value in frozen["probe"]["seeds"]],
            "K": int(frozen["probe"]["K"]),
        }
        if actual_probe != expected_probe:
            raise ContractError(
                "test_probe h/seed/K differ from frozen_attack_config.json"
            )


def enforce_prerequisites(
    args: argparse.Namespace,
    stage: str,
    task: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed before importing torch or constructing a runtime."""

    if stage == "grid":
        return
    repo_root = Path(__file__).resolve().parents[2]
    current_source_hash = _cached_source_hash(args, repo_root)
    current_split_hash = (
        verify_split_manifest(load_json(args.split_manifest))
        if args.split_manifest and Path(args.split_manifest).is_file()
        else None
    )

    def current_gate(
        path: str,
        gate_type: str,
        *,
        required_hashes: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        gate = require_passed_gate(path, gate_type=gate_type, required_hashes=required_hashes)
        if gate.get("source_hash") != current_source_hash:
            raise ContractError(f"{gate_type} source hash is stale for the current code")
        _authenticate_gate_completion(
            args,
            task,
            gate_path=Path(path),
            gate_type=gate_type,
            gate=gate,
            current_source_hash=current_source_hash,
        )
        return gate

    if args.workflow == "smoke" and stage not in {"split"}:
        current_gate(
            args.engineering_gate,
            "engineering_gate",
            required_hashes={"split_manifest_hash": current_split_hash},
        )
    elif args.workflow == "pilot" and stage not in {"split"}:
        engineering = current_gate(
            args.engineering_gate,
            "engineering_gate",
            required_hashes={"split_manifest_hash": current_split_hash},
        )
        smoke = current_gate(
            args.smoke_gate,
            "smoke_gate",
            required_hashes={"engineering_gate_hash": engineering["gate_content_hash"], "split_manifest_hash": current_split_hash},
        )
        if stage == "aggregate":
            probe = current_gate(
                args.probe_gate,
                "probe_gate",
                required_hashes={
                    "engineering_gate_hash": engineering["gate_content_hash"],
                    "smoke_gate_hash": smoke["gate_content_hash"],
                    "split_manifest_hash": current_split_hash,
                },
            )
            frozen_probe = _authenticated_frozen_probe_config(
                args, task, current_source_hash=current_source_hash
            )
            if probe.get("frozen_config_hash") != frozen_probe.get("content_hash"):
                raise ContractError("pilot aggregate probe gate/config hashes differ")
    elif args.workflow == "attacks":
        engineering = current_gate(
            args.engineering_gate,
            "engineering_gate",
            required_hashes={"split_manifest_hash": current_split_hash},
        )
        smoke = current_gate(
            args.smoke_gate,
            "smoke_gate",
            required_hashes={"engineering_gate_hash": engineering["gate_content_hash"], "split_manifest_hash": current_split_hash},
        )
        probe_gate = current_gate(
            args.probe_gate,
            "probe_gate",
            required_hashes={
                "engineering_gate_hash": engineering["gate_content_hash"],
                "smoke_gate_hash": smoke["gate_content_hash"],
                "split_manifest_hash": current_split_hash,
            },
        )
        frozen_probe = _authenticated_frozen_probe_config(
            args, task, current_source_hash=current_source_hash
        )
        if probe_gate.get("frozen_config_hash") != frozen_probe.get("content_hash"):
            raise ContractError("probe gate and frozen probe configuration hashes are incompatible")
        if stage in {"clean", "test_probe", "test", "thresholds", "analyze", "validate"}:
            attack_gate = current_gate(
                args.attack_freeze_gate,
                "attack_freeze_gate",
                required_hashes={
                    "split_manifest_hash": current_split_hash,
                    "probe_gate_hash": probe_gate["gate_content_hash"],
                },
            )
            frozen_attack = _authenticated_frozen_attack_config(
                args, task, current_source_hash=current_source_hash
            )
            if attack_gate.get("frozen_attack_config_hash") != frozen_attack.get("content_hash"):
                raise ContractError("attack-freeze gate and frozen attack configuration hashes differ")
            if args.tuning_override:
                raise ContractError("frozen test-probe/test-attack stages reject tuning overrides")
            _assert_frozen_attack_arguments(
                args, frozen_attack, stage=stage
            )
    elif args.workflow == "expansion":
        current_gate(args.pilot_gate, "pilot_gate")
        current_gate(args.attack_validation_gate, "attack_validation_gate")
    elif args.workflow == "aggregate" and stage != "verify":
        gate_path = (
            Path(args.aggregate_verification_gate)
            if args.aggregate_verification_gate
            else Path(args.out_root) / f"aggregate_verification_{args.aggregate_phase}_gate.json"
        )
        gate = current_gate(str(gate_path), "aggregate_verification_gate")
        if gate.get("aggregate_phase") != args.aggregate_phase:
            raise ContractError("aggregate verification gate was produced for a different phase")
        required = set(REQUIRED_AGGREGATE_STAGES[args.aggregate_phase])
        verified = gate.get("verified_stages")
        if not isinstance(verified, list) or not required.issubset(set(verified)):
            raise ContractError(
                "aggregate verification gate does not cover the complete source workflow"
            )
        expected_scope_hash = content_hash(
            {
                "aggregate_phase": args.aggregate_phase,
                "verified_stages": verified,
                "source_hash": current_source_hash,
            },
            domain="linkradius:aggregate_verification_scope:v1",
        )
        if gate.get("verification_scope_hash") != expected_scope_hash:
            raise ContractError("aggregate verification gate scope hash is stale")


def _build_grid_config(args: argparse.Namespace, stage: str | None = None) -> GridConfig:
    base = config_from_namespace(args)
    selected_stage = stage or args.stage
    if selected_stage == "grid":
        selected_stage = args.grid_target_stage or GRID_DEFAULT_STAGE[args.workflow]
        if (
            selected_stage not in WORKFLOW_STAGES[args.workflow]
            or selected_stage in {"grid", "all"}
            or selected_stage in GRID_ALIAS
        ):
            raise ContractError(
                "--grid-target-stage must name a concrete canonical stage in the selected workflow"
            )
    if args.workflow == "pilot" and selected_stage == "validate":
        # ``validate`` is the user-facing idempotent alias for the one canonical
        # validation-only probe gate.  It must not mint a second config key or
        # rewrite the same global probe gate from a different task identity.
        selected_stage = "validate_probe"
    values = {
        **asdict(base),
        "stage": selected_stage,
        "runner_config_hash": _runner_config_digest(
            args,
            workflow=args.workflow,
            stage=GRID_ALIAS.get(selected_stage, selected_stage),
        ),
    }
    canonical_stage = GRID_ALIAS.get(selected_stage, selected_stage)
    consumes_execution = canonical_stage in {
        "clean",
        "replay",
        "causal",
        "probe",
        "probe_calibration",
        "gradient",
        "attack",
        "val",
        "test_probe",
        "test",
    }
    manifests = dict(values["execution_manifests"]) if consumes_execution else {}
    candidate_partitions = base.partitions or ("attack_train", "validation", "test")
    if consumes_execution:
        for partition in candidate_partitions:
            manifest_args = (
                _producer_namespace(args, "pilot")
                if args.workflow == "attacks"
                and canonical_stage == "val"
                and partition == "validation"
                else args
            )
            path = _execution_manifest_path(manifest_args, partition)
            if path:
                manifests.setdefault(partition, path)
    values["execution_manifests"] = manifests
    return GridConfig(**values)


def _split_task(args: argparse.Namespace, task_dir: Path, task: Mapping[str, Any], repo_root: Path) -> None:
    records = load_gpqa_raw_records()
    split = build_split_manifest(records, seed=args.seed)
    canonical_path = Path(args.split_manifest) if args.split_manifest else Path(args.out_root) / "split_manifest.json"
    digest = create_or_verify(canonical_path, split, overwrite=args.overwrite)
    pointer = {"split_manifest_path": str(canonical_path.resolve()), "split_manifest_hash": digest}
    atomic_write_json(task_dir / "split_manifest.json", split, overwrite=args.overwrite)
    atomic_write_json(task_dir / "split_result.json", pointer, overwrite=args.overwrite)
    publish_completion(
        task_dir,
        config_hash=str(task["config_key"]),
        source_hash_value=_cached_source_hash(args, repo_root),
        artifact_paths=["manifest.json", "command.txt", "split_result.json", "split_manifest.json"],
        extra={"array_index": int(task["array_index"]), "split_manifest_hash": digest},
        overwrite=args.overwrite,
    )


def _records_for_task(
    args: argparse.Namespace, task: Mapping[str, Any], repo_root: Path
) -> list[Mapping[str, Any]]:
    split, _ = _authenticated_split_manifest(args, task, repo_root)
    partition_rows = split["partitions"][task["partition"]]
    raw_ids = [str(row["raw_sample_id"] if isinstance(row, Mapping) else row) for row in partition_rows]
    execution_path = (
        _execution_manifest_path(args, str(task["partition"]), task)
        if task.get("stage") == "clean"
        else ""
    )
    if execution_path:
        _, execution, _ = _authenticated_execution_manifest(
            args, str(task["partition"]), task, repo_root
        )
        if execution["partition"] != task["partition"]:
            raise ContractError("selected execution manifest partition differs from task")
        boundary = execution["batch_boundaries"][int(task["execution_batch_id"])]
        raw_ids = list(execution["ordered_raw_sample_ids"])[int(boundary["start"]):int(boundary["stop"])]
    else:
        start = int(task["execution_batch_id"] or 0) * int(args.batch_size)
        raw_ids = raw_ids[start : start + int(args.batch_size)]
    if not raw_ids:
        raise ContractError("selected grid task has an empty execution batch")
    records = load_gpqa_raw_records()
    by_id = {str(record["raw_sample_id"]): record for record in records}
    missing = [raw_id for raw_id in raw_ids if raw_id not in by_id]
    if missing:
        raise ContractError(f"raw dataset no longer contains frozen IDs: {missing[:5]}")
    return [by_id[raw_id] for raw_id in raw_ids]


def _runtime(args: argparse.Namespace, *, requested_edge: str | None = None):
    # Deliberately lazy: all split/manifest/gate/edge checks run first.
    from RecursiveMAS.inference_utils import inference_mas
    from RecursiveMAS.inference_utils.linkradius_runtime import LinkRadiusRuntime, RuntimeConfig

    inference_mas.configure_runtime_reproducibility(int(args.seed), deterministic=True)

    role_devices = _resolved_role_devices(args)
    config = RuntimeConfig(
        rounds=args.rounds_runtime,
        latent_steps=args.latent_length,
        batch_size=args.batch_size,
        style=args.style,
        dataset=args.dataset_runtime,
        seed=int(args.seed),
        deterministic=True,
        device=args.device,
        planner_device=role_devices["planner"],
        critic_device=role_devices["critic"],
        solver_device=role_devices["solver"],
        terminal_solver_device=role_devices["terminal_solver"],
        relay_transfer_mode=args.relay_transfer_mode,
        autograd_memory_mode=args.autograd_memory_mode,
        dtype="auto",
        outer_dtype="auto",
        enable_thinking=False,
        choice_old_prompt=2,
        solver_pre_question=0,
        do_sample=False,
        round_label_mode=args.round_label_mode,
    )
    runtime = LinkRadiusRuntime(config)
    if requested_edge:
        runtime.prevalidate_edges([requested_edge])
    runtime.load_system(trust_remote_code=bool(args.trust_remote_code))
    return runtime


def _finite_or_none(value: Any) -> float | None:
    """Return a JSON-safe finite float for diagnostic-only statistics."""

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _finite_value_summary(values: Sequence[float]) -> dict[str, float | None]:
    """Summarize already-filtered finite values without publishing NaN/Inf."""

    if not values:
        return {
            "finite_min": None,
            "finite_max": None,
            "finite_mean": None,
            "finite_abs_max": None,
            "finite_l2_norm": None,
        }
    norm = 0.0
    for value in values:
        norm = math.hypot(norm, value)
    try:
        mean = math.fsum(value / len(values) for value in values)
    except OverflowError:
        mean = math.nan
    return {
        "finite_min": _finite_or_none(min(values)),
        "finite_max": _finite_or_none(max(values)),
        "finite_mean": _finite_or_none(mean),
        "finite_abs_max": _finite_or_none(max(abs(value) for value in values)),
        "finite_l2_norm": _finite_or_none(norm),
    }


def _plain_tensor_row_finiteness(value: Any, sample_index: int) -> dict[str, Any]:
    """Pure-Python fallback used by CPU tests and tensor-like diagnostics."""

    try:
        row = value[sample_index]
    except (IndexError, KeyError, TypeError) as exc:
        raise ContractError(
            f"relay tensor does not contain sample index {sample_index}"
        ) from exc

    shape: list[int] = []
    cursor = row
    while isinstance(cursor, (list, tuple)):
        shape.append(len(cursor))
        if not cursor:
            break
        cursor = cursor[0]

    entries: list[tuple[list[int], float]] = []

    def visit(item: Any, path: list[int]) -> None:
        if isinstance(item, (list, tuple)):
            for offset, child in enumerate(item):
                visit(child, [*path, offset])
            return
        try:
            entries.append((path, float(item)))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ContractError(
                f"relay tensor diagnostic encountered a non-numeric value at {path}"
            ) from exc

    visit(row, [])
    finite_entries = [(path, number) for path, number in entries if math.isfinite(number)]
    nonfinite_entries = [
        (path, number) for path, number in entries if not math.isfinite(number)
    ]
    nan_count = sum(math.isnan(number) for _, number in nonfinite_entries)
    posinf_count = sum(number == math.inf for _, number in nonfinite_entries)
    neginf_count = sum(number == -math.inf for _, number in nonfinite_entries)

    latent_steps: list[dict[str, Any]] = []
    if shape:
        for latent_step in range(shape[0]):
            step_values = [
                number
                for path, number in entries
                if path and path[0] == latent_step
            ]
            step_finite = [number for number in step_values if math.isfinite(number)]
            step_nonfinite = len(step_values) - len(step_finite)
            latent_steps.append(
                {
                    "latent_step": latent_step,
                    "all_finite": step_nonfinite == 0,
                    "finite_count": len(step_finite),
                    "nonfinite_count": step_nonfinite,
                    **_finite_value_summary(step_finite),
                }
            )

    return {
        "shape": shape,
        "stored_dtype": "python_float",
        "numel": len(entries),
        "all_finite": not nonfinite_entries,
        "finite_count": len(finite_entries),
        "nonfinite_count": len(nonfinite_entries),
        "nan_count": nan_count,
        "posinf_count": posinf_count,
        "neginf_count": neginf_count,
        "first_nonfinite_index": (
            nonfinite_entries[0][0] if nonfinite_entries else None
        ),
        "first_nonfinite_latent_step": (
            nonfinite_entries[0][0][0]
            if nonfinite_entries and nonfinite_entries[0][0]
            else None
        ),
        "latent_step_stats": latent_steps,
        **_finite_value_summary([number for _, number in finite_entries]),
    }


def _tensor_row_finiteness(value: Any, sample_index: int) -> dict[str, Any]:
    """Compute compact, JSON-safe finiteness statistics for one relay row."""

    try:
        import torch
    except ModuleNotFoundError:  # Pure scheduling/test environments omit PyTorch.
        torch = None  # type: ignore[assignment]

    if torch is None or not torch.is_tensor(value):
        return _plain_tensor_row_finiteness(value, sample_index)
    if sample_index < 0 or sample_index >= int(value.size(0)):
        raise ContractError(
            f"relay tensor does not contain sample index {sample_index}"
        )

    row = value[sample_index].detach().to(device="cpu").double()
    finite_mask = torch.isfinite(row)
    nonfinite_mask = ~finite_mask
    finite_values = row[finite_mask]
    nonfinite_locations = torch.nonzero(nonfinite_mask, as_tuple=False)

    def tensor_summary(values: Any) -> dict[str, float | None]:
        if int(values.numel()) == 0:
            return _finite_value_summary([])
        return {
            "finite_min": _finite_or_none(values.min().item()),
            "finite_max": _finite_or_none(values.max().item()),
            "finite_mean": _finite_or_none(values.mean().item()),
            "finite_abs_max": _finite_or_none(values.abs().max().item()),
            "finite_l2_norm": _finite_or_none(
                torch.linalg.vector_norm(values).item()
            ),
        }

    latent_steps: list[dict[str, Any]] = []
    if row.ndim:
        flattened = row.reshape(int(row.size(0)), -1)
        for latent_step in range(int(flattened.size(0))):
            step = flattened[latent_step]
            step_mask = torch.isfinite(step)
            step_finite = step[step_mask]
            step_nonfinite = int((~step_mask).sum().item())
            latent_steps.append(
                {
                    "latent_step": latent_step,
                    "all_finite": step_nonfinite == 0,
                    "finite_count": int(step_finite.numel()),
                    "nonfinite_count": step_nonfinite,
                    **tensor_summary(step_finite),
                }
            )

    first_index = (
        [int(item) for item in nonfinite_locations[0].tolist()]
        if int(nonfinite_locations.size(0))
        else None
    )
    return {
        "shape": [int(item) for item in row.shape],
        "stored_dtype": str(value.dtype).replace("torch.", ""),
        "numel": int(row.numel()),
        "all_finite": not bool(nonfinite_mask.any().item()),
        "finite_count": int(finite_mask.sum().item()),
        "nonfinite_count": int(nonfinite_mask.sum().item()),
        "nan_count": int(torch.isnan(row).sum().item()),
        "posinf_count": int((torch.isinf(row) & (row > 0)).sum().item()),
        "neginf_count": int((torch.isinf(row) & (row < 0)).sum().item()),
        "first_nonfinite_index": first_index,
        "first_nonfinite_latent_step": (
            first_index[0] if first_index else None
        ),
        "latent_step_stats": latent_steps,
        **tensor_summary(finite_values),
    }


def _forward_finiteness_diagnostics(
    trajectory: Any,
    sample_index: int,
    *,
    scorer_numerically_valid: bool,
) -> dict[str, Any]:
    """Locate the first observed non-finite sequential forward boundary."""

    transports = getattr(trajectory, "transport_messages", None)
    receivers = getattr(trajectory, "receiver_reference_messages", None)
    dtype_metadata = getattr(trajectory, "edge_dtypes", None)
    if not isinstance(transports, Mapping) or not isinstance(receivers, Mapping):
        raise ContractError("trajectory relay mappings are missing")
    if not isinstance(dtype_metadata, Mapping):
        raise ContractError("trajectory relay dtype metadata is missing")

    def edge_id(value: Any) -> str:
        return str(getattr(value, "edge_id", value))

    def by_edge_id(mapping: Mapping[Any, Any], *, field: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in mapping.items():
            identifier = edge_id(key)
            if identifier in result:
                raise ContractError(f"duplicate {field} relay edge: {identifier}")
            result[identifier] = value
        return result

    transport_by_id = by_edge_id(transports, field="transport")
    receiver_by_id = by_edge_id(receivers, field="receiver")
    dtype_by_id = by_edge_id(dtype_metadata, field="dtype")
    if set(transport_by_id) != set(receiver_by_id) or set(transport_by_id) != set(
        dtype_by_id
    ):
        raise ContractError("trajectory relay diagnostics found inconsistent edge sets")

    site_order = {"p2c": 0, "c2s": 1, "s2p": 2}

    def chronological(identifier: str) -> tuple[int, int]:
        try:
            site, raw_round = identifier.split("@", 1)
            return int(raw_round), site_order[site]
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"invalid trajectory relay edge ID: {identifier}") from exc

    ordered_edges = sorted(transport_by_id, key=chronological)
    edge_diagnostics: dict[str, Any] = {}
    first_nonfinite: dict[str, Any] | None = None
    all_relays_finite = True
    role_by_site = {
        "p2c": ("planner", "critic"),
        "c2s": ("critic", "solver"),
        "s2p": ("solver", "planner"),
    }
    action_by_site = {
        "c2s": "critic",
        "s2p": "solver_feedback",
    }

    for identifier in ordered_edges:
        round_idx, _ = chronological(identifier)
        site = identifier.split("@", 1)[0]
        source_role, consumer_role = role_by_site[site]
        action = (
            "planner_initial"
            if site == "p2c" and round_idx == 0
            else "planner_feedback"
            if site == "p2c"
            else action_by_site[site]
        )
        transport_stats = _tensor_row_finiteness(
            transport_by_id[identifier], sample_index
        )
        receiver_stats = _tensor_row_finiteness(
            receiver_by_id[identifier], sample_index
        )
        metadata = dtype_by_id[identifier]
        def metadata_value(field: str, default: Any = None) -> Any:
            if isinstance(metadata, Mapping):
                return metadata.get(field, default)
            return getattr(metadata, field, default)

        transport_dtype = metadata_value("transport_dtype")
        consumer_dtype = metadata_value("consumer_dtype")
        requested_transfer_mode = metadata_value(
            "requested_transfer_mode", "direct"
        )
        realized_transfer_mode = metadata_value(
            "realized_transfer_mode", "direct"
        )
        edge_diagnostics[identifier] = {
            "action": action,
            "source_role": source_role,
            "consumer_role": consumer_role,
            "declared_transport_dtype": str(transport_dtype),
            "declared_consumer_dtype": str(consumer_dtype),
            "requested_transfer_mode": str(requested_transfer_mode),
            "realized_transfer_mode": str(realized_transfer_mode),
            "transport": transport_stats,
            "receiver": receiver_stats,
        }
        for interface, stats in (
            ("transport", transport_stats),
            ("receiver", receiver_stats),
        ):
            if not stats["all_finite"]:
                all_relays_finite = False
                if first_nonfinite is None:
                    first_nonfinite = {
                        "stage": action,
                        "edge_id": identifier,
                        "source_role": source_role,
                        "consumer_role": consumer_role,
                        "interface": interface,
                        "first_nonfinite_index": stats[
                            "first_nonfinite_index"
                        ],
                        "first_nonfinite_latent_step": stats[
                            "first_nonfinite_latent_step"
                        ],
                    }

    terminal_edge = f"c2s@{int(trajectory.rounds) - 1}"
    if first_nonfinite is None and not scorer_numerically_valid:
        first_nonfinite = {
            "stage": "terminal_solver_scoring",
            "edge_id": terminal_edge,
            "source_role": "solver",
            "consumer_role": None,
            "interface": "forced_choice_scores",
            "first_nonfinite_index": None,
            "first_nonfinite_latent_step": None,
        }

    return {
        "schema_version": "linkradius.forward_finiteness.v2",
        "all_relay_interfaces_finite": all_relays_finite,
        "scorer_numerically_valid": bool(scorer_numerically_valid),
        "all_observed_numeric_outputs_finite": bool(
            all_relays_finite and scorer_numerically_valid
        ),
        "terminal_input_edge": terminal_edge,
        "first_nonfinite": first_nonfinite,
        "edges": edge_diagnostics,
    }


def _trajectory_rows(
    trajectory: Any,
    *,
    task: Mapping[str, Any],
    args: argparse.Namespace | None = None,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    scores = _to_plain(trajectory.clean_scoring.scores)
    summed_logprobs = _to_plain(trajectory.clean_scoring.summed_logprobs)
    mean_logprobs = _to_plain(trajectory.clean_scoring.mean_logprobs)
    token_counts = _to_plain(trajectory.clean_scoring.token_counts)
    scorer_metadata = _to_plain(trajectory.clean_scoring.metadata)
    trajectory_provenance = trajectory.provenance
    model_hash = trajectory_provenance.get("model_hash")
    adapter_hash = trajectory_provenance.get("adapter_hash")
    system_resolution = trajectory_provenance.get("system_resolution")
    scorer_hash = trajectory_provenance.get("scorer_hash") or content_hash(
        scorer_metadata, domain="linkradius:scorer:v1"
    )
    prompt_hash = content_hash(
        trajectory_provenance.get("release_settings", {}),
        domain="linkradius:prompt_settings:v1",
    )
    exclusion_reasons = list(
        trajectory_provenance.get("execution_exclusion_reasons", [])
    )
    labels = list(trajectory.clean_scoring.labels)
    source_digest = (
        _cached_source_hash(args, repo_root)
        if args is not None and repo_root is not None
        else None
    )
    rows = []
    screening_stage = str(task.get("stage")) in {"discover", "screen", "screen_clean"}
    for index, sample_id in enumerate(trajectory.sample_ids):
        generation = trajectory.clean_generation_audit[index] if trajectory.clean_generation_audit else {}

        nonfinite_fields: list[str] = []

        def public_number(value: Any, *, field: str) -> float | None:
            number = float(value)
            if math.isfinite(number):
                return number
            nonfinite_fields.append(field)
            return None

        option_scores = {
            label: public_number(scores[index][pos], field=f"option_scores.{label}")
            for pos, label in enumerate(labels)
        }
        summed_scores = {
            label: public_number(
                summed_logprobs[index][pos],
                field=f"summed_option_logprobs.{label}",
            )
            for pos, label in enumerate(labels)
        }
        mean_scores = {
            label: public_number(
                mean_logprobs[index][pos],
                field=f"mean_option_logprobs.{label}",
            )
            for pos, label in enumerate(labels)
        }
        margins = {
            label: public_number(value, field=f"margins.{label}")
            for label, value in trajectory.clean_margins[index].items()
        }
        scorer_numerically_valid = not nonfinite_fields
        forward_finiteness = _forward_finiteness_diagnostics(
            trajectory,
            index,
            scorer_numerically_valid=scorer_numerically_valid,
        )
        if (
            nonfinite_fields
            or not forward_finiteness["all_relay_interfaces_finite"]
        ) and not screening_stage:
            first = forward_finiteness.get("first_nonfinite")
            first_text = (
                f"; first observed at {first.get('stage')} "
                f"{first.get('edge_id')} {first.get('interface')}"
                if isinstance(first, Mapping)
                else ""
            )
            raise ContractError(
                f"{task.get('stage')} trajectory row {trajectory.raw_sample_ids[index]} "
                "contains non-finite forward values: "
                + (", ".join(nonfinite_fields) or "relay tensor")
                + first_text
            )
        gold = trajectory.gold_labels[index]
        prediction = (
            trajectory.clean_scoring.predictions[index]
            if scorer_numerically_valid
            else None
        )
        strict_choice = generation.get("strict_choice")
        strict_valid = bool(strict_choice in {"A", "B", "C", "D"}) and not bool(
            generation.get("answer_invalid", False)
        )
        finite_margins = {
            label: value for label, value in margins.items() if value is not None
        }
        rows.append(
            {
                "schema_version": "linkradius.v1",
                "record_type": "sample",
                "raw_sample_id": trajectory.raw_sample_ids[index],
                "sample_id": sample_id,
                "raw_index": trajectory.raw_indices[index],
                "gold": gold,
                "strict_generated_choice": strict_choice,
                "strict_generated_valid": strict_valid,
                "strict_generated_correct": bool(strict_valid and strict_choice == gold),
                "answer_invalid": generation.get("answer_invalid", not bool(generation)),
                "answer_conflict": generation.get("answer_conflict", False),
                "scorer_prediction": prediction,
                "score_tie": (
                    bool(trajectory.clean_scoring.score_ties[index])
                    if scorer_numerically_valid
                    else False
                ),
                "scorer_correct": bool(scorer_numerically_valid and prediction == gold),
                "scorer_numerically_valid": scorer_numerically_valid,
                "scorer_nonfinite_fields": sorted(nonfinite_fields),
                "forward_finiteness": forward_finiteness,
                "option_scores": option_scores,
                "summed_option_logprobs": summed_scores,
                "mean_option_logprobs": mean_scores,
                "option_token_counts": {
                    label: int(token_counts[index][pos])
                    for pos, label in enumerate(labels)
                },
                "scorer_metadata": scorer_metadata,
                "margins": margins,
                "minimum_margin": (
                    min(finite_margins.values())
                    if scorer_numerically_valid
                    else None
                ),
                "binding_competitor": (
                    min(finite_margins, key=finite_margins.get)
                    if scorer_numerically_valid
                    else None
                ),
                "analysis_eligible": bool(trajectory.analysis_eligibility_mask[index]),
                "exclusion_reason": (
                    str(exclusion_reasons[index])
                    if index < len(exclusion_reasons)
                    else ("" if trajectory.analysis_eligibility_mask[index] else "unspecified_ineligible")
                ),
                "phase": args.workflow if args is not None else task.get("workflow"),
                "partition": task.get("partition"),
                "dataset": task.get("dataset"),
                "source_split": "train",
                "style": args.style if args is not None else task.get("style"),
                "method": args.method if args is not None else task.get("method"),
                "R": int(task["R"]),
                "split_manifest_hash": trajectory.provenance.get("split_manifest_hash"),
                "execution_manifest_hash": trajectory.execution_manifest_hash,
                "ordered_cohort_hash": trajectory.provenance.get("global_ordered_cohort_hash") or trajectory.ordered_cohort_hash,
                "batch_boundary_hash": trajectory.provenance.get("global_batch_boundary_hash") or trajectory.batch_boundary_hash,
                "source_hash": source_digest,
                "model_hash": model_hash,
                "adapter_hash": adapter_hash,
                "system_resolution": (
                    dict(system_resolution)
                    if isinstance(system_resolution, Mapping)
                    else None
                ),
                "scorer_hash": scorer_hash,
                "prompt_hash": prompt_hash,
                "config_hash": task.get("config_key"),
                "task": dict(task),
                "generation_audit": _to_plain(generation),
            }
        )
    return rows


def _atomic_torch_save(path: Path, value: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _load_trajectory(path: str | Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _completed_clean_trajectories(
    args: argparse.Namespace,
    task: Mapping[str, Any],
    *,
    producer_workflow: str | None = None,
    execution_batch_ids: Iterable[int] | None = None,
) -> list[Path]:
    producer_args = _producer_namespace(args, producer_workflow)
    clean_root = (
        _phase_root(producer_args)
        / str(task["dataset"])
        / f"R{int(task['R'])}"
        / str(task["partition"])
        / "clean"
    )
    execution_path = _execution_manifest_path(
        producer_args, str(task["partition"]), task
    )
    if not execution_path:
        raise ContractError("clean trajectory authentication requires an execution manifest")
    expected_execution_hash = verify_execution_manifest(load_json(execution_path))
    expected_tasks = {
        int(candidate.execution_batch_id): candidate.as_dict()
        for candidate in build_grid(_build_grid_config(producer_args, "clean"))
        if candidate.dataset == str(task["dataset"])
        and int(candidate.R) == int(task["R"])
        and candidate.partition == str(task["partition"])
        and candidate.execution_batch_id is not None
    }
    if not expected_tasks:
        raise ContractError("canonical clean grid has no tasks for the requested partition")
    selected_batch_ids = (
        set(expected_tasks)
        if execution_batch_ids is None
        else {int(value) for value in execution_batch_ids}
    )
    unknown_batch_ids = selected_batch_ids - set(expected_tasks)
    if unknown_batch_ids:
        raise ContractError(
            "requested execution batch is absent from the canonical clean grid: "
            f"{sorted(unknown_batch_ids)}"
        )
    repo_root = Path(__file__).resolve().parents[2]
    found: dict[int, Path] = {}
    for batch_id, expected_task in sorted(expected_tasks.items()):
        if batch_id not in selected_batch_ids:
            continue
        expected_dir = task_output_dir(producer_args, expected_task)
        completion_path = expected_dir / ".complete.json"
        if not completion_path.is_file():
            continue
        completion = verify_completion(
            expected_dir, expected_config_hash=str(expected_task["config_key"])
        )
        if completion.get("source_hash") != _cached_source_hash(producer_args, repo_root):
            raise ContractError("clean trajectory completion has a stale source hash")
        declared = [
            artifact
            for artifact in completion["artifacts"]
            if artifact.get("path") == "clean_trajectory.pt"
        ]
        if len(declared) != 1:
            raise ContractError("clean trajectory is not uniquely declared by its completion")
        manifest_entries = [
            artifact
            for artifact in completion["artifacts"]
            if artifact.get("path") == "manifest.json"
        ]
        manifest = load_json(expected_dir / "manifest.json")
        if (
            len(manifest_entries) != 1
            or manifest.get("task") != expected_task
            or completion.get("config_hash") != expected_task["config_key"]
            or int(completion.get("array_index", -1))
            != int(expected_task["array_index"])
            or int(completion.get("execution_batch_id", -1)) != batch_id
        ):
            raise ContractError("clean trajectory completion does not match its canonical clean task")
        trajectory_path = expected_dir / "clean_trajectory.pt"
        trajectory = _load_trajectory(trajectory_path)
        if trajectory.execution_manifest_hash != expected_execution_hash:
            raise ContractError("clean trajectory references a different execution manifest")
        if tuple(int(value) for value in trajectory.ordered_batch_ids) != (batch_id,):
            raise ContractError("clean trajectory does not contain exactly its canonical frozen batch")
        found[batch_id] = trajectory_path
    return [found[key] for key in sorted(found)]


def _resolve_trajectory_path(
    args: argparse.Namespace,
    task: Mapping[str, Any],
    *,
    producer_workflow: str | None = None,
) -> Path:
    batch_id = int(task["execution_batch_id"])
    authenticated = _completed_clean_trajectories(
        args,
        task,
        producer_workflow=producer_workflow,
        execution_batch_ids=(batch_id,),
    )
    if args.trajectory:
        requested = Path(args.trajectory).resolve()
        matches = [path for path in authenticated if path.resolve() == requested]
        if len(matches) != 1:
            raise ContractError(
                "explicit --trajectory must be the authenticated artifact of the exact canonical clean task"
            )
        completion = verify_completion(matches[0].parent)
        if int(completion["execution_batch_id"]) != batch_id:
            raise ContractError("explicit --trajectory is for a different execution batch")
        return matches[0]
    if len(authenticated) == 1:
        return authenticated[0]
    raise ContractError(
        f"no compatible completed clean trajectory found for execution batch {batch_id}; "
        "pass --trajectory explicitly"
    )


def _mismatch_intervention(
    args: argparse.Namespace, trajectory: Any, edge: str
) -> tuple[Any, list[dict[str, Any]]]:
    import torch
    from RecursiveMAS.inference_utils.linkradius import (
        deterministic_donor_assignments,
        mismatch_intervention,
    )
    from RecursiveMAS.inference_utils.linkradius_runtime import ReplayIntervention

    trajectories = [trajectory]
    seen_paths: set[str] = set()
    donor_paths = list(args.donor_trajectory)
    authenticated_paths = _completed_clean_trajectories(args, args._active_task)
    authenticated_by_resolved = {
        str(path.resolve()): path for path in authenticated_paths
    }
    if not donor_paths:
        # Automatically use every completed whole-batch trajectory from the same
        # frozen execution manifest; no row is extracted and repadded.
        donor_paths = [str(path) for path in authenticated_paths]
    for path in donor_paths:
        resolved = str(Path(path).resolve())
        if resolved not in authenticated_by_resolved:
            raise ContractError(
                "explicit --donor-trajectory must be an authenticated canonical clean artifact"
            )
        active_path = str(getattr(args, "_active_trajectory_path", "") or "")
        if resolved in seen_paths or (active_path and resolved == active_path):
            continue
        seen_paths.add(resolved)
        donor_trajectory = _load_trajectory(authenticated_by_resolved[resolved])
        if donor_trajectory.execution_manifest_hash != trajectory.execution_manifest_hash:
            raise ContractError("mismatch donor trajectory has a different execution manifest")
        if donor_trajectory.rounds != trajectory.rounds:
            raise ContractError("mismatch donor trajectory has a different horizon")
        trajectories.append(donor_trajectory)
    execution_path = _execution_manifest_path(args, str(args._active_task["partition"]), args._active_task)
    if not execution_path:
        raise ContractError("mismatch control requires the complete frozen execution manifest")
    execution = load_json(execution_path)
    expected_batch_ids = set(range(len(execution["batch_boundaries"])))
    available_batch_ids = {
        int(batch_id)
        for source in trajectories
        for batch_id in source.ordered_batch_ids
    }
    if available_batch_ids != expected_batch_ids:
        missing = sorted(expected_batch_ids - available_batch_ids)
        extra = sorted(available_batch_ids - expected_batch_ids)
        raise ContractError(
            f"mismatch donor set must contain every frozen execution batch exactly; missing={missing}, extra={extra}"
        )
    records = []
    tensors: dict[str, Any] = {}
    for source in trajectories:
        message = source.message(edge, receiver=True)
        for index, raw_id in enumerate(source.raw_sample_ids):
            if raw_id in tensors:
                continue
            tensor = message[index].float().cpu()
            tensors[raw_id] = tensor
            records.append(
                {
                    "raw_sample_id": raw_id,
                    "partition": str(args._active_task["partition"]),
                    "gold": source.gold_labels[index],
                    "edge_id": edge,
                    "R": source.rounds,
                    "tensor_shape": list(tensor.shape),
                    "length_bucket": int(tensor.shape[0]),
                }
            )
    assignments = deterministic_donor_assignments(records, donor_seed=int(args.donor_seed))
    clean = trajectory.message(edge, receiver=True).float().cpu()
    replacements = []
    metadata: list[dict[str, Any]] = []
    for index, raw_id in enumerate(trajectory.raw_sample_ids):
        assignment = assignments[raw_id]
        if not assignment.available or assignment.donor_id is None:
            # The row is explicitly unavailable and excluded downstream.  The
            # clean value merely keeps the whole frozen batch executable; it is
            # never reported as a successful mismatch intervention.
            replacements.append(clean[index])
            metadata.append(
                {
                    "available": False,
                    "intervention_unavailable": True,
                    "unavailable_reason": assignment.reason,
                    "recipient_id": raw_id,
                    "donor_id": None,
                }
            )
            continue
        requested, diagnostics = mismatch_intervention(
            clean[index], tensors[assignment.donor_id]
        )
        if requested is None:
            replacements.append(clean[index])
            metadata.append(
                {
                    "available": False,
                    "intervention_unavailable": True,
                    "unavailable_reason": diagnostics.reason,
                    "recipient_id": raw_id,
                    "donor_id": assignment.donor_id,
                    **_to_plain(diagnostics),
                }
            )
        else:
            replacements.append(requested)
            metadata.append(
                {
                    "available": True,
                    "intervention_unavailable": False,
                    "unavailable_reason": "",
                    "recipient_id": raw_id,
                    "donor_id": assignment.donor_id,
                    "donor_gold": next(record["gold"] for record in records if record["raw_sample_id"] == assignment.donor_id),
                    **_to_plain(diagnostics),
                }
            )
    return ReplayIntervention(
        mode="replacement",
        replacement=torch.stack(replacements, dim=0),
        metadata={"control_mode": "label_matched_mismatch", "donor_seed": int(args.donor_seed)},
    ), metadata


def _capture_stage(args: argparse.Namespace, task_dir: Path, task: Mapping[str, Any], repo_root: Path) -> None:
    is_clean = task.get("stage") == "clean"
    execution_path = (
        _execution_manifest_path(args, str(task["partition"]), task)
        if is_clean
        else ""
    )
    if is_clean and not execution_path:
        raise ContractError("clean capture requires a frozen execution manifest")
    records = _records_for_task(args, task, repo_root)
    execution_hash = "screening_unfrozen"
    eligibility = [True] * len(records)
    execution = None
    frozen_slice: slice | None = None
    if execution_path:
        _, execution, execution_hash = _authenticated_execution_manifest(
            args, str(task["partition"]), task, repo_root
        )
        boundary = execution["batch_boundaries"][int(task["execution_batch_id"])]
        frozen_slice = slice(int(boundary["start"]), int(boundary["stop"]))
        eligibility = list(execution["analysis_eligible"])[frozen_slice]
    runtime = _runtime(args)
    try:
        trajectory = runtime.capture_clean(
            sample_ids=[record["raw_sample_id"] for record in records],
            raw_sample_ids=[record["raw_sample_id"] for record in records],
            raw_indices=[record["raw_index"] for record in records],
            questions=[record["rendered_question"] for record in records],
            gold_labels=[record["gold_label"] for record in records],
            option_permutations=[list(record["option_permutation"]) for record in records],
            choice_metadata=[
                {
                    "option_texts": list(record["option_texts"]),
                    "permutation_algorithm": record["option_permutation_algorithm"],
                    "raw_id_algorithm": record["raw_id_algorithm"],
                }
                for record in records
            ],
            execution_manifest_hash=execution_hash,
            ordered_batch_ids=[int(task["execution_batch_id"] or 0)],
            batch_boundaries=[(0, len(records))],
            analysis_eligibility_mask=eligibility,
            include_generation=True,
            provenance={
                "split_manifest_hash": verify_split_manifest(load_json(args.split_manifest)),
                "global_ordered_cohort_hash": execution.get("ordered_cohort_hash") if execution else None,
                "global_batch_boundary_hash": execution.get("batch_boundary_hash") if execution else None,
                "execution_exclusion_reasons": (
                    list(execution["exclusion_reasons"])[frozen_slice]
                    if execution is not None and frozen_slice is not None
                    else []
                ),
                "release_settings": {
                    "dtype": "auto",
                    "outer_dtype": "auto",
                    "enable_thinking": 0,
                    "solver_pre_question": 0,
                    "choice_old_prompt": 2,
                    "deterministic": 1,
                    "max_new_tokens": 4000,
                    "answer_retry": True,
                    "seed": int(args.seed),
                    "batch_size": int(args.batch_size),
                    "latent_length": int(args.latent_length),
                },
            },
        )
    finally:
        runtime.unload()
    rows = _trajectory_rows(trajectory, task=task, args=args, repo_root=repo_root)
    if task["stage"] == "clean" and execution is not None and frozen_slice is not None:
        fresh_annotated, _ = annotate_screening_rows(rows)
        frozen_dual = list(execution["screening_dual_correct"])[frozen_slice]
        changed = [
            rows[index]["raw_sample_id"]
            for index, (expected, fresh) in enumerate(
                zip(frozen_dual, (row["dual_correct"] for row in fresh_annotated))
            )
            if expected is not None and bool(expected) != bool(fresh)
        ]
        if changed:
            raise ContractError(
                "fresh clean dual-correct status differs from screening under the frozen execution "
                f"settings: {', '.join(changed)}"
            )
    if args.workflow == "attacks" and task["stage"] == "clean":
        frozen_attack = _authenticated_frozen_attack_config(
            args,
            task,
            current_source_hash=_cached_source_hash(args, repo_root),
        )
        clean_system_identity = _common_system_identity(
            rows, where="held-out clean capture"
        )
        if clean_system_identity != frozen_attack.get("system_identity"):
            raise ContractError(
                "held-out clean capture resolved a different model/adapter/"
                "scorer/prompt identity than the frozen attack protocol"
            )
    if task["stage"] in {"discover", "screen", "screen_clean"}:
        annotated, summary = annotate_screening_rows(rows)
        artifact_name = "screening_rows.jsonl"
        atomic_write_jsonl(
            task_dir / artifact_name,
            [*annotated, {"record_type": "shard_metadata", "array_index": task["array_index"], "config_key": task["config_key"], "row_count": len(annotated), "summary": summary}],
            overwrite=args.overwrite,
        )
        artifacts = ["manifest.json", "command.txt", artifact_name]
    else:
        artifact_name = "clean_baseline.jsonl"
        atomic_write_jsonl(
            task_dir / artifact_name,
            [*rows, {"record_type": "shard_metadata", "array_index": task["array_index"], "config_key": task["config_key"], "row_count": len(rows)}],
            overwrite=args.overwrite,
        )
        _atomic_torch_save(task_dir / "clean_trajectory.pt", trajectory)
        artifacts = ["manifest.json", "command.txt", artifact_name, "clean_trajectory.pt"]
    publish_completion(
        task_dir,
        config_hash=str(task["config_key"]),
        source_hash_value=_cached_source_hash(args, repo_root),
        artifact_paths=artifacts,
        row_counts={artifact_name: len(rows) + 1},
        extra={"array_index": int(task["array_index"]), "execution_batch_id": task["execution_batch_id"]},
        overwrite=args.overwrite,
    )


def _row_envelope(
    *,
    args: argparse.Namespace,
    task: Mapping[str, Any],
    trajectory: Any,
    index: int,
    repo_root: Path,
    intervention_mode: str,
    requested: Mapping[str, Any] | None = None,
    realized: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    edge_id = str(task.get("edge_id") or "")
    site = str(task.get("site") or "")
    scorer_metadata = _to_plain(trajectory.clean_scoring.metadata)
    relay = trajectory.message(edge_id, receiver=True) if edge_id else None
    T = int(relay.shape[-2]) if relay is not None else None
    D = int(relay.shape[-1]) if relay is not None else None
    subspace_name = task.get("subspace") or "receiver_tensor"
    subspace_spec = {
        "name": subspace_name,
        "edge_id": edge_id,
        "T": T,
        "D": D,
        "q": (D if subspace_name == "channel_broadcast" else (T * D if T is not None and D is not None else None)),
    }
    provenance = trajectory.provenance
    exclusion_reasons = list(provenance.get("execution_exclusion_reasons", []))
    model_hash = provenance.get("model_hash")
    adapter_hash = provenance.get("adapter_hash")
    system_resolution = provenance.get("system_resolution")
    if not model_hash or not adapter_hash or not isinstance(system_resolution, Mapping):
        raise ContractError("trajectory lacks runtime-resolved model/adapter/system provenance")
    return {
        "schema_version": "linkradius.v1",
        "record_type": "sample",
        "run_id": str(task["config_key"]),
        "phase": args.workflow,
        "partition": str(task["partition"]),
        "raw_sample_id": trajectory.raw_sample_ids[index],
        "sample_id": trajectory.sample_ids[index],
        "raw_index": int(trajectory.raw_indices[index]),
        "analysis_eligible": bool(trajectory.analysis_eligibility_mask[index]),
        "exclusion_reason": (
            str(exclusion_reasons[index])
            if index < len(exclusion_reasons)
            else ("" if trajectory.analysis_eligibility_mask[index] else "unspecified_ineligible")
        ),
        "dataset": str(task["dataset"]),
        "source_split": "train",
        "style": args.style,
        "method": args.method,
        "R": int(task["R"]),
        "site": site,
        "code_round": int(task.get("code_round") or 0),
        "paper_round": int(task.get("paper_round") or 1),
        "edge_id": edge_id,
        "split_manifest_hash": provenance.get("split_manifest_hash"),
        "execution_manifest_hash": trajectory.execution_manifest_hash,
        "ordered_cohort_hash": provenance.get("global_ordered_cohort_hash") or trajectory.ordered_cohort_hash,
        "batch_boundary_hash": provenance.get("global_batch_boundary_hash") or trajectory.batch_boundary_hash,
        "source_hash": _cached_source_hash(args, repo_root),
        "config_hash": str(task["config_key"]),
        "model_hash": model_hash,
        "adapter_hash": adapter_hash,
        "system_resolution": dict(system_resolution),
        "prompt_hash": content_hash(provenance.get("release_settings", {}), domain="linkradius:prompt_settings:v1"),
        "scorer_hash": provenance.get("scorer_hash") or content_hash(scorer_metadata, domain="linkradius:scorer:v1"),
        "scorer_metadata": scorer_metadata,
        "subspace_hash": content_hash(subspace_spec, domain="linkradius:subspace:v1"),
        "intervention_mode": intervention_mode,
        "intervention_family": task.get("attack_family"),
        "requested_intervention": dict(requested or {}),
        "realized_intervention": dict(realized or {}),
        "runtime": provenance.get("runtime_config", {}),
        "failure": None,
        "warnings": [],
    }


def _score_rows(
    scoring: Any,
    margins: Sequence[Mapping[str, float]],
    trajectory: Any,
    *,
    args: argparse.Namespace,
    task: Mapping[str, Any],
    repo_root: Path,
    intervention_mode: str,
    requested: Mapping[str, Any] | None = None,
    realized_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    values = _to_plain(scoring.scores)
    summed = _to_plain(scoring.summed_logprobs)
    means = _to_plain(scoring.mean_logprobs)
    counts = _to_plain(scoring.token_counts)
    scoring_metadata = _to_plain(scoring.metadata)
    rows = []
    for idx, raw_id in enumerate(trajectory.raw_sample_ids):
        option_scores = {label: float(values[idx][pos]) for pos, label in enumerate(scoring.labels)}
        generation = {}
        rows.append(
            {
                **_row_envelope(
                    args=args,
                    task=task,
                    trajectory=trajectory,
                    index=idx,
                    repo_root=repo_root,
                    intervention_mode=intervention_mode,
                    requested=requested,
                    realized=(realized_rows[idx] if realized_rows else {}),
                ),
                "gold": trajectory.gold_labels[idx],
                "strict_generated_choice": generation.get("strict_choice"),
                "strict_generated_valid": None,
                "strict_generated_correct": None,
                "option_scores": option_scores,
                "summed_option_logprobs": {
                    label: float(summed[idx][pos])
                    for pos, label in enumerate(scoring.labels)
                },
                "mean_option_logprobs": {
                    label: float(means[idx][pos])
                    for pos, label in enumerate(scoring.labels)
                },
                "option_token_counts": {
                    label: int(counts[idx][pos])
                    for pos, label in enumerate(scoring.labels)
                },
                "scorer_metadata": scoring_metadata,
                "scorer_prediction": scoring.predictions[idx],
                "scorer_correct": scoring.predictions[idx] == trajectory.gold_labels[idx],
                "score_tie": scoring.score_ties[idx],
                "margins": {label: float(value) for label, value in margins[idx].items()},
                "minimum_margin": min(float(value) for value in margins[idx].values()),
                "binding_competitor": min(margins[idx], key=margins[idx].get),
            }
        )
    return rows


def _autograd_sample_index(trajectory: Any) -> int:
    """Select one eligible row while retaining its complete frozen batch."""

    for index, eligible in enumerate(trajectory.analysis_eligibility_mask):
        if bool(eligible):
            return index
    raise ContractError("autograd stage has no analysis-eligible row in this execution batch")


def _fresh_dual_correct_trajectory_indices(trajectory: Any) -> list[int]:
    """Recompute the predeclared analysis cohort from frozen clean outcomes."""

    selected: list[int] = []
    for index, gold in enumerate(trajectory.gold_labels):
        generation = (
            trajectory.clean_generation_audit[index]
            if index < len(trajectory.clean_generation_audit)
            else {}
        )
        margins = trajectory.clean_margins[index]
        if (
            bool(trajectory.analysis_eligibility_mask[index])
            and generation.get("strict_choice") == gold
            and not bool(generation.get("answer_invalid", True))
            and not bool(generation.get("answer_conflict", False))
            and trajectory.clean_scoring.predictions[index] == gold
            and not bool(trajectory.clean_scoring.score_ties[index])
            and isinstance(margins, Mapping)
            and bool(margins)
            and all(
                math.isfinite(float(value)) and float(value) > 0.0
                for value in margins.values()
            )
        ):
            selected.append(index)
    return selected


def _publish_outcome_excluded_task(
    args: argparse.Namespace,
    task_dir: Path,
    task: Mapping[str, Any],
    repo_root: Path,
    *,
    artifact_name: str,
) -> None:
    """Complete a frozen grid cell that has no fresh dual-correct test row."""

    rows = [
        {
            "record_type": "shard_metadata",
            "array_index": task["array_index"],
            "config_key": task["config_key"],
            "row_count": 0,
            "skip_reason": "no_fresh_dual_correct_row",
        }
    ]
    atomic_write_jsonl(task_dir / artifact_name, rows, overwrite=args.overwrite)
    publish_completion(
        task_dir,
        config_hash=str(task["config_key"]),
        source_hash_value=_cached_source_hash(args, repo_root),
        artifact_paths=["manifest.json", "command.txt", artifact_name],
        row_counts={artifact_name: 1},
        extra={
            "array_index": int(task["array_index"]),
            "execution_batch_id": task["execution_batch_id"],
            "analysis_row_count": 0,
        },
        overwrite=args.overwrite,
    )


def _replay_stage(args: argparse.Namespace, task_dir: Path, task: Mapping[str, Any], repo_root: Path) -> None:
    args._active_task = task
    trajectory_path = _resolve_trajectory_path(args, task)
    args._active_trajectory_path = str(trajectory_path.resolve())
    trajectory = _load_trajectory(trajectory_path)
    execution_path = _execution_manifest_path(args, str(task["partition"]), task)
    if not execution_path:
        raise ContractError(f"{task['stage']} requires the matching --execution-manifest-path")
    _, _, authenticated_execution_hash = _authenticated_execution_manifest(
        args, str(task["partition"]), task, repo_root
    )
    if trajectory.execution_manifest_hash != authenticated_execution_hash:
        raise ContractError("trajectory and execution manifest hashes differ")
    edge = str(task["edge_id"])
    selected_probe_indices: list[int] | None = None
    if task["stage"] in {"probe", "probe_calibration", "test_probe"}:
        selected_probe_indices = list(range(len(trajectory.raw_sample_ids)))
        if args.workflow == "attacks" and task["stage"] == "test_probe":
            selected_probe_indices = _fresh_dual_correct_trajectory_indices(
                trajectory
            )
            if not selected_probe_indices:
                _publish_outcome_excluded_task(
                    args,
                    task_dir,
                    task,
                    repo_root,
                    artifact_name="probe_runs.jsonl",
                )
                return
    runtime = _runtime(args, requested_edge=edge)
    try:
        if task["stage"] in {"replay", "causal"}:
            mode = str(task["intervention_mode"])
            if mode == "additive_zero":
                import torch
                from RecursiveMAS.inference_utils.linkradius_runtime import ReplayIntervention

                delta = torch.zeros_like(trajectory.message(edge, receiver=True))
                intervention: Any = ReplayIntervention(mode="additive", delta=delta)
                mismatch_metadata = None
            elif mode == "mismatch":
                intervention, mismatch_metadata = _mismatch_intervention(args, trajectory, edge)
            else:
                intervention = mode
                mismatch_metadata = None
            result = runtime.replay(trajectory, edge, intervention)
            realized_metadata = (
                [
                    {**_to_plain(runtime_value), **_to_plain(donor_value)}
                    for runtime_value, donor_value in zip(
                        result.intervention_metadata, mismatch_metadata
                    )
                ]
                if mismatch_metadata is not None
                else [_to_plain(value) for value in result.intervention_metadata]
            )
            rows = _score_rows(
                result.scoring,
                result.margins,
                trajectory,
                args=args,
                task=task,
                repo_root=repo_root,
                intervention_mode=mode,
                realized_rows=realized_metadata,
            )
            for row, metadata in zip(rows, result.intervention_metadata):
                row.update({"edge_id": edge, "intervention_mode": mode, "intervention_metadata": _to_plain(metadata)})
            if mismatch_metadata is not None:
                for row, metadata in zip(rows, mismatch_metadata):
                    row.update(metadata)
            artifact_name = "replay_runs.jsonl" if task["stage"] == "replay" else "causal_runs.jsonl"
        elif task["stage"] in {"probe", "probe_calibration", "test_probe"}:
            if selected_probe_indices is None:
                raise ContractError("probe selection was not initialized")
            rows = []
            for direction_id in range(int(task["K"])):
                probe = runtime.run_antithetic_probe(
                    trajectory,
                    edge,
                    h=float(task["h"]),
                    global_seed=int(task["seed"]),
                    probe_seed=int(task["probe_seed"]),
                    direction_id=direction_id,
                    subspace_name=str(task["subspace"]),
                )
                plus_rows = _score_rows(
                    probe.plus.scoring,
                    probe.plus.margins,
                    trajectory,
                    args=args,
                    task=task,
                    repo_root=repo_root,
                    intervention_mode="additive_antithetic",
                    requested={"h": task["h"], "sign": 1, "probe_seed": task["probe_seed"], "direction_id": direction_id, "subspace": task["subspace"]},
                    realized_rows=[_to_plain(value) for value in probe.plus_diagnostics],
                )
                minus_rows = _score_rows(
                    probe.minus.scoring,
                    probe.minus.margins,
                    trajectory,
                    args=args,
                    task=task,
                    repo_root=repo_root,
                    intervention_mode="additive_antithetic",
                    requested={"h": task["h"], "sign": -1, "probe_seed": task["probe_seed"], "direction_id": direction_id, "subspace": task["subspace"]},
                    realized_rows=[_to_plain(value) for value in probe.minus_diagnostics],
                )
                for index in selected_probe_indices:
                    raw_id = trajectory.raw_sample_ids[index]
                    plus_run_id = content_hash(
                        {"task": task["config_key"], "raw_sample_id": raw_id, "direction_id": direction_id, "sign": 1},
                        domain="linkradius:signed_probe_run:v1",
                    )
                    minus_run_id = content_hash(
                        {"task": task["config_key"], "raw_sample_id": raw_id, "direction_id": direction_id, "sign": -1},
                        domain="linkradius:signed_probe_run:v1",
                    )
                    plus_rows[index]["run_id"] = plus_run_id
                    minus_rows[index]["run_id"] = minus_run_id
                    plus_rows[index]["q"] = int(probe.subspace["q"])
                    minus_rows[index]["q"] = int(probe.subspace["q"])
                    plus_rows[index]["subspace_id"] = probe.subspace["subspace_id"]
                    minus_rows[index]["subspace_id"] = probe.subspace["subspace_id"]
                    plus_diagnostics = _bind_probe_run_ids(
                        probe.plus_diagnostics[index], run_id=plus_run_id
                    )
                    minus_diagnostics = _bind_probe_run_ids(
                        probe.minus_diagnostics[index], run_id=minus_run_id
                    )
                    pair_diagnostics = _bind_probe_run_ids(
                        probe.pair_diagnostics[index],
                        plus_run_id=plus_run_id,
                        minus_run_id=minus_run_id,
                    )
                    plus_rows[index]["realized_intervention"] = plus_diagnostics
                    minus_rows[index]["realized_intervention"] = minus_diagnostics
                    derivatives: dict[str, float | None] = {}
                    derivative_error = None
                    try:
                        t_plus = float(pair_diagnostics["t_plus"])
                        t_minus = float(pair_diagnostics["t_minus"])
                        separation = t_plus - t_minus
                        if not math.isfinite(separation) or separation <= 0:
                            raise ContractError("probe pair has non-positive realized separation")
                        derivatives = {
                            competitor: (
                                float(plus_rows[index]["margins"][competitor])
                                - float(minus_rows[index]["margins"][competitor])
                            )
                            / separation
                            for competitor in plus_rows[index]["margins"]
                        }
                    except (KeyError, TypeError, ValueError, ContractError) as exc:
                        derivative_error = str(exc)
                        derivatives = {
                            competitor: None for competitor in plus_rows[index]["margins"]
                        }
                        if bool(pair_diagnostics.get("accepted", False)):
                            raise ContractError(
                                "accepted probe pair lacks a valid realized-coordinate derivative"
                            ) from exc
                    rows.extend(
                        [
                            {**plus_rows[index], "edge_id": edge, "sign": 1, "direction_id": direction_id, "probe_seed": task["probe_seed"], "h": task["h"], "diagnostics": plus_diagnostics},
                            {**minus_rows[index], "edge_id": edge, "sign": -1, "direction_id": direction_id, "probe_seed": task["probe_seed"], "h": task["h"], "diagnostics": minus_diagnostics},
                            {
                                **_row_envelope(
                                    args=args,
                                    task=task,
                                    trajectory=trajectory,
                                    index=index,
                                    repo_root=repo_root,
                                    intervention_mode="additive_antithetic_pair",
                                    requested={"h": task["h"], "probe_seed": task["probe_seed"], "direction_id": direction_id, "subspace": task["subspace"]},
                                    realized=pair_diagnostics,
                                ),
                                "record_type": "probe_pair",
                                "run_id": content_hash(
                                    {"plus_run_id": plus_run_id, "minus_run_id": minus_run_id},
                                    domain="linkradius:probe_pair_run:v1",
                                ),
                                "gold": trajectory.gold_labels[index],
                                "clean_margins": trajectory.clean_margins[index],
                                "plus_run_id": plus_run_id,
                                "minus_run_id": minus_run_id,
                                "direction_id": direction_id,
                                "probe_seed": task["probe_seed"],
                                "h": task["h"],
                                "q": int(probe.subspace["q"]),
                                "subspace_id": probe.subspace["subspace_id"],
                                **pair_diagnostics,
                                "central_differences": derivatives,
                                "derivative_error": derivative_error,
                                "margins_plus": plus_rows[index]["margins"],
                                "margins_minus": minus_rows[index]["margins"],
                            },
                        ]
                    )
            artifact_name = "probe_runs.jsonl"
        elif task["stage"] == "gradient":
            import torch
            from RecursiveMAS.inference_utils import linkradius as lr
            from RecursiveMAS.inference_utils.linkradius_runtime import ReplayIntervention

            sample_index = _autograd_sample_index(trajectory)
            gradient = runtime.autograd_gradient(
                trajectory,
                edge,
                sample_index=sample_index,
            )
            gradient_tensor = gradient.gradient.float().cpu()
            subspace = lr.get_subspace(
                str(task.get("subspace") or "full_tensor"),
                int(gradient_tensor.shape[-2]),
                int(gradient_tensor.shape[-1]),
            )
            gradient_coordinates = subspace.adjoint(gradient_tensor[0])
            subspace_gradient_norm = float(
                torch.linalg.vector_norm(gradient_coordinates).item()
            )
            if subspace_gradient_norm:
                direction = subspace.lift(
                    gradient_coordinates / subspace_gradient_norm
                )
            else:
                direction = torch.zeros_like(gradient_tensor[0])
            clean = trajectory.message(edge, receiver=True).float().cpu()
            target = gradient.target_label
            finite_difference = None
            consumer_dtype = trajectory.dtype_metadata(edge).consumer_dtype
            for radius in args.finite_difference_radii:
                if subspace_gradient_norm == 0:
                    break
                plus_delta = torch.zeros_like(clean)
                minus_delta = torch.zeros_like(clean)
                plus_delta[sample_index] = lr.requested_additive_delta(
                    clean[sample_index], direction, h=radius, sign=1
                )
                minus_delta[sample_index] = lr.requested_additive_delta(
                    clean[sample_index], direction, h=radius, sign=-1
                )
                plus_diag = lr.realized_delta_diagnostics(clean[sample_index], plus_delta[sample_index], consumer_dtype=consumer_dtype, lifted_unit_direction=direction)
                minus_diag = lr.realized_delta_diagnostics(clean[sample_index], minus_delta[sample_index], consumer_dtype=consumer_dtype, lifted_unit_direction=direction)
                if plus_diag.collapsed or minus_diag.collapsed:
                    continue
                separation = float(plus_diag.realized_signed_coordinate) - float(minus_diag.realized_signed_coordinate)
                if separation <= 0:
                    continue
                plus = runtime.replay(trajectory, edge, ReplayIntervention(mode="additive", delta=plus_delta), differentiable=False)
                minus = runtime.replay(trajectory, edge, ReplayIntervention(mode="additive", delta=minus_delta), differentiable=False)
                plus_margin = plus.margins[sample_index][target]
                minus_margin = minus.margins[sample_index][target]
                derivative = (plus_margin - minus_margin) / separation
                autograd_scaled = subspace_gradient_norm * float(torch.linalg.vector_norm(clean[sample_index]).item())
                relative_error = abs(derivative - autograd_scaled) / max(abs(derivative), abs(autograd_scaled), 1e-12)
                finite_difference = {
                    "h": radius,
                    "target": target,
                    "realized_separation": separation,
                    "finite_difference_derivative": derivative,
                    "autograd_dimensionless_derivative": autograd_scaled,
                    "relative_error": relative_error,
                    "agrees": relative_error <= args.autograd_fd_relative_tolerance,
                    "plus_diagnostics": plus_diag.to_dict(),
                    "minus_diagnostics": minus_diag.to_dict(),
                }
                break
            pgd_summary: dict[str, Any]
            try:
                pgd = runtime.autograd_pgd(
                    trajectory,
                    edge,
                    epsilon=args.engineering_pgd_epsilon,
                    steps=args.pgd_steps,
                    subspace_name=str(task.get("subspace") or "full_tensor"),
                    sample_index=sample_index,
                )
            except (RuntimeError, ValueError) as exc:
                pgd_summary = {"supported": False, "failure": str(exc), "autograd_semantics": "not_substituted"}
            else:
                pgd_summary = {
                    "supported": True,
                    "epsilon": pgd.epsilon,
                    "steps": pgd.steps,
                    "step_size": pgd.step_size,
                    "autograd_semantics": pgd.autograd_semantics,
                    "sample_index": pgd.sample_index,
                    "sample_id": pgd.sample_id,
                    "strongest_target": pgd.strongest_target,
                    "targets": [
                        {
                            "target_label": value.target_label,
                            "initial_margin": value.initial_margin,
                            "final_margin": value.final_margin,
                            "improved": value.improved,
                            "requested_delta_norm": value.requested_delta_norm,
                            "realized_delta_norm": value.realized_delta_norm,
                            "budget": value.budget,
                            "budget_respected": value.budget_respected,
                        }
                        for value in pgd.targets
                    ],
                }
            row = {
                "record_type": "gradient",
                "edge_id": edge,
                "sample_index": sample_index,
                "sample_id": trajectory.sample_ids[sample_index],
                "raw_sample_id": trajectory.raw_sample_ids[sample_index],
                "frozen_batch_size": len(trajectory.sample_ids),
                "gold_label": gradient.gold_label,
                "target_label": gradient.target_label,
                "objective_name": gradient.objective_name,
                "objective_value": gradient.objective_value,
                "gradient_norm": gradient.gradient_norm,
                "subspace_gradient_norm": subspace_gradient_norm,
                "gradient_sha256": content_hash(_to_plain(gradient_tensor), domain="linkradius:gradient_tensor:v1"),
                "subspace_gradient_sha256": content_hash(
                    _to_plain(gradient_coordinates),
                    domain="linkradius:subspace_gradient:v1",
                ),
                "autograd_semantics": gradient.autograd_semantics,
                "subspace": subspace.name,
                "subspace_id": subspace.subspace_id,
                "q": subspace.q,
                "split_manifest_hash": trajectory.provenance.get("split_manifest_hash"),
                "execution_manifest_hash": trajectory.execution_manifest_hash,
                "ordered_cohort_hash": trajectory.provenance.get("global_ordered_cohort_hash")
                or trajectory.ordered_cohort_hash,
                "batch_boundary_hash": trajectory.provenance.get("global_batch_boundary_hash")
                or trajectory.batch_boundary_hash,
                "source_hash": _cached_source_hash(args, repo_root),
                "finite_difference": finite_difference,
                "pgd": pgd_summary,
            }
            rows = [row]
            artifact_name = "gradient_runs.jsonl"
        else:
            raise ContractError(f"unsupported runtime replay stage: {task['stage']}")
    finally:
        runtime.unload()
    sample_count = len(rows)
    for row in rows:
        if row.get("record_type") == "sample":
            validate_intervention_row(row)
    rows.append(
        {"record_type": "shard_metadata", "array_index": task["array_index"], "config_key": task["config_key"], "row_count": sample_count}
    )
    atomic_write_jsonl(task_dir / artifact_name, rows, overwrite=args.overwrite)
    publish_completion(
        task_dir,
        config_hash=str(task["config_key"]),
        source_hash_value=_cached_source_hash(args, repo_root),
        artifact_paths=["manifest.json", "command.txt", artifact_name],
        row_counts={artifact_name: len(rows)},
        extra={"array_index": int(task["array_index"]), "execution_batch_id": task["execution_batch_id"]},
        overwrite=args.overwrite,
    )


def _attack_stage(args: argparse.Namespace, task_dir: Path, task: Mapping[str, Any], repo_root: Path) -> None:
    """Run every frozen dose for one batch/edge/family under one runtime load."""

    args._active_task = task
    producer_workflow = (
        "pilot"
        if args.workflow == "attacks" and str(task["partition"]) == "validation"
        else None
    )
    producer_args = _producer_namespace(args, producer_workflow)
    trajectory_path = _resolve_trajectory_path(
        args, task, producer_workflow=producer_workflow
    )
    args._active_trajectory_path = str(trajectory_path.resolve())
    trajectory = _load_trajectory(trajectory_path)
    execution_path = _execution_manifest_path(
        producer_args, str(task["partition"]), task
    )
    authenticated_execution_hash = (
        _authenticated_execution_manifest(
            producer_args, str(task["partition"]), task, repo_root
        )[2]
        if execution_path
        else None
    )
    if not execution_path or trajectory.execution_manifest_hash != authenticated_execution_hash:
        raise ContractError("attack trajectory is not tied to the selected execution manifest")
    edge = str(task["edge_id"])
    family = str(task["attack_family"])
    metadata = task.get("metadata")
    raw_budgets = (
        metadata.get("attack_epsilons", [])
        if isinstance(metadata, Mapping)
        else []
    )
    if not raw_budgets and task.get("epsilon") is not None:
        raw_budgets = [task["epsilon"]]
    budgets = [float(value) for value in raw_budgets]
    if (
        not budgets
        or any(not math.isfinite(value) or value <= 0.0 for value in budgets)
        or budgets != sorted(set(budgets))
    ):
        raise ContractError("attack budgets must be unique, positive, finite, and increasing")
    eligible_indices = (
        _fresh_dual_correct_trajectory_indices(trajectory)
        if args.workflow == "attacks"
        else [
            index
            for index, eligible in enumerate(trajectory.analysis_eligibility_mask)
            if bool(eligible)
        ]
    )
    if not eligible_indices:
        _publish_outcome_excluded_task(
            args,
            task_dir,
            task,
            repo_root,
            artifact_name="attack_results.jsonl",
        )
        return
    runtime = _runtime(args, requested_edge=edge)
    rows: list[dict[str, Any]] = []
    try:
        if family == "random_independent":
            import torch
            from RecursiveMAS.inference_utils import linkradius as lr
            from RecursiveMAS.inference_utils.linkradius_runtime import ReplayIntervention

            clean = trajectory.message(edge, receiver=True).float().cpu()
            subspace = lr.get_subspace(
                str(task["subspace"]), int(clean.size(1)), int(clean.size(2))
            )
            attack_seed = int(task["seed"]) + int(args.random_attack_seed_offset)
            directions = []
            direction_seeds = []
            for raw_id in trajectory.raw_sample_ids:
                direction = lr.sample_stable_unit_direction(
                    attack_seed,
                    raw_id,
                    edge,
                    subspace,
                    probe_seed=0,
                    direction_id=0,
                    purpose="random_independent_attack",
                )
                directions.append(subspace.lift(direction).float().cpu())
                direction_seeds.append(
                    lr.stable_intervention_seed(
                        attack_seed,
                        raw_id,
                        edge,
                        probe_seed=0,
                        direction_id=0,
                        purpose="random_independent_attack",
                    )
                )
            consumer_dtype = trajectory.dtype_metadata(edge).consumer_dtype
            for epsilon in budgets:
                delta = torch.stack(
                    [
                        lr.postcast_budget_fitted_delta(
                            clean[index],
                            directions[index],
                            relative_budget=epsilon,
                            consumer_dtype=consumer_dtype,
                        )
                        for index in range(len(trajectory.raw_sample_ids))
                    ],
                    dim=0,
                )
                result = runtime.replay(
                    trajectory,
                    edge,
                    ReplayIntervention(
                        mode="additive",
                        delta=delta,
                        metadata={
                            "attack_family": family,
                            "requested_epsilon": epsilon,
                            "attack_seed": attack_seed,
                        },
                    ),
                )
                diagnostics = [
                    {
                        **lr.realized_delta_diagnostics(
                            clean[index],
                            delta[index],
                            consumer_dtype=consumer_dtype,
                            lifted_unit_direction=directions[index],
                        ).to_dict(),
                        "attack_seed": attack_seed,
                        "direction_seed": direction_seeds[index],
                        "requested_epsilon": epsilon,
                    }
                    for index in range(len(trajectory.raw_sample_ids))
                ]
                scored = _score_rows(
                    result.scoring,
                    result.margins,
                    trajectory,
                    args=args,
                    task=task,
                    repo_root=repo_root,
                    intervention_mode="additive",
                    requested={
                        "requested_epsilon": epsilon,
                        "attack_family": family,
                        "seed_domain": "random_independent_attack",
                    },
                    realized_rows=diagnostics,
                )
                for index in eligible_indices:
                    row = scored[index]
                    row.update(
                        {
                            "run_id": content_hash(
                                {
                                    "task": task["config_key"],
                                    "raw_sample_id": row["raw_sample_id"],
                                    "requested_epsilon": epsilon,
                                },
                                domain="linkradius:attack_run:v1",
                            ),
                            "attack_family": family,
                            "attack_seed": attack_seed,
                            "attack_restart": 0,
                            "requested_epsilon": epsilon,
                            "realized_epsilon": row["realized_intervention"].get(
                                "realized_relative_norm"
                            ),
                            "clean_margins": {
                                str(label): float(value)
                                for label, value in trajectory.clean_margins[index].items()
                            },
                            "minimum_clean_margin": min(
                                float(value)
                                for value in trajectory.clean_margins[index].values()
                            ),
                            "flipped": float(row["minimum_margin"]) <= 0.0,
                        }
                    )
                    rows.append(row)
        elif family == "pgd_autograd":
            import torch

            labels = list(trajectory.clean_scoring.labels)
            clean_receiver = trajectory.message(edge, receiver=True).float().cpu()
            for sample_index in eligible_indices:
                clean_norm = float(
                    torch.linalg.vector_norm(clean_receiver[sample_index]).item()
                )
                for epsilon in budgets:
                    try:
                        result = runtime.autograd_pgd(
                            trajectory,
                            edge,
                            epsilon=epsilon,
                            steps=args.pgd_steps,
                            subspace_name=str(task["subspace"]),
                            sample_index=sample_index,
                        )
                    except (RuntimeError, ValueError, IndexError) as exc:
                        if args.workflow == "attacks":
                            raise ContractError(
                                "PGD attack execution failed; no completed attack "
                                "artifact will be published for "
                                f"raw_sample_id={trajectory.raw_sample_ids[sample_index]}, "
                                f"edge={edge}, requested_epsilon={epsilon}: {exc}"
                            ) from exc
                        rows.append(
                            {
                                **_row_envelope(
                                    args=args,
                                    task=task,
                                    trajectory=trajectory,
                                    index=sample_index,
                                    repo_root=repo_root,
                                    intervention_mode="pgd_autograd",
                                    requested={
                                        "requested_epsilon": epsilon,
                                        "steps": args.pgd_steps,
                                    },
                                ),
                                "record_type": "unsupported",
                                "attack_family": family,
                                "attack_seed": int(task["seed"]),
                                "attack_restart": 0,
                                "requested_epsilon": epsilon,
                                "gold": trajectory.gold_labels[sample_index],
                                "option_scores": {},
                                "margins": {},
                                "failure": "unsupported_or_oom",
                                "unsupported_reason": str(exc),
                                "autograd_semantics": "not_substituted",
                                "sample_index": sample_index,
                                "frozen_batch_size": len(trajectory.sample_ids),
                            }
                        )
                        continue
                    strongest = next(
                        (
                            value
                            for value in result.targets
                            if value.target_label == result.strongest_target
                        ),
                        result.targets[0],
                    )
                    gold = trajectory.gold_labels[sample_index]

                    def scored_values(target_result: Any) -> tuple[dict[str, float], dict[str, float], str | None, bool]:
                        option_scores = {
                            label: float(target_result.scores[pos])
                            for pos, label in enumerate(labels)
                        }
                        margins = {
                            label: option_scores[gold] - option_scores[label]
                            for label in labels
                            if label != gold
                        }
                        maximum = max(option_scores.values())
                        winners = [
                            label for label in labels if option_scores[label] == maximum
                        ]
                        return (
                            option_scores,
                            margins,
                            winners[0] if len(winners) == 1 else None,
                            len(winners) > 1,
                        )

                    option_scores, margins, prediction, score_tie = scored_values(
                        strongest
                    )
                    realized_epsilon = (
                        float(strongest.realized_delta_norm) / clean_norm
                        if clean_norm
                        else None
                    )
                    requested_epsilon = (
                        float(strongest.requested_delta_norm) / clean_norm
                        if clean_norm
                        else None
                    )
                    realized = {
                        "target_label": strongest.target_label,
                        "initial_margin": float(strongest.initial_margin),
                        "final_margin": float(strongest.final_margin),
                        "requested_delta_norm": float(strongest.requested_delta_norm),
                        "realized_delta_norm": float(strongest.realized_delta_norm),
                        "requested_relative_norm": requested_epsilon,
                        "realized_relative_norm": realized_epsilon,
                        "absolute_budget": float(strongest.budget),
                        "budget_respected": bool(strongest.budget_respected),
                    }
                    summary = {
                        **_row_envelope(
                            args=args,
                            task=task,
                            trajectory=trajectory,
                            index=sample_index,
                            repo_root=repo_root,
                            intervention_mode="pgd_autograd",
                            requested={
                                "requested_epsilon": epsilon,
                                "steps": args.pgd_steps,
                                "step_size": result.step_size,
                                "initialization": "zero",
                                "restart": 0,
                            },
                            realized=realized,
                        ),
                        "run_id": content_hash(
                            {
                                "task": task["config_key"],
                                "raw_sample_id": trajectory.raw_sample_ids[sample_index],
                                "requested_epsilon": epsilon,
                            },
                            domain="linkradius:attack_run:v1",
                        ),
                        "gold": gold,
                        "option_scores": option_scores,
                        "margins": margins,
                        "minimum_margin": min(margins.values()),
                        "binding_competitor": min(margins, key=margins.get),
                        "scorer_prediction": prediction,
                        "scorer_correct": prediction == gold,
                        "score_tie": score_tie,
                        "strict_generated_choice": None,
                        "strict_generated_valid": None,
                        "strict_generated_correct": None,
                        "autograd_semantics": result.autograd_semantics,
                        "attack_family": family,
                        "attack_seed": int(task["seed"]),
                        "attack_restart": 0,
                        "requested_epsilon": epsilon,
                        "realized_epsilon": realized_epsilon,
                        "clean_margins": {
                            str(label): float(value)
                            for label, value in trajectory.clean_margins[sample_index].items()
                        },
                        "minimum_clean_margin": min(
                            float(value)
                            for value in trajectory.clean_margins[sample_index].values()
                        ),
                        "flipped": min(margins.values()) <= 0.0,
                        "sample_index": sample_index,
                        "frozen_batch_size": len(trajectory.sample_ids),
                        "pgd_target_count": len(result.targets),
                    }
                    rows.append(summary)
                    for target_result in result.targets:
                        target_scores, target_margins, target_prediction, target_tie = scored_values(
                            target_result
                        )
                        target_realized_epsilon = (
                            float(target_result.realized_delta_norm) / clean_norm
                            if clean_norm
                            else None
                        )
                        target_requested_epsilon = (
                            float(target_result.requested_delta_norm) / clean_norm
                            if clean_norm
                            else None
                        )
                        rows.append(
                            {
                                **summary,
                                "record_type": "attack_target",
                                "run_id": content_hash(
                                    {
                                        "summary_run_id": summary["run_id"],
                                        "target_label": target_result.target_label,
                                    },
                                    domain="linkradius:attack_target_run:v1",
                                ),
                                "target_label": target_result.target_label,
                                "competitor": target_result.target_label,
                                "option_scores": target_scores,
                                "margins": target_margins,
                                "minimum_margin": float(
                                    target_margins[target_result.target_label]
                                ),
                                "binding_competitor": target_result.target_label,
                                "scorer_prediction": target_prediction,
                                "scorer_correct": target_prediction == gold,
                                "score_tie": target_tie,
                                "flipped": float(
                                    target_margins[target_result.target_label]
                                )
                                <= 0.0,
                                "realized_epsilon": target_realized_epsilon,
                                "realized_intervention": {
                                    "target_label": target_result.target_label,
                                    "initial_margin": float(
                                        target_result.initial_margin
                                    ),
                                    "final_margin": float(target_result.final_margin),
                                    "requested_delta_norm": float(
                                        target_result.requested_delta_norm
                                    ),
                                    "realized_delta_norm": float(
                                        target_result.realized_delta_norm
                                    ),
                                    "requested_relative_norm": target_requested_epsilon,
                                    "realized_relative_norm": target_realized_epsilon,
                                    "absolute_budget": float(target_result.budget),
                                    "budget_respected": bool(
                                        target_result.budget_respected
                                    ),
                                },
                            }
                        )
        else:
            raise ContractError(
                f"failure-boundary v1 does not support attack family: {family}"
            )
    finally:
        runtime.unload()
    sample_count = len(rows)
    for row in rows:
        if row.get("record_type") == "sample":
            validate_intervention_row(row)
    rows.append(
        {
            "record_type": "shard_metadata",
            "array_index": task["array_index"],
            "config_key": task["config_key"],
            "row_count": sample_count,
        }
    )
    atomic_write_jsonl(
        task_dir / "attack_results.jsonl", rows, overwrite=args.overwrite
    )
    publish_completion(
        task_dir,
        config_hash=str(task["config_key"]),
        source_hash_value=_cached_source_hash(args, repo_root),
        artifact_paths=["manifest.json", "command.txt", "attack_results.jsonl"],
        row_counts={"attack_results.jsonl": len(rows)},
        extra={
            "array_index": int(task["array_index"]),
            "execution_batch_id": task["execution_batch_id"],
        },
        overwrite=args.overwrite,
    )


def _test_outcome_evidence(test_root: Path) -> list[str]:
    """Return any durable sign that a held-out outcome stage has started.

    A completion marker alone is too weak: a killed job can already have
    written a manifest, log, partial/final outcome artifact, or pending marker.
    Once that happens the output root is contaminated for a pre-outcome freeze,
    even if the task never finalized successfully.
    """

    evidence: list[str] = []
    for stage in ("clean", "test_probe", "test"):
        stage_root = test_root / stage
        if not stage_root.is_dir():
            continue
        evidence.extend(
            str(path)
            for path in sorted(stage_root.rglob("*"))
            if path.is_file() or path.is_symlink()
        )
    return evidence


def _freeze_heldout_execution(
    args: argparse.Namespace,
    task_dir: Path,
    task: Mapping[str, Any],
    repo_root: Path,
) -> None:
    """Freeze the raw held-out cohort without reading a single test outcome."""

    if str(task["partition"]) != "test":
        raise ContractError("held-out execution freeze is restricted to test")
    test_root = (
        _phase_root(args)
        / str(task["dataset"])
        / f"R{int(task['R'])}"
        / "test"
    )
    contaminated = _test_outcome_evidence(test_root)
    if contaminated:
        raise ContractError(
            "held-out execution must be frozen before any test outcome task; "
            f"found {contaminated[:3]}"
        )
    split, split_hash = _authenticated_split_manifest(args, task, repo_root)
    partition_rows = list(split["partitions"]["test"])
    screening_rows = []
    for value in partition_rows:
        row = value if isinstance(value, Mapping) else {"raw_sample_id": value}
        raw_id = str(row["raw_sample_id"])
        screening_rows.append(
            {
                "raw_sample_id": raw_id,
                "sample_id": raw_id,
                "raw_index": row.get("raw_index"),
                "analysis_eligible": True,
                "dual_correct": None,
                "exclusion_reason": "",
            }
        )
    freeze_protocol = {
        "schema_version": "linkradius.heldout_execution_freeze.v1",
        "partition": "test",
        "split_manifest_hash": split_hash,
        "batch_size": int(args.batch_size),
        "selection": "all_raw_test_rows_before_outcomes",
        "outcomes_observed": False,
    }
    manifest = build_execution_manifest(
        split_manifest=split,
        partition="test",
        screening_rows=screening_rows,
        batch_size=int(args.batch_size),
        batches_per_shard=int(args.batches_per_shard),
        screening_config_hash=content_hash(
            freeze_protocol, domain="linkradius:heldout_execution_freeze:v1"
        ),
        screening_run_hash=content_hash(
            [row["raw_sample_id"] for row in screening_rows],
            domain="linkradius:heldout_raw_cohort:v1",
        ),
        retain_all_partition_rows=True,
    )
    output = (
        Path(args.execution_output)
        if args.execution_output
        else test_root / "execution_manifest.json"
    )
    if output.exists() and not args.overwrite:
        existing_hash = verify_execution_manifest(
            load_json(output), split_manifest=split
        )
        if existing_hash != manifest["content_hash"]:
            raise ContractError("existing held-out execution manifest is incompatible")
    else:
        atomic_write_json(output, manifest, overwrite=True)
    atomic_write_json(
        task_dir / "execution_manifest.json", manifest, overwrite=args.overwrite
    )
    atomic_write_json(
        task_dir / "freeze_execution_result.json",
        {
            "path": str(output.resolve()),
            "content_hash": manifest["content_hash"],
            "outcomes_observed": False,
            "row_count": len(screening_rows),
        },
        overwrite=args.overwrite,
    )
    publish_completion(
        task_dir,
        config_hash=str(task["config_key"]),
        source_hash_value=_cached_source_hash(args, repo_root),
        artifact_paths=[
            "manifest.json",
            "command.txt",
            "freeze_execution_result.json",
            "execution_manifest.json",
        ],
        extra={
            "array_index": int(task["array_index"]),
            "execution_manifest_hash": manifest["content_hash"],
        },
        overwrite=args.overwrite,
    )


def _freeze_execution(args: argparse.Namespace, task_dir: Path, task: Mapping[str, Any], repo_root: Path) -> None:
    if args.workflow == "attacks":
        _freeze_heldout_execution(args, task_dir, task, repo_root)
        return
    screening_stage = {
        "engineering": "discover",
        "smoke": "screen",
        "pilot": "screen_clean",
    }[args.workflow]
    screening_paths = list(args.screening_jsonl)
    if not screening_paths:
        screening_root = (
            _phase_root(args)
            / str(task["dataset"])
            / f"R{int(task['R'])}"
            / str(task["partition"])
            / screening_stage
        )
        screening_paths = [
            str(path)
            for path in sorted(screening_root.rglob("screening_rows.jsonl"))
            if (path.parent / ".complete.json").is_file()
        ]
    if not screening_paths:
        raise ContractError("freeze_execution found no completed screening JSONL shards")
    expected_tasks = {
        candidate.array_index: candidate.as_dict()
        for candidate in build_grid(_build_grid_config(args, screening_stage))
        if candidate.dataset == str(task["dataset"])
        and int(candidate.R) == int(task["R"])
        and candidate.partition == str(task["partition"])
    }
    indexed_shards: dict[int, list[dict[str, Any]]] = {}
    for raw_path in screening_paths:
        path = Path(raw_path)
        if path.name != "screening_rows.jsonl":
            raise ContractError("screening shards must be named screening_rows.jsonl")
        completion = verify_completion(path.parent)
        if completion.get("source_hash") != _cached_source_hash(args, repo_root):
            raise ContractError("screening shard has a stale source hash")
        declared = [
            artifact
            for artifact in completion["artifacts"]
            if artifact.get("path") == path.name
        ]
        manifest_entries = [
            artifact
            for artifact in completion["artifacts"]
            if artifact.get("path") == "manifest.json"
        ]
        if len(declared) != 1 or len(manifest_entries) != 1:
            raise ContractError(
                "screening shard and manifest must be uniquely declared by their completion"
            )
        shard_rows = load_jsonl(path)
        metadata = [row for row in shard_rows if row.get("record_type") == "shard_metadata"]
        if len(metadata) != 1:
            raise ContractError("screening shard must contain one shard_metadata row")
        array_index = int(metadata[0].get("array_index", -1))
        config_key = str(metadata[0].get("config_key") or "")
        expected_task = expected_tasks.get(array_index)
        if expected_task is None or expected_task["config_key"] != config_key:
            raise ContractError("screening shard index/config does not match the canonical grid")
        manifest = load_json(path.parent / "manifest.json")
        sample_rows = [row for row in shard_rows if row.get("record_type") == "sample"]
        if (
            path.parent.resolve() != task_output_dir(args, expected_task).resolve()
            or manifest.get("task") != expected_task
            or completion.get("config_hash") != expected_task["config_key"]
            or int(completion.get("array_index", -1)) != array_index
            or int(completion.get("execution_batch_id", -1))
            != int(expected_task["execution_batch_id"])
            or int(metadata[0].get("row_count", -1)) != len(sample_rows)
            or int(declared[0].get("row_count", -1)) != len(shard_rows)
        ):
            raise ContractError(
                "screening shard completion, task manifest, metadata, or row count is incompatible"
            )
        if array_index in indexed_shards:
            raise ContractError("duplicate screening array index")
        indexed_shards[array_index] = sample_rows
    if set(indexed_shards) != set(expected_tasks):
        missing = sorted(set(expected_tasks) - set(indexed_shards))
        raise ContractError(f"freeze_execution screening grid is incomplete: {missing}")
    rows = [
        row
        for array_index in sorted(indexed_shards)
        for row in indexed_shards[array_index]
    ]
    max_eligible = 1 if args.workflow == "engineering" else (args.max_eligible or None)
    if args.workflow == "smoke" and (max_eligible is None or not 10 <= max_eligible <= 20):
        raise ContractError("smoke freeze_execution requires --max-eligible in the frozen range 10..20")
    annotated, summary = annotate_screening_rows(rows, max_eligible=max_eligible)
    if args.workflow == "smoke" and not 10 <= int(summary["dual_correct_count"]) <= 20:
        raise ContractError(
            f"smoke requires 10..20 dual-correct rows; screening produced {summary['dual_correct_count']}"
        )
    manifest_rows = annotated
    if args.workflow == "engineering":
        manifest_rows = [row for row in annotated if row["analysis_eligible"]]
        if len(manifest_rows) != 1:
            raise ContractError(
                f"engineering freeze requires exactly one fixed dual-correct row; found {len(manifest_rows)}"
            )
    authenticated_split, _ = _authenticated_split_manifest(args, task, repo_root)
    if args.workflow == "pilot":
        partition_ids = {
            str(row["raw_sample_id"] if isinstance(row, Mapping) else row)
            for row in authenticated_split["partitions"][str(task["partition"])]
        }
        screened_ids = [str(row.get("raw_sample_id") or "") for row in rows]
        if (
            len(screened_ids) != len(set(screened_ids))
            or set(screened_ids) != partition_ids
        ):
            missing = sorted(partition_ids - set(screened_ids))
            extra = sorted(set(screened_ids) - partition_ids)
            raise ContractError(
                "pilot screen_clean must cover every raw partition row exactly once; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
    manifest = build_execution_manifest(
        split_manifest=authenticated_split,
        partition=str(task["partition"]),
        screening_rows=manifest_rows,
        batch_size=1 if args.workflow == "engineering" else args.batch_size,
        batches_per_shard=args.batches_per_shard,
        screening_config_hash=content_hash(summary, domain="linkradius:screening_config:v1"),
        screening_run_hash=summary["ordered_raw_id_hash"],
        retain_all_partition_rows=args.workflow == "pilot" and args.retain_all_partition_rows,
    )
    output = (
        Path(args.execution_output)
        if args.execution_output
        else _phase_root(args)
        / str(task["dataset"])
        / f"R{int(task['R'])}"
        / str(task["partition"])
        / "execution_manifest.json"
    )
    if output.exists() and not args.overwrite:
        existing_hash = verify_execution_manifest(load_json(output), split_manifest=authenticated_split)
        if existing_hash != manifest["content_hash"]:
            raise ContractError("existing execution manifest is incompatible")
    else:
        atomic_write_json(output, manifest, overwrite=True)
    atomic_write_json(
        task_dir / "execution_manifest.json", manifest, overwrite=args.overwrite
    )
    atomic_write_json(task_dir / "freeze_execution_result.json", {"path": str(output.resolve()), "content_hash": manifest["content_hash"], "screening_summary": summary}, overwrite=args.overwrite)
    publish_completion(
        task_dir,
        config_hash=str(task["config_key"]),
        source_hash_value=_cached_source_hash(args, repo_root),
        artifact_paths=["manifest.json", "command.txt", "freeze_execution_result.json", "execution_manifest.json"],
        extra={"array_index": int(task["array_index"]), "execution_manifest_hash": manifest["content_hash"]},
        overwrite=args.overwrite,
    )


def _validation_pgd_straddle(
    pgd_rows: Sequence[Mapping[str, Any]],
    budgets: Sequence[float],
) -> dict[str, Any]:
    """Count validation boundaries bracketed on the same example/edge curve."""

    if not budgets:
        raise ContractError("validation PGD straddle requires budgets")
    minimum_budget, maximum_budget = float(budgets[0]), float(budgets[-1])
    curves: dict[tuple[str, str], dict[float, Mapping[str, Any]]] = {}
    for row in pgd_rows:
        curve_key = (str(row["raw_sample_id"]), str(row["edge_id"]))
        budget = float(row["requested_epsilon"])
        curve = curves.setdefault(curve_key, {})
        if budget in curve:
            raise ContractError(f"duplicate validation PGD curve cell: {curve_key}/{budget}")
        curve[budget] = row
    incomplete = [
        curve_key
        for curve_key, curve in curves.items()
        if minimum_budget not in curve or maximum_budget not in curve
    ]
    if incomplete:
        raise ContractError(
            f"validation PGD curves lack endpoint budgets: {incomplete[:3]}"
        )
    safe_at_min = sum(
        float(curve[minimum_budget]["minimum_margin"]) > 0.0
        for curve in curves.values()
    )
    crossed_at_max = sum(
        float(curve[maximum_budget]["minimum_margin"]) <= 0.0
        for curve in curves.values()
    )
    straddled = [
        curve_key
        for curve_key, curve in curves.items()
        if float(curve[minimum_budget]["minimum_margin"]) > 0.0
        and float(curve[maximum_budget]["minimum_margin"]) <= 0.0
    ]
    return {
        "curve_count": len(curves),
        "safe_at_smallest": safe_at_min,
        "crossed_at_largest": crossed_at_max,
        "paired_straddled_curves": len(straddled),
        "paired_straddled_curve_fraction": (
            len(straddled) / len(curves) if curves else 0.0
        ),
    }


def _freeze_attack_stage(
    args: argparse.Namespace,
    task_dir: Path,
    task: Mapping[str, Any],
    repo_root: Path,
) -> None:
    """Freeze the RQ2 attack/probe protocol using validation evidence only."""

    current_source = _cached_source_hash(args, repo_root)
    if int(task["R"]) != 2:
        raise ContractError("failure-boundary v1 is frozen at R=2")
    families = str(args.attack_families).split()
    if families != ["pgd_autograd", "random_independent"]:
        raise ContractError(
            "failure-boundary v1 requires exactly: pgd_autograd random_independent"
        )
    budgets = [float(value) for value in str(args.attack_epsilons).split()]
    if (
        not budgets
        or budgets != sorted(set(budgets))
        or any(not math.isfinite(value) or value <= 0.0 for value in budgets)
    ):
        raise ContractError(
            "validation attack budgets must be unique, positive, finite, and increasing"
        )
    if int(args.pgd_steps) < 1:
        raise ContractError("PGD steps must be positive")

    test_root = (
        _phase_root(args)
        / str(task["dataset"])
        / f"R{int(task['R'])}"
        / "test"
    )
    contaminated = _test_outcome_evidence(test_root)
    if contaminated:
        raise ContractError(
            "freeze_attack refuses pre-existing test outcome evidence: "
            f"{contaminated[:3]}"
        )

    grid_report = _verify_source_stage_grid(
        args, task, repo_root, stage="val", partition="validation"
    )
    validation_root = (
        _phase_root(args)
        / str(task["dataset"])
        / f"R{int(task['R'])}"
        / "validation"
        / "val"
    )
    all_rows = _completed_rows(
        validation_root,
        "attack_results.jsonl",
        expected_source_hash=current_source,
        expected_config_keys=_current_stage_config_keys(
            args, task, stage="val", partition="validation"
        ),
    )
    bad_records = [
        row
        for row in all_rows
        if row.get("record_type") not in {
            "sample",
            "attack_target",
            "shard_metadata",
        }
        or row.get("failure")
    ]
    if bad_records:
        first = bad_records[0]
        raise ContractError(
            "validation attack evidence contains an unsupported/failed row: "
            f"{first.get('unsupported_reason') or first.get('failure') or first.get('record_type')}"
        )
    rows = [row for row in all_rows if row.get("record_type") == "sample"]
    if not rows:
        raise ContractError("freeze_attack found no validation attack samples")
    pgd_target_report = _validate_pgd_target_evidence(
        all_rows, where="validation attack evidence"
    )

    pilot_args = _producer_namespace(args, "pilot")
    split, split_hash = _authenticated_split_manifest(pilot_args, task, repo_root)
    _, validation_execution, validation_execution_hash = (
        _authenticated_execution_manifest(
            pilot_args, "validation", task, repo_root
        )
    )
    _, test_execution, test_execution_hash = _authenticated_execution_manifest(
        args, "test", task, repo_root
    )
    frozen_probe = _authenticated_frozen_probe_config(
        args, task, current_source_hash=current_source
    )
    expected_probe_protocol = {
        "dataset": str(task["dataset"]),
        "R": int(task["R"]),
        "style": str(args.style),
        "method": str(args.method),
        "seed": int(task["seed"]),
        "batch_size": int(args.batch_size),
        "latent_length": int(args.latent_length),
        "subspace": str(args.subspace),
        "runtime": {
            "role_devices": _resolved_role_devices(args),
            "relay_transfer_mode": str(args.relay_transfer_mode),
            "autograd_memory_mode": str(args.autograd_memory_mode),
            "trust_remote_code": int(args.trust_remote_code),
            "round_label_mode": str(args.round_label_mode),
            "environment": _runtime_environment_identity(),
        },
    }
    observed_probe_protocol = {
        key: frozen_probe.get(key) for key in expected_probe_protocol
    }
    if observed_probe_protocol != expected_probe_protocol:
        raise ContractError(
            "frozen Phase-3 probe protocol is incompatible with the "
            "failure-boundary attack configuration"
        )
    if not set(EARLY_R2_EDGES).issubset(set(frozen_probe.get("grid_edges", []))):
        raise ContractError(
            "frozen Phase-3 probe grid does not cover every early R=2 edge"
        )
    validation_system_identity = _common_system_identity(
        all_rows, where="validation attack evidence"
    )
    if validation_system_identity != frozen_probe.get("system_identity"):
        raise ContractError(
            "validation attacks resolved a different model/adapter/scorer/"
            "prompt identity than the frozen Phase-3 probe protocol"
        )
    attack_cast_thresholds = frozen_probe["acceptance_thresholds"]
    eligible_ids = {
        str(raw_id)
        for raw_id, eligible in zip(
            validation_execution["ordered_raw_sample_ids"],
            validation_execution["analysis_eligible"],
        )
        if bool(eligible)
    }
    edges = list(EARLY_R2_EDGES)
    expected = {
        (raw_id, edge, family, epsilon)
        for raw_id in eligible_ids
        for edge in edges
        for family in families
        for epsilon in budgets
    }
    observed: dict[tuple[str, str, str, float], Mapping[str, Any]] = {}
    nonfinite: list[str] = []
    budget_violations: list[str] = []
    attack_quality_violations: list[str] = []
    for row in rows:
        key = (
            str(row.get("raw_sample_id")),
            str(row.get("edge_id")),
            str(row.get("attack_family")),
            float(row.get("requested_epsilon")),
        )
        if key in observed:
            raise ContractError(f"duplicate validation attack cell: {key}")
        observed[key] = row
        margins = row.get("margins")
        if (
            not isinstance(margins, Mapping)
            or not margins
            or any(
                not math.isfinite(float(value)) for value in margins.values()
            )
        ):
            nonfinite.append(str(key))
        realized = row.get("realized_epsilon")
        if realized is None or not math.isfinite(float(realized)):
            nonfinite.append(f"{key}:realized_epsilon")
        elif float(realized) > float(key[3]) + 1e-6 * max(1.0, float(key[3])):
            budget_violations.append(str(key))
        elif float(realized) <= 0.0:
            attack_quality_violations.append(f"{key}:collapsed_budget")
        if row.get("attack_family") == "random_independent":
            diagnostics = row.get("realized_intervention")
            if not isinstance(diagnostics, Mapping):
                attack_quality_violations.append(f"{key}:missing_diagnostics")
            else:
                cosine = diagnostics.get("requested_realized_cosine")
                off_direction = diagnostics.get("off_direction_relative")
                if (
                    bool(diagnostics.get("collapsed", True))
                    or cosine is None
                    or not math.isfinite(float(cosine))
                    or float(cosine)
                    < float(
                        attack_cast_thresholds[
                            "minimum_requested_realized_cosine"
                        ]
                    )
                    or off_direction is None
                    or not math.isfinite(float(off_direction))
                    or float(off_direction)
                    > float(
                        attack_cast_thresholds["maximum_off_direction_relative"]
                    )
                ):
                    attack_quality_violations.append(
                        f"{key}:random_postcast_direction"
                    )
        elif row.get("attack_family") == "pgd_autograd":
            requested = row.get("requested_intervention")
            diagnostics = row.get("realized_intervention")
            if (
                row.get("autograd_semantics") != "relaxed_autograd"
                or int(row.get("pgd_target_count", -1)) != 3
                or not isinstance(requested, Mapping)
                or int(requested.get("steps", -1)) != int(args.pgd_steps)
                or int(requested.get("restart", -1)) != 0
                or not isinstance(diagnostics, Mapping)
                or diagnostics.get("budget_respected") is not True
            ):
                attack_quality_violations.append(f"{key}:pgd_protocol")
        if row.get("analysis_eligible") is not True:
            raise ContractError(f"validation attack emitted an ineligible row: {key}")
    realized_curves: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for key, row in observed.items():
        curve = key[:3]
        realized_curves.setdefault(curve, []).append(
            (float(key[3]), float(row.get("realized_epsilon", float("nan"))))
        )
    nonincreasing_realized_curves: list[str] = []
    for curve, values in realized_curves.items():
        ordered_realized = [
            realized for _, realized in sorted(values, key=lambda item: item[0])
        ]
        if any(
            not math.isfinite(right) or right <= left
            for left, right in zip(ordered_realized, ordered_realized[1:])
        ):
            # Requested budgets define the pre-registered attack curve.  A
            # constrained optimizer can legitimately finish inside its ball,
            # so achieved norms are diagnostic rather than required to be
            # monotone.  Such curves retain requested-grid thresholds, while
            # downstream actual-norm thresholds are ordered by achieved norm.
            nonincreasing_realized_curves.append(str(curve))
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    if (
        missing
        or extra
        or nonfinite
        or budget_violations
        or attack_quality_violations
    ):
        raise ContractError(
            "validation attack cube is invalid; "
            f"missing={missing[:3]}, extra={extra[:3]}, "
            f"nonfinite={nonfinite[:3]}, budget_violations={budget_violations[:3]}, "
            f"quality_violations={attack_quality_violations[:3]}"
        )
    provenance = _provenance_check(
        all_rows,
        partition="validation",
        allowed_raw_ids=eligible_ids,
        split_hash=split_hash,
        execution_hash=validation_execution_hash,
        current_source_hash=current_source,
        ordered_cohort_hash=validation_execution["ordered_cohort_hash"],
        batch_boundary_hash=validation_execution["batch_boundary_hash"],
    )
    if not provenance["passed"]:
        raise ContractError(f"validation attack provenance failed: {provenance}")

    pgd_rows = [row for row in rows if row["attack_family"] == "pgd_autograd"]
    straddle = _validation_pgd_straddle(pgd_rows, budgets)
    safe_at_min = int(straddle["safe_at_smallest"])
    crossed_at_max = int(straddle["crossed_at_largest"])
    paired_straddled_curves = int(straddle["paired_straddled_curves"])
    minimum_straddled_curves = 1
    if paired_straddled_curves < minimum_straddled_curves:
        raise ContractError(
            "validation PGD grid does not straddle an observed boundary on "
            "the same raw-example/edge curve: "
            f"safe_at_min={safe_at_min}, crossed_at_max={crossed_at_max}; "
            f"paired_straddled_curves={paired_straddled_curves}, "
            f"required={minimum_straddled_curves}; "
            "widen ATTACK_EPSILONS and rerun validation"
        )

    probe_seeds = [int(value) for value in frozen_probe["probe_seeds"]]
    if len(probe_seeds) < 3 or len(set(probe_seeds)) != len(probe_seeds):
        raise ContractError(
            "failure-boundary evaluation requires at least three unique "
            "probe seeds in the frozen Phase-3 protocol"
        )
    primary_probe_seed = min(probe_seeds)
    evidence_rows = sorted(
        rows,
        key=lambda row: (
            str(row["raw_sample_id"]),
            str(row["edge_id"]),
            str(row["attack_family"]),
            float(row["requested_epsilon"]),
        ),
    )
    config: dict[str, Any] = {
        "schema_version": "linkradius.frozen_attack_config.v1",
        "experiment_scope": "rq2_failure_boundary_pgd_plus_random_v1",
        "dataset": str(task["dataset"]),
        "R": int(task["R"]),
        "style": str(args.style),
        "method": str(args.method),
        "seed": int(task["seed"]),
        "batch_size": int(args.batch_size),
        "latent_length": int(args.latent_length),
        "edges": edges,
        "subspace": str(args.subspace),
        "runtime": {
            "role_devices": _resolved_role_devices(args),
            "relay_transfer_mode": str(args.relay_transfer_mode),
            "autograd_memory_mode": str(args.autograd_memory_mode),
            "trust_remote_code": int(args.trust_remote_code),
            "round_label_mode": str(args.round_label_mode),
            "environment": _runtime_environment_identity(),
        },
        "attack_families": families,
        "attack_epsilons": budgets,
        "threshold_tie_rule": "minimum_pairwise_gold_margin_le_zero",
        "pgd": {
            "steps": int(args.pgd_steps),
            "restarts": 1,
            "initialization": "zero",
            "step_size_rule": "2_times_relative_budget_times_reference_norm_div_steps",
            "target_rule": "all_three_wrong_labels_keep_minimum_margin",
        },
        "random_independent": {
            "seed_offset": int(args.random_attack_seed_offset),
            "direction_count": 1,
            "direction_reused_across_budgets": True,
            "seed_domain": "random_independent_attack",
        },
        "probe": {
            "h": float(frozen_probe["selected_h"]),
            "K": int(frozen_probe["K"]),
            "seeds": probe_seeds,
            "primary_seed": primary_probe_seed,
            "acceptance_thresholds": frozen_probe["acceptance_thresholds"],
            "frozen_probe_config_hash": frozen_probe["content_hash"],
        },
        "validation_summary": {
            "eligible_raw_ids": len(eligible_ids),
            "attack_cells": len(rows),
            "safe_at_smallest_pgd_budget": safe_at_min,
            "crossed_at_largest_pgd_budget": crossed_at_max,
            "paired_straddled_pgd_curves": paired_straddled_curves,
            "paired_straddled_pgd_curve_fraction": straddle[
                "paired_straddled_curve_fraction"
            ],
            "minimum_paired_straddled_pgd_curves": minimum_straddled_curves,
            "nonincreasing_realized_curves": len(
                nonincreasing_realized_curves
            ),
        },
        "system_identity": validation_system_identity,
        "test_outcomes_accessed_before_freeze": False,
        "source_hash": current_source,
        "split_manifest_hash": split_hash,
        "validation_execution_manifest_hash": validation_execution_hash,
        "test_execution_manifest_hash": test_execution_hash,
        "test_ordered_cohort_hash": test_execution["ordered_cohort_hash"],
        "test_batch_boundary_hash": test_execution["batch_boundary_hash"],
        "validation_attack_evidence_hash": content_hash(
            evidence_rows, domain="linkradius:validation_attack_evidence:v1"
        ),
        "validation_grid_verification": grid_report,
    }
    config["content_hash"] = content_hash(
        config, domain="linkradius:frozen_attack_config:v1"
    )
    output = Path(args.frozen_attack_config)
    if output.exists() and not args.overwrite:
        existing = load_json(output)
        if existing.get("content_hash") != config["content_hash"]:
            raise ContractError(
                "refusing to overwrite an incompatible frozen attack configuration"
            )
    else:
        atomic_write_json(output, config, overwrite=True)

    engineering = require_passed_gate(
        args.engineering_gate, gate_type="engineering_gate"
    )
    smoke = require_passed_gate(args.smoke_gate, gate_type="smoke_gate")
    probe_gate = require_passed_gate(args.probe_gate, gate_type="probe_gate")
    checks = [
        {"name": "no_test_outcomes_before_freeze", "passed": True},
        {"name": "validation_attack_grid_complete", "passed": True, **grid_report},
        {"name": "validation_attack_cube_exact", "passed": True, "cells": len(rows)},
        {"name": "validation_attack_provenance", **provenance},
        pgd_target_report,
        {
            "name": "validation_budget_grid_straddles_boundary",
            "passed": True,
            "safe_at_smallest": safe_at_min,
            "crossed_at_largest": crossed_at_max,
            "paired_straddled_curves": paired_straddled_curves,
            "paired_straddled_curve_fraction": straddle[
                "paired_straddled_curve_fraction"
            ],
            "minimum_required": minimum_straddled_curves,
        },
        {
            "name": "heldout_execution_frozen_without_outcomes",
            "passed": True,
            "test_rows": len(test_execution["ordered_raw_sample_ids"]),
        },
    ]
    gate = make_gate(
        gate_type="attack_freeze_gate",
        checks=checks,
        config_hash=str(task["config_key"]),
        source_hash=current_source,
        prerequisite_hashes={
            "engineering_gate_hash": engineering["gate_content_hash"],
            "smoke_gate_hash": smoke["gate_content_hash"],
            "probe_gate_hash": probe_gate["gate_content_hash"],
            "frozen_attack_config_hash": config["content_hash"],
            "split_manifest_hash": split_hash,
            "validation_execution_manifest_hash": validation_execution_hash,
            "test_execution_manifest_hash": test_execution_hash,
        },
    )
    gate_path = Path(args.attack_freeze_gate)
    atomic_write_json(gate_path, gate, overwrite=args.overwrite)
    _record_global_pointer_completion(
        args,
        task_dir,
        task,
        repo_root,
        pointer_name="attack_freeze_result.json",
        pointer={
            "passed": True,
            "frozen_attack_config": str(output.resolve()),
            "content_hash": config["content_hash"],
            "attack_freeze_gate": str(gate_path.resolve()),
            "gate_content_hash": gate["gate_content_hash"],
        },
    )


def _completed_rows(
    root: Path,
    filename: str,
    *,
    expected_source_hash: str,
    expected_config_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for completion_path in sorted(root.rglob(".complete.json")):
        manifest_path = completion_path.parent / "manifest.json"
        if expected_config_keys is not None:
            if not manifest_path.is_file():
                continue
            candidate_manifest = load_json(manifest_path)
            candidate_task = candidate_manifest.get("task", {})
            if (
                candidate_task.get("stage") != root.name
                or str(candidate_task.get("config_key") or "")
                not in expected_config_keys
            ):
                # Content-addressed historical attempts are intentionally
                # retained, but only the currently requested canonical grid
                # may contribute rows to a freeze or aggregate.
                continue
        path = completion_path.parent / filename
        completion = verify_completion(completion_path.parent)
        if completion.get("source_hash") != expected_source_hash:
            raise ContractError(f"completed {filename} shard has a stale source hash")
        declared = [
            artifact
            for artifact in completion["artifacts"]
            if artifact.get("path") == filename
        ]
        declared_manifest = [
            artifact
            for artifact in completion["artifacts"]
            if artifact.get("path") == "manifest.json"
        ]
        if (
            len(declared) != 1
            or "row_count" not in declared[0]
            or len(declared_manifest) != 1
            or not path.is_file()
            or path.name != filename
        ):
            raise ContractError(
                f"{filename} must be uniquely declared by its completion record"
            )
        shard_rows = load_jsonl(path)
        manifest = load_json(manifest_path)
        manifest_task = manifest.get("task")
        metadata = [
            row
            for row in shard_rows
            if row.get("record_type") == "shard_metadata"
        ]
        sample_count = sum(
            row.get("record_type") != "shard_metadata" for row in shard_rows
        )
        if (
            not isinstance(manifest_task, Mapping)
            or manifest_task.get("stage") != root.name
            or completion.get("config_hash") != manifest_task.get("config_key")
            or int(completion.get("array_index", -1))
            != int(manifest_task.get("array_index", -2))
            or path.parent.name != manifest_task.get("config_key")
            or len(metadata) != 1
            or int(metadata[0].get("array_index", -1))
            != int(manifest_task.get("array_index", -2))
            or metadata[0].get("config_key") != manifest_task.get("config_key")
            or int(metadata[0].get("row_count", -1)) != sample_count
            or int(declared[0]["row_count"]) != len(shard_rows)
        ):
            raise ContractError(
                f"{filename} completion/task identity or shard row count is incompatible"
            )
        rows.extend(shard_rows)
    return rows


def _current_stage_config_keys(
    args: argparse.Namespace,
    task: Mapping[str, Any],
    *,
    stage: str,
    partition: str,
) -> set[str]:
    """Return the exact content-addressed keys in the current canonical grid."""

    return {
        str(candidate.config_key)
        for candidate in build_grid(_build_grid_config(args, stage))
        if candidate.dataset == str(task["dataset"])
        and int(candidate.R) == int(task["R"])
        and candidate.partition == partition
    }


def _verify_source_stage_grid(
    args: argparse.Namespace,
    task: Mapping[str, Any],
    repo_root: Path,
    *,
    stage: str,
    partition: str,
) -> dict[str, Any]:
    """Verify the exact current grid for one dataset/R/partition/stage."""

    from .validate_stage import verify_expected_completions

    expected = [
        candidate
        for candidate in build_grid(_build_grid_config(args, stage))
        if candidate.dataset == str(task["dataset"])
        and int(candidate.R) == int(task["R"])
        and candidate.partition == partition
    ]
    if not expected:
        raise ContractError(f"the canonical {stage} grid has no {partition} tasks")
    stage_root = (
        _phase_root(args)
        / str(task["dataset"])
        / f"R{int(task['R'])}"
        / partition
        / stage
    )
    records = []
    expected_keys = _current_stage_config_keys(
        args, task, stage=stage, partition=partition
    )
    if stage_root.is_dir():
        for completion_path in sorted(stage_root.rglob(".complete.json")):
            manifest_path = completion_path.parent / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = load_json(manifest_path)
            manifest_task = manifest.get("task", {})
            if manifest_task.get("stage") != stage:
                continue
            if str(manifest_task.get("config_key") or "") not in expected_keys:
                # Content-addressed historical attempts intentionally coexist
                # with a retuned validation grid.  Only the exact current task
                # keys participate in the current freeze/audit.
                continue
            records.append((completion_path, verify_completion(completion_path.parent)))
    report = verify_expected_completions(
        expected,
        records,
        expected_source_hash=_cached_source_hash(args, repo_root),
    )
    if not report["passed"]:
        raise ContractError(
            f"{stage}/{partition} completion grid is missing, duplicate, unexpected, or stale: {report}"
        )
    return {"stage": stage, "partition": partition, **report}


def _probe_evidence_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    selected = [
        dict(row)
        for row in rows
        if row.get("record_type") in {"sample", "probe_pair"}
    ]
    selected.sort(
        key=lambda row: (
            str(row.get("record_type")),
            str(row.get("run_id")),
        )
    )
    return content_hash(selected, domain="linkradius:validation_probe_evidence:v1")


def _probe_pair_csv_rows(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        derivatives = pair.get("central_differences", {})
        rows.append(
            {
                "raw_sample_id": pair.get("raw_sample_id"),
                "sample_id": pair.get("sample_id"),
                "edge_id": pair.get("edge_id"),
                "h": pair.get("h"),
                "probe_seed": pair.get("probe_seed"),
                "direction_id": pair.get("direction_id"),
                "q": pair.get("q"),
                "subspace_id": pair.get("subspace_id"),
                "plus_run_id": pair.get("plus_run_id"),
                "minus_run_id": pair.get("minus_run_id"),
                "t_plus": pair.get("t_plus"),
                "t_minus": pair.get("t_minus"),
                "realized_separation": pair.get("realized_separation"),
                "plus_cosine": pair.get("plus_requested_realized_cosine"),
                "minus_cosine": pair.get("minus_requested_realized_cosine"),
                "plus_off_direction_relative": pair.get("plus_off_direction_relative"),
                "minus_off_direction_relative": pair.get("minus_off_direction_relative"),
                "antipodality": pair.get("antipodality"),
                "plus_collapsed": pair.get("plus_collapsed"),
                "minus_collapsed": pair.get("minus_collapsed"),
                "accepted": pair.get("accepted"),
                "rejection_reasons": json.dumps(
                    pair.get("rejection_reasons", []), separators=(",", ":")
                ),
                "central_difference_A": derivatives.get("A"),
                "central_difference_B": derivatives.get("B"),
                "central_difference_C": derivatives.get("C"),
                "central_difference_D": derivatives.get("D"),
                "analysis_eligible": pair.get("analysis_eligible"),
                "split_manifest_hash": pair.get("split_manifest_hash"),
                "execution_manifest_hash": pair.get("execution_manifest_hash"),
                "ordered_cohort_hash": pair.get("ordered_cohort_hash"),
                "batch_boundary_hash": pair.get("batch_boundary_hash"),
                "source_hash": pair.get("source_hash"),
            }
        )
    return rows


def _merge_public_stage_jsonl(
    args: argparse.Namespace,
    task: Mapping[str, Any],
    *,
    stage: str,
    partition: str,
    filename: str,
    output: Path,
) -> int:
    from .merge_shards import merge_to_path

    expected = [
        candidate.as_dict()
        for candidate in build_grid(_build_grid_config(args, stage))
        if candidate.dataset == str(task["dataset"])
        and int(candidate.R) == int(task["R"])
        and candidate.partition == partition
    ]
    root = (
        _phase_root(args)
        / str(task["dataset"])
        / f"R{int(task['R'])}"
        / partition
        / stage
    )
    shards = sorted(root.rglob(filename))
    merge_to_path(
        shards,
        expected_tasks=expected,
        output=output,
        require_completion=True,
        overwrite=args.overwrite,
        expected_source_hash=_cached_source_hash(
            args, Path(__file__).resolve().parents[2]
        ),
    )
    return len(load_jsonl(output))


def _provenance_check(
    rows: Sequence[Mapping[str, Any]],
    *,
    partition: str,
    allowed_raw_ids: set[str],
    split_hash: str,
    execution_hash: str,
    current_source_hash: str,
    ordered_cohort_hash: str | None = None,
    batch_boundary_hash: str | None = None,
) -> dict[str, Any]:
    relevant = [
        row
        for row in rows
        if row.get("record_type") in {"sample", "probe_pair", "attack_target"}
    ]
    violations: list[str] = []
    for row in relevant:
        raw_id = str(row.get("raw_sample_id") or "")
        if raw_id not in allowed_raw_ids:
            violations.append(f"raw_id:{raw_id}")
        for field, expected in (
            ("partition", partition),
            ("split_manifest_hash", split_hash),
            ("execution_manifest_hash", execution_hash),
            ("source_hash", current_source_hash),
            ("ordered_cohort_hash", ordered_cohort_hash),
            ("batch_boundary_hash", batch_boundary_hash),
        ):
            if expected is not None and row.get(field) != expected:
                violations.append(f"{raw_id}:{field}")
    return {
        "name": f"{partition}_row_provenance",
        "passed": bool(relevant) and not violations,
        "row_count": len(relevant),
        "violation_count": len(violations),
        "violation_examples": sorted(set(violations))[:20],
    }


_SYSTEM_IDENTITY_FIELDS = (
    "model_hash",
    "adapter_hash",
    "scorer_hash",
    "prompt_hash",
    "system_resolution",
)


def _common_system_identity(
    rows: Sequence[Mapping[str, Any]],
    *,
    where: str,
) -> dict[str, Any]:
    """Require one exact model/adapter/scorer/prompt identity across rows."""

    identities: dict[str, dict[str, Any]] = {}
    relevant = [
        row
        for row in rows
        if row.get("record_type") != "shard_metadata"
        and row.get("raw_sample_id") not in (None, "")
    ]
    for row in relevant:
        identity = {field: row.get(field) for field in _SYSTEM_IDENTITY_FIELDS}
        if (
            any(identity[field] in (None, "") for field in _SYSTEM_IDENTITY_FIELDS)
            or not isinstance(identity["system_resolution"], Mapping)
        ):
            raise ContractError(f"{where} row is missing resolved system identity")
        key = content_hash(identity, domain="linkradius:system_identity:v1")
        identities[key] = identity
    if not relevant or len(identities) != 1:
        raise ContractError(
            f"{where} must contain exactly one resolved system identity; "
            f"rows={len(relevant)}, identities={len(identities)}"
        )
    identity_hash, identity = next(iter(identities.items()))
    return {**identity, "content_hash": identity_hash}


def _validate_pgd_target_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    where: str,
) -> dict[str, Any]:
    """Authenticate every PGD edge summary against all three target runs."""

    def coordinate(row: Mapping[str, Any]) -> tuple[str, str, float, int, int]:
        return (
            str(row.get("raw_sample_id") or ""),
            str(row.get("edge_id") or ""),
            float(row.get("requested_epsilon")),
            int(row.get("attack_seed")),
            int(row.get("attack_restart")),
        )

    summaries: dict[tuple[str, str, float, int, int], Mapping[str, Any]] = {}
    targets: dict[
        tuple[str, str, float, int, int], dict[str, Mapping[str, Any]]
    ] = {}
    for row in rows:
        if row.get("attack_family") != "pgd_autograd":
            if row.get("record_type") == "attack_target":
                raise ContractError(f"{where} contains a non-PGD attack_target row")
            continue
        kind = row.get("record_type")
        if kind not in {"sample", "attack_target"}:
            continue
        try:
            key = coordinate(row)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{where} PGD row has invalid coordinates") from exc
        if not key[0] or not key[1]:
            raise ContractError(f"{where} PGD row has empty coordinates")
        if kind == "sample":
            if key in summaries:
                raise ContractError(f"{where} has a duplicate PGD summary: {key}")
            summaries[key] = row
            continue
        label = str(row.get("target_label") or "")
        by_label = targets.setdefault(key, {})
        if not label or label in by_label:
            raise ContractError(
                f"{where} has an empty/duplicate PGD target label at {key}"
            )
        by_label[label] = row

    if not summaries:
        raise ContractError(f"{where} contains no PGD edge summaries")
    if set(targets) != set(summaries):
        raise ContractError(
            f"{where} PGD target coordinates differ from summary coordinates"
        )
    for key, summary in summaries.items():
        gold = str(summary.get("gold") or "")
        summary_margins = summary.get("margins")
        summary_scores = summary.get("option_scores")
        if gold not in {"A", "B", "C", "D"} or not isinstance(
            summary_margins, Mapping
        ) or not isinstance(summary_scores, Mapping):
            raise ContractError(f"{where} PGD summary has invalid gold/margins: {key}")
        expected_labels = {label for label in ("A", "B", "C", "D") if label != gold}
        try:
            parsed_summary_scores = {
                str(name): float(value)
                for name, value in dict(summary_scores).items()
            }
            parsed_summary_margins = {
                str(name): float(value)
                for name, value in dict(summary_margins).items()
            }
            summary_realized_epsilon = float(summary.get("realized_epsilon"))
        except (TypeError, ValueError) as exc:
            raise ContractError(
                f"{where} malformed PGD summary numerics at {key}"
            ) from exc
        if (
            set(parsed_summary_scores) != {"A", "B", "C", "D"}
            or set(parsed_summary_margins) != expected_labels
            or any(
                not math.isfinite(value)
                for value in (
                    *parsed_summary_scores.values(),
                    *parsed_summary_margins.values(),
                    summary_realized_epsilon,
                )
            )
            or any(
                not math.isclose(
                    parsed_summary_margins[label],
                    parsed_summary_scores[gold] - parsed_summary_scores[label],
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for label in expected_labels
            )
        ):
            raise ContractError(
                f"{where} PGD summary scores/margins are inconsistent at {key}"
            )
        by_label = targets[key]
        if set(by_label) != expected_labels or int(
            summary.get("pgd_target_count", -1)
        ) != len(expected_labels):
            raise ContractError(
                f"{where} PGD target set is incomplete at {key}: "
                f"expected={sorted(expected_labels)}, observed={sorted(by_label)}"
            )
        target_minima: dict[str, float] = {}
        for label in sorted(expected_labels):
            target = by_label[label]
            margins = target.get("margins")
            scores = target.get("option_scores")
            diagnostics = target.get("realized_intervention")
            clean_margins = target.get("clean_margins")
            try:
                parsed_margins = {
                    str(name): float(value)
                    for name, value in dict(margins or {}).items()
                }
                parsed_scores = {
                    str(name): float(value)
                    for name, value in dict(scores or {}).items()
                }
                declared_target_margin = float(target.get("minimum_margin"))
                realized_epsilon = float(target.get("realized_epsilon"))
                requested_relative = float(
                    dict(diagnostics or {}).get("requested_relative_norm")
                )
                realized_relative = float(
                    dict(diagnostics or {}).get("realized_relative_norm")
                )
                requested_delta = float(
                    dict(diagnostics or {}).get("requested_delta_norm")
                )
                realized_delta = float(
                    dict(diagnostics or {}).get("realized_delta_norm")
                )
                absolute_budget = float(
                    dict(diagnostics or {}).get("absolute_budget")
                )
                initial_margin = float(
                    dict(diagnostics or {}).get("initial_margin")
                )
                final_margin = float(
                    dict(diagnostics or {}).get("final_margin")
                )
                parsed_clean_margins = {
                    str(name): float(value)
                    for name, value in dict(clean_margins or {}).items()
                }
            except (TypeError, ValueError) as exc:
                raise ContractError(
                    f"{where} malformed PGD target numerics at {key}/{label}"
                ) from exc
            tolerance = 1e-6 * max(1.0, float(key[2]))
            if (
                str(target.get("gold") or "") != gold
                or not isinstance(margins, Mapping)
                or not isinstance(scores, Mapping)
                or not isinstance(diagnostics, Mapping)
                or not isinstance(clean_margins, Mapping)
                or set(parsed_margins) != expected_labels
                or set(parsed_scores) != {"A", "B", "C", "D"}
                or set(parsed_clean_margins) != expected_labels
                or any(
                    not math.isfinite(value)
                    for value in (
                        *parsed_margins.values(),
                        *parsed_scores.values(),
                        declared_target_margin,
                        realized_epsilon,
                        requested_relative,
                        realized_relative,
                        requested_delta,
                        realized_delta,
                        absolute_budget,
                        initial_margin,
                        final_margin,
                        *parsed_clean_margins.values(),
                    )
                )
                or any(
                    not math.isclose(
                        parsed_margins[competitor],
                        parsed_scores[gold] - parsed_scores[competitor],
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    for competitor in expected_labels
                )
                or str(target.get("competitor") or "") != label
                or str(target.get("binding_competitor") or "") != label
                or declared_target_margin != parsed_margins[label]
                or not math.isclose(
                    initial_margin,
                    parsed_clean_margins[label],
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    final_margin,
                    parsed_margins[label],
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or str(diagnostics.get("target_label") or "") != label
                or diagnostics.get("budget_respected") is not True
                or requested_delta <= 0.0
                or realized_delta <= 0.0
                or absolute_budget <= 0.0
                or requested_delta
                > absolute_budget + 1e-6 * max(1.0, absolute_budget)
                or realized_delta
                > absolute_budget + 1e-6 * max(1.0, absolute_budget)
                or requested_relative <= 0.0
                or requested_relative > float(key[2]) + tolerance
                or realized_epsilon <= 0.0
                or realized_epsilon > float(key[2]) + tolerance
                or not math.isclose(
                    realized_epsilon,
                    realized_relative,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    realized_delta / requested_delta,
                    realized_relative / requested_relative,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    absolute_budget / (requested_delta / requested_relative),
                    float(key[2]),
                    rel_tol=1e-6,
                    abs_tol=tolerance,
                )
            ):
                raise ContractError(
                    f"{where} malformed PGD target evidence at {key}/{label}"
                )
            target_minima[label] = min(parsed_margins.values())
        strongest_label = min(sorted(expected_labels), key=target_minima.get)
        strongest = by_label[strongest_label]
        realized = summary.get("realized_intervention")
        try:
            declared_summary_minimum = float(summary.get("minimum_margin"))
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{where} malformed PGD summary numerics at {key}") from exc
        if (
            summary.get("option_scores") != strongest.get("option_scores")
            or summary.get("margins") != strongest.get("margins")
            or declared_summary_minimum != target_minima[strongest_label]
            or not isinstance(realized, Mapping)
            or str(realized.get("target_label") or "") != strongest_label
            or summary_realized_epsilon
            != float(strongest.get("realized_epsilon"))
            or dict(realized) != dict(strongest.get("realized_intervention") or {})
        ):
            raise ContractError(
                f"{where} PGD summary is not the strongest full-margin target at {key}"
            )
    return {
        "name": "pgd_target_evidence",
        "passed": True,
        "summary_rows": len(summaries),
        "target_rows": sum(len(value) for value in targets.values()),
    }


def _clean_execution_coverage(
    clean_rows: Sequence[Mapping[str, Any]],
    execution_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one fresh clean row, its order, and eligibility to every frozen row."""

    expected_ids = [str(value) for value in execution_manifest["ordered_raw_sample_ids"]]
    expected_eligibility = [bool(value) for value in execution_manifest["analysis_eligible"]]
    boundaries = execution_manifest["batch_boundaries"]
    observed_by_id: dict[str, Mapping[str, Any]] = {}
    observed_by_batch: dict[int, list[str]] = {}
    violations: list[str] = []
    for row in clean_rows:
        raw_id = str(row.get("raw_sample_id") or "")
        if not raw_id or raw_id in observed_by_id:
            violations.append(f"duplicate_or_empty:{raw_id}")
            continue
        observed_by_id[raw_id] = row
        row_task = row.get("task")
        if not isinstance(row_task, Mapping):
            violations.append(f"{raw_id}:missing_task")
            continue
        try:
            batch_id = int(row_task["execution_batch_id"])
        except (KeyError, TypeError, ValueError):
            violations.append(f"{raw_id}:invalid_batch")
            continue
        observed_by_batch.setdefault(batch_id, []).append(raw_id)
    if set(observed_by_id) != set(expected_ids) or len(clean_rows) != len(expected_ids):
        violations.append("raw_id_cardinality")
    for index, raw_id in enumerate(expected_ids):
        row = observed_by_id.get(raw_id)
        if row is None:
            continue
        if bool(row.get("analysis_eligible", False)) != expected_eligibility[index]:
            violations.append(f"{raw_id}:eligibility")
        if row.get("ordered_cohort_hash") != execution_manifest.get("ordered_cohort_hash"):
            violations.append(f"{raw_id}:ordered_cohort_hash")
        if row.get("batch_boundary_hash") != execution_manifest.get("batch_boundary_hash"):
            violations.append(f"{raw_id}:batch_boundary_hash")
    for boundary in boundaries:
        batch_id = int(boundary["execution_batch_id"])
        expected_batch = expected_ids[int(boundary["start"]):int(boundary["stop"])]
        if observed_by_batch.get(batch_id, []) != expected_batch:
            violations.append(f"batch_order:{batch_id}")
    return {
        "name": "fresh_clean_execution_coverage",
        "passed": bool(expected_ids) and not violations,
        "observed_rows": len(clean_rows),
        "expected_rows": len(expected_ids),
        "eligible_rows": sum(expected_eligibility),
        "violation_count": len(violations),
        "violation_examples": sorted(set(violations))[:20],
    }


def _all_clean_scorer_agreement(
    clean_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure agreement before the dual-correct analysis mask is applied."""

    labels = {"A", "B", "C", "D"}
    comparable = [
        row
        for row in clean_rows
        if row.get("strict_generated_choice") in labels
        and bool(row.get("strict_generated_valid", False))
        and row.get("scorer_prediction") in labels
        and not bool(row.get("score_tie", False))
    ]
    agreements = sum(
        row["strict_generated_choice"] == row["scorer_prediction"]
        for row in comparable
    )
    return {
        "comparable_rows": len(comparable),
        "total_clean_rows": len(clean_rows),
        "agreement": agreements / len(comparable) if comparable else 0.0,
        "agreement_all_rows": agreements / len(clean_rows) if clean_rows else 0.0,
        "comparable_coverage": len(comparable) / len(clean_rows) if clean_rows else 0.0,
        "disagreement_rows": len(comparable) - agreements,
        "invalid_generation_rows": sum(
            not bool(row.get("strict_generated_valid", False)) for row in clean_rows
        ),
        "scorer_tie_or_invalid_rows": sum(
            bool(row.get("score_tie", False))
            or row.get("scorer_prediction") not in labels
            for row in clean_rows
        ),
        "analysis_ineligible_rows": sum(
            not bool(row.get("analysis_eligible", False)) for row in clean_rows
        ),
    }


def _identity_replay_check(
    clean_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    clean_by_id: dict[str, Mapping[str, Any]] = {}
    for row in clean_rows:
        raw_id = str(row.get("raw_sample_id") or "")
        if not raw_id or raw_id in clean_by_id:
            raise ContractError("clean validation rows must have unique raw sample IDs")
        clean_by_id[raw_id] = row
    identities = [row for row in replay_rows if row.get("intervention_mode") == "identity"]
    mismatches: list[str] = []
    maximum_error = 0.0
    for row in identities:
        raw_id = str(row.get("raw_sample_id") or "")
        clean = clean_by_id.get(raw_id)
        if clean is None:
            mismatches.append(f"{raw_id}:missing_clean")
            continue
        clean_scores, replay_scores = clean.get("option_scores"), row.get("option_scores")
        if not isinstance(clean_scores, Mapping) or not isinstance(replay_scores, Mapping):
            mismatches.append(f"{raw_id}:missing_scores")
            continue
        if set(clean_scores) != set(replay_scores):
            mismatches.append(f"{raw_id}:score_labels")
            continue
        for label in clean_scores:
            error = abs(float(clean_scores[label]) - float(replay_scores[label]))
            maximum_error = max(maximum_error, error)
            if error > tolerance:
                mismatches.append(f"{raw_id}:{row.get('edge_id')}:{label}")
        if row.get("scorer_prediction") != clean.get("scorer_prediction"):
            mismatches.append(f"{raw_id}:{row.get('edge_id')}:prediction")
    return {
        "name": "clean_identity_replay",
        "passed": bool(identities) and not mismatches,
        "identity_rows": len(identities),
        "maximum_score_error": maximum_error,
        "tolerance": float(tolerance),
        "mismatch_count": len(mismatches),
        "mismatch_examples": sorted(set(mismatches))[:20],
    }


def _record_global_pointer_completion(
    args: argparse.Namespace,
    task_dir: Path,
    task: Mapping[str, Any],
    repo_root: Path,
    *,
    pointer_name: str,
    pointer: Mapping[str, Any],
) -> None:
    atomic_write_json(task_dir / pointer_name, dict(pointer), overwrite=args.overwrite)
    authenticated_hashes = {
        key: value
        for key, value in pointer.items()
        if key in {"gate_content_hash", "content_hash"}
    }
    publish_completion(
        task_dir,
        config_hash=str(task["config_key"]),
        source_hash_value=_cached_source_hash(args, repo_root),
        artifact_paths=["manifest.json", "command.txt", pointer_name],
        extra={"array_index": int(task["array_index"]), **authenticated_hashes},
        overwrite=args.overwrite,
    )


def _engineering_validate_stage(
    args: argparse.Namespace, task_dir: Path, task: Mapping[str, Any], repo_root: Path
) -> None:
    from .validate_engineering import assemble_engineering_evidence, validate_engineering_evidence

    artifact_root = (
        _phase_root(args)
        / str(task["dataset"])
        / f"R{int(task['R'])}"
    )
    evidence = assemble_engineering_evidence(
        artifact_root,
        legacy_equivalence_path=args.legacy_equivalence or None,
        expected_latent_steps=args.latent_length,
    )
    report, checks = validate_engineering_evidence(evidence)
    engineering_grid_checks = [
        _verify_source_stage_grid(
            args,
            task,
            repo_root,
            stage=stage,
            partition=partition,
        )
        for stage, partition in (
            ("split", "global"),
            ("discover", "validation"),
            ("freeze_execution", "validation"),
            ("clean", "validation"),
            ("replay", "validation"),
            ("probe", "validation"),
            ("gradient", "validation"),
        )
    ]
    checks.extend(
        {
            "name": f"exact_grid:{item['stage']}",
            **{key: value for key, value in item.items() if key != "stage"},
        }
        for item in engineering_grid_checks
    )
    _, split_hash = _authenticated_split_manifest(args, task, repo_root)
    _, execution_manifest, execution_hash = _authenticated_execution_manifest(
        args, "validation", task, repo_root
    )
    report.pop("report_content_hash", None)
    report["execution_provenance"] = {
        "split_manifest_hash": split_hash,
        "execution_manifest_hash": execution_hash,
        "ordered_cohort_hash": execution_manifest["ordered_cohort_hash"],
        "batch_boundary_hash": execution_manifest["batch_boundary_hash"],
    }
    report["report_content_hash"] = content_hash(
        report, domain="linkradius:engineering_report:v1"
    )
    report_path = Path(args.out_root) / "engineering_report.json"
    gate_path = Path(args.engineering_gate)
    gate = make_gate(
        gate_type="engineering_gate",
        checks=checks,
        config_hash=str(task["config_key"]),
        source_hash=_cached_source_hash(args, repo_root),
        prerequisite_hashes={
            "engineering_report_hash": report["report_content_hash"],
            "split_manifest_hash": split_hash,
            "execution_manifest_hash": execution_hash,
            "ordered_cohort_hash": execution_manifest["ordered_cohort_hash"],
            "batch_boundary_hash": execution_manifest["batch_boundary_hash"],
        },
    )
    atomic_write_json(report_path, report, overwrite=args.overwrite)
    atomic_write_json(gate_path, gate, overwrite=args.overwrite)
    if not gate["passed"]:
        raise ContractError("engineering gate failed; inspect engineering_report.json")
    clean_row_count = _merge_public_stage_jsonl(
        args,
        task,
        stage="clean",
        partition="validation",
        filename="clean_baseline.jsonl",
        output=task_dir / "clean_baseline.jsonl",
    )
    atomic_write_json(
        task_dir / "engineering_validation_result.json",
        {
            "passed": gate["passed"],
            "engineering_report": str(report_path.resolve()),
            "engineering_gate": str(gate_path.resolve()),
            "gate_content_hash": gate["gate_content_hash"],
        },
        overwrite=args.overwrite,
    )
    publish_completion(
        task_dir,
        config_hash=str(task["config_key"]),
        source_hash_value=_cached_source_hash(args, repo_root),
        artifact_paths=[
            "manifest.json",
            "command.txt",
            "engineering_validation_result.json",
            "clean_baseline.jsonl",
        ],
        row_counts={"clean_baseline.jsonl": clean_row_count},
        extra={
            "array_index": int(task["array_index"]),
            "gate_content_hash": gate["gate_content_hash"],
        },
        overwrite=args.overwrite,
    )


def _smoke_validate_stage(
    args: argparse.Namespace, task_dir: Path, task: Mapping[str, Any], repo_root: Path
) -> None:
    root = _phase_root(args) / str(task["dataset"]) / f"R{int(task['R'])}"
    grid_checks = [
        _verify_source_stage_grid(
            args,
            task,
            repo_root,
            stage=stage,
            partition=partition,
        )
        for stage, partition in (
            ("clean", "validation"),
            ("causal", "validation"),
            ("probe", "validation"),
            ("gradient", "validation"),
            ("attack", "validation"),
            ("estimate", "global"),
            ("aggregate", "global"),
        )
    ]
    validation_root = root / "validation"
    expected_source = _cached_source_hash(args, repo_root)
    clean = [row for row in _completed_rows(validation_root / "clean", "clean_baseline.jsonl", expected_source_hash=expected_source) if row.get("record_type") == "sample"]
    causal = [row for row in _completed_rows(validation_root / "causal", "causal_runs.jsonl", expected_source_hash=expected_source) if row.get("record_type") == "sample"]
    probes = _completed_rows(validation_root / "probe", "probe_runs.jsonl", expected_source_hash=expected_source)
    gradients = [row for row in _completed_rows(validation_root / "gradient", "gradient_runs.jsonl", expected_source_hash=expected_source) if row.get("record_type") == "gradient"]
    all_attacks = _completed_rows(validation_root / "attack", "attack_results.jsonl", expected_source_hash=expected_source)
    attacks = [row for row in all_attacks if row.get("record_type") == "sample"]
    scorer_agreement = _all_clean_scorer_agreement(clean)
    early = {"p2c@0", "c2s@0", "s2p@0"}
    split_manifest = load_json(args.split_manifest)
    split_hash = verify_split_manifest(split_manifest)
    _, execution_manifest, execution_hash = _authenticated_execution_manifest(
        args, "validation", task, repo_root
    )
    validation_ids = {
        str(row["raw_sample_id"] if isinstance(row, Mapping) else row)
        for row in split_manifest["partitions"]["validation"]
    }
    current_source_hash = expected_source
    provenance_checks = [
        {
            **_provenance_check(
                rows,
                partition="validation",
                allowed_raw_ids=validation_ids,
                split_hash=split_hash,
                execution_hash=execution_hash,
                current_source_hash=current_source_hash,
                ordered_cohort_hash=execution_manifest["ordered_cohort_hash"],
                batch_boundary_hash=execution_manifest["batch_boundary_hash"],
            ),
            "name": f"{name}_provenance",
        }
        for name, rows in (
            ("clean", clean),
            ("causal", causal),
            ("probe", probes),
            ("attack", attacks),
        )
    ]
    terminal_gradients = [row for row in gradients if row.get("edge_id") == "c2s@1"]
    cast_surviving_gradients = []
    for row in terminal_gradients:
        finite_difference = row.get("finite_difference")
        if not isinstance(finite_difference, Mapping):
            continue
        plus = finite_difference.get("plus_diagnostics")
        minus = finite_difference.get("minus_diagnostics")
        if (
            row.get("autograd_semantics") == "continuous_consumer_input"
            and finite_difference.get("agrees") is True
            and isinstance(plus, Mapping)
            and isinstance(minus, Mapping)
            and plus.get("collapsed") is False
            and minus.get("collapsed") is False
            and float(finite_difference.get("realized_separation", 0.0)) > 0.0
        ):
            cast_surviving_gradients.append(row)
    pgd_rows = [
        row
        for row in attacks
        if row.get("attack_family") == "pgd_autograd"
        and row.get("edge_id") == "c2s@1"
        and not row.get("failure")
    ]
    valid_pgd = []
    for row in pgd_rows:
        realized = row.get("realized_intervention")
        try:
            objective_improved = float(realized["final_margin"]) < float(
                realized["initial_margin"]
            )
        except (KeyError, TypeError, ValueError):
            objective_improved = False
        if (
            row.get("autograd_semantics") == "relaxed_autograd"
            and isinstance(realized, Mapping)
            and realized.get("budget_respected") is True
            and realized.get("improved") is True
            and objective_improved
        ):
            valid_pgd.append(row)
    unsupported_pgd = [
        row
        for row in all_attacks
        if row.get("record_type") == "unsupported"
        and (
            row.get("attack_family") == "pgd_autograd"
            or row.get("intervention_mode") == "pgd_autograd"
        )
    ]
    random_rows = [row for row in attacks if row.get("attack_family") == "random_independent"]
    from .probe_validation import reclassify_probe_pairs

    smoke_pairs = reclassify_probe_pairs(
        probes,
        {
            "minimum_requested_realized_cosine": 0.0,
            "maximum_off_direction_relative": 1.0,
            "minimum_signed_separation": 0.0,
            "minimum_antipodality": -1.0,
            "version": "linkradius_probe_thresholds_v1",
        },
    )
    smoke_probe_acceptance = sum(bool(row["accepted"]) for row in smoke_pairs) / len(smoke_pairs)
    eligible_raw_ids = tuple(
        str(raw_id)
        for raw_id, eligible in zip(
            execution_manifest["ordered_raw_sample_ids"],
            execution_manifest["analysis_eligible"],
        )
        if bool(eligible)
    )
    expected_probe_configurations = {
        (raw_id, edge, float(h), int(seed), direction)
        for raw_id in eligible_raw_ids
        for edge in early
        for h in args.probe_radii.split()
        for seed in args.probe_seeds.split()
        for direction in range(int(args.K))
    }
    actual_probe_configurations = {
        (
            str(row["raw_sample_id"]),
            str(row["edge_id"]),
            float(row["h"]),
            int(row["probe_seed"]),
            int(row["direction_id"]),
        )
        for row in smoke_pairs
    }
    from .aggregate_causal_use import eligible_complete_causal_rows

    complete_causal = eligible_complete_causal_rows(
        causal,
        expected_edges=tuple(sorted(early)),
        expected_modes=tuple(args.interventions.split()),
        expected_raw_ids=eligible_raw_ids,
    )

    expected_random_units = {
        (raw_id, edge, float(epsilon))
        for raw_id in eligible_raw_ids
        for edge in early
        for epsilon in args.attack_epsilons.split()
    }
    actual_random_units = {
        (
            str(row.get("raw_sample_id") or ""),
            str(row.get("edge_id") or ""),
            float(row.get("requested_epsilon")),
        )
        for row in random_rows
        if bool(row.get("analysis_eligible", False))
    }
    random_unit_count = sum(
        bool(row.get("analysis_eligible", False)) for row in random_rows
    )

    eligible_batch_representatives: dict[int, str] = {}
    for boundary in execution_manifest["batch_boundaries"]:
        start, stop = int(boundary["start"]), int(boundary["stop"])
        for offset in range(start, stop):
            if bool(execution_manifest["analysis_eligible"][offset]):
                eligible_batch_representatives[int(boundary["execution_batch_id"])] = str(
                    execution_manifest["ordered_raw_sample_ids"][offset]
                )
                break
    expected_gradient_units = {
        (raw_id, "c2s@1") for raw_id in eligible_batch_representatives.values()
    }
    actual_gradient_units = {
        (str(row.get("raw_sample_id") or ""), str(row.get("edge_id") or ""))
        for row in gradients
    }
    expected_pgd_units = {
        (raw_id, "c2s@1", float(epsilon))
        for raw_id in eligible_batch_representatives.values()
        for epsilon in args.attack_epsilons.split()
    }
    actual_pgd_units = {
        (
            str(row.get("raw_sample_id") or ""),
            str(row.get("edge_id") or ""),
            float(row.get("requested_epsilon")),
        )
        for row in pgd_rows
    }
    import csv

    _, estimate_artifacts = _authenticated_canonical_task_artifacts(
        args,
        task,
        repo_root,
        stage="estimate",
        filenames=("linkradius_edges.csv", "linkradius_competitors.csv"),
    )
    with estimate_artifacts["linkradius_edges.csv"].open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        estimate_rows = list(csv.DictReader(handle))
    expected_primary_units = {
        (str(raw_id), edge)
        for raw_id, eligible in zip(
            execution_manifest["ordered_raw_sample_ids"],
            execution_manifest["analysis_eligible"],
        )
        if bool(eligible)
        for edge in early
    }
    primary_units = {
        (str(row.get("raw_sample_id")), str(row.get("edge_id")))
        for row in estimate_rows
        if str(row.get("primary_available", "")).lower() in {"1", "true", "yes"}
        and int(row.get("requested_K", -1)) == int(args.K)
    }
    expected_causal_modes = set(args.interventions.split())
    clean_coverage = _clean_execution_coverage(clean, execution_manifest)
    checks = [
        clean_coverage,
        {"name": "dual_correct_cohort_size", "passed": 10 <= sum(bool(row.get("analysis_eligible")) for row in clean) <= 20},
        {"name": "early_causal_grid", "passed": bool(complete_causal) and {row.get("edge_id") for row in complete_causal} == early and {row.get("intervention_mode") for row in complete_causal} == expected_causal_modes},
        {"name": "early_probe_grid", "passed": {row.get("edge_id") for row in probes if row.get("record_type") == "sample"} == early},
        {"name": "eligible_probe_configuration_coverage", "passed": actual_probe_configurations == expected_probe_configurations and len(smoke_pairs) == len(expected_probe_configurations), "observed": len(smoke_pairs), "expected": len(expected_probe_configurations)},
        {"name": "probe_pair_cast_survival", "passed": bool(smoke_pairs) and smoke_probe_acceptance >= args.minimum_probe_acceptance, "acceptance": smoke_probe_acceptance, "n": len(smoke_pairs)},
        {"name": "primary_linkradius_coverage", "passed": primary_units == expected_primary_units, "observed_units": len(primary_units), "expected_units": len(expected_primary_units)},
        {"name": "terminal_gradient_grid", "passed": actual_gradient_units == expected_gradient_units and len(gradients) == len(expected_gradient_units), "observed": len(gradients), "expected": len(expected_gradient_units)},
        {"name": "terminal_relaxed_autograd_cast_survival", "passed": bool(cast_surviving_gradients) and len(cast_surviving_gradients) == len(terminal_gradients), "successful": len(cast_surviving_gradients), "total": len(terminal_gradients)},
        {"name": "independent_random_attack_grid", "passed": actual_random_units == expected_random_units and random_unit_count == len(expected_random_units), "observed": random_unit_count, "expected": len(expected_random_units)},
        {"name": "autograd_pgd_success", "passed": actual_pgd_units == expected_pgd_units and len(pgd_rows) == len(expected_pgd_units) and not unsupported_pgd and len(valid_pgd) == len(pgd_rows), "successful": len(valid_pgd), "total": len(pgd_rows), "expected": len(expected_pgd_units), "unsupported": len(unsupported_pgd)},
        {"name": "scorer_generated_agreement", "passed": scorer_agreement["total_clean_rows"] > 0 and scorer_agreement["agreement_all_rows"] >= args.minimum_scorer_agreement, **scorer_agreement, "minimum_all_row_agreement": args.minimum_scorer_agreement},
        *[
            {"name": f"exact_grid:{report['stage']}", **{key: value for key, value in report.items() if key not in {"stage"}}}
            for report in grid_checks
        ],
        *provenance_checks,
    ]
    gate = make_gate(
        gate_type="smoke_gate",
        checks=checks,
        config_hash=str(task["config_key"]),
        source_hash=current_source_hash,
        prerequisite_hashes={
            "engineering_gate_hash": require_passed_gate(args.engineering_gate, gate_type="engineering_gate")["gate_content_hash"],
            "split_manifest_hash": split_hash,
            "execution_manifest_hash": execution_hash,
            "ordered_cohort_hash": execution_manifest["ordered_cohort_hash"],
            "batch_boundary_hash": execution_manifest["batch_boundary_hash"],
        },
    )
    gate_path = Path(args.smoke_gate)
    atomic_write_json(gate_path, gate, overwrite=args.overwrite)
    if not gate["passed"]:
        raise ContractError("smoke gate failed; required completed smoke artifacts are missing or incompatible")
    _record_global_pointer_completion(
        args,
        task_dir,
        task,
        repo_root,
        pointer_name="smoke_validation_result.json",
        pointer={"passed": gate["passed"], "smoke_gate": str(gate_path.resolve()), "gate_content_hash": gate["gate_content_hash"]},
    )


def _freeze_probe_stage(
    args: argparse.Namespace, task_dir: Path, task: Mapping[str, Any], repo_root: Path
) -> None:
    from .probe_validation import (
        calibrate_probe_configuration,
        probe_autograd_agreement,
        reclassify_probe_pairs,
        select_causally_useful_edges,
    )
    from .aggregate_causal_use import eligible_complete_causal_rows

    if not -1.0 <= args.minimum_rank_stability <= 1.0:
        raise ContractError("--minimum-rank-stability must lie in [-1,1]")
    if not 0.0 <= args.minimum_binding_stability <= 1.0:
        raise ContractError("--minimum-binding-stability must lie in [0,1]")
    if args.minimum_stability_comparisons < 1:
        raise ContractError("--minimum-stability-comparisons must be positive")
    if not 0.0 <= args.minimum_scorer_agreement <= 1.0:
        raise ContractError("--minimum-scorer-agreement must lie in [0,1]")
    if not 0.0 <= args.minimum_autograd_agreement <= 1.0:
        raise ContractError("--minimum-autograd-agreement must lie in [0,1]")
    if not 0.0 <= args.maximum_probe_autograd_relative_error <= 1.0:
        raise ContractError("--maximum-probe-autograd-relative-error must lie in [0,1]")
    if args.identity_replay_tolerance < 0.0:
        raise ContractError("--identity-replay-tolerance must be non-negative")
    root = _phase_root(args) / str(task["dataset"]) / f"R{int(task['R'])}"
    if (root / "test").exists() and any((root / "test").rglob("*.jsonl")):
        raise ContractError("freeze_probe refuses to read or coexist with Phase-3 test outcome artifacts")
    probe_grid = _verify_source_stage_grid(
        args,
        task,
        repo_root,
        stage="probe_calibration",
        partition="validation",
    )
    causal_grid = _verify_source_stage_grid(
        args,
        task,
        repo_root,
        stage="causal",
        partition="validation",
    )
    gradient_grid = _verify_source_stage_grid(
        args,
        task,
        repo_root,
        stage="gradient",
        partition="validation",
    )
    expected_source = _cached_source_hash(args, repo_root)
    probe_rows = _completed_rows(root / "validation" / "probe_calibration", "probe_runs.jsonl", expected_source_hash=expected_source)
    causal_rows = [
        row
        for row in _completed_rows(root / "validation" / "causal", "causal_runs.jsonl", expected_source_hash=expected_source)
        if row.get("record_type") == "sample"
    ]
    gradient_rows = [
        row
        for row in _completed_rows(
            root / "validation" / "gradient", "gradient_runs.jsonl", expected_source_hash=expected_source
        )
        if row.get("record_type") == "gradient"
    ]
    split_manifest, split_hash = _authenticated_split_manifest(args, task, repo_root)
    _, execution_manifest, execution_hash = _authenticated_execution_manifest(
        args, "validation", task, repo_root
    )
    eligible_raw_ids = tuple(
        str(raw_id)
        for raw_id, eligible in zip(
            execution_manifest["ordered_raw_sample_ids"],
            execution_manifest["analysis_eligible"],
        )
        if bool(eligible)
    )
    expected_edges = tuple(
        f"{site}@{round_idx}"
        for site, round_idx in canonical_edge_pairs(int(task["R"]))
    )
    expected_probe_configurations = tuple(
        (edge, float(h), int(seed), direction)
        for edge in expected_edges
        for h in args.probe_radii.split()
        for seed in args.probe_seeds.split()
        for direction in range(int(args.K))
    )
    calibration, _ = calibrate_probe_configuration(
        probe_rows,
        candidate_K=tuple(value for value in (4, 8, 16, 32) if value <= int(args.K)),
        minimum_acceptance=args.minimum_probe_acceptance,
        lower_quantile=args.probe_threshold_lower_quantile,
        upper_quantile=args.probe_threshold_upper_quantile,
        expected_raw_ids=eligible_raw_ids,
        expected_configurations=expected_probe_configurations,
    )
    validation_ids = {
        str(row["raw_sample_id"] if isinstance(row, Mapping) else row)
        for row in split_manifest["partitions"]["validation"]
    }
    provenance = _provenance_check(
        probe_rows,
        partition="validation",
        allowed_raw_ids=validation_ids,
        split_hash=split_hash,
        execution_hash=execution_hash,
        current_source_hash=_cached_source_hash(args, repo_root),
        ordered_cohort_hash=execution_manifest["ordered_cohort_hash"],
        batch_boundary_hash=execution_manifest["batch_boundary_hash"],
    )
    if not provenance["passed"]:
        raise ContractError(f"validation probe provenance failed: {provenance}")
    causal_provenance = _provenance_check(
        causal_rows,
        partition="validation",
        allowed_raw_ids=validation_ids,
        split_hash=split_hash,
        execution_hash=execution_hash,
        current_source_hash=_cached_source_hash(args, repo_root),
        ordered_cohort_hash=execution_manifest["ordered_cohort_hash"],
        batch_boundary_hash=execution_manifest["batch_boundary_hash"],
    )
    if not causal_provenance["passed"]:
        raise ContractError(f"validation causal provenance failed: {causal_provenance}")
    complete_causal_rows = eligible_complete_causal_rows(
        causal_rows,
        expected_edges=expected_edges,
        expected_modes=tuple(args.interventions.split()),
        expected_raw_ids=eligible_raw_ids,
    )
    causal_rule = select_causally_useful_edges(
        complete_causal_rows,
        expected_edges=expected_edges,
        minimum_pairs=args.minimum_causal_pairs,
        minimum_accuracy_effect=args.minimum_causal_accuracy_effect,
        minimum_margin_effect=args.minimum_causal_margin_effect,
    )
    autograd_agreement = probe_autograd_agreement(
        reclassify_probe_pairs(probe_rows, calibration["acceptance_thresholds"]),
        gradient_rows,
        selected_h=float(calibration["selected_h"]),
        selected_K=int(calibration["selected_K"]),
    )
    system_identity = _common_system_identity(
        [*probe_rows, *causal_rows, *gradient_rows],
        where="Phase-3 validation evidence",
    )
    config = {
        "schema_version": "linkradius.frozen_probe_config.v1",
        "dataset": task["dataset"],
        "R": int(task["R"]),
        "style": str(args.style),
        "method": str(args.method),
        "seed": int(task["seed"]),
        "batch_size": int(args.batch_size),
        "latent_length": int(args.latent_length),
        "runtime": {
            "role_devices": _resolved_role_devices(args),
            "relay_transfer_mode": str(args.relay_transfer_mode),
            "autograd_memory_mode": str(args.autograd_memory_mode),
            "trust_remote_code": int(args.trust_remote_code),
            "round_label_mode": str(args.round_label_mode),
            "environment": _runtime_environment_identity(),
        },
        "system_identity": system_identity,
        "selected_h": calibration["selected_h"],
        "K": calibration["selected_K"],
        "nested_K": calibration["nested_K"],
        "candidate_K": calibration["candidate_K"],
        "usable_h": calibration["usable_h"],
        "subspace": args.subspace,
        "probe_seeds": calibration["probe_seeds"],
        "grid_probe_radii": [float(value) for value in args.probe_radii.split()],
        "grid_probe_seeds": [int(value) for value in args.probe_seeds.split()],
        "grid_probe_K": int(args.K),
        "grid_edges": list(expected_edges),
        "acceptance_thresholds": calibration["acceptance_thresholds"],
        "probe_calibration": calibration,
        "probe_autograd_agreement": autograd_agreement,
        "causally_useful_edge_rule": causal_rule,
        "causal_positive_control_required": not bool(
            causal_rule.get("useful_edges")
        ),
        "stability_criteria": {
            "minimum_rank_correlation": float(args.minimum_rank_stability),
            "minimum_binding_agreement": float(args.minimum_binding_stability),
            "minimum_comparisons_per_dimension": int(args.minimum_stability_comparisons),
            "maximum_probe_autograd_relative_error": float(args.maximum_probe_autograd_relative_error),
        },
        "gate_criteria": {
            "minimum_probe_acceptance": float(args.minimum_probe_acceptance),
            "minimum_scorer_agreement": float(args.minimum_scorer_agreement),
            "minimum_autograd_agreement": float(args.minimum_autograd_agreement),
            "identity_replay_tolerance": float(args.identity_replay_tolerance),
        },
        "scorer_normalization": "mean",
        "score_tie_tolerance": 0.0,
        "evidence_partitions": ["validation"],
        "test_accessed": False,
        "source_hash": _cached_source_hash(args, repo_root),
        "split_manifest_hash": split_hash,
        "validation_execution_manifest_hash": execution_hash,
        "validation_ordered_cohort_hash": execution_manifest["ordered_cohort_hash"],
        "validation_batch_boundary_hash": execution_manifest["batch_boundary_hash"],
        "validation_probe_evidence_hash": _probe_evidence_hash(probe_rows),
        "validation_causal_evidence_hash": content_hash(
            sorted(causal_rows, key=lambda row: str(row.get("run_id"))),
            domain="linkradius:validation_causal_evidence:v1",
        ),
        "probe_grid_verification": probe_grid,
        "causal_grid_verification": causal_grid,
        "gradient_grid_verification": gradient_grid,
        "validation_probe_provenance": provenance,
        "validation_causal_provenance": causal_provenance,
    }
    config["content_hash"] = content_hash(config, domain="linkradius:frozen_probe_config:v1")
    output = Path(args.frozen_config)
    if output.exists() and not args.overwrite:
        existing = load_json(output)
        if existing.get("content_hash") != config["content_hash"]:
            raise ContractError("refusing to overwrite an incompatible frozen probe configuration")
    else:
        atomic_write_json(output, config, overwrite=True)
    _record_global_pointer_completion(
        args,
        task_dir,
        task,
        repo_root,
        pointer_name="freeze_probe_result.json",
        pointer={"frozen_config": str(output.resolve()), "content_hash": config["content_hash"]},
    )


def _validate_probe_stage(
    args: argparse.Namespace, task_dir: Path, task: Mapping[str, Any], repo_root: Path
) -> None:
    from .probe_validation import (
        calibrate_probe_configuration,
        probe_autograd_agreement,
        select_causally_useful_edges,
        stability_checks,
    )
    from .aggregate_causal_use import eligible_complete_causal_rows

    current_source_hash = _cached_source_hash(args, repo_root)
    frozen = _authenticated_frozen_probe_config(
        args, task, current_source_hash=current_source_hash
    )
    expected_hash = str(frozen["content_hash"])
    root = _phase_root(args) / str(task["dataset"]) / f"R{int(task['R'])}"
    if any((root / "test").rglob("*.jsonl")) if (root / "test").exists() else False:
        raise ContractError("validate_probe refuses Phase-3 test outcomes")
    grid_checks = [
        _verify_source_stage_grid(
            args,
            task,
            repo_root,
            stage=stage,
            partition="validation",
        )
        for stage in ("clean", "causal", "probe_calibration", "gradient")
    ]
    clean = [row for row in _completed_rows(root / "validation" / "clean", "clean_baseline.jsonl", expected_source_hash=current_source_hash) if row.get("record_type") == "sample"]
    replay = [row for row in _completed_rows(root / "validation" / "causal", "causal_runs.jsonl", expected_source_hash=current_source_hash) if row.get("record_type") == "sample"]
    probe_rows = _completed_rows(root / "validation" / "probe_calibration", "probe_runs.jsonl", expected_source_hash=current_source_hash)
    gradient_rows = [
        row
        for row in _completed_rows(root / "validation" / "gradient", "gradient_runs.jsonl", expected_source_hash=current_source_hash)
        if row.get("record_type") == "gradient"
    ]
    split_manifest, split_hash = _authenticated_split_manifest(args, task, repo_root)
    _, execution_manifest, execution_hash = _authenticated_execution_manifest(
        args, "validation", task, repo_root
    )
    eligible_raw_ids = tuple(
        str(raw_id)
        for raw_id, eligible in zip(
            execution_manifest["ordered_raw_sample_ids"],
            execution_manifest["analysis_eligible"],
        )
        if bool(eligible)
    )
    expected_probe_configurations = tuple(
        (str(edge), float(h), int(seed), direction)
        for edge in frozen["grid_edges"]
        for h in frozen["grid_probe_radii"]
        for seed in frozen["grid_probe_seeds"]
        for direction in range(int(frozen["grid_probe_K"]))
    )
    calibration, classified_pairs = calibrate_probe_configuration(
        probe_rows,
        candidate_K=tuple(int(value) for value in frozen["candidate_K"]),
        minimum_acceptance=float(frozen["gate_criteria"]["minimum_probe_acceptance"]),
        lower_quantile=float(frozen["probe_calibration"]["threshold_derivation"]["lower_quantile"]),
        upper_quantile=float(frozen["probe_calibration"]["threshold_derivation"]["upper_quantile"]),
        expected_raw_ids=eligible_raw_ids,
        expected_configurations=expected_probe_configurations,
    )
    pair_rows = [
        row
        for row in classified_pairs
        if float(row.get("h", -1)) == float(frozen["selected_h"])
        and int(row.get("direction_id", -1)) < int(frozen["K"])
    ]
    scorer_agreement = _all_clean_scorer_agreement(clean)
    acceptance = sum(bool(row.get("accepted")) for row in pair_rows) / len(pair_rows) if pair_rows else 0.0
    identity_check = _identity_replay_check(
        clean,
        replay,
        tolerance=float(frozen["gate_criteria"]["identity_replay_tolerance"]),
    )
    validation_ids = {
        str(row["raw_sample_id"] if isinstance(row, Mapping) else row)
        for row in split_manifest["partitions"]["validation"]
    }
    provenance_checks = [
        _provenance_check(
            rows,
            partition="validation",
            allowed_raw_ids=validation_ids,
            split_hash=split_hash,
            execution_hash=execution_hash,
            current_source_hash=current_source_hash,
            ordered_cohort_hash=execution_manifest["ordered_cohort_hash"],
            batch_boundary_hash=execution_manifest["batch_boundary_hash"],
        )
        for rows in (clean, replay, probe_rows)
    ]
    finite_difference_rows = [
        row
        for row in gradient_rows
        if isinstance(row.get("finite_difference"), Mapping)
    ]
    autograd_agreement = (
        sum(bool(row["finite_difference"].get("agrees")) for row in finite_difference_rows)
        / len(gradient_rows)
        if gradient_rows
        else 0.0
    )
    criteria = frozen["stability_criteria"]
    gate_criteria = frozen["gate_criteria"]
    frozen_causal_rule = frozen["causally_useful_edge_rule"]
    complete_causal_rows = eligible_complete_causal_rows(
        replay,
        expected_edges=tuple(str(value) for value in frozen_causal_rule["expected_edges"]),
        expected_modes=tuple(args.interventions.split()),
        expected_raw_ids=eligible_raw_ids,
    )
    causal_rule = select_causally_useful_edges(
        complete_causal_rows,
        expected_edges=tuple(str(value) for value in frozen_causal_rule["expected_edges"]),
        minimum_pairs=int(frozen_causal_rule["minimum_pairs"]),
        minimum_accuracy_effect=float(frozen_causal_rule["minimum_accuracy_effect"]),
        minimum_margin_effect=float(frozen_causal_rule["minimum_margin_effect"]),
    )
    probe_autograd = probe_autograd_agreement(
        classified_pairs,
        gradient_rows,
        selected_h=float(frozen["selected_h"]),
        selected_K=int(frozen["K"]),
    )
    checks = [
        {"name": "frozen_config_hash", "passed": True, "hash": expected_hash},
        {"name": "no_test_access", "passed": True},
        {
            "name": "source_and_manifest_freeze",
            "passed": frozen.get("split_manifest_hash") == split_hash
            and frozen.get("validation_execution_manifest_hash") == execution_hash
            and frozen.get("validation_ordered_cohort_hash")
            == execution_manifest.get("ordered_cohort_hash")
            and frozen.get("validation_batch_boundary_hash")
            == execution_manifest.get("batch_boundary_hash"),
        },
        _clean_execution_coverage(clean, execution_manifest),
        {
            "name": "validation_probe_evidence_identity",
            "passed": frozen.get("validation_probe_evidence_hash")
            == _probe_evidence_hash(probe_rows),
        },
        {
            "name": "validation_calibration_reproduced",
            "passed": calibration.get("content_hash")
            == frozen.get("probe_calibration", {}).get("content_hash")
            and calibration.get("selected_h") == frozen.get("selected_h")
            and calibration.get("selected_K") == frozen.get("K"),
        },
        {
            "name": "causally_useful_edge_rule_reproduced",
            "passed": causal_rule.get("content_hash")
            == frozen_causal_rule.get("content_hash")
            and frozen.get("validation_causal_evidence_hash")
            == content_hash(
                sorted(replay, key=lambda row: str(row.get("run_id"))),
                domain="linkradius:validation_causal_evidence:v1",
            ),
            "useful_edges": causal_rule.get("useful_edges", []),
            "positive_control_required": not bool(
                causal_rule.get("useful_edges")
            ),
        },
        {"name": "scorer_generation_agreement", "passed": scorer_agreement["total_clean_rows"] > 0 and scorer_agreement["agreement_all_rows"] >= float(gate_criteria["minimum_scorer_agreement"]), **scorer_agreement, "minimum_all_row_agreement": float(gate_criteria["minimum_scorer_agreement"])},
        identity_check,
        {"name": "probe_acceptance", "passed": bool(pair_rows) and acceptance >= float(gate_criteria["minimum_probe_acceptance"]), "acceptance": acceptance, "n": len(pair_rows)},
        {
            "name": "autograd_reference_agreement",
            "passed": bool(finite_difference_rows)
            and autograd_agreement >= float(gate_criteria["minimum_autograd_agreement"]),
            "agreement": autograd_agreement,
            "usable": len(finite_difference_rows),
            "total": len(gradient_rows),
        },
        {
            "name": "probe_susceptibility_autograd_agreement",
            "passed": probe_autograd.get("content_hash")
            == frozen.get("probe_autograd_agreement", {}).get("content_hash")
            and int(probe_autograd.get("comparison_count", 0))
            >= int(criteria["minimum_comparisons_per_dimension"])
            and float(probe_autograd.get("usable_finite_difference_coverage", 0.0))
            >= float(gate_criteria["minimum_autograd_agreement"])
            and float(probe_autograd.get("matched_gradient_coverage", 0.0))
            >= float(gate_criteria["minimum_autograd_agreement"])
            and probe_autograd.get("median_relative_error") is not None
            and float(probe_autograd["median_relative_error"])
            <= float(criteria["maximum_probe_autograd_relative_error"]),
            "comparison_count": probe_autograd.get("comparison_count", 0),
            "total_gradient_rows": probe_autograd.get("total_gradient_rows", 0),
            "usable_finite_difference_coverage": probe_autograd.get("usable_finite_difference_coverage", 0.0),
            "matched_gradient_coverage": probe_autograd.get("matched_gradient_coverage", 0.0),
            "median_relative_error": probe_autograd.get("median_relative_error"),
            "maximum_allowed": criteria["maximum_probe_autograd_relative_error"],
        },
        *[
            {"name": f"exact_grid:{report['stage']}", **{key: value for key, value in report.items() if key not in {"stage"}}}
            for report in grid_checks
        ],
        *provenance_checks,
        *stability_checks(
            calibration["stability"],
            minimum_rank_correlation=float(criteria["minimum_rank_correlation"]),
            minimum_binding_agreement=float(criteria["minimum_binding_agreement"]),
            minimum_comparisons=int(criteria["minimum_comparisons_per_dimension"]),
        ),
    ]
    engineering = require_passed_gate(args.engineering_gate, gate_type="engineering_gate")
    smoke = require_passed_gate(args.smoke_gate, gate_type="smoke_gate")
    gate = make_gate(
        gate_type="probe_gate",
        checks=checks,
        config_hash=str(task["config_key"]),
        source_hash=current_source_hash,
        prerequisite_hashes={
            "engineering_gate_hash": engineering["gate_content_hash"],
            "smoke_gate_hash": smoke["gate_content_hash"],
            "frozen_config_hash": expected_hash,
            "split_manifest_hash": split_hash,
            "execution_manifest_hash": execution_hash,
            "ordered_cohort_hash": execution_manifest["ordered_cohort_hash"],
            "batch_boundary_hash": execution_manifest["batch_boundary_hash"],
        },
    )
    gate_path = Path(args.probe_gate)
    atomic_write_json(gate_path, gate, overwrite=args.overwrite)
    if not gate["passed"]:
        raise ContractError("probe validation gate failed")
    _record_global_pointer_completion(
        args,
        task_dir,
        task,
        repo_root,
        pointer_name="probe_validation_result.json",
        pointer={"passed": gate["passed"], "probe_gate": str(gate_path.resolve()), "gate_content_hash": gate["gate_content_hash"]},
    )


def _smoke_estimate_stage(
    args: argparse.Namespace, task_dir: Path, task: Mapping[str, Any], repo_root: Path
) -> None:
    from collections import defaultdict
    from .estimate_linkradius import estimate_from_pair_rows
    from .io_utils import atomic_write_csv
    from .merge_shards import merge_to_path
    from .probe_validation import reclassify_probe_pairs
    from .aggregate_causal_use import common_provenance

    root = _phase_root(args) / str(task["dataset"]) / f"R{int(task['R'])}"
    _verify_source_stage_grid(
        args,
        task,
        repo_root,
        stage="probe",
        partition="validation",
    )
    pairs = reclassify_probe_pairs(
        _completed_rows(root / "validation" / "probe", "probe_runs.jsonl", expected_source_hash=_cached_source_hash(args, repo_root)),
        {
            "minimum_requested_realized_cosine": 0.0,
            "maximum_off_direction_relative": 1.0,
            "minimum_signed_separation": 0.0,
            "minimum_antipodality": -1.0,
            "version": "linkradius_probe_thresholds_v1",
        },
    )
    probe_provenance = common_provenance(pairs)
    groups: dict[tuple[str, str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        key = (
            str(row["raw_sample_id"]),
            str(row["edge_id"]),
            int(row["probe_seed"]),
            float(row["h"]),
        )
        groups[key].append(row)
    competitor_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        first = group[0]
        clean_margins = {label: float(value) for label, value in first["clean_margins"].items()}
        normalized = [
            {
                "direction_id": int(row["direction_id"]),
                "accepted": bool(row.get("accepted", False)),
                "margins_plus": row["margins_plus"],
                "margins_minus": row["margins_minus"],
                "t_plus": row["t_plus"],
                "t_minus": row["t_minus"],
            }
            for row in group
        ]
        try:
            estimate = estimate_from_pair_rows(
                normalized,
                clean_margins=clean_margins,
                q=int(first["q"]),
                requested_K=int(args.K),
            )
        except ContractError:
            continue
        if not bool(estimate["primary_available"]):
            continue
        common = {
            **probe_provenance,
            "raw_sample_id": key[0],
            "edge_id": key[1],
            "probe_seed": key[2],
            "h": key[3],
            "requested_K": int(args.K),
            "K_eff": estimate["K_eff"],
            "primary_available": estimate["primary_available"],
        }
        competitor_rows.extend({**common, **value} for value in estimate["competitors"])
        edge_rows.append(
            {
                **common,
                "edge_radius": estimate["edge_radius"],
                "binding_competitor": estimate["binding_competitor"],
            }
        )
    execution_path = _execution_manifest_path(args, "validation", task)
    if not execution_path:
        raise ContractError("smoke estimate requires the validation execution manifest")
    execution = load_json(execution_path)
    expected_units = {
        (str(raw_id), edge)
        for raw_id, eligible in zip(
            execution["ordered_raw_sample_ids"], execution["analysis_eligible"]
        )
        if bool(eligible)
        for edge in ("p2c@0", "c2s@0", "s2p@0")
    }
    observed_units = {
        (str(row["raw_sample_id"]), str(row["edge_id"])) for row in edge_rows
    }
    if not edge_rows or observed_units != expected_units:
        raise ContractError(
            "smoke requires at least one complete primary K-direction estimate for every eligible sample/early edge"
        )
    atomic_write_csv(task_dir / "linkradius_competitors.csv", competitor_rows, overwrite=args.overwrite)
    atomic_write_csv(task_dir / "linkradius_edges.csv", edge_rows, overwrite=args.overwrite)
    expected_probe_tasks = [
        candidate.as_dict()
        for candidate in build_grid(_build_grid_config(args, "probe"))
        if candidate.dataset == str(task["dataset"])
        and int(candidate.R) == int(task["R"])
        and candidate.partition == "validation"
    ]
    probe_shards = sorted(
        (root / "validation" / "probe").rglob("probe_runs.jsonl")
    )
    merge_to_path(
        probe_shards,
        expected_tasks=expected_probe_tasks,
        output=task_dir / "probe_runs.jsonl",
        require_completion=True,
        overwrite=args.overwrite,
        expected_source_hash=_cached_source_hash(args, repo_root),
    )
    atomic_write_csv(
        task_dir / "probe_pairs.csv",
        _probe_pair_csv_rows(pairs),
        overwrite=args.overwrite,
    )
    atomic_write_json(
        task_dir / "estimate_result.json",
        {"competitor_rows": len(competitor_rows), "edge_rows": len(edge_rows)},
        overwrite=args.overwrite,
    )
    publish_completion(
        task_dir,
        config_hash=str(task["config_key"]),
        source_hash_value=_cached_source_hash(args, repo_root),
        artifact_paths=["manifest.json", "command.txt", "estimate_result.json", "linkradius_competitors.csv", "linkradius_edges.csv", "probe_runs.jsonl", "probe_pairs.csv"],
        row_counts={"probe_runs.jsonl": len(load_jsonl(task_dir / "probe_runs.jsonl"))},
        extra={"array_index": int(task["array_index"])},
        overwrite=args.overwrite,
    )


def _smoke_aggregate_stage(
    args: argparse.Namespace, task_dir: Path, task: Mapping[str, Any], repo_root: Path
) -> None:
    from .aggregate_causal_use import aggregate_causal_rows, eligible_complete_causal_rows
    from .io_utils import atomic_write_csv
    from .merge_shards import merge_to_path

    root = _phase_root(args) / str(task["dataset"]) / f"R{int(task['R'])}"
    _verify_source_stage_grid(
        args,
        task,
        repo_root,
        stage="causal",
        partition="validation",
    )
    causal = eligible_complete_causal_rows(
        _completed_rows(root / "validation" / "causal", "causal_runs.jsonl", expected_source_hash=_cached_source_hash(args, repo_root)),
        expected_edges=("p2c@0", "c2s@0", "s2p@0"),
        expected_modes=tuple(args.interventions.split()),
    )
    paired, summaries = aggregate_causal_rows(causal, bootstrap_draws=args.bootstrap_draws, seed=int(task["seed"]))
    atomic_write_csv(task_dir / "causal_use_rows.csv", paired, overwrite=args.overwrite)
    atomic_write_csv(task_dir / "causal_use_summary.csv", summaries, overwrite=args.overwrite)
    clean_row_count = _merge_public_stage_jsonl(
        args,
        task,
        stage="clean",
        partition="validation",
        filename="clean_baseline.jsonl",
        output=task_dir / "clean_baseline.jsonl",
    )
    atomic_write_json(task_dir / "aggregate_result.json", {"paired_rows": len(paired), "summary_rows": len(summaries)}, overwrite=args.overwrite)
    publish_completion(
        task_dir,
        config_hash=str(task["config_key"]),
        source_hash_value=_cached_source_hash(args, repo_root),
        artifact_paths=["manifest.json", "command.txt", "aggregate_result.json", "causal_use_rows.csv", "causal_use_summary.csv", "clean_baseline.jsonl"],
        row_counts={"clean_baseline.jsonl": clean_row_count},
        extra={"array_index": int(task["array_index"])},
        overwrite=args.overwrite,
    )


def _pilot_aggregate_stage(
    args: argparse.Namespace, task_dir: Path, task: Mapping[str, Any], repo_root: Path
) -> None:
    """Publish validation-only causal and probe diagnostics after probe freeze."""

    from collections import defaultdict
    from .aggregate_causal_use import (
        aggregate_causal_rows,
        common_provenance,
        eligible_complete_causal_rows,
    )
    from .estimate_linkradius import estimate_from_pair_rows
    from .io_utils import atomic_write_csv
    from .probe_validation import reclassify_probe_pairs

    probe_gate = require_passed_gate(args.probe_gate, gate_type="probe_gate")
    if probe_gate.get("source_hash") != _cached_source_hash(args, repo_root):
        raise ContractError("pilot aggregate requires the current probe gate")
    frozen = _authenticated_frozen_probe_config(
        args,
        task,
        current_source_hash=_cached_source_hash(args, repo_root),
    )
    if probe_gate.get("frozen_config_hash") != frozen.get("content_hash"):
        raise ContractError("pilot aggregate probe gate/config hashes differ")
    for stage in ("causal", "probe_calibration"):
        _verify_source_stage_grid(
            args,
            task,
            repo_root,
            stage=stage,
            partition="validation",
        )
    root = _phase_root(args) / str(task["dataset"]) / f"R{int(task['R'])}" / "validation"
    edges = tuple(
        f"{site}@{round_idx}" for site, round_idx in canonical_edge_pairs(int(task["R"]))
    )
    _, execution_manifest, _ = _authenticated_execution_manifest(
        args, "validation", task, repo_root
    )
    eligible_raw_ids = tuple(
        str(raw_id)
        for raw_id, eligible in zip(
            execution_manifest["ordered_raw_sample_ids"],
            execution_manifest["analysis_eligible"],
        )
        if bool(eligible)
    )
    causal = eligible_complete_causal_rows(
        _completed_rows(root / "causal", "causal_runs.jsonl", expected_source_hash=_cached_source_hash(args, repo_root)),
        expected_edges=edges,
        expected_modes=tuple(args.interventions.split()),
        expected_raw_ids=eligible_raw_ids,
    )
    paired, causal_summary = aggregate_causal_rows(
        causal, bootstrap_draws=args.bootstrap_draws, seed=int(task["seed"])
    )
    probe_rows = _completed_rows(root / "probe_calibration", "probe_runs.jsonl", expected_source_hash=_cached_source_hash(args, repo_root))
    classified = [
        row
        for row in reclassify_probe_pairs(
            probe_rows, frozen["acceptance_thresholds"]
        )
        if bool(row.get("analysis_eligible", False))
    ]
    if not classified:
        raise ContractError("pilot aggregate found no eligible validation probe pairs")
    probe_provenance = common_provenance(classified)
    groups: dict[tuple[str, float, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in classified:
        groups[(str(row["edge_id"]), float(row["h"]), int(row["probe_seed"]))].append(row)
    diagnostics = []
    for (edge, h, seed), rows in sorted(groups.items()):
        accepted = [row for row in rows if bool(row.get("accepted"))]
        diagnostics.append(
            {
                **probe_provenance,
                "edge_id": edge,
                "h": h,
                "probe_seed": seed,
                "pair_count": len(rows),
                "accepted_count": len(accepted),
                "acceptance_rate": len(accepted) / len(rows),
                "mean_realized_separation": sum(float(row["realized_separation"]) for row in accepted) / len(accepted) if accepted else None,
                "mean_antipodality": sum(float(row["antipodality"]) for row in accepted) / len(accepted) if accepted else None,
            }
        )
    estimate_groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in classified:
        if (
            float(row.get("h", -1.0)) == float(frozen["selected_h"])
            and int(row.get("direction_id", -1)) < int(frozen["K"])
        ):
            estimate_groups[
                (
                    str(row["raw_sample_id"]),
                    str(row["edge_id"]),
                    int(row["probe_seed"]),
                )
            ].append(row)
    competitor_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    for (raw_id, edge, seed), rows in sorted(estimate_groups.items()):
        first = rows[0]
        clean_margins = {
            str(label): float(value)
            for label, value in first["clean_margins"].items()
        }
        normalized = [
            {
                "direction_id": int(row["direction_id"]),
                "accepted": bool(row.get("accepted", False)),
                "margins_plus": row["margins_plus"],
                "margins_minus": row["margins_minus"],
                "t_plus": row["t_plus"],
                "t_minus": row["t_minus"],
            }
            for row in rows
        ]
        try:
            estimate = estimate_from_pair_rows(
                normalized,
                clean_margins=clean_margins,
                q=int(first["q"]),
                requested_K=int(frozen["K"]),
            )
        except ContractError:
            continue
        common = {
            **probe_provenance,
            "raw_sample_id": raw_id,
            "edge_id": edge,
            "probe_seed": seed,
            "h": float(frozen["selected_h"]),
            "requested_K": int(frozen["K"]),
            "K_eff": estimate["K_eff"],
            "primary_available": estimate["primary_available"],
        }
        competitor_rows.extend({**common, **value} for value in estimate["competitors"])
        edge_rows.append(
            {
                **common,
                "edge_radius": estimate["edge_radius"],
                "binding_competitor": estimate["binding_competitor"],
            }
        )
    if not any(bool(row["primary_available"]) for row in edge_rows):
        raise ContractError("pilot aggregate has no complete primary LinkRadius estimate")
    atomic_write_csv(task_dir / "causal_use_rows.csv", paired, overwrite=args.overwrite)
    atomic_write_csv(task_dir / "causal_use_summary.csv", causal_summary, overwrite=args.overwrite)
    atomic_write_csv(task_dir / "probe_diagnostics.csv", diagnostics, overwrite=args.overwrite)
    atomic_write_csv(task_dir / "linkradius_competitors.csv", competitor_rows, overwrite=args.overwrite)
    atomic_write_csv(task_dir / "linkradius_edges.csv", edge_rows, overwrite=args.overwrite)
    expected_probe_tasks = [
        candidate.as_dict()
        for candidate in build_grid(_build_grid_config(args, "probe_calibration"))
        if candidate.dataset == str(task["dataset"])
        and int(candidate.R) == int(task["R"])
        and candidate.partition == "validation"
    ]
    probe_shards = sorted((root / "probe_calibration").rglob("probe_runs.jsonl"))
    merge_to_path(
        probe_shards,
        expected_tasks=expected_probe_tasks,
        output=task_dir / "probe_runs.jsonl",
        require_completion=True,
        overwrite=args.overwrite,
        expected_source_hash=_cached_source_hash(args, repo_root),
    )
    atomic_write_csv(
        task_dir / "probe_pairs.csv",
        _probe_pair_csv_rows(classified),
        overwrite=args.overwrite,
    )
    clean_row_count = _merge_public_stage_jsonl(
        args,
        task,
        stage="clean",
        partition="validation",
        filename="clean_baseline.jsonl",
        output=task_dir / "clean_baseline.jsonl",
    )
    atomic_write_json(
        task_dir / "aggregate_result.json",
        {
            "partition": "validation",
            "causal_pair_rows": len(paired),
            "causal_summary_rows": len(causal_summary),
            "probe_diagnostic_rows": len(diagnostics),
            "probe_pair_rows": len(classified),
            "linkradius_competitor_rows": len(competitor_rows),
            "linkradius_edge_rows": len(edge_rows),
            "clean_public_rows": clean_row_count,
            "frozen_config_hash": frozen["content_hash"],
        },
        overwrite=args.overwrite,
    )
    publish_completion(
        task_dir,
        config_hash=str(task["config_key"]),
        source_hash_value=_cached_source_hash(args, repo_root),
        artifact_paths=[
            "manifest.json",
            "command.txt",
            "aggregate_result.json",
            "causal_use_rows.csv",
            "causal_use_summary.csv",
            "probe_diagnostics.csv",
            "linkradius_competitors.csv",
            "linkradius_edges.csv",
            "probe_runs.jsonl",
            "probe_pairs.csv",
            "clean_baseline.jsonl",
        ],
        row_counts={
            "probe_runs.jsonl": len(load_jsonl(task_dir / "probe_runs.jsonl")),
            "clean_baseline.jsonl": clean_row_count,
        },
        extra={"array_index": int(task["array_index"])},
        overwrite=args.overwrite,
    )


def _aggregate_source_root(args: argparse.Namespace, task: Mapping[str, Any]) -> Path:
    if args.aggregate_phase not in {"engineering", "smoke", "pilot", "attacks"}:
        raise ContractError("--aggregate-phase must name an implemented experiment phase")
    return (
        Path(args.out_root).resolve()
        / args.aggregate_phase
        / str(task["dataset"])
        / f"R{int(task['R'])}"
    )


def _aggregate_gate_path(args: argparse.Namespace) -> Path:
    return (
        Path(args.aggregate_verification_gate)
        if args.aggregate_verification_gate
        else Path(args.out_root) / f"aggregate_verification_{args.aggregate_phase}_gate.json"
    )


def _verified_aggregate_directories(
    args: argparse.Namespace,
    task: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, list[Path]]:
    """Recheck the gate inventory and return only its exact source directories."""

    source_root = _aggregate_source_root(args, task)
    gate_path = _aggregate_gate_path(args)
    gate = require_passed_gate(
        gate_path, gate_type="aggregate_verification_gate"
    )
    current_source_hash = _cached_source_hash(args, repo_root)
    if gate.get("source_hash") != current_source_hash:
        raise ContractError("aggregate verification inventory has a stale source hash")
    _authenticate_gate_completion(
        args,
        task,
        gate_path=gate_path,
        gate_type="aggregate_verification_gate",
        gate=gate,
        current_source_hash=current_source_hash,
    )
    if gate.get("aggregate_phase") != args.aggregate_phase:
        raise ContractError(
            "aggregate verification inventory was produced for a different phase"
        )
    required_stages = set(REQUIRED_AGGREGATE_STAGES[args.aggregate_phase])
    verified_stages = gate.get("verified_stages")
    if not isinstance(verified_stages, list) or not required_stages.issubset(
        set(verified_stages)
    ):
        raise ContractError("aggregate verification inventory has incomplete scope")
    expected_scope_hash = content_hash(
        {
            "aggregate_phase": args.aggregate_phase,
            "verified_stages": verified_stages,
            "source_hash": current_source_hash,
        },
        domain="linkradius:aggregate_verification_scope:v1",
    )
    if gate.get("verification_scope_hash") != expected_scope_hash:
        raise ContractError("aggregate verification inventory scope hash is stale")
    expected_by_stage: dict[str, list[Mapping[str, Any]]] = {}
    for check in gate.get("checks", []):
        name = str(check.get("name") or "")
        if not name.startswith("expected_stage:"):
            continue
        stage = name.split(":", 1)[1]
        inventory = check.get("completion_inventory")
        if not isinstance(inventory, list):
            raise ContractError("aggregate verification gate lacks a completion inventory")
        expected_by_stage[stage] = inventory
    if set(expected_by_stage) != set(gate.get("verified_stages", [])):
        raise ContractError("aggregate gate stage and inventory scopes differ")
    expected_inventory_hash = content_hash(
        [
            {"stage": stage, "items": expected_by_stage[stage]}
            for stage in gate["verified_stages"]
        ],
        domain="linkradius:aggregate_completion_inventory:v1",
    )
    if gate.get("completion_inventory_hash") != expected_inventory_hash:
        raise ContractError("aggregate completion inventory hash is stale")

    actual_by_stage: dict[str, list[dict[str, Any]]] = {
        stage: [] for stage in gate["verified_stages"]
    }
    directories: dict[str, list[Path]] = {
        stage: [] for stage in gate["verified_stages"]
    }
    for completion_path in sorted(source_root.rglob(".complete.json")):
        manifest_path = completion_path.parent / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = load_json(manifest_path)
        stage = manifest.get("task", {}).get("stage")
        if stage not in actual_by_stage:
            continue
        record = verify_completion(completion_path.parent)
        relative = completion_path.parent.relative_to(source_root)
        if ".." in relative.parts:
            raise ContractError("aggregate inventory escaped its source root")
        actual_by_stage[str(stage)].append(
            {
                "directory": relative.as_posix(),
                "completion_sha256": file_sha256(completion_path),
                "config_hash": str(record["config_hash"]),
                "source_hash": str(record["source_hash"]),
            }
        )
        directories[str(stage)].append(completion_path.parent)
    for stage in gate["verified_stages"]:
        if actual_by_stage[stage] != expected_by_stage[stage]:
            raise ContractError(
                f"source completions changed after aggregate verification for stage {stage}"
            )
    return directories


def _required_verified_artifacts(
    directories: Mapping[str, Sequence[Path]],
    *,
    stage: str,
    filename: str,
    expected_source_hash: str | None = None,
) -> list[Path]:
    """Require one completion-declared artifact in every verified task."""

    if stage not in directories or not directories[stage]:
        raise ContractError(f"aggregate verification did not cover stage {stage}")
    paths: list[Path] = []
    for directory in directories[stage]:
        completion = verify_completion(directory)
        if (
            expected_source_hash is not None
            and completion.get("source_hash") != expected_source_hash
        ):
            raise ContractError(f"{filename} completion has a stale source hash")
        declared = [
            artifact
            for artifact in completion["artifacts"]
            if artifact.get("path") == filename
        ]
        path = directory / filename
        if len(declared) != 1 or not path.is_file():
            raise ContractError(
                f"every verified {stage} task must declare exactly one {filename}"
            )
        paths.append(path)
    return paths


def _single_required_verified_artifact(
    directories: Mapping[str, Sequence[Path]],
    *,
    stage: str,
    filename: str,
    expected_source_hash: str | None = None,
) -> Path:
    paths = _required_verified_artifacts(
        directories,
        stage=stage,
        filename=filename,
        expected_source_hash=expected_source_hash,
    )
    if len(paths) != 1:
        raise ContractError(
            f"aggregate requires exactly one verified {stage} source declaring "
            f"{filename}; found {len(paths)}"
        )
    return paths[0]


def _rows_from_verified_directories(
    directories: Mapping[str, Sequence[Path]],
    *,
    stage: str,
    filename: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _required_verified_artifacts(
        directories, stage=stage, filename=filename
    ):
        rows.extend(load_jsonl(path))
    return rows


def _aggregate_verify_stage(
    args: argparse.Namespace, task_dir: Path, task: Mapping[str, Any], repo_root: Path
) -> None:
    from .validate_stage import verify_expected_completions

    source_root = _aggregate_source_root(args, task)
    required_stages = REQUIRED_AGGREGATE_STAGES[args.aggregate_phase]
    requested_stages = tuple(args.verify_stages.split())
    if len(requested_stages) != len(set(requested_stages)):
        raise ContractError("--verify-stages may not contain duplicates")
    missing_required = [stage for stage in required_stages if stage not in requested_stages]
    if missing_required:
        raise ContractError(
            "aggregate verification requires the complete source workflow; "
            f"missing stages: {' '.join(missing_required)}"
        )
    stage_names = required_stages + tuple(
        stage for stage in requested_stages if stage not in required_stages
    )
    execution_manifests: dict[str, str] = {}
    for partition in ("attack_train", "validation", "test"):
        path = source_root / partition / "execution_manifest.json"
        if path.is_file():
            execution_manifests[partition] = str(path)
    stage_reports = []
    current_source_hash = _cached_source_hash(args, repo_root)
    for stage in stage_names:
        stage_execution_manifests = (
            execution_manifests
            if stage
            in {
                "clean",
                "replay",
                "causal",
                "probe",
                "probe_calibration",
                "gradient",
                "attack",
                "val",
                "test_probe",
                "test",
            }
            else {}
        )
        config = GridConfig(
            workflow=args.aggregate_phase,
            stage=stage,
            datasets=(str(task["dataset"]),),
            rounds=(int(task["R"]),),
            seeds=(int(task["seed"]),),
            partitions=(),
            num_batches=args.num_batches,
            batch_counts={key: int(value) for key, value in (item.split("=", 1) for item in args.batch_count)},
            probe_radii=tuple(float(value) for value in args.probe_radii.split()),
            probe_seeds=tuple(int(value) for value in args.probe_seeds.split()),
            K=int(args.K),
            subspace=args.subspace,
            interventions=tuple(args.interventions.split()),
            attack_families=tuple(args.attack_families.split()),
            attack_epsilons=tuple(float(value) for value in args.attack_epsilons.split()),
            execution_manifests=stage_execution_manifests,
            style=args.style,
            method=args.method,
            batch_size=args.batch_size,
            latent_length=args.latent_length,
            discovery_batches=args.discovery_batches,
            runner_config_hash=_runner_config_digest(
                args,
                workflow=args.aggregate_phase,
                stage=stage,
            ),
        )
        expected = build_grid(config)
        records = []
        for completion_path in source_root.rglob(".complete.json"):
            manifest_path = completion_path.parent / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = load_json(manifest_path)
            if manifest.get("task", {}).get("stage") != stage:
                continue
            records.append((completion_path, verify_completion(completion_path.parent)))
        report = verify_expected_completions(
            expected,
            records,
            expected_source_hash=current_source_hash,
        )
        completion_inventory = [
            {
                "directory": completion_path.parent.relative_to(source_root).as_posix(),
                "completion_sha256": file_sha256(completion_path),
                "config_hash": str(record["config_hash"]),
                "source_hash": str(record["source_hash"]),
            }
            for completion_path, record in sorted(
                records, key=lambda item: item[0].relative_to(source_root).as_posix()
            )
        ]
        stage_reports.append(
            {"stage": stage, **report, "completion_inventory": completion_inventory}
        )
    verification = {
        "schema_version": "linkradius.aggregate_verification.v1",
        "aggregate_phase": args.aggregate_phase,
        "source_hash": current_source_hash,
        "verified_stages": list(stage_names),
        "completion_inventory_hash": content_hash(
            [
                {"stage": report["stage"], "items": report["completion_inventory"]}
                for report in stage_reports
            ],
            domain="linkradius:aggregate_completion_inventory:v1",
        ),
        "passed": all(report["passed"] for report in stage_reports),
        "stages": stage_reports,
    }
    atomic_write_json(task_dir / "verification.json", verification, overwrite=args.overwrite)
    if not verification["passed"]:
        missing = {
            report["stage"]: report["missing_array_indices"]
            for report in stage_reports
            if report["missing_array_indices"]
        }
        stale = {
            report["stage"]: report["stale_array_indices"]
            for report in stage_reports
            if report["stale_array_indices"]
        }
        raise ContractError(f"aggregate verification failed; missing={missing}, stale={stale}")
    verification_gate = make_gate(
        gate_type="aggregate_verification_gate",
        checks=[
            {"name": f"expected_stage:{report['stage']}", "passed": report["passed"], **{key: value for key, value in report.items() if key not in {"stage", "passed"}}}
            for report in stage_reports
        ],
        config_hash=str(task["config_key"]),
        source_hash=current_source_hash,
        prerequisite_hashes={
            "aggregate_phase": args.aggregate_phase,
            "verified_stages": list(stage_names),
            "verification_scope_hash": content_hash(
                {
                    "aggregate_phase": args.aggregate_phase,
                    "verified_stages": list(stage_names),
                    "source_hash": current_source_hash,
                },
                domain="linkradius:aggregate_verification_scope:v1",
            ),
            "completion_inventory_hash": verification["completion_inventory_hash"],
        },
    )
    gate_path = _aggregate_gate_path(args)
    atomic_write_json(gate_path, verification_gate, overwrite=args.overwrite)
    local_gate_name = "aggregate_verification_gate.json"
    pointer_name = "aggregate_verification_result.json"
    atomic_write_json(
        task_dir / local_gate_name, verification_gate, overwrite=args.overwrite
    )
    atomic_write_json(
        task_dir / pointer_name,
        {
            "schema_version": "linkradius.aggregate_verification_result.v1",
            "aggregate_phase": args.aggregate_phase,
            "aggregate_verification_gate": str(gate_path.resolve()),
            "gate_content_hash": verification_gate["gate_content_hash"],
            "local_gate_sha256": file_sha256(task_dir / local_gate_name),
            "source_hash": current_source_hash,
        },
        overwrite=args.overwrite,
    )
    publish_completion(
        task_dir,
        config_hash=str(task["config_key"]),
        source_hash_value=current_source_hash,
        artifact_paths=[
            "manifest.json",
            "command.txt",
            "verification.json",
            local_gate_name,
            pointer_name,
        ],
        extra={
            "array_index": int(task["array_index"]),
            "gate_content_hash": verification_gate["gate_content_hash"],
            "aggregate_verification_gate_hash": verification_gate[
                "gate_content_hash"
            ],
        },
        overwrite=args.overwrite,
    )


def _aggregate_cpu_stage(
    args: argparse.Namespace, task_dir: Path, task: Mapping[str, Any], repo_root: Path
) -> None:
    import csv
    from .io_utils import atomic_write_csv, atomic_write_text

    source_root = _aggregate_source_root(args, task)
    verified_directories = _verified_aggregate_directories(args, task, repo_root)
    stage = str(task["stage"])
    artifacts = ["manifest.json", "command.txt"]
    if stage == "causal":
        from .aggregate_causal_use import aggregate_causal_rows, eligible_complete_causal_rows

        raw_rows = [
            row
            for row in _rows_from_verified_directories(
                verified_directories, stage="causal", filename="causal_runs.jsonl"
            )
        ]
        expected_edges = (
            ("p2c@0", "c2s@0", "s2p@0")
            if args.aggregate_phase == "smoke"
            else tuple(
                f"{site}@{round_idx}"
                for site, round_idx in canonical_edge_pairs(int(task["R"]))
            )
        )
        rows = eligible_complete_causal_rows(
            raw_rows,
            expected_edges=expected_edges,
            expected_modes=tuple(args.interventions.split()),
        )
        paired, summaries = aggregate_causal_rows(rows, bootstrap_draws=args.bootstrap_draws, seed=int(task["seed"]))
        atomic_write_csv(task_dir / "causal_use_rows.csv", paired, overwrite=args.overwrite)
        atomic_write_csv(task_dir / "causal_use_summary.csv", summaries, overwrite=args.overwrite)
        artifacts.extend(("causal_use_rows.csv", "causal_use_summary.csv"))
    elif stage == "linkradius":
        linkradius_source_stage = (
            "aggregate" if args.aggregate_phase == "pilot" else "estimate"
        )
        current_source_hash = _cached_source_hash(args, repo_root)
        edges_path = _single_required_verified_artifact(
            verified_directories,
            stage=linkradius_source_stage,
            filename="linkradius_edges.csv",
            expected_source_hash=current_source_hash,
        )
        competitors_path = _single_required_verified_artifact(
            verified_directories,
            stage=linkradius_source_stage,
            filename="linkradius_competitors.csv",
            expected_source_hash=current_source_hash,
        )
        atomic_write_text(task_dir / "linkradius_edges.csv", edges_path.read_text(encoding="utf-8"), overwrite=args.overwrite)
        atomic_write_text(task_dir / "linkradius_competitors.csv", competitors_path.read_text(encoding="utf-8"), overwrite=args.overwrite)
        artifacts.extend(("linkradius_edges.csv", "linkradius_competitors.csv"))
    elif stage == "attacks":
        from .aggregate_attack_thresholds import aggregate_thresholds

        attack_stage = "test" if args.aggregate_phase == "attacks" else "attack"
        rows = [
            row
            for row in _rows_from_verified_directories(
                verified_directories,
                stage=attack_stage,
                filename="attack_results.jsonl",
            )
            if row.get("record_type") == "sample"
            and bool(row.get("analysis_eligible", False))
            and not row.get("failure")
        ]
        if not rows:
            raise ContractError("no completed compatible attack sample rows")
        thresholds = aggregate_thresholds(rows)
        _write_union_csv(
            task_dir / "attack_thresholds.csv",
            thresholds,
            overwrite=args.overwrite,
        )
        artifacts.append("attack_thresholds.csv")
    elif stage == "metrics":
        from .aggregate_causal_use import common_provenance

        probe_stage = "probe_calibration" if args.aggregate_phase == "pilot" else "probe"
        clean = [
            row
            for row in _rows_from_verified_directories(
                verified_directories, stage="clean", filename="clean_baseline.jsonl"
            )
            if row.get("record_type") == "sample"
        ]
        raw_probes = [
            row
            for row in _rows_from_verified_directories(
                verified_directories, stage=probe_stage, filename="probe_runs.jsonl"
            )
        ]
        probes = [
            row
            for row in raw_probes
            if row.get("record_type") == "probe_pair"
            and bool(row.get("analysis_eligible", False))
        ]
        signed = [
            row
            for row in raw_probes
            if row.get("record_type") == "sample"
            and row.get("intervention_mode") == "additive_antithetic"
            and row.get("sign") in {-1, 1}
            and bool(row.get("analysis_eligible", False))
        ]
        if not clean or not probes or not signed:
            raise ContractError("diagnostic metrics require clean, signed-probe, and pair artifacts")
        provenance = common_provenance([*signed, *probes])
        diagnostics = [
            row.get("diagnostics", row.get("realized_intervention")) for row in signed
        ]
        if not all(isinstance(value, Mapping) for value in diagnostics):
            raise ContractError("signed probe diagnostics are incomplete")
        parsed_diagnostics = [value for value in diagnostics if isinstance(value, Mapping)]
        norm_ratios = [
            float(value["realized_delta_norm"]) / float(value["requested_delta_norm"])
            for value in parsed_diagnostics
            if float(value.get("requested_delta_norm", 0.0)) > 0.0
            and math.isfinite(float(value.get("realized_delta_norm", float("nan"))))
        ]
        cosines = [
            float(value["requested_realized_cosine"])
            for value in parsed_diagnostics
            if value.get("requested_realized_cosine") is not None
            and math.isfinite(float(value["requested_realized_cosine"]))
        ]
        off_direction = [
            float(value["off_direction_relative"])
            for value in parsed_diagnostics
            if value.get("off_direction_relative") is not None
            and math.isfinite(float(value["off_direction_relative"]))
        ]
        antipodalities = [
            float(row["antipodality"])
            for row in probes
            if row.get("antipodality") is not None
            and math.isfinite(float(row["antipodality"]))
        ]
        if not norm_ratios or not cosines or not antipodalities:
            raise ContractError("cast diagnostics lack finite norm, cosine, or antipodality values")
        agreement = _all_clean_scorer_agreement(clean)
        exclusion_counts: dict[str, int] = {}
        for row in clean:
            reason = str(row.get("exclusion_reason") or "included")
            exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
        comparable = [
            row
            for row in clean
            if row.get("strict_generated_choice") in {"A", "B", "C", "D"}
            and bool(row.get("strict_generated_valid", False))
        ]
        metrics = [
            {**provenance, "metric": "strict_generated_accuracy", "value": sum(row.get("strict_generated_choice") == row.get("gold") for row in comparable) / len(comparable) if comparable else None, "n": len(comparable)},
            {**provenance, "metric": "forced_choice_accuracy", "value": sum(row.get("scorer_prediction") == row.get("gold") for row in clean) / len(clean), "n": len(clean)},
            {**provenance, "metric": "scorer_generated_agreement_comparable", "value": agreement["agreement"] if agreement["comparable_rows"] else None, "n": agreement["comparable_rows"]},
            {**provenance, "metric": "scorer_generated_agreement_all_rows", "value": agreement["agreement_all_rows"], "n": agreement["total_clean_rows"]},
            {**provenance, "metric": "scorer_generated_comparable_coverage", "value": agreement["comparable_coverage"], "n": agreement["total_clean_rows"]},
            {**provenance, "metric": "dual_correct_rows", "value": sum(bool(row.get("analysis_eligible", False)) for row in clean), "n": len(clean)},
            {**provenance, "metric": "analysis_excluded_rows", "value": agreement["analysis_ineligible_rows"], "n": len(clean)},
            {**provenance, "metric": "invalid_generation_rows", "value": agreement["invalid_generation_rows"], "n": len(clean)},
            {**provenance, "metric": "scorer_tie_or_invalid_rows", "value": agreement["scorer_tie_or_invalid_rows"], "n": len(clean)},
            {**provenance, "metric": "probe_cast_survival_rate", "value": sum(not bool(value.get("collapsed", False)) for value in parsed_diagnostics) / len(parsed_diagnostics), "n": len(parsed_diagnostics)},
            {**provenance, "metric": "probe_actual_requested_norm_ratio_mean", "value": sum(norm_ratios) / len(norm_ratios), "n": len(norm_ratios)},
            {**provenance, "metric": "probe_requested_realized_cosine_mean", "value": sum(cosines) / len(cosines), "n": len(cosines)},
            {**provenance, "metric": "probe_off_direction_relative_mean", "value": sum(off_direction) / len(off_direction) if off_direction else None, "n": len(off_direction)},
            {**provenance, "metric": "probe_antipodality_mean", "value": sum(antipodalities) / len(antipodalities), "n": len(antipodalities)},
            {**provenance, "metric": "probe_collapse_rate", "value": sum(bool(value.get("collapsed", False)) for value in parsed_diagnostics) / len(parsed_diagnostics), "n": len(parsed_diagnostics)},
            {**provenance, "metric": "probe_rejection_rate", "value": sum(not bool(row.get("accepted")) for row in probes) / len(probes), "n": len(probes)},
            *[
                {
                    **provenance,
                    "metric": f"execution_exclusion_reason:{reason}",
                    "value": count,
                    "n": len(clean),
                }
                for reason, count in sorted(exclusion_counts.items())
            ],
        ]
        atomic_write_csv(task_dir / "diagnostic_metrics.csv", metrics, overwrite=args.overwrite)
        artifacts.append("diagnostic_metrics.csv")
    elif stage == "system_curves":
        from .build_system_curves import build_predicted_system_curves

        linkradius_source_stage = (
            "aggregate" if args.aggregate_phase == "pilot" else "estimate"
        )
        edges_path = _single_required_verified_artifact(
            verified_directories,
            stage=linkradius_source_stage,
            filename="linkradius_edges.csv",
            expected_source_hash=_cached_source_hash(args, repo_root),
        )
        with edges_path.open("r", encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
        deduplicated: dict[tuple[str, str], Mapping[str, Any]] = {}
        for row in sorted(raw_rows, key=lambda value: (value["raw_sample_id"], value["edge_id"], value.get("h", ""), value.get("probe_seed", ""))):
            deduplicated.setdefault((row["raw_sample_id"], row["edge_id"]), row)
        curves, summaries = build_predicted_system_curves(
            list(deduplicated.values()),
            [float(value) for value in args.attack_epsilons.split()],
            fixed_edge=args.fixed_edge or "p2c@0",
            useful_edges=args.useful_edges.split() or ("p2c@0", "c2s@0", "s2p@0"),
        )
        atomic_write_csv(task_dir / "system_curves.csv", curves, overwrite=args.overwrite)
        atomic_write_csv(task_dir / "system_curve_summary.csv", summaries, overwrite=args.overwrite)
        artifacts.extend(("system_curves.csv", "system_curve_summary.csv"))
    else:
        raise ContractError(f"unsupported CPU aggregate stage: {stage}")
    publish_completion(
        task_dir,
        config_hash=str(task["config_key"]),
        source_hash_value=_cached_source_hash(args, repo_root),
        artifact_paths=artifacts,
        extra={"array_index": int(task["array_index"])},
        overwrite=args.overwrite,
    )


def _write_union_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    overwrite: bool,
) -> None:
    fieldnames = sorted({str(key) for row in rows for key in row})
    atomic_write_csv(
        path,
        rows,
        fieldnames=fieldnames,
        overwrite=overwrite,
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fresh_dual_correct_ids(
    clean_rows: Sequence[Mapping[str, Any]],
) -> set[str]:
    from .select_clean_correct import classify_screening_row

    return {
        str(row["raw_sample_id"])
        for row in clean_rows
        if row.get("record_type") == "sample"
        and classify_screening_row(row)[0]
    }


def _attack_threshold_stage(
    args: argparse.Namespace,
    task_dir: Path,
    task: Mapping[str, Any],
    repo_root: Path,
) -> None:
    from .aggregate_attack_thresholds import aggregate_thresholds

    frozen = _authenticated_frozen_attack_config(
        args,
        task,
        current_source_hash=_cached_source_hash(args, repo_root),
    )
    for stage in ("clean", "test"):
        _verify_source_stage_grid(
            args, task, repo_root, stage=stage, partition="test"
        )
    root = (
        _phase_root(args)
        / str(task["dataset"])
        / f"R{int(task['R'])}"
        / "test"
    )
    clean_rows = _completed_rows(
        root / "clean",
        "clean_baseline.jsonl",
        expected_source_hash=_cached_source_hash(args, repo_root),
    )
    eligible_ids = _fresh_dual_correct_ids(clean_rows)
    if not eligible_ids:
        raise ContractError(
            "held-out test contains no fresh dual-correct rows for boundary analysis"
        )
    attack_rows = [
        row
        for row in _completed_rows(
            root / "test",
            "attack_results.jsonl",
            expected_source_hash=_cached_source_hash(args, repo_root),
        )
        if row.get("record_type") != "shard_metadata"
        and str(row.get("raw_sample_id") or "") in eligible_ids
    ]
    if _common_system_identity(
        [*clean_rows, *attack_rows], where="held-out threshold evidence"
    ) != frozen.get("system_identity"):
        raise ContractError(
            "held-out threshold evidence differs from the frozen system identity"
        )
    pgd_target_report = _validate_pgd_target_evidence(
        attack_rows, where="held-out threshold evidence"
    )
    thresholds = aggregate_thresholds(
        attack_rows,
        requested_budget_grid=[float(value) for value in frozen["attack_epsilons"]],
        tie_tolerance=0.0,
    )
    if not thresholds:
        raise ContractError("failure-threshold aggregation produced no rows")
    _write_union_csv(
        task_dir / "failure_thresholds.csv",
        thresholds,
        overwrite=args.overwrite,
    )
    summary = {
        "schema_version": "linkradius.failure_threshold_summary.v1",
        "eligible_raw_sample_count": len(eligible_ids),
        "threshold_rows": len(thresholds),
        "edge_summary_rows": sum(
            row.get("curve_kind") == "edge_summary" for row in thresholds
        ),
        "attack_target_rows": sum(
            row.get("curve_kind") == "attack_target" for row in thresholds
        ),
        "right_censored_rows": sum(
            row.get("crossing_status") == "right_censored" for row in thresholds
        ),
        "nonmonotonic_rows": sum(bool(row.get("nonmonotonic")) for row in thresholds),
        "realized_interval_unavailable_rows": sum(
            row.get("realized_interval_available") is not True
            for row in thresholds
        ),
        "frozen_attack_config_hash": frozen["content_hash"],
        "pgd_target_evidence": pgd_target_report,
    }
    atomic_write_json(
        task_dir / "threshold_summary.json", summary, overwrite=args.overwrite
    )
    publish_completion(
        task_dir,
        config_hash=str(task["config_key"]),
        source_hash_value=_cached_source_hash(args, repo_root),
        artifact_paths=[
            "manifest.json",
            "command.txt",
            "failure_thresholds.csv",
            "threshold_summary.json",
        ],
        extra={"array_index": int(task["array_index"])},
        overwrite=args.overwrite,
    )


def _attack_analyze_stage(
    args: argparse.Namespace,
    task_dir: Path,
    task: Mapping[str, Any],
    repo_root: Path,
) -> None:
    from .assemble_failure_boundary import assemble_failure_boundary
    from .evaluate_failure_boundary import (
        binary_auprc,
        binary_auroc,
        calibration_bins,
        cluster_bootstrap,
        family_budget_metrics,
        interval_censored_concordance,
        mean_over_probe_seeds,
        site_ranking_metrics,
        threshold_spearman,
    )

    current_source = _cached_source_hash(args, repo_root)
    frozen = _authenticated_frozen_attack_config(
        args, task, current_source_hash=current_source
    )
    for stage in ("clean", "test_probe", "test"):
        _verify_source_stage_grid(
            args, task, repo_root, stage=stage, partition="test"
        )
    root = (
        _phase_root(args)
        / str(task["dataset"])
        / f"R{int(task['R'])}"
        / "test"
    )
    clean_rows = _completed_rows(
        root / "clean", "clean_baseline.jsonl", expected_source_hash=current_source
    )
    probe_rows = _completed_rows(
        root / "test_probe", "probe_runs.jsonl", expected_source_hash=current_source
    )
    attack_rows = _completed_rows(
        root / "test", "attack_results.jsonl", expected_source_hash=current_source
    )
    bad_attack_rows = [
        row
        for row in attack_rows
        if row.get("record_type") not in {
            "sample",
            "attack_target",
            "shard_metadata",
        }
        or row.get("failure")
    ]
    if bad_attack_rows:
        raise ContractError("held-out attack evidence contains failed/unsupported rows")
    if _common_system_identity(
        [*clean_rows, *probe_rows, *attack_rows],
        where="held-out failure-boundary evidence",
    ) != frozen.get("system_identity"):
        raise ContractError(
            "held-out evidence differs from the frozen system identity"
        )
    _validate_pgd_target_evidence(
        attack_rows, where="held-out failure-boundary evidence"
    )
    assembly = assemble_failure_boundary(
        clean_rows,
        probe_rows,
        attack_rows,
        frozen_edges=[str(value) for value in frozen["edges"]],
        frozen_budgets=[float(value) for value in frozen["attack_epsilons"]],
        frozen_families=[str(value) for value in frozen["attack_families"]],
        requested_K=int(frozen["probe"]["K"]),
        selected_h=float(frozen["probe"]["h"]),
        probe_seeds=[int(value) for value in frozen["probe"]["seeds"]],
        probe_acceptance_thresholds=frozen["probe"]["acceptance_thresholds"],
    )
    frozen_probe_seeds = tuple(int(value) for value in frozen["probe"]["seeds"])
    units = [dict(row) for row in assembly["prediction_units"]]
    predictors = [dict(row) for row in assembly["edge_predictors"]]
    probe_exclusions = [dict(row) for row in assembly["probe_exclusions"]]
    seeds_by_edge: dict[tuple[str, str], set[int]] = {}
    for predictor in predictors:
        key = (
            str(predictor["raw_sample_id"]),
            str(predictor["edge_id"]),
        )
        seeds_by_edge.setdefault(key, set()).add(int(predictor["probe_seed"]))
    complete_seed_set = set(frozen_probe_seeds)
    seed_complete_edges = {
        key for key, observed_seeds in seeds_by_edge.items()
        if observed_seeds == complete_seed_set
    }
    for (raw_id, edge_id), observed_seeds in sorted(seeds_by_edge.items()):
        if (raw_id, edge_id) in seed_complete_edges:
            continue
        probe_exclusions.append(
            {
                "raw_sample_id": raw_id,
                "raw_id": raw_id,
                "edge_id": edge_id,
                "probe_seed": None,
                "reason": "incomplete_across_frozen_probe_seeds",
                "observed_probe_seeds": sorted(observed_seeds),
                "required_probe_seeds": list(frozen_probe_seeds),
            }
        )
    predictors = [
        row
        for row in predictors
        if (str(row["raw_sample_id"]), str(row["edge_id"]))
        in seed_complete_edges
    ]
    units = [
        row
        for row in units
        if (str(row["raw_sample_id"]), str(row["edge_id"]))
        in seed_complete_edges
    ]
    if not units or not predictors:
        raise ContractError(
            "held-out boundary assembly produced no probe-seed-complete "
            "evaluation units"
        )
    maximum_budget = max(float(value) for value in frozen["attack_epsilons"])
    finite_radii = [
        float(row["edge_radius"])
        for row in predictors
        if math.isfinite(float(row["edge_radius"]))
    ]
    radius_sentinel = 2.0 * max([maximum_budget, *finite_radii])
    finite_inverse_susceptibilities = [
        1.0 / float(row["maximum_susceptibility"])
        for row in predictors
        if float(row["maximum_susceptibility"]) > 0.0
    ]
    susceptibility_sentinel = 2.0 * max(
        [maximum_budget, *finite_inverse_susceptibilities]
    )
    for row in units:
        radius = float(row["edge_radius"])
        row["metric_edge_radius"] = (
            radius if math.isfinite(radius) else radius_sentinel
        )
    for row in predictors:
        radius = float(row["edge_radius"])
        row["metric_edge_radius"] = (
            radius if math.isfinite(radius) else radius_sentinel
        )

    _, threshold_paths = _authenticated_canonical_task_artifacts(
        args,
        task,
        repo_root,
        stage="thresholds",
        filenames=("failure_thresholds.csv", "threshold_summary.json"),
    )
    threshold_rows = [
        row
        for row in _read_csv_rows(threshold_paths["failure_thresholds.csv"])
        if row.get("curve_kind") == "edge_summary"
    ]
    predictors_by_edge: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for predictor in predictors:
        key = (
            str(predictor["raw_sample_id"]),
            str(predictor["edge_id"]),
        )
        predictors_by_edge.setdefault(key, []).append(predictor)
    for values in predictors_by_edge.values():
        values.sort(key=lambda value: int(value["probe_seed"]))
    threshold_eval: list[dict[str, Any]] = []
    threshold_exclusions: list[dict[str, Any]] = []
    all_thresholds: set[tuple[str, str, str]] = set()
    evaluated_thresholds: set[tuple[str, str, str, int]] = set()
    for row in threshold_rows:
        raw_id = str(row["raw_sample_id"])
        edge = str(row["edge_id"])
        family = str(row["attack_family"])
        key = (raw_id, edge, family)
        if key in all_thresholds:
            raise ContractError(f"duplicate edge threshold curve: {key}")
        all_thresholds.add(key)
        edge_predictors = predictors_by_edge.get((raw_id, edge), [])
        if not edge_predictors:
            # A complete held-out probe prefix can be present but rejected by
            # the frozen cast-quality rule.  Its attacks and threshold remain
            # authenticated, while the raw-edge unit is transparently excluded
            # from prediction metrics.
            continue
        realized_available = str(
            row.get("realized_interval_available", "")
        ).strip().lower() in {"1", "true", "yes"}
        if not realized_available:
            threshold_exclusions.append(
                {
                    "raw_sample_id": raw_id,
                    "edge_id": edge,
                    "attack_family": family,
                    "reason": "realized_threshold_interval_unavailable",
                    "realized_grid_status": row.get("realized_grid_status"),
                    "requested_interval_lower": row.get(
                        "requested_interval_lower"
                    ),
                    "requested_interval_upper": row.get(
                        "requested_interval_upper"
                    ),
                }
            )

        def optional_float(value: Any) -> float | None:
            return None if value in (None, "") else float(value)

        for predictor in edge_predictors:
            probe_seed = int(predictor["probe_seed"])
            evaluated_thresholds.add((*key, probe_seed))
            susceptibility = float(predictor["maximum_susceptibility"])
            common_threshold = {
                **row,
                "raw_id": raw_id,
                "probe_seed": probe_seed,
                "requested_threshold_lower": optional_float(
                    row.get("requested_interval_lower")
                ),
                "requested_threshold_upper": optional_float(
                    row.get("requested_interval_upper")
                ),
                "linkradius": float(predictor["metric_edge_radius"]),
                "margin_only": float(predictor["minimum_clean_margin"]),
                "susceptibility_only": (
                    1.0 / susceptibility
                    if susceptibility > 0.0
                    else susceptibility_sentinel
                ),
            }
            # Requested-grid metrics are a complete pre-registered sensitivity
            # analysis.  Actual post-cast norms are the manuscript's primary
            # coordinate and are included whenever the complete curve reports them.
            threshold_eval.append(
                {
                    **common_threshold,
                    "budget_coordinate": "requested_grid",
                    "threshold_lower": optional_float(
                        row.get("requested_interval_lower")
                    ),
                    "threshold_upper": optional_float(
                        row.get("requested_interval_upper")
                    ),
                }
            )
            if realized_available:
                threshold_eval.append(
                    {
                        **common_threshold,
                        "budget_coordinate": "realized_postcast",
                        "threshold_lower": optional_float(
                            row.get("realized_interval_lower")
                        ),
                        "threshold_upper": optional_float(
                            row.get("realized_interval_upper")
                        ),
                    }
                )
    expected_all_thresholds = {
        (raw_id, edge, str(family))
        for raw_id in assembly["eligible_raw_sample_ids"]
        for edge in frozen["edges"]
        for family in frozen["attack_families"]
    }
    if all_thresholds != expected_all_thresholds:
        missing = sorted(expected_all_thresholds - all_thresholds)
        extra = sorted(all_thresholds - expected_all_thresholds)
        raise ContractError(
            "held-out threshold cube is incomplete: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    expected_evaluated_thresholds = {
        (
            str(predictor["raw_sample_id"]),
            str(predictor["edge_id"]),
            str(family),
            int(predictor["probe_seed"]),
        )
        for predictor in predictors
        for family in frozen["attack_families"]
    }
    if evaluated_thresholds != expected_evaluated_thresholds:
        missing = sorted(expected_evaluated_thresholds - evaluated_thresholds)
        extra = sorted(evaluated_thresholds - expected_evaluated_thresholds)
        raise ContractError(
            f"threshold/predictor join is incomplete: missing={missing[:3]}, extra={extra[:3]}"
        )

    flip_metrics: list[dict[str, Any]] = []
    for probe_seed in frozen_probe_seeds:
        seed_units = [
            row for row in units if int(row["probe_seed"]) == probe_seed
        ]
        for metric in family_budget_metrics(
            seed_units, radius_field="metric_edge_radius"
        ):
            flip_metrics.append({"probe_seed": probe_seed, **metric})
    threshold_metrics: list[dict[str, Any]] = []
    frozen_edge_set = {str(value) for value in frozen["edges"]}

    def complete_site_rows(
        rows: Sequence[Mapping[str, Any]],
        *,
        require_all_probe_seeds: bool = False,
    ) -> list[Mapping[str, Any]]:
        observed: dict[tuple[str, int], set[str]] = {}
        for row in rows:
            observed.setdefault(
                (str(row["raw_id"]), int(row["probe_seed"])), set()
            ).add(
                str(row["edge_id"])
            )
        present_seeds = {seed for _, seed in observed}
        required_seeds = (
            set(frozen_probe_seeds)
            if require_all_probe_seeds
            else present_seeds
        )
        complete_ids = {
            raw_id
            for raw_id, _ in observed
            if all(
                observed.get((raw_id, seed)) == frozen_edge_set
                for seed in required_seeds
            )
        }
        return [row for row in rows if str(row["raw_id"]) in complete_ids]

    for probe_seed in frozen_probe_seeds:
        for budget_coordinate in ("realized_postcast", "requested_grid"):
            for family in frozen["attack_families"]:
                family_rows = [
                    row
                    for row in threshold_eval
                    if int(row["probe_seed"]) == probe_seed
                    and row["attack_family"] == family
                    and row["budget_coordinate"] == budget_coordinate
                ]
                family_site_rows = complete_site_rows(family_rows)
                for predictor_name in (
                    "linkradius",
                    "margin_only",
                    "susceptibility_only",
                ):
                    concordance = interval_censored_concordance(
                        family_rows,
                        prediction_field=predictor_name,
                    )
                    correlation = threshold_spearman(
                        family_rows,
                        prediction_field=predictor_name,
                    )
                    ranking = site_ranking_metrics(
                        family_site_rows,
                        prediction_field=predictor_name,
                    )
                    threshold_metrics.extend(
                        [
                            {
                                "probe_seed": probe_seed,
                                "budget_coordinate": budget_coordinate,
                                "attack_family": family,
                                "predictor": predictor_name,
                                "metric": "interval_censored_concordance",
                                "n": concordance["n"],
                                "support": concordance["comparable_pairs"],
                                "value": concordance["concordance"],
                            },
                            {
                                "probe_seed": probe_seed,
                                "budget_coordinate": budget_coordinate,
                                "attack_family": family,
                                "predictor": predictor_name,
                                "metric": "crossed_threshold_spearman",
                                "n": correlation["n"],
                                "support": correlation["n"],
                                "value": correlation["spearman"],
                            },
                            {
                                "probe_seed": probe_seed,
                                "budget_coordinate": budget_coordinate,
                                "attack_family": family,
                                "predictor": predictor_name,
                                "metric": "vulnerable_site_top1",
                                "n": ranking["groups"],
                                "support": ranking["top1_groups"],
                                "value": ranking["top1_accuracy"],
                            },
                            {
                                "probe_seed": probe_seed,
                                "budget_coordinate": budget_coordinate,
                                "attack_family": family,
                                "predictor": predictor_name,
                                "metric": "vulnerable_site_pair_concordance",
                                "n": ranking["groups"],
                                "support": ranking["comparable_site_pairs"],
                                "value": ranking["site_kendall"],
                            },
                        ]
                    )

    calibration: list[dict[str, Any]] = []
    for probe_seed in frozen_probe_seeds:
        for family in frozen["attack_families"]:
            family_units = [
                row
                for row in units
                if int(row["probe_seed"]) == probe_seed
                and row["attack_family"] == family
            ]
            calibration_scopes = [
                ("overall", family_units),
                *[
                    (
                        str(edge),
                        [row for row in family_units if row["edge_id"] == edge],
                    )
                    for edge in frozen["edges"]
                ],
            ]
            for edge_scope, scoped_units in calibration_scopes:
                if not scoped_units:
                    continue
                raw_count = len({str(row["raw_id"]) for row in scoped_units})
                for row in calibration_bins(
                    scoped_units,
                    score_field="linkradius_score",
                    num_bins=min(10, max(1, raw_count)),
                ):
                    calibration.append(
                        {
                            "probe_seed": probe_seed,
                            "attack_family": family,
                            "edge_id": edge_scope,
                            **row,
                        }
                    )

    paired_intervals: list[dict[str, Any]] = []
    for family in frozen["attack_families"]:
        for budget in frozen["attack_epsilons"]:
            stratum = [
                row
                for row in units
                if row["attack_family"] == family
                and float(row["requested_epsilon"]) == float(budget)
            ]
            labels = [bool(row["flipped"]) for row in stratum]
            if not labels or all(labels) or not any(labels):
                continue
            for baseline in ("margin_score", "susceptibility_score"):
                for metric_name, metric_fn in (
                    ("auroc", binary_auroc),
                    ("auprc", binary_auprc),
                ):

                    def paired_statistic(
                        sampled: Sequence[Mapping[str, Any]],
                        *,
                        _baseline: str = baseline,
                        _metric: Any = metric_fn,
                    ) -> float:
                        def seed_contrast(
                            seed_rows: Sequence[Mapping[str, Any]],
                        ) -> float:
                            sampled_labels = [
                                bool(row["flipped"]) for row in seed_rows
                            ]
                            return _metric(
                                sampled_labels,
                                [
                                    float(row["linkradius_score"])
                                    for row in seed_rows
                                ],
                            ) - _metric(
                                sampled_labels,
                                [float(row[_baseline]) for row in seed_rows],
                            )

                        return mean_over_probe_seeds(
                            sampled,
                            seed_contrast,
                            expected_seeds=frozen_probe_seeds,
                        )

                    try:
                        interval = cluster_bootstrap(
                            stratum,
                            paired_statistic,
                            cluster_field="raw_id",
                            repetitions=int(args.bootstrap_draws),
                            seed=int(task["seed"]),
                        )
                    except ContractError:
                        continue
                    paired_intervals.append(
                        {
                            "probe_seed_aggregation": "equal_mean_over_frozen_seeds",
                            "probe_seed_count": len(frozen_probe_seeds),
                            "budget_coordinate": "realized_postcast",
                            "attack_family": family,
                            "requested_epsilon": float(budget),
                            "metric": metric_name,
                            "contrast": f"linkradius_minus_{baseline}",
                            **interval,
                        }
                    )

    threshold_statistics = (
        (
            "interval_censored_concordance",
            lambda sampled, field: interval_censored_concordance(
                sampled, prediction_field=field
            )["concordance"],
        ),
        (
            "crossed_threshold_spearman",
            lambda sampled, field: threshold_spearman(
                sampled, prediction_field=field
            )["spearman"],
        ),
        (
            "vulnerable_site_top1",
            lambda sampled, field: site_ranking_metrics(
                sampled, prediction_field=field
            )["top1_accuracy"],
        ),
        (
            "vulnerable_site_pair_concordance",
            lambda sampled, field: site_ranking_metrics(
                sampled, prediction_field=field
            )["site_kendall"],
        ),
    )
    for budget_coordinate in ("realized_postcast", "requested_grid"):
        for family in frozen["attack_families"]:
            stratum = [
                row
                for row in threshold_eval
                if row["attack_family"] == family
                and row["budget_coordinate"] == budget_coordinate
            ]
            for baseline in ("margin_only", "susceptibility_only"):
                for metric_name, metric_fn in threshold_statistics:

                    def paired_threshold_statistic(
                        sampled: Sequence[Mapping[str, Any]],
                        *,
                        _baseline: str = baseline,
                        _metric: Any = metric_fn,
                    ) -> float:
                        return mean_over_probe_seeds(
                            sampled,
                            lambda seed_rows: float(
                                _metric(seed_rows, "linkradius")
                            )
                            - float(_metric(seed_rows, _baseline)),
                            expected_seeds=frozen_probe_seeds,
                        )

                    try:
                        bootstrap_stratum = (
                            complete_site_rows(
                                stratum, require_all_probe_seeds=True
                            )
                            if metric_name.startswith("vulnerable_site_")
                            else stratum
                        )
                        interval = cluster_bootstrap(
                            bootstrap_stratum,
                            paired_threshold_statistic,
                            cluster_field="raw_id",
                            repetitions=int(args.bootstrap_draws),
                            seed=int(task["seed"]),
                        )
                    except ContractError:
                        continue
                    paired_intervals.append(
                        {
                            "probe_seed_aggregation": "equal_mean_over_frozen_seeds",
                            "probe_seed_count": len(frozen_probe_seeds),
                            "budget_coordinate": budget_coordinate,
                            "attack_family": family,
                            "requested_epsilon": None,
                            "metric": metric_name,
                            "contrast": f"linkradius_minus_{baseline}",
                            **interval,
                        }
                    )

    _write_union_csv(
        task_dir / "prediction_units.csv", units, overwrite=args.overwrite
    )
    _write_union_csv(
        task_dir / "edge_predictors.csv", predictors, overwrite=args.overwrite
    )
    _write_union_csv(
        task_dir / "probe_exclusions.csv",
        probe_exclusions,
        overwrite=args.overwrite,
    )
    _write_union_csv(
        task_dir / "threshold_prediction_rows.csv",
        threshold_eval,
        overwrite=args.overwrite,
    )
    _write_union_csv(
        task_dir / "threshold_exclusions.csv",
        threshold_exclusions,
        overwrite=args.overwrite,
    )
    _write_union_csv(
        task_dir / "flip_prediction_metrics.csv",
        flip_metrics,
        overwrite=args.overwrite,
    )
    _write_union_csv(
        task_dir / "threshold_prediction_metrics.csv",
        threshold_metrics,
        overwrite=args.overwrite,
    )
    _write_union_csv(
        task_dir / "calibration_bins.csv", calibration, overwrite=args.overwrite
    )
    _write_union_csv(
        task_dir / "paired_bootstrap_intervals.csv",
        paired_intervals,
        overwrite=args.overwrite,
    )
    # A final gate must distinguish "files were produced" from "the central
    # RQ2 statistics are actually estimable".  Require the named metrics in
    # the proposal, meaningful (albeit pilot-scale) support, and paired CIs
    # against *both* components for every required metric.  The sign of those
    # contrasts is deliberately not gated: a valid negative result must pass.
    minimum_support = {
        "evaluated_raw_examples": 10,
        "flip_positive_raw_examples": 2,
        "flip_negative_raw_examples": 2,
        "interval_censored_concordance": 3,
        "crossed_threshold_spearman": 3,
        "vulnerable_site_top1": 3,
        "vulnerable_site_pair_concordance": 3,
    }

    def finite_interval(row: Mapping[str, Any]) -> bool:
        return all(
            math.isfinite(float(row.get(field, float("nan"))))
            for field in ("estimate", "ci_lower", "ci_upper")
        )

    required_baselines = ("margin", "susceptibility")
    pgd_flip_candidates: list[dict[str, Any]] = []
    for budget in (float(value) for value in frozen["attack_epsilons"]):
        point_rows = [
            row
            for row in flip_metrics
            if row.get("attack_family") == "pgd_autograd"
            and row.get("predictor") == "linkradius"
            and float(row.get("requested_epsilon")) == budget
        ]
        point_by_seed = {int(row["probe_seed"]): row for row in point_rows}
        all_seed_point_metrics_finite = bool(
            len(point_rows) == len(frozen_probe_seeds)
            and set(point_by_seed) == set(frozen_probe_seeds)
            and all(
                math.isfinite(float(row.get("auroc", float("nan"))))
                and math.isfinite(float(row.get("auprc", float("nan"))))
                for row in point_by_seed.values()
            )
        )
        if not all_seed_point_metrics_finite:
            continue
        budget_units = [
            row
            for row in units
            if row.get("attack_family") == "pgd_autograd"
            and float(row.get("requested_epsilon")) == budget
        ]
        per_seed_flip_support = {
            seed: {
                "raw_examples": len(
                    {
                        str(row["raw_id"])
                        for row in budget_units
                        if int(row["probe_seed"]) == seed
                    }
                ),
                "positive_raw_examples": len(
                    {
                        str(row["raw_id"])
                        for row in budget_units
                        if int(row["probe_seed"]) == seed
                        and bool(row["flipped"])
                    }
                ),
                "negative_raw_examples": len(
                    {
                        str(row["raw_id"])
                        for row in budget_units
                        if int(row["probe_seed"]) == seed
                        and not bool(row["flipped"])
                    }
                ),
            }
            for seed in frozen_probe_seeds
        }
        required_flip_cis = {
            (metric, baseline): any(
                row.get("budget_coordinate") == "realized_postcast"
                and row.get("attack_family") == "pgd_autograd"
                and row.get("metric") == metric
                and row.get("requested_epsilon") not in (None, "")
                and float(row.get("requested_epsilon")) == budget
                and row.get("contrast")
                == f"linkradius_minus_{baseline}_score"
                and finite_interval(row)
                for row in paired_intervals
            )
            for metric in ("auroc", "auprc")
            for baseline in required_baselines
        }
        if (
            all(
                support["raw_examples"]
                >= minimum_support["evaluated_raw_examples"]
                and support["positive_raw_examples"]
                >= minimum_support["flip_positive_raw_examples"]
                and support["negative_raw_examples"]
                >= minimum_support["flip_negative_raw_examples"]
                for support in per_seed_flip_support.values()
            )
            and all_seed_point_metrics_finite
            and all(required_flip_cis.values())
        ):
            pgd_flip_candidates.append(
                {
                    "requested_epsilon": budget,
                    "per_seed_support": per_seed_flip_support,
                    "finite_probe_seeds": sorted(point_by_seed),
                    "paired_ci_coverage": {
                        f"{metric}_vs_{baseline}": passed
                        for (metric, baseline), passed in required_flip_cis.items()
                    },
                }
            )

    required_threshold_metrics = (
        "interval_censored_concordance",
        "crossed_threshold_spearman",
        "vulnerable_site_top1",
        "vulnerable_site_pair_concordance",
    )
    pgd_threshold_support: dict[str, dict[str, Any]] = {}
    for metric in required_threshold_metrics:
        matches = [
            row
            for row in threshold_metrics
            if row.get("budget_coordinate") == "realized_postcast"
            and row.get("attack_family") == "pgd_autograd"
            and row.get("predictor") == "linkradius"
            and row.get("metric") == metric
        ]
        matches_by_seed = {int(row["probe_seed"]): row for row in matches}
        paired_coverage = {
            baseline: any(
                row.get("budget_coordinate") == "realized_postcast"
                and row.get("attack_family") == "pgd_autograd"
                and row.get("requested_epsilon") in (None, "")
                and row.get("metric") == metric
                and row.get("contrast") == f"linkradius_minus_{baseline}_only"
                and finite_interval(row)
                for row in paired_intervals
            )
            for baseline in required_baselines
        }
        per_seed = {
            seed: {
                "support": int(matches_by_seed.get(seed, {}).get("support", 0)),
                "value": float(
                    matches_by_seed.get(seed, {}).get("value", float("nan"))
                ),
            }
            for seed in frozen_probe_seeds
        }
        all_seeds_present = (
            len(matches) == len(frozen_probe_seeds)
            and set(matches_by_seed) == set(frozen_probe_seeds)
        )
        minimum_observed_support = min(
            (value["support"] for value in per_seed.values()),
            default=0,
        )
        every_seed_finite = bool(
            all_seeds_present
            and all(math.isfinite(value["value"]) for value in per_seed.values())
        )
        pgd_threshold_support[metric] = {
            "per_seed": per_seed,
            "minimum_observed_support": minimum_observed_support,
            "minimum_required": minimum_support[metric],
            "all_frozen_seeds_present": all_seeds_present,
            "every_seed_finite": every_seed_finite,
            "paired_ci_coverage": paired_coverage,
            "passed": bool(
                every_seed_finite
                and minimum_observed_support >= minimum_support[metric]
                and all(paired_coverage.values())
            ),
        }

    evaluated_raw_ids_by_seed = {
        seed: {
            str(row["raw_sample_id"])
            for row in predictors
            if int(row["probe_seed"]) == seed
        }
        for seed in frozen_probe_seeds
    }
    common_evaluated_raw_ids = set.intersection(
        *(set(values) for values in evaluated_raw_ids_by_seed.values())
    )
    evaluated_raw_examples = len(common_evaluated_raw_ids)
    scientific_support = {
        "minimum_support": minimum_support,
        "evaluated_raw_examples": evaluated_raw_examples,
        "evaluated_raw_examples_per_seed": {
            seed: len(values)
            for seed, values in evaluated_raw_ids_by_seed.items()
        },
        "pgd_fully_supported_flip_budgets": pgd_flip_candidates,
        "pgd_actual_threshold_metrics": pgd_threshold_support,
    }
    scientific_support["passed"] = bool(
        evaluated_raw_examples >= minimum_support["evaluated_raw_examples"]
        and pgd_flip_candidates
        and all(
            result["passed"] for result in pgd_threshold_support.values()
        )
    )
    result = {
        "schema_version": "linkradius.failure_boundary_analysis.v1",
        "frozen_attack_config_hash": frozen["content_hash"],
        "probe_seeds": list(frozen_probe_seeds),
        "probe_seed_metric_strategy": "per_seed",
        "probe_seed_bootstrap_strategy": (
            "raw_id_cluster_bootstrap_of_equal_seed_mean"
        ),
        "eligible_raw_sample_count": len(assembly["eligible_raw_sample_ids"]),
        "evaluated_raw_sample_count": evaluated_raw_examples,
        "any_seed_evaluated_raw_sample_count": len(
            assembly["evaluated_raw_sample_ids"]
        ),
        "excluded_raw_sample_count": len(assembly["excluded_raw_sample_ids"]),
        "probe_exclusion_rows": len(probe_exclusions),
        "prediction_units": len(units),
        "edge_predictors": len(predictors),
        "threshold_rows": len(threshold_eval),
        "threshold_exclusion_rows": len(threshold_exclusions),
        "flip_metric_rows": len(flip_metrics),
        "threshold_metric_rows": len(threshold_metrics),
        "paired_bootstrap_rows": len(paired_intervals),
        "scientific_support": scientific_support,
        "scientific_status": (
            "estimable" if scientific_support["passed"] else "underpowered"
        ),
    }
    atomic_write_json(
        task_dir / "analysis_result.json", result, overwrite=args.overwrite
    )
    artifacts = [
        "manifest.json",
        "command.txt",
        "analysis_result.json",
        "prediction_units.csv",
        "edge_predictors.csv",
        "probe_exclusions.csv",
        "threshold_prediction_rows.csv",
        "threshold_exclusions.csv",
        "flip_prediction_metrics.csv",
        "threshold_prediction_metrics.csv",
        "calibration_bins.csv",
        "paired_bootstrap_intervals.csv",
    ]
    publish_completion(
        task_dir,
        config_hash=str(task["config_key"]),
        source_hash_value=current_source,
        artifact_paths=artifacts,
        extra={"array_index": int(task["array_index"])},
        overwrite=args.overwrite,
    )


def _attack_validate_stage(
    args: argparse.Namespace,
    task_dir: Path,
    task: Mapping[str, Any],
    repo_root: Path,
) -> None:
    current_source = _cached_source_hash(args, repo_root)
    frozen = _authenticated_frozen_attack_config(
        args, task, current_source_hash=current_source
    )
    grid_reports = [
        _verify_source_stage_grid(
            args, task, repo_root, stage=stage, partition="test"
        )
        for stage in ("clean", "test_probe", "test")
    ]
    _, execution, execution_hash = _authenticated_execution_manifest(
        args, "test", task, repo_root
    )
    root = (
        _phase_root(args)
        / str(task["dataset"])
        / f"R{int(task['R'])}"
        / "test"
    )
    clean_rows = [
        row
        for row in _completed_rows(
            root / "clean",
            "clean_baseline.jsonl",
            expected_source_hash=current_source,
        )
        if row.get("record_type") == "sample"
    ]
    coverage = _clean_execution_coverage(clean_rows, execution)
    if not coverage["passed"]:
        raise ContractError(f"held-out clean execution coverage failed: {coverage}")
    if _common_system_identity(
        clean_rows, where="held-out clean validation"
    ) != frozen.get("system_identity"):
        raise ContractError(
            "held-out clean validation differs from the frozen system identity"
        )
    split, split_hash = _authenticated_split_manifest(args, task, repo_root)
    partition_sets = {
        name: {
            str(value["raw_sample_id"] if isinstance(value, Mapping) else value)
            for value in split["partitions"][name]
        }
        for name in ("attack_train", "validation", "test")
    }
    disjoint = not (
        partition_sets["attack_train"] & partition_sets["validation"]
        or partition_sets["attack_train"] & partition_sets["test"]
        or partition_sets["validation"] & partition_sets["test"]
    )
    if not disjoint:
        raise ContractError("split partitions overlap at final attack validation")
    _, threshold_paths = _authenticated_canonical_task_artifacts(
        args,
        task,
        repo_root,
        stage="thresholds",
        filenames=("failure_thresholds.csv", "threshold_summary.json"),
    )
    _, analysis_paths = _authenticated_canonical_task_artifacts(
        args,
        task,
        repo_root,
        stage="analyze",
        filenames=(
            "analysis_result.json",
            "prediction_units.csv",
            "edge_predictors.csv",
            "probe_exclusions.csv",
            "threshold_prediction_rows.csv",
            "threshold_exclusions.csv",
            "flip_prediction_metrics.csv",
            "threshold_prediction_metrics.csv",
            "calibration_bins.csv",
            "paired_bootstrap_intervals.csv",
        ),
    )
    threshold_summary = load_json(threshold_paths["threshold_summary.json"])
    analysis_result = load_json(analysis_paths["analysis_result.json"])
    artifacts_exist = all(
        path.is_file() for path in (*threshold_paths.values(), *analysis_paths.values())
    )
    essential_analysis_names = (
        "analysis_result.json",
        "prediction_units.csv",
        "edge_predictors.csv",
        "threshold_prediction_rows.csv",
        "flip_prediction_metrics.csv",
        "threshold_prediction_metrics.csv",
        "calibration_bins.csv",
    )
    essential_artifacts_nonempty = all(
        analysis_paths[name].stat().st_size > 0
        for name in essential_analysis_names
    )
    checks = [
        {
            "name": "frozen_attack_protocol_identity",
            "passed": frozen.get("source_hash") == current_source
            and frozen.get("split_manifest_hash") == split_hash
            and frozen.get("test_execution_manifest_hash") == execution_hash,
        },
        {"name": "split_partitions_disjoint", "passed": disjoint},
        coverage,
        *[
            {
                "name": f"exact_grid:{report['stage']}",
                **{key: value for key, value in report.items() if key != "stage"},
            }
            for report in grid_reports
        ],
        {
            "name": "threshold_artifacts_complete",
            "passed": int(threshold_summary.get("threshold_rows", 0)) > 0,
            "threshold_rows": threshold_summary.get("threshold_rows", 0),
        },
        {
            "name": "analysis_artifacts_complete",
            "passed": artifacts_exist
            and essential_artifacts_nonempty
            and int(analysis_result.get("prediction_units", 0)) > 0
            and int(analysis_result.get("edge_predictors", 0)) > 0
            and int(analysis_result.get("flip_metric_rows", 0)) > 0
            and int(analysis_result.get("threshold_metric_rows", 0)) > 0,
            "prediction_units": analysis_result.get("prediction_units", 0),
            "edge_predictors": analysis_result.get("edge_predictors", 0),
        },
        {
            "name": "primary_failure_boundary_metrics_estimable",
            "passed": analysis_result.get("scientific_status") == "estimable"
            and isinstance(analysis_result.get("scientific_support"), Mapping)
            and analysis_result["scientific_support"].get("passed") is True,
            "scientific_status": analysis_result.get("scientific_status"),
            "support": analysis_result.get("scientific_support"),
        },
    ]
    attack_freeze_gate = require_passed_gate(
        args.attack_freeze_gate, gate_type="attack_freeze_gate"
    )
    gate = make_gate(
        gate_type="attack_validation_gate",
        checks=checks,
        config_hash=str(task["config_key"]),
        source_hash=current_source,
        prerequisite_hashes={
            "attack_freeze_gate_hash": attack_freeze_gate["gate_content_hash"],
            "frozen_attack_config_hash": frozen["content_hash"],
            "split_manifest_hash": split_hash,
            "test_execution_manifest_hash": execution_hash,
        },
    )
    gate_path = Path(args.attack_validation_gate)
    atomic_write_json(gate_path, gate, overwrite=args.overwrite)
    if not gate["passed"]:
        raise ContractError("held-out attack validation gate failed")
    _record_global_pointer_completion(
        args,
        task_dir,
        task,
        repo_root,
        pointer_name="attack_validation_result.json",
        pointer={
            "passed": True,
            "attack_validation_gate": str(gate_path.resolve()),
            "gate_content_hash": gate["gate_content_hash"],
        },
    )


def execute_task(args: argparse.Namespace, task: Mapping[str, Any], task_dir: Path, repo_root: Path) -> None:
    stage = str(task["stage"])
    if stage == "split":
        _split_task(args, task_dir, task, repo_root)
    elif stage == "freeze_execution":
        _freeze_execution(args, task_dir, task, repo_root)
    elif stage in {"discover", "screen", "screen_clean", "clean"}:
        _capture_stage(args, task_dir, task, repo_root)
    elif stage in {"replay", "causal", "probe", "probe_calibration", "test_probe", "gradient"}:
        _replay_stage(args, task_dir, task, repo_root)
    elif stage == "attack" and args.workflow == "smoke":
        _attack_stage(args, task_dir, task, repo_root)
    elif stage in {"val", "test"} and args.workflow == "attacks":
        _attack_stage(args, task_dir, task, repo_root)
    elif stage == "freeze_attack" and args.workflow == "attacks":
        _freeze_attack_stage(args, task_dir, task, repo_root)
    elif stage == "thresholds" and args.workflow == "attacks":
        _attack_threshold_stage(args, task_dir, task, repo_root)
    elif stage == "analyze" and args.workflow == "attacks":
        _attack_analyze_stage(args, task_dir, task, repo_root)
    elif stage == "validate" and args.workflow == "attacks":
        _attack_validate_stage(args, task_dir, task, repo_root)
    elif stage == "validate" and args.workflow == "engineering":
        _engineering_validate_stage(args, task_dir, task, repo_root)
    elif stage == "validate" and args.workflow == "smoke":
        _smoke_validate_stage(args, task_dir, task, repo_root)
    elif stage == "estimate" and args.workflow == "smoke":
        _smoke_estimate_stage(args, task_dir, task, repo_root)
    elif stage == "aggregate" and args.workflow == "smoke":
        _smoke_aggregate_stage(args, task_dir, task, repo_root)
    elif stage == "aggregate" and args.workflow == "pilot":
        _pilot_aggregate_stage(args, task_dir, task, repo_root)
    elif stage == "freeze_probe" and args.workflow == "pilot":
        _freeze_probe_stage(args, task_dir, task, repo_root)
    elif stage in {"validate_probe", "validate"} and args.workflow == "pilot":
        _validate_probe_stage(args, task_dir, task, repo_root)
    elif stage == "verify" and args.workflow == "aggregate":
        _aggregate_verify_stage(args, task_dir, task, repo_root)
    elif stage in {"causal", "linkradius", "attacks", "metrics", "system_curves"} and args.workflow == "aggregate":
        _aggregate_cpu_stage(args, task_dir, task, repo_root)
    elif args.workflow == "expansion":
        raise ContractError(
            "unsupported_pending_gate: expansion model execution is disabled; "
            "no result was written"
        )
    else:
        raise ContractError(
            f"stage {args.workflow}/{stage} is CPU analysis or validation; invoke its dedicated module"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_grid_arguments(parser)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--style", default="sequential_light")
    parser.add_argument("--method", default="ours_recursive")
    parser.add_argument("--dataset-runtime", default="gpqa")
    parser.add_argument("--rounds-runtime", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--latent-length", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--planner-device", default="")
    parser.add_argument("--critic-device", default="")
    parser.add_argument("--solver-device", default="")
    parser.add_argument("--terminal-solver-device", default="")
    parser.add_argument(
        "--relay-transfer-mode",
        choices=("direct", "cpu_staged"),
        default="cpu_staged",
        help=(
            "relay copy path: cpu_staged uses differentiable GPU->CPU float32->GPU "
            "copies for distinct CUDA role devices"
        ),
    )
    parser.add_argument(
        "--autograd-memory-mode",
        choices=("none", "checkpoint"),
        default="none",
        help=(
            "memory policy for differentiable model forwards; checkpoint uses "
            "non-reentrant activation recomputation"
        ),
    )
    parser.add_argument("--trust-remote-code", type=int, choices=(0, 1), default=1)
    parser.add_argument("--round-label-mode", choices=("legacy", "actual"), default="legacy")
    parser.add_argument("--out-root", default="outputs/linkradius")
    parser.add_argument("--split-manifest", default="")
    parser.add_argument("--execution-manifest-path", default="")
    parser.add_argument("--execution-output", default="")
    parser.add_argument("--screening-jsonl", action="append", default=[])
    parser.add_argument("--trajectory", default="")
    parser.add_argument("--donor-trajectory", action="append", default=[])
    parser.add_argument("--donor-seed", type=int, default=42)
    parser.add_argument("--random-attack-seed-offset", type=int, default=1000000)
    parser.add_argument("--pgd-steps", type=int, default=5)
    parser.add_argument("--finite-difference-radii", type=float, nargs="+", default=[1e-3, 3e-3])
    parser.add_argument("--autograd-fd-relative-tolerance", type=float, default=0.25)
    parser.add_argument("--engineering-pgd-epsilon", type=float, default=3e-3)
    parser.add_argument("--legacy-equivalence", default="")
    parser.add_argument("--minimum-scorer-agreement", type=float, default=0.5)
    parser.add_argument("--minimum-probe-acceptance", type=float, default=0.5)
    parser.add_argument("--probe-threshold-lower-quantile", type=float, default=0.05)
    parser.add_argument("--probe-threshold-upper-quantile", type=float, default=0.95)
    parser.add_argument("--minimum-rank-stability", type=float, default=0.5)
    parser.add_argument("--minimum-binding-stability", type=float, default=0.5)
    parser.add_argument("--minimum-stability-comparisons", type=int, default=1)
    parser.add_argument("--minimum-autograd-agreement", type=float, default=0.5)
    parser.add_argument("--maximum-probe-autograd-relative-error", type=float, default=0.5)
    parser.add_argument("--identity-replay-tolerance", type=float, default=1e-6)
    parser.add_argument("--minimum-causal-pairs", type=int, default=1)
    parser.add_argument("--minimum-causal-accuracy-effect", type=float, default=0.0)
    parser.add_argument("--minimum-causal-margin-effect", type=float, default=0.0)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--aggregate-phase", default="smoke")
    parser.add_argument(
        "--verify-stages",
        default="split screen freeze_execution clean causal probe gradient attack estimate aggregate validate",
    )
    parser.add_argument("--fixed-edge", default="")
    parser.add_argument("--useful-edges", default="")
    parser.add_argument("--aggregate-verification-gate", default="")
    parser.add_argument("--batches-per-shard", type=int, default=1)
    parser.add_argument("--max-eligible", type=int, default=0)
    parser.add_argument("--retain-all-partition-rows", type=int, choices=(0, 1), default=1)
    parser.add_argument("--engineering-gate", default="outputs/linkradius/engineering_gate.json")
    parser.add_argument("--smoke-gate", default="outputs/linkradius/smoke_gate.json")
    parser.add_argument("--probe-gate", default="outputs/linkradius/probe_gate.json")
    parser.add_argument("--attack-freeze-gate", default="outputs/linkradius/attack_freeze_gate.json")
    parser.add_argument("--attack-validation-gate", default="outputs/linkradius/attack_validation_gate.json")
    parser.add_argument("--pilot-gate", default="outputs/linkradius/pilot_gate.json")
    parser.add_argument("--frozen-config", default="")
    parser.add_argument("--frozen-attack-config", default="")
    parser.add_argument("--tuning-override", action="append", default=[])
    parser.add_argument("--reuse-complete", type=int, choices=(0, 1), default=1)
    parser.add_argument("--overwrite", type=int, choices=(0, 1), default=0)
    parser.add_argument("--print-output-dir", action="store_true")
    parser.add_argument("--defer-completion", action="store_true")
    parser.add_argument("--grid-format", choices=("tsv", "json", "count"), default="tsv")
    parser.add_argument("--grid-target-stage", default="")
    return parser


def _argv_without_control_flags(argv: Sequence[str]) -> list[str]:
    result: list[str] = []
    skip_next = False
    controls = {"--stage", "--task-id", "--grid-format"}
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in controls:
            skip_next = True
            continue
        if token == "--print-output-dir":
            continue
        result.append(token)
    return result


def _run_child_with_authenticated_logs(
    command: Sequence[str],
    *,
    overwrite: bool,
) -> int:
    """Run an ``all`` child and finalize the same artifacts as a shell task."""

    child = list(command)
    if "--defer-completion" not in child:
        child.append("--defer-completion")
    preflight = subprocess.run(
        [*child, "--print-output-dir"],
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if preflight.returncode != 0:
        sys.stdout.write(preflight.stdout)
        return preflight.returncode
    task_dir_text = preflight.stdout.strip()
    if not task_dir_text:
        raise ContractError("child output-directory preflight returned an empty path")
    task_dir = Path(task_dir_text)
    task_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        task_dir / ".launcher_command.pending.txt",
        " ".join(shlex.quote(value) for value in child) + "\n",
        overwrite=True,
    )
    completed = subprocess.run(
        child,
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    sys.stdout.write(completed.stdout)
    atomic_write_text(
        task_dir / ".run.log.pending", completed.stdout, overwrite=True
    )
    if completed.returncode != 0:
        return completed.returncode
    _finalize_deferred_completion(task_dir)
    return 0


def _run_local_all(args: argparse.Namespace, original_argv: Sequence[str]) -> int:
    if args.workflow not in {"engineering", "smoke"}:
        raise ContractError("LR_STAGE=all is restricted to the one-example engineering and tiny smoke workflows")
    stage_order = (
        ("split", "discover", "freeze_execution", "clean", "replay", "probe", "gradient", "validate")
        if args.workflow == "engineering"
        else ("split", "screen", "freeze_execution", "clean", "causal", "probe", "gradient", "attack", "estimate", "aggregate", "validate")
    )
    base = _argv_without_control_flags(original_argv)
    for stage in stage_order:
        stage_args = argparse.Namespace(**vars(args))
        stage_args.stage = stage
        count = len(build_grid(_build_grid_config(stage_args, stage)))
        for task_id in range(count):
            command = [
                sys.executable,
                "-m",
                "experiments.linkradius.run_linkradius",
                *base,
                "--stage",
                stage,
                "--task-id",
                str(task_id),
            ]
            return_code = _run_child_with_authenticated_logs(
                command, overwrite=bool(args.overwrite)
            )
            if return_code != 0:
                return return_code
    return 0


def _run_aggregate_all(args: argparse.Namespace, original_argv: Sequence[str]) -> int:
    base = _argv_without_control_flags(original_argv)
    for stage in ("verify", "causal", "linkradius", "attacks", "metrics", "system_curves"):
        command = [
            sys.executable,
            "-m",
            "experiments.linkradius.run_linkradius",
            *base,
            "--stage",
            stage,
            "--task-id",
            "0",
        ]
        return_code = _run_child_with_authenticated_logs(
            command, overwrite=bool(args.overwrite)
        )
        if return_code != 0:
            return return_code
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    global _DEFER_COMPLETION
    original_argv = list(sys.argv[1:] if argv is None else argv)
    if original_argv[:1] == ["--finalize-completion-dir"]:
        if len(original_argv) != 2:
            raise ContractError("--finalize-completion-dir requires exactly one task directory")
        print(
            json.dumps(
                _finalize_deferred_completion(original_argv[1]), sort_keys=True
            )
        )
        return 0
    args = build_parser().parse_args(original_argv)
    _DEFER_COMPLETION = bool(args.defer_completion)
    args.overwrite = bool(args.overwrite)
    args.reuse_complete = bool(args.reuse_complete)
    args.retain_all_partition_rows = bool(args.retain_all_partition_rows)
    args.dataset_runtime = args.datasets.split()[0]
    if args.workflow == "attacks":
        attack_datasets = str(args.datasets).split()
        attack_rounds = [int(value) for value in str(args.rounds).split()]
        attack_seeds = [int(value) for value in str(args.seeds).split()]
        if (
            attack_datasets != ["gpqa"]
            or attack_rounds != [2]
            or len(attack_seeds) != 1
        ):
            raise ContractError(
                "failure-boundary v1 requires exactly DATASETS=gpqa, "
                "ROUNDS=2, and one seed"
            )
        if str(args.execution_manifest_path).strip():
            raise ContractError(
                "attacks workflow rejects the unqualified EXECUTION_MANIFEST "
                "override; use the canonical per-partition manifests"
            )
        if int(args.batches_per_shard) != 1:
            raise ContractError(
                "failure-boundary v1 requires BATCHES_PER_SHARD=1"
            )
    if args.workflow == "pilot" and "test" in args.partitions.split():
        raise ContractError("Phase-3 pilot stages categorically reject the test partition")
    rounds = [int(value) for value in args.rounds.split()]
    args.rounds_runtime = rounds[0]
    args.seed = int(args.seeds.split()[0])
    if args.stage == "all":
        if args.print_output_dir:
            print(_phase_root(args) / "local_all")
            return 0
        if args.workflow == "aggregate":
            return _run_aggregate_all(args, original_argv)
        return _run_local_all(args, original_argv)
    config = _build_grid_config(args)
    tasks = build_grid(config)
    if args.stage == "grid" or args.stage in GRID_ALIAS:
        grid_stage = (
            args.grid_target_stage or GRID_DEFAULT_STAGE[args.workflow]
            if args.stage == "grid"
            else GRID_ALIAS[args.stage]
        )
        if args.workflow == "attacks" and grid_stage in {
            "clean",
            "test_probe",
            "test",
        }:
            enforce_prerequisites(args, grid_stage)
        if args.grid_format == "count":
            print(len(tasks))
        elif args.grid_format == "json":
            print(json.dumps([task.as_dict() for task in tasks], sort_keys=True))
        else:
            print(f"total_tasks\t{len(tasks)}")
            print(f"max_array_index\t{len(tasks) - 1}")
            print(grid_tsv(tasks))
        return 0
    task = select_task(tasks, args.task_id).as_dict()
    args.rounds_runtime = int(task["R"])
    args.seed = int(task["seed"])
    args.dataset_runtime = str(task["dataset"])
    task_dir = task_output_dir(args, task)
    enforce_prerequisites(args, str(task["stage"]), task)
    if args.print_output_dir:
        print(task_dir)
        return 0
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "RecursiveMAS" / "run.py").is_file():
        raise ContractError(f"repository root validation failed: {repo_root}")
    task_dir.mkdir(parents=True, exist_ok=True)
    if (task_dir / ".complete.json").exists():
        if not args.overwrite and args.reuse_complete and compatible_complete(
            task_dir,
            expected_config_hash=task["config_key"],
            expected_source_hash=_cached_source_hash(args, repo_root),
        ):
            print(json.dumps({"status": "reused_complete", "task_dir": str(task_dir)}, sort_keys=True))
            return 0
        if not args.overwrite:
            raise ContractError("incompatible/existing completed output; set OVERWRITE=1 or use a new OUT_ROOT")
    if args.defer_completion:
        atomic_write_text(task_dir / "warnings.txt", "", overwrite=args.overwrite)
    manifest = _task_manifest(args, task, repo_root)
    atomic_write_json(task_dir / "manifest.json", manifest, overwrite=args.overwrite)
    command = " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv]) + "\n"
    atomic_write_text(task_dir / "command.txt", command, overwrite=args.overwrite)
    execute_task(args, task, task_dir, repo_root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[linkradius:error] {exc}", file=sys.stderr)
        raise SystemExit(2)
