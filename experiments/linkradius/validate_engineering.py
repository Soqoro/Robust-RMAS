#!/usr/bin/env python3
"""Validate every mandatory one-example engineering invariant."""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io_utils import atomic_write_json, content_hash, load_json, source_hash
from .schemas import ContractError, validate_sha256
from .validate_stage import make_gate


MANDATORY_CHECKS = (
    "edge_counts",
    "captured_all_r2_edges",
    "identity_replay_scores",
    "zero_additive_scores",
    "repeated_scoring_deterministic",
    "finite_scores_and_margins",
    "strict_and_scored_predictions_recorded",
    "direction_identity_invariant",
    "antithetic_cast_survival",
    "invalid_terminal_s2p_rejected_preload",
    "early_replay_descendants_only",
    "terminal_autograd_finite_difference_agreement",
    "pgd_budget_and_objective",
    "legacy_release_equivalence",
    "legacy_latent_contagion_regression",
)


EXPECTED_R2_EDGES = frozenset({"p2c@0", "c2s@0", "s2p@0", "p2c@1", "c2s@1"})
ENGINEERING_PROBE_RADII = frozenset({1e-3, 3e-3})
ENGINEERING_PROBE_K = 8
ENGINEERING_AUTOGRAD_FD_RELATIVE_TOLERANCE = 0.25
ENGINEERING_PROBE_THRESHOLDS = {
    "minimum_requested_realized_cosine": -1.0,
    "maximum_off_direction_relative": 1e300,
    "minimum_signed_separation": 0.0,
    "minimum_antipodality": -1.0,
}
ENGINEERING_ARTIFACT_NAMES = frozenset(
    {
        "clean_trajectory.pt",
        "clean_baseline.jsonl",
        "replay_runs.jsonl",
        "probe_runs.jsonl",
        "gradient_runs.jsonl",
    }
)


@dataclass(frozen=True)
class _AuthenticatedArtifact:
    path: Path
    completion: Mapping[str, Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _authenticated_artifacts(
    root: Path,
    name: str,
    *,
    expected_source_hash: str,
) -> list[_AuthenticatedArtifact]:
    """Return every named artifact, rejecting anything not currently authenticated."""

    from .io_utils import verify_completion

    if name not in ENGINEERING_ARTIFACT_NAMES:
        raise ContractError(f"unsupported engineering artifact inventory: {name}")
    authenticated: dict[Path, _AuthenticatedArtifact] = {}
    config_paths: dict[str, Path] = {}
    for completion_path in sorted(root.rglob(".complete.json")):
        try:
            unverified = load_json(completion_path)
        except (OSError, TypeError, ValueError) as exc:
            raise ContractError(f"invalid engineering completion record: {completion_path}") from exc
        if not isinstance(unverified, Mapping):
            raise ContractError(f"invalid engineering completion record: {completion_path}")
        raw_artifacts = unverified.get("artifacts")
        if not isinstance(raw_artifacts, Sequence) or isinstance(raw_artifacts, (str, bytes)):
            raise ContractError(f"invalid engineering completion artifact list: {completion_path}")
        declared_names = [
            artifact.get("path")
            for artifact in raw_artifacts
            if isinstance(artifact, Mapping)
        ]
        if name not in declared_names:
            continue
        try:
            completion = verify_completion(completion_path.parent)
        except (OSError, TypeError, ValueError, ContractError) as exc:
            raise ContractError(
                f"engineering artifact completion is invalid: {completion_path}: {exc}"
            ) from exc
        if completion.get("source_hash") != expected_source_hash:
            raise ContractError(f"stale-source engineering artifact: {completion_path.parent / name}")
        declared = [
            artifact
            for artifact in completion.get("artifacts", ())
            if isinstance(artifact, Mapping) and artifact.get("path") == name
        ]
        if len(declared) != 1:
            raise ContractError(
                f"engineering artifact is not uniquely declared by completion: "
                f"{completion_path.parent / name}"
            )
        path = completion_path.parent / name
        config_hash = str(completion.get("config_hash", ""))
        if config_hash in config_paths:
            raise ContractError(
                f"duplicate completed {name} configuration at {config_paths[config_hash]} "
                f"and {path}"
            )
        config_paths[config_hash] = path
        authenticated[path] = _AuthenticatedArtifact(path=path, completion=completion)

    for path in sorted(root.rglob(name)):
        if path not in authenticated:
            raise ContractError(f"orphan engineering artifact has no declaring completion: {path}")
    return [authenticated[path] for path in sorted(authenticated)]


def _authenticated_jsonl_rows(
    artifacts: Sequence[_AuthenticatedArtifact],
    *,
    expected_source_hash: str,
    allowed_record_types: set[str],
    expected_stage: str,
) -> list[dict[str, Any]]:
    from .io_utils import load_jsonl

    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        task = _authenticated_task(
            artifact,
            expected_source_hash=expected_source_hash,
            expected_stage=expected_stage,
        )
        loaded = load_jsonl(artifact.path)
        metadata = [row for row in loaded if row.get("record_type") == "shard_metadata"]
        real = [row for row in loaded if row.get("record_type") != "shard_metadata"]
        if len(metadata) != 1:
            raise ContractError(f"{artifact.path} must contain exactly one shard_metadata row")
        shard = metadata[0]
        if isinstance(shard.get("row_count"), bool) or shard.get("row_count") != len(real):
            raise ContractError(f"{artifact.path} shard row_count is stale")
        if shard.get("config_key") != artifact.completion.get("config_hash"):
            raise ContractError(f"{artifact.path} shard/completion config hashes differ")
        if (
            "array_index" in artifact.completion
            and shard.get("array_index") != artifact.completion.get("array_index")
        ):
            raise ContractError(f"{artifact.path} shard/completion array indices differ")
        if not real:
            raise ContractError(f"{artifact.path} contains no real evidence rows")
        for row in real:
            record_type = row.get("record_type")
            if record_type not in allowed_record_types:
                raise ContractError(
                    f"{artifact.path} contains unexpected record_type {record_type!r}"
                )
            if "config_hash" in row and row["config_hash"] != artifact.completion.get(
                "config_hash"
            ):
                raise ContractError(f"{artifact.path} row/completion config hashes differ")
            if "source_hash" in row and row["source_hash"] != expected_source_hash:
                raise ContractError(f"{artifact.path} contains a stale-source row")
            for field in (
                "dataset",
                "partition",
                "style",
                "method",
                "R",
                "edge_id",
                "h",
                "probe_seed",
            ):
                if field in row and task.get(field) != row[field]:
                    raise ContractError(f"{artifact.path} row/task {field} values differ")
        rows.extend(real)
    return rows


def _authenticated_task(
    artifact: _AuthenticatedArtifact,
    *,
    expected_source_hash: str,
    expected_stage: str,
) -> Mapping[str, Any]:
    declared_manifests = [
        item
        for item in artifact.completion.get("artifacts", ())
        if isinstance(item, Mapping) and item.get("path") == "manifest.json"
    ]
    if len(declared_manifests) != 1:
        raise ContractError(f"{artifact.path} completion must authenticate manifest.json")
    manifest_path = artifact.path.parent / "manifest.json"
    manifest = load_json(manifest_path)
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != "linkradius.task_manifest.v1":
        raise ContractError(f"{artifact.path} has an invalid task manifest")
    if manifest.get("source_hash") != expected_source_hash:
        raise ContractError(f"{artifact.path} task manifest is stale")
    task = manifest.get("task")
    if not isinstance(task, Mapping):
        raise ContractError(f"{artifact.path} task manifest has no task object")
    if task.get("workflow") != "engineering" or task.get("stage") != expected_stage:
        raise ContractError(f"{artifact.path} is not an engineering/{expected_stage} task")
    if task.get("config_key") != artifact.completion.get("config_hash"):
        raise ContractError(f"{artifact.path} task/completion config hashes differ")
    for field in ("array_index", "execution_batch_id"):
        if field in artifact.completion and task.get(field) != artifact.completion[field]:
            raise ContractError(f"{artifact.path} task/completion {field} values differ")
    return task


def _require_unique_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    key_fields: Sequence[str],
    where: str,
) -> None:
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        if any(value is None or value == "" for value in key):
            raise ContractError(f"{where} row is missing unique-key fields {tuple(key_fields)!r}")
        if key in seen:
            raise ContractError(f"duplicate {where} row for key {key!r}")
        seen.add(key)


def _shared_trajectory_provenance(trajectory: Any) -> dict[str, Any]:
    provenance = getattr(trajectory, "provenance", None)
    if not isinstance(provenance, Mapping):
        raise ContractError("clean trajectory provenance is missing")
    shared = {
        "split_manifest_hash": provenance.get("split_manifest_hash"),
        "execution_manifest_hash": getattr(trajectory, "execution_manifest_hash", None),
        "ordered_cohort_hash": provenance.get("global_ordered_cohort_hash")
        or getattr(trajectory, "ordered_cohort_hash", None),
        "batch_boundary_hash": provenance.get("global_batch_boundary_hash")
        or getattr(trajectory, "batch_boundary_hash", None),
        "model_hash": provenance.get("model_hash"),
        "scorer_hash": provenance.get("scorer_hash"),
    }
    for field, value in shared.items():
        validate_sha256(value, field=f"trajectory.{field}")
    return shared


def _bind_rows_to_trajectory(
    rows: Sequence[Mapping[str, Any]],
    *,
    trajectory: Any,
    shared: Mapping[str, Any],
    expected_source_hash: str,
) -> None:
    sample_id = str(trajectory.sample_ids[0])
    raw_sample_id = str(trajectory.raw_sample_ids[0])
    raw_index = int(trajectory.raw_indices[0])
    for row in rows:
        if row.get("sample_id") != sample_id or row.get("raw_sample_id") != raw_sample_id:
            raise ContractError("engineering evidence mixes samples")
        if "raw_index" in row and row["raw_index"] != raw_index:
            raise ContractError("engineering evidence raw_index differs from clean trajectory")
        if "source_hash" in row and row["source_hash"] != expected_source_hash:
            raise ContractError("engineering evidence mixes source revisions")
        if "R" in row and row["R"] != trajectory.rounds:
            raise ContractError("engineering evidence mixes horizons")
        if "analysis_eligible" in row and row["analysis_eligible"] is not True:
            raise ContractError("engineering evidence row is not analysis eligible")
        for field, expected in shared.items():
            if field in row and row[field] != expected:
                raise ContractError(f"engineering evidence mixes {field}")


def _bind_task_to_trajectory(
    task: Mapping[str, Any],
    *,
    trajectory: Any,
    shared: Mapping[str, Any],
    expected_latent_steps: int,
) -> None:
    expected = {
        "R": 2,
        "batch_size": 1,
        "latent_length": expected_latent_steps,
        "execution_manifest_hash": shared["execution_manifest_hash"],
    }
    for field, value in expected.items():
        if task.get(field) != value:
            raise ContractError(f"engineering task {field} differs from the clean trajectory")
    batch_ids = list(getattr(trajectory, "ordered_batch_ids", ()))
    if len(batch_ids) != 1 or task.get("execution_batch_id") != batch_ids[0]:
        raise ContractError("engineering task uses a different execution batch")
    runtime = trajectory.provenance.get("runtime_config", {})
    for field in ("dataset", "style"):
        if field in runtime and task.get(field) != runtime[field]:
            raise ContractError(f"engineering task {field} differs from runtime provenance")


def _score_signature(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        field: row.get(field)
        for field in (
            "option_scores",
            "scorer_prediction",
            "scorer_correct",
            "score_tie",
            "margins",
            "minimum_margin",
            "binding_competitor",
        )
    }


def _legacy_latent_contagion_regression_check() -> dict[str, Any]:
    """Exercise legacy defaults, perturbation behavior, and emitted schema literals safely."""

    try:
        import torch

        from RecursiveMAS.inference_utils import inference_mas
        from RecursiveMAS.inference_utils.latent_contagion import (
            PerturbConfig,
            maybe_perturb,
            stable_seed,
        )

        recursive_root = str(_repo_root() / "RecursiveMAS")
        inserted = recursive_root not in sys.path
        if inserted:
            sys.path.insert(0, recursive_root)
        try:
            release_run = importlib.import_module("RecursiveMAS.run")
        finally:
            if inserted:
                sys.path.remove(recursive_root)
        parser = release_run.build_parser()
        parsed = parser.parse_args(["--style", "sequential_light", "--dataset", "gpqa"])
        expected_defaults = {
            "lc_mode": "none",
            "lc_site": "",
            "lc_epsilon": 0.0,
            "lc_round": 0,
            "lc_seed": 42,
            "lc_direction": "random",
            "lc_steering_bank": "",
            "lc_steering_method": "",
            "lc_steering_id": "",
            "lc_trace_path": "",
            "lc_trace_sites": "p2c,c2s,s2p",
            "lc_trace_rounds": "0",
            "lc_trace_dtype": "float16",
        }
        defaults_ok = all(getattr(parsed, key, None) == value for key, value in expected_defaults.items())
        actions = {action.dest: action for action in parser._actions}
        choices_ok = (
            set(actions["lc_mode"].choices or ()) == {"none", "one_shot", "persistent"}
            and set(actions["lc_site"].choices or ()) == {"", "p2c", "c2s", "s2p"}
            and set(actions["lc_direction"].choices or ()) == {"random", "bank"}
        )

        defaults = PerturbConfig()
        config_defaults_ok = (
            defaults.mode == "none"
            and defaults.site == ""
            and defaults.epsilon == 0.0
            and defaults.round_idx == 0
            and defaults.seed == 42
            and defaults.enabled is False
            and defaults.direction == "random"
            and defaults.steering_bank is None
        )
        relay = torch.arange(1, 13, dtype=torch.float32).reshape(1, 2, 6)
        unchanged, disabled_metadata = maybe_perturb(relay, defaults, "p2c", 0, 0)
        disabled_ok = unchanged is relay and disabled_metadata is None
        enabled = PerturbConfig(
            mode="one_shot",
            site="p2c",
            epsilon=0.1,
            round_idx=0,
            seed=42,
            enabled=True,
            direction="random",
        )
        perturbed_a, metadata_a = maybe_perturb(relay, enabled, "p2c", 0, 0)
        perturbed_b, metadata_b = maybe_perturb(relay, enabled, "p2c", 0, 0)
        skipped, skipped_metadata = maybe_perturb(relay, enabled, "p2c", 1, 0)
        metadata_fields = {
            "mode",
            "direction",
            "steering_method",
            "steering_id",
            "applied",
            "site",
            "round_idx",
            "epsilon",
            "x_norm_mean",
            "delta_norm_mean",
            "relative_delta_norm_mean",
            "batch_start",
        }
        perturbation_ok = (
            torch.equal(perturbed_a, perturbed_b)
            and not torch.equal(perturbed_a, relay)
            and metadata_a == metadata_b
            and isinstance(metadata_a, Mapping)
            and set(metadata_a) == metadata_fields
            and metadata_a.get("applied") is True
            and math.isclose(
                float(metadata_a.get("relative_delta_norm_mean", math.nan)),
                0.1,
                rel_tol=1e-5,
                abs_tol=1e-5,
            )
            and skipped is relay
            and skipped_metadata is None
            and stable_seed(42, "p2c", 0, 0) == stable_seed(42, "p2c", 0, 0)
            and stable_seed(42, "p2c", 0, 0) != stable_seed(42, "c2s", 0, 0)
        )

        main_tree = ast.parse(inspect.getsource(inference_mas.main))
        emitted_literals = {
            node.value
            for node in ast.walk(main_tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        required_jsonl_fields = {
            "sample_id",
            "lc_mode",
            "lc_site",
            "lc_epsilon",
            "lc_round",
            "lc_seed",
            "lc_direction",
            "lc_steering_method",
            "lc_steering_id",
            "lc_enabled",
            "attack_config_sha256",
            "attack_config",
            "generation_config_sha256",
            "evaluation_config_sha256",
            "answer_invalid",
            "type",
            "summary",
        }
        schema_ok = required_jsonl_fields <= emitted_literals
    except Exception as exc:
        return {"passed": False, "detail": f"legacy regression check failed: {exc}"}
    return {
        "passed": defaults_ok
        and choices_ok
        and config_defaults_ok
        and disabled_ok
        and perturbation_ok
        and schema_ok,
        "detail": {
            "parser_defaults": defaults_ok,
            "parser_choices": choices_ok,
            "perturb_config_defaults": config_defaults_ok,
            "disabled_identity": disabled_ok,
            "deterministic_perturbation_schema": perturbation_ok,
            "jsonl_schema_literals": schema_ok,
            "required_jsonl_fields": sorted(required_jsonl_fields),
        },
    }


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ContractError(f"{field} must be finite")
    return parsed


def _validate_terminal_finite_difference(
    terminal: Mapping[str, Any] | None,
    *,
    trajectory: Any,
) -> dict[str, Any]:
    if terminal is None:
        return {"passed": False, "detail": "missing unique terminal gradient row"}
    if terminal.get("autograd_semantics") != "continuous_consumer_input":
        return {
            "passed": False,
            "detail": "terminal autograd is not labeled continuous_consumer_input",
        }
    gradient_norm = _finite_float(terminal.get("gradient_norm"), field="gradient_norm")
    if gradient_norm <= 0:
        return {"passed": False, "detail": "terminal gradient norm is not positive"}
    fd = terminal.get("finite_difference")
    if not isinstance(fd, Mapping):
        return {"passed": False, "detail": "missing terminal finite-difference evidence"}
    plus = fd.get("plus_diagnostics")
    minus = fd.get("minus_diagnostics")
    if not isinstance(plus, Mapping) or not isinstance(minus, Mapping):
        raise ContractError("finite difference is missing realized cast diagnostics")
    if plus.get("collapsed") is not False or minus.get("collapsed") is not False:
        raise ContractError("finite-difference offsets did not both survive consumer casting")
    expected_dtype = trajectory.dtype_metadata("c2s@1").consumer_dtype.replace("torch.", "")
    if plus.get("consumer_dtype") != expected_dtype or minus.get("consumer_dtype") != expected_dtype:
        raise ContractError("finite-difference diagnostics use a different consumer dtype")

    h = _finite_float(fd.get("h"), field="finite_difference.h")
    if h <= 0:
        raise ContractError("finite_difference.h must be positive")
    plus_requested = _finite_float(
        plus.get("requested_signed_coordinate"),
        field="finite_difference.plus.requested_signed_coordinate",
    )
    minus_requested = _finite_float(
        minus.get("requested_signed_coordinate"),
        field="finite_difference.minus.requested_signed_coordinate",
    )
    plus_realized = _finite_float(
        plus.get("realized_signed_coordinate"),
        field="finite_difference.plus.realized_signed_coordinate",
    )
    minus_realized = _finite_float(
        minus.get("realized_signed_coordinate"),
        field="finite_difference.minus.realized_signed_coordinate",
    )
    if not math.isclose(plus_requested, h, rel_tol=1e-6, abs_tol=1e-9) or not math.isclose(
        minus_requested, -h, rel_tol=1e-6, abs_tol=1e-9
    ):
        raise ContractError("finite-difference requested signed coordinates differ from +/-h")
    if plus_realized <= 0 or minus_realized >= 0:
        raise ContractError("finite-difference realized offsets are not oppositely signed")
    separation = plus_realized - minus_realized
    recorded_separation = _finite_float(
        fd.get("realized_separation"), field="finite_difference.realized_separation"
    )
    if not math.isclose(
        recorded_separation, separation, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ContractError("finite-difference realized separation is stale")
    for sign_name, diagnostics in (("plus", plus), ("minus", minus)):
        realized_norm = _finite_float(
            diagnostics.get("realized_delta_norm"),
            field=f"finite_difference.{sign_name}.realized_delta_norm",
        )
        cosine = _finite_float(
            diagnostics.get("requested_realized_cosine"),
            field=f"finite_difference.{sign_name}.requested_realized_cosine",
        )
        if realized_norm <= 0 or not -1.0 <= cosine <= 1.0:
            raise ContractError("finite-difference cast diagnostics are invalid")

    derivative = _finite_float(
        fd.get("finite_difference_derivative"),
        field="finite_difference.finite_difference_derivative",
    )
    autograd = _finite_float(
        fd.get("autograd_dimensionless_derivative"),
        field="finite_difference.autograd_dimensionless_derivative",
    )
    expected_relative_error = abs(derivative - autograd) / max(
        abs(derivative), abs(autograd), 1e-12
    )
    recorded_relative_error = _finite_float(
        fd.get("relative_error"), field="finite_difference.relative_error"
    )
    if not math.isclose(
        recorded_relative_error,
        expected_relative_error,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ContractError("finite-difference relative error is not reproducible")
    expected_agreement = (
        expected_relative_error <= ENGINEERING_AUTOGRAD_FD_RELATIVE_TOLERANCE
    )
    if fd.get("agrees") is not expected_agreement:
        raise ContractError("finite-difference agreement flag is stale")
    if fd.get("target") != terminal.get("target_label"):
        raise ContractError("finite-difference target differs from the gradient target")
    return {
        "passed": expected_agreement,
        "detail": {
            **dict(fd),
            "recomputed_realized_separation": separation,
            "recomputed_relative_error": expected_relative_error,
            "agreement_tolerance": ENGINEERING_AUTOGRAD_FD_RELATIVE_TOLERANCE,
            "cast_survived": True,
        },
    }


def assemble_engineering_evidence(
    root: str | Path,
    *,
    score_tolerance: float = 1e-4,
    legacy_equivalence_path: str | Path | None = None,
    expected_latent_steps: int = 32,
) -> dict[str, Any]:
    """Assemble the 15 checks from completed real artifacts plus pure invariants."""

    if (
        isinstance(score_tolerance, bool)
        or not isinstance(score_tolerance, (int, float))
        or not math.isfinite(float(score_tolerance))
        or float(score_tolerance) < 0
    ):
        raise ContractError("score_tolerance must be finite and non-negative")
    if (
        isinstance(expected_latent_steps, bool)
        or not isinstance(expected_latent_steps, int)
        or expected_latent_steps < 1
    ):
        raise ContractError("expected_latent_steps must be a positive integer")
    base = Path(root)
    current_source_hash = source_hash(_repo_root())
    checks: dict[str, dict[str, Any]] = {}
    try:
        from RecursiveMAS.inference_utils import linkradius as lr
        from RecursiveMAS.inference_utils.linkradius_runtime import LinkRadiusRuntime, RuntimeConfig

        checks["edge_counts"] = {
            "passed": len(lr.valid_edges(1)) == 2 and len(lr.valid_edges(2)) == 5,
            "detail": {"R1": len(lr.valid_edges(1)), "R2": len(lr.valid_edges(2))},
        }
        unloaded = LinkRadiusRuntime(RuntimeConfig(rounds=2))
        try:
            unloaded.prevalidate_edges(["s2p@1"])
            invalid_passed = False
        except ValueError:
            invalid_passed = unloaded.system is None
        checks["invalid_terminal_s2p_rejected_preload"] = {"passed": invalid_passed}
        expected_schedule = tuple(step.operation for step in lr.replay_schedule("p2c@0", 2))
        checks["early_replay_descendants_only"] = {
            "passed": expected_schedule == ("critic", "solver_feedback", "planner_feedback", "critic", "score_final"),
            "detail": list(expected_schedule),
        }
    except Exception as exc:
        checks["edge_counts"] = {"passed": False, "detail": str(exc)}
        checks["invalid_terminal_s2p_rejected_preload"] = {"passed": False, "detail": str(exc)}
        checks["early_replay_descendants_only"] = {"passed": False, "detail": str(exc)}

    trajectory_artifacts = _authenticated_artifacts(
        base, "clean_trajectory.pt", expected_source_hash=current_source_hash
    )
    clean_artifacts = _authenticated_artifacts(
        base, "clean_baseline.jsonl", expected_source_hash=current_source_hash
    )
    if len(trajectory_artifacts) != 1 or len(clean_artifacts) != 1:
        raise ContractError(
            "engineering validation requires exactly one completed current-source clean "
            f"trajectory/baseline pair; found {len(trajectory_artifacts)}/{len(clean_artifacts)}"
        )
    trajectory_artifact = trajectory_artifacts[0]
    if trajectory_artifact.path.parent != clean_artifacts[0].path.parent:
        raise ContractError("clean trajectory and baseline are not from one completed task")
    clean_task = _authenticated_task(
        trajectory_artifact,
        expected_source_hash=current_source_hash,
        expected_stage="clean",
    )

    try:
        import torch

        try:
            trajectory = torch.load(
                trajectory_artifact.path, map_location="cpu", weights_only=False
            )
        except TypeError:
            trajectory = torch.load(trajectory_artifact.path, map_location="cpu")
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ContractError(f"cannot load authenticated clean trajectory: {exc}") from exc

    if (
        getattr(trajectory, "rounds", None) != 2
        or len(getattr(trajectory, "sample_ids", ())) != 1
        or len(getattr(trajectory, "raw_sample_ids", ())) != 1
        or len(getattr(trajectory, "raw_indices", ())) != 1
        or list(getattr(trajectory, "analysis_eligibility_mask", ())) != [True]
    ):
        raise ContractError("engineering clean trajectory must contain one eligible R=2 sample")
    provenance = getattr(trajectory, "provenance", None)
    runtime_config = provenance.get("runtime_config") if isinstance(provenance, Mapping) else None
    if not isinstance(runtime_config, Mapping):
        raise ContractError("clean trajectory is missing recorded runtime configuration")
    recorded_latent_steps = runtime_config.get("latent_steps")
    if (
        isinstance(recorded_latent_steps, bool)
        or not isinstance(recorded_latent_steps, int)
        or recorded_latent_steps != expected_latent_steps
    ):
        raise ContractError(
            "clean trajectory recorded latent_steps does not match the engineering design"
        )

    receiver_messages = getattr(trajectory, "receiver_reference_messages", None)
    transport_messages = getattr(trajectory, "transport_messages", None)
    if not isinstance(receiver_messages, Mapping) or not isinstance(transport_messages, Mapping):
        raise ContractError("clean trajectory relay mappings are missing")
    receiver_edges = {edge.edge_id for edge in receiver_messages}
    transport_edges = {edge.edge_id for edge in transport_messages}

    def valid_relay_shape(value: Any) -> bool:
        return (
            getattr(value, "ndim", None) == 3
            and int(value.shape[0]) == 1
            and int(value.shape[1]) == recorded_latent_steps
            and int(value.shape[2]) > 0
        )

    shapes_ok = all(valid_relay_shape(value) for value in receiver_messages.values()) and all(
        valid_relay_shape(value) for value in transport_messages.values()
    )
    if receiver_edges != EXPECTED_R2_EDGES or transport_edges != EXPECTED_R2_EDGES or not shapes_ok:
        raise ContractError(
            "clean trajectory must contain exactly five R=2 transport/receiver relays with "
            f"shape [1,{recorded_latent_steps},D]"
        )
    checks["captured_all_r2_edges"] = {
        "passed": True,
        "detail": {
            "edges": sorted(receiver_edges),
            "latent_steps": recorded_latent_steps,
            "shapes_ok": True,
        },
    }

    shared = _shared_trajectory_provenance(trajectory)
    _bind_task_to_trajectory(
        clean_task,
        trajectory=trajectory,
        shared=shared,
        expected_latent_steps=expected_latent_steps,
    )
    scores = _to_score_mapping(trajectory.clean_scoring)
    if len(scores) != 1 or len(getattr(trajectory, "clean_margins", ())) != 1:
        raise ContractError("clean trajectory scoring must contain exactly one row")
    margins_finite = all(
        math.isfinite(float(value))
        for margins in trajectory.clean_margins
        for value in margins.values()
    )
    checks["finite_scores_and_margins"] = {
        "passed": all(math.isfinite(value) for row in scores for value in row.values())
        and margins_finite
    }
    audit = trajectory.clean_generation_audit[0] if trajectory.clean_generation_audit else {}
    strict_choice = audit.get("strict_choice")
    scorer_prediction = trajectory.clean_scoring.predictions[0]
    score_tie = bool(trajectory.clean_scoring.score_ties[0])
    comparable = strict_choice in {"A", "B", "C", "D"} and scorer_prediction in {
        "A",
        "B",
        "C",
        "D",
    } and not score_tie
    checks["strict_and_scored_predictions_recorded"] = {
        "passed": "strict_choice" in audit
        and len(trajectory.clean_scoring.predictions) == 1
        and len(trajectory.clean_scoring.score_ties) == 1
        and comparable
        and strict_choice == scorer_prediction,
        "detail": {
            "total_clean_rows": 1,
            "comparable_rows": int(comparable),
            "agreement": (
                float(strict_choice == scorer_prediction) if comparable else None
            ),
            "agreement_all_rows": float(
                comparable and strict_choice == scorer_prediction
            ),
            "comparable_coverage": float(comparable),
            "invalid_generation_rows": int(strict_choice not in {"A", "B", "C", "D"}),
            "scorer_tie_or_invalid_rows": int(
                score_tie or scorer_prediction not in {"A", "B", "C", "D"}
            ),
            "eligibility_filter_applied": False,
        },
    }
    edge = lr.Edge("p2c", 0)
    relay = trajectory.message(edge, receiver=True)
    subspace = lr.get_subspace(
        "full_tensor", int(relay.shape[1]), int(relay.shape[2])
    )
    direction_a = lr.sample_stable_unit_direction(
        42,
        trajectory.raw_sample_ids[0],
        edge,
        subspace,
        probe_seed=101,
        direction_id=0,
    )
    direction_b = lr.sample_stable_unit_direction(
        42,
        trajectory.raw_sample_ids[0],
        edge,
        subspace,
        probe_seed=101,
        direction_id=0,
    )
    checks["direction_identity_invariant"] = {
        "passed": bool(torch.equal(direction_a, direction_b))
    }

    clean_rows = _authenticated_jsonl_rows(
        clean_artifacts,
        expected_source_hash=current_source_hash,
        allowed_record_types={"sample"},
        expected_stage="clean",
    )
    if len(clean_rows) != 1:
        raise ContractError("engineering clean baseline must contain exactly one sample row")
    _bind_rows_to_trajectory(
        clean_rows,
        trajectory=trajectory,
        shared=shared,
        expected_source_hash=current_source_hash,
    )
    clean = clean_rows[0]
    clean_expected_signature = {
        "option_scores": scores[0],
        "scorer_prediction": trajectory.clean_scoring.predictions[0],
        "scorer_correct": trajectory.clean_scoring.predictions[0]
        == trajectory.gold_labels[0],
        "score_tie": bool(trajectory.clean_scoring.score_ties[0]),
        "margins": dict(trajectory.clean_margins[0]),
        "minimum_margin": min(trajectory.clean_margins[0].values()),
        "binding_competitor": min(
            trajectory.clean_margins[0], key=trajectory.clean_margins[0].get
        ),
    }
    if _score_signature(clean) != clean_expected_signature or clean.get("gold") != trajectory.gold_labels[0]:
        raise ContractError("clean baseline row disagrees with its authenticated trajectory")

    replay_artifacts = _authenticated_artifacts(
        base, "replay_runs.jsonl", expected_source_hash=current_source_hash
    )
    for artifact in replay_artifacts:
        _bind_task_to_trajectory(
            _authenticated_task(
                artifact,
                expected_source_hash=current_source_hash,
                expected_stage="replay",
            ),
            trajectory=trajectory,
            shared=shared,
            expected_latent_steps=expected_latent_steps,
        )
    replay_rows = _authenticated_jsonl_rows(
        replay_artifacts,
        expected_source_hash=current_source_hash,
        allowed_record_types={"sample"},
        expected_stage="replay",
    )
    _bind_rows_to_trajectory(
        replay_rows,
        trajectory=trajectory,
        shared=shared,
        expected_source_hash=current_source_hash,
    )
    _require_unique_rows(
        replay_rows,
        key_fields=("raw_sample_id", "intervention_mode", "edge_id"),
        where="engineering replay",
    )
    unknown_modes = {
        row.get("intervention_mode") for row in replay_rows
    } - {"identity", "additive_zero"}
    if unknown_modes:
        raise ContractError(f"engineering replay contains unexpected modes: {sorted(unknown_modes)!r}")

    def scores_match(row: Mapping[str, Any]) -> bool:
        option_scores = row.get("option_scores")
        clean_scores = clean.get("option_scores")
        if not isinstance(option_scores, Mapping) or not isinstance(clean_scores, Mapping):
            return False
        try:
            return set(option_scores) == set(clean_scores) == set("ABCD") and all(
                math.isfinite(float(option_scores[label]))
                and abs(float(option_scores[label]) - float(clean_scores[label]))
                <= float(score_tolerance)
                for label in "ABCD"
            )
        except (TypeError, ValueError):
            return False

    identity = [row for row in replay_rows if row.get("intervention_mode") == "identity"]
    zero = [row for row in replay_rows if row.get("intervention_mode") == "additive_zero"]
    checks["identity_replay_scores"] = {
        "passed": len(identity) == 5
        and {row.get("edge_id") for row in identity} == EXPECTED_R2_EDGES
        and all(scores_match(row) for row in identity),
        "detail": {"rows": len(identity), "edges": sorted(row.get("edge_id") for row in identity)},
    }
    checks["zero_additive_scores"] = {
        "passed": len(zero) == 5
        and {row.get("edge_id") for row in zero} == EXPECTED_R2_EDGES
        and all(scores_match(row) for row in zero),
        "detail": {"rows": len(zero), "edges": sorted(row.get("edge_id") for row in zero)},
    }
    repeated_rows = [*identity, *zero]
    checks["repeated_scoring_deterministic"] = {
        "passed": len(repeated_rows) == 10
        and all(_score_signature(row) == _score_signature(clean) for row in repeated_rows),
        "detail": {"independent_replays": len(repeated_rows), "comparison": "exact"},
    }

    probe_artifacts = _authenticated_artifacts(
        base, "probe_runs.jsonl", expected_source_hash=current_source_hash
    )
    probe_task_keys: set[tuple[float, int]] = set()
    for artifact in probe_artifacts:
        task = _authenticated_task(
            artifact,
            expected_source_hash=current_source_hash,
            expected_stage="probe",
        )
        _bind_task_to_trajectory(
            task,
            trajectory=trajectory,
            shared=shared,
            expected_latent_steps=expected_latent_steps,
        )
        h = task.get("h")
        probe_seed = task.get("probe_seed")
        if (
            isinstance(h, bool)
            or not isinstance(h, (int, float))
            or float(h) not in ENGINEERING_PROBE_RADII
            or isinstance(probe_seed, bool)
            or not isinstance(probe_seed, int)
            or task.get("K") != ENGINEERING_PROBE_K
            or task.get("edge_id") != "p2c@0"
            or (task.get("metadata") or {}).get("direction_ids")
            != list(range(ENGINEERING_PROBE_K))
        ):
            raise ContractError("engineering probe task differs from the K=8/two-radius design")
        task_key = (float(h), int(probe_seed))
        if task_key in probe_task_keys:
            raise ContractError(f"duplicate engineering probe task for {task_key!r}")
        probe_task_keys.add(task_key)

        task_rows = _authenticated_jsonl_rows(
            [artifact],
            expected_source_hash=current_source_hash,
            allowed_record_types={"sample", "probe_pair"},
            expected_stage="probe",
        )
        task_signed = [row for row in task_rows if row.get("record_type") == "sample"]
        task_pairs = [row for row in task_rows if row.get("record_type") == "probe_pair"]
        expected_signed_keys = {
            (direction_id, sign)
            for direction_id in range(ENGINEERING_PROBE_K)
            for sign in (-1, 1)
        }
        observed_signed_keys = {
            (row.get("direction_id"), row.get("sign")) for row in task_signed
        }
        observed_pair_directions = {row.get("direction_id") for row in task_pairs}
        if (
            len(task_signed) != 2 * ENGINEERING_PROBE_K
            or observed_signed_keys != expected_signed_keys
            or len(task_pairs) != ENGINEERING_PROBE_K
            or observed_pair_directions != set(range(ENGINEERING_PROBE_K))
        ):
            raise ContractError(
                "each engineering probe task must contain exactly one +/- pair for directions 0..7"
            )
    probe_rows = _authenticated_jsonl_rows(
        probe_artifacts,
        expected_source_hash=current_source_hash,
        allowed_record_types={"sample", "probe_pair"},
        expected_stage="probe",
    )
    _bind_rows_to_trajectory(
        probe_rows,
        trajectory=trajectory,
        shared=shared,
        expected_source_hash=current_source_hash,
    )
    _require_unique_rows(
        probe_rows,
        key_fields=("record_type", "run_id"),
        where="engineering probe",
    )
    signed = [row for row in probe_rows if row.get("record_type") == "sample"]
    pairs = [row for row in probe_rows if row.get("record_type") == "probe_pair"]
    _require_unique_rows(
        signed,
        key_fields=("raw_sample_id", "edge_id", "probe_seed", "direction_id", "sign", "h"),
        where="engineering signed probe",
    )
    _require_unique_rows(
        pairs,
        key_fields=("raw_sample_id", "edge_id", "probe_seed", "direction_id", "h"),
        where="engineering probe pair",
    )
    radii_to_seeds = {
        radius: {seed for h, seed in probe_task_keys if h == radius}
        for radius in ENGINEERING_PROBE_RADII
    }
    if probe_task_keys and (
        {h for h, _ in probe_task_keys} != ENGINEERING_PROBE_RADII
        or not all(radii_to_seeds.values())
        or len({frozenset(seeds) for seeds in radii_to_seeds.values()}) != 1
    ):
        raise ContractError(
            "engineering probe tasks must cover both radii with the same probe-seed inventory"
        )
    if probe_rows:
        from .probe_validation import reclassify_probe_pairs

        reclassified_pairs = reclassify_probe_pairs(
            probe_rows,
            ENGINEERING_PROBE_THRESHOLDS,
        )
    else:
        reclassified_pairs = []
    accepted_by_radius = {
        radius: sum(
            1
            for row in reclassified_pairs
            if float(row["h"]) == radius and bool(row.get("accepted"))
        )
        for radius in ENGINEERING_PROBE_RADII
    }
    checks["antithetic_cast_survival"] = {
        "passed": bool(probe_task_keys)
        and len(reclassified_pairs) == len(pairs)
        and any(count > 0 for count in accepted_by_radius.values()),
        "detail": {
            "signed_rows": len(signed),
            "pair_rows": len(pairs),
            "task_keys": sorted(probe_task_keys),
            "accepted_pairs_by_radius": accepted_by_radius,
            "thresholds": dict(ENGINEERING_PROBE_THRESHOLDS),
        },
    }

    gradient_artifacts = _authenticated_artifacts(
        base, "gradient_runs.jsonl", expected_source_hash=current_source_hash
    )
    for artifact in gradient_artifacts:
        _bind_task_to_trajectory(
            _authenticated_task(
                artifact,
                expected_source_hash=current_source_hash,
                expected_stage="gradient",
            ),
            trajectory=trajectory,
            shared=shared,
            expected_latent_steps=expected_latent_steps,
        )
    gradient_rows = _authenticated_jsonl_rows(
        gradient_artifacts,
        expected_source_hash=current_source_hash,
        allowed_record_types={"gradient"},
        expected_stage="gradient",
    )
    _bind_rows_to_trajectory(
        gradient_rows,
        trajectory=trajectory,
        shared=shared,
        expected_source_hash=current_source_hash,
    )
    _require_unique_rows(
        gradient_rows,
        key_fields=("record_type", "sample_id", "edge_id"),
        where="engineering gradient",
    )
    terminal_rows = [row for row in gradient_rows if row.get("edge_id") == "c2s@1"]
    terminal = terminal_rows[0] if len(terminal_rows) == 1 else None
    checks["terminal_autograd_finite_difference_agreement"] = (
        _validate_terminal_finite_difference(terminal, trajectory=trajectory)
    )
    pgd = (terminal or {}).get("pgd") or {}
    pgd_targets = pgd.get("targets", [])
    checks["pgd_budget_and_objective"] = {
        "passed": bool(pgd.get("supported"))
        and bool(pgd_targets)
        and all(bool(item.get("budget_respected")) for item in pgd_targets)
        and any(bool(item.get("improved")) for item in pgd_targets),
        "detail": pgd,
    }

    legacy_path = Path(legacy_equivalence_path) if legacy_equivalence_path else base / "legacy_equivalence.json"
    if legacy_path.is_file():
        try:
            from .compare_legacy_equivalence import verify_equivalence_report

            legacy = verify_equivalence_report(
                load_json(legacy_path),
                repo_root=_repo_root(),
                expected_trajectory_path=trajectory_artifact.path,
            )
        except (OSError, RuntimeError, TypeError, ValueError, ContractError) as exc:
            checks["legacy_release_equivalence"] = {
                "passed": False,
                "detail": f"legacy equivalence authentication failed: {exc}",
            }
        else:
            checks["legacy_release_equivalence"] = {
                "passed": True,
                "detail": {
                    "report_content_hash": legacy["report_content_hash"],
                    "legacy_trace_sha256": legacy["legacy_trace_sha256"],
                    "legacy_results_sha256": legacy["legacy_results_sha256"],
                    "checks": legacy["checks"],
                },
            }
    else:
        checks["legacy_release_equivalence"] = {
            "passed": False,
            "detail": "missing real legacy-versus-LinkRadius equivalence artifact",
        }
    checks["legacy_latent_contagion_regression"] = (
        _legacy_latent_contagion_regression_check()
    )
    return {
        "checks": checks,
        "sample_id": trajectory.sample_ids[0],
        "edge_ids": sorted(EXPECTED_R2_EDGES),
        "runtime": getattr(trajectory, "provenance", {}),
    }


def _to_score_mapping(scoring: Any) -> list[dict[str, float]]:
    values = scoring.scores.detach().float().cpu().tolist() if hasattr(scoring.scores, "detach") else scoring.scores
    return [
        {label: float(row[index]) for index, label in enumerate(scoring.labels)}
        for row in values
    ]


def validate_engineering_evidence(evidence: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_checks = evidence.get("checks", {})
    if isinstance(raw_checks, Sequence) and not isinstance(raw_checks, (str, bytes)):
        by_name = {str(check.get("name")): check for check in raw_checks if isinstance(check, Mapping)}
    elif isinstance(raw_checks, Mapping):
        by_name = {
            str(name): ({"name": name, **value} if isinstance(value, Mapping) else {"name": name, "passed": value is True})
            for name, value in raw_checks.items()
        }
    else:
        by_name = {}
    checks = []
    for name in MANDATORY_CHECKS:
        source = dict(by_name.get(name, {}))
        checks.append(
            {
                "name": name,
                "passed": source.get("passed") is True,
                "detail": source.get("detail", "missing mandatory evidence" if not source else ""),
                **{key: value for key, value in source.items() if key not in {"name", "passed", "detail"}},
            }
        )
    report = {
        "schema_version": "linkradius.engineering_report.v1",
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "sample_id": evidence.get("sample_id"),
        "edge_ids": evidence.get("edge_ids", []),
        "runtime": evidence.get("runtime", {}),
        "warnings": evidence.get("warnings", []),
    }
    report["report_content_hash"] = content_hash(report, domain="linkradius:engineering_report:v1")
    return report, checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", default="")
    parser.add_argument("--artifact-root", default="")
    parser.add_argument("--legacy-equivalence", default="")
    parser.add_argument("--expected-latent-steps", type=int, default=32)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--gate-output", required=True)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.evidence) == bool(args.artifact_root):
        raise ContractError("provide exactly one of --evidence or --artifact-root")
    evidence = (
        load_json(args.evidence)
        if args.evidence
        else assemble_engineering_evidence(
            args.artifact_root,
            legacy_equivalence_path=args.legacy_equivalence or None,
            expected_latent_steps=args.expected_latent_steps,
        )
    )
    report, checks = validate_engineering_evidence(evidence)
    gate = make_gate(
        gate_type="engineering_gate",
        checks=checks,
        config_hash=args.config_hash,
        source_hash=args.source_hash,
        prerequisite_hashes={"engineering_report_hash": report["report_content_hash"]},
    )
    atomic_write_json(args.report_output, report, overwrite=args.overwrite)
    atomic_write_json(args.gate_output, gate, overwrite=args.overwrite)
    print(json.dumps({"passed": gate["passed"], "report": str(Path(args.report_output).resolve())}, sort_keys=True))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
