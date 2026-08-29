"""Versioned public schemas and fail-closed validators for LinkRadius."""

from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "linkradius.v1"
SPLIT_MANIFEST_VERSION = "linkradius.split_manifest.v1"
EXECUTION_MANIFEST_VERSION = "linkradius.execution_manifest.v1"
COMPLETION_VERSION = "linkradius.completion.v1"
GATE_VERSION = "linkradius.gate.v1"
GRID_VERSION = "linkradius.grid.v1"

PARTITIONS = ("attack_train", "validation", "test")
EARLY_R2_EDGES = ("p2c@0", "c2s@0", "s2p@0")
CHOICE_LABELS = ("A", "B", "C", "D")
PORTABLE_SYSTEM_IDENTITY_VERSION = "linkradius_portable_system_identity_v1"
CLEAN_CORRECT_POLICIES = ("forced_margin", "dual_correct")
DEFAULT_CLEAN_CORRECT_POLICY = "forced_margin"

_DERIVED_FLOAT_REL_TOL = 1e-12
_DERIVED_FLOAT_ABS_TOL = 1e-12


class ContractError(ValueError):
    """Raised when an artifact violates a LinkRadius public contract."""


def _required(mapping: Mapping[str, Any], fields: Iterable[str], *, where: str) -> None:
    missing = [name for name in fields if name not in mapping]
    if missing:
        raise ContractError(f"{where} is missing required fields: {', '.join(missing)}")


def _nonempty_text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ContractError(f"{field} must be nonempty")
    return text


def _strict_nonempty_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a nonempty string")
    return value


def _strict_int(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ContractError(f"{field} must be at least {minimum}")
    return value


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{field} must be a number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ContractError(f"{field} must be finite")
    return numeric


def validate_sha256(value: Any, *, field: str) -> str:
    text = _strict_nonempty_text(value, field=field)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ContractError(f"{field} must be a lowercase SHA-256 hex digest")
    return text


def validate_split_manifest(manifest: Mapping[str, Any]) -> None:
    _required(
        manifest,
        ("schema_version", "dataset", "source_split", "seed", "ratios", "partitions"),
        where="split manifest",
    )
    if manifest["schema_version"] != SPLIT_MANIFEST_VERSION:
        raise ContractError(f"unsupported split manifest schema: {manifest['schema_version']!r}")
    partitions = manifest["partitions"]
    if not isinstance(partitions, Mapping) or set(partitions) != set(PARTITIONS):
        raise ContractError(f"split partitions must contain exactly {PARTITIONS!r}")
    all_ids: list[str] = []
    for name in PARTITIONS:
        rows = partitions[name]
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ContractError(f"partition {name!r} must be a list")
        for row in rows:
            if isinstance(row, Mapping):
                raw_id = _nonempty_text(row.get("raw_sample_id"), field=f"{name}.raw_sample_id")
            else:
                raw_id = _nonempty_text(row, field=f"{name}.raw_sample_id")
            all_ids.append(raw_id)
    if len(all_ids) != len(set(all_ids)):
        raise ContractError("split manifest partitions overlap or contain duplicate raw IDs")
    declared = manifest.get("num_records", len(all_ids))
    if isinstance(declared, bool) or int(declared) != len(all_ids):
        raise ContractError("split manifest num_records does not match partition rows")


def validate_execution_manifest(manifest: Mapping[str, Any]) -> None:
    _required(
        manifest,
        (
            "schema_version",
            "split_manifest_hash",
            "partition",
            "ordered_raw_sample_ids",
            "ordered_sample_ids",
            "batch_size",
            "batch_boundaries",
            "padding_policy",
            "array_shards",
            "analysis_eligible",
            "screening_dual_correct",
            "exclusion_reasons",
            "screening_config_hash",
        ),
        where="execution manifest",
    )
    if manifest["schema_version"] != EXECUTION_MANIFEST_VERSION:
        raise ContractError(
            f"unsupported execution manifest schema: {manifest['schema_version']!r}"
        )
    validate_sha256(manifest["split_manifest_hash"], field="split_manifest_hash")
    if manifest["partition"] not in PARTITIONS:
        raise ContractError(f"invalid execution partition: {manifest['partition']!r}")
    raw_ids = list(manifest["ordered_raw_sample_ids"])
    sample_ids = list(manifest["ordered_sample_ids"])
    eligible = list(manifest["analysis_eligible"])
    dual_correct = list(manifest["screening_dual_correct"])
    has_cohort_policy = "clean_correct_policy" in manifest
    clean_correct_policy = manifest.get("clean_correct_policy", "dual_correct")
    if clean_correct_policy not in CLEAN_CORRECT_POLICIES:
        raise ContractError(
            "clean_correct_policy must be one of "
            f"{CLEAN_CORRECT_POLICIES!r}, got: {clean_correct_policy!r}"
        )
    if has_cohort_policy and "screening_clean_correct" not in manifest:
        raise ContractError(
            "execution manifests declaring clean_correct_policy must record "
            "screening_clean_correct"
        )
    if has_cohort_policy and "screening_forced_margin_correct" not in manifest:
        raise ContractError(
            "execution manifests declaring clean_correct_policy must record "
            "screening_forced_margin_correct"
        )
    clean_correct = list(
        manifest.get(
            "screening_clean_correct",
            dual_correct if clean_correct_policy == "dual_correct" else [None] * len(raw_ids),
        )
    )
    forced_margin_correct = list(
        manifest.get("screening_forced_margin_correct", [None] * len(raw_ids))
    )
    generated_choices = list(
        manifest.get("screening_generated_choices", [None] * len(raw_ids))
    )
    scorer_predictions = list(
        manifest.get("screening_scorer_predictions", [None] * len(raw_ids))
    )
    reasons = list(manifest["exclusion_reasons"])
    n = len(raw_ids)
    if not (
        n
        == len(sample_ids)
        == len(eligible)
        == len(dual_correct)
        == len(clean_correct)
        == len(forced_margin_correct)
        == len(generated_choices)
        == len(scorer_predictions)
        == len(reasons)
    ):
        raise ContractError("execution row arrays must have identical lengths")
    if len(set(raw_ids)) != n or len(set(sample_ids)) != n:
        raise ContractError("execution manifest contains duplicate raw/sample IDs")
    if isinstance(manifest["batch_size"], bool) or int(manifest["batch_size"]) <= 0:
        raise ContractError("batch_size must be a positive integer")

    boundaries = list(manifest["batch_boundaries"])
    expected_start = 0
    batch_ids: list[int] = []
    for expected_id, boundary in enumerate(boundaries):
        if not isinstance(boundary, Mapping):
            raise ContractError("batch boundaries must be objects")
        _required(boundary, ("execution_batch_id", "start", "stop"), where="batch boundary")
        batch_id = int(boundary["execution_batch_id"])
        start, stop = int(boundary["start"]), int(boundary["stop"])
        if batch_id != expected_id or start != expected_start or not (start < stop <= n):
            raise ContractError("batch boundaries must be contiguous, ordered, nonempty, and in range")
        expected_start = stop
        batch_ids.append(batch_id)
    if n and expected_start != n:
        raise ContractError("batch boundaries do not cover every execution row")
    if not n and boundaries:
        raise ContractError("empty execution manifests cannot contain batch boundaries")

    assigned: list[int] = []
    for shard_idx, shard in enumerate(manifest["array_shards"]):
        if not isinstance(shard, Mapping):
            raise ContractError("array shards must be objects")
        _required(shard, ("array_index", "execution_batch_ids"), where="array shard")
        if int(shard["array_index"]) != shard_idx:
            raise ContractError("array shard indices must be zero-based and contiguous")
        ids = [int(value) for value in shard["execution_batch_ids"]]
        if not ids:
            raise ContractError("an array shard may not be empty")
        assigned.extend(ids)
    if sorted(assigned) != batch_ids or len(assigned) != len(set(assigned)):
        raise ContractError("array shards must assign each whole execution batch exactly once")
    for is_eligible, is_clean_correct, reason in zip(
        eligible, clean_correct, reasons
    ):
        if not isinstance(is_eligible, bool):
            raise ContractError("analysis_eligible entries must be booleans")
        if is_clean_correct is False and is_eligible:
            raise ContractError(
                "a screening-clean-incorrect row cannot be analysis eligible"
            )
        if is_eligible and str(reason or "").strip():
            raise ContractError("eligible rows cannot have an exclusion reason")
        if not is_eligible and not str(reason or "").strip():
            raise ContractError("ineligible rows require an explicit exclusion reason")
    if any(value is not None and not isinstance(value, bool) for value in dual_correct):
        raise ContractError("screening_dual_correct entries must be booleans or null for unscreened fillers")
    if any(value is not None and not isinstance(value, bool) for value in clean_correct):
        raise ContractError(
            "screening_clean_correct entries must be booleans or null for unscreened fillers"
        )
    if any(
        value is not None and not isinstance(value, bool)
        for value in forced_margin_correct
    ):
        raise ContractError(
            "screening_forced_margin_correct entries must be booleans or null "
            "for unscreened fillers"
        )
    selected_status = (
        forced_margin_correct
        if clean_correct_policy == "forced_margin"
        else dual_correct
    )
    for index, (selected, recorded) in enumerate(
        zip(selected_status, clean_correct)
    ):
        if selected is not None and recorded is not None and selected != recorded:
            raise ContractError(
                "screening_clean_correct disagrees with the selected "
                f"{clean_correct_policy} endpoint at row {index}"
            )
    labels = {"A", "B", "C", "D"}
    if any(value is not None and value not in labels for value in generated_choices):
        raise ContractError(
            "screening_generated_choices entries must be A-D or null"
        )
    if any(value is not None and value not in labels for value in scorer_predictions):
        raise ContractError(
            "screening_scorer_predictions entries must be A-D or null"
        )


INTERVENTION_REQUIRED_FIELDS = (
    "schema_version",
    "record_type",
    "run_id",
    "phase",
    "partition",
    "raw_sample_id",
    "sample_id",
    "raw_index",
    "dataset",
    "source_split",
    "style",
    "method",
    "R",
    "site",
    "code_round",
    "paper_round",
    "edge_id",
    "split_manifest_hash",
    "execution_manifest_hash",
    "ordered_cohort_hash",
    "batch_boundary_hash",
    "config_hash",
    "source_hash",
    "scorer_hash",
    "subspace_hash",
    "model_hash",
    "adapter_hash",
    "prompt_hash",
    "system_resolution",
    "analysis_eligible",
    "strict_generated_choice",
    "strict_generated_valid",
    "strict_generated_correct",
    "gold",
    "scorer_prediction",
    "scorer_correct",
    "score_tie",
    "option_scores",
    "margins",
    "minimum_margin",
    "binding_competitor",
    "intervention_mode",
    "intervention_family",
    "requested_intervention",
    "realized_intervention",
    "runtime",
    "failure",
    "warnings",
)


def _stable_json_sha256(value: Any, *, field: str) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{field} must contain canonical JSON values") from exc
    return hashlib.sha256(encoded).hexdigest()


def _validate_system_resolution(row: Mapping[str, Any]) -> None:
    resolution = row["system_resolution"]
    if not isinstance(resolution, Mapping):
        raise ContractError("system_resolution must be an object")
    _required(
        resolution,
        ("identity_kind", "style", "family", "dataset", "model_identity", "adapter_identity"),
        where="system_resolution",
    )
    identity_kind = _strict_nonempty_text(
        resolution["identity_kind"], field="system_resolution.identity_kind"
    )
    if identity_kind != PORTABLE_SYSTEM_IDENTITY_VERSION:
        raise ContractError(
            "successful intervention rows require a portable resolved-system identity"
        )

    style = _strict_nonempty_text(resolution["style"], field="system_resolution.style")
    dataset = _strict_nonempty_text(
        resolution["dataset"], field="system_resolution.dataset"
    )
    _strict_nonempty_text(resolution["family"], field="system_resolution.family")
    if style != row["style"] or dataset != row["dataset"]:
        raise ContractError("system_resolution style/dataset differs from the sample row")

    model_identity = resolution["model_identity"]
    adapter_identity = resolution["adapter_identity"]
    if not isinstance(model_identity, Mapping) or not isinstance(adapter_identity, Mapping):
        raise ContractError("resolved model_identity and adapter_identity must be objects")
    _required(model_identity, ("artifacts",), where="system_resolution.model_identity")
    _required(
        adapter_identity,
        ("inner_artifacts", "outer_artifacts"),
        where="system_resolution.adapter_identity",
    )
    if not isinstance(model_identity["artifacts"], Mapping) or not model_identity["artifacts"]:
        raise ContractError("resolved model_identity.artifacts must be a nonempty object")
    if not isinstance(adapter_identity["inner_artifacts"], Mapping) or not isinstance(
        adapter_identity["outer_artifacts"], Mapping
    ):
        raise ContractError("resolved adapter artifact inventories must be objects")

    for name, identity in (
        ("model_identity", model_identity),
        ("adapter_identity", adapter_identity),
    ):
        for common_field in ("identity_kind", "style", "family", "dataset"):
            if identity.get(common_field) != resolution[common_field]:
                raise ContractError(
                    f"system_resolution.{name}.{common_field} differs from its parent identity"
                )

    if _stable_json_sha256(model_identity, field="system_resolution.model_identity") != row[
        "model_hash"
    ]:
        raise ContractError("model_hash does not match system_resolution.model_identity")
    if _stable_json_sha256(
        adapter_identity, field="system_resolution.adapter_identity"
    ) != row["adapter_hash"]:
        raise ContractError("adapter_hash does not match system_resolution.adapter_identity")


def _validate_strict_generation(row: Mapping[str, Any], *, gold: str) -> None:
    choice = row["strict_generated_choice"]
    valid = row["strict_generated_valid"]
    correct = row["strict_generated_correct"]
    if choice is None and valid is None and correct is None:
        return
    if valid is not None and not isinstance(valid, bool):
        raise ContractError("strict_generated_valid must be a boolean or null")
    if correct is not None and not isinstance(correct, bool):
        raise ContractError("strict_generated_correct must be a boolean or null")
    if choice is None:
        if valid is not False or correct is not False:
            raise ContractError(
                "an attempted generation without a strict choice must be invalid and incorrect"
            )
        return
    if not isinstance(choice, str) or choice not in CHOICE_LABELS:
        raise ContractError("strict_generated_choice must be A/B/C/D or null")
    if valid is not True:
        raise ContractError("a strict generated choice must be marked valid")
    if correct is not (choice == gold):
        raise ContractError("strict_generated_correct disagrees with the strict choice and gold")


def _validate_runtime(row: Mapping[str, Any]) -> tuple[float, float]:
    runtime = row["runtime"]
    if not isinstance(runtime, Mapping):
        raise ContractError("runtime must be an object")
    _required(
        runtime,
        ("rounds", "style", "dataset", "score_tie_atol", "score_tie_rtol"),
        where="runtime",
    )
    if _strict_int(runtime["rounds"], field="runtime.rounds", minimum=1) != row["R"]:
        raise ContractError("runtime.rounds differs from R")
    if runtime["style"] != row["style"] or runtime["dataset"] != row["dataset"]:
        raise ContractError("runtime style/dataset differs from the sample row")
    atol = _finite_number(runtime["score_tie_atol"], field="runtime.score_tie_atol")
    rtol = _finite_number(runtime["score_tie_rtol"], field="runtime.score_tie_rtol")
    if atol < 0 or rtol < 0:
        raise ContractError("runtime score tie tolerances must be non-negative")
    _stable_json_sha256(runtime, field="runtime")
    return atol, rtol


def validate_intervention_row(row: Mapping[str, Any]) -> None:
    _required(row, INTERVENTION_REQUIRED_FIELDS, where="intervention row")
    if row["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"unsupported sample schema: {row['schema_version']!r}")
    if row["record_type"] != "sample":
        raise ContractError("intervention row record_type must be 'sample'")
    for field in (
        "split_manifest_hash",
        "execution_manifest_hash",
        "ordered_cohort_hash",
        "batch_boundary_hash",
        "config_hash",
        "source_hash",
        "scorer_hash",
        "subspace_hash",
        "model_hash",
        "adapter_hash",
        "prompt_hash",
    ):
        validate_sha256(row[field], field=field)

    for field in (
        "run_id",
        "phase",
        "raw_sample_id",
        "sample_id",
        "dataset",
        "source_split",
        "style",
        "method",
        "site",
        "edge_id",
        "intervention_mode",
    ):
        _strict_nonempty_text(row[field], field=field)
    if row["partition"] not in PARTITIONS:
        raise ContractError(f"invalid intervention partition: {row['partition']!r}")
    _strict_int(row["raw_index"], field="raw_index", minimum=0)
    rounds = _strict_int(row["R"], field="R", minimum=1)
    code_round = _strict_int(row["code_round"], field="code_round", minimum=0)
    paper_round = _strict_int(row["paper_round"], field="paper_round", minimum=1)
    if row["site"] not in {"p2c", "c2s", "s2p"}:
        raise ContractError("site must be p2c, c2s, or s2p")
    if code_round >= rounds or (row["site"] == "s2p" and code_round >= rounds - 1):
        raise ContractError("edge round is not valid for R")
    if row["edge_id"] != f"{row['site']}@{code_round}":
        raise ContractError("edge_id does not match site/code_round")
    if paper_round != code_round + 1:
        raise ContractError("paper_round must equal code_round + 1")

    if not isinstance(row["analysis_eligible"], bool):
        raise ContractError("analysis_eligible must be a boolean")
    family = row["intervention_family"]
    if family is not None:
        _strict_nonempty_text(family, field="intervention_family")
    for field in ("requested_intervention", "realized_intervention"):
        if not isinstance(row[field], Mapping):
            raise ContractError(f"{field} must be an object")
        _stable_json_sha256(row[field], field=field)
    if row["failure"] is not None:
        raise ContractError("a successful sample row must have failure=null")
    warnings = row["warnings"]
    if not isinstance(warnings, list) or any(
        not isinstance(warning, str) or not warning.strip() for warning in warnings
    ):
        raise ContractError("warnings must be a list of nonempty strings")

    _validate_system_resolution(row)
    tie_atol, tie_rtol = _validate_runtime(row)

    scores = row["option_scores"]
    if not isinstance(scores, Mapping) or set(scores) != set(CHOICE_LABELS):
        raise ContractError("option_scores must contain exactly A, B, C, and D")
    numeric_scores = {
        label: _finite_number(scores[label], field=f"option_scores.{label}")
        for label in CHOICE_LABELS
    }
    gold = row["gold"]
    if not isinstance(gold, str) or gold not in CHOICE_LABELS:
        raise ContractError("gold must be one of A/B/C/D")
    _validate_strict_generation(row, gold=gold)

    maximum = max(numeric_scores.values())
    winners = tuple(
        label
        for label in CHOICE_LABELS
        if math.isclose(
            numeric_scores[label], maximum, rel_tol=tie_rtol, abs_tol=tie_atol
        )
    )
    expected_tie = len(winners) != 1
    expected_prediction = None if expected_tie else winners[0]
    if not isinstance(row["score_tie"], bool) or row["score_tie"] is not expected_tie:
        raise ContractError("score_tie disagrees with option_scores and runtime tolerances")
    prediction = row["scorer_prediction"]
    if prediction is not None and (
        not isinstance(prediction, str) or prediction not in CHOICE_LABELS
    ):
        raise ContractError("scorer_prediction must be A/B/C/D or null")
    if prediction != expected_prediction:
        raise ContractError("scorer_prediction disagrees with option_scores")
    if not isinstance(row["scorer_correct"], bool):
        raise ContractError("scorer_correct must be a boolean")
    if row["scorer_correct"] is not (expected_prediction == gold):
        raise ContractError("scorer_correct disagrees with scorer_prediction and gold")

    expected_competitors = tuple(label for label in CHOICE_LABELS if label != gold)
    margins = row["margins"]
    if not isinstance(margins, Mapping) or set(margins) != set(expected_competitors):
        raise ContractError("margins must contain exactly the three non-gold competitors")
    numeric_margins = {
        label: _finite_number(margins[label], field=f"margins.{label}")
        for label in expected_competitors
    }
    expected_margins = {
        label: numeric_scores[gold] - numeric_scores[label]
        for label in expected_competitors
    }
    for label in expected_competitors:
        if not math.isclose(
            numeric_margins[label],
            expected_margins[label],
            rel_tol=_DERIVED_FLOAT_REL_TOL,
            abs_tol=_DERIVED_FLOAT_ABS_TOL,
        ):
            raise ContractError(f"margin for {label} disagrees with gold and option scores")
    expected_minimum = min(expected_margins.values())
    minimum = _finite_number(row["minimum_margin"], field="minimum_margin")
    if not math.isclose(
        minimum,
        expected_minimum,
        rel_tol=_DERIVED_FLOAT_REL_TOL,
        abs_tol=_DERIVED_FLOAT_ABS_TOL,
    ):
        raise ContractError("minimum_margin does not equal the minimum gold margin")
    expected_binding = min(expected_competitors, key=expected_margins.__getitem__)
    if row["binding_competitor"] != expected_binding:
        raise ContractError("binding_competitor is not the minimum-margin competitor")


@dataclass(frozen=True)
class ArtifactExpectation:
    path: str
    sha256: str
    row_count: int | None = None


def validate_completion_record(
    record: Mapping[str, Any],
    *,
    expected_config_hash: str | None = None,
    expected_artifacts: Sequence[ArtifactExpectation] | None = None,
) -> None:
    _required(
        record,
        ("schema_version", "status", "config_hash", "source_hash", "artifacts", "completed_at"),
        where="completion record",
    )
    if record["schema_version"] != COMPLETION_VERSION or record["status"] != "complete":
        raise ContractError("completion record must have the current schema and complete status")
    config_hash = validate_sha256(record["config_hash"], field="config_hash")
    validate_sha256(record["source_hash"], field="source_hash")
    if expected_config_hash is not None and config_hash != expected_config_hash:
        raise ContractError("completion config hash is stale")
    artifacts = record["artifacts"]
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        raise ContractError("completion artifacts must be a list")
    seen: set[str] = set()
    by_path: dict[str, Mapping[str, Any]] = {}
    for artifact in artifacts:
        _required(artifact, ("path", "sha256", "size_bytes"), where="completion artifact")
        path = _nonempty_text(artifact["path"], field="artifact.path")
        if path in seen:
            raise ContractError(f"duplicate completion artifact path: {path}")
        seen.add(path)
        by_path[path] = artifact
        validate_sha256(artifact["sha256"], field=f"artifacts[{path}].sha256")
        if int(artifact["size_bytes"]) < 0:
            raise ContractError("artifact size cannot be negative")
    for expected in expected_artifacts or ():
        actual = by_path.get(expected.path)
        if actual is None:
            raise ContractError(f"completion is missing expected artifact: {expected.path}")
        if actual["sha256"] != expected.sha256:
            raise ContractError(f"completion artifact hash is stale: {expected.path}")
        if expected.row_count is not None and int(actual.get("row_count", -1)) != expected.row_count:
            raise ContractError(f"completion artifact row count differs: {expected.path}")


def validate_gate(
    gate: Mapping[str, Any],
    *,
    gate_type: str | None = None,
    required_hashes: Mapping[str, str] | None = None,
) -> None:
    _required(
        gate,
        ("schema_version", "gate_type", "passed", "config_hash", "source_hash", "checks", "gate_content_hash"),
        where="gate",
    )
    if gate["schema_version"] != GATE_VERSION:
        raise ContractError(f"unsupported gate schema: {gate['schema_version']!r}")
    if gate_type is not None and gate["gate_type"] != gate_type:
        raise ContractError(f"expected {gate_type!r} gate, got {gate['gate_type']!r}")
    if gate["passed"] is not True:
        raise ContractError(f"gate {gate['gate_type']!r} did not pass")
    validate_sha256(gate["config_hash"], field="gate.config_hash")
    validate_sha256(gate["source_hash"], field="gate.source_hash")
    payload = dict(gate)
    supplied_gate_hash = payload.pop("gate_content_hash")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected_gate_hash = hashlib.sha256(
        b"linkradius:gate_content:v1\0" + encoded
    ).hexdigest()
    if supplied_gate_hash != expected_gate_hash:
        raise ContractError("gate_content_hash is missing or stale")
    checks = gate["checks"]
    if not isinstance(checks, Sequence) or not checks:
        raise ContractError("passed gate must contain checks")
    failed = [check for check in checks if not isinstance(check, Mapping) or check.get("passed") is not True]
    if failed:
        raise ContractError("passed gate contains a failed or malformed check")
    for field, expected in (required_hashes or {}).items():
        actual = gate.get(field)
        if actual != expected:
            raise ContractError(f"gate prerequisite hash mismatch for {field}")
