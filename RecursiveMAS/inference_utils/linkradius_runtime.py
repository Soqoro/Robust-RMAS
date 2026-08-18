"""Persistent sequential runtime used by the LinkRadius experiments.

The release inference path intentionally loads and unloads one agent at a time.
That is a good default for ordinary evaluation, but it cannot resume a saved
trajectory or retain an autograd graph across agents.  This module provides a
separate, persistent runtime with an explicit receiver-interface boundary.

The module is deliberately importable without PyTorch.  Pure scheduling and
tokenisation helpers are used by CPU-only manifest/tests; model-backed methods
call :func:`_require_torch` and fail with an actionable error when PyTorch is
not installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Match the release deterministic CUDA contract before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:  # Keep experiment CLI/schema imports usable in the lightweight test env.
    import torch
except ModuleNotFoundError:  # pragma: no cover - covered by import smoke tests.
    torch = None  # type: ignore[assignment]

from . import linkradius as lr


SCORER_VERSION = "linkradius_forced_choice_v1"
TRAJECTORY_VERSION = "linkradius_clean_trajectory_v1"
RUNTIME_VERSION = "linkradius_persistent_sequential_v2"
SYSTEM_IDENTITY_VERSION = "linkradius_portable_system_identity_v1"
DEFAULT_SCORER_PREFIX = "Final Choice: \\boxed{"
DEFAULT_VERBALIZERS: Mapping[str, str] = {
    "A": "A}",
    "B": "B}",
    "C": "C}",
    "D": "D}",
}
CHOICE_LABELS: Tuple[str, ...] = tuple(DEFAULT_VERBALIZERS)
SEQUENTIAL_ROLES: Tuple[str, ...] = ("planner", "critic", "solver")
EDGE_CONSUMER_ROLES: Mapping[str, str] = {
    "p2c": "critic",
    "c2s": "solver",
    "s2p": "planner",
}


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "LinkRadius model execution requires PyTorch. Install the RecursiveMAS "
            "runtime dependencies or use the project's recursivemas environment."
        )
    return torch


def _stable_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_artifact_identity(path: Any, *, repo_id: Optional[str] = None) -> Dict[str, Any]:
    """Return an artifact identity that is independent of the local cache root.

    Hugging Face cache paths carry either an immutable snapshot revision or a
    content-addressed blob ID.  Local files/directories fall back to exact
    content hashes.  Absolute paths are deliberately excluded; callers may log
    them separately as diagnostics, but they are not scientific identity.
    """

    expanded = Path(os.path.expanduser(str(path)))
    requested = expanded.absolute().as_posix()
    resolved_path = expanded.resolve(strict=False)
    resolved = resolved_path.as_posix()
    snapshot_pattern = re.compile(
        r"(?:^|/)(?P<repo>models--[^/]+)/snapshots/"
        r"(?P<revision>[^/]+)(?:/(?P<relative>.*))?$"
    )
    blob_pattern = re.compile(
        r"(?:^|/)(?P<repo>models--[^/]+)/blobs/(?P<blob>[^/]+)$"
    )
    snapshot_match = snapshot_pattern.search(requested) or snapshot_pattern.search(resolved)
    blob_match = blob_pattern.search(requested) or blob_pattern.search(resolved)

    def cache_repo(match: Any) -> str:
        return match.group("repo").removeprefix("models--").replace("--", "/")

    if snapshot_match is not None:
        return {
            "kind": "hf_snapshot",
            "repo_id": str(repo_id or cache_repo(snapshot_match)),
            "snapshot_revision": snapshot_match.group("revision"),
            "snapshot_relative_path": snapshot_match.group("relative") or ".",
        }
    if blob_match is not None:
        return {
            "kind": "hf_blob",
            "repo_id": str(repo_id or cache_repo(blob_match)),
            "blob_id": blob_match.group("blob"),
        }

    identity: Dict[str, Any] = {
        "kind": "local_content",
        "repo_id": str(repo_id) if repo_id is not None else None,
    }
    if not resolved_path.exists():
        identity.update(exists=False, missing_name=resolved_path.name)
        return identity
    identity["exists"] = True
    if resolved_path.is_file():
        identity.update(
            artifact_type="file",
            size=int(resolved_path.stat().st_size),
            sha256=_sha256_file(resolved_path),
        )
        return identity
    if not resolved_path.is_dir():
        identity.update(artifact_type="unsupported")
        return identity

    entries: List[Tuple[str, int, str]] = []
    for item in sorted(
        (candidate for candidate in resolved_path.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(resolved_path).as_posix(),
    ):
        relative = item.relative_to(resolved_path).as_posix()
        entries.append((relative, int(item.stat().st_size), _sha256_file(item)))
    identity.update(
        artifact_type="directory",
        entry_count=len(entries),
        content_sha256=_stable_json_hash(entries),
    )
    return identity


def _normalise_ids(value: Any) -> List[int]:
    """Normalize the common tokenizer ``input_ids`` return shapes."""

    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError("Tokenizer result does not contain input_ids")
        return _normalise_ids(value["input_ids"])
    if hasattr(value, "tolist"):
        return _normalise_ids(value.tolist())
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            if len(value) != 1:
                raise ValueError("Expected one tokenized string, received a batch")
            return _normalise_ids(value[0])
        return [int(item) for item in value]
    raise TypeError(f"Unsupported input_ids type: {type(value)!r}")


def _normalise_offsets(value: Any) -> Optional[List[Tuple[int, int]]]:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and value and isinstance(value[0], list):
        # Offsets are either [[start, end], ...] or batched one level deeper.
        if value[0] and isinstance(value[0][0], (list, tuple)):
            if len(value) != 1:
                return None
            value = value[0]
    if not isinstance(value, list):
        return None
    try:
        return [(int(pair[0]), int(pair[1])) for pair in value]
    except (TypeError, ValueError, IndexError):
        return None


def _tokenize_no_specials(
    tokenizer: Any,
    text: str,
    *,
    request_offsets: bool = False,
) -> Tuple[List[int], Optional[List[Tuple[int, int]]]]:
    kwargs: Dict[str, Any] = {"add_special_tokens": False}
    if request_offsets:
        kwargs["return_offsets_mapping"] = True
    try:
        encoded = tokenizer(text, **kwargs)
    except (TypeError, NotImplementedError, ValueError):
        if not request_offsets:
            raise
        encoded = tokenizer(text, add_special_tokens=False)
    ids = _normalise_ids(encoded)
    offsets = None
    if request_offsets and isinstance(encoded, Mapping):
        offsets = _normalise_offsets(encoded.get("offset_mapping"))
        if offsets is not None and len(offsets) != len(ids):
            offsets = None
    return ids, offsets


def longest_common_token_prefix(sequences: Sequence[Sequence[int]]) -> int:
    """Return the number of identical leading tokens in every sequence."""

    if not sequences:
        raise ValueError("sequences must not be empty")
    limit = min(len(sequence) for sequence in sequences)
    for index in range(limit):
        first = int(sequences[0][index])
        if any(int(sequence[index]) != first for sequence in sequences[1:]):
            return index
    return limit


@dataclass(frozen=True)
class CandidateEncoding:
    """Joint prefix/candidate encoding and the candidate-dependent token span."""

    label: str
    verbalizer: str
    joint_text: str
    token_ids: Tuple[int, ...]
    candidate_start: int
    candidate_end: int
    span_method: str
    offsets: Optional[Tuple[Tuple[int, int], ...]] = None

    @property
    def candidate_token_ids(self) -> Tuple[int, ...]:
        return self.token_ids[self.candidate_start : self.candidate_end]

    @property
    def token_count(self) -> int:
        return self.candidate_end - self.candidate_start


def tokenize_joint_candidates(
    tokenizer: Any,
    prefix: str = DEFAULT_SCORER_PREFIX,
    verbalizers: Mapping[str, str] = DEFAULT_VERBALIZERS,
) -> Dict[str, CandidateEncoding]:
    """Jointly tokenize a scorer prefix and every candidate verbalizer.

    Fast-tokenizer character offsets are preferred.  A token overlapping the
    prefix/candidate character boundary is candidate-dependent and is included.
    When reliable offsets are unavailable, the longest token prefix common to
    *all* joint encodings is used.  Tokenizing the verbalizer on its own is never
    used because BPE/SentencePiece merges can cross the boundary.
    """

    if not verbalizers:
        raise ValueError("verbalizers must not be empty")
    labels = [str(label).upper() for label in verbalizers]
    if len(labels) != len(set(labels)):
        raise ValueError("verbalizer labels must be unique after normalization")

    raw: Dict[str, Tuple[str, str, List[int], Optional[List[Tuple[int, int]]]]] = {}
    for original_label, verbalizer_value in verbalizers.items():
        label = str(original_label).upper()
        verbalizer = str(verbalizer_value)
        if not verbalizer:
            raise ValueError(f"Empty verbalizer for {label}")
        joint_text = str(prefix) + verbalizer
        ids, offsets = _tokenize_no_specials(tokenizer, joint_text, request_offsets=True)
        if not ids:
            raise ValueError(f"Tokenizer produced no tokens for candidate {label}")
        raw[label] = (verbalizer, joint_text, ids, offsets)

    if len(raw) == 1:
        # A prefix-only comparison still detects a token whose BPE merge crosses
        # the boundary.  This avoids treating the full one-item sequence as its
        # own (and therefore empty-span) common prefix.
        prefix_ids, _ = _tokenize_no_specials(tokenizer, str(prefix), request_offsets=False)
        only_ids = next(iter(raw.values()))[2]
        common_start = longest_common_token_prefix((prefix_ids, only_ids))
    else:
        common_start = longest_common_token_prefix([item[2] for item in raw.values()])
    encodings: Dict[str, CandidateEncoding] = {}
    prefix_chars = len(prefix)
    for label in labels:
        verbalizer, joint_text, ids, offsets = raw[label]
        span_start: Optional[int] = None
        span_method = "longest_common_token_prefix"
        if offsets is not None:
            overlapping = [
                index
                for index, (start, end) in enumerate(offsets)
                if end > prefix_chars and start < len(joint_text)
            ]
            if overlapping:
                # Require one contiguous suffix; unusual special-token offsets
                # fail closed into the deterministic LCP fallback.
                proposed = overlapping[0]
                if overlapping == list(range(proposed, len(ids))):
                    span_start = proposed
                    span_method = "offset_mapping"
        if span_start is None:
            span_start = common_start
        if span_start >= len(ids):
            raise ValueError(
                "Candidate-dependent token span is empty. The tokenizer does not "
                f"distinguish verbalizer {label!r} under the frozen prefix."
            )
        encodings[label] = CandidateEncoding(
            label=label,
            verbalizer=verbalizer,
            joint_text=joint_text,
            token_ids=tuple(ids),
            candidate_start=int(span_start),
            candidate_end=len(ids),
            span_method=span_method,
            offsets=tuple(offsets) if offsets is not None else None,
        )
    return encodings


def tokenize_joint_candidate(
    tokenizer: Any,
    prefix: str,
    candidate: str,
    *,
    label: str = "candidate",
    comparison_candidates: Optional[Sequence[str]] = None,
) -> CandidateEncoding:
    """Convenience wrapper for one candidate with an optional LCP comparison set."""

    comparisons = list(comparison_candidates or ())
    if candidate not in comparisons:
        comparisons.append(candidate)
    mapping = {f"candidate_{index}": value for index, value in enumerate(comparisons)}
    selected_key = next(key for key, value in mapping.items() if value == candidate)
    selected = tokenize_joint_candidates(tokenizer, prefix, mapping)[selected_key.upper()]
    return CandidateEncoding(
        label=str(label),
        verbalizer=selected.verbalizer,
        joint_text=selected.joint_text,
        token_ids=selected.token_ids,
        candidate_start=selected.candidate_start,
        candidate_end=selected.candidate_end,
        span_method=selected.span_method,
        offsets=selected.offsets,
    )


def prediction_from_scores(
    scores: Sequence[float],
    labels: Sequence[str] = CHOICE_LABELS,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> Tuple[Optional[str], bool]:
    """Return the unique argmax, or ``(None, True)`` for a numerical tie."""

    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must have the same non-zero length")
    if atol < 0 or rtol < 0:
        raise ValueError("tie tolerances must be non-negative")
    numeric = [float(value) for value in scores]
    if not all(math.isfinite(value) for value in numeric):
        return None, False
    maximum = max(numeric)
    winners = [
        index
        for index, value in enumerate(numeric)
        if math.isclose(value, maximum, rel_tol=rtol, abs_tol=atol)
    ]
    if len(winners) != 1:
        return None, True
    return str(labels[winners[0]]), False


def choice_margins(
    scores: Sequence[float],
    gold_label: str,
    labels: Sequence[str] = CHOICE_LABELS,
) -> Dict[str, float]:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    gold = str(gold_label).upper()
    normalized = [str(label).upper() for label in labels]
    if gold not in normalized:
        raise ValueError(f"Unsupported gold choice: {gold_label!r}")
    gold_score = float(scores[normalized.index(gold)])
    return {
        label: gold_score - float(scores[index])
        for index, label in enumerate(normalized)
        if label != gold
    }


def causal_token_log_probs(
    logits: Any,
    target_token_ids: Any,
    target_positions: Any,
) -> Any:
    """Gather causal token log-probabilities with float32 ``log_softmax``.

    ``target_positions`` are positions of the *tokens* in the input sequence;
    logits at ``position - 1`` score each target.  ``logits`` may have shape
    ``[L,V]`` or ``[B,L,V]``.  This helper intentionally does not accept a
    pre-shifted convention, eliminating a common off-by-one scorer bug.
    """

    t = _require_torch()
    logits_tensor = logits if t.is_tensor(logits) else t.as_tensor(logits)
    ids = target_token_ids if t.is_tensor(target_token_ids) else t.as_tensor(target_token_ids)
    positions = target_positions if t.is_tensor(target_positions) else t.as_tensor(target_positions)
    if logits_tensor.ndim == 2:
        logits_tensor = logits_tensor.unsqueeze(0)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    if logits_tensor.ndim != 3 or ids.ndim != 2 or positions.ndim != 2:
        raise ValueError("Expected logits [B,L,V] and ids/positions [B,K]")
    if ids.shape != positions.shape or ids.size(0) != logits_tensor.size(0):
        raise ValueError("Batch/token dimensions do not agree")
    prediction_positions = positions.to(dtype=t.long, device=logits_tensor.device) - 1
    if bool((prediction_positions < 0).any()) or bool(
        (prediction_positions >= logits_tensor.size(1)).any()
    ):
        raise ValueError("Every target token must have a preceding in-range logit")
    batch_index = t.arange(logits_tensor.size(0), device=logits_tensor.device).unsqueeze(1)
    selected_logits = logits_tensor[batch_index, prediction_positions]
    log_probs = t.log_softmax(selected_logits.float(), dim=-1)
    return log_probs.gather(-1, ids.to(device=logits_tensor.device, dtype=t.long).unsqueeze(-1)).squeeze(-1)


@dataclass(frozen=True)
class ReplayStep:
    action: str
    round_idx: Optional[int] = None
    consumes: Optional[str] = None
    produces: Optional[str] = None

    @property
    def token(self) -> str:
        if self.round_idx is None:
            return self.action
        return f"{self.action}@{self.round_idx}"


def replay_schedule(edge: Any, rounds: int) -> Tuple[ReplayStep, ...]:
    """Return the exact descendants recomputed after intervention at ``edge``."""

    parsed = lr.parse_edge(edge) if not isinstance(edge, lr.Edge) else edge
    lr.validate_edge(parsed, rounds)
    steps: List[ReplayStep] = []
    round_idx = parsed.round_idx
    if parsed.site == "p2c":
        steps.append(ReplayStep("critic", round_idx, parsed.edge_id, f"c2s@{round_idx}"))
        if round_idx == rounds - 1:
            steps.append(ReplayStep("score_final", round_idx, f"c2s@{round_idx}", None))
            return tuple(steps)
        steps.append(
            ReplayStep("solver_feedback", round_idx, f"c2s@{round_idx}", f"s2p@{round_idx}")
        )
    elif parsed.site == "c2s":
        if round_idx == rounds - 1:
            steps.append(ReplayStep("score_final", round_idx, parsed.edge_id, None))
            return tuple(steps)
        steps.append(
            ReplayStep("solver_feedback", round_idx, parsed.edge_id, f"s2p@{round_idx}")
        )
    elif parsed.site == "s2p":
        pass
    else:  # ``validate_edge`` should make this unreachable.
        raise ValueError(f"Unsupported LinkRadius edge site: {parsed.site!r}")

    for next_round in range(round_idx + 1, rounds):
        steps.append(
            ReplayStep(
                "planner_feedback",
                next_round,
                f"s2p@{next_round - 1}",
                f"p2c@{next_round}",
            )
        )
        steps.append(
            ReplayStep("critic", next_round, f"p2c@{next_round}", f"c2s@{next_round}")
        )
        if next_round < rounds - 1:
            steps.append(
                ReplayStep(
                    "solver_feedback",
                    next_round,
                    f"c2s@{next_round}",
                    f"s2p@{next_round}",
                )
            )
    steps.append(ReplayStep("score_final", rounds - 1, f"c2s@{rounds - 1}", None))
    return tuple(steps)


def replay_schedule_tokens(edge: Any, rounds: int) -> Tuple[str, ...]:
    return tuple(step.token for step in replay_schedule(edge, rounds))


@dataclass
class RuntimeConfig:
    rounds: int = 2
    latent_steps: int = 32
    batch_size: int = 16
    style: str = "sequential_light"
    dataset: str = "gpqa"
    seed: int = 42
    deterministic: bool = True
    device: str = "cuda"
    planner_device: str = ""
    critic_device: str = ""
    solver_device: str = ""
    dtype: str = "auto"
    outer_dtype: str = "auto"
    enable_thinking: bool = False
    choice_old_prompt: int = 2
    solver_pre_question: int = 0
    mas_shape: str = "chain"
    prompt_footer: str = ""
    role_response_regime: str = "neutral"
    role_response_regime_path: str = ""
    round_label_mode: str = "legacy"
    max_new_tokens: int = 4000
    do_sample: bool = False
    temperature: float = 0.6
    top_p: float = 0.95
    answer_retry: bool = True
    retry_max_new_tokens: int = 16
    scorer_prefix: str = DEFAULT_SCORER_PREFIX
    scorer_normalization: str = "mean"
    score_tie_atol: float = 0.0
    score_tie_rtol: float = 0.0
    scorer_version: str = SCORER_VERSION

    def __post_init__(self) -> None:
        self.device = str(self.device).strip()
        if not self.device:
            raise ValueError("device must not be empty")
        for role in SEQUENTIAL_ROLES:
            field_name = f"{role}_device"
            explicit = str(getattr(self, field_name) or "").strip()
            # Canonicalize fallback placement immediately so semantically
            # identical configurations hash identically in saved trajectories.
            setattr(self, field_name, explicit or self.device)
        if int(self.rounds) < 1:
            raise ValueError("rounds must be at least 1")
        if int(self.latent_steps) < 1:
            raise ValueError("latent_steps must be at least 1")
        if int(self.batch_size) < 1:
            raise ValueError("batch_size must be at least 1")
        if self.scorer_normalization not in {"mean", "sum"}:
            raise ValueError("scorer_normalization must be 'mean' or 'sum'")
        if self.round_label_mode not in {"legacy", "actual"}:
            raise ValueError("round_label_mode must be 'legacy' or 'actual'")
        if self.score_tie_atol < 0 or self.score_tie_rtol < 0:
            raise ValueError("score tie tolerances must be non-negative")
        if not self.scorer_prefix:
            raise ValueError("scorer_prefix must not be empty")

    def device_for_role(self, role: str) -> str:
        normalized = str(role).strip().lower()
        if normalized not in SEQUENTIAL_ROLES:
            raise ValueError(f"unknown sequential role: {role!r}")
        return str(getattr(self, f"{normalized}_device"))

    def resolved_role_devices(self) -> Dict[str, str]:
        return {role: self.device_for_role(role) for role in SEQUENTIAL_ROLES}


@dataclass
class EdgeDtypeMetadata:
    transport_dtype: str
    consumer_dtype: str


@dataclass
class RelayEmission:
    """Live relay tensors plus the exact consumer-interface representation."""

    transport: Any
    receiver: Any
    transport_dtype: str
    consumer_dtype: str


@dataclass
class ForcedChoiceBatch:
    labels: Tuple[str, ...]
    scores: Any
    summed_logprobs: Any
    mean_logprobs: Any
    token_counts: Any
    predictions: Tuple[Optional[str], ...]
    score_ties: Tuple[bool, ...]
    encodings: Mapping[str, CandidateEncoding]
    metadata: Mapping[str, Any]

    def detached(self) -> "ForcedChoiceBatch":
        if torch is None:
            return self

        def maybe(value: Any) -> Any:
            return value.detach().to(device="cpu") if torch.is_tensor(value) else value

        return ForcedChoiceBatch(
            labels=self.labels,
            scores=maybe(self.scores),
            summed_logprobs=maybe(self.summed_logprobs),
            mean_logprobs=maybe(self.mean_logprobs),
            token_counts=maybe(self.token_counts),
            predictions=self.predictions,
            score_ties=self.score_ties,
            encodings=self.encodings,
            metadata=dict(self.metadata),
        )

    def margins(self, gold_labels: Sequence[str]) -> List[Dict[str, float]]:
        if len(gold_labels) != len(self.predictions):
            raise ValueError("gold_labels length does not match score batch")
        values = self.scores
        if torch is not None and torch.is_tensor(values):
            values = values.detach().float().cpu().tolist()
        return [choice_margins(row, gold, self.labels) for row, gold in zip(values, gold_labels)]


@dataclass
class CleanTrajectory:
    schema_version: str
    runtime_version: str
    rounds: int
    sample_ids: List[str]
    raw_sample_ids: List[str]
    raw_indices: List[int]
    questions: List[str]
    gold_labels: List[str]
    option_permutations: List[Any]
    choice_metadata: List[Any]
    execution_manifest_hash: str
    ordered_batch_ids: List[Any]
    batch_boundaries: List[Tuple[int, int]]
    analysis_eligibility_mask: List[bool]
    transport_messages: Dict[Any, Any]
    receiver_reference_messages: Dict[Any, Any]
    edge_dtypes: Dict[Any, EdgeDtypeMetadata]
    clean_scoring: ForcedChoiceBatch
    clean_margins: List[Dict[str, float]]
    clean_generation_audit: List[Mapping[str, Any]]
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = set(lr.valid_edges(self.rounds))
        if set(self.transport_messages) != expected:
            missing = expected - set(self.transport_messages)
            extra = set(self.transport_messages) - expected
            raise ValueError(f"Invalid transport edge set; missing={missing}, extra={extra}")
        if set(self.receiver_reference_messages) != expected:
            missing = expected - set(self.receiver_reference_messages)
            extra = set(self.receiver_reference_messages) - expected
            raise ValueError(f"Invalid receiver edge set; missing={missing}, extra={extra}")
        if set(self.edge_dtypes) != expected:
            raise ValueError("edge_dtypes must describe every and only valid edge")
        count = len(self.sample_ids)
        fields = (
            self.raw_sample_ids,
            self.raw_indices,
            self.questions,
            self.gold_labels,
            self.option_permutations,
            self.choice_metadata,
            self.analysis_eligibility_mask,
        )
        if any(len(value) != count for value in fields):
            raise ValueError("All trajectory row fields must have the same length")
        if len(set(self.sample_ids)) != count or len(set(self.raw_sample_ids)) != count:
            raise ValueError("Trajectory sample_ids and raw_sample_ids must each be unique")
        validate_batch_boundaries(self.batch_boundaries, count)

    def message(self, edge: Any, *, receiver: bool = True) -> Any:
        parsed = lr.parse_edge(edge) if not isinstance(edge, lr.Edge) else edge
        lr.validate_edge(parsed, self.rounds)
        mapping = self.receiver_reference_messages if receiver else self.transport_messages
        return mapping[parsed]

    def dtype_metadata(self, edge: Any) -> EdgeDtypeMetadata:
        parsed = lr.parse_edge(edge) if not isinstance(edge, lr.Edge) else edge
        lr.validate_edge(parsed, self.rounds)
        return self.edge_dtypes[parsed]

    @property
    def ordered_cohort_hash(self) -> str:
        return _stable_json_hash(self.sample_ids)

    @property
    def batch_boundary_hash(self) -> str:
        return _stable_json_hash(self.batch_boundaries)


@dataclass
class ReplayIntervention:
    mode: str = "identity"
    delta: Any = None
    replacement: Any = None
    donor_indices: Optional[Sequence[int]] = None
    seed: int = 42
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayResult:
    edge: Any
    schedule: Tuple[ReplayStep, ...]
    scoring: ForcedChoiceBatch
    margins: List[Dict[str, float]]
    intervention_metadata: List[Mapping[str, Any]]
    intervened_receiver: Any
    recomputed_transport_messages: Dict[Any, Any]
    recomputed_receiver_messages: Dict[Any, Any]
    generation_audit: List[Mapping[str, Any]] = field(default_factory=list)


@dataclass
class _ReplayTerminalState:
    """Live downstream replay state before terminal candidate scoring."""

    edge: Any
    schedule: Tuple[ReplayStep, ...]
    boundaries: List[Tuple[int, int]]
    intervention_metadata: List[Mapping[str, Any]]
    intervened_receiver: Any
    terminal_receiver: Any
    recomputed_transport_messages: Dict[Any, Any]
    recomputed_receiver_messages: Dict[Any, Any]


@dataclass
class GradientResult:
    edge: Any
    gold_label: str
    target_label: str
    objective_name: str
    objective_value: float
    gradient: Any
    gradient_norm: float
    autograd_semantics: str
    sample_index: int = 0
    sample_id: Optional[str] = None


@dataclass
class PGDTargetResult:
    target_label: str
    initial_margin: float
    final_margin: float
    improved: bool
    requested_delta_norm: float
    realized_delta_norm: float
    budget: float
    budget_respected: bool
    scores: List[float]
    adversarial_receiver: Any


@dataclass
class PGDResult:
    edge: Any
    epsilon: float
    subspace: str
    q: int
    steps: int
    step_size: float
    autograd_semantics: str
    targets: List[PGDTargetResult]
    strongest_target: Optional[str]
    sample_index: int = 0
    sample_id: Optional[str] = None


@dataclass
class AntitheticProbeResult:
    edge: Any
    h: float
    global_seed: int
    probe_seed: int
    direction_id: int
    subspace: Mapping[str, Any]
    plus: ReplayResult
    minus: ReplayResult
    plus_diagnostics: List[Mapping[str, Any]]
    minus_diagnostics: List[Mapping[str, Any]]
    pair_diagnostics: List[Mapping[str, Any]]


class InterventionUnavailable(RuntimeError):
    """A control requested a scientifically invalid fallback-free intervention."""

    def __init__(self, message: str, metadata: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(message)
        self.metadata = list(metadata)


def _default_boundaries(total: int, batch_size: int) -> List[Tuple[int, int]]:
    return [(start, min(start + batch_size, total)) for start in range(0, total, batch_size)]


def validate_batch_boundaries(
    boundaries: Sequence[Sequence[int]],
    total: int,
) -> List[Tuple[int, int]]:
    normalized = [(int(pair[0]), int(pair[1])) for pair in boundaries]
    cursor = 0
    for start, end in normalized:
        if start != cursor or end <= start or end > total:
            raise ValueError(
                "Batch boundaries must be a contiguous, ordered, exact partition "
                f"of [0,{total}); got {(start, end)} after cursor {cursor}"
            )
        cursor = end
    if cursor != total or (total and not normalized):
        raise ValueError(f"Batch boundaries end at {cursor}, expected {total}")
    return normalized


def _dtype_name(dtype: Any) -> str:
    text = str(dtype)
    return text.removeprefix("torch.")


def _torch_dtype_from_name(name: str) -> Any:
    t = _require_torch()
    normalized = str(name).removeprefix("torch.").lower()
    aliases = {
        "float": "float32",
        "half": "float16",
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }
    normalized = aliases.get(normalized, normalized)
    value = getattr(t, normalized, None)
    if value is None or not isinstance(value, t.dtype):
        raise ValueError(f"Unsupported stored torch dtype: {name!r}")
    return value


def _module_dtype(module: Any, fallback: Any) -> Any:
    try:
        parameter = next(module.parameters())
    except (AttributeError, StopIteration):
        return fallback
    return parameter.dtype


def _run_adapter(module: Any, value: Any, output_dtype: Any) -> Any:
    module_dtype = _module_dtype(module, value.dtype)
    adapted = module(value.to(dtype=module_dtype))
    return adapted.to(dtype=output_dtype) if adapted.dtype != output_dtype else adapted


def _pad_left_ids(
    sequences: Sequence[Sequence[int]],
    pad_id: int,
    device: Any,
) -> Tuple[Any, Any]:
    t = _require_torch()
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("Cannot pad an empty token sequence")
    maximum = max(len(sequence) for sequence in sequences)
    ids = t.full((len(sequences), maximum), int(pad_id), dtype=t.long, device=device)
    mask = t.zeros((len(sequences), maximum), dtype=t.long, device=device)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        ids[row, maximum - length :] = t.as_tensor(sequence, dtype=t.long, device=device)
        mask[row, maximum - length :] = 1
    return ids, mask


def _pad_left_embeds(sequences: Sequence[Any], device: Any) -> Tuple[Any, Any, List[int]]:
    t = _require_torch()
    if not sequences or any(int(sequence.size(0)) == 0 for sequence in sequences):
        raise ValueError("Cannot pad an empty embedding sequence")
    hidden = int(sequences[0].size(-1))
    dtype = sequences[0].dtype
    lengths = [int(sequence.size(0)) for sequence in sequences]
    maximum = max(lengths)
    embeds = t.zeros((len(sequences), maximum, hidden), dtype=dtype, device=device)
    mask = t.zeros((len(sequences), maximum), dtype=t.long, device=device)
    for row, (sequence, length) in enumerate(zip(sequences, lengths)):
        if int(sequence.size(-1)) != hidden:
            raise ValueError("Embedding sequences have inconsistent hidden dimensions")
        embeds[row, maximum - length :] = sequence.to(device=device, dtype=dtype)
        mask[row, maximum - length :] = 1
    return embeds, mask, lengths


def _token_ids_to_embeds(embed_layer: Any, ids: Sequence[int], device: Any, dtype: Any) -> Any:
    t = _require_torch()
    hidden = int(embed_layer.weight.size(-1))
    if not ids:
        return t.empty((0, hidden), device=device, dtype=dtype)
    token_tensor = t.as_tensor(ids, dtype=t.long, device=device).unsqueeze(0)
    return embed_layer(token_tensor)[0].to(dtype=dtype)


def _normalise_template_text(tokenizer: Any, value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return _normalise_template_text(tokenizer, value["input_ids"])
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if not value:
            return ""
        if isinstance(value[0], list):
            if len(value) != 1:
                raise ValueError("Expected one chat template result")
            return _normalise_template_text(tokenizer, value[0])
        if isinstance(value[0], str):
            return "".join(value)
        return tokenizer.decode([int(item) for item in value], skip_special_tokens=False)
    raise TypeError(f"Unsupported chat-template result: {type(value)!r}")


def _apply_chat_template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    tokenize: bool,
    enable_thinking: bool,
) -> Any:
    kwargs = {
        "tokenize": tokenize,
        "add_generation_prompt": True,
        "enable_thinking": bool(enable_thinking),
    }
    def invoke(template_messages: Sequence[Mapping[str, str]]) -> Any:
        try:
            return tokenizer.apply_chat_template(list(template_messages), **kwargs)
        except TypeError as error:
            if "enable_thinking" not in str(error):
                raise
            fallback = dict(kwargs)
            fallback.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(list(template_messages), **fallback)

    try:
        return invoke(messages)
    except Exception as error:
        if "Conversation roles must alternate" not in str(error):
            raise
        system_parts: List[str] = []
        merged: List[Mapping[str, str]] = []
        inserted = False
        for message in messages:
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            if role == "system":
                if content:
                    system_parts.append(content)
                continue
            if role == "user" and system_parts and not inserted:
                merged.append(
                    {"role": "user", "content": "\n\n".join(system_parts + [content])}
                )
                inserted = True
            else:
                merged.append(dict(message))
        if system_parts and not inserted:
            merged.insert(0, {"role": "user", "content": "\n\n".join(system_parts)})
        return invoke(merged)


def _render_chat_text(tokenizer: Any, system_prompt: str, user_prompt: str, thinking: bool) -> str:
    result = _apply_chat_template(
        tokenizer,
        (
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ),
        tokenize=False,
        enable_thinking=thinking,
    )
    return _normalise_template_text(tokenizer, result)


def _render_chat_ids(tokenizer: Any, system_prompt: str, user_prompt: str, thinking: bool) -> List[int]:
    result = _apply_chat_template(
        tokenizer,
        (
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ),
        tokenize=True,
        enable_thinking=thinking,
    )
    if isinstance(result, str):
        return _tokenize_no_specials(tokenizer, result)[0]
    return _normalise_ids(result)


def _split_prompt_by_slots(
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    slots: Sequence[str],
    thinking: bool,
) -> List[List[int]]:
    # This intentionally mirrors the release path: render the full chat text,
    # locate literal slot sentinels, then tokenize each surrounding text segment.
    rendered = _render_chat_text(tokenizer, system_prompt, user_prompt, thinking)
    pieces: List[str] = []
    cursor = 0
    for slot in slots:
        position = rendered.find(slot, cursor)
        if position < 0:
            raise RuntimeError(f"Could not locate latent slot {slot!r} in rendered prompt")
        pieces.append(rendered[cursor:position])
        cursor = position + len(slot)
    pieces.append(rendered[cursor:])
    return [_tokenize_no_specials(tokenizer, piece)[0] if piece else [] for piece in pieces]


class LinkRadiusRuntime:
    """Persistent implementation of the released sequential latent chronology.

    Constructing the runtime never loads a model.  This is important for invalid
    grid entries: :meth:`prevalidate_edges`, :meth:`capture_clean`, and
    :meth:`replay` validate the complete edge request before any lazy load.
    ``system`` can be a real :class:`LoadedMASSystem` or a duck-typed toy system.
    """

    def __init__(self, config: Optional[RuntimeConfig] = None, *, system: Any = None) -> None:
        self.config = config or RuntimeConfig()
        # Force canonical edge construction now, before a caller can load models.
        self._valid_edges = tuple(lr.valid_edges(self.config.rounds))
        self.system = system
        self._owns_system = False
        self._stage_audit_hook: Optional[Callable[[Mapping[str, Any]], None]] = None
        if system is not None:
            self._validate_system(system)
            self._freeze_system_parameters(system)

    @classmethod
    def from_repository(
        cls,
        config: Optional[RuntimeConfig] = None,
        *,
        requested_edges: Sequence[Any] = (),
        trust_remote_code: bool = True,
    ) -> "LinkRadiusRuntime":
        runtime = cls(config=config)
        runtime.prevalidate_edges(requested_edges)
        runtime.load_system(trust_remote_code=trust_remote_code)
        return runtime

    def prevalidate_edges(self, edges: Iterable[Any]) -> Tuple[Any, ...]:
        parsed: List[Any] = []
        for value in edges:
            edge = lr.parse_edge(value) if not isinstance(value, lr.Edge) else value
            lr.validate_edge(edge, self.config.rounds)
            parsed.append(edge)
        return tuple(parsed)

    def load_system(self, *, trust_remote_code: bool = True) -> Any:
        if self.system is not None:
            return self.system
        _require_torch()
        # Imports are intentionally lazy so CPU-only grid tools do not require
        # torch/transformers/datasets. Package-relative imports are supported by
        # system_loader, with this path bootstrap retained for old checkouts.
        try:
            from .. import system_loader
        except ImportError:
            package_dir = Path(__file__).resolve().parents[1]
            if str(package_dir) not in sys.path:
                sys.path.insert(0, str(package_dir))
            from .. import system_loader
        from . import inference_mas as base_runtime

        base_runtime.configure_runtime_reproducibility(
            int(self.config.seed),
            bool(self.config.deterministic),
        )
        self.system = system_loader.load_mas_system(
            style=self.config.style,
            dataset=self.config.dataset,
            device=self.config.device,
            role_devices=self.config.resolved_role_devices(),
            dtype=self.config.dtype,
            outer_dtype=self.config.outer_dtype,
            trust_remote_code=trust_remote_code,
        )
        self._owns_system = True
        self._validate_system(self.system)
        self._freeze_system_parameters(self.system)
        return self.system

    def unload(self) -> None:
        if self.system is None:
            return
        if self._owns_system:
            from .. import system_loader

            system_loader.unload_mas_system(self.system)
        self.system = None
        self._owns_system = False

    close = unload

    def __enter__(self) -> "LinkRadiusRuntime":
        if self.system is None:
            self.load_system()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.unload()

    def _validate_system(self, system: Any) -> None:
        family = getattr(system, "family", "sequential")
        if family != "sequential":
            raise ValueError(f"LinkRadiusRuntime requires a sequential system, got {family!r}")
        agents = getattr(system, "agents", {})
        adapters = getattr(system, "outer_adapters", {})
        missing_agents = {"planner", "critic", "solver"} - set(agents)
        missing_adapters = {"outer_12", "outer_23", "outer_31"} - set(adapters)
        if missing_agents or missing_adapters:
            raise ValueError(
                f"Incomplete sequential system: agents={missing_agents}, adapters={missing_adapters}"
            )

    @staticmethod
    def _freeze_system_parameters(system: Any) -> None:
        modules: List[Any] = []
        for agent in system.agents.values():
            modules.extend((agent.model, agent.inner_adapter))
        modules.extend(system.outer_adapters.values())
        for module in modules:
            if module is None:
                continue
            if hasattr(module, "eval"):
                module.eval()
            if hasattr(module, "parameters"):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def _ensure_system(self) -> Any:
        return self.system if self.system is not None else self.load_system()

    def _validate_trajectory_compatibility(self, trajectory: CleanTrajectory) -> None:
        """Reject replay under settings different from the clean capture.

        The execution-manifest hash fixes rows and padding, but it does not by
        itself fix prompts, latent steps, scorer normalization, or the resolved
        system.  Those settings are therefore checked independently before any
        descendant computation or terminal-only gradient is allowed.
        """

        if not isinstance(trajectory, CleanTrajectory):
            raise TypeError("replay requires a versioned CleanTrajectory")
        if trajectory.schema_version != TRAJECTORY_VERSION:
            raise ValueError(
                f"trajectory schema mismatch: {trajectory.schema_version!r} != {TRAJECTORY_VERSION!r}"
            )
        if trajectory.runtime_version != RUNTIME_VERSION:
            raise ValueError(
                f"trajectory runtime mismatch: {trajectory.runtime_version!r} != {RUNTIME_VERSION!r}"
            )
        expected_config = asdict(self.config)
        actual_config = trajectory.provenance.get("runtime_config")
        actual_hash = trajectory.provenance.get("runtime_config_sha256")
        expected_hash = _stable_json_hash(expected_config)
        if not isinstance(actual_config, Mapping) or dict(actual_config) != expected_config:
            raise ValueError(
                "trajectory runtime configuration differs from the replay runtime"
            )
        if actual_hash != expected_hash:
            raise ValueError("trajectory runtime configuration hash is missing or stale")
        scorer = trajectory.provenance.get("scorer")
        if not isinstance(scorer, Mapping):
            raise ValueError("trajectory scorer provenance is missing")
        expected_scorer_fields = {
            "scorer_version": self.config.scorer_version,
            "normalization": self.config.scorer_normalization,
            "prefix": self.config.scorer_prefix,
            "tie_atol": float(self.config.score_tie_atol),
            "tie_rtol": float(self.config.score_tie_rtol),
        }
        for field_name, expected in expected_scorer_fields.items():
            if scorer.get(field_name) != expected:
                raise ValueError(
                    f"trajectory scorer setting {field_name!r} differs from replay"
                )
        if trajectory.provenance.get("scorer_hash") != _stable_json_hash(dict(scorer)):
            raise ValueError("trajectory scorer provenance hash is missing or stale")
        saved_resolution = trajectory.provenance.get("system_resolution")
        if not isinstance(saved_resolution, Mapping):
            raise ValueError("trajectory resolved-system provenance is missing")
        # A real captured trajectory records portable HF revision/blob identities.
        # Load the requested system now (after edge/config validation) so an
        # updated checkpoint/adapter cannot masquerade as a compatible replay.
        if (
            saved_resolution.get("identity_kind") == SYSTEM_IDENTITY_VERSION
            and self.system is None
        ):
            self.load_system()
        current_system = self._system_provenance()
        for field_name in ("model_hash", "adapter_hash", "system_resolution"):
            if trajectory.provenance.get(field_name) != current_system[field_name]:
                raise ValueError(
                    f"trajectory {field_name} differs from the currently resolved system"
                )

    def _system_provenance(self) -> Dict[str, Any]:
        """Describe the persistent system with cache-root-independent identities.

        Compatibility hashes contain repository/snapshot/blob or exact local
        content identities. Resolved absolute paths remain available under the
        separate ``system_diagnostic_paths`` field and never affect those hashes.
        """

        system = self.system
        if system is None:
            payload = {
                "identity_kind": "unresolved_runtime_without_loaded_system",
                "runtime_class": f"{type(self).__module__}.{type(self).__qualname__}",
                "style": self.config.style,
                "dataset": self.config.dataset,
            }
            return {
                "system_resolution": payload,
                "model_hash": _stable_json_hash({**payload, "component": "models"}),
                "adapter_hash": _stable_json_hash({**payload, "component": "adapters"}),
                "system_diagnostic_paths": {},
            }
        paths = getattr(system, "paths", None)
        if paths is None:
            payload: Dict[str, Any] = {
                "identity_kind": "unresolved_injected_system",
                "style": str(getattr(system, "style", self.config.style)),
                "family": str(getattr(system, "family", "sequential")),
                "dataset": str(getattr(system, "dataset", self.config.dataset)),
            }
            return {
                "system_resolution": payload,
                "model_hash": _stable_json_hash({**payload, "component": "models"}),
                "adapter_hash": _stable_json_hash({**payload, "component": "adapters"}),
                "system_diagnostic_paths": {},
            }

        def resolved_mapping(value: Any) -> Dict[str, str]:
            return {
                str(key): str(Path(path).resolve())
                for key, path in sorted(dict(value or {}).items())
            }

        repo_ids = {
            str(key): str(value)
            for key, value in sorted(dict(getattr(paths, "repo_ids", {}) or {}).items())
        }
        repo_paths = resolved_mapping(getattr(paths, "repo_paths", {}))
        inner_paths = resolved_mapping(getattr(paths, "inner_adapter_paths", {}))
        outer_paths = resolved_mapping(getattr(paths, "outer_adapter_paths", {}))
        common = {
            "identity_kind": SYSTEM_IDENTITY_VERSION,
            "style": str(getattr(paths, "style", self.config.style)),
            "family": str(getattr(paths, "family", getattr(system, "family", "sequential"))),
            "dataset": str(getattr(paths, "dataset", self.config.dataset)),
        }
        agent_roles = set(getattr(system, "agents", {}))
        model_repo_keys = [
            key for key in sorted(repo_paths) if key in agent_roles
        ]
        if not model_repo_keys:
            model_repo_keys = [key for key in sorted(repo_paths) if key != "outer"]
        model_artifacts = {
            key: _portable_artifact_identity(
                repo_paths[key],
                repo_id=repo_ids.get(key),
            )
            for key in model_repo_keys
        }
        inner_artifacts = {
            key: _portable_artifact_identity(
                path,
                repo_id=repo_ids.get(key),
            )
            for key, path in sorted(inner_paths.items())
        }
        outer_repo_id = repo_ids.get("outer")
        outer_artifacts = {
            key: _portable_artifact_identity(path, repo_id=outer_repo_id)
            for key, path in sorted(outer_paths.items())
        }
        model_identity = {**common, "artifacts": model_artifacts}
        adapter_identity = {
            **common,
            "inner_artifacts": inner_artifacts,
            "outer_artifacts": outer_artifacts,
        }
        resolution = {
            **common,
            "model_identity": model_identity,
            "adapter_identity": adapter_identity,
        }
        return {
            "system_resolution": resolution,
            "model_hash": _stable_json_hash(model_identity),
            "adapter_hash": _stable_json_hash(adapter_identity),
            "system_diagnostic_paths": {
                "repo_paths": repo_paths,
                "inner_adapter_paths": inner_paths,
                "outer_adapter_paths": outer_paths,
            },
        }

    @property
    def device(self) -> Any:
        """Backward-compatible alias for the planner's execution device."""

        return self.role_device("planner")

    def role_device(self, role: str) -> Any:
        """Return the actual device assigned to one sequential agent role."""

        t = _require_torch()
        normalized = str(role).strip().lower()
        configured = self.config.device_for_role(normalized)
        system = self.system
        if system is None:
            return t.device(configured)
        declared = dict(getattr(system, "role_devices", {}) or {}).get(normalized)
        if declared is not None:
            return t.device(declared)
        agent = system.agents[normalized]
        for parameter in agent.model.parameters():
            return parameter.device
        return t.device(configured)

    @staticmethod
    def edge_consumer_role(edge: Any) -> str:
        parsed = lr.parse_edge(edge) if not isinstance(edge, lr.Edge) else edge
        try:
            return EDGE_CONSUMER_ROLES[parsed.site]
        except KeyError as exc:  # Defensive for duck-typed/legacy Edge values.
            raise ValueError(f"unknown relay edge site: {parsed.site!r}") from exc

    def edge_consumer_device(self, edge: Any) -> Any:
        return self.role_device(self.edge_consumer_role(edge))

    def set_stage_audit_hook(
        self,
        hook: Optional[Callable[[Mapping[str, Any]], None]],
    ) -> None:
        self._stage_audit_hook = hook

    def _audit(self, action: str, round_idx: int, phase: str) -> None:
        if self._stage_audit_hook is not None:
            self._stage_audit_hook(
                {"action": action, "round_idx": int(round_idx), "phase": phase}
            )

    def _args(self) -> SimpleNamespace:
        return SimpleNamespace(
            choice_old_prompt=int(self.config.choice_old_prompt),
            solver_pre_question=int(self.config.solver_pre_question),
            mas_shape=self.config.mas_shape,
            role_response_regime=self.config.role_response_regime,
            role_response_regime_path=self.config.role_response_regime_path,
            planner_feedback_round_label_mode=self.config.round_label_mode,
        )

    def _decorate_prompt(self, prompt: str, role: str) -> str:
        from ..prompts import apply_role_response_regime

        prompt = apply_role_response_regime(
            prompt,
            regime=self.config.role_response_regime,
            role=role,
            custom_path=self.config.role_response_regime_path,
        )
        footer = str(self.config.prompt_footer or "").strip()
        return prompt.rstrip() + (f"\n\n{footer}" if footer else "")

    def _planner_initial_prompts(self, questions: Sequence[str]) -> List[List[int]]:
        from ..prompts import SYSTEM_PROMPT, build_math_planner_prompt

        tokenizer = self._ensure_system().agents["planner"].tokenizer
        return [
            _render_chat_ids(
                tokenizer,
                SYSTEM_PROMPT,
                self._decorate_prompt(build_math_planner_prompt(question), "planner"),
                self.config.enable_thinking,
            )
            for question in questions
        ]

    def _slot_prompts(
        self,
        role: str,
        questions: Sequence[str],
        round_idx: int,
    ) -> List[List[List[int]]]:
        from ..prompts import (
            FEEDBACK_SLOT,
            PLANNER_SLOT,
            REFINED_SLOT,
            SYSTEM_PROMPT,
            build_math_planner_prompt_with_feedback_slot,
            build_math_refiner_prompt_with_slot,
            build_math_solver_prompt_with_slots,
        )

        agent = self._ensure_system().agents[role]
        if role == "critic":
            slot = PLANNER_SLOT
            builder = lambda question: build_math_refiner_prompt_with_slot(question)
            prompt_role = "critic"
        elif role == "solver":
            slot = REFINED_SLOT
            builder = lambda question: build_math_solver_prompt_with_slots(
                question,
                args=self._args(),
                mas_shape=self.config.mas_shape,
            )
            prompt_role = "solver"
        elif role == "planner":
            slot = FEEDBACK_SLOT
            builder = lambda question: build_math_planner_prompt_with_feedback_slot(
                question,
                round_idx=round_idx,
                round_label_mode=self.config.round_label_mode,
            )
            prompt_role = "planner"
        else:
            raise ValueError(f"Unsupported sequential role: {role}")
        return [
            _split_prompt_by_slots(
                agent.tokenizer,
                SYSTEM_PROMPT,
                self._decorate_prompt(builder(question), prompt_role),
                (slot,),
                self.config.enable_thinking,
            )
            for question in questions
        ]

    def _latent_rollout(
        self,
        model: Any,
        inner_adapter: Any,
        input_embeds: Any,
        attention_mask: Any,
    ) -> Any:
        t = _require_torch()
        hidden_states: List[Any] = []
        for _ in range(int(self.config.latent_steps)):
            kwargs = {
                "inputs_embeds": input_embeds,
                "attention_mask": attention_mask,
                "output_hidden_states": True,
                "use_cache": False,
                "return_dict": True,
            }
            try:
                outputs = model(logits_to_keep=1, **kwargs)
            except TypeError:
                outputs = model(**kwargs)
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            hidden_states.append(last_hidden.unsqueeze(1))
            next_embed = _run_adapter(inner_adapter, last_hidden, input_embeds.dtype).unsqueeze(1)
            input_embeds = t.cat((input_embeds, next_embed), dim=1)
            next_mask = t.ones(
                (attention_mask.size(0), 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = t.cat((attention_mask, next_mask), dim=1)
        return t.cat(hidden_states, dim=1)

    def _emit_initial_planner(
        self,
        questions: Sequence[str],
        boundaries: Sequence[Tuple[int, int]],
        *,
        differentiable: bool,
    ) -> RelayEmission:
        t = _require_torch()
        system = self._ensure_system()
        agent = system.agents["planner"]
        consumer = system.agents["critic"]
        planner_device = self.role_device("planner")
        consumer_device = self.role_device("critic")
        embed_layer = agent.model.get_input_embeddings()
        embed_dtype = embed_layer.weight.dtype
        prompt_ids = self._planner_initial_prompts(questions)
        batches: List[Any] = []
        self._audit("planner_initial", 0, "start")
        with t.set_grad_enabled(differentiable):
            for start, end in boundaries:
                ids, mask = _pad_left_ids(
                    prompt_ids[start:end],
                    agent.tokenizer.pad_token_id,
                    planner_device,
                )
                hidden = self._latent_rollout(
                    agent.model,
                    agent.inner_adapter,
                    embed_layer(ids),
                    mask,
                )
                self_state = _run_adapter(agent.inner_adapter, hidden, embed_dtype)
                batches.append(
                    _run_adapter(system.outer_adapters["outer_12"], self_state, embed_dtype)
                )
        transport = t.cat(batches, dim=0)
        consumer_dtype = consumer.model.get_input_embeddings().weight.dtype
        receiver = transport.to(device=consumer_device, dtype=consumer_dtype)
        self._audit("planner_initial", 0, "end")
        return RelayEmission(
            transport=transport,
            receiver=receiver,
            transport_dtype=_dtype_name(transport.dtype),
            consumer_dtype=_dtype_name(consumer_dtype),
        )

    def _emit_from_slot(
        self,
        *,
        role: str,
        questions: Sequence[str],
        incoming: Any,
        round_idx: int,
        boundaries: Sequence[Tuple[int, int]],
        outer_key: str,
        consumer_role: str,
        transport_dtype: Any,
        action: str,
        differentiable: bool,
    ) -> RelayEmission:
        t = _require_torch()
        system = self._ensure_system()
        agent = system.agents[role]
        consumer = system.agents[consumer_role]
        role_device = self.role_device(role)
        consumer_device = self.role_device(consumer_role)
        embed_layer = agent.model.get_input_embeddings()
        embed_dtype = embed_layer.weight.dtype
        if int(incoming.size(-1)) != int(embed_layer.weight.size(-1)):
            raise ValueError(
                f"{action} incoming hidden dimension {incoming.size(-1)} does not "
                f"match {role} embeddings {embed_layer.weight.size(-1)}"
            )
        segments = self._slot_prompts(role, questions, round_idx)
        outputs: List[Any] = []
        self._audit(action, round_idx, "start")
        with t.set_grad_enabled(differentiable):
            for start, end in boundaries:
                sequences: List[Any] = []
                for index in range(start, end):
                    prefix_ids, suffix_ids = segments[index]
                    prefix = _token_ids_to_embeds(embed_layer, prefix_ids, role_device, embed_dtype)
                    suffix = _token_ids_to_embeds(embed_layer, suffix_ids, role_device, embed_dtype)
                    relay = incoming[index].to(device=role_device, dtype=embed_dtype)
                    sequences.append(t.cat((prefix, relay, suffix), dim=0))
                embedded, mask, _ = _pad_left_embeds(sequences, role_device)
                hidden = self._latent_rollout(
                    agent.model,
                    agent.inner_adapter,
                    embedded,
                    mask,
                )
                self_state = _run_adapter(agent.inner_adapter, hidden, embed_dtype)
                outputs.append(
                    _run_adapter(
                        system.outer_adapters[outer_key],
                        self_state,
                        transport_dtype,
                    )
                )
        transport = t.cat(outputs, dim=0)
        consumer_dtype = consumer.model.get_input_embeddings().weight.dtype
        receiver = transport.to(device=consumer_device, dtype=consumer_dtype)
        self._audit(action, round_idx, "end")
        return RelayEmission(
            transport=transport,
            receiver=receiver,
            transport_dtype=_dtype_name(transport.dtype),
            consumer_dtype=_dtype_name(consumer_dtype),
        )

    # Public stage methods are intentionally concrete and individually
    # overridable, which also makes the scheduler testable with tiny toy stages.
    def run_initial_planner(
        self,
        questions: Sequence[str],
        *,
        batch_boundaries: Optional[Sequence[Sequence[int]]] = None,
        differentiable: bool = False,
    ) -> RelayEmission:
        boundaries = validate_batch_boundaries(
            batch_boundaries or _default_boundaries(len(questions), self.config.batch_size),
            len(questions),
        )
        return self._emit_initial_planner(questions, boundaries, differentiable=differentiable)

    def run_critic(
        self,
        questions: Sequence[str],
        planner_message: Any,
        round_idx: int,
        *,
        batch_boundaries: Optional[Sequence[Sequence[int]]] = None,
        differentiable: bool = False,
    ) -> RelayEmission:
        system = self._ensure_system()
        boundaries = validate_batch_boundaries(
            batch_boundaries or _default_boundaries(len(questions), self.config.batch_size),
            len(questions),
        )
        critic_dtype = system.agents["critic"].model.get_input_embeddings().weight.dtype
        return self._emit_from_slot(
            role="critic",
            questions=questions,
            incoming=planner_message,
            round_idx=round_idx,
            boundaries=boundaries,
            outer_key="outer_23",
            consumer_role="solver",
            transport_dtype=critic_dtype,
            action="critic",
            differentiable=differentiable,
        )

    def run_solver_feedback(
        self,
        questions: Sequence[str],
        critic_message: Any,
        round_idx: int,
        *,
        batch_boundaries: Optional[Sequence[Sequence[int]]] = None,
        differentiable: bool = False,
    ) -> RelayEmission:
        t = _require_torch()
        boundaries = validate_batch_boundaries(
            batch_boundaries or _default_boundaries(len(questions), self.config.batch_size),
            len(questions),
        )
        return self._emit_from_slot(
            role="solver",
            questions=questions,
            incoming=critic_message,
            round_idx=round_idx,
            boundaries=boundaries,
            outer_key="outer_31",
            consumer_role="planner",
            transport_dtype=t.float32,
            action="solver_feedback",
            differentiable=differentiable,
        )

    def run_planner_feedback(
        self,
        questions: Sequence[str],
        solver_message: Any,
        round_idx: int,
        *,
        batch_boundaries: Optional[Sequence[Sequence[int]]] = None,
        differentiable: bool = False,
    ) -> RelayEmission:
        system = self._ensure_system()
        boundaries = validate_batch_boundaries(
            batch_boundaries or _default_boundaries(len(questions), self.config.batch_size),
            len(questions),
        )
        planner_dtype = system.agents["planner"].model.get_input_embeddings().weight.dtype
        return self._emit_from_slot(
            role="planner",
            questions=questions,
            incoming=solver_message,
            round_idx=round_idx,
            boundaries=boundaries,
            outer_key="outer_12",
            consumer_role="critic",
            transport_dtype=planner_dtype,
            action="planner_feedback",
            differentiable=differentiable,
        )

    def build_terminal_input_embeddings(
        self,
        questions: Sequence[str],
        critic_message: Any,
    ) -> List[Any]:
        """Build the exact final-solver embedding sequences before continuation."""

        t = _require_torch()
        system = self._ensure_system()
        agent = system.agents["solver"]
        solver_device = self.role_device("solver")
        embed_layer = agent.model.get_input_embeddings()
        embed_dtype = embed_layer.weight.dtype
        if len(questions) != int(critic_message.size(0)):
            raise ValueError("Question and terminal relay batch sizes do not agree")
        if int(critic_message.size(-1)) != int(embed_layer.weight.size(-1)):
            raise ValueError("Terminal c2s dimension does not match solver embeddings")
        segments = self._slot_prompts("solver", questions, self.config.rounds - 1)
        sequences: List[Any] = []
        for index, (prefix_ids, suffix_ids) in enumerate(segments):
            prefix = _token_ids_to_embeds(embed_layer, prefix_ids, solver_device, embed_dtype)
            suffix = _token_ids_to_embeds(embed_layer, suffix_ids, solver_device, embed_dtype)
            relay = critic_message[index].to(device=solver_device, dtype=embed_dtype)
            sequences.append(t.cat((prefix, relay, suffix), dim=0))
        return sequences

    def _scorer_metadata(
        self,
        tokenizer: Any,
        encodings: Mapping[str, CandidateEncoding],
    ) -> Dict[str, Any]:
        chat_template = getattr(tokenizer, "chat_template", None)
        verbalizers = {label: encoding.verbalizer for label, encoding in encodings.items()}
        return {
            "scorer_version": self.config.scorer_version,
            "normalization": self.config.scorer_normalization,
            "prefix": self.config.scorer_prefix,
            "scorer_prefix_sha256": _sha256_text(self.config.scorer_prefix),
            "chat_template_sha256": _stable_json_hash(chat_template),
            "verbalizer_sha256": {
                label: _sha256_text(value) for label, value in verbalizers.items()
            },
            "verbalizers_sha256": _stable_json_hash(verbalizers),
            "joint_token_ids": {
                label: list(encoding.token_ids)
                for label, encoding in encodings.items()
            },
            "candidate_token_ids": {
                label: list(encoding.candidate_token_ids)
                for label, encoding in encodings.items()
            },
            "candidate_token_spans": {
                label: [encoding.candidate_start, encoding.candidate_end]
                for label, encoding in encodings.items()
            },
            "candidate_token_counts": {
                label: encoding.token_count
                for label, encoding in encodings.items()
            },
            "candidate_span_methods": {
                label: encoding.span_method
                for label, encoding in encodings.items()
            },
            "tie_atol": float(self.config.score_tie_atol),
            "tie_rtol": float(self.config.score_tie_rtol),
            "joint_tokenization": True,
            "causal_alignment": "logits[t-1]_scores_token[t]",
            "log_softmax_dtype": "float32",
        }

    def score_terminal(
        self,
        questions: Sequence[str],
        critic_message: Any,
        *,
        batch_boundaries: Optional[Sequence[Sequence[int]]] = None,
        differentiable: bool = False,
        verbalizers: Mapping[str, str] = DEFAULT_VERBALIZERS,
        candidate_labels: Optional[Sequence[str]] = None,
    ) -> ForcedChoiceBatch:
        """Teacher-force and score every frozen A/B/C/D continuation.

        The prefix and candidate are tokenized jointly. Candidate token log
        probabilities are gathered with causal next-token alignment after an
        explicit float32 ``log_softmax``.  The returned score tensor retains its
        graph only when ``differentiable=True``. ``candidate_labels`` is an
        internal memory-bounded gradient facility: tokenization still uses the
        complete frozen A/B/C/D comparison set, but only the requested columns
        are forwarded through the solver. Ordinary scoring always leaves it
        unset and therefore retains the exact four-way scorer.
        """

        t = _require_torch()
        system = self._ensure_system()
        agent = system.agents["solver"]
        model = agent.model
        tokenizer = agent.tokenizer
        solver_device = self.role_device("solver")
        embed_layer = model.get_input_embeddings()
        embed_dtype = embed_layer.weight.dtype
        boundaries = validate_batch_boundaries(
            batch_boundaries or _default_boundaries(len(questions), self.config.batch_size),
            len(questions),
        )
        normalized_verbalizers = {
            str(label).upper(): str(value) for label, value in verbalizers.items()
        }
        if tuple(normalized_verbalizers) != CHOICE_LABELS:
            raise ValueError("The GPQA scorer requires ordered verbalizers A, B, C, D")
        selected_labels = tuple(
            str(label).upper()
            for label in (CHOICE_LABELS if candidate_labels is None else candidate_labels)
        )
        if (
            not selected_labels
            or len(set(selected_labels)) != len(selected_labels)
            or set(selected_labels) - set(CHOICE_LABELS)
        ):
            raise ValueError(
                "candidate_labels must be a non-empty unique subset of A/B/C/D"
            )
        encodings = tokenize_joint_candidates(
            tokenizer,
            self.config.scorer_prefix,
            normalized_verbalizers,
        )
        base_sequences = self.build_terminal_input_embeddings(questions, critic_message)
        sum_columns: List[Any] = []
        mean_columns: List[Any] = []
        count_columns: List[Any] = []
        self._audit("score_final", self.config.rounds - 1, "start")
        with t.set_grad_enabled(differentiable):
            for label in selected_labels:
                encoding = encodings[label]
                continuation = _token_ids_to_embeds(
                    embed_layer,
                    encoding.token_ids,
                    solver_device,
                    embed_dtype,
                )
                batch_sums: List[Any] = []
                batch_means: List[Any] = []
                batch_counts: List[Any] = []
                for start, end in boundaries:
                    sequences = [
                        t.cat((base_sequences[index], continuation), dim=0)
                        for index in range(start, end)
                    ]
                    padded, mask, lengths = _pad_left_embeds(sequences, solver_device)
                    keep = max(len(encoding.token_ids) + 1, encoding.token_count + 1)
                    kwargs = {
                        "inputs_embeds": padded,
                        "attention_mask": mask,
                        "use_cache": False,
                        "return_dict": True,
                    }
                    try:
                        output = model(logits_to_keep=keep, **kwargs)
                    except TypeError:
                        output = model(**kwargs)
                    logits = output.logits
                    # Models supporting logits_to_keep return a suffix. Models
                    # ignoring it return the full sequence; both are handled.
                    logit_origin = int(padded.size(1)) - int(logits.size(1))
                    target_positions: List[List[int]] = []
                    target_ids: List[List[int]] = []
                    for sequence_length in lengths:
                        pad_offset = int(padded.size(1)) - sequence_length
                        base_length = sequence_length - len(encoding.token_ids)
                        positions = [
                            pad_offset + base_length + token_index - logit_origin
                            for token_index in range(
                                encoding.candidate_start,
                                encoding.candidate_end,
                            )
                        ]
                        target_positions.append(positions)
                        target_ids.append(list(encoding.candidate_token_ids))
                    token_log_probs = causal_token_log_probs(
                        logits,
                        t.as_tensor(target_ids, dtype=t.long, device=solver_device),
                        t.as_tensor(target_positions, dtype=t.long, device=solver_device),
                    )
                    batch_sums.append(token_log_probs.sum(dim=-1))
                    batch_means.append(token_log_probs.mean(dim=-1))
                    batch_counts.append(
                        t.full(
                            (end - start,),
                            encoding.token_count,
                            dtype=t.long,
                            device=solver_device,
                        )
                    )
                sum_columns.append(t.cat(batch_sums, dim=0))
                mean_columns.append(t.cat(batch_means, dim=0))
                count_columns.append(t.cat(batch_counts, dim=0))
        summed = t.stack(sum_columns, dim=-1).float()
        means = t.stack(mean_columns, dim=-1).float()
        counts = t.stack(count_columns, dim=-1)
        scores = means if self.config.scorer_normalization == "mean" else summed
        rows = scores.detach().float().cpu().tolist()
        selections = [
            prediction_from_scores(
                row,
                selected_labels,
                atol=self.config.score_tie_atol,
                rtol=self.config.score_tie_rtol,
            )
            for row in rows
        ]
        self._audit("score_final", self.config.rounds - 1, "end")
        return ForcedChoiceBatch(
            labels=selected_labels,
            scores=scores,
            summed_logprobs=summed,
            mean_logprobs=means,
            token_counts=counts,
            predictions=tuple(item[0] for item in selections),
            score_ties=tuple(item[1] for item in selections),
            encodings=encodings,
            metadata=self._scorer_metadata(tokenizer, encodings),
        )

    @staticmethod
    def _generation_kwargs(tokenizer: Any, config: RuntimeConfig, max_tokens: int) -> Dict[str, Any]:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_tokens),
            "do_sample": bool(config.do_sample),
            "pad_token_id": pad_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.0,
        }
        if config.do_sample:
            kwargs.update(temperature=float(config.temperature), top_p=float(config.top_p))
        return kwargs

    def generate_terminal(
        self,
        questions: Sequence[str],
        critic_message: Any,
        *,
        batch_boundaries: Optional[Sequence[Sequence[int]]] = None,
    ) -> List[str]:
        """Run the ordinary deterministic release generation at terminal c2s."""

        t = _require_torch()
        system = self._ensure_system()
        agent = system.agents["solver"]
        solver_device = self.role_device("solver")
        boundaries = validate_batch_boundaries(
            batch_boundaries or _default_boundaries(len(questions), self.config.batch_size),
            len(questions),
        )
        sequences = self.build_terminal_input_embeddings(questions, critic_message)
        kwargs = self._generation_kwargs(
            agent.tokenizer,
            self.config,
            self.config.max_new_tokens,
        )
        texts: List[str] = []
        self._audit("generate_final", self.config.rounds - 1, "start")
        with t.no_grad():
            for start, end in boundaries:
                padded, mask, _ = _pad_left_embeds(sequences[start:end], solver_device)
                generated = agent.model.generate(
                    inputs_embeds=padded,
                    attention_mask=mask,
                    **kwargs,
                )
                generated_ids = generated.sequences if hasattr(generated, "sequences") else generated
                if int(generated_ids.size(1)) > int(self.config.max_new_tokens):
                    generated_ids = generated_ids[:, int(mask.size(1)) :]
                decoded = agent.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                texts.extend(str(text).strip() for text in decoded)
        self._audit("generate_final", self.config.rounds - 1, "end")
        return texts

    def _retry_generated_choices(self, outputs: Sequence[str]) -> Tuple[List[str], List[bool]]:
        """Mirror the release ``--ans`` choice retry using the persistent solver."""

        t = _require_torch()
        from .answer_utils import extract_choice_answer

        pending = [
            index
            for index, text in enumerate(outputs)
            if extract_choice_answer(text, default=None) is None
        ]
        attempted = [index in set(pending) for index in range(len(outputs))]
        if not pending:
            return list(outputs), attempted
        agent = self._ensure_system().agents["solver"]
        solver_device = self.role_device("solver")
        prompts = [
            f"{str(outputs[index]).rstrip()}\n{self.config.scorer_prefix}" for index in pending
        ]
        kwargs = self._generation_kwargs(
            agent.tokenizer,
            self.config,
            self.config.retry_max_new_tokens,
        )
        updated = list(outputs)
        with t.no_grad():
            for offset in range(0, len(prompts), self.config.batch_size):
                prompt_batch = prompts[offset : offset + self.config.batch_size]
                encoded = agent.tokenizer(
                    prompt_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                )
                if hasattr(encoded, "to"):
                    encoded = encoded.to(solver_device)
                input_ids = encoded["input_ids"].to(solver_device)
                mask = encoded["attention_mask"].to(solver_device)
                generated = agent.model.generate(
                    input_ids=input_ids,
                    attention_mask=mask,
                    **kwargs,
                )
                generated_ids = generated.sequences if hasattr(generated, "sequences") else generated
                suffix_ids = generated_ids[:, int(input_ids.size(1)) :]
                suffixes = agent.tokenizer.batch_decode(suffix_ids, skip_special_tokens=True)
                for local_index, suffix in enumerate(suffixes):
                    output_index = pending[offset + local_index]
                    updated[output_index] = prompt_batch[local_index] + str(suffix)
        return updated, attempted

    def audit_ordinary_generation(
        self,
        questions: Sequence[str],
        critic_message: Any,
        *,
        batch_boundaries: Optional[Sequence[Sequence[int]]] = None,
    ) -> List[Mapping[str, Any]]:
        first_pass = self.generate_terminal(
            questions,
            critic_message,
            batch_boundaries=batch_boundaries,
        )
        if self.config.answer_retry:
            final_outputs, attempted = self._retry_generated_choices(first_pass)
        else:
            final_outputs, attempted = list(first_pass), [False] * len(first_pass)
        audits: List[Mapping[str, Any]] = []
        for first, final, did_retry in zip(first_pass, final_outputs, attempted):
            first_result = lr.parse_strict_choice(first)
            final_result = lr.parse_strict_choice(final)
            audits.append(
                {
                    "first_pass_text": first,
                    "first_pass_strict": asdict(first_result),
                    "retry_attempted": bool(did_retry),
                    "retry_text": final if did_retry else None,
                    "final_text": final,
                    "strict_choice": final_result.choice,
                    "answer_invalid": bool(final_result.answer_invalid),
                    "answer_conflict": bool(final_result.answer_conflict),
                    "strict_result": asdict(final_result),
                    "checker_version": final_result.checker_version,
                }
            )
        return audits

    @staticmethod
    def _store_tensor(value: Any) -> Any:
        _require_torch()
        return value.detach().to(device="cpu", dtype=torch.float32).contiguous()

    @classmethod
    def _store_emission(
        cls,
        edge: Any,
        emission: RelayEmission,
        transport: Dict[Any, Any],
        receiver: Dict[Any, Any],
        dtypes: Dict[Any, EdgeDtypeMetadata],
    ) -> None:
        if edge in transport or edge in receiver or edge in dtypes:
            raise RuntimeError(f"Clean edge captured more than once: {edge.edge_id}")
        transport[edge] = cls._store_tensor(emission.transport)
        receiver[edge] = cls._store_tensor(emission.receiver)
        dtypes[edge] = EdgeDtypeMetadata(
            transport_dtype=emission.transport_dtype,
            consumer_dtype=emission.consumer_dtype,
        )

    def capture_clean(
        self,
        *,
        sample_ids: Sequence[str],
        questions: Sequence[str],
        gold_labels: Sequence[str],
        raw_sample_ids: Optional[Sequence[str]] = None,
        raw_indices: Optional[Sequence[int]] = None,
        option_permutations: Optional[Sequence[Any]] = None,
        choice_metadata: Optional[Sequence[Any]] = None,
        execution_manifest_hash: str = "",
        ordered_batch_ids: Optional[Sequence[Any]] = None,
        batch_boundaries: Optional[Sequence[Sequence[int]]] = None,
        analysis_eligibility_mask: Optional[Sequence[bool]] = None,
        include_generation: bool = True,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> CleanTrajectory:
        """Execute and capture every valid clean relay exactly once."""

        count = len(sample_ids)
        if len(questions) != count or len(gold_labels) != count:
            raise ValueError("sample_ids, questions, and gold_labels lengths must agree")
        if count == 0:
            raise ValueError("A clean trajectory cannot be empty")
        normalized_gold = [str(value).upper() for value in gold_labels]
        if any(value not in CHOICE_LABELS for value in normalized_gold):
            raise ValueError("Every GPQA gold label must be one of A/B/C/D")
        boundaries = validate_batch_boundaries(
            batch_boundaries or _default_boundaries(count, self.config.batch_size),
            count,
        )
        # Validate all canonical edges before _ensure_system can trigger a load.
        self.prevalidate_edges(self._valid_edges)
        transports: Dict[Any, Any] = {}
        receivers: Dict[Any, Any] = {}
        dtype_metadata: Dict[Any, EdgeDtypeMetadata] = {}

        p = self.run_initial_planner(
            questions,
            batch_boundaries=boundaries,
            differentiable=False,
        )
        edge = lr.Edge("p2c", 0)
        self._store_emission(edge, p, transports, receivers, dtype_metadata)
        current_p = p.receiver
        final_c = None
        for round_idx in range(self.config.rounds):
            if round_idx > 0:
                # current_s is assigned by every nonterminal preceding round.
                p = self.run_planner_feedback(
                    questions,
                    current_s,
                    round_idx,
                    batch_boundaries=boundaries,
                    differentiable=False,
                )
                edge = lr.Edge("p2c", round_idx)
                self._store_emission(edge, p, transports, receivers, dtype_metadata)
                current_p = p.receiver
            c = self.run_critic(
                questions,
                current_p,
                round_idx,
                batch_boundaries=boundaries,
                differentiable=False,
            )
            edge = lr.Edge("c2s", round_idx)
            self._store_emission(edge, c, transports, receivers, dtype_metadata)
            final_c = c.receiver
            if round_idx < self.config.rounds - 1:
                s = self.run_solver_feedback(
                    questions,
                    c.receiver,
                    round_idx,
                    batch_boundaries=boundaries,
                    differentiable=False,
                )
                edge = lr.Edge("s2p", round_idx)
                self._store_emission(edge, s, transports, receivers, dtype_metadata)
                current_s = s.receiver
        if final_c is None:
            raise RuntimeError("Sequential capture did not produce a terminal c2s relay")
        scoring = self.score_terminal(
            questions,
            final_c,
            batch_boundaries=boundaries,
            differentiable=False,
        ).detached()
        margins = scoring.margins(normalized_gold)
        generation = (
            self.audit_ordinary_generation(
                questions,
                final_c,
                batch_boundaries=boundaries,
            )
            if include_generation
            else []
        )
        raw_ids = list(raw_sample_ids or sample_ids)
        indices = list(raw_indices if raw_indices is not None else range(count))
        permutations = list(option_permutations or [None] * count)
        metadata = list(choice_metadata or [None] * count)
        mask = list(analysis_eligibility_mask or [True] * count)
        batch_ids = list(
            ordered_batch_ids
            if ordered_batch_ids is not None
            else range(len(boundaries))
        )
        if len(batch_ids) != len(boundaries):
            raise ValueError("ordered_batch_ids must have one entry per batch boundary")
        runtime_provenance = {
            "runtime_config": asdict(self.config),
            "runtime_config_sha256": _stable_json_hash(asdict(self.config)),
            "role_devices": self.config.resolved_role_devices(),
            "scorer": dict(scoring.metadata),
            "scorer_hash": _stable_json_hash(dict(scoring.metadata)),
            "transport_storage_dtype": "float32",
            "receiver_reference_storage_dtype": "float32",
            "receiver_reference_convention": "post_consumer_cast",
        }
        runtime_provenance.update(self._system_provenance())
        extra_provenance = dict(provenance or {})
        reserved = set(runtime_provenance) & set(extra_provenance)
        if reserved:
            raise ValueError(
                "caller provenance cannot replace runtime-owned fields: "
                + ", ".join(sorted(reserved))
            )
        runtime_provenance.update(extra_provenance)
        return CleanTrajectory(
            schema_version=TRAJECTORY_VERSION,
            runtime_version=RUNTIME_VERSION,
            rounds=self.config.rounds,
            sample_ids=list(sample_ids),
            raw_sample_ids=raw_ids,
            raw_indices=indices,
            questions=list(questions),
            gold_labels=normalized_gold,
            option_permutations=permutations,
            choice_metadata=metadata,
            execution_manifest_hash=str(execution_manifest_hash),
            ordered_batch_ids=batch_ids,
            batch_boundaries=boundaries,
            analysis_eligibility_mask=[bool(value) for value in mask],
            transport_messages=transports,
            receiver_reference_messages=receivers,
            edge_dtypes=dtype_metadata,
            clean_scoring=scoring,
            clean_margins=margins,
            clean_generation_audit=generation,
            provenance=runtime_provenance,
        )

    @staticmethod
    def _tensor_row_stats(value: Any) -> Tuple[float, float, float]:
        row = value.float()
        return (
            float(row.mean().detach().cpu()),
            float(row.std(unbiased=False).detach().cpu()),
            float(torch.linalg.vector_norm(row).detach().cpu()),
        )

    def _intervention_from_value(self, value: Any) -> ReplayIntervention:
        if isinstance(value, ReplayIntervention):
            return value
        if isinstance(value, str):
            return ReplayIntervention(mode=value)
        if callable(value):
            return ReplayIntervention(mode="callable", replacement=value)
        if torch is not None and torch.is_tensor(value):
            return ReplayIntervention(mode="replacement", replacement=value)
        raise TypeError(f"Unsupported replay intervention: {type(value)!r}")

    def _apply_intervention(
        self,
        trajectory: CleanTrajectory,
        edge: Any,
        intervention_value: Any,
    ) -> Tuple[Any, List[Mapping[str, Any]]]:
        t = _require_torch()
        intervention = self._intervention_from_value(intervention_value)
        clean_stored = trajectory.message(edge, receiver=True)
        execution_device = (
            self.edge_consumer_device(edge)
            if self.system is not None
            else getattr(clean_stored, "device", t.device("cpu"))
        )
        clean = clean_stored.to(device=execution_device, dtype=t.float32)
        consumer_dtype = _torch_dtype_from_name(trajectory.dtype_metadata(edge).consumer_dtype)
        mode = str(intervention.mode).strip().lower()
        row_metadata: List[Dict[str, Any]] = []

        if mode == "identity":
            requested = clean
        elif mode == "zero":
            requested = t.zeros_like(clean)
        elif mode == "additive":
            if intervention.delta is None:
                raise ValueError("additive intervention requires delta")
            delta = intervention.delta
            if not t.is_tensor(delta):
                delta = t.as_tensor(delta, dtype=t.float32, device=execution_device)
            delta = delta.to(device=execution_device, dtype=t.float32)
            if delta.shape != clean.shape:
                if clean.size(0) == 1 and delta.shape == clean.shape[1:]:
                    delta = delta.unsqueeze(0)
                else:
                    raise ValueError(
                        f"Additive delta shape {tuple(delta.shape)} != relay {tuple(clean.shape)}"
                    )
            requested = clean + delta
        elif mode in {"replacement", "mismatch"}:
            if mode == "mismatch" and intervention.replacement is not None:
                raise ValueError(
                    "mode='mismatch' does not accept an unchecked replacement tensor; "
                    "provide deterministic donor_indices, or use mode='replacement' "
                    "after validating an external donor bank"
                )
            if mode == "replacement" and intervention.replacement is None:
                raise ValueError("replacement intervention requires replacement")
            if intervention.replacement is not None:
                replacement = intervention.replacement
                if callable(replacement):
                    replacement = replacement(clean, trajectory, edge)
                if not t.is_tensor(replacement):
                    replacement = t.as_tensor(replacement, dtype=t.float32)
                requested = replacement.to(device=execution_device, dtype=t.float32)
                if requested.shape != clean.shape:
                    raise ValueError("Replacement relay shape does not match recipient relay")
            elif intervention.donor_indices is not None:
                donors = [int(value) for value in intervention.donor_indices]
                if len(donors) != int(clean.size(0)):
                    raise ValueError("donor_indices must have one entry per recipient")
                partition = str(
                    trajectory.provenance.get("partition")
                    or trajectory.provenance.get("split_partition")
                    or trajectory.execution_manifest_hash
                    or "single_trajectory"
                )
                donor_records = [
                    {
                        "raw_sample_id": raw_id,
                        "partition": partition,
                        "gold": trajectory.gold_labels[index],
                        "edge_id": edge.edge_id,
                        "R": trajectory.rounds,
                        "tensor_shape": list(clean[index].shape),
                        "length_bucket": int(clean[index].shape[0]),
                    }
                    for index, raw_id in enumerate(trajectory.raw_sample_ids)
                ]
                assignments = lr.deterministic_donor_assignments(
                    donor_records,
                    donor_seed=int(intervention.seed),
                )
                raw_id_to_index = {
                    raw_id: index
                    for index, raw_id in enumerate(trajectory.raw_sample_ids)
                }
                mapping_failures: List[Mapping[str, Any]] = []
                for recipient, donor in enumerate(donors):
                    recipient_id = trajectory.raw_sample_ids[recipient]
                    assignment = assignments[recipient_id]
                    expected_index = (
                        raw_id_to_index.get(assignment.donor_id)
                        if assignment.donor_id is not None
                        else None
                    )
                    if not assignment.available:
                        mapping_failures.append(
                            {
                                "recipient_index": recipient,
                                "recipient_raw_sample_id": recipient_id,
                                "donor_index": donor,
                                "available": False,
                                "reason": assignment.reason,
                            }
                        )
                    elif donor != expected_index:
                        raise ValueError(
                            "mismatch donor_indices do not match the deterministic "
                            f"cyclic mapping for recipient {recipient_id!r}: "
                            f"expected {expected_index}, received {donor}"
                        )
                if mapping_failures:
                    raise InterventionUnavailable(
                        "Mismatch control unavailable under the deterministic donor mapping",
                        mapping_failures,
                    )
                requested_rows: List[Any] = []
                unavailable: List[Mapping[str, Any]] = []
                for recipient, donor in enumerate(donors):
                    if donor < 0 or donor >= int(clean.size(0)) or donor == recipient:
                        unavailable.append(
                            {"recipient_index": recipient, "donor_index": donor, "available": False,
                             "reason": "invalid_or_self_donor"}
                        )
                        continue
                    if trajectory.raw_sample_ids[donor] == trajectory.raw_sample_ids[recipient]:
                        unavailable.append(
                            {"recipient_index": recipient, "donor_index": donor, "available": False,
                             "reason": "self_donor_id"}
                        )
                        continue
                    if trajectory.gold_labels[donor] != trajectory.gold_labels[recipient]:
                        unavailable.append(
                            {"recipient_index": recipient, "donor_index": donor, "available": False,
                             "reason": "gold_label_mismatch"}
                        )
                        continue
                    source = clean[donor]
                    target = clean[recipient]
                    source_norm = t.linalg.vector_norm(source)
                    target_norm = t.linalg.vector_norm(target)
                    if float(source_norm.detach().cpu()) == 0.0 or float(target_norm.detach().cpu()) == 0.0:
                        unavailable.append(
                            {"recipient_index": recipient, "donor_index": donor, "available": False,
                             "reason": "zero_source_or_target_norm"}
                        )
                        continue
                    requested_rows.append(source * (target_norm / source_norm))
                    row_metadata.append(
                        {
                            "recipient_index": recipient,
                            "donor_index": donor,
                            "donor_sample_id": trajectory.sample_ids[donor],
                            "donor_raw_sample_id": trajectory.raw_sample_ids[donor],
                            "recipient_gold_label": trajectory.gold_labels[recipient],
                            "donor_gold_label": trajectory.gold_labels[donor],
                            "donor_seed": int(intervention.seed),
                            "donor_mapping_version": assignments[
                                trajectory.raw_sample_ids[recipient]
                            ].version,
                            "available": True,
                            "source_norm": float(source_norm.detach().cpu()),
                            "target_norm": float(target_norm.detach().cpu()),
                        }
                    )
                if unavailable:
                    raise InterventionUnavailable(
                        "Mismatch control unavailable; no cross-label/self fallback was used",
                        row_metadata + unavailable,
                    )
                requested = t.stack(requested_rows, dim=0)
            else:
                raise ValueError("mismatch intervention requires donor_indices")
        elif mode == "moment_noise":
            requested_rows = []
            for index, clean_row in enumerate(clean):
                requested_row, noise_diagnostics = lr.moment_noise_intervention(
                    clean_row,
                    global_seed=intervention.seed,
                    raw_sample_id=trajectory.raw_sample_ids[index],
                    edge=edge,
                    return_diagnostics=True,
                )
                requested_row = requested_row.to(execution_device)
                requested_rows.append(requested_row)
                row_metadata.append(
                    {
                        "sample_id": trajectory.sample_ids[index],
                        **noise_diagnostics.to_dict(),
                    }
                )
            requested = t.stack(requested_rows, dim=0)
        elif mode == "callable":
            requested = intervention.replacement(clean, trajectory, edge)
            if not t.is_tensor(requested) or requested.shape != clean.shape:
                raise ValueError("Callable intervention must return a same-shaped tensor")
        else:
            raise ValueError(f"Unsupported LinkRadius intervention mode: {mode!r}")

        if requested.shape != clean.shape:
            raise ValueError("Requested intervention does not match clean relay shape")
        live = requested.to(device=execution_device, dtype=consumer_dtype)
        realized = live.float() - clean
        for index in range(int(clean.size(0))):
            clean_norm = t.linalg.vector_norm(clean[index].float())
            requested_delta = requested[index].float() - clean[index]
            realized_delta = realized[index]
            requested_norm = t.linalg.vector_norm(requested_delta)
            realized_norm = t.linalg.vector_norm(realized_delta)
            base = row_metadata[index] if len(row_metadata) == int(clean.size(0)) else {
                "sample_id": trajectory.sample_ids[index]
            }
            base.update(
                {
                    "mode": mode,
                    "edge": edge.edge_id,
                    "clean_norm": float(clean_norm.detach().cpu()),
                    "requested_delta_norm": float(requested_norm.detach().cpu()),
                    "realized_delta_norm": float(realized_norm.detach().cpu()),
                    "realized_relative_norm": (
                        float((realized_norm / clean_norm).detach().cpu())
                        if float(clean_norm.detach().cpu()) > 0.0
                        else math.inf if float(realized_norm.detach().cpu()) > 0.0 else 0.0
                    ),
                    "collapsed": bool(float(realized_norm.detach().cpu()) == 0.0),
                    "realized_value_norm": float(
                        t.linalg.vector_norm(live[index].float()).detach().cpu()
                    ),
                    "consumer_dtype": trajectory.dtype_metadata(edge).consumer_dtype,
                    "transport_dtype": trajectory.dtype_metadata(edge).transport_dtype,
                }
            )
            base.setdefault("sample_id", trajectory.sample_ids[index])
            if mode == "moment_noise":
                realized_mean, realized_std, realized_value_norm = self._tensor_row_stats(
                    live[index].float()
                )
                base.update(
                    {
                        "realized_mean": realized_mean,
                        "realized_std": realized_std,
                        "realized_norm": realized_value_norm,
                    }
                )
            base.update(intervention.metadata)
            if len(row_metadata) != int(clean.size(0)):
                row_metadata.append(base)
        return live, row_metadata

    @classmethod
    def _record_recomputed(
        cls,
        edge: Any,
        emission: RelayEmission,
        transport: Dict[Any, Any],
        receiver: Dict[Any, Any],
    ) -> None:
        transport[edge] = cls._store_tensor(emission.transport)
        receiver[edge] = cls._store_tensor(emission.receiver)

    def _replay_to_terminal(
        self,
        trajectory: CleanTrajectory,
        edge: Any,
        intervention: Any = "identity",
        *,
        differentiable: bool = False,
    ) -> _ReplayTerminalState:
        """Inject one relay and recompute descendants without terminal scoring."""

        parsed = lr.parse_edge(edge) if not isinstance(edge, lr.Edge) else edge
        # This happens before _apply_intervention or any default stage can load.
        lr.validate_edge(parsed, self.config.rounds)
        self._validate_trajectory_compatibility(trajectory)
        if trajectory.rounds != self.config.rounds:
            raise ValueError("Trajectory/runtime round counts differ")
        schedule = replay_schedule(parsed, self.config.rounds)
        boundaries = validate_batch_boundaries(
            trajectory.batch_boundaries,
            len(trajectory.questions),
        )
        injected, intervention_metadata = self._apply_intervention(
            trajectory,
            parsed,
            intervention,
        )
        recomputed_transport: Dict[Any, Any] = {}
        recomputed_receiver: Dict[Any, Any] = {}
        round_idx = parsed.round_idx
        final_c = None

        if parsed.site == "p2c":
            c = self.run_critic(
                trajectory.questions,
                injected,
                round_idx,
                batch_boundaries=boundaries,
                differentiable=differentiable,
            )
            self._record_recomputed(
                lr.Edge("c2s", round_idx), c, recomputed_transport, recomputed_receiver
            )
            final_c = c.receiver
            if round_idx < self.config.rounds - 1:
                s = self.run_solver_feedback(
                    trajectory.questions,
                    c.receiver,
                    round_idx,
                    batch_boundaries=boundaries,
                    differentiable=differentiable,
                )
                self._record_recomputed(
                    lr.Edge("s2p", round_idx), s, recomputed_transport, recomputed_receiver
                )
                current_s = s.receiver
        elif parsed.site == "c2s":
            final_c = injected
            if round_idx < self.config.rounds - 1:
                s = self.run_solver_feedback(
                    trajectory.questions,
                    injected,
                    round_idx,
                    batch_boundaries=boundaries,
                    differentiable=differentiable,
                )
                self._record_recomputed(
                    lr.Edge("s2p", round_idx), s, recomputed_transport, recomputed_receiver
                )
                current_s = s.receiver
        else:  # s2p
            current_s = injected

        if round_idx < self.config.rounds - 1:
            for next_round in range(round_idx + 1, self.config.rounds):
                p = self.run_planner_feedback(
                    trajectory.questions,
                    current_s,
                    next_round,
                    batch_boundaries=boundaries,
                    differentiable=differentiable,
                )
                self._record_recomputed(
                    lr.Edge("p2c", next_round), p, recomputed_transport, recomputed_receiver
                )
                c = self.run_critic(
                    trajectory.questions,
                    p.receiver,
                    next_round,
                    batch_boundaries=boundaries,
                    differentiable=differentiable,
                )
                self._record_recomputed(
                    lr.Edge("c2s", next_round), c, recomputed_transport, recomputed_receiver
                )
                final_c = c.receiver
                if next_round < self.config.rounds - 1:
                    s = self.run_solver_feedback(
                        trajectory.questions,
                        c.receiver,
                        next_round,
                        batch_boundaries=boundaries,
                        differentiable=differentiable,
                    )
                    self._record_recomputed(
                        lr.Edge("s2p", next_round),
                        s,
                        recomputed_transport,
                        recomputed_receiver,
                    )
                    current_s = s.receiver
        if final_c is None:
            raise RuntimeError("Replay schedule did not produce terminal c2s")
        return _ReplayTerminalState(
            edge=parsed,
            schedule=schedule,
            boundaries=boundaries,
            intervention_metadata=intervention_metadata,
            intervened_receiver=injected,
            terminal_receiver=final_c,
            recomputed_transport_messages=recomputed_transport,
            recomputed_receiver_messages=recomputed_receiver,
        )

    def replay(
        self,
        trajectory: CleanTrajectory,
        edge: Any,
        intervention: Any = "identity",
        *,
        differentiable: bool = False,
        include_generation: bool = False,
    ) -> ReplayResult:
        """Inject one receiver-interface relay and recompute only descendants."""

        state = self._replay_to_terminal(
            trajectory,
            edge,
            intervention,
            differentiable=differentiable,
        )
        scoring = self.score_terminal(
            trajectory.questions,
            state.terminal_receiver,
            batch_boundaries=state.boundaries,
            differentiable=differentiable,
        )
        margins = scoring.margins(trajectory.gold_labels)
        generation = (
            self.audit_ordinary_generation(
                trajectory.questions,
                state.terminal_receiver,
                batch_boundaries=state.boundaries,
            )
            if include_generation
            else []
        )
        return ReplayResult(
            edge=state.edge,
            schedule=state.schedule,
            scoring=scoring if differentiable else scoring.detached(),
            margins=margins,
            intervention_metadata=state.intervention_metadata,
            intervened_receiver=self._store_tensor(state.intervened_receiver),
            recomputed_transport_messages=state.recomputed_transport_messages,
            recomputed_receiver_messages=state.recomputed_receiver_messages,
            generation_audit=generation,
        )

    def run_antithetic_probe(
        self,
        trajectory: CleanTrajectory,
        edge: Any,
        *,
        h: float,
        global_seed: int,
        probe_seed: int,
        direction_id: int,
        subspace_name: str = "full_tensor",
        thresholds: Any = None,
    ) -> AntitheticProbeResult:
        """Replay one sample-stable direction with exactly the +h and -h signs."""

        t = _require_torch()
        parsed = lr.validate_edge(edge, self.config.rounds)
        if not math.isfinite(float(h)) or float(h) < 0:
            raise ValueError("h must be finite and non-negative")
        clean = trajectory.message(parsed, receiver=True).float().cpu()
        if clean.ndim != 3 or int(clean.size(0)) != len(trajectory.sample_ids):
            raise ValueError("Stored receiver relay must have shape [N,T,D]")
        subspace = lr.get_subspace(
            subspace_name,
            int(clean.size(1)),
            int(clean.size(2)),
        )
        directions: List[Any] = []
        plus_deltas: List[Any] = []
        minus_deltas: List[Any] = []
        seeds: List[int] = []
        for index, raw_id in enumerate(trajectory.raw_sample_ids):
            seed = lr.stable_intervention_seed(
                global_seed,
                raw_id,
                parsed,
                probe_seed=probe_seed,
                direction_id=direction_id,
                purpose="probe",
            )
            direction = lr.sample_stable_unit_direction(
                global_seed,
                raw_id,
                parsed,
                subspace,
                probe_seed=probe_seed,
                direction_id=direction_id,
                purpose="probe",
            )
            lifted = subspace.lift(direction).float().cpu()
            directions.append(lifted)
            plus_deltas.append(
                lr.requested_additive_delta(clean[index], lifted, h=float(h), sign=1)
            )
            minus_deltas.append(
                lr.requested_additive_delta(clean[index], lifted, h=float(h), sign=-1)
            )
            seeds.append(int(seed))
        plus_delta = t.stack(plus_deltas, dim=0)
        minus_delta = t.stack(minus_deltas, dim=0)
        shared_metadata = {
            "requested_h": float(h),
            "global_seed": int(global_seed),
            "probe_seed": int(probe_seed),
            "direction_id": int(direction_id),
            "subspace_id": subspace.subspace_id,
            "q": int(subspace.q),
        }
        plus = self.replay(
            trajectory,
            parsed,
            ReplayIntervention(
                mode="additive",
                delta=plus_delta,
                metadata={**shared_metadata, "sign": 1},
            ),
        )
        minus = self.replay(
            trajectory,
            parsed,
            ReplayIntervention(
                mode="additive",
                delta=minus_delta,
                metadata={**shared_metadata, "sign": -1},
            ),
        )
        consumer_dtype = trajectory.dtype_metadata(parsed).consumer_dtype
        resolved_thresholds = thresholds or lr.ProbeAcceptanceThresholds()
        plus_records: List[Mapping[str, Any]] = []
        minus_records: List[Mapping[str, Any]] = []
        pair_records: List[Mapping[str, Any]] = []
        for index, direction in enumerate(directions):
            plus_diag = lr.realized_delta_diagnostics(
                clean[index],
                plus_delta[index],
                consumer_dtype=consumer_dtype,
                lifted_unit_direction=direction,
            )
            minus_diag = lr.realized_delta_diagnostics(
                clean[index],
                minus_delta[index],
                consumer_dtype=consumer_dtype,
                lifted_unit_direction=direction,
            )
            pair_diag = lr.probe_pair_diagnostics(
                plus_diag,
                minus_diag,
                resolved_thresholds,
            )
            common = {
                "sample_id": trajectory.sample_ids[index],
                "raw_sample_id": trajectory.raw_sample_ids[index],
                "edge": parsed.edge_id,
                "requested_h": float(h),
                "seed": seeds[index],
                "global_seed": int(global_seed),
                "probe_seed": int(probe_seed),
                "direction_id": int(direction_id),
                "subspace_id": subspace.subspace_id,
                "q": int(subspace.q),
                "transport_dtype": trajectory.dtype_metadata(parsed).transport_dtype,
                "consumer_dtype": consumer_dtype,
            }
            run_identity = {
                "sample_id": trajectory.sample_ids[index],
                "edge": parsed.edge_id,
                "h": float(h),
                "global_seed": int(global_seed),
                "probe_seed": int(probe_seed),
                "direction_id": int(direction_id),
                "subspace_id": subspace.subspace_id,
            }
            plus_run_id = _stable_json_hash({**run_identity, "sign": 1})
            minus_run_id = _stable_json_hash({**run_identity, "sign": -1})
            plus_records.append(
                {**common, "sign": 1, "run_id": plus_run_id, **plus_diag.to_dict()}
            )
            minus_records.append(
                {**common, "sign": -1, "run_id": minus_run_id, **minus_diag.to_dict()}
            )
            margins_plus = dict(plus.margins[index])
            margins_minus = dict(minus.margins[index])
            derivatives = None
            if pair_diag.accepted and pair_diag.realized_separation is not None:
                derivatives = {
                    competitor: lr.central_difference(
                        margins_plus[competitor],
                        margins_minus[competitor],
                        float(pair_diag.t_plus),
                        float(pair_diag.t_minus),
                    )
                    for competitor in margins_plus
                }
            pair_records.append(
                {
                    **common,
                    "plus_run_id": plus_run_id,
                    "minus_run_id": minus_run_id,
                    "margins_plus": margins_plus,
                    "margins_minus": margins_minus,
                    "directional_derivatives": derivatives,
                    **pair_diag.to_dict(),
                }
            )
        return AntitheticProbeResult(
            edge=parsed,
            h=float(h),
            global_seed=int(global_seed),
            probe_seed=int(probe_seed),
            direction_id=int(direction_id),
            subspace=subspace.to_dict(),
            plus=plus,
            minus=minus,
            plus_diagnostics=plus_records,
            minus_diagnostics=minus_records,
            pair_diagnostics=pair_records,
        )

    @staticmethod
    def _score_margin_tensor(
        scoring: ForcedChoiceBatch,
        gold_label: str,
        target_label: str,
        *,
        sample_index: int = 0,
    ) -> Any:
        gold = str(gold_label).upper()
        target = str(target_label).upper()
        if gold not in scoring.labels or target not in scoring.labels or gold == target:
            raise ValueError("gold and target must be distinct A/B/C/D labels")
        if sample_index < 0 or sample_index >= int(scoring.scores.size(0)):
            raise IndexError("sample_index is outside the forced-choice score batch")
        return (
            scoring.scores[sample_index, scoring.labels.index(gold)]
            - scoring.scores[sample_index, scoring.labels.index(target)]
        )

    def _sequential_terminal_margin_gradient(
        self,
        *,
        questions: Sequence[str],
        terminal_receiver: Any,
        gradient_input: Any,
        gold_label: str,
        target_label: str,
        sample_index: int,
        batch_boundaries: Sequence[Sequence[int]],
    ) -> Tuple[float, Any]:
        """Differentiate a margin while retaining only one scorer graph.

        The downstream replay graph is shared by both passes. The gold pass
        retains that shared graph, then releases its solver-scoring branch
        before the target pass is constructed. This computes the same
        ``gold_score - target_score`` derivative without simultaneously
        retaining four candidate forward graphs on the solver device.
        """

        t = _require_torch()
        gold = str(gold_label).upper()
        target = str(target_label).upper()
        if gold not in CHOICE_LABELS or target not in CHOICE_LABELS or gold == target:
            raise ValueError("gold and target must be distinct A/B/C/D labels")

        component_values: List[Any] = []
        margin_gradient = None
        for component_index, (label, sign) in enumerate(((gold, 1.0), (target, -1.0))):
            scoring = self.score_terminal(
                questions,
                terminal_receiver,
                batch_boundaries=batch_boundaries,
                differentiable=True,
                candidate_labels=(label,),
            )
            if label not in scoring.labels:
                raise RuntimeError(
                    f"memory-bounded scorer omitted requested candidate {label}"
                )
            component = scoring.scores[
                sample_index,
                scoring.labels.index(label),
            ]
            component_gradient = t.autograd.grad(
                component,
                gradient_input,
                # The target pass must traverse the same downstream replay
                # graph, but the gold scorer branch itself can be dropped as
                # soon as this iteration's local references are released.
                retain_graph=component_index == 0,
                create_graph=False,
                allow_unused=False,
            )[0]
            component_values.append(component.detach().float().cpu())
            signed_gradient = component_gradient if sign > 0 else -component_gradient
            margin_gradient = (
                signed_gradient
                if margin_gradient is None
                else margin_gradient + signed_gradient
            )
            del component, component_gradient, scoring, signed_gradient

        if margin_gradient is None:  # Defensive; the fixed loop always runs twice.
            raise RuntimeError("memory-bounded margin gradient produced no components")
        objective_value = float(component_values[0] - component_values[1])
        return objective_value, margin_gradient

    @staticmethod
    def _validate_autograd_sample(trajectory: CleanTrajectory, sample_index: int) -> int:
        if isinstance(sample_index, bool):
            raise TypeError("sample_index must be an integer")
        selected = int(sample_index)
        if selected < 0 or selected >= len(trajectory.sample_ids):
            raise IndexError("sample_index is outside the frozen execution batch")
        if not trajectory.analysis_eligibility_mask[selected]:
            raise ValueError("autograd sample_index must select an analysis-eligible row")
        return selected

    @staticmethod
    def _default_target(trajectory: CleanTrajectory, sample_index: int = 0) -> str:
        selected = LinkRadiusRuntime._validate_autograd_sample(trajectory, sample_index)
        gold = trajectory.gold_labels[selected]
        scores = trajectory.clean_scoring.scores
        if torch is not None and torch.is_tensor(scores):
            row = scores[selected].detach().float().cpu().tolist()
        else:
            row = list(scores[selected])
        candidates = [
            (float(row[index]), label)
            for index, label in enumerate(trajectory.clean_scoring.labels)
            if label != gold
        ]
        return max(candidates)[1]

    def terminal_gradient(
        self,
        trajectory: CleanTrajectory,
        *,
        target_label: Optional[str] = None,
        sample_index: int = 0,
    ) -> GradientResult:
        """Continuous terminal gradient in the unchanged frozen batch context."""

        t = _require_torch()
        edge = lr.Edge("c2s", self.config.rounds - 1)
        lr.validate_edge(edge, self.config.rounds)  # before the model is ensured
        self._validate_trajectory_compatibility(trajectory)
        selected = self._validate_autograd_sample(trajectory, sample_index)
        target = str(target_label or self._default_target(trajectory, selected)).upper()
        gold = trajectory.gold_labels[selected]
        consumer_dtype = _torch_dtype_from_name(
            trajectory.dtype_metadata(edge).consumer_dtype
        )
        leaf = trajectory.message(edge, receiver=True).to(
            device=self.role_device("solver"),
            dtype=consumer_dtype,
        ).detach()
        leaf.requires_grad_(True)
        objective_value, full_gradient = self._sequential_terminal_margin_gradient(
            questions=trajectory.questions,
            terminal_receiver=leaf,
            gradient_input=leaf,
            gold_label=gold,
            target_label=target,
            sample_index=selected,
            batch_boundaries=trajectory.batch_boundaries,
        )
        gradient = full_gradient[selected : selected + 1]
        return GradientResult(
            edge=edge,
            gold_label=gold,
            target_label=target,
            objective_name="gold_minus_target_margin",
            objective_value=objective_value,
            gradient=gradient.detach().float().cpu(),
            gradient_norm=float(t.linalg.vector_norm(gradient.float()).detach().cpu()),
            autograd_semantics="continuous_consumer_input",
            sample_index=selected,
            sample_id=trajectory.sample_ids[selected],
        )

    def autograd_gradient(
        self,
        trajectory: CleanTrajectory,
        edge: Any,
        *,
        target_label: Optional[str] = None,
        sample_index: int = 0,
    ) -> GradientResult:
        """Attempt an autograd reference through the full downstream replay."""

        t = _require_torch()
        parsed = lr.validate_edge(edge, self.config.rounds)
        self._validate_trajectory_compatibility(trajectory)
        selected = self._validate_autograd_sample(trajectory, sample_index)
        terminal = lr.Edge("c2s", self.config.rounds - 1)
        if parsed == terminal:
            return self.terminal_gradient(
                trajectory,
                target_label=target_label,
                sample_index=selected,
            )
        target = str(target_label or self._default_target(trajectory, selected)).upper()
        gold = trajectory.gold_labels[selected]
        consumer_dtype = _torch_dtype_from_name(
            trajectory.dtype_metadata(parsed).consumer_dtype
        )
        leaf = trajectory.message(parsed, receiver=True).to(
            device=self.edge_consumer_device(parsed),
            dtype=consumer_dtype,
        ).detach()
        leaf.requires_grad_(True)
        state = self._replay_to_terminal(
            trajectory,
            parsed,
            ReplayIntervention(mode="replacement", replacement=leaf),
            differentiable=True,
        )
        objective_value, full_gradient = self._sequential_terminal_margin_gradient(
            questions=trajectory.questions,
            terminal_receiver=state.terminal_receiver,
            gradient_input=leaf,
            gold_label=gold,
            target_label=target,
            sample_index=selected,
            batch_boundaries=state.boundaries,
        )
        gradient = full_gradient[selected : selected + 1]
        return GradientResult(
            edge=parsed,
            gold_label=gold,
            target_label=target,
            objective_name="gold_minus_target_margin",
            objective_value=objective_value,
            gradient=gradient.detach().float().cpu(),
            gradient_norm=float(t.linalg.vector_norm(gradient.float()).detach().cpu()),
            autograd_semantics="relaxed_autograd",
            sample_index=selected,
            sample_id=trajectory.sample_ids[selected],
        )

    @staticmethod
    def _fit_realized_budget(
        coefficients: Any,
        clean: Any,
        subspace: Any,
        consumer_dtype: Any,
        budget: float,
    ) -> Any:
        """Shrink coefficients until the actual post-cast norm is feasible."""

        t = _require_torch()
        if budget == 0.0:
            return t.zeros_like(coefficients)

        def realized_norm(candidate: Any) -> float:
            requested = subspace.lift(candidate).reshape_as(clean)
            realized = (clean + requested).to(dtype=consumer_dtype).float() - clean
            return float(t.linalg.vector_norm(realized).detach().cpu())

        tolerance = 1e-7 * max(1.0, budget)
        if realized_norm(coefficients) <= budget + tolerance:
            return coefficients
        lower = 0.0
        upper = 1.0
        best = t.zeros_like(coefficients)
        for _ in range(32):
            scale = (lower + upper) / 2.0
            candidate = coefficients * scale
            if realized_norm(candidate) <= budget + tolerance:
                lower = scale
                best = candidate
            else:
                upper = scale
        return best

    def autograd_pgd(
        self,
        trajectory: CleanTrajectory,
        edge: Any,
        *,
        epsilon: float,
        steps: int = 20,
        step_size: Optional[float] = None,
        subspace_name: str = "full_tensor",
        targets: Optional[Sequence[str]] = None,
        sample_index: int = 0,
    ) -> PGDResult:
        """Projected gradient attack, never a derivative-free substitute.

        ``epsilon`` is relative to the frozen receiver-reference Frobenius norm.
        An explicit ``step_size`` is in absolute subspace-coordinate units; the
        default is ``2 * epsilon * ||z_ref|| / steps``. Every update is projected
        in the declared isometric subspace and then shrunk, if needed, so its
        realized post-consumer-cast norm also respects the scientific budget.
        """

        t = _require_torch()
        parsed = lr.validate_edge(edge, self.config.rounds)
        self._validate_trajectory_compatibility(trajectory)
        selected = self._validate_autograd_sample(trajectory, sample_index)
        if not math.isfinite(float(epsilon)) or float(epsilon) < 0:
            raise ValueError("epsilon must be finite and non-negative")
        if isinstance(steps, bool) or int(steps) < 1:
            raise ValueError("steps must be a positive integer")
        clean_batch_cpu = trajectory.message(parsed, receiver=True).float().cpu()
        if clean_batch_cpu.ndim != 3:
            raise ValueError("PGD relay must have shape [N,T,D]")
        clean_cpu = clean_batch_cpu[selected : selected + 1]
        clean_norm = float(t.linalg.vector_norm(clean_cpu).item())
        if clean_norm == 0.0:
            raise ValueError("Relative-budget PGD is unavailable for a zero-norm relay")
        budget = float(epsilon) * clean_norm
        absolute_step = (
            float(step_size)
            if step_size is not None
            else (2.0 * budget / int(steps) if steps else 0.0)
        )
        if not math.isfinite(absolute_step) or absolute_step < 0:
            raise ValueError("step_size must be finite and non-negative")
        subspace = lr.get_subspace(
            subspace_name,
            int(clean_cpu.size(1)),
            int(clean_cpu.size(2)),
        )
        gold = trajectory.gold_labels[selected]
        target_values = [str(value).upper() for value in (targets or CHOICE_LABELS) if str(value).upper() != gold]
        if set(target_values) - set(CHOICE_LABELS) or len(target_values) != len(set(target_values)):
            raise ValueError("PGD targets must be unique wrong A/B/C/D labels")
        consumer_dtype = _torch_dtype_from_name(
            trajectory.dtype_metadata(parsed).consumer_dtype
        )
        execution_device = self.edge_consumer_device(parsed)
        clean_batch = clean_batch_cpu.to(device=execution_device, dtype=t.float32)
        clean = clean_batch[selected : selected + 1]
        terminal = parsed == lr.Edge("c2s", self.config.rounds - 1)
        target_results: List[PGDTargetResult] = []

        def inject_selected(requested_row: Any) -> Any:
            return t.cat(
                (
                    clean_batch[:selected],
                    requested_row,
                    clean_batch[selected + 1 :],
                ),
                dim=0,
            )

        def terminal_receiver_for(requested_row: Any) -> Any:
            requested_receiver = inject_selected(requested_row)
            live = requested_receiver.to(dtype=consumer_dtype)
            if terminal:
                return live
            state = self._replay_to_terminal(
                trajectory,
                parsed,
                ReplayIntervention(mode="replacement", replacement=requested_receiver),
                differentiable=True,
            )
            return state.terminal_receiver

        clean_scores = trajectory.clean_scoring.scores
        if t.is_tensor(clean_scores):
            clean_row = clean_scores[selected].detach().float().cpu().tolist()
        else:
            clean_row = list(clean_scores[selected])
        for target in target_values:
            initial_margin = choice_margins(clean_row, gold, CHOICE_LABELS)[target]
            coefficients = t.zeros(subspace.q, dtype=t.float32, device=execution_device)
            for _ in range(int(steps)):
                coefficients = coefficients.detach().requires_grad_(True)
                delta = subspace.lift(coefficients).reshape_as(clean)
                terminal_receiver = terminal_receiver_for(clean + delta)
                _, gradient = self._sequential_terminal_margin_gradient(
                    questions=trajectory.questions,
                    terminal_receiver=terminal_receiver,
                    gradient_input=coefficients,
                    gold_label=gold,
                    target_label=target,
                    sample_index=selected,
                    batch_boundaries=trajectory.batch_boundaries,
                )
                del terminal_receiver
                gradient_norm = t.linalg.vector_norm(gradient)
                if not bool(t.isfinite(gradient_norm)) or float(gradient_norm.detach().cpu()) == 0.0:
                    coefficients = coefficients.detach()
                    break
                candidate = coefficients - absolute_step * gradient / gradient_norm
                candidate = lr.project_subspace_coefficients(candidate, budget).detach()
                coefficients = self._fit_realized_budget(
                    candidate,
                    clean,
                    subspace,
                    consumer_dtype,
                    budget,
                ).detach()
            requested_delta = subspace.lift(coefficients).reshape_as(clean)
            adversarial = (clean + requested_delta).to(dtype=consumer_dtype)
            realized_delta = adversarial.float() - clean
            adversarial_batch = inject_selected(clean + requested_delta).to(
                dtype=consumer_dtype
            )
            with t.set_grad_enabled(False):
                if terminal:
                    final_scoring = self.score_terminal(
                        trajectory.questions,
                        adversarial_batch,
                        batch_boundaries=trajectory.batch_boundaries,
                        differentiable=False,
                    )
                else:
                    final_scoring = self.replay(
                        trajectory,
                        parsed,
                        ReplayIntervention(
                            mode="replacement",
                            replacement=inject_selected(
                                (clean + requested_delta).detach()
                            ),
                        ),
                        differentiable=False,
                    ).scoring
            final_row = (
                final_scoring.scores[selected].detach().float().cpu().tolist()
            )
            final_margin = choice_margins(final_row, gold, CHOICE_LABELS)[target]
            requested_norm = float(t.linalg.vector_norm(requested_delta).detach().cpu())
            realized_norm = float(t.linalg.vector_norm(realized_delta).detach().cpu())
            target_results.append(
                PGDTargetResult(
                    target_label=target,
                    initial_margin=float(initial_margin),
                    final_margin=float(final_margin),
                    improved=bool(final_margin < initial_margin),
                    requested_delta_norm=requested_norm,
                    realized_delta_norm=realized_norm,
                    budget=budget,
                    budget_respected=bool(realized_norm <= budget + 1e-7 * max(1.0, budget)),
                    scores=[float(value) for value in final_row],
                    adversarial_receiver=adversarial.detach().float().cpu(),
                )
            )
        strongest = min(target_results, key=lambda item: item.final_margin).target_label if target_results else None
        return PGDResult(
            edge=parsed,
            epsilon=float(epsilon),
            subspace=subspace.name,
            q=int(subspace.q),
            steps=int(steps),
            step_size=absolute_step,
            # Coefficients are float32 and pass through the consumer cast before
            # scoring, even at terminal c2s. Backward is therefore explicitly a
            # straight-through continuous relaxation of the quantized map.
            autograd_semantics="relaxed_autograd",
            targets=target_results,
            strongest_target=strongest,
            sample_index=selected,
            sample_id=trajectory.sample_ids[selected],
        )

    terminal_pgd = autograd_pgd


def clean_audit_rows(trajectory: CleanTrajectory) -> List[Dict[str, Any]]:
    """Build transparent dual-correct eligibility and scorer-agreement rows."""

    scores = trajectory.clean_scoring.scores
    if torch is not None and torch.is_tensor(scores):
        score_rows = scores.detach().float().cpu().tolist()
    else:
        score_rows = [list(row) for row in scores]
    generation = trajectory.clean_generation_audit
    rows: List[Dict[str, Any]] = []
    for index, (sample_id, gold, score_row) in enumerate(
        zip(trajectory.sample_ids, trajectory.gold_labels, score_rows)
    ):
        generated = generation[index].get("strict_choice") if index < len(generation) else None
        generated_invalid = (
            bool(generation[index].get("answer_invalid", True))
            if index < len(generation)
            else True
        )
        generated_conflict = (
            bool(generation[index].get("answer_conflict", False))
            if index < len(generation)
            else False
        )
        scored = trajectory.clean_scoring.predictions[index]
        score_tie = bool(trajectory.clean_scoring.score_ties[index])
        margins = trajectory.clean_margins[index]
        reasons: List[str] = []
        analysis_eligible = bool(trajectory.analysis_eligibility_mask[index])
        analysis_reasons = [] if analysis_eligible else ["execution_manifest_ineligible"]
        if index >= len(generation):
            reasons.append("ordinary_generation_not_recorded")
        elif generated_conflict:
            reasons.append("generated_answer_conflict")
        elif generated_invalid or generated is None:
            reasons.append("generated_answer_invalid")
        elif generated != gold:
            reasons.append("generated_answer_incorrect")
        if score_tie:
            reasons.append("forced_choice_score_tie")
        elif scored is None:
            reasons.append("forced_choice_prediction_invalid")
        elif scored != gold:
            reasons.append("forced_choice_prediction_incorrect")
        if not all(math.isfinite(float(value)) for value in score_row):
            reasons.append("nonfinite_forced_choice_score")
        if not all(math.isfinite(float(value)) for value in margins.values()):
            reasons.append("nonfinite_clean_margin")
        elif any(float(value) <= 0 for value in margins.values()):
            reasons.append("nonpositive_clean_margin")
        dual_correct = not reasons
        rows.append(
            {
                "sample_id": sample_id,
                "raw_sample_id": trajectory.raw_sample_ids[index],
                "gold_label": gold,
                "strict_generated_choice": generated,
                "forced_choice_prediction": scored,
                "score_tie": score_tie,
                "generated_scored_agreement": (
                    generated == scored if generated is not None and scored is not None else None
                ),
                "generated_correct": generated == gold if generated is not None else False,
                "scored_correct": scored == gold if scored is not None else False,
                "dual_correct": dual_correct,
                "dual_correct_exclusion_reasons": list(reasons),
                "analysis_eligible": analysis_eligible,
                "analysis_exclusion_reasons": analysis_reasons,
                "analysis_dual_correct": bool(dual_correct and analysis_eligible),
                "exclusion_reasons": reasons,
            }
        )
    return rows


def save_trajectory(path: Any, trajectory: CleanTrajectory) -> None:
    """Persist the typed trajectory without narrowing any float32 relay tensor."""

    t = _require_torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    t.save(
        {
            "artifact_type": "linkradius_clean_trajectory",
            "schema_version": trajectory.schema_version,
            "trajectory": trajectory,
        },
        destination,
    )


def load_trajectory(path: Any) -> CleanTrajectory:
    t = _require_torch()
    payload = t.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("artifact_type") != "linkradius_clean_trajectory":
        raise ValueError("Not a LinkRadius clean trajectory artifact")
    trajectory = payload.get("trajectory")
    if not isinstance(trajectory, CleanTrajectory):
        raise ValueError("Trajectory artifact has an unexpected payload type")
    if trajectory.schema_version != TRAJECTORY_VERSION:
        raise ValueError(
            f"Unsupported trajectory schema {trajectory.schema_version!r}; expected {TRAJECTORY_VERSION!r}"
        )
    return trajectory


# Explicit name used by early design notes and some external runners.
PersistentSequentialRuntime = LinkRadiusRuntime


__all__ = [
    "AntitheticProbeResult",
    "CHOICE_LABELS",
    "CandidateEncoding",
    "CleanTrajectory",
    "DEFAULT_SCORER_PREFIX",
    "DEFAULT_VERBALIZERS",
    "EdgeDtypeMetadata",
    "ForcedChoiceBatch",
    "GradientResult",
    "InterventionUnavailable",
    "LinkRadiusRuntime",
    "PGDResult",
    "PGDTargetResult",
    "PersistentSequentialRuntime",
    "RUNTIME_VERSION",
    "RelayEmission",
    "ReplayIntervention",
    "ReplayResult",
    "ReplayStep",
    "RuntimeConfig",
    "SCORER_VERSION",
    "SYSTEM_IDENTITY_VERSION",
    "TRAJECTORY_VERSION",
    "causal_token_log_probs",
    "choice_margins",
    "clean_audit_rows",
    "load_trajectory",
    "longest_common_token_prefix",
    "prediction_from_scores",
    "replay_schedule",
    "replay_schedule_tokens",
    "save_trajectory",
    "tokenize_joint_candidate",
    "tokenize_joint_candidates",
    "validate_batch_boundaries",
]
