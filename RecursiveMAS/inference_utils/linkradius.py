"""Pure, versioned foundations for the LinkRadius experiments.

This module deliberately contains no model-loading code.  In particular, importing
it must be safe in CPU-only aggregation and manifest jobs.  PyTorch is optional at
import time and is required only by tensor-valued perturbation helpers.

The public conventions in this file are part of the experiment provenance.  Do not
change an existing version constant when changing semantics; introduce a new
version instead.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from fractions import Fraction
import hashlib
import json
import math
import random
import re
import unicodedata
from typing import Any, Callable, Optional

try:  # Keep split/manifest/aggregation utilities importable without PyTorch.
    import torch
except Exception:  # pragma: no cover - exercised by the system Python in CI jobs.
    torch = None  # type: ignore[assignment]


LINKRADIUS_UTILS_VERSION = "linkradius_utils_v1"
EDGE_SCHEMA_VERSION = "linkradius_edge_v1"
REPLAY_SCHEDULE_VERSION = "linkradius_replay_schedule_v1"
GPQA_RAW_ID_VERSION = "linkradius_gpqa_raw_id_v1"
GPQA_OPTION_PERMUTATION_VERSION = "gpqa_md5_stem_seed_v1"
GPQA_SPLIT_VERSION = "linkradius_gpqa_split_v1"
STRICT_CHOICE_VERSION = "linkradius_choice_v2"
SUBSPACE_VERSION = "linkradius_subspace_v1"
DIRECTION_VERSION = "linkradius_direction_v1"
INTERVENTION_SEED_VERSION = "linkradius_intervention_seed_v1"
MOMENT_NOISE_VERSION = "moment_noise_v1"
DONOR_MAPPING_VERSION = "linkradius_cyclic_derangement_v1"
DELTA_DIAGNOSTICS_VERSION = "linkradius_realized_delta_v1"
PROBE_PAIR_VERSION = "linkradius_probe_pair_v1"
ESTIMATOR_VERSION = "linkradius_estimator_v1"

EDGE_SITES = ("p2c", "c2s", "s2p")
CHOICE_LABELS = ("A", "B", "C", "D")
GPQA_PARTITIONS = ("attack_train", "validation", "test")
GPQA_SPLIT_RATIOS = (Fraction(2, 5), Fraction(1, 5), Fraction(2, 5))


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _domain_hash(domain: bytes, value: Any) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


# ---------------------------------------------------------------------------
# Sequential chronology and replay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    """One transmitted sequential-system handoff, using zero-based code rounds."""

    site: str
    round_idx: int
    schema_version: str = field(default=EDGE_SCHEMA_VERSION, init=False, compare=False)

    def __post_init__(self) -> None:
        site = str(self.site).strip().lower()
        if site not in EDGE_SITES:
            raise ValueError(f"Unknown LinkRadius edge site {self.site!r}; expected one of {EDGE_SITES}.")
        if isinstance(self.round_idx, bool):
            raise ValueError("round_idx must be a non-negative integer, not bool.")
        round_idx = int(self.round_idx)
        if round_idx != self.round_idx or round_idx < 0:
            raise ValueError(f"round_idx must be a non-negative integer, got {self.round_idx!r}.")
        object.__setattr__(self, "site", site)
        object.__setattr__(self, "round_idx", round_idx)

    @property
    def edge_id(self) -> str:
        return f"{self.site}@{self.round_idx}"

    @property
    def token(self) -> str:
        """Filesystem-friendly, unambiguous edge token."""

        return f"{self.site}_r{self.round_idx}"

    @property
    def code_round(self) -> int:
        return self.round_idx

    @property
    def paper_round(self) -> int:
        return self.round_idx + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "site": self.site,
            "round_idx": self.round_idx,
            "code_round": self.code_round,
            "paper_round": self.paper_round,
            "edge_id": self.edge_id,
            "edge_token": self.token,
        }

    def __str__(self) -> str:
        return self.edge_id


def parse_edge(value: Edge | str | Mapping[str, Any]) -> Edge:
    if isinstance(value, Edge):
        return value
    if isinstance(value, Mapping):
        if "site" not in value:
            raise ValueError("Edge mapping is missing 'site'.")
        round_value = value.get("round_idx", value.get("code_round"))
        if round_value is None:
            raise ValueError("Edge mapping is missing 'round_idx'/'code_round'.")
        return Edge(str(value["site"]), int(round_value))
    text = str(value).strip().lower()
    match = re.fullmatch(r"(p2c|c2s|s2p)\s*(?:@|:|_r)\s*(\d+)", text)
    if match is None:
        raise ValueError(f"Invalid edge {value!r}; expected e.g. 'p2c@0'.")
    return Edge(match.group(1), int(match.group(2)))


def valid_edges(R: int) -> tuple[Edge, ...]:
    """Return the exact chronological edge set for a sequential horizon ``R``."""

    if isinstance(R, bool) or int(R) != R or int(R) < 1:
        raise ValueError(f"R must be a positive integer, got {R!r}.")
    horizon = int(R)
    edges: list[Edge] = []
    for round_idx in range(horizon):
        edges.append(Edge("p2c", round_idx))
        edges.append(Edge("c2s", round_idx))
        if round_idx < horizon - 1:
            edges.append(Edge("s2p", round_idx))
    return tuple(edges)


def validate_edge(edge: Edge | str | Mapping[str, Any], R: int) -> Edge:
    parsed = parse_edge(edge)
    allowed = valid_edges(R)
    if parsed not in allowed:
        allowed_text = ", ".join(item.edge_id for item in allowed)
        raise ValueError(
            f"Invalid intervention edge {parsed.edge_id!r} for R={int(R)}. "
            f"Valid edges: [{allowed_text}]."
        )
    return parsed


@dataclass(frozen=True)
class ReplayStep:
    operation: str
    round_idx: int
    consumes_edge: Optional[Edge] = None
    produces_edge: Optional[Edge] = None
    schema_version: str = field(default=REPLAY_SCHEDULE_VERSION, init=False, compare=False)

    @property
    def code_round(self) -> int:
        return self.round_idx

    @property
    def paper_round(self) -> int:
        return self.round_idx + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "round_idx": self.round_idx,
            "code_round": self.code_round,
            "paper_round": self.paper_round,
            "consumes_edge": self.consumes_edge.edge_id if self.consumes_edge else None,
            "produces_edge": self.produces_edge.edge_id if self.produces_edge else None,
        }


def replay_schedule(edge: Edge | str | Mapping[str, Any], R: int) -> tuple[ReplayStep, ...]:
    """Return exactly the downstream computations after replacing ``edge``.

    No clean descendant is reused and no upstream operation is included.  Edge
    validation occurs before a schedule is constructed, so a terminal ``s2p`` can
    never be mislabeled as an intervention run.
    """

    selected = validate_edge(edge, R)
    horizon = int(R)
    r = selected.round_idx
    steps: list[ReplayStep] = []

    if selected.site == "p2c":
        c_edge = Edge("c2s", r)
        steps.append(ReplayStep("critic", r, consumes_edge=selected, produces_edge=c_edge))
        if r == horizon - 1:
            steps.append(ReplayStep("score_final", r, consumes_edge=c_edge))
            return tuple(steps)
        s_edge = Edge("s2p", r)
        steps.append(ReplayStep("solver_feedback", r, consumes_edge=c_edge, produces_edge=s_edge))
    elif selected.site == "c2s":
        c_edge = selected
        if r == horizon - 1:
            steps.append(ReplayStep("score_final", r, consumes_edge=c_edge))
            return tuple(steps)
        s_edge = Edge("s2p", r)
        steps.append(ReplayStep("solver_feedback", r, consumes_edge=c_edge, produces_edge=s_edge))
    else:
        s_edge = selected

    for round_idx in range(r + 1, horizon):
        p_edge = Edge("p2c", round_idx)
        steps.append(ReplayStep("planner_feedback", round_idx, consumes_edge=s_edge, produces_edge=p_edge))
        c_edge = Edge("c2s", round_idx)
        steps.append(ReplayStep("critic", round_idx, consumes_edge=p_edge, produces_edge=c_edge))
        if round_idx < horizon - 1:
            s_edge = Edge("s2p", round_idx)
            steps.append(
                ReplayStep("solver_feedback", round_idx, consumes_edge=c_edge, produces_edge=s_edge)
            )

    steps.append(ReplayStep("score_final", horizon - 1, consumes_edge=c_edge))
    return tuple(steps)


def replay_operations(edge: Edge | str | Mapping[str, Any], R: int) -> tuple[str, ...]:
    """Compact operation names, useful for audits and unit tests."""

    return tuple(step.operation for step in replay_schedule(edge, R))


# ---------------------------------------------------------------------------
# Stable GPQA raw identity, release-equivalent rendering, and raw splitting
# ---------------------------------------------------------------------------


_GPQA_CANONICAL_KEYS = (
    "question",
    "correct_answer",
    "incorrect_answer_1",
    "incorrect_answer_2",
    "incorrect_answer_3",
)

_GPQA_SOURCE_KEYS: dict[str, tuple[str, ...]] = {
    "question": ("Question", "question", "query", "prompt"),
    "correct_answer": ("Correct Answer", "correct_answer"),
    "incorrect_answer_1": ("Incorrect Answer 1", "incorrect_answer_1"),
    "incorrect_answer_2": ("Incorrect Answer 2", "incorrect_answer_2"),
    "incorrect_answer_3": ("Incorrect Answer 3", "incorrect_answer_3"),
}

_GPQA_NATIVE_ID_KEYS = (
    "id",
    "ID",
    "question_id",
    "Question ID",
    "record_id",
    "Record ID",
    "sample_id",
)


def _first_nonempty(record: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def normalize_gpqa_identity_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def canonical_gpqa_identity_object(record: Mapping[str, Any]) -> dict[str, str]:
    canonical = {
        target: normalize_gpqa_identity_text(_first_nonempty(record, source_keys))
        for target, source_keys in _GPQA_SOURCE_KEYS.items()
    }
    missing = [key for key in _GPQA_CANONICAL_KEYS if not canonical[key]]
    if missing:
        raise ValueError(f"GPQA record has empty required field(s): {', '.join(missing)}.")
    return canonical


def stable_gpqa_raw_id(record: Mapping[str, Any]) -> str:
    """Return a native GPQA ID, or the specified domain-separated SHA-256 ID."""

    # Validate content even when a source ID exists: malformed records are not safe
    # to include merely because they happen to carry metadata.
    canonical_gpqa_identity_object(record)
    native_id = _first_nonempty(record, _GPQA_NATIVE_ID_KEYS)
    if native_id:
        return normalize_gpqa_identity_text(native_id)
    payload = _canonical_json_bytes(canonical_gpqa_identity_object(record))
    return hashlib.sha256(b"linkradius:gpqa_raw_id:v1\0" + payload).hexdigest()


def gpqa_option_permutation(question: str, seed: int = 42, shuffle_options: bool = True) -> tuple[int, ...]:
    """Mirror the existing GPQA MD5 stem-and-seed presentation permutation."""

    if not shuffle_options:
        return (0, 1, 2, 3)
    order = list(range(4))
    seed_hex = hashlib.md5(f"{int(seed)}::{str(question)}".encode("utf-8")).hexdigest()
    rng = random.Random(int(seed_hex[:16], 16))
    rng.shuffle(order)
    return tuple(order)


def render_gpqa_options(
    question: str,
    source_options: Sequence[str],
    permutation: Sequence[int],
) -> tuple[str, str, tuple[str, ...]]:
    if len(source_options) != 4 or len(permutation) != 4 or set(permutation) != set(range(4)):
        raise ValueError("GPQA rendering requires four options and a permutation of 0..3.")
    labels = CHOICE_LABELS
    option_texts = tuple(str(source_options[index]) for index in permutation)
    gold_label = labels[tuple(permutation).index(0)]
    lines = [f"{label}. {text}" for label, text in zip(labels, option_texts)]
    rendered = f"{str(question).strip()}\n" + "\n".join(lines)
    rendered = rendered.rstrip()
    if "choose the correct option" not in rendered.lower():
        rendered += "\n\nChoose the correct option (A/B/C/D)."
    return rendered, gold_label, option_texts


@dataclass(frozen=True)
class GPQARawRecord(Mapping[str, Any]):
    raw_sample_id: str
    raw_index: int
    question: str
    correct_answer: str
    incorrect_answer_1: str
    incorrect_answer_2: str
    incorrect_answer_3: str
    option_permutation: tuple[int, ...]
    option_texts: tuple[str, ...]
    gold_label: str
    rendered_question: str
    raw_id_algorithm: str
    option_permutation_algorithm: str = GPQA_OPTION_PERMUTATION_VERSION
    schema_version: str = "linkradius_gpqa_raw_record_v1"

    @property
    def source_options(self) -> tuple[str, ...]:
        return (
            self.correct_answer,
            self.incorrect_answer_1,
            self.incorrect_answer_2,
            self.incorrect_answer_3,
        )

    # Mapping behavior makes the record convenient for JSON/manifests while keeping
    # a typed, immutable contract for runtime code.
    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_options"] = list(self.source_options)
        result["option_permutation"] = list(self.option_permutation)
        result["option_texts"] = list(self.option_texts)
        return result

    def __getitem__(self, key: str) -> Any:
        if key == "source_options":
            return self.source_options
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def build_gpqa_raw_record(
    row: Mapping[str, Any],
    raw_index: int,
    seed: int = 42,
    shuffle_options: bool = True,
) -> GPQARawRecord:
    if isinstance(raw_index, bool) or int(raw_index) != raw_index or int(raw_index) < 0:
        raise ValueError(f"raw_index must be a non-negative integer, got {raw_index!r}.")
    canonical = canonical_gpqa_identity_object(row)
    # Rendering intentionally uses the legacy stripped source text, rather than the
    # NFKC identity representation, to avoid changing release inputs.
    question = _first_nonempty(row, _GPQA_SOURCE_KEYS["question"])
    source_options = tuple(
        _first_nonempty(row, _GPQA_SOURCE_KEYS[key])
        for key in _GPQA_CANONICAL_KEYS[1:]
    )
    permutation = gpqa_option_permutation(question, seed=seed, shuffle_options=shuffle_options)
    rendered, gold_label, option_texts = render_gpqa_options(question, source_options, permutation)
    has_native_id = bool(_first_nonempty(row, _GPQA_NATIVE_ID_KEYS))
    return GPQARawRecord(
        raw_sample_id=stable_gpqa_raw_id(row),
        raw_index=int(raw_index),
        question=question,
        correct_answer=source_options[0],
        incorrect_answer_1=source_options[1],
        incorrect_answer_2=source_options[2],
        incorrect_answer_3=source_options[3],
        option_permutation=permutation,
        option_texts=option_texts,
        gold_label=gold_label,
        rendered_question=rendered,
        raw_id_algorithm="native_source_id" if has_native_id else GPQA_RAW_ID_VERSION,
    )


def validate_unique_raw_ids(records_or_ids: Sequence[GPQARawRecord | Mapping[str, Any] | str]) -> tuple[str, ...]:
    ids: list[str] = []
    for item in records_or_ids:
        if isinstance(item, str):
            raw_id = item
        elif isinstance(item, GPQARawRecord):
            raw_id = item.raw_sample_id
        else:
            value = item.get("raw_sample_id")
            if value is None:
                raise ValueError("Raw record mapping is missing 'raw_sample_id'.")
            raw_id = str(value)
        if not raw_id:
            raise ValueError("raw_sample_id must be nonempty.")
        ids.append(raw_id)
    duplicates = sorted({raw_id for raw_id in ids if ids.count(raw_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate GPQA raw IDs: {duplicates}.")
    return tuple(ids)


def _ratio_fraction(value: Fraction | int | float | str) -> Fraction:
    if isinstance(value, Fraction):
        return value
    return Fraction(str(value))


def largest_remainder_counts(
    total: int,
    ratios: Sequence[Fraction | int | float | str] = GPQA_SPLIT_RATIOS,
) -> tuple[int, ...]:
    """Allocate ``total`` by Hamilton's largest-remainder rule."""

    if isinstance(total, bool) or int(total) != total or int(total) < 0:
        raise ValueError(f"total must be a non-negative integer, got {total!r}.")
    fractions = tuple(_ratio_fraction(value) for value in ratios)
    if not fractions or any(value < 0 for value in fractions):
        raise ValueError("ratios must be a nonempty sequence of non-negative values.")
    ratio_sum = sum(fractions, Fraction(0))
    if ratio_sum <= 0:
        raise ValueError("ratios must have a positive sum.")
    quotas = [Fraction(int(total)) * value / ratio_sum for value in fractions]
    counts = [quota.numerator // quota.denominator for quota in quotas]
    remaining = int(total) - sum(counts)
    order = sorted(
        range(len(quotas)),
        key=lambda index: (-(quotas[index] - counts[index]), index),
    )
    for index in order[:remaining]:
        counts[index] += 1
    return tuple(counts)


def split_raw_ids(
    raw_ids: Sequence[str],
    seed: int = 42,
    ratios: Sequence[Fraction | int | float | str] = GPQA_SPLIT_RATIOS,
    partition_names: Sequence[str] = GPQA_PARTITIONS,
) -> dict[str, tuple[str, ...]]:
    """Split the entire raw-ID universe before any clean-correct filtering."""

    ids = validate_unique_raw_ids([str(value) for value in raw_ids])
    if len(partition_names) != len(ratios):
        raise ValueError("partition_names and ratios must have the same length.")
    if len(set(partition_names)) != len(partition_names):
        raise ValueError("partition_names must be unique.")
    shuffled = sorted(ids)
    random.Random(int(seed)).shuffle(shuffled)
    counts = largest_remainder_counts(len(shuffled), ratios)
    result: dict[str, tuple[str, ...]] = {}
    cursor = 0
    for name, count in zip(partition_names, counts):
        result[str(name)] = tuple(shuffled[cursor : cursor + count])
        cursor += count
    if cursor != len(ids):  # defensive assertion with a user-facing failure.
        raise AssertionError("Raw split allocation was not exhaustive.")
    return result


def raw_split_assignment(
    raw_ids: Sequence[str],
    seed: int = 42,
) -> dict[str, str]:
    split = split_raw_ids(raw_ids, seed=seed)
    return {raw_id: partition for partition, values in split.items() for raw_id in values}


# ---------------------------------------------------------------------------
# Strict generated-choice parsing and deterministic score ties
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChoiceMatch:
    label: str
    form: str
    start: int
    end: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrictChoiceResult:
    choice: Optional[str]
    answer_invalid: bool
    answer_conflict: bool
    matched_spans: tuple[ChoiceMatch, ...]
    checker_version: str = STRICT_CHOICE_VERSION

    @property
    def valid(self) -> bool:
        return not self.answer_invalid and self.choice is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "choice": self.choice,
            "answer_invalid": self.answer_invalid,
            "answer_conflict": self.answer_conflict,
            "matched_spans": [match.to_dict() for match in self.matched_spans],
            "checker_version": self.checker_version,
        }


@dataclass(frozen=True)
class StrictChoiceCheck:
    parsed: StrictChoiceResult
    gold: str
    is_correct: bool

    @property
    def choice(self) -> Optional[str]:
        return self.parsed.choice

    @property
    def answer_invalid(self) -> bool:
        return self.parsed.answer_invalid

    @property
    def answer_conflict(self) -> bool:
        return self.parsed.answer_conflict

    def to_dict(self) -> dict[str, Any]:
        result = self.parsed.to_dict()
        result.update({"gold": self.gold, "is_correct": self.is_correct})
        return result


_BOXED_CHOICE_RE = re.compile(r"\\boxed\s*\{\s*([A-D])\s*\}", re.IGNORECASE)
_EXPLICIT_CHOICE_RE = re.compile(
    r"(?im)^\s*(?:final\s+(?:choice|answer)|answer|choice|option)\s*[:\-]\s*"
    r"(?:\\boxed\s*\{\s*)?([A-D])(?:\s*\})?\s*[.!]?\s*$"
)
_STANDALONE_CHOICE_RE = re.compile(
    r"^\s*(?:[\(\[]\s*)?([A-D])(?:\s*[\)\]])?\s*[.!]?\s*$",
    re.IGNORECASE,
)
_TERMINAL_BOX_SUFFIX_RE = re.compile(
    r"(?:\\[)\]]\s*)?[\s.!?]*"
)


def parse_strict_choice(text: Any) -> StrictChoiceResult:
    """Parse only declared, unambiguous final-answer forms.

    Unlike the legacy evaluator, this function never defaults an invalid answer to
    A.  Ordinary A/B/C/D mentions in reasoning are intentionally ignored.
    """

    source = "" if text is None else str(text)
    matches: list[ChoiceMatch] = []
    seen_spans: set[tuple[int, int, str]] = set()

    def add_match(match: re.Match[str], form: str, *, offset: int = 0) -> None:
        label = match.group(1).upper()
        start, end = offset + match.start(), offset + match.end()
        key = (start, end, label)
        if key not in seen_spans:
            seen_spans.add(key)
            matches.append(ChoiceMatch(label, form, start, end, match.group(0)))

    # Parse only the contiguous terminal answer block.  Working backwards makes
    # an explicit final line (or repeated explicit lines) auditable while a later
    # reasoning/qualification line invalidates an earlier apparent answer.  A
    # boxed label embedded in prose is accepted only when the last boxed span is
    # itself terminal on that line; all boxed spans in such a terminal line are
    # retained so contradictory declarations fail closed.
    offset = 0
    nonempty_lines: list[tuple[str, int]] = []
    for line in source.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        if content.strip():
            nonempty_lines.append((content, offset))
        offset += len(line)
    if not nonempty_lines and source.strip():
        nonempty_lines.append((source, 0))

    for line, line_offset in reversed(nonempty_lines):
        explicit = _EXPLICIT_CHOICE_RE.fullmatch(line)
        if explicit is not None:
            add_match(explicit, "explicit_final", offset=line_offset)
            continue
        standalone = _STANDALONE_CHOICE_RE.fullmatch(line)
        if standalone is not None:
            add_match(standalone, "standalone_terminal", offset=line_offset)
            continue
        boxed = list(_BOXED_CHOICE_RE.finditer(line))
        if boxed and _TERMINAL_BOX_SUFFIX_RE.fullmatch(
            line[boxed[-1].end() :]
        ):
            for match in boxed:
                add_match(match, "boxed_terminal", offset=line_offset)
            continue
        break

    matches.sort(key=lambda item: (item.start, item.end, item.form))
    labels = {match.label for match in matches}
    conflict = len(labels) > 1
    choice = next(iter(labels)) if len(labels) == 1 else None
    return StrictChoiceResult(
        choice=choice,
        answer_invalid=(choice is None),
        answer_conflict=conflict,
        matched_spans=tuple(matches),
    )


def strict_choice_parse(text: Any) -> StrictChoiceResult:
    """Compatibility alias with the versioned result shape."""

    return parse_strict_choice(text)


def check_strict_choice(text: Any, gold: str) -> StrictChoiceCheck:
    normalized_gold = str(gold).strip().upper()
    if normalized_gold not in CHOICE_LABELS:
        raise ValueError(f"gold must be one of {CHOICE_LABELS}, got {gold!r}.")
    parsed = parse_strict_choice(text)
    return StrictChoiceCheck(parsed, normalized_gold, parsed.choice == normalized_gold)


@dataclass(frozen=True)
class ChoiceScoreDecision:
    prediction: Optional[str]
    score_tie: bool
    tied_labels: tuple[str, ...]
    scores: dict[str, float]
    tie_tolerance: float


def forced_choice_prediction(
    scores: Mapping[str, float],
    tie_tolerance: float = 0.0,
) -> ChoiceScoreDecision:
    if float(tie_tolerance) < 0 or not math.isfinite(float(tie_tolerance)):
        raise ValueError("tie_tolerance must be finite and non-negative.")
    normalized = {label: float(scores[label]) for label in CHOICE_LABELS if label in scores}
    if set(normalized) != set(CHOICE_LABELS):
        raise ValueError("scores must contain exactly A, B, C, and D.")
    if not all(math.isfinite(value) for value in normalized.values()):
        raise ValueError("All forced-choice scores must be finite.")
    maximum = max(normalized.values())
    tied = tuple(label for label in CHOICE_LABELS if maximum - normalized[label] <= tie_tolerance)
    prediction = tied[0] if len(tied) == 1 else None
    return ChoiceScoreDecision(prediction, len(tied) > 1, tied, normalized, float(tie_tolerance))


# ---------------------------------------------------------------------------
# Isometric perturbation subspaces and stable directions
# ---------------------------------------------------------------------------


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("This LinkRadius tensor operation requires PyTorch.")
    return torch


def _is_torch_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


@dataclass(frozen=True)
class PerturbationSubspace:
    name: str
    token_count: int
    hidden_dim: int
    version: str = SUBSPACE_VERSION

    def __post_init__(self) -> None:
        name = str(self.name).strip().lower()
        if name not in {"full_tensor", "channel_broadcast"}:
            raise ValueError("subspace name must be 'full_tensor' or 'channel_broadcast'.")
        if isinstance(self.token_count, bool) or int(self.token_count) != self.token_count or int(self.token_count) < 1:
            raise ValueError("token_count must be a positive integer.")
        if isinstance(self.hidden_dim, bool) or int(self.hidden_dim) != self.hidden_dim or int(self.hidden_dim) < 1:
            raise ValueError("hidden_dim must be a positive integer.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "token_count", int(self.token_count))
        object.__setattr__(self, "hidden_dim", int(self.hidden_dim))

    @property
    def q(self) -> int:
        return self.token_count * self.hidden_dim if self.name == "full_tensor" else self.hidden_dim

    @property
    def effective_dimension(self) -> int:
        return self.q

    @property
    def subspace_id(self) -> str:
        return f"{self.name}_v1:T{self.token_count}:D{self.hidden_dim}"

    def lift(self, coefficients: Any) -> Any:
        """Apply the isometric lift B_e to one or a batch of q-vectors."""

        if _is_torch_tensor(coefficients):
            if coefficients.shape[-1] != self.q:
                raise ValueError(f"Expected coefficient dimension q={self.q}, got {coefficients.shape[-1]}.")
            if self.name == "full_tensor":
                return coefficients.reshape(*coefficients.shape[:-1], self.token_count, self.hidden_dim)
            channels = coefficients.reshape(*coefficients.shape[:-1], 1, self.hidden_dim)
            return channels.expand(*coefficients.shape[:-1], self.token_count, self.hidden_dim) / math.sqrt(
                self.token_count
            )

        # Lightweight standard-library path for manifest/self-tests without torch.
        values = tuple(float(value) for value in coefficients)
        if len(values) != self.q:
            raise ValueError(f"Expected coefficient dimension q={self.q}, got {len(values)}.")
        if self.name == "full_tensor":
            return tuple(
                tuple(values[row * self.hidden_dim : (row + 1) * self.hidden_dim])
                for row in range(self.token_count)
            )
        scale = math.sqrt(self.token_count)
        row = tuple(value / scale for value in values)
        return tuple(row for _ in range(self.token_count))

    def adjoint(self, relay_tensor: Any) -> Any:
        """Apply ``B_e^T`` to a relay tensor.

        This is the coordinate-space gradient used when an autograd reference is
        compared with probes in a declared subspace. In particular, a
        ``channel_broadcast`` experiment compares against the norm of
        ``sum_t g_t / sqrt(T)``, not the full-tensor gradient norm.
        """

        if _is_torch_tensor(relay_tensor):
            if tuple(relay_tensor.shape[-2:]) != (
                self.token_count,
                self.hidden_dim,
            ):
                raise ValueError(
                    "Expected relay trailing shape "
                    f"[{self.token_count},{self.hidden_dim}], got "
                    f"{tuple(relay_tensor.shape[-2:])}."
                )
            if self.name == "full_tensor":
                return relay_tensor.reshape(*relay_tensor.shape[:-2], self.q)
            return relay_tensor.sum(dim=-2) / math.sqrt(self.token_count)

        rows = tuple(tuple(float(value) for value in row) for row in relay_tensor)
        if len(rows) != self.token_count or any(
            len(row) != self.hidden_dim for row in rows
        ):
            raise ValueError(
                "Expected relay shape "
                f"[{self.token_count},{self.hidden_dim}] for the subspace adjoint."
            )
        if self.name == "full_tensor":
            return tuple(value for row in rows for value in row)
        scale = math.sqrt(self.token_count)
        return tuple(
            sum(row[channel] for row in rows) / scale
            for channel in range(self.hidden_dim)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "token_count": self.token_count,
            "hidden_dim": self.hidden_dim,
            "q": self.q,
            "subspace_id": self.subspace_id,
        }


def get_subspace(name: str, token_count: int, hidden_dim: int) -> PerturbationSubspace:
    return PerturbationSubspace(name, token_count, hidden_dim)


def stable_intervention_seed(
    global_seed: int,
    raw_sample_id: str,
    edge: Edge | str | Mapping[str, Any],
    probe_seed: int = 0,
    direction_id: int = 0,
    purpose: str = "probe",
) -> int:
    selected = parse_edge(edge)
    if not str(raw_sample_id):
        raise ValueError("raw_sample_id must be nonempty.")
    if isinstance(direction_id, bool) or int(direction_id) != direction_id or int(direction_id) < 0:
        raise ValueError("direction_id must be a non-negative integer.")
    payload = {
        "direction_id": int(direction_id),
        "global_seed": int(global_seed),
        "probe_seed": int(probe_seed),
        "purpose": str(purpose),
        "raw_sample_id": str(raw_sample_id),
        "round_idx": selected.round_idx,
        "site": selected.site,
        "version": INTERVENTION_SEED_VERSION,
    }
    digest = hashlib.sha256(
        b"linkradius:intervention_seed:v1\0" + _canonical_json_bytes(payload)
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63)


def sample_stable_unit_direction(
    global_seed: int,
    raw_sample_id: str,
    edge: Edge | str | Mapping[str, Any],
    subspace: PerturbationSubspace,
    probe_seed: int = 0,
    direction_id: int = 0,
    purpose: str = "probe",
) -> Any:
    """Sample one CPU float32 q-vector, independent of batching and sharding."""

    seed = stable_intervention_seed(
        global_seed,
        raw_sample_id,
        edge,
        probe_seed=probe_seed,
        direction_id=direction_id,
        purpose=purpose,
    )
    if torch is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        direction = torch.randn(subspace.q, generator=generator, dtype=torch.float32, device="cpu")
        norm = torch.linalg.vector_norm(direction)
        if not bool(torch.isfinite(norm)) or float(norm.item()) == 0.0:
            raise RuntimeError("Stable direction generator produced a non-finite or zero vector.")
        return direction / norm

    # Import-only CPU environments still get a deterministic, testable direction.
    rng = random.Random(seed)
    values = [rng.gauss(0.0, 1.0) for _ in range(subspace.q)]
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm == 0.0:
        raise RuntimeError("Stable direction generator produced a non-finite or zero vector.")
    return tuple(value / norm for value in values)


def sample_stable_lifted_direction(
    global_seed: int,
    raw_sample_id: str,
    edge: Edge | str | Mapping[str, Any],
    subspace: PerturbationSubspace,
    probe_seed: int = 0,
    direction_id: int = 0,
    purpose: str = "probe",
) -> Any:
    return subspace.lift(
        sample_stable_unit_direction(
            global_seed,
            raw_sample_id,
            edge,
            subspace,
            probe_seed=probe_seed,
            direction_id=direction_id,
            purpose=purpose,
        )
    )


def project_subspace_coefficients(coefficients: Any, radius: float) -> Any:
    if float(radius) < 0 or not math.isfinite(float(radius)):
        raise ValueError("radius must be finite and non-negative.")
    if _is_torch_tensor(coefficients):
        norm = torch.linalg.vector_norm(coefficients, dim=-1, keepdim=True)
        scale = torch.clamp(float(radius) / norm.clamp_min(torch.finfo(coefficients.dtype).tiny), max=1.0)
        return coefficients * scale
    values = tuple(float(value) for value in coefficients)
    norm = math.sqrt(sum(value * value for value in values))
    scale = min(1.0, float(radius) / norm) if norm else 1.0
    return tuple(value * scale for value in values)


# ---------------------------------------------------------------------------
# Relay controls and requested/realized perturbation diagnostics
# ---------------------------------------------------------------------------


def _float32_tensor(value: Any) -> Any:
    t = _require_torch()
    if not isinstance(value, t.Tensor):
        value = t.as_tensor(value)
    return value.to(dtype=t.float32)


def _flat_norm(value: Any) -> Any:
    return torch.linalg.vector_norm(value.reshape(-1))


def identity_intervention(z_ref: Any) -> Any:
    return _float32_tensor(z_ref).clone()


def zero_intervention(z_ref: Any) -> Any:
    return torch.zeros_like(_float32_tensor(z_ref))


def additive_intervention(z_ref: Any, requested_delta: Any) -> Any:
    reference = _float32_tensor(z_ref)
    delta = _float32_tensor(requested_delta).to(device=reference.device)
    if tuple(reference.shape) != tuple(delta.shape):
        raise ValueError(
            f"requested_delta shape {tuple(delta.shape)} does not match z_ref {tuple(reference.shape)}."
        )
    return reference + delta


def requested_additive_delta(
    z_ref: Any,
    lifted_unit_direction: Any,
    h: float,
    sign: int,
) -> Any:
    if sign not in (-1, 1):
        raise ValueError("sign must be exactly +1 or -1.")
    if float(h) < 0 or not math.isfinite(float(h)):
        raise ValueError("h must be finite and non-negative.")
    reference = _float32_tensor(z_ref)
    direction = _float32_tensor(lifted_unit_direction).to(device=reference.device)
    if tuple(reference.shape) != tuple(direction.shape):
        raise ValueError("lifted direction and z_ref must have the same shape.")
    direction_norm = _flat_norm(direction)
    if not torch.isfinite(direction_norm) or float(direction_norm.item()) == 0.0:
        raise ValueError("lifted direction must have a finite, non-zero norm.")
    direction = direction / direction_norm
    return int(sign) * float(h) * _flat_norm(reference) * direction


def postcast_budget_fitted_delta(
    z_ref: Any,
    lifted_unit_direction: Any,
    *,
    relative_budget: float,
    consumer_dtype: Any,
    iterations: int = 40,
) -> Any:
    """Fit a directional delta to the largest feasible post-cast norm.

    The returned float32 request follows ``lifted_unit_direction``, while the
    budget is enforced on ``cast(z_ref + delta) - z_ref``.  A short expansion
    followed by bisection compensates for low-precision rounding that can make
    a pre-cast normalized request either exceed or substantially undershoot the
    declared realized budget.
    """

    if not math.isfinite(float(relative_budget)) or float(relative_budget) < 0.0:
        raise ValueError("relative_budget must be finite and non-negative")
    if isinstance(iterations, bool) or int(iterations) < 1:
        raise ValueError("iterations must be a positive integer")
    reference = _float32_tensor(z_ref)
    direction = _float32_tensor(lifted_unit_direction).to(device=reference.device)
    if tuple(reference.shape) != tuple(direction.shape):
        raise ValueError("lifted direction and z_ref must have the same shape")
    direction_norm = _flat_norm(direction)
    if not torch.isfinite(direction_norm) or float(direction_norm.item()) == 0.0:
        raise ValueError("lifted direction must have a finite, non-zero norm")
    unit = direction / direction_norm
    clean_norm = float(_flat_norm(reference).item())
    target = float(relative_budget) * clean_norm
    if target == 0.0:
        return torch.zeros_like(reference)

    def realized_norm(scale: float) -> float:
        realized = (
            cast_receiver_tensor(reference + float(scale) * unit, consumer_dtype)
            .to(dtype=torch.float32)
            - reference
        )
        value = float(_flat_norm(realized).item())
        return value if math.isfinite(value) else math.inf

    tolerance = 1e-7 * max(1.0, target)
    lower = 0.0
    upper = target
    best = 0.0
    # The nominal coefficient is normally already close.  Expansion matters
    # only when the consumer cast rounds much of a small request away.
    for _ in range(16):
        if realized_norm(upper) <= target + tolerance:
            best = upper
            lower = upper
            upper *= 2.0
        else:
            break
    for _ in range(int(iterations)):
        midpoint = (lower + upper) / 2.0
        if realized_norm(midpoint) <= target + tolerance:
            best = midpoint
            lower = midpoint
        else:
            upper = midpoint
    return float(best) * unit


@dataclass(frozen=True)
class MomentNoiseDiagnostics:
    seed: int
    requested_mean: float
    requested_std: float
    requested_norm: float
    target_mean: float
    target_std: float
    zero_variance: bool
    version: str = MOMENT_NOISE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def moment_noise_intervention(
    z_ref: Any,
    *,
    global_seed: int,
    raw_sample_id: str,
    edge: Edge | str | Mapping[str, Any],
    return_diagnostics: bool = False,
) -> Any:
    """Generate deterministic per-recipient global-moment-matched noise.

    Standard deviations are population standard deviations (``correction=0``)
    over the complete valid [T,D] tensor.  This definition is frozen in v1.
    """

    reference = _float32_tensor(z_ref)
    if reference.ndim != 2 or reference.numel() == 0:
        raise ValueError(f"z_ref must be a nonempty [T,D] tensor, got {tuple(reference.shape)}.")
    seed = stable_intervention_seed(
        global_seed,
        raw_sample_id,
        edge,
        purpose=MOMENT_NOISE_VERSION,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    raw = torch.randn(tuple(reference.shape), generator=generator, dtype=torch.float32, device="cpu")
    raw_mean = raw.mean()
    raw_std = raw.std(correction=0)
    target_mean = reference.mean()
    target_std = reference.std(correction=0)
    zero_variance = float(target_std.item()) == 0.0
    if zero_variance:
        requested = torch.full_like(raw, float(target_mean.item()))
    else:
        requested = (raw - raw_mean) / raw_std * target_std.detach().cpu() + target_mean.detach().cpu()
    requested = requested.to(device=reference.device, dtype=torch.float32)
    diagnostics = MomentNoiseDiagnostics(
        seed=seed,
        requested_mean=float(requested.mean().item()),
        requested_std=float(requested.std(correction=0).item()),
        requested_norm=float(_flat_norm(requested).item()),
        target_mean=float(target_mean.item()),
        target_std=float(target_std.item()),
        zero_variance=zero_variance,
    )
    if return_diagnostics:
        return requested, diagnostics
    return requested


@dataclass(frozen=True)
class DonorAssignment:
    recipient_id: str
    donor_id: Optional[str]
    available: bool
    reason: Optional[str]
    stratum: tuple[Any, ...]
    version: str = DONOR_MAPPING_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _record_value(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _donor_stratum(record: Mapping[str, Any]) -> tuple[Any, ...]:
    raw_edge = _record_value(record, "edge", "edge_id")
    if raw_edge is None:
        site = _record_value(record, "site")
        round_idx = _record_value(record, "round_idx", "code_round", "round")
        if site is None or round_idx is None:
            raise ValueError("Donor record requires edge/edge_id or site and round_idx.")
        raw_edge = Edge(str(site), int(round_idx)).edge_id
    else:
        raw_edge = parse_edge(raw_edge).edge_id
    shape = _record_value(record, "tensor_shape", "shape")
    if shape is None:
        raise ValueError("Donor record requires tensor_shape/shape.")
    shape_tuple = tuple(int(value) for value in shape)
    if not shape_tuple or any(value < 1 for value in shape_tuple):
        raise ValueError(f"Invalid donor tensor shape {shape!r}.")
    partition = _record_value(record, "partition")
    gold = _record_value(record, "gold_label", "gold")
    horizon = _record_value(record, "R", "horizon")
    if partition is None or gold is None or horizon is None:
        raise ValueError("Donor record requires partition, gold_label/gold, and R/horizon.")
    normalized_gold = str(gold).upper()
    if normalized_gold not in CHOICE_LABELS:
        raise ValueError(f"Donor gold label must be one of {CHOICE_LABELS}, got {gold!r}.")
    validate_edge(raw_edge, int(horizon))
    length_bucket = _record_value(record, "length_bucket")
    if length_bucket is None:
        length_bucket = shape_tuple[0]
    return (
        str(partition),
        normalized_gold,
        raw_edge,
        int(horizon),
        shape_tuple,
        str(length_bucket),
    )


def deterministic_donor_assignments(
    records: Sequence[Mapping[str, Any]],
    donor_seed: int = 42,
) -> dict[str, DonorAssignment]:
    strata: dict[tuple[Any, ...], list[str]] = {}
    seen: set[str] = set()
    for record in records:
        recipient_value = _record_value(record, "raw_sample_id", "sample_id")
        if recipient_value is None or not str(recipient_value):
            raise ValueError("Donor record requires nonempty raw_sample_id/sample_id.")
        recipient_id = str(recipient_value)
        if recipient_id in seen:
            raise ValueError(f"Duplicate donor recipient ID {recipient_id!r}.")
        seen.add(recipient_id)
        strata.setdefault(_donor_stratum(record), []).append(recipient_id)

    result: dict[str, DonorAssignment] = {}
    for stratum, ids in strata.items():
        ordered = sorted(
            ids,
            key=lambda raw_id: hashlib.sha256(
                b"linkradius:donor_order:v1\0"
                + _canonical_json_bytes({"donor_seed": int(donor_seed), "raw_sample_id": raw_id})
            ).digest(),
        )
        if len(ordered) < 2:
            recipient_id = ordered[0]
            result[recipient_id] = DonorAssignment(
                recipient_id, None, False, "stratum_too_small", stratum
            )
            continue
        for index, recipient_id in enumerate(ordered):
            donor_id = ordered[(index + 1) % len(ordered)]
            result[recipient_id] = DonorAssignment(recipient_id, donor_id, True, None, stratum)
    return result


def deterministic_donor_mapping(
    records: Sequence[Mapping[str, Any]],
    donor_seed: int = 42,
) -> dict[str, Optional[str]]:
    return {
        recipient_id: assignment.donor_id
        for recipient_id, assignment in deterministic_donor_assignments(records, donor_seed).items()
    }


@dataclass(frozen=True)
class MismatchDiagnostics:
    available: bool
    reason: Optional[str]
    source_norm: float
    target_norm: float
    requested_norm: Optional[float]
    scale: Optional[float]


def mismatch_intervention(z_ref: Any, donor_ref: Any) -> tuple[Optional[Any], MismatchDiagnostics]:
    reference = _float32_tensor(z_ref)
    donor = _float32_tensor(donor_ref).to(device=reference.device)
    if tuple(reference.shape) != tuple(donor.shape):
        raise ValueError(f"Donor shape {tuple(donor.shape)} does not match recipient {tuple(reference.shape)}.")
    target_norm = float(_flat_norm(reference).item())
    source_norm = float(_flat_norm(donor).item())
    if target_norm == 0.0:
        return None, MismatchDiagnostics(False, "recipient_zero_norm", source_norm, target_norm, None, None)
    if source_norm == 0.0:
        return None, MismatchDiagnostics(False, "donor_zero_norm", source_norm, target_norm, None, None)
    scale = target_norm / source_norm
    requested = donor * scale
    return requested, MismatchDiagnostics(
        True,
        None,
        source_norm,
        target_norm,
        float(_flat_norm(requested).item()),
        scale,
    )


def _resolve_torch_dtype(dtype: Any) -> Any:
    if dtype is None:
        return None
    if torch is not None and isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).replace("torch.", "").strip().lower()
    aliases = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
    key = aliases.get(key, key)
    if key not in {"float32", "float16", "bfloat16", "float64"}:
        raise ValueError(f"Unsupported consumer dtype {dtype!r}.")
    return getattr(_require_torch(), key)


def cast_receiver_tensor(value: Any, consumer_dtype: Any) -> Any:
    tensor = _float32_tensor(value)
    dtype = _resolve_torch_dtype(consumer_dtype)
    return tensor if dtype is None else tensor.to(dtype=dtype)


@dataclass(frozen=True)
class RealizedDeltaDiagnostics:
    clean_norm: float
    requested_delta_norm: float
    requested_relative_norm: float
    realized_delta_norm: float
    realized_relative_norm: float
    requested_signed_coordinate: Optional[float]
    realized_signed_coordinate: Optional[float]
    signed_projection: Optional[float]
    requested_realized_cosine: Optional[float]
    off_direction_residual_norm: Optional[float]
    off_direction_relative: Optional[float]
    collapsed: bool
    consumer_dtype: str
    requested_delta: Any = field(repr=False, compare=False)
    realized_delta: Any = field(repr=False, compare=False)
    realized_value: Any = field(repr=False, compare=False)
    version: str = DELTA_DIAGNOSTICS_VERSION

    def to_dict(self) -> dict[str, Any]:
        # Avoid dataclasses.asdict here: it deep-copies tensors, which is expensive
        # and fails for some non-leaf tensors.  Public records contain diagnostics,
        # never the in-memory tensor payloads.
        return {
            "clean_norm": self.clean_norm,
            "requested_delta_norm": self.requested_delta_norm,
            "requested_relative_norm": self.requested_relative_norm,
            "realized_delta_norm": self.realized_delta_norm,
            "realized_relative_norm": self.realized_relative_norm,
            "requested_signed_coordinate": self.requested_signed_coordinate,
            "realized_signed_coordinate": self.realized_signed_coordinate,
            "signed_projection": self.signed_projection,
            "requested_realized_cosine": self.requested_realized_cosine,
            "off_direction_residual_norm": self.off_direction_residual_norm,
            "off_direction_relative": self.off_direction_relative,
            "collapsed": self.collapsed,
            "consumer_dtype": self.consumer_dtype,
            "version": self.version,
        }


def realized_delta_diagnostics(
    z_ref: Any,
    requested_delta: Any,
    *,
    consumer_dtype: Any,
    lifted_unit_direction: Optional[Any] = None,
    collapse_tolerance: float = 0.0,
) -> RealizedDeltaDiagnostics:
    """Apply the consumer cast and measure offsets in the frozen z_ref convention."""

    if float(collapse_tolerance) < 0:
        raise ValueError("collapse_tolerance must be non-negative.")
    reference = _float32_tensor(z_ref)
    requested = _float32_tensor(requested_delta).to(device=reference.device)
    if tuple(reference.shape) != tuple(requested.shape):
        raise ValueError("z_ref and requested_delta must have the same shape.")
    cast_value = cast_receiver_tensor(reference + requested, consumer_dtype)
    realized_value = cast_value.to(dtype=torch.float32)
    realized = realized_value - reference
    clean_norm = float(_flat_norm(reference).item())
    requested_norm = float(_flat_norm(requested).item())
    realized_norm = float(_flat_norm(realized).item())
    requested_relative = requested_norm / clean_norm if clean_norm > 0 else 0.0
    realized_relative = realized_norm / clean_norm if clean_norm > 0 else 0.0

    requested_coordinate: Optional[float] = None
    realized_coordinate: Optional[float] = None
    projection: Optional[float] = None
    cosine: Optional[float] = None
    residual_norm: Optional[float] = None
    residual_relative: Optional[float] = None
    if lifted_unit_direction is not None:
        direction = _float32_tensor(lifted_unit_direction).to(device=reference.device)
        if tuple(direction.shape) != tuple(reference.shape):
            raise ValueError("lifted_unit_direction and z_ref must have the same shape.")
        direction_norm = _flat_norm(direction)
        if not torch.isfinite(direction_norm) or float(direction_norm.item()) == 0.0:
            raise ValueError("lifted_unit_direction must be finite and non-zero.")
        unit = direction / direction_norm
        requested_projection = float(torch.sum(requested * unit).item())
        projection = float(torch.sum(realized * unit).item())
        requested_coordinate = requested_projection / clean_norm if clean_norm > 0 else None
        realized_coordinate = projection / clean_norm if clean_norm > 0 else None
        residual = realized - projection * unit
        residual_norm = float(_flat_norm(residual).item())
        residual_relative = residual_norm / realized_norm if realized_norm > 0 else 0.0
        if requested_norm > 0 and realized_norm > 0:
            cosine = float(torch.sum(requested * realized).item()) / (requested_norm * realized_norm)
            cosine = max(-1.0, min(1.0, cosine))

    return RealizedDeltaDiagnostics(
        clean_norm=clean_norm,
        requested_delta_norm=requested_norm,
        requested_relative_norm=requested_relative,
        realized_delta_norm=realized_norm,
        realized_relative_norm=realized_relative,
        requested_signed_coordinate=requested_coordinate,
        realized_signed_coordinate=realized_coordinate,
        signed_projection=projection,
        requested_realized_cosine=cosine,
        off_direction_residual_norm=residual_norm,
        off_direction_relative=residual_relative,
        collapsed=realized_norm <= float(collapse_tolerance),
        consumer_dtype=str(_resolve_torch_dtype(consumer_dtype)).replace("torch.", ""),
        requested_delta=requested,
        realized_delta=realized,
        realized_value=realized_value,
    )


@dataclass(frozen=True)
class ProbeAcceptanceThresholds:
    minimum_requested_realized_cosine: float = 0.0
    maximum_off_direction_relative: float = 1.0
    minimum_signed_separation: float = 0.0
    minimum_antipodality: float = -1.0
    version: str = "linkradius_probe_thresholds_v1"

    def __post_init__(self) -> None:
        if not -1 <= self.minimum_requested_realized_cosine <= 1:
            raise ValueError("minimum_requested_realized_cosine must lie in [-1,1].")
        if self.maximum_off_direction_relative < 0:
            raise ValueError("maximum_off_direction_relative must be non-negative.")
        if self.minimum_signed_separation < 0:
            raise ValueError("minimum_signed_separation must be non-negative.")
        if not -1 <= self.minimum_antipodality <= 1:
            raise ValueError("minimum_antipodality must lie in [-1,1].")


@dataclass(frozen=True)
class ProbePairDiagnostics:
    t_plus: Optional[float]
    t_minus: Optional[float]
    realized_separation: Optional[float]
    antipodality: Optional[float]
    accepted: bool
    rejection_reasons: tuple[str, ...]
    version: str = PROBE_PAIR_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_pair_diagnostics(
    plus: RealizedDeltaDiagnostics,
    minus: RealizedDeltaDiagnostics,
    thresholds: ProbeAcceptanceThresholds = ProbeAcceptanceThresholds(),
) -> ProbePairDiagnostics:
    reasons: list[str] = []
    if plus.collapsed:
        reasons.append("plus_collapsed")
    if minus.collapsed:
        reasons.append("minus_collapsed")
    for name, item in (("plus", plus), ("minus", minus)):
        if item.requested_realized_cosine is None:
            reasons.append(f"{name}_cosine_unavailable")
        elif item.requested_realized_cosine < thresholds.minimum_requested_realized_cosine:
            reasons.append(f"{name}_cosine_below_threshold")
        if item.off_direction_relative is None:
            reasons.append(f"{name}_residual_unavailable")
        elif item.off_direction_relative > thresholds.maximum_off_direction_relative:
            reasons.append(f"{name}_residual_above_threshold")

    t_plus = plus.realized_signed_coordinate
    t_minus = minus.realized_signed_coordinate
    separation = None if t_plus is None or t_minus is None else t_plus - t_minus
    if separation is None:
        reasons.append("separation_unavailable")
    elif separation <= 0:
        reasons.append("nonpositive_signed_separation")
    elif separation < thresholds.minimum_signed_separation:
        reasons.append("separation_below_threshold")

    antipodality: Optional[float] = None
    plus_norm = plus.realized_delta_norm
    minus_norm = minus.realized_delta_norm
    if plus_norm > 0 and minus_norm > 0:
        dot = float(torch.sum(plus.realized_delta * (-minus.realized_delta)).item())
        antipodality = max(-1.0, min(1.0, dot / (plus_norm * minus_norm)))
        if antipodality < thresholds.minimum_antipodality:
            reasons.append("antipodality_below_threshold")
    else:
        reasons.append("antipodality_unavailable")

    return ProbePairDiagnostics(
        t_plus=t_plus,
        t_minus=t_minus,
        realized_separation=separation,
        antipodality=antipodality,
        accepted=not reasons,
        rejection_reasons=tuple(dict.fromkeys(reasons)),
    )


# ---------------------------------------------------------------------------
# Antithetic estimator, susceptibilities, and competitor-specific radii
# ---------------------------------------------------------------------------


def central_difference(
    margin_plus: float,
    margin_minus: float,
    t_plus: float,
    t_minus: float,
) -> float:
    values = tuple(float(value) for value in (margin_plus, margin_minus, t_plus, t_minus))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Central-difference inputs must be finite.")
    separation = values[2] - values[3]
    if separation <= 0:
        raise ValueError("Realized signed separation t_plus - t_minus must be positive.")
    return (values[0] - values[1]) / separation


def susceptibility_squared(directional_derivatives: Sequence[float], q: int) -> float:
    if isinstance(q, bool) or int(q) != q or int(q) < 1:
        raise ValueError("q must be a positive integer.")
    derivatives = tuple(float(value) for value in directional_derivatives)
    if not derivatives:
        raise ValueError("At least one accepted directional derivative is required.")
    if not all(math.isfinite(value) for value in derivatives):
        raise ValueError("Directional derivatives must be finite.")
    return int(q) * math.fsum(value * value for value in derivatives) / len(derivatives)


def estimate_susceptibility(directional_derivatives: Sequence[float], q: int) -> float:
    return math.sqrt(susceptibility_squared(directional_derivatives, q))


@dataclass(frozen=True)
class CompetitorRadiusEstimate:
    competitor: str
    clean_margin: float
    susceptibility_squared: float
    susceptibility: float
    radius: float
    zero_susceptibility: bool
    k_eff: int
    q: int
    version: str = ESTIMATOR_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LinkRadiusEstimate:
    competitors: dict[str, CompetitorRadiusEstimate]
    radius: float
    binding_competitor: str
    binding_competitors: tuple[str, ...]
    k_eff: int
    q: int
    version: str = ESTIMATOR_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "competitors": {key: value.to_dict() for key, value in self.competitors.items()},
            "radius": self.radius,
            "binding_competitor": self.binding_competitor,
            "binding_competitors": list(self.binding_competitors),
            "k_eff": self.k_eff,
            "q": self.q,
            "version": self.version,
        }


def estimate_competitor_radii(
    clean_margins: Mapping[str, float],
    directional_derivatives: Mapping[str, Sequence[float]],
    q: int,
) -> LinkRadiusEstimate:
    if not clean_margins:
        raise ValueError("clean_margins must contain every wrong-option competitor.")
    if set(clean_margins) != set(directional_derivatives):
        raise ValueError("clean_margins and directional_derivatives must have identical competitors.")
    derivative_lengths = {len(tuple(values)) for values in directional_derivatives.values()}
    if len(derivative_lengths) != 1 or not derivative_lengths or next(iter(derivative_lengths)) < 1:
        raise ValueError("Every competitor must have the same positive K_eff.")
    k_eff = next(iter(derivative_lengths))
    estimates: dict[str, CompetitorRadiusEstimate] = {}
    for competitor in sorted(clean_margins):
        margin = float(clean_margins[competitor])
        if not math.isfinite(margin) or margin <= 0:
            raise ValueError(
                f"Clean margin for competitor {competitor!r} must be finite and positive; got {margin}."
            )
        derivatives = tuple(float(value) for value in directional_derivatives[competitor])
        chi2 = susceptibility_squared(derivatives, q)
        chi = math.sqrt(chi2)
        zero = chi == 0.0
        radius = math.inf if zero else margin / chi
        estimates[competitor] = CompetitorRadiusEstimate(
            competitor=competitor,
            clean_margin=margin,
            susceptibility_squared=chi2,
            susceptibility=chi,
            radius=radius,
            zero_susceptibility=zero,
            k_eff=k_eff,
            q=int(q),
        )
    radius = min(item.radius for item in estimates.values())
    bindings = tuple(
        key
        for key in sorted(estimates)
        if estimates[key].radius == radius
        or math.isclose(estimates[key].radius, radius, rel_tol=1e-12, abs_tol=0.0)
    )
    return LinkRadiusEstimate(estimates, radius, bindings[0], bindings, k_eff, int(q))


estimate_linkradius = estimate_competitor_radii


@dataclass(frozen=True)
class ProbePairObservation:
    direction_id: int
    t_plus: float
    t_minus: float
    margins_plus: Mapping[str, float]
    margins_minus: Mapping[str, float]
    accepted: bool
    rejection_reason: Optional[str] = None

    def derivatives(self) -> dict[str, float]:
        if set(self.margins_plus) != set(self.margins_minus):
            raise ValueError("Plus/minus probe margins have different competitors.")
        return {
            competitor: central_difference(
                self.margins_plus[competitor],
                self.margins_minus[competitor],
                self.t_plus,
                self.t_minus,
            )
            for competitor in self.margins_plus
        }


@dataclass(frozen=True)
class ProbePrefixEstimate:
    requested_k: int
    k_eff: int
    primary_available: bool
    primary: Optional[LinkRadiusEstimate]
    incomplete_sensitivity: Optional[LinkRadiusEstimate]
    missing_direction_ids: tuple[int, ...]
    rejected_direction_ids: tuple[int, ...]
    version: str = "linkradius_probe_prefix_v1"


def estimate_probe_prefix(
    clean_margins: Mapping[str, float],
    pairs: Sequence[ProbePairObservation],
    q: int,
    requested_k: int,
) -> ProbePrefixEstimate:
    """Estimate a nested 0..K-1 prefix without relabeling incomplete results."""

    if isinstance(requested_k, bool) or int(requested_k) != requested_k or int(requested_k) < 1:
        raise ValueError("requested_k must be a positive integer.")
    for competitor, margin_value in clean_margins.items():
        margin = float(margin_value)
        if not math.isfinite(margin) or margin <= 0:
            raise ValueError(
                f"Clean margin for competitor {competitor!r} must be finite and positive; got {margin}."
            )
    by_id: dict[int, ProbePairObservation] = {}
    for pair in pairs:
        direction_id = int(pair.direction_id)
        if direction_id in by_id:
            raise ValueError(f"Duplicate probe direction_id={direction_id}.")
        by_id[direction_id] = pair
    wanted = tuple(range(int(requested_k)))
    missing = tuple(direction_id for direction_id in wanted if direction_id not in by_id)
    rejected = tuple(
        direction_id
        for direction_id in wanted
        if direction_id in by_id and not by_id[direction_id].accepted
    )
    accepted = [by_id[direction_id] for direction_id in wanted if direction_id in by_id and by_id[direction_id].accepted]
    derivatives: dict[str, list[float]] = {competitor: [] for competitor in clean_margins}
    for pair in accepted:
        values = pair.derivatives()
        if set(values) != set(clean_margins):
            raise ValueError("Probe competitors do not match clean margins.")
        for competitor, value in values.items():
            derivatives[competitor].append(value)
    incomplete = (
        estimate_competitor_radii(clean_margins, derivatives, q) if accepted else None
    )
    primary_available = not missing and not rejected
    return ProbePrefixEstimate(
        requested_k=int(requested_k),
        k_eff=len(accepted),
        primary_available=primary_available,
        primary=incomplete if primary_available else None,
        incomplete_sensitivity=incomplete,
        missing_direction_ids=missing,
        rejected_direction_ids=rejected,
    )


__all__ = [
    "CHOICE_LABELS",
    "DELTA_DIAGNOSTICS_VERSION",
    "DIRECTION_VERSION",
    "DONOR_MAPPING_VERSION",
    "EDGE_SITES",
    "ESTIMATOR_VERSION",
    "Edge",
    "GPQARawRecord",
    "GPQA_PARTITIONS",
    "GPQA_RAW_ID_VERSION",
    "GPQA_SPLIT_VERSION",
    "LINKRADIUS_UTILS_VERSION",
    "MOMENT_NOISE_VERSION",
    "PerturbationSubspace",
    "ProbeAcceptanceThresholds",
    "ProbePairObservation",
    "ReplayStep",
    "STRICT_CHOICE_VERSION",
    "StrictChoiceResult",
    "additive_intervention",
    "build_gpqa_raw_record",
    "canonical_gpqa_identity_object",
    "cast_receiver_tensor",
    "central_difference",
    "check_strict_choice",
    "deterministic_donor_assignments",
    "deterministic_donor_mapping",
    "estimate_competitor_radii",
    "estimate_linkradius",
    "estimate_probe_prefix",
    "estimate_susceptibility",
    "forced_choice_prediction",
    "get_subspace",
    "gpqa_option_permutation",
    "identity_intervention",
    "largest_remainder_counts",
    "mismatch_intervention",
    "moment_noise_intervention",
    "normalize_gpqa_identity_text",
    "parse_edge",
    "parse_strict_choice",
    "postcast_budget_fitted_delta",
    "probe_pair_diagnostics",
    "project_subspace_coefficients",
    "raw_split_assignment",
    "realized_delta_diagnostics",
    "render_gpqa_options",
    "replay_operations",
    "replay_schedule",
    "requested_additive_delta",
    "sample_stable_lifted_direction",
    "sample_stable_unit_direction",
    "split_raw_ids",
    "stable_gpqa_raw_id",
    "stable_intervention_seed",
    "strict_choice_parse",
    "susceptibility_squared",
    "valid_edges",
    "validate_edge",
    "validate_unique_raw_ids",
    "zero_intervention",
]
