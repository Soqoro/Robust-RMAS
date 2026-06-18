#!/usr/bin/env python3
"""Estimate Experiment D role response profiles from clean/probe traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


ROLES = ["planner", "critic", "solver"]
ROLE_INDEX = {role: idx for idx, role in enumerate(ROLES)}
SUMMARY_COLUMNS = [
    "dataset",
    "R",
    "seed",
    "quantity_type",
    "role",
    "sender_role",
    "receiver_role",
    "site",
    "round_idx",
    "epsilon",
    "count",
    "mean",
    "std",
    "median",
    "q75",
    "q90",
    "q95",
]
ROW_COLUMNS = [
    "dataset",
    "R",
    "seed",
    "sample_id",
    "sample_index",
    "quantity_type",
    "role",
    "sender_role",
    "receiver_role",
    "site",
    "round_idx",
    "epsilon",
    "delta_norm",
    "response_norm",
    "ratio",
    "clean_correct",
    "response_kind",
    "alpha_proxy",
]
LAMBDA_COLUMNS = [
    "dataset",
    "R",
    "seed",
    "site",
    "probe_target",
    "round_idx",
    "epsilon",
    "gain_quantile",
    "H",
    "tau",
    "Lambda",
    "bound_min_1_lambda2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean_jsonl", required=True)
    parser.add_argument("--clean_trace", required=True)
    parser.add_argument("--probe_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--rounds", default="")
    parser.add_argument("--epsilons", default="")
    parser.add_argument("--quantiles", default="0.5,0.75,0.9,0.95")
    parser.add_argument("--clean_correct_only", type=int, default=1, choices=[0, 1])
    parser.add_argument("--lambda_grid_from_experiment_c", default="")
    parser.add_argument(
        "--tau_proxy",
        default="clean_clean_floor",
        choices=["none", "constant", "terminal_drift_quantile", "clean_clean_floor"],
    )
    parser.add_argument("--lambda_stabilizer", type=float, default=1e-8)
    return parser.parse_args()


def parse_csv_items(text: str) -> List[str]:
    return [item.strip() for item in str(text or "").replace(" ", ",").split(",") if item.strip()]


def parse_float_list(text: str, default: Sequence[float]) -> List[float]:
    items = parse_csv_items(text)
    if not items:
        return [float(x) for x in default]
    return [float(item) for item in items]


def parse_int_list(text: str) -> List[int]:
    return [int(item) for item in parse_csv_items(text)]


def to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "f", "no", "n", "off"}:
            return False
        if text in {"", "none", "null", "nan"}:
            return None
    return None


def finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def torch_load_cpu(path: Path) -> Dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Expected torch file to load as dict: {path}")
    return obj


def load_clean_jsonl(path: Path) -> Tuple[Dict[str, bool], bool, List[str]]:
    correctness: Dict[str, bool] = {}
    warnings: List[str] = []
    saw_correctness_field = False
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"Skipping malformed JSONL line {line_number}: {exc}")
                continue
            if not isinstance(record, dict) or str(record.get("type", "")).lower() == "summary":
                continue
            sample_id = record.get("sample_id")
            if sample_id is None:
                continue
            correct_value = None
            for field in ("exact_match", "correct", "is_correct"):
                if field in record:
                    correct_value = to_bool(record.get(field))
                    saw_correctness_field = True
                    break
            if correct_value is not None:
                correctness[str(sample_id)] = bool(correct_value)
    if not saw_correctness_field:
        warnings.append(
            "No exact_match/correct/is_correct field found in clean_jsonl; clean-correct filtering keeps all samples."
        )
    return correctness, saw_correctness_field, warnings


def sample_ids(trace: Mapping[str, Any], path: Path) -> List[str]:
    raw = trace.get("sample_ids")
    if not isinstance(raw, list):
        raise ValueError(f"Trace file is missing list sample_ids: {path}")
    return [str(x) for x in raw]


def sample_indices(trace: Mapping[str, Any], count: int) -> List[int]:
    raw = trace.get("sample_indices")
    if isinstance(raw, list) and len(raw) == count:
        return [int(x) for x in raw]
    return list(range(count))


def per_sample_norm(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 0:
        raise ValueError("Expected a batched tensor.")
    return torch.linalg.vector_norm(x.detach().float().reshape(x.size(0), -1), dim=1)


def canonical_site(site: str) -> str:
    if site == "refiner_self":
        return "critic_self"
    return str(site)


def state_site_to_role(site: str) -> str:
    site = canonical_site(site)
    if site == "planner_self":
        return "planner"
    if site == "critic_self":
        return "critic"
    if site == "solver_self":
        return "solver"
    raise ValueError(f"Unsupported state site: {site}")


def message_site_roles(site: str) -> Tuple[str, str]:
    if site == "p2c":
        return "planner", "critic"
    if site == "c2s":
        return "critic", "solver"
    if site == "s2p":
        return "solver", "planner"
    raise ValueError(f"Unsupported message site: {site}")


def round_lookup(round_map: Any, round_idx: int) -> Optional[Any]:
    if not isinstance(round_map, dict):
        return None
    if round_idx in round_map:
        return round_map[round_idx]
    key = str(round_idx)
    if key in round_map:
        return round_map[key]
    return None


def lookup_round_tensor(trace: Mapping[str, Any], section: str, key: str, round_idx: int) -> Optional[torch.Tensor]:
    container = trace.get(section)
    if not isinstance(container, dict):
        return None
    value = container.get(key)
    round_value = round_lookup(value, round_idx)
    if isinstance(round_value, torch.Tensor):
        return round_value.detach().float()
    return None


def lookup_terminal_tensor(trace: Mapping[str, Any], key: str) -> Optional[torch.Tensor]:
    terminal = trace.get("terminal")
    if not isinstance(terminal, dict):
        return None
    value = terminal.get(key)
    if isinstance(value, torch.Tensor):
        return value.detach().float()
    return None


def select_rows(x: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    if not indices:
        return x[:0]
    index_tensor = torch.tensor([int(i) for i in indices], dtype=torch.long)
    return x.index_select(0, index_tensor)


def get_probe_delta_norm(probe_trace: Mapping[str, Any], site: str, round_idx: int) -> Optional[torch.Tensor]:
    probe_deltas = probe_trace.get("probe_deltas")
    if not isinstance(probe_deltas, dict):
        return None
    site_entry = probe_deltas.get(canonical_site(site))
    round_entry = round_lookup(site_entry, round_idx)
    if isinstance(round_entry, torch.Tensor):
        return round_entry.detach().float()
    if isinstance(round_entry, dict):
        for key in ("delta_norm", "actual_delta_norm", "norm"):
            value = round_entry.get(key)
            if isinstance(value, torch.Tensor):
                return value.detach().float()
    return None


def compute_input_delta_norm(
    clean_trace: Mapping[str, Any],
    probe_trace: Mapping[str, Any],
    target: str,
    site: str,
    round_idx: int,
) -> Optional[torch.Tensor]:
    site = canonical_site(site)
    if target == "message":
        clean_x = lookup_round_tensor(clean_trace, "messages", site, round_idx)
        probe_x = lookup_round_tensor(probe_trace, "messages", site, round_idx)
    elif target == "state":
        role = state_site_to_role(site)
        clean_x = lookup_round_tensor(clean_trace, "states", role, round_idx)
        probe_x = lookup_round_tensor(probe_trace, "states", role, round_idx)
    elif target == "terminal" and site == "final_c2s":
        clean_x = lookup_terminal_tensor(clean_trace, "final_c2s")
        probe_x = lookup_terminal_tensor(probe_trace, "final_c2s")
    else:
        return None
    if clean_x is None or probe_x is None or tuple(clean_x.shape) != tuple(probe_x.shape):
        return None
    return per_sample_norm(probe_x - clean_x)


def quantile(values: Sequence[float], q: float) -> float:
    finite = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not finite:
        return float("nan")
    if len(finite) == 1:
        return finite[0]
    pos = float(q) * (len(finite) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return finite[lo]
    frac = pos - lo
    return finite[lo] * (1.0 - frac) + finite[hi] * frac


def discover_probe_traces(root: Path, warnings: List[str]) -> List[Dict[str, Any]]:
    probes: List[Dict[str, Any]] = []
    if not root.exists():
        warnings.append(f"Probe root does not exist: {root}")
        return probes
    for path in sorted(root.rglob("*.pt")):
        try:
            trace = torch_load_cpu(path)
        except Exception as exc:
            warnings.append(f"Skipping unreadable probe trace {path}: {exc}")
            continue
        metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
        probe_meta = metadata.get("probe") if isinstance(metadata.get("probe"), dict) else {}
        mode = str(probe_meta.get("probe_mode", probe_meta.get("mode", "none")))
        target = str(probe_meta.get("probe_target", "none"))
        site = canonical_site(str(probe_meta.get("probe_site", "none")))
        try:
            round_idx = int(probe_meta.get("probe_round", -1))
        except (TypeError, ValueError):
            round_idx = -1
        epsilon = finite_or_none(probe_meta.get("epsilon"))
        seed = probe_meta.get("seed", metadata.get("seed"))
        if mode != "one_shot" or target == "none" or site == "none" or round_idx < 0 or epsilon is None:
            continue
        probes.append(
            {
                "path": path,
                "trace": trace,
                "metadata": metadata,
                "target": target,
                "site": site,
                "round_idx": round_idx,
                "epsilon": float(epsilon),
                "seed": seed,
            }
        )
    if not probes:
        warnings.append(f"No role-profile one_shot probe traces found under {root}")
    return probes


def alignment_for_probe(
    clean_trace: Mapping[str, Any],
    clean_path: Path,
    probe_trace: Mapping[str, Any],
    probe_path: Path,
    correctness: Mapping[str, bool],
    correctness_available: bool,
    clean_correct_only: bool,
    warnings: List[str],
) -> Tuple[List[str], List[int], List[int], List[bool], List[int]]:
    clean_ids = sample_ids(clean_trace, clean_path)
    probe_ids = sample_ids(probe_trace, probe_path)
    clean_lookup = {sample_id: idx for idx, sample_id in enumerate(clean_ids)}
    probe_lookup = {sample_id: idx for idx, sample_id in enumerate(probe_ids)}
    clean_sample_indices = sample_indices(clean_trace, len(clean_ids))
    out_ids: List[str] = []
    clean_indices: List[int] = []
    probe_indices: List[int] = []
    correct_flags: List[bool] = []
    original_sample_indices: List[int] = []
    for sample_id in clean_ids:
        if sample_id not in probe_lookup:
            continue
        is_correct = bool(correctness.get(sample_id, True))
        if clean_correct_only and correctness_available and not is_correct:
            continue
        clean_i = clean_lookup[sample_id]
        out_ids.append(sample_id)
        clean_indices.append(clean_i)
        probe_indices.append(probe_lookup[sample_id])
        correct_flags.append(is_correct)
        original_sample_indices.append(clean_sample_indices[clean_i])
    if not out_ids:
        warnings.append(f"No aligned samples after filtering for probe trace: {probe_path}")
    return out_ids, clean_indices, probe_indices, correct_flags, original_sample_indices


def add_ratio_rows(
    rows: List[Dict[str, Any]],
    warnings: List[str],
    *,
    dataset: str,
    R: int,
    seed: Any,
    clean_trace: Mapping[str, Any],
    probe_info: Mapping[str, Any],
    clean_path: Path,
    correctness: Mapping[str, bool],
    correctness_available: bool,
    clean_correct_only: bool,
    quantity_type: str,
    role: str,
    sender_role: Optional[str],
    receiver_role: Optional[str],
    site: str,
    round_idx: int,
    response_kind: str,
    clean_response: Optional[torch.Tensor],
    probe_response: Optional[torch.Tensor],
    eps_floor: float,
    alpha_proxy: bool = False,
) -> int:
    probe_trace = probe_info["trace"]
    probe_path = probe_info["path"]
    if clean_response is None or probe_response is None:
        warnings.append(
            f"Skipping {quantity_type} site={site} round={round_idx}: missing {response_kind} tensor."
        )
        return 0
    sample_id_list, clean_indices, probe_indices, correct_flags, original_indices = alignment_for_probe(
        clean_trace=clean_trace,
        clean_path=clean_path,
        probe_trace=probe_trace,
        probe_path=probe_path,
        correctness=correctness,
        correctness_available=correctness_available,
        clean_correct_only=clean_correct_only,
        warnings=warnings,
    )
    if not sample_id_list:
        return 0
    saved_delta = get_probe_delta_norm(probe_trace, site, round_idx)
    if saved_delta is None:
        saved_delta = compute_input_delta_norm(
            clean_trace,
            probe_trace,
            target=str(probe_info["target"]),
            site=site,
            round_idx=round_idx,
        )
        if saved_delta is None:
            warnings.append(
                f"Skipping {quantity_type} site={site} round={round_idx}: missing saved delta norms."
            )
            return 0
        warnings.append(
            f"Using recomputed input deltas for {quantity_type} site={site} round={round_idx}; "
            "saved actual delta norms were absent."
        )
    try:
        clean_resp_aligned = select_rows(clean_response, clean_indices)
        probe_resp_aligned = select_rows(probe_response, probe_indices)
        delta_aligned = select_rows(saved_delta.reshape(saved_delta.size(0), -1), probe_indices).reshape(-1)
    except Exception as exc:
        warnings.append(f"Skipping {quantity_type} site={site} round={round_idx}: alignment failed: {exc}")
        return 0
    if tuple(clean_resp_aligned.shape) != tuple(probe_resp_aligned.shape):
        warnings.append(
            f"Skipping {quantity_type} site={site} round={round_idx}: response shape mismatch "
            f"{tuple(clean_resp_aligned.shape)} vs {tuple(probe_resp_aligned.shape)}."
        )
        return 0
    response_norm = per_sample_norm(probe_resp_aligned - clean_resp_aligned)
    denom = torch.clamp(delta_aligned.detach().float(), min=float(eps_floor))
    ratios = response_norm / denom
    for local_idx, sample_id in enumerate(sample_id_list):
        rows.append(
            {
                "dataset": dataset,
                "R": int(R),
                "seed": seed,
                "sample_id": sample_id,
                "sample_index": original_indices[local_idx],
                "quantity_type": quantity_type,
                "role": role,
                "sender_role": sender_role,
                "receiver_role": receiver_role,
                "site": canonical_site(site),
                "round_idx": int(round_idx),
                "epsilon": float(probe_info["epsilon"]),
                "delta_norm": float(delta_aligned[local_idx].item()),
                "response_norm": float(response_norm[local_idx].item()),
                "ratio": float(ratios[local_idx].item()),
                "clean_correct": bool(correct_flags[local_idx]),
                "response_kind": response_kind,
                "alpha_proxy": bool(alpha_proxy),
            }
        )
    return len(sample_id_list)


def estimate_rows(
    clean_trace: Mapping[str, Any],
    clean_trace_path: Path,
    probes: Sequence[Mapping[str, Any]],
    correctness: Mapping[str, bool],
    correctness_available: bool,
    args: argparse.Namespace,
    warnings: List[str],
) -> List[Dict[str, Any]]:
    metadata = clean_trace.get("metadata") if isinstance(clean_trace.get("metadata"), dict) else {}
    R = int(metadata.get("R") or (parse_int_list(args.rounds)[0] if args.rounds else 1))
    seed = metadata.get("seed", "")
    rows: List[Dict[str, Any]] = []
    eps_floor = max(float(args.lambda_stabilizer), 1e-12)
    for probe in probes:
        target = str(probe["target"])
        site = canonical_site(str(probe["site"]))
        round_idx = int(probe["round_idx"])
        if target == "message" and site in {"p2c", "c2s", "s2p"}:
            sender, receiver = message_site_roles(site)
            if site == "p2c":
                clean_resp = lookup_round_tensor(clean_trace, "states", "critic", round_idx)
                probe_resp = lookup_round_tensor(probe["trace"], "states", "critic", round_idx)
                add_ratio_rows(
                    rows,
                    warnings,
                    dataset=args.dataset,
                    R=R,
                    seed=seed,
                    clean_trace=clean_trace,
                    probe_info=probe,
                    clean_path=clean_trace_path,
                    correctness=correctness,
                    correctness_available=correctness_available,
                    clean_correct_only=bool(args.clean_correct_only),
                    quantity_type="beta",
                    role="critic",
                    sender_role=sender,
                    receiver_role=receiver,
                    site=site,
                    round_idx=round_idx,
                    response_kind="critic_state",
                    clean_response=clean_resp,
                    probe_response=probe_resp,
                    eps_floor=eps_floor,
                )
            elif site == "c2s":
                if round_idx < R - 1:
                    clean_resp = lookup_round_tensor(clean_trace, "states", "solver", round_idx)
                    probe_resp = lookup_round_tensor(probe["trace"], "states", "solver", round_idx)
                    response_kind = "solver_feedback_state"
                else:
                    clean_resp = lookup_terminal_tensor(clean_trace, "solver_prefill_last_hidden")
                    probe_resp = lookup_terminal_tensor(probe["trace"], "solver_prefill_last_hidden")
                    response_kind = "terminal_solver_state"
                add_ratio_rows(
                    rows,
                    warnings,
                    dataset=args.dataset,
                    R=R,
                    seed=seed,
                    clean_trace=clean_trace,
                    probe_info=probe,
                    clean_path=clean_trace_path,
                    correctness=correctness,
                    correctness_available=correctness_available,
                    clean_correct_only=bool(args.clean_correct_only),
                    quantity_type="beta",
                    role="solver",
                    sender_role=sender,
                    receiver_role=receiver,
                    site=site,
                    round_idx=round_idx,
                    response_kind=response_kind,
                    clean_response=clean_resp,
                    probe_response=probe_resp,
                    eps_floor=eps_floor,
                )
            elif site == "s2p":
                if round_idx >= R - 1:
                    warnings.append(f"Skipping invalid s2p probe at round {round_idx} for R={R}.")
                    continue
                clean_resp = lookup_round_tensor(clean_trace, "states", "planner", round_idx + 1)
                probe_resp = lookup_round_tensor(probe["trace"], "states", "planner", round_idx + 1)
                add_ratio_rows(
                    rows,
                    warnings,
                    dataset=args.dataset,
                    R=R,
                    seed=seed,
                    clean_trace=clean_trace,
                    probe_info=probe,
                    clean_path=clean_trace_path,
                    correctness=correctness,
                    correctness_available=correctness_available,
                    clean_correct_only=bool(args.clean_correct_only),
                    quantity_type="beta",
                    role="planner",
                    sender_role=sender,
                    receiver_role=receiver,
                    site=site,
                    round_idx=round_idx,
                    response_kind="planner_next_state",
                    clean_response=clean_resp,
                    probe_response=probe_resp,
                    eps_floor=eps_floor,
                )
        if target == "state" and site in {"planner_self", "critic_self", "solver_self"}:
            role = state_site_to_role(site)
            if role in {"planner", "critic"}:
                clean_resp = lookup_round_tensor(clean_trace, "states", role, round_idx + 1)
                probe_resp = lookup_round_tensor(probe["trace"], "states", role, round_idx + 1)
                response_kind = f"{role}_next_state"
            else:
                clean_resp = lookup_round_tensor(clean_trace, "states", "solver", round_idx + 1)
                probe_resp = lookup_round_tensor(probe["trace"], "states", "solver", round_idx + 1)
                response_kind = "solver_next_state"
                if clean_resp is None or probe_resp is None:
                    clean_resp = lookup_terminal_tensor(clean_trace, "solver_prefill_last_hidden")
                    probe_resp = lookup_terminal_tensor(probe["trace"], "solver_prefill_last_hidden")
                    response_kind = "terminal_solver_state"
            add_ratio_rows(
                rows,
                warnings,
                dataset=args.dataset,
                R=R,
                seed=seed,
                clean_trace=clean_trace,
                probe_info=probe,
                clean_path=clean_trace_path,
                correctness=correctness,
                correctness_available=correctness_available,
                clean_correct_only=bool(args.clean_correct_only),
                quantity_type="alpha_proxy",
                role=role,
                sender_role=role,
                receiver_role=role,
                site=site,
                round_idx=round_idx,
                response_kind=response_kind,
                clean_response=clean_resp,
                probe_response=probe_resp,
                eps_floor=eps_floor,
                alpha_proxy=True,
            )
        if target == "terminal" and site == "final_c2s":
            clean_resp = lookup_terminal_tensor(clean_trace, "solver_prefill_last_hidden")
            probe_resp = lookup_terminal_tensor(probe["trace"], "solver_prefill_last_hidden")
            add_ratio_rows(
                rows,
                warnings,
                dataset=args.dataset,
                R=R,
                seed=seed,
                clean_trace=clean_trace,
                probe_info=probe,
                clean_path=clean_trace_path,
                correctness=correctness,
                correctness_available=correctness_available,
                clean_correct_only=bool(args.clean_correct_only),
                quantity_type="q",
                role="solver",
                sender_role="critic",
                receiver_role="solver",
                site=site,
                round_idx=round_idx,
                response_kind="solver_prefill_last_hidden",
                clean_response=clean_resp,
                probe_response=probe_resp,
                eps_floor=eps_floor,
            )

        if target in {"message", "state"}:
            clean_terminal = lookup_terminal_tensor(clean_trace, "solver_prefill_last_hidden")
            probe_terminal = lookup_terminal_tensor(probe["trace"], "solver_prefill_last_hidden")
            if clean_terminal is not None and probe_terminal is not None:
                if target == "message" and site in {"p2c", "c2s", "s2p"}:
                    sender, receiver = message_site_roles(site)
                    role = sender
                elif target == "state" and site in {"planner_self", "critic_self", "solver_self"}:
                    role = state_site_to_role(site)
                    sender = role
                    receiver = role
                else:
                    continue
                add_ratio_rows(
                    rows,
                    warnings,
                    dataset=args.dataset,
                    R=R,
                    seed=seed,
                    clean_trace=clean_trace,
                    probe_info=probe,
                    clean_path=clean_trace_path,
                    correctness=correctness,
                    correctness_available=correctness_available,
                    clean_correct_only=bool(args.clean_correct_only),
                    quantity_type="q_path",
                    role=role,
                    sender_role=sender,
                    receiver_role=receiver,
                    site=site,
                    round_idx=round_idx,
                    response_kind="solver_prefill_last_hidden_path",
                    clean_response=clean_terminal,
                    probe_response=probe_terminal,
                    eps_floor=eps_floor,
                )
    if not any(row["quantity_type"] == "alpha_proxy" for row in rows):
        warnings.append("No alpha_proxy rows were estimated; state-probe traces may be absent.")
    return rows


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    group_columns = SUMMARY_COLUMNS[:10]
    grouped: Dict[Tuple[Any, ...], List[float]] = defaultdict(list)
    for row in rows:
        ratio = finite_or_none(row.get("ratio"))
        if ratio is None:
            continue
        key = tuple(row.get(col) for col in group_columns)
        grouped[key].append(ratio)
    summaries: List[Dict[str, Any]] = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        if not values:
            continue
        record = {col: key[idx] for idx, col in enumerate(group_columns)}
        record["count"] = len(values)
        record["mean"] = statistics.fmean(values)
        record["std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        record["median"] = quantile(values, 0.5)
        record["q75"] = quantile(values, 0.75)
        record["q90"] = quantile(values, 0.90)
        record["q95"] = quantile(values, 0.95)
        summaries.append(record)
    return summaries


def select_gain(
    rows: Sequence[Mapping[str, Any]],
    *,
    quantity_type: str,
    role: Optional[str] = None,
    sender_role: Optional[str] = None,
    receiver_role: Optional[str] = None,
    site: Optional[str] = None,
    round_idx: Optional[int] = None,
    epsilon: Optional[float] = None,
    gain_quantile: float = 0.5,
) -> Optional[float]:
    values: List[float] = []
    fallback_values: List[float] = []
    for row in rows:
        if row.get("quantity_type") != quantity_type:
            continue
        if role is not None and row.get("role") != role:
            continue
        if sender_role is not None and row.get("sender_role") != sender_role:
            continue
        if receiver_role is not None and row.get("receiver_role") != receiver_role:
            continue
        if site is not None and row.get("site") != site:
            continue
        if round_idx is not None and int(row.get("round_idx", -999999)) != int(round_idx):
            continue
        ratio = finite_or_none(row.get("ratio"))
        if ratio is None:
            continue
        fallback_values.append(ratio)
        if epsilon is None or math.isclose(float(row.get("epsilon")), float(epsilon), rel_tol=1e-9, abs_tol=1e-12):
            values.append(ratio)
    source = values if values else fallback_values
    if not source:
        return None
    return quantile(source, gain_quantile)


def matrix_for_round(rows: Sequence[Mapping[str, Any]], round_idx: int, epsilon: float, q: float) -> List[List[float]]:
    alpha_planner = select_gain(rows, quantity_type="alpha_proxy", role="planner", round_idx=round_idx, epsilon=epsilon, gain_quantile=q) or 0.0
    alpha_critic = select_gain(rows, quantity_type="alpha_proxy", role="critic", round_idx=round_idx, epsilon=epsilon, gain_quantile=q) or 0.0
    alpha_solver = select_gain(rows, quantity_type="alpha_proxy", role="solver", round_idx=round_idx, epsilon=epsilon, gain_quantile=q) or 0.0
    beta_cp = select_gain(
        rows,
        quantity_type="beta",
        sender_role="planner",
        receiver_role="critic",
        site="p2c",
        round_idx=round_idx,
        epsilon=epsilon,
        gain_quantile=q,
    ) or 0.0
    beta_sc = select_gain(
        rows,
        quantity_type="beta",
        sender_role="critic",
        receiver_role="solver",
        site="c2s",
        round_idx=round_idx,
        epsilon=epsilon,
        gain_quantile=q,
    ) or 0.0
    beta_ps = 0.0
    if round_idx > 0:
        beta_ps = select_gain(
            rows,
            quantity_type="beta",
            sender_role="solver",
            receiver_role="planner",
            site="s2p",
            round_idx=round_idx - 1,
            epsilon=epsilon,
            gain_quantile=q,
        ) or 0.0
    return [
        [alpha_planner, 0.0, beta_ps],
        [beta_cp, alpha_critic, 0.0],
        [0.0, beta_sc, alpha_solver],
    ]


def mat_vec(M: Sequence[Sequence[float]], v: Sequence[float]) -> List[float]:
    return [sum(float(M[i][j]) * float(v[j]) for j in range(len(v))) for i in range(len(M))]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def injection_basis(site: str, target: str, round_idx: int) -> Optional[Tuple[List[float], int]]:
    site = canonical_site(site)
    v = [0.0, 0.0, 0.0]
    if target == "message":
        if site == "p2c":
            v[ROLE_INDEX["critic"]] = 1.0
            return v, round_idx
        if site == "c2s":
            v[ROLE_INDEX["solver"]] = 1.0
            return v, round_idx + 1
        if site == "s2p":
            v[ROLE_INDEX["planner"]] = 1.0
            return v, round_idx + 1
    if target == "state":
        role = state_site_to_role(site)
        v[ROLE_INDEX[role]] = 1.0
        if role == "solver":
            return v, round_idx + 1
        return v, round_idx
    if target == "terminal" and site == "final_c2s":
        v[ROLE_INDEX["solver"]] = 1.0
        return v, 10**9
    return None


def tau_value(args: argparse.Namespace, warnings: List[str]) -> Tuple[float, Dict[str, Any]]:
    if args.tau_proxy == "none":
        return float("nan"), {"kind": "none", "value": None}
    if args.tau_proxy == "constant":
        return 1.0, {"kind": "constant", "value": 1.0}
    warnings.append(f"tau_proxy={args.tau_proxy} has no clean-clean rerun traces in this CLI; using constant fallback 1.0.")
    return 1.0, {"kind": "constant_fallback", "requested": args.tau_proxy, "value": 1.0}


def lambda_predictions(
    rows: Sequence[Mapping[str, Any]],
    probes: Sequence[Mapping[str, Any]],
    R: int,
    dataset: str,
    seed: Any,
    quantiles: Sequence[float],
    tau: float,
) -> List[Dict[str, Any]]:
    measured = sorted(
        {
            (canonical_site(str(p["site"])), str(p["target"]), int(p["round_idx"]), float(p["epsilon"]))
            for p in probes
        },
        key=lambda item: (item[1], item[0], item[2], item[3]),
    )
    out: List[Dict[str, Any]] = []
    for site, target, round_idx, epsilon in measured:
        basis = injection_basis(site, target, round_idx)
        if basis is None:
            continue
        for q in quantiles:
            q_solver = select_gain(rows, quantity_type="q", role="solver", epsilon=epsilon, gain_quantile=q) or 0.0
            q_vec = [0.0, 0.0, q_solver]
            v, start_round = basis
            if target == "terminal" and site == "final_c2s":
                H = q_solver
            else:
                for matrix_round in range(int(start_round), int(R)):
                    M = matrix_for_round(rows, matrix_round, epsilon, q)
                    v = mat_vec(M, v)
                H = dot(q_vec, v)
            if math.isfinite(tau) and tau > 0:
                lambda_value = float(epsilon) * float(H) / float(tau)
                bound = min(1.0, lambda_value * lambda_value)
            else:
                lambda_value = float("nan")
                bound = float("nan")
            out.append(
                {
                    "dataset": dataset,
                    "R": int(R),
                    "seed": seed,
                    "site": site,
                    "probe_target": target,
                    "round_idx": int(round_idx),
                    "epsilon": float(epsilon),
                    "gain_quantile": float(q),
                    "H": float(H),
                    "tau": tau,
                    "Lambda": lambda_value,
                    "bound_min_1_lambda2": bound,
                }
            )
    return out


def build_profile(
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    clean_trace: Mapping[str, Any],
    probes: Sequence[Mapping[str, Any]],
    warnings: Sequence[str],
    tau_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    metadata = clean_trace.get("metadata") if isinstance(clean_trace.get("metadata"), dict) else {}
    R = int(metadata.get("R") or (parse_int_list(args.rounds)[0] if args.rounds else 1))
    stabilizer = float(args.lambda_stabilizer)
    alpha: Dict[str, Any] = {role: {} for role in ROLES}
    beta: Dict[str, Any] = {}
    q_profile: Dict[str, Any] = {
        "planner_direct": 0.0,
        "critic_direct": 0.0,
        "solver": {},
        "note": "planner and critic do not directly feed the final decoder in this implementation.",
    }
    for summary in summaries:
        quantity = summary.get("quantity_type")
        site = summary.get("site")
        round_key = str(summary.get("round_idx"))
        eps_key = str(summary.get("epsilon"))
        stats = {
            "count": summary.get("count"),
            "median": summary.get("median"),
            "q75": summary.get("q75"),
            "q90": summary.get("q90"),
            "q95": summary.get("q95"),
            "mean": summary.get("mean"),
        }
        if quantity == "alpha_proxy":
            alpha.setdefault(str(summary.get("role")), {}).setdefault(round_key, {})[eps_key] = stats
        elif quantity == "beta":
            key = f"{summary.get('receiver_role')}<-{summary.get('sender_role')}"
            beta.setdefault(key, {}).setdefault(round_key, {})[eps_key] = stats
        elif quantity == "q" and site == "final_c2s":
            q_profile["solver"].setdefault(round_key, {})[eps_key] = stats

    tau = finite_or_none(tau_meta.get("value"))
    kappa: Dict[str, Optional[float]] = {}
    pi_out = {"planner": [("critic", "p2c")], "critic": [("solver", "c2s")], "solver": [("planner", "s2p")]}
    for role in ROLES:
        alpha_gain = select_gain(rows, quantity_type="alpha_proxy", role=role, gain_quantile=0.5)
        q_gain = 0.0 if role in {"planner", "critic"} else select_gain(rows, quantity_type="q", role="solver", gain_quantile=0.5)
        outgoing: List[float] = []
        for receiver, site in pi_out[role]:
            beta_gain = select_gain(
                rows,
                quantity_type="beta",
                sender_role=role,
                receiver_role=receiver,
                site=site,
                gain_quantile=0.5,
            )
            if beta_gain is not None:
                outgoing.append(beta_gain)
        if alpha_gain is None or q_gain is None or tau is None or tau <= 0 or not outgoing:
            kappa[role] = None
            continue
        outgoing_term = sum(outgoing) / len(outgoing)
        numerator = (float(q_gain) + stabilizer) * (float(alpha_gain) + outgoing_term + stabilizer)
        denominator = float(tau) + stabilizer
        kappa[role] = math.log(max(numerator, stabilizer) / max(denominator, stabilizer))

    return {
        "metadata": {
            "experiment": "d",
            "schema_version": 1,
            "dataset": args.dataset,
            "R": R,
            "seed": metadata.get("seed"),
            "clean_trace_metadata": metadata,
            "probe_count": len(probes),
            "lambda_indexing_convention": (
                "p2c injects e_critic at round r and propagates through M[r..R-1]; "
                "c2s injects e_solver at round r and propagates through M[r+1..R-1]; "
                "s2p injects e_planner at round r+1 and propagates through M[r+1..R-1]."
            ),
            "warnings": list(warnings),
        },
        "alpha": alpha,
        "alpha_proxy": True,
        "beta": beta,
        "q": q_profile,
        "tau_proxy": dict(tau_meta),
        "kappa": kappa,
    }


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [json_sanitize(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def main() -> int:
    args = parse_args()
    clean_trace_path = Path(args.clean_trace)
    clean_jsonl_path = Path(args.clean_jsonl)
    probe_root = Path(args.probe_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    if args.lambda_grid_from_experiment_c:
        warnings.append(
            "--lambda_grid_from_experiment_c is recorded for provenance but not consumed by this estimator."
        )

    clean_trace = torch_load_cpu(clean_trace_path)
    correctness, correctness_available, correctness_warnings = load_clean_jsonl(clean_jsonl_path)
    warnings.extend(correctness_warnings)
    probes = discover_probe_traces(probe_root, warnings)
    metadata = clean_trace.get("metadata") if isinstance(clean_trace.get("metadata"), dict) else {}
    R = int(metadata.get("R") or (parse_int_list(args.rounds)[0] if args.rounds else 1))
    seed = metadata.get("seed", "")
    quantiles = parse_float_list(args.quantiles, default=[0.5, 0.75, 0.9, 0.95])

    rows = estimate_rows(
        clean_trace=clean_trace,
        clean_trace_path=clean_trace_path,
        probes=probes,
        correctness=correctness,
        correctness_available=correctness_available,
        args=args,
        warnings=warnings,
    )
    summaries = summarize_rows(rows)
    tau, tau_meta = tau_value(args, warnings)
    lambda_rows = lambda_predictions(
        rows=rows,
        probes=probes,
        R=R,
        dataset=args.dataset,
        seed=seed,
        quantiles=quantiles,
        tau=tau,
    )
    profile = build_profile(
        rows=rows,
        summaries=summaries,
        args=args,
        clean_trace=clean_trace,
        probes=probes,
        warnings=warnings,
        tau_meta=tau_meta,
    )

    write_csv(out_dir / "role_profile_rows.csv", rows, ROW_COLUMNS)
    write_csv(out_dir / "role_profile_summary.csv", summaries, SUMMARY_COLUMNS)
    write_csv(out_dir / "lambda_predictions.csv", lambda_rows, LAMBDA_COLUMNS)
    with (out_dir / "role_profile.json").open("w", encoding="utf-8") as handle:
        json.dump(json_sanitize(profile), handle, indent=2, sort_keys=True)
        handle.write("\n")
    manifest = {
        "clean_jsonl": str(clean_jsonl_path),
        "clean_trace": str(clean_trace_path),
        "probe_root": str(probe_root),
        "out_dir": str(out_dir),
        "dataset": args.dataset,
        "R": R,
        "seed": seed,
        "probe_count": len(probes),
        "row_count": len(rows),
        "summary_count": len(summaries),
        "lambda_row_count": len(lambda_rows),
        "warnings": warnings,
        "probe_paths": [str(probe["path"]) for probe in probes],
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(json_sanitize(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"[role-profile-estimator] rows={len(rows)} summaries={len(summaries)} lambda_rows={len(lambda_rows)}")
    print(f"[role-profile-estimator] wrote outputs to {out_dir}")
    if warnings:
        print(f"[role-profile-estimator] warnings={len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
