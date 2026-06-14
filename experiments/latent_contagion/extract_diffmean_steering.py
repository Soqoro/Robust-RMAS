#!/usr/bin/env python3
"""Extract DiffMean attack-associated steering directions from latent traces."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch


EXAMPLE_USAGE = """\
Example:
  python experiments/latent_contagion/extract_diffmean_steering.py \\
    --clean_jsonl outputs/latent_contagion/diffmean_calibration/math500_R2/clean_R2.jsonl \\
    --attack_jsonl outputs/latent_contagion/diffmean_calibration/math500_R2/attack_R2.jsonl \\
    --clean_trace outputs/latent_contagion/diffmean_calibration/math500_R2/clean_R2_trace.pt \\
    --attack_trace outputs/latent_contagion/diffmean_calibration/math500_R2/attack_R2_trace.pt \\
    --out_bank outputs/latent_contagion/diffmean_calibration/math500_R2/diffmean_R2_math500_role_aligned_target_hit.pt \\
    --sites p2c,c2s,s2p \\
    --rounds 0 \\
    --filter target_hit \\
    --target_answer 999999999 \\
    --calibration_R 2 \\
    --steering_id diffmean_R2_math500_role_aligned_target_hit
"""


def parse_csv_items(text: str) -> List[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def parse_sites(text: str) -> List[str]:
    sites = parse_csv_items(text)
    allowed = {"p2c", "c2s", "s2p"}
    unknown = [site for site in sites if site not in allowed]
    if unknown:
        raise ValueError(f"Unsupported --sites value(s): {unknown}. Allowed sites: {sorted(allowed)}")
    return sites


def parse_rounds(text: str) -> List[int]:
    rounds: List[int] = []
    for item in parse_csv_items(text):
        try:
            round_idx = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid --rounds item {item!r}; expected integers.") from exc
        if round_idx < 0:
            raise ValueError("--rounds must contain non-negative integers.")
        rounds.append(round_idx)
    return rounds


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, int):
        return value != 0
    if isinstance(value, float):
        return math.isfinite(value) and value != 0.0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "f", "no", "n", "off", "", "none", "null", "nan"}:
            return False
    return bool(value)


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def _strip_matching_outer(text: str, left: str, right: str) -> Optional[str]:
    text = text.strip()
    if not (text.startswith(left) and text.endswith(right)):
        return None
    return text[len(left) : len(text) - len(right)]


def _strip_outer_braces(text: str) -> Optional[str]:
    text = text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    depth = 0
    for idx, char in enumerate(text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and idx != len(text) - 1:
                return None
            if depth < 0:
                return None
    if depth != 0:
        return None
    return text[1:-1]


def _strip_macro_wrapper(text: str, macro: str) -> Optional[str]:
    text = text.strip()
    prefix = f"\\{macro}"
    if not text.startswith(prefix):
        return None
    rest = text[len(prefix) :].lstrip()
    if not rest.startswith("{"):
        return None

    depth = 0
    for idx, char in enumerate(rest):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                trailing = rest[idx + 1 :].strip()
                if trailing:
                    return None
                return rest[1:idx]
            if depth < 0:
                return None
    return None


def _unescape_known_latex_markers(text: str) -> str:
    for marker in ("boxed", "fbox", "left", "right"):
        text = text.replace(f"\\\\{marker}", f"\\{marker}")
    for marker in ("(", ")", "[", "]"):
        text = text.replace(f"\\\\{marker}", f"\\{marker}")
    return text


def strip_boxed(text: Any) -> str:
    out = "" if text is None else _unescape_known_latex_markers(str(text).strip())
    while True:
        changed = False
        for macro in ("boxed", "fbox"):
            inner = _strip_macro_wrapper(out, macro)
            if inner is not None:
                out = inner.strip()
                changed = True
                break
        if not changed:
            return out


def normalize_answer_for_target(text: Any) -> str:
    out = "" if text is None else _unescape_known_latex_markers(str(text).strip())
    for _ in range(16):
        previous = out
        out = out.strip()
        out = _unescape_known_latex_markers(out)
        out = out.replace("\\left", "").replace("\\right", "")
        out = strip_boxed(out)

        for left, right in (("\\(", "\\)"), ("\\[", "\\]"), ("$$", "$$"), ("$", "$")):
            inner = _strip_matching_outer(out, left, right)
            if inner is not None:
                out = inner.strip()
                break

        inner_braced = _strip_outer_braces(out)
        if inner_braced is not None:
            out = inner_braced.strip()

        if out == previous:
            break

    out = out.replace("\\left", "").replace("\\right", "")
    out = strip_boxed(out)
    out = out.replace(",", "")
    out = re.sub(r"\s+", "", out)
    return out


def is_target_answer(record: Mapping[str, Any], target_answer: str) -> bool:
    normalized_target = normalize_answer_for_target(target_answer)
    for field_name in ("pred_answer_parsed", "final_answer"):
        if field_name not in record:
            continue
        if normalize_answer_for_target(record.get(field_name)) == normalized_target:
            return True
    return False


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            if str(record.get("type", "")).lower() == "summary":
                continue
            sample_id = record.get("sample_id")
            if sample_id is None or str(sample_id).strip() == "":
                continue
            record["__line_number"] = line_number
            records.append(record)
    return records


def records_by_sample_id(records: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    out: Dict[str, Mapping[str, Any]] = {}
    for record in records:
        sample_id = str(record.get("sample_id"))
        if sample_id not in out:
            out[sample_id] = record
    return out


def torch_load_cpu(path: Path) -> Dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Expected trace file to load as dict: {path}")
    return obj


def index_by_sample_id(trace: Mapping[str, Any], path: Path) -> Tuple[List[str], Dict[str, int]]:
    sample_ids_raw = trace.get("sample_ids")
    if not isinstance(sample_ids_raw, list):
        raise ValueError(f"Trace file is missing list sample_ids: {path}")
    sample_ids = [str(sample_id) for sample_id in sample_ids_raw]
    index: Dict[str, int] = {}
    for idx, sample_id in enumerate(sample_ids):
        if sample_id not in index:
            index[sample_id] = idx
    return sample_ids, index


def lookup_latent(trace: Mapping[str, Any], site: str, round_idx: int, path: Path) -> torch.Tensor:
    latents = trace.get("latents")
    if not isinstance(latents, dict):
        raise ValueError(f"Trace file is missing latents dict: {path}")
    if site not in latents:
        available = ", ".join(str(key) for key in sorted(latents, key=str))
        raise ValueError(f"Trace file {path} has no site={site!r}. Available sites: [{available}]")
    rounds = latents[site]
    if not isinstance(rounds, dict):
        raise ValueError(f"Trace file {path} site={site!r} is not a round dict.")
    if round_idx in rounds:
        tensor = rounds[round_idx]
    elif str(round_idx) in rounds:
        tensor = rounds[str(round_idx)]
    else:
        available = ", ".join(str(key) for key in sorted(rounds, key=str))
        raise ValueError(
            f"Trace file {path} site={site!r} has no round={round_idx}. "
            f"Available rounds: [{available}]"
        )
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Trace entry {path} site={site!r} round={round_idx} is not a tensor.")
    return tensor


def find_latent(trace: Mapping[str, Any], site: str, round_idx: int) -> Optional[torch.Tensor]:
    latents = trace.get("latents")
    if not isinstance(latents, dict):
        return None
    rounds = latents.get(site)
    if not isinstance(rounds, dict):
        return None
    if round_idx in rounds:
        tensor = rounds[round_idx]
    elif str(round_idx) in rounds:
        tensor = rounds[str(round_idx)]
    else:
        return None
    return tensor if isinstance(tensor, torch.Tensor) else None


def collect_trace_tensors(
    clean_trace: Mapping[str, Any],
    attack_trace: Mapping[str, Any],
    clean_trace_path: Path,
    attack_trace_path: Path,
    sites: Sequence[str],
    rounds: Sequence[int],
) -> Tuple[Dict[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor]], List[Dict[str, Any]]]:
    tensors: Dict[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor]] = {}
    skipped: List[Dict[str, Any]] = []
    for site in sites:
        for round_idx in rounds:
            clean_tensor = find_latent(clean_trace, site, int(round_idx))
            attack_tensor = find_latent(attack_trace, site, int(round_idx))
            if clean_tensor is None and attack_tensor is None:
                skipped.append(
                    {
                        "site": site,
                        "round_idx": int(round_idx),
                        "reason": "missing_in_both_traces",
                    }
                )
                continue
            if clean_tensor is None or attack_tensor is None:
                missing_path = clean_trace_path if clean_tensor is None else attack_trace_path
                raise ValueError(
                    f"Trace pair is incomplete for site={site!r} round={int(round_idx)}; "
                    f"missing in {missing_path}"
                )
            tensors[(site, int(round_idx))] = (clean_tensor, attack_tensor)
    return tensors, skipped


def select_rows(tensor: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    if not indices:
        return tensor[:0].float()
    index_tensor = torch.tensor([int(idx) for idx in indices], dtype=torch.long)
    return tensor.index_select(0, index_tensor).float()


def paired_ids_from_sources(
    clean_json: Mapping[str, Mapping[str, Any]],
    attack_json: Mapping[str, Mapping[str, Any]],
    clean_trace_ids: Sequence[str],
    clean_trace_index: Mapping[str, int],
    attack_trace_index: Mapping[str, int],
) -> List[str]:
    allowed = (
        set(clean_json)
        & set(attack_json)
        & set(clean_trace_index)
        & set(attack_trace_index)
    )
    paired: List[str] = []
    seen = set()
    for sample_id in clean_trace_ids:
        if sample_id in allowed and sample_id not in seen:
            paired.append(sample_id)
            seen.add(sample_id)
    return paired


def correctness(record: Mapping[str, Any]) -> bool:
    return parse_bool(first_nonempty(record.get("is_correct"), record.get("correct")))


def build_site_direction(
    trace_tensors: Mapping[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor]],
    site: str,
    round_idx: int,
    clean_indices: Sequence[int],
    attack_indices: Sequence[int],
    selected_filter: str,
    target_answer: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    clean_tensor, attack_tensor = trace_tensors[(site, int(round_idx))]

    if not clean_indices:
        raise ValueError(f"No valid paired trace rows for site={site!r} round={round_idx}.")

    clean_latent = select_rows(clean_tensor, clean_indices)
    attack_latent = select_rows(attack_tensor, attack_indices)
    if tuple(clean_latent.shape) != tuple(attack_latent.shape):
        raise ValueError(
            f"Clean/attack latent shapes differ for site={site!r} round={round_idx}: "
            f"{tuple(clean_latent.shape)} vs {tuple(attack_latent.shape)}"
        )

    delta = attack_latent - clean_latent
    diffmean_raw = delta.mean(dim=0).float()
    diffmean_norm_before = torch.linalg.vector_norm(diffmean_raw)
    diffmean = diffmean_raw / diffmean_norm_before.clamp_min(1e-12)
    per_sample_delta_norm = torch.linalg.vector_norm(delta.reshape(delta.size(0), -1), dim=1)

    stats = {
        "n_pairs": int(delta.size(0)),
        "mean_delta_norm": float(per_sample_delta_norm.mean().item()) if per_sample_delta_norm.numel() else 0.0,
        "median_delta_norm": float(per_sample_delta_norm.median().item()) if per_sample_delta_norm.numel() else 0.0,
        "diffmean_norm_before_normalization": float(diffmean_norm_before.item()),
        "selected_filter": selected_filter,
        "target_answer": str(target_answer),
    }
    return diffmean.cpu(), stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract DiffMean steering directions from clean and directly attacked latent traces.",
        epilog=EXAMPLE_USAGE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--clean_jsonl", required=True)
    parser.add_argument("--attack_jsonl", required=True)
    parser.add_argument("--clean_trace", required=True)
    parser.add_argument("--attack_trace", required=True)
    parser.add_argument("--out_bank", required=True)
    parser.add_argument("--sites", default="p2c,c2s,s2p")
    parser.add_argument("--rounds", default="0")
    parser.add_argument(
        "--filter",
        default="all_valid_pairs",
        choices=["all_valid_pairs", "target_hit", "clean_correct_target_hit"],
    )
    parser.add_argument("--target_answer", default="999999999")
    parser.add_argument("--min_pairs", type=int, default=1)
    parser.add_argument("--calibration_R", type=int, default=2)
    parser.add_argument("--steering_id", default="diffmean_R2_math500_role_aligned")
    return parser.parse_args()


def has_valid_trace_pair(
    clean_idx: int,
    attack_idx: int,
    trace_tensors: Mapping[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor]],
) -> bool:
    for clean_tensor, attack_tensor in trace_tensors.values():
        if clean_idx >= clean_tensor.size(0) or attack_idx >= attack_tensor.size(0):
            return False
    return True


def main() -> None:
    args = parse_args()
    if args.min_pairs < 1:
        raise ValueError("--min_pairs must be at least 1.")
    sites = parse_sites(args.sites)
    rounds = parse_rounds(args.rounds)

    clean_jsonl_path = Path(args.clean_jsonl)
    attack_jsonl_path = Path(args.attack_jsonl)
    clean_trace_path = Path(args.clean_trace)
    attack_trace_path = Path(args.attack_trace)
    out_bank_path = Path(args.out_bank)

    clean_records = records_by_sample_id(load_jsonl_records(clean_jsonl_path))
    attack_records = records_by_sample_id(load_jsonl_records(attack_jsonl_path))
    clean_trace = torch_load_cpu(clean_trace_path)
    attack_trace = torch_load_cpu(attack_trace_path)
    clean_trace_ids, clean_trace_index = index_by_sample_id(clean_trace, clean_trace_path)
    _, attack_trace_index = index_by_sample_id(attack_trace, attack_trace_path)
    trace_tensors, skipped_trace_entries = collect_trace_tensors(
        clean_trace=clean_trace,
        attack_trace=attack_trace,
        clean_trace_path=clean_trace_path,
        attack_trace_path=attack_trace_path,
        sites=sites,
        rounds=rounds,
    )
    if not trace_tensors:
        raise ValueError("No requested site/round trace tensors were found.")

    paired_ids = paired_ids_from_sources(
        clean_json=clean_records,
        attack_json=attack_records,
        clean_trace_ids=clean_trace_ids,
        clean_trace_index=clean_trace_index,
        attack_trace_index=attack_trace_index,
    )
    if not paired_ids:
        raise ValueError("No valid sample_id pairs found across JSONL and trace files.")

    valid_sample_ids: List[str] = []
    selected_sample_ids: List[str] = []
    selected_clean_indices: List[int] = []
    selected_attack_indices: List[int] = []

    n_clean_correct = 0
    n_attack_wrong_by_existing_judge = 0
    n_clean_correct_attack_wrong_by_existing_judge = 0
    n_raw_contains_target = 0
    n_strict_target_hit = 0
    n_clean_correct_target_hit = 0

    for sample_id in paired_ids:
        clean_idx = clean_trace_index[sample_id]
        attack_idx = attack_trace_index[sample_id]
        if not has_valid_trace_pair(clean_idx, attack_idx, trace_tensors):
            continue

        valid_sample_ids.append(sample_id)

        clean_record = clean_records[sample_id]
        attack_record = attack_records[sample_id]
        clean_is_correct = correctness(clean_record)
        attack_is_wrong = not correctness(attack_record)
        target_hit = is_target_answer(attack_record, str(args.target_answer))
        clean_correct_target_hit = clean_is_correct and target_hit
        raw_contains_target = str(args.target_answer) in str(attack_record.get("raw_final_output", ""))

        n_clean_correct += int(clean_is_correct)
        n_attack_wrong_by_existing_judge += int(attack_is_wrong)
        n_clean_correct_attack_wrong_by_existing_judge += int(clean_is_correct and attack_is_wrong)
        n_raw_contains_target += int(raw_contains_target)
        n_strict_target_hit += int(target_hit)
        n_clean_correct_target_hit += int(clean_correct_target_hit)

        include = False
        if args.filter == "all_valid_pairs":
            include = True
        elif args.filter == "target_hit":
            include = target_hit
        elif args.filter == "clean_correct_target_hit":
            include = clean_correct_target_hit
        else:
            raise ValueError(f"Unsupported --filter: {args.filter}")

        if include:
            selected_sample_ids.append(sample_id)
            selected_clean_indices.append(clean_idx)
            selected_attack_indices.append(attack_idx)

    diagnostics = {
        "n_total_matched": len(paired_ids),
        "n_valid_trace_pairs": len(valid_sample_ids),
        "n_clean_correct": n_clean_correct,
        "n_attack_wrong_by_existing_judge": n_attack_wrong_by_existing_judge,
        "n_clean_correct_attack_wrong_by_existing_judge": n_clean_correct_attack_wrong_by_existing_judge,
        "n_raw_contains_target": n_raw_contains_target,
        "n_strict_target_hit": n_strict_target_hit,
        "n_clean_correct_target_hit": n_clean_correct_target_hit,
        "n_selected_pairs": len(selected_sample_ids),
    }

    if len(selected_sample_ids) < int(args.min_pairs):
        raise ValueError(
            f"Only {len(selected_sample_ids)} pairs selected for filter {args.filter}; "
            f"need at least min_pairs={args.min_pairs}."
        )

    directions: Dict[str, Dict[int, torch.Tensor]] = {site: {} for site in sites}
    stats: Dict[str, Dict[int, Dict[str, Any]]] = {site: {} for site in sites}
    for site, round_idx in sorted(trace_tensors, key=lambda item: (item[0], item[1])):
        direction, site_stats = build_site_direction(
            trace_tensors=trace_tensors,
            site=site,
            round_idx=round_idx,
            clean_indices=selected_clean_indices,
            attack_indices=selected_attack_indices,
            selected_filter=args.filter,
            target_answer=str(args.target_answer),
        )
        directions[site][int(round_idx)] = direction
        stats[site][int(round_idx)] = site_stats

    available_rounds_by_site = {
        site: sorted(int(round_idx) for round_idx in directions.get(site, {}))
        for site in sites
    }

    metadata = {
        "source": "attack-associated-diffmean",
        "dataset": "math500",
        "calibration_R": int(args.calibration_R),
        "sites": sites,
        "rounds": rounds,
        "available_rounds_by_site": available_rounds_by_site,
        "skipped_trace_entries": skipped_trace_entries,
        "filter": args.filter,
        "target_answer": str(args.target_answer),
        "steering_id": args.steering_id,
        **diagnostics,
        "selected_sample_ids": selected_sample_ids,
        "selected_sample_indices": selected_clean_indices,
        "selected_clean_trace_indices": selected_clean_indices,
        "selected_attack_trace_indices": selected_attack_indices,
        # Legacy aliases retained for older analysis scripts.
        "n_pairs": len(selected_sample_ids),
        "n_total_pairs": len(paired_ids),
        "n_attack_wrong": n_attack_wrong_by_existing_judge,
        "n_clean_correct_attack_wrong": n_clean_correct_attack_wrong_by_existing_judge,
    }

    bank = {
        "metadata": metadata,
        "directions": {
            "diffmean": directions,
        },
        "stats": stats,
    }

    out_dir = os.path.dirname(str(out_bank_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print("[diffmean] diagnostics " + " ".join(f"{key}={value}" for key, value in diagnostics.items()))
    torch.save(bank, out_bank_path)
    print(
        f"[diffmean] wrote {out_bank_path} "
        f"pairs={len(selected_sample_ids)} filter={args.filter} "
        f"sites={','.join(sites)} rounds={','.join(str(r) for r in rounds)} "
        f"skipped={len(skipped_trace_entries)}"
    )


if __name__ == "__main__":
    main()
