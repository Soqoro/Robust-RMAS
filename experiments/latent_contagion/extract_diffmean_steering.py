#!/usr/bin/env python3
"""Extract DiffMean attack-associated steering directions from latent traces."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch


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
    clean_trace: Mapping[str, Any],
    attack_trace: Mapping[str, Any],
    clean_trace_path: Path,
    attack_trace_path: Path,
    site: str,
    round_idx: int,
    paired_ids: Sequence[str],
    clean_trace_index: Mapping[str, int],
    attack_trace_index: Mapping[str, int],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    clean_tensor = lookup_latent(clean_trace, site, round_idx, clean_trace_path)
    attack_tensor = lookup_latent(attack_trace, site, round_idx, attack_trace_path)

    site_ids: List[str] = []
    clean_indices: List[int] = []
    attack_indices: List[int] = []
    for sample_id in paired_ids:
        clean_idx = clean_trace_index[sample_id]
        attack_idx = attack_trace_index[sample_id]
        if clean_idx >= clean_tensor.size(0) or attack_idx >= attack_tensor.size(0):
            continue
        site_ids.append(sample_id)
        clean_indices.append(clean_idx)
        attack_indices.append(attack_idx)

    if not site_ids:
        raise ValueError(f"No valid paired trace rows for site={site!r} round={round_idx}.")

    clean_latent = select_rows(clean_tensor, clean_indices)
    attack_latent = select_rows(attack_tensor, attack_indices)
    if tuple(clean_latent.shape) != tuple(attack_latent.shape):
        raise ValueError(
            f"Clean/attack latent shapes differ for site={site!r} round={round_idx}: "
            f"{tuple(clean_latent.shape)} vs {tuple(attack_latent.shape)}"
        )

    delta = attack_latent - clean_latent
    diffmean = delta.mean(dim=0)
    diffmean_norm_before = torch.linalg.vector_norm(diffmean.float())
    diffmean = diffmean.float() / diffmean_norm_before.clamp_min(1e-12)
    per_sample_delta_norm = torch.linalg.vector_norm(delta.reshape(delta.size(0), -1), dim=1)

    stats = {
        "n_pairs": int(delta.size(0)),
        "mean_delta_norm": float(per_sample_delta_norm.mean().item()) if per_sample_delta_norm.numel() else 0.0,
        "diffmean_norm_before_normalization": float(diffmean_norm_before.item()),
    }
    return diffmean.cpu(), stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract DiffMean steering directions from clean and directly attacked latent traces."
    )
    parser.add_argument("--clean_jsonl", required=True)
    parser.add_argument("--attack_jsonl", required=True)
    parser.add_argument("--clean_trace", required=True)
    parser.add_argument("--attack_trace", required=True)
    parser.add_argument("--out_bank", required=True)
    parser.add_argument("--sites", default="p2c,c2s,s2p")
    parser.add_argument("--rounds", default="0")
    parser.add_argument("--filter", default="all_valid_pairs", choices=["all_valid_pairs"])
    parser.add_argument("--calibration_R", type=int, default=2)
    parser.add_argument("--steering_id", default="diffmean_R2_math500_role_aligned")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    paired_ids = paired_ids_from_sources(
        clean_json=clean_records,
        attack_json=attack_records,
        clean_trace_ids=clean_trace_ids,
        clean_trace_index=clean_trace_index,
        attack_trace_index=attack_trace_index,
    )
    if not paired_ids:
        raise ValueError("No valid sample_id pairs found across JSONL and trace files.")

    n_clean_correct = 0
    n_attack_wrong = 0
    n_clean_correct_attack_wrong = 0
    for sample_id in paired_ids:
        clean_is_correct = correctness(clean_records[sample_id])
        attack_is_wrong = not correctness(attack_records[sample_id])
        n_clean_correct += int(clean_is_correct)
        n_attack_wrong += int(attack_is_wrong)
        n_clean_correct_attack_wrong += int(clean_is_correct and attack_is_wrong)

    directions: Dict[str, Dict[int, torch.Tensor]] = {site: {} for site in sites}
    stats: Dict[str, Dict[int, Dict[str, Any]]] = {site: {} for site in sites}
    for site in sites:
        for round_idx in rounds:
            direction, site_stats = build_site_direction(
                clean_trace=clean_trace,
                attack_trace=attack_trace,
                clean_trace_path=clean_trace_path,
                attack_trace_path=attack_trace_path,
                site=site,
                round_idx=round_idx,
                paired_ids=paired_ids,
                clean_trace_index=clean_trace_index,
                attack_trace_index=attack_trace_index,
            )
            directions[site][int(round_idx)] = direction
            stats[site][int(round_idx)] = site_stats

    bank = {
        "metadata": {
            "source": "attack-associated-diffmean",
            "dataset": "math500",
            "calibration_R": int(args.calibration_R),
            "sites": sites,
            "rounds": rounds,
            "filter": args.filter,
            "steering_id": args.steering_id,
            "n_pairs": len(paired_ids),
            "n_total_pairs": len(paired_ids),
            "n_clean_correct": n_clean_correct,
            "n_attack_wrong": n_attack_wrong,
            "n_clean_correct_attack_wrong": n_clean_correct_attack_wrong,
        },
        "directions": {
            "diffmean": directions,
        },
        "stats": stats,
    }

    out_dir = os.path.dirname(str(out_bank_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(bank, out_bank_path)
    print(
        f"[diffmean] wrote {out_bank_path} "
        f"pairs={len(paired_ids)} sites={','.join(sites)} rounds={','.join(str(r) for r in rounds)}"
    )


if __name__ == "__main__":
    main()
