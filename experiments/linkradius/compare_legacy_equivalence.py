#!/usr/bin/env python3
"""Authenticate a real release-run trace against one frozen LinkRadius capture."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io_utils import atomic_write_json, content_hash, file_sha256, load_jsonl, source_hash
from .schemas import ContractError


EQUIVALENCE_VERSION = "linkradius.legacy_equivalence.v1"
EXPECTED_R2_EDGES = ("p2c@0", "c2s@0", "s2p@0", "p2c@1", "c2s@1")


def _torch_load(path: str | Path) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in the full environment.
        raise ContractError("legacy equivalence requires PyTorch") from exc
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Older PyTorch.
        return torch.load(path, map_location="cpu")


def _check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), **detail}


def _legacy_rows(path: str | Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    rows = load_jsonl(path)
    samples = [row for row in rows if row.get("type") != "summary"]
    summaries = [row for row in rows if row.get("type") == "summary"]
    if len(samples) != 1 or len(summaries) != 1:
        raise ContractError(
            "legacy equivalence JSONL must contain exactly one sample and one summary"
        )
    return samples[0], summaries[0]


def _trace_tensor(trace: Mapping[str, Any], edge_id: str) -> Any:
    site, round_text = edge_id.split("@", 1)
    by_site = trace.get("latents", {}).get(site)
    if not isinstance(by_site, Mapping):
        raise ContractError(f"legacy trace is missing site {site!r}")
    round_idx = int(round_text)
    if round_idx in by_site:
        return by_site[round_idx]
    if str(round_idx) in by_site:
        return by_site[str(round_idx)]
    raise ContractError(f"legacy trace is missing {edge_id}")


def _tensor_comparison(
    current: Any,
    legacy: Any,
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, dict[str, Any]]:
    import torch

    current_tensor = current.detach().float().cpu()
    legacy_tensor = legacy.detach().float().cpu()
    same_shape = tuple(current_tensor.shape) == tuple(legacy_tensor.shape)
    finite = bool(
        torch.isfinite(current_tensor).all() and torch.isfinite(legacy_tensor).all()
    )
    if same_shape and current_tensor.numel():
        difference = (current_tensor - legacy_tensor).abs()
        max_abs = float(difference.max().item())
        scale = float(torch.maximum(current_tensor.abs(), legacy_tensor.abs()).max().item())
        limit = float(atol) + float(rtol) * scale
        close = finite and max_abs <= limit
    elif same_shape:
        max_abs, limit, close = 0.0, float(atol), finite
    else:
        max_abs, limit, close = math.inf, float(atol), False
    return close, {
        "current_shape": list(current_tensor.shape),
        "legacy_shape": list(legacy_tensor.shape),
        "finite": finite,
        "max_abs_error": max_abs,
        "tolerance": limit,
    }


def _result_settings(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    generation = summary.get("generation_config")
    if not isinstance(generation, Mapping):
        raise ContractError("legacy summary is missing generation_config provenance")
    return generation


def build_equivalence_report(
    *,
    trajectory_path: str | Path,
    legacy_trace_path: str | Path,
    legacy_results_path: str | Path,
    repo_root: str | Path,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Recompute every equivalence check; no trusted boolean input is accepted."""

    if atol < 0 or rtol < 0 or not math.isfinite(atol) or not math.isfinite(rtol):
        raise ContractError("legacy equivalence tolerances must be finite and non-negative")
    trajectory_file = Path(trajectory_path).resolve()
    trace_file = Path(legacy_trace_path).resolve()
    results_file = Path(legacy_results_path).resolve()
    trajectory = _torch_load(trajectory_file)
    trace = _torch_load(trace_file)
    if not isinstance(trace, Mapping):
        raise ContractError("legacy trace must be a mapping")
    sample, summary = _legacy_rows(results_file)
    if len(getattr(trajectory, "sample_ids", ())) != 1:
        raise ContractError("legacy equivalence requires one frozen LinkRadius row")

    from RecursiveMAS.inference_utils import inference_mas
    from RecursiveMAS.inference_utils import linkradius as lr

    raw_index = int(trajectory.raw_indices[0])
    runtime_config = trajectory.provenance.get("runtime_config", {})
    generation_config = _result_settings(summary)
    checks: list[dict[str, Any]] = []

    trace_metadata = trace.get("metadata", {})
    expected_trace_metadata = {
        "dataset": "gpqa",
        "style": str(runtime_config.get("style", "sequential_light")),
        "method": "ours_recursive",
        "R": int(getattr(trajectory, "rounds", -1)),
        "seed": int(runtime_config.get("seed", 42)),
        "num_samples": 1,
        "trace_dtype": "float32",
    }
    checks.append(
        _check(
            "legacy_trace_settings",
            isinstance(trace_metadata, Mapping)
            and all(trace_metadata.get(key) == value for key, value in expected_trace_metadata.items())
            and set(trace_metadata.get("trace_sites", ())) == {"p2c", "c2s", "s2p"}
            and set(int(value) for value in trace_metadata.get("trace_rounds", ())) == {0, 1},
            expected=expected_trace_metadata,
            observed=dict(trace_metadata) if isinstance(trace_metadata, Mapping) else trace_metadata,
        )
    )
    checks.append(
        _check(
            "legacy_selected_raw_index",
            list(trace.get("sample_indices", ())) == [raw_index]
            and str(sample.get("sample_id", "")).endswith(f":{raw_index}"),
            raw_index=raw_index,
            trace_sample_indices=list(trace.get("sample_indices", ())),
            legacy_sample_id=sample.get("sample_id"),
        )
    )

    experiment = generation_config.get("experiment", {})
    dataset_config = generation_config.get("dataset", {})
    generation = generation_config.get("generation", {})
    expected_generation = {
        "seed": int(runtime_config.get("seed", 42)),
        "num_rollouts": 1,
        "batch_size": 1,
        "max_new_tokens": int(runtime_config.get("max_new_tokens", 4000)),
        "do_sample": bool(runtime_config.get("do_sample", False)),
        "ans": bool(runtime_config.get("answer_retry", True)),
        "enable_thinking": bool(runtime_config.get("enable_thinking", False)),
    }
    provenance_hash_valid = (
        summary.get("generation_config_sha256")
        == inference_mas.stable_json_sha256(generation_config)
    )
    checks.append(
        _check(
            "legacy_release_configuration",
            provenance_hash_valid
            and experiment.get("style") == runtime_config.get("style")
            and experiment.get("method") == "ours_recursive"
            and experiment.get("num_recursive_rounds") == trajectory.rounds
            and experiment.get("latent_steps") == runtime_config.get("latent_steps")
            and experiment.get("choice_old_prompt") == runtime_config.get("choice_old_prompt")
            and experiment.get("solver_pre_question") == runtime_config.get("solver_pre_question")
            and experiment.get("planner_feedback_round_label_mode") == runtime_config.get("round_label_mode")
            and dataset_config.get("split") == "train"
            and dataset_config.get("gpqa_option_shuffle") is True
            and all(generation.get(key) == value for key, value in expected_generation.items()),
            provenance_hash_valid=provenance_hash_valid,
            expected_generation=expected_generation,
            experiment=dict(experiment) if isinstance(experiment, Mapping) else experiment,
            dataset=dict(dataset_config) if isinstance(dataset_config, Mapping) else dataset_config,
            generation=dict(generation) if isinstance(generation, Mapping) else generation,
        )
    )

    checks.append(
        _check(
            "legacy_question_and_gold",
            sample.get("question") == trajectory.questions[0]
            and sample.get("ground_truth") == trajectory.gold_labels[0],
            question_equal=sample.get("question") == trajectory.questions[0],
            legacy_gold=sample.get("ground_truth"),
            linkradius_gold=trajectory.gold_labels[0],
        )
    )

    generation_audit = trajectory.clean_generation_audit[0]
    legacy_text = str(sample.get("raw_final_output", ""))
    current_text = str(generation_audit.get("final_text", ""))
    legacy_choice = lr.parse_strict_choice(legacy_text)
    current_choice = lr.parse_strict_choice(current_text)
    checks.append(
        _check(
            "legacy_generation_exact",
            legacy_text == current_text,
            legacy_text_sha256=inference_mas.stable_json_sha256(legacy_text),
            linkradius_text_sha256=inference_mas.stable_json_sha256(current_text),
        )
    )
    checks.append(
        _check(
            "legacy_strict_choice",
            legacy_choice.choice == current_choice.choice
            and legacy_choice.answer_invalid == current_choice.answer_invalid
            and legacy_choice.answer_conflict == current_choice.answer_conflict,
            legacy=legacy_choice.to_dict(),
            linkradius=current_choice.to_dict(),
        )
    )

    expected_edges = set(EXPECTED_R2_EDGES)
    actual_trace_edges: set[str] = set()
    for site, values in trace.get("latents", {}).items():
        if isinstance(values, Mapping):
            actual_trace_edges.update(f"{site}@{int(round_idx)}" for round_idx in values)
    checks.append(
        _check(
            "legacy_trace_edge_set",
            actual_trace_edges == expected_edges,
            expected=sorted(expected_edges),
            observed=sorted(actual_trace_edges),
        )
    )
    for edge_id in EXPECTED_R2_EDGES:
        legacy_tensor = _trace_tensor(trace, edge_id)
        transport_ok, transport_detail = _tensor_comparison(
            trajectory.message(edge_id, receiver=False),
            legacy_tensor,
            atol=atol,
            rtol=rtol,
        )
        receiver_from_legacy = lr.cast_receiver_tensor(
            legacy_tensor,
            trajectory.dtype_metadata(edge_id).consumer_dtype,
        )
        receiver_ok, receiver_detail = _tensor_comparison(
            trajectory.message(edge_id, receiver=True),
            receiver_from_legacy,
            atol=atol,
            rtol=rtol,
        )
        checks.append(
            _check(
                f"legacy_relay:{edge_id}",
                transport_ok and receiver_ok,
                transport=transport_detail,
                receiver=receiver_detail,
            )
        )

    report: dict[str, Any] = {
        "schema_version": EQUIVALENCE_VERSION,
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "raw_sample_id": trajectory.raw_sample_ids[0],
        "raw_index": raw_index,
        "sample_id": trajectory.sample_ids[0],
        "atol": float(atol),
        "rtol": float(rtol),
        "trajectory_path": str(trajectory_file),
        "trajectory_sha256": file_sha256(trajectory_file),
        "legacy_trace_path": str(trace_file),
        "legacy_trace_sha256": file_sha256(trace_file),
        "legacy_results_path": str(results_file),
        "legacy_results_sha256": file_sha256(results_file),
        "source_hash": source_hash(repo_root),
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }
    report["report_content_hash"] = content_hash(
        report, domain="linkradius:legacy_equivalence_report:v1"
    )
    return report


def verify_equivalence_report(
    report: Mapping[str, Any],
    *,
    repo_root: str | Path,
    expected_trajectory_path: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild a report from its hashed artifacts and require exact agreement."""

    if report.get("schema_version") != EQUIVALENCE_VERSION:
        raise ContractError("unsupported legacy-equivalence report schema")
    payload = dict(report)
    supplied_hash = payload.pop("report_content_hash", None)
    if supplied_hash != content_hash(payload, domain="linkradius:legacy_equivalence_report:v1"):
        raise ContractError("legacy-equivalence report hash is missing or stale")
    if expected_trajectory_path is not None:
        expected_path = str(Path(expected_trajectory_path).resolve())
        if report.get("trajectory_path") != expected_path:
            raise ContractError("legacy-equivalence report references a different trajectory")
    rebuilt = build_equivalence_report(
        trajectory_path=report["trajectory_path"],
        legacy_trace_path=report["legacy_trace_path"],
        legacy_results_path=report["legacy_results_path"],
        repo_root=repo_root,
        atol=float(report["atol"]),
        rtol=float(report["rtol"]),
        created_at=str(report["created_at"]),
    )
    if rebuilt != dict(report):
        raise ContractError("legacy-equivalence report does not match recomputed artifacts")
    if rebuilt["passed"] is not True:
        raise ContractError("legacy-versus-LinkRadius equivalence did not pass")
    return rebuilt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--legacy-trace", required=True)
    parser.add_argument("--legacy-results", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--output", required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = build_equivalence_report(
        trajectory_path=args.trajectory,
        legacy_trace_path=args.legacy_trace,
        legacy_results_path=args.legacy_results,
        repo_root=args.repo_root,
        atol=args.atol,
        rtol=args.rtol,
    )
    atomic_write_json(args.output, report, overwrite=args.overwrite)
    print(json.dumps({"path": str(Path(args.output).resolve()), "passed": report["passed"]}, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
