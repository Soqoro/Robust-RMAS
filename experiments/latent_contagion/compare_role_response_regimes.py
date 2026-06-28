#!/usr/bin/env python3
"""Compare Experiment E role-response regimes across attack and profile outputs."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


EXPECTED_ORDER = ["amplifying", "neutral", "corrective"]
HANDOFF_SITES = ["p2c", "c2s", "s2p"]
PROFILE_COLUMNS = [
    "dataset",
    "role_response_regime",
    "R",
    "seed",
    "profile_epsilon",
    "gain_quantile",
    "mean_lambda_handoff",
    "p2c_lambda",
    "c2s_lambda",
    "s2p_lambda",
    "final_c2s_lambda",
    "mean_beta_handoff",
    "beta_critic_from_planner",
    "beta_solver_from_critic",
    "beta_planner_from_solver",
    "q_solver_direct",
    "n_lambda_sites",
    "n_beta_sites",
]
ATTACK_COLUMNS = [
    "dataset",
    "role_response_regime",
    "R",
    "site",
    "attack_eps_mode",
    "excess_asrcc",
    "raw_asrcc",
    "clean_floor",
    "clean_accuracy",
    "perturbed_accuracy",
    "max_excess_asrcc",
    "epsilon50",
    "n_conditions",
]
ORDERING_COLUMNS = [
    "dataset",
    "R",
    "site",
    "lambda_order",
    "asr_order",
    "lambda_rank_match_expected",
    "asr_rank_match_expected",
    "expected_order",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack_aggregate_dir", required=True)
    parser.add_argument("--profile_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--regimes", default="neutral,amplifying,corrective")
    parser.add_argument("--profile_epsilon", type=float, default=1e-3)
    parser.add_argument("--gain_quantile", type=float, default=0.5)
    parser.add_argument(
        "--attack_eps_mode",
        choices=["same_as_profile", "mean_positive", "max_positive"],
        default="mean_positive",
    )
    return parser.parse_args()


def parse_regimes(text: str) -> List[str]:
    return [item.strip().lower() for item in str(text or "").replace(" ", ",").split(",") if item.strip()]


def read_csvs(paths: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def finite_mean(values: Iterable[Any]) -> float:
    numbers = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    numbers = numbers[np.isfinite(numbers)]
    if numbers.empty:
        return float("nan")
    return float(numbers.mean())


def finite_max(values: Iterable[Any]) -> float:
    numbers = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    numbers = numbers[np.isfinite(numbers)]
    if numbers.empty:
        return float("nan")
    return float(numbers.max())


def close_to(series: pd.Series, value: float) -> pd.Series:
    numbers = pd.to_numeric(series, errors="coerce")
    return np.isclose(numbers.astype(float), float(value), rtol=1e-9, atol=1e-12)


def path_token(path: Path, prefix: str) -> str:
    for part in path.parts:
        if part.startswith(prefix):
            return part[len(prefix) :]
    return ""


def infer_regime_from_path(path: Path, default: str = "neutral") -> str:
    text = "/" + str(path).replace("\\", "/").strip("/").lower() + "/"
    for regime in EXPECTED_ORDER:
        if f"/{regime}/" in text:
            return regime
    return default


def ensure_regime_column(df: pd.DataFrame, paths_default: str = "neutral") -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "role_response_regime" not in out.columns:
        out["role_response_regime"] = paths_default
    out["role_response_regime"] = out["role_response_regime"].fillna(paths_default).astype(str).str.lower()
    return out


def find_attack_files(attack_aggregate_dir: Path, dataset: str) -> Tuple[List[Path], List[Path]]:
    if attack_aggregate_dir.is_file():
        per_condition = [attack_aggregate_dir]
        epsilon50 = []
    else:
        per_condition = sorted(attack_aggregate_dir.glob(f"{dataset}_*_per_condition.csv"))
        epsilon50 = sorted(attack_aggregate_dir.glob(f"{dataset}_*_epsilon50.csv"))
        if not per_condition:
            per_condition = sorted(attack_aggregate_dir.rglob("*per_condition.csv"))
        if not epsilon50:
            epsilon50 = sorted(attack_aggregate_dir.rglob("*epsilon50.csv"))
    return per_condition, epsilon50


def load_attack_tables(attack_aggregate_dir: Path, dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[Path]]:
    per_condition_paths, epsilon50_paths = find_attack_files(attack_aggregate_dir, dataset)
    per_condition = ensure_regime_column(read_csvs(per_condition_paths))
    epsilon50 = ensure_regime_column(read_csvs(epsilon50_paths))
    return per_condition, epsilon50, per_condition_paths + epsilon50_paths


def first_value(df: pd.DataFrame, column: str, default: Any = "") -> Any:
    if df.empty or column not in df.columns:
        return default
    values = df[column].dropna()
    if values.empty:
        return default
    return values.iloc[0]


def mean_summary(
    summary: pd.DataFrame,
    *,
    quantity_type: str,
    epsilon: float,
    site: Optional[str] = None,
    sender_role: Optional[str] = None,
    receiver_role: Optional[str] = None,
    role: Optional[str] = None,
) -> float:
    if summary.empty:
        return float("nan")
    frame = summary[summary["quantity_type"].astype(str) == quantity_type].copy()
    if "epsilon" in frame.columns:
        frame = frame[close_to(frame["epsilon"], epsilon)]
    if site is not None and "site" in frame.columns:
        frame = frame[frame["site"].astype(str) == site]
    if sender_role is not None and "sender_role" in frame.columns:
        frame = frame[frame["sender_role"].astype(str) == sender_role]
    if receiver_role is not None and "receiver_role" in frame.columns:
        frame = frame[frame["receiver_role"].astype(str) == receiver_role]
    if role is not None and "role" in frame.columns:
        frame = frame[frame["role"].astype(str) == role]
    return finite_mean(frame.get("mean", pd.Series(dtype=float)))


def lambda_value(lambdas: pd.DataFrame, site: str, epsilon: float, gain_quantile: float) -> float:
    if lambdas.empty:
        return float("nan")
    frame = lambdas.copy()
    if "site" in frame.columns:
        frame = frame[frame["site"].astype(str) == site]
    if "epsilon" in frame.columns:
        frame = frame[close_to(frame["epsilon"], epsilon)]
    if "gain_quantile" in frame.columns:
        frame = frame[close_to(frame["gain_quantile"], gain_quantile)]
    if "lambda_mode" in frame.columns and (frame["lambda_mode"].astype(str) == "end_to_end_q_path").any():
        frame = frame[frame["lambda_mode"].astype(str) == "end_to_end_q_path"]
    return finite_mean(frame.get("Lambda", pd.Series(dtype=float)))


def load_profile_summary(profile_root: Path, dataset: str, regimes: Sequence[str], profile_epsilon: float, gain_quantile: float) -> Tuple[pd.DataFrame, List[Path]]:
    rows: List[Dict[str, Any]] = []
    files_read: List[Path] = []
    for regime in regimes:
        summary_paths = sorted((profile_root / regime / "summaries" / dataset).glob("R*/seed*/role_profile_summary.csv"))
        for summary_path in summary_paths:
            summary_dir = summary_path.parent
            lambda_path = summary_dir / "lambda_predictions.csv"
            try:
                summary = pd.read_csv(summary_path)
            except Exception:
                continue
            try:
                lambdas = pd.read_csv(lambda_path)
            except Exception:
                lambdas = pd.DataFrame()
            files_read.append(summary_path)
            if lambda_path.exists():
                files_read.append(lambda_path)
            summary = ensure_regime_column(summary, regime)
            lambdas = ensure_regime_column(lambdas, regime) if not lambdas.empty else lambdas
            R = first_value(summary, "R", path_token(summary_dir, "R"))
            seed = first_value(summary, "seed", path_token(summary_dir, "seed"))
            p2c_lambda = lambda_value(lambdas, "p2c", profile_epsilon, gain_quantile)
            c2s_lambda = lambda_value(lambdas, "c2s", profile_epsilon, gain_quantile)
            s2p_lambda = lambda_value(lambdas, "s2p", profile_epsilon, gain_quantile)
            final_c2s_lambda = lambda_value(lambdas, "final_c2s", profile_epsilon, gain_quantile)
            beta_critic = mean_summary(
                summary,
                quantity_type="beta",
                epsilon=profile_epsilon,
                site="p2c",
                sender_role="planner",
                receiver_role="critic",
            )
            beta_solver = mean_summary(
                summary,
                quantity_type="beta",
                epsilon=profile_epsilon,
                site="c2s",
                sender_role="critic",
                receiver_role="solver",
            )
            beta_planner = mean_summary(
                summary,
                quantity_type="beta",
                epsilon=profile_epsilon,
                site="s2p",
                sender_role="solver",
                receiver_role="planner",
            )
            q_solver = mean_summary(
                summary,
                quantity_type="q",
                epsilon=profile_epsilon,
                site="final_c2s",
                role="solver",
            )
            lambda_values = [p2c_lambda, c2s_lambda, s2p_lambda]
            beta_values = [beta_critic, beta_solver, beta_planner]
            rows.append(
                {
                    "dataset": dataset,
                    "role_response_regime": regime,
                    "R": int(R) if str(R).strip().isdigit() else R,
                    "seed": seed,
                    "profile_epsilon": float(profile_epsilon),
                    "gain_quantile": float(gain_quantile),
                    "mean_lambda_handoff": finite_mean(lambda_values),
                    "p2c_lambda": p2c_lambda,
                    "c2s_lambda": c2s_lambda,
                    "s2p_lambda": s2p_lambda,
                    "final_c2s_lambda": final_c2s_lambda,
                    "mean_beta_handoff": finite_mean(beta_values),
                    "beta_critic_from_planner": beta_critic,
                    "beta_solver_from_critic": beta_solver,
                    "beta_planner_from_solver": beta_planner,
                    "q_solver_direct": q_solver,
                    "n_lambda_sites": int(sum(math.isfinite(x) for x in lambda_values)),
                    "n_beta_sites": int(sum(math.isfinite(x) for x in beta_values)),
                }
            )
    return pd.DataFrame(rows, columns=PROFILE_COLUMNS), files_read


def summarize_attacks(
    per_condition: pd.DataFrame,
    epsilon50: pd.DataFrame,
    dataset: str,
    regimes: Sequence[str],
    attack_eps_mode: str,
    profile_epsilon: float,
) -> pd.DataFrame:
    if per_condition.empty:
        return pd.DataFrame(columns=ATTACK_COLUMNS)
    frame = ensure_regime_column(per_condition)
    if "dataset" in frame.columns:
        frame = frame[frame["dataset"].astype(str) == dataset]
    frame = frame[frame["role_response_regime"].isin(regimes)]
    frame["eps_numeric"] = pd.to_numeric(frame.get("eps"), errors="coerce")
    frame = frame[np.isfinite(frame["eps_numeric"])]
    positive = frame[frame["eps_numeric"] > 0].copy()
    eps50 = ensure_regime_column(epsilon50) if not epsilon50.empty else pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    group_cols = ["dataset", "role_response_regime", "R", "site"]
    for key, group in frame.groupby(group_cols, dropna=False, sort=True):
        dataset_value, regime, R, site = key
        group_positive = group[group["eps_numeric"] > 0].copy()
        if attack_eps_mode == "same_as_profile":
            selected = group_positive[close_to(group_positive["eps_numeric"], profile_epsilon)]
        elif attack_eps_mode == "max_positive":
            if group_positive.empty:
                selected = group_positive
            else:
                selected = group_positive[group_positive["eps_numeric"] == group_positive["eps_numeric"].max()]
        else:
            selected = group_positive
        if selected.empty:
            continue
        eps50_value = float("nan")
        if not eps50.empty:
            match = eps50[
                (eps50.get("dataset", "").astype(str) == str(dataset_value))
                & (eps50.get("role_response_regime", "").astype(str) == str(regime))
                & (eps50.get("R", "").astype(str) == str(R))
                & (eps50.get("site", "").astype(str) == str(site))
            ]
            eps50_value = finite_mean(match.get("epsilon50", pd.Series(dtype=float)))
        rows.append(
            {
                "dataset": dataset_value,
                "role_response_regime": regime,
                "R": R,
                "site": site,
                "attack_eps_mode": attack_eps_mode,
                "excess_asrcc": finite_mean(selected.get("excess_asrcc", pd.Series(dtype=float))),
                "raw_asrcc": finite_mean(selected.get("asrcc", pd.Series(dtype=float))),
                "clean_floor": finite_mean(selected.get("clean_flip_floor", pd.Series(dtype=float))),
                "clean_accuracy": finite_mean(selected.get("clean_accuracy", pd.Series(dtype=float))),
                "perturbed_accuracy": finite_mean(selected.get("perturbed_accuracy", pd.Series(dtype=float))),
                "max_excess_asrcc": finite_max(group_positive.get("excess_asrcc", pd.Series(dtype=float))),
                "epsilon50": eps50_value,
                "n_conditions": int(len(selected)),
            }
        )
    return pd.DataFrame(rows, columns=ATTACK_COLUMNS)


def profile_long(profile: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in profile.iterrows():
        base = {
            "dataset": row["dataset"],
            "role_response_regime": row["role_response_regime"],
            "R": row["R"],
            "seed": row["seed"],
            "q_solver_direct": row["q_solver_direct"],
            "mean_lambda_handoff": row["mean_lambda_handoff"],
            "mean_beta_handoff": row["mean_beta_handoff"],
        }
        site_specs = [
            ("p2c", row["p2c_lambda"], row["beta_critic_from_planner"]),
            ("c2s", row["c2s_lambda"], row["beta_solver_from_critic"]),
            ("s2p", row["s2p_lambda"], row["beta_planner_from_solver"]),
            ("final_c2s", row["final_c2s_lambda"], row["q_solver_direct"]),
            ("mean_handoff", row["mean_lambda_handoff"], row["mean_beta_handoff"]),
        ]
        for site, lamb, beta in site_specs:
            rows.append({**base, "site": site, "Lambda": lamb, "beta": beta})
    return pd.DataFrame(rows)


def attack_with_mean_handoff(attack: pd.DataFrame) -> pd.DataFrame:
    if attack.empty:
        return attack
    rows = [attack]
    mean_rows: List[Dict[str, Any]] = []
    for key, group in attack[attack["site"].isin(HANDOFF_SITES)].groupby(
        ["dataset", "role_response_regime", "R"], dropna=False, sort=True
    ):
        dataset, regime, R = key
        mean_rows.append(
            {
                "dataset": dataset,
                "role_response_regime": regime,
                "R": R,
                "site": "mean_handoff",
                "attack_eps_mode": first_value(group, "attack_eps_mode", ""),
                "excess_asrcc": finite_mean(group["excess_asrcc"]),
                "raw_asrcc": finite_mean(group["raw_asrcc"]),
                "clean_floor": finite_mean(group["clean_floor"]),
                "clean_accuracy": finite_mean(group["clean_accuracy"]),
                "perturbed_accuracy": finite_mean(group["perturbed_accuracy"]),
                "max_excess_asrcc": finite_max(group["max_excess_asrcc"]),
                "epsilon50": finite_mean(group["epsilon50"]),
                "n_conditions": int(pd.to_numeric(group["n_conditions"], errors="coerce").fillna(0).sum()),
            }
        )
    if mean_rows:
        rows.append(pd.DataFrame(mean_rows, columns=ATTACK_COLUMNS))
    return pd.concat(rows, ignore_index=True, sort=False)


def rank_order(values: Mapping[str, float]) -> str:
    finite = [(regime, value) for regime, value in values.items() if math.isfinite(float(value))]
    finite.sort(key=lambda item: (-float(item[1]), item[0]))
    return " > ".join(regime for regime, _ in finite)


def matches_expected(values: Mapping[str, float], tolerance: float = 1e-12) -> bool:
    try:
        amp = float(values["amplifying"])
        neutral = float(values["neutral"])
        corrective = float(values["corrective"])
    except (KeyError, TypeError, ValueError):
        return False
    if not all(math.isfinite(x) for x in (amp, neutral, corrective)):
        return False
    return amp + tolerance >= neutral and neutral + tolerance >= corrective


def build_ordering(profile_l: pd.DataFrame, attack_m: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    keys = set()
    if not profile_l.empty:
        keys.update((dataset, row.R, row.site) for row in profile_l.itertuples())
    if not attack_m.empty:
        keys.update((dataset, row.R, row.site) for row in attack_m.itertuples())
    for dataset_value, R, site in sorted(keys, key=lambda item: (str(item[1]), str(item[2]))):
        lambda_group = profile_l[(profile_l["R"].astype(str) == str(R)) & (profile_l["site"].astype(str) == str(site))]
        asr_group = attack_m[(attack_m["R"].astype(str) == str(R)) & (attack_m["site"].astype(str) == str(site))]
        lambda_values = {
            regime: finite_mean(lambda_group[lambda_group["role_response_regime"] == regime]["Lambda"])
            for regime in EXPECTED_ORDER
        }
        asr_values = {
            regime: finite_mean(asr_group[asr_group["role_response_regime"] == regime]["excess_asrcc"])
            for regime in EXPECTED_ORDER
        }
        rows.append(
            {
                "dataset": dataset_value,
                "R": R,
                "site": site,
                "lambda_order": rank_order(lambda_values),
                "asr_order": rank_order(asr_values),
                "lambda_rank_match_expected": matches_expected(lambda_values),
                "asr_rank_match_expected": matches_expected(asr_values),
                "expected_order": " > ".join(EXPECTED_ORDER),
            }
        )
    return pd.DataFrame(rows, columns=ORDERING_COLUMNS)


def robust_pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 2 or np.nanstd(x_arr) == 0 or np.nanstd(y_arr) == 0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def robust_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_series = pd.Series(x, dtype=float)
    y_series = pd.Series(y, dtype=float)
    mask = np.isfinite(x_series) & np.isfinite(y_series)
    if int(mask.sum()) < 2:
        return float("nan")
    xr = x_series[mask].rank(method="average")
    yr = y_series[mask].rank(method="average")
    return robust_pearson(xr.to_numpy(), yr.to_numpy())


def prediction_quality(profile: pd.DataFrame, attack_m: pd.DataFrame, joined: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def add_row(pool: str, frame: pd.DataFrame, x_col: str, y_col: str) -> None:
        if frame.empty or x_col not in frame.columns or y_col not in frame.columns:
            x_values: List[float] = []
            y_values: List[float] = []
        else:
            x_values = pd.to_numeric(frame[x_col], errors="coerce").tolist()
            y_values = pd.to_numeric(frame[y_col], errors="coerce").tolist()
        rows.append(
            {
                "dataset": dataset,
                "condition_pool": pool,
                "pearson_lambda_excess_asrcc": robust_pearson(x_values, y_values),
                "spearman_lambda_excess_asrcc": robust_spearman(x_values, y_values),
                "n": int(np.sum(np.isfinite(x_values) & np.isfinite(y_values))) if x_values else 0,
            }
        )

    attack_mean = attack_m[attack_m["site"] == "mean_handoff"].copy() if not attack_m.empty else pd.DataFrame()
    horizon = profile.merge(
        attack_mean[["dataset", "role_response_regime", "R", "excess_asrcc"]],
        on=["dataset", "role_response_regime", "R"],
        how="inner",
    ) if not profile.empty and not attack_mean.empty else pd.DataFrame()
    add_row("horizon_mean", horizon, "mean_lambda_handoff", "excess_asrcc")

    site_horizon = joined[joined["site"].isin(HANDOFF_SITES)].copy() if not joined.empty else pd.DataFrame()
    add_row("site_horizon", site_horizon, "Lambda", "excess_asrcc")

    if not site_horizon.empty:
        regime_mean = site_horizon.groupby("role_response_regime", as_index=False).agg(
            Lambda=("Lambda", "mean"),
            excess_asrcc=("excess_asrcc", "mean"),
        )
    else:
        regime_mean = pd.DataFrame()
    add_row("regime_mean", regime_mean, "Lambda", "excess_asrcc")
    return pd.DataFrame(rows)


def write_readme(
    path: Path,
    *,
    files_read: Sequence[Path],
    regimes: Sequence[str],
    present_regimes: Sequence[str],
    profile_epsilon: float,
    gain_quantile: float,
    attack_eps_mode: str,
    warnings: Sequence[str],
) -> None:
    missing = [regime for regime in regimes if regime not in set(present_regimes)]
    lines = [
        "Experiment E role-response regime comparison",
        "",
        "Files read:",
    ]
    lines.extend(f"- {file_path}" for file_path in files_read)
    if not files_read:
        lines.append("- <none>")
    lines.extend(
        [
            "",
            f"Requested regimes: {', '.join(regimes)}",
            f"Present regimes: {', '.join(present_regimes) if present_regimes else '<none>'}",
            f"Missing regimes: {', '.join(missing) if missing else '<none>'}",
            f"Profile epsilon: {profile_epsilon:g}",
            f"Gain quantile: {gain_quantile:g}",
            f"Attack epsilon mode: {attack_eps_mode}",
            "",
            "Caveats:",
            "- Correlations are descriptive and return NaN for fewer than two finite non-constant points.",
            "- Attack summaries depend on the aggregate per_condition CSV and its clean baseline matching.",
            "- Missing regimes or missing profile/attack files are allowed and produce partial outputs.",
        ]
    )
    if warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    regimes = parse_regimes(args.regimes)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []

    attack, epsilon50, attack_files = load_attack_tables(Path(args.attack_aggregate_dir), args.dataset)
    profile, profile_files = load_profile_summary(
        Path(args.profile_root),
        args.dataset,
        regimes,
        float(args.profile_epsilon),
        float(args.gain_quantile),
    )
    attack_summary = summarize_attacks(
        attack,
        epsilon50,
        args.dataset,
        regimes,
        args.attack_eps_mode,
        float(args.profile_epsilon),
    )
    attack_mean = attack_with_mean_handoff(attack_summary)
    prof_long = profile_long(profile) if not profile.empty else pd.DataFrame()
    if prof_long.empty or attack_mean.empty:
        joined = pd.DataFrame()
    else:
        joined = prof_long.merge(
            attack_mean,
            on=["dataset", "role_response_regime", "R", "site"],
            how="outer",
        )
    ordering = build_ordering(prof_long, attack_mean, args.dataset)
    quality = prediction_quality(profile, attack_mean, joined, args.dataset)

    if profile.empty:
        warnings.append("No role profile summaries were found.")
    if attack_summary.empty:
        warnings.append("No attack aggregate rows were summarized.")
    present_regimes = sorted(
        set(profile.get("role_response_regime", pd.Series(dtype=str)).dropna().astype(str))
        | set(attack_summary.get("role_response_regime", pd.Series(dtype=str)).dropna().astype(str))
    )

    profile.to_csv(out_dir / "role_regime_profile_summary.csv", index=False)
    attack_summary.to_csv(out_dir / "role_regime_attack_summary.csv", index=False)
    joined.to_csv(out_dir / "role_regime_joined.csv", index=False)
    ordering.to_csv(out_dir / "role_regime_ordering.csv", index=False)
    quality.to_csv(out_dir / "role_regime_prediction_quality.csv", index=False)
    write_readme(
        out_dir / "README.txt",
        files_read=attack_files + profile_files,
        regimes=regimes,
        present_regimes=present_regimes,
        profile_epsilon=float(args.profile_epsilon),
        gain_quantile=float(args.gain_quantile),
        attack_eps_mode=args.attack_eps_mode,
        warnings=warnings,
    )

    print("mean Lambda by regime")
    if profile.empty:
        print("<none>")
    else:
        print(profile.groupby("role_response_regime")["mean_lambda_handoff"].mean().to_string())
    print("mean excess ASR by regime")
    if attack_summary.empty:
        print("<none>")
    else:
        print(attack_summary.groupby("role_response_regime")["excess_asrcc"].mean().to_string())
    print("rank match status")
    if ordering.empty:
        print("<none>")
    else:
        lambda_ok = bool(ordering["lambda_rank_match_expected"].fillna(False).all())
        asr_ok = bool(ordering["asr_rank_match_expected"].fillna(False).all())
        print(f"lambda_all_match={lambda_ok} asr_all_match={asr_ok}")
    print(f"out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
