#!/bin/bash
#SBATCH --job-name=latent-contagion-e
#SBATCH -p NA100q
#SBATCH -w node01
#SBATCH --output=logs/latent_contagion_e_%A_%a.out
#SBATCH --error=logs/latent_contagion_e_%A_%a.err

# Experiment E: prompt-induced role-response regimes.
#
# Print attack grid:
#   E_STAGE=attack_grid ROLE_RESPONSE_REGIMES="neutral amplifying corrective" \
#     ROUNDS="1 2 3 4 5" DATASETS="math500" \
#     bash experiments/latent_contagion/run_experiment_e.sh
#
# Run attacks:
#   GPU_LIST="0 1 2 3" E_STAGE=attack ROLE_RESPONSE_REGIMES="neutral amplifying corrective" \
#     ROUNDS="1 2 3 4 5" DATASETS="math500" LC_ROUNDS="0" \
#     sbatch --array=0-N%4 experiments/latent_contagion/run_experiment_e.sh
#
# Role-profile clean:
#   GPU_LIST="0 1 2 3" E_STAGE=profile_clean sbatch --array=0-N%4 experiments/latent_contagion/run_experiment_e.sh
#
# Role-profile probes:
#   GPU_LIST="0 1 2 3" E_STAGE=profile_probe ROLE_EPSILONS="1e-3" \
#     ROLE_PROBE_TARGETS="message terminal" sbatch --array=0-N%4 experiments/latent_contagion/run_experiment_e.sh
#
# Estimate profiles:
#   E_STAGE=profile_estimate sbatch --array=0-N%4 experiments/latent_contagion/run_experiment_e.sh
#
# Aggregate and compare:
#   E_STAGE=aggregate_attack bash experiments/latent_contagion/run_experiment_e.sh
#   E_STAGE=compare bash experiments/latent_contagion/run_experiment_e.sh

set -euo pipefail

mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-python}"
STYLE="${STYLE:-sequential_light}"
METHOD="${METHOD:-ours_recursive}"
DATASETS="${DATASETS:-math500}"
ROUNDS="${ROUNDS:-1 2 3 4 5}"
SEEDS="${SEEDS:-42}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LATENT_LENGTH="${LATENT_LENGTH:-48}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
OUT_ROOT="${OUT_ROOT:-outputs/latent_contagion/experiment_e}"

ROLE_RESPONSE_REGIMES="${ROLE_RESPONSE_REGIMES:-neutral amplifying corrective}"
ROLE_RESPONSE_REGIME_PATH="${ROLE_RESPONSE_REGIME_PATH:-}"

E_STAGE="${E_STAGE:-attack}"

SITES="${SITES:-p2c c2s s2p}"
EPSILONS="${EPSILONS:-0 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1}"
LC_MODE="${LC_MODE:-one_shot}"
LC_ROUNDS="${LC_ROUNDS:-0}"
LC_DIRECTION="${LC_DIRECTION:-bank}"
LC_STEERING_METHOD="${LC_STEERING_METHOD:-diffmean}"
CALIB_ROOT="${CALIB_ROOT:-outputs/latent_contagion/diffmean_calibration}"
STEERING_FILTER="${STEERING_FILTER:-clean_correct_attack_wrong}"
ATTACK_BANK_MODE="${ATTACK_BANK_MODE:-fixed_neutral}"
LC_STEERING_BANK="${LC_STEERING_BANK:-}"
LC_STEERING_ID="${LC_STEERING_ID:-}"
ATTACK_RUN_SUBDIR="${ATTACK_RUN_SUBDIR:-structured_${STEERING_FILTER}_${ATTACK_BANK_MODE}}"

ROLE_EPSILONS="${ROLE_EPSILONS:-1e-3}"
ROLE_PROBE_TARGETS="${ROLE_PROBE_TARGETS:-message terminal}"
ROLE_PROBE_SITES_MESSAGE="${ROLE_PROBE_SITES_MESSAGE:-p2c c2s s2p}"
ROLE_PROBE_SITES_STATE="${ROLE_PROBE_SITES_STATE:-planner_self critic_self solver_self}"
ROLE_PROBE_SITES_TERMINAL="${ROLE_PROBE_SITES_TERMINAL:-final_c2s}"
ROLE_PROBE_ROUND_MODE="${ROLE_PROBE_ROUND_MODE:-first}"
ROLE_PROBE_ROUNDS="${ROLE_PROBE_ROUNDS:-}"
ROLE_TRACE_DTYPE="${ROLE_TRACE_DTYPE:-float32}"
ROLE_PROFILE_DIRECTION="${ROLE_PROFILE_DIRECTION:-random}"
ROLE_PROFILE_QUANTILES="${ROLE_PROFILE_QUANTILES:-0.5 0.75 0.9 0.95}"
CLEAN_CORRECT_ONLY="${CLEAN_CORRECT_ONLY:-1}"
TAU_PROXY="${TAU_PROXY:-clean_clean_floor}"
LAMBDA_STABILIZER="${LAMBDA_STABILIZER:-1e-8}"
LAMBDA_MODE="${LAMBDA_MODE:-end_to_end_q_path}"
LAMBDA_MISSING_GAIN_POLICY="${LAMBDA_MISSING_GAIN_POLICY:-nan}"
LAMBDA_Q_SOURCE="${LAMBDA_Q_SOURCE:-direct_q}"
ALLOW_RECOMPUTED_INPUT_DELTA="${ALLOW_RECOMPUTED_INPUT_DELTA:-0}"
LAMBDA_GRID_FROM_EXPERIMENT_C="${LAMBDA_GRID_FROM_EXPERIMENT_C:-}"
GAIN_QUANTILE="${GAIN_QUANTILE:-0.5}"
ATTACK_EPS_MODE="${ATTACK_EPS_MODE:-mean_positive}"

GPU_LIST="${GPU_LIST:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
ESTIMATE_EXTRA_ARGS="${ESTIMATE_EXTRA_ARGS:-}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

die() {
  echo "[error] $*" >&2
  exit 2
}

validate_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    die "$name must be a positive integer, got: $value"
  fi
}

validate_nonnegative_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    die "$name must be a non-negative integer, got: $value"
  fi
}

validate_stage() {
  case "$E_STAGE" in
    attack_grid|attack|profile_clean_grid|profile_clean|profile_probe_grid|profile_probe|profile_estimate_grid|profile_estimate|aggregate_attack|compare|all) ;;
    *) die "E_STAGE must be one of: attack_grid, attack, profile_clean_grid, profile_clean, profile_probe_grid, profile_probe, profile_estimate_grid, profile_estimate, aggregate_attack, compare, all. Got: $E_STAGE" ;;
  esac
}

validate_regime() {
  case "$1" in
    neutral|amplifying|corrective|custom) ;;
    *) die "ROLE_RESPONSE_REGIMES item must be neutral, amplifying, corrective, or custom. Got: $1" ;;
  esac
}

validate_target() {
  case "$1" in
    message|state|terminal) ;;
    *) die "ROLE_PROBE_TARGETS item must be message, state, or terminal. Got: $1" ;;
  esac
}

validate_site_for_target() {
  local target="$1"
  local site="$2"
  case "$target:$site" in
    message:p2c|message:c2s|message:s2p) ;;
    state:planner_self|state:critic_self|state:refiner_self|state:solver_self) ;;
    terminal:final_c2s) ;;
    *) die "Invalid probe target/site pair: target=$target site=$site" ;;
  esac
}

validate_lc_site() {
  case "$1" in
    p2c|c2s|s2p) ;;
    *) die "SITES item must be one of p2c, c2s, s2p. Got: $1" ;;
  esac
}

validate_grid_inputs() {
  local regime rounds seed site target eps lc_round normalized
  for regime in $ROLE_RESPONSE_REGIMES; do
    validate_regime "$regime"
  done
  for rounds in $ROUNDS; do
    validate_positive_int "ROUNDS item" "$rounds"
  done
  for seed in $SEEDS; do
    validate_nonnegative_int "SEEDS item" "$seed"
  done
  for site in $SITES; do
    validate_lc_site "$site"
  done
  if [[ "$LC_ROUNDS" != "all" ]]; then
    normalized="${LC_ROUNDS//,/ }"
    if [[ -z "${normalized//[[:space:]]/}" ]]; then
      die "LC_ROUNDS must be 'all' or a list of zero-based non-negative integers."
    fi
    for lc_round in $normalized; do
      validate_nonnegative_int "LC_ROUNDS item" "$lc_round"
    done
  fi
  case "$ATTACK_BANK_MODE" in
    fixed_neutral|per_regime|explicit) ;;
    *) die "ATTACK_BANK_MODE must be fixed_neutral, per_regime, or explicit. Got: $ATTACK_BANK_MODE" ;;
  esac
  for eps in $ROLE_EPSILONS; do
    "$PYTHON_BIN" - "$eps" <<'PY' >/dev/null || die "ROLE_EPSILONS items must be finite non-negative floats. Got: $eps"
import math
import sys
value = float(sys.argv[1])
if not math.isfinite(value) or value < 0:
    raise SystemExit(2)
PY
  done
  for target in $ROLE_PROBE_TARGETS; do
    validate_target "$target"
    for site in $(probe_sites_for_target "$target"); do
      validate_site_for_target "$target" "$site"
    done
  done
  case "$ROLE_TRACE_DTYPE" in
    float32|float16|bfloat16) ;;
    *) die "ROLE_TRACE_DTYPE must be float32, float16, or bfloat16. Got: $ROLE_TRACE_DTYPE" ;;
  esac
  case "$ROLE_PROFILE_DIRECTION" in
    random) ;;
    *) die "ROLE_PROFILE_DIRECTION currently supports only random. Got: $ROLE_PROFILE_DIRECTION" ;;
  esac
  case "$ROLE_PROBE_ROUND_MODE" in
    first|all|custom) ;;
    *) die "ROLE_PROBE_ROUND_MODE must be one of: first, all, custom. Got: $ROLE_PROBE_ROUND_MODE" ;;
  esac
  if [[ "$ROLE_PROBE_ROUND_MODE" == "custom" && -z "$ROLE_PROBE_ROUNDS" ]]; then
    die "ROLE_PROBE_ROUNDS must be set when ROLE_PROBE_ROUND_MODE=custom."
  fi
  case "$LAMBDA_MODE" in
    end_to_end_q_path|factorized_timevarying|stationary_round0|both) ;;
    *) die "LAMBDA_MODE must be one of: end_to_end_q_path, factorized_timevarying, stationary_round0, both. Got: $LAMBDA_MODE" ;;
  esac
  case "$LAMBDA_MISSING_GAIN_POLICY" in
    nan|zero) ;;
    *) die "LAMBDA_MISSING_GAIN_POLICY must be nan or zero. Got: $LAMBDA_MISSING_GAIN_POLICY" ;;
  esac
  case "$LAMBDA_Q_SOURCE" in
    direct_q|q_path_fallback) ;;
    *) die "LAMBDA_Q_SOURCE must be direct_q or q_path_fallback. Got: $LAMBDA_Q_SOURCE" ;;
  esac
  case "$ALLOW_RECOMPUTED_INPUT_DELTA" in
    0|1) ;;
    *) die "ALLOW_RECOMPUTED_INPUT_DELTA must be 0 or 1. Got: $ALLOW_RECOMPUTED_INPUT_DELTA" ;;
  esac
  case "$ATTACK_EPS_MODE" in
    same_as_profile|mean_positive|max_positive) ;;
    *) die "ATTACK_EPS_MODE must be same_as_profile, mean_positive, or max_positive. Got: $ATTACK_EPS_MODE" ;;
  esac
}

probe_sites_for_target() {
  case "$1" in
    message) echo "$ROLE_PROBE_SITES_MESSAGE" ;;
    state) echo "$ROLE_PROBE_SITES_STATE" ;;
    terminal) echo "$ROLE_PROBE_SITES_TERMINAL" ;;
    *) die "Unknown probe target: $1" ;;
  esac
}

valid_lc_round_for_site_R() {
  local site="$1"
  local rounds="$2"
  local lc_round="$3"
  validate_positive_int "ROUNDS item" "$rounds"
  validate_nonnegative_int "LC_ROUNDS item" "$lc_round"
  case "$site" in
    p2c|c2s) (( lc_round < rounds )) ;;
    s2p) (( rounds > 1 && lc_round < rounds - 1 )) ;;
    *) die "SITES item must be one of p2c, c2s, s2p. Got: $site" ;;
  esac
}

candidate_lc_rounds_for_config() {
  local site="$1"
  local rounds="$2"
  validate_positive_int "ROUNDS item" "$rounds"
  if [[ "$LC_ROUNDS" == "all" ]]; then
    local max_round
    case "$site" in
      p2c|c2s) max_round=$((rounds - 1)) ;;
      s2p) max_round=$((rounds - 2)) ;;
      *) die "SITES item must be one of p2c, c2s, s2p. Got: $site" ;;
    esac
    if (( max_round < 0 )); then
      return 0
    fi
    local lc_round
    for ((lc_round = 0; lc_round <= max_round; lc_round++)); do
      echo "$lc_round"
    done
    return 0
  fi

  local normalized="${LC_ROUNDS//,/ }"
  local lc_round seen
  seen=" "
  for lc_round in $normalized; do
    validate_nonnegative_int "LC_ROUNDS item" "$lc_round"
    if valid_lc_round_for_site_R "$site" "$rounds" "$lc_round"; then
      if [[ "$seen" != *" $lc_round "* ]]; then
        echo "$lc_round"
        seen="${seen}${lc_round} "
      fi
    fi
  done
}

all_probe_rounds_for_config() {
  local target="$1"
  local site="$2"
  local rounds="$3"
  validate_positive_int "ROUNDS item" "$rounds"
  case "$target:$site" in
    message:p2c|message:c2s|state:planner_self|state:critic_self|state:refiner_self)
      seq 0 $((rounds - 1))
      ;;
    message:s2p|state:solver_self)
      if (( rounds > 1 )); then
        seq 0 $((rounds - 2))
      fi
      ;;
    terminal:final_c2s)
      echo $((rounds - 1))
      ;;
    *)
      die "Invalid probe target/site pair: target=$target site=$site"
      ;;
  esac
}

first_probe_rounds_for_config() {
  local target="$1"
  local site="$2"
  local rounds="$3"
  validate_positive_int "ROUNDS item" "$rounds"
  case "$target:$site" in
    message:p2c|message:c2s|state:planner_self|state:critic_self|state:refiner_self)
      echo 0
      ;;
    message:s2p|state:solver_self)
      if (( rounds > 1 )); then
        echo 0
      fi
      ;;
    terminal:final_c2s)
      echo $((rounds - 1))
      ;;
    *)
      die "Invalid probe target/site pair: target=$target site=$site"
      ;;
  esac
}

custom_probe_rounds_for_config() {
  local target="$1"
  local site="$2"
  local rounds="$3"
  local requested valid
  if [[ "$target:$site" == "terminal:final_c2s" ]]; then
    echo $((rounds - 1))
    return 0
  fi
  for requested in $ROLE_PROBE_ROUNDS; do
    for valid in $(all_probe_rounds_for_config "$target" "$site" "$rounds"); do
      if [[ "$requested" == "$valid" ]]; then
        echo "$requested"
      fi
    done
  done | awk '!seen[$0]++'
}

candidate_probe_rounds_for_config() {
  case "$ROLE_PROBE_ROUND_MODE" in
    first) first_probe_rounds_for_config "$@" ;;
    all) all_probe_rounds_for_config "$@" ;;
    custom) custom_probe_rounds_for_config "$@" ;;
    *) die "ROLE_PROBE_ROUND_MODE must be one of: first, all, custom. Got: $ROLE_PROBE_ROUND_MODE" ;;
  esac
}

sanitize_token() {
  echo "$1" | sed 's/[^A-Za-z0-9._+-]/_/g'
}

join_csv_words() {
  local out=""
  local item
  for item in "$@"; do
    if [[ -z "$out" ]]; then
      out="$item"
    else
      out="${out},${item}"
    fi
  done
  echo "$out"
}

first_word() {
  local first=""
  read -r first _ <<< "$1"
  echo "$first"
}

write_command_txt() {
  local path="$1"
  shift
  mkdir -p "$(dirname "$path")"
  : > "$path"
  printf '%q ' "$@" >> "$path"
  printf '\n' >> "$path"
}

write_manifest_json() {
  local path="$1"
  shift
  mkdir -p "$(dirname "$path")"
  "$PYTHON_BIN" - "$path" "$@" <<'PY'
import json
import os
import sys

path = sys.argv[1]
manifest = {}
for item in sys.argv[2:]:
    key, _, value = item.partition("=")
    manifest[key] = value
manifest["cwd"] = os.getcwd()
with open(path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

append_common_optional_args() {
  if [[ -n "${SAMPLE_SEED:-}" ]]; then
    cmd+=(--sample_seed "$SAMPLE_SEED")
  fi
  if [[ -n "${TEMPERATURE:-}" ]]; then
    cmd+=(--temperature "$TEMPERATURE")
  fi
  if [[ -n "${TOP_P:-}" ]]; then
    cmd+=(--top_p "$TOP_P")
  fi
  if [[ -n "${TOP_K:-}" ]]; then
    cmd+=(--top_k "$TOP_K")
  fi
  if [[ -n "${DEVICE:-}" ]]; then
    cmd+=(--device "$DEVICE")
  fi
  if [[ -n "$EXTRA_ARGS" ]]; then
    read -r -a extra_args_array <<< "$EXTRA_ARGS"
    cmd+=("${extra_args_array[@]}")
  fi
}

configure_environment() {
  validate_nonnegative_int "SLURM_ARRAY_TASK_ID" "$TASK_ID"
  if [[ -n "$GPU_LIST" ]]; then
    read -r -a gpu_array <<< "$GPU_LIST"
    if (( ${#gpu_array[@]} == 0 )); then
      die "GPU_LIST was set but no GPU ids were parsed."
    fi
    gpu_index=$((TASK_ID % ${#gpu_array[@]}))
    export CUDA_VISIBLE_DEVICES="${gpu_array[$gpu_index]}"
  else
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
  fi
  export CONDA_NO_PLUGINS=true
  export TMPDIR="${SLURM_TMPDIR:-/tmp}"
  export PYTHONNOUSERSITE=1
  unset PYTHONPATH || true
  mkdir -p "$TMPDIR"
}

select_attack_bank() {
  local regime="$1"
  local dataset="$2"
  local rounds="$3"
  case "$ATTACK_BANK_MODE" in
    fixed_neutral)
      LC_STEERING_ID_EFFECTIVE="${LC_STEERING_ID:-diffmean_R${rounds}_${dataset}_role_aligned_${STEERING_FILTER}}"
      LC_STEERING_BANK_EFFECTIVE="${LC_STEERING_BANK:-$CALIB_ROOT/${dataset}_R${rounds}/${LC_STEERING_ID_EFFECTIVE}.pt}"
      ;;
    per_regime)
      LC_STEERING_ID_EFFECTIVE="${LC_STEERING_ID:-diffmean_R${rounds}_${dataset}_${regime}_role_aligned_${STEERING_FILTER}}"
      LC_STEERING_BANK_EFFECTIVE="${LC_STEERING_BANK:-$CALIB_ROOT/${regime}/${dataset}_R${rounds}/${LC_STEERING_ID_EFFECTIVE}.pt}"
      ;;
    explicit)
      if [[ -z "$LC_STEERING_BANK" ]]; then
        die "ATTACK_BANK_MODE=explicit requires LC_STEERING_BANK."
      fi
      LC_STEERING_BANK_EFFECTIVE="$LC_STEERING_BANK"
      if [[ -n "$LC_STEERING_ID" ]]; then
        LC_STEERING_ID_EFFECTIVE="$LC_STEERING_ID"
      else
        LC_STEERING_ID_EFFECTIVE="$(basename "$LC_STEERING_BANK_EFFECTIVE" .pt)"
      fi
      ;;
    *)
      die "Unsupported ATTACK_BANK_MODE=$ATTACK_BANK_MODE"
      ;;
  esac
}

total_attack_jobs() {
  local total=0
  local regime dataset site eps rounds lc_round seed
  for regime in $ROLE_RESPONSE_REGIMES; do
    for dataset in $DATASETS; do
      for site in $SITES; do
        for eps in $EPSILONS; do
          for rounds in $ROUNDS; do
            for lc_round in $(candidate_lc_rounds_for_config "$site" "$rounds"); do
              for seed in $SEEDS; do
                total=$((total + 1))
              done
            done
          done
        done
      done
    done
  done
  echo "$total"
}

print_attack_grid() {
  local index=0
  local regime dataset site eps rounds lc_round seed total
  total="$(total_attack_jobs)"
  echo "total_attack_jobs=$total"
  echo "array_index regime dataset site epsilon R lc_round seed"
  for regime in $ROLE_RESPONSE_REGIMES; do
    for dataset in $DATASETS; do
      for site in $SITES; do
        for eps in $EPSILONS; do
          for rounds in $ROUNDS; do
            for lc_round in $(candidate_lc_rounds_for_config "$site" "$rounds"); do
              for seed in $SEEDS; do
                echo "$index $regime $dataset $site $eps $rounds $lc_round $seed"
                index=$((index + 1))
              done
            done
          done
        done
      done
    done
  done
}

select_attack_config() {
  local target_index="$1"
  local index=0
  local regime dataset site eps rounds lc_round seed
  for regime in $ROLE_RESPONSE_REGIMES; do
    for dataset in $DATASETS; do
      for site in $SITES; do
        for eps in $EPSILONS; do
          for rounds in $ROUNDS; do
            for lc_round in $(candidate_lc_rounds_for_config "$site" "$rounds"); do
              for seed in $SEEDS; do
                if (( index == target_index )); then
                  ROLE_RESPONSE_REGIME="$regime"
                  DATASET="$dataset"
                  SITE="$site"
                  EPS="$eps"
                  R="$rounds"
                  LC_ROUND_EFFECTIVE="$lc_round"
                  SEED="$seed"
                  return 0
                fi
                index=$((index + 1))
              done
            done
          done
        done
      done
    done
  done
  return 1
}

total_profile_clean_jobs() {
  local total=0
  local regime dataset rounds seed
  for regime in $ROLE_RESPONSE_REGIMES; do
    for dataset in $DATASETS; do
      for rounds in $ROUNDS; do
        for seed in $SEEDS; do
          total=$((total + 1))
        done
      done
    done
  done
  echo "$total"
}

print_profile_clean_grid() {
  local index=0
  local regime dataset rounds seed total
  total="$(total_profile_clean_jobs)"
  echo "total_profile_clean_jobs=$total"
  echo "array_index regime dataset R seed"
  for regime in $ROLE_RESPONSE_REGIMES; do
    for dataset in $DATASETS; do
      for rounds in $ROUNDS; do
        for seed in $SEEDS; do
          echo "$index $regime $dataset $rounds $seed"
          index=$((index + 1))
        done
      done
    done
  done
}

select_profile_clean_config() {
  local target_index="$1"
  local index=0
  local regime dataset rounds seed
  for regime in $ROLE_RESPONSE_REGIMES; do
    for dataset in $DATASETS; do
      for rounds in $ROUNDS; do
        for seed in $SEEDS; do
          if (( index == target_index )); then
            ROLE_RESPONSE_REGIME="$regime"
            DATASET="$dataset"
            R="$rounds"
            SEED="$seed"
            return 0
          fi
          index=$((index + 1))
        done
      done
    done
  done
  return 1
}

total_profile_probe_jobs() {
  local total=0
  local regime dataset rounds seed target site probe_round eps
  for regime in $ROLE_RESPONSE_REGIMES; do
    for dataset in $DATASETS; do
      for rounds in $ROUNDS; do
        for seed in $SEEDS; do
          for target in $ROLE_PROBE_TARGETS; do
            for site in $(probe_sites_for_target "$target"); do
              for probe_round in $(candidate_probe_rounds_for_config "$target" "$site" "$rounds"); do
                for eps in $ROLE_EPSILONS; do
                  total=$((total + 1))
                done
              done
            done
          done
        done
      done
    done
  done
  echo "$total"
}

print_profile_probe_grid() {
  local index=0
  local regime dataset rounds seed target site probe_round eps total
  total="$(total_profile_probe_jobs)"
  echo "total_profile_probe_jobs=$total"
  echo "array_index regime dataset R seed probe_target probe_site probe_round epsilon"
  for regime in $ROLE_RESPONSE_REGIMES; do
    for dataset in $DATASETS; do
      for rounds in $ROUNDS; do
        for seed in $SEEDS; do
          for target in $ROLE_PROBE_TARGETS; do
            for site in $(probe_sites_for_target "$target"); do
              for probe_round in $(candidate_probe_rounds_for_config "$target" "$site" "$rounds"); do
                for eps in $ROLE_EPSILONS; do
                  echo "$index $regime $dataset $rounds $seed $target $site $probe_round $eps"
                  index=$((index + 1))
                done
              done
            done
          done
        done
      done
    done
  done
}

select_profile_probe_config() {
  local target_index="$1"
  local index=0
  local regime dataset rounds seed target site probe_round eps
  for regime in $ROLE_RESPONSE_REGIMES; do
    for dataset in $DATASETS; do
      for rounds in $ROUNDS; do
        for seed in $SEEDS; do
          for target in $ROLE_PROBE_TARGETS; do
            for site in $(probe_sites_for_target "$target"); do
              for probe_round in $(candidate_probe_rounds_for_config "$target" "$site" "$rounds"); do
                for eps in $ROLE_EPSILONS; do
                  if (( index == target_index )); then
                    ROLE_RESPONSE_REGIME="$regime"
                    DATASET="$dataset"
                    R="$rounds"
                    SEED="$seed"
                    PROBE_TARGET="$target"
                    PROBE_SITE="$site"
                    PROBE_ROUND="$probe_round"
                    EPS="$eps"
                    return 0
                  fi
                  index=$((index + 1))
                done
              done
            done
          done
        done
      done
    done
  done
  return 1
}

total_profile_estimate_jobs() {
  total_profile_clean_jobs
}

select_profile_estimate_config() {
  select_profile_clean_config "$1"
}

run_attack_config() {
  local regime="$1"
  local dataset="$2"
  local site="$3"
  local eps="$4"
  local rounds="$5"
  local lc_round="$6"
  local seed="$7"
  local run_dir result_jsonl meta_dir command_path manifest_path run_log
  select_attack_bank "$regime" "$dataset" "$rounds"
  if [[ "$LC_DIRECTION" == "bank" && ! -f "$LC_STEERING_BANK_EFFECTIVE" ]]; then
    die "Selected steering bank does not exist: $LC_STEERING_BANK_EFFECTIVE"
  fi
  run_dir="$OUT_ROOT/attacks/$dataset/$ATTACK_RUN_SUBDIR/$regime"
  result_jsonl="$run_dir/site=${site}_eps=${eps}_R=${rounds}_lc_round=${lc_round}_seed=${seed}.jsonl"
  meta_dir="$run_dir/runs/site=${site}_eps=$(sanitize_token "$eps")_R=${rounds}_lc_round=${lc_round}_seed=${seed}"
  command_path="$meta_dir/command.txt"
  manifest_path="$meta_dir/manifest.json"
  run_log="$meta_dir/run.log"
  mkdir -p "$run_dir" "$meta_dir"

  cmd=(
    "$PYTHON_BIN" RecursiveMAS/run.py
    --style "$STYLE"
    --dataset "$dataset"
    --method "$METHOD"
    --num_recursive_rounds "$rounds"
    --num_samples "$NUM_SAMPLES"
    --batch_size "$BATCH_SIZE"
    --latent_length "$LATENT_LENGTH"
    --seed "$seed"
    --trust_remote_code "$TRUST_REMOTE_CODE"
    --deterministic 1
    --role_response_regime "$regime"
    --role_response_regime_path "$ROLE_RESPONSE_REGIME_PATH"
    --lc_mode "$LC_MODE"
    --lc_site "$site"
    --lc_epsilon "$eps"
    --lc_round "$lc_round"
    --lc_seed "$seed"
    --lc_direction "$LC_DIRECTION"
    --result_jsonl "$result_jsonl"
  )
  if [[ -n "$LC_STEERING_BANK_EFFECTIVE" ]]; then
    cmd+=(--lc_steering_bank "$LC_STEERING_BANK_EFFECTIVE")
  fi
  if [[ -n "$LC_STEERING_METHOD" ]]; then
    cmd+=(--lc_steering_method "$LC_STEERING_METHOD")
  fi
  if [[ -n "$LC_STEERING_ID_EFFECTIVE" ]]; then
    cmd+=(--lc_steering_id "$LC_STEERING_ID_EFFECTIVE")
  fi
  append_common_optional_args
  write_command_txt "$command_path" "${cmd[@]}"
  write_manifest_json "$manifest_path" \
    "stage=attack" "style=$STYLE" "method=$METHOD" "dataset=$dataset" \
    "R=$rounds" "site=$site" "epsilon=$eps" "lc_round=$lc_round" "seed=$seed" \
    "role_response_regime=$regime" "role_response_regime_path=$ROLE_RESPONSE_REGIME_PATH" \
    "attack_bank_mode=$ATTACK_BANK_MODE" "selected_steering_bank=$LC_STEERING_BANK_EFFECTIVE" \
    "selected_steering_id=$LC_STEERING_ID_EFFECTIVE" "lc_mode=$LC_MODE" \
    "lc_direction=$LC_DIRECTION" "lc_steering_method=$LC_STEERING_METHOD" \
    "steering_filter=$STEERING_FILTER" "result_jsonl=$result_jsonl" "run_log=$run_log" \
    "num_samples=$NUM_SAMPLES" "batch_size=$BATCH_SIZE" "latent_length=$LATENT_LENGTH" \
    "trust_remote_code=$TRUST_REMOTE_CODE" "gpu_list=$GPU_LIST" \
    "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}" "extra_args=$EXTRA_ARGS"

  echo "===== Experiment E attack :: regime=$regime dataset=$dataset site=$site eps=$eps R=$rounds lc_round=$lc_round seed=$seed ====="
  echo "[experiment_e] result_jsonl=$result_jsonl"
  echo "[experiment_e] selected_steering_bank=$LC_STEERING_BANK_EFFECTIVE"
  printf '[experiment_e] command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}" 2>&1 | tee "$run_log"
}

run_profile_clean_config() {
  local regime="$1"
  local dataset="$2"
  local rounds="$3"
  local seed="$4"
  local run_dir result_jsonl trace_path command_path manifest_path run_log
  run_dir="$OUT_ROOT/role_profile/$regime/clean/$dataset/R${rounds}/seed${seed}"
  result_jsonl="$run_dir/result.jsonl"
  trace_path="$run_dir/role_trace.pt"
  command_path="$run_dir/command.txt"
  manifest_path="$run_dir/manifest.json"
  run_log="$run_dir/run.log"
  mkdir -p "$run_dir"

  cmd=(
    "$PYTHON_BIN" RecursiveMAS/run.py
    --style "$STYLE"
    --dataset "$dataset"
    --method "$METHOD"
    --num_recursive_rounds "$rounds"
    --num_samples "$NUM_SAMPLES"
    --batch_size "$BATCH_SIZE"
    --latent_length "$LATENT_LENGTH"
    --seed "$seed"
    --trust_remote_code "$TRUST_REMOTE_CODE"
    --deterministic 1
    --role_response_regime "$regime"
    --role_response_regime_path "$ROLE_RESPONSE_REGIME_PATH"
    --lc_mode none
    --lc_direction random
    --result_jsonl "$result_jsonl"
    --role_profile_trace_path "$trace_path"
    --role_profile_trace_dtype "$ROLE_TRACE_DTYPE"
    --role_profile_trace_messages 1
    --role_profile_trace_states 1
    --role_profile_trace_terminal 1
    --role_profile_probe_mode none
    --role_profile_probe_target none
    --role_profile_probe_site none
    --role_profile_probe_round -1
    --role_profile_epsilon 0
    --role_profile_seed "$seed"
    --role_profile_direction "$ROLE_PROFILE_DIRECTION"
  )
  append_common_optional_args
  write_command_txt "$command_path" "${cmd[@]}"
  write_manifest_json "$manifest_path" \
    "stage=profile_clean" "style=$STYLE" "method=$METHOD" "dataset=$dataset" \
    "R=$rounds" "seed=$seed" "role_response_regime=$regime" \
    "role_response_regime_path=$ROLE_RESPONSE_REGIME_PATH" "result_jsonl=$result_jsonl" \
    "trace_path=$trace_path" "run_log=$run_log" "role_trace_dtype=$ROLE_TRACE_DTYPE" \
    "num_samples=$NUM_SAMPLES" "batch_size=$BATCH_SIZE" "latent_length=$LATENT_LENGTH" \
    "trust_remote_code=$TRUST_REMOTE_CODE" "gpu_list=$GPU_LIST" \
    "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}" "extra_args=$EXTRA_ARGS"

  echo "===== Experiment E profile clean :: regime=$regime dataset=$dataset R=$rounds seed=$seed ====="
  printf '[experiment_e] command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}" 2>&1 | tee "$run_log"
}

run_profile_probe_config() {
  local regime="$1"
  local dataset="$2"
  local rounds="$3"
  local seed="$4"
  local probe_target="$5"
  local probe_site="$6"
  local probe_round="$7"
  local eps="$8"
  local eps_token run_dir result_jsonl trace_path command_path manifest_path run_log
  eps_token="$(sanitize_token "$eps")"
  run_dir="$OUT_ROOT/role_profile/$regime/probes/$dataset/R${rounds}/seed${seed}/${probe_target}/${probe_site}/round${probe_round}/eps${eps_token}"
  result_jsonl="$run_dir/result.jsonl"
  trace_path="$run_dir/role_trace.pt"
  command_path="$run_dir/command.txt"
  manifest_path="$run_dir/manifest.json"
  run_log="$run_dir/run.log"
  mkdir -p "$run_dir"

  cmd=(
    "$PYTHON_BIN" RecursiveMAS/run.py
    --style "$STYLE"
    --dataset "$dataset"
    --method "$METHOD"
    --num_recursive_rounds "$rounds"
    --num_samples "$NUM_SAMPLES"
    --batch_size "$BATCH_SIZE"
    --latent_length "$LATENT_LENGTH"
    --seed "$seed"
    --trust_remote_code "$TRUST_REMOTE_CODE"
    --deterministic 1
    --role_response_regime "$regime"
    --role_response_regime_path "$ROLE_RESPONSE_REGIME_PATH"
    --lc_mode none
    --lc_direction random
    --result_jsonl "$result_jsonl"
    --role_profile_trace_path "$trace_path"
    --role_profile_trace_dtype "$ROLE_TRACE_DTYPE"
    --role_profile_trace_messages 1
    --role_profile_trace_states 1
    --role_profile_trace_terminal 1
    --role_profile_probe_mode one_shot
    --role_profile_probe_target "$probe_target"
    --role_profile_probe_site "$probe_site"
    --role_profile_probe_round "$probe_round"
    --role_profile_epsilon "$eps"
    --role_profile_seed "$seed"
    --role_profile_direction "$ROLE_PROFILE_DIRECTION"
  )
  append_common_optional_args
  write_command_txt "$command_path" "${cmd[@]}"
  write_manifest_json "$manifest_path" \
    "stage=profile_probe" "style=$STYLE" "method=$METHOD" "dataset=$dataset" \
    "R=$rounds" "seed=$seed" "probe_target=$probe_target" "probe_site=$probe_site" \
    "probe_round=$probe_round" "epsilon=$eps" "role_response_regime=$regime" \
    "role_response_regime_path=$ROLE_RESPONSE_REGIME_PATH" "result_jsonl=$result_jsonl" \
    "trace_path=$trace_path" "run_log=$run_log" "role_trace_dtype=$ROLE_TRACE_DTYPE" \
    "role_probe_round_mode=$ROLE_PROBE_ROUND_MODE" "num_samples=$NUM_SAMPLES" \
    "batch_size=$BATCH_SIZE" "latent_length=$LATENT_LENGTH" \
    "trust_remote_code=$TRUST_REMOTE_CODE" "gpu_list=$GPU_LIST" \
    "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}" "extra_args=$EXTRA_ARGS"

  echo "===== Experiment E profile probe :: regime=$regime dataset=$dataset R=$rounds seed=$seed target=$probe_target site=$probe_site round=$probe_round eps=$eps ====="
  printf '[experiment_e] command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}" 2>&1 | tee "$run_log"
}

run_profile_estimate_config() {
  local regime="$1"
  local dataset="$2"
  local rounds="$3"
  local seed="$4"
  local clean_dir clean_jsonl clean_trace probe_root summary_dir command_path manifest_path run_log eps_csv quantiles_csv
  clean_dir="$OUT_ROOT/role_profile/$regime/clean/$dataset/R${rounds}/seed${seed}"
  clean_jsonl="$clean_dir/result.jsonl"
  clean_trace="$clean_dir/role_trace.pt"
  probe_root="$OUT_ROOT/role_profile/$regime/probes/$dataset/R${rounds}/seed${seed}"
  summary_dir="$OUT_ROOT/role_profile/$regime/summaries/$dataset/R${rounds}/seed${seed}"
  command_path="$summary_dir/command.txt"
  manifest_path="$summary_dir/manifest.json"
  run_log="$summary_dir/run.log"
  mkdir -p "$summary_dir"
  eps_csv="$(join_csv_words $ROLE_EPSILONS)"
  quantiles_csv="$(join_csv_words $ROLE_PROFILE_QUANTILES)"
  if [[ ! -f "$clean_jsonl" ]]; then
    die "Missing clean JSONL for estimation: $clean_jsonl"
  fi
  if [[ ! -f "$clean_trace" ]]; then
    die "Missing clean role trace for estimation: $clean_trace"
  fi

  cmd=(
    "$PYTHON_BIN" experiments/latent_contagion/estimate_role_response_profile.py
    --clean_jsonl "$clean_jsonl"
    --clean_trace "$clean_trace"
    --probe_root "$probe_root"
    --out_dir "$summary_dir"
    --dataset "$dataset"
    --role_response_regime "$regime"
    --role_response_regime_path "$ROLE_RESPONSE_REGIME_PATH"
    --rounds "$rounds"
    --epsilons "$eps_csv"
    --quantiles "$quantiles_csv"
    --clean_correct_only "$CLEAN_CORRECT_ONLY"
    --lambda_grid_from_experiment_c "$LAMBDA_GRID_FROM_EXPERIMENT_C"
    --tau_proxy "$TAU_PROXY"
    --lambda_stabilizer "$LAMBDA_STABILIZER"
    --lambda_mode "$LAMBDA_MODE"
    --lambda_missing_gain_policy "$LAMBDA_MISSING_GAIN_POLICY"
    --lambda_q_source "$LAMBDA_Q_SOURCE"
    --allow_recomputed_input_delta "$ALLOW_RECOMPUTED_INPUT_DELTA"
  )
  if [[ -n "$ESTIMATE_EXTRA_ARGS" ]]; then
    read -r -a estimate_extra_args_array <<< "$ESTIMATE_EXTRA_ARGS"
    cmd+=("${estimate_extra_args_array[@]}")
  fi
  write_command_txt "$command_path" "${cmd[@]}"
  write_manifest_json "$manifest_path" \
    "stage=profile_estimate" "dataset=$dataset" "R=$rounds" "seed=$seed" \
    "role_response_regime=$regime" "role_response_regime_path=$ROLE_RESPONSE_REGIME_PATH" \
    "clean_jsonl=$clean_jsonl" "clean_trace=$clean_trace" "probe_root=$probe_root" \
    "summary_dir=$summary_dir" "role_epsilons=$ROLE_EPSILONS" \
    "role_profile_quantiles=$ROLE_PROFILE_QUANTILES" "clean_correct_only=$CLEAN_CORRECT_ONLY" \
    "tau_proxy=$TAU_PROXY" "lambda_stabilizer=$LAMBDA_STABILIZER" \
    "lambda_mode=$LAMBDA_MODE" "lambda_missing_gain_policy=$LAMBDA_MISSING_GAIN_POLICY" \
    "lambda_q_source=$LAMBDA_Q_SOURCE" "allow_recomputed_input_delta=$ALLOW_RECOMPUTED_INPUT_DELTA" \
    "lambda_grid_from_experiment_c=$LAMBDA_GRID_FROM_EXPERIMENT_C" \
    "role_probe_round_mode=$ROLE_PROBE_ROUND_MODE" "estimate_extra_args=$ESTIMATE_EXTRA_ARGS" \
    "run_log=$run_log"

  echo "===== Experiment E profile estimate :: regime=$regime dataset=$dataset R=$rounds seed=$seed ====="
  printf '[experiment_e] command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}" 2>&1 | tee "$run_log"
}

run_aggregate_attack() {
  local dataset aggregate_dir command_path manifest_path run_log
  for dataset in $DATASETS; do
    aggregate_dir="$OUT_ROOT/aggregate/attacks/$dataset"
    command_path="$aggregate_dir/command.txt"
    manifest_path="$aggregate_dir/manifest.json"
    run_log="$aggregate_dir/run.log"
    mkdir -p "$aggregate_dir"
    cmd=(
      "$PYTHON_BIN" experiments/latent_contagion/aggregate_latent_contagion.py
      --root "$OUT_ROOT/attacks"
      --dataset "$dataset"
      --subdir "$ATTACK_RUN_SUBDIR"
      --out_dir "$aggregate_dir"
      --label "experiment_e_${ATTACK_RUN_SUBDIR}"
      --make_plots false
    )
    write_command_txt "$command_path" "${cmd[@]}"
    write_manifest_json "$manifest_path" \
      "stage=aggregate_attack" "dataset=$dataset" "out_dir=$aggregate_dir" \
      "attack_run_subdir=$ATTACK_RUN_SUBDIR" "role_response_regimes=$ROLE_RESPONSE_REGIMES" \
      "run_log=$run_log"
    echo "===== Experiment E aggregate attack :: dataset=$dataset ====="
    printf '[experiment_e] command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}" 2>&1 | tee "$run_log"
  done
}

run_compare() {
  local dataset comparison_dir command_path manifest_path run_log profile_epsilon regimes_csv
  profile_epsilon="$(first_word "$ROLE_EPSILONS")"
  regimes_csv="$(join_csv_words $ROLE_RESPONSE_REGIMES)"
  for dataset in $DATASETS; do
    comparison_dir="$OUT_ROOT/comparison/$dataset"
    command_path="$comparison_dir/command.txt"
    manifest_path="$comparison_dir/manifest.json"
    run_log="$comparison_dir/run.log"
    mkdir -p "$comparison_dir"
    cmd=(
      "$PYTHON_BIN" experiments/latent_contagion/compare_role_response_regimes.py
      --attack_aggregate_dir "$OUT_ROOT/aggregate/attacks/$dataset"
      --profile_root "$OUT_ROOT/role_profile"
      --out_dir "$comparison_dir"
      --dataset "$dataset"
      --regimes "$regimes_csv"
      --profile_epsilon "$profile_epsilon"
      --gain_quantile "$GAIN_QUANTILE"
      --attack_eps_mode "$ATTACK_EPS_MODE"
    )
    write_command_txt "$command_path" "${cmd[@]}"
    write_manifest_json "$manifest_path" \
      "stage=compare" "dataset=$dataset" "out_dir=$comparison_dir" \
      "role_response_regimes=$ROLE_RESPONSE_REGIMES" "profile_epsilon=$profile_epsilon" \
      "gain_quantile=$GAIN_QUANTILE" "attack_eps_mode=$ATTACK_EPS_MODE" "run_log=$run_log"
    echo "===== Experiment E compare :: dataset=$dataset ====="
    printf '[experiment_e] command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}" 2>&1 | tee "$run_log"
  done
}

run_all_local() {
  local regime dataset site eps rounds lc_round seed target probe_site probe_round role_eps
  for regime in $ROLE_RESPONSE_REGIMES; do
    for dataset in $DATASETS; do
      for site in $SITES; do
        for eps in $EPSILONS; do
          for rounds in $ROUNDS; do
            for lc_round in $(candidate_lc_rounds_for_config "$site" "$rounds"); do
              for seed in $SEEDS; do
                run_attack_config "$regime" "$dataset" "$site" "$eps" "$rounds" "$lc_round" "$seed"
              done
            done
          done
        done
      done
    done
  done
  for regime in $ROLE_RESPONSE_REGIMES; do
    for dataset in $DATASETS; do
      for rounds in $ROUNDS; do
        for seed in $SEEDS; do
          run_profile_clean_config "$regime" "$dataset" "$rounds" "$seed"
        done
      done
    done
  done
  for regime in $ROLE_RESPONSE_REGIMES; do
    for dataset in $DATASETS; do
      for rounds in $ROUNDS; do
        for seed in $SEEDS; do
          for target in $ROLE_PROBE_TARGETS; do
            for probe_site in $(probe_sites_for_target "$target"); do
              for probe_round in $(candidate_probe_rounds_for_config "$target" "$probe_site" "$rounds"); do
                for role_eps in $ROLE_EPSILONS; do
                  run_profile_probe_config "$regime" "$dataset" "$rounds" "$seed" "$target" "$probe_site" "$probe_round" "$role_eps"
                done
              done
            done
          done
        done
      done
    done
  done
  for regime in $ROLE_RESPONSE_REGIMES; do
    for dataset in $DATASETS; do
      for rounds in $ROUNDS; do
        for seed in $SEEDS; do
          run_profile_estimate_config "$regime" "$dataset" "$rounds" "$seed"
        done
      done
    done
  done
  run_aggregate_attack
  run_compare
}

validate_stage
validate_grid_inputs

case "$E_STAGE" in
  attack_grid)
    echo "[experiment_e] stage=$E_STAGE out_root=$OUT_ROOT"
    print_attack_grid
    exit 0
    ;;
  profile_clean_grid)
    echo "[experiment_e] stage=$E_STAGE out_root=$OUT_ROOT"
    print_profile_clean_grid
    exit 0
    ;;
  profile_probe_grid)
    echo "[experiment_e] stage=$E_STAGE out_root=$OUT_ROOT role_probe_round_mode=$ROLE_PROBE_ROUND_MODE"
    print_profile_probe_grid
    exit 0
    ;;
  profile_estimate_grid)
    echo "[experiment_e] stage=$E_STAGE out_root=$OUT_ROOT"
    print_profile_clean_grid
    exit 0
    ;;
  aggregate_attack)
    echo "[experiment_e] stage=$E_STAGE out_root=$OUT_ROOT"
    run_aggregate_attack
    exit 0
    ;;
  compare)
    echo "[experiment_e] stage=$E_STAGE out_root=$OUT_ROOT"
    run_compare
    exit 0
    ;;
esac

configure_environment

echo "Using TMPDIR=$TMPDIR"
ls -ld "$TMPDIR" || true
echo "[experiment_e] stage=$E_STAGE out_root=$OUT_ROOT"
echo "[experiment_e] style=$STYLE method=$METHOD datasets=$DATASETS rounds=$ROUNDS seeds=$SEEDS"
echo "[experiment_e] role_response_regimes=$ROLE_RESPONSE_REGIMES role_response_regime_path=${ROLE_RESPONSE_REGIME_PATH:-<empty>}"
echo "[experiment_e] num_samples=$NUM_SAMPLES batch_size=$BATCH_SIZE latent_length=$LATENT_LENGTH"
echo "[experiment_e] attack sites=$SITES epsilons=$EPSILONS lc_rounds=$LC_ROUNDS bank_mode=$ATTACK_BANK_MODE"
echo "[experiment_e] profile targets=$ROLE_PROBE_TARGETS role_epsilons=$ROLE_EPSILONS trace_dtype=$ROLE_TRACE_DTYPE"
echo "[experiment_e] role_probe_round_mode=$ROLE_PROBE_ROUND_MODE role_probe_rounds=${ROLE_PROBE_ROUNDS:-<empty>}"
echo "[experiment_e] lambda_mode=$LAMBDA_MODE missing_gain_policy=$LAMBDA_MISSING_GAIN_POLICY q_source=$LAMBDA_Q_SOURCE"
echo "[experiment_e] gpu_list=${GPU_LIST:-<empty>} task_id=$TASK_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

echo "===== nvidia-smi -L ====="
nvidia-smi -L || true
echo "===== initial nvidia-smi ====="
nvidia-smi || true

if [[ "$E_STAGE" == "all" ]]; then
  run_all_local
  echo "[experiment_e] all stages complete."
  exit 0
fi

TOTAL_TASKS=0
case "$E_STAGE" in
  attack) TOTAL_TASKS="$(total_attack_jobs)" ;;
  profile_clean) TOTAL_TASKS="$(total_profile_clean_jobs)" ;;
  profile_probe) TOTAL_TASKS="$(total_profile_probe_jobs)" ;;
  profile_estimate) TOTAL_TASKS="$(total_profile_estimate_jobs)" ;;
esac

if (( TOTAL_TASKS <= 0 )); then
  die "No valid Experiment E tasks for E_STAGE=$E_STAGE."
fi
if (( TASK_ID >= TOTAL_TASKS )); then
  die "Array index $TASK_ID is out of range for E_STAGE=$E_STAGE; total tasks: $TOTAL_TASKS"
fi

ROLE_RESPONSE_REGIME=""
DATASET=""
R=""
SEED=""
SITE=""
EPS=""
LC_ROUND_EFFECTIVE=""
PROBE_TARGET=""
PROBE_SITE=""
PROBE_ROUND=""

case "$E_STAGE" in
  attack)
    select_attack_config "$TASK_ID" || die "Failed to select attack config for task $TASK_ID"
    echo "[experiment_e] task_id=$TASK_ID/$TOTAL_TASKS selected regime=$ROLE_RESPONSE_REGIME dataset=$DATASET site=$SITE eps=$EPS R=$R lc_round=$LC_ROUND_EFFECTIVE seed=$SEED"
    run_attack_config "$ROLE_RESPONSE_REGIME" "$DATASET" "$SITE" "$EPS" "$R" "$LC_ROUND_EFFECTIVE" "$SEED"
    ;;
  profile_clean)
    select_profile_clean_config "$TASK_ID" || die "Failed to select profile_clean config for task $TASK_ID"
    echo "[experiment_e] task_id=$TASK_ID/$TOTAL_TASKS selected regime=$ROLE_RESPONSE_REGIME dataset=$DATASET R=$R seed=$SEED"
    run_profile_clean_config "$ROLE_RESPONSE_REGIME" "$DATASET" "$R" "$SEED"
    ;;
  profile_probe)
    select_profile_probe_config "$TASK_ID" || die "Failed to select profile_probe config for task $TASK_ID"
    echo "[experiment_e] task_id=$TASK_ID/$TOTAL_TASKS selected regime=$ROLE_RESPONSE_REGIME dataset=$DATASET R=$R seed=$SEED target=$PROBE_TARGET site=$PROBE_SITE round=$PROBE_ROUND eps=$EPS"
    run_profile_probe_config "$ROLE_RESPONSE_REGIME" "$DATASET" "$R" "$SEED" "$PROBE_TARGET" "$PROBE_SITE" "$PROBE_ROUND" "$EPS"
    ;;
  profile_estimate)
    select_profile_estimate_config "$TASK_ID" || die "Failed to select profile_estimate config for task $TASK_ID"
    echo "[experiment_e] task_id=$TASK_ID/$TOTAL_TASKS selected regime=$ROLE_RESPONSE_REGIME dataset=$DATASET R=$R seed=$SEED"
    run_profile_estimate_config "$ROLE_RESPONSE_REGIME" "$DATASET" "$R" "$SEED"
    ;;
esac

echo "[experiment_e] complete."
