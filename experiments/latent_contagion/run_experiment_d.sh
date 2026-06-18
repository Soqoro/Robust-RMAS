#!/bin/bash
#SBATCH --job-name=latent-contagion-d
#SBATCH -p NA100q
#SBATCH -w node01
#SBATCH --output=logs/latent_contagion_d_%A_%a.out
#SBATCH --error=logs/latent_contagion_d_%A_%a.err

# Experiment D: role response profile estimation.
#
# Clean traces:
#   D_STAGE=clean sbatch --array=0-N experiments/latent_contagion/run_experiment_d.sh
#
# Probe traces:
#   D_STAGE=probe sbatch --array=0-N experiments/latent_contagion/run_experiment_d.sh
#
# Estimate:
#   D_STAGE=estimate bash experiments/latent_contagion/run_experiment_d.sh
#
# Smoke test:
#   D_STAGE=all NUM_SAMPLES=2 BATCH_SIZE=1 ROUNDS="1" ROLE_EPSILONS="1e-3" \
#     bash experiments/latent_contagion/run_experiment_d.sh

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
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
OUT_ROOT="${OUT_ROOT:-outputs/latent_contagion/experiment_d}"
D_STAGE="${D_STAGE:-clean}"

ROLE_EPSILONS="${ROLE_EPSILONS:-1e-4 3e-4 1e-3}"
ROLE_PROBE_TARGETS="${ROLE_PROBE_TARGETS:-message state terminal}"
ROLE_PROBE_SITES_MESSAGE="${ROLE_PROBE_SITES_MESSAGE:-p2c c2s s2p}"
ROLE_PROBE_SITES_STATE="${ROLE_PROBE_SITES_STATE:-planner_self critic_self solver_self}"
ROLE_PROBE_SITES_TERMINAL="${ROLE_PROBE_SITES_TERMINAL:-final_c2s}"
ROLE_TRACE_DTYPE="${ROLE_TRACE_DTYPE:-float32}"
ROLE_PROFILE_DIRECTION="${ROLE_PROFILE_DIRECTION:-random}"
ROLE_PROFILE_QUANTILES="${ROLE_PROFILE_QUANTILES:-0.5 0.75 0.9 0.95}"
CLEAN_CORRECT_ONLY="${CLEAN_CORRECT_ONLY:-1}"
TAU_PROXY="${TAU_PROXY:-clean_clean_floor}"
LAMBDA_STABILIZER="${LAMBDA_STABILIZER:-1e-8}"
LAMBDA_GRID_FROM_EXPERIMENT_C="${LAMBDA_GRID_FROM_EXPERIMENT_C:-}"

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
  case "$D_STAGE" in
    clean|probe|estimate|all) ;;
    *) die "D_STAGE must be one of: clean, probe, estimate, all. Got: $D_STAGE" ;;
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

validate_grid_inputs() {
  local rounds target site eps seed
  for rounds in $ROUNDS; do
    validate_positive_int "ROUNDS item" "$rounds"
  done
  for seed in $SEEDS; do
    validate_nonnegative_int "SEEDS item" "$seed"
  done
  for eps in $ROLE_EPSILONS; do
    if ! "$PYTHON_BIN" - "$eps" <<'PY' >/dev/null; then
import math
import sys
value = float(sys.argv[1])
if not math.isfinite(value) or value < 0:
    raise SystemExit(2)
PY
      die "ROLE_EPSILONS items must be finite non-negative floats. Got: $eps"
    fi
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
}

probe_sites_for_target() {
  case "$1" in
    message) echo "$ROLE_PROBE_SITES_MESSAGE" ;;
    state) echo "$ROLE_PROBE_SITES_STATE" ;;
    terminal) echo "$ROLE_PROBE_SITES_TERMINAL" ;;
    *) die "Unknown probe target: $1" ;;
  esac
}

candidate_probe_rounds_for_config() {
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

sanitize_token() {
  echo "$1" | sed 's/[^A-Za-z0-9._+-]/_/g'
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

clean_dir_for() {
  echo "$OUT_ROOT/clean/$1/R$2/seed$3"
}

total_clean_jobs() {
  local total=0
  local dataset rounds seed
  for dataset in $DATASETS; do
    for rounds in $ROUNDS; do
      for seed in $SEEDS; do
        total=$((total + 1))
      done
    done
  done
  echo "$total"
}

select_clean_config() {
  local target="$1"
  local index=0
  local dataset rounds seed
  for dataset in $DATASETS; do
    for rounds in $ROUNDS; do
      for seed in $SEEDS; do
        if (( index == target )); then
          DATASET="$dataset"
          R="$rounds"
          SEED="$seed"
          return 0
        fi
        index=$((index + 1))
      done
    done
  done
  return 1
}

total_probe_jobs() {
  local total=0
  local dataset rounds seed target site probe_round eps
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
  echo "$total"
}

select_probe_config() {
  local target_index="$1"
  local index=0
  local dataset rounds seed target site probe_round eps
  for dataset in $DATASETS; do
    for rounds in $ROUNDS; do
      for seed in $SEEDS; do
        for target in $ROLE_PROBE_TARGETS; do
          for site in $(probe_sites_for_target "$target"); do
            for probe_round in $(candidate_probe_rounds_for_config "$target" "$site" "$rounds"); do
              for eps in $ROLE_EPSILONS; do
                if (( index == target_index )); then
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
  return 1
}

total_estimate_jobs() {
  total_clean_jobs
}

select_estimate_config() {
  select_clean_config "$1"
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

run_clean_config() {
  local dataset="$1"
  local rounds="$2"
  local seed="$3"
  local run_dir result_jsonl trace_path run_log command_path manifest_path
  run_dir="$(clean_dir_for "$dataset" "$rounds" "$seed")"
  result_jsonl="$run_dir/result.jsonl"
  trace_path="$run_dir/role_trace.pt"
  run_log="$run_dir/run.log"
  command_path="$run_dir/command.txt"
  manifest_path="$run_dir/manifest.json"
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
    "stage=clean" "style=$STYLE" "method=$METHOD" "dataset=$dataset" "R=$rounds" \
    "seed=$seed" "num_samples=$NUM_SAMPLES" "batch_size=$BATCH_SIZE" \
    "latent_length=$LATENT_LENGTH" "model_name_or_path=$MODEL_NAME_OR_PATH" \
    "result_jsonl=$result_jsonl" "trace_path=$trace_path" "run_log=$run_log" \
    "role_trace_dtype=$ROLE_TRACE_DTYPE" "gpu_list=$GPU_LIST" \
    "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}" "extra_args=$EXTRA_ARGS"

  echo "===== Experiment D clean :: dataset=$dataset R=$rounds seed=$seed ====="
  echo "[experiment_d] result_jsonl=$result_jsonl"
  echo "[experiment_d] trace_path=$trace_path"
  printf '[experiment_d] command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}" 2>&1 | tee "$run_log"
}

run_probe_config() {
  local dataset="$1"
  local rounds="$2"
  local seed="$3"
  local probe_target="$4"
  local probe_site="$5"
  local probe_round="$6"
  local eps="$7"
  local eps_token run_dir result_jsonl trace_path run_log command_path manifest_path
  eps_token="$(sanitize_token "$eps")"
  run_dir="$OUT_ROOT/probes/$dataset/R${rounds}/seed${seed}/${probe_target}/${probe_site}/round${probe_round}/eps${eps_token}"
  result_jsonl="$run_dir/result.jsonl"
  trace_path="$run_dir/role_trace.pt"
  run_log="$run_dir/run.log"
  command_path="$run_dir/command.txt"
  manifest_path="$run_dir/manifest.json"
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
    "stage=probe" "style=$STYLE" "method=$METHOD" "dataset=$dataset" "R=$rounds" \
    "seed=$seed" "probe_target=$probe_target" "probe_site=$probe_site" \
    "probe_round=$probe_round" "epsilon=$eps" "num_samples=$NUM_SAMPLES" \
    "batch_size=$BATCH_SIZE" "latent_length=$LATENT_LENGTH" \
    "model_name_or_path=$MODEL_NAME_OR_PATH" "result_jsonl=$result_jsonl" \
    "trace_path=$trace_path" "run_log=$run_log" "role_trace_dtype=$ROLE_TRACE_DTYPE" \
    "gpu_list=$GPU_LIST" "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}" \
    "extra_args=$EXTRA_ARGS"

  echo "===== Experiment D probe :: dataset=$dataset R=$rounds seed=$seed target=$probe_target site=$probe_site round=$probe_round eps=$eps ====="
  echo "[experiment_d] result_jsonl=$result_jsonl"
  echo "[experiment_d] trace_path=$trace_path"
  printf '[experiment_d] command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}" 2>&1 | tee "$run_log"
}

run_estimate_config() {
  local dataset="$1"
  local rounds="$2"
  local seed="$3"
  local clean_dir clean_jsonl clean_trace probe_root summary_dir run_log command_path manifest_path eps_csv quantiles_csv
  clean_dir="$(clean_dir_for "$dataset" "$rounds" "$seed")"
  clean_jsonl="$clean_dir/result.jsonl"
  clean_trace="$clean_dir/role_trace.pt"
  probe_root="$OUT_ROOT/probes/$dataset/R${rounds}/seed${seed}"
  summary_dir="$OUT_ROOT/summaries/$dataset/R${rounds}/seed${seed}"
  run_log="$summary_dir/run.log"
  command_path="$summary_dir/command.txt"
  manifest_path="$summary_dir/manifest.json"
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
    --rounds "$rounds"
    --epsilons "$eps_csv"
    --quantiles "$quantiles_csv"
    --clean_correct_only "$CLEAN_CORRECT_ONLY"
    --lambda_grid_from_experiment_c "$LAMBDA_GRID_FROM_EXPERIMENT_C"
    --tau_proxy "$TAU_PROXY"
    --lambda_stabilizer "$LAMBDA_STABILIZER"
  )
  if [[ -n "$ESTIMATE_EXTRA_ARGS" ]]; then
    read -r -a estimate_extra_args_array <<< "$ESTIMATE_EXTRA_ARGS"
    cmd+=("${estimate_extra_args_array[@]}")
  fi
  write_command_txt "$command_path" "${cmd[@]}"
  write_manifest_json "$manifest_path" \
    "stage=estimate" "dataset=$dataset" "R=$rounds" "seed=$seed" \
    "clean_jsonl=$clean_jsonl" "clean_trace=$clean_trace" "probe_root=$probe_root" \
    "summary_dir=$summary_dir" "role_epsilons=$ROLE_EPSILONS" \
    "role_profile_quantiles=$ROLE_PROFILE_QUANTILES" "clean_correct_only=$CLEAN_CORRECT_ONLY" \
    "tau_proxy=$TAU_PROXY" "lambda_stabilizer=$LAMBDA_STABILIZER" \
    "lambda_grid_from_experiment_c=$LAMBDA_GRID_FROM_EXPERIMENT_C" \
    "estimate_extra_args=$ESTIMATE_EXTRA_ARGS"

  echo "===== Experiment D estimate :: dataset=$dataset R=$rounds seed=$seed ====="
  echo "[experiment_d] summary_dir=$summary_dir"
  printf '[experiment_d] command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}" 2>&1 | tee "$run_log"
}

run_all_local() {
  local dataset rounds seed target site probe_round eps
  for dataset in $DATASETS; do
    for rounds in $ROUNDS; do
      for seed in $SEEDS; do
        run_clean_config "$dataset" "$rounds" "$seed"
      done
    done
  done
  for dataset in $DATASETS; do
    for rounds in $ROUNDS; do
      for seed in $SEEDS; do
        for target in $ROLE_PROBE_TARGETS; do
          for site in $(probe_sites_for_target "$target"); do
            for probe_round in $(candidate_probe_rounds_for_config "$target" "$site" "$rounds"); do
              for eps in $ROLE_EPSILONS; do
                run_probe_config "$dataset" "$rounds" "$seed" "$target" "$site" "$probe_round" "$eps"
              done
            done
          done
        done
      done
    done
  done
  for dataset in $DATASETS; do
    for rounds in $ROUNDS; do
      for seed in $SEEDS; do
        run_estimate_config "$dataset" "$rounds" "$seed"
      done
    done
  done
}

validate_stage
validate_grid_inputs
configure_environment

echo "Using TMPDIR=$TMPDIR"
ls -ld "$TMPDIR" || true
echo "[experiment_d] stage=$D_STAGE out_root=$OUT_ROOT"
echo "[experiment_d] style=$STYLE method=$METHOD datasets=$DATASETS rounds=$ROUNDS seeds=$SEEDS"
echo "[experiment_d] num_samples=$NUM_SAMPLES batch_size=$BATCH_SIZE latent_length=$LATENT_LENGTH"
echo "[experiment_d] targets=$ROLE_PROBE_TARGETS epsilons=$ROLE_EPSILONS trace_dtype=$ROLE_TRACE_DTYPE"
echo "[experiment_d] gpu_list=${GPU_LIST:-<empty>} task_id=$TASK_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

if [[ "$D_STAGE" != "estimate" ]]; then
  echo "===== nvidia-smi -L ====="
  nvidia-smi -L || true
  echo "===== initial nvidia-smi ====="
  nvidia-smi || true
fi

if [[ "$D_STAGE" == "all" ]]; then
  run_all_local
  echo "[experiment_d] all stages complete."
  exit 0
fi

TOTAL_TASKS=0
case "$D_STAGE" in
  clean)
    TOTAL_TASKS="$(total_clean_jobs)"
    ;;
  probe)
    TOTAL_TASKS="$(total_probe_jobs)"
    ;;
  estimate)
    TOTAL_TASKS="$(total_estimate_jobs)"
    ;;
esac

if (( TOTAL_TASKS <= 0 )); then
  die "No valid Experiment D tasks for D_STAGE=$D_STAGE. Check rounds/sites; s2p is inactive for R=1."
fi
if (( TASK_ID >= TOTAL_TASKS )); then
  die "Array index $TASK_ID is out of range for D_STAGE=$D_STAGE; total tasks: $TOTAL_TASKS"
fi

DATASET=""
R=""
SEED=""
PROBE_TARGET=""
PROBE_SITE=""
PROBE_ROUND=""
EPS=""

case "$D_STAGE" in
  clean)
    select_clean_config "$TASK_ID" || die "Failed to select clean config for task $TASK_ID"
    echo "[experiment_d] task_id=$TASK_ID/$TOTAL_TASKS selected dataset=$DATASET R=$R seed=$SEED"
    run_clean_config "$DATASET" "$R" "$SEED"
    ;;
  probe)
    select_probe_config "$TASK_ID" || die "Failed to select probe config for task $TASK_ID"
    echo "[experiment_d] task_id=$TASK_ID/$TOTAL_TASKS selected dataset=$DATASET R=$R seed=$SEED target=$PROBE_TARGET site=$PROBE_SITE round=$PROBE_ROUND eps=$EPS"
    run_probe_config "$DATASET" "$R" "$SEED" "$PROBE_TARGET" "$PROBE_SITE" "$PROBE_ROUND" "$EPS"
    ;;
  estimate)
    select_estimate_config "$TASK_ID" || die "Failed to select estimate config for task $TASK_ID"
    echo "[experiment_d] task_id=$TASK_ID/$TOTAL_TASKS selected dataset=$DATASET R=$R seed=$SEED"
    run_estimate_config "$DATASET" "$R" "$SEED"
    ;;
esac

echo "[experiment_d] complete."
