#!/bin/bash
#SBATCH --job-name=latent-contagion-c
#SBATCH -p NA100q
#SBATCH -w node01
#SBATCH --output=logs/latent_contagion_c_%A_%a.out
#SBATCH --error=logs/latent_contagion_c_%A_%a.err

# Experiment C: DiffMean bank-direction one-shot latent steering phase diagram.
#
# Total jobs = valid combinations of
#   DATASETS x SITES x EPSILONS x ROUNDS x LC_ROUNDS x SEEDS.
# LC_ROUNDS are zero-based latent injection/calibration round indices. For
# recursive depth R=5, p2c/c2s rounds are 0..4 and s2p rounds are 0..3.
# Local smoke:
#   SLURM_ARRAY_TASK_ID=0 NUM_SAMPLES=2 bash experiments/latent_contagion/run_experiment_c.sh
# Slurm:
#   sbatch --array=0-$((TOTAL_TASKS - 1)) experiments/latent_contagion/run_experiment_c.sh

set -euo pipefail

mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-python}"
STYLE="${STYLE:-sequential_light}"
METHOD="${METHOD:-ours_recursive}"
DATASETS="${DATASETS:-math500}"
SITES="${SITES:-p2c c2s s2p}"
EPSILONS="${EPSILONS:-0 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1}"
ROUNDS="${ROUNDS:-1 2 3 4 5}"
SEEDS="${SEEDS:-42}"
LC_MODE="${LC_MODE:-one_shot}"
LC_ROUNDS="${LC_ROUNDS:-${LC_ROUND:-0}}"
LC_DIRECTION="${LC_DIRECTION:-bank}"
LC_STEERING_METHOD="${LC_STEERING_METHOD:-diffmean}"
CALIB_ROOT="${CALIB_ROOT:-outputs/latent_contagion/diffmean_calibration}"
STEERING_FILTER="${STEERING_FILTER:-clean_correct_attack_wrong}"
LC_STEERING_ID="${LC_STEERING_ID:-}"
LC_STEERING_BANK="${LC_STEERING_BANK:-}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LATENT_LENGTH="${LATENT_LENGTH:-48}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
GPU_LIST="${GPU_LIST:-}"
OUT_ROOT="${OUT_ROOT:-${OUT_DIR:-outputs/latent_contagion/experiment_c}}"
RUN_SUBDIR="${RUN_SUBDIR:-diffmean_${STEERING_FILTER}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]]; then
  echo "[error] SLURM_ARRAY_TASK_ID must be a non-negative integer, got: $TASK_ID" >&2
  exit 2
fi

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

valid_lc_round_for_site_R() {
  local site="$1"
  local rounds="$2"
  local lc_round="$3"
  validate_positive_int "ROUNDS item" "$rounds"
  validate_nonnegative_int "LC_ROUNDS item" "$lc_round"
  case "$site" in
    p2c|c2s)
      (( lc_round < rounds ))
      ;;
    s2p)
      # s2p is the feedback edge and is inactive in the final zero-based round.
      (( rounds > 1 && lc_round < rounds - 1 ))
      ;;
    *)
      die "SITES item must be one of p2c, c2s, s2p. Got: $site"
      ;;
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
  if [[ -z "$normalized" ]]; then
    die "LC_ROUNDS must be 'all' or a list of zero-based non-negative integers."
  fi
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

count_skipped_invalid_configs() {
  local skipped=0
  local dataset site eps rounds seed lc_round normalized
  if [[ "$LC_ROUNDS" == "all" ]]; then
    for dataset in $DATASETS; do
      for site in $SITES; do
        case "$site" in
          p2c|c2s|s2p) ;;
          *) die "SITES item must be one of p2c, c2s, s2p. Got: $site" ;;
        esac
        for eps in $EPSILONS; do
          for rounds in $ROUNDS; do
            validate_positive_int "ROUNDS item" "$rounds"
            if [[ "$site" == "s2p" ]] && (( rounds <= 1 )); then
              for seed in $SEEDS; do
                skipped=$((skipped + 1))
              done
            fi
          done
        done
      done
    done
    echo "$skipped"
    return 0
  fi

  normalized="${LC_ROUNDS//,/ }"
  for dataset in $DATASETS; do
    for site in $SITES; do
      for eps in $EPSILONS; do
        for rounds in $ROUNDS; do
          validate_positive_int "ROUNDS item" "$rounds"
          for lc_round in $normalized; do
            validate_nonnegative_int "LC_ROUNDS item" "$lc_round"
            if ! valid_lc_round_for_site_R "$site" "$rounds" "$lc_round"; then
              for seed in $SEEDS; do
                skipped=$((skipped + 1))
              done
            fi
          done
        done
      done
    done
  done
  echo "$skipped"
}

validate_grid_inputs() {
  local site rounds lc_round normalized
  for site in $SITES; do
    case "$site" in
      p2c|c2s|s2p) ;;
      *) die "SITES item must be one of p2c, c2s, s2p. Got: $site" ;;
    esac
  done
  for rounds in $ROUNDS; do
    validate_positive_int "ROUNDS item" "$rounds"
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
}

validate_grid_inputs

if [[ -n "$GPU_LIST" ]]; then
  read -r -a gpu_array <<< "$GPU_LIST"
  if (( ${#gpu_array[@]} == 0 )); then
    echo "[error] GPU_LIST was set but no GPU ids were parsed." >&2
    exit 2
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

echo "Using TMPDIR=$TMPDIR"
mkdir -p "$TMPDIR" || true
ls -ld "$TMPDIR" || true

total_jobs() {
  local total=0
  local dataset site eps rounds lc_round seed
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
  echo "$total"
}

select_config() {
  local target="$1"
  local index=0
  local dataset site eps rounds lc_round seed
  for dataset in $DATASETS; do
    for site in $SITES; do
      for eps in $EPSILONS; do
        for rounds in $ROUNDS; do
          for lc_round in $(candidate_lc_rounds_for_config "$site" "$rounds"); do
            for seed in $SEEDS; do
              if (( index == target )); then
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
  return 1
}

TOTAL_TASKS="$(total_jobs)"
SKIPPED_INVALID_CONFIGS="$(count_skipped_invalid_configs)"
if (( TOTAL_TASKS <= 0 )); then
  echo "[error] no valid experiment_c tasks after applying LC_ROUNDS=$LC_ROUNDS." >&2
  echo "[error] zero-based convention: p2c/c2s use 0..R-1; s2p uses 0..R-2 and is inactive for R=1." >&2
  exit 2
fi
if (( TASK_ID >= TOTAL_TASKS )); then
  echo "[error] array index $TASK_ID is out of range." >&2
  echo "[error] total number of tasks: $TOTAL_TASKS" >&2
  exit 2
fi

DATASET=""
SITE=""
EPS=""
R=""
LC_ROUND_EFFECTIVE=""
SEED=""
select_config "$TASK_ID"

LC_STEERING_ID_EFFECTIVE="${LC_STEERING_ID:-diffmean_R${R}_${DATASET}_role_aligned_${STEERING_FILTER}}"
LC_STEERING_BANK_EFFECTIVE="${LC_STEERING_BANK:-$CALIB_ROOT/${DATASET}_R${R}/${LC_STEERING_ID_EFFECTIVE}.pt}"

if [[ "$LC_DIRECTION" == "bank" && ! -f "$LC_STEERING_BANK_EFFECTIVE" ]]; then
  echo "[error] LC_STEERING_BANK_EFFECTIVE does not exist: $LC_STEERING_BANK_EFFECTIVE" >&2
  echo "[error] Run experiments/latent_contagion/run_experiment_c_calibration.sh for dataset=$DATASET R=$R, or set LC_STEERING_BANK." >&2
  exit 2
fi

RUN_DIR="$OUT_ROOT/$DATASET/$RUN_SUBDIR"
LOG_DIR="$RUN_DIR/logs"
RESULT_JSONL="$RUN_DIR/site=${SITE}_eps=${EPS}_R=${R}_lc_round=${LC_ROUND_EFFECTIVE}_seed=${SEED}.jsonl"
RUN_LOG="$LOG_DIR/site=${SITE}_eps=${EPS}_R=${R}_lc_round=${LC_ROUND_EFFECTIVE}_seed=${SEED}.log"

mkdir -p "$RUN_DIR" "$LOG_DIR"

echo "[experiment_c] out_root=$OUT_ROOT"
echo "[experiment_c] task_id=$TASK_ID/$TOTAL_TASKS"
echo "[experiment_c] style=$STYLE method=$METHOD"
echo "[experiment_c] datasets=$DATASETS"
echo "[experiment_c] sites=$SITES"
echo "[experiment_c] epsilons=$EPSILONS"
echo "[experiment_c] rounds=$ROUNDS"
echo "[experiment_c] lc_rounds=$LC_ROUNDS (zero-based; p2c/c2s=0..R-1, s2p=0..R-2)"
echo "[experiment_c] seeds=$SEEDS"
echo "[experiment_c] skipped_invalid_lc_configs=$SKIPPED_INVALID_CONFIGS"
echo "[experiment_c] lc_mode=$LC_MODE lc_round=$LC_ROUND_EFFECTIVE lc_direction=$LC_DIRECTION run_subdir=$RUN_SUBDIR"
echo "[experiment_c] selected_steering_bank=$LC_STEERING_BANK_EFFECTIVE"
echo "[experiment_c] lc_steering_method=$LC_STEERING_METHOD selected_steering_id=$LC_STEERING_ID_EFFECTIVE"
echo "[experiment_c] selected dataset=$DATASET site=$SITE eps=$EPS rounds=$R lc_round=$LC_ROUND_EFFECTIVE seed=$SEED"
echo "[experiment_c] num_samples=$NUM_SAMPLES batch_size=$BATCH_SIZE latent_length=$LATENT_LENGTH"
echo "[experiment_c] gpu_list=${GPU_LIST:-<empty>}"
echo "[experiment_c] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

echo "===== nvidia-smi -L ====="
nvidia-smi -L || true
echo "===== initial nvidia-smi ====="
nvidia-smi || true

{
  echo "style=$STYLE"
  echo "method=$METHOD"
  echo "datasets=$DATASETS"
  echo "sites=$SITES"
  echo "epsilons=$EPSILONS"
  echo "rounds=$ROUNDS"
  echo "lc_rounds=$LC_ROUNDS"
  echo "lc_round_index_base=0"
  echo "seeds=$SEEDS"
  echo "lc_mode=$LC_MODE"
  echo "lc_round=$LC_ROUND_EFFECTIVE"
  echo "lc_direction=$LC_DIRECTION"
  echo "steering_filter=$STEERING_FILTER"
  echo "lc_steering_bank_override=$LC_STEERING_BANK"
  echo "lc_steering_bank=$LC_STEERING_BANK_EFFECTIVE"
  echo "lc_steering_method=$LC_STEERING_METHOD"
  echo "lc_steering_id_override=$LC_STEERING_ID"
  echo "lc_steering_id=$LC_STEERING_ID_EFFECTIVE"
  echo "run_subdir=$RUN_SUBDIR"
  echo "task_id=$TASK_ID"
  echo "total_tasks=$TOTAL_TASKS"
  echo "skipped_invalid_lc_configs=$SKIPPED_INVALID_CONFIGS"
  echo "selected_dataset=$DATASET"
  echo "selected_site=$SITE"
  echo "selected_epsilon=$EPS"
  echo "selected_rounds=$R"
  echo "selected_recursive_R=$R"
  echo "selected_lc_round=$LC_ROUND_EFFECTIVE"
  echo "lc_round_effective=$LC_ROUND_EFFECTIVE"
  echo "selected_steering_bank=$LC_STEERING_BANK_EFFECTIVE"
  echo "selected_steering_id=$LC_STEERING_ID_EFFECTIVE"
  echo "selected_seed=$SEED"
  echo "num_samples=$NUM_SAMPLES"
  echo "batch_size=$BATCH_SIZE"
  echo "latent_length=$LATENT_LENGTH"
  echo "trust_remote_code=$TRUST_REMOTE_CODE"
  echo "gpu_list=$GPU_LIST"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "extra_args=$EXTRA_ARGS"
} > "$RUN_DIR/manifest_task_${TASK_ID}.txt"

cmd=(
  "$PYTHON_BIN" RecursiveMAS/run.py
  --style "$STYLE"
  --dataset "$DATASET"
  --method "$METHOD"
  --num_recursive_rounds "$R"
  --num_samples "$NUM_SAMPLES"
  --batch_size "$BATCH_SIZE"
  --latent_length "$LATENT_LENGTH"
  --seed "$SEED"
  --trust_remote_code "$TRUST_REMOTE_CODE"
  --deterministic 1
  --lc_mode "$LC_MODE"
  --lc_site "$SITE"
  --lc_epsilon "$EPS"
  --lc_round "$LC_ROUND_EFFECTIVE"
  --lc_seed "$SEED"
  --lc_direction "$LC_DIRECTION"
  --result_jsonl "$RESULT_JSONL"
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

echo
echo "===== $DATASET :: experiment_c $LC_MODE site=$SITE eps=$EPS R=$R lc_round=$LC_ROUND_EFFECTIVE seed=$SEED ====="
echo "[experiment_c] result_jsonl=$RESULT_JSONL"
echo "[experiment_c] run_log=$RUN_LOG"
printf '[experiment_c] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}" 2>&1 | tee "$RUN_LOG"

echo
echo "[experiment_c] complete. JSONL log: $RESULT_JSONL"
