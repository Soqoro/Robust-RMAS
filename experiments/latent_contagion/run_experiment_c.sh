#!/bin/bash
#SBATCH --job-name=latent-contagion-c
#SBATCH -p NA100q
#SBATCH -w node01
#SBATCH --output=logs/latent_contagion_c_%A_%a.out
#SBATCH --error=logs/latent_contagion_c_%A_%a.err

# Experiment C: DiffMean bank-direction one-shot latent steering phase diagram.
#
# Total jobs = (#DATASETS * #SITES * #EPSILONS * #ROUNDS * #SEEDS).
# Total jobs with defaults = 1 dataset * 3 sites * 8 eps * 5 rounds * 1 seed = 120.
# Local smoke:
#   SLURM_ARRAY_TASK_ID=0 NUM_SAMPLES=2 bash experiments/latent_contagion/run_experiment_c.sh
# Slurm:
#   sbatch --array=0-119 experiments/latent_contagion/run_experiment_c.sh

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
LC_ROUND="${LC_ROUND:-0}"
LC_DIRECTION="${LC_DIRECTION:-bank}"
LC_STEERING_METHOD="${LC_STEERING_METHOD:-diffmean}"
CALIBRATION_R="${CALIBRATION_R:-2}"
CALIB_ROOT="${CALIB_ROOT:-outputs/latent_contagion/diffmean_calibration}"
LC_STEERING_ID="${LC_STEERING_ID:-diffmean_R${CALIBRATION_R}_math500_role_aligned_target_hit}"
LC_STEERING_BANK="${LC_STEERING_BANK:-$CALIB_ROOT/math500_R${CALIBRATION_R}/${LC_STEERING_ID}.pt}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LATENT_LENGTH="${LATENT_LENGTH:-48}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
GPU_LIST="${GPU_LIST:-}"
OUT_ROOT="${OUT_ROOT:-${OUT_DIR:-outputs/latent_contagion/experiment_c}}"
RUN_SUBDIR="${RUN_SUBDIR:-diffmean_R${CALIBRATION_R}_target_hit}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]]; then
  echo "[error] SLURM_ARRAY_TASK_ID must be a non-negative integer, got: $TASK_ID" >&2
  exit 2
fi

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

if [[ "$LC_DIRECTION" == "bank" && ! -f "$LC_STEERING_BANK" ]]; then
  echo "[error] LC_STEERING_BANK does not exist: $LC_STEERING_BANK" >&2
  echo "[error] Run experiments/latent_contagion/run_experiment_c_calibration.sh first, or set LC_STEERING_BANK." >&2
  exit 2
fi

total_jobs() {
  local total=0
  local dataset site eps rounds seed
  for dataset in $DATASETS; do
    for site in $SITES; do
      for eps in $EPSILONS; do
        for rounds in $ROUNDS; do
          for seed in $SEEDS; do
            total=$((total + 1))
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
  local dataset site eps rounds seed
  for dataset in $DATASETS; do
    for site in $SITES; do
      for eps in $EPSILONS; do
        for rounds in $ROUNDS; do
          for seed in $SEEDS; do
            if (( index == target )); then
              DATASET="$dataset"
              SITE="$site"
              EPS="$eps"
              R="$rounds"
              SEED="$seed"
              return 0
            fi
            index=$((index + 1))
          done
        done
      done
    done
  done
  return 1
}

TOTAL_TASKS="$(total_jobs)"
if (( TASK_ID >= TOTAL_TASKS )); then
  echo "[error] array index $TASK_ID is out of range." >&2
  echo "[error] total number of tasks: $TOTAL_TASKS" >&2
  exit 2
fi

DATASET=""
SITE=""
EPS=""
R=""
SEED=""
select_config "$TASK_ID"

RUN_DIR="$OUT_ROOT/$DATASET/$RUN_SUBDIR"
LOG_DIR="$RUN_DIR/logs"
RESULT_JSONL="$RUN_DIR/site=${SITE}_eps=${EPS}_R=${R}_seed=${SEED}.jsonl"
RUN_LOG="$LOG_DIR/site=${SITE}_eps=${EPS}_R=${R}_seed=${SEED}.log"

mkdir -p "$RUN_DIR" "$LOG_DIR"

echo "[experiment_c] out_root=$OUT_ROOT"
echo "[experiment_c] task_id=$TASK_ID/$TOTAL_TASKS"
echo "[experiment_c] style=$STYLE method=$METHOD"
echo "[experiment_c] datasets=$DATASETS"
echo "[experiment_c] sites=$SITES"
echo "[experiment_c] epsilons=$EPSILONS"
echo "[experiment_c] rounds=$ROUNDS"
echo "[experiment_c] seeds=$SEEDS"
echo "[experiment_c] lc_mode=$LC_MODE lc_round=$LC_ROUND lc_direction=$LC_DIRECTION run_subdir=$RUN_SUBDIR"
echo "[experiment_c] lc_steering_bank=$LC_STEERING_BANK"
echo "[experiment_c] lc_steering_method=$LC_STEERING_METHOD lc_steering_id=$LC_STEERING_ID"
echo "[experiment_c] selected dataset=$DATASET site=$SITE eps=$EPS rounds=$R seed=$SEED"
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
  echo "seeds=$SEEDS"
  echo "lc_mode=$LC_MODE"
  echo "lc_round=$LC_ROUND"
  echo "lc_direction=$LC_DIRECTION"
  echo "lc_steering_bank=$LC_STEERING_BANK"
  echo "lc_steering_method=$LC_STEERING_METHOD"
  echo "lc_steering_id=$LC_STEERING_ID"
  echo "run_subdir=$RUN_SUBDIR"
  echo "task_id=$TASK_ID"
  echo "total_tasks=$TOTAL_TASKS"
  echo "selected_dataset=$DATASET"
  echo "selected_site=$SITE"
  echo "selected_epsilon=$EPS"
  echo "selected_rounds=$R"
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
  --lc_round "$LC_ROUND"
  --lc_seed "$SEED"
  --lc_direction "$LC_DIRECTION"
  --result_jsonl "$RESULT_JSONL"
)

if [[ -n "$LC_STEERING_BANK" ]]; then
  cmd+=(--lc_steering_bank "$LC_STEERING_BANK")
fi
if [[ -n "$LC_STEERING_METHOD" ]]; then
  cmd+=(--lc_steering_method "$LC_STEERING_METHOD")
fi
if [[ -n "$LC_STEERING_ID" ]]; then
  cmd+=(--lc_steering_id "$LC_STEERING_ID")
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
echo "===== $DATASET :: experiment_c $LC_MODE site=$SITE eps=$EPS R=$R seed=$SEED ====="
echo "[experiment_c] result_jsonl=$RESULT_JSONL"
echo "[experiment_c] run_log=$RUN_LOG"
printf '[experiment_c] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}" 2>&1 | tee "$RUN_LOG"

echo
echo "[experiment_c] complete. JSONL log: $RESULT_JSONL"
