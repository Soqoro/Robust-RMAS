#!/bin/bash
#SBATCH --job-name=latent-contagion-c-calib
#SBATCH -p NA100q
#SBATCH -w node01
#SBATCH --output=logs/latent_contagion_c_calibration_%A_%a.out
#SBATCH --error=logs/latent_contagion_c_calibration_%A_%a.err

# Experiment C calibration: collect clean/direct-attacked R=2 latent traces and
# extract a DiffMean steering bank.
#
# Slurm:
#   CALIB_JOB=$(sbatch --parsable --array=0-1 experiments/latent_contagion/run_experiment_c_calibration.sh)
#   CALIB_STAGE=extract sbatch --dependency=afterok:$CALIB_JOB experiments/latent_contagion/run_experiment_c_calibration.sh
#
# Local smoke:
#   CALIB_STAGE=clean NUM_SAMPLES=2 bash experiments/latent_contagion/run_experiment_c_calibration.sh
#   CALIB_STAGE=attack NUM_SAMPLES=2 bash experiments/latent_contagion/run_experiment_c_calibration.sh
#   CALIB_STAGE=extract bash experiments/latent_contagion/run_experiment_c_calibration.sh

set -euo pipefail

mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-python}"
STYLE="${STYLE:-sequential_light}"
METHOD="${METHOD:-ours_recursive}"
DATASET="${DATASET:-math500}"
CALIBRATION_R="${CALIBRATION_R:-2}"
SEED="${SEED:-42}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LATENT_LENGTH="${LATENT_LENGTH:-48}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
TRACE_SITES="${TRACE_SITES:-p2c,c2s,s2p}"
TRACE_ROUNDS="${TRACE_ROUNDS:-0}"
TRACE_DTYPE="${TRACE_DTYPE:-float16}"
GPU_LIST="${GPU_LIST:-}"
FILTER="${FILTER:-all_valid_pairs}"
TARGET_ANSWER="${TARGET_ANSWER:-999999999}"
MIN_PAIRS="${MIN_PAIRS:-1}"
ATTACK_SUFFIX_PATH="${ATTACK_SUFFIX_PATH:-experiments/latent_contagion/attack_suffix_math_role_aligned.txt}"
DEFAULT_STEERING_ID="diffmean_R${CALIBRATION_R}_${DATASET}_role_aligned"
if [[ "$FILTER" == "target_hit" ]]; then
  DEFAULT_STEERING_ID="${DEFAULT_STEERING_ID}_target_hit"
elif [[ "$FILTER" == "clean_correct_target_hit" ]]; then
  DEFAULT_STEERING_ID="${DEFAULT_STEERING_ID}_clean_correct_target_hit"
fi
STEERING_ID="${STEERING_ID:-$DEFAULT_STEERING_ID}"
CALIB_ROOT="${CALIB_ROOT:-outputs/latent_contagion/diffmean_calibration}"
CALIB_SUBDIR="${CALIB_SUBDIR:-${DATASET}_R${CALIBRATION_R}}"
CALIB_DIR="$CALIB_ROOT/$CALIB_SUBDIR"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRACT_EXTRA_ARGS="${EXTRACT_EXTRA_ARGS:-}"

CLEAN_JSONL="${CLEAN_JSONL:-$CALIB_DIR/clean_R${CALIBRATION_R}.jsonl}"
ATTACK_JSONL="${ATTACK_JSONL:-$CALIB_DIR/attack_R${CALIBRATION_R}.jsonl}"
CLEAN_TRACE="${CLEAN_TRACE:-$CALIB_DIR/clean_R${CALIBRATION_R}_trace.pt}"
ATTACK_TRACE="${ATTACK_TRACE:-$CALIB_DIR/attack_R${CALIBRATION_R}_trace.pt}"
OUT_BANK="${OUT_BANK:-$CALIB_DIR/${STEERING_ID}.pt}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ -z "${CALIB_STAGE:-}" ]]; then
  if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]]; then
    echo "[error] SLURM_ARRAY_TASK_ID must be a non-negative integer, got: $TASK_ID" >&2
    exit 2
  fi
  case "$TASK_ID" in
    0) CALIB_STAGE="clean" ;;
    1) CALIB_STAGE="attack" ;;
    *)
      echo "[error] calibration array index $TASK_ID is out of range. Use --array=0-1." >&2
      exit 2
      ;;
  esac
fi

case "$CALIB_STAGE" in
  clean|attack|extract) ;;
  *)
    echo "[error] CALIB_STAGE must be one of: clean, attack, extract. Got: $CALIB_STAGE" >&2
    exit 2
    ;;
esac

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

mkdir -p "$CALIB_DIR" "$CALIB_DIR/logs" "$TMPDIR"

RUN_LOG="$CALIB_DIR/logs/${CALIB_STAGE}_R${CALIBRATION_R}_seed=${SEED}.log"

echo "Using TMPDIR=$TMPDIR"
ls -ld "$TMPDIR" || true

echo "[experiment_c_calibration] calib_stage=$CALIB_STAGE"
echo "[experiment_c_calibration] calib_dir=$CALIB_DIR"
echo "[experiment_c_calibration] style=$STYLE method=$METHOD dataset=$DATASET"
echo "[experiment_c_calibration] calibration_R=$CALIBRATION_R seed=$SEED"
echo "[experiment_c_calibration] num_samples=$NUM_SAMPLES batch_size=$BATCH_SIZE latent_length=$LATENT_LENGTH"
echo "[experiment_c_calibration] trace_sites=$TRACE_SITES trace_rounds=$TRACE_ROUNDS trace_dtype=$TRACE_DTYPE"
echo "[experiment_c_calibration] steering_id=$STEERING_ID out_bank=$OUT_BANK"
echo "[experiment_c_calibration] gpu_list=${GPU_LIST:-<empty>} task_id=$TASK_ID"
echo "[experiment_c_calibration] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

{
  echo "calib_stage=$CALIB_STAGE"
  echo "style=$STYLE"
  echo "method=$METHOD"
  echo "dataset=$DATASET"
  echo "calibration_R=$CALIBRATION_R"
  echo "seed=$SEED"
  echo "num_samples=$NUM_SAMPLES"
  echo "batch_size=$BATCH_SIZE"
  echo "latent_length=$LATENT_LENGTH"
  echo "trust_remote_code=$TRUST_REMOTE_CODE"
  echo "trace_sites=$TRACE_SITES"
  echo "trace_rounds=$TRACE_ROUNDS"
  echo "trace_dtype=$TRACE_DTYPE"
  echo "gpu_list=$GPU_LIST"
  echo "task_id=$TASK_ID"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "attack_suffix_path=$ATTACK_SUFFIX_PATH"
  echo "filter=$FILTER"
  echo "target_answer=$TARGET_ANSWER"
  echo "min_pairs=$MIN_PAIRS"
  echo "steering_id=$STEERING_ID"
  echo "clean_jsonl=$CLEAN_JSONL"
  echo "attack_jsonl=$ATTACK_JSONL"
  echo "clean_trace=$CLEAN_TRACE"
  echo "attack_trace=$ATTACK_TRACE"
  echo "out_bank=$OUT_BANK"
  echo "extra_args=$EXTRA_ARGS"
  echo "extract_extra_args=$EXTRACT_EXTRA_ARGS"
} > "$CALIB_DIR/manifest_${CALIB_STAGE}.txt"

if [[ "$CALIB_STAGE" == "extract" ]]; then
  for required_path in "$CLEAN_JSONL" "$ATTACK_JSONL" "$CLEAN_TRACE" "$ATTACK_TRACE"; do
    if [[ ! -f "$required_path" ]]; then
      echo "[error] required calibration input is missing: $required_path" >&2
      exit 2
    fi
  done

  cmd=(
    "$PYTHON_BIN" experiments/latent_contagion/extract_diffmean_steering.py
    --clean_jsonl "$CLEAN_JSONL"
    --attack_jsonl "$ATTACK_JSONL"
    --clean_trace "$CLEAN_TRACE"
    --attack_trace "$ATTACK_TRACE"
    --out_bank "$OUT_BANK"
    --sites "$TRACE_SITES"
    --rounds "$TRACE_ROUNDS"
    --filter "$FILTER"
    --target_answer "$TARGET_ANSWER"
    --min_pairs "$MIN_PAIRS"
    --calibration_R "$CALIBRATION_R"
    --steering_id "$STEERING_ID"
  )
  if [[ -n "$EXTRACT_EXTRA_ARGS" ]]; then
    read -r -a extract_extra_args_array <<< "$EXTRACT_EXTRA_ARGS"
    cmd+=("${extract_extra_args_array[@]}")
  fi
else
  RESULT_JSONL="$CLEAN_JSONL"
  TRACE_PATH="$CLEAN_TRACE"
  QUESTION_SUFFIX_ARGS=()
  if [[ "$CALIB_STAGE" == "attack" ]]; then
    RESULT_JSONL="$ATTACK_JSONL"
    TRACE_PATH="$ATTACK_TRACE"
    QUESTION_SUFFIX_ARGS=(--question_suffix_path "$ATTACK_SUFFIX_PATH")
  fi

  cmd=(
    "$PYTHON_BIN" RecursiveMAS/run.py
    --style "$STYLE"
    --dataset "$DATASET"
    --method "$METHOD"
    --num_recursive_rounds "$CALIBRATION_R"
    --num_samples "$NUM_SAMPLES"
    --batch_size "$BATCH_SIZE"
    --latent_length "$LATENT_LENGTH"
    --seed "$SEED"
    --trust_remote_code "$TRUST_REMOTE_CODE"
    --deterministic 1
    --lc_mode none
    --lc_direction random
    --result_jsonl "$RESULT_JSONL"
    --lc_trace_path "$TRACE_PATH"
    --lc_trace_sites "$TRACE_SITES"
    --lc_trace_rounds "$TRACE_ROUNDS"
    --lc_trace_dtype "$TRACE_DTYPE"
    "${QUESTION_SUFFIX_ARGS[@]}"
  )

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
fi

echo "===== nvidia-smi -L ====="
nvidia-smi -L || true
echo "===== initial nvidia-smi ====="
nvidia-smi || true

echo
echo "===== Experiment C calibration :: $CALIB_STAGE ====="
echo "[experiment_c_calibration] run_log=$RUN_LOG"
printf '[experiment_c_calibration] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}" 2>&1 | tee "$RUN_LOG"

echo
echo "[experiment_c_calibration] complete stage=$CALIB_STAGE"
if [[ "$CALIB_STAGE" == "extract" ]]; then
  echo "[experiment_c_calibration] steering bank: $OUT_BANK"
fi
