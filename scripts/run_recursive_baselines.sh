#!/bin/bash
#SBATCH --job-name=recursive-baselines
#SBATCH -p NA100q
#SBATCH -w node01
#SBATCH --output=logs/recursive_baselines_%j.out
#SBATCH --error=logs/recursive_baselines_%j.err

set -euo pipefail

mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-python}"
STYLE="${STYLE:-sequential_light}"
DATASETS="${DATASETS:-math500 gpqa medqa mbppplus}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
SEED="${SEED:-42}"
SAMPLE_SEED="${SAMPLE_SEED:--1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
TEXT_RECURSION_ROUNDS="${TEXT_RECURSION_ROUNDS:-3}"
OUT_ROOT="${OUT_ROOT:-logs/recursive_baselines/$(date +%Y%m%d_%H%M%S)}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export CUDA_VISIBLE_DEVICES=7
export CONDA_NO_PLUGINS=true
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

echo "Using TMPDIR=$TMPDIR"
mkdir -p "$TMPDIR" || true
ls -ld "$TMPDIR" || true

mkdir -p "$OUT_ROOT"

echo "[baseline] out_root=$OUT_ROOT"
echo "[baseline] style=$STYLE"
echo "[baseline] datasets=$DATASETS"
echo "[baseline] num_samples=$NUM_SAMPLES"
echo "[baseline] seed=$SEED sample_seed=$SAMPLE_SEED"
echo "[baseline] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

echo "===== nvidia-smi -L ====="
nvidia-smi -L || true
echo "===== initial nvidia-smi ====="
nvidia-smi || true

{
  echo "style=$STYLE"
  echo "datasets=$DATASETS"
  echo "num_samples=$NUM_SAMPLES"
  echo "seed=$SEED"
  echo "sample_seed=$SAMPLE_SEED"
  echo "text_recursion_rounds=$TEXT_RECURSION_ROUNDS"
  echo "extra_args=$EXTRA_ARGS"
} > "$OUT_ROOT/manifest.txt"

run_one() {
  local dataset="$1"
  local label="$2"
  local system_name="$3"
  local method="$4"
  local rounds="$5"

  local safe_dataset="${dataset//\//_}"
  local out_dir="$OUT_ROOT/$safe_dataset"
  local result_jsonl="$out_dir/${label}.jsonl"
  local run_log="$out_dir/${label}.log"

  mkdir -p "$out_dir"

  local cmd=(
    "$PYTHON_BIN" RecursiveMAS/run.py
    --style "$STYLE"
    --dataset "$dataset"
    --method "$method"
    --system_name "$system_name"
    --num_recursive_rounds "$rounds"
    --num_samples "$NUM_SAMPLES"
    --seed "$SEED"
    --sample_seed "$SAMPLE_SEED"
    --trust_remote_code "$TRUST_REMOTE_CODE"
    --result_jsonl "$result_jsonl"
  )

  if [[ -n "${BATCH_SIZE:-}" ]]; then
    cmd+=(--batch_size "$BATCH_SIZE")
  fi
  if [[ -n "${LATENT_LENGTH:-}" ]]; then
    cmd+=(--latent_length "$LATENT_LENGTH")
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
  echo "===== $dataset :: $system_name ====="
  echo "[baseline] result_jsonl=$result_jsonl"
  printf '[baseline] command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}" 2>&1 | tee "$run_log"
}

for dataset in $DATASETS; do
  run_one "$dataset" "single_final_agent" "Single final agent" "single" 1
  run_one "$dataset" "traditional_text_mas" "Traditional text MAS" "text" 1
  run_one "$dataset" "recursive_text_mas_r${TEXT_RECURSION_ROUNDS}" "Recursive-TextMAS" "text_recursive" "$TEXT_RECURSION_ROUNDS"

  for rounds in 1 2 3; do
    run_one "$dataset" "recursivemas_r${rounds}" "RecursiveMAS" "ours_recursive" "$rounds"
  done

  run_one "$dataset" "recursivemas_no_feedback_r3" "RecursiveMAS no-feedback" "ours_recursive_no_feedback" 3
done

echo
echo "[baseline] complete. JSONL logs are under $OUT_ROOT"
