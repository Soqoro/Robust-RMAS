#!/usr/bin/env bash
#
# Experiment B: one-shot latent contagion phase diagram.
#
# Total jobs = (#DATASETS * #SITES * #EPSILONS * #ROUNDS * #SEEDS).
# Total jobs with defaults = 2 datasets * 3 sites * 8 eps * 5 rounds * 1 seed = 240.
# Local smoke:
#   SLURM_ARRAY_TASK_ID=0 NUM_SAMPLES=2 bash experiments/latent_contagion/run_experiment_b.sh
# Slurm:
#   sbatch --array=0-239 experiments/latent_contagion/run_experiment_b.sh
#
#SBATCH --job-name=latent_contagion_b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DATASETS="${DATASETS:-math500 gpqa}"
SITES="${SITES:-p2c c2s s2p}"
EPSILONS="${EPSILONS:-0 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1}"
ROUNDS="${ROUNDS:-1 2 3 4 5}"
SEEDS="${SEEDS:-42}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LATENT_LENGTH="${LATENT_LENGTH:-48}"
STYLE="${STYLE:-sequential_light}"
METHOD="${METHOD:-ours_recursive}"
OUT_DIR="${OUT_DIR:-outputs/latent_contagion/experiment_b}"

read -r -a DATASET_LIST <<< "${DATASETS}"
read -r -a SITE_LIST <<< "${SITES}"
read -r -a EPSILON_LIST <<< "${EPSILONS}"
read -r -a ROUND_LIST <<< "${ROUNDS}"
read -r -a SEED_LIST <<< "${SEEDS}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]]; then
  echo "[error] SLURM_ARRAY_TASK_ID must be a non-negative integer, got: ${TASK_ID}" >&2
  exit 2
fi

TOTAL_TASKS=0
for _DATASET in "${DATASET_LIST[@]}"; do
  for _SITE in "${SITE_LIST[@]}"; do
    for _EPS in "${EPSILON_LIST[@]}"; do
      for _R in "${ROUND_LIST[@]}"; do
        for _SEED in "${SEED_LIST[@]}"; do
          TOTAL_TASKS=$((TOTAL_TASKS + 1))
        done
      done
    done
  done
done

if (( TASK_ID >= TOTAL_TASKS )); then
  echo "[error] array index ${TASK_ID} is out of range." >&2
  echo "[error] total number of tasks: ${TOTAL_TASKS}" >&2
  exit 2
fi

DATASET=""
SITE=""
EPS=""
R=""
SEED=""
INDEX=0
FOUND=0
for _DATASET in "${DATASET_LIST[@]}"; do
  for _SITE in "${SITE_LIST[@]}"; do
    for _EPS in "${EPSILON_LIST[@]}"; do
      for _R in "${ROUND_LIST[@]}"; do
        for _SEED in "${SEED_LIST[@]}"; do
          if (( INDEX == TASK_ID )); then
            DATASET="${_DATASET}"
            SITE="${_SITE}"
            EPS="${_EPS}"
            R="${_R}"
            SEED="${_SEED}"
            FOUND=1
            break 5
          fi
          INDEX=$((INDEX + 1))
        done
      done
    done
  done
done

if (( FOUND != 1 )); then
  echo "[error] failed to resolve array index ${TASK_ID}." >&2
  echo "[error] total number of tasks: ${TOTAL_TASKS}" >&2
  exit 2
fi

RUN_DIR="${OUT_DIR}/${DATASET}/oneshot"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

RESULT_JSONL="${RUN_DIR}/site=${SITE}_eps=${EPS}_R=${R}_seed=${SEED}.jsonl"
JOB_TAG="${SLURM_ARRAY_JOB_ID:-local}_${TASK_ID}"
STDOUT_LOG="${LOG_DIR}/site=${SITE}_eps=${EPS}_R=${R}_seed=${SEED}_${JOB_TAG}.out"
STDERR_LOG="${LOG_DIR}/site=${SITE}_eps=${EPS}_R=${R}_seed=${SEED}_${JOB_TAG}.err"

echo "[config] task_id=${TASK_ID}/${TOTAL_TASKS}"
echo "[config] dataset=${DATASET}"
echo "[config] site=${SITE}"
echo "[config] epsilon=${EPS}"
echo "[config] recursive_rounds=${R}"
echo "[config] seed=${SEED}"
echo "[config] num_samples=${NUM_SAMPLES}"
echo "[config] batch_size=${BATCH_SIZE}"
echo "[config] latent_length=${LATENT_LENGTH}"
echo "[config] style=${STYLE}"
echo "[config] method=${METHOD}"
echo "[config] result_jsonl=${RESULT_JSONL}"
echo "[config] stdout_log=${STDOUT_LOG}"
echo "[config] stderr_log=${STDERR_LOG}"

python RecursiveMAS/run.py \
  --style "${STYLE}" \
  --dataset "${DATASET}" \
  --method "${METHOD}" \
  --num_recursive_rounds "${R}" \
  --num_samples "${NUM_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --latent_length "${LATENT_LENGTH}" \
  --seed "${SEED}" \
  --deterministic 1 \
  --lc_mode one_shot \
  --lc_site "${SITE}" \
  --lc_epsilon "${EPS}" \
  --lc_round 0 \
  --lc_seed "${SEED}" \
  --result_jsonl "${RESULT_JSONL}" \
  >"${STDOUT_LOG}" \
  2>"${STDERR_LOG}"

echo "[done] wrote ${RESULT_JSONL}"
