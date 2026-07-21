#!/bin/bash
#SBATCH --job-name=latent-contagion-c-calib-array
#SBATCH -p NA100q
#SBATCH -w node01
#SBATCH --output=logs/latent_contagion_c_calibration_array_%A_%a.out
#SBATCH --error=logs/latent_contagion_c_calibration_array_%A_%a.err

# Run Experiment C calibration depths across a fixed number of Slurm workers.
# Each worker owns a strided subset of CALIBRATION_ROUNDS and executes
# clean -> attack -> extract sequentially for every assigned depth.
#
# Example (four manually assigned GPUs):
#   DATASET=gpqa GPU_LIST="0 1 2 3" CALIBRATION_ROUNDS="1 2 3 4 5" \
#     sbatch --array=0-3%4 \
#     experiments/latent_contagion/run_experiment_c_calibration_array.sh

set -euo pipefail

mkdir -p logs

CALIB_SCRIPT="${CALIB_SCRIPT:-experiments/latent_contagion/run_experiment_c_calibration.sh}"
CALIBRATION_ROUNDS="${CALIBRATION_ROUNDS:-1 2 3 4 5}"
CALIBRATION_DRY_RUN="${CALIBRATION_DRY_RUN:-0}"
WORKER_ID="${SLURM_ARRAY_TASK_ID:-0}"
WORKER_COUNT="${CALIBRATION_WORKERS:-${SLURM_ARRAY_TASK_COUNT:-1}}"

die() {
  echo "[error] $*" >&2
  exit 2
}

if ! [[ "$WORKER_ID" =~ ^[0-9]+$ ]]; then
  die "SLURM_ARRAY_TASK_ID must be a non-negative integer, got: $WORKER_ID"
fi
if ! [[ "$WORKER_COUNT" =~ ^[0-9]+$ ]] || (( WORKER_COUNT < 1 )); then
  die "CALIBRATION_WORKERS/SLURM_ARRAY_TASK_COUNT must be positive, got: $WORKER_COUNT"
fi
if (( WORKER_ID >= WORKER_COUNT )); then
  die "worker id $WORKER_ID is outside worker count $WORKER_COUNT"
fi
case "$CALIBRATION_DRY_RUN" in
  0|1) ;;
  *) die "CALIBRATION_DRY_RUN must be 0 or 1, got: $CALIBRATION_DRY_RUN" ;;
esac
if [[ ! -f "$CALIB_SCRIPT" ]]; then
  die "calibration script not found: $CALIB_SCRIPT"
fi

read -r -a calibration_rounds_array <<< "$CALIBRATION_ROUNDS"
if (( ${#calibration_rounds_array[@]} == 0 )); then
  die "CALIBRATION_ROUNDS must contain at least one positive integer"
fi

echo "[experiment_c_calibration_array] worker=$WORKER_ID/$WORKER_COUNT"
echo "[experiment_c_calibration_array] rounds=$CALIBRATION_ROUNDS dataset=${DATASET:-math500}"
echo "[experiment_c_calibration_array] gpu_list=${GPU_LIST:-<empty>}"

assigned=0
for index in "${!calibration_rounds_array[@]}"; do
  if (( index % WORKER_COUNT != WORKER_ID )); then
    continue
  fi

  calibration_r="${calibration_rounds_array[$index]}"
  if ! [[ "$calibration_r" =~ ^[0-9]+$ ]] || (( calibration_r < 1 )); then
    die "CALIBRATION_ROUNDS items must be positive integers, got: $calibration_r"
  fi
  assigned=$((assigned + 1))

  for stage in clean attack extract; do
    echo "[experiment_c_calibration_array] R=$calibration_r stage=$stage"
    if [[ "$CALIBRATION_DRY_RUN" == "0" ]]; then
      CALIB_STAGE="$stage" CALIBRATION_R="$calibration_r" bash "$CALIB_SCRIPT"
    fi
  done
done

if (( assigned == 0 )); then
  echo "[experiment_c_calibration_array] worker=$WORKER_ID has no assigned calibration depths"
else
  echo "[experiment_c_calibration_array] worker=$WORKER_ID complete assigned_depths=$assigned"
fi
