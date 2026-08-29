#!/bin/bash
#SBATCH --job-name=linkradius-pilot
#SBATCH -p NA100q
#SBATCH -w node01
#SBATCH --output=logs/linkradius_pilot_%A_%a.out
#SBATCH --error=logs/linkradius_pilot_%A_%a.err

set -euo pipefail

LR_STAGE="${LR_STAGE:-grid}"
STYLE="${STYLE:-sequential_light}"
METHOD="${METHOD:-ours_recursive}"
DATASETS="${DATASETS:-gpqa}"
ROUNDS="${ROUNDS:-2}"
SEEDS="${SEEDS:-42}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LATENT_LENGTH="${LATENT_LENGTH:-32}"
PARTITIONS="${PARTITIONS:-validation}"
NUM_BATCHES="${NUM_BATCHES:-40}"
BATCH_COUNTS="${BATCH_COUNTS:-validation=40}"
K="${K:-8}"
GRADIENT_REFERENCE_BATCHES="${GRADIENT_REFERENCE_BATCHES:-2}"
PROBE_RADII="${PROBE_RADII:-1e-3 3e-3}"
PROBE_SEEDS="${PROBE_SEEDS:-101 202 303}"
CLEAN_CORRECT_POLICY="${CLEAN_CORRECT_POLICY:-forced_margin}"
CLEAN_STABILITY_POLICY="${CLEAN_STABILITY_POLICY:-empirical}"
VALIDATION_TIER="${VALIDATION_TIER:-empirical}"
INTERVENTIONS="${INTERVENTIONS:-identity mismatch zero}"
INCLUDE_GENERATION="${INCLUDE_GENERATION:-0}"

lr_entrypoint_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  lr_submit_root="${LINKRADIUS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
  if [[ -z "$lr_submit_root" ]]; then
    echo "[error] SLURM_SUBMIT_DIR is unavailable; export LINKRADIUS_REPO_ROOT" >&2
    exit 2
  fi
  lr_entrypoint_dir="$lr_submit_root/experiments/linkradius"
fi
if [[ ! -f "$lr_entrypoint_dir/linkradius_common.sh" ]]; then
  echo "[error] cannot find linkradius_common.sh under $lr_entrypoint_dir" >&2
  exit 2
fi
source "$lr_entrypoint_dir/linkradius_common.sh"
unset lr_entrypoint_dir lr_submit_root
lr_validate_stage "$LR_STAGE" split screen_clean_grid screen_clean freeze_execution clean_grid clean causal_grid causal probe_calibration_grid probe_calibration gradient_grid gradient freeze_probe validate_probe aggregate validate grid
lr_run_entrypoint pilot "$LR_STAGE"
