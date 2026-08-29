#!/bin/bash
#SBATCH --job-name=linkradius-expansion
#SBATCH -p NA100q
#SBATCH -w node01
#SBATCH --output=logs/linkradius_expansion_%A_%a.out
#SBATCH --error=logs/linkradius_expansion_%A_%a.err

set -euo pipefail

LR_STAGE="${LR_STAGE:-grid}"
STYLE="${STYLE:-sequential_light}"
METHOD="${METHOD:-ours_recursive}"
DATASETS="${DATASETS:-gpqa}"
ROUNDS="${ROUNDS:-1 2 3 4 5}"
SEEDS="${SEEDS:-42}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LATENT_LENGTH="${LATENT_LENGTH:-32}"
CLEAN_CORRECT_POLICY="${CLEAN_CORRECT_POLICY:-forced_margin}"
CLEAN_STABILITY_POLICY="${CLEAN_STABILITY_POLICY:-strict}"
VALIDATION_TIER="${VALIDATION_TIER:-empirical}"

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
lr_validate_stage "$LR_STAGE" r2 r4 rounds medqa systems prompts protection grid
lr_run_entrypoint expansion "$LR_STAGE"
