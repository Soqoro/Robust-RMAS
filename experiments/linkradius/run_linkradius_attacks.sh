#!/bin/bash
#SBATCH --job-name=linkradius-attacks
#SBATCH -p NA100q
#SBATCH -w node01
#SBATCH --output=logs/linkradius_attacks_%A_%a.out
#SBATCH --error=logs/linkradius_attacks_%A_%a.err

set -euo pipefail

LR_STAGE="${LR_STAGE:-grid}"
STYLE="${STYLE:-sequential_light}"
METHOD="${METHOD:-ours_recursive}"
DATASETS="${DATASETS:-gpqa}"
ROUNDS="${ROUNDS:-2}"
SEEDS="${SEEDS:-42}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LATENT_LENGTH="${LATENT_LENGTH:-32}"
K="${K:-32}"
BATCH_COUNTS="${BATCH_COUNTS:-test=5}"
ATTACK_FAMILIES="${ATTACK_FAMILIES:-universal_margin diffmean pca pgd_autograd random_independent}"

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
lr_validate_stage "$LR_STAGE" train_grid train val_grid val freeze_attack test_probe_grid test_probe test_grid test thresholds analyze validate grid
lr_run_entrypoint attacks "$LR_STAGE"
