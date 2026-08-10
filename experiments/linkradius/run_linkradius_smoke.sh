#!/bin/bash
#SBATCH --job-name=linkradius-smoke
#SBATCH -p NA100q
#SBATCH -w node01
#SBATCH --output=logs/linkradius_smoke_%A_%a.out
#SBATCH --error=logs/linkradius_smoke_%A_%a.err

set -euo pipefail

LR_STAGE="${LR_STAGE:-grid}"
STYLE="${STYLE:-sequential_light}"
METHOD="${METHOD:-ours_recursive}"
DATASETS="${DATASETS:-gpqa}"
ROUNDS="${ROUNDS:-2}"
SEEDS="${SEEDS:-42}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LATENT_LENGTH="${LATENT_LENGTH:-32}"
PARTITIONS="${PARTITIONS:-validation}"
NUM_BATCHES="${NUM_BATCHES:-2}"
MAX_ELIGIBLE="${MAX_ELIGIBLE:-16}"
K="${K:-8}"
PROBE_RADII="${PROBE_RADII:-1e-3 3e-3}"
PROBE_SEEDS="${PROBE_SEEDS:-101 202}"

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
lr_validate_stage "$LR_STAGE" split screen freeze_execution clean causal_grid causal probe_grid probe gradient_grid gradient attack_grid attack estimate aggregate validate all grid
lr_run_entrypoint smoke "$LR_STAGE"
