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

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/linkradius_common.sh"
lr_validate_stage "$LR_STAGE" r2 r4 rounds medqa systems prompts protection grid
lr_run_entrypoint expansion "$LR_STAGE"
