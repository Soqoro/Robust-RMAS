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

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/linkradius_common.sh"
lr_validate_stage "$LR_STAGE" train_grid train val_grid val freeze_attack test_probe_grid test_probe test_grid test thresholds analyze validate grid
lr_run_entrypoint attacks "$LR_STAGE"
