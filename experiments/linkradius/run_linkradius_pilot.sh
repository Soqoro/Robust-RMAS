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
BATCH_SIZE="${BATCH_SIZE:-16}"
LATENT_LENGTH="${LATENT_LENGTH:-32}"
PARTITIONS="${PARTITIONS:-attack_train validation}"
NUM_BATCHES="${NUM_BATCHES:-5}"
BATCH_COUNTS="${BATCH_COUNTS:-attack_train=5 validation=3}"
K="${K:-32}"
PROBE_RADII="${PROBE_RADII:-1e-3 3e-3}"
PROBE_SEEDS="${PROBE_SEEDS:-101 202 303}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/linkradius_common.sh"
lr_validate_stage "$LR_STAGE" split screen_clean_grid screen_clean freeze_execution clean_grid clean causal_grid causal probe_calibration_grid probe_calibration gradient_grid gradient freeze_probe validate_probe aggregate validate grid
lr_run_entrypoint pilot "$LR_STAGE"
