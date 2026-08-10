#!/bin/bash
#SBATCH --job-name=linkradius-engineering
#SBATCH -p NA100q
#SBATCH -w node01
#SBATCH --output=logs/linkradius_engineering_%A_%a.out
#SBATCH --error=logs/linkradius_engineering_%A_%a.err

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
NUM_BATCHES="${NUM_BATCHES:-1}"
DISCOVERY_BATCHES="${DISCOVERY_BATCHES:-20}"
K="${K:-8}"
PROBE_RADII="${PROBE_RADII:-1e-3 3e-3}"
PROBE_SEEDS="${PROBE_SEEDS:-101}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/linkradius_common.sh"
lr_validate_stage "$LR_STAGE" split discover freeze_execution clean replay probe gradient validate all grid
lr_run_entrypoint engineering "$LR_STAGE"
