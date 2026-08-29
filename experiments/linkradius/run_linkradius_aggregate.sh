#!/bin/bash
#SBATCH --job-name=linkradius-aggregate
#SBATCH -p NA100q
#SBATCH -w node01
#SBATCH --output=logs/linkradius_aggregate_%A_%a.out
#SBATCH --error=logs/linkradius_aggregate_%A_%a.err

set -euo pipefail

LR_STAGE="${LR_STAGE:-verify}"
AGGREGATE_PHASE="${AGGREGATE_PHASE:-smoke}"
DATASETS="${DATASETS:-gpqa}"
ROUNDS="${ROUNDS:-2}"
SEEDS="${SEEDS:-42}"
CLEAN_CORRECT_POLICY="${CLEAN_CORRECT_POLICY:-forced_margin}"
CLEAN_STABILITY_POLICY="${CLEAN_STABILITY_POLICY:-empirical}"
VALIDATION_TIER="${VALIDATION_TIER:-empirical}"
case "$AGGREGATE_PHASE" in
  engineering)
    CLEAN_CORRECT_POLICY="dual_correct"
    CLEAN_STABILITY_POLICY="strict"
    VALIDATION_TIER="certification"
    BATCH_SIZE="${BATCH_SIZE:-1}"
    NUM_BATCHES="${NUM_BATCHES:-1}"
    PROBE_SEEDS="${PROBE_SEEDS:-101}"
    ;;
  smoke)
    BATCH_SIZE="${BATCH_SIZE:-16}"
    NUM_BATCHES="${NUM_BATCHES:-2}"
    MAX_ELIGIBLE="${MAX_ELIGIBLE:-16}"
    PROBE_SEEDS="${PROBE_SEEDS:-101 202}"
    INCLUDE_GENERATION="${INCLUDE_GENERATION:-0}"
    ;;
  pilot)
    BATCH_SIZE="${BATCH_SIZE:-1}"
    PARTITIONS="${PARTITIONS:-validation}"
    NUM_BATCHES="${NUM_BATCHES:-40}"
    BATCH_COUNTS="${BATCH_COUNTS:-validation=40}"
    K="${K:-8}"
    GRADIENT_REFERENCE_BATCHES="${GRADIENT_REFERENCE_BATCHES:-2}"
    PROBE_SEEDS="${PROBE_SEEDS:-101 202 303}"
    INTERVENTIONS="${INTERVENTIONS:-identity mismatch zero}"
    INCLUDE_GENERATION="${INCLUDE_GENERATION:-0}"
    ;;
  attacks)
    BATCH_SIZE="${BATCH_SIZE:-1}"
    NUM_BATCHES="${NUM_BATCHES:-1}"
    BATCH_COUNTS="${BATCH_COUNTS:-test=5}"
    K="${K:-8}"
    ATTACK_FAMILIES="${ATTACK_FAMILIES:-pgd_autograd random_independent}"
    INTERVENTIONS="${INTERVENTIONS:-identity mismatch zero}"
    INCLUDE_GENERATION="${INCLUDE_GENERATION:-0}"
    ;;
esac

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
lr_validate_stage "$LR_STAGE" verify causal linkradius attacks metrics system_curves all
lr_run_entrypoint aggregate "$LR_STAGE"
