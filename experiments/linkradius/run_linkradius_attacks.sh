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
BATCH_COUNTS="${BATCH_COUNTS:-}"
ATTACK_FAMILIES="${ATTACK_FAMILIES:-pgd_autograd random_independent}"
ATTACK_EPSILONS="${ATTACK_EPSILONS:-3e-4 1e-3 3e-3 1e-2 3e-2 1e-1}"
PGD_STEPS="${PGD_STEPS:-20}"

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

# After validation has frozen the attack protocol, make every later command
# consume its exact h/seeds/K and attack grid.  This avoids accidental drift
# from launcher defaults (for example K=32 after Phase 3 selected K=8).
case "$LR_STAGE" in
  clean_grid|clean|test_probe_grid|test_probe|test_grid|test|thresholds|analyze|validate|grid)
    if [[ -f "$FROZEN_ATTACK_CONFIG" ]]; then
      mapfile -t lr_frozen_values < <(
        "$PYTHON_BIN" - "$FROZEN_ATTACK_CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    frozen = json.load(handle)

print(frozen["probe"]["h"])
print(" ".join(str(value) for value in frozen["probe"]["seeds"]))
print(frozen["probe"]["K"])
print(" ".join(str(value) for value in frozen["attack_families"]))
print(" ".join(str(value) for value in frozen["attack_epsilons"]))
print(frozen["pgd"]["steps"])
print(frozen["random_independent"]["seed_offset"])
print(frozen["subspace"])
PY
      )
      if [[ "${#lr_frozen_values[@]}" -ne 8 ]]; then
        echo "[error] could not hydrate the frozen attack protocol" >&2
        exit 2
      fi
      PROBE_RADII="${lr_frozen_values[0]}"
      PROBE_SEEDS="${lr_frozen_values[1]}"
      K="${lr_frozen_values[2]}"
      ATTACK_FAMILIES="${lr_frozen_values[3]}"
      ATTACK_EPSILONS="${lr_frozen_values[4]}"
      PGD_STEPS="${lr_frozen_values[5]}"
      RANDOM_ATTACK_SEED_OFFSET="${lr_frozen_values[6]}"
      SUBSPACE="${lr_frozen_values[7]}"
      unset lr_frozen_values
    fi
    ;;
esac

lr_validate_stage "$LR_STAGE" split freeze_execution val_grid val freeze_attack clean_grid clean test_probe_grid test_probe test_grid test thresholds analyze validate grid
lr_run_entrypoint attacks "$LR_STAGE"
