#!/bin/bash
#SBATCH --job-name=latent-contagion-c
#SBATCH -p NA100q
#SBATCH -w node01
#SBATCH --output=logs/latent_contagion_c_%A_%a.out
#SBATCH --error=logs/latent_contagion_c_%A_%a.err

# Experiment C: DiffMean bank-direction one-shot latent steering phase diagram.
#
# The default grid contains two fixed clean runs (reference and control) per
# DATASET/R/SEED, followed by valid positive-epsilon attack configurations.
# Zero entries in EPSILONS are accepted for compatibility but are not attack
# jobs. Print the exact array size and mapping with:
#   C_STAGE=grid bash experiments/latent_contagion/run_experiment_c.sh
# Complete canonical clean JSONLs are reused. After changing code, models, or
# generation/evaluation settings, use a fresh OUT_ROOT (recommended) or set
# OVERWRITE_CLEAN=1 and rerun the matching attack jobs.
# With the defaults this is 10 clean jobs + 98 attack jobs = 108 tasks.
# LC_ROUNDS are zero-based latent injection/calibration round indices. For
# recursive depth R=5, p2c/c2s rounds are 0..4 and s2p rounds are 0..3.
# Local smoke:
#   SLURM_ARRAY_TASK_ID=0 NUM_SAMPLES=2 bash experiments/latent_contagion/run_experiment_c.sh
# Slurm:
#   sbatch --array=0-107 experiments/latent_contagion/run_experiment_c.sh
# Recompute the final index from C_STAGE=grid after changing any grid variable.
# Aggregate after the array completes:
#   python experiments/latent_contagion/aggregate_latent_contagion.py \
#     --root outputs/latent_contagion/experiment_c --dataset math500 \
#     --subdir diffmean_clean_correct_attack_wrong \
#     --clean_reference_root outputs/latent_contagion/experiment_c/clean/reference \
#     --clean_control_root outputs/latent_contagion/experiment_c/clean/control

set -euo pipefail

mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-python}"
STYLE="${STYLE:-sequential_light}"
METHOD="${METHOD:-ours_recursive}"
DATASETS="${DATASETS:-math500}"
SITES="${SITES:-p2c c2s s2p}"
EPSILONS="${EPSILONS:-0 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1}"
ROUNDS="${ROUNDS:-1 2 3 4 5}"
SEEDS="${SEEDS:-42}"
LC_MODE="${LC_MODE:-one_shot}"
LC_ROUNDS="${LC_ROUNDS:-${LC_ROUND:-0}}"
LC_DIRECTION="${LC_DIRECTION:-bank}"
LC_STEERING_METHOD="${LC_STEERING_METHOD:-diffmean}"
CALIB_ROOT="${CALIB_ROOT:-outputs/latent_contagion/diffmean_calibration}"
STEERING_FILTER="${STEERING_FILTER:-clean_correct_attack_wrong}"
LC_STEERING_ID="${LC_STEERING_ID:-}"
LC_STEERING_BANK="${LC_STEERING_BANK:-}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LATENT_LENGTH="${LATENT_LENGTH:-48}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
GPU_LIST="${GPU_LIST:-}"
OUT_ROOT="${OUT_ROOT:-${OUT_DIR:-outputs/latent_contagion/experiment_c}}"
RUN_SUBDIR="${RUN_SUBDIR:-diffmean_${STEERING_FILTER}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
C_STAGE="${C_STAGE:-run}"
OVERWRITE_CLEAN="${OVERWRITE_CLEAN:-0}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]]; then
  echo "[error] SLURM_ARRAY_TASK_ID must be a non-negative integer, got: $TASK_ID" >&2
  exit 2
fi

die() {
  echo "[error] $*" >&2
  exit 2
}

case "$C_STAGE" in
  run|grid) ;;
  *) die "C_STAGE must be run or grid. Got: $C_STAGE" ;;
esac

case "$STYLE" in
  sequential_light|sequential_scaled) ;;
  *) die "STYLE must use the sequential inference family (sequential_light or sequential_scaled). Got: $STYLE" ;;
esac

case "$METHOD" in
  ours_recursive|ours_recursive_no_feedback) ;;
  *) die "METHOD must be a recursive latent method (ours_recursive or ours_recursive_no_feedback). Got: $METHOD" ;;
esac

case "$OVERWRITE_CLEAN" in
  0|1) ;;
  *) die "OVERWRITE_CLEAN must be 0 or 1. Got: $OVERWRITE_CLEAN" ;;
esac

if ! POSITIVE_EPSILONS="$("$PYTHON_BIN" - "$EPSILONS" <<'PY'
import math
import sys

seen = set()
positive = []
for token in sys.argv[1].split():
    try:
        value = float(token)
    except ValueError:
        print(f"invalid EPSILONS item: {token!r}", file=sys.stderr)
        raise SystemExit(2)
    if not math.isfinite(value) or value < 0.0:
        print(f"EPSILONS item must be finite and non-negative: {token!r}", file=sys.stderr)
        raise SystemExit(2)
    if value > 0.0 and value not in seen:
        seen.add(value)
        positive.append(token)
print(" ".join(positive))
PY
)"; then
  die "EPSILONS items must be finite non-negative numbers."
fi

clean_result_is_reusable() {
  "$PYTHON_BIN" - "$@" <<'PY'
import json
import math
from pathlib import Path
import sys

path = Path(sys.argv[1])

def reject(reason):
    print(f"[clean-cache] {path}: {reason}", file=sys.stderr)
    raise SystemExit(1)

samples = []
summaries = []
try:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                reject(f"line {line_number} is invalid JSON ({exc})")
            if not isinstance(record, dict):
                reject(f"line {line_number} is not a JSON object")
            if str(record.get("type", "")).lower() == "summary":
                summaries.append(record)
            else:
                samples.append(record)
except OSError as exc:
    reject(f"cannot be read ({exc})")

if not samples:
    reject("contains no sample rows")
if len(summaries) != 1:
    reject(f"requires exactly one summary row, found {len(summaries)}")
summary = summaries[0]
declared_total = summary.get("num_samples", summary.get("n_total"))
if isinstance(declared_total, bool):
    reject("summary sample count is invalid")
try:
    declared_total = int(declared_total)
except (TypeError, ValueError):
    reject("summary has no valid num_samples/n_total")
if declared_total != len(samples):
    reject(f"summary declares {declared_total} samples but file has {len(samples)}")

expected_dataset, expected_style, expected_method = sys.argv[2:5]
expected_rounds, expected_seed = sys.argv[5:7]
expected_regime, requested_samples = sys.argv[7:9]
for field, expected in (
    ("dataset", expected_dataset),
    ("style", expected_style),
    ("method", expected_method),
):
    actual = str(summary.get(field, "")).strip()
    if actual != expected:
        reject(f"summary {field}={actual!r} does not match selected {expected!r}")
actual_regime = str(summary.get("role_response_regime", "")).strip().lower()
if actual_regime != expected_regime.lower():
    reject(
        f"summary role_response_regime={actual_regime!r} "
        f"does not match selected {expected_regime.lower()!r}"
    )
for field, actual, expected in (
    ("recursion_rounds", summary.get("recursion_rounds", summary.get("R")), expected_rounds),
    ("seed", summary.get("seed"), expected_seed),
):
    try:
        actual_int = int(actual)
        expected_int = int(expected)
    except (TypeError, ValueError):
        reject(f"summary has invalid {field} metadata")
    if actual_int != expected_int:
        reject(f"summary {field}={actual_int} does not match selected {expected_int}")
try:
    requested_samples_int = int(requested_samples)
except ValueError:
    reject(f"selected NUM_SAMPLES={requested_samples!r} is invalid")
if requested_samples_int >= 0 and declared_total != requested_samples_int:
    reject(
        f"summary num_samples={declared_total} does not match requested {requested_samples_int}"
    )

required_provenance = (
    "provenance_schema_version",
    "sample_cohort_sha256",
    "sample_ids_sha256",
    "questions_sha256",
    "ground_truths_sha256",
    "generation_config_sha256",
    "evaluation_config_sha256",
    "evaluation_protocol",
    "attack_config_sha256",
)
for field in required_provenance:
    values = [str(record.get(field, "")).strip() for record in [*samples, summary]]
    if not all(values):
        reject(f"is missing {field} provenance")
    if len(set(values)) != 1:
        reject(f"has inconsistent {field} provenance")

sample_ids = [str(record.get("sample_id", "")).strip() for record in samples]
if not all(sample_ids) or len(set(sample_ids)) != len(sample_ids):
    reject("has missing or duplicate sample_id values")
for record in samples:
    if not str(record.get("sample_input_sha256", "")).strip():
        reject("is missing per-sample input provenance")
    try:
        epsilon = float(record.get("lc_epsilon"))
    except (TypeError, ValueError):
        reject("has missing or invalid clean epsilon")
    if not math.isfinite(epsilon) or epsilon != 0.0:
        reject("contains a nonzero clean epsilon")
    if bool(record.get("lc_enabled", False)):
        reject("contains an enabled latent-contagion sample")
if str(summary.get("question_suffix_path", "")).strip() or str(summary.get("prompt_footer_path", "")).strip():
    reject("contains a direct prompt attack")
attack_config = summary.get("attack_config")
if not isinstance(attack_config, dict):
    reject("is missing attack_config metadata")
latent_config = attack_config.get("latent_contagion")
probe_config = attack_config.get("role_profile_probe")
if not isinstance(latent_config, dict) or str(latent_config.get("mode", "")).lower() != "none":
    reject("contains a latent-contagion attack configuration")
if float(latent_config.get("epsilon", float("nan"))) != 0.0:
    reject("contains a nonzero latent-contagion attack configuration")
if not isinstance(probe_config, dict) or str(probe_config.get("mode", "")).lower() != "none":
    reject("contains a role-profile probe configuration")
if float(probe_config.get("epsilon", float("nan"))) != 0.0:
    reject("contains a nonzero role-profile probe configuration")
PY
}

validate_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    die "$name must be a positive integer, got: $value"
  fi
}

validate_nonnegative_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    die "$name must be a non-negative integer, got: $value"
  fi
}

valid_lc_round_for_site_R() {
  local site="$1"
  local rounds="$2"
  local lc_round="$3"
  validate_positive_int "ROUNDS item" "$rounds"
  validate_nonnegative_int "LC_ROUNDS item" "$lc_round"
  case "$site" in
    p2c|c2s)
      (( lc_round < rounds ))
      ;;
    s2p)
      # s2p is the feedback edge and is inactive in the final zero-based round.
      (( rounds > 1 && lc_round < rounds - 1 ))
      ;;
    *)
      die "SITES item must be one of p2c, c2s, s2p. Got: $site"
      ;;
  esac
}

candidate_lc_rounds_for_config() {
  local site="$1"
  local rounds="$2"
  validate_positive_int "ROUNDS item" "$rounds"

  if [[ "$LC_ROUNDS" == "all" ]]; then
    local max_round
    case "$site" in
      p2c|c2s) max_round=$((rounds - 1)) ;;
      s2p) max_round=$((rounds - 2)) ;;
      *) die "SITES item must be one of p2c, c2s, s2p. Got: $site" ;;
    esac
    if (( max_round < 0 )); then
      return 0
    fi
    local lc_round
    for ((lc_round = 0; lc_round <= max_round; lc_round++)); do
      echo "$lc_round"
    done
    return 0
  fi

  local normalized="${LC_ROUNDS//,/ }"
  if [[ -z "$normalized" ]]; then
    die "LC_ROUNDS must be 'all' or a list of zero-based non-negative integers."
  fi
  local lc_round seen
  seen=" "
  for lc_round in $normalized; do
    validate_nonnegative_int "LC_ROUNDS item" "$lc_round"
    if valid_lc_round_for_site_R "$site" "$rounds" "$lc_round"; then
      if [[ "$seen" != *" $lc_round "* ]]; then
        echo "$lc_round"
        seen="${seen}${lc_round} "
      fi
    fi
  done
}

count_skipped_invalid_configs() {
  local skipped=0
  local dataset site eps rounds seed lc_round normalized
  if [[ "$LC_ROUNDS" == "all" ]]; then
    for dataset in $DATASETS; do
      for site in $SITES; do
        case "$site" in
          p2c|c2s|s2p) ;;
          *) die "SITES item must be one of p2c, c2s, s2p. Got: $site" ;;
        esac
        for eps in $POSITIVE_EPSILONS; do
          for rounds in $ROUNDS; do
            validate_positive_int "ROUNDS item" "$rounds"
            if [[ "$site" == "s2p" ]] && (( rounds <= 1 )); then
              for seed in $SEEDS; do
                skipped=$((skipped + 1))
              done
            fi
          done
        done
      done
    done
    echo "$skipped"
    return 0
  fi

  normalized="${LC_ROUNDS//,/ }"
  for dataset in $DATASETS; do
    for site in $SITES; do
      for eps in $POSITIVE_EPSILONS; do
        for rounds in $ROUNDS; do
          validate_positive_int "ROUNDS item" "$rounds"
          for lc_round in $normalized; do
            validate_nonnegative_int "LC_ROUNDS item" "$lc_round"
            if ! valid_lc_round_for_site_R "$site" "$rounds" "$lc_round"; then
              for seed in $SEEDS; do
                skipped=$((skipped + 1))
              done
            fi
          done
        done
      done
    done
  done
  echo "$skipped"
}

validate_grid_inputs() {
  local site rounds lc_round normalized
  for site in $SITES; do
    case "$site" in
      p2c|c2s|s2p) ;;
      *) die "SITES item must be one of p2c, c2s, s2p. Got: $site" ;;
    esac
  done
  for rounds in $ROUNDS; do
    validate_positive_int "ROUNDS item" "$rounds"
  done
  if [[ "$LC_ROUNDS" != "all" ]]; then
    normalized="${LC_ROUNDS//,/ }"
    if [[ -z "${normalized//[[:space:]]/}" ]]; then
      die "LC_ROUNDS must be 'all' or a list of zero-based non-negative integers."
    fi
    for lc_round in $normalized; do
      validate_nonnegative_int "LC_ROUNDS item" "$lc_round"
    done
  fi
}

validate_grid_inputs

if [[ -n "$GPU_LIST" ]]; then
  read -r -a gpu_array <<< "$GPU_LIST"
  if (( ${#gpu_array[@]} == 0 )); then
    echo "[error] GPU_LIST was set but no GPU ids were parsed." >&2
    exit 2
  fi
  gpu_index=$((TASK_ID % ${#gpu_array[@]}))
  export CUDA_VISIBLE_DEVICES="${gpu_array[$gpu_index]}"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
fi
export CONDA_NO_PLUGINS=true
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

echo "Using TMPDIR=$TMPDIR"
mkdir -p "$TMPDIR" || true
ls -ld "$TMPDIR" || true

total_jobs() {
  local total=0
  local clean_role dataset site eps rounds lc_round seed
  for clean_role in reference control; do
    for dataset in $DATASETS; do
      for rounds in $ROUNDS; do
        for seed in $SEEDS; do
          total=$((total + 1))
        done
      done
    done
  done
  for dataset in $DATASETS; do
    for site in $SITES; do
      for eps in $POSITIVE_EPSILONS; do
        for rounds in $ROUNDS; do
          for lc_round in $(candidate_lc_rounds_for_config "$site" "$rounds"); do
            for seed in $SEEDS; do
              total=$((total + 1))
            done
          done
        done
      done
    done
  done
  echo "$total"
}

select_config() {
  local target="$1"
  local index=0
  local clean_role dataset site eps rounds lc_round seed
  for clean_role in reference control; do
    for dataset in $DATASETS; do
      for rounds in $ROUNDS; do
        for seed in $SEEDS; do
          if (( index == target )); then
            JOB_KIND="clean"
            CLEAN_ROLE="$clean_role"
            DATASET="$dataset"
            SITE=""
            EPS="0"
            R="$rounds"
            LC_ROUND_EFFECTIVE=""
            SEED="$seed"
            return 0
          fi
          index=$((index + 1))
        done
      done
    done
  done
  for dataset in $DATASETS; do
    for site in $SITES; do
      for eps in $POSITIVE_EPSILONS; do
        for rounds in $ROUNDS; do
          for lc_round in $(candidate_lc_rounds_for_config "$site" "$rounds"); do
            for seed in $SEEDS; do
              if (( index == target )); then
                JOB_KIND="attack"
                CLEAN_ROLE=""
                DATASET="$dataset"
                SITE="$site"
                EPS="$eps"
                R="$rounds"
                LC_ROUND_EFFECTIVE="$lc_round"
                SEED="$seed"
                return 0
              fi
              index=$((index + 1))
            done
          done
        done
      done
    done
  done
  return 1
}

print_grid() {
  local index=0
  local clean_role dataset site eps rounds lc_round seed
  echo "total_jobs=$(total_jobs)"
  echo "array_index job_kind clean_role dataset site epsilon R lc_round seed"
  for clean_role in reference control; do
    for dataset in $DATASETS; do
      for rounds in $ROUNDS; do
        for seed in $SEEDS; do
          echo "$index clean $clean_role $dataset - 0 $rounds - $seed"
          index=$((index + 1))
        done
      done
    done
  done
  for dataset in $DATASETS; do
    for site in $SITES; do
      for eps in $POSITIVE_EPSILONS; do
        for rounds in $ROUNDS; do
          for lc_round in $(candidate_lc_rounds_for_config "$site" "$rounds"); do
            for seed in $SEEDS; do
              echo "$index attack - $dataset $site $eps $rounds $lc_round $seed"
              index=$((index + 1))
            done
          done
        done
      done
    done
  done
}

TOTAL_TASKS="$(total_jobs)"
SKIPPED_INVALID_CONFIGS="$(count_skipped_invalid_configs)"
if [[ "$C_STAGE" == "grid" ]]; then
  echo "skipped_invalid_attack_configs=$SKIPPED_INVALID_CONFIGS"
  print_grid
  exit 0
fi
if (( TOTAL_TASKS <= 0 )); then
  echo "[error] no valid experiment_c tasks after applying LC_ROUNDS=$LC_ROUNDS." >&2
  echo "[error] zero-based convention: p2c/c2s use 0..R-1; s2p uses 0..R-2 and is inactive for R=1." >&2
  exit 2
fi
if (( TASK_ID >= TOTAL_TASKS )); then
  echo "[error] array index $TASK_ID is out of range." >&2
  echo "[error] total number of tasks: $TOTAL_TASKS" >&2
  exit 2
fi

JOB_KIND=""
CLEAN_ROLE=""
DATASET=""
SITE=""
EPS=""
R=""
LC_ROUND_EFFECTIVE=""
SEED=""
select_config "$TASK_ID"

LC_STEERING_ID_EFFECTIVE=""
LC_STEERING_BANK_EFFECTIVE=""
if [[ "$JOB_KIND" == "attack" ]]; then
  LC_STEERING_ID_EFFECTIVE="${LC_STEERING_ID:-diffmean_R${R}_${DATASET}_role_aligned_${STEERING_FILTER}}"
  LC_STEERING_BANK_EFFECTIVE="${LC_STEERING_BANK:-$CALIB_ROOT/${DATASET}_R${R}/${LC_STEERING_ID_EFFECTIVE}.pt}"
  if [[ "$LC_DIRECTION" == "bank" && ! -f "$LC_STEERING_BANK_EFFECTIVE" ]]; then
    echo "[error] LC_STEERING_BANK_EFFECTIVE does not exist: $LC_STEERING_BANK_EFFECTIVE" >&2
    echo "[error] Run experiments/latent_contagion/run_experiment_c_calibration.sh for dataset=$DATASET R=$R, or set LC_STEERING_BANK." >&2
    exit 2
  fi
fi

if [[ "$JOB_KIND" == "clean" ]]; then
  LC_MODE_EFFECTIVE="none"
  LC_DIRECTION_EFFECTIVE="random"
  RUN_DIR="$OUT_ROOT/clean/$CLEAN_ROLE/$DATASET/R${R}/seed${SEED}"
  LOG_DIR="$RUN_DIR"
  RESULT_JSONL="$RUN_DIR/result.jsonl"
  RUN_LOG="$RUN_DIR/run.log"
  MANIFEST_PATH="$RUN_DIR/manifest.txt"
else
  LC_MODE_EFFECTIVE="$LC_MODE"
  LC_DIRECTION_EFFECTIVE="$LC_DIRECTION"
  RUN_DIR="$OUT_ROOT/$DATASET/$RUN_SUBDIR"
  LOG_DIR="$RUN_DIR/logs"
  RESULT_JSONL="$RUN_DIR/site=${SITE}_eps=${EPS}_R=${R}_lc_round=${LC_ROUND_EFFECTIVE}_seed=${SEED}.jsonl"
  RUN_LOG="$LOG_DIR/site=${SITE}_eps=${EPS}_R=${R}_lc_round=${LC_ROUND_EFFECTIVE}_seed=${SEED}.log"
  MANIFEST_PATH="$RUN_DIR/manifest_task_${TASK_ID}.txt"
fi

CLEAN_RESULT_ACTION="not_applicable"
if [[ "$JOB_KIND" == "clean" ]]; then
  CLEAN_RESULT_ACTION="create"
  if [[ -e "$RESULT_JSONL" ]]; then
    if [[ "$OVERWRITE_CLEAN" == "1" ]]; then
      CLEAN_RESULT_ACTION="overwrite"
    elif clean_result_is_reusable \
      "$RESULT_JSONL" "$DATASET" "$STYLE" "$METHOD" "$R" "$SEED" "neutral" "$NUM_SAMPLES"; then
      CLEAN_RESULT_ACTION="reuse"
    else
      die "Existing canonical clean JSONL is incomplete, incompatible, or lacks provenance: $RESULT_JSONL. Set OVERWRITE_CLEAN=1 to replace it."
    fi
  fi
fi

if [[ "$CLEAN_RESULT_ACTION" == "reuse" ]]; then
  echo "[experiment_c] reusing complete compatible canonical clean JSONL: $RESULT_JSONL"
  exit 0
fi

mkdir -p "$RUN_DIR" "$LOG_DIR"

echo "[experiment_c] out_root=$OUT_ROOT"
echo "[experiment_c] task_id=$TASK_ID/$TOTAL_TASKS"
echo "[experiment_c] job_kind=$JOB_KIND clean_role=${CLEAN_ROLE:-<none>}"
echo "[experiment_c] overwrite_clean=$OVERWRITE_CLEAN clean_result_action=$CLEAN_RESULT_ACTION"
echo "[experiment_c] style=$STYLE method=$METHOD"
echo "[experiment_c] datasets=$DATASETS"
echo "[experiment_c] sites=$SITES"
echo "[experiment_c] epsilons=$EPSILONS"
echo "[experiment_c] rounds=$ROUNDS"
echo "[experiment_c] lc_rounds=$LC_ROUNDS (zero-based; p2c/c2s=0..R-1, s2p=0..R-2)"
echo "[experiment_c] seeds=$SEEDS"
echo "[experiment_c] skipped_invalid_lc_configs=$SKIPPED_INVALID_CONFIGS"
echo "[experiment_c] lc_mode=$LC_MODE_EFFECTIVE lc_round=${LC_ROUND_EFFECTIVE:-<none>} lc_direction=$LC_DIRECTION_EFFECTIVE run_subdir=$RUN_SUBDIR"
echo "[experiment_c] selected_steering_bank=$LC_STEERING_BANK_EFFECTIVE"
echo "[experiment_c] lc_steering_method=$LC_STEERING_METHOD selected_steering_id=$LC_STEERING_ID_EFFECTIVE"
echo "[experiment_c] selected dataset=$DATASET site=${SITE:-<none>} eps=$EPS rounds=$R lc_round=${LC_ROUND_EFFECTIVE:-<none>} seed=$SEED"
echo "[experiment_c] num_samples=$NUM_SAMPLES batch_size=$BATCH_SIZE latent_length=$LATENT_LENGTH"
echo "[experiment_c] gpu_list=${GPU_LIST:-<empty>}"
echo "[experiment_c] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

echo "===== nvidia-smi -L ====="
nvidia-smi -L || true
echo "===== initial nvidia-smi ====="
nvidia-smi || true

{
  echo "style=$STYLE"
  echo "method=$METHOD"
  echo "datasets=$DATASETS"
  echo "sites=$SITES"
  echo "epsilons=$EPSILONS"
  echo "rounds=$ROUNDS"
  echo "lc_rounds=$LC_ROUNDS"
  echo "lc_round_index_base=0"
  echo "seeds=$SEEDS"
  echo "lc_mode=$LC_MODE_EFFECTIVE"
  echo "lc_round=$LC_ROUND_EFFECTIVE"
  echo "lc_direction=$LC_DIRECTION_EFFECTIVE"
  echo "steering_filter=$STEERING_FILTER"
  echo "lc_steering_bank_override=$LC_STEERING_BANK"
  echo "lc_steering_bank=$LC_STEERING_BANK_EFFECTIVE"
  echo "lc_steering_method=$LC_STEERING_METHOD"
  echo "lc_steering_id_override=$LC_STEERING_ID"
  echo "lc_steering_id=$LC_STEERING_ID_EFFECTIVE"
  echo "run_subdir=$RUN_SUBDIR"
  echo "job_kind=$JOB_KIND"
  echo "clean_role=$CLEAN_ROLE"
  echo "overwrite_clean=$OVERWRITE_CLEAN"
  echo "clean_result_action=$CLEAN_RESULT_ACTION"
  echo "task_id=$TASK_ID"
  echo "total_tasks=$TOTAL_TASKS"
  echo "skipped_invalid_lc_configs=$SKIPPED_INVALID_CONFIGS"
  echo "selected_dataset=$DATASET"
  echo "selected_site=$SITE"
  echo "selected_epsilon=$EPS"
  echo "selected_rounds=$R"
  echo "selected_recursive_R=$R"
  echo "selected_lc_round=$LC_ROUND_EFFECTIVE"
  echo "lc_round_effective=$LC_ROUND_EFFECTIVE"
  echo "selected_steering_bank=$LC_STEERING_BANK_EFFECTIVE"
  echo "selected_steering_id=$LC_STEERING_ID_EFFECTIVE"
  echo "selected_seed=$SEED"
  echo "num_samples=$NUM_SAMPLES"
  echo "batch_size=$BATCH_SIZE"
  echo "latent_length=$LATENT_LENGTH"
  echo "trust_remote_code=$TRUST_REMOTE_CODE"
  echo "gpu_list=$GPU_LIST"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "extra_args=$EXTRA_ARGS"
} > "$MANIFEST_PATH"

cmd=(
  "$PYTHON_BIN" RecursiveMAS/run.py
)
if [[ -n "$EXTRA_ARGS" ]]; then
  read -r -a extra_args_array <<< "$EXTRA_ARGS"
  cmd+=("${extra_args_array[@]}")
fi
cmd+=(
  --style "$STYLE"
  --dataset "$DATASET"
  --method "$METHOD"
  --num_recursive_rounds "$R"
  --num_samples "$NUM_SAMPLES"
  --batch_size "$BATCH_SIZE"
  --latent_length "$LATENT_LENGTH"
  --seed "$SEED"
  --trust_remote_code "$TRUST_REMOTE_CODE"
  --deterministic 1
  --role_response_regime neutral
  --role_response_regime_path ""
  --result_jsonl "$RESULT_JSONL"
)

if [[ "$JOB_KIND" == "attack" ]]; then
  cmd+=(
    --lc_mode "$LC_MODE"
    --lc_site "$SITE"
    --lc_epsilon "$EPS"
    --lc_round "$LC_ROUND_EFFECTIVE"
    --lc_seed "$SEED"
    --lc_direction "$LC_DIRECTION"
  )
  if [[ -n "$LC_STEERING_BANK_EFFECTIVE" ]]; then
    cmd+=(--lc_steering_bank "$LC_STEERING_BANK_EFFECTIVE")
  fi
  if [[ -n "$LC_STEERING_METHOD" ]]; then
    cmd+=(--lc_steering_method "$LC_STEERING_METHOD")
  fi
  if [[ -n "$LC_STEERING_ID_EFFECTIVE" ]]; then
    cmd+=(--lc_steering_id "$LC_STEERING_ID_EFFECTIVE")
  fi
fi
if [[ -n "${SAMPLE_SEED:-}" ]]; then
  cmd+=(--sample_seed "$SAMPLE_SEED")
fi
if [[ -n "${TEMPERATURE:-}" ]]; then
  cmd+=(--temperature "$TEMPERATURE")
fi
if [[ -n "${TOP_P:-}" ]]; then
  cmd+=(--top_p "$TOP_P")
fi
if [[ -n "${TOP_K:-}" ]]; then
  cmd+=(--top_k "$TOP_K")
fi
if [[ -n "${DEVICE:-}" ]]; then
  cmd+=(--device "$DEVICE")
fi
if [[ "$JOB_KIND" == "clean" ]]; then
  # Keep both fixed controls attack-free even if EXTRA_ARGS has overrides.
  cmd+=(
    --deterministic 1
    --question_suffix_path ""
    --prompt_footer_path ""
    --lc_mode none --lc_epsilon 0 --lc_seed "$SEED" --lc_direction random
    --role_profile_probe_mode none --role_profile_probe_target none
    --role_profile_probe_site none --role_profile_probe_round -1
    --role_profile_epsilon 0 --role_profile_seed "$SEED"
  )
fi

echo
if [[ "$JOB_KIND" == "clean" ]]; then
  echo "===== $DATASET :: experiment_c clean role=$CLEAN_ROLE R=$R seed=$SEED ====="
else
  echo "===== $DATASET :: experiment_c $LC_MODE site=$SITE eps=$EPS R=$R lc_round=$LC_ROUND_EFFECTIVE seed=$SEED ====="
fi
echo "[experiment_c] result_jsonl=$RESULT_JSONL"
echo "[experiment_c] run_log=$RUN_LOG"
printf '[experiment_c] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}" 2>&1 | tee "$RUN_LOG"

echo
echo "[experiment_c] complete. JSONL log: $RESULT_JSONL"
