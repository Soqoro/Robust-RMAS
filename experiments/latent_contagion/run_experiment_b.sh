#!/bin/bash
#SBATCH --job-name=latent-contagion-b
#SBATCH -p NA100q
#SBATCH -w node01
#SBATCH --output=logs/latent_contagion_b_%A_%a.out
#SBATCH --error=logs/latent_contagion_b_%A_%a.err

# Experiment B: one-shot latent contagion phase diagram.
#
# The default grid contains two fixed clean runs (reference and control) for
# every DATASET/R/SEED plus all positive-epsilon attack configurations. Zero
# entries in EPSILONS are accepted for compatibility but are not attack jobs.
# Total jobs with defaults = 2 * (2 datasets * 5 rounds * 1 seed)
#                          + 2 datasets * 3 sites * 7 positive eps * 5 rounds
#                          = 230.
# Print the exact grid and total before submitting:
#   B_STAGE=grid bash experiments/latent_contagion/run_experiment_b.sh
# Complete canonical clean JSONLs are reused. After changing code, models, or
# generation/evaluation settings, use a fresh OUT_ROOT (recommended) or set
# OVERWRITE_CLEAN=1 and rerun the matching attack jobs.
# Local smoke:
#   SLURM_ARRAY_TASK_ID=0 NUM_SAMPLES=2 bash experiments/latent_contagion/run_experiment_b.sh
# Slurm:
#   sbatch --array=0-229 experiments/latent_contagion/run_experiment_b.sh
# Aggregate one dataset after the array completes (repeat for each dataset):
#   python experiments/latent_contagion/aggregate_latent_contagion.py \
#     --root outputs/latent_contagion/experiment_b --dataset math500 --subdir oneshot \
#     --clean_reference_root outputs/latent_contagion/experiment_b/clean/reference \
#     --clean_control_root outputs/latent_contagion/experiment_b/clean/control

set -euo pipefail

mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-python}"
STYLE="${STYLE:-sequential_light}"
METHOD="${METHOD:-ours_recursive}"
DATASETS="${DATASETS:-math500 gpqa}"
SITES="${SITES:-p2c c2s s2p}"
EPSILONS="${EPSILONS:-0 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1}"
ROUNDS="${ROUNDS:-1 2 3 4 5}"
SEEDS="${SEEDS:-42}"
LC_MODE="${LC_MODE:-one_shot}"
LC_ROUND="${LC_ROUND:-0}"
LC_DIRECTION="${LC_DIRECTION:-random}"
LC_STEERING_BANK="${LC_STEERING_BANK:-}"
LC_STEERING_METHOD="${LC_STEERING_METHOD:-}"
LC_STEERING_ID="${LC_STEERING_ID:-}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LATENT_LENGTH="${LATENT_LENGTH:-48}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
OUT_ROOT="${OUT_ROOT:-${OUT_DIR:-outputs/latent_contagion/experiment_b}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
B_STAGE="${B_STAGE:-run}"
OVERWRITE_CLEAN="${OVERWRITE_CLEAN:-0}"

if [[ -z "${RUN_SUBDIR:-}" ]]; then
  if [[ "$LC_MODE" == "one_shot" ]]; then
    RUN_SUBDIR="oneshot"
  else
    RUN_SUBDIR="$LC_MODE"
  fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export CONDA_NO_PLUGINS=true
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

echo "Using TMPDIR=$TMPDIR"
mkdir -p "$TMPDIR" || true
ls -ld "$TMPDIR" || true

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]]; then
  echo "[error] SLURM_ARRAY_TASK_ID must be a non-negative integer, got: $TASK_ID" >&2
  exit 2
fi

die() {
  echo "[error] $*" >&2
  exit 2
}

case "$B_STAGE" in
  run|grid) ;;
  *) die "B_STAGE must be run or grid. Got: $B_STAGE" ;;
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

total_jobs() {
  local total=0
  local clean_role dataset site eps rounds seed
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
          for seed in $SEEDS; do
            total=$((total + 1))
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
  local clean_role dataset site eps rounds seed
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
          for seed in $SEEDS; do
            if (( index == target )); then
              JOB_KIND="attack"
              CLEAN_ROLE=""
              DATASET="$dataset"
              SITE="$site"
              EPS="$eps"
              R="$rounds"
              SEED="$seed"
              return 0
            fi
            index=$((index + 1))
          done
        done
      done
    done
  done
  return 1
}

print_grid() {
  local index=0
  local clean_role dataset site eps rounds seed
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
          for seed in $SEEDS; do
            echo "$index attack - $dataset $site $eps $rounds $LC_ROUND $seed"
            index=$((index + 1))
          done
        done
      done
    done
  done
}

TOTAL_TASKS="$(total_jobs)"
if [[ "$B_STAGE" == "grid" ]]; then
  print_grid
  exit 0
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
SEED=""
select_config "$TASK_ID"

if [[ "$JOB_KIND" == "clean" ]]; then
  LC_MODE_EFFECTIVE="none"
  LC_ROUND_EFFECTIVE=""
  LC_DIRECTION_EFFECTIVE="random"
  RUN_DIR="$OUT_ROOT/clean/$CLEAN_ROLE/$DATASET/R${R}/seed${SEED}"
  LOG_DIR="$RUN_DIR"
  RESULT_JSONL="$RUN_DIR/result.jsonl"
  RUN_LOG="$RUN_DIR/run.log"
  MANIFEST_PATH="$RUN_DIR/manifest.txt"
else
  LC_MODE_EFFECTIVE="$LC_MODE"
  LC_ROUND_EFFECTIVE="$LC_ROUND"
  LC_DIRECTION_EFFECTIVE="$LC_DIRECTION"
  RUN_DIR="$OUT_ROOT/$DATASET/$RUN_SUBDIR"
  LOG_DIR="$RUN_DIR/logs"
  RESULT_JSONL="$RUN_DIR/site=${SITE}_eps=${EPS}_R=${R}_lc_round=${LC_ROUND}_seed=${SEED}.jsonl"
  RUN_LOG="$LOG_DIR/site=${SITE}_eps=${EPS}_R=${R}_lc_round=${LC_ROUND}_seed=${SEED}.log"
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
  echo "[experiment_b] reusing complete compatible canonical clean JSONL: $RESULT_JSONL"
  exit 0
fi

mkdir -p "$RUN_DIR" "$LOG_DIR"

echo "[experiment_b] out_root=$OUT_ROOT"
echo "[experiment_b] task_id=$TASK_ID/$TOTAL_TASKS"
echo "[experiment_b] job_kind=$JOB_KIND clean_role=${CLEAN_ROLE:-<none>}"
echo "[experiment_b] overwrite_clean=$OVERWRITE_CLEAN clean_result_action=$CLEAN_RESULT_ACTION"
echo "[experiment_b] style=$STYLE method=$METHOD"
echo "[experiment_b] datasets=$DATASETS"
echo "[experiment_b] sites=$SITES"
echo "[experiment_b] epsilons=$EPSILONS"
echo "[experiment_b] rounds=$ROUNDS"
echo "[experiment_b] seeds=$SEEDS"
echo "[experiment_b] lc_mode=$LC_MODE_EFFECTIVE lc_round=${LC_ROUND_EFFECTIVE:-<none>} lc_direction=$LC_DIRECTION_EFFECTIVE run_subdir=$RUN_SUBDIR"
echo "[experiment_b] lc_steering_bank=${LC_STEERING_BANK:-<empty>}"
echo "[experiment_b] lc_steering_method=${LC_STEERING_METHOD:-<empty>} lc_steering_id=${LC_STEERING_ID:-<empty>}"
echo "[experiment_b] selected dataset=$DATASET site=${SITE:-<none>} eps=$EPS rounds=$R seed=$SEED"
echo "[experiment_b] num_samples=$NUM_SAMPLES batch_size=$BATCH_SIZE latent_length=$LATENT_LENGTH"
echo "[experiment_b] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

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
  echo "seeds=$SEEDS"
  echo "lc_mode=$LC_MODE_EFFECTIVE"
  echo "lc_round=$LC_ROUND_EFFECTIVE"
  echo "lc_direction=$LC_DIRECTION_EFFECTIVE"
  echo "lc_steering_bank=$LC_STEERING_BANK"
  echo "lc_steering_method=$LC_STEERING_METHOD"
  echo "lc_steering_id=$LC_STEERING_ID"
  echo "run_subdir=$RUN_SUBDIR"
  echo "job_kind=$JOB_KIND"
  echo "clean_role=$CLEAN_ROLE"
  echo "overwrite_clean=$OVERWRITE_CLEAN"
  echo "clean_result_action=$CLEAN_RESULT_ACTION"
  echo "task_id=$TASK_ID"
  echo "total_tasks=$TOTAL_TASKS"
  echo "selected_dataset=$DATASET"
  echo "selected_site=$SITE"
  echo "selected_epsilon=$EPS"
  echo "selected_rounds=$R"
  echo "selected_seed=$SEED"
  echo "num_samples=$NUM_SAMPLES"
  echo "batch_size=$BATCH_SIZE"
  echo "latent_length=$LATENT_LENGTH"
  echo "trust_remote_code=$TRUST_REMOTE_CODE"
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
    --lc_round "$LC_ROUND"
    --lc_seed "$SEED"
    --lc_direction "$LC_DIRECTION"
  )
  if [[ -n "$LC_STEERING_BANK" ]]; then
    cmd+=(--lc_steering_bank "$LC_STEERING_BANK")
  fi
  if [[ -n "$LC_STEERING_METHOD" ]]; then
    cmd+=(--lc_steering_method "$LC_STEERING_METHOD")
  fi
  if [[ -n "$LC_STEERING_ID" ]]; then
    cmd+=(--lc_steering_id "$LC_STEERING_ID")
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
  # Keep clean controls attack-free even if EXTRA_ARGS contains overrides.
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
  echo "===== $DATASET :: clean role=$CLEAN_ROLE R=$R seed=$SEED ====="
else
  echo "===== $DATASET :: $LC_MODE site=$SITE eps=$EPS R=$R lc_round=$LC_ROUND seed=$SEED ====="
fi
echo "[experiment_b] result_jsonl=$RESULT_JSONL"
echo "[experiment_b] run_log=$RUN_LOG"
printf '[experiment_b] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}" 2>&1 | tee "$RUN_LOG"

echo
echo "[experiment_b] complete. JSONL log: $RESULT_JSONL"
