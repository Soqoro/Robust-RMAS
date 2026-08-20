#!/bin/bash
# Shared, source-only launcher for the six LinkRadius entrypoints.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "[error] linkradius_common.sh must be sourced by a LinkRadius entrypoint" >&2
  exit 2
fi

LR_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LR_REPO_ROOT="$(cd "$LR_SCRIPT_DIR/../.." && pwd)"
if [[ ! -f "$LR_REPO_ROOT/RecursiveMAS/run.py" ]]; then
  echo "[error] could not resolve RecursiveMAS repository root from $LR_SCRIPT_DIR" >&2
  return 2
fi
cd "$LR_REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
STYLE="${STYLE:-sequential_light}"
METHOD="${METHOD:-ours_recursive}"
DATASETS="${DATASETS:-${DATASET:-gpqa}}"
ROUNDS="${ROUNDS:-2}"
SEEDS="${SEEDS:-42}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LATENT_LENGTH="${LATENT_LENGTH:-32}"
OUT_ROOT="${OUT_ROOT:-outputs/linkradius}"
GPU_LIST="${GPU_LIST:-}"
DEVICE="${DEVICE:-cuda:0}"
PLANNER_DEVICE="${PLANNER_DEVICE:-}"
CRITIC_DEVICE="${CRITIC_DEVICE:-}"
SOLVER_DEVICE="${SOLVER_DEVICE:-}"
TERMINAL_SOLVER_DEVICE="${TERMINAL_SOLVER_DEVICE:-}"
RELAY_TRANSFER_MODE="${RELAY_TRANSFER_MODE:-cpu_staged}"
AUTOGRAD_MEMORY_MODE="${AUTOGRAD_MEMORY_MODE:-none}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
PROBE_RADII="${PROBE_RADII:-1e-3 3e-3}"
PROBE_SEEDS="${PROBE_SEEDS:-101 202}"
K="${K:-8}"
SUBSPACE="${SUBSPACE:-full_tensor}"
NUM_BATCHES="${NUM_BATCHES:-1}"
DISCOVERY_BATCHES="${DISCOVERY_BATCHES:-20}"
PARTITIONS="${PARTITIONS:-}"
ATTACK_FAMILIES="${ATTACK_FAMILIES:-random_independent pgd_autograd}"
ATTACK_EPSILONS="${ATTACK_EPSILONS:-1e-3 3e-3 1e-2}"
REUSE_COMPLETE="${REUSE_COMPLETE:-1}"
OVERWRITE="${OVERWRITE:-0}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$OUT_ROOT/split_manifest.json}"
EXECUTION_MANIFEST="${EXECUTION_MANIFEST:-}"
TRAJECTORY="${TRAJECTORY:-}"
LEGACY_EQUIVALENCE="${LEGACY_EQUIVALENCE:-}"
SCREENING_JSONL="${SCREENING_JSONL:-}"
MAX_ELIGIBLE="${MAX_ELIGIBLE:-0}"
BATCH_COUNTS="${BATCH_COUNTS:-}"
ENGINEERING_GATE="${ENGINEERING_GATE:-$OUT_ROOT/engineering_gate.json}"
SMOKE_GATE="${SMOKE_GATE:-$OUT_ROOT/smoke_gate.json}"
PROBE_GATE="${PROBE_GATE:-$OUT_ROOT/probe_gate.json}"
ATTACK_FREEZE_GATE="${ATTACK_FREEZE_GATE:-$OUT_ROOT/attack_freeze_gate.json}"
ATTACK_VALIDATION_GATE="${ATTACK_VALIDATION_GATE:-$OUT_ROOT/attack_validation_gate.json}"
PILOT_GATE="${PILOT_GATE:-$OUT_ROOT/pilot_gate.json}"
FROZEN_CONFIG="${FROZEN_CONFIG:-$OUT_ROOT/frozen_config.json}"
FROZEN_ATTACK_CONFIG="${FROZEN_ATTACK_CONFIG:-$OUT_ROOT/frozen_attack_config.json}"
AGGREGATE_PHASE="${AGGREGATE_PHASE:-smoke}"
case "$AGGREGATE_PHASE" in
  engineering) LR_REQUIRED_VERIFY_STAGES="split discover freeze_execution clean replay probe gradient validate" ;;
  smoke) LR_REQUIRED_VERIFY_STAGES="split screen freeze_execution clean causal probe gradient attack estimate aggregate validate" ;;
  pilot) LR_REQUIRED_VERIFY_STAGES="split screen_clean freeze_execution clean causal probe_calibration gradient freeze_probe validate_probe aggregate" ;;
  attacks) LR_REQUIRED_VERIFY_STAGES="train val freeze_attack test_probe test thresholds analyze validate" ;;
  *) LR_REQUIRED_VERIFY_STAGES="" ;;
esac
VERIFY_STAGES="${VERIFY_STAGES:-$LR_REQUIRED_VERIFY_STAGES}"
GRID_TARGET_STAGE="${GRID_TARGET_STAGE:-}"

export CONDA_NO_PLUGINS=true
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

lr_die() {
  echo "[error] $*" >&2
  return 2
}

lr_validate_nonnegative() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    lr_die "$name must be a non-negative integer, got: $value"
  fi
}

lr_validate_positive() {
  local name="$1"
  local value="$2"
  lr_validate_nonnegative "$name" "$value" || return
  if (( value < 1 )); then
    lr_die "$name must be positive, got: $value"
  fi
}

lr_validate_bool() {
  case "$2" in
    0|1) ;;
    *) lr_die "$1 must be 0 or 1, got: $2" ;;
  esac
}

lr_validate_relay_transfer_mode() {
  case "$RELAY_TRANSFER_MODE" in
    direct|cpu_staged) ;;
    *) lr_die "RELAY_TRANSFER_MODE must be direct or cpu_staged, got: $RELAY_TRANSFER_MODE" ;;
  esac
}

lr_validate_autograd_memory_mode() {
  case "$AUTOGRAD_MEMORY_MODE" in
    none|checkpoint) ;;
    *) lr_die "AUTOGRAD_MEMORY_MODE must be none or checkpoint, got: $AUTOGRAD_MEMORY_MODE" ;;
  esac
}

lr_validate_stage() {
  local selected="$1"
  shift
  local allowed
  for allowed in "$@"; do
    if [[ "$selected" == "$allowed" ]]; then
      return 0
    fi
  done
  lr_die "unsupported LR_STAGE=$selected; allowed: $*"
}

lr_is_gpu_stage() {
  local workflow="$1"
  local stage="$2"
  # The aggregate entrypoint is categorically CPU-only even when a summary
  # stage shares a name (for example, "causal") with a GPU experiment stage.
  if [[ "$workflow" == "aggregate" ]]; then
    return 1
  fi
  case "$stage" in
    discover|screen|screen_clean|clean|replay|causal|probe|probe_calibration|gradient|attack|train|val|test_probe|test) return 0 ;;
    *) return 1 ;;
  esac
}

lr_validate_gpu_configuration() {
  if [[ -n "$GPU_LIST" ]] && {
    [[ -n "$PLANNER_DEVICE" ]] || [[ -n "$CRITIC_DEVICE" ]] || [[ -n "$SOLVER_DEVICE" ]] || [[ -n "$TERMINAL_SOLVER_DEVICE" ]]
  }; then
    lr_die "GPU_LIST masks each array task to one GPU and cannot be combined with per-role devices; leave GPU_LIST empty and request all role GPUs in one scheduler allocation"
    return
  fi
}

lr_configure_gpu() {
  local task_id="$1"
  lr_validate_gpu_configuration || return
  if [[ -n "$GPU_LIST" ]]; then
    local -a gpu_values
    read -r -a gpu_values <<< "$GPU_LIST"
    if (( ${#gpu_values[@]} == 0 )); then
      lr_die "GPU_LIST was supplied but contained no device IDs"
    fi
    local gpu
    for gpu in "${gpu_values[@]}"; do
      lr_validate_nonnegative "GPU_LIST item" "$gpu" || return
    done
    export CUDA_VISIBLE_DEVICES="${gpu_values[$((task_id % ${#gpu_values[@]}))]}"
  fi
  # If GPU_LIST is empty, preserve the scheduler-provided visibility mask.
}

lr_build_command() {
  local workflow="$1"
  local stage="$2"
  local task_id="$3"
  LR_COMMAND=("$PYTHON_BIN" -m experiments.linkradius.run_linkradius)
  if [[ -n "$EXTRA_ARGS" ]]; then
    local -a extras
    read -r -a extras <<< "$EXTRA_ARGS"
    LR_COMMAND+=("${extras[@]}")
  fi
  # Authoritative experiment flags deliberately follow optional extras.
  LR_COMMAND+=(
    --workflow "$workflow"
    --stage "$stage"
    --task-id "$task_id"
    --style "$STYLE"
    --method "$METHOD"
    --datasets "$DATASETS"
    --rounds "$ROUNDS"
    --seeds "$SEEDS"
    --batch-size "$BATCH_SIZE"
    --latent-length "$LATENT_LENGTH"
    --out-root "$OUT_ROOT"
    --split-manifest "$SPLIT_MANIFEST"
    --num-batches "$NUM_BATCHES"
    --discovery-batches "$DISCOVERY_BATCHES"
    --probe-radii "$PROBE_RADII"
    --probe-seeds "$PROBE_SEEDS"
    --K "$K"
    --subspace "$SUBSPACE"
    --attack-families "$ATTACK_FAMILIES"
    --attack-epsilons "$ATTACK_EPSILONS"
    --reuse-complete "$REUSE_COMPLETE"
    --overwrite "$OVERWRITE"
    --max-eligible "$MAX_ELIGIBLE"
    --device "$DEVICE"
    --planner-device "$PLANNER_DEVICE"
    --critic-device "$CRITIC_DEVICE"
    --solver-device "$SOLVER_DEVICE"
    --terminal-solver-device "$TERMINAL_SOLVER_DEVICE"
    --relay-transfer-mode "$RELAY_TRANSFER_MODE"
    --autograd-memory-mode "$AUTOGRAD_MEMORY_MODE"
    --engineering-gate "$ENGINEERING_GATE"
    --smoke-gate "$SMOKE_GATE"
    --probe-gate "$PROBE_GATE"
    --attack-freeze-gate "$ATTACK_FREEZE_GATE"
    --attack-validation-gate "$ATTACK_VALIDATION_GATE"
    --pilot-gate "$PILOT_GATE"
    --frozen-config "$FROZEN_CONFIG"
    --frozen-attack-config "$FROZEN_ATTACK_CONFIG"
    --aggregate-phase "$AGGREGATE_PHASE"
    --verify-stages "$VERIFY_STAGES"
  )
  if [[ -n "$PARTITIONS" ]]; then
    LR_COMMAND+=(--partitions "$PARTITIONS")
  fi
  if [[ -n "$BATCH_COUNTS" ]]; then
    local batch_count
    for batch_count in $BATCH_COUNTS; do
      LR_COMMAND+=(--batch-count "$batch_count")
    done
  fi
  if [[ -n "$GRID_TARGET_STAGE" ]]; then
    LR_COMMAND+=(--grid-target-stage "$GRID_TARGET_STAGE")
  fi
  if [[ -n "$EXECUTION_MANIFEST" ]]; then
    LR_COMMAND+=(--execution-manifest-path "$EXECUTION_MANIFEST")
  fi
  if [[ -n "$TRAJECTORY" ]]; then
    LR_COMMAND+=(--trajectory "$TRAJECTORY")
  fi
  if [[ -n "$LEGACY_EQUIVALENCE" ]]; then
    LR_COMMAND+=(--legacy-equivalence "$LEGACY_EQUIVALENCE")
  fi
  if [[ -n "$SCREENING_JSONL" ]]; then
    local screening_path
    for screening_path in $SCREENING_JSONL; do
      LR_COMMAND+=(--screening-jsonl "$screening_path")
    done
  fi
}

lr_run_entrypoint() {
  local workflow="$1"
  local stage="$2"
  local task_id="${SLURM_ARRAY_TASK_ID:-0}"
  lr_validate_nonnegative "SLURM_ARRAY_TASK_ID" "$task_id" || return
  lr_validate_positive "BATCH_SIZE" "$BATCH_SIZE" || return
  lr_validate_positive "LATENT_LENGTH" "$LATENT_LENGTH" || return
  lr_validate_positive "K" "$K" || return
  lr_validate_bool "REUSE_COMPLETE" "$REUSE_COMPLETE" || return
  lr_validate_bool "OVERWRITE" "$OVERWRITE" || return
  lr_validate_relay_transfer_mode || return
  lr_validate_autograd_memory_mode || return
  lr_validate_gpu_configuration || return

  lr_build_command "$workflow" "$stage" "$task_id"
  if [[ "$stage" == "grid" || "$stage" == *_grid ]]; then
    "${LR_COMMAND[@]}" --grid-format tsv
    return
  fi

  local defer_completion=1
  if [[ "$stage" == "all" ]]; then
    defer_completion=0
  else
    LR_COMMAND+=(--defer-completion)
  fi

  lr_configure_gpu "$task_id" || return
  if lr_is_gpu_stage "$workflow" "$stage"; then
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
  fi

  local task_dir
  task_dir="$("${LR_COMMAND[@]}" --print-output-dir)"
  mkdir -p "$task_dir"
  local launcher_path="$task_dir/launcher_command.txt"
  local log_path="$task_dir/run.log"
  if (( defer_completion )); then
    launcher_path="$task_dir/.launcher_command.pending.txt"
    log_path="$task_dir/.run.log.pending"
  fi
  printf '%q ' "${LR_COMMAND[@]}" > "$launcher_path"
  printf '\n' >> "$launcher_path"
  set -o pipefail
  "${LR_COMMAND[@]}" 2>&1 | tee "$log_path"
  if (( defer_completion )); then
    "$PYTHON_BIN" -m experiments.linkradius.run_linkradius \
      --finalize-completion-dir "$task_dir"
  fi
}
