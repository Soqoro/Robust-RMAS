# LinkRadius experiments

This package implements the staged, auditable LinkRadius pipeline for the
released `sequential_light` baseline: Qwen3-1.7B planner,
Llama-3.2-1B critic, Qwen2.5-Math-1.5B solver, and learned `p2c`, `c2s`, and
`s2p` outer adapters. The GPQA release settings are seed 42, batch size 16,
latent length 32, source split `train`, deterministic generation, and
`choice_old_prompt=2`.

The reference relay is always the clean post-consumer-cast tensor, stored in
float32. Live transport dtypes are preserved. Probe derivatives use realized
signed separation after the consumer cast, and the primary K-direction
estimate exists only when every direction in `0..K-1` passes the frozen pair
checks. All learned banks use `attack_train`; probe/scorer/edge choices use
`validation`; Phase 3 refuses test outcomes.

The exact sequential edge set is `p2c@r,c2s@r` for `0 <= r < R` plus `s2p@r`
for `0 <= r < R-1`. Thus R=2 has exactly five edges and never `s2p@1`.

## Before running

Direct `bash` commands may run from anywhere; each script resolves and enters
the repository root, unsets `PYTHONPATH`, and invokes the package with
`python -m`. Because SLURM executes a spool copy of an `sbatch` script, submit
from the repository root so `SLURM_SUBMIT_DIR` identifies the checkout. If that
is not possible, export `LINKRADIUS_REPO_ROOT=/absolute/path/to/Robust-RMAS`
before submission. Create the SLURM log directory **before** submission because
SLURM opens output files before the script starts:

```bash
cd /path/to/Robust-RMAS
mkdir -p logs
```

Important environment variables are `PYTHON_BIN`, `LINKRADIUS_REPO_ROOT`, `STYLE`, `METHOD`,
`DATASETS`, `ROUNDS`, `SEEDS`, `BATCH_SIZE`, `LATENT_LENGTH`, `OUT_ROOT`,
`GPU_LIST`, `PLANNER_DEVICE`, `CRITIC_DEVICE`, `SOLVER_DEVICE`, `PROBE_RADII`,
`PROBE_SEEDS`, `K`, and `SUBSPACE`. `GPU_LIST` is a whitespace-separated
physical-device list for round-robin, one-GPU-per-array-task execution. If it
is empty, scheduler-provided `CUDA_VISIBLE_DEVICES` is preserved; inside a
masked job the runtime uses logical CUDA indices.

### Multi-GPU sequential model placement

The persistent sequential runtime can place its three agents on different
logical devices. For example:

```bash
export PLANNER_DEVICE=cuda:0
export CRITIC_DEVICE=cuda:1
export SOLVER_DEVICE=cuda:2
unset GPU_LIST
```

This places the planner, critic, and solver (plus each role's inner adapter) on
three devices. Each outer adapter stays on its source role's device, and its
relay is copied directly to the consumer device without detaching the autograd
graph. The CUDA indices are logical indices within the scheduler's
`CUDA_VISIBLE_DEVICES`, not necessarily physical GPU ordinals.

Run one process in one Slurm task and request three GPUs, for example with
`--ntasks=1 --gres=gpu:3` when those are the site's allocation flags. Keep
`GPU_LIST` empty: `GPU_LIST="0 1 2"` still means round-robin single-GPU array
routing and does not pool memory. On a four-GPU node, throttle a two-element
gradient array with `--array=0-1%1` so only one three-GPU element runs at once.

The resolved role topology is authenticated experiment configuration. Export
the same `STYLE`, `LATENT_LENGTH`, `PLANNER_DEVICE`, `CRITIC_DEVICE`, and
`SOLVER_DEVICE` values for every command in a fresh workflow, including CPU
freeze/validation/aggregation commands that reconstruct upstream task keys.
Changing only the physical node is safe when the same three logical devices
remain available. Changing the logical role map requires a new output root.

Differentiable gradient and PGD objectives use the exact frozen
`gold_score - target_score` margin but evaluate and backpropagate the gold and
target solver candidates sequentially. This keeps at most one differentiable
terminal scorer graph resident at a time. Ordinary clean, replay,
finite-difference, and final PGD scoring remains the complete four-way A/B/C/D
scorer; the memory optimization does not change reported choice scores or
candidate selection.

Every array task is a whole frozen execution batch plus one intervention
configuration. A probe task contains both signs for all nested directions, so
the scheduler never splits an antithetic pair. Outputs are written beside a
temporary file and atomically renamed. Reuse occurs only when `.complete.json`
validates the config hash, source hash, artifact hashes, and row counts.

## Phase 1: engineering

The default discovery grid examines twenty validation candidates one at a time
and freezes the first dual-correct row. The array bounds below are the exact
defaults; if you change a grid variable, regenerate and inspect the canonical
mapping before submission. Run the stages in order, waiting for each array and
manually inspecting failures before advancing:

```bash
LR_STAGE=split bash experiments/linkradius/run_linkradius_engineering.sh
LR_STAGE=discover sbatch --array=0-19 experiments/linkradius/run_linkradius_engineering.sh
LR_STAGE=freeze_execution bash experiments/linkradius/run_linkradius_engineering.sh
LR_STAGE=clean bash experiments/linkradius/run_linkradius_engineering.sh
LR_STAGE=replay sbatch --array=0-9 experiments/linkradius/run_linkradius_engineering.sh
LR_STAGE=grid bash experiments/linkradius/run_linkradius_engineering.sh       # probe: 2 tasks, max 1
LR_STAGE=probe sbatch --array=0-1 experiments/linkradius/run_linkradius_engineering.sh
LR_STAGE=gradient sbatch --array=0-1 experiments/linkradius/run_linkradius_engineering.sh
```

`LR_STAGE=grid` is the engineering probe mapping. The other default array sizes
are discovery 20, replay 10, and gradient 2; they are generated by the same
canonical Python grid used for task selection. `LR_STAGE=all` is a local,
sequential convenience for this small workflow; it never submits jobs and is
not a substitute for inspecting the real GPU checks.

Validation writes:

```text
OUT_ROOT/engineering_report.json
OUT_ROOT/engineering_gate.json
```

The gate fails if any real-data invariant is absent, including an authenticated
legacy-equivalence artifact. After the engineering `clean` stage, set
`CLEAN_TRAJECTORY` to its completed `clean_trajectory.pt` and set `RAW_INDEX`
to the trajectory's recorded `raw_indices[0]`. Then produce the comparison with
the release runner itself:

```bash
export PYTHON_BIN="${PYTHON_BIN:-python}"
export OUT_ROOT="${OUT_ROOT:-outputs/linkradius}"
export CLEAN_TRAJECTORY=/absolute/path/to/clean_trajectory.pt
export RAW_INDEX=17                         # use the recorded value, not this example
export LEGACY_RESULTS="$OUT_ROOT/legacy_release.jsonl"
export LEGACY_TRACE="$OUT_ROOT/legacy_release_trace.pt"

"$PYTHON_BIN" RecursiveMAS/run.py \
  --style sequential_light \
  --dataset gpqa --dataset_split train \
  --method ours_recursive --num_recursive_rounds 2 \
  --num_samples 1 --sample_indices "$RAW_INDEX" \
  --batch_size 1 --latent_length 32 \
  --seed 42 --deterministic 1 --device cuda:0 \
  --result_jsonl "$LEGACY_RESULTS" \
  --lc_trace_path "$LEGACY_TRACE" \
  --lc_trace_sites p2c,c2s,s2p \
  --lc_trace_rounds 0,1 --lc_trace_dtype float32

"$PYTHON_BIN" -m experiments.linkradius.compare_legacy_equivalence \
  --trajectory "$CLEAN_TRAJECTORY" \
  --legacy-trace "$LEGACY_TRACE" \
  --legacy-results "$LEGACY_RESULTS" \
  --output "$OUT_ROOT/legacy_equivalence.json"

LEGACY_EQUIVALENCE="$OUT_ROOT/legacy_equivalence.json" \
  LR_STAGE=validate bash experiments/linkradius/run_linkradius_engineering.sh
```

The comparator reopens and hashes all three artifacts, recomputes the exact
generation/strict-choice/five-relay checks, and binds the report to the current
source tree. Regenerate it after any result-affecting source change. CPU toy
checks are not accepted as a substitute for this audit or for real terminal
finite differences.

## Phase 2: smoke

Smoke requires the passed engineering gate. It screens fixed validation
batches, keeps their filler rows and boundaries, and marks 10--20 rows eligible
(default 16).

```bash
LR_STAGE=split bash experiments/linkradius/run_linkradius_smoke.sh
LR_STAGE=screen sbatch --array=0-1 experiments/linkradius/run_linkradius_smoke.sh
LR_STAGE=freeze_execution bash experiments/linkradius/run_linkradius_smoke.sh
LR_STAGE=clean sbatch --array=0-1 experiments/linkradius/run_linkradius_smoke.sh
LR_STAGE=causal_grid bash experiments/linkradius/run_linkradius_smoke.sh
SMOKE_CAUSAL_MAX="$(LR_STAGE=causal_grid bash experiments/linkradius/run_linkradius_smoke.sh | awk -F '\t' '$1=="max_array_index" {print $2}')"
LR_STAGE=causal sbatch --array="0-${SMOKE_CAUSAL_MAX}" experiments/linkradius/run_linkradius_smoke.sh
LR_STAGE=probe_grid bash experiments/linkradius/run_linkradius_smoke.sh
SMOKE_PROBE_MAX="$(LR_STAGE=probe_grid bash experiments/linkradius/run_linkradius_smoke.sh | awk -F '\t' '$1=="max_array_index" {print $2}')"
LR_STAGE=probe sbatch --array="0-${SMOKE_PROBE_MAX}" experiments/linkradius/run_linkradius_smoke.sh
LR_STAGE=gradient_grid bash experiments/linkradius/run_linkradius_smoke.sh
SMOKE_GRADIENT_MAX="$(LR_STAGE=gradient_grid bash experiments/linkradius/run_linkradius_smoke.sh | awk -F '\t' '$1=="max_array_index" {print $2}')"
LR_STAGE=gradient sbatch --array="0-${SMOKE_GRADIENT_MAX}" experiments/linkradius/run_linkradius_smoke.sh
LR_STAGE=attack_grid bash experiments/linkradius/run_linkradius_smoke.sh
SMOKE_ATTACK_MAX="$(LR_STAGE=attack_grid bash experiments/linkradius/run_linkradius_smoke.sh | awk -F '\t' '$1=="max_array_index" {print $2}')"
LR_STAGE=attack sbatch --array="0-${SMOKE_ATTACK_MAX}" experiments/linkradius/run_linkradius_smoke.sh
LR_STAGE=estimate bash experiments/linkradius/run_linkradius_smoke.sh
LR_STAGE=aggregate bash experiments/linkradius/run_linkradius_smoke.sh
LR_STAGE=validate bash experiments/linkradius/run_linkradius_smoke.sh
```

The causal/probe smoke grid contains only `p2c@0,c2s@0,s2p@0`. Terminal
`c2s@1` appears separately for autograd gradient/PGD reference jobs. Unsupported
early autograd or incompatible batch-size attempts are explicitly recorded and
never relabeled as derivative-free attacks.

Print the probe grid and safely stride it over manually selected GPUs:

```bash
LR_STAGE=probe_grid bash experiments/linkradius/run_linkradius_smoke.sh
# Read max_array_index from the command above, then:
MAX_INDEX=23
mkdir -p logs
for gpu in 0 1 2 3; do
  GPU_LIST="$gpu" \
  LR_STAGE=probe \
  sbatch --array="${gpu}-${MAX_INDEX}:4%1" \
    experiments/linkradius/run_linkradius_smoke.sh
done
```

This manual striding matches the established workflow. `%4` plus modulo device
selection does not itself guarantee distinct GPUs across concurrently scheduled
tasks. On a cluster with GRES, prefer one scheduler-assigned GPU per task:

```bash
GPU_LIST="" LR_STAGE=probe \
sbatch --gres=gpu:1 --array="0-${MAX_INDEX}%4" \
  experiments/linkradius/run_linkradius_smoke.sh
```

## Phase 3: validation-only pilot calibration

Phase 3 requires both prior gates and categorically rejects `test` in
`PARTITIONS`. Its default batch counts match GPQA Diamond's 79/40
attack-train/validation split at batch size 16.

```bash
# CPU split; identical to the already frozen canonical split.
LR_STAGE=split bash experiments/linkradius/run_linkradius_pilot.sh

# Print, submit, and wait for each GPU array.
LR_STAGE=screen_clean_grid bash experiments/linkradius/run_linkradius_pilot.sh
LR_STAGE=screen_clean sbatch --array=0-7 experiments/linkradius/run_linkradius_pilot.sh
LR_STAGE=freeze_execution sbatch --array=0-1 experiments/linkradius/run_linkradius_pilot.sh

LR_STAGE=clean_grid bash experiments/linkradius/run_linkradius_pilot.sh
LR_STAGE=clean sbatch --array=0-7 experiments/linkradius/run_linkradius_pilot.sh
LR_STAGE=causal_grid bash experiments/linkradius/run_linkradius_pilot.sh
PILOT_CAUSAL_MAX="$(LR_STAGE=causal_grid bash experiments/linkradius/run_linkradius_pilot.sh | awk -F '\t' '$1=="max_array_index" {print $2}')"
LR_STAGE=causal sbatch --array="0-${PILOT_CAUSAL_MAX}" experiments/linkradius/run_linkradius_pilot.sh
LR_STAGE=probe_calibration_grid bash experiments/linkradius/run_linkradius_pilot.sh
PILOT_PROBE_MAX="$(LR_STAGE=probe_calibration_grid bash experiments/linkradius/run_linkradius_pilot.sh | awk -F '\t' '$1=="max_array_index" {print $2}')"
LR_STAGE=probe_calibration sbatch --array="0-${PILOT_PROBE_MAX}" experiments/linkradius/run_linkradius_pilot.sh
LR_STAGE=gradient_grid bash experiments/linkradius/run_linkradius_pilot.sh
PILOT_GRADIENT_MAX="$(LR_STAGE=gradient_grid bash experiments/linkradius/run_linkradius_pilot.sh | awk -F '\t' '$1=="max_array_index" {print $2}')"
LR_STAGE=gradient sbatch --array="0-${PILOT_GRADIENT_MAX}" experiments/linkradius/run_linkradius_pilot.sh

# Only after all attack-train/validation arrays completed and were inspected:
LR_STAGE=freeze_probe bash experiments/linkradius/run_linkradius_pilot.sh
LR_STAGE=validate_probe bash experiments/linkradius/run_linkradius_pilot.sh
LR_STAGE=aggregate bash experiments/linkradius/run_linkradius_pilot.sh
```

Mechanical dependencies can use parsable job IDs:

```bash
screen_job=$(LR_STAGE=screen_clean sbatch --parsable --array=0-7 \
  experiments/linkradius/run_linkradius_pilot.sh)
LR_STAGE=freeze_execution sbatch --dependency="afterok:${screen_job}" --array=0-1 \
  experiments/linkradius/run_linkradius_pilot.sh
```

Do not use dependencies to bypass scientific review. Inspect validation evidence
before `freeze_probe`; do not submit test jobs until the attack design is also
frozen in Phase 4. Successful probe jobs alone do not create `pilot_gate.json`.

Phase-3 outputs include canonical per-partition `execution_manifest.json`,
`frozen_config.json`, and `probe_gate.json` under `OUT_ROOT` and the hashed task
tree.

## Phase 4, aggregation, and expansion

`run_linkradius_attacks.sh` exposes all required grids and enforces
`probe_gate.json`, `frozen_config.json`, and (for test stages)
`attack_freeze_gate.json` plus the exact frozen-attack hash. Model-backed Phase-4
construction remains deliberately `unsupported_pending_gate` in this pass; it
does not emit empty/fabricated science.

`run_linkradius_expansion.sh` requires compatible passed `pilot_gate.json` and
`attack_validation_gate.json`. Its R=2/R=4/all-round grids omit terminal `s2p`,
but execution is likewise fail-closed until Phase 4 is genuinely complete.
The shell's pre-gate `LR_STAGE=grid` view is deliberately the structural R=2
mapping only; it does not authorize or claim readiness for R=4 or all-round
execution.

The CPU aggregation entrypoint never initializes CUDA or calls `nvidia-smi`:

```bash
AGGREGATE_PHASE=smoke LR_STAGE=verify \
  bash experiments/linkradius/run_linkradius_aggregate.sh
AGGREGATE_PHASE=smoke LR_STAGE=metrics \
  bash experiments/linkradius/run_linkradius_aggregate.sh
AGGREGATE_PHASE=smoke LR_STAGE=all \
  bash experiments/linkradius/run_linkradius_aggregate.sh
```

`verify` regenerates the expected grids and reports exact missing/stale array
indices before any summary is published. Run `metrics` after verification for
the CPU-only cast-quality and clean-agreement diagnostics; `all` includes it.
Individual stages are `causal`, `linkradius`, `attacks`, `metrics`, and
`system_curves`.

## Output layout

Task artifacts use:

```text
OUT_ROOT/<phase>/<dataset>/R<R>/<partition>/<stage>/<edge-token>/<config-hash>/
```

Each task directory contains `manifest.json`, `command.txt`,
`launcher_command.txt`, `run.log`, `warnings.txt`, its stage artifacts, and an
authenticated `.complete.json`. Canonical lifecycle files live at stable paths
such as:

```text
OUT_ROOT/split_manifest.json
OUT_ROOT/<phase>/gpqa/R2/<partition>/execution_manifest.json
OUT_ROOT/engineering_report.json
OUT_ROOT/engineering_gate.json
OUT_ROOT/smoke_gate.json
OUT_ROOT/frozen_config.json
OUT_ROOT/probe_gate.json
```

Set `REUSE_COMPLETE=1` to reuse only validated compatible tasks. Existing
incompatible output fails unless `OVERWRITE=1`; using a fresh `OUT_ROOT` after
source/configuration changes is recommended.
