# LinkRadius experiments

This package implements the staged, auditable LinkRadius pipeline for the
released `sequential_light` baseline and explicitly configured stronger-model
follow-ups such as `sequential_scaled`. The GPQA release baseline uses seed 42,
batch size 16, latent length 32, source split `train`, deterministic generation,
and `choice_old_prompt=2`; every deviation is recorded in task identity and may
not be mixed with the baseline.

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
`GPU_LIST`, `PLANNER_DEVICE`, `CRITIC_DEVICE`, `SOLVER_DEVICE`,
`TERMINAL_SOLVER_DEVICE`, `RELAY_TRANSFER_MODE`, `AUTOGRAD_MEMORY_MODE`,
`PROBE_RADII`, `PROBE_SEEDS`, `K`, and `SUBSPACE`.
`GPU_LIST` is a whitespace-separated
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
export TERMINAL_SOLVER_DEVICE=cuda:3
export RELAY_TRANSFER_MODE=cpu_staged
export AUTOGRAD_MEMORY_MODE=checkpoint
unset GPU_LIST
```

This four-device configuration places the planner, critic, and recurrent
solver-feedback model on the first three devices. It loads a frozen replica of
the identical solver checkpoint on the fourth device for terminal scoring and
generation. This separates the solver-feedback and terminal-scoring activation
graphs. Each outer adapter stays on its source role's device. By default, a
relay crossing two CUDA devices is copied source GPU -> CPU float32 ->
destination GPU consumer dtype, using differentiable tensor copies without a
detach. A same-device relay remains a direct cast/copy. The CUDA indices are
logical indices within the scheduler's
`CUDA_VISIBLE_DEVICES`, not necessarily physical GPU ordinals.

Run one process in one Slurm task and request four GPUs, for example with
`--ntasks=1 --gres=gpu:4` when those are the site's allocation flags. Keep
`GPU_LIST` empty: `GPU_LIST="0 1 2 3"` still means round-robin single-GPU array
routing and does not pool memory. On a four-GPU node, throttle a two-element
gradient array with `--array=0-1%1` so only one four-GPU element runs at once.

`AUTOGRAD_MEMORY_MODE=checkpoint` applies PyTorch non-reentrant activation
checkpointing only to differentiable latent and terminal-scorer forwards. It
recomputes model activations during backward rather than retaining every
intermediate activation from all latent steps. Non-differentiable discovery,
clean capture, replay, probes, and final scoring retain their ordinary forward
path. Leaving `TERMINAL_SOLVER_DEVICE` empty reuses `SOLVER_DEVICE`; leaving
`AUTOGRAD_MEMORY_MODE=none` preserves the legacy memory behavior.

The resolved role topology is authenticated experiment configuration. Export
the same `STYLE`, `LATENT_LENGTH`, `PLANNER_DEVICE`, `CRITIC_DEVICE`, and
`SOLVER_DEVICE` values, plus the same `RELAY_TRANSFER_MODE`, for every command
in a fresh workflow. The same `TERMINAL_SOLVER_DEVICE` and
`AUTOGRAD_MEMORY_MODE` values are also required, including for CPU
freeze/validation/aggregation commands that reconstruct upstream task keys.
`cpu_staged` is the supported default;
`direct` is retained for controlled diagnostics and has a distinct task hash.
Changing only the physical node is safe when the same four logical devices
remain available. Changing the logical role map, terminal replica placement,
checkpoint mode, or transfer mode requires a new output root.

Differentiable gradient and PGD objectives use the exact frozen
`gold_score - target_score` margin but evaluate and backpropagate the gold and
target solver candidates sequentially. This keeps at most one differentiable
terminal scorer graph resident at a time. Ordinary clean, replay,
finite-difference, and final PGD scoring remains the complete four-way A/B/C/D
scorer. With checkpoint mode enabled, each differentiable model forward is
recomputed during backward; neither memory optimization changes reported choice
scores or candidate selection.

Every array task is a whole frozen execution batch plus one intervention
configuration. A probe task contains both signs for all nested directions, so
the scheduler never splits an antithetic pair. Outputs are written beside a
temporary file and atomically renamed. Reuse occurs only when `.complete.json`
validates the config hash, source hash, artifact hashes, and row counts.
`OVERWRITE=1` takes precedence over reuse and executes the task again. GPU-task
identity also binds the Python executable and the installed inference-backend
package versions. Frozen probe/attack protocols additionally bind the exact
resolved model, adapter, scorer, and prompt identities, so changing conda
environments or a mutable Hugging Face snapshot cannot silently mix results.

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

Discovery/screening treats a non-finite scorer output as an auditable invalid
row rather than a serialization failure: the affected public numeric fields
are `null`, `scorer_nonfinite_fields` names the exact failures, and the row is
excluded as `scorer_nonfinite`. This tolerance is screening-only. A non-finite
score or relay in an authenticated clean/frozen capture remains a hard error.

Every fresh screening row also publishes `forward_finiteness`. Its `edges`
mapping follows execution chronology (`p2c@0,c2s@0,s2p@0,...`) and separately
summarizes the source-side `transport` and destination-side `receiver` tensor.
Diagnostics include finite/NaN/+Inf/-Inf counts, the first bad coordinate and
latent step, the requested and realized relay-transfer modes, and JSON-safe
magnitude statistics for every latent step. The
top-level `first_nonfinite` therefore distinguishes a source-agent failure, a
consumer cast/transfer failure, and a terminal-only forced-choice failure.
These statistics detach and inspect stored clean relays; they do not alter the
forward computation or make a non-finite row eligible.

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
export DEVICE="${DEVICE:-cuda:0}"
export PLANNER_DEVICE="${PLANNER_DEVICE:-$DEVICE}"
export CRITIC_DEVICE="${CRITIC_DEVICE:-$DEVICE}"
export SOLVER_DEVICE="${SOLVER_DEVICE:-$DEVICE}"
export TERMINAL_SOLVER_DEVICE="${TERMINAL_SOLVER_DEVICE:-$SOLVER_DEVICE}"
export RELAY_TRANSFER_MODE="${RELAY_TRANSFER_MODE:-cpu_staged}"

"$PYTHON_BIN" RecursiveMAS/run.py \
  --style sequential_light \
  --dataset gpqa --dataset_split train \
  --method ours_recursive --num_recursive_rounds 2 \
  --num_samples 1 --sample_indices "$RAW_INDEX" \
  --batch_size 1 --latent_length 32 \
  --seed 42 --deterministic 1 --device "$DEVICE" \
  --planner-device "$PLANNER_DEVICE" \
  --critic-device "$CRITIC_DEVICE" \
  --solver-device "$SOLVER_DEVICE" \
  --terminal-solver-device "$TERMINAL_SOLVER_DEVICE" \
  --relay-transfer-mode "$RELAY_TRANSFER_MODE" \
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

Run the release command inside an allocation that exposes every logical device
named by the frozen trajectory. The comparator reopens and hashes all three
artifacts, requires the release results and trace to authenticate that exact
role topology and relay-transfer policy, recomputes the exact
generation/strict-choice/five-relay checks, and binds the report to the current
release source tree. The relay tolerances are fixed at `atol=rtol=1e-5`; the
validator rejects reports created with looser values. Regenerate it after any
result-affecting source change. CPU toy
checks are not accepted as a substitute for this audit or for real terminal
finite differences.

## Phase 2: smoke

Smoke requires the passed engineering gate. It screens fixed validation
batches, keeps their filler rows and boundaries, and marks 10--20 rows eligible
(default 16).

```bash
LR_STAGE=split bash experiments/linkradius/run_linkradius_smoke.sh
SMOKE_SCREEN_MAX="$(GRID_TARGET_STAGE=screen LR_STAGE=grid bash experiments/linkradius/run_linkradius_smoke.sh | awk -F '\t' '$1=="max_array_index" {print $2}')"
LR_STAGE=screen sbatch --array="0-${SMOKE_SCREEN_MAX}" experiments/linkradius/run_linkradius_smoke.sh
LR_STAGE=freeze_execution bash experiments/linkradius/run_linkradius_smoke.sh
SMOKE_CLEAN_MAX="$(GRID_TARGET_STAGE=clean LR_STAGE=grid bash experiments/linkradius/run_linkradius_smoke.sh | awk -F '\t' '$1=="max_array_index" {print $2}')"
LR_STAGE=clean sbatch --array="0-${SMOKE_CLEAN_MAX}" experiments/linkradius/run_linkradius_smoke.sh
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

For the four-GPU scaled run with `BATCH_SIZE=1`, export
`NUM_BATCHES=40 MAX_ELIGIBLE=16`; the screen and clean arrays are then `0-39`.
Keep `%1` and request all four GPUs for each element. Do not let
`MAX_ELIGIBLE=16` leak into Phase 3.

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
PILOT_SCREEN_MAX="$(LR_STAGE=screen_clean_grid bash experiments/linkradius/run_linkradius_pilot.sh | awk -F '\t' '$1=="max_array_index" {print $2}')"
LR_STAGE=screen_clean sbatch --array="0-${PILOT_SCREEN_MAX}" experiments/linkradius/run_linkradius_pilot.sh
LR_STAGE=freeze_execution sbatch --array=0-1 experiments/linkradius/run_linkradius_pilot.sh

LR_STAGE=clean_grid bash experiments/linkradius/run_linkradius_pilot.sh
PILOT_CLEAN_MAX="$(LR_STAGE=clean_grid bash experiments/linkradius/run_linkradius_pilot.sh | awk -F '\t' '$1=="max_array_index" {print $2}')"
LR_STAGE=clean sbatch --array="0-${PILOT_CLEAN_MAX}" experiments/linkradius/run_linkradius_pilot.sh
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

For `BATCH_SIZE=1`, first unset `NUM_BATCHES MAX_ELIGIBLE`, then export
`PARTITIONS="attack_train validation"` and
`BATCH_COUNTS="attack_train=79 validation=40"` for every Phase-3 command.
The screen/clean grids then contain 119 tasks (`0-118`) and execution freeze
contains two (`0-1`). Export at least three fixed direction seeds, for example
`PROBE_SEEDS="101 202 303"`, before starting the fresh workflow; Phase 4 will
reject a Phase-3 probe freeze with fewer than three seeds.

Mechanical dependencies can use parsable job IDs:

```bash
screen_job=$(LR_STAGE=screen_clean sbatch --parsable --array="0-${PILOT_SCREEN_MAX}" \
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

Phase 4 is the minimal held-out RQ2 experiment: does the validation-frozen
LinkRadius predict the real per-example failure boundary? It evaluates the
three early R2 edges (`p2c@0`, `c2s@0`, `s2p@0`) against per-example white-box
PGD and a stable random direction drawn from a seed domain independent of the
probe directions. Every attack task sweeps the complete frozen budget grid
under one model load. PGD optimizes all three wrong labels and keeps the
candidate with the smallest full pairwise margin. Random perturbations are fit
to the post-consumer-cast budget, and primary analyses use the realized norm.

The lifecycle is strictly ordered:

```text
split -> freeze_execution -> val -> freeze_attack -> clean -> test_probe
      -> test -> thresholds -> analyze -> validate
```

`freeze_execution` fixes every raw test ID before observing an outcome. `val`
uses only the already frozen validation trajectories. `freeze_attack` verifies
the exact validation cube and requires the smallest and largest PGD budgets to
straddle at least one boundary on the same raw-example/edge curve. It requires
and freezes at least three Phase-3 probe seeds together with the attack/runtime
settings, and refuses to run if any test clean/probe/attack artifact exists,
including an unfinished job's manifest or pending log. Only then may any test
outcome be opened. Fresh test dual-correctness is recomputed from primitive
clean results; frozen grid cells without a fresh dual-correct row publish an
authenticated exclusion completion without loading the models.

For the four-GPU `sequential_scaled`, batch-size-one configuration, keep the
same logical role map used in Phases 1--3 and submit one array element at a time:

```bash
unset GPU_LIST PARTITIONS BATCH_COUNTS EXECUTION_MANIFEST TRAJECTORY SCREENING_JSONL
unset NUM_BATCHES MAX_ELIGIBLE
mkdir -p logs
SBATCH_GPU=(-p PA100q -w node02 --nodes=1 --ntasks=1 --gres=gpu:4)
export ATTACK_FAMILIES="pgd_autograd random_independent"
export ATTACK_EPSILONS="3e-4 1e-3 3e-3 1e-2 3e-2 1e-1"
export PGD_STEPS=20
export RANDOM_ATTACK_SEED_OFFSET=1000000

OVERWRITE=1 LR_STAGE=split \
  bash experiments/linkradius/run_linkradius_attacks.sh
OVERWRITE=1 LR_STAGE=freeze_execution \
  bash experiments/linkradius/run_linkradius_attacks.sh

LR_STAGE=val_grid bash experiments/linkradius/run_linkradius_attacks.sh
VAL_MAX="$(LR_STAGE=val_grid bash experiments/linkradius/run_linkradius_attacks.sh |
  awk -F '\t' '$1=="max_array_index" {print $2}')"
OVERWRITE=1 LR_STAGE=val \
  sbatch "${SBATCH_GPU[@]}" --array="0-${VAL_MAX}%1" \
    experiments/linkradius/run_linkradius_attacks.sh

# Wait for and verify every validation task before freezing.
OVERWRITE=1 LR_STAGE=freeze_attack \
  bash experiments/linkradius/run_linkradius_attacks.sh

# The launcher authenticates frozen_attack_config.json and hydrates its exact
# one h, all frozen probe seeds, K, plus the attack grid for every post-freeze
# command.

LR_STAGE=clean_grid bash experiments/linkradius/run_linkradius_attacks.sh
CLEAN_MAX="$(LR_STAGE=clean_grid bash experiments/linkradius/run_linkradius_attacks.sh |
  awk -F '\t' '$1=="max_array_index" {print $2}')"
OVERWRITE=1 LR_STAGE=clean \
  sbatch "${SBATCH_GPU[@]}" --array="0-${CLEAN_MAX}%1" \
    experiments/linkradius/run_linkradius_attacks.sh

# Wait for and audit every clean completion before submitting test probes.
LR_STAGE=test_probe_grid bash experiments/linkradius/run_linkradius_attacks.sh
TEST_PROBE_MAX="$(LR_STAGE=test_probe_grid bash experiments/linkradius/run_linkradius_attacks.sh |
  awk -F '\t' '$1=="max_array_index" {print $2}')"
OVERWRITE=1 LR_STAGE=test_probe \
  sbatch "${SBATCH_GPU[@]}" --array="0-${TEST_PROBE_MAX}%1" \
    experiments/linkradius/run_linkradius_attacks.sh

# Wait for and audit every test-probe completion before submitting attacks.
LR_STAGE=test_grid bash experiments/linkradius/run_linkradius_attacks.sh
TEST_MAX="$(LR_STAGE=test_grid bash experiments/linkradius/run_linkradius_attacks.sh |
  awk -F '\t' '$1=="max_array_index" {print $2}')"
OVERWRITE=1 LR_STAGE=test \
  sbatch "${SBATCH_GPU[@]}" --array="0-${TEST_MAX}%1" \
    experiments/linkradius/run_linkradius_attacks.sh

# Wait for and audit every held-out attack completion before CPU analysis.

OVERWRITE=1 LR_STAGE=thresholds bash experiments/linkradius/run_linkradius_attacks.sh
OVERWRITE=1 LR_STAGE=analyze bash experiments/linkradius/run_linkradius_attacks.sh
OVERWRITE=1 LR_STAGE=validate bash experiments/linkradius/run_linkradius_attacks.sh
```

Each wait comment is a mandatory synchronization point. Require the printed
grid's exact completion set before moving on. With GPQA's 79 held-out rows and batch size 1, the expected upper
bounds are usually 78 for clean, 710 for three-seed/one-radius test probes, and
473 for two attack families; always trust the printed authenticated grid.

Phase-4 analysis writes interval-censored thresholds, per-budget AUROC/AUPRC,
crossed-threshold Spearman correlation, interval concordance, complete-edge
within-example site ranking, overall and per-edge calibration bins, transparent
probe/threshold exclusions, and paired raw-ID cluster-bootstrap contrasts against clean
margin alone and susceptibility alone. Primary files are
`failure_thresholds.csv`, `prediction_units.csv`, `edge_predictors.csv`,
`probe_exclusions.csv`, `threshold_exclusions.csv`, `flip_prediction_metrics.csv`,
`threshold_prediction_metrics.csv`, `calibration_bins.csv`, and
`paired_bootstrap_intervals.csv` under their authenticated task directories.
Actual post-cast norm is the primary threshold coordinate; the complete
requested-grid threshold analysis is emitted as a labeled sensitivity check.
Point metrics and calibration are reported separately for every frozen probe
seed, on the raw/edge cohort with a complete accepted prefix for every frozen
seed. Paired intervals resample raw examples as clusters and average the
seed-specific contrast within each bootstrap replicate; probe-seed realizations
are therefore never counted as independent examples.
The final gate separately requires all three probe-seed realizations to have
finite, minimally supported PGD AUROC/AUPRC, threshold Spearman, censored
concordance, and site-ranking results, plus paired intervals against both
component baselines for every required metric. If those are not estimable, the
artifacts remain auditable but the run is labeled `underpowered` and the gate
does not pass.

GPQA provides only 79 raw held-out rows before fresh dual-correct filtering, so
this configuration cannot guarantee the proposal's 64--128 clean-correct test
target. Treat it as a minimal, potentially underpowered RQ2 pilot and report the
eligible denominator, censoring, and exclusion counts; it is not by itself a
definitive theorem test.

This is deliberately the minimal RQ2 pilot, not the RQ3 transfer experiment:
universal, DiffMean, and PCA banks remain future work and must not be inferred
from the PGD/random-null result.

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
