# LinkRadius RQ2 failure-boundary handoff

Last updated: 2026-08-25 (Asia/Singapore)

## Objective

The next experiment is the proposal's most important RQ2 test:

> Does a validation-frozen LinkRadius estimate predict held-out, per-example
> empirical failure boundaries?

This is a fresh `sequential_scaled`, GPQA, R=2, four-80GB-GPU run. It evaluates
the first three valid handoffs (`p2c@0`, `c2s@0`, `s2p@0`) against deterministic
per-example white-box PGD and an independently seeded random-direction null.
The primary budget coordinate is the achieved post-consumer-cast relative norm.

This is an RQ2 pilot, not the proposal's later RQ3 transfer experiment.
Universal, DiffMean, and PCA attacks are not part of this run.

## What was implemented

Phase 4 now has an authenticated lifecycle:

```text
split -> freeze_execution -> val -> freeze_attack -> clean -> test_probe
      -> test -> thresholds -> analyze -> validate
```

The implementation enforces all of the following:

- raw attack-train/validation/test IDs are split before clean-correct filtering;
- all held-out test IDs are frozen before any test outcome is opened;
- even an interrupted test job's manifest, log, or artifact contaminates and
  blocks a late pre-outcome freeze;
- validation alone freezes the PGD/random budget grid and attack settings;
- the smallest/largest validation budgets must bracket a boundary on the same
  raw-example/edge PGD curve;
- PGD attacks all three wrong choices and retains the target with the smallest
  full pairwise gold margin;
- every PGD target's scores, margins, achieved norm, budget, and target identity
  are authenticated;
- requested budgets are a preregistered sensitivity coordinate, while achieved
  post-cast norms are primary;
- threshold crossing uses `margin <= 0`, with interval censoring, right
  censoring, and re-entry/non-monotonicity recorded;
- at least three independent frozen probe-direction seeds are required;
- primary analysis keeps the raw/edge cohort complete across all frozen seeds;
- point metrics are reported per probe seed;
- paired confidence intervals cluster by raw example and take an equal mean of
  seed-specific contrasts, so probe seeds are not counted as independent data;
- LinkRadius is compared with clean margin alone and susceptibility alone;
- the final gate requires supported PGD AUROC/AUPRC, threshold Spearman,
  interval-censored concordance, and within-example site-ranking metrics for
  every frozen probe seed, plus paired intervals against both baselines;
- a valid negative result is allowed, but an unestimable result is labeled
  `underpowered` and cannot pass the scientific gate.

Re-run verification after transferring the checkout:

```bash
python -m unittest discover -s experiments/linkradius/tests
python -m compileall -q RecursiveMAS experiments/linkradius
bash -n experiments/linkradius/linkradius_common.sh \
  experiments/linkradius/run_linkradius_engineering.sh \
  experiments/linkradius/run_linkradius_smoke.sh \
  experiments/linkradius/run_linkradius_pilot.sh \
  experiments/linkradius/run_linkradius_attacks.sh
git diff --check
```

## Scientific limitations that must remain explicit

- The earlier `sequential_light` GPQA smoke run failed its 10--20
  dual-correct-row gate. Do not relabel it as a pass.
- `sequential_scaled` is a stronger-model follow-up, not a replacement for that
  failed release baseline.
- GPQA has only 79 raw test rows under this 40/20/40 split, before fresh
  clean-correct filtering. It therefore cannot guarantee the proposal's
  64--128 clean-correct target. The final analysis may correctly finish as
  underpowered.
- A failure of the scientific gate does not mean the implementation failed. It
  can mean too few eligible examples, too little class variation, too few
  comparable threshold intervals, or unstable probe realizations.

## Source identity

Current base commit:

```text
16693df0af6a034ca1a304544ff7d71306bfc225
```

The result-affecting changes are currently in the working tree. Transfer or
commit every change under `RecursiveMAS` and `experiments/linkradius` together.
Do not begin a run from a partially copied tree. The final expected source hash
is filled in below after the last audit:

```text
f1f6228b9379913bec11a832762148bb0b96fd0cbec7f83dcd52d5f8bdf6958d
```

`handoff.md` itself is outside the result-affecting source hash.

## Fresh-root environment

Run all stages from one persistent shell. The intended cluster placement is
`PA100q` on `node02`, with one process reserving four GPUs. CUDA indices below
are logical indices inside Slurm's `CUDA_VISIBLE_DEVICES`; they are not physical
GPU numbers.

```bash
cd ~/Robust-RMAS
conda activate recursivemas

export PYTHON_BIN="$CONDA_PREFIX/bin/python"
export LINKRADIUS_REPO_ROOT="$PWD"
export OUT_ROOT="$PWD/outputs/linkradius-rq2-scaled-4x80-v1"

export STYLE=sequential_scaled
export METHOD=ours_recursive
export DATASETS=gpqa
export ROUNDS=2
export SEEDS=42
export BATCH_SIZE=1
export LATENT_LENGTH=48

export DEVICE=cuda:0
export PLANNER_DEVICE=cuda:0
export CRITIC_DEVICE=cuda:1
export SOLVER_DEVICE=cuda:2
export TERMINAL_SOLVER_DEVICE=cuda:3
export RELAY_TRANSFER_MODE=cpu_staged
export AUTOGRAD_MEMORY_MODE=checkpoint

export DISCOVERY_BATCHES=20
export PROBE_RADII="1e-3 3e-3"
export PROBE_SEEDS="101 202 303"
export K=8
export SUBSPACE=full_tensor

export ATTACK_FAMILIES="pgd_autograd random_independent"
export ATTACK_EPSILONS="3e-4 1e-3 3e-3 1e-2 3e-2 1e-1"
export PGD_STEPS=20
export RANDOM_ATTACK_SEED_OFFSET=1000000

export SPLIT_MANIFEST="$OUT_ROOT/split_manifest.json"
export ENGINEERING_GATE="$OUT_ROOT/engineering_gate.json"
export SMOKE_GATE="$OUT_ROOT/smoke_gate.json"
export PROBE_GATE="$OUT_ROOT/probe_gate.json"
export PILOT_GATE="$OUT_ROOT/pilot_gate.json"
export FROZEN_CONFIG="$OUT_ROOT/frozen_config.json"
export ATTACK_FREEZE_GATE="$OUT_ROOT/attack_freeze_gate.json"
export FROZEN_ATTACK_CONFIG="$OUT_ROOT/frozen_attack_config.json"
export ATTACK_VALIDATION_GATE="$OUT_ROOT/attack_validation_gate.json"
export LEGACY_EQUIVALENCE="$OUT_ROOT/legacy_equivalence.json"

unset GPU_LIST EXECUTION_MANIFEST TRAJECTORY SCREENING_JSONL
unset PARTITIONS BATCH_COUNTS NUM_BATCHES MAX_ELIGIBLE

mkdir -p logs
SBATCH_GPU=(-p PA100q -w node02 --nodes=1 --ntasks=1 --gres=gpu:4)
```

Do not install FLA or switch conda environments after starting this root. GPU
task identity binds the Python executable and inference-backend package
versions. Do not edit any file under `RecursiveMAS` or `experiments/linkradius`
after the first completion is written.

Verify the transferred source before the first stage:

```bash
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
from experiments.linkradius.io_utils import source_hash

expected = "f1f6228b9379913bec11a832762148bb0b96fd0cbec7f83dcd52d5f8bdf6958d"
actual = source_hash(Path.cwd())
print({"expected": expected, "actual": actual, "match": actual == expected})
raise SystemExit(actual != expected)
PY
```

## First command

```bash
OVERWRITE=1 LR_STAGE=split \
bash experiments/linkradius/run_linkradius_engineering.sh
```

This is CPU-only. It creates and authenticates the deterministic GPQA
attack-train/validation/test split. It does not load a model or inspect any
held-out outcome. Wait for its success and show the output to Codex before
running the next command.

## Step-by-step stage map

The new Codex session should issue one command at a time, wait for completion,
and audit the exact grid before advancing. Never guess an array bound; print the
canonical grid and read `max_array_index`.

### Phase 1: engineering

1. `split` (the first command above).
2. `discover` on validation, initially task 0, then the remaining canonical
   tasks.
3. `freeze_execution` to select exactly one dual-correct row.
4. `clean` on the frozen row.
5. Produce a release-runner trace for that exact raw index using the identical
   four-device topology, then run `compare_legacy_equivalence`.
6. `replay`, `probe`, and `gradient` GPU grids.
7. `validate` to create `engineering_gate.json`.

Every GPU submission uses this shape and `%1`:

```bash
OVERWRITE=1 LR_STAGE=<stage> \
sbatch "${SBATCH_GPU[@]}" --array="0-<max>%1" \
  experiments/linkradius/run_linkradius_engineering.sh
```

### Phase 2: smoke

Before Phase 2:

```bash
export NUM_BATCHES=40
export MAX_ELIGIBLE=16
```

Run, in order: `split`, `screen`, `freeze_execution`, `clean`, `causal`,
`probe`, `gradient`, `attack`, `estimate`, `aggregate`, `validate`. Print each
dynamic `*_grid` immediately before the matching GPU stage. Every GPU array
again requests all four GPUs and uses `%1`. The smoke gate requires 10--20
fresh dual-correct validation rows; stop honestly if it fails.

### Phase 3: validation-only probe calibration

Remove the Phase-2 limits and cover the complete attack-train/validation raw
partitions:

```bash
unset NUM_BATCHES MAX_ELIGIBLE
export PARTITIONS="attack_train validation"
export BATCH_COUNTS="attack_train=79 validation=40"
```

Run, in order: `split`, `screen_clean`, `freeze_execution` for array indices
0 and 1, `clean`, `causal`, `probe_calibration`, `gradient`, `freeze_probe`,
`validate_probe`, and `aggregate`. The three probe seeds must remain
`101 202 303`. Do not access test outcomes in this phase.

### Phase 4: held-out RQ2 boundary test

Reset variables that would leak the Phase-3 scope:

```bash
unset PARTITIONS BATCH_COUNTS NUM_BATCHES MAX_ELIGIBLE
unset EXECUTION_MANIFEST TRAJECTORY SCREENING_JSONL GPU_LIST
```

Then run, in order:

1. `split` (CPU, verifies the same split).
2. `freeze_execution` (CPU, freezes all 79 raw test IDs before outcomes).
3. `val_grid`, then `val` on four-GPU Slurm tasks.
4. `freeze_attack` (CPU). If validation does not bracket a same-curve PGD
   boundary, widen `ATTACK_EPSILONS`, rerun the current validation grid with
   `OVERWRITE=1`, and freeze again. Historical retune directories are ignored.
5. `clean_grid`, then held-out `clean` GPU tasks.
6. `test_probe_grid`, then held-out `test_probe` GPU tasks. With 79 one-row
   batches, three edges, one frozen radius, and three seeds, the usual maximum
   array index is 710.
7. `test_grid`, then held-out `test` GPU tasks. With two families and three
   edges, the usual maximum index is 473; each task sweeps the full budget grid
   under one model load.
8. `thresholds`, `analyze`, then `validate` on CPU.

The exact Phase-4 commands are maintained in
`experiments/linkradius/README.md`. `freeze_attack`, `thresholds`, `analyze`,
and `validate` all fail closed on missing, stale, duplicate, or incompatible
completions.

## Slurm and memory rules

- Use `sbatch`, not an interactive `srun`, for long model runs.
- `--gres=gpu:4` asks Slurm for four GPUs for the one process. It does not pool
  them automatically; the code explicitly places the three roles and terminal
  solver replica on four logical devices.
- `%1` prevents two four-GPU array elements from overlapping on the node.
- Slurm reservations cannot evict unmanaged processes. Inspect the `nvidia-smi`
  snapshot at the beginning of every job log. If another process occupies an
  allocated device, cancel that element and rerun the exact array index after
  the device is clear.
- Switching physical nodes is safe only if the same code, conda environment,
  model snapshots, four logical role devices, transfer mode, and checkpoint
  mode are preserved. Node names themselves are not in the scientific hash.
- Use `OVERWRITE=1` only for the exact failed/current task. Content-addressed
  historical results can coexist, but cannot satisfy the current grid.

## Main outputs and interpretation

After Phase 4 analysis, locate the authenticated task directory containing:

```text
failure_thresholds.csv
prediction_units.csv
edge_predictors.csv
probe_exclusions.csv
threshold_exclusions.csv
flip_prediction_metrics.csv
threshold_prediction_metrics.csv
calibration_bins.csv
paired_bootstrap_intervals.csv
analysis_result.json
```

The final root-level result is `attack_validation_gate.json`.

- `passed=true` means the required metrics were estimable, not that LinkRadius
  won. Inspect estimates and paired CIs to answer the scientific question.
- A positive LinkRadius-minus-baseline contrast favors LinkRadius.
- If a paired 95% interval includes zero, improvement over that baseline is not
  statistically supported.
- `scientific_status=underpowered` means report the eligible denominator,
  censoring, exclusions, and support; do not turn it into a positive or negative
  theorem claim.

## Prompt for another Codex session

Use this when continuing on a different machine:

> Read `handoff.md` and `experiments/linkradius/README.md` completely. Inspect
> the repository status and current output root. Verify the recorded
> LinkRadius source hash before doing anything. Continue the fresh
> `linkradius-rq2-scaled-4x80-v1` experiment one authenticated stage at a time.
> Use `PA100q`, `node02`, one Slurm task with four GPUs, and `%1`. Never skip a
> freeze/gate, mix environments or source hashes, infer success from a Slurm
> exit code alone, or open test outcomes before `freeze_attack` passes.
