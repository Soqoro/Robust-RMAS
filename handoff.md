# LinkRadius empirical-core handoff

Last updated: 2026-08-30 (Asia/Singapore)

## Scientific objective

The corrected proposal's most important experiment is:

> Does a validation-frozen LinkRadius estimate predict held-out,
> per-example/per-site empirical failure boundaries better than clean margin
> or susceptibility alone?

This is an empirical risk-ranking claim. It is not a theorem certification,
bitwise reproducibility claim, or proof that LinkRadius is a universal safety
radius. A negative or underpowered result is still valid evidence when the
cohort, exclusions, censoring, and uncertainty are reported honestly.

## Decisions from the proposal review

Use `forced_margin` as the primary clean-correct endpoint. A row is eligible
when forced-choice scoring is finite and every gold-versus-wrong option margin
is strictly positive. Free-text generation is diagnostic only and should be
disabled for the main census and robustness run. The old `dual_correct`
endpoint mixed scorer correctness with generation correctness/formatting and
discarded scientifically usable rows.

Use the empirical validation tier for the paper run:

```bash
export VALIDATION_TIER=empirical
export CLEAN_CORRECT_POLICY=forced_margin
export CLEAN_STABILITY_POLICY=empirical
export INCLUDE_GENERATION=0
```

Keep the certification tier and `dual_correct` only for release-engineering or
sensitivity checks. Exact legacy generation/tensor equality, strict rerun
identity, the same physical GPU ordinals, and the same node are not prerequisites
for the empirical paper claim.

The essential safeguards remain:

- split raw IDs into attack-train/validation/test before any filtering;
- choose probe radii, attacks, budgets, scorer, and sites using no test outcome;
- freeze the complete protocol before opening held-out outcomes;
- use the same post-consumer-cast subspace and relative-norm definition for
  probes and attacks, and record achieved rather than only requested norms;
- authenticate code, environment, model/adapter identities, logical role map,
  manifests, and artifact hashes;
- report full dose-response curves, exclusions, interval/right censoring, and
  non-monotone re-entry;
- compare LinkRadius with clean margin and susceptibility on identical rows;
- bootstrap by raw example so edges and probe seeds are not treated as
  independent examples.

## Model choice: run a fresh census first

Do not select `sequential_scaled` from the earlier low `sequential_light`
"dual-correct" rate. Those runs were generation-gated, so they do not establish
that the forced-choice margins are poor.

Start a fresh `sequential_light`, GPQA, R=2 output root and run a
forced-margin screening census with generation disabled. Record:

- total rows and finite-scorer rows;
- forced-margin-positive count and margin distribution by split;
- the three early-site causal audit on a small validation subset;
- exclusions by primitive reason, without collapsing them into generation
  failures.

Prefer `sequential_light` if it yields a usable forced-margin cohort and a
nontrivial mismatch/zero causal signal. If it remains too sparse or
non-informative, start a separate `sequential_scaled` root and state the actual
model/checkpoints in the manuscript. Never combine the two roots as if they
were one preregistered system. A practical small pilot can proceed below the
proposal's aspirational sample target, but must be labeled underpowered.

GPQA Diamond currently contributes 79 attack-train, 40 validation, and 79 test
raw rows. Consequently the proposed 64--128 clean-correct test examples cannot
be guaranteed after filtering. MedQA is a later independent replication, not
a substitute for getting the primary GPQA design correct.

## Minimal experiment sequence

Use a fresh output root after the policy patch is merged, tested, and synced to
the cluster. Do not reuse a root created under `dual_correct` or a different
source hash.

### 0. Feasibility census

Run the raw split and `screen_clean` stages for `sequential_light` with the
four policy variables above. Inspect the forced-margin yield and numerical
finiteness before committing GPU time to every downstream task. Generation is
not needed for this stage. The first PGD-plus-random experiment needs no learned
attack bank, so Phase 3 should cover the complete 40-row `validation` partition
only (`BATCH_SIZE=1`, `BATCH_COUNTS=validation=40`). Reserve `attack_train` for
a later universal-attack follow-up.

### 1. RQ0 causal-use audit

At R=2, use only the early relay sites:

```text
p2c@0, c2s@0, s2p@0
```

The minimum intervention set is:

```bash
export INTERVENTIONS="identity mismatch zero"
```

Identity replay is the control, label-matched mismatch is the decisive
message-specificity intervention, and zero is the destructive control.
Moment-matched noise is optional follow-up evidence, not a reason to block the
main experiment. A negative audit does not erase the all-edge held-out test,
but it changes the claim: report RQ2 only as predictive association unless a
preregistered positive control later supports a causal useful-link reading.

### 2. RQ1 estimator calibration

Use two post-cast radii, three fixed probe seeds, and a feasible directional
count:

```bash
export PROBE_SEEDS="101 202 303"
export K=8
export GRADIENT_REFERENCE_BATCHES=2
```

The K-direction symmetric finite-difference estimate is primary. Run exact
autograd only on two fixed validation batches as a reference check. Do not
require exact autograd for every example/site. Treat it as a nonblocking
diagnostic in empirical mode. Increase from K=8 to K=16 only in a fresh
confirmatory root if validation shows a material stability gain without
unacceptable probe attrition. Freeze the selected probe radius
and estimator settings using validation data only.

In empirical mode, radius/seed/K stability and the exact-autograd subset are
diagnostics, not gates that erase a negative result. Calibration must still
produce a finite usable estimator and pass the frozen cast-quality/coverage
contracts. Certification mode retains the stricter stability thresholds.

Empirical Phase 4 freezes a minimum accepted-direction count before test access:
75% of K, with a floor of four (so 6 of 8 here). It retains and labels partial
`K_eff` estimates instead of discarding an entire unit after one cast-quality
rejection. Certification mode still requires all K. Report the `K_eff` and
exclusion tables alongside the primary result.

### 3. RQ2 held-out failure-boundary test

Use the same three early sites. Freeze validation-selected attacks and budgets,
then evaluate test examples with:

```bash
export ATTACK_FAMILIES="pgd_autograd random_independent"
```

PGD is the main per-example white-box attack; the independent random direction
is the negative-control family and must use seeds outside the probe seed domain.
Sweep the complete frozen budget grid under each model load. The primary
threshold coordinate is achieved post-cast relative norm. Preserve left/
interval/right censoring rather than replacing censored thresholds with an
arbitrary scalar.

The primary evidence is interval-censored concordance or another censoring-
aware ranking statistic, supported by threshold Spearman (where crossed),
AUROC/AUPRC across budgets, within-example site ranking, and calibration bins.
Every metric must be shown for LinkRadius, clean margin, and susceptibility;
paired uncertainty is clustered by raw example.

### 4. Only after the core result

If RQ0--RQ2 are interpretable, add one independent universal attack as the
strongest next experiment. Then replicate on MedQA, and only afterward study
R=4/system-level curves, prompts, architectures, or protection policies.
DiffMean/PCA banks and broad architecture sweeps are not prerequisites for the
first paper result.

## Lifecycle and gates

The existing authenticated stage order remains useful:

```text
pilot:   split -> screen_clean -> freeze_execution -> clean -> causal
         -> probe_calibration -> gradient(subset) -> freeze_probe
         -> validate_probe -> aggregate

heldout: split -> freeze_execution -> val -> freeze_attack -> clean
         -> test_probe -> test -> thresholds -> analyze -> validate
```

In empirical mode, Phase 1 release engineering and the large smoke phase are
optional diagnostics; they must not block the paper solely because text or
relay tensors differ bitwise across otherwise authenticated jobs. The pilot and
held-out phases still fail closed on missing/stale artifacts, test leakage,
wrong logical topology, non-finite forced scores, changed model artifacts, or
an unfrozen protocol.

For every dynamic GPU stage, print the canonical `*_grid`, read its
`max_array_index`, submit exactly that array, wait for completion, and audit the
completion set before advancing. Do not infer scientific completion from a
Slurm exit code alone.

## Slurm and memory rules

- Use `sbatch` for long jobs. One process may reserve four GPUs with the site's
  equivalent of `--nodes=1 --ntasks=1 --gres=gpu:4`.
- Four GPUs do not automatically pool memory. The code must map planner,
  critic, recurrent solver, and terminal solver replica to logical devices
  `cuda:0..3` and use the configured checkpoint/relay-transfer modes.
- Keep `GPU_LIST` unset for multi-GPU role placement. It denotes round-robin
  single-GPU array routing, not model-parallel memory pooling.
- Use array concurrency `%1` when every element reserves all four GPUs.
- A physical node or GPU-ordinal change is acceptable for the empirical run if
  the environment, artifacts, logical role topology, and frozen configuration
  are unchanged. Record placement for provenance; do not gate the paper on it.
- Slurm cannot protect an allocated GPU from unmanaged processes. Inspect the
  initial `nvidia-smi` snapshot and rerun only failed array indices after a
  genuine hardware/occupancy failure.
- Use `OVERWRITE=1` only for the intended current task. After a result-affecting
  code/configuration change, use a fresh output root.

## Verification before the fresh root

The empirical-policy patch is integrated locally. On 2026-08-30, the complete
LinkRadius unit suite passed with 261 tests and 78 dependency-gated skips;
compileall, shell syntax checks, parser smoke checks, and `git diff --check`
also passed. There is deliberately no hard-coded expected source hash here:
verify the synced cluster checkout immediately before creating the fresh root.
Run:

```bash
git status --short --untracked-files=all -- RecursiveMAS experiments/linkradius
python -m unittest discover -s experiments/linkradius/tests
python -m compileall -q RecursiveMAS experiments/linkradius
bash -n experiments/linkradius/linkradius_common.sh \
  experiments/linkradius/run_linkradius_engineering.sh \
  experiments/linkradius/run_linkradius_smoke.sh \
  experiments/linkradius/run_linkradius_pilot.sh \
  experiments/linkradius/run_linkradius_attacks.sh
git diff --check
```

After those checks pass, record the final commit and `source_hash(Path.cwd())`,
sync the complete tree, and create the fresh root. Do not copy an old expected
hash or launch command from chat history. Documentation edits and `#SBATCH`
placement directives no longer change the scientific source hash; executable
Python/shell/config changes still do.

## Outputs and interpretation

The core Phase-4 outputs are the authenticated versions of:

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
attack_validation_gate.json
```

`passed=true` means required metrics were estimable; it does not mean
LinkRadius beat the baselines. Inspect estimates and paired confidence
intervals. If an interval for the LinkRadius-minus-baseline contrast includes
zero, improvement is not supported. If the run is underpowered, report the
eligible denominator, censoring pattern, class support, and interval width
without converting it into a positive theorem claim.

## Prompt for another Codex session

> Read `handoff.md` and `experiments/linkradius/README.md` completely. Inspect
> the working tree and finish verifying the empirical-policy patch. Do not use
> a stale source hash or an output root created under `dual_correct`. First run
> a fresh `sequential_light` forced-margin census with generation disabled.
> Guide me one authenticated command at a time. Keep raw split/freeze/no-test-
> tuning/provenance/censoring safeguards, but do not require exact legacy
> equivalence, the same physical GPUs, or bitwise clean reruns for the paper.
