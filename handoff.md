# LinkRadius experiment handoff

Last updated: 2026-08-17 (Asia/Singapore)

## How to use this file

When continuing from another computer, open the same repository revision and
tell the new Codex session:

> Read `handoff.md`, inspect the repository state, and continue from the current
> checkpoint. Do not skip a stage or modify authenticated experiment source.

The model runs and large outputs live on the cluster, not on the desktop or
laptop. The safest way to continue is to SSH from either computer into the same
cluster checkout:

```text
/export/home2/suaq0001/Robust-RMAS
```

Do not copy `outputs/`, model checkpoints, or SLURM logs through Git. Use Git
only for source code, tests, documentation, and this handoff.

## Scientific status

### Original baseline

The preregistered `sequential_light` run under `outputs/linkradius-v3` failed
the mandatory Phase 2 smoke gate. After correcting the strict parser, the full
40-row validation screening cohort contained only 3 dual-correct rows; the
required range is 10--20. That result remains a failed baseline and must not be
relabelled as a pass. Do not resume its causal, probe, attack, or pilot stages.

### Current follow-up

The next run is a separate exploratory stronger-model condition:

```text
OUT_ROOT=outputs/linkradius-scaled-mp-v1
STYLE=sequential_scaled
METHOD=ours_recursive
DATASETS=gpqa
ROUNDS=2
SEEDS=42
LATENT_LENGTH=48
```

`sequential_scaled` uses a Gemma-3 4B planner, Llama-3.2 3B critic, Qwen-3.5
4B solver, and the scaled learned outer adapters. It must be reported as a
follow-up condition, not as a replacement for the failed light baseline.

The previous `outputs/linkradius-scaled-v1` attempt is archived incomplete. Its
single-GPU gradient task 1 OOMed, and the model-parallel source update makes its
authenticated artifacts stale. Do not resume or copy those artifacts into the
new root.

## Current checkpoint

Historical `scaled-v1` engineering stages completed before the source change:

- `split`
- `discover` (missing tasks 11, 13, and 18 were rerun)
- `freeze_execution`
- `clean`
- `replay`
- probe `grid`
- `probe`

Engineering gradient task 1 (`p2c@0`, the full downstream autograd replay) is
currently blocked. It exhausted an idle 80 GB A100: PyTorch had 78.09 GiB
actively allocated, only 646 MiB reserved-but-unused, and less than 1 MiB free.
This is a true-capacity OOM, not allocator fragmentation. Task 0 completion is
not yet confirmed in chat.

The scaled Qwen3.5 solver also reported that its fast linear-attention path was
unavailable. The active environment is PyTorch 2.9.0+cu128, Transformers 5.3.0,
and Triton 3.5.0; `flash-linear-attention`, `fla-core`, and `causal-conv1d` are
not installed, and system `nvcc` is unavailable. Do not accept a gradient
artifact produced by mixing a new kernel environment into `scaled-v1`.

The previous gradient submission was:

```bash
cd /export/home2/suaq0001/Robust-RMAS
conda activate recursivemas

OUT_ROOT="$PWD/outputs/linkradius-scaled-v1" \
STYLE=sequential_scaled \
LATENT_LENGTH=48 \
OVERWRITE=1 \
GPU_LIST="1 2" \
LR_STAGE=gradient \
sbatch -p PA10080q -w node04 --array=0-1%2 \
experiments/linkradius/run_linkradius_engineering.sh
```

The code now supports source-hashed three-GPU role placement. The planner,
critic, and solver can use logical `cuda:0`, `cuda:1`, and `cuda:2`; learned
outer adapters remain on their source role and relays cross devices without
detaching autograd. Start the complete engineering workflow again under
`outputs/linkradius-scaled-mp-v1`. Export these values for every command,
including CPU freeze and validation commands:

```bash
export OUT_ROOT="$PWD/outputs/linkradius-scaled-mp-v1"
export STYLE=sequential_scaled
export LATENT_LENGTH=48
export DEVICE=cuda:0
export PLANNER_DEVICE=cuda:0
export CRITIC_DEVICE=cuda:1
export SOLVER_DEVICE=cuda:2
export GPU_LIST=
```

Each GPU stage needs one Slurm task containing three GPUs, using the site's
allocation flag such as `--ntasks=1 --gres=gpu:3`. `GPU_LIST` must remain empty;
it is single-GPU array routing and is deliberately rejected with role placement.
Changing the node is allowed when the same logical three-device topology is
available. Adjust `-p` and `-w` on `sbatch`; do not edit launcher directives.

## Next engineering work after gradient succeeds

Engineering validation requires an authenticated legacy-equivalence run. Find
the unique current clean trajectory and its raw index:

```bash
python - <<'PY'
from pathlib import Path
import torch

from experiments.linkradius.io_utils import source_hash, verify_completion

repo = Path.cwd().resolve()
root = repo / "outputs/linkradius-scaled-mp-v1/engineering/gpqa/R2/validation/clean"
current_source = source_hash(repo)
candidates = []

for path in sorted(root.rglob("clean_trajectory.pt")):
    try:
        completion = verify_completion(path.parent)
    except Exception:
        continue
    declared = {
        item.get("path")
        for item in completion.get("artifacts", [])
        if isinstance(item, dict)
    }
    if (
        completion.get("source_hash") == current_source
        and path.name in declared
    ):
        candidates.append(path.resolve())

if len(candidates) != 1:
    raise SystemExit(f"expected one authenticated clean trajectory, found {candidates}")

trajectory_path = candidates[0]
trajectory = torch.load(trajectory_path, map_location="cpu", weights_only=False)
print(f"export CLEAN_TRAJECTORY={trajectory_path}")
print(f"export RAW_INDEX={int(trajectory.raw_indices[0])}")
PY
```

Copy the two printed `export` lines into the shell, then run the release path on
a GPU. The scaled style and latent length are mandatory:

```bash
export OUT_ROOT="$PWD/outputs/linkradius-scaled-mp-v1"
export LEGACY_RESULTS="$OUT_ROOT/legacy_release.jsonl"
export LEGACY_TRACE="$OUT_ROOT/legacy_release_trace.pt"

python RecursiveMAS/run.py \
  --style sequential_scaled \
  --dataset gpqa --dataset_split train \
  --method ours_recursive --num_recursive_rounds 2 \
  --num_samples 1 --sample_indices "$RAW_INDEX" \
  --batch_size 1 --latent_length 48 \
  --seed 42 --deterministic 1 --device cuda:0 \
  --result_jsonl "$LEGACY_RESULTS" \
  --lc_trace_path "$LEGACY_TRACE" \
  --lc_trace_sites p2c,c2s,s2p \
  --lc_trace_rounds 0,1 --lc_trace_dtype float32

python -m experiments.linkradius.compare_legacy_equivalence \
  --trajectory "$CLEAN_TRAJECTORY" \
  --legacy-trace "$LEGACY_TRACE" \
  --legacy-results "$LEGACY_RESULTS" \
  --output "$OUT_ROOT/legacy_equivalence.json"
```

If the comparator reports `"passed": true`, validate Phase 1:

```bash
OUT_ROOT="$PWD/outputs/linkradius-scaled-mp-v1" \
STYLE=sequential_scaled \
LATENT_LENGTH=48 \
PLANNER_DEVICE=cuda:0 \
CRITIC_DEVICE=cuda:1 \
SOLVER_DEVICE=cuda:2 \
LEGACY_EQUIVALENCE="$PWD/outputs/linkradius-scaled-mp-v1/legacy_equivalence.json" \
OVERWRITE=1 \
LR_STAGE=validate \
bash experiments/linkradius/run_linkradius_engineering.sh
```

Do not start scaled Phase 2 unless this produces a passed
`outputs/linkradius-scaled-mp-v1/engineering_gate.json`.

## Source and artifact rules

- Completion files authenticate the configuration, source hash, artifacts, and
  row counts. A file merely existing is not proof that its task succeeded.
- Switching nodes does not change the source hash when every node sees the same
  shared checkout and model files.
- Editing any tracked LinkRadius/RecursiveMAS experiment source during a phase
  makes earlier gates stale. Scheduler choices belong on the `sbatch` command
  line, not inside `.sh` files.
- Do not mix artifacts from `linkradius-v3`, `linkradius-scaled-v1`, and
  `linkradius-scaled-mp-v1`.
- Do not reuse discovery output as the clean reference trajectory.
- Do not continue after a failed engineering or smoke gate.
- The strict parser should report `STRICT_CHOICE_VERSION=linkradius_choice_v2`.
- The engineering validator must pass `args.latent_length` as
  `expected_latent_steps`; this is required for scaled GPQA's length 48.

Quick source check:

```bash
python - <<'PY'
from pathlib import Path
from RecursiveMAS.inference_utils.linkradius import STRICT_CHOICE_VERSION
from experiments.linkradius.io_utils import source_hash

print("strict_parser:", STRICT_CHOICE_VERSION)
print("source_hash:", source_hash(Path.cwd()))
PY

git status --short --untracked-files=all RecursiveMAS experiments/linkradius
```

## Moving between desktop and laptop

1. Commit and push source changes plus `handoff.md` to a private Git branch.
2. On the other computer, pull the same branch and read this file before doing
   anything.
3. SSH into the same cluster checkout and continue there so all SLURM outputs
   and model paths remain available.
4. Do not pull a different source revision into the cluster while authenticated
   jobs from the current revision are still running.
5. Before leaving a computer, update the **Current checkpoint** section with
   completed/failed array indices, job IDs, node/GPU choices, and the exact next
   command; commit and push that update.

For a persistent cluster shell, use `tmux` or another cluster-approved terminal
multiplexer. SLURM jobs continue independently after SSH disconnects, but a
plain interactive command may not.
